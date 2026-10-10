from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class QwenLoraWeights:
    matrices: dict[str, Any]
    deltas: dict[str, Any]


def parse_qwen_lora(path: Path) -> QwenLoraWeights:
    """Read every local tensor on CPU; never delegate a mixed file to Diffusers.

    Old Qwen module names already match the transformer. Alpha/rank is folded
    into B; PEFT therefore receives its default unit intrinsic scale. Request
    strength is applied separately, exactly once, by set_adapters.
    """
    import torch
    from safetensors import safe_open

    source: dict[str, Any] = {}
    suffixes = (".lora_down.weight", ".lora_up.weight", ".lora_A.weight",
                ".lora_B.weight", ".alpha", ".diff_b", ".diff")
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        for key in handle.keys():  # noqa: SIM118 -- safe_open exposes keys, not iteration
            name = key.removeprefix("diffusion_model.").removeprefix("transformer.")
            if not name.endswith(suffixes) or name in source:
                raise ValueError(f"Unknown or duplicate Qwen LoRA key: {key}")
            value = handle.get_tensor(key)
            # Scalar integer alphas are valid, but matrices/deltas must float.
            if (not value.is_floating_point() and not name.endswith(".alpha")) or not torch.isfinite(value).all().item():
                raise ValueError(f"Nonfinite or nonfloating Qwen LoRA tensor: {key}")
            source[name] = value
    if not source:
        raise ValueError("Qwen LoRA contains no tensors.")

    groups: dict[str, dict[str, Any]] = {}
    deltas: dict[str, Any] = {}
    for name, value in source.items():
        suffix = next(suffix for suffix in suffixes if name.endswith(suffix))
        module = name.removesuffix(suffix)
        if not module:
            raise ValueError(f"Empty Qwen target: {name}")
        if suffix in {".diff", ".diff_b"}:
            target = module + (".bias" if suffix == ".diff_b" else ".weight")
            if target in deltas:
                raise ValueError(f"Duplicate Qwen direct target: {target}")
            deltas[target] = value
        else:
            groups.setdefault(module, {})[suffix] = value

    matrices: dict[str, Any] = {}
    for module, values in groups.items():
        old = {".lora_down.weight", ".lora_up.weight"}
        normalized = {".lora_A.weight", ".lora_B.weight"}
        keys = set(values) - {".alpha"}
        if keys == old:
            a, b = values[".lora_down.weight"], values[".lora_up.weight"]
        elif keys == normalized:
            a, b = values[".lora_A.weight"], values[".lora_B.weight"]
        else:
            raise ValueError(f"Incomplete or mixed Qwen LoRA pair: {module}")
        if a.ndim != 2 or b.ndim != 2 or min(a.shape) <= 0 or min(b.shape) <= 0 or a.shape[0] != b.shape[1]:
            raise ValueError(f"Invalid Qwen LoRA rank/shapes: {module}")
        rank = a.shape[0]
        alpha = values.get(".alpha")
        if alpha is not None and alpha.numel() != 1:
            raise ValueError(f"Qwen LoRA alpha must be scalar: {module}")
        scale = float(alpha.item()) / rank if alpha is not None else 1.0
        if not math.isfinite(scale):
            raise ValueError(f"Qwen LoRA alpha/rank must be finite: {module}")
        # Do not invent rank16 or reshape/split any module. Retain the actual
        # pair and use at least float32 to avoid fp16 scale overflow/underflow.
        b = b.to(dtype=torch.float64 if b.dtype == torch.float64 else torch.float32) * scale
        if not torch.isfinite(b).all().item():
            raise ValueError(f"Qwen LoRA scaled B is nonfinite: {module}")
        matrices[f"transformer.{module}.lora_A.weight"] = a
        matrices[f"transformer.{module}.lora_B.weight"] = b
    return QwenLoraWeights(matrices, deltas)


def qwen_parameters(transformer: Any) -> dict[str, Any]:
    """Canonical checkpoint names even while the previous PEFT adapter exists."""
    result = {}
    for name, parameter in transformer.named_parameters():
        if ".lora_" in name:
            continue
        name = name.replace(".base_layer.", ".")
        if name in result:
            raise ValueError(f"Ambiguous Qwen parameter: {name}")
        result[name] = parameter
    return result


def validate_qwen_lora(
    transformer: Any, weights: QwenLoraWeights, *, require_materialized: bool = True,
) -> dict[str, Any]:
    """Verify real module/parameter dimensions without allocating a base model."""
    import torch

    parameters = qwen_parameters(transformer)
    modules = dict(transformer.named_modules())

    def target(name: str, shape: tuple[int, ...]) -> Any:
        parameter = parameters.get(name)
        if parameter is None:
            raise ValueError(f"Unknown Qwen transformer parameter: {name}")
        if require_materialized and parameter.is_meta:
            raise ValueError(f"Qwen parameter is still offloaded to meta: {name}")
        if not parameter.is_floating_point() or tuple(parameter.shape) != shape:
            raise ValueError(f"Qwen target shape/dtype mismatch: {name}: {shape} vs {tuple(parameter.shape)}")
        return parameter

    direct = {name: target(name, tuple(delta.shape)) for name, delta in weights.deltas.items()}
    for key, a in weights.matrices.items():
        if not key.endswith(".lora_A.weight"):
            continue
        module_name = key.removeprefix("transformer.").removesuffix(".lora_A.weight")
        b = weights.matrices[f"transformer.{module_name}.lora_B.weight"]
        parameter = target(f"{module_name}.weight", (b.shape[0], a.shape[1]))
        module = modules.get(module_name)
        base = getattr(module, "base_layer", module)
        if not isinstance(base, torch.nn.Linear) or getattr(base, "weight", None) is not parameter:
            raise ValueError(f"Qwen LoRA requires an unquantized Linear target: {module_name}")
        for value in (a, b):
            if not torch.isfinite(value.to(dtype=parameter.dtype)).all().item():
                raise ValueError(f"Qwen LoRA overflows target dtype: {module_name}")
    return direct


@dataclass
class QwenDirectState:
    """Exact CPU originals plus live handles retained across PEFT injection."""
    originals: dict[str, Any] = field(default_factory=dict)
    parameters: dict[str, Any] = field(default_factory=dict)

    def restore(self, transformer: Any) -> None:
        import torch

        # Accelerate detach can replace Parameter objects when materializing
        # meta weights. Rebind after hook removal; never write a stale handle.
        current = qwen_parameters(transformer)
        with torch.no_grad():
            for name, original in self.originals.items():
                parameter = current.get(name)
                if parameter is None or parameter.is_meta or tuple(parameter.shape) != tuple(original.shape):
                    raise ValueError(f"Cannot restore Qwen direct parameter: {name}")
                parameter.copy_(original.to(device=parameter.device, dtype=parameter.dtype))
        self.clear()

    def prepare(self, transformer: Any, adapters: list[tuple[QwenLoraWeights, float]]) -> dict[str, Any]:
        """Aggregate all signed unit deltas and validate before any mutation."""
        import torch

        parameters = qwen_parameters(transformer)
        totals: dict[str, Any] = {}
        for weights, strength in adapters:
            if not math.isfinite(strength):
                raise ValueError("Qwen LoRA strength must be finite.")
            validate_qwen_lora(transformer, weights)
            for name, delta in weights.deltas.items():
                dtype = torch.float64 if parameters[name].dtype == torch.float64 else torch.float32
                weighted = delta.to(dtype=dtype) * strength
                totals[name] = totals[name] + weighted if name in totals else weighted
        updates = {}
        for name, total in totals.items():
            parameter = parameters[name]
            original = self.originals.get(name)
            if original is None:
                original = parameter.detach().to(device="cpu").clone()
            value = (original.to(dtype=total.dtype) + total).to(dtype=parameter.dtype)
            if not torch.isfinite(total).all().item() or not torch.isfinite(value).all().item():
                raise ValueError(f"Qwen direct update overflows target dtype: {name}")
            updates[name] = value
        return updates

    def capture(self, transformer: Any, updates: dict[str, Any]) -> None:
        parameters = qwen_parameters(transformer)
        self.parameters = {name: parameters[name] for name in updates}
        self.originals = {
            name: parameter.detach().to(device="cpu").clone()
            for name, parameter in self.parameters.items()
        }

    def apply(self, updates: dict[str, Any]) -> None:
        import torch

        with torch.no_grad():
            for name, value in updates.items():
                parameter = self.parameters[name]
                parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))

    def clear(self) -> None:
        self.originals.clear()
        self.parameters.clear()