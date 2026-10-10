from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path
from typing import Any

from photo_edit_studio.config import settings

QWEN21_OFFICIAL_EXTRACT_KEY = "qwen-2.1-turbo-official-extract"
QWEN21_OFFICIAL_EXTRACT_LORA = {
    "version_id": 3394831,
    "file_id": 3284648,
    "filename": "qwen_image_2.1_turbo_lora_avg_rank_178_bf16.safetensors",
    "url": "https://civitai.com/api/download/models/3394831?fileId=3284648",
    "size": 913314512,
    "sha256": "208dd43250e1e01467ba190572ae2e7107a7870ec1ff026bc65f791fa1e80e95",
}


def qwen21_official_extract_path() -> Path:
    return settings.lora_dir / "qwen21-official" / QWEN21_OFFICIAL_EXTRACT_LORA["filename"]


def validate_qwen21_official_extract_lora(path: Path) -> None:
    """Verify the mandatory pin once per load, without materializing tensors."""
    from safetensors import safe_open

    pin = QWEN21_OFFICIAL_EXTRACT_LORA
    if not path.is_file():
        raise FileNotFoundError(f"Mandatory official extracted LoRA is missing at {path}.")
    if path.stat().st_size != pin["size"]:
        raise ValueError("Official extracted LoRA size mismatch.")
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != pin["sha256"]:
        raise ValueError("Official extracted LoRA SHA256 mismatch.")
    with safe_open(path, framework="pt", device="cpu") as handle:
        if not handle.keys():
            raise ValueError("Official extracted LoRA has an empty Safetensors header.")


def _validate_qwen21_extracted_targets(
    transformer: Any, lora: dict[str, Any], deltas: dict[str, Any],
    *, require_materialized: bool = True,
) -> dict[str, Any]:
    """Validate every target without installing adapters or changing parameters."""
    import torch

    parameters = dict(transformer.named_parameters())
    modules = dict(transformer.named_modules())
    direct_parameters = {}

    def target(name: str, shape: tuple[int, ...]) -> Any:
        parameter = parameters.get(name)
        if parameter is None:
            raise ValueError(f"Extracted weights target an unknown transformer parameter: {name}")
        if require_materialized and parameter.is_meta:
            raise ValueError(f"Extracted weights require a materialized parameter, not meta: {name}")
        if not parameter.is_floating_point() or tuple(parameter.shape) != shape:
            raise ValueError(f"Extracted weight shape/dtype mismatch for {name}: {shape} vs {tuple(parameter.shape)}")
        return parameter

    for name, delta in deltas.items():
        direct_parameters[name] = target(name, tuple(delta.shape))
    for key, a in lora.items():
        if not key.endswith(".lora_A.weight"):
            continue
        module_name = key.removeprefix("transformer.").removesuffix(".lora_A.weight")
        b = lora[f"transformer.{module_name}.lora_B.weight"]
        if a.ndim != 2 or b.ndim != 2 or a.shape[0] <= 0 or a.shape[0] != b.shape[1]:
            raise ValueError(f"Invalid extracted LoRA matrix pair: {module_name}")
        parameter = target(f"{module_name}.weight", (b.shape[0], a.shape[1]))
        module = modules.get(module_name)
        if module is None or getattr(module, "weight", None) is not parameter:
            raise ValueError(f"Extracted LoRA target is not a weighted module: {module_name}")
        if not torch.isfinite(a).all().item() or not torch.isfinite(b).all().item():
            raise ValueError(f"Nonfinite converted extracted LoRA weights: {module_name}")
    return direct_parameters


def _split_qwen21_fused_lora(lora: dict[str, Any]) -> dict[str, Any]:
    """Match the installed checkpoint converter's gate-first SwiGLU split."""
    normalized = dict(lora)
    for key in list(lora):
        if not key.endswith(".img_mlp.gate_up.lora_A.weight"):
            continue
        b_key = key.replace(".lora_A.weight", ".lora_B.weight")
        a, b = normalized.pop(key), normalized.pop(b_key)
        if a.ndim != 2 or b.ndim != 2 or b.shape[0] % 2:
            raise ValueError(f"Invalid fused extracted SwiGLU matrix pair: {key}")
        gate, up = b.chunk(2, dim=0)
        for module, value in (("gate_layer", gate), ("proj", up)):
            new_a = key.replace(".gate_up.", f".{module}.")
            new_b = b_key.replace(".gate_up.", f".{module}.")
            if new_a in normalized or new_b in normalized:
                raise ValueError(f"Colliding extracted SwiGLU targets: {new_a}")
            normalized[new_a] = a
            normalized[new_b] = value
    return normalized


def split_qwen21_fused_lora(lora: dict[str, Any]) -> dict[str, Any]:
    """Normalize matrix-only Qwen 2.1 adapters using the gate-first converter."""
    return _split_qwen21_fused_lora(lora)


def apply_qwen21_extracted_weights(pipe: Any, path: Path) -> None:
    """Apply the mixed extraction before offload; dispose of the pipe on failure.

    Pin verification belongs to the dedicated adapter, so small synthetic subsets
    can exercise this loader. All source keys, pairs, scales and actual targets
    are checked before either a direct update or a LoRA installation occurs.
    """
    import torch
    from diffusers.loaders.lora_conversion_utils import (
        _convert_non_diffusers_qwen_lora_to_diffusers,
    )
    from safetensors import safe_open

    if not all(callable(getattr(pipe, name, None)) for name in ("load_lora_weights", "set_adapters")):
        raise ValueError("Extracted weights require a LoRA-capable pipeline.")
    source: dict[str, Any] = {}
    deltas: dict[str, Any] = {}
    suffixes = (".lora_down.weight", ".lora_up.weight", ".alpha")
    # The pin contains only these 65 direct updates. Subsets are supported, but
    # unknown deltas (including other normalization layers) must fail closed.
    norm_delta = re.compile(r"transformer_blocks\.(?:[0-9]|[12][0-9]|3[01])\.attn\.norm_[qk]\.diff")
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        for key in handle.keys():  # noqa: SIM118 -- safe_open exposes keys, not iteration
            name = key.removeprefix("diffusion_model.")
            if name in source:
                raise ValueError(f"Duplicate normalized extracted key: {name}")
            if not name.endswith(suffixes) and not (
                norm_delta.fullmatch(name) or name == "txt_in.text_norm.diff"
            ):
                raise ValueError(f"Unsupported extracted weight key: {key}")
            value = handle.get_tensor(key)
            if not value.is_floating_point() or not torch.isfinite(value).all().item():
                raise ValueError(f"Nonfinite/nonfloating extracted weight: {key}")
            source[name] = value
    if not source:
        raise ValueError("Extracted weights are empty.")

    lora_source: dict[str, Any] = {}
    bases = set()
    for name, value in source.items():
        if name.endswith(".diff"):
            # QwenImage21TextProjection really owns text_norm (not txt_norm).
            deltas[name.removesuffix(".diff") + ".weight"] = value
        else:
            suffix = next(suffix for suffix in suffixes if name.endswith(suffix))
            bases.add(name.removesuffix(suffix))
            lora_source[name] = value

    zero_alpha = set()
    for base in bases:
        keys = [base + suffix for suffix in suffixes]
        if not all(key in lora_source for key in keys):
            raise ValueError(f"Incomplete extracted down/up/alpha triple: {base}")
        down, up, alpha = (lora_source[key] for key in keys)
        if down.ndim != 2 or up.ndim != 2 or down.shape[0] <= 0 or down.shape[0] != up.shape[1]:
            raise ValueError(f"Invalid extracted LoRA matrix pair: {base}")
        if alpha.numel() != 1:
            raise ValueError(f"Extracted LoRA alpha must be scalar: {base}")
        scale = float(alpha.item())
        if not math.isfinite(scale) or scale < 0:
            raise ValueError(f"Extracted LoRA alpha must be finite and nonnegative: {base}")
        if scale > 0 and scale / down.shape[0] == 0:
            raise ValueError(f"Extracted LoRA alpha/rank underflows converter scaling: {base}")
        if scale == 0:
            # The installed converter's scale-balancing loop never terminates
            # for zero. Convert at unit scale, then zero B: the exact zero update.
            zero_alpha.add(base)
            lora_source[base + ".alpha"] = torch.tensor(float(down.shape[0]), dtype=torch.float64)

    # Do not send any .diff keys through the legacy LoRA converter. It consumes
    # alpha/rank into A/B, so the resulting adapter must be used at weight 1.
    lora = _convert_non_diffusers_qwen_lora_to_diffusers(dict(lora_source)) if bases else {}
    expected = {
        f"transformer.{base}.lora_{letter}.weight"
        for base in bases for letter in ("A", "B")
    }
    if set(lora) != expected:
        raise ValueError("Installed Qwen LoRA converter lost or renamed extracted matrix targets.")
    for base in zero_alpha:
        key = f"transformer.{base}.lora_B.weight"
        lora[key] = torch.zeros_like(lora[key])
    lora = _split_qwen21_fused_lora(lora)
    direct_parameters = _validate_qwen21_extracted_targets(pipe.transformer, lora, deltas)

    with torch.no_grad():
        for name, parameter in direct_parameters.items():
            parameter.add_(deltas[name].to(device=parameter.device, dtype=parameter.dtype))
    if lora:
        # Passing the normalized dictionary avoids re-converting the mixed file.
        pipe.load_lora_weights(lora, adapter_name="official_extract", local_files_only=True)
        pipe.set_adapters(["official_extract"], adapter_weights=[1.0])