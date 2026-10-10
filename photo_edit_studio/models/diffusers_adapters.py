from __future__ import annotations

import inspect
import json
import logging
import math
from pathlib import Path
from typing import Any

from photo_edit_studio.config import settings
from photo_edit_studio.models.base import ModelAdapter
from photo_edit_studio.models.memory import cuda_available, cuda_vram_gb
from photo_edit_studio.models.qwen21_official_extract import (
    QWEN21_OFFICIAL_EXTRACT_KEY,
    apply_qwen21_extracted_weights,
    qwen21_official_extract_path,
    split_qwen21_fused_lora,
    validate_qwen21_official_extract_lora,
)
from photo_edit_studio.models.qwen_aio import (
    materialize_qwen_rope,
    resolve_qwen_aio_checkpoint,
    transformer_state_from_comfy_aio,
)
from photo_edit_studio.models.registry import MODEL_SPECS
from photo_edit_studio.progress import GenerationProgress
from photo_edit_studio.swap import QWEN21_BFS_PINS, swap_profile, validate_qwen21_bfs_file
from photo_edit_studio.types import GenerationRequest

logger = logging.getLogger(__name__)

IDENTITY_SUFFIX = (
    " Preserve the subject's exact identity, facial geometry, skin texture, pose, body "
    "proportions, hands, camera angle, and lighting unless explicitly requested otherwise."
)
FRAMING_SUFFIX = (
    " Keep the original full image framing and composition: retain the same camera distance, "
    "subject scale and position, and all visible clothing, body parts, and background at the "
    "edges. Do not crop, zoom in, or reframe the source photograph."
)


def _runtime() -> tuple[Any, Any]:
    import diffusers
    import torch

    return diffusers, torch


def _pipeline_class(diffusers: Any, *names: str) -> Any:
    for name in names:
        value = getattr(diffusers, name, None)
        if value is not None:
            return value
    raise RuntimeError(
        f"Installed Diffusers does not provide {', '.join(names)}. Reinstall the project requirements."
    )


def _local_path(path: Path, label: str) -> str:
    if not path.exists():
        raise FileNotFoundError(
            f"{label} is not downloaded at {path}. Run scripts/download_models.py first."
        )
    return str(path)


def _supported_call(pipe: Any, kwargs: dict[str, Any]) -> Any:
    signature = inspect.signature(pipe.__call__)
    accepts_any = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    if accepts_any:
        return pipe(**kwargs)
    return pipe(**{key: value for key, value in kwargs.items() if key in signature.parameters})


def _extract_images(output: Any) -> list[Any]:
    images = getattr(output, "images", None)
    if images is None and isinstance(output, tuple) and output:
        images = output[0]
    if images is None:
        raise RuntimeError("The pipeline returned no images.")
    return list(images)


def _progress_kwargs(
    pipe: Any, progress: GenerationProgress | None, image_index: int, count: int, steps: int = 8
) -> dict[str, Any]:
    if progress is None or "callback_on_step_end" not in inspect.signature(pipe.__call__).parameters:
        return {}

    def on_step_end(_pipe: Any, step: int, _timestep: Any, tensors: dict[str, Any]) -> dict[str, Any]:
        progress.inference(image_index, count, step + 1, steps)
        return tensors

    return {"callback_on_step_end": on_step_end, "callback_on_step_end_tensor_inputs": []}


class DiffusersAdapter(ModelAdapter):
    pipeline_names: tuple[str, ...] = ()

    def load(self) -> None:
        self._validate_hardware()
        diffusers, torch = _runtime()
        pipeline_class = _pipeline_class(diffusers, *self.pipeline_names)
        logger.info("Diffusers pipeline load begin")
        self.pipe = pipeline_class.from_pretrained(
            _local_path(self.spec.local_path, self.spec.label),
            torch_dtype=torch.bfloat16,
            local_files_only=True,
            low_cpu_mem_usage=True,
            requires_safety_checker=False,
            safety_checker=None,
        )
        logger.info("Diffusers pipeline load end")
        logger.info("Diffusers pipeline offload configuration begin")
        self.configure_memory()
        logger.info("Diffusers pipeline offload configuration end")

    def _validate_hardware(self) -> None:
        if not cuda_available():
            raise RuntimeError(
                "A CUDA-capable NVIDIA GPU is required. Install the CUDA PyTorch wheel and verify the driver."
            )
        vram = cuda_vram_gb()
        if vram + 0.5 < self.spec.minimum_vram_gb:
            raise RuntimeError(
                f"{self.spec.label} needs about {self.spec.minimum_vram_gb} GB VRAM; "
                f"detected {vram:.1f} GB."
            )

    def generate(
        self, request: GenerationRequest, progress: GenerationProgress | None = None
    ) -> list[Any]:
        if self.pipe is None:
            if progress is not None:
                progress.update(0.05, f"Loading {self.spec.label} · this may take a few minutes")
            self.load()
        if progress is not None:
            progress.update(0.12, "Preparing LoRAs and prompt")
        self.apply_loras(request)
        if request.workflow == "swap":
            parameters = inspect.signature(self.pipe.__call__).parameters
            if "image" not in parameters:
                raise RuntimeError(
                    f"{self.spec.label} pipeline has no two-image conditioning; "
                    "cannot perform a head swap with this Diffusers version."
                )
        if request.workflow == "text" and self.spec.key != "flux-klein-4b":
            raise ValueError(f"{self.spec.label} has no supported text-to-image path.")
        _, torch = _runtime()
        text_only = request.workflow == "text"
        prompt = request.prompt
        if not text_only:
            prompt += FRAMING_SUFFIX
            if request.preserve_identity:
                prompt += IDENTITY_SUFFIX
        images: list[Any] = []
        for index in range(request.count):
            if progress is not None:
                progress.inference(index, request.count, 0, request.steps)
            generator = torch.Generator(device="cpu").manual_seed(request.seed + index)
            kwargs = {
                "prompt": prompt,
                "negative_prompt": request.negative_prompt or None,
                "width": request.width,
                "height": request.height,
                "num_inference_steps": request.steps,
                "guidance_scale": request.guidance,
                "true_cfg_scale": request.true_cfg,
                "strength": request.strength,
                "generator": generator,
                "num_images_per_prompt": 1,
            }
            if not text_only:
                kwargs["image"] = request.images if len(request.images) > 1 else request.images[0]
                kwargs["images"] = request.images
            kwargs.update(_progress_kwargs(self.pipe, progress, index, request.count, request.steps))
            with torch.no_grad():
                images.extend(_extract_images(_supported_call(self.pipe, kwargs)))
        return images


class QwenAdapter(DiffusersAdapter):
    pipeline_names = ("QwenImageEditPlusPipeline", "QwenImageEditPipeline")

    def __init__(self, spec: Any) -> None:
        super().__init__(spec)
        from photo_edit_studio.models.qwen2511_lora import QwenDirectState

        self._direct_state = QwenDirectState()

    def apply_loras(self, request: GenerationRequest) -> None:
        from photo_edit_studio.models.qwen2511_lora import parse_qwen_lora

        signature = self.lora_signature(request)
        if signature == self._lora_signature:
            return
        if self.pipe is None:
            raise RuntimeError("Load the Qwen pipeline before applying LoRAs.")
        pipe = self.pipe
        try:
            # Sequential offload stores authoritative weights outside the meta
            # parameters. Detach BEFORE validation/capture/restore/PEFT loading.
            remove_hooks = getattr(pipe, "remove_all_hooks", None)
            if not callable(remove_hooks):
                raise TypeError("Qwen LoRA changes require pipeline.remove_all_hooks().")
            remove_hooks()
            unload = getattr(pipe, "unload_lora_weights", None)
            load = getattr(pipe, "load_lora_weights", None)
            set_adapters = getattr(pipe, "set_adapters", None)
            if not callable(unload) or (request.loras and not all(callable(method) for method in (load, set_adapters))):
                raise TypeError(f"{self.spec.label} does not expose Diffusers LoRA support.")
            names = [item.adapter_name for item in request.loras]
            if any(not name for name in names) or len(set(names)) != len(names):
                raise ValueError("Qwen LoRA adapter names must be nonempty and unique.")
            adapters = [(parse_qwen_lora(item.path), float(item.weight)) for item in request.loras]
            # Validate every file, every target, and the aggregate against the
            # original checkpoint before replacing the previous adapter set.
            updates = self._direct_state.prepare(pipe.transformer, adapters)
            self._direct_state.restore(pipe.transformer)
            unload()
            self._direct_state.capture(pipe.transformer, updates)
            matrix_names: list[str] = []
            matrix_weights: list[float] = []
            for item, (weights, strength) in zip(request.loras, adapters, strict=True):
                if weights.matrices:
                    load(weights.matrices, adapter_name=item.adapter_name, local_files_only=True)
                    matrix_names.append(item.adapter_name)
                    matrix_weights.append(strength)
            if matrix_names:
                set_adapters(matrix_names, adapter_weights=matrix_weights)
            # PEFT retains original base Parameter objects: use the handles
            # captured before injection, not names renamed to .base_layer.
            self._direct_state.apply(updates)
            self.configure_memory()
            self._lora_signature = signature
        except Exception:
            # Fail closed, including partial loader/offload failures. Do not
            # expose a mutated pipeline or retain base-model parameter handles.
            self.unload()
            raise

    def unload(self) -> None:
        super().unload()
        self._direct_state.clear()


QWEN21_OFFICIAL_SIGMAS = (
    1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568,
)
QWEN21_OFFICIAL_UPGRADE = (
    "Official Qwen 2.1 Turbo requires upstream Diffusers PR #14950, merge commit "
    "da1d3829cf08d4f329b526d89e17cc035c049d8d or a compatible descendant, "
    "and Transformers >=5.17,<6. A dev version label alone is not sufficient. "
    "Upgrade the environment explicitly after checking other model compatibility."
)


def validate_qwen21_official_request(request: GenerationRequest) -> None:
    """Reject unsupported operations before model selection/allocation."""
    if request.model_key not in {"qwen-2.1-turbo-official", QWEN21_OFFICIAL_EXTRACT_KEY}:
        raise ValueError("Unsupported official Qwen 2.1 Turbo model key.")
    swap = request.workflow == "swap" and request.model_key == "qwen-2.1-turbo-official"
    if not swap and (request.workflow not in {"standard", "text"} or request.swap_kind is not None):
        raise ValueError("This official Qwen request supports only Edit, Combine and Text; no swaps.")
    if swap:
        if request.swap_kind not in {"Head", "Body"} or len(request.images) != 2:
            raise ValueError("Official BFS swap requires Head or Body and exactly two ordered images.")
        profile = swap_profile(request.model_key, request.swap_kind)
        if len(request.loras) != 1:
            raise ValueError("Official swap requires exactly one mandatory BFS LoRA; no optional LoRAs.")
        lora = request.loras[0]
        expected = settings.lora_dir / profile.family / profile.filename
        if (lora.name != profile.filename or lora.path.resolve() != expected.resolve()
                or lora.adapter_name != "bfs_swap"):
            raise ValueError("Official swap requires the exact matching BFS swap profile LoRA.")
        if not math.isfinite(lora.weight) or not 0.1 <= lora.weight <= 1.5:
            raise ValueError("Official BFS weight must be finite and in the positive range 0.1–1.5.")
    if type(request.steps) is not int or request.steps != 8:
        raise ValueError("Official Qwen 2.1 Turbo requires exactly eight saved-sigma steps.")
    if request.guidance != 1 or request.true_cfg != 1:
        raise ValueError("Official Qwen 2.1 Turbo requires guidance 1 and True CFG 1.")
    if request.negative_prompt:
        raise ValueError("Official Qwen 2.1 Turbo requires an empty negative prompt.")
    if request.loras and not swap:
        raise ValueError("Optional LoRAs are unsupported for Official Qwen 2.1 Turbo.")
    if request.dlss is not None:
        raise ValueError("DLSS is unsupported for the official Diffusers pipeline.")
    if request.workflow == "text":
        if request.images:
            raise ValueError("Official text generation does not accept images.")
    elif not swap and not 1 <= len(request.images) <= 3:
        raise ValueError("Official Edit/Combine requires one to three ordered images.")


def validate_qwen21_official_config(path: Path) -> None:
    """Verify required snapshot configuration without allocating model weights."""
    try:
        index = json.loads((path / "model_index.json").read_text(encoding="utf-8"))
        transformer = json.loads((path / "transformer" / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError("Official Qwen snapshot requires model_index.json and transformer/config.json.") from exc
    if index.get("_class_name") != "QwenImage21Pipeline" or index.get("sample_sigmas") != list(QWEN21_OFFICIAL_SIGMAS):
        raise RuntimeError("Official Qwen model_index.json must retain the exact publisher eight sample_sigmas.")
    if transformer.get("causal_condition") is not True:
        raise RuntimeError("Official Qwen transformer requires causal_condition=true for KV caching.")


def _qwen21_bfs_target(key: str) -> tuple[str, str]:
    """Accept only the publisher's image MLP and image-attention targets."""
    name = key.removeprefix("diffusion_model.").removeprefix("transformer.")
    parts = name.split(".", 2)
    suffix = next((s for s in (".lora_A.weight", ".lora_B.weight") if name.endswith(s)), None)
    if (suffix is None or len(parts) != 3 or parts[0] != "transformer_blocks"
            or not parts[1].isascii() or not parts[1].isdigit()
            or str(int(parts[1])) != parts[1]
            or parts[2].removesuffix(suffix) not in {
                "attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0",
                "img_mlp.gate_up", "img_mlp.out",
            }):
        raise ValueError(f"Unknown Qwen BFS target: {key}")
    return name.removesuffix(suffix), suffix


def _validate_qwen21_bfs_header(path: Path, kind: str, transformer: Any) -> None:
    """Validate the complete normalized schema using shapes only, never tensor data."""
    import torch
    from safetensors import safe_open

    groups: dict[str, dict[str, tuple[int, ...]]] = {}
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        if (len(keys) != 2 * QWEN21_BFS_PINS[kind]["pairs"]
                or any(not key.endswith((".lora_A.weight", ".lora_B.weight")) for key in keys)):
            raise ValueError("Official BFS requires the pinned A/B-only matrix pairs.")
        for key in keys:
            module, suffix = _qwen21_bfs_target(key)
            values = groups.setdefault(module, {})
            if suffix in values:
                raise ValueError(f"Unknown or duplicate Qwen LoRA key: {key}")
            tensor = handle.get_slice(key)
            if tensor.get_dtype() not in {"F16", "BF16", "F32", "F64"}:
                raise ValueError(f"Nonfinite or nonfloating Qwen LoRA tensor: {key}")
            values[suffix] = tuple(tensor.get_shape())
    modules = dict(transformer.named_modules())
    for name, values in groups.items():
        if set(values) != {".lora_A.weight", ".lora_B.weight"}:
            raise ValueError(f"Incomplete or mixed Qwen LoRA pair: {name}")
        a, b = values[".lora_A.weight"], values[".lora_B.weight"]
        if len(a) != 2 or len(b) != 2 or min(a) <= 0 or min(b) <= 0 or a[0] != b[1]:
            raise ValueError(f"Invalid Qwen LoRA rank/shapes: {name}")
        targets = [(name, b[0])]
        if name.endswith(".img_mlp.gate_up"):
            if b[0] % 2:
                raise ValueError(f"Invalid fused extracted SwiGLU matrix pair: {name}")
            targets = [(name.removesuffix("gate_up") + target, b[0] // 2)
                       for target in ("gate_layer", "proj")]
        for target, outputs in targets:
            module = modules.get(target)
            base = getattr(module, "base_layer", module)
            if base is None:
                raise ValueError(f"Unknown Qwen transformer parameter: {target}.weight")
            if not isinstance(base, torch.nn.Linear):
                raise ValueError(f"Qwen LoRA requires an unquantized Linear target: {target}")  # noqa: TRY004 -- invalid asset target
            shape = (outputs, a[1])
            if not base.weight.is_floating_point() or tuple(base.weight.shape) != shape:
                raise ValueError(f"Qwen target shape/dtype mismatch: {target}: {shape} vs {tuple(base.weight.shape)}")


def _preflight_qwen21_bfs(path: Path, kind: str, config_path: Path,
                        diffusers: Any, pipeline_class: Any) -> None:
    """Construct only meta parameters; no checkpoint, offload hooks, or large weights."""
    import torch
    from accelerate import init_empty_weights

    config = json.loads(config_path.read_text(encoding="utf-8"))
    transformer_class = getattr(diffusers, "QwenImage21Transformer2DModel", None)
    with init_empty_weights(include_buffers=True), torch.device("meta"):
        if transformer_class is not None:
            transformer = transformer_class.from_config(config)
        else:
            # A self-contained pipeline may supply its own transformer implementation.
            # Inspect that implementation on meta too, never through from_pretrained.
            transformer = pipeline_class(sample_sigmas=list(QWEN21_OFFICIAL_SIGMAS)).transformer
    _validate_qwen21_bfs_header(path, kind, transformer)


class Qwen21OfficialAdapter(DiffusersAdapter):
    pipeline_names = ("QwenImage21Pipeline",)

    def __init__(self, spec: Any) -> None:
        super().__init__(spec)
        self._pending_request: GenerationRequest | None = None

    def load(self) -> None:
        pending = self._pending_request
        if pending is not None and pending.workflow == "swap":
            validate_qwen21_official_request(pending)
            validate_qwen21_bfs_file(pending.loras[0].path, pending.swap_kind)
        diffusers, torch = _runtime()
        try:
            pipeline_class = _pipeline_class(diffusers, *self.pipeline_names)
        except (ImportError, RuntimeError) as exc:
            raise RuntimeError(QWEN21_OFFICIAL_UPGRADE) from exc
        if "sample_sigmas" not in inspect.signature(pipeline_class.__init__).parameters:
            raise RuntimeError(QWEN21_OFFICIAL_UPGRADE)
        required = {
            "prompt", "image", "width", "height", "num_inference_steps", "true_cfg_scale",
            "use_kv_cache", "generator", "num_images_per_prompt", "callback_on_step_end",
            "callback_on_step_end_tensor_inputs",
        }
        if not required.issubset(inspect.signature(pipeline_class.__call__).parameters):
            raise RuntimeError(QWEN21_OFFICIAL_UPGRADE)
        if pending is not None and pending.workflow == "swap" and not all(
            callable(getattr(pipeline_class, name, None))
            for name in ("remove_all_hooks", "unload_lora_weights", "load_lora_weights", "set_adapters")
        ):
            raise RuntimeError("Official BFS requires a LoRA-capable QwenImage21Pipeline. " + QWEN21_OFFICIAL_UPGRADE)
        path = self.spec.local_path
        validate_qwen21_official_config(path)
        if pending is not None and pending.workflow == "swap":
            _preflight_qwen21_bfs(
                pending.loras[0].path, pending.swap_kind,
                path / "transformer" / "config.json", diffusers, pipeline_class,
            )
        self._validate_hardware()
        # The saved pipeline sampling grid is authoritative: never pass sigmas,
        # replace the scheduler, or attach the unrelated six-step Turbo LoRA.
        pipe = pipeline_class.from_pretrained(
            _local_path(path, self.spec.label),
            torch_dtype=torch.bfloat16,
            local_files_only=True,
            low_cpu_mem_usage=True,
        )
        if getattr(pipe.config, "sample_sigmas", None) != list(QWEN21_OFFICIAL_SIGMAS):
            raise RuntimeError("Loaded official Qwen pipeline lost its exact saved sample_sigmas; refusing inference.")
        if getattr(pipe.transformer.config, "causal_condition", None) is not True:
            raise RuntimeError("Loaded official Qwen transformer does not support causal KV caching.")
        self.pipe = pipe
        try:
            self._prepare_before_offload()
            self.configure_memory()
        except Exception:
            self.unload()
            raise

    def _prepare_before_offload(self) -> None:
        """Attach mandatory BFS while checkpoint parameters are materialized."""
        if self._pending_request is not None and self._pending_request.workflow == "swap":
            self._install_bfs(self._pending_request)
            self._lora_signature = self.lora_signature(self._pending_request)

    def _install_bfs(self, request: GenerationRequest) -> None:
        from safetensors import safe_open

        from photo_edit_studio.models.qwen2511_lora import (
            QwenLoraWeights,
            parse_qwen_lora,
            validate_qwen_lora,
        )

        item = request.loras[0]
        with safe_open(str(item.path), framework="pt", device="cpu") as handle:
            keys = list(handle.keys())
        # These pins contain A/B only: no alpha scaling or direct deltas.
        if (len(keys) != 2 * QWEN21_BFS_PINS[request.swap_kind]["pairs"]
                or any(not key.endswith((".lora_A.weight", ".lora_B.weight")) for key in keys)):
            raise ValueError("Official BFS requires the pinned A/B-only matrix pairs.")
        for key in keys:
            _qwen21_bfs_target(key)
        weights = parse_qwen_lora(item.path)
        if weights.deltas:
            raise ValueError("Official BFS must not contain direct deltas.")
        normalized = QwenLoraWeights(split_qwen21_fused_lora(weights.matrices), {})
        validate_qwen_lora(self.pipe.transformer, normalized)
        self.pipe.load_lora_weights(normalized.matrices, adapter_name="bfs_swap", local_files_only=True)
        self.pipe.set_adapters(["bfs_swap"], adapter_weights=[float(item.weight)])

    def apply_loras(self, request: GenerationRequest) -> None:
        validate_qwen21_official_request(request)
        if request.model_key != self.spec.key:
            raise ValueError("Official Qwen request key mismatch.")
        signature = self.lora_signature(request)
        if signature == self._lora_signature:
            return
        try:
            if request.loras:
                validate_qwen21_bfs_file(request.loras[0].path, request.swap_kind)
            if not all(callable(getattr(self.pipe, name, None)) for name in
                       ("remove_all_hooks", "unload_lora_weights", "load_lora_weights", "set_adapters")):
                raise TypeError("Official BFS changes require hook removal and Diffusers LoRA support.")
            self.pipe.remove_all_hooks()
            self.pipe.unload_lora_weights()
            if request.loras:
                self._install_bfs(request)
            self.configure_memory()
            self._lora_signature = signature
        except Exception:
            self.unload()
            raise

    def unload(self) -> None:
        super().unload()
        self._pending_request = None

    def generate(
        self, request: GenerationRequest, progress: GenerationProgress | None = None
    ) -> list[Any]:
        validate_qwen21_official_request(request)
        if request.model_key != self.spec.key:
            raise ValueError("Official Qwen request key does not match the selected adapter.")
        if self.pipe is None:
            if progress is not None:
                progress.update(0.05, f"Loading {self.spec.label}")
            self._pending_request = request
            try:
                self.load()
            finally:
                self._pending_request = None
        if self.spec.key == "qwen-2.1-turbo-official" and self.lora_signature(request) != self._lora_signature:
            self.apply_loras(request)
        if getattr(self.pipe.config, "sample_sigmas", None) != list(QWEN21_OFFICIAL_SIGMAS):
            raise RuntimeError("Official Qwen saved sample_sigmas changed; refusing inference.")
        if getattr(self.pipe.transformer.config, "causal_condition", None) is not True:
            raise RuntimeError("Official Qwen causal KV cache configuration changed; refusing inference.")
        _, torch = _runtime()
        images: list[Any] = []
        for index in range(request.count):
            if progress is not None:
                progress.inference(index, request.count, 0, 8)
            kwargs = {
                "prompt": request.prompt,
                "width": request.width,
                "height": request.height,
                "num_inference_steps": 8,
                "true_cfg_scale": 1.0,
                "use_kv_cache": True,
                "generator": torch.Generator(device="cpu").manual_seed(request.seed + index),
                "num_images_per_prompt": 1,
            }
            if request.workflow != "text":
                kwargs["image"] = request.images
            kwargs.update(_progress_kwargs(self.pipe, progress, index, request.count, steps=8))
            with torch.no_grad():
                # Essential kwargs must never be silently filtered by _supported_call.
                images.extend(_extract_images(self.pipe(**kwargs)))
        return images


class Qwen21OfficialExtractAdapter(Qwen21OfficialAdapter):
    """Explicit experimental stacking on the full official Turbo checkpoint."""

    def load(self) -> None:
        if self.spec.key != QWEN21_OFFICIAL_EXTRACT_KEY:
            raise ValueError("Extracted LoRA requires its dedicated official profile.")
        path = qwen21_official_extract_path()
        # Missing/invalid mandatory weights must fail before checkpoint allocation.
        validate_qwen21_official_extract_lora(path)
        diffusers, _ = _runtime()
        pipeline_class = _pipeline_class(diffusers, *self.pipeline_names)
        if not all(callable(getattr(pipeline_class, name, None))
                   for name in ("load_lora_weights", "set_adapters")):
            raise RuntimeError("Official extracted LoRA requires a LoRA-capable QwenImage21Pipeline. " + QWEN21_OFFICIAL_UPGRADE)
        try:
            super().load()
        except Exception:
            self.unload()
            raise

    def _prepare_before_offload(self) -> None:
        apply_qwen21_extracted_weights(self.pipe, qwen21_official_extract_path())

    def apply_loras(self, request: GenerationRequest) -> None:
        # Parent's official generate does not call this; never let the generic
        # optional-LoRA path unload the mandatory adapter if reused in future.
        validate_qwen21_official_request(request)
        if request.model_key != self.spec.key:
            raise ValueError("Official extracted LoRA request key mismatch.")


class FluxKleinAdapter(DiffusersAdapter):
    pipeline_names = ("Flux2KleinPipeline", "Flux2Pipeline")

    def load(self) -> None:
        self._validate_hardware()
        diffusers, torch = _runtime()
        try:
            from transformers import Qwen3Config, Qwen3ForCausalLM
        except ImportError as exc:
            raise RuntimeError("The installed Transformers build does not support Qwen3.") from exc

        encoder_path = resolve_flux_klein_text_encoder(settings.flux_klein_4b_text_encoder)
        encoder_config_path = self.spec.local_path / "text_encoder"
        # Otherwise Transformers attempts to find gguf_file as the configuration
        # inside text_encoder/. Keep Klein's configuration and external weights separate.
        encoder_config = Qwen3Config.from_pretrained(
            str(encoder_config_path.resolve()), local_files_only=True,
        )
        logger.info("FLUX text encoder load begin")
        try:
            text_encoder = Qwen3ForCausalLM.from_pretrained(
                str(encoder_config_path.resolve()),
                config=encoder_config,
                gguf_file=str(encoder_path),
                dtype=torch.bfloat16,
                local_files_only=True,
                low_cpu_mem_usage=True,
            )
        except ImportError as exc:
            raise RuntimeError(
                "Loading the FLUX.2 Klein GGUF text encoder requires the 'gguf' package. "
                "Rerun scripts/setup.ps1."
            ) from exc
        logger.info("FLUX text encoder load end")
        pipeline_class = _pipeline_class(diffusers, *self.pipeline_names)
        logger.info("FLUX pipeline load begin")
        self.pipe = pipeline_class.from_pretrained(
            _local_path(self.spec.local_path, self.spec.label),
            text_encoder=text_encoder,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
            low_cpu_mem_usage=True,
            requires_safety_checker=False,
            safety_checker=None,
        )
        logger.info("FLUX pipeline load end")
        logger.info("FLUX pipeline offload configuration begin")
        self.configure_memory()
        logger.info("FLUX pipeline offload configuration end")


def resolve_flux_klein_text_encoder(configured_path: Path) -> Path:
    candidates = [configured_path]
    name = configured_path.name
    if "-alb-" in name:
        candidates.append(configured_path.with_name(name.replace("-alb-", "-abl-")))
    elif "-abl-" in name:
        candidates.append(configured_path.with_name(name.replace("-abl-", "-alb-")))
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "FLUX.2 Klein 4B custom text encoder was not found. "
        f"Expected one of: {searched}."
    )


class Krea2Adapter(DiffusersAdapter):
    pipeline_names = ("Krea2Pipeline",)

    def generate(
        self, request: GenerationRequest, progress: GenerationProgress | None = None
    ) -> list[Any]:
        if self.pipe is None:
            if progress is not None:
                progress.update(0.05, f"Loading {self.spec.label} · this may take a few minutes")
            self.load()
        if progress is not None:
            progress.update(0.12, "Preparing LoRAs and prompt")
        self.apply_loras(request)
        _, torch = _runtime()
        images: list[Any] = []
        for index in range(request.count):
            if progress is not None:
                progress.inference(index, request.count, 0, request.steps)
            generator = torch.Generator(device="cpu").manual_seed(request.seed + index)
            kwargs = {
                "prompt": request.prompt,
                "negative_prompt": request.negative_prompt or None,
                "width": request.width,
                "height": request.height,
                "num_inference_steps": request.steps,
                "guidance_scale": request.guidance,
                "generator": generator,
                "num_images_per_prompt": 1,
            }
            kwargs.update(_progress_kwargs(self.pipe, progress, index, request.count, request.steps))
            with torch.no_grad():
                images.extend(_extract_images(_supported_call(self.pipe, kwargs)))
        return images


class QwenAioAdapter(QwenAdapter):
    def load(self) -> None:
        logger.info("Qwen AIO load begin")
        self._validate_hardware()
        diffusers, torch = _runtime()
        base_spec = MODEL_SPECS["qwen-2511"]
        base_path = _local_path(base_spec.local_path, base_spec.label)
        checkpoint = resolve_qwen_aio_checkpoint(self.spec.local_path)
        transformer_class = _pipeline_class(diffusers, "QwenImageTransformer2DModel")
        logger.info("Qwen AIO official transformer load begin")
        transformer = transformer_class.from_pretrained(
            base_path,
            subfolder="transformer",
            torch_dtype=torch.bfloat16,
            local_files_only=True,
            low_cpu_mem_usage=True,
            safety_checker=None,
            requires_safety_checker=False,
        )
        logger.info("Qwen AIO official transformer load end")
        logger.info("Qwen AIO state preparation begin")
        state = transformer_state_from_comfy_aio(checkpoint, torch.bfloat16)
        logger.info("Qwen AIO state preparation end: keys=%d", len(state))
        logger.info("Qwen AIO state apply begin")
        incompatible = transformer.load_state_dict(
            state, strict=False
        )
        del state
        logger.info(
            "Qwen AIO state apply end: missing_keys=%d unexpected_keys=%d",
            len(incompatible.missing_keys),
            len(incompatible.unexpected_keys),
        )
        logger.info("Qwen AIO rope materialization begin")
        materialize_qwen_rope(transformer)
        logger.info("Qwen AIO rope materialization end")
        pipeline_class = _pipeline_class(diffusers, *self.pipeline_names)
        logger.info("Qwen AIO pipeline load begin")
        self.pipe = pipeline_class.from_pretrained(
            base_path,
            transformer=transformer,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
            low_cpu_mem_usage=True,
            safety_checker=None,
            requires_safety_checker=False
        )
        logger.info("Qwen AIO pipeline load end")
        logger.info("Qwen AIO pipeline offload configuration begin")
        self.configure_memory()
        logger.info("Qwen AIO pipeline offload configuration end")
        logger.info("Qwen AIO load end")
