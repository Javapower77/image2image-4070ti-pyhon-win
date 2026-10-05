from __future__ import annotations

import inspect
import logging
from pathlib import Path
from typing import Any

from photo_edit_studio.config import settings
from photo_edit_studio.models.base import ModelAdapter
from photo_edit_studio.models.memory import cuda_available, cuda_vram_gb
from photo_edit_studio.models.qwen_aio import (
    materialize_qwen_rope,
    resolve_qwen_aio_checkpoint,
    transformer_state_from_comfy_aio,
)
from photo_edit_studio.models.registry import MODEL_SPECS
from photo_edit_studio.progress import GenerationProgress
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
    pipe: Any, progress: GenerationProgress | None, image_index: int, count: int, steps: int
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
