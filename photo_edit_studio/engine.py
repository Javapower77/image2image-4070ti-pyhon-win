from __future__ import annotations

import json
import secrets
import time
from datetime import UTC, datetime
from pathlib import Path

from PIL import PngImagePlugin

from photo_edit_studio.config import settings
from photo_edit_studio.dlss import validate_dlss_request
from photo_edit_studio.image_utils import (
    MAX_OUTPUT_SIDE,
    combine_canvas_size,
    composite_with_mask,
    constrain_output_size,
    normalize_image,
    scaled_output_size,
)
from photo_edit_studio.models import MODEL_SPECS, model_manager
from photo_edit_studio.models.memory import release_cuda
from photo_edit_studio.models.registry import TEXT_TO_IMAGE_KEYS
from photo_edit_studio.progress import GenerationProgress
from photo_edit_studio.restoration import restore_faces
from photo_edit_studio.types import GenerationRequest, GenerationResult


def generate(
    request: GenerationRequest,
    restoration: str,
    restore_weight: float,
    progress: GenerationProgress | None = None,
) -> GenerationResult:
    dlss = validate_dlss_request(request)
    if progress is not None:
        progress.update(0.01, "Preparing request and sizing images")
    spec = MODEL_SPECS[request.model_key]
    if request.model_key == "krea-2-turbo" and request.workflow == "standard":
        request.workflow = "krea-text"
    if request.workflow in {"text", "krea-text"}:
        if request.model_key not in TEXT_TO_IMAGE_KEYS or (
            request.workflow == "krea-text" and request.model_key != "krea-2-turbo"
        ):
            raise ValueError("Choose a text-capable model for Create from text.")
        if request.images:
            raise ValueError("Create from text does not accept source images.")
    if request.workflow == "krea-remix":
        from photo_edit_studio.models.comfy_swap import krea_remix_size, validate_krea_remix_request

        validate_krea_remix_request(request)
        if restoration != "Off":
            raise ValueError("Krea remix returns generated output only; face restoration must be Off.")
        request.compose = False  # Composition already happened in the uploaded canvas.
        request.width, request.height = krea_remix_size(request)
        input_max_side = 1024
    elif request.workflow == "krea-reference":
        if spec.key != "krea-2-turbo" or not 1 <= len(request.images) <= 2:
            raise ValueError("Krea reference edit requires a source and at most one reference.")
        request.compose = False
        request.mask = None
        request.width, request.height = scaled_output_size(
            request.images[0].size, request.size_multiplier
        )
        input_max_side = max(request.width, request.height)
    elif request.workflow == "swap":
        if len(request.images) != 2:
            raise ValueError("Swap requires a body image and a face/person reference image.")
        request.compose = False
        request.mask = None
        request.width, request.height = scaled_output_size(
            request.images[0].size, request.size_multiplier
        )
        input_max_side = max(request.width, request.height)
    elif request.workflow in {"text", "krea-text"} or spec.task == "text-to-image":
        request.compose = True
        request.mask = None
        request.width, request.height = combine_canvas_size(
            request.output_resolution, request.output_aspect_ratio
        )
        input_max_side = MAX_OUTPUT_SIDE
    elif not request.images:
        if request.compose:
            raise ValueError("Upload 1–3 images to combine.")
        raise ValueError("Upload a source image.")
    elif request.compose:
        request.width, request.height = combine_canvas_size(
            request.output_resolution, request.output_aspect_ratio
        )
        request.mask = None
        input_max_side = MAX_OUTPUT_SIDE
    else:
        request.width, request.height = scaled_output_size(
            request.images[0].size, request.size_multiplier
        )
        input_max_side = max(request.width, request.height)
    requested_size = (request.width, request.height)
    canvas_workflow = request.compose
    request.width, request.height = constrain_output_size(
        requested_size,
        max_side=(settings.combine_max_output_side if canvas_workflow else settings.max_output_side),
        max_pixels=(settings.combine_max_output_pixels if canvas_workflow else settings.max_output_pixels),
    )
    if (request.width, request.height) != requested_size and request.workflow != "krea-remix":
        input_max_side = max(request.width, request.height)
    request.count = min(request.count, settings.max_batch_count)
    input_limit = (
        1 if request.workflow == "krea-remix" else
        2 if request.workflow in {"swap", "krea-reference"} else spec.max_images
    )
    request.images = [
        normalize_image(image, max_side=input_max_side)
        for image in request.images[:input_limit]
    ]
    if request.seed < 0:
        request.seed = secrets.randbelow(2**31 - 1)

    started = time.perf_counter()
    if request.workflow in {"swap", "krea-reference", "krea-text", "krea-remix"} and request.model_key == "krea-2-turbo":
        from photo_edit_studio.models.comfy_swap import ComfyKreaSwapAdapter

        model_manager.unload()
        adapter = ComfyKreaSwapAdapter(spec)
    else:
        adapter = model_manager.get(request.model_key)
    images = adapter.generate(request, progress=progress)
    if request.workflow == "krea-remix" and len(images) != 1:
        raise RuntimeError("Krea remix must return exactly one generated output.")
    if progress is not None:
        progress.update(0.87, "Finishing generated images")
    notes: list[str] = []
    if (request.width, request.height) != requested_size:
        notes.append(
            f"Low-VRAM limit reduced {requested_size[0]}×{requested_size[1]} to "
            f"{request.width}×{request.height}."
        )
    if request.workflow == "krea-remix":
        notes.append(
            f"Krea Ostris remix of one uploaded canvas, capped at 1024: {request.width}×{request.height}. "
            "Remix LoRA first pass only; mandatory first LoRA on both passes; no diffusion upscale."
        )
        notes.append(
            "Refinement re-encodes the same prompt without image references: this removes reference_latents "
            "but also vision-derived text context, approximating reference-metadata removal. "
            "Euler kl_optimal then simple schedules are an adapted handoff, not exact upstream parity."
        )
    elif request.workflow == "krea-reference":
        notes.append(f"Krea reference edit with {len(request.images)} source/reference image(s).")
    elif request.workflow == "swap":
        notes.append(f"{request.swap_kind} swap: body/scene first, face/person reference second.")
    elif request.workflow in {"text", "krea-text"} or spec.task == "text-to-image":
        notes.append(
            f"Text-to-image output {request.width}×{request.height} "
            f"({request.output_resolution}, {request.output_aspect_ratio} canvas)."
        )
    elif request.compose:
        notes.append(
            f"Output {request.width}×{request.height} "
            f"({request.output_resolution}, {request.output_aspect_ratio} canvas)."
        )
        notes.append(f"Combined {len(request.images)} reference image(s) into one new picture.")
    else:
        notes.append(
            f"Output {request.width}×{request.height} from image-1 aspect ×{request.size_multiplier}."
        )
    if request.mask is not None:
        # composite_with_mask fits source and mask to generated.size, not the
        # diffusion canvas; never normalize/downsize enhanced output here.
        images = [composite_with_mask(request.images[0], image, request.mask) for image in images]
        notes.append("Mask composited after generation; white areas contain the edit.")
    if restoration != "Off":
        if progress is not None:
            progress.update(0.91, "Restoring faces")
        release_cuda()
        enhanced_sizes = [image.size for image in images]
        images = [restore_faces(image, restoration, restore_weight) for image in images]
        if dlss is not None and [image.size for image in images] != enhanced_sizes:
            raise RuntimeError("Face restoration changed DLSS output dimensions; refusing silent resizing.")
        notes.append(f"Applied {restoration} post-processing.")
        release_cuda()
    if dlss is not None:
        notes.append(
            f"DLSS 5 enhancement ({dlss['upscaling_mode']}) after diffusion at "
            f"{request.width}×{request.height}; final sizes: "
            + ", ".join(f"{image.width}×{image.height}" for image in images)
            + ". Enhancement scaling is separate from the diffusion VRAM budget; "
            "upscaling requires additional runtime/GPU memory."
        )
    if request.loras:
        notes.append(
            "LoRAs: " + ", ".join(f"{spec.name}@{spec.weight:g}" for spec in request.loras)
        )
    if request.model_key == "firered-1.1":
        from photo_edit_studio.comfy_assets import FIRERED_LIGHTNING, FIRERED_TRANSFORMER

        notes.append(f"GGUF Q4_K_M: {FIRERED_TRANSFORMER}; automatic 8-step LoRA: {FIRERED_LIGHTNING}.")
    elif request.model_key == "qwen-2.1-turbo":
        from photo_edit_studio.comfy_assets import QWEN21_TURBO_LORA

        notes.append(f"INT8 Qwen Image 2.1; unmerged Viggle r256 LoRA: {QWEN21_TURBO_LORA}; fixed six-step schedule.")
        if request.workflow == "swap":
            notes.append(
                "Two-image BFS Head swap with Qwen21-BFS_Head_v1.1.safetensors; results may vary."
                if request.swap_kind == "Head" else
                "Two-image BFS Body swap with Qwen21-BFS_Body_v1.1.safetensors; results may vary."
            )

    elapsed = time.perf_counter() - started
    result = GenerationResult(images, request.seed, elapsed, request.model_key, notes=notes)
    if progress is not None:
        progress.update(0.96, "Saving results and metadata locally")
    result.saved_paths = _save(result, request)
    if progress is not None:
        progress.update(1.0, "Complete · results saved locally")
    return result


def _save(result: GenerationResult, request: GenerationRequest) -> list[Path]:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = settings.output_dir / f"{stamp}_{request.seed}"
    run_dir.mkdir(parents=True, exist_ok=False)
    metadata = {
        "model": MODEL_SPECS[request.model_key].repo_id,
        "seed": result.seed,
        "width": request.width,
        "height": request.height,
        "size_multiplier": request.size_multiplier,
        "compose": request.compose,
        "output_resolution": request.output_resolution if request.compose else None,
        "output_aspect_ratio": request.output_aspect_ratio if request.compose else None,
        "steps": request.steps,
        "guidance": request.guidance,
        "true_cfg": request.true_cfg,
        "strength": request.strength,
        "elapsed_seconds": round(result.elapsed_seconds, 3),
        "input_count": len(request.images),
        "workflow": request.workflow,
        "swap_kind": request.swap_kind,
        "loras": [{"name": spec.name, "weight": spec.weight} for spec in request.loras],
        "prompt_stored": False,
    }
    if request.dlss is not None and request.dlss["enabled"]:
        metadata["dlss"] = request.dlss.copy()
        metadata["diffusion_size"] = [request.width, request.height]
        metadata["enhanced_output_sizes"] = [list(image.size) for image in result.images]
        metadata["enhancement_scaling_separate_from_diffusion_budget"] = True
    paths: list[Path] = []
    for index, image in enumerate(result.images, start=1):
        path = run_dir / f"result_{index}.png"
        pnginfo = PngImagePlugin.PngInfo()
        pnginfo.add_text("generation", json.dumps(metadata))
        image.save(path, pnginfo=pnginfo)
        paths.append(path)
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return paths


def metadata_text(result: GenerationResult) -> str:
    payload = {
        "model": MODEL_SPECS[result.model_key].label,
        "seed": result.seed,
        "elapsed_seconds": round(result.elapsed_seconds, 2),
        "outputs": [str(path) for path in result.saved_paths],
        "notes": result.notes,
    }
    return json.dumps(payload, indent=2)
