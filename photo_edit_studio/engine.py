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
    diffusion_output_size,
    normalize_image,
    scaled_output_size,
)
from photo_edit_studio.krea_character_sheet import (
    KREA_CHARACTER_SHEET_PROFILES,
    KREA_CHARACTER_SHEET_REVISION,
    KREA_CHARACTER_SHEET_SIZE,
    KREA_CHARACTER_SHEET_SOURCE,
    KREA_DYNAMIC_TEMPLATE_SHA256,
    validate_krea_character_sheet_request,
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
    spec = MODEL_SPECS[request.model_key]
    if spec.family == "qwen21-official":
        from photo_edit_studio.models.diffusers_adapters import validate_qwen21_official_request

        validate_qwen21_official_request(request)
    dlss = validate_dlss_request(request)
    if progress is not None:
        progress.update(0.01, "Preparing request and sizing images")
    qwen_sheet = request.workflow == "qwen-character-sheet"
    if request.model_key == "qwen-2.1-sheet" and not qwen_sheet:
        raise ValueError("qwen-2.1-sheet is only available for qwen-character-sheet.")
    if request.model_key == "krea-2-turbo" and request.workflow == "standard":
        request.workflow = "krea-text"
    if request.workflow in {"text", "krea-text"}:
        if request.model_key not in TEXT_TO_IMAGE_KEYS or (
            request.workflow == "krea-text" and request.model_key != "krea-2-turbo"
        ):
            raise ValueError("Choose a text-capable model for Create from text.")
        if request.images:
            raise ValueError("Create from text does not accept source images.")
    sheet = request.workflow in KREA_CHARACTER_SHEET_PROFILES
    if qwen_sheet:
        from photo_edit_studio.qwen_character_sheet import (
            qwen_character_sheet_size,
            validate_qwen_character_sheet_request,
        )

        if type(request.seed) is int and request.seed < 0:
            request.seed = secrets.randbelow(2**31 - 1)
        validate_qwen_character_sheet_request(request)
        if restoration != "Off":
            raise ValueError("Qwen character sheets require face restoration Off.")
        request.compose = False
        request.width, request.height = qwen_character_sheet_size(request.sheet_megapixels)
        input_max_side = 1536
    elif sheet:
        if type(request.seed) is int and request.seed < 0:
            request.seed = secrets.randbelow(2**31 - 1)
        validate_krea_character_sheet_request(request)
        if restoration != "Off":
            raise ValueError("Krea character sheets require face restoration Off.")
        request.compose = False
        request.width, request.height = KREA_CHARACTER_SHEET_SIZE
        input_max_side = 1024
    elif request.workflow == "krea-all2real":
        from photo_edit_studio.krea_all2real import validate_krea_all2real_request
        from photo_edit_studio.models.comfy_swap import krea_remix_size

        # The original graph needs a concrete seed plus a non-wrapping seed+1.
        # Reject malformed seeds through its validator, not silent coercion.
        if type(request.seed) is int and request.seed < 0:
            request.seed = secrets.randbelow(2**31 - 1)
        validate_krea_all2real_request(request)
        if restoration != "Off":
            raise ValueError("All2Real returns generated output only; face restoration must be Off.")
        request.compose = False
        # This is the uploaded SOURCE budget, not a text/empty-latent canvas.
        # Graph 73 then scales to 1MP and graph 75 applies Flux aspect buckets;
        # the original Wan VAE decodes/upscales between passes and at output.
        request.width, request.height = krea_remix_size(request)
        input_max_side = 1024
    elif request.workflow == "krea-remix":
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
    # Sheet profiles own their explicit 1536x1024 budget, independent of the
    # global 1024 edit cap. Never downsize their empty latent to the source AR.
    if not (sheet or qwen_sheet):
        request.width, request.height = diffusion_output_size(
            requested_size,
            request.size_multiplier,
            family=spec.family,
            workflow=request.workflow,
            canvas=canvas_workflow,
            pre_sized=True,
        )
    if (request.width, request.height) != requested_size and request.workflow not in {"krea-remix", "krea-all2real"}:
        input_max_side = max(request.width, request.height)
    request.count = min(request.count, settings.max_batch_count)
    input_limit = (
        1 if sheet or qwen_sheet or request.workflow in {"krea-remix", "krea-all2real"} else
        2 if request.workflow in {"swap", "krea-reference"} else spec.max_images
    )
    request.images = [
        normalize_image(image, max_side=input_max_side)
        for image in request.images[:input_limit]
    ]
    if request.seed < 0:
        request.seed = secrets.randbelow(2**31 - 1)

    started = time.perf_counter()
    if qwen_sheet:
        from photo_edit_studio.qwen_character_sheet import ComfyQwenCharacterSheetAdapter

        model_manager.unload()
        adapter = ComfyQwenCharacterSheetAdapter(spec)
    elif (sheet or request.workflow in {"swap", "krea-reference", "krea-text", "krea-remix", "krea-all2real"}) and request.model_key == "krea-2-turbo":
        from photo_edit_studio.models.comfy_swap import ComfyKreaSwapAdapter

        model_manager.unload()
        adapter = ComfyKreaSwapAdapter(spec)
    else:
        adapter = model_manager.get(request.model_key)
    images = adapter.generate(request, progress=progress)
    if qwen_sheet:
        if len(images) != 1:
            raise RuntimeError("Qwen character sheets must return exactly one generated output.")
        if dlss is None and images[0].size != requested_size:
            raise RuntimeError("Qwen character-sheet native output differs from canvas; refusing silent resizing.")
    if sheet:
        if len(images) != 1:
            raise RuntimeError("Krea character sheets must return exactly one generated output.")
        if dlss is None and images[0].size != KREA_CHARACTER_SHEET_SIZE:
            raise RuntimeError("Krea character-sheet native output must be 1536x1024; refusing silent resizing.")
    if request.workflow == "krea-all2real" and len(images) != 1:
        raise RuntimeError("All2Real must return exactly one generated output.")
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
    if qwen_sheet:
        notes.append(
            f"Qwen {request.sheet_layout}/{request.sheet_prompt_mode} sheet: full BF16, "
            f"25 steps CFG 1, res_multistep/beta; native {request.width}×{request.height}. "
            "Optional ModelOnly LoRAs precede attention/cache; no mandatory Turbo adapter. "
            "Aspect-preserved source <=1536 replaces KJ total_pixels resizing; both encoders "
            "share that upload. Qwen conditioning resolution=1536 is an area target and may "
            "internally exceed 1536 on the longest side. DeGrid auto runs before optional DLSS. "
            "Auto uses thinking with greedy decoding (publisher sampling adaptation); "
            "static replaces the example Ayaka name with the supplied name or a neutral reference. "
            "Full BF16 at 3.4–6 MP may require substantial host RAM/VRAM; 12 GB is not a fit guarantee."
        )
    elif sheet:
        policy = ("manual structured caption" if request.prompt.strip() else "publisher-template greedy VLM caption") if request.workflow == "krea-dynamic-sheet" else "fixed QuadView trigger plus optional customization"
        notes.append(
            f"{request.workflow}: native diffusion canvas 1536×1024; one aspect-preserved "
            f"source <=1024 ({request.images[0].width}×{request.images[0].height}). "
            "Shared FP8 Krea checkpoint; mandatory first adapter, selected sheet adapter, "
            "then optional adapters. Grounding 0 on both conditioners; pixel-space fit to target latent."
        )
        notes.append(
            f"Prompt policy: {policy}; CFG {request.guidance:g} "
            "(negative conditioning ignored at CFG 1). No identity/Remix/MoreReal or restoration. "
            "Dynamic sheet is experimental; text and view consistency are not guaranteed. "
            "Source aspect preservation and greedy captioning are adaptations of the pinned publisher workflows."
        )
    elif request.workflow == "krea-all2real":
        notes.append(
            f"All2Real source budget capped at 1024: {request.width}×{request.height}; "
            f"actual uploaded source {request.images[0].width}×{request.images[0].height}. "
            "These are not diffusion/output dimensions: the original graph scales to 1MP, "
            "blurs and applies FluxKontext aspect-bucket scaling. No text canvas is selected."
        )
        notes.append(
            "Original INT8 model, FP8 vision encoder, Wan upscale VAE and skin-detail model; "
            "mandatory first LoRA before Ostris, then MoreReal and selected optional LoRAs "
            "on the same model chain for both er_sde passes (kl_optimal then simple). "
            "Refinement removes reference metadata without re-encoding text."
        )
        notes.append(
            "The original Wan VAE decodes with its native upscale between passes and at output; "
            "actual output dimensions change independently of the source budget. Core VAEDecode "
            "equivalence to the upstream VAEUtils tile=False/upscale=-1 wrapper is not established. "
            "Final returned size: " + ", ".join(f"{image.width}×{image.height}" for image in images) + "."
        )
    elif request.workflow == "krea-remix":
        notes.append(
            f"Krea Ostris remix of one uploaded canvas, capped at 1024: {request.width}×{request.height}. "
            "Remix and optional LoRAs first pass only; mandatory first LoRA on both passes; no diffusion upscale."
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
        dlss_input = (
            "after All2Real Wan decode and skin detail (source budget "
            f"{request.width}×{request.height}, not the DLSS input size)"
            if request.workflow == "krea-all2real" else
            f"after diffusion at {request.width}×{request.height}"
        )
        notes.append(
            f"DLSS 5 enhancement ({dlss['upscaling_mode']}) {dlss_input}; final sizes: "
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
    elif spec.family == "qwen21-official":
        notes.append(
            "Official Qwen Image 2.1 Turbo: full BF16, publisher's saved eight-sigma schedule, "
            "True CFG 1 and causal KV cache; no six-step or optional LoRAs. "
            "CPU offload reused; 12 GB VRAM fit and live inference unverified."
        )
        if request.workflow == "swap":
            from photo_edit_studio.swap import QWEN21_BFS_PINS, swap_profile

            profile = swap_profile(request.model_key, request.swap_kind)
            pin = QWEN21_BFS_PINS[request.swap_kind]
            notes.append(
                f"Two-image BFS {request.swap_kind} v{pin['version']}: "
                f"{profile.filename}@{request.loras[0].weight:g}; "
                "Source 1 = target body/scene, Source 2 = replacement reference. "
                f"Training template: {profile.trigger}"
            )
        if spec.loader == "qwen21_official_extract":
            from photo_edit_studio.models.qwen21_official_extract import (
                QWEN21_OFFICIAL_EXTRACT_LORA,
            )

            pin = QWEN21_OFFICIAL_EXTRACT_LORA
            notes.append(
                f"Experimental stack (not original base): official Turbo checkpoint + mandatory "
                f"{pin['filename']}@1; Civitai version {pin['version_id']}, file {pin['file_id']}, "
                f"SHA256 {pin['sha256']}. Compatibility/live inference unverified."
            )
    elif spec.family == "qwen21":
        from photo_edit_studio.comfy_assets import (
            QWEN21_R128_KEY,
            QWEN21_R128_TURBO_LORA,
            QWEN21_TURBO_LORA,
        )

        if request.model_key == QWEN21_R128_KEY:
            notes.append(f"INT8 Qwen Image 2.1; isHeSatoshi Turbo r128 (Civitai compatibility modification by tsolful): {QWEN21_R128_TURBO_LORA}; res_2s_ode, ManualSigmas, six steps, CFG 1; Qwen Research License.")
        else:
            notes.append(f"INT8 Qwen Image 2.1; unmerged Viggle r256 LoRA: {QWEN21_TURBO_LORA}; fixed six-step schedule.")
        if request.workflow == "swap":
            from photo_edit_studio.swap import swap_profile

            notes.append(
                f"Two-image BFS {request.swap_kind} swap with "
                f"{swap_profile(request.model_key, request.swap_kind).filename}; results may vary."
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
        "model_key": request.model_key,
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
    if MODEL_SPECS[request.model_key].family == "qwen21-official":
        from photo_edit_studio.models.diffusers_adapters import QWEN21_OFFICIAL_SIGMAS

        metadata.update(
            sample_sigmas=list(QWEN21_OFFICIAL_SIGMAS),
            actual_output_sizes=[list(image.size) for image in result.images],
            use_kv_cache=True,
            scheduler_source="checkpoint",
        )
        if request.workflow == "swap":
            from photo_edit_studio.swap import QWEN21_BFS_PINS, swap_profile

            profile = swap_profile(request.model_key, request.swap_kind)
            pin = QWEN21_BFS_PINS[request.swap_kind]
            metadata.update(
                mandatory_adapter={
                    "name": "bfs_swap", "filename": profile.filename,
                    "version": pin["version"], "weight": request.loras[0].weight,
                    "sha256": pin["sha256"], "size_bytes": pin["size"],
                },
                ordered_image_sources=["Source 1: target body/scene", "Source 2: replacement reference"],
                swap_trigger=profile.trigger,
            )
        if MODEL_SPECS[request.model_key].loader == "qwen21_official_extract":
            from photo_edit_studio.models.qwen21_official_extract import (
                QWEN21_OFFICIAL_EXTRACT_LORA,
            )

            pin = QWEN21_OFFICIAL_EXTRACT_LORA
            metadata.update(
                experimental_stack=True,
                checkpoint_source="official Turbo (not original base)",
                mandatory_adapter={
                    "name": "official_extract", "filename": pin["filename"], "weight": 1.0,
                    "version_id": pin["version_id"], "file_id": pin["file_id"],
                    "sha256": pin["sha256"], "size_bytes": pin["size"],
                },
            )
    if request.workflow in KREA_CHARACTER_SHEET_PROFILES:
        metadata.update(
            character_sheet_source=KREA_CHARACTER_SHEET_SOURCE,
            character_sheet_revision=KREA_CHARACTER_SHEET_REVISION,
            native_diffusion_size=list(KREA_CHARACTER_SHEET_SIZE),
            actual_output_sizes=[list(image.size) for image in result.images],
            uploaded_source_size=list(request.images[0].size),
            source_aspect_preserved=True,
            source_max_side=1024,
            grounding_px=0,
            fit_mode="fit",
            sheet_adapter=KREA_CHARACTER_SHEET_PROFILES[request.workflow],
            sheet_adapter_weight=next(
                (float(lora.weight) for lora in request.loras
                 if lora.name.casefold() == KREA_CHARACTER_SHEET_PROFILES[request.workflow].casefold()),
                1.0,
            ),
            mandatory_first_lora_weight=request.krea_first_lora_weight,
            sampler="lcm" if request.workflow == "krea-dynamic-sheet" else "euler",
            scheduler="simple",
            negative_conditioning_active=request.guidance != 1,
            prompt_policy=("manual-structured" if request.prompt.strip() else "auto-publisher-vlm-greedy")
            if request.workflow == "krea-dynamic-sheet" else "fixed-trigger-plus-customization",
            experimental=request.workflow == "krea-dynamic-sheet",
        )
        if request.workflow == "krea-dynamic-sheet":
            metadata["caption_template_sha256"] = KREA_DYNAMIC_TEMPLATE_SHA256
            metadata["vlm_sampling_mode"] = None if request.prompt.strip() else "off"
            metadata["vlm_max_length"] = None if request.prompt.strip() else 2048
    if request.workflow == "qwen-character-sheet":
        from photo_edit_studio.qwen_character_sheet import QWEN_CHARACTER_SHEET_SHA256

        metadata.update(
            sheet_layout=request.sheet_layout,
            sheet_prompt_mode=request.sheet_prompt_mode,
            sheet_megapixels=request.sheet_megapixels,
            sheet_archive_sha256=QWEN_CHARACTER_SHEET_SHA256,
            native_diffusion_size=[request.width, request.height],
            actual_output_sizes=[list(image.size) for image in result.images],
            uploaded_source_size=list(request.images[0].size),
            source_max_side=1536,
            source_aspect_preserved=True,
            source_preprocessing="aspect-preserved local upload replaces KJ total_pixels",
            weight_precision="full BF16",
            sampler="res_multistep",
            scheduler="beta",
            denoise=1.0,
            attention="comfy kitchen attention",
            cache_device="auto",
            cache_dtype="default",
            degridding="auto",
            vlm_sampling_mode="off" if request.sheet_prompt_mode == "Auto" else None,
            vlm_thinking=request.sheet_prompt_mode == "Auto",
            vlm_max_length=(2560 if request.sheet_layout == "Production" else 2048)
            if request.sheet_prompt_mode == "Auto" else None,
            negative_conditioning_active=False,
        )
    if request.workflow == "krea-all2real":
        metadata["source_budget"] = [request.width, request.height]
        metadata["uploaded_source_size"] = list(request.images[0].size)
        metadata["graph_source_scaling"] = "1MP then FluxKontext aspect buckets"
        metadata["native_wan_vae_upscale_requested"] = True
        metadata["upstream_vae_wrapper_parity_verified"] = False
        metadata["actual_output_sizes"] = [list(image.size) for image in result.images]
        metadata["width_height_are_source_budget"] = True
    if MODEL_SPECS[request.model_key].family == "qwen21":
        from photo_edit_studio.comfy_assets import (
            QWEN21_R128_FILE_ID,
            QWEN21_R128_KEY,
            QWEN21_R128_SOURCE,
            QWEN21_R128_TURBO_LORA,
            QWEN21_TURBO_LORA,
            QWEN21_VIGGLE_REPO,
        )
        from photo_edit_studio.comfy_workflows import QWEN21_R128_SIGMAS, QWEN21_TURBO_SIGMAS

        r128 = request.model_key == QWEN21_R128_KEY
        metadata["mandatory_turbo_lora"] = QWEN21_R128_TURBO_LORA if r128 else QWEN21_TURBO_LORA
        metadata["mandatory_turbo_weight"] = 1.0
        metadata["sampler"] = "res_2s_ode" if r128 else "euler"
        metadata["sigma_node"] = "ManualSigmas" if r128 else "ViggleTurboSigmas"
        metadata["sigmas"] = QWEN21_R128_SIGMAS if r128 else QWEN21_TURBO_SIGMAS
        metadata["turbo_source"] = QWEN21_R128_SOURCE if r128 else QWEN21_VIGGLE_REPO
        if r128:
            metadata["turbo_file_id"] = QWEN21_R128_FILE_ID
            metadata["turbo_credit"] = "isHeSatoshi; compatibility modification by tsolful"
            metadata["license"] = "Qwen Research License Agreement (non-commercial)"
    if request.dlss is not None and request.dlss["enabled"]:
        metadata["dlss"] = request.dlss.copy()
        if request.workflow == "krea-all2real":
            metadata["dlss_input_stage"] = "after native Wan VAE decode and skin detail"
        else:
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
        "model_key": result.model_key,
        "seed": result.seed,
        "elapsed_seconds": round(result.elapsed_seconds, 2),
        "outputs": [str(path) for path in result.saved_paths],
        "notes": result.notes,
    }
    return json.dumps(payload, indent=2)
