"""Original-assets All2Real API graph, independent of UI/backend integration.

Adapted from workflows/Krea2-all2real.json: the exposed integer 9 is a seed,
not a step count. Both passes share the complete adapter chain and total steps.
StudioAll2RealVAEDecode unpacks the original Wan VAE's normalized packed RGB12
via pixel shuffle, using a copy-only tiled allocation fix for OOM fallback.
Full VAEUtils wrapper equivalence has NOT been established. No replacement
VAE, identity or Remix adapter is used.
"""

from __future__ import annotations

import math
import uuid
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from photo_edit_studio.types import GenerationRequest

KREA_ALL2REAL_MODEL = "krea2_turbo_int8_convrot-b19a4f0be264.safetensors"
KREA_ALL2REAL_ENCODER = "qwen3vl_4b_fp8_scaled.safetensors"
KREA_ALL2REAL_VAE = "Wan2.1_VAE_upscale2x_imageonly_real_v1.safetensors"
KREA_ALL2REAL_SKIN_MODEL = "1x-ITF-SkinDiffDetail-Lite-v1.pth"
KREA_ALL2REAL_FIRST_LORA = "Krea2_ALWAYS_LOAD_FIRST.safetensors"
KREA_ALL2REAL_MORE_REAL = "Krea2-MoreReal.safetensors"
KREA_ALL2REAL_FILES = (
    f"diffusion_models/{KREA_ALL2REAL_MODEL}",
    f"text_encoders/{KREA_ALL2REAL_ENCODER}",
    f"vae/{KREA_ALL2REAL_VAE}",
    f"upscale_models/{KREA_ALL2REAL_SKIN_MODEL}",
    f"loras/{KREA_ALL2REAL_FIRST_LORA}",
)


def krea_all2real_template() -> dict[str, Any]:
    """Return a fresh API template; configure adds the mandatory first LoRA.

    MoreReal initially uses its basename, matching the shared Krea LoRA helpers.
    Before queueing, the caller must resolve registered names and verify that
    this basename identifies models/loras/krea2/Krea2-MoreReal.safetensors, not a
    different same-named vendor file (qualified krea2/ names may be necessary).
    """
    return {
        "55": {"class_type": "UNETLoader", "inputs": {
            "unet_name": KREA_ALL2REAL_MODEL, "weight_dtype": "default",
        }},
        "56": {"class_type": "CLIPLoader", "inputs": {
            "clip_name": KREA_ALL2REAL_ENCODER, "type": "krea2", "device": "default",
        }},
        "57": {"class_type": "VAELoader", "inputs": {"vae_name": KREA_ALL2REAL_VAE}},
        "72": {"class_type": "LoadImage", "inputs": {"image": "source.png"}},
        "73": {"class_type": "ImageScaleToTotalPixels", "inputs": {
            "image": ["72", 0], "upscale_method": "bicubic", "megapixels": 1.0,
            "resolution_steps": 16,
        }},
        "74": {"class_type": "ImageBlur", "inputs": {
            "image": ["73", 0], "blur_radius": 1, "sigma": 1.0,
        }},
        "75": {"class_type": "FluxKontextImageScale", "inputs": {"image": ["74", 0]}},
        "79": {"class_type": "Krea2OstrisEditModelPatch", "inputs": {
            "model": ["55", 0], "kv_cache": True,
        }},
        "71": {"class_type": "LoraLoaderModelOnly", "inputs": {
            "model": ["79", 0], "lora_name": KREA_ALL2REAL_MORE_REAL, "strength_model": 0.9,
        }},
        "82": {"class_type": "VAEEncode", "inputs": {"pixels": ["75", 0], "vae": ["57", 0]}},
        "84": {"class_type": "TextEncodeKrea2OstrisEdit", "inputs": {
            "clip": ["56", 0], "vae": ["57", 0], "image1": ["75", 0],
            "prompt": "photorealistic",
        }},
        "85": {"class_type": "FluxKontextMultiReferenceLatentMethod", "inputs": {
            "conditioning": ["84", 0], "reference_latents_method": "index_timestep_zero",
        }},
        "86": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["85", 0]}},
        "87": {"class_type": "StudioAll2RealRemoveReferences", "inputs": {
            "conditioning": ["85", 0],
        }},
        "53": {"class_type": "KSamplerAdvanced", "inputs": {
            "model": ["71", 0], "positive": ["85", 0], "negative": ["86", 0],
            "latent_image": ["82", 0], "add_noise": "enable", "noise_seed": 9,
            "steps": 11, "cfg": 1.0, "sampler_name": "er_sde", "scheduler": "kl_optimal",
            "start_at_step": 1, "end_at_step": 8, "return_with_leftover_noise": "disable",
        }},
        "88": {"class_type": "StudioAll2RealVAEDecode", "inputs": {"samples": ["53", 0], "vae": ["57", 0]}},
        "89": {"class_type": "VAEEncode", "inputs": {"pixels": ["88", 0], "vae": ["57", 0]}},
        "90": {"class_type": "StudioAll2RealLatentNoise", "inputs": {
            "samples": ["89", 0], "noise_std": 0.3, "seed": 10,
        }},
        "54": {"class_type": "KSamplerAdvanced", "inputs": {
            "model": ["71", 0], "positive": ["87", 0], "negative": ["86", 0],
            "latent_image": ["90", 0], "add_noise": "enable", "noise_seed": 9,
            "steps": 11, "cfg": 1.0, "sampler_name": "er_sde", "scheduler": "simple",
            "start_at_step": 9, "end_at_step": 11, "return_with_leftover_noise": "disable",
        }},
        "91": {"class_type": "StudioAll2RealVAEDecode", "inputs": {"samples": ["54", 0], "vae": ["57", 0]}},
        "92": {"class_type": "UpscaleModelLoader", "inputs": {
            "model_name": KREA_ALL2REAL_SKIN_MODEL,
        }},
        "93": {"class_type": "StudioAll2RealSkinDetail", "inputs": {
            "image": ["91", 0], "upscale_model": ["92", 0],
        }},
        "29": {"class_type": "SaveImage", "inputs": {
            "images": ["93", 0], "filename_prefix": "photo_edit_krea_all2real",
        }},
    }


def validate_krea_all2real_request(req: GenerationRequest) -> None:
    """Reject unsupported inputs before uploads or workflow mutation."""
    from photo_edit_studio.config import settings
    from photo_edit_studio.models.comfy_swap import _validate_krea_loras

    if req.model_key != "krea-2-turbo" or req.workflow != "krea-all2real":
        raise ValueError("All2Real requires model_key='krea-2-turbo' and workflow='krea-all2real'.")
    if len(req.images) != 1 or req.mask is not None or req.count != 1:
        raise ValueError("All2Real requires exactly one source, no mask and count=1.")
    if req.swap_kind is not None:
        raise ValueError("All2Real is not an identity swap workflow.")
    if req.size_multiplier != 1:
        raise ValueError("All2Real uses the original Wan VAE upscale only; size_multiplier must be 1.")
    if type(req.steps) is not int or not 5 <= req.steps <= 40:
        raise ValueError("All2Real total steps must be between 5 and 40 (default 11).")
    # Leave room for the explicit, non-wrapping seed+1 used by latent injection.
    if type(req.seed) is not int or not 0 <= req.seed < 2**64 - 1:
        raise ValueError("All2Real seed must be an integer between 0 and 2**64-2.")
    if req.guidance != 1 or req.negative_prompt.strip():
        raise ValueError("All2Real requires guidance=1 and no negative prompt (ConditioningZeroOut).")
    if not math.isfinite(req.krea_first_lora_weight) or not 0 < req.krea_first_lora_weight <= 2:
        raise ValueError("Mandatory Krea LoRA weight must be finite, greater than 0 and at most 2.")
    if len(req.loras) > 5:
        raise ValueError("All2Real supports at most five optional LoRA selections.")
    _validate_krea_loras(req.loras)
    more_real = settings.lora_dir / "krea2" / KREA_ALL2REAL_MORE_REAL
    if not more_real.is_file():
        raise FileNotFoundError(f"Required All2Real MoreReal LoRA is missing: {more_real}")
    # preserve_identity is a general UI default, not an instruction to add an
    # identity adapter here. Width/height/strength do not replace the source path.


def configure_krea_all2real_graph(
    graph: dict[str, Any], req: GenerationRequest, image_names: tuple[str, ...]
) -> dict[str, Any]:
    """Configure a fresh template in place, committing only after validation.

    Selecting MoreReal at a nonzero optional weight adjusts its required node,
    exactly like reference edit; a zero-weight slot is inactive and retains .9.
    The mandatory adapter is inserted BEFORE Ostris, then MoreReal and optionals
    are applied to BOTH samplers. No text re-encode occurs between passes.
    """
    from photo_edit_studio.models.comfy_swap import apply_krea_loras, apply_mandatory_krea_lora

    validate_krea_all2real_request(req)
    if len(image_names) != 1 or not isinstance(image_names[0], str) or not image_names[0].strip():
        raise ValueError("All2Real needs exactly one nonempty uploaded image name.")
    # Fail closed: a substituted asset or changed processing link must not be
    # silently accepted as this original-assets workflow. Start from a fresh
    # template for each request; this also prevents duplicate mandatory adapters.
    if graph != krea_all2real_template():
        raise ValueError("All2Real requires a fresh, unmodified krea_all2real_template().")
    configured = deepcopy(graph)
    configured["72"]["inputs"]["image"] = image_names[0]
    configured["84"]["inputs"]["prompt"] = req.prompt
    for node_id in ("53", "54"):
        configured[node_id]["inputs"].update(noise_seed=req.seed, steps=req.steps)
    configured["53"]["inputs"]["end_at_step"] = req.steps - 3
    configured["54"]["inputs"].update(start_at_step=req.steps - 2, end_at_step=req.steps)
    configured["90"]["inputs"]["seed"] = req.seed + 1
    configured["29"]["inputs"]["filename_prefix"] = f"photo_edit_krea_all2real_{uuid.uuid4().hex}"
    apply_mandatory_krea_lora(configured, req)
    apply_krea_loras(configured, "71", req.loras)
    graph.clear()
    graph.update(configured)
    return graph


def missing_krea_all2real_assets(root: str | Path) -> list[str]:
    """Check exact originals under a ComfyUI root and the project MoreReal file.

    No downloads or fallback assets. The helper's shared project settings locate
    MoreReal, consistent with _validate_krea_loras and ComfyUI extra model paths.
    Ostris/core node availability must additionally be checked via object_info.
    """
    from photo_edit_studio.config import settings

    base = Path(root)
    paths = [base / "models" / file for file in KREA_ALL2REAL_FILES]
    paths.append(settings.lora_dir / "krea2" / KREA_ALL2REAL_MORE_REAL)
    paths.append(base / "custom_nodes" / "photo_edit_all2real.py")
    return [str(path) for path in paths if not path.is_file()]