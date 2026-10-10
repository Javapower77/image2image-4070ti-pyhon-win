"""Pinned CharacterSheet API graphs, using the studio's shared FP8 Krea assets.

Adaptations: preserve source aspect at <=1024 rather than KJ's square resize;
ground both conditioners at native source size; prime pixel-space fit against
the fixed 1536x1024 target. Dynamic captions use greedy core TextGenerate rather
than the publisher's sampled mode. No identity, Remix, or MoreReal adapter.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

from photo_edit_studio.comfy_assets import (
    COMFY_KREA_FILES,
    KREA_EDIT_FILE,
    KREA_FIRST_LORA_FILE,
    KREA_REMIX_LORA_FILE,
)
from photo_edit_studio.config import settings

if TYPE_CHECKING:
    from photo_edit_studio.types import GenerationRequest

KREA_CHARACTER_SHEET_SOURCE = "https://huggingface.co/Alissonerdx/CharacterSheet"
KREA_CHARACTER_SHEET_REVISION = "3dc4295163dacc924d213168d67bf16850fd954f"
KREA_CHARACTER_SHEET_PROFILES = {
    "krea-quadview": "QuadView_krea2_v1.safetensors",
    "krea-dynamic-sheet": "DynamicCharacterSheet_krea2_v1.safetensors",
}
KREA_CHARACTER_SHEET_SIZE = (1536, 1024)
KREA_CHARACTER_SHEET_MAX_PIXELS = 1536 * 1024
KREA_DYNAMIC_TEMPLATE_SHA256 = "d144cda8af10c5c6e3fe8472d541f619a751f597cba298d581e23dbe5901b706"
KREA_QUADVIEW_TRIGGER = (
    "Convert the character in the image to a Character Sheet showing a face close-up, "
    "front full body, side full body and back full body views"
)
KREA_DYNAMIC_NEGATIVE = "imperfect text, bad text, blur"
KREA_CHARACTER_SHEET_ASSET_HINT = (
    "Install the full krea-character-sheets download preset (shared FP8 checkpoint, "
    "qwen3vl_4b_fp8_scaled encoder, qwen_image_vae, mandatory first adapter, selected "
    "sheet adapter and pinned Dynamic workflow). Each sheet adapter is about 0.85 GiB. "
    "No automatic downloads or INT8/identity/Remix/MoreReal substitutions."
)


def _profile(workflow: str) -> str:
    try:
        return KREA_CHARACTER_SHEET_PROFILES[workflow]
    except KeyError as exc:
        raise ValueError(f"Unknown Krea character-sheet workflow: {workflow}") from exc


def dynamic_template_path() -> Path:
    return settings.model_dir / "workflows" / "charactersheet" / "DynamicCharacterSheet_krea2_v1.json"


def load_dynamic_caption_template() -> str:
    """Read the full publisher template verbatim, only after verifying raw bytes."""
    path = dynamic_template_path()
    if not path.is_file():
        raise FileNotFoundError(f"Missing pinned Dynamic caption workflow: {path}. {KREA_CHARACTER_SHEET_ASSET_HINT}")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != KREA_DYNAMIC_TEMPLATE_SHA256:
        raise ValueError(
            f"Dynamic caption workflow SHA256 mismatch: {path}; require revision "
            f"{KREA_CHARACTER_SHEET_REVISION} (unaltered bytes)."
        )
    nodes = json.loads(raw)["nodes"]
    matches = [node for node in nodes if node.get("id") == 184
               and node.get("type") == "PrimitiveStringMultiline"]
    if len(matches) != 1:
        raise ValueError("Pinned Dynamic workflow must contain caption-template node 184.")
    template = matches[0]["widgets_values"][0]
    if not isinstance(template, str) or not template.strip():
        raise ValueError("Pinned Dynamic caption template is empty or invalid.")
    return template


def missing_krea_character_sheet_assets(root: str | Path, workflow: str) -> list[str]:
    """Check shared standard weights, not the identity-edit asset bundle."""
    base = Path(root) / "models"
    paths = [base / name for name in COMFY_KREA_FILES]
    paths += [base / "loras" / KREA_FIRST_LORA_FILE,
              settings.lora_dir / "krea2" / _profile(workflow)]
    if workflow == "krea-dynamic-sheet":
        paths.append(dynamic_template_path())
    return [str(path) for path in paths if not path.is_file()]


def validate_krea_character_sheet_request(req: GenerationRequest) -> None:
    from photo_edit_studio.models.comfy_swap import _validate_krea_loras

    selected = _profile(req.workflow)
    if req.model_key != "krea-2-turbo":
        raise ValueError("Krea character sheets require model_key='krea-2-turbo'.")
    if len(req.images) != 1 or req.mask is not None or req.count != 1:
        raise ValueError("Krea character sheets require exactly one source, no mask and count=1.")
    if req.swap_kind is not None or req.size_multiplier != 1:
        raise ValueError("Krea character sheets are not identity swaps; size_multiplier must be 1.")
    if type(req.steps) is not int or not 4 <= req.steps <= 40:
        raise ValueError("Krea character-sheet steps must be 4–40 (publisher default 10).")
    if type(req.seed) is not int or not 0 <= req.seed < 2**64:
        raise ValueError("Krea character-sheet seed must be an integer between 0 and 2**64-1.")
    if not math.isfinite(req.guidance) or req.guidance <= 0:
        raise ValueError("Krea character-sheet CFG must be finite and positive (default 1).")
    if not math.isfinite(req.krea_first_lora_weight) or not 0 < req.krea_first_lora_weight <= 2:
        raise ValueError("Mandatory Krea LoRA weight must be finite, greater than 0 and at most 2.")
    if len(req.loras) > 5:
        raise ValueError("Krea character sheets support at most five optional LoRA selections.")
    forbidden = {name.casefold() for name in KREA_CHARACTER_SHEET_PROFILES.values()
                 if name != selected}
    forbidden.update(name.casefold() for name in (
        KREA_EDIT_FILE, KREA_REMIX_LORA_FILE, "Krea2-MoreReal.safetensors",
    ))
    if any(lora.name.casefold() in forbidden for lora in req.loras):
        raise ValueError("Do not combine sheet profiles or add identity-edit, Remix or MoreReal adapters.")
    _validate_krea_loras(req.loras)
    if req.workflow == "krea-dynamic-sheet" and req.prompt.strip():
        prompt = req.prompt.strip()
        headers = ("[TASK: ENTITY_SHEET_GENERATION]", "[TEMPLATE: MULTI_ANGLE_ENTITY_SHEET_V1]")
        if (any(header not in prompt for header in headers)
                or any(not re.search(rf"\[{field}:\s*[^\]\s][^\]]*\]", prompt)
                       for field in ("ENTITY_TYPE", "ENTITY_ID"))):
            raise ValueError(
                "Dynamic manual prompt requires [TASK: ENTITY_SHEET_GENERATION], "
                "[TEMPLATE: MULTI_ANGLE_ENTITY_SHEET_V1], [ENTITY_TYPE: ...] and "
                "[ENTITY_ID: ...] headers. Leave prompt blank for publisher-template VLM captioning."
            )


def krea_character_sheet_template(workflow: str = "krea-quadview") -> dict[str, Any]:
    """Fresh native-canvas API graph; node 72 is already aspect-normalized on upload."""
    selected = _profile(workflow)
    width, height = KREA_CHARACTER_SHEET_SIZE
    if width > 1536 or height > 1024 or width * height > KREA_CHARACTER_SHEET_MAX_PIXELS:
        raise ValueError("Character-sheet diffusion canvas exceeds its explicit 1536x1024 budget.")
    return {
        "55": {"class_type": "UNETLoader", "inputs": {
            "unet_name": COMFY_KREA_FILES[0].rsplit("/", 1)[-1], "weight_dtype": "default",
        }},
        "56": {"class_type": "CLIPLoader", "inputs": {
            "clip_name": COMFY_KREA_FILES[1].rsplit("/", 1)[-1], "type": "krea2", "device": "default",
        }},
        "57": {"class_type": "VAELoader", "inputs": {"vae_name": COMFY_KREA_FILES[2].rsplit("/", 1)[-1]}},
        "72": {"class_type": "LoadImage", "inputs": {"image": "source.png"}},
        "73": {"class_type": "VAEEncode", "inputs": {"pixels": ["72", 0], "vae": ["57", 0]}},
        "71": {"class_type": "LoraLoaderModelOnly", "inputs": {
            "model": ["55", 0], "lora_name": selected, "strength_model": 1.0,
        }},
        "82": {"class_type": "EmptySD3LatentImage", "inputs": {
            "width": width, "height": height, "batch_size": 1,
        }},
        "120": {"class_type": "Krea2EditModelPatch", "inputs": {
            "model": ["71", 0], "source_latent": ["73", 0], "vae": ["57", 0],
            "source_image": ["72", 0], "target_latent": ["82", 0],
            "ref_boost": 1.0, "ref_boost_a": 1.0, "fit_mode": "fit",
        }},
        "84": {"class_type": "Krea2EditGroundedEncode", "inputs": {
            "clip": ["56", 0], "image": ["72", 0], "prompt": KREA_QUADVIEW_TRIGGER,
            "grounding_px": 0,
        }},
        "85": {"class_type": "Krea2EditGroundedEncode", "inputs": {
            "clip": ["56", 0], "image": ["72", 0], "prompt": "",
            "grounding_px": 0,
        }},
        "53": {"class_type": "KSampler", "inputs": {
            "model": ["120", 0], "positive": ["84", 0], "negative": ["85", 0],
            "latent_image": ["82", 0], "seed": 0, "steps": 10, "cfg": 1.0,
            "sampler_name": "lcm" if workflow == "krea-dynamic-sheet" else "euler",
            "scheduler": "simple", "denoise": 1.0,
        }},
        "54": {"class_type": "VAEDecode", "inputs": {"samples": ["53", 0], "vae": ["57", 0]}},
        "29": {"class_type": "SaveImage", "inputs": {
            "images": ["54", 0], "filename_prefix": "photo_edit_krea_sheet",
        }},
    }


def configure_krea_character_sheet_graph(
    graph: dict[str, Any], req: GenerationRequest, image_names: tuple[str, ...]
) -> dict[str, Any]:
    from photo_edit_studio.models.comfy_swap import apply_krea_loras, apply_mandatory_krea_lora

    validate_krea_character_sheet_request(req)
    if len(image_names) != 1 or not isinstance(image_names[0], str) or not image_names[0].strip():
        raise ValueError("Krea character sheets need exactly one nonempty uploaded image name.")
    if graph != krea_character_sheet_template(req.workflow):
        raise ValueError("Krea character sheets require a fresh, unmodified API template.")
    graph["72"]["inputs"]["image"] = image_names[0]
    graph["53"]["inputs"].update(seed=req.seed, steps=req.steps, cfg=req.guidance)
    if req.workflow == "krea-dynamic-sheet":
        # Validate the pinned asset even for manual bypass: the profile owns it.
        template = load_dynamic_caption_template()
        graph["85"]["inputs"]["prompt"] = req.negative_prompt.strip() or KREA_DYNAMIC_NEGATIVE
        if req.prompt.strip():
            graph["84"]["inputs"]["prompt"] = req.prompt.strip()
        else:
            graph["160"] = {"class_type": "TextGenerate", "inputs": {
                "clip": ["56", 0], "image": ["72", 0], "prompt": template,
                "max_length": 2048, "sampling_mode": "off",
                "thinking": False, "use_default_template": True,
            }}
            graph["84"]["inputs"]["prompt"] = ["160", 0]
    else:
        graph["84"]["inputs"]["prompt"] = (
            KREA_QUADVIEW_TRIGGER + ("\n" + req.prompt.strip() if req.prompt.strip() else "")
        )
        graph["85"]["inputs"]["prompt"] = req.negative_prompt
    graph["29"]["inputs"]["filename_prefix"] = f"photo_edit_{req.workflow.replace('-', '_')}_{uuid.uuid4().hex}"
    apply_mandatory_krea_lora(graph, req)
    apply_krea_loras(graph, "71", req.loras)
    # An explicitly selected required adapter adjusts its existing node even
    # at zero; it is never duplicated in an optional chain.
    for lora in req.loras:
        if lora.name.casefold() == _profile(req.workflow).casefold():
            graph["71"]["inputs"]["strength_model"] = float(lora.weight)
    # Bind the selected sheet and optionals to the project library, not same-named
    # vendor files. The mandatory adapter remains in the vendor LoRA directory.
    for node in graph.values():
        if (node["class_type"] == "LoraLoaderModelOnly"
                and node["inputs"]["lora_name"] != KREA_FIRST_LORA_FILE):
            node["inputs"]["lora_name"] = "krea2/" + node["inputs"]["lora_name"]
    return graph