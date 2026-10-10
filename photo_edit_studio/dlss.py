"""Optional ComfyUI DLSS 5 postprocessing; never imports or installs the runtime."""

from __future__ import annotations

import math
from typing import Any

import httpx

from photo_edit_studio.types import GenerationRequest

# Names, labels and bounds mirror the cloned pack's V3 node schemas.
DLSS_CHOICES = {
    "upscaling_mode": (
        "1x (DLAA / native)", "1.5x (Quality)", "1.724x (Balanced)",
        "2x (Performance)", "3x (Ultra Performance)",
    ),
    "nr_preset": ("Default", "Preset #1", "Preset #2", "Preset #3"),
    "nr_style": ("Default", "Natural", "Cinematic"),
    "dlss_model_preset": ("Default", "J", "K", "L", "M"),
    "motion": ("auto", "optical_flow", "none"),
}
DLSS_DEFAULTS: dict[str, Any] = {
    "enabled": True,
    "upscaling_mode": DLSS_CHOICES["upscaling_mode"][0],
    "nr_preset": "Default",
    "nr_style": "Default",
    "nr_intensity": 1.0,
    "local_tone_strength": 1.0,
    "local_structure_strength": 1.5,
    "skin_structure_strength": 2.0,
    "automatic_mask": True,
    "dlss_model_preset": "M",
    "motion": "auto",
    "scene_change_threshold": 0.24,
    "warmup_frames": 0,
    "runtime_dir": "",
    "verify_neural_rendering": True,
}
_RANGES = {
    "nr_intensity": (0.0, 2.0),
    "local_tone_strength": (0.0, 2.0),
    "local_structure_strength": (0.0, 2.0),
    "skin_structure_strength": (-1.0, 2.0),
    "scene_change_threshold": (0.01, 1.0),
}
_INSTALL_HINT = (
    "Install the Blueforcer/ComfyUI-DLSS5-Enhancer node pack with "
    "scripts/setup-comfy.ps1 and restart ComfyUI (V3 comfy_api.latest required). "
    "The proprietary DLSS 5 Visual Enhancer v3.0 runtime is manual and is not "
    "downloaded by setup: review its licenses and upstream install_runtime.py "
    "instructions yourself. Set runtime_dir to the folder containing nvngx.dll, "
    "or use DLSS5_RUNTIME_DIR/config.json. A supported NVIDIA RTX 30/40/50 GPU, "
    "current driver and Windows D3D12 are required."
)


def validate_dlss_settings(settings: dict[str, Any] | None) -> dict[str, Any] | None:
    """Return a fresh canonical dictionary; None disables, an empty dict enables 1x."""
    if settings is None:
        return None
    if not isinstance(settings, dict):
        raise ValueError("DLSS settings must be a dictionary or None.")  # noqa: TRY004 -- uniform validation API
    unknown = settings.keys() - DLSS_DEFAULTS.keys()
    if unknown:
        raise ValueError(f"Unknown DLSS settings: {', '.join(sorted(map(str, unknown)))}.")
    values = {**DLSS_DEFAULTS, **settings}
    for name in ("enabled", "automatic_mask", "verify_neural_rendering"):
        if not isinstance(values[name], bool):
            raise ValueError(f"DLSS {name} must be a boolean.")  # noqa: TRY004 -- uniform validation API
    for name, choices in DLSS_CHOICES.items():
        if not isinstance(values[name], str) or values[name] not in choices:
            raise ValueError(f"DLSS {name} must be one of: {', '.join(choices)}.")
    for name, (minimum, maximum) in _RANGES.items():
        value = values[name]
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or not minimum <= value <= maximum):
            raise ValueError(f"DLSS {name} must be a finite number between {minimum:g} and {maximum:g}.")
        values[name] = float(value)
    warmup = values["warmup_frames"]
    if isinstance(warmup, bool) or not isinstance(warmup, int) or not 0 <= warmup <= 16:
        raise ValueError("DLSS warmup_frames must be an integer between 0 and 16.")
    runtime = values["runtime_dir"]
    if not isinstance(runtime, str) or "\x00" in runtime:
        raise ValueError("DLSS runtime_dir must be a path string without NUL characters.")
    # This path belongs to the Comfy server, so do not require local existence.
    values["runtime_dir"] = runtime.strip()
    return values


def validate_dlss_request(request: GenerationRequest) -> dict[str, Any] | None:
    """Validate before any image upload, model allocation or backend startup."""
    request.dlss = validate_dlss_settings(request.dlss)
    if request.dlss is None or not request.dlss["enabled"]:
        return None
    from photo_edit_studio.models.registry import MODEL_SPECS

    spec = MODEL_SPECS.get(request.model_key)
    supported = (
        request.model_key == "qwen-2.1-sheet" and request.workflow == "qwen-character-sheet"
        or
        spec is not None and spec.family == "qwen21"
        and request.workflow in {"standard", "text", "swap"}
        or request.model_key == "firered-1.1" and request.workflow == "standard"
        or request.model_key == "krea-2-turbo"
        and request.workflow in {"standard", "krea-text", "krea-reference", "krea-remix", "krea-all2real", "krea-quadview", "krea-dynamic-sheet", "swap"}
    )
    if not supported:
        raise ValueError(
            "DLSS enhancement supports only current ComfyUI Qwen 2.1, FireRed and "
            "Krea reference/remix/All2Real/character-sheet/text/swap workflows; it is not supported by Diffusers. "
            "Disable DLSS (dlss=None) or select a supported ComfyUI workflow."
        )
    return request.dlss


def add_dlss_nodes(graph: dict[str, Any], request: GenerationRequest) -> dict[str, Any]:
    """Splice IMAGE -> settings/enhancer -> SaveImage 29 in place, preserving its ID."""
    values = validate_dlss_request(request)
    if values is None:
        return graph
    saver = graph.get("29")
    if not isinstance(saver, dict) or saver.get("class_type") != "SaveImage":
        raise ValueError("DLSS requires the existing SaveImage node 29.")
    source = saver.get("inputs", {}).get("images")
    if (not isinstance(source, list) or len(source) != 2
            or str(source[0]) not in graph):
        raise ValueError("DLSS requires SaveImage 29 to have a connected IMAGE source.")
    if any(node.get("class_type") in {"DLSS5Settings", "DLSS5EnhanceImages"}
           for node in graph.values()):
        raise ValueError("DLSS nodes already exist in this graph; refusing duplicate enhancement.")
    next_id = max((int(key) for key in graph if key.isdecimal()), default=0) + 1
    settings_id, enhance_id = str(next_id), str(next_id + 1)
    graph[settings_id] = {
        "class_type": "DLSS5Settings",
        "inputs": {key: value for key, value in values.items()
                   if key not in {"enabled", "verify_neural_rendering"}},
    }
    graph[enhance_id] = {
        "class_type": "DLSS5EnhanceImages",
        "inputs": {"images": source.copy(), "settings": [settings_id, 0],
                   "verify_neural_rendering": values["verify_neural_rendering"]},
    }
    saver["inputs"]["images"] = [enhance_id, 0]
    return graph


def preflight_dlss(client: httpx.Client, request: GenerationRequest) -> None:
    """Check node registration only, without loading/executing proprietary binaries."""
    if validate_dlss_request(request) is None:
        return
    try:
        response = client.get("/object_info")
        response.raise_for_status()
        registered = response.json()
        if not isinstance(registered, dict):
            raise ValueError("Expected a node registry dictionary.")  # noqa: TRY004 -- invalid API payload
    except (httpx.HTTPError, ValueError) as exc:
        raise RuntimeError(f"Cannot verify DLSS nodes via ComfyUI /object_info. {_INSTALL_HINT}") from exc
    missing = {"DLSS5Settings", "DLSS5EnhanceImages"} - registered.keys()
    if missing:
        raise RuntimeError(f"ComfyUI is missing DLSS nodes: {', '.join(sorted(missing))}. {_INSTALL_HINT}")