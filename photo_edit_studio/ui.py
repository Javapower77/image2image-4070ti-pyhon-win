from __future__ import annotations

import queue
import re
import threading
import time
from datetime import UTC, datetime
from typing import Any

import gradio as gr
from PIL import Image

from photo_edit_studio.config import ROOT, settings
from photo_edit_studio.engine import generate, metadata_text
from photo_edit_studio.image_utils import combine_canvas_size, scaled_output_size
from photo_edit_studio.loras import (
    MAX_LORAS,
    NONE_CHOICE,
    assign_new_slots,
    import_lora_files,
    library_status,
    lora_choices,
    selected_loras,
)
from photo_edit_studio.models import MODEL_SPECS, model_manager
from photo_edit_studio.models.registry import TEXT_TO_IMAGE_KEYS
from photo_edit_studio.progress import GenerationProgress
from photo_edit_studio.swap import SWAP_PROFILES, swap_lora, swap_profile
from photo_edit_studio.types import GenerationRequest

EDIT_MODE = "Edit source"
COMBINE_MODE = "Combine images"
TEXT_MODE = "Create from text"
KREA_EDIT_MODE = "Krea reference edit"
SWAP_MODE = "Swap head / body"
EDIT_PLACEHOLDER = (
    "Replace the background with a softly lit Paris café; preserve the "
    "person's exact identity, pose, and skin texture."
)
COMBINE_PLACEHOLDER = (
    "Create a new picture: the person from image 1 wearing the outfit from "
    "image 2, standing in the scene from image 3."
)
TEXT_PLACEHOLDER = (
    "A cinematic photograph of a small red fox walking through fresh snow at golden hour, "
    "soft atmospheric light, detailed fur, natural colors."
)
SWAP_PLACEHOLDER = "Preserve the target scene, body pose, lighting, and natural skin texture."

WORKFLOW_DESCRIPTIONS = {
    EDIT_MODE: "Edit your source photograph while keeping its framing. Add references or an optional mask.",
    COMBINE_MODE: "Blend up to three images into one new composition. Name image 1, 2, and 3 in your prompt.",
    TEXT_MODE: "Describe the picture you want to create; no source photograph is needed.",
    KREA_EDIT_MODE: "Edit a Krea source with an optional second reference through local ComfyUI.",
    SWAP_MODE: "Use the target scene as Picture 1 and the donor as Picture 2. Only use images with consent.",
}
UI_ASSETS = ROOT / "photo_edit_studio" / "assets" / "ui"
WORKFLOW_ICONS = {
    EDIT_MODE: "edit.svg", COMBINE_MODE: "combine.svg", TEXT_MODE: "text.svg",
    KREA_EDIT_MODE: "krea.svg", SWAP_MODE: "swap.svg",
}


def _release_badge() -> str:
    changelog = ROOT / "CHANGELOG.md"
    if not changelog.is_file():
        return "Beta"
    text = changelog.read_text(encoding="utf-8")
    match = re.search(r"^## \[(\d+\.\d+\.\d+)\]", text, re.MULTILINE)
    version = match.group(1) if match else "Unreleased"
    modified = datetime.fromtimestamp(changelog.stat().st_mtime, tz=UTC).strftime("%Y-%m-%d")
    return f"Beta · {version} · {modified}"


def _mode_description(mode: str) -> str:
    return WORKFLOW_DESCRIPTIONS[mode]


def _workflow_tool_value(mode: str) -> str:
    """Decode a native Gradio workflow button's stable mode value."""
    if mode not in WORKFLOW_DESCRIPTIONS:
        raise ValueError(f"Unknown workflow: {mode}")
    return mode


def _model_icon_name(key: str) -> str:
    if key == "firered-1.1":
        return "FireRed"
    if key == "flux-klein-4b":
        return "FLUX"
    if key == "krea-2-turbo":
        return "Krea"
    return "Qwen"


def _model_toolbar_label(mode: str, model_key: str, swap_key: str) -> str:
    if mode == KREA_EDIT_MODE:
        return "Krea · fixed"
    key = swap_key if mode == SWAP_MODE else model_key
    return f"{_model_icon_name(key)}  ▾"


MODEL_ICON_FILES = {
    "krea-2-turbo": "krea.svg",
    "firered-1.1": "firered.svg",
    "flux-klein-4b": "flux.svg",
}
SWAP_MODEL_KEYS = ("qwen-2511", "flux-klein-4b", "krea-2-turbo", "qwen-2.1-turbo")


def _model_icon_file(key: str) -> str:
    return MODEL_ICON_FILES.get(key, "qwen.svg")


def _model_icon_for_mode(mode: str, model_key: str, swap_key: str) -> Any:
    key = "krea-2-turbo" if mode == KREA_EDIT_MODE else swap_key if mode == SWAP_MODE else model_key
    return gr.update(icon=UI_ASSETS / _model_icon_file(key))


def _model_keys_for_mode(mode: str) -> tuple[str, ...]:
    if mode == TEXT_MODE:
        return TEXT_TO_IMAGE_KEYS
    if mode in {EDIT_MODE, COMBINE_MODE}:
        return tuple(key for key, spec in MODEL_SPECS.items() if spec.task == "image-to-image")
    return tuple(MODEL_SPECS)


def _model_choice_update(visible: bool, selected: bool) -> Any:
    classes = ["model-choice", *HIDDEN] if not visible else ["model-choice"]
    return gr.update(elem_classes=classes, variant="primary" if visible and selected else "secondary")


def _model_picker_visibility(mode: str, model_key: str, swap_key: str) -> tuple[Any, ...]:
    allowed = set(_model_keys_for_mode(mode))
    show_models = mode not in {SWAP_MODE, KREA_EDIT_MODE}
    show_swap = mode == SWAP_MODE
    return (
        *(_model_choice_update(show_models and key in allowed, key == model_key) for key in MODEL_SPECS),
        *(_model_choice_update(show_swap, key == swap_key) for key in SWAP_MODEL_KEYS),
    )


def _prompt_tool_visibility(mode: str) -> tuple[Any, ...]:
    swap = mode == SWAP_MODE
    return (
        gr.update(elem_classes=["toolbar-icon", *HIDDEN] if not swap else ["toolbar-icon"]),
        gr.update(elem_classes=["toolbar-icon", *HIDDEN] if not swap else ["toolbar-icon"]),
        gr.update(visible=mode != KREA_EDIT_MODE),
    )


def _toolbar_mode_updates(mode: str) -> tuple[Any, ...]:
    canvas = mode in {COMBINE_MODE, TEXT_MODE}
    return (
        gr.update(value=_mode_description(mode)),
        *_prompt_tool_visibility(mode)[:2],
        gr.update(visible=not canvas),
        gr.update(visible=canvas),
        None,
        *_popover_updates(None),
    )


def _workflow_buttons(mode: str) -> tuple[Any, ...]:
    return tuple(
        gr.update(variant="primary" if key == mode else "secondary")
        for key in WORKFLOW_DESCRIPTIONS
    )


def _size_preview_for_mode(
    mode: str, multiplier: int, resolution: str, aspect_ratio: str,
    source: Image.Image | None, krea_source: Image.Image | None,
    swap_body: Image.Image | None, combine_1: Image.Image | None,
    combine_2: Image.Image | None, combine_3: Image.Image | None,
) -> str:
    if mode in {COMBINE_MODE, TEXT_MODE}:
        return _canvas_size_preview(mode, resolution, combine_1, combine_2, combine_3, aspect_ratio)
    active_source = (
        swap_body if mode == SWAP_MODE else krea_source if mode == KREA_EDIT_MODE else source
    )
    return _size_preview(EDIT_MODE, multiplier, resolution, active_source, None, None, None)


PANEL_NAMES = ("settings", "variants", "size", "model", "swap", "bfs")


def _popover_updates(name: str | None) -> tuple[Any, ...]:
    return tuple(gr.update(visible=key == name) for key in PANEL_NAMES)


def _toggle_popover(name: str, current: str | None) -> tuple[Any, ...]:
    selected = None if current == name else name
    return (selected, *_popover_updates(selected))


def _close_popovers() -> tuple[Any, ...]:
    return (None, *_popover_updates(None))


def _toggle_run_details(is_open: bool) -> tuple[bool, Any]:
    next_open = not is_open
    return next_open, gr.update(visible=next_open)


def _progress_markup(message: str, fraction: float, elapsed: float) -> str:
    from html import escape

    pct = round(max(0.0, min(fraction, 1.0)) * 100)
    return (
        '<section class="run-progress" role="status" aria-live="polite">'
        f'<span>{escape(message)}</span><span>{pct}% · {elapsed:.1f}s</span>'
        f'<progress value="{pct}" max="100" aria-label="Generation progress"></progress>'
        '</section>'
    )


def _stream_run(fn: Any, *args: Any):
    """Stream existing inference callbacks to the new progress card without altering inference."""
    events: queue.Queue[tuple[str, Any]] = queue.Queue()
    started = time.monotonic()

    def notify(fraction: float | None, *, desc: str = "") -> None:
        events.put(("progress", (fraction, desc)))

    def run() -> None:
        try:
            events.put(("result", fn(*args, progress=notify)))
        except Exception as exc:  # noqa: BLE001 - forward inference errors to the UI generator
            events.put(("error", exc))

    threading.Thread(target=run, daemon=True).start()
    fraction = 0.0
    message = "Preparing generation…"
    yield gr.update(), gr.update(), gr.update(), _progress_markup(message, fraction, 0.0)
    while True:
        try:
            kind, data = events.get(timeout=0.2)
        except queue.Empty:
            yield gr.update(), gr.update(), gr.update(), _progress_markup(
                message, fraction, time.monotonic() - started
            )
            continue
        if kind == "error":
            message = str(data).splitlines()[0]
            yield gr.update(), gr.update(), gr.update(), _progress_markup(
                f"Failed: {message}", fraction, time.monotonic() - started
            )
            raise data
        if kind == "result":
            images, metadata, status = data
            yield images, metadata, status, _progress_markup("Completed", 1.0, time.monotonic() - started)
            return
        value, message = data
        if value is not None:
            fraction = value
        yield gr.update(), gr.update(), gr.update(), _progress_markup(message, fraction, time.monotonic() - started)


CSS = (ROOT / "photo_edit_studio" / "assets" / "ui" / "studio.css").read_text(encoding="utf-8")


def build_theme() -> gr.themes.Soft:
        return gr.themes.Soft(
                primary_hue="teal", secondary_hue="cyan", neutral_hue="slate",
                font=["Inter", "Segoe UI", "Arial", "sans-serif"],
        ).set(
                body_background_fill="#f2f6f5",
                body_background_fill_dark="#101d24",
                body_text_color="#18323c",
                body_text_color_dark="#e9f3f4",
                body_text_color_subdued="#465e69",
                body_text_color_subdued_dark="#b8ccd2",
                block_background_fill="#ffffff",
                block_background_fill_dark="#1b2b34",
                border_color_primary="#d4e1e2",
                border_color_primary_dark="#3b535e",
                block_label_text_color="#223c46",
                block_label_text_color_dark="#e9f3f4",
                input_placeholder_color="#526977",
                input_placeholder_color_dark="#afc5cb",
        )

HIDDEN = ["mode-hidden"]
DEFAULT_PROGRESS = gr.Progress()


def _visibility(hidden: bool) -> Any:
    return gr.update(elem_classes=HIDDEN if hidden else [])


def _model_changed(key: str) -> tuple[Any, ...]:
    spec = MODEL_SPECS[key]
    choices = lora_choices(key)
    return (
        gr.update(value=spec.default_steps),
        gr.update(value=spec.default_guidance),
        gr.update(value=spec.default_true_cfg),
        (
            f"**{spec.label}** · text-only locally; edit with up to 2 images via ComfyUI · "
            f"{spec.license_note}"
            if key == "krea-2-turbo" else
            f"**{spec.label}** · up to {spec.max_images} inputs · {spec.license_note}"
        ),
        library_status(key),
        gr.update(value=TEXT_MODE if spec.task == "text-to-image" else EDIT_MODE),
        *[gr.update(choices=choices, value=NONE_CHOICE) for _ in range(MAX_LORAS)],
    )


def _standard_model_changed(key: str, mode: str) -> tuple[Any, ...]:
    values = _model_changed(key)
    if mode == KREA_EDIT_MODE:
        return (*values[:3], gr.update(value="", elem_classes=HIDDEN), values[4], gr.update(), *values[6:])
    if mode == SWAP_MODE:
        return (*values[:5], gr.update(), *values[6:])
    if mode == TEXT_MODE:
        return (*values[:5], gr.update(value=TEXT_MODE), *values[6:])
    return (*values[:5], gr.update(), *values[6:])


def _swap_model_changed(model_key: str, kind: str) -> tuple[Any, ...]:
    valid = [name for key, name in SWAP_PROFILES if key == model_key]
    kind = kind if kind in valid else valid[0]
    spec = MODEL_SPECS[model_key]
    choices = lora_choices(model_key)
    return (
        gr.update(choices=valid, value=kind),
        gr.update(value=10 if model_key == "krea-2-turbo" else spec.default_steps),
        gr.update(value=1.0 if model_key == "krea-2-turbo" else spec.default_guidance),
        gr.update(value=spec.default_true_cfg),
        _swap_model_info(model_key, kind),
        library_status(model_key),
        *[gr.update(choices=choices, value=NONE_CHOICE) for _ in range(MAX_LORAS)],
    )


def _mode_model_defaults(mode: str, model_key: str, swap_model_key: str, kind: str) -> tuple[Any, ...]:
    if mode == SWAP_MODE:
        return _swap_model_changed(swap_model_key, kind)
    if mode == KREA_EDIT_MODE:
        return (
            gr.update(value="Head"),
            gr.update(value=10), gr.update(value=1.0), gr.update(value=0.0),
                        gr.update(value="", elem_classes=HIDDEN),
            library_status("krea-2-turbo"),
            *[gr.update(choices=lora_choices("krea-2-turbo"), value=NONE_CHOICE)
              for _ in range(MAX_LORAS)],
        )
    if mode in {EDIT_MODE, COMBINE_MODE}:
        model_key = _model_choices_for_mode(mode, model_key)["value"]
    values = _model_changed(model_key)
    return (gr.update(value="Head"), *values[:3], gr.update(value=values[3], elem_classes=[]), values[4], *values[6:])


def _swap_kind_changed(model_key: str, kind: str) -> str:
    return _swap_model_info(model_key, kind)


def _swap_model_info(model_key: str, kind: str) -> str:
    profile = swap_profile(model_key, kind)
    spec = MODEL_SPECS[model_key]
    if model_key == "qwen-2.1-turbo":
        return (
            f"**{spec.label}** · Viggle Turbo r256 · {kind} BFS LoRA {profile.filename} "
            "· body image sets output aspect"
        )
    return f"**{spec.label}** · {profile.filename} · body image sets output aspect"


def _upload_loras(model_key: str, files: Any, *current_names: str) -> tuple[Any, ...]:
    saved = import_lora_files(model_key, files)
    names = assign_new_slots(list(current_names), saved)
    choices = lora_choices(model_key)
    return (
        library_status(model_key, saved),
        *[gr.update(choices=choices, value=name) for name in names],
    )


def _upload_loras_for_mode(
    mode: str, model_key: str, swap_model_key: str, files: Any, *current_names: str
) -> tuple[Any, ...]:
    return _upload_loras(
        swap_model_key if mode == SWAP_MODE else "krea-2-turbo" if mode == KREA_EDIT_MODE else model_key,
        files, *current_names,
    )


def _as_pil_images(value: Any) -> list[Image.Image]:
    if value is None:
        return []
    if isinstance(value, Image.Image):
        return [value]
    images: list[Image.Image] = []
    for item in value:
        if isinstance(item, Image.Image):
            images.append(item)
        elif isinstance(item, (tuple, list)) and item and isinstance(item[0], Image.Image):
            images.append(item[0])
    return images


def _collect_images(
    mode: str,
    source: Image.Image | None,
    references: Any,
    combine_1: Image.Image | None,
    combine_2: Image.Image | None,
    combine_3: Image.Image | None,
) -> list[Image.Image]:
    if mode == TEXT_MODE:
        return []
    if mode == COMBINE_MODE:
        return [image for image in (combine_1, combine_2, combine_3) if image is not None]
    images = [source] if source is not None else []
    images.extend(_as_pil_images(references))
    return images


def _size_preview(
    mode: str,
    multiplier: int,
    resolution: str,
    source: Image.Image | None,
    combine_1: Image.Image | None,
    combine_2: Image.Image | None,
    combine_3: Image.Image | None,
    aspect_ratio: str = "1:1",
) -> str:
    if mode in {COMBINE_MODE, TEXT_MODE}:
        width, height = combine_canvas_size(str(resolution), str(aspect_ratio))
        return f"Output **{width} × {height}** · {resolution} · {aspect_ratio} canvas"
    images = _collect_images(mode, source, None, combine_1, combine_2, combine_3)
    if not images:
        return "Upload an image to lock output size to its aspect ratio."
    width, height = scaled_output_size(images[0].size, int(multiplier))
    src_w, src_h = images[0].size
    return (
        f"Output **{width} × {height}** · source {src_w} × {src_h} · "
        f"aspect preserved at ×{int(multiplier)}"
    )


def _edit_size_preview(multiplier: int, source: Image.Image | None) -> str:
    return _size_preview(EDIT_MODE, int(multiplier), "1K", source, None, None, None)


def _combine_size_preview(
    resolution: str,
    combine_1: Image.Image | None,
    combine_2: Image.Image | None,
    combine_3: Image.Image | None,
    aspect_ratio: str = "1:1",
) -> str:
    return _size_preview(
        COMBINE_MODE, 1, str(resolution), None, combine_1, combine_2, combine_3,
        str(aspect_ratio),
    )


def _canvas_size_preview(
    mode: str,
    resolution: str,
    combine_1: Image.Image | None,
    combine_2: Image.Image | None,
    combine_3: Image.Image | None,
    aspect_ratio: str,
) -> str:
    return _size_preview(
        mode, 1, resolution, None, combine_1, combine_2, combine_3, aspect_ratio
    )


def _mode_changed(
    mode: str, resolution: str = "1K", aspect_ratio: str = "1:1"
) -> tuple[Any, ...]:
    is_edit = mode == EDIT_MODE
    is_krea_edit = mode == KREA_EDIT_MODE
    is_text = mode == TEXT_MODE
    is_swap = mode == SWAP_MODE
    preview = (
        "Upload an image to lock output size to its aspect ratio."
        if is_edit or is_krea_edit
        else _canvas_size_preview(mode, resolution, None, None, None, aspect_ratio)
    )
    return (
        _visibility(hidden=not is_edit),
        _visibility(hidden=mode != COMBINE_MODE),
        _visibility(hidden=not is_swap),
        _visibility(hidden=not is_krea_edit),
        gr.update(
            placeholder=(
                EDIT_PLACEHOLDER
                if is_edit
                else "Edit Picture 1 using Picture 2 as a reference; preserve the scene and pose."
                if is_krea_edit else SWAP_PLACEHOLDER if is_swap else TEXT_PLACEHOLDER if is_text else COMBINE_PLACEHOLDER
            ),
            label=(
                "Editing instruction"
                if is_edit or is_krea_edit
                else "Additional swap instructions" if is_swap else "Image prompt" if is_text else "Combination instruction"
            ),
        ),
        gr.update(
            value=(
                "Generate edit"
                if is_edit or is_krea_edit
                else "Generate swap" if is_swap else "Create image" if is_text else "Generate combination"
            ),
            elem_classes=["generate-action", *HIDDEN] if is_swap or is_krea_edit else ["generate-action"],
        ),
        _visibility(hidden=not (is_edit or is_swap or is_krea_edit)),
        _visibility(hidden=is_edit or is_krea_edit),
        preview,
        gr.update(elem_classes=["generate-action", *HIDDEN] if not is_swap else ["generate-action"]),
        _visibility(hidden=True),
        _visibility(hidden=True),
        gr.update(elem_classes=["generate-action", *HIDDEN] if not is_krea_edit else ["generate-action"]),
        _visibility(hidden=not is_krea_edit),
        _visibility(hidden=mode not in {COMBINE_MODE, TEXT_MODE}),
    )


def _model_choices_for_mode(mode: str, selected_key: str) -> Any:
    keys = _model_keys_for_mode(mode)
    choices = [(MODEL_SPECS[key].label, key) for key in keys]
    value = selected_key if selected_key in keys else keys[0]
    return gr.update(choices=choices, value=value)


def _turbo_controls_for_model(mode: str, model_key: str, swap_model_key: str) -> tuple[Any, ...]:
    """Fix controls that must not depart from Viggle's exact six-step schedule."""
    key = swap_model_key if mode == SWAP_MODE else model_key
    turbo = key == "qwen-2.1-turbo"
    return (
        gr.update(value=6 if turbo else MODEL_SPECS[key].default_steps, interactive=not turbo),
        gr.update(value=1.0 if turbo else MODEL_SPECS[key].default_guidance, interactive=not turbo),
        gr.update(value=1.0 if turbo else MODEL_SPECS[key].default_true_cfg, interactive=not turbo),
        gr.update(value="", interactive=not turbo) if turbo else gr.update(interactive=True),
    )


def _krea_first_visibility(mode: str, swap_model_key: str, model_key: str) -> Any:
    return _visibility(
        hidden=not ((mode == TEXT_MODE and model_key == "krea-2-turbo")
                    or mode == KREA_EDIT_MODE
                    or (mode == SWAP_MODE and swap_model_key == "krea-2-turbo"))
    )


def _run(
    mode: str,
    source: Image.Image | None,
    references: Any,
    combine_1: Image.Image | None,
    combine_2: Image.Image | None,
    combine_3: Image.Image | None,
    mask: Image.Image | None,
    model_key: str,
    prompt: str,
    negative_prompt: str,
    size_multiplier: int,
    output_resolution: str,
    steps: int,
    guidance: float,
    true_cfg: float,
    strength: float,
    seed: int,
    count: int,
    preserve_identity: bool,
    restoration: str,
    restore_weight: float,
    lora_name_1: str = NONE_CHOICE,
    lora_name_2: str = NONE_CHOICE,
    lora_name_3: str = NONE_CHOICE,
    lora_name_4: str = NONE_CHOICE,
    lora_name_5: str = NONE_CHOICE,
    lora_weight_1: float = 1.0,
    lora_weight_2: float = 1.0,
    lora_weight_3: float = 1.0,
    lora_weight_4: float = 1.0,
    lora_weight_5: float = 1.0,
    krea_first_weight: float = 1.0,
    output_aspect_ratio: str = "1:1",
    progress: gr.Progress = DEFAULT_PROGRESS,
) -> tuple[list[Image.Image], str, str]:
    progress(0, desc="Starting generation")
    if not prompt.strip():
        raise ValueError("Enter an instruction.")
    spec = MODEL_SPECS[model_key]
    if spec.task == "text-to-image" and mode not in {TEXT_MODE, KREA_EDIT_MODE}:
        raise ValueError(f"{spec.label} is not available in {mode} mode.")
    if mode == TEXT_MODE and model_key not in TEXT_TO_IMAGE_KEYS:
        raise ValueError("Choose a text-capable model for Create from text.")
    if mode == KREA_EDIT_MODE and model_key != "krea-2-turbo":
        raise ValueError("Krea reference edit requires Krea 2 Turbo.")
    compose = mode in {COMBINE_MODE, TEXT_MODE}
    if mode == KREA_EDIT_MODE:
        if mask is not None:
            raise ValueError("Krea ComfyUI reference edit does not support the optional mask; remove it before generating.")
        images = _collect_images(EDIT_MODE, source, references, combine_1, combine_2, combine_3)
        if not 1 <= len(images) <= 2:
            raise ValueError("Krea reference editing needs a source and at most one reference.")
        compose = False
    else:
        images = _collect_images(mode, source, references, combine_1, combine_2, combine_3)
    request = GenerationRequest(
        model_key=model_key,
        prompt=prompt,
        negative_prompt=negative_prompt,
        images=images,
        mask=None if compose or mode == KREA_EDIT_MODE else mask,
        width=1024,
        height=1024,
        steps=int(steps),
        guidance=guidance,
        true_cfg=true_cfg,
        strength=strength,
        seed=int(seed),
        count=int(count),
        preserve_identity=preserve_identity and mode != TEXT_MODE,
        size_multiplier=int(size_multiplier),
        compose=compose,
        output_resolution=str(output_resolution),
        output_aspect_ratio=str(output_aspect_ratio),
        loras=selected_loras(
            model_key,
            [lora_name_1, lora_name_2, lora_name_3, lora_name_4, lora_name_5],
            [lora_weight_1, lora_weight_2, lora_weight_3, lora_weight_4, lora_weight_5],
        ),
        workflow=(
            "krea-reference" if mode == KREA_EDIT_MODE
            else "krea-text" if mode == TEXT_MODE and model_key == "krea-2-turbo"
            else "text" if mode == TEXT_MODE else "standard"
        ),
        krea_first_lora_weight=krea_first_weight,
    )
    result = generate(
        request,
        restoration,
        restore_weight,
        progress=GenerationProgress(lambda fraction, message: progress(fraction, desc=message)),
    )
    return result.images, metadata_text(result), model_manager.status


def _run_swap(
    body: Image.Image | None,
    reference: Image.Image | None,
    model_key: str,
    kind: str,
    instruction: str,
    negative_prompt: str,
    multiplier: int,
    steps: int,
    guidance: float,
    true_cfg: float,
    strength: float,
    seed: int,
    preserve_identity: bool,
    restoration: str,
    restore_weight: float,
    bfs_weight: float,
    krea_first_weight: float,
    *extra_lora_inputs: Any,
    progress: gr.Progress = DEFAULT_PROGRESS,
) -> tuple[list[Image.Image], str, str]:
    if body is None or reference is None:
        raise ValueError("Upload both the target body/scene and the replacement face/person.")
    if model_key == "qwen-2.1-turbo" and kind not in {"Head", "Body"}:
        raise ValueError("Qwen Image 2.1 Turbo supports Head or Body BFS swaps.")
    profile = swap_profile(model_key, kind)
    if model_key == "qwen-2.1-turbo" and negative_prompt.strip():
        raise ValueError("Viggle Turbo requires an empty negative prompt for swap.")
    if model_key == "krea-2-turbo" and negative_prompt.strip():
        raise ValueError(
            "Krea ComfyUI swap uses the negative prompt in its exported workflow; "
            "leave this field empty or edit the workflow in ComfyUI."
        )
    extra = selected_loras(
        model_key,
        list(extra_lora_inputs[:MAX_LORAS]),
        list(extra_lora_inputs[MAX_LORAS:]),
    )
    if model_key == "qwen-2.1-turbo":
        extra = [lora for lora in extra if lora.name.casefold() != profile.filename.casefold()]
        if len(extra) >= MAX_LORAS:
            raise ValueError(
                f"Qwen 2.1 {kind} swap supports at most four optional LoRAs in addition "
                f"to the required BFS {kind} adapter."
            )
    if model_key == "krea-2-turbo":
        prompt = instruction.strip()
    else:
        prompt = f"{profile.trigger} {instruction.strip()}".strip()
    if preserve_identity:
        prompt += " Preserve the target body's pose, clothing, lighting, and scene."
    request = GenerationRequest(
        model_key=model_key,
        prompt=prompt,
        negative_prompt=negative_prompt,
        images=[body, reference],
        mask=None,
        width=1024,
        height=1024,
        steps=int(steps),
        guidance=guidance,
        true_cfg=true_cfg,
        strength=strength,
        seed=int(seed),
        preserve_identity=False,
        size_multiplier=int(multiplier),
        loras=[swap_lora(model_key, kind, bfs_weight)] + extra,
        workflow="swap",
        swap_kind=kind,
        krea_first_lora_weight=krea_first_weight,
    )
    progress(0, desc="Preparing two-image swap")
    result = generate(
        request,
        restoration,
        restore_weight,
        progress=GenerationProgress(lambda fraction, message: progress(fraction, desc=message)),
    )
    return result.images, metadata_text(result), model_manager.status


def _run_krea_reference(
    source: Image.Image | None,
    reference: Image.Image | None,
    prompt: str,
    negative_prompt: str,
    multiplier: int,
    steps: int,
    guidance: float,
    seed: int,
    restoration: str,
    restore_weight: float,
    krea_first_weight: float = 1.0,
    *extra_lora_inputs: Any,
    progress: gr.Progress = DEFAULT_PROGRESS,
) -> tuple[list[Image.Image], str, str]:
    if source is None:
        raise ValueError("Upload Picture 1 (the source image) for Krea reference editing.")
    if not prompt.strip():
        raise ValueError("Enter an edit instruction.")
    request = GenerationRequest(
        model_key="krea-2-turbo",
        prompt=prompt.strip(),
        negative_prompt=negative_prompt,
        images=[source] + ([reference] if reference is not None else []),
        mask=None,
        width=1024,
        height=1024,
        steps=int(steps),
        guidance=guidance,
        true_cfg=0.0,
        strength=1.0,
        seed=int(seed),
        size_multiplier=int(multiplier),
        loras=selected_loras(
            "krea-2-turbo",
            list(extra_lora_inputs[:MAX_LORAS]),
            list(extra_lora_inputs[MAX_LORAS:]),
        ),
        workflow="krea-reference",
        krea_first_lora_weight=krea_first_weight,
    )
    progress(0, desc="Preparing Krea reference edit")
    result = generate(
        request,
        restoration,
        restore_weight,
        progress=GenerationProgress(lambda fraction, message: progress(fraction, desc=message)),
    )
    return result.images, metadata_text(result), model_manager.status


def build_app() -> gr.Blocks:
    model_choices = _model_choices_for_mode(EDIT_MODE, "qwen-2511-aio")["choices"]
    with gr.Blocks(title="Local Photo Edit Studio") as app:
        gr.HTML(
            "<header class='studio-header'><strong>Local Photo Edit Studio</strong>"
            f"<span class='badge'>{_release_badge()}</span>"
            "<p>Powered by <b>Javapower</b></p></header>"
        )
        with gr.Column(elem_classes="studio-shell", elem_id="studio-shell"):
            mode = gr.State(EDIT_MODE)
            with gr.Row(elem_id="studio-top"):
                with gr.Row(elem_classes="workflow-toolbar", elem_id="studio-workflows"):
                    workflow_buttons = {
                        key: gr.Button(
                            key, icon=UI_ASSETS / WORKFLOW_ICONS[key], size="sm",
                            variant="primary" if key == EDIT_MODE else "secondary",
                            elem_classes="workflow-choice",
                        ) for key in WORKFLOW_DESCRIPTIONS
                    }
                with gr.Group(elem_id="studio-progress"):
                    progress_card = gr.HTML(_progress_markup("Ready to create", 0, 0), elem_id="studio-progress-card")
                    details_trigger = gr.Button(
                        "Run details", icon=UI_ASSETS / "run-details.svg", size="sm",
                        elem_id="studio-details-trigger",
                    )
            details_open = gr.State(False)
            with gr.Column(visible=False, elem_classes="run-details-panel", elem_id="studio-run-details") as details_panel:
                gr.Markdown("**Run details** · prompts are not stored")
                metadata = gr.Code(label="Run metadata", language="json")
                status = gr.Markdown("No model loaded")
            description = gr.Markdown(_mode_description(EDIT_MODE), elem_classes="workflow-description", elem_id="studio-description")
            output = gr.Gallery(label="Generated images", columns=2, object_fit="contain", height="auto", elem_id="studio-results")
            with gr.Group(elem_id="studio-uploads"):
                with gr.Group() as edit_group:
                    gr.Markdown(
                        "Edits use the uploaded image's full canvas. A model can still reframe the "
                        "subject; describe any framing you need to retain in the instruction."
                    )
                    with gr.Row():
                        source = gr.Image(type="pil", label="Source photo", height=390)
                        mask = gr.Image(
                            type="pil",
                            image_mode="L",
                            label="Optional edit mask (white = edit)",
                            height=390,
                        )
                    references = gr.Gallery(
                        label="Optional references · clean face crop, clothing, object, or scene",
                        type="pil",
                        columns=4,
                        rows=1,
                        height=180,
                    )
                with gr.Group(elem_classes=HIDDEN) as combine_group:
                    gr.Markdown(
                        "Upload 1–3 images and describe how to combine them into one new picture. "
                        "Refer to them as image 1, image 2, and image 3. Choose a 1K or 2K canvas."
                    )
                    with gr.Row():
                        combine_1 = gr.Image(type="pil", label="Image 1", height=260)
                        combine_2 = gr.Image(type="pil", label="Image 2", height=260)
                        combine_3 = gr.Image(type="pil", label="Image 3", height=260)
                with gr.Group(elem_classes=HIDDEN) as swap_group:
                    gr.Markdown(
                        "**Two images:** the target body/scene is Picture 1; the donor face/person "
                        "is Picture 2. Head swap changes the head; Body swap (Krea or Qwen 2.1 BFS) "
                        "changes the person. Use only images you have permission to edit."
                    )
                    with gr.Row():
                        swap_body = gr.Image(type="pil", label="Picture 1 · target body / scene", height=300)
                        swap_reference = gr.Image(type="pil", label="Picture 2 · replacement face / person", height=300)
                with gr.Group(elem_classes=HIDDEN) as krea_edit_group:
                    gr.Markdown(
                        "Krea 2 reference editing runs through local ComfyUI. "
                        "Picture 1 sets the scene and output aspect; Picture 2 is optional. "
                        "The identity-edit LoRA stays active; additional Krea LoRAs selected below "
                        "are applied after it. The edit-strength control does not apply here."
                    )
                    with gr.Row():
                        krea_source = gr.Image(type="pil", label="Picture 1 · source scene", height=300)
                        krea_reference = gr.Image(type="pil", label="Picture 2 · optional reference", height=300)
            with gr.Column(elem_id="studio-composer"):
                with gr.Group(elem_classes="prompt-surface", elem_id="studio-prompt"):
                    prompt = gr.Textbox(
                        label="Editing instruction",
                        placeholder=EDIT_PLACEHOLDER,
                        lines=4,
                    )
                    with gr.Row(elem_classes="prompt-toolbar", elem_id="studio-prompt-toolbar"):
                        settings_trigger = gr.Button("Settings", icon=UI_ASSETS / "settings.svg", size="sm", elem_classes="toolbar-icon")
                        variants_trigger = gr.Button("Variants", icon=UI_ASSETS / "variants.svg", size="sm", elem_classes="toolbar-icon")
                        size_trigger = gr.Button("Size", icon=UI_ASSETS / "size.svg", size="sm", elem_classes="toolbar-icon")
                        swap_trigger = gr.Button("Replace", icon=UI_ASSETS / "swap.svg", size="sm", elem_classes=["toolbar-icon", *HIDDEN])
                        bfs_trigger = gr.Button("BFS weight", icon=UI_ASSETS / "weight.svg", size="sm", elem_classes=["toolbar-icon", *HIDDEN])
                        model_trigger = gr.Button("Qwen  ▾", icon=UI_ASSETS / "qwen.svg", size="sm", elem_classes="toolbar-icon")
                        generate_button = gr.Button(
                            "Generate edit",
                            icon=UI_ASSETS / "generate.svg",
                            variant="primary",
                            size="sm",
                            elem_classes="generate-action",
                        )
                        swap_button = gr.Button(
                            "Generate swap",
                            icon=UI_ASSETS / "generate.svg",
                            variant="primary",
                            size="sm",
                            elem_classes=["generate-action", *HIDDEN],
                        )
                        krea_edit_button = gr.Button(
                            "Generate Krea edit",
                            icon=UI_ASSETS / "generate.svg",
                            variant="primary",
                            size="sm",
                            elem_classes=["generate-action", *HIDDEN],
                        )
                active_panel = gr.State(None)
                with gr.Group(visible=False, elem_classes="tool-panel", elem_id="studio-panel-settings") as settings_panel:
                    gr.Markdown("**Advanced model controls** · tune only if the selected model supports it.")
                    steps = gr.Slider(1, 60, value=40, step=1, label="Inference steps")
                    guidance = gr.Slider(0, 10, value=1.0, step=0.1, label="Guidance / CFG")
                    true_cfg = gr.Slider(0, 10, value=4.0, step=0.1, label="True CFG (Qwen/FireRed)")
                    strength = gr.Slider(0.05, 1.0, value=0.8, step=0.05, label="Edit strength (when supported)")
                with gr.Group(visible=False, elem_classes="tool-panel", elem_id="studio-panel-variants") as variants_panel:
                    gr.Markdown(f"**Variants** · up to {settings.max_batch_count} output(s) per request on this GPU profile.")
                    count = gr.Number(value=1, precision=0, label=f"Outputs (maximum {settings.max_batch_count})", interactive=False)
                with gr.Group(visible=False, elem_classes="tool-panel", elem_id="studio-panel-size") as size_panel:
                    size_multiplier = gr.Radio(
                        choices=[("×1 (recommended)", 1), ("×2 (VRAM-capped)", 2)], value=1,
                        label="Size multiplier · source aspect preserved",
                    )
                    with gr.Group(visible=False) as combine_canvas_controls:
                        output_resolution = gr.Radio(choices=["1K", "2K"], value="1K", label="Output resolution")
                        aspect_ratio = gr.Radio(choices=["1:1", "9:16", "16:9"], value="1:1", label="Aspect ratio")
                with gr.Group(visible=False, elem_classes="tool-panel", elem_id="studio-panel-model") as model_panel:
                    model_key = gr.Dropdown(
                        model_choices, value="qwen-2511-aio", label="Model",
                        show_label=False, elem_classes=HIDDEN,
                    )
                    swap_model = gr.Dropdown(
                        [(MODEL_SPECS[key].label, key) for key in SWAP_MODEL_KEYS],
                        value="qwen-2511", label="Swap model",
                        show_label=False, elem_classes=HIDDEN,
                    )
                    model_choice_buttons = {
                        key: gr.Button(
                            MODEL_SPECS[key].label,
                            icon=UI_ASSETS / _model_icon_file(key),
                            size="sm",
                            variant="primary" if key == "qwen-2511-aio" else "secondary",
                            elem_classes="model-choice" if MODEL_SPECS[key].task == "image-to-image" else ["model-choice", *HIDDEN],
                        )
                        for key in MODEL_SPECS
                    }
                    swap_choice_buttons = {
                        key: gr.Button(
                            MODEL_SPECS[key].label,
                            icon=UI_ASSETS / _model_icon_file(key),
                            size="sm",
                            variant="primary" if key == "qwen-2511" else "secondary",
                            elem_classes=["model-choice", *HIDDEN],
                        )
                        for key in SWAP_MODEL_KEYS
                    }
                    krea_edit_info = gr.Markdown("**Krea 2 Turbo · reference edit via local ComfyUI**", elem_classes=HIDDEN)
                with gr.Group(visible=False, elem_classes="tool-panel", elem_id="studio-panel-swap") as swap_panel:
                    swap_kind = gr.Radio(["Head"], value="Head", label="Replace")
                with gr.Group(visible=False, elem_classes="tool-panel", elem_id="studio-panel-bfs") as bfs_panel:
                    bfs_weight = gr.Slider(0.1, 1.5, value=1.0, step=0.05, label="BFS adapter weight")
                with gr.Group(elem_classes="model-description-card", elem_id="studio-model-info"):
                    model_info = gr.Markdown()
                    size_info = gr.Markdown("Upload an image to lock output size to its aspect ratio.")
                with gr.Group(elem_classes="negative-prompt-card", elem_id="studio-negative"):
                    negative_prompt = gr.Textbox(
                        label="Negative prompt (model-dependent)",
                        placeholder="identity drift, altered facial structure, waxy skin, extra fingers",
                        lines=2,
                    )
            with gr.Group(elem_classes="studio-options", elem_id="studio-options"):
                with gr.Accordion("Advanced LoRAs · up to 5", open=False, elem_id="studio-accordion-loras"):
                    krea_first_weight = gr.Slider(
                        0.05, 2.0, value=1.0, step=0.05,
                        label="Krea2_ALWAYS_LOAD_FIRST · mandatory weight",
                        info="Krea only · always applied before the identity-edit, BFS, and optional LoRAs",
                        elem_classes=HIDDEN,
                    )
                    lora_upload = gr.File(
                        label="Upload LoRA files for the selected model family",
                        file_count="multiple",
                        file_types=[".safetensors", ".pt", ".bin"],
                        type="filepath",
                    )
                    lora_status = gr.Markdown()
                    lora_names: list[gr.Dropdown] = []
                    lora_weights: list[gr.Slider] = []
                    for index in range(1, MAX_LORAS + 1):
                        with gr.Row():
                            lora_names.append(
                                gr.Dropdown(
                                    choices=[NONE_CHOICE],
                                    value=NONE_CHOICE,
                                    label=f"LoRA {index}",
                                    scale=3,
                                )
                            )
                            lora_weights.append(
                                gr.Slider(
                                    0,
                                    2,
                                    value=1.0,
                                    step=0.05,
                                    label=f"Weight {index}",
                                    scale=2,
                                )
                            )
                    gr.Markdown(
                        "LoRAs stay on disk under `models/loras/<family>/`. "
                        "Use adapters trained for the selected family; weights of 0 skip a slot. "
                        "Krea reference edit and swap chain selected Krea LoRAs after the required base adapter."
                    )
                with gr.Accordion("Face restoration", open=False, elem_id="studio-accordion-face"):
                    restoration = gr.Radio(["Off", "GFPGAN"], value="Off", label="Post-process")
                    restore_weight = gr.Slider(
                        0, 1, value=0.35, step=0.05, label="Restoration fidelity weight"
                    )
                    gr.Markdown(
                        "Use sparingly: restoration can improve micro-details but may also "
                        "shift identity."
                    )
                with gr.Accordion("Other options", open=False, elem_id="studio-accordion-other"):
                    seed = gr.Number(value=-1, precision=0, label="Seed (-1 random)")
                    preserve_identity = gr.Checkbox(value=True, label="Identity-preservation prompt")
                    unload_button = gr.Button("Unload model / release VRAM")
                    clear_button = gr.ClearButton(value="Clear")

        inputs = [
            mode,
            source,
            references,
            combine_1,
            combine_2,
            combine_3,
            mask,
            model_key,
            prompt,
            negative_prompt,
            size_multiplier,
            output_resolution,
            steps,
            guidance,
            true_cfg,
            strength,
            seed,
            count,
            preserve_identity,
            restoration,
            restore_weight,
            *lora_names,
            *lora_weights,
            krea_first_weight,
            aspect_ratio,
        ]
        generate_button.click(
            lambda *values: _stream_run(_run, *values), inputs=inputs,
            outputs=[output, metadata, status, progress_card], show_progress="hidden",
        )
        swap_inputs = [
            swap_body, swap_reference, swap_model, swap_kind, prompt, negative_prompt,
            size_multiplier, steps, guidance, true_cfg, strength, seed,
            preserve_identity, restoration, restore_weight, bfs_weight,
            krea_first_weight,
            *lora_names, *lora_weights,
        ]
        swap_button.click(
            lambda *values: _stream_run(_run_swap, *values), inputs=swap_inputs,
            outputs=[output, metadata, status, progress_card], show_progress="hidden",
        )
        krea_edit_button.click(
            lambda *values: _stream_run(_run_krea_reference, *values),
            inputs=[krea_source, krea_reference, prompt, negative_prompt, size_multiplier,
                    steps, guidance, seed, restoration, restore_weight, krea_first_weight,
                    *lora_names, *lora_weights],
            outputs=[output, metadata, status, progress_card],
            show_progress="hidden",
        )
        model_outputs = [
            steps,
            guidance,
            true_cfg,
            model_info,
            lora_status,
            mode,
            *lora_names,
        ]
        mode_outputs = [
            edit_group,
            combine_group,
            swap_group,
            krea_edit_group,
            prompt,
            generate_button,
            size_multiplier,
            output_resolution,
            size_info,
            swap_button,
            model_key,
            swap_model,
            krea_edit_button,
            krea_edit_info,
            combine_canvas_controls,
        ]
        popover_outputs = [
            active_panel, settings_panel, variants_panel, size_panel, model_panel,
            swap_panel, bfs_panel,
        ]
        details_trigger.click(
            _toggle_run_details, inputs=details_open, outputs=[details_open, details_panel],
            show_progress="hidden",
        )
        for workflow, button in workflow_buttons.items():
            button.click(lambda key=workflow: key, outputs=mode, show_progress="hidden").then(
                _workflow_buttons, inputs=mode, outputs=list(workflow_buttons.values()),
            )
        for name, button in (
            ("settings", settings_trigger), ("variants", variants_trigger),
            ("size", size_trigger), ("model", model_trigger),
            ("swap", swap_trigger), ("bfs", bfs_trigger),
        ):
            button.click(
                lambda current, panel=name: _toggle_popover(panel, current),
                inputs=[active_panel], outputs=popover_outputs, show_progress="hidden",
            )
        picker_outputs = [*model_choice_buttons.values(), *swap_choice_buttons.values()]
        for key, button in model_choice_buttons.items():
            button.click(lambda selected=key: gr.update(value=selected), outputs=model_key, show_progress="hidden")
        for key, button in swap_choice_buttons.items():
            button.click(lambda selected=key: gr.update(value=selected), outputs=swap_model, show_progress="hidden")
        model_change = model_key.change(
            _standard_model_changed, inputs=[model_key, mode], outputs=model_outputs
        )
        model_change.then(
            _mode_changed, inputs=[mode, output_resolution, aspect_ratio], outputs=mode_outputs
        ).then(
            _krea_first_visibility, inputs=[mode, swap_model, model_key], outputs=krea_first_weight,
        ).then(
            _turbo_controls_for_model, inputs=[mode, model_key, swap_model],
            outputs=[steps, guidance, true_cfg, negative_prompt],
        ).then(
            _close_popovers, outputs=popover_outputs,
        ).then(
            _model_toolbar_label, inputs=[mode, model_key, swap_model], outputs=model_trigger,
        ).then(
            _prompt_tool_visibility, inputs=mode,
            outputs=[swap_trigger, bfs_trigger, model_trigger],
        ).then(
            _model_icon_for_mode, inputs=[mode, model_key, swap_model], outputs=model_trigger,
        ).then(
            _model_picker_visibility, inputs=[mode, model_key, swap_model], outputs=picker_outputs,
        ).then(
            _size_preview_for_mode,
            inputs=[mode, size_multiplier, output_resolution, aspect_ratio, source,
                    krea_source, swap_body, combine_1, combine_2, combine_3],
            outputs=size_info,
        )
        mode.change(
            _mode_changed,
            inputs=[mode, output_resolution, aspect_ratio],
            outputs=mode_outputs,
        ).then(
            _model_choices_for_mode, inputs=[mode, model_key], outputs=model_key,
        ).then(
            _mode_model_defaults,
            inputs=[mode, model_key, swap_model, swap_kind],
            outputs=[swap_kind, steps, guidance, true_cfg, model_info, lora_status, *lora_names],
        ).then(
            _krea_first_visibility, inputs=[mode, swap_model, model_key], outputs=krea_first_weight,
        ).then(
            _turbo_controls_for_model, inputs=[mode, model_key, swap_model],
            outputs=[steps, guidance, true_cfg, negative_prompt],
        ).then(
            _toolbar_mode_updates, inputs=[mode],
            outputs=[description, swap_trigger, bfs_trigger,
                     size_multiplier, combine_canvas_controls, *popover_outputs],
        ).then(
            _model_toolbar_label, inputs=[mode, model_key, swap_model], outputs=model_trigger,
        ).then(
            _prompt_tool_visibility, inputs=mode,
            outputs=[swap_trigger, bfs_trigger, model_trigger],
        ).then(
            _model_icon_for_mode, inputs=[mode, model_key, swap_model], outputs=model_trigger,
        ).then(
            _model_picker_visibility, inputs=[mode, model_key, swap_model], outputs=picker_outputs,
        ).then(
            _size_preview_for_mode,
            inputs=[mode, size_multiplier, output_resolution, aspect_ratio, source,
                    krea_source, swap_body, combine_1, combine_2, combine_3],
            outputs=size_info,
        )
        swap_model.change(
            _swap_model_changed,
            inputs=[swap_model, swap_kind],
            outputs=[swap_kind, steps, guidance, true_cfg, model_info, lora_status, *lora_names],
        ).then(
            _krea_first_visibility, inputs=[mode, swap_model, model_key], outputs=krea_first_weight,
        ).then(
            _turbo_controls_for_model, inputs=[mode, model_key, swap_model],
            outputs=[steps, guidance, true_cfg, negative_prompt],
        ).then(
            _close_popovers, outputs=popover_outputs,
        ).then(
            _model_toolbar_label, inputs=[mode, model_key, swap_model], outputs=model_trigger,
        ).then(
            _prompt_tool_visibility, inputs=mode,
            outputs=[swap_trigger, bfs_trigger, model_trigger],
        ).then(
            _model_icon_for_mode, inputs=[mode, model_key, swap_model], outputs=model_trigger,
        ).then(
            _model_picker_visibility, inputs=[mode, model_key, swap_model], outputs=picker_outputs,
        )
        swap_kind.change(
            _swap_kind_changed, inputs=[swap_model, swap_kind], outputs=model_info
        )
        size_preview_inputs = [
            mode, size_multiplier, output_resolution, aspect_ratio, source,
            krea_source, swap_body, combine_1, combine_2, combine_3,
        ]
        for control in [size_multiplier, source, krea_source, swap_body,
                        output_resolution, combine_1, combine_2, combine_3, aspect_ratio]:
            control.change(
                _size_preview_for_mode, inputs=size_preview_inputs, outputs=size_info
            )
        lora_upload.upload(
            _upload_loras_for_mode,
            inputs=[mode, model_key, swap_model, lora_upload, *lora_names],
            outputs=[lora_status, *lora_names],
        )
        unload_button.click(model_manager.unload, outputs=status)
        clear_button.add(
            [
                source,
                references,
                combine_1,
                combine_2,
                combine_3,
                swap_body,
                swap_reference,
                krea_source,
                krea_reference,
                mask,
                prompt,
                negative_prompt,
                output,
                metadata,
            ]
        )
        clear_button.click(
            lambda: _progress_markup("Ready to create", 0, 0),
            outputs=progress_card, show_progress="hidden",
        )
        app.load(_model_changed, inputs=model_key, outputs=model_outputs)
    return app
