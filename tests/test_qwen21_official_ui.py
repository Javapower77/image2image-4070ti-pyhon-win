from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from photo_edit_studio import models, ui
from photo_edit_studio.config import settings
from photo_edit_studio.dlss import DLSS_DEFAULTS
from photo_edit_studio.loras import NONE_CHOICE
from photo_edit_studio.models.diffusers_adapters import (
    Qwen21OfficialAdapter,
    Qwen21OfficialExtractAdapter,
)
from photo_edit_studio.models.registry import MODEL_SPECS
from photo_edit_studio.swap import swap_profile
from photo_edit_studio.types import GenerationRequest, GenerationResult, LoraSpec

KEY = "qwen-2.1-turbo-official"
MODES = (ui.EDIT_MODE, ui.COMBINE_MODE, ui.TEXT_MODE)


@pytest.fixture(autouse=True, params=["qwen-2.1-turbo-official", "qwen-2.1-turbo-official-extract"])
def official_profile(request, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(globals(), "KEY", request.param)


def test_official_spec_and_manager_selection_never_load_weights(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*_args: object, **_kwargs: object) -> None:
        pytest.fail("Model selection must not load weights or initialize a runtime")

    monkeypatch.setattr(models, "release_cuda", lambda: None)
    monkeypatch.setattr(Qwen21OfficialAdapter, "load", forbidden)
    monkeypatch.setattr(Qwen21OfficialExtractAdapter, "load", forbidden)
    spec = MODEL_SPECS[KEY]
    extracted = KEY.endswith("-extract")
    assert (spec.key, spec.repo_id, spec.family, spec.loader, spec.task) == (
        KEY, "Qwen/Qwen-Image-2.1-Turbo", "qwen21-official",
        "qwen21_official_extract" if extracted else "qwen21_official", "image-to-image",
    )
    assert (spec.default_steps, spec.default_guidance, spec.default_true_cfg, spec.max_images) == (8, 1, 1, 3)
    manager = models.ModelManager()
    adapter = manager.get(KEY)
    assert type(adapter) is (Qwen21OfficialExtractAdapter if extracted else Qwen21OfficialAdapter)
    assert adapter.spec is spec
    assert adapter.pipe is None
    assert manager.get(KEY) is adapter


@pytest.mark.parametrize("mode", MODES)
def test_official_is_selectable_and_visible_in_supported_modes(mode: str) -> None:
    choices = ui._model_choices_for_mode(mode, KEY)
    assert choices["value"] == KEY
    assert (MODEL_SPECS[KEY].label, KEY) in choices["choices"]
    assert ui._model_icon_for_mode(mode, KEY, "flux-klein-4b")["icon"].name == "qwen.svg"
    visibility = ui._model_picker_visibility(mode, KEY, "flux-klein-4b")
    selected = visibility[list(MODEL_SPECS).index(KEY)]
    assert "mode-hidden" not in selected["elem_classes"]
    assert selected["variant"] == "primary"


@pytest.mark.parametrize("mode", [ui.SWAP_MODE, ui.KREA_EDIT_MODE, ui.SHEET_MODE])
def test_official_is_excluded_from_unsupported_modes_and_swaps(mode: str) -> None:
    assert (KEY in ui.SWAP_MODEL_KEYS) is (not KEY.endswith("-extract"))
    assert KEY not in ui._model_keys_for_mode(mode)
    selected = ui._model_picker_visibility(mode, KEY, "flux-klein-4b")[list(MODEL_SPECS).index(KEY)]
    assert "mode-hidden" in selected["elem_classes"]


@pytest.mark.parametrize("kind", ["Head", "Body"])
def test_unstacked_official_has_separate_swap_picker_and_locked_defaults(kind: str) -> None:
    key = "qwen-2.1-turbo-official"
    visibility = ui._model_picker_visibility(ui.SWAP_MODE, "qwen-2511", key)
    selected = visibility[len(MODEL_SPECS) + ui.SWAP_MODEL_KEYS.index(key)]
    assert "mode-hidden" not in selected["elem_classes"]
    assert selected["variant"] == "primary"
    changes = ui._swap_model_changed(key, kind)
    assert changes[0]["choices"] == ["Head", "Body"] and changes[0]["value"] == kind
    assert [change["value"] for change in changes[1:4]] == [8, 1, 1]
    assert all(slot["choices"] == [NONE_CHOICE] and slot["value"] == NONE_CHOICE
               and slot["interactive"] is False for slot in changes[6:])
    controls = ui._turbo_controls_for_model(ui.SWAP_MODE, "flux-klein-4b", key)
    assert [control["value"] for control in controls] == [8, 1, 1, ""]
    assert all(control["interactive"] is False for control in controls)
    extras = ui._official_extras_for_model(ui.SWAP_MODE, "flux-klein-4b", key)
    assert all(control["interactive"] is False for control in extras[:-1])
    assert extras[1]["value"] is None
    assert extras[7 + list(DLSS_DEFAULTS).index("enabled")]["value"] is False
    assert extras[-1] is None
    assert ui._dlss_controls_for_mode(ui.SWAP_MODE, "flux-klein-4b", key) is None
    assert "mode-hidden" not in ui._prompt_tool_visibility(ui.SWAP_MODE)[1]["elem_classes"]


@pytest.mark.parametrize("mode,default", [
    (ui.EDIT_MODE, "qwen-2511"), (ui.COMBINE_MODE, "qwen-2511"), (ui.TEXT_MODE, "krea-2-turbo"),
])
def test_official_inclusion_does_not_replace_default_fallback(mode: str, default: str) -> None:
    assert ui._model_choices_for_mode(mode, "unknown-model")["value"] == default
    assert ui._model_choices_for_mode(mode, default)["value"] == default


@pytest.mark.parametrize("mode", MODES)
def test_official_locks_eight_steps_unit_guidance_cfg_and_empty_negative(mode: str) -> None:
    controls = ui._turbo_controls_for_model(mode, KEY, "flux-klein-4b")
    assert [control["value"] for control in controls] == [8, 1.0, 1.0, ""]
    assert all(control["interactive"] is False for control in controls)
    unlocked = ui._turbo_controls_for_model(mode, "flux-klein-4b", "flux-klein-4b")
    assert all(control["interactive"] is True for control in unlocked)


@pytest.mark.parametrize("mode", MODES)
def test_official_clears_and_disables_optional_loras_uploads_and_dlss(mode: str) -> None:
    changes = ui._standard_model_changed(KEY, mode)
    assert [change["value"] for change in changes[:3]] == [8, 1, 1]
    slots = changes[6:]
    assert len(slots) == 5
    assert all(slot["choices"] == [NONE_CHOICE] and slot["value"] == NONE_CHOICE
               and slot["interactive"] is False for slot in slots)
    extras = ui._official_extras_for_model(mode, KEY, "flux-klein-4b")
    strength, upload, *remaining = extras
    weights, dlss_controls, state = remaining[:5], remaining[5:-1], remaining[-1]
    assert strength["interactive"] is False
    assert upload["interactive"] is False and upload["value"] is None
    assert len(weights) == 5 and all(weight["interactive"] is False for weight in weights)
    assert len(dlss_controls) == len(DLSS_DEFAULTS)
    assert all(control["interactive"] is False for control in dlss_controls)
    assert dlss_controls[list(DLSS_DEFAULTS).index("enabled")]["value"] is False
    assert state is None  # Disabled official DLSS is None, not {"enabled": False}.
    assert ui._dlss_controls_for_mode(mode, KEY, "flux-klein-4b") is None
    unlocked = ui._official_extras_for_model(mode, "flux-klein-4b", "flux-klein-4b")
    assert all(control["interactive"] is True for control in unlocked[:-1])


def test_real_app_has_official_picker_buttons_and_independent_bfs_lock_event_handlers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    monkeypatch.setattr(settings, "lora_dir", tmp_path)
    app = ui.build_app()
    buttons = [component for component in app.config["components"]
               if component["type"] == "button" and component["props"].get("value") == MODEL_SPECS[KEY].label]
    assert len(buttons) == (1 if KEY.endswith("-extract") else 2)
    assert "model-choice" in buttons[0]["props"]["elem_classes"]
    assert buttons[0]["props"]["icon"]["path"].endswith("qwen.svg")
    bfs = [component for component in app.config["components"]
           if component["type"] == "slider"
           and component["props"].get("label") == "BFS adapter weight"]
    assert len(bfs) == 1
    assert bfs[0]["props"].get("interactive") is not False
    assert (bfs[0]["props"]["minimum"], bfs[0]["props"]["maximum"],
            bfs[0]["props"]["value"]) == (0.1, 1.5, 1.0)
    for name in ("_turbo_controls_for_model", "_official_extras_for_model"):
        handlers = [fn for fn in app.fns.values() if getattr(fn.fn, "__name__", None) == name]
        assert len(handlers) == 3  # Mode, standard model and swap-model change chains.
        assert all(len(handler.inputs) == 3 for handler in handlers)
        expected_outputs = 4 if name == "_turbo_controls_for_model" else 8 + len(DLSS_DEFAULTS)
        assert all(len(handler.outputs) == expected_outputs for handler in handlers)
        assert all(bfs[0]["id"] not in [output._id for output in handler.outputs]
               for handler in handlers)


@pytest.mark.parametrize("mode", MODES)
def test_ui_dispatch_preserves_order_and_uses_official_request_defaults(
    mode: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    monkeypatch.setattr(settings, "lora_dir", tmp_path)
    seen: list[GenerationRequest] = []

    def fake_generate(req: GenerationRequest, *_args: object, **_kwargs: object) -> GenerationResult:
        seen.append(req)
        return GenerationResult([Image.new("RGB", (64, 64))], req.seed, 0, KEY)

    monkeypatch.setattr(ui, "generate", fake_generate)
    images = [Image.new("RGB", (64, 64), color) for color in ("red", "green", "blue")]
    ui._run(
        mode=mode, source=images[0], references=images[1:],
        combine_1=images[0], combine_2=images[1], combine_3=images[2], mask=None,
        model_key=KEY, prompt="Change colors", negative_prompt="", size_multiplier=2,
        output_resolution="2K", output_aspect_ratio="9:16", steps=8, guidance=1, true_cfg=1,
        strength=0.8, seed=19, count=1, preserve_identity=True,
        restoration="Off", restore_weight=0.35, progress=lambda *_args, **_kwargs: None,
    )
    assert len(seen) == 1
    req = seen[0]
    assert req.model_key == KEY
    assert req.images == ([] if mode == ui.TEXT_MODE else images)
    assert req.workflow == ("text" if mode == ui.TEXT_MODE else "standard")
    assert req.compose is (mode != ui.EDIT_MODE)
    assert (req.steps, req.guidance, req.true_cfg, req.negative_prompt) == (8, 1, 1, "")
    assert req.loras == [] and req.dlss is None
    assert (req.output_resolution, req.output_aspect_ratio, req.size_multiplier) == ("2K", "9:16", 2)


def test_official_extracted_swap_and_all_uploads_reject_before_downstream_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*_args: object, **_kwargs: object) -> None:
        pytest.fail("Unsupported official operation reached downstream work")

    monkeypatch.setattr(ui, "generate", forbidden)
    monkeypatch.setattr(ui, "import_lora_files", forbidden)
    with pytest.raises(ValueError, match="does not support swap"):
        ui._run_swap(None, None, "qwen-2.1-turbo-official-extract", "Head", "colors", "", 2, 8, 1, 1, 0.8, 19,
                     True, "Off", 0.35, 1, 1)
    with pytest.raises(ValueError, match="does not support optional LoRA uploads"):
        ui._upload_loras(KEY, [Path("never-read.safetensors")], *([NONE_CHOICE] * 5))


@pytest.mark.parametrize("kind", ["Head", "Body"])
@pytest.mark.parametrize("weight", [0.1, 0.65, 1.5])
def test_official_swap_dispatch_preserves_sources_and_independent_bfs_weight(
    kind: str, weight: float, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    key = "qwen-2.1-turbo-official"
    monkeypatch.setattr(settings, "lora_dir", tmp_path)
    profile = swap_profile(key, kind)
    seen: list[GenerationRequest] = []

    def mandatory(model_key: str, swap_kind: str, received_weight: float) -> LoraSpec:
        assert (model_key, swap_kind, received_weight) == (key, kind, weight)
        return LoraSpec(profile.filename, tmp_path / "qwen21" / profile.filename, weight, "bfs_swap")

    def fake_generate(req: GenerationRequest, *_args: object, **_kwargs: object) -> GenerationResult:
        seen.append(req)
        return GenerationResult([Image.new("RGB", (64, 64))], req.seed, 0, key)

    monkeypatch.setattr(ui, "swap_lora", mandatory)
    monkeypatch.setattr(ui, "generate", fake_generate)
    monkeypatch.setattr(ui, "selected_loras", lambda *_args: pytest.fail("Official optional slots resolved"))
    body, reference = Image.new("RGB", (128, 192), "red"), Image.new("RGB", (192, 128), "blue")
    ui._run_swap(
        body, reference, key, kind, "Change colors", "", 2, 8, 1, 1, 0.8, 19,
        True, "Off", 0.35, weight, 0.25, *([NONE_CHOICE] * 5), *([0.9] * 5),
        progress=lambda *_args, **_kwargs: None,
    )
    assert len(seen) == 1
    req = seen[0]
    assert req.images == [body, reference]
    assert req.images[0] is body and req.images[1] is reference
    assert (req.model_key, req.workflow, req.swap_kind) == (key, "swap", kind)
    assert (req.steps, req.guidance, req.true_cfg, req.negative_prompt) == (8, 1, 1, "")
    assert len(req.loras) == 1 and req.loras[0].weight == weight
    assert req.loras[0].adapter_name == "bfs_swap"
    assert req.prompt == (
        profile.trigger + " Change colors Preserve the target body's pose, clothing, lighting, and scene."
    )
    assert req.dlss is None and req.mask is None and req.compose is False
    assert req.preserve_identity is False


@pytest.mark.parametrize("extra,dlss,match", [
    (["style.safetensors"] + [NONE_CHOICE] * 4, None, "optional LoRAs"),
    ([NONE_CHOICE] * 5, {"enabled": False}, "DLSS"),
    ([NONE_CHOICE] * 5, {"enabled": True}, "DLSS"),
])
def test_official_swap_extras_block_before_generation(
    extra: list[str], dlss: dict | None, match: str,
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    key = "qwen-2.1-turbo-official"
    monkeypatch.setattr(settings, "lora_dir", tmp_path)
    profile = swap_profile(key, "Head")
    monkeypatch.setattr(ui, "swap_lora", lambda *_args: LoraSpec(
        profile.filename, tmp_path / "qwen21" / profile.filename, 0.65, "bfs_swap",
    ))
    monkeypatch.setattr(ui, "generate", lambda *_args, **_kwargs: pytest.fail("Official extras generated"))
    monkeypatch.setattr(ui, "selected_loras", lambda *_args: pytest.fail("Optional slots resolved"))
    with pytest.raises(ValueError, match=match):
        ui._run_swap(
            Image.new("RGB", (64, 64)), Image.new("RGB", (64, 64)), key, "Head", "colors", "",
            2, 8, 1, 1, 0.8, 19, True, "Off", 0.35, 0.65, 1,
            *extra, *([1.0] * 5), dlss=dlss,
        )