from __future__ import annotations

import inspect
from xml.etree import ElementTree

import pytest
from PIL import Image

from photo_edit_studio.models import MODEL_SPECS
from photo_edit_studio.ui import (
    COMBINE_MODE,
    CSS,
    EDIT_MODE,
    KREA_EDIT_MODE,
    SHEET_MODE,
    SWAP_MODE,
    TEXT_MODE,
    _close_popovers,
    _mode_changed,
    _model_icon_file,
    _model_icon_for_mode,
    _model_picker_visibility,
    _model_toolbar_label,
    _popover_updates,
    _progress_markup,
    _size_preview_for_mode,
    _standard_model_changed,
    _stream_click,
    _stream_run,
    _toggle_popover,
    _toolbar_mode_updates,
    _workflow_buttons,
    build_app,
)


def test_workflow_icons_and_layout_wired_into_generation() -> None:
    app = build_app()
    components = app.config["components"]
    component_by_id = {component["id"]: component for component in components}
    buttons = {
        component["props"].get("value"): component
        for component in components if component["type"] == "button"
    }
    workflows = (EDIT_MODE, COMBINE_MODE, TEXT_MODE, KREA_EDIT_MODE, SWAP_MODE, SHEET_MODE)
    assert all(buttons[key]["props"]["icon"]["path"].endswith(".svg") for key in workflows)
    assert "grid-template-areas" in CSS
    assert "'top options'" in CSS
    assert "'description options'" in CSS
    assert "#studio-top {grid-area:top" in CSS
    assert "#studio-progress {flex:1 1 0" in CSS
    assert "#studio-progress .styler, #studio-progress-card" in CSS
    assert "#studio-options {grid-area:options" in CSS
    assert "#studio-composer {grid-area:prompt" in CSS
    assert "#studio-model-info {width:100%!important" in CSS
    assert "#studio-negative {width:100%!important" in CSS
    assert "background:#1b1913" in CSS
    assert 'content:"➜"' not in CSS
    assert "#studio-options {grid-area:options; align-self:start; width:100%!important; margin:0!important; background:transparent" in CSS
    assert "#studio-options .gr-accordion .form, #studio-options .gr-accordion .block, #studio-options .gr-accordion .wrap {background:#fff" in CSS
    assert "#studio-options .gr-accordion .form {gap:0!important" in CSS
    assert "#studio-accordion-loras {--studio-acc-icon:" in CSS
    assert "#studio-accordion-face {--studio-acc-icon:" in CSS
    assert "#studio-accordion-other {--studio-acc-icon:" in CSS
    assert {c["props"].get("elem_id") for c in components if c["type"] == "accordion"} >= {
        "studio-accordion-loras", "studio-accordion-face", "studio-accordion-other"
    }
    assert ".run-progress progress {width:100%!important" in CSS
    assert "grid-template-columns:minmax(0,7fr) minmax(0,3fr)" in CSS
    assert "button.workflow-choice" in CSS
    assert len([x for x in components if x["type"] == "group" and "tool-panel" in x["props"].get("elem_classes", [])]) == 6
    assert all(
        component["props"]["visible"] is False
        for component in components if component["type"] == "group"
        and "tool-panel" in component["props"].get("elem_classes", [])
    )
    assert all(buttons[name]["props"].get("icon") for name in ("Settings", "Variants", "Size", "Replace", "BFS weight"))
    assert all(next(c for c in components if c["type"] == "group" and c["props"].get("elem_id") == f"studio-panel-{name}")["props"]["visible"] is False
               for name in ("settings", "variants", "size", "model", "swap", "bfs"))
    assert component_by_id[next(c["id"] for c in components if c["props"].get("label") == "Outputs (maximum 1)")]["props"]["interactive"] is False
    for name in ("Generate edit", "Generate swap", "Generate Krea edit", "Generate character sheet"):
        target = buttons[name]["id"]
        assert buttons[name]["props"]["icon"]["path"].endswith("generate.svg")
        events = [event for event in app.config["dependencies"] if (target, "click") in event["targets"]]
        assert len(events) == 1
        assert len(events[0]["outputs"]) == 4  # gallery, run details, model status, progress
        assert inspect.isgeneratorfunction(app.fns[events[0]["id"]].fn)
    workflow_state = next(c["id"] for c in components if c["type"] == "state")
    mode_event = next(event for event in app.config["dependencies"] if (workflow_state, "change") in event["targets"])
    assert len(mode_event["outputs"]) == len(set(mode_event["outputs"]))
    details = next(c for c in components if c["props"].get("elem_id") == "studio-run-details")
    trigger = next(c for c in components if c["props"].get("elem_id") == "studio-details-trigger")
    assert details["type"] == "column"
    assert not details["props"]["visible"]
    assert trigger["type"] == "button" and trigger["props"]["icon"]["path"].endswith("run-details.svg")
    assert any((trigger["id"], "click") in event["targets"] and details["id"] in event["outputs"]
               for event in app.config["dependencies"])


def test_reference_svg_assets_are_valid() -> None:
    from photo_edit_studio.config import ROOT

    assets = ROOT / "photo_edit_studio" / "assets" / "ui"
    for name in ("edit", "combine", "text", "krea", "swap", "notebook", "settings", "variants", "size", "weight", "flux", "firered", "generate", "qwen"):
        path = assets / f"{name}.svg"
        assert path.is_file()
        svg = ElementTree.parse(path).getroot()
        assert svg.tag.endswith("svg")
        assert svg.attrib.get("viewBox")
    assert ElementTree.parse(assets / "qwen.svg").getroot().attrib["viewBox"] == "0 0 128 128"


def test_popovers_open_one_at_a_time_and_toggle_closed() -> None:
    current, *panels = _toggle_popover("model", None)
    assert current == "model"
    assert [panel["visible"] for panel in panels] == [False, False, False, True, False, False]
    current, *panels = _toggle_popover("size", current)
    assert current == "size" and panels[2]["visible"] is True and panels[3]["visible"] is False
    current, *panels = _toggle_popover("size", current)
    assert current is None and not any(panel["visible"] for panel in panels)
    assert len(_close_popovers()) == 7
    assert len(_popover_updates(None)) == 6


@pytest.mark.parametrize("mode", [EDIT_MODE, COMBINE_MODE, TEXT_MODE, KREA_EDIT_MODE, SWAP_MODE, SHEET_MODE])
def test_toolbar_reacts_to_each_workflow(mode: str) -> None:
    updates = _toolbar_mode_updates(mode)
    assert len(updates) == 14
    assert updates[0]["value"]
    assert updates[1]["elem_classes"] == (["toolbar-icon"] if mode == SWAP_MODE else ["toolbar-icon", "mode-hidden"])
    assert updates[3]["visible"] is (mode not in {COMBINE_MODE, TEXT_MODE, SHEET_MODE})
    assert updates[4]["visible"] is (mode in {COMBINE_MODE, TEXT_MODE})
    assert all(button["variant"] == ("primary" if key == mode else "secondary")
               for key, button in zip((EDIT_MODE, COMBINE_MODE, TEXT_MODE, KREA_EDIT_MODE, SWAP_MODE, SHEET_MODE), _workflow_buttons(mode), strict=True))


def test_model_icon_and_label_follow_active_selection() -> None:
    assert _model_toolbar_label(KREA_EDIT_MODE, "qwen-2511", "flux-klein-4b") == "Krea · fixed"
    assert "Qwen" in _model_toolbar_label(EDIT_MODE, "qwen-2.1-turbo", "flux-klein-4b")
    assert "FLUX" in _model_toolbar_label(SWAP_MODE, "qwen-2511", "flux-klein-4b")
    assert _model_icon_for_mode(EDIT_MODE, "firered-1.1", "qwen-2511")["icon"].name == "firered.svg"
    assert _model_icon_file("qwen-2511-aio") == "qwen.svg"
    assert _model_icon_file("krea-2-turbo") == "krea.svg"


def test_model_picker_rows_use_brand_icons() -> None:
    app = build_app()
    buttons = {
        component["props"].get("value"): component
        for component in app.config["components"]
        if component["type"] == "button"
    }
    expected = {
        "krea-2-turbo": "krea.svg",
        "qwen-2511": "qwen.svg",
        "qwen-2511-aio": "qwen.svg",
        "qwen-2.1-turbo": "qwen.svg",
        "firered-1.1": "firered.svg",
        "flux-klein-4b": "flux.svg",
    }
    for key, icon in expected.items():
        button = buttons[MODEL_SPECS[key].label]
        assert "model-choice" in button["props"].get("elem_classes", [])
        assert button["props"]["icon"]["path"].endswith(icon)
    assert "#studio-prompt-toolbar button.toolbar-icon:hover" in CSS
    assert "#studio-prompt-toolbar button.toolbar-icon:hover, #studio-prompt-toolbar button.toolbar-icon[aria-expanded=\"true\"] {background:#f0efec" in CSS
    assert "#studio-panel-model button.model-choice:hover {background:#f0efec" in CSS
    assert '#studio-panel-model .block:has(input[role="combobox"]) {display:none!important;}' in CSS
    assert "#studio-panel-model .dropdown" not in CSS


def test_model_picker_visibility_follows_workflow() -> None:
    edit = _model_picker_visibility(EDIT_MODE, "qwen-2511-aio", "qwen-2511")
    text = _model_picker_visibility(TEXT_MODE, "krea-2-turbo", "qwen-2511")
    swap = _model_picker_visibility(SWAP_MODE, "qwen-2511-aio", "flux-klein-4b")
    krea = _model_picker_visibility(KREA_EDIT_MODE, "qwen-2511-aio", "qwen-2511")
    keys = list(MODEL_SPECS)
    from photo_edit_studio.ui import SWAP_MODEL_KEYS

    assert len(edit) == len(keys) + len(SWAP_MODEL_KEYS)
    assert "mode-hidden" not in edit[keys.index("qwen-2511-aio")]["elem_classes"]
    assert edit[keys.index("qwen-2511-aio")]["variant"] == "primary"
    assert "mode-hidden" in edit[keys.index("krea-2-turbo")]["elem_classes"]
    assert "mode-hidden" not in text[keys.index("krea-2-turbo")]["elem_classes"]
    assert "mode-hidden" in text[keys.index("qwen-2511-aio")]["elem_classes"]
    assert all("mode-hidden" in update["elem_classes"] for update in swap[:len(keys)])
    assert "mode-hidden" not in swap[len(keys) + 1]["elem_classes"]
    assert swap[len(keys) + 1]["variant"] == "primary"
    assert all("mode-hidden" in update["elem_classes"] for update in krea)


@pytest.mark.parametrize("mode", [EDIT_MODE, COMBINE_MODE, TEXT_MODE, KREA_EDIT_MODE, SWAP_MODE, SHEET_MODE])
def test_native_model_dropdowns_stay_hidden_for_every_workflow(mode: str) -> None:
    changes = _mode_changed(mode)
    assert len(changes) == 17
    assert changes[10]["elem_classes"] == ["mode-hidden"]
    assert changes[11]["elem_classes"] == ["mode-hidden"]


def test_visual_popover_layer_loaded_only_for_ui() -> None:
    from photo_edit_studio.config import ROOT

    path = ROOT / "photo_edit_studio" / "assets" / "ui" / "studio-popovers.js"
    source = path.read_text(encoding="utf-8")
    assert "studio-panel-${name}" in source
    assert "Escape" in source and "aria-expanded" in source
    assert "requestAnimationFrame" in source


def test_size_popover_uses_the_active_workflow_source() -> None:
    edit = Image.new("RGB", (1024, 512))
    krea = Image.new("RGB", (512, 1024))
    swap = Image.new("RGB", (768, 1024))
    args = (1, "1K", "1:1", edit, krea, swap, None, None, None)
    assert "1024 × 512" in _size_preview_for_mode(EDIT_MODE, *args)
    assert "512 × 1024" in _size_preview_for_mode(KREA_EDIT_MODE, *args)
    assert "768 × 1024" in _size_preview_for_mode(SWAP_MODE, *args)
    assert "1024 × 1024" in _size_preview_for_mode(TEXT_MODE, *args)


def test_model_refresh_does_not_reset_combine_to_edit() -> None:
    assert "value" not in _standard_model_changed("qwen-2511", COMBINE_MODE)[5]
    assert "value" not in _standard_model_changed("qwen-2511", EDIT_MODE)[5]
    assert _standard_model_changed("qwen-2.1-turbo", TEXT_MODE)[5]["value"] == TEXT_MODE


@pytest.mark.parametrize(
    ("mode", "submit_position"),
    [(EDIT_MODE, 5), (COMBINE_MODE, 5), (TEXT_MODE, 5), (SWAP_MODE, 9), (KREA_EDIT_MODE, 12), (SHEET_MODE, 16)],
)
def test_every_workflow_has_a_visible_submit_button(mode: str, submit_position: int) -> None:
    changes = _mode_changed(mode)
    assert len(changes) == 17
    if submit_position == 5:
        assert "mode-hidden" not in changes[5]["elem_classes"]
    else:
        assert "mode-hidden" not in changes[submit_position]["elem_classes"]
    if mode in {SWAP_MODE, KREA_EDIT_MODE, SHEET_MODE}:
        assert "mode-hidden" in changes[5]["elem_classes"]
    else:
        assert "mode-hidden" in changes[9]["elem_classes"]


def test_stream_click_is_a_generator_function() -> None:
    def generate(*_args: object, progress: object) -> tuple[list[str], str, str]:
        return ["image"], "{}", "Loaded"

    click = _stream_click(generate)
    assert inspect.isgeneratorfunction(click)
    events = list(click())
    assert events[-1][:3] == (["image"], "{}", "Loaded")


def test_streamed_progress_and_failure_keep_inference_contract() -> None:
    def generate(*_args: object, progress: object) -> tuple[list[str], str, str]:
        progress(0.5, desc="Sampling step")
        return ["image"], "{}", "Loaded"

    events = list(_stream_run(generate))
    assert "Sampling step" in events[1][3]
    assert events[-1][:3] == (["image"], "{}", "Loaded")
    assert "100%" in events[-1][3]
    assert "&lt;script&gt;" in _progress_markup("<script>", 0.5, 1)

    def fail(*_args: object, progress: object) -> None:
        raise ValueError("Invalid model")

    result = _stream_run(fail)
    next(result)
    assert "Failed: Invalid model" in next(result)[3]
    with pytest.raises(ValueError, match="Invalid model"):
        next(result)
