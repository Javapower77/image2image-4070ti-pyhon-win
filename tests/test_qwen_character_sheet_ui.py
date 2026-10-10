from __future__ import annotations

import inspect
from pathlib import Path
from xml.etree import ElementTree

import pytest
from PIL import Image

from photo_edit_studio import ui
from photo_edit_studio.config import ROOT, settings
from photo_edit_studio.models import MODEL_SPECS
from photo_edit_studio.types import GenerationResult


def test_sixth_mode_exact_notebook_svg_and_selection_event():
    modes = list(ui.WORKFLOW_DESCRIPTIONS)
    assert len(modes) == 6 and modes[-1] == ui.SHEET_MODE
    assert ui.WORKFLOW_ICONS[ui.SHEET_MODE] == "notebook.svg"
    path = ROOT / "photo_edit_studio/assets/ui/notebook.svg"
    svg = ElementTree.parse(path).getroot()
    assert {key: svg.attrib[key] for key in ("width", "height", "viewBox")} == {
        "width": "24", "height": "24", "viewBox": "0 0 24 24"}
    assert svg.attrib["stroke"] == "currentColor"
    assert svg[0].attrib["d"] == "M5 1.5h15a2 2 0 0 1 2 2v17a2 2 0 0 1-2 2H5z"
    app = ui.build_app()
    components = app.config["components"]
    button = next(c for c in components if c["type"] == "button" and c["props"].get("value") == ui.SHEET_MODE)
    cached = Path(button["props"]["icon"]["path"])
    assert cached.name == path.name and cached.read_bytes() == path.read_bytes()
    event = next(e for e in app.config["dependencies"] if (button["id"], "click") in e["targets"])
    assert app.fns[event["id"]].fn() == ui.SHEET_MODE
    state = event["outputs"][0]
    changes = [e for e in app.config["dependencies"] if (state, "change") in e["targets"]]
    assert any(len(e["outputs"]) == 17 for e in changes)
    toolbar_event = next(e for e in app.config["dependencies"]
                         if app.fns[e["id"]].fn is ui._toolbar_mode_updates)
    assert len(toolbar_event["outputs"]) == 14
    assert toolbar_event["inputs"] == [state]
    # Gradio .then chains register dependencies, not another state.change target.
    assert toolbar_event["trigger_after"] is not None
    submit = next(c for c in components if c["type"] == "button" and c["props"].get("value") == "Generate character sheet")
    click = next(e for e in app.config["dependencies"] if (submit["id"], "click") in e["targets"])
    assert len(click["outputs"]) == 4
    assert inspect.isgeneratorfunction(app.fns[click["id"]].fn)


def test_sheet_controls_are_dedicated_and_options_exact():
    app = ui.build_app()
    props = [c["props"] for c in app.config["components"]]
    for choices, default in [(["Simple", "Production"], "Simple"),
                             (["Static", "Auto"], "Static"), ([1.0, 3.4, 6.0], 3.4)]:
        matching = [p for p in props if [choice[1] for choice in p.get("choices", [])] == choices]
        assert len(matching) == 1
        assert matching[0]["value"] == default
    changes = ui._mode_changed(ui.SHEET_MODE)
    assert len(changes) == 17 and "mode-hidden" not in changes[15]["elem_classes"]
    assert "mode-hidden" not in changes[16]["elem_classes"]
    assert all("mode-hidden" in changes[i]["elem_classes"] for i in (0, 1, 2, 3, 6, 7, 14))
    assert all("mode-hidden" in changes[i]["elem_classes"] for i in (5, 9, 12))
    toolbar = ui._toolbar_mode_updates(ui.SHEET_MODE)
    assert len(toolbar) == 14
    assert all("mode-hidden" in toolbar[i]["elem_classes"] for i in (12, 13))
    assert all("mode-hidden" in row["elem_classes"] for row in ui._model_picker_visibility(
        ui.SHEET_MODE, "qwen-2511", "flux-klein-4b"))
    assert "value" not in ui._model_choices_for_mode(ui.SHEET_MODE, "qwen-2511")
    assert "value" not in ui._standard_model_changed("qwen-2511", ui.SHEET_MODE)[5]
    assert ui._effective_model(ui.SHEET_MODE, "qwen-2511", "flux-klein-4b") == "qwen-2.1-sheet"
    assert ui._model_toolbar_label(ui.SHEET_MODE, "qwen-2511", "flux-klein-4b") == "Qwen · fixed"
    assert ui._lora_model_for_mode(ui.SHEET_MODE, "qwen-2511", "flux-klein-4b") == "qwen-2.1-turbo"
    for mode in (ui.EDIT_MODE, ui.COMBINE_MODE, ui.TEXT_MODE, ui.KREA_EDIT_MODE, ui.SWAP_MODE):
        assert "qwen-2.1-sheet" not in ui._model_keys_for_mode(mode)
        assert "mode-hidden" in ui._mode_changed(mode)[15]["elem_classes"]


def test_sheet_ui_request_uses_qwen21_library_and_own_fixed_controls(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "lora_dir", tmp_path)
    library = tmp_path / "qwen21"
    library.mkdir()
    (library / "style.safetensors").touch()
    captured = []

    def generate(req, restoration, weight, **kwargs):
        captured.append(req)
        assert restoration == "Off" and weight == 0
        return GenerationResult([Image.new("RGB", (64, 64))], req.seed, 0.1, req.model_key)

    monkeypatch.setattr(ui, "generate", generate)
    source = Image.new("RGB", (400, 800))
    result = ui._run_qwen_sheet(source, "Production", "Auto", " Name ", " Description ",
        6, " Custom ", "ignored negative", 25, 1, 7,
        "style.safetensors", *([ui.NONE_CHOICE] * 4), *([0.5] + [1.0] * 4),
        progress=lambda *a, **kw: None)
    assert len(result) == 3 and len(captured) == 1
    req = captured[0]
    assert req.model_key == "qwen-2.1-sheet" and req.workflow == "qwen-character-sheet"
    assert req.images == [source] and req.mask is None
    assert (req.width, req.height) == (3072, 2048)
    assert req.steps == 25 and req.guidance == req.true_cfg == 1
    assert req.count == req.size_multiplier == 1 and req.preserve_identity is False
    assert req.negative_prompt == "" and req.prompt == "Custom"
    assert req.sheet_entity_name == "Name" and req.sheet_character_description == "Description"
    assert req.loras[0].path == library / "style.safetensors"
    assert req.loras[0].weight == 0.5
    assert MODEL_SPECS[req.model_key].task == "character-sheet"
    assert MODEL_SPECS[req.model_key].family == "qwen21-sheet"
    with pytest.raises(ValueError, match="Upload one source"):
        ui._run_qwen_sheet(None, "Simple", "Static", "", "", 1, "", "", 25, 1, 1)