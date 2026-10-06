from types import SimpleNamespace

import pytest
from PIL import Image

from photo_edit_studio import ui


def test_remix_ui_fixes_model_and_ignores_unrelated_controls(monkeypatch):
    calls = []

    def generate(request, restoration, weight, progress):
        calls.append((request, restoration, weight))
        return SimpleNamespace(images=["generated"])

    monkeypatch.setattr(ui, "generate", generate)
    monkeypatch.setattr(ui, "metadata_text", lambda result: "metadata")
    selections = []
    selected = [object() for _ in range(5)]
    monkeypatch.setattr(ui, "selected_loras", lambda model, names, weights:
                        selections.append((model, names, weights)) or selected)
    source = Image.new("RGB", (400, 600))
    result = ui._run_krea_edit(
        "Composition remix", source, Image.new("RGB", (50, 50)), "", "ignored",
        2, 9, 1.0, 666, "GFPGAN", 0.5, 0.8,
        "one", "two", "three", "four", "five", 0.1, 0.2, 0.3, 0.4, 0.5,
        progress=lambda *args, **kwargs: None,
    )
    request, restoration, weight = calls[0]
    assert request.model_key == "krea-2-turbo"
    assert request.workflow == "krea-remix"
    assert request.images == [source]
    assert request.prompt == "remix"
    assert request.negative_prompt == ""
    assert request.size_multiplier == request.count == 1
    assert request.loras is selected
    assert selections == [("krea-2-turbo", ["one", "two", "three", "four", "five"],
                           [0.1, 0.2, 0.3, 0.4, 0.5])]
    assert request.krea_first_lora_weight == 0.8
    assert (restoration, weight) == ("Off", 0.0)
    assert result[:2] == (["generated"], "metadata")


def test_remix_ui_requires_canvas():
    with pytest.raises(ValueError, match="Upload Picture 1.*Composition remix"):
        ui._run_krea_edit(
            "Composition remix", None, None, "remix", "", 1, 9, 1, 666, "Off", 0,
        )


def test_krea_reference_ui_still_delegates(monkeypatch):
    calls = []
    monkeypatch.setattr(ui, "_run_krea_reference", lambda *args, **kwargs: calls.append(args) or "result")
    source = Image.new("RGB", (64, 64))
    assert ui._run_krea_edit(
        "Reference edit", source, None, "edit", "", 1, 10, 1, 666, "Off", 0,
    ) == "result"
    assert calls[0][0] is source


def test_remix_defaults_are_restored_only_in_krea_mode():
    updates = ui._krea_operation_for_mode(ui.KREA_EDIT_MODE, "Composition remix")
    assert updates[0]["visible"] is False
    assert updates[1]["value"] == 9
    assert updates[2]["value"] == 1.0
    assert updates[3]["value"] == 0.0
    assert all("value" not in update for update in ui._krea_operation_for_mode(ui.EDIT_MODE, "Composition remix"))