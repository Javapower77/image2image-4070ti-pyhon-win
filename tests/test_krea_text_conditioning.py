from types import SimpleNamespace

import pytest

from photo_edit_studio import ui
from photo_edit_studio.comfy_workflows import krea_text_template
from photo_edit_studio.models import comfy_swap
from photo_edit_studio.models.registry import MODEL_SPECS


def test_krea_ui_defaults_keep_positive_conditioning():
    assert MODEL_SPECS["krea-2-turbo"].default_guidance == 1.0
    assert ui._model_changed("krea-2-turbo")[1]["value"] == 1.0
    assert ui._turbo_controls_for_model(ui.TEXT_MODE, "krea-2-turbo", "qwen-2511")[1]["value"] == 1.0
    assert ui._turbo_controls_for_model(ui.SWAP_MODE, "qwen-2511", "krea-2-turbo")[1]["value"] == 1.0


@pytest.mark.parametrize("cfg", [0.0, -1.0, float("nan"), float("inf")])
def test_krea_text_rejects_prompt_discarding_cfg(cfg):
    request = SimpleNamespace(model_key="krea-2-turbo", images=[], guidance=cfg)
    with pytest.raises(ValueError, match="CFG greater than 0"):
        comfy_swap.configure_krea_text_graph(krea_text_template(), request)


@pytest.mark.parametrize("cfg", [0.5, 1.0, 2.0])
def test_krea_text_preserves_prompt_and_positive_connection(monkeypatch, cfg):
    monkeypatch.setattr(comfy_swap, "apply_mandatory_krea_lora", lambda graph, request: graph)
    monkeypatch.setattr(comfy_swap, "apply_krea_loras", lambda graph, anchor, loras: graph)
    request = SimpleNamespace(model_key="krea-2-turbo", images=[], guidance=cfg,
                              prompt="An indoor ceramic figurine on a wooden table",
                              negative_prompt="", seed=42, steps=8, width=1024,
                              height=1024, loras=[])
    graph = comfy_swap.configure_krea_text_graph(krea_text_template(), request)
    assert graph["84"]["inputs"]["text"] == request.prompt
    assert graph["53"]["inputs"]["positive"] == ["84", 0]
    assert graph["53"]["inputs"]["cfg"] == cfg
