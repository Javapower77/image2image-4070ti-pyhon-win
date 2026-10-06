from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from unittest.mock import Mock

import pytest
from PIL import Image

from photo_edit_studio import ui
from photo_edit_studio.comfy_assets import KREA_FIRST_LORA_FILE
from photo_edit_studio.comfy_workflows import krea_reference_template, qwen21_turbo_template
from photo_edit_studio.config import settings
from photo_edit_studio.loras import family_for, selected_loras
from photo_edit_studio.models.comfy_swap import (
    apply_krea_loras,
    configure_krea_reference_graph,
    configure_qwen21_graph,
)
from photo_edit_studio.models.diffusers_adapters import QwenAdapter
from photo_edit_studio.models.registry import MODEL_SPECS
from photo_edit_studio.swap import swap_lora, swap_profile
from photo_edit_studio.types import GenerationRequest, LoraSpec

PROFILES = ("qwen-2.1-turbo", "qwen-2.1-turbo-r128")
FAMILIES = ("qwen-2511", "qwen-2511-aio", "flux-klein-4b", "krea-2-turbo", *PROFILES)
WEIGHTS = [-2.0, -0.5, 0.0, 1.0, 2.0]
INVALID = [-2.1, 2.1, float("nan"), float("inf"), float("-inf")]


def selection(root: Path, model_key: str, weights: list[float]) -> list[LoraSpec]:
    family = root / family_for(model_key)
    family.mkdir(parents=True, exist_ok=True)
    names = [f"slot{index}.safetensors" for index in range(len(weights))]
    for name in names:
        (family / name).touch()
    return selected_loras(model_key, names, weights, root=root)


def request(model_key: str, workflow: str = "standard") -> GenerationRequest:
    count = 0 if workflow == "text" else 2 if workflow == "swap" else 1
    return GenerationRequest(
        model_key=model_key, workflow=workflow, prompt="edit", negative_prompt="",
        images=[Image.new("RGB", (64, 64)) for _ in range(count)], mask=None,
        width=1024, height=1024, steps=6, guidance=1, true_cfg=1, strength=1, seed=1,
        swap_kind="Head" if workflow == "swap" else None,
    )


@pytest.mark.parametrize("model_key", FAMILIES)
def test_picker_preserves_signed_weights_and_skips_only_zero(tmp_path, model_key):
    loras = selection(tmp_path, model_key, WEIGHTS)
    assert [lora.weight for lora in loras] == [-2, -0.5, 1, 2]
    assert [lora.name for lora in loras] == [f"slot{i}.safetensors" for i in (0, 1, 3, 4)]
    assert [lora.adapter_name for lora in loras] == ["slot0_1", "slot1_2", "slot3_4", "slot4_5"]


@pytest.mark.parametrize("model_key", FAMILIES)
@pytest.mark.parametrize("weight", INVALID)
def test_picker_rejects_nonfinite_and_out_of_range(tmp_path, model_key, weight):
    with pytest.raises(ValueError, match="finite and between -2 and 2"):
        selection(tmp_path, model_key, [weight])


def test_five_optional_sliders_are_signed_and_required_controls_unchanged(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "lora_dir", tmp_path)
    sliders = {
        component["props"]["label"]: component["props"]
        for component in ui.build_app().config["components"] if component["type"] == "slider"
    }
    for index in range(1, 6):
        props = sliders[f"Weight {index}"]
        assert (props["minimum"], props["maximum"], props["step"], props["value"]) == (-2, 2, 0.05, 1)
    first = sliders["Krea2_ALWAYS_LOAD_FIRST · mandatory weight"]
    assert (first["minimum"], first["maximum"]) == (0.05, 2)
    bfs = sliders["BFS adapter weight"]
    assert (bfs["minimum"], bfs["maximum"]) == (0.1, 1.5)


@pytest.mark.parametrize("model_key", PROFILES)
@pytest.mark.parametrize("workflow", ["standard", "text", "swap"])
def test_qwen_signed_chain_preserves_turbo_and_bfs(tmp_path, monkeypatch, model_key, workflow):
    monkeypatch.setattr(settings, "lora_dir", tmp_path)
    req = request(model_key, workflow)
    req.loras = selection(tmp_path, model_key, WEIGHTS)
    if workflow == "swap":
        (tmp_path / "qwen21" / swap_profile(model_key, "Head").filename).touch()
        req.loras.insert(0, swap_lora(model_key, "Head", 0.7))
    graph = qwen21_turbo_template(model_key)
    required = {key: deepcopy(graph[key]) for key in ("2", "9", "10")}
    configure_qwen21_graph(graph, req, tuple(f"{i}.png" for i in range(len(req.images))))
    assert all(graph[key] == value for key, value in required.items())
    current = graph["8"]["inputs"]["model"][0]
    for lora in reversed(req.loras):
        inputs = graph[current]["inputs"]
        assert inputs["lora_name"].replace("\\", "/") == f"qwen21/{lora.name}"
        assert inputs["strength_model"] == lora.weight
        current = inputs["model"][0]
    assert current == "2"
    assert not any(node["inputs"].get("lora_name", "").endswith("slot2.safetensors")
                   for node in graph.values())


@pytest.mark.parametrize("model_key", PROFILES)
@pytest.mark.parametrize("weight", INVALID)
def test_qwen_direct_request_validates_optional_weight(tmp_path, monkeypatch, model_key, weight):
    monkeypatch.setattr(settings, "lora_dir", tmp_path)
    path = tmp_path / "qwen21" / "style.safetensors"
    path.parent.mkdir()
    path.touch()
    req = request(model_key)
    req.loras = [LoraSpec(path.name, path, weight, "style")]
    with pytest.raises(ValueError, match="finite and between -2 and 2"):
        configure_qwen21_graph(qwen21_turbo_template(model_key), req, ("source.png",))


@pytest.mark.parametrize("model_key", PROFILES)
def test_qwen_direct_zero_optional_is_not_chained(tmp_path, monkeypatch, model_key):
    monkeypatch.setattr(settings, "lora_dir", tmp_path)
    lora = selection(tmp_path, model_key, [-0.5])[0]
    req = request(model_key)
    req.loras = [LoraSpec(lora.name, lora.path, 0, lora.adapter_name)]
    graph = configure_qwen21_graph(qwen21_turbo_template(model_key), req, ("source.png",))
    assert graph["8"]["inputs"]["model"] == ["2", 0]
    assert "40" not in graph


def test_krea_reference_signed_chain_and_base_override(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "lora_dir", tmp_path)
    monkeypatch.setattr(settings, "comfy_dir", tmp_path / "comfy")
    first = settings.comfy_dir / "models" / "loras" / KREA_FIRST_LORA_FILE
    first.parent.mkdir(parents=True)
    first.touch()
    req = request("krea-2-turbo", "krea-reference")
    req.loras = selection(tmp_path, req.model_key, WEIGHTS)
    base = tmp_path / "krea2" / settings.comfy_reference_lora
    base.touch()
    req.loras.insert(0, LoraSpec(base.name, base, -0.5, "base"))
    graph = configure_krea_reference_graph(krea_reference_template(), req, ("source.png",))
    assert graph["71"]["inputs"]["strength_model"] == -0.5
    current = graph["79"]["inputs"]["model"][0]
    for lora in reversed(req.loras[1:]):
        inputs = graph[current]["inputs"]
        assert (inputs["lora_name"], inputs["strength_model"]) == (lora.name, lora.weight)
        current = inputs["model"][0]
    assert current == "71"
    mandatory = graph["71"]["inputs"]["model"][0]
    assert graph[mandatory]["inputs"]["strength_model"] == 1


def test_direct_krea_zero_does_not_disable_base_or_add_node(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "lora_dir", tmp_path)
    lora = selection(tmp_path, "krea-2-turbo", [-0.5])[0]
    base = lora.path.parent / settings.comfy_reference_lora
    base.touch()
    graph = krea_reference_template()
    before = deepcopy(graph)
    apply_krea_loras(graph, "71", [LoraSpec(base.name, base, 0, "base"),
                                   LoraSpec(lora.name, lora.path, 0, "style")])
    assert graph == before


@pytest.mark.parametrize("model_key", ["qwen-2511", "flux-klein-4b"])
def test_diffusers_receives_signed_weights_in_slot_order_and_reloads(tmp_path, model_key):
    # Exercise the common Diffusers method without loading weights or running inference.
    adapter = QwenAdapter(MODEL_SPECS[model_key])
    adapter.pipe = Mock()
    req = request(model_key)
    req.loras = selection(tmp_path, model_key, WEIGHTS)
    adapter.apply_loras(req)
    assert adapter.pipe.load_lora_weights.call_count == 4
    assert [call.kwargs["adapter_name"] for call in adapter.pipe.load_lora_weights.call_args_list] == [
        lora.adapter_name for lora in req.loras
    ]
    adapter.pipe.set_adapters.assert_called_once_with(
        [lora.adapter_name for lora in req.loras], adapter_weights=[-2, -0.5, 1, 2],
    )
    adapter.apply_loras(req)
    assert adapter.pipe.set_adapters.call_count == 1
    req.loras = selection(tmp_path, model_key, [2])
    adapter.apply_loras(req)
    adapter.pipe.unload_lora_weights.assert_called_once()
    adapter.pipe.set_adapters.assert_called_with(["slot0_1"], adapter_weights=[2])