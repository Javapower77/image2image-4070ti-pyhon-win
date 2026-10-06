from __future__ import annotations

import json

import httpx
import pytest
from PIL import Image

from photo_edit_studio.comfy_assets import (
    QWEN21_FILES,
    QWEN21_R128_KEY,
    QWEN21_R128_SAMPLER_EXTENSION,
    QWEN21_R128_TURBO_LORA,
    QWEN21_TURBO_LORA,
    missing_qwen21_assets,
)
from photo_edit_studio.comfy_workflows import QWEN21_R128_SIGMAS, qwen21_turbo_template
from photo_edit_studio.config import settings
from photo_edit_studio.models import MODEL_SPECS, ModelManager
from photo_edit_studio.models.comfy_swap import (
    ComfyQwen21Adapter,
    configure_qwen21_graph,
    preflight_qwen21_r128_graph,
)
from photo_edit_studio.swap import swap_profile
from photo_edit_studio.types import GenerationRequest, LoraSpec


def request(images=0, workflow="text", kind=None, compose=False):
    return GenerationRequest(
        model_key=QWEN21_R128_KEY, prompt="Recolor the shirt", negative_prompt="",
        images=[Image.new("RGB", (64, 64)) for _ in range(images)], mask=None,
        width=1024, height=768, steps=6, guidance=1.0, true_cfg=1.0,
        strength=0.8, seed=17, workflow=workflow, swap_kind=kind, compose=compose,
    )


def registered_graph(graph):
    """Mock object_info choices and field schemas without importing ComfyUI/GPU."""
    registered = {}
    for node in graph.values():
        fields = registered.setdefault(node["class_type"], {"input": {"required": {}}})["input"]["required"]
        for field, value in node["inputs"].items():
            if field.endswith("_name"):
                choices = fields.setdefault(field, [[], {}])[0]
                if value not in choices:
                    choices.append(value)
            else:
                fields[field] = ["STRING" if isinstance(value, str) else "INT", {}]
    return registered


def test_distinct_registry_and_adapter_identity():
    manager = ModelManager()
    old = manager.get("qwen-2.1-turbo")
    new = manager.get(QWEN21_R128_KEY)
    assert old is not new
    assert isinstance(new, ComfyQwen21Adapter)
    assert new.spec.key == QWEN21_R128_KEY
    assert new.spec.family == "qwen21"
    assert new.spec.label == "Qwen Image 2.1 + Turbo r128 (Civitai)"
    assert "isHeSatoshi" in new.spec.license_note
    assert "Research License" in new.spec.license_note


def test_template_profile_isolation_and_default_unchanged():
    old = qwen21_turbo_template()
    assert old == qwen21_turbo_template("qwen-2.1-turbo")
    new = qwen21_turbo_template(QWEN21_R128_KEY)
    assert new["2"] == {"class_type": "LoraLoaderModelOnly", "inputs": {
        "model": ["1", 0], "lora_name": QWEN21_R128_TURBO_LORA, "strength_model": 1.0,
    }}
    assert new["9"]["inputs"]["sampler_name"] == "res_2s_ode"
    assert new["10"] == {"class_type": "ManualSigmas", "inputs": {"sigmas": QWEN21_R128_SIGMAS}}
    assert QWEN21_R128_SIGMAS == "1.0,0.9375,0.875,0.75,0.5,0.25,0.0"
    assert len(QWEN21_R128_SIGMAS.split(",")) - 1 == 6
    assert new["8"]["class_type"] == "BasicGuider"  # CFG 1, positive only.
    assert new["5"]["inputs"]["negative_prompt"] == ""
    for node in ("1", "3", "4", "5", "6", "7", "8", "11", "12"):
        assert new[node] == old[node]
    assert not any("Viggle" in node["class_type"] for node in new.values())
    assert old["2"]["inputs"]["lora_name"] == QWEN21_TURBO_LORA
    assert old["9"]["inputs"]["sampler_name"] == "euler"
    assert old["10"]["inputs"]["latent"] == old["11"]["inputs"]["latent_image"]
    with pytest.raises(ValueError, match="Unknown Qwen"):
        qwen21_turbo_template("unknown")


@pytest.mark.parametrize("images,workflow,kind,compose", [
    (0, "text", None, True), (1, "standard", None, False),
    (3, "standard", None, True), (2, "swap", "Head", False), (2, "swap", "Body", False),
])
def test_all_modes_with_optional_chain(monkeypatch, tmp_path, images, workflow, kind, compose):
    monkeypatch.setattr(settings, "lora_dir", tmp_path)
    library = tmp_path / "qwen21"
    library.mkdir()
    req = request(images, workflow, kind, compose)
    for index in range(5):
        filename = swap_profile(req.model_key, kind).filename if workflow == "swap" and index == 0 else f"style{index}.safetensors"
        path = library / filename
        path.touch()
        req.loras.append(LoraSpec(path.name, path, 0.7, f"style{index}"))
    graph = configure_qwen21_graph(qwen21_turbo_template(req.model_key), req, tuple(f"ref{i}.png" for i in range(images)))
    assert graph["2"]["inputs"]["strength_model"] == 1.0
    assert graph["40"]["inputs"]["model"] == ["2", 0]
    assert graph["8"]["inputs"]["model"] == ["44", 0]
    assert graph["6"]["inputs"]["width"] == 1024
    assert graph["6"]["inputs"]["height"] == 768
    assert graph["11"]["inputs"]["sigmas"] == ["10", 0]
    for i in range(1, images + 1):
        assert graph["5"]["inputs"][f"images.image_{i}"] == [str(30 + i), 0]
    info = registered_graph(graph)
    info["LoraLoaderModelOnly"]["input"]["required"]["lora_name"][0] = [
        name.replace("/", "\\") for name in info["LoraLoaderModelOnly"]["input"]["required"]["lora_name"][0]
    ]
    preflight_qwen21_r128_graph(graph, info)


def test_preflight_accepts_live_v3_sampler_combo():
    graph = qwen21_turbo_template("qwen-2.1-turbo-r128")
    info = registered_graph(graph)
    definition = info["KSamplerSelect"]["input"]["required"]["sampler_name"]
    definition[:] = ["COMBO", {"options": ["euler", "res_2s_ode"]}]
    preflight_qwen21_r128_graph(graph, info)
    definition[1]["options"] = ["euler"]
    with pytest.raises(RuntimeError, match="sampler|registered"):
        preflight_qwen21_r128_graph(graph, info)


@pytest.mark.parametrize("field,value", [("steps", 4), ("true_cfg", 2), ("guidance", 2), ("negative_prompt", "bad")])
def test_fixed_controls_rejected(field, value):
    req = request()
    setattr(req, field, value)
    with pytest.raises(ValueError, match="Civitai Turbo r128"):
        configure_qwen21_graph(qwen21_turbo_template(req.model_key), req, ())


@pytest.mark.parametrize("node", ["2", "9", "10"])
def test_wrong_profile_graph_rejected(node):
    graph = qwen21_turbo_template(QWEN21_R128_KEY)
    graph[node] = qwen21_turbo_template()[node]
    with pytest.raises(ValueError, match="profile-specific"):
        configure_qwen21_graph(graph, request(), ())


def test_asset_checks_do_not_require_viggle_or_extension_directory(tmp_path):
    for file in (*QWEN21_FILES[:3], f"loras/{QWEN21_R128_TURBO_LORA}"):
        path = tmp_path / "models" / file
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    assert missing_qwen21_assets(tmp_path, QWEN21_R128_KEY) == []
    assert len(missing_qwen21_assets(tmp_path)) == 2  # old r256 + Viggle node still required
    (tmp_path / "models" / "loras" / QWEN21_R128_TURBO_LORA).unlink()
    assert missing_qwen21_assets(tmp_path, QWEN21_R128_KEY) == [str(tmp_path / "models" / "loras" / QWEN21_R128_TURBO_LORA)]
    with pytest.raises(ValueError, match="Unknown Qwen"):
        missing_qwen21_assets(tmp_path, "unknown")


@pytest.mark.parametrize("failure", ["node", "sampler", "sampler_schema", "input", "asset"])
def test_preflight_fails_closed(failure):
    graph = qwen21_turbo_template(QWEN21_R128_KEY)
    info = registered_graph(graph)
    if failure == "node":
        del info["ManualSigmas"]
    elif failure == "sampler":
        info["KSamplerSelect"]["input"]["required"]["sampler_name"][0] = ["euler"]
    elif failure == "sampler_schema":
        info["KSamplerSelect"]["input"]["required"]["sampler_name"][0] = "STRING"
    elif failure == "input":
        del info["ManualSigmas"]["input"]["required"]["sigmas"]
    else:
        info["LoraLoaderModelOnly"]["input"]["required"]["lora_name"][0] = [QWEN21_TURBO_LORA]
    with pytest.raises(RuntimeError, match="qwen-2.1-turbo-r128") as error:
        preflight_qwen21_r128_graph(graph, info)
    assert QWEN21_R128_SAMPLER_EXTENSION in str(error.value)
    assert "no sampler substitution" in str(error.value)


@pytest.mark.parametrize("failure", ["node", "sampler", "asset"])
def test_adapter_preflight_happens_before_upload(monkeypatch, failure):
    from photo_edit_studio.models import comfy_swap

    req = request(1, "standard")
    graph = configure_qwen21_graph(qwen21_turbo_template(req.model_key), req, ("pending.png",))
    info = registered_graph(graph)
    if failure == "node":
        del info["ManualSigmas"]
    elif failure == "sampler":
        info["KSamplerSelect"]["input"]["required"]["sampler_name"][0] = ["euler"]
    else:
        info["LoraLoaderModelOnly"]["input"]["required"]["lora_name"][0] = []
    calls = []

    class Client:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def get(self, path):
            calls.append(path)
            return httpx.Response(200, json=info if path == "/object_info" else {}, request=httpx.Request("GET", "http://localhost" + path))

        def post(self, path, **kwargs):
            pytest.fail(f"Unexpected upload/queue before dependency rejection: {path}")

    monkeypatch.setattr(comfy_swap, "cuda_available", lambda: True)
    monkeypatch.setattr(comfy_swap, "cuda_vram_gb", lambda: 12)
    monkeypatch.setattr(comfy_swap, "missing_qwen21_assets", lambda root, key: [])
    monkeypatch.setattr(comfy_swap, "ensure_backend", lambda **kwargs: calls.append(kwargs["model_key"]))
    monkeypatch.setattr(comfy_swap.httpx, "Client", Client)
    with pytest.raises(RuntimeError, match="qwen-2.1-turbo-r128"):
        ComfyQwen21Adapter(MODEL_SPECS[req.model_key]).generate(req)
    assert calls == [QWEN21_R128_KEY, "/system_stats", "/object_info"]


def test_backend_uses_r128_asset_checks_not_krea(monkeypatch, tmp_path):
    from photo_edit_studio import comfy_backend

    monkeypatch.setattr(settings, "comfy_dir", tmp_path)
    (tmp_path / "main.py").touch()
    monkeypatch.setattr(comfy_backend, "_process", None)
    monkeypatch.setattr(comfy_backend, "is_ready", lambda: False)
    seen = []
    monkeypatch.setattr(comfy_backend, "missing_qwen21_assets", lambda root, key: seen.append(key) or ["r128 missing"])
    monkeypatch.setattr(comfy_backend, "missing_assets", lambda root: pytest.fail("Krea fallback"))
    with pytest.raises(FileNotFoundError, match="mandatory r128"):
        comfy_backend.ensure_backend(enabled=True, model_key=QWEN21_R128_KEY)
    assert seen == [QWEN21_R128_KEY]


@pytest.mark.parametrize("key", ["qwen-2.1-turbo", QWEN21_R128_KEY])
def test_profile_metadata_and_notes(monkeypatch, tmp_path, key):
    from photo_edit_studio import engine

    req = request()
    req.model_key = key
    req.width = req.height = 64
    req.compose = True
    monkeypatch.setattr(settings, "output_dir", tmp_path)

    class Adapter:
        def generate(self, req, progress=None):
            return [Image.new("RGB", (64, 64))]

    monkeypatch.setattr(engine.model_manager, "get", lambda key: Adapter())
    result = engine.generate(req, "Off", 0.35)
    metadata = json.loads((result.saved_paths[0].parent / "metadata.json").read_text())
    assert metadata["model_key"] == key
    assert metadata["model"] == "Qwen/Qwen-Image-2.1"
    assert metadata["mandatory_turbo_weight"] == 1.0
    assert metadata["mandatory_turbo_lora"] == (QWEN21_R128_TURBO_LORA if key == QWEN21_R128_KEY else QWEN21_TURBO_LORA)
    assert metadata["sampler"] == ("res_2s_ode" if key == QWEN21_R128_KEY else "euler")
    assert any(("isHeSatoshi" if key == QWEN21_R128_KEY else "unmerged Viggle r256") in note for note in result.notes)
    if key == QWEN21_R128_KEY:
        assert metadata["turbo_file_id"] == 3273779
        assert metadata["sigmas"] == QWEN21_R128_SIGMAS
    with Image.open(result.saved_paths[0]) as output:
        assert json.loads(output.info["generation"])["model_key"] == key


def test_preflight_supports_declared_v3_autogrow_image_inputs():
    req = request(2, "standard")
    graph = configure_qwen21_graph(qwen21_turbo_template(req.model_key), req, ("a.png", "b.png"))
    info = registered_graph(graph)
    fields = info["TextEncodeQwenImage21"]["input"]["required"]
    del fields["images.image_1"]
    del fields["images.image_2"]
    fields["images"] = ["COMFY_AUTOGROW_V3", {"template": {
        "names": ["image_1", "image_2"], "min": 0,
        "input": {"required": {"image": ["IMAGE", {}]}},
    }}]
    preflight_qwen21_r128_graph(graph, info)
    graph["5"]["inputs"]["images.image_3"] = ["31", 0]
    with pytest.raises(RuntimeError, match="lacks input images.image_3"):
        preflight_qwen21_r128_graph(graph, info)


@pytest.mark.parametrize("kind", ["Head", "Body"])
@pytest.mark.parametrize("failure", ["missing", "wrong", "zero", "duplicate", "other"])
def test_swap_requires_matching_bfs_before_optional_loras(monkeypatch, tmp_path, kind, failure):
    monkeypatch.setattr(settings, "lora_dir", tmp_path)
    library = tmp_path / "qwen21"
    library.mkdir()
    req = request(2, "swap", kind)
    bfs = swap_profile(req.model_key, kind).filename
    other = swap_profile(req.model_key, "Body" if kind == "Head" else "Head").filename
    for name in (bfs, other):
        (library / name).touch()
    if failure != "missing":
        name = other if failure == "wrong" else bfs
        req.loras = [LoraSpec(name, library / name, 0 if failure == "zero" else 1, "bfs")]
    if failure in {"duplicate", "other"}:
        name = bfs if failure == "duplicate" else other
        req.loras.append(LoraSpec(name, library / name, 1, "extra"))
    with pytest.raises(ValueError, match="BFS|swap requires"):
        configure_qwen21_graph(qwen21_turbo_template(req.model_key), req, ("a.png", "b.png"))