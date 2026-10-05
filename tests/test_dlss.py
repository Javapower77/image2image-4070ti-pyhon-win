from __future__ import annotations

import json
from copy import deepcopy
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from PIL import Image

from photo_edit_studio import engine
from photo_edit_studio.comfy_workflows import (
    firered_edit_template,
    krea_reference_template,
    krea_remix_template,
    krea_text_template,
    qwen21_turbo_template,
)
from photo_edit_studio.config import settings
from photo_edit_studio.dlss import (
    DLSS_CHOICES,
    DLSS_DEFAULTS,
    add_dlss_nodes,
    preflight_dlss,
    validate_dlss_settings,
)
from photo_edit_studio.models import MODEL_SPECS, comfy_swap
from photo_edit_studio.types import GenerationRequest


def request(model="qwen-2.1-turbo", workflow="standard", dlss=None):
    return GenerationRequest(
        model_key=model, prompt="colors", negative_prompt="",
        images=[] if workflow in {"text", "krea-text"} else [Image.new("RGB", (512, 512))],
        mask=None, width=512, height=512, steps=6, guidance=1, true_cfg=1,
        strength=0.8, seed=17, workflow=workflow, dlss=dlss,
    )


def test_default_contract_is_opt_in_and_does_not_mutate_settings():
    assert request().dlss is None
    assert validate_dlss_settings(None) is None
    values = validate_dlss_settings({})
    assert values == DLSS_DEFAULTS
    assert values is not DLSS_DEFAULTS
    assert values["upscaling_mode"] == "1x (DLAA / native)"
    original = {"runtime_dir": "  D:\\runtime  "}
    assert validate_dlss_settings(original)["runtime_dir"] == "D:\\runtime"
    assert original["runtime_dir"].startswith("  ")


@pytest.mark.parametrize("name", DLSS_CHOICES)
def test_enum_validation(name):
    for choice in DLSS_CHOICES[name]:
        assert validate_dlss_settings({name: choice})[name] == choice
    for value in ("invalid", 1, [], None):
        with pytest.raises(ValueError, match=name):
            validate_dlss_settings({name: value})


@pytest.mark.parametrize("name,low,high", [
    ("nr_intensity", 0, 2), ("local_tone_strength", 0, 2),
    ("local_structure_strength", 0, 2), ("skin_structure_strength", -1, 2),
    ("scene_change_threshold", 0.01, 1),
])
def test_numeric_ranges(name, low, high):
    for value in (low, high):
        assert validate_dlss_settings({name: value})[name] == value
    for value in (low - 0.01, high + 0.01, float("nan"), float("inf"), True, "1", None):
        with pytest.raises(ValueError, match=name):
            validate_dlss_settings({name: value})


@pytest.mark.parametrize("name", ["enabled", "automatic_mask", "verify_neural_rendering"])
def test_strict_bools(name):
    for value in (0, 1, "false", None):
        with pytest.raises(ValueError, match=name):
            validate_dlss_settings({name: value})
    assert validate_dlss_settings({name: False})[name] is False


@pytest.mark.parametrize("value", [-1, 17, 1.5, True, "0", None])
def test_warmup_integer_bounds(value):
    with pytest.raises(ValueError, match="warmup_frames"):
        validate_dlss_settings({"warmup_frames": value})


def test_warmup_runtime_and_unknown_settings():
    for value in (0, 16):
        assert validate_dlss_settings({"warmup_frames": value})["warmup_frames"] == value
    for value in (Path("runtime"), None, "bad\x00path"):
        with pytest.raises(ValueError, match="runtime_dir"):
            validate_dlss_settings({"runtime_dir": value})
    with pytest.raises(ValueError, match="Unknown DLSS"):
        validate_dlss_settings({"flow_width": 640})  # Not an exposed node input.
    with pytest.raises(ValueError, match="dictionary"):
        validate_dlss_settings([])


@pytest.mark.parametrize("factory,model,workflow", [
    (qwen21_turbo_template, "qwen-2.1-turbo", "standard"),
    (qwen21_turbo_template, "qwen-2.1-turbo", "text"),
    (qwen21_turbo_template, "qwen-2.1-turbo", "swap"),
    (firered_edit_template, "firered-1.1", "standard"),
    (krea_reference_template, "krea-2-turbo", "krea-reference"),
    (krea_reference_template, "krea-2-turbo", "swap"),
    (krea_text_template, "krea-2-turbo", "krea-text"),
    (krea_remix_template, "krea-2-turbo", "krea-remix"),
])
def test_graph_placement_and_disabled_identity(factory, model, workflow):
    graph = factory()
    original = deepcopy(graph)
    for disabled in (None, {"enabled": False}):
        assert add_dlss_nodes(graph, request(model, workflow, disabled)) is graph
        assert graph == original
    source = original["29"]["inputs"]["images"]
    req = request(model, workflow, {"upscaling_mode": "2x (Performance)"})
    assert add_dlss_nodes(graph, req) is graph
    enhance_id = graph["29"]["inputs"]["images"][0]
    enhance = graph[enhance_id]
    assert enhance["class_type"] == "DLSS5EnhanceImages"
    assert enhance["inputs"]["images"] == source
    assert enhance["inputs"]["verify_neural_rendering"] is True
    config = graph[enhance["inputs"]["settings"][0]]
    assert config["class_type"] == "DLSS5Settings"
    assert config["inputs"] == {key: value for key, value in req.dlss.items()
                                if key not in {"enabled", "verify_neural_rendering"}}
    assert len(graph) == len(original) + 2
    for key in original.keys() - {"29"}:
        assert graph[key] == original[key]
    assert graph["29"]["inputs"]["filename_prefix"] == original["29"]["inputs"]["filename_prefix"]
    with pytest.raises(ValueError, match="already exist"):
        add_dlss_nodes(graph, req)


def test_graph_errors_and_unsupported_engine_fail_before_loading(monkeypatch):
    with pytest.raises(ValueError, match="SaveImage"):
        add_dlss_nodes({}, request(dlss={}))
    with pytest.raises(ValueError, match="connected IMAGE"):
        add_dlss_nodes({"29": {"class_type": "SaveImage", "inputs": {}}}, request(dlss={}))
    monkeypatch.setattr(engine.model_manager, "get", lambda *_: pytest.fail("Must fail before model loading"))
    with pytest.raises(ValueError, match="Diffusers"):
        engine.generate(request("qwen-2511", dlss={}), "Off", 0.35)
    with pytest.raises(ValueError, match="warmup_frames"):
        engine.generate(request(dlss={"warmup_frames": 17}), "Off", 0.35)


def test_preflight_is_noop_when_disabled():
    client = SimpleNamespace(get=lambda *_: pytest.fail("No network for disabled DLSS"))
    preflight_dlss(client, request())
    preflight_dlss(client, request(dlss={"enabled": False}))


@pytest.mark.parametrize("registry,status", [({}, 200), ({"DLSS5Settings": {}}, 200), ([], 200), ({}, 500)])
def test_preflight_missing_nodes_and_network_errors(registry, status):
    with httpx.Client(base_url="http://127.0.0.1:8188", transport=httpx.MockTransport(
        lambda _: httpx.Response(status, json=registry)
    )) as client, pytest.raises(RuntimeError, match="runtime.*manual|manual.*runtime") as error:
        preflight_dlss(client, request(dlss={}))
    assert "setup-comfy.ps1" in str(error.value)
    assert "restart ComfyUI" in str(error.value)
    assert "nvngx.dll" in str(error.value)


@pytest.mark.parametrize("model,workflow", [
    ("qwen-2.1-turbo", "standard"), ("qwen-2.1-turbo", "text"),
    ("qwen-2.1-turbo", "swap"), ("firered-1.1", "standard"),
    ("krea-2-turbo", "krea-reference"), ("krea-2-turbo", "krea-text"),
    ("krea-2-turbo", "krea-remix"), ("krea-2-turbo", "swap"),
])
@pytest.mark.parametrize("missing", [False, True])
def test_adapters_preflight_before_upload_and_route_to_enhanced_save29(monkeypatch, model, workflow, missing):
    req = request(model, workflow, {})
    if workflow == "swap":
        req.images *= 2
        req.swap_kind = "Head"
    calls = []
    data = BytesIO()
    Image.new("RGB", (1024, 1024), "red").save(data, format="PNG")
    registry = {} if missing else {"DLSS5Settings": {}, "DLSS5EnhanceImages": {}}
    registry.update({name: {} for name in ("TextEncodeQwenImage21", "ViggleTurboLora", "ViggleTurboSigmas")})

    def handler(http_request):
        path = http_request.url.path
        calls.append(path)
        if path == "/system_stats":
            return httpx.Response(200, json={})
        if path == "/object_info":
            return httpx.Response(200, json=registry)
        if path == "/upload/image":
            return httpx.Response(200, json={"name": "uploaded.png"})
        if path == "/prompt":
            graph = json.loads(http_request.content)["prompt"]
            assert graph[graph["29"]["inputs"]["images"][0]]["class_type"] == "DLSS5EnhanceImages"
            return httpx.Response(200, json={"prompt_id": "job"})
        if path == "/history/job":
            return httpx.Response(200, json={"job": {"outputs": {
                "29": {"images": [{"filename": "enhanced.png"}]},
                "99": {"images": [{"filename": "original.png"}]},
            }}})
        assert path == "/view" and http_request.url.params["filename"] == "enhanced.png"
        return httpx.Response(200, content=data.getvalue())

    original_client = httpx.Client
    monkeypatch.setattr(comfy_swap.httpx, "Client", lambda **kwargs: original_client(
        **kwargs, transport=httpx.MockTransport(handler),
    ))
    monkeypatch.setattr(comfy_swap, "ensure_backend", lambda **_: None)
    monkeypatch.setattr(comfy_swap, "cuda_available", lambda: True)
    monkeypatch.setattr(comfy_swap, "cuda_vram_gb", lambda: 24)
    for name in ("missing_qwen21_assets", "missing_firered_assets", "missing_krea_remix_assets"):
        monkeypatch.setattr(comfy_swap, name, lambda *_: [])
    # Isolate routing from model weights/LoRA files; placement has real template tests above.
    def configured(graph, *_args, **_kwargs):
        return graph
    for name in ("configure_qwen21_graph", "configure_firered_graph", "configure_krea_graph",
                 "configure_krea_reference_graph", "configure_krea_text_graph", "configure_krea_remix_graph"):
        monkeypatch.setattr(comfy_swap, name, configured)
    monkeypatch.setattr(comfy_swap, "preflight_krea_remix_graph", lambda *_: None)
    adapter_type = {"qwen-2.1-turbo": comfy_swap.ComfyQwen21Adapter,
                    "firered-1.1": comfy_swap.ComfyFireRedAdapter}.get(model, comfy_swap.ComfyKreaSwapAdapter)
    adapter = adapter_type(MODEL_SPECS[model])
    if missing:
        with pytest.raises(RuntimeError, match="missing DLSS nodes"):
            adapter.generate(req)
        assert "/upload/image" not in calls and "/prompt" not in calls
    else:
        images = adapter.generate(req)
        assert images[0].size == (1024, 1024)
        assert calls.index("/object_info") < calls.index("/prompt")
        if req.images:
            assert calls.index("/object_info") < calls.index("/upload/image")


def test_engine_keeps_enhanced_dimensions_and_scales_mask_and_metadata(monkeypatch, tmp_path):
    req = request(dlss={"upscaling_mode": "3x (Ultra Performance)"})
    req.images = [Image.new("RGB", (512, 512), "blue")]
    req.mask = Image.new("L", (512, 512), 0)
    req.mask.paste(255, (256, 0, 512, 512))
    adapter = SimpleNamespace(generate=lambda *_args, **_kwargs: [Image.new("RGB", (2304, 2304), "red")])
    monkeypatch.setattr(engine.model_manager, "get", lambda *_: adapter)
    monkeypatch.setattr(settings, "output_dir", tmp_path)
    result = engine.generate(req, "Off", 0.35)
    assert result.images[0].size == (2304, 2304)  # Above normalize_image's default 2048 cap.
    assert result.images[0].getpixel((10, 10)) == (0, 0, 255)
    assert result.images[0].getpixel((2290, 10)) == (255, 0, 0)
    metadata = json.loads(result.saved_paths[0].with_name("metadata.json").read_text())
    assert metadata["diffusion_size"] == [req.width, req.height]
    assert metadata["enhanced_output_sizes"] == [[2304, 2304]]
    assert metadata["dlss"]["upscaling_mode"] == "3x (Ultra Performance)"
    assert any("separate from the diffusion VRAM budget" in note for note in result.notes)


def test_restoration_cannot_silently_downsize_dlss(monkeypatch):
    monkeypatch.setattr(engine.model_manager, "get", lambda *_: SimpleNamespace(
        generate=lambda *_args, **_kwargs: [Image.new("RGB", (1024, 1024))],
    ))
    monkeypatch.setattr(engine, "restore_faces", lambda *_: Image.new("RGB", (512, 512)))
    monkeypatch.setattr(engine, "release_cuda", lambda: None)
    with pytest.raises(RuntimeError, match="refusing silent resizing"):
        engine.generate(request(dlss={}), "GFPGAN", 0.35)


def test_setup_only_installs_node_pack_not_runtime():
    script = (Path(__file__).parents[1] / "scripts" / "setup-comfy.ps1").read_text()
    assert "https://github.com/Blueforcer/ComfyUI-DLSS5-Enhancer.git" in script
    assert '$dlssNodes "requirements.txt"' in script
    assert not any("install_runtime.py" in line and line.lstrip().startswith("&")
                   for line in script.splitlines())
    assert "Runtime is explicit/manual" in script