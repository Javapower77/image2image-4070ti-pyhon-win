"""All2Real contracts only: temporary assets, mocked HTTP and CPU tensor stubs."""

from __future__ import annotations

import importlib.util
import json
import sys
from copy import deepcopy
from io import BytesIO
from pathlib import Path
from types import ModuleType, SimpleNamespace

import httpx
import pytest
from PIL import Image

from photo_edit_studio import comfy_backend, engine, ui
from photo_edit_studio.config import settings
from photo_edit_studio.krea_all2real import (
    KREA_ALL2REAL_ENCODER,
    KREA_ALL2REAL_FILES,
    KREA_ALL2REAL_FIRST_LORA,
    KREA_ALL2REAL_MODEL,
    KREA_ALL2REAL_MORE_REAL,
    KREA_ALL2REAL_SKIN_MODEL,
    KREA_ALL2REAL_VAE,
    configure_krea_all2real_graph,
    krea_all2real_template,
    missing_krea_all2real_assets,
)
from photo_edit_studio.models import comfy_swap
from photo_edit_studio.models.registry import MODEL_SPECS
from photo_edit_studio.types import GenerationRequest, LoraSpec

ROOT = Path(__file__).resolve().parents[1]


def request(**changes):
    req = GenerationRequest(
        model_key="krea-2-turbo", workflow="krea-all2real",
        images=[Image.new("RGB", (1536, 768), "blue")], prompt="natural photographic light",
        negative_prompt="", mask=None, width=2048, height=2048, steps=11,
        guidance=1, true_cfg=0, strength=1, seed=42, krea_first_lora_weight=0.65,
    )
    for name, value in changes.items():
        setattr(req, name, value)
    return req


@pytest.fixture
def assets(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "comfy_dir", tmp_path / "comfy")
    monkeypatch.setattr(settings, "lora_dir", tmp_path / "library")
    monkeypatch.setattr(settings, "comfy_url", "http://127.0.0.1:8188")
    paths = [settings.comfy_dir / "models" / name for name in KREA_ALL2REAL_FILES]
    paths += [settings.lora_dir / "krea2" / KREA_ALL2REAL_MORE_REAL,
              settings.comfy_dir / "custom_nodes" / "photo_edit_all2real.py"]
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    return paths


def optional_loras(weights):
    result = []
    for index, weight in enumerate(weights):
        path = settings.lora_dir / "krea2" / f"style{index}.safetensors"
        path.touch()
        result.append(LoraSpec(path.name, path, weight, f"slot{index}"))
    return result


def registered(graph, *, qualified=False):
    """Expose real enum checks, not a bypass of graph preflight."""
    info = {}
    for node in graph.values():
        fields = info.setdefault(node["class_type"], {"input": {"required": {}}})["input"]["required"]
        for name in node["inputs"]:
            fields[name] = ("ANY",)
    for cls, field in (("UNETLoader", "unet_name"), ("CLIPLoader", "clip_name"),
                       ("VAELoader", "vae_name"), ("UpscaleModelLoader", "model_name"),
                       ("LoraLoaderModelOnly", "lora_name")):
        choices = []
        for node in graph.values():
            if node["class_type"] == cls:
                name = node["inputs"][field]
                if qualified and cls == "LoraLoaderModelOnly" and name != KREA_ALL2REAL_FIRST_LORA:
                    name = "krea2\\" + name
                choices.append(name)
        info[cls]["input"]["required"][field] = (choices,)
    info["KSamplerAdvanced"]["input"]["required"].update(
        sampler_name=(["er_sde"],), scheduler=(["kl_optimal", "simple"],),
    )
    info["LoadImage"]["input"]["required"]["image"] = (["old.png"],)
    return info


def install_http(monkeypatch, handler):
    original = httpx.Client
    monkeypatch.setattr(comfy_swap.httpx, "Client", lambda **kwargs: original(
        **kwargs, transport=httpx.MockTransport(handler),
    ))
    starts = []
    monkeypatch.setattr(comfy_swap, "ensure_backend", lambda **kwargs: starts.append(kwargs))
    return starts


def adapter():
    return comfy_swap.ComfyKreaSwapAdapter(MODEL_SPECS["krea-2-turbo"])


def test_original_assets_match_editor_workflow_and_templates_are_independent():
    workflow = json.loads((ROOT / "workflows" / "Krea2-all2real.json").read_text(encoding="utf-8"))
    nodes = workflow["nodes"] + [node for subgraph in workflow["definitions"]["subgraphs"]
                                 for node in subgraph["nodes"]]
    for cls, name in (("UNETLoader", KREA_ALL2REAL_MODEL), ("CLIPLoader", KREA_ALL2REAL_ENCODER),
                      ("VAELoader", KREA_ALL2REAL_VAE), ("UpscaleModelLoader", KREA_ALL2REAL_SKIN_MODEL)):
        assert any(node["type"] == cls and node["widgets_values"][0] == name for node in nodes)
    assert any(node["type"] == "PrimitiveInt" and node["widgets_values"][0] == 9 for node in nodes)
    assert any(node["type"] == "Image Tiled" and node["widgets_values"] == [4] for node in nodes)
    assert KREA_ALL2REAL_FILES == (
        "diffusion_models/krea2_turbo_int8_convrot-b19a4f0be264.safetensors",
        "text_encoders/qwen3vl_4b_fp8_scaled.safetensors",
        "vae/Wan2.1_VAE_upscale2x_imageonly_real_v1.safetensors",
        "upscale_models/1x-ITF-SkinDiffDetail-Lite-v1.pth",
        "loras/Krea2_ALWAYS_LOAD_FIRST.safetensors",
    )
    first, second = krea_all2real_template(), krea_all2real_template()
    assert first["53"]["inputs"]["noise_seed"] == 9
    assert first["90"]["inputs"]["seed"] == 10
    first["53"]["inputs"]["steps"] = 99
    assert second["53"]["inputs"]["steps"] == 11


@pytest.mark.parametrize("steps", [5, 11, 40])
@pytest.mark.parametrize("weights", [[], [0], [-2, -0.5, 0, 1.25, 2]])
def test_graph_exact_source_path_shared_chain_and_schedules(assets, steps, weights):
    req = request(steps=steps, loras=optional_loras(weights))
    graph = configure_krea_all2real_graph(krea_all2real_template(), req, ("uploaded.png",))
    mandatory = graph["79"]["inputs"]["model"][0]
    assert graph[mandatory]["inputs"] == {
        "model": ["55", 0], "lora_name": KREA_ALL2REAL_FIRST_LORA, "strength_model": 0.65,
    }
    assert graph["55"]["inputs"]["unet_name"] == KREA_ALL2REAL_MODEL
    assert graph["56"]["inputs"] == {"clip_name": KREA_ALL2REAL_ENCODER, "type": "krea2", "device": "default"}
    assert graph["57"]["inputs"]["vae_name"] == KREA_ALL2REAL_VAE
    assert graph["79"]["inputs"]["kv_cache"] is True
    assert graph["71"]["inputs"] == {
        "model": ["79", 0], "lora_name": KREA_ALL2REAL_MORE_REAL, "strength_model": 0.9,
    }
    assert graph["53"]["inputs"]["model"] == graph["54"]["inputs"]["model"]
    current = graph["54"]["inputs"]["model"][0]
    active = [lora for lora in req.loras if lora.weight != 0]
    for lora in reversed(active):
        assert graph[current]["inputs"]["lora_name"] == lora.name
        assert graph[current]["inputs"]["strength_model"] == lora.weight
        current = graph[current]["inputs"]["model"][0]
    assert current == "71"
    assert sum(node["class_type"] == "LoraLoaderModelOnly" for node in graph.values()) == 2 + len(active)
    assert graph["72"]["inputs"]["image"] == "uploaded.png"
    assert graph["73"]["inputs"] == {
        "image": ["72", 0], "upscale_method": "bicubic", "megapixels": 1.0, "resolution_steps": 16,
    }
    assert graph["74"]["inputs"] == {"image": ["73", 0], "blur_radius": 1, "sigma": 1.0}
    assert graph["75"]["inputs"]["image"] == ["74", 0]
    assert graph["82"]["inputs"] == {"pixels": ["75", 0], "vae": ["57", 0]}
    assert graph["84"]["inputs"] == {
        "clip": ["56", 0], "vae": ["57", 0], "image1": ["75", 0], "prompt": req.prompt,
    }
    assert graph["85"]["inputs"] == {"conditioning": ["84", 0], "reference_latents_method": "index_timestep_zero"}
    assert graph["86"]["inputs"]["conditioning"] == ["85", 0]
    assert graph["87"]["inputs"]["conditioning"] == ["85", 0]
    for node_id, positive, latent, scheduler, start, end in (
        ("53", "85", "82", "kl_optimal", 1, steps - 3),
        ("54", "87", "90", "simple", steps - 2, steps),
    ):
        inputs = graph[node_id]["inputs"]
        assert inputs == {
            "model": graph["53"]["inputs"]["model"], "positive": [positive, 0], "negative": ["86", 0],
            "latent_image": [latent, 0], "add_noise": "enable", "noise_seed": 42, "steps": steps,
            "cfg": 1.0, "sampler_name": "er_sde", "scheduler": scheduler,
            "start_at_step": start, "end_at_step": end, "return_with_leftover_noise": "disable",
        }
    assert graph["88"]["inputs"] == {"samples": ["53", 0], "vae": ["57", 0]}
    assert graph["88"]["class_type"] == graph["91"]["class_type"] == "StudioAll2RealVAEDecode"
    assert graph["89"]["inputs"] == {"pixels": ["88", 0], "vae": ["57", 0]}
    assert graph["90"]["inputs"] == {"samples": ["89", 0], "noise_std": 0.3, "seed": 43}
    assert graph["91"]["inputs"] == {"samples": ["54", 0], "vae": ["57", 0]}
    assert graph["92"]["inputs"]["model_name"] == KREA_ALL2REAL_SKIN_MODEL
    assert graph["93"]["inputs"] == {"image": ["91", 0], "upscale_model": ["92", 0]}
    assert graph["29"]["inputs"]["images"] == ["93", 0]
    assert graph["29"]["inputs"]["filename_prefix"].startswith("photo_edit_krea_all2real_")
    assert sum(node["class_type"] == "SaveImage" for node in graph.values()) == 1
    assert not any(node["class_type"] in {"CLIPTextEncode", "EmptyLatentImage", "Krea2EditModelPatch"}
                   for node in graph.values())
    assert not any("identity" in node["inputs"].get("lora_name", "").lower()
                   or "remix" in node["inputs"].get("lora_name", "").lower() for node in graph.values())
    comfy_swap.preflight_krea_remix_graph(graph, registered(graph), label="All2Real")


@pytest.mark.parametrize("weight,expected", [(0, 0.9), (-0.5, -0.5), (1.2, 1.2)])
def test_more_real_optional_selection_adjusts_required_node_without_duplicate(assets, weight, expected):
    path = settings.lora_dir / "krea2" / KREA_ALL2REAL_MORE_REAL
    req = request(loras=[LoraSpec(path.name, path, weight, "more_real")])
    graph = configure_krea_all2real_graph(krea_all2real_template(), req, ("source.png",))
    assert graph["71"]["inputs"]["strength_model"] == expected
    assert graph["53"]["inputs"]["model"] == graph["54"]["inputs"]["model"] == ["71", 0]
    assert sum(node["class_type"] == "LoraLoaderModelOnly" for node in graph.values()) == 2


@pytest.mark.parametrize("field,value", [
    ("model_key", "flux-klein-4b"),
    ("images", []), ("images", [Image.new("RGB", (8, 8))] * 2),
    ("count", 0), ("count", 2), ("mask", Image.new("L", (8, 8))),
    ("swap_kind", "Head"), ("size_multiplier", 2),
    ("steps", 4), ("steps", 41), ("steps", 11.0), ("steps", True),
    ("guidance", 0), ("guidance", 2), ("guidance", float("nan")),
    ("guidance", float("inf")), ("negative_prompt", "bad"),
    ("seed", -1), ("seed", 2**64 - 1), ("seed", 1.5), ("seed", True),
    ("krea_first_lora_weight", 0), ("krea_first_lora_weight", 2.1),
    ("krea_first_lora_weight", float("nan")), ("krea_first_lora_weight", float("inf")),
])
def test_invalid_requests_fail_before_backend_or_upload_and_keep_graph(assets, monkeypatch, field, value):
    req = request(**{field: value})
    monkeypatch.setattr(comfy_swap, "ensure_backend", lambda **kwargs: pytest.fail("must not start backend"))
    monkeypatch.setattr(comfy_swap.httpx, "Client", lambda **kwargs: pytest.fail("must not upload"))
    with pytest.raises(ValueError):
        adapter().generate(req)
    graph = krea_all2real_template()
    before = deepcopy(graph)
    with pytest.raises(ValueError):
        configure_krea_all2real_graph(graph, req, ("source.png",))
    assert graph == before


def test_all2real_configurator_rejects_other_workflow(assets):
    # The shared adapter legitimately routes krea-remix to its own validator.
    graph = krea_all2real_template()
    before = deepcopy(graph)
    with pytest.raises(ValueError, match="workflow='krea-all2real'"):
        configure_krea_all2real_graph(graph, request(workflow="krea-remix"), ("source.png",))
    assert graph == before


@pytest.mark.parametrize("seed", [0, 2**64 - 2])
def test_seed_boundaries_preserve_nonwrapping_noise_seed_plus_one(assets, seed):
    graph = configure_krea_all2real_graph(krea_all2real_template(), request(seed=seed), ("source.png",))
    assert graph["53"]["inputs"]["noise_seed"] == graph["54"]["inputs"]["noise_seed"] == seed
    assert graph["90"]["inputs"]["seed"] == seed + 1


@pytest.mark.parametrize("case", ["six", "mandatory", "duplicate", "foreign", "name", "suffix",
                                  "missing", "nan", "inf", "negative_inf", "low", "high"])
def test_invalid_loras_fail_before_upload_and_graph_mutation(assets, monkeypatch, case):
    loras = optional_loras([0.7])
    lora = loras[0]
    error = ValueError
    if case == "six":
        loras = optional_loras([1] * 6)
    elif case == "mandatory":
        loras = [LoraSpec(KREA_ALL2REAL_FIRST_LORA, lora.path, 1, "reserved")]
    elif case == "duplicate":
        loras += loras
    elif case == "foreign":
        path = settings.lora_dir / "flux" / lora.name
        path.parent.mkdir()
        path.touch()
        loras = [LoraSpec(path.name, path, 1, "foreign")]
    elif case == "name":
        loras = [LoraSpec("different.safetensors", lora.path, 1, "name")]
    elif case == "suffix":
        path = lora.path.with_suffix(".pt")
        path.touch()
        loras = [LoraSpec(path.name, path, 1, "suffix")]
    elif case == "missing":
        lora.path.unlink()
        error = FileNotFoundError
    else:
        weight = {"nan": float("nan"), "inf": float("inf"), "negative_inf": float("-inf"),
                  "low": -2.1, "high": 2.1}[case]
        loras = [LoraSpec(lora.name, lora.path, weight, "invalid")]
    req = request(loras=loras)
    monkeypatch.setattr(comfy_swap, "ensure_backend", lambda **kwargs: pytest.fail("must not start"))
    with pytest.raises(error):
        adapter().generate(req)
    graph = krea_all2real_template()
    before = deepcopy(graph)
    with pytest.raises(error):
        configure_krea_all2real_graph(graph, req, ("source.png",))
    assert graph == before


@pytest.mark.parametrize("names", [(), ("",), ("  ",), (None,), ("a", "b")])
def test_bad_upload_names_do_not_mutate_template(assets, names):
    graph = krea_all2real_template()
    before = deepcopy(graph)
    with pytest.raises(ValueError, match="uploaded image name"):
        configure_krea_all2real_graph(graph, request(), names)
    assert graph == before


@pytest.mark.parametrize("node,field,value", [
    ("55", "unet_name", "krea2_turbo_fp8_scaled.safetensors"),
    ("57", "vae_name", "qwen_image_vae.safetensors"), ("75", "image", ["72", 0]),
])
def test_substituted_graph_and_reconfiguration_fail_closed(assets, node, field, value):
    graph = krea_all2real_template()
    graph[node]["inputs"][field] = value
    before = deepcopy(graph)
    with pytest.raises(ValueError, match="fresh, unmodified"):
        configure_krea_all2real_graph(graph, request(), ("source.png",))
    assert graph == before
    graph = configure_krea_all2real_graph(krea_all2real_template(), request(), ("source.png",))
    before = deepcopy(graph)
    with pytest.raises(ValueError, match="fresh, unmodified"):
        configure_krea_all2real_graph(graph, request(), ("other.png",))
    assert graph == before


@pytest.mark.parametrize("index", range(7))
def test_exact_missing_asset_never_falls_back_or_starts_backend(assets, monkeypatch, index):
    assert missing_krea_all2real_assets(settings.comfy_dir) == []
    missing = assets[index]
    missing.unlink()
    missing.with_name("fallback" + missing.suffix).touch()
    assert missing_krea_all2real_assets(settings.comfy_dir) == [str(missing)]
    monkeypatch.setattr(comfy_swap, "ensure_backend", lambda **kwargs: pytest.fail("must not start"))
    with pytest.raises(FileNotFoundError):
        adapter().generate(request())
    # Check even when autostart is disabled or a live process could be reused.
    with pytest.raises(FileNotFoundError, match="no downloads or substitutions"):
        comfy_backend.ensure_backend(enabled=False, workflow="krea-all2real")


@pytest.mark.parametrize("missing", ["Krea2OstrisEditModelPatch", "TextEncodeKrea2OstrisEdit",
                                     "StudioAll2RealRemoveReferences", "StudioAll2RealLatentNoise",
                                     "StudioAll2RealSkinDetail", "FluxKontextImageScale", "StudioAll2RealVAEDecode"])
def test_missing_nodes_fail_http_preflight_before_upload(assets, monkeypatch, missing):
    graph = configure_krea_all2real_graph(krea_all2real_template(), request(), ("source.png",))
    info = registered(graph, qualified=True)
    info.pop(missing)
    info["Krea2EditModelPatch"] = {"input": {"required": {}}}
    calls = []

    def handler(req):
        calls.append(req.url.path)
        assert req.method == "GET"
        return httpx.Response(200, json=info if req.url.path == "/object_info" else {})

    install_http(monkeypatch, handler)
    with pytest.raises(RuntimeError, match=missing):
        adapter().generate(request())
    assert calls == ["/system_stats", "/object_info"]


@pytest.mark.parametrize("failure", ["sampler", "scheduler", "kv_cache", "unqualified"])
def test_incompatible_registered_inputs_fail_before_upload(assets, monkeypatch, failure):
    graph = configure_krea_all2real_graph(krea_all2real_template(), request(), ("source.png",))
    info = registered(graph, qualified=True)
    if failure == "kv_cache":
        info["Krea2OstrisEditModelPatch"]["input"]["required"].pop("kv_cache")
    elif failure == "unqualified":
        info["LoraLoaderModelOnly"]["input"]["required"]["lora_name"] = (
            [KREA_ALL2REAL_FIRST_LORA, KREA_ALL2REAL_MORE_REAL],
        )
    else:
        field = "sampler_name" if failure == "sampler" else "scheduler"
        info["KSamplerAdvanced"]["input"]["required"][field] = (["euler" if failure == "sampler" else "simple"],)
    calls = []

    def handler(req):
        calls.append(req.url.path)
        assert req.method == "GET"
        return httpx.Response(200, json=info if req.url.path == "/object_info" else {})

    install_http(monkeypatch, handler)
    with pytest.raises(RuntimeError):
        adapter().generate(request())
    assert calls == ["/system_stats", "/object_info"]


@pytest.mark.parametrize("output_count", [0, 1, 2])
def test_http_preflight_upload_and_queue_real_graph_returns_only_output29(assets, monkeypatch, output_count):
    req = request(loras=optional_loras([-0.4, 1.2]))
    expected = configure_krea_all2real_graph(krea_all2real_template(), req, ("source.png",))
    info = registered(expected, qualified=True)
    calls = []
    image = BytesIO()
    Image.new("RGB", (1280, 640), "red").save(image, format="PNG")

    def handler(http_request):
        path = http_request.url.path
        calls.append(path)
        if path == "/system_stats":
            return httpx.Response(200, json={})
        if path == "/object_info":
            return httpx.Response(200, json=info)
        if path == "/upload/image":
            assert b"image/png" in http_request.content
            start = http_request.content.index(b"\x89PNG")
            with Image.open(BytesIO(http_request.content[start:])) as uploaded:
                assert uploaded.size == (1024, 512)
                assert uploaded.getpixel((0, 0)) == (0, 0, 255)
            return httpx.Response(200, json={"name": "uploaded.png"})
        if path == "/prompt":
            payload = json.loads(http_request.content)["prompt"]
            expected["72"]["inputs"]["image"] = "uploaded.png"
            expected["29"]["inputs"]["filename_prefix"] = payload["29"]["inputs"]["filename_prefix"]
            for node in expected.values():
                if node["class_type"] == "LoraLoaderModelOnly" and node["inputs"]["lora_name"] != KREA_ALL2REAL_FIRST_LORA:
                    node["inputs"]["lora_name"] = "krea2\\" + node["inputs"]["lora_name"]
            assert payload == expected
            return httpx.Response(200, json={"prompt_id": "job"})
        if path == "/history/job":
            return httpx.Response(200, json={"job": {"outputs": {
                "29": {"images": [{"filename": "generated.png"}] * output_count},
                "72": {"images": [{"filename": "source.png"}]},
                "88": {"images": [{"filename": "intermediate.png"}]},
            }}})
        assert path == "/view" and http_request.url.params["filename"] == "generated.png"
        return httpx.Response(200, content=image.getvalue())

    starts = install_http(monkeypatch, handler)
    if output_count == 1:
        results = adapter().generate(req)
        assert len(results) == 1 and results[0].size == (1280, 640)
        assert results[0].getpixel((0, 0)) == (255, 0, 0)
        assert calls[-1] == "/view"
    else:
        with pytest.raises(RuntimeError, match="SaveImage 29 must return exactly one"):
            adapter().generate(req)
        assert "/view" not in calls
    assert starts == [{"workflow": "krea-all2real"}]
    assert calls[:4] == ["/system_stats", "/object_info", "/upload/image", "/prompt"]


@pytest.mark.parametrize("prompt,expected", [("", "photorealistic"), ("  realistic sunlight  ", "realistic sunlight")])
def test_ui_forwards_five_optional_names_weights_and_fixes_operation(assets, monkeypatch, prompt, expected):
    selections, calls = [], []
    selected = optional_loras([-0.5, 0.2, 0.3, 0.4, 0.5])
    monkeypatch.setattr(ui, "selected_loras", lambda model, names, weights:
                        selections.append((model, names, weights)) or selected)
    monkeypatch.setattr(ui, "generate", lambda req, restoration, weight, progress:
                        calls.append((req, restoration, weight)) or SimpleNamespace(images=["output"]))
    monkeypatch.setattr(ui, "metadata_text", lambda result: "metadata")
    source = Image.new("RGB", (400, 600))
    dlss = {"enabled": False}
    result = ui._run_krea_edit(
        "All2Real", source, Image.new("RGB", (8, 8)), prompt, "ignored negative",
        3, 11, 1, -1, "GFPGAN", 0.8, 0.7,
        "one", "two", "three", "four", "five", -0.5, 0.2, 0.3, 0.4, 0.5,
        progress=lambda *args, **kwargs: None, dlss=dlss,
    )
    req, restoration, weight = calls[0]
    assert (req.workflow, req.model_key, req.prompt) == ("krea-all2real", "krea-2-turbo", expected)
    assert req.images == [source] and req.images[0] is source
    assert req.loras is selected and req.dlss is dlss
    assert selections == [("krea-2-turbo", ["one", "two", "three", "four", "five"], [-0.5, 0.2, 0.3, 0.4, 0.5])]
    assert req.steps == 11 and req.seed == -1 and req.guidance == 1
    assert req.krea_first_lora_weight == 0.7
    assert req.count == req.size_multiplier == req.strength == 1
    assert req.mask is req.swap_kind is None
    assert not req.compose and not req.preserve_identity
    assert req.negative_prompt == "" and req.true_cfg == 0
    assert (restoration, weight) == ("Off", 0.0)
    assert result[:2] == (["output"], "metadata")


def test_ui_defaults_hide_second_reference_and_require_source():
    updates = ui._krea_operation_for_mode(ui.KREA_EDIT_MODE, "All2Real")
    assert updates[0]["visible"] is False
    assert updates[1]["value"] == 11 and updates[2]["value"] == 1
    assert updates[3]["value"] == 0
    assert all("value" not in update for update in ui._krea_operation_for_mode(ui.EDIT_MODE, "All2Real"))
    with pytest.raises(ValueError, match="Upload Picture 1"):
        ui._run_krea_edit("All2Real", None, None, "", "", 1, 11, 1, 9, "Off", 0)


def test_engine_retains_source_routes_comfy_and_resolves_random_seed(assets, monkeypatch, tmp_path):
    req = request(seed=-1, compose=True)
    source = req.images[0]
    source_bytes = source.tobytes()
    monkeypatch.setattr(settings, "output_dir", tmp_path / "outputs")
    monkeypatch.setattr(settings, "max_output_side", 1024)
    monkeypatch.setattr(settings, "max_output_pixels", 1024**2)
    monkeypatch.setattr(engine.secrets, "randbelow", lambda upper: 123456)
    monkeypatch.setattr(engine.model_manager, "get", lambda *args: pytest.fail("must not load Diffusers"))
    unloaded = []
    monkeypatch.setattr(engine.model_manager, "unload", lambda: unloaded.append(True))
    generated = Image.new("RGB", (2048, 1024), "red")

    def generate(self, actual, progress=None):
        assert actual.workflow == "krea-all2real" and not actual.compose
        assert actual.seed == 123456
        assert actual.images[0].size == (1024, 512)
        assert actual.images[0].getpixel((0, 0)) == (0, 0, 255)
        graph = configure_krea_all2real_graph(krea_all2real_template(), actual, ("source.png",))
        assert graph["53"]["inputs"]["noise_seed"] == graph["54"]["inputs"]["noise_seed"] == 123456
        assert graph["90"]["inputs"]["seed"] == 123457
        return [generated]

    monkeypatch.setattr(comfy_swap.ComfyKreaSwapAdapter, "generate", generate)
    result = engine.generate(req, "Off", 0)
    assert unloaded == [True] and result.images == [generated] and result.seed == 123456
    assert source.size == (1536, 768) and source.tobytes() == source_bytes
    assert result.images[0].size == (2048, 1024)
    metadata = json.loads((result.saved_paths[0].parent / "metadata.json").read_text())
    assert metadata["source_budget"] == metadata["uploaded_source_size"] == [1024, 512]
    assert metadata["actual_output_sizes"] == [[2048, 1024]]
    assert metadata["native_wan_vae_upscale_requested"] and metadata["width_height_are_source_budget"]
    assert metadata["upstream_vae_wrapper_parity_verified"] is False
    assert metadata["graph_source_scaling"] == "1MP then FluxKontext aspect buckets"
    assert any("not diffusion/output dimensions" in note for note in result.notes)
    assert any("not established" in note for note in result.notes)


@pytest.mark.parametrize("case", ["images", "count", "mask", "steps", "guidance", "restoration"])
def test_engine_rejects_invalid_before_truncation_or_model_loading(assets, monkeypatch, case):
    changes = {"images": [Image.new("RGB", (8, 8))] * 2, "count": 2,
               "mask": Image.new("L", (8, 8)), "steps": 4, "guidance": 2}
    req = request(**({case: changes[case]} if case != "restoration" else {}))
    monkeypatch.setattr(engine.model_manager, "get", lambda *args: pytest.fail("must not load"))
    monkeypatch.setattr(engine.model_manager, "unload", lambda: pytest.fail("must not unload"))
    with pytest.raises(ValueError):
        engine.generate(req, "GFPGAN" if case == "restoration" else "Off", 0)


@pytest.fixture
def custom_nodes():
    spec = importlib.util.spec_from_file_location("all2real_test_nodes", ROOT / "scripts" / "comfy_all2real_nodes.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_custom_reference_removal_preserves_embeddings_metadata_and_input(custom_nodes):
    embedding, other_embedding, nested = object(), object(), {"keep": [1, 2]}
    metadata = {"reference_latents": [object()], "reference_latents_method": "index_timestep_zero",
                "pooled_output": object(), "strength": 0.8, "custom": nested}
    conditioning = [[embedding, metadata], [other_embedding, {"custom": nested}]]
    result, = custom_nodes.StudioAll2RealRemoveReferences().remove_references(conditioning)
    assert result is not conditioning and result[0] is not conditioning[0]
    assert result[0][0] is embedding and result[1][0] is other_embedding
    assert result[0][1] == {key: value for key, value in metadata.items() if not key.startswith("reference_latents")}
    assert result[0][1] is not metadata and result[0][1]["custom"] is nested
    assert result[1][1] == conditioning[1][1] and result[1][1] is not conditioning[1][1]
    assert "reference_latents" in metadata and "reference_latents_method" in metadata
    assert set(custom_nodes.NODE_CLASS_MAPPINGS) == {
        "StudioAll2RealRemoveReferences", "StudioAll2RealLatentNoise", "StudioAll2RealSkinDetail",
        "StudioAll2RealVAEDecode",
    }


@pytest.mark.parametrize("dtype_name", ["float32", "float16"])
def test_custom_noise_deterministic_seed_no_tensor_metadata_or_rng_mutation(custom_nodes, dtype_name):
    torch = pytest.importorskip("torch")
    tensor = torch.arange(24, dtype=getattr(torch, dtype_name)).reshape(1, 2, 3, 4)
    original = tensor.clone()
    mask = object()
    samples = {"samples": tensor, "noise_mask": mask, "batch_index": [5]}
    rng = torch.random.get_rng_state().clone()
    node = custom_nodes.StudioAll2RealLatentNoise()
    first, = node.inject_noise(samples, 0.3, 43)
    second, = node.inject_noise(samples, 0.3, 43)
    different, = node.inject_noise(samples, 0.3, 44)
    zero, = node.inject_noise(samples, 0, 43)
    expected_noise = torch.randn(tensor.shape, generator=torch.Generator(device="cpu").manual_seed(43), dtype=torch.float32)
    assert torch.equal(first["samples"], tensor + expected_noise.to(tensor.dtype) * 0.3)
    assert torch.equal(first["samples"], second["samples"])
    assert not torch.equal(first["samples"], different["samples"])
    assert torch.equal(zero["samples"], tensor)
    assert torch.equal(tensor, original) and torch.equal(torch.random.get_rng_state(), rng)
    assert first is not samples and first["samples"].data_ptr() != tensor.data_ptr()
    assert first["samples"].dtype == tensor.dtype and first["samples"].device == tensor.device
    assert first["noise_mask"] is mask and first["batch_index"] is samples["batch_index"]


@pytest.mark.parametrize("std,seed", [(-0.1, 1), (2.1, 1), (float("nan"), 1), (float("inf"), 1),
                                    (0.3, -1), (0.3, 2**64), (0.3, True), (0.3, 1.5)])
def test_custom_noise_rejects_invalid_parameters(custom_nodes, std, seed):
    torch = pytest.importorskip("torch")
    with pytest.raises(ValueError):
        custom_nodes.StudioAll2RealLatentNoise().inject_noise({"samples": torch.zeros(1, 2, 2, 2)}, std, seed)


def test_custom_noise_rejects_integer_latents(custom_nodes):
    torch = pytest.importorskip("torch")
    with pytest.raises(ValueError, match="floating-point"):
        custom_nodes.StudioAll2RealLatentNoise().inject_noise({"samples": torch.zeros(1, dtype=torch.int64)}, 0.3, 1)


@pytest.fixture
def upscaler_stub(monkeypatch):
    calls = []

    class ImageUpscaleWithModel:
        def upscale(self, *, upscale_model, image):
            calls.append((upscale_model, image.clone(), image.is_contiguous()))
            return (image + len(calls),)

    package = ModuleType("comfy_extras")
    module = ModuleType("comfy_extras.nodes_upscale_model")
    module.ImageUpscaleWithModel = ImageUpscaleWithModel
    monkeypatch.setitem(sys.modules, "comfy_extras", package)
    monkeypatch.setitem(sys.modules, "comfy_extras.nodes_upscale_model", module)
    return calls, module


@pytest.mark.parametrize("shape", [(1, 4, 6, 3), (2, 5, 7, 3)])
def test_custom_skin_detail_four_tiles_order_preserves_batch_and_odd_edges(custom_nodes, upscaler_stub, shape):
    torch = pytest.importorskip("torch")
    calls, _ = upscaler_stub
    image = torch.arange(torch.tensor(shape).prod().item(), dtype=torch.float32).reshape(shape)
    original = image.clone()
    model = SimpleNamespace(scale=1)
    result, = custom_nodes.StudioAll2RealSkinDetail().enhance(image, model)
    h, w = shape[1] // 2, shape[2] // 2
    tiles = (image[:, :h, :w], image[:, :h, w:], image[:, h:, :w], image[:, h:, w:])
    assert len(calls) == 4
    for (actual_model, tile, contiguous), expected in zip(calls, tiles):
        assert actual_model is model and contiguous and torch.equal(tile, expected)
    expected = torch.cat((torch.cat((tiles[0] + 1, tiles[1] + 2), dim=2),
                          torch.cat((tiles[2] + 3, tiles[3] + 4), dim=2)), dim=1)
    assert torch.equal(result, expected) and tuple(result.shape) == shape
    assert torch.equal(image, original)


@pytest.mark.parametrize("shape,scale", [((0, 4, 4, 3), 1), ((1, 1, 4, 3), 1),
                                       ((1, 4, 1, 3), 1), ((4, 4, 3), 1), ((1, 4, 4, 3), 2)])
def test_custom_skin_detail_rejects_bad_images_or_non1x_model(custom_nodes, upscaler_stub, shape, scale):
    torch = pytest.importorskip("torch")
    calls, _ = upscaler_stub
    with pytest.raises(ValueError):
        custom_nodes.StudioAll2RealSkinDetail().enhance(torch.zeros(shape), SimpleNamespace(scale=scale))
    assert calls == []


def test_custom_skin_detail_rejects_shape_changing_upscaler(custom_nodes, upscaler_stub):
    torch = pytest.importorskip("torch")
    _, module = upscaler_stub
    module.ImageUpscaleWithModel = lambda: SimpleNamespace(upscale=lambda **kwargs: (torch.zeros(1, 1, 1, 3),))
    with pytest.raises(ValueError, match="preserve each tile"):
        custom_nodes.StudioAll2RealSkinDetail().enhance(torch.zeros(1, 4, 4, 3), SimpleNamespace(scale=1))


def test_custom_nodes_vendor_exact_sync_and_lazy_comfy_imports(monkeypatch):
    assert (ROOT / "scripts/comfy_all2real_nodes.py").read_bytes() == (
        ROOT / "vendor/ComfyUI/custom_nodes/photo_edit_all2real.py"
    ).read_bytes()
    # Importing registration must not import either optional runtime package.
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "comfy", None)
    spec = importlib.util.spec_from_file_location(
        "all2real_lazy_test", ROOT / "scripts/comfy_all2real_nodes.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.StudioAll2RealVAEDecode.INPUT_TYPES() == {
        "required": {"samples": ("LATENT",), "vae": ("VAE",)},
    }


@pytest.mark.parametrize("temporal", [False, True])
@pytest.mark.parametrize("raw", [False, True])
def test_custom_decode_packed_mapping_flatten_normalization_and_no_mutation(custom_nodes, temporal, raw):
    torch = pytest.importorskip("torch")
    shape = (2, 3, 2, 3, 12) if temporal else (2, 2, 3, 12)
    packed = torch.arange(torch.tensor(shape).prod().item(), dtype=torch.float32).reshape(shape)
    packed /= packed.numel() - 1
    decoded = packed * 2 - 1 if raw else packed.clone()
    before = decoded.clone()
    latent = torch.zeros(2, 16, 1, 2, 3)
    metadata = {"batch_index": [5], "noise_mask": object()}
    samples = {"samples": latent, **metadata}
    seen = []

    class VAE:
        latent_dim, output_channels, conv_out_channels = 3, 3, 12

        def decode(self, value):
            seen.append(self)
            assert value is latent
            self.copy_only = True
            assert self.decode_tiled_3d.__func__ is custom_nodes._decode_tiled_3d
            return decoded

        def decode_tiled_3d(self, *args, **kwargs):
            pytest.fail("original tiled method must not run")

    vae = VAE()
    original_attributes = vae.__dict__.copy()
    output, = custom_nodes.StudioAll2RealVAEDecode().decode(samples, vae)
    flat = packed.reshape(-1, *packed.shape[-3:])
    expected = torch.empty(flat.shape[0], flat.shape[1] * 2, flat.shape[2] * 2, 3)
    # Independent mapping oracle: c*4 + dy*2 + dx -> RGB channel c.
    for channel in range(3):
        for dy in range(2):
            for dx in range(2):
                expected[:, dy::2, dx::2, channel] = flat[..., channel * 4 + dy * 2 + dx]
    torch.testing.assert_close(output, expected)
    assert seen[0] is not vae and vae.__dict__ == original_attributes
    assert vae.decode_tiled_3d.__func__ is VAE.decode_tiled_3d
    assert torch.equal(decoded, before) and torch.count_nonzero(latent) == 0
    assert samples["samples"] is latent
    assert samples["batch_index"] is metadata["batch_index"]
    assert samples["noise_mask"] is metadata["noise_mask"]


def test_custom_decode_rgb_passthrough_does_not_shuffle_or_double_normalize(custom_nodes):
    torch = pytest.importorskip("torch")
    image = torch.tensor([0.0, 0.25, 1.0]).expand(1, 2, 3, 3).clone()

    class VAE:
        latent_dim, output_channels, conv_out_channels = 3, 3, 3

        def decode(self, samples):
            assert "decode_tiled_3d" not in self.__dict__
            return image

    output, = custom_nodes.StudioAll2RealVAEDecode().decode({"samples": object()}, VAE())
    assert output is image and torch.equal(output[0, 0, 0], torch.tensor([0.0, 0.25, 1.0]))


@pytest.mark.parametrize("shape", [(1, 2, 2, 0), (1, 2, 2, 4), (1, 2, 2, 9),
                                       (1, 2, 2, 13), (2, 2, 12), (1, 1, 1, 2, 2, 12)])
def test_custom_decode_invalid_output_shapes_fail(custom_nodes, shape):
    torch = pytest.importorskip("torch")
    vae = SimpleNamespace(output_channels=3, decode=lambda samples: torch.zeros(shape))
    with pytest.raises(ValueError, match="channels|BHWC"):
        custom_nodes.StudioAll2RealVAEDecode().decode({"samples": object()}, vae)


def test_custom_decode_non_rgb_vae_fails_before_decode(custom_nodes):
    vae = SimpleNamespace(output_channels=4, decode=lambda samples: pytest.fail("must not decode"))
    with pytest.raises(ValueError, match="output_channels=3"):
        custom_nodes.StudioAll2RealVAEDecode().decode({"samples": object()}, vae)


def test_custom_decode_mocked_oom_dispatch_allocates_twelve_tiled_channels(custom_nodes, monkeypatch):
    torch = pytest.importorskip("torch")
    calls = []
    raw = torch.linspace(-1, 1, 24).reshape(1, 12, 1, 1, 2)
    latent = torch.zeros(1, 16, 1, 1, 2, dtype=torch.float64)

    def tiled_scale_multidim(samples, decode_fn, **kwargs):
        calls.append(kwargs)
        assert samples is latent
        decoded = decode_fn(samples)
        assert decoded.dtype == torch.float32
        # Model the allocation/shape check performed by Comfy's tiler.
        allocation = torch.empty(1, kwargs["out_channels"], 1, 1, 2)
        allocation.copy_(decoded)
        return allocation

    module = ModuleType("comfy.utils")
    module.tiled_scale_multidim = tiled_scale_multidim
    monkeypatch.setitem(sys.modules, "comfy", ModuleType("comfy"))
    monkeypatch.setitem(sys.modules, "comfy.utils", module)
    first_stage_calls = []

    class VAE:
        latent_dim, output_channels, conv_out_channels = 3, 3, 12
        vae_dtype, device, output_device = torch.float32, "cpu", "cpu"
        upscale_ratio = (1, 8, 8)
        upscale_index_formula = (object(), object(), object())

        def vae_output_dtype(self):
            return torch.float32

        def process_output(self, images):
            return (images + 1) / 2

        def decode(self, samples):
            try:
                raise torch.OutOfMemoryError("mock regular decode OOM")
            except torch.OutOfMemoryError:
                images = self.decode_tiled_3d(samples, tile_t=1, tile_x=7, tile_y=9,
                                              overlap=(1, 2, 3))
            return images.movedim(1, -1)

        def decode_tiled_3d(self, *args, **kwargs):
            pytest.fail("original allocation3 would fail")

    def model_decode(value):
        first_stage_calls.append(value)
        return raw

    vae = VAE()
    vae.first_stage_model = SimpleNamespace(decode=model_decode)
    output, = custom_nodes.StudioAll2RealVAEDecode().decode({"samples": latent}, vae)
    assert output.shape == (1, 2, 4, 3) and len(first_stage_calls) == 1
    assert first_stage_calls[0].dtype == torch.float32
    assert calls == [{"tile": (1, 7, 9), "overlap": (1, 2, 3),
                      "upscale_amount": vae.upscale_ratio, "out_channels": 12,
                      "index_formulas": vae.upscale_index_formula, "output_device": "cpu"}]
    assert "decode_tiled_3d" not in vae.__dict__
    expected = torch.nn.functional.pixel_shuffle(
        ((raw + 1) / 2).squeeze(2), 2
    ).movedim(1, -1)
    torch.testing.assert_close(output, expected)


@pytest.mark.parametrize("channels", [1, 4, 12])
def test_custom_skin_requires_rgb_before_model_execution(custom_nodes, upscaler_stub, channels):
    torch = pytest.importorskip("torch")
    calls, _ = upscaler_stub
    # Even a model without scale must get the helpful channel error first.
    with pytest.raises(ValueError, match="final RGB3.*StudioAll2RealVAEDecode"):
        custom_nodes.StudioAll2RealSkinDetail().enhance(torch.zeros(1, 4, 4, channels), object())
    assert calls == []