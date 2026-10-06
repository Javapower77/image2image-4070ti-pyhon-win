from __future__ import annotations

from io import BytesIO
from pathlib import Path

import httpx
import pytest
from PIL import Image

from photo_edit_studio import comfy_backend, engine
from photo_edit_studio.comfy_assets import (
    COMFY_KREA_FILES,
    KREA_FIRST_LORA_FILE,
    KREA_REMIX_LORA_FILE,
    missing_krea_remix_assets,
)
from photo_edit_studio.comfy_workflows import krea_remix_template
from photo_edit_studio.config import settings
from photo_edit_studio.loras import NONE_CHOICE, selected_loras
from photo_edit_studio.models import comfy_swap
from photo_edit_studio.models.registry import MODEL_SPECS
from photo_edit_studio.types import GenerationRequest, LoraSpec


def request() -> GenerationRequest:
    return GenerationRequest(
        model_key="krea-2-turbo", workflow="krea-remix", images=[Image.new("RGB", (1536, 768))],
        prompt="remix", negative_prompt="", mask=None, width=2048, height=2048,
        steps=9, guidance=1, true_cfg=1, strength=1, seed=42, size_multiplier=1,
        krea_first_lora_weight=0.65,
    )


@pytest.fixture
def assets(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "comfy_dir", tmp_path)
    monkeypatch.setattr(settings, "lora_dir", tmp_path / "models" / "loras")
    for file in (*COMFY_KREA_FILES, f"loras/{KREA_FIRST_LORA_FILE}", f"loras/{KREA_REMIX_LORA_FILE}"):
        path = tmp_path / "models" / file
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    return tmp_path


def optional_loras(assets, weights):
    family = assets / "models" / "loras" / "krea2"
    family.mkdir(parents=True, exist_ok=True)
    result = []
    for index, weight in enumerate(weights, start=1):
        path = family / f"slot{index}.safetensors"
        path.touch()
        result.append(LoraSpec(path.name, path, weight, f"slot_{index}"))
    return result


def registered(graph):
    result = {}
    for node in graph.values():
        inputs = result.setdefault(node["class_type"], {"input": {"required": {}}})["input"]["required"]
        for key in node["inputs"]:
            inputs[key] = ("ANY",)
    # Exercise real enum registration, including the future upload name exception.
    result["LoadImage"]["input"]["required"]["image"] = (["old.png"],)
    result["LoraLoaderModelOnly"]["input"]["required"]["lora_name"] = (
        [node["inputs"]["lora_name"] for node in graph.values()
         if node["class_type"] == "LoraLoaderModelOnly"],
    )
    result["KSamplerAdvanced"]["input"]["required"]["scheduler"] = (["simple", "kl_optimal"],)
    return result


@pytest.mark.parametrize("steps", [3, 9, 14])
def test_graph_has_source_latent_two_passes_and_exact_adapter_order(assets, steps):
    req = request()
    req.steps = steps
    graph = comfy_swap.configure_krea_remix_graph(krea_remix_template(), req, ("canvas.png",))
    mandatory = graph["80"]["inputs"]["model"][0]
    assert graph[mandatory]["inputs"] == {
        "model": ["55", 0], "lora_name": KREA_FIRST_LORA_FILE, "strength_model": 0.65,
    }
    assert graph["71"]["inputs"]["model"] == [mandatory, 0]
    assert graph["71"]["inputs"]["lora_name"] == KREA_REMIX_LORA_FILE
    assert graph["79"]["inputs"] == {"model": ["71", 0], "kv_cache": True}
    assert graph["80"]["inputs"] == {"model": [mandatory, 0], "kv_cache": True}
    assert graph["84"]["inputs"] == {
        "clip": ["56", 0], "prompt": "remix", "vae": ["57", 0], "image1": ["73", 0],
    }
    assert graph["73"]["inputs"]["image"] == ["72", 0]
    assert (graph["73"]["inputs"]["width"], graph["73"]["inputs"]["height"]) == (1024, 512)
    assert graph["74"]["inputs"]["pixels"] == ["73", 0]
    assert graph["53"]["inputs"] == {
        "model": ["79", 0], "positive": ["84", 0], "negative": ["85", 0],
        "latent_image": ["74", 0], "add_noise": "enable", "noise_seed": 42, "steps": steps,
        "cfg": 1, "sampler_name": "euler", "scheduler": "kl_optimal", "start_at_step": 1,
        "end_at_step": steps - 1, "return_with_leftover_noise": "enable",
    }
    refine = graph["54"]["inputs"]
    assert refine["model"] == ["80", 0] and refine["latent_image"] == ["53", 0]
    assert (refine["start_at_step"], refine["end_at_step"], refine["steps"]) == (steps - 1, steps, steps)
    assert refine["scheduler"] == "simple" and refine["add_noise"] == "disable"
    assert refine["return_with_leftover_noise"] == "disable"
    assert graph["86"] == {"class_type": "CLIPTextEncode", "inputs": {"clip": ["56", 0], "text": "remix"}}
    assert refine["positive"] == ["86", 0]
    assert graph["58"]["inputs"]["samples"] == ["54", 0]
    assert graph["29"]["inputs"]["images"] == ["58", 0]
    classes = [node["class_type"] for node in graph.values()]
    assert classes.count("SaveImage") == 1 and classes.count("LoadImage") == 1
    assert not any("NO8D" in cls or "DetailDaemon" in cls or "Krea2Edit" in cls or "Empty" in cls for cls in classes)
    comfy_swap.preflight_krea_remix_graph(graph, registered(graph))


@pytest.mark.parametrize("weights", [[], [0.7], [-2.0, -0.5, 1.0, 1.25, 2.0]])
def test_optional_chain_is_ordered_first_pass_only(assets, weights):
    req = request()
    req.loras = optional_loras(assets, weights)
    template = krea_remix_template()
    remix_weight = template["71"]["inputs"]["strength_model"]
    refinement = template["80"]
    graph = comfy_swap.configure_krea_remix_graph(template, req, ("canvas.png",))
    mandatory = graph["80"]["inputs"]["model"][0]
    assert graph[mandatory]["inputs"] == {
        "model": ["55", 0], "lora_name": KREA_FIRST_LORA_FILE, "strength_model": 0.65,
    }
    assert graph["71"]["inputs"] == {
        "model": [mandatory, 0], "lora_name": KREA_REMIX_LORA_FILE,
        "strength_model": remix_weight,
    }
    # Walk backward from the first-pass consumer to prove the exact slot order.
    current = graph["79"]["inputs"]["model"][0]
    for lora in reversed(req.loras):
        node = graph[current]
        assert node["class_type"] == "LoraLoaderModelOnly"
        assert node["inputs"]["lora_name"] == lora.name
        assert node["inputs"]["strength_model"] == lora.weight
        current = node["inputs"]["model"][0]
    assert current == "71"
    assert graph["79"]["inputs"]["kv_cache"] is True
    assert graph["80"] is refinement
    assert graph["80"]["inputs"] == {"model": [mandatory, 0], "kv_cache": True}
    assert graph["54"]["inputs"]["model"] == ["80", 0]
    assert graph["53"]["inputs"]["model"] == ["79", 0]
    assert sum(node["class_type"] == "LoraLoaderModelOnly" for node in graph.values()) == 2 + len(weights)
    comfy_swap.preflight_krea_remix_graph(graph, registered(graph))


def test_zero_weight_and_empty_slots_keep_existing_selection_semantics(assets):
    loras = optional_loras(assets, [0, 0.8, 1.2])
    req = request()
    req.loras = selected_loras(
        req.model_key,
        [loras[0].name, NONE_CHOICE, loras[1].name, None, loras[2].name],
        [0, 1, 0.8, 1, 1.2],
    )
    assert req.loras == [
        LoraSpec(loras[1].name, loras[1].path, 0.8, "slot2_3"),
        LoraSpec(loras[2].name, loras[2].path, 1.2, "slot3_5"),
    ]
    graph = comfy_swap.configure_krea_remix_graph(krea_remix_template(), req, ("canvas.png",))
    assert loras[0].name not in [
        node["inputs"].get("lora_name") for node in graph.values()
    ]
    last = graph["79"]["inputs"]["model"][0]
    assert graph[last]["inputs"]["lora_name"] == loras[2].name
    previous = graph[last]["inputs"]["model"][0]
    assert graph[previous]["inputs"]["lora_name"] == loras[1].name
    assert graph[previous]["inputs"]["model"] == ["71", 0]


def test_dlss_is_after_refinement_and_does_not_change_adapter_chain(assets):
    from copy import deepcopy

    req = request()
    req.loras = optional_loras(assets, [0.2, 0.65, 1.0, 1.25, 2.0])
    req.dlss = {}
    graph = comfy_swap.configure_krea_remix_graph(krea_remix_template(), req, ("canvas.png",))
    before = deepcopy(graph)
    comfy_swap.add_dlss_nodes(graph, req)
    enhancer = graph["29"]["inputs"]["images"][0]
    assert graph[enhancer]["class_type"] == "DLSS5EnhanceImages"
    assert graph[enhancer]["inputs"]["images"] == ["58", 0]
    assert graph["58"]["inputs"]["samples"] == ["54", 0]
    assert all(graph[node_id] == node for node_id, node in before.items() if node_id != "29")


def test_direct_zero_optional_does_not_change_remix_chain(assets):
    req = request()
    req.loras = optional_loras(assets, [0])
    graph = comfy_swap.configure_krea_remix_graph(krea_remix_template(), req, ("canvas.png",))
    assert graph["79"]["inputs"]["model"] == ["71", 0]
    assert graph["71"]["inputs"]["strength_model"] == 1
    assert sum(node["class_type"] == "LoraLoaderModelOnly" for node in graph.values()) == 2


@pytest.mark.parametrize("field,value,match", [
    ("model_key", "flux-klein-4b", "only model"),
    ("images", [], "exactly one"),
    ("images", [Image.new("RGB", (64, 64))] * 2, "second image"),
    ("mask", Image.new("L", (64, 64)), "mask"),
    ("count", 2, "single output"),
    ("size_multiplier", 2, "no upscale"),
    ("steps", 2, "steps"),
    ("guidance", 2, "guidance"),
    ("negative_prompt", "bad", "negative prompt"),
    ("krea_first_lora_weight", 0, "greater than 0"),
    ("krea_first_lora_weight", float("nan"), "greater than 0"),
    ("loras", [LoraSpec("extra.safetensors", Path("extra.safetensors"), 1, "extra")], "Krea2"),
])
def test_engine_rejects_unsupported_before_loading_or_silently_truncating(monkeypatch, field, value, match):
    req = request()
    setattr(req, field, value)
    monkeypatch.setattr(engine.model_manager, "get", lambda *args: pytest.fail("must not load"))
    monkeypatch.setattr(engine.model_manager, "unload", lambda: pytest.fail("must not unload"))
    with pytest.raises(ValueError, match=match):
        engine.generate(req, "Off", 0.35)


@pytest.mark.parametrize("case,match", [
    ("too_many", "at most 5"),
    ("mandatory", "mandatory Krea LoRA"),
    ("remix", "required Remix LoRA"),
    ("duplicate", "more than once"),
    ("wrong_family", "Krea2"),
    ("wrong_name", "Krea2"),
    ("wrong_suffix", "Krea2"),
    ("missing", "Selected Krea LoRA is missing"),
    ("negative", "between -2 and 2"),
    ("above_max", "between -2 and 2"),
    ("nan", "finite"),
    ("inf", "finite"),
    ("negative_inf", "finite"),
])
def test_invalid_optional_loras_fail_before_loading_or_graph_mutation(monkeypatch, assets, case, match):
    req = request()
    req.loras = optional_loras(assets, [0.7])
    lora = req.loras[0]
    if case == "too_many":
        req.loras = optional_loras(assets, [1] * 6)
    elif case in {"mandatory", "remix"}:
        name = KREA_FIRST_LORA_FILE if case == "mandatory" else KREA_REMIX_LORA_FILE
        # Case-insensitive reserved-name checks must precede path validation.
        req.loras = [LoraSpec(name.upper(), lora.path.parent / name.upper(), 0.3, "reserved")]
    elif case == "duplicate":
        req.loras.append(LoraSpec(lora.name, lora.path, 1.2, "duplicate"))
    elif case == "wrong_family":
        path = assets / "models" / "loras" / "flux" / lora.name
        path.parent.mkdir()
        path.touch()
        req.loras = [LoraSpec(path.name, path, 1, "foreign")]
    elif case == "wrong_name":
        req.loras = [LoraSpec("other.safetensors", lora.path, 1, "mismatch")]
    elif case == "wrong_suffix":
        path = lora.path.with_suffix(".pt")
        path.touch()
        req.loras = [LoraSpec(path.name, path, 1, "suffix")]
    elif case == "missing":
        lora.path.unlink()
    else:
        weight = {"negative": -2.1, "above_max": 2.1, "nan": float("nan"),
                  "inf": float("inf"), "negative_inf": float("-inf")}[case]
        req.loras = [LoraSpec(lora.name, lora.path, weight, lora.adapter_name)]
    monkeypatch.setattr(engine.model_manager, "get", lambda *args: pytest.fail("must not load"))
    monkeypatch.setattr(engine.model_manager, "unload", lambda: pytest.fail("must not unload"))
    error = FileNotFoundError if case == "missing" else ValueError
    with pytest.raises(error, match=match):
        engine.generate(req, "Off", 0.35)
    graph = krea_remix_template()
    original_ids = set(graph)
    original_remix = graph["71"]["inputs"].copy()
    with pytest.raises(error, match=match):
        comfy_swap.configure_krea_remix_graph(graph, req, ("canvas.png",))
    assert set(graph) == original_ids
    assert graph["71"]["inputs"] == original_remix


def test_engine_routes_remix_not_text_and_preserves_upload(monkeypatch, tmp_path):
    req = request()
    source = req.images[0]
    req.compose = True
    monkeypatch.setattr(settings, "max_output_side", 2048)
    monkeypatch.setattr(settings, "max_output_pixels", 2048 ** 2)
    monkeypatch.setattr(settings, "output_dir", tmp_path)
    monkeypatch.setattr(engine.model_manager, "get", lambda *args: pytest.fail("must use Comfy"))
    unloaded = []
    monkeypatch.setattr(engine.model_manager, "unload", lambda: unloaded.append(True))
    generated = Image.new("RGB", (1024, 512), "red")

    def generate(self, actual, progress=None):
        assert actual.workflow == "krea-remix" and not actual.compose
        assert len(actual.images) == 1 and actual.images[0].size == (1024, 512)
        assert (actual.width, actual.height) == (1024, 512)
        return [generated]

    monkeypatch.setattr(comfy_swap.ComfyKreaSwapAdapter, "generate", generate)
    result = engine.generate(req, "Off", 0.35)
    assert unloaded == [True] and result.images == [generated]
    assert source.size == (1536, 768) and source.getpixel((0, 0)) == (0, 0, 0)
    assert any("approximating" in note for note in result.notes)
    assert len(result.saved_paths) == 1


def test_remix_no_restoration(monkeypatch):
    with pytest.raises(ValueError, match="restoration must be Off"):
        engine.generate(request(), "GFPGAN", 0.35)


def test_graph_honors_custom_prompt_and_lower_memory_guards(assets, monkeypatch):
    req = request()
    req.prompt = "Harmonize the composed canvas"
    monkeypatch.setattr(settings, "max_output_side", 768)
    monkeypatch.setattr(settings, "max_output_pixels", 768 * 384)
    graph = comfy_swap.configure_krea_remix_graph(krea_remix_template(), req, ("canvas.png",))
    assert graph["84"]["inputs"]["prompt"] == graph["86"]["inputs"]["text"] == req.prompt
    assert (graph["73"]["inputs"]["width"], graph["73"]["inputs"]["height"]) == (768, 384)


def test_missing_node_fails_before_upload_or_queue(monkeypatch, assets):
    calls = []

    def handler(http_request):
        calls.append(http_request.url.path)
        assert http_request.method == "GET"
        return httpx.Response(200, json={})

    original_client = httpx.Client
    monkeypatch.setattr(comfy_swap.httpx, "Client", lambda **kwargs: original_client(
        **kwargs, transport=httpx.MockTransport(handler),
    ))
    monkeypatch.setattr(comfy_swap, "ensure_backend", lambda **kwargs: None)
    with pytest.raises(RuntimeError, match="Ostris"):
        comfy_swap.ComfyKreaSwapAdapter(MODEL_SPECS["krea-2-turbo"]).generate(request())
    assert calls == ["/system_stats", "/object_info"]


def test_assets_require_remix_not_identity(assets):
    assert missing_krea_remix_assets(assets) == []
    (assets / "models" / "loras" / KREA_REMIX_LORA_FILE).unlink()
    with pytest.raises(FileNotFoundError, match="identity-edit is not a substitute"):
        comfy_swap.ComfyKreaSwapAdapter(MODEL_SPECS["krea-2-turbo"]).generate(request())


@pytest.mark.parametrize("missing", ["Krea2OstrisEditModelPatch", "TextEncodeKrea2OstrisEdit"])
def test_missing_ostris_nodes_are_not_substituted(assets, missing):
    graph = comfy_swap.configure_krea_remix_graph(krea_remix_template(), request(), ("source.png",))
    info = registered(graph)
    info.pop(missing)
    info["Krea2EditModelPatch"] = {}
    with pytest.raises(RuntimeError, match="Identity-edit nodes cannot substitute"):
        comfy_swap.preflight_krea_remix_graph(graph, info)


def test_preflight_checks_kv_cache_and_scheduler(assets):
    graph = comfy_swap.configure_krea_remix_graph(krea_remix_template(), request(), ("source.png",))
    info = registered(graph)
    info["Krea2OstrisEditModelPatch"]["input"]["required"].pop("kv_cache")
    with pytest.raises(RuntimeError, match="lacks input kv_cache"):
        comfy_swap.preflight_krea_remix_graph(graph, info)
    info = registered(graph)
    info["KSamplerAdvanced"]["input"]["required"]["scheduler"] = (["simple"],)
    with pytest.raises(RuntimeError, match="scheduler=kl_optimal"):
        comfy_swap.preflight_krea_remix_graph(graph, info)


def test_adapter_preflights_before_upload_and_returns_only_save29(monkeypatch, assets):
    req = request()
    req.loras = optional_loras(assets, [0.4, 1.2])
    graph = comfy_swap.configure_krea_remix_graph(krea_remix_template(), req, ("source.png",))
    calls = []
    started = []
    image = BytesIO()
    Image.new("RGB", (1024, 512), "red").save(image, format="PNG")

    def handler(http_request):
        path = http_request.url.path
        calls.append(path)
        if path == "/system_stats":
            return httpx.Response(200, json={})
        if path == "/object_info":
            return httpx.Response(200, json=registered(graph))
        if path == "/upload/image":
            return httpx.Response(200, json={"name": "uploaded.png"})
        if path == "/prompt":
            import json
            payload = json.loads(http_request.content)["prompt"]
            assert payload["72"]["inputs"]["image"] == "uploaded.png"
            assert payload["84"]["class_type"] == "TextEncodeKrea2OstrisEdit"
            assert payload["79"] == graph["79"]
            assert payload["80"] == graph["80"]
            for node_id, node in graph.items():
                if node["class_type"] == "LoraLoaderModelOnly":
                    assert payload[node_id] == node
            return httpx.Response(200, json={"prompt_id": "job"})
        if path == "/history/job":
            return httpx.Response(200, json={"job": {"outputs": {
                "29": {"images": [{"filename": "generated.png"}]},
                "72": {"images": [{"filename": "source.png"}]},
            }}})
        assert path == "/view" and http_request.url.params["filename"] == "generated.png"
        return httpx.Response(200, content=image.getvalue())

    original_client = httpx.Client
    monkeypatch.setattr(comfy_swap.httpx, "Client", lambda **kwargs: original_client(
        **kwargs, transport=httpx.MockTransport(handler),
    ))
    monkeypatch.setattr(comfy_swap, "ensure_backend", lambda **kwargs: started.append(kwargs))
    outputs = comfy_swap.ComfyKreaSwapAdapter(MODEL_SPECS["krea-2-turbo"]).generate(req)
    assert started == [{"workflow": "krea-remix"}]
    assert calls.index("/object_info") < calls.index("/upload/image") < calls.index("/prompt")
    assert len(outputs) == 1 and outputs[0].getpixel((0, 0)) == (255, 0, 0)


def test_remix_autostart_does_not_require_identity_asset(monkeypatch, assets):
    (assets / "main.py").touch()
    monkeypatch.setattr(comfy_backend, "_process", None)
    monkeypatch.setattr(comfy_backend, "is_ready", lambda: False)
    monkeypatch.setattr(comfy_backend, "missing_assets", lambda root: pytest.fail("identity preflight is wrong"))
    (assets / "models" / "loras" / KREA_REMIX_LORA_FILE).unlink()
    with pytest.raises(FileNotFoundError, match=KREA_REMIX_LORA_FILE):
        comfy_backend.ensure_backend(enabled=True, workflow="krea-remix")


def test_setup_installs_ostris_nodes_without_remix_download():
    script = (Path(__file__).parents[1] / "scripts" / "setup-comfy.ps1").read_text()
    assert "https://github.com/ostris/ComfyUI-Krea2-Ostris-Edit.git" in script
    assert "manually place Krea2-Remix_Patreon.safetensors" in script