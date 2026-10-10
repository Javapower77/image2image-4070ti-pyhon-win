"""Offline contracts for the dedicated BF16 sheet route; no publisher text copied."""
from __future__ import annotations

import hashlib
import io
import json
import warnings
import zipfile
from copy import deepcopy

import httpx
import pytest
from PIL import Image

from photo_edit_studio import engine
from photo_edit_studio import qwen_character_sheet as sheet
from photo_edit_studio.config import settings
from photo_edit_studio.models import MODEL_SPECS
from photo_edit_studio.types import GenerationRequest, LoraSpec


def request(**changes):
    values = {"model_key": sheet.QWEN_CHARACTER_SHEET_KEY,
              "workflow": sheet.QWEN_CHARACTER_SHEET_WORKFLOW,
              "prompt": "private customization", "negative_prompt": "",
              "images": [Image.new("RGB", (2400, 1200))], "mask": None,
              "width": 64, "height": 64, "steps": 25, "guidance": 1.0, "true_cfg": 1.0,
              "strength": 1.0, "seed": 17, "sheet_entity_name": "Test Entity"}
    values.update(changes)
    return GenerationRequest(**values)


def documents():
    result = {}
    for profile, (filename, group, static_id) in sheet.QWEN_CHARACTER_SHEET_PROFILES.items():
        # Two subgraphs, including decoy IDs, ensure extraction is scoped.
        result[filename] = {"definitions": {"subgraphs": [
            {"id": "decoy", "nodes": [{"id": 478, "widgets_values": ["wrong"]}]},
            {"id": group, "nodes": [
                {"id": 478, "widgets_values_named": {"value": f"{profile} synthetic system"},
                 "widgets_values": ["wrong fallback"]},
                {"id": static_id, "widgets_values": [f"{profile} synthetic Ayaka sheet"]},
            ]},
        ]}}
    return result


def install_archive(monkeypatch, docs=None, extras=()):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, document in (documents() if docs is None else docs).items():
            archive.writestr(name, json.dumps(document))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            for name, content in extras:
                archive.writestr(name, content)
    raw = buffer.getvalue()
    path = sheet.qwen_character_sheet_archive_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    monkeypatch.setattr(sheet, "QWEN_CHARACTER_SHEET_SHA256", hashlib.sha256(raw).hexdigest())
    return path


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "model_dir", tmp_path / "models")
    monkeypatch.setattr(settings, "lora_dir", tmp_path / "loras")
    monkeypatch.setattr(settings, "comfy_dir", tmp_path / "comfy")
    monkeypatch.setattr(settings, "output_dir", tmp_path / "outputs")
    monkeypatch.setattr(settings, "comfy_url", "http://127.0.0.1:8188")
    # Every HTTP test below must explicitly replace this fail-closed client.
    monkeypatch.setattr(sheet.httpx, "Client", lambda **kw: pytest.fail("unmocked HTTP"))
    install_archive(monkeypatch)


@pytest.mark.parametrize("profile", ["Production", "Simple"])
def test_prompt_bundle_scoped_system_and_static(profile):
    assert sheet.load_qwen_character_sheet_prompts(profile) == (
        f"{profile} synthetic system", f"{profile} synthetic Ayaka sheet")


@pytest.mark.parametrize("mutation", ["group-missing", "group-duplicate", "node-missing",
                                      "node-duplicate", "blank", "nonstring"])
def test_parser_rejects_ambiguous_or_invalid_prompt(monkeypatch, mutation):
    docs = documents()
    filename, _, _ = sheet.QWEN_CHARACTER_SHEET_PROFILES["Simple"]
    groups = docs[filename]["definitions"]["subgraphs"]
    nodes = groups[1]["nodes"]
    if mutation == "group-missing":
        groups.pop()
    elif mutation == "group-duplicate":
        groups.append(deepcopy(groups[1]))
    elif mutation == "node-missing":
        nodes.pop(0)
    elif mutation == "node-duplicate":
        nodes.append(deepcopy(nodes[0]))
    else:
        nodes[0]["widgets_values_named"]["value"] = " " if mutation == "blank" else 42
    install_archive(monkeypatch, docs)
    with pytest.raises(ValueError):
        sheet.load_qwen_character_sheet_prompts("Simple")


def test_parser_fallback_and_named_precedence(monkeypatch):
    docs = documents()
    filename, _, _ = sheet.QWEN_CHARACTER_SHEET_PROFILES["Simple"]
    node = docs[filename]["definitions"]["subgraphs"][1]["nodes"][0]
    del node["widgets_values_named"]
    node["widgets_values"] = ["fallback system"]
    install_archive(monkeypatch, docs)
    assert sheet.load_qwen_character_sheet_prompts()[0] == "fallback system"


def test_parser_hash_first_and_no_extraction(monkeypatch):
    monkeypatch.setattr(zipfile.ZipFile, "extract", lambda *a, **kw: pytest.fail("extract"))
    monkeypatch.setattr(zipfile.ZipFile, "extractall", lambda *a, **kw: pytest.fail("extractall"))
    assert sheet.load_qwen_character_sheet_prompts()[0] == "Simple synthetic system"
    sheet.qwen_character_sheet_archive_path().write_bytes(b"changed")
    monkeypatch.setattr(zipfile, "ZipFile", lambda *a, **kw: pytest.fail("parsed before hash"))
    with pytest.raises(ValueError, match="SHA256"):
        sheet.load_qwen_character_sheet_prompts()


@pytest.mark.parametrize("kind", ["missing", "duplicate", "many", "archive-limit", "member-limit"])
def test_parser_archive_guards(monkeypatch, kind):
    filename = sheet.QWEN_CHARACTER_SHEET_PROFILES["Simple"][0]
    if kind == "missing":
        sheet.qwen_character_sheet_archive_path().unlink()
        error = FileNotFoundError
    else:
        error = ValueError
        extras = [(filename, "{}")] if kind == "duplicate" else (
            [(f"extra{i}", "") for i in range(127)] if kind == "many" else [])
        install_archive(monkeypatch, extras=extras)
        if kind == "archive-limit":
            monkeypatch.setattr(sheet, "_MAX_ARCHIVE_BYTES", 10)
        if kind == "member-limit":
            monkeypatch.setattr(sheet, "_MAX_MEMBER_BYTES", 10)
    with pytest.raises(error):
        sheet.load_qwen_character_sheet_prompts()


@pytest.mark.parametrize("mp,size", [(1, (1248, 832)), (3.4, (2304, 1536)), (6, (3072, 2048))])
def test_binary_megapixel_canvas(mp, size):
    assert sheet.qwen_character_sheet_size(mp) == size


@pytest.mark.parametrize("mp", [True, "3.4", None, 0, 6.1, float("nan"), float("inf")])
def test_invalid_canvas(mp):
    with pytest.raises(ValueError):
        sheet.qwen_character_sheet_size(mp)


@pytest.mark.parametrize("profile", ["Production", "Simple"])
@pytest.mark.parametrize("mode", ["Static", "Auto"])
def test_exact_graph_profiles(profile, mode):
    req = request(sheet_layout=profile, sheet_prompt_mode=mode,
                  sheet_character_description="private description")
    graph = sheet.configure_qwen_character_sheet_graph(req, "folder/source.png")
    expected = {"1": "UNETLoader", "2": "ModelAttentionBackend", "3": "QwenImage21Cache",
                "4": "CLIPLoader", "5": "VAELoader", "6": "LoadImage",
                "8": "TextEncodeQwenImage21", "9": "EmptyLatentImage", "10": "KSampler",
                "11": "VAEDecode", "12": "VAEDeGrid", "29": "SaveImage"}
    if mode == "Auto":
        expected.update({"13": "CLIPLoader", "14": "TextGenerate"})
    assert {key: node["class_type"] for key, node in graph.items()} == expected
    assert graph["1"]["inputs"] == {"unet_name": "qwen_image_2.1_bf16.safetensors", "weight_dtype": "default"}
    assert graph["4"]["inputs"]["clip_name"] == "qwen3vl_8b_bf16.safetensors"
    assert graph["5"]["inputs"]["vae_name"] == "qwen_image_2.1_vae_bf16.safetensors"
    assert graph["2"]["inputs"] == {"model": ["1", 0], "attention": "comfy kitchen attention"}
    assert graph["3"]["inputs"] == {"model": ["2", 0], "device": "auto", "dtype": "default"}
    assert graph["6"]["inputs"]["image"] == "folder/source.png"
    assert graph["8"]["inputs"]["images.image_1"] == ["6", 0]
    assert graph["8"]["inputs"]["resolution"] == 1536
    assert graph["9"]["inputs"] == {"width": 2304, "height": 1536, "batch_size": 1}
    assert graph["10"]["inputs"] == {"model": ["3", 0], "positive": ["8", 0], "negative": ["8", 1],
        "latent_image": ["9", 0], "seed": 17, "steps": 25, "cfg": 1.0, "sampler_name": "res_multistep",
        "scheduler": "beta", "denoise": 1.0}
    assert graph["11"]["inputs"] == {"samples": ["10", 0], "vae": ["5", 0]}
    assert graph["12"]["inputs"] == {"image": ["11", 0], "enabled": True, "mode": "auto",
        "limit": 0.02, "skip_when_clean": True, "grid_gain": 10, "grid_view": "4x zoom"}
    assert graph["29"]["inputs"]["images"] == ["12", 0]  # never diagnostic output 1
    assert graph["29"]["inputs"]["filename_prefix"].startswith("photo_edit_qwen_sheet_")
    assert "turbo" not in json.dumps(graph).lower()
    if mode == "Static":
        assert graph["8"]["inputs"]["prompt"] == f"{profile} synthetic Test Entity sheet\n\nCustomization:\nprivate customization"
        assert "Ayaka" not in graph["8"]["inputs"]["prompt"]
    else:
        assert graph["8"]["inputs"]["prompt"] == ["14", 0]
        assert graph["14"]["inputs"] == {"clip": ["13", 0], "image": ["6", 0],
            "prompt": ("Entity name: Test Entity" if profile == "Production" else "private description")
                   + "\n\nCustomization:\nprivate customization",
            "system_prompt": f"{profile} synthetic system", "thinking": True, "use_default_template": True,
            "max_length": 2560 if profile == "Production" else 2048, "sampling_mode": "off"}
        assert graph["13"]["inputs"]["clip_name"] == "qwen3.5_4b_int8_convrot.safetensors"


def test_static_neutral_identity_and_no_customization():
    graph = sheet.configure_qwen_character_sheet_graph(request(prompt=" ", sheet_entity_name=""))
    assert graph["8"]["inputs"]["prompt"] == "Simple synthetic the reference entity sheet"


@pytest.mark.parametrize("field,value", [
    ("model_key", "qwen-2.1-turbo"), ("workflow", "standard"), ("sheet_layout", "Other"),
    ("sheet_prompt_mode", "Manual"), ("images", []), ("images", [Image.new("RGB", (1, 1))] * 2),
    ("mask", Image.new("L", (1, 1))), ("count", 2), ("count", True), ("count", 1.0),
    ("size_multiplier", 2), ("swap_kind", "Head"), ("steps", 6), ("steps", 25.0),
    ("guidance", 0), ("guidance", 2), ("guidance", float("nan")), ("true_cfg", 0),
    ("negative_prompt", "bad"), ("seed", -1), ("seed", True), ("seed", 2**64),
    ("prompt", "bad\x00"), ("sheet_entity_name", None), ("sheet_character_description", "\x00"),
])
def test_request_guards(field, value):
    with pytest.raises(ValueError):
        sheet.validate_qwen_character_sheet_request(request(**{field: value}))


def lora(name, weight=1):
    path = settings.lora_dir / "qwen21" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    return LoraSpec(name, path, weight, name)


def test_modelonly_chain_slot_order_zero_skip():
    req = request(loras=[lora("a.safetensors", -0.5), lora("zero.safetensors", 0), lora("b.safetensors", 2)])
    graph = sheet.configure_qwen_character_sheet_graph(req)
    assert graph["40"]["inputs"] == {"model": ["1", 0], "lora_name": "qwen21/a.safetensors", "strength_model": -0.5}
    assert "41" not in graph
    assert graph["42"]["inputs"]["model"] == ["40", 0]
    assert graph["2"]["inputs"]["model"] == ["42", 0]


@pytest.mark.parametrize("kind", ["duplicate", "outside", "extension", "missing", "weight", "six"])
def test_optional_lora_guards(kind):
    item = lora("a.safetensors")
    items = [item]
    if kind == "duplicate":
        items *= 2
    elif kind == "outside":
        items = [LoraSpec(item.name, settings.lora_dir / item.name, 1, "a")]
    elif kind == "extension":
        items = [lora("a.pt")]
    elif kind == "missing":
        item.path.unlink()
    elif kind == "weight":
        items = [lora("a.safetensors", float("inf"))]
    else:
        items = [lora(f"{i}.safetensors") for i in range(6)]
    with pytest.raises((ValueError, FileNotFoundError)):
        sheet.validate_qwen_character_sheet_request(request(loras=items))


def install_weights(mode="Static"):
    names = sheet.QWEN_CHARACTER_SHEET_FILES + ((sheet.QWEN_CHARACTER_SHEET_PE,) if mode == "Auto" else ())
    for name in names:
        path = settings.comfy_dir / "models" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()


def test_pe_required_only_for_auto():
    install_weights()
    assert sheet.missing_qwen_character_sheet_assets(settings.comfy_dir) == []
    assert sheet.missing_qwen_character_sheet_assets(settings.comfy_dir, "Auto") == [
        str(settings.comfy_dir / "models" / sheet.QWEN_CHARACTER_SHEET_PE)]
    with pytest.raises(ValueError):
        sheet.missing_qwen_character_sheet_assets(settings.comfy_dir, "Other")


def registry(graph):
    """Synthetic core-style V3 registry, not a live-node compatibility claim."""
    result = {}
    for node in graph.values():
        fields = result.setdefault(node["class_type"], {"input": {"required": {}}})["input"]["required"]
        for field, value in node["inputs"].items():
            if field in {"prompt", "system_prompt", "negative_prompt", "filename_prefix"}:
                fields[field] = ("STRING",)
            elif isinstance(value, str):
                choices = fields.setdefault(field, ([],))[0]
                if value not in choices:
                    choices.append(value)
            else:
                fields[field] = ("ANY",)
    encode = result["TextEncodeQwenImage21"]["input"]["required"]
    del encode["images.image_1"]
    encode["images"] = ("COMFY_AUTOGROW_V3", {"template": {"names": ["image_1"],
        "input": {"required": {"image": ("IMAGE",)}}}})
    result["KSampler"]["input"]["required"]["sampler_name"] = ("COMBO", {"options": ["res_multistep"]})
    if "TextGenerate" in result:
        result["TextGenerate"]["input"]["required"]["sampling_mode"] = (
            "COMFY_DYNAMICCOMBO_V3", {"options": [{"key": "off", "inputs": {}}]})
    return result


@pytest.mark.parametrize("mode", ["Static", "Auto"])
def test_actual_preflight_v3_schema_does_not_mutate_registry(mode):
    graph = sheet.configure_qwen_character_sheet_graph(request(sheet_prompt_mode=mode))
    registered = registry(graph)
    original = deepcopy(registered)
    sheet.preflight_qwen_character_sheet_graph(graph, registered)
    assert registered == original


@pytest.mark.parametrize("kind", ["node", "field", "sampler", "sampled"])
def test_preflight_fails_closed(kind):
    graph = sheet.configure_qwen_character_sheet_graph(request(sheet_prompt_mode="Auto"))
    registered = registry(graph)
    if kind == "node":
        del registered["VAEDeGrid"]
    elif kind == "field":
        del registered["VAEDeGrid"]["input"]["required"]["limit"]
    elif kind == "sampler":
        registered["KSampler"]["input"]["required"]["sampler_name"] = (["euler"],)
    else:
        registered["TextGenerate"]["input"]["required"]["sampling_mode"][1]["options"][0]["inputs"] = {
            "required": {"temperature": ("FLOAT",)}}
    with pytest.raises(RuntimeError):
        sheet.preflight_qwen_character_sheet_graph(graph, registered)


def adapter():
    return sheet.ComfyQwenCharacterSheetAdapter(MODEL_SPECS[sheet.QWEN_CHARACTER_SHEET_KEY])


def test_missing_assets_before_backend_or_upload(monkeypatch):
    from photo_edit_studio import comfy_backend
    monkeypatch.setattr(comfy_backend, "ensure_backend", lambda **kw: pytest.fail("started before assets"))
    with pytest.raises(FileNotFoundError, match="assets missing"):
        adapter().generate(request())
    with pytest.raises(RuntimeError, match="ComfyUI API"):
        adapter().load()


@pytest.mark.parametrize("cuda,vram", [(False, 24), (True, 8)])
def test_gpu_eligibility_before_backend(monkeypatch, cuda, vram):
    from photo_edit_studio import comfy_backend

    install_weights()
    monkeypatch.setattr(sheet, "cuda_available", lambda: cuda)
    monkeypatch.setattr(sheet, "cuda_vram_gb", lambda: vram)
    monkeypatch.setattr(comfy_backend, "ensure_backend", lambda **kw: pytest.fail("backend before GPU guard"))
    with pytest.raises(RuntimeError, match="CUDA"):
        adapter().generate(request())


def mock_adapter_http(monkeypatch, req, failure=None):
    from httpx._client import Client

    from photo_edit_studio import comfy_backend
    install_weights(req.sheet_prompt_mode)
    monkeypatch.setattr(sheet, "cuda_available", lambda: True)
    monkeypatch.setattr(sheet, "cuda_vram_gb", lambda: 12)
    starts = []
    monkeypatch.setattr(comfy_backend, "ensure_backend", lambda **kw: starts.append(kw))
    graph = sheet.configure_qwen_character_sheet_graph(req)
    calls, queued = [], []

    def handler(call):
        calls.append(call.url.path)
        if call.url.path == "/system_stats":
            return httpx.Response(200, json={})
        if call.url.path == "/object_info":
            registered = registry(graph)
            if failure == "preflight":
                del registered["VAEDeGrid"]
            return httpx.Response(200, json=[] if failure == "registry" else registered)
        if call.url.path == "/upload/image":
            # Inspect actual multipart PNG, proving aspect-preserved upload.
            start = call.content.index(b"\x89PNG")
            end = call.content.index(b"IEND", start) + 8
            with Image.open(io.BytesIO(call.content[start:end])) as image:
                assert image.size == (1536, 768)
            return httpx.Response(200, json={"name": "uploaded.png", "subfolder": "refs"})
        if call.url.path == "/prompt":
            queued.append(json.loads(call.content)["prompt"])
            return httpx.Response(400, text="rejected") if failure == "queue" else httpx.Response(200, json={"prompt_id": "job"})
        if call.url.path == "/history/job":
            outputs = [{"filename": "sheet.png", "subfolder": "sheets", "type": "output"}]
            if failure == "multiple":
                outputs *= 2
            return httpx.Response(200, json={"job": {
                "status": {"status_str": "error" if failure == "execution" else "success"},
                "outputs": {"12": {"images": [{"filename": "diagnostic.png"}]},
                            "29": {"images": outputs}}}})
        assert call.url.path == "/view"
        assert dict(call.url.params) == {"filename": "sheet.png", "subfolder": "sheets", "type": "output"}
        data = io.BytesIO()
        size = (64, 64) if failure == "size" else sheet.qwen_character_sheet_size(req.sheet_megapixels)
        Image.new("RGB", size).save(data, format="PNG")
        return httpx.Response(200, content=data.getvalue())

    monkeypatch.setattr(sheet.httpx, "Client", lambda **kw: Client(transport=httpx.MockTransport(handler), **kw))
    return calls, queued, starts


@pytest.mark.parametrize("mode", ["Static", "Auto"])
def test_http_preflight_upload_queue_saver29_retrieval(monkeypatch, mode):
    req = request(sheet_prompt_mode=mode)
    calls, queued, starts = mock_adapter_http(monkeypatch, req)
    result = adapter().generate(req)
    assert [image.size for image in result] == [(2304, 1536)]
    assert calls == ["/system_stats", "/object_info", "/upload/image", "/prompt", "/history/job", "/view"]
    assert queued[0]["6"]["inputs"]["image"] == "refs/uploaded.png"
    assert queued[0]["29"]["inputs"]["images"] == ["12", 0]
    assert starts == [{"model_key": req.model_key, "workflow": req.workflow, "sheet_prompt_mode": mode}]


@pytest.mark.parametrize("failure", ["preflight", "registry", "queue", "multiple", "execution", "size"])
def test_http_failures_no_silent_fallback(monkeypatch, failure):
    req = request()
    calls, _, _ = mock_adapter_http(monkeypatch, req, failure)
    with pytest.raises(RuntimeError):
        adapter().generate(req)
    if failure in {"preflight", "registry"}:
        assert calls == ["/system_stats", "/object_info"]


@pytest.mark.parametrize("mp", [1, 3.4, 6])
@pytest.mark.parametrize("mode", ["Static", "Auto"])
def test_engine_native_budget_and_private_output_metadata(monkeypatch, mp, mode):
    req = request(sheet_megapixels=mp, sheet_prompt_mode=mode, compose=True)
    size = sheet.qwen_character_sheet_size(mp)
    captured = []

    def generate(self, actual, progress=None):
        captured.append(actual)
        assert (actual.width, actual.height) == size
        assert actual.images[0].size == (1536, 768)
        assert actual.compose is False
        return [Image.new("RGB", size)]

    monkeypatch.setattr(sheet.ComfyQwenCharacterSheetAdapter, "generate", generate)
    monkeypatch.setattr(engine.model_manager, "unload", lambda: None)
    monkeypatch.setattr(engine.model_manager, "get", lambda *a: pytest.fail("normal model loader"))
    monkeypatch.setattr(engine, "diffusion_output_size", lambda *a, **kw: pytest.fail("global clamp"))
    monkeypatch.setattr(engine, "restore_faces", lambda *a: pytest.fail("restoration"))
    result = engine.generate(req, "Off", 0)
    assert len(captured) == 1 and result.images[0].size == size
    path = result.saved_paths[0]
    metadata = json.loads((path.parent / "metadata.json").read_text())
    with Image.open(path) as image:
        assert image.size == size
        assert json.loads(image.info["generation"]) == metadata
    assert metadata["native_diffusion_size"] == list(size)
    assert metadata["actual_output_sizes"] == [list(size)]
    assert metadata["uploaded_source_size"] == [1536, 768]
    assert metadata["source_max_side"] == 1536 and metadata["source_aspect_preserved"] is True
    assert metadata["weight_precision"] == "full BF16"
    assert metadata["sheet_archive_sha256"] == sheet.QWEN_CHARACTER_SHEET_SHA256
    assert metadata["sampler"] == "res_multistep" and metadata["scheduler"] == "beta"
    assert metadata["negative_conditioning_active"] is False
    assert metadata["vlm_sampling_mode"] == ("off" if mode == "Auto" else None)
    assert metadata["vlm_thinking"] is (mode == "Auto")
    assert metadata["vlm_max_length"] == (2048 if mode == "Auto" else None)
    assert metadata["prompt_stored"] is False
    assert "private customization" not in json.dumps(metadata)
    assert "Test Entity" not in json.dumps(metadata)
    assert "mandatory_turbo_lora" not in metadata
    assert "Low-VRAM limit" not in " ".join(result.notes)


@pytest.mark.parametrize("failure", ["restoration", "size", "count"])
def test_engine_rejects_incompatible_outputs(monkeypatch, failure):
    def generate(self, actual, progress=None):
        return [Image.new("RGB", (64, 64))] if failure == "size" else []
    monkeypatch.setattr(sheet.ComfyQwenCharacterSheetAdapter, "generate", generate)
    monkeypatch.setattr(engine.model_manager, "unload", lambda: None)
    with pytest.raises(ValueError if failure == "restoration" else RuntimeError):
        engine.generate(request(), "GFPGAN" if failure == "restoration" else "Off", 0)