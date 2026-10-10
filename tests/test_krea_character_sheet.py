"""Character-sheet contracts: fake assets, mocked HTTP, no model inference/downloads."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from io import BytesIO
from types import SimpleNamespace

import httpx
import pytest
from PIL import Image

from photo_edit_studio import comfy_backend, engine, ui
from photo_edit_studio import krea_character_sheet as sheets
from photo_edit_studio.comfy_assets import COMFY_KREA_FILES, KREA_FIRST_LORA_FILE
from photo_edit_studio.config import settings
from photo_edit_studio.models import comfy_swap
from photo_edit_studio.models.registry import MODEL_SPECS
from photo_edit_studio.types import GenerationRequest, LoraSpec

PROFILES = tuple(sheets.KREA_CHARACTER_SHEET_PROFILES)
OPERATIONS = ("QuadView — Krea 2", "DynamicCharacterSheet (experimental)")
MANUAL = (
    "[TASK: ENTITY_SHEET_GENERATION]\n[TEMPLATE: MULTI_ANGLE_ENTITY_SHEET_V1]\n"
    "[ENTITY_TYPE: character]\n[ENTITY_ID: blue coat]"
)
CAPTION = "Full fake publisher template\nPreserve every line and trailing whitespace.  \n"


def request(**changes):
    req = GenerationRequest(
        model_key="krea-2-turbo", workflow="krea-quadview",
        images=[Image.new("RGB", (768, 1536), "blue")], prompt="", negative_prompt="",
        mask=None, width=512, height=512, steps=10, guidance=1, true_cfg=0,
        strength=1, seed=42, krea_first_lora_weight=0.65,
    )
    for name, value in changes.items():
        setattr(req, name, value)
    return req


@pytest.fixture
def assets(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "comfy_dir", tmp_path / "comfy")
    monkeypatch.setattr(settings, "model_dir", tmp_path / "models")
    monkeypatch.setattr(settings, "lora_dir", tmp_path / "library")
    monkeypatch.setattr(settings, "output_dir", tmp_path / "outputs")
    monkeypatch.setattr(settings, "comfy_url", "http://127.0.0.1:8188")
    shared = [settings.comfy_dir / "models" / name for name in COMFY_KREA_FILES]
    shared.append(settings.comfy_dir / "models/loras" / KREA_FIRST_LORA_FILE)
    profiles = [settings.lora_dir / "krea2" / name
                for name in sheets.KREA_CHARACTER_SHEET_PROFILES.values()]
    for path in shared + profiles:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    template = sheets.dynamic_template_path()
    template.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps({"nodes": [
        {"id": 184, "type": "PrimitiveStringMultiline", "widgets_values": [CAPTION]},
        {"id": 185, "type": "PrimitiveStringMultiline", "widgets_values": ["decoy"]},
    ]}).encode()
    template.write_bytes(raw)
    monkeypatch.setattr(sheets, "KREA_DYNAMIC_TEMPLATE_SHA256", hashlib.sha256(raw).hexdigest())
    return shared, profiles, template


def optional_loras(weights):
    result = []
    for index, weight in enumerate(weights):
        path = settings.lora_dir / "krea2" / f"style{index}.safetensors"
        path.touch()
        result.append(LoraSpec(path.name, path, weight, f"slot{index}"))
    return result


def configured(req):
    return sheets.configure_krea_character_sheet_graph(
        sheets.krea_character_sheet_template(req.workflow), req, ("source.png",),
    )


def registered(graph):
    info = {}
    for node in graph.values():
        fields = info.setdefault(node["class_type"], {"input": {"required": {}}})["input"]["required"]
        fields.update({name: ("ANY",) for name in node["inputs"]})
    for cls, field in (("UNETLoader", "unet_name"), ("CLIPLoader", "clip_name"),
                       ("VAELoader", "vae_name"), ("LoraLoaderModelOnly", "lora_name")):
        choices = [node["inputs"][field].replace("/", "\\") for node in graph.values()
                   if node["class_type"] == cls]
        info[cls]["input"]["required"][field] = (choices,)
    info["KSampler"]["input"]["required"].update(
        sampler_name=(["euler", "lcm"],), scheduler=(["simple"],),
    )
    info["LoadImage"]["input"]["required"]["image"] = (["old.png"],)
    if "TextGenerate" in info:
        info["TextGenerate"]["input"]["required"]["sampling_mode"] = (
            "COMFY_DYNAMICCOMBO_V3", {"options": [{"key": "off", "inputs": {}}]},
        )
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


@pytest.mark.parametrize("workflow", PROFILES)
@pytest.mark.parametrize("weights", [[], [0], [-2, -0.5, 0, 1.25, 2]])
def test_active_model_chain_and_native_source_conditioning(assets, workflow, weights):
    req = request(workflow=workflow, loras=optional_loras(weights), steps=17, guidance=0.5)
    graph = configured(req)
    assert graph["53"]["inputs"]["model"] == ["120", 0]
    current = graph["120"]["inputs"]["model"][0]
    active = [lora for lora in req.loras if lora.weight != 0]
    for lora in reversed(active):
        assert graph[current]["inputs"]["lora_name"] == "krea2/" + lora.name
        assert graph[current]["inputs"]["strength_model"] == lora.weight
        current = graph[current]["inputs"]["model"][0]
    assert current == "71"
    profile = graph["71"]["inputs"]
    assert profile["lora_name"] == "krea2/" + sheets.KREA_CHARACTER_SHEET_PROFILES[workflow]
    assert profile["strength_model"] == 1
    mandatory = graph[profile["model"][0]]["inputs"]
    assert mandatory == {"model": ["55", 0], "lora_name": KREA_FIRST_LORA_FILE, "strength_model": 0.65}
    assert graph["55"]["inputs"]["unet_name"] == COMFY_KREA_FILES[0].rsplit("/", 1)[-1]
    assert graph["56"]["inputs"]["clip_name"] == COMFY_KREA_FILES[1].rsplit("/", 1)[-1]
    assert graph["57"]["inputs"]["vae_name"] == COMFY_KREA_FILES[2].rsplit("/", 1)[-1]
    assert graph["82"]["inputs"] == {"width": 1536, "height": 1024, "batch_size": 1}
    assert graph["73"]["inputs"] == {"pixels": ["72", 0], "vae": ["57", 0]}
    patch = graph["120"]["inputs"]
    assert {key: value for key, value in patch.items() if key != "model"} == {
        "source_latent": ["73", 0], "vae": ["57", 0], "source_image": ["72", 0],
        "target_latent": ["82", 0], "ref_boost": 1, "ref_boost_a": 1, "fit_mode": "fit",
    }
    for node_id in ("84", "85"):
        assert graph[node_id]["inputs"]["image"] == ["72", 0]
        assert graph[node_id]["inputs"]["grounding_px"] == 0
    assert graph["53"]["inputs"] == {
        "model": ["120", 0], "positive": ["84", 0], "negative": ["85", 0],
        "latent_image": ["82", 0], "seed": 42, "steps": 17, "cfg": 0.5,
        "sampler_name": "lcm" if workflow == PROFILES[1] else "euler",
        "scheduler": "simple", "denoise": 1,
    }
    assert graph["54"]["inputs"] == {"samples": ["53", 0], "vae": ["57", 0]}
    assert graph["29"]["inputs"]["images"] == ["54", 0]
    assert sum(node["class_type"] == "LoraLoaderModelOnly" for node in graph.values()) == 2 + len(active)
    assert sum(node["class_type"] == "SaveImage" for node in graph.values()) == 1
    assert not any(node["class_type"] in {"PreviewImage", "ImageScale", "Krea2OstrisEditModelPatch"}
                   for node in graph.values())
    comfy_swap.preflight_krea_remix_graph(graph, registered(graph), label=workflow)


@pytest.mark.parametrize("workflow", PROFILES)
@pytest.mark.parametrize("weight", [0, -0.5, 1.2])
def test_selected_profile_weight_overrides_without_duplicate(assets, workflow, weight):
    path = settings.lora_dir / "krea2" / sheets.KREA_CHARACTER_SHEET_PROFILES[workflow]
    graph = configured(request(workflow=workflow, loras=[LoraSpec(path.name, path, weight, "sheet")]))
    assert graph["71"]["inputs"]["strength_model"] == weight
    assert graph["120"]["inputs"]["model"] == ["71", 0]
    assert sum(node["class_type"] == "LoraLoaderModelOnly" for node in graph.values()) == 2


@pytest.mark.parametrize("workflow", PROFILES)
@pytest.mark.parametrize("steps,seed,cfg", [(4, 0, 0.25), (40, 2**64 - 1, 2)])
def test_valid_boundary_values_are_not_coerced(assets, workflow, steps, seed, cfg):
    graph = configured(request(workflow=workflow, steps=steps, seed=seed, guidance=cfg))
    assert graph["53"]["inputs"]["steps"] == steps
    assert graph["53"]["inputs"]["seed"] == seed
    assert graph["53"]["inputs"]["cfg"] == cfg


def test_templates_are_independent_and_unknown_profile_is_rejected():
    first, second = sheets.krea_character_sheet_template(), sheets.krea_character_sheet_template()
    first["82"]["inputs"]["width"] = 99
    assert second["82"]["inputs"]["width"] == 1536
    with pytest.raises(ValueError, match="Unknown"):
        sheets.krea_character_sheet_template("not-a-profile")


@pytest.mark.parametrize("workflow", PROFILES)
def test_unselected_profile_and_identity_bundle_are_not_runtime_requirements(assets, workflow):
    other = assets[1][1 - PROFILES.index(workflow)]
    other.unlink()
    if workflow == PROFILES[0]:
        assets[2].unlink()
    assert sheets.missing_krea_character_sheet_assets(settings.comfy_dir, workflow) == []
    assert configured(request(workflow=workflow))["71"]["inputs"]["lora_name"].endswith(
        sheets.KREA_CHARACTER_SHEET_PROFILES[workflow],
    )


@pytest.mark.parametrize("case", ["missing", "hash"])
def test_dynamic_template_checked_by_backend_even_when_disabled(assets, case):
    if case == "missing":
        assets[2].unlink()
    else:
        assets[2].write_bytes(assets[2].read_bytes() + b" ")
    with pytest.raises(FileNotFoundError if case == "missing" else ValueError):
        comfy_backend.ensure_backend(enabled=False, workflow=PROFILES[1])


@pytest.mark.parametrize("prompt", ["", "  blue coat and plain background  "])
def test_quad_trigger_is_always_retained_and_customization_appended(assets, prompt):
    graph = configured(request(prompt=prompt, negative_prompt="bad pose"))
    assert graph["84"]["inputs"]["prompt"] == sheets.KREA_QUADVIEW_TRIGGER + (
        "\n" + prompt.strip() if prompt.strip() else ""
    )
    assert graph["85"]["inputs"]["prompt"] == "bad pose"
    assert "160" not in graph


@pytest.mark.parametrize("prompt", ["", "  \n", MANUAL])
def test_dynamic_positive_caption_link_or_manual_bypass(assets, prompt):
    graph = configured(request(workflow=PROFILES[1], prompt=prompt))
    assert sheets.load_dynamic_caption_template() == CAPTION
    assert graph["85"]["inputs"]["prompt"] == sheets.KREA_DYNAMIC_NEGATIVE
    if prompt.strip():
        assert "160" not in graph
        assert graph["84"]["inputs"]["prompt"] == MANUAL
    else:
        assert graph["84"]["inputs"]["prompt"] == ["160", 0]
        assert graph["160"] == {"class_type": "TextGenerate", "inputs": {
            "clip": ["56", 0], "image": ["72", 0], "prompt": CAPTION,
            "max_length": 2048, "sampling_mode": "off", "thinking": False,
            "use_default_template": True,
        }}
    custom = configured(request(workflow=PROFILES[1], prompt=prompt, negative_prompt="  bad pose  "))
    assert custom["85"]["inputs"]["prompt"] == "bad pose"


@pytest.mark.parametrize("prompt", ["ordinary description", MANUAL.replace("[ENTITY_ID: blue coat]", ""),
                                  MANUAL.replace("character", " "), MANUAL.replace("blue coat", " "),
                                  MANUAL.replace("MULTI_ANGLE_ENTITY_SHEET_V1", "other")])
def test_dynamic_rejects_incomplete_manual_headers(assets, prompt):
    with pytest.raises(ValueError, match="headers"):
        configured(request(workflow=PROFILES[1], prompt=prompt))


@pytest.mark.parametrize("case", ["hash", "missing", "node", "duplicate", "type", "empty", "not-string"])
def test_pinned_template_failures_even_for_manual_bypass(assets, monkeypatch, case):
    path = assets[2]
    error = ValueError
    if case == "missing":
        path.unlink()
        error = FileNotFoundError
    elif case == "hash":
        path.write_bytes(path.read_bytes() + b" ")
    else:
        nodes = [{"id": 184, "type": "PrimitiveStringMultiline", "widgets_values": [CAPTION]}]
        if case == "node":
            nodes[0]["id"] = 183
        elif case == "duplicate":
            nodes *= 2
        elif case == "type":
            nodes[0]["type"] = "Other"
        else:
            nodes[0]["widgets_values"] = [" " if case == "empty" else 12]
        raw = json.dumps({"nodes": nodes}).encode()
        path.write_bytes(raw)
        monkeypatch.setattr(sheets, "KREA_DYNAMIC_TEMPLATE_SHA256", hashlib.sha256(raw).hexdigest())
    monkeypatch.setattr(comfy_swap, "ensure_backend", lambda **kwargs: pytest.fail("must not start"))
    monkeypatch.setattr(comfy_swap.httpx, "Client", lambda **kwargs: pytest.fail("must not upload"))
    with pytest.raises(error):
        adapter().generate(request(workflow=PROFILES[1], prompt=MANUAL))


@pytest.mark.parametrize("workflow", PROFILES)
@pytest.mark.parametrize("field,value", [
    ("model_key", "flux-klein-4b"), ("images", []), ("images", [Image.new("RGB", (8, 8))] * 2),
    ("mask", Image.new("L", (8, 8))), ("count", 0), ("count", 2),
    ("size_multiplier", 0), ("size_multiplier", 2), ("swap_kind", "Head"),
    ("steps", 3), ("steps", 41), ("steps", 10.0), ("steps", True),
    ("guidance", 0), ("guidance", -1), ("guidance", float("nan")), ("guidance", float("inf")),
    ("seed", -1), ("seed", 2**64), ("seed", True), ("seed", 1.5),
    ("krea_first_lora_weight", 0), ("krea_first_lora_weight", 2.1),
    ("krea_first_lora_weight", float("nan")), ("krea_first_lora_weight", float("inf")),
])
def test_invalid_requests_fail_before_start_upload_and_graph_mutation(assets, monkeypatch, workflow, field, value):
    req = request(workflow=workflow, **{field: value})
    monkeypatch.setattr(comfy_swap, "ensure_backend", lambda **kwargs: pytest.fail("must not start"))
    monkeypatch.setattr(comfy_swap.httpx, "Client", lambda **kwargs: pytest.fail("must not upload"))
    with pytest.raises(ValueError):
        adapter().generate(req)
    graph = sheets.krea_character_sheet_template(workflow)
    before = deepcopy(graph)
    with pytest.raises(ValueError):
        sheets.configure_krea_character_sheet_graph(graph, req, ("source.png",))
    assert graph == before


@pytest.mark.parametrize("workflow", PROFILES)
@pytest.mark.parametrize("case", ["other-profile", "identity", "remix", "more-real", "mandatory", "six",
                                  "duplicate", "foreign", "name", "suffix", "missing", "nan", "low", "high"])
def test_invalid_optional_stacks(assets, monkeypatch, workflow, case):
    loras = optional_loras([0.7])
    lora = loras[0]
    error = ValueError
    forbidden = {"other-profile": sheets.KREA_CHARACTER_SHEET_PROFILES[PROFILES[workflow == PROFILES[0]]],
                 "identity": sheets.KREA_EDIT_FILE, "remix": sheets.KREA_REMIX_LORA_FILE,
                 "more-real": "Krea2-MoreReal.safetensors", "mandatory": KREA_FIRST_LORA_FILE}
    if case in forbidden:
        path = lora.path.with_name(forbidden[case])
        path.touch()
        loras = [LoraSpec(path.name, path, 1, case)]
    elif case == "six":
        loras = optional_loras([1] * 6)
    elif case == "duplicate":
        loras *= 2
    elif case == "missing":
        lora.path.unlink()
        error = FileNotFoundError
    elif case in {"foreign", "name", "suffix"}:
        path = lora.path
        if case == "foreign":
            path = settings.lora_dir / "flux" / path.name
            path.parent.mkdir()
        elif case == "suffix":
            path = path.with_suffix(".pt")
        path.touch()
        loras = [LoraSpec("wrong.safetensors" if case == "name" else path.name, path, 1, case)]
    else:
        loras = [LoraSpec(lora.name, lora.path, {"nan": float("nan"), "low": -2.1, "high": 2.1}[case], case)]
    monkeypatch.setattr(comfy_swap, "ensure_backend", lambda **kwargs: pytest.fail("must not start"))
    with pytest.raises(error):
        adapter().generate(request(workflow=workflow, loras=loras))


@pytest.mark.parametrize("names", [(), ("",), ("  ",), (None,), ("a", "b")])
def test_invalid_upload_names_and_changed_templates(assets, names):
    graph = sheets.krea_character_sheet_template()
    before = deepcopy(graph)
    with pytest.raises(ValueError, match="uploaded image name"):
        sheets.configure_krea_character_sheet_graph(graph, request(), names)
    assert graph == before
    graph["82"]["inputs"]["width"] = 1024
    with pytest.raises(ValueError, match="fresh, unmodified"):
        sheets.configure_krea_character_sheet_graph(graph, request(), ("source.png",))


@pytest.mark.parametrize("workflow", PROFILES)
@pytest.mark.parametrize("index", range(5))
def test_missing_required_files_before_start_even_with_autostart_disabled(assets, monkeypatch, workflow, index):
    paths = assets[0] + [assets[1][PROFILES.index(workflow)]]
    paths[index].unlink()
    assert sheets.missing_krea_character_sheet_assets(settings.comfy_dir, workflow) == [str(paths[index])]
    monkeypatch.setattr(comfy_swap, "ensure_backend", lambda **kwargs: pytest.fail("must not start"))
    monkeypatch.setattr(comfy_swap.httpx, "Client", lambda **kwargs: pytest.fail("must not upload"))
    with pytest.raises(FileNotFoundError):
        adapter().generate(request(workflow=workflow))
    with pytest.raises(FileNotFoundError):
        comfy_backend.ensure_backend(enabled=False, workflow=workflow)


@pytest.mark.parametrize("failure", ["Krea2EditModelPatch", "Krea2EditGroundedEncode", "TextGenerate",
                                     "sampler", "scheduler", "grounding", "unqualified", "dynamic-branch"])
def test_missing_nodes_and_incompatible_schema_before_upload(assets, monkeypatch, failure):
    req = request(workflow=PROFILES[1])
    info = registered(configured(req))
    if failure in info:
        info.pop(failure)
    elif failure in {"sampler", "scheduler"}:
        field = "sampler_name" if failure == "sampler" else "scheduler"
        info["KSampler"]["input"]["required"][field] = (["unsupported"],)
    elif failure == "grounding":
        info["Krea2EditGroundedEncode"]["input"]["required"].pop("grounding_px")
    elif failure == "unqualified":
        info["LoraLoaderModelOnly"]["input"]["required"]["lora_name"] = ([KREA_FIRST_LORA_FILE],)
    else:
        info["TextGenerate"]["input"]["required"]["sampling_mode"][1]["options"][0]["inputs"] = {
            "required": {"seed": ("INT",)},
        }
    calls = []

    def handler(req):
        calls.append(req.url.path)
        assert req.method == "GET"
        return httpx.Response(200, json=info if req.url.path == "/object_info" else {})

    install_http(monkeypatch, handler)
    with pytest.raises(RuntimeError):
        adapter().generate(req)
    assert calls == ["/system_stats", "/object_info"]


@pytest.mark.parametrize("workflow,prompt", [(PROFILES[0], "custom coat"), (PROFILES[1], ""), (PROFILES[1], MANUAL)])
@pytest.mark.parametrize("output_count", [0, 1, 2])
def test_actual_http_payload_and_only_save29_output(assets, monkeypatch, workflow, prompt, output_count):
    req = request(workflow=workflow, prompt=prompt, loras=optional_loras([-0.4, 1.2]))
    expected = configured(req)
    info = registered(expected)
    comfy_swap.preflight_krea_remix_graph(expected, info, label=workflow)
    calls = []
    image = BytesIO()
    Image.new("RGB", (1536, 1024), "red").save(image, format="PNG")

    def handler(http_request):
        path = http_request.url.path
        calls.append(path)
        if path == "/system_stats":
            return httpx.Response(200, json={})
        if path == "/object_info":
            return httpx.Response(200, json=info)
        if path == "/upload/image":
            start = http_request.content.index(b"\x89PNG")
            with Image.open(BytesIO(http_request.content[start:])) as uploaded:
                assert uploaded.size == (512, 1024)
                assert uploaded.getpixel((0, 0)) == (0, 0, 255)
            return httpx.Response(200, json={"name": "uploaded.png"})
        if path == "/prompt":
            payload = json.loads(http_request.content)["prompt"]
            expected["72"]["inputs"]["image"] = "uploaded.png"
            prefix = payload["29"]["inputs"]["filename_prefix"]
            assert prefix.startswith("photo_edit_" + workflow.replace("-", "_") + "_")
            expected["29"]["inputs"]["filename_prefix"] = prefix
            assert payload == expected
            return httpx.Response(200, json={"prompt_id": "job"})
        if path == "/history/job":
            return httpx.Response(200, json={"job": {"outputs": {
                "29": {"images": [{"filename": "sheet.png", "subfolder": "sheets", "type": "output"}] * output_count},
                "72": {"images": [{"filename": "source.png"}]},
                "54": {"images": [{"filename": "preview.png", "type": "temp"}]},
            }}})
        assert path == "/view"
        assert dict(http_request.url.params) == {"filename": "sheet.png", "subfolder": "sheets", "type": "output"}
        return httpx.Response(200, content=image.getvalue())

    starts = install_http(monkeypatch, handler)
    if output_count == 1:
        result = adapter().generate(req)
        assert len(result) == 1 and result[0].size == (1536, 1024)
        assert result[0].getpixel((0, 0)) == (255, 0, 0)
    else:
        with pytest.raises(RuntimeError, match="SaveImage 29 must return exactly one"):
            adapter().generate(req)
        assert "/view" not in calls
    assert starts == [{"workflow": workflow}]
    assert calls[:4] == ["/system_stats", "/object_info", "/upload/image", "/prompt"]


@pytest.mark.parametrize("workflow,prompt", [(PROFILES[0], "coat"), (PROFILES[1], ""), (PROFILES[1], MANUAL)])
def test_engine_seed_before_validation_no_clamp_source_retained_actual_metadata(assets, monkeypatch, workflow, prompt):
    req = request(workflow=workflow, prompt=prompt, seed=-1, compose=True)
    source = req.images[0]
    original = source.tobytes()
    monkeypatch.setattr(settings, "max_output_side", 1024)
    monkeypatch.setattr(settings, "max_output_pixels", 1024**2)
    monkeypatch.setattr(engine.secrets, "randbelow", lambda upper: 123456)
    monkeypatch.setattr(engine, "diffusion_output_size", lambda *args, **kwargs: pytest.fail("must not clamp"))
    monkeypatch.setattr(engine.model_manager, "get", lambda *args: pytest.fail("must not load Diffusers"))
    validation = engine.validate_krea_character_sheet_request
    validated = []

    def validate(actual):
        assert actual.seed == 123456
        validated.append(True)
        validation(actual)

    monkeypatch.setattr(engine, "validate_krea_character_sheet_request", validate)
    unloaded = []
    monkeypatch.setattr(engine.model_manager, "unload", lambda: unloaded.append(True))
    generated = Image.new("RGB", (1536, 1024), "red")

    def generate(self, actual, progress=None):
        assert actual.images[0].size == (512, 1024)
        assert (actual.width, actual.height) == (1536, 1024) and not actual.compose
        assert actual.images[0].getpixel((0, 0)) == (0, 0, 255)
        assert configured(actual)["53"]["inputs"]["seed"] == 123456
        return [generated]

    monkeypatch.setattr(comfy_swap.ComfyKreaSwapAdapter, "generate", generate)
    result = engine.generate(req, "Off", 0)
    assert validated == unloaded == [True]
    assert result.images == [generated] and result.seed == 123456
    assert source.size == (768, 1536) and source.tobytes() == original
    metadata = json.loads((result.saved_paths[0].parent / "metadata.json").read_text())
    assert metadata["native_diffusion_size"] == [1536, 1024]
    assert metadata["actual_output_sizes"] == [[1536, 1024]]
    assert metadata["uploaded_source_size"] == [512, 1024]
    assert metadata["source_aspect_preserved"] and metadata["source_max_side"] == 1024
    assert metadata["grounding_px"] == 0 and metadata["fit_mode"] == "fit"
    assert metadata["character_sheet_revision"] == "3dc4295163dacc924d213168d67bf16850fd954f"
    assert metadata["sheet_adapter"] == sheets.KREA_CHARACTER_SHEET_PROFILES[workflow]
    assert metadata["sheet_adapter_weight"] == 1 and metadata["mandatory_first_lora_weight"] == 0.65
    assert metadata["sampler"] == ("lcm" if workflow == PROFILES[1] else "euler")
    assert metadata["scheduler"] == "simple" and not metadata["negative_conditioning_active"]
    assert metadata["prompt_stored"] is False
    if workflow == PROFILES[1]:
        assert metadata["experimental"]
        assert metadata["caption_template_sha256"] == engine.KREA_DYNAMIC_TEMPLATE_SHA256
        assert metadata["prompt_policy"] == ("manual-structured" if prompt else "auto-publisher-vlm-greedy")
        assert metadata["vlm_sampling_mode"] == (None if prompt else "off")
        assert metadata["vlm_max_length"] == (None if prompt else 2048)
    else:
        assert metadata["prompt_policy"] == "fixed-trigger-plus-customization"
        assert not metadata["experimental"]
    assert any("1536×1024" in note for note in result.notes)


@pytest.mark.parametrize("changes,restoration", [({"images": []}, "Off"), ({"count": 2}, "Off"),
                                                ({"steps": 3}, "Off"), ({"guidance": 0}, "Off"),
                                                ({"size_multiplier": 2}, "Off"), ({}, "GFPGAN")])
def test_engine_rejects_before_truncation_or_loading(assets, monkeypatch, changes, restoration):
    monkeypatch.setattr(engine.model_manager, "unload", lambda: pytest.fail("must not unload"))
    with pytest.raises(ValueError):
        engine.generate(request(**changes), restoration, 0)


@pytest.mark.parametrize("outputs", [[], [Image.new("RGB", (1024, 1024))],
                                     [Image.new("RGB", (1536, 1024))] * 2])
def test_engine_rejects_wrong_native_size_or_count_without_resizing(assets, monkeypatch, outputs):
    monkeypatch.setattr(engine.model_manager, "unload", lambda: None)
    monkeypatch.setattr(comfy_swap.ComfyKreaSwapAdapter, "generate", lambda *args, **kwargs: outputs)
    with pytest.raises(RuntimeError, match="exactly one|1536x1024"):
        engine.generate(request(), "Off", 0)


@pytest.mark.parametrize("operation,workflow", list(zip(OPERATIONS, PROFILES, strict=True)))
def test_ui_forwarding_defaults_per_mode_and_fixed_preview(assets, monkeypatch, operation, workflow):
    selected = optional_loras([-0.5, 0.2, 0.3, 0.4, 0.5])
    selections, calls = [], []
    monkeypatch.setattr(ui, "selected_loras", lambda model, names, weights:
                        selections.append((model, names, weights)) or selected)
    monkeypatch.setattr(ui, "generate", lambda req, restoration, weight, progress:
                        calls.append((req, restoration, weight)) or SimpleNamespace(images=["sheet"]))
    monkeypatch.setattr(ui, "metadata_text", lambda result: "metadata")
    source = Image.new("RGB", (400, 600))
    result = ui._run_krea_edit(
        operation, source, Image.new("RGB", (8, 8)), "  caption  ", "negative",
        3, 10, 1, -1, "GFPGAN", 0.8, 0.7,
        "one", "two", "three", "four", "five", -0.5, 0.2, 0.3, 0.4, 0.5,
        progress=lambda *args, **kwargs: None,
    )
    req, restoration, weight = calls[0]
    assert (req.workflow, req.model_key, req.prompt) == (workflow, "krea-2-turbo", "caption")
    assert req.images == [source] and req.images[0] is source
    assert req.loras is selected and req.negative_prompt == "negative"
    assert selections == [("krea-2-turbo", ["one", "two", "three", "four", "five"], [-0.5, 0.2, 0.3, 0.4, 0.5])]
    assert (req.width, req.height, req.steps, req.guidance, req.seed) == (1536, 1024, 10, 1, -1)
    assert req.count == req.size_multiplier == req.strength == 1
    assert req.mask is req.swap_kind is None and not req.preserve_identity
    assert (restoration, weight, req.krea_first_lora_weight) == ("Off", 0, 0.7)
    assert result[:2] == (["sheet"], "metadata")
    updates = ui._krea_operation_for_mode(ui.KREA_EDIT_MODE, operation)
    assert updates[0]["visible"] is False
    assert [update["value"] for update in updates[1:]] == [10, 1, 0]
    for mode in (ui.EDIT_MODE, ui.COMBINE_MODE, ui.TEXT_MODE, ui.SWAP_MODE):
        assert all("value" not in update and "visible" not in update
                   for update in ui._krea_operation_for_mode(mode, operation))
    preview = ui._size_preview_for_mode(
        ui.KREA_EDIT_MODE, 3, "2K", "1:1", None, source, None, None, None, None,
        krea_operation=operation,
    )
    assert "1536×1024" in preview and "multiplier does not apply" in preview
    with pytest.raises(ValueError, match="Upload Picture 1"):
        ui._run_krea_edit(operation, None, None, "", "", 1, 10, 1, 9, "Off", 0)


def test_ui_registers_both_operations_and_change_preview_events():
    app = ui.build_app()
    component = next(c for c in app.config["components"]
                     if c["props"].get("label") == "Krea 2 operation")
    choices = [choice[0] if isinstance(choice, (tuple, list)) else choice
               for choice in component["props"]["choices"]]
    assert all(operation in choices for operation in OPERATIONS)
    events = [event for event in app.config["dependencies"]
              if (component["id"], "change") in event["targets"]]
    functions = [app.fns[event["id"]].fn for event in events]
    assert ui._krea_operation_for_mode in functions
    assert ui._size_preview_for_mode in functions


def test_fractional_cfg_metadata_reports_negative_conditioning(assets, monkeypatch):
    monkeypatch.setattr(engine.model_manager, "unload", lambda: None)
    monkeypatch.setattr(comfy_swap.ComfyKreaSwapAdapter, "generate",
                        lambda *args, **kwargs: [Image.new("RGB", (1536, 1024))])
    result = engine.generate(request(guidance=0.5), "Off", 0)
    metadata = json.loads((result.saved_paths[0].parent / "metadata.json").read_text())
    assert metadata["negative_conditioning_active"] is True


def test_ui_zero_weight_skips_optional_slot_and_retains_required_sheet(assets, monkeypatch):
    calls = []
    monkeypatch.setattr(ui, "generate", lambda req, restoration, weight, progress:
                        calls.append(req) or SimpleNamespace(images=[]))
    monkeypatch.setattr(ui, "metadata_text", lambda result: "metadata")
    name = sheets.KREA_CHARACTER_SHEET_PROFILES[PROFILES[0]]
    ui._run_krea_edit(
        OPERATIONS[0], Image.new("RGB", (8, 8)), None, "", "", 1, 10, 1, 42, "Off", 0,
        1, name, None, None, None, None, 0, 0, 0, 0, 0,
        progress=lambda *args, **kwargs: None,
    )
    assert configured(calls[0])["71"]["inputs"]["strength_model"] == 1