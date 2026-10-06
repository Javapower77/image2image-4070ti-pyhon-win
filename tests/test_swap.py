from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from photo_edit_studio.comfy_assets import KREA_FIRST_LORA_FILE
from photo_edit_studio.comfy_workflows import krea_reference_template, krea_text_template
from photo_edit_studio.models.comfy_swap import (
    _api_graph,
    _local_comfy_url,
    apply_krea_loras,
    apply_mandatory_krea_lora,
    configure_krea_graph,
    configure_krea_reference_graph,
    configure_krea_text_graph,
)
from photo_edit_studio.swap import SWAP_PROFILES, swap_lora, swap_profile
from photo_edit_studio.types import GenerationRequest, LoraSpec


def test_swap_profiles_target_exact_model_families() -> None:
    assert set(SWAP_PROFILES) == {
        ("qwen-2511", "Head"), ("flux-klein-4b", "Head"),
        ("krea-2-turbo", "Head"), ("krea-2-turbo", "Body"),
        ("qwen-2.1-turbo", "Head"), ("qwen-2.1-turbo", "Body"),
        ("qwen-2.1-turbo-r128", "Head"), ("qwen-2.1-turbo-r128", "Body"),
    }
    with pytest.raises(ValueError, match="does not support"):
        swap_profile("qwen-2511-aio", "Head")


def test_swap_missing_required_lora_fails_before_inference(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from photo_edit_studio.config import settings

    monkeypatch.setattr(settings, "lora_dir", tmp_path)
    with pytest.raises(FileNotFoundError, match="--bfs-swap"):
        swap_lora("flux-klein-4b", "Head")


def _request(kind: str) -> GenerationRequest:
    return GenerationRequest(
        model_key="krea-2-turbo", prompt="additional details", negative_prompt="",
        images=[Image.new("RGB", (64, 64)), Image.new("RGB", (64, 64))],
        mask=None, width=1024, height=1024, steps=10, guidance=1.0,
        true_cfg=0.0, strength=0.8, seed=55, workflow="swap", swap_kind=kind,
    )


@pytest.mark.parametrize("kind,body_id,reference_id,lora_id", [
    ("Head", "72", "90", "111"), ("Body", "72", "139", "127"),
])
def test_krea_graph_injects_matching_assets(
    kind: str, body_id: str, reference_id: str, lora_id: str
) -> None:
    profile = swap_profile("krea-2-turbo", kind)
    request = _request(kind)
    request.loras = [LoraSpec(profile.filename, Path(profile.filename), 0.8, "bfs_swap")]
    graph = {
        body_id: {"class_type": "LoadImage", "inputs": {"image": "old"}},
        reference_id: {"class_type": "LoadImage", "inputs": {"image": "old"}},
        "55": {"class_type": "UNETLoader", "inputs": {}},
        lora_id: {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["55", 0]}},
        "79": {"class_type": "Krea2EditModelPatch", "inputs": {"model": [lora_id, 0]}},
        "119": {"class_type": "Krea2EditGroundedEncode", "inputs": {}},
        "53": {"class_type": "KSampler", "inputs": {}},
        "29": {"class_type": "SaveImage", "inputs": {}},
    }
    if kind == "Body":
        graph["145"] = {"class_type": "PrimitiveNode", "inputs": {}}
    updated = configure_krea_graph(graph, request, ("body.png", "face.png"))
    assert updated[body_id]["inputs"]["image"] == "body.png"
    assert updated[reference_id]["inputs"]["image"] == "face.png"
    assert updated[lora_id]["inputs"]["lora_name"] == profile.filename
    assert updated[lora_id]["inputs"]["strength_model"] == 0.8
    assert updated["53"]["inputs"]["seed"] == 55
    assert updated["119"]["inputs"]["prompt"].startswith(profile.trigger)


def test_comfy_ui_format_is_rejected() -> None:
    with pytest.raises(ValueError, match="API-format"):
        _api_graph({"nodes": []})


def test_krea_swap_retains_both_images_in_engine(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from photo_edit_studio import engine
    from photo_edit_studio.config import settings

    request = _request("Head")
    request.loras = [LoraSpec("bfs.safetensors", tmp_path / "bfs.safetensors", 1.0, "bfs_swap")]
    seen: list[int] = []

    def fake_generate(self: object, request: GenerationRequest, progress: object = None) -> list[Image.Image]:
        seen.append(len(request.images))
        return [Image.new("RGB", (64, 64))]

    monkeypatch.setattr("photo_edit_studio.models.comfy_swap.ComfyKreaSwapAdapter.generate", fake_generate)
    monkeypatch.setattr(settings, "output_dir", tmp_path)
    engine.generate(request, "Off", 0.35)
    assert seen == [2]


def _reference_graph() -> dict[str, dict]:
    return {
    "55": {"class_type": "UNETLoader", "inputs": {}},
        "72": {"class_type": "LoadImage", "inputs": {"image": "old.png"}},
        "90": {"class_type": "LoadImage", "inputs": {"image": "old-ref.png"}},
        "92": {"class_type": "VAEEncode", "inputs": {"pixels": ["90", 0]}},
        "79": {"class_type": "Krea2EditModelPatch", "inputs": {
            "model": ["71", 0], "source_latent_b": ["92", 0], "source_image_b": ["90", 0],
        }},
        "84": {"class_type": "Krea2EditGroundedEncode", "inputs": {"image_b": ["90", 0]}},
        "85": {"class_type": "Krea2EditGroundedEncode", "inputs": {"image_b": ["90", 0]}},
        "53": {"class_type": "KSampler", "inputs": {}},
        "82": {"class_type": "EmptySD3LatentImage", "inputs": {}},
        "29": {"class_type": "SaveImage", "inputs": {}},
        "71": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["55", 0]}},
    }


def test_krea_reference_edit_uses_both_references() -> None:
    request = _request("Head")
    request.workflow = "krea-reference"
    updated = configure_krea_reference_graph(_reference_graph(), request, ("scene.png", "person.png"))
    assert updated["72"]["inputs"]["image"] == "scene.png"
    assert updated["90"]["inputs"]["image"] == "person.png"
    assert updated["84"]["inputs"]["prompt"] == request.prompt
    assert updated["53"]["inputs"]["steps"] == request.steps
    assert updated["82"]["inputs"]["width"] == request.width


def test_packaged_reference_graph_is_api_ready() -> None:
    request = _request("Head")
    request.workflow = "krea-reference"
    graph = _api_graph(krea_reference_template())
    result = configure_krea_reference_graph(graph, request, ("target.png", "reference.png"))
    assert result["79"]["inputs"]["source_latent_b"] == ["92", 0]
    assert result["29"]["inputs"]["images"] == ["54", 0]
    assert result["82"]["inputs"]["height"] == request.height


def test_reference_graph_uses_selected_loras_in_order(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from photo_edit_studio.config import settings

    monkeypatch.setattr(settings, "lora_dir", tmp_path)
    family = tmp_path / "krea2"
    family.mkdir()
    first = family / "style.safetensors"
    second = family / "character.safetensors"
    first.touch()
    second.touch()
    request = _request("Head")
    request.workflow = "krea-reference"
    request.loras = [
        LoraSpec(first.name, first, 0.7, "style_1"),
        LoraSpec(second.name, second, 1.25, "character_2"),
    ]
    graph = configure_krea_reference_graph(krea_reference_template(), request, ("a.png", "b.png"))
    assert graph["71"]["inputs"]["lora_name"] == settings.comfy_reference_lora
    assert graph["93"]["inputs"] == {
        "model": ["71", 0], "lora_name": "style.safetensors", "strength_model": 0.7,
    }
    assert graph["94"]["inputs"] == {
        "model": ["93", 0], "lora_name": "character.safetensors", "strength_model": 1.25,
    }
    assert graph["79"]["inputs"]["model"] == ["94", 0]


def test_optional_swap_lora_uses_bfs_as_anchor(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from photo_edit_studio.config import settings

    monkeypatch.setattr(settings, "lora_dir", tmp_path)
    family = tmp_path / "krea2"
    family.mkdir()
    extra = family / "style.safetensors"
    extra.touch()
    request = _request("Head")
    profile = swap_profile("krea-2-turbo", "Head")
    request.loras = [
        LoraSpec(profile.filename, family / profile.filename, 0.9, "bfs_swap"),
        LoraSpec(extra.name, extra, 0.4, "style_1"),
    ]
    graph = configure_krea_graph(krea_reference_template(), request, ("body.png", "face.png"))
    assert graph["71"]["inputs"]["lora_name"] == profile.filename
    assert graph["93"]["inputs"]["model"] == ["71", 0]
    assert graph["93"]["inputs"]["strength_model"] == 0.4
    assert graph["79"]["inputs"]["model"] == ["93", 0]


def test_swap_bfs_weight_is_not_overridden_by_optional_slot(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from photo_edit_studio.config import settings

    monkeypatch.setattr(settings, "lora_dir", tmp_path)
    family = tmp_path / "krea2"
    family.mkdir()
    profile = swap_profile("krea-2-turbo", "Head")
    base = family / profile.filename
    base.touch()
    request = _request("Head")
    request.loras = [
        LoraSpec(base.name, base, 0.9, "bfs_swap"),
        LoraSpec(base.name, base, 0.2, "slot_1"),
    ]
    graph = configure_krea_graph(krea_reference_template(), request, ("body.png", "face.png"))
    assert graph["71"]["inputs"]["strength_model"] == 0.9
    assert graph["79"]["inputs"]["model"] == ["71", 0]


def test_optional_krea_lora_adjusts_required_base_weight(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from photo_edit_studio.config import settings

    monkeypatch.setattr(settings, "lora_dir", tmp_path)
    family = tmp_path / "krea2"
    family.mkdir()
    base = family / settings.comfy_reference_lora
    base.touch()
    extra = family / "style.safetensors"
    extra.touch()
    graph = configure_krea_reference_graph(
        krea_reference_template(),
        GenerationRequest(
            model_key="krea-2-turbo", prompt="edit", negative_prompt="", images=[],
            mask=None, width=1024, height=1024, steps=10, guidance=1.0,
            true_cfg=0, strength=1, seed=0,
            loras=[LoraSpec(base.name, base, 0.6, "base"), LoraSpec(extra.name, extra, 0.8, "style")],
            workflow="krea-reference",
        ),
        ("source.png",),
    )
    assert graph["71"]["inputs"]["strength_model"] == 0.6
    added_id = graph["79"]["inputs"]["model"][0]
    assert graph[added_id]["inputs"] == {
        "model": ["71", 0], "lora_name": "style.safetensors", "strength_model": 0.8,
    }
    assert added_id != "71"
    assert len([node for node in graph.values() if node["class_type"] == "LoraLoaderModelOnly"]) == 3


def test_optional_krea_lora_rejects_duplicates_and_other_families(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from photo_edit_studio.config import settings

    monkeypatch.setattr(settings, "lora_dir", tmp_path)
    family = tmp_path / "krea2"
    family.mkdir()
    base = family / settings.comfy_reference_lora
    base.touch()
    with pytest.raises(ValueError, match="more than once"):
        apply_krea_loras(krea_reference_template(), "71", [
            LoraSpec(base.name, base, 1.0, "base"), LoraSpec(base.name, base, 0.5, "duplicate")
        ])
    foreign = tmp_path / "flux" / "other.safetensors"
    foreign.parent.mkdir()
    foreign.touch()
    with pytest.raises(ValueError, match="Krea2"):
        apply_krea_loras(krea_reference_template(), "71", [
            LoraSpec(foreign.name, foreign, 1.0, "foreign")
        ])


def test_krea_reference_ui_passes_lora_selection_to_engine(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from types import SimpleNamespace

    from photo_edit_studio import ui
    from photo_edit_studio.config import settings

    monkeypatch.setattr(settings, "lora_dir", tmp_path)
    library = tmp_path / "krea2"
    library.mkdir()
    adapter = library / "style.safetensors"
    adapter.touch()
    captured: list[GenerationRequest] = []

    def fake_generate(request: GenerationRequest, *_args: object, **_kwargs: object) -> object:
        captured.append(request)
        return SimpleNamespace(images=[Image.new("RGB", (64, 64))])

    monkeypatch.setattr(ui, "generate", fake_generate)
    monkeypatch.setattr(ui, "metadata_text", lambda _result: "{}")
    ui._run_krea_reference(
        Image.new("RGB", (64, 64)), None, "Edit scene", "", 1, 10, 1.0, -1,
        "Off", 0.35, 1.0,
        "style.safetensors", *(["(none)"] * 4), 0.65, *([1.0] * 4),
        progress=lambda *_args, **_kwargs: None,
    )
    assert [(lora.name, lora.weight) for lora in captured[0].loras] == [
        ("style.safetensors", 0.65)
    ]


def test_krea_reference_edit_rejects_unwired_second_image() -> None:
    graph = _reference_graph()
    graph["84"]["inputs"].pop("image_b")
    with pytest.raises(TypeError, match="disconnected"):
        configure_krea_reference_graph(graph, _request("Head"), ("scene.png", "person.png"))


def test_krea_reference_edit_accepts_only_source_image() -> None:
    graph = configure_krea_reference_graph(_reference_graph(), _request("Head"), ("scene.png",))
    assert "90" not in graph and "92" not in graph
    assert "image_b" not in graph["84"]["inputs"]


def test_krea_reference_bridge_requires_loopback() -> None:
    assert _local_comfy_url("http://127.0.0.1:8188/") == "http://127.0.0.1:8188"
    with pytest.raises(ValueError, match="loopback"):
        _local_comfy_url("https://public-comfy.example.com")


def test_krea_reference_engine_uses_comfy_backend(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from photo_edit_studio import engine
    from photo_edit_studio.config import settings

    request = _request("Head")
    request.workflow = "krea-reference"
    request.loras = []
    seen: list[int] = []

    def fake_generate(self: object, request: GenerationRequest, progress: object = None) -> list[Image.Image]:
        seen.append(len(request.images))
        return [Image.new("RGB", (64, 64))]

    monkeypatch.setattr("photo_edit_studio.models.comfy_swap.ComfyKreaSwapAdapter.generate", fake_generate)
    monkeypatch.setattr(settings, "output_dir", tmp_path)
    engine.generate(request, "Off", 0.35)
    assert seen == [2]


@pytest.mark.parametrize("kind", ["Head", "Body"])
def test_builtin_krea_swap_graph_reuses_krea_edit_template(kind: str) -> None:
    request = _request(kind)
    profile = swap_profile("krea-2-turbo", kind)
    request.loras = [LoraSpec(profile.filename, Path(profile.filename), 0.8, "bfs_swap")]
    graph = configure_krea_graph(krea_reference_template(), request, ("scene.png", "person.png"))
    assert graph["71"]["inputs"]["lora_name"] == profile.filename
    assert graph["84"]["inputs"]["prompt"].startswith(profile.trigger)
    assert graph["90"]["inputs"]["image"] == "person.png"


def _upstream_lora_chain(graph: dict, consumer_id: str) -> list[tuple[str, float]]:
    names = []
    node_id = graph[consumer_id]["inputs"]["model"][0]
    while graph[node_id]["class_type"] == "LoraLoaderModelOnly":
        inputs = graph[node_id]["inputs"]
        names.append((inputs["lora_name"], inputs["strength_model"]))
        node_id = inputs["model"][0]
    assert graph[node_id]["class_type"] == "UNETLoader"
    return list(reversed(names))


def test_mandatory_adapter_is_first_for_krea_text(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from photo_edit_studio.config import settings

    monkeypatch.setattr(settings, "comfy_dir", tmp_path)
    folder = tmp_path / "models" / "loras"
    folder.mkdir(parents=True)
    (folder / KREA_FIRST_LORA_FILE).touch()
    request = _request("Head")
    request.workflow = "krea-text"
    request.images = []
    request.width, request.height = 2048, 1152
    request.krea_first_lora_weight = 0.45
    monkeypatch.setattr(settings, "lora_dir", tmp_path / "library")
    extra_dir = settings.lora_dir / "krea2"
    extra_dir.mkdir(parents=True)
    extra = extra_dir / "style.safetensors"
    extra.touch()
    request.loras = [LoraSpec(extra.name, extra, 0.7, "style")]
    graph = configure_krea_text_graph(krea_text_template(), request)
    assert _upstream_lora_chain(graph, "53") == [
        (KREA_FIRST_LORA_FILE, 0.45), (extra.name, 0.7),
    ]
    assert graph["84"]["inputs"]["text"] == request.prompt
    assert (graph["82"]["inputs"]["width"], graph["82"]["inputs"]["height"]) == (
        2048, 1152
    )


@pytest.mark.parametrize("kind", ["Head", "Body"])
def test_mandatory_adapter_precedes_bfs_and_optional_for_krea_swap(
    kind: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from photo_edit_studio.config import settings

    monkeypatch.setattr(settings, "comfy_dir", tmp_path)
    monkeypatch.setattr(settings, "lora_dir", tmp_path / "library")
    first_dir = tmp_path / "models" / "loras"
    first_dir.mkdir(parents=True)
    (first_dir / KREA_FIRST_LORA_FILE).touch()
    extras_dir = settings.lora_dir / "krea2"
    extras_dir.mkdir(parents=True)
    extra = extras_dir / "style.safetensors"
    extra.touch()
    request = _request(kind)
    request.krea_first_lora_weight = 1.3
    profile = swap_profile("krea-2-turbo", kind)
    request.loras = [
        LoraSpec(profile.filename, extras_dir / profile.filename, 0.8, "bfs"),
        LoraSpec(extra.name, extra, 0.35, "style"),
    ]
    graph = configure_krea_graph(krea_reference_template(), request, ("body.png", "face.png"))
    assert _upstream_lora_chain(graph, "79") == [
        (KREA_FIRST_LORA_FILE, 1.3), (profile.filename, 0.8), (extra.name, 0.35),
    ]


def test_mandatory_adapter_precedes_identity_lora_in_krea_reference(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from photo_edit_studio.config import settings

    monkeypatch.setattr(settings, "comfy_dir", tmp_path)
    folder = tmp_path / "models" / "loras"
    folder.mkdir(parents=True)
    (folder / KREA_FIRST_LORA_FILE).touch()
    request = _request("Head")
    request.workflow = "krea-reference"
    request.krea_first_lora_weight = 0.75
    graph = configure_krea_reference_graph(krea_reference_template(), request, ("a.png", "b.png"))
    assert _upstream_lora_chain(graph, "79") == [
        (KREA_FIRST_LORA_FILE, 0.75), (settings.comfy_reference_lora, 1.0),
    ]


def test_mandatory_adapter_missing_or_disabled_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from photo_edit_studio.config import settings

    monkeypatch.setattr(settings, "comfy_dir", tmp_path)
    request = _request("Head")
    with pytest.raises(FileNotFoundError, match=KREA_FIRST_LORA_FILE):
        apply_mandatory_krea_lora(krea_text_template(), request)
    folder = tmp_path / "models" / "loras"
    folder.mkdir(parents=True)
    (folder / KREA_FIRST_LORA_FILE).touch()
    request.krea_first_lora_weight = 0
    with pytest.raises(ValueError, match="greater than 0"):
        apply_mandatory_krea_lora(krea_text_template(), request)