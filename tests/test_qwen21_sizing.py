from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from photo_edit_studio import engine
from photo_edit_studio.comfy_workflows import qwen21_turbo_template
from photo_edit_studio.config import Settings, settings
from photo_edit_studio.dlss import add_dlss_nodes
from photo_edit_studio.image_utils import diffusion_output_size
from photo_edit_studio.models.comfy_swap import configure_qwen21_graph
from photo_edit_studio.swap import QWEN21_BFS_HEAD_FILE
from photo_edit_studio.types import GenerationRequest, LoraSpec
from photo_edit_studio.ui import COMBINE_MODE, EDIT_MODE, SWAP_MODE, TEXT_MODE, _size_preview_for_mode

QWEN_KEYS = ("qwen-2.1-turbo", "qwen-2.1-turbo-r128")
SOURCE_SIZE = (3784, 4852)


@pytest.fixture
def limits(monkeypatch: pytest.MonkeyPatch) -> None:
    # Isolate these assertions from the user's environment/.env limits.
    for name, value in {
        "max_output_side": 1024, "max_output_pixels": 1048576,
        "qwen21_x2_max_output_side": 2048, "qwen21_x2_max_output_pixels": 4194304,
        "combine_max_output_side": 2048, "combine_max_output_pixels": 4194304,
    }.items():
        monkeypatch.setattr(settings, name, value)


def preview(req: GenerationRequest) -> str:
    mode = TEXT_MODE if req.workflow == "text" else SWAP_MODE if req.workflow == "swap" else COMBINE_MODE if req.compose else EDIT_MODE
    source = req.images[0] if req.images else None
    # Populate inactive slots too: swap must select its own body/model.
    return _size_preview_for_mode(
        mode, req.size_multiplier, req.output_resolution, req.output_aspect_ratio,
        source, None, source, source, None, None,
        "qwen-2511" if mode == SWAP_MODE else req.model_key, req.model_key,
    )


@pytest.mark.parametrize("key", QWEN_KEYS)
@pytest.mark.parametrize("workflow", ["standard", "swap", "combine", "text"])
@pytest.mark.parametrize("multiplier", [1, 2, 3])
def test_preview_engine_and_latent_agree_before_source_normalization(
    key: str, workflow: str, multiplier: int, limits: None,
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    monkeypatch.setattr(settings, "lora_dir", tmp_path)
    library = tmp_path / "qwen21"
    library.mkdir()
    bfs_path = library / QWEN21_BFS_HEAD_FILE
    bfs_path.write_bytes(b"mock")
    source = Image.new("RGB", SOURCE_SIZE)
    refs = [Image.new("RGB", (768, 512))] if workflow in {"swap", "combine"} else []
    req = GenerationRequest(
        model_key=key, prompt="Change colors", negative_prompt="",
        images=[] if workflow == "text" else [source, *refs], mask=None,
        width=64, height=64, steps=6, guidance=1, true_cfg=1, strength=0.8, seed=17,
        size_multiplier=multiplier, compose=workflow == "combine",
        workflow="standard" if workflow == "combine" else workflow,
        swap_kind="Head" if workflow == "swap" else None,
        output_resolution="2K", output_aspect_ratio="9:16",
        loras=[LoraSpec(QWEN21_BFS_HEAD_FILE, bfs_path, 1, "bfs")] if workflow == "swap" else [],
    )
    expected = (1152, 2048) if workflow in {"combine", "text"} else (1600, 2048) if multiplier == 2 else (768, 1024)
    assert f"{expected[0]} × {expected[1]}" in preview(req)
    if req.images:
        assert req.images[0].size == SOURCE_SIZE
    observed: list[tuple[int, int]] = []

    class Adapter:
        def generate(self, request: GenerationRequest, progress: object = None) -> list[Image.Image]:
            assert (request.width, request.height) == expected
            graph = configure_qwen21_graph(
                qwen21_turbo_template(key), request,
                tuple(f"image{i}.png" for i in range(len(request.images))),
            )
            latent = graph["6"]["inputs"]
            observed.append((latent["width"], latent["height"]))
            assert graph["5"]["inputs"]["resolution"] == 1024
            assert graph["11"]["inputs"]["latent_image"] == ["6", 0]
            if request.images:
                assert max(request.images[0].size) == max(expected)
            assert not any("upscale" in node["class_type"].lower() for node in graph.values())
            return [Image.new("RGB", expected)]

    monkeypatch.setattr(engine.model_manager, "get", lambda _key: Adapter())
    monkeypatch.setattr(engine, "_save", lambda *_args: [])
    result = engine.generate(req, "Off", 0.35)
    assert observed == [expected]
    assert result.images[0].size == expected
    assert source.size == SOURCE_SIZE  # Original upload was not modified.


@pytest.mark.parametrize("key", ["qwen-2511", "flux-klein-4b", "firered-1.1"])
@pytest.mark.parametrize("workflow", ["standard", "swap"])
def test_other_models_keep_1024_source_budget(
    key: str, workflow: str, limits: None, monkeypatch: pytest.MonkeyPatch,
) -> None:
    req = GenerationRequest(
        model_key=key, prompt="colors", negative_prompt="", mask=None,
        images=[Image.new("RGB", SOURCE_SIZE)] + ([Image.new("RGB", (512, 512))] if workflow == "swap" else []),
        width=64, height=64, steps=6, guidance=1, true_cfg=1, strength=0.8, seed=17,
        size_multiplier=2, workflow=workflow,
    )
    assert "768 × 1024" in preview(req)

    class Adapter:
        def generate(self, request: GenerationRequest, progress: object = None) -> list[Image.Image]:
            assert (request.width, request.height) == (768, 1024)
            assert max(request.images[0].size) == 1024
            return [Image.new("RGB", (768, 1024))]

    monkeypatch.setattr(engine.model_manager, "get", lambda _key: Adapter())
    monkeypatch.setattr(engine, "_save", lambda *_args: [])
    engine.generate(req, "Off", 0.35)


def test_qwen_limits_are_configurable_and_do_not_leak(limits: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PHOTO_EDIT_QWEN21_X2_MAX_OUTPUT_SIDE", "1536")
    monkeypatch.setenv("PHOTO_EDIT_QWEN21_X2_MAX_OUTPUT_PIXELS", "2097152")
    configured = Settings(_env_file=None)
    assert configured.qwen21_x2_max_output_side == 1536
    assert configured.qwen21_x2_max_output_pixels == 2097152
    monkeypatch.setattr(settings, "qwen21_x2_max_output_side", 1536)
    assert diffusion_output_size(SOURCE_SIZE, 2, family="qwen21") == (1152, 1536)
    monkeypatch.setattr(settings, "qwen21_x2_max_output_side", 2048)
    monkeypatch.setattr(settings, "qwen21_x2_max_output_pixels", 1048576)
    assert diffusion_output_size(SOURCE_SIZE, 2, family="qwen21") == (896, 1152)
    assert diffusion_output_size(SOURCE_SIZE, 1, family="qwen21") == (768, 1024)
    assert diffusion_output_size(SOURCE_SIZE, 3, family="qwen21") == (768, 1024)
    assert diffusion_output_size(SOURCE_SIZE, 2, family="qwen") == (768, 1024)
    assert diffusion_output_size((2048, 2048), 2, family="qwen21", canvas=True) == (2048, 2048)


@pytest.mark.parametrize("key", QWEN_KEYS)
@pytest.mark.parametrize("multiplier,expected", [(1, (768, 1024)), (2, (1600, 2048))])
def test_dlss_remains_separate_from_diffusion(
    key: str, multiplier: int, expected: tuple[int, int], limits: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    req = GenerationRequest(
        model_key=key, prompt="colors", negative_prompt="", mask=None,
        images=[Image.new("RGB", SOURCE_SIZE)], width=64, height=64,
        steps=6, guidance=1, true_cfg=1, strength=0.8, seed=17,
        size_multiplier=multiplier, dlss={"enabled": True, "upscaling_mode": "2x (Performance)"},
    )
    assert f"{expected[0]} × {expected[1]}" in preview(req)

    class Adapter:
        def generate(self, request: GenerationRequest, progress: object = None) -> list[Image.Image]:
            graph = configure_qwen21_graph(qwen21_turbo_template(key), request, ("source.png",))
            graph = add_dlss_nodes(graph, request)
            assert (graph["6"]["inputs"]["width"], graph["6"]["inputs"]["height"]) == expected
            assert graph["5"]["inputs"]["resolution"] == 1024
            assert any(node["class_type"] == "DLSS5EnhanceImages" for node in graph.values())
            return [Image.new("RGB", (expected[0] * 2, expected[1] * 2))]

    monkeypatch.setattr(engine.model_manager, "get", lambda _key: Adapter())
    monkeypatch.setattr(engine, "_save", lambda *_args: [])
    result = engine.generate(req, "Off", 0.35)
    assert result.images[0].size == (expected[0] * 2, expected[1] * 2)
    assert (req.width, req.height) == expected
    assert any("separate from the diffusion" in note for note in result.notes)