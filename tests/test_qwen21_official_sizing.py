from __future__ import annotations

import json
from contextlib import nullcontext
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from photo_edit_studio import engine, ui
from photo_edit_studio.config import settings
from photo_edit_studio.image_utils import diffusion_output_size
from photo_edit_studio.models import diffusers_adapters as adapters
from photo_edit_studio.models.registry import MODEL_SPECS
from photo_edit_studio.types import GenerationRequest, LoraSpec

KEY = "qwen-2.1-turbo-official"
SOURCE_SIZE = (3784, 4852)
# Independent publisher literal: never derive the expected grid from production.
PUBLISHER_SIGMAS = (1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568)


@pytest.fixture(autouse=True, params=["qwen-2.1-turbo-official", "qwen-2.1-turbo-official-extract"])
def official_profile(request, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(globals(), "KEY", request.param)


@pytest.fixture
def limits(monkeypatch: pytest.MonkeyPatch) -> None:
    for name, value in {
        "max_output_side": 1024, "max_output_pixels": 1048576,
        "qwen21_x2_max_output_side": 2048, "qwen21_x2_max_output_pixels": 4194304,
        "combine_max_output_side": 2048, "combine_max_output_pixels": 4194304,
    }.items():
        monkeypatch.setattr(settings, name, value)


def request(workflow: str = "standard", multiplier: int = 2) -> GenerationRequest:
    return GenerationRequest(
        model_key=KEY, prompt="Change colors", negative_prompt="",
        images=[] if workflow == "text" else [Image.new("RGB", SOURCE_SIZE)], mask=None,
        width=64, height=64, steps=8, guidance=1, true_cfg=1, strength=0.8, seed=17,
        size_multiplier=multiplier, compose=workflow == "combine",
        workflow="standard" if workflow == "combine" else workflow,
        output_resolution="2K", output_aspect_ratio="9:16",
    )


@pytest.mark.parametrize("workflow", ["standard", "combine", "text"])
@pytest.mark.parametrize("multiplier", [1, 2, 3])
def test_shared_preview_and_engine_size_original_upload_before_normalization(
    workflow: str, multiplier: int, limits: None, monkeypatch: pytest.MonkeyPatch,
) -> None:
    req = request(workflow, multiplier)
    original = req.images[0] if req.images else None
    expected = (1152, 2048) if workflow != "standard" else (1600, 2048) if multiplier == 2 else (768, 1024)
    mode = {"standard": ui.EDIT_MODE, "combine": ui.COMBINE_MODE, "text": ui.TEXT_MODE}[workflow]
    preview = ui._size_preview_for_mode(
        mode, multiplier, "2K", "9:16", original, None, None, original, None, None,
        KEY, "qwen-2511",
    )
    assert f"{expected[0]} × {expected[1]}" in preview
    events: list[str] = []
    real_sizing, real_normalize = engine.diffusion_output_size, engine.normalize_image

    def spy_sizing(*args: object, **kwargs: object) -> tuple[int, int]:
        assert not events
        assert all(image.size == SOURCE_SIZE for image in req.images)
        assert kwargs["family"] == "qwen21-official"
        events.append("size")
        return real_sizing(*args, **kwargs)

    def spy_normalize(image: Image.Image, *, max_side: int) -> Image.Image:
        assert events[0] == "size"
        assert (req.width, req.height) == expected
        events.append("normalize")
        return real_normalize(image, max_side=max_side)

    class Adapter:
        def generate(self, received: GenerationRequest, progress: object = None) -> list[Image.Image]:
            assert (received.width, received.height) == expected
            assert received.steps == 8
            if received.images:
                assert max(received.images[0].size) == max(expected)
            events.append("adapter")
            return [Image.new("RGB", expected)]

    monkeypatch.setattr(engine, "diffusion_output_size", spy_sizing)
    monkeypatch.setattr(engine, "normalize_image", spy_normalize)
    monkeypatch.setattr(engine.model_manager, "get", lambda key: Adapter() if key == KEY else pytest.fail(key))
    monkeypatch.setattr(engine, "_save", lambda *_args: [])
    result = engine.generate(req, "Off", 0.35)
    assert events == (["size", "adapter"] if workflow == "text" else ["size", "normalize", "adapter"])
    assert result.images[0].size == expected
    if original is not None:
        assert original.size == SOURCE_SIZE


def test_official_x2_caps_are_shared_configurable_and_do_not_leak(
    limits: None, monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert diffusion_output_size(SOURCE_SIZE, 2, family="qwen21-official") == (1600, 2048)
    monkeypatch.setattr(settings, "qwen21_x2_max_output_side", 1536)
    assert diffusion_output_size(SOURCE_SIZE, 2, family="qwen21-official") == (1152, 1536)
    monkeypatch.setattr(settings, "qwen21_x2_max_output_side", 2048)
    monkeypatch.setattr(settings, "qwen21_x2_max_output_pixels", 1048576)
    assert diffusion_output_size(SOURCE_SIZE, 2, family="qwen21-official") == (896, 1152)
    for multiplier in (1, 3):
        assert diffusion_output_size(SOURCE_SIZE, multiplier, family="qwen21-official") == (768, 1024)
    assert diffusion_output_size(SOURCE_SIZE, 2, family="qwen") == (768, 1024)
    assert diffusion_output_size((1152, 2048), 2, family="qwen21-official", canvas=True) == (1152, 2048)


@pytest.fixture
def fake_official(monkeypatch: pytest.MonkeyPatch) -> tuple[adapters.Qwen21OfficialAdapter, list[dict]]:
    calls: list[dict] = []

    class Generator:
        def __init__(self, *, device: str) -> None:
            assert device == "cpu"

        def manual_seed(self, seed: int) -> int:
            return seed

    class Pipe:
        config = SimpleNamespace(sample_sigmas=list(PUBLISHER_SIGMAS))
        transformer = SimpleNamespace(config=SimpleNamespace(causal_condition=True))

        def __call__(self, **kwargs: object) -> SimpleNamespace:
            calls.append(kwargs)
            return SimpleNamespace(images=[Image.new("RGB", (640, 768))])

    def forbidden(*_args: object, **_kwargs: object) -> None:
        pytest.fail("Fake official generation must not load weights or optional adapters")

    monkeypatch.setattr(adapters, "_runtime", lambda: (None, SimpleNamespace(Generator=Generator, no_grad=nullcontext)))
    adapter_class = adapters.Qwen21OfficialExtractAdapter if KEY.endswith("-extract") else adapters.Qwen21OfficialAdapter
    adapter = adapter_class(MODEL_SPECS[KEY])
    adapter.pipe = Pipe()
    monkeypatch.setattr(adapter, "load", forbidden)
    monkeypatch.setattr(adapter, "apply_loras", forbidden)
    return adapter, calls


@pytest.mark.parametrize("workflow", ["standard", "combine", "text"])
def test_fake_pipeline_retains_publisher_grid_and_runs_eight_step_causal_cache(
    workflow: str, fake_official: tuple[adapters.Qwen21OfficialAdapter, list[dict]],
) -> None:
    adapter, calls = fake_official
    req = request(workflow)
    req.images = [] if workflow == "text" else [Image.new("RGB", (64, 64), color) for color in ("red", "blue", "green")]
    req.width, req.height, req.count = 1152, 2048, 2
    assert adapters.QWEN21_OFFICIAL_SIGMAS == PUBLISHER_SIGMAS
    assert [image.size for image in adapter.generate(req)] == [(640, 768), (640, 768)]
    assert len(calls) == 2
    for index, call in enumerate(calls):
        assert (call["width"], call["height"], call["num_inference_steps"]) == (1152, 2048, 8)
        assert call["true_cfg_scale"] == 1 and call["use_kv_cache"] is True
        assert call["generator"] == 17 + index and call["num_images_per_prompt"] == 1
        assert call["prompt"] == req.prompt
        assert not {"sigmas", "sample_sigmas", "timesteps", "guidance_scale", "negative_prompt", "strength"}.intersection(call)
        if workflow == "text":
            assert "image" not in call
        else:
            assert call["image"] is req.images
    assert tuple(adapter.pipe.config.sample_sigmas) == PUBLISHER_SIGMAS


@pytest.mark.parametrize("field,value,match", [
    ("steps", 6, "eight"), ("guidance", 2, "guidance 1"), ("true_cfg", 2, "True CFG 1"),
    ("negative_prompt", "blur", "empty negative"),
    ("swap_kind", "Head", "no swaps"), ("guidance", -1, "guidance 1"),
    ("true_cfg", -1, "True CFG 1"), ("dlss", {"enabled": False}, "DLSS"),
    ("dlss", {"enabled": True}, "DLSS"),
    ("loras", [LoraSpec("style", Path("never-read.safetensors"), 1, "style")], "LoRAs"),
])
def test_invalid_official_requests_fail_before_model_selection(
    field: str, value: object, match: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    req = request()
    setattr(req, field, value)
    before = {field.name: getattr(req, field.name) for field in fields(req)}
    original_images = list(req.images)
    monkeypatch.setattr(engine, "diffusion_output_size", lambda *_args, **_kwargs: pytest.fail("Invalid request resized"))
    monkeypatch.setattr(engine, "normalize_image", lambda *_args, **_kwargs: pytest.fail("Invalid request normalized"))
    monkeypatch.setattr(engine.model_manager, "get", lambda *_args: pytest.fail("Invalid request selected a model"))
    with pytest.raises(ValueError, match=match):
        engine.generate(req, "Off", 0.35)
    assert {field.name: getattr(req, field.name) for field in fields(req)} == before
    assert req.images == original_images
    assert all(image.size == SOURCE_SIZE for image in req.images)


def test_incomplete_swap_rejected_before_model_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    req = request("swap")
    monkeypatch.setattr(engine.model_manager, "get", lambda *_args: pytest.fail("Invalid swap selected a model"))
    match = "no swaps" if KEY.endswith("-extract") else "exactly two ordered images"
    with pytest.raises(ValueError, match=match):
        engine.generate(req, "Off", 0.35)


@pytest.fixture
def saved_official(
    fake_official: tuple[adapters.Qwen21OfficialAdapter, list[dict]], limits: None,
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> tuple[dict, dict, list[list[int]]]:
    adapter, calls = fake_official
    monkeypatch.setattr(engine.model_manager, "get", lambda _key: adapter)
    monkeypatch.setattr(settings, "output_dir", tmp_path)
    monkeypatch.setattr(settings, "max_batch_count", 2)
    req = request()
    req.count = 2
    result = engine.generate(req, "Off", 0.35)
    assert len(calls) == 2 and all(call["num_inference_steps"] == 8 for call in calls)
    assert all((call["width"], call["height"]) == (1600, 2048) for call in calls)
    metadata = json.loads((result.saved_paths[0].parent / "metadata.json").read_text(encoding="utf-8"))
    png_payloads, sizes = [], []
    for path in result.saved_paths:
        with Image.open(path) as image:
            sizes.append(list(image.size))
            png_payloads.append(json.loads(image.info["generation"]))
    assert len(png_payloads) == 2 and all(payload == metadata for payload in png_payloads)
    assert sizes == [[640, 768], [640, 768]]  # Deliberately differ from requested canvas.
    return metadata, png_payloads[0], sizes


@pytest.mark.parametrize("channel", ["json", "png"])
def test_saved_generic_controls_remain_eight_and_requested_dimensions_are_distinct(
    channel: str, saved_official: tuple[dict, dict, list[list[int]]],
) -> None:
    metadata = saved_official[0 if channel == "json" else 1]
    assert (metadata["steps"], metadata["guidance"], metadata["true_cfg"]) == (8, 1, 1)
    assert (metadata["width"], metadata["height"]) == (1600, 2048)
    assert metadata["model"] == "Qwen/Qwen-Image-2.1-Turbo"
    assert metadata["model_key"] == KEY and metadata["loras"] == []
    assert "dlss" not in metadata
    assert metadata["scheduler_source"] == "checkpoint" and metadata["use_kv_cache"] is True
    if KEY.endswith("-extract"):
        assert metadata["experimental_stack"] is True
        assert metadata["checkpoint_source"] == "official Turbo (not original base)"
        mandatory = metadata["mandatory_adapter"]
        assert mandatory["weight"] == 1.0
        assert mandatory["version_id"] == 3394831 and mandatory["file_id"] == 3284648
        assert mandatory["filename"] == "qwen_image_2.1_turbo_lora_avg_rank_178_bf16.safetensors"
        assert mandatory["size_bytes"] == 913314512
        assert mandatory["sha256"].upper() == "208DD43250E1E01467BA190572AE2E7107A7870EC1FF026BC65F791FA1E80E95"
    else:
        assert "mandatory_adapter" not in metadata and "experimental_stack" not in metadata


@pytest.mark.parametrize("channel", ["json", "png"])
def test_saved_official_metadata_records_actual_publisher_sigma_grid(
    channel: str, saved_official: tuple[dict, dict, list[list[int]]],
) -> None:
    metadata = saved_official[0 if channel == "json" else 1]
    # Proposed official schema: sample_sigmas follows publisher model_index.json,
    # not the unrelated Comfy `sigmas`/ManualSigmas fields. Missing is a regression.
    assert "sample_sigmas" in metadata, "Official metadata must expose publisher sample_sigmas, not merely steps=8"
    assert metadata["sample_sigmas"] == list(PUBLISHER_SIGMAS)


@pytest.mark.parametrize("channel", ["json", "png"])
def test_saved_official_metadata_records_actual_returned_dimensions(
    channel: str, saved_official: tuple[dict, dict, list[list[int]]],
) -> None:
    metadata = saved_official[0 if channel == "json" else 1]
    # Reuse the existing sheet/All2Real schema: width/height stay requested canvas;
    # actual_output_sizes is an ordered list of [width, height] for every output.
    assert "actual_output_sizes" in metadata, "Requested canvas is not a substitute for actual output dimensions"
    assert metadata["actual_output_sizes"] == saved_official[2]