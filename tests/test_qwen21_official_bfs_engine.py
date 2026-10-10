from __future__ import annotations

import json
from dataclasses import fields, replace
from pathlib import Path

import pytest
from PIL import Image

from photo_edit_studio import engine, ui
from photo_edit_studio.config import settings
from photo_edit_studio.models import diffusers_adapters as adapters
from photo_edit_studio.types import GenerationRequest, LoraSpec

KEY = "qwen-2.1-turbo-official"
SIGMAS = [1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568]
# Publisher literals deliberately independent of production profiles and pins.
PROFILES = {
    "Head": {
        "filename": "bfs_head_v1.1_qwen_2.1.safetensors",
        "version": "1.1", "size_bytes": 260096144,
        "sha256": "d1d748d5601077f3b6d05766f6823510e901970d916afa404a97e85dc92fa88e",
        "trigger": (
            "head_swap: start with <image1> as the base image, keeping its lighting, "
            "environment, and background. remove the head from <image1> completely and "
            "replace it with the head from <image2>, strictly preserving the hair, eye "
            "color, nose structure from <image2>. copy the direction of the eye, head "
            "rotation, micro expressions from <image1>, high quality, sharp details, 4k"
        ),
    },
    "Body": {
        "filename": "bfs_body_swap_v1.0_qwen_2.1.safetensors",
        "version": "1.0", "size_bytes": 209753576,
        "sha256": "7dc0a53aba4dbc70c204f498936a619d226559da11011f748c389189af46b664",
        "trigger": (
            "body_swap: start with <image1> as the base image, keeping its lighting, "
            "environment, and background. replace the body from <image1> with the body "
            "from <image2>, strictly preserving pose, background, lightning and structure "
            "from <image2>. copy the pose, direction of the eye, head rotation, micro "
            "expressions from <image1>"
        ),
    },
}


@pytest.fixture(autouse=True)
def isolated_runtime(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for name, value in {
        "max_output_side": 1024, "max_output_pixels": 1048576,
        "qwen21_x2_max_output_side": 2048, "qwen21_x2_max_output_pixels": 4194304,
        "lora_dir": tmp_path / "loras", "output_dir": tmp_path / "outputs",
        "max_batch_count": 2,
    }.items():
        monkeypatch.setattr(settings, name, value)

    def forbidden(*_args: object, **_kwargs: object) -> None:
        pytest.fail("Engine/UI tests must not allocate weights, invoke a live runtime or Comfy backend")

    monkeypatch.setattr(adapters, "_runtime", forbidden)
    monkeypatch.setattr(engine.model_manager, "get", forbidden)
    monkeypatch.setattr(engine.model_manager, "unload", forbidden)


def swap_request(kind: str = "Head", weight: float = 0.65, multiplier: int = 2) -> GenerationRequest:
    filename = PROFILES[kind]["filename"]
    return GenerationRequest(
        model_key=KEY, prompt=PROFILES[kind]["trigger"] + " Keep colors.", negative_prompt="",
        images=[Image.new("RGB", (3784, 4852), "red"), Image.new("RGB", (1600, 800), "blue")],
        mask=None, width=64, height=64, steps=8, guidance=1, true_cfg=1, strength=0.8,
        seed=17, count=2, size_multiplier=multiplier, compose=False,
        workflow="swap", swap_kind=kind,
        loras=[LoraSpec(filename, settings.lora_dir / "qwen21" / filename, weight, "bfs_swap")],
    )


@pytest.mark.parametrize("kind", ["Head", "Body"])
@pytest.mark.parametrize("weight", [0.1, 0.65, 1.5])
@pytest.mark.parametrize("multiplier,expected", [(1, (768, 1024)), (2, (1600, 2048)), (3, (768, 1024))])
def test_official_swap_order_sizing_and_saved_json_png_provenance(
    kind: str, weight: float, multiplier: int, expected: tuple[int, int],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    req = swap_request(kind, weight, multiplier)
    originals = list(req.images)
    preview = ui._size_preview_for_mode(
        ui.SWAP_MODE, multiplier, "2K", "9:16", None, None, originals[0], None, None, None,
        "flux-klein-4b", KEY,
    )
    assert f"{expected[0]} × {expected[1]}" in preview
    events: list[str] = []
    real_sizing, real_normalize = engine.diffusion_output_size, engine.normalize_image

    def sizing(*args: object, **kwargs: object) -> tuple[int, int]:
        assert not events
        assert req.images == originals
        assert [image.size for image in req.images] == [(3784, 4852), (1600, 800)]
        assert (kwargs["family"], kwargs["workflow"], kwargs["canvas"]) == ("qwen21-official", "swap", False)
        events.append("size")
        return real_sizing(*args, **kwargs)

    def normalize(image: Image.Image, *, max_side: int) -> Image.Image:
        assert events[0] == "size" and (req.width, req.height) == expected
        assert image is originals[len(events) - 1]
        assert max_side == max(expected)
        events.append("normalize")
        return real_normalize(image, max_side=max_side)

    returned_sizes = [(640, 768), (704, 832)]

    class Adapter:
        def generate(self, received: GenerationRequest, progress: object = None) -> list[Image.Image]:
            assert received is req
            assert events == ["size", "normalize", "normalize"]
            assert (received.width, received.height) == expected
            assert len(received.images) == 2
            assert [image.getpixel((0, 0)) for image in received.images] == [(255, 0, 0), (0, 0, 255)]
            assert max(received.images[0].size) == max(expected)
            assert received.images[1].width == 2 * received.images[1].height
            assert (received.steps, received.guidance, received.true_cfg) == (8, 1, 1)
            assert received.loras == [LoraSpec(
                PROFILES[kind]["filename"], settings.lora_dir / "qwen21" / PROFILES[kind]["filename"],
                weight, "bfs_swap",
            )]
            assert received.prompt == PROFILES[kind]["trigger"] + " Keep colors."
            assert received.dlss is None and received.mask is None and not received.compose
            events.append("adapter")
            return [Image.new("RGB", size) for size in returned_sizes]

    def select(key: str) -> Adapter:
        assert key == KEY
        return Adapter()

    monkeypatch.setattr(engine, "diffusion_output_size", sizing)
    monkeypatch.setattr(engine, "normalize_image", normalize)
    monkeypatch.setattr(engine.model_manager, "get", select)
    result = engine.generate(req, "Off", 0.35)
    assert events == ["size", "normalize", "normalize", "adapter"]
    assert [image.size for image in originals] == [(3784, 4852), (1600, 800)]
    assert [image.size for image in result.images] == returned_sizes
    assert len(result.saved_paths) == 2
    metadata = json.loads((result.saved_paths[0].parent / "metadata.json").read_text(encoding="utf-8"))
    for index, path in enumerate(result.saved_paths):
        with Image.open(path) as image:
            assert image.size == returned_sizes[index]
            assert json.loads(image.info["generation"]) == metadata
    assert (metadata["model_key"], metadata["model"]) == (KEY, "Qwen/Qwen-Image-2.1-Turbo")
    assert (metadata["workflow"], metadata["swap_kind"], metadata["input_count"]) == ("swap", kind, 2)
    assert (metadata["steps"], metadata["guidance"], metadata["true_cfg"]) == (8, 1, 1)
    assert (metadata["width"], metadata["height"], metadata["size_multiplier"]) == (*expected, multiplier)
    assert metadata["actual_output_sizes"] == [[640, 768], [704, 832]]
    assert metadata["compose"] is False
    assert metadata["scheduler_source"] == "checkpoint" and metadata["use_kv_cache"] is True
    assert metadata["sample_sigmas"] == SIGMAS
    assert metadata["mandatory_adapter"] == {
        "name": "bfs_swap", "filename": PROFILES[kind]["filename"], "version": PROFILES[kind]["version"],
        "sha256": PROFILES[kind]["sha256"], "size_bytes": PROFILES[kind]["size_bytes"], "weight": weight,
    }
    assert metadata["loras"] == [{"name": PROFILES[kind]["filename"], "weight": weight}]
    assert metadata["ordered_image_sources"] == ["Source 1: target body/scene", "Source 2: replacement reference"]
    assert metadata["swap_trigger"] == PROFILES[kind]["trigger"]
    assert metadata["prompt_stored"] is False
    assert not {
        "experimental_stack", "checkpoint_source", "sigmas", "timesteps", "sampler", "scheduler",
        "comfy", "comfy_workflow", "comfy_prompt", "dlss", "turbo_lora",
    }.intersection(metadata)
    assert "official_extract" not in json.dumps(metadata)
    assert any(f"Two-image BFS {kind} v{PROFILES[kind]['version']}" in note for note in result.notes)


@pytest.mark.parametrize("field,value,match", [
    ("model_key", "qwen-2.1-turbo-official-extract", "no swaps"),
    ("swap_kind", None, "Head or Body"), ("swap_kind", "Face", "Head or Body"),
    ("steps", 6, "eight"), ("guidance", 2, "guidance 1"), ("true_cfg", 2, "True CFG 1"),
    ("negative_prompt", "blur", "empty negative"),
    ("dlss", {"enabled": False}, "DLSS"), ("dlss", {"enabled": True}, "DLSS"),
])
def test_official_swap_invalid_controls_reject_before_sizing_selection_or_save(
    field: str, value: object, match: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    req = swap_request()
    setattr(req, field, value)
    assert_rejected_unchanged(req, match, monkeypatch)


def assert_rejected_unchanged(req: GenerationRequest, match: str, monkeypatch: pytest.MonkeyPatch) -> None:
    before = {item.name: getattr(req, item.name) for item in fields(req)}

    def forbidden(*_args: object, **_kwargs: object) -> None:
        pytest.fail("Invalid official swap reached sizing, normalization or saving")

    monkeypatch.setattr(engine, "diffusion_output_size", forbidden)
    monkeypatch.setattr(engine, "normalize_image", forbidden)
    monkeypatch.setattr(engine, "_save", forbidden)
    with pytest.raises(ValueError, match=match):
        engine.generate(req, "Off", 0.35)
    assert {item.name: getattr(req, item.name) for item in fields(req)} == before


@pytest.mark.parametrize("count", [0, 1, 3])
def test_official_swap_requires_exactly_two_sources(count: int, monkeypatch: pytest.MonkeyPatch) -> None:
    req = swap_request()
    req.images = [Image.new("RGB", (64, 64)) for _ in range(count)]
    assert_rejected_unchanged(req, "exactly two ordered images", monkeypatch)


@pytest.mark.parametrize("defect", ["missing", "optional", "filename", "path", "adapter", "other-kind"])
def test_official_swap_requires_exact_matching_mandatory_bfs_only(
    defect: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    req = swap_request()
    item = req.loras[0]
    if defect == "missing":
        req.loras = []
    elif defect == "optional":
        req.loras.append(LoraSpec("style", Path("never-read.safetensors"), 1, "style"))
    else:
        change = {
            "filename": {"name": "wrong.safetensors"},
            "path": {"path": settings.lora_dir / "elsewhere" / item.name},
            "adapter": {"adapter_name": "official_extract"},
            "other-kind": {"name": PROFILES["Body"]["filename"],
                           "path": settings.lora_dir / "qwen21" / PROFILES["Body"]["filename"]},
        }[defect]
        req.loras = [replace(item, **change)]
    match = "exactly one mandatory BFS" if defect in {"missing", "optional"} else "exact matching BFS"
    assert_rejected_unchanged(req, match, monkeypatch)


@pytest.mark.parametrize("weight", [0, -1, 0.09, 1.51, float("nan"), float("inf"), -float("inf")])
def test_official_swap_rejects_invalid_bfs_weight(weight: float, monkeypatch: pytest.MonkeyPatch) -> None:
    assert_rejected_unchanged(swap_request(weight=weight), "BFS weight", monkeypatch)