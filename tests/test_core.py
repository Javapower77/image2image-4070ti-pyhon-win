from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from photo_edit_studio.engine import generate
from photo_edit_studio.image_utils import composite_with_mask, normalize_image
from photo_edit_studio.models import MODEL_SPECS, ModelManager
from photo_edit_studio.types import GenerationRequest
from photo_edit_studio.ui import (
    COMBINE_MODE,
    EDIT_MODE,
    KREA_EDIT_MODE,
    TEXT_MODE,
    _canvas_size_preview,
    _collect_images,
    _combine_size_preview,
    _mode_changed,
    _mode_model_defaults,
    _model_choices_for_mode,
    _run,
    _size_preview,
    _standard_model_changed,
)


def test_registry_has_requested_models() -> None:
    assert set(MODEL_SPECS) == {
        "qwen-2511",
        "qwen-2511-aio",
        "qwen-2.1-turbo",
        "qwen-2.1-turbo-r128",
        "firered-1.1",
        "flux-klein-4b",
        "krea-2-turbo",
    }


def test_manager_selects_firered_gguf_backend() -> None:
    from photo_edit_studio.models.comfy_swap import ComfyFireRedAdapter

    adapter = ModelManager().get("firered-1.1")
    assert isinstance(adapter, ComfyFireRedAdapter)
    assert MODEL_SPECS["firered-1.1"].default_steps == 8
    assert MODEL_SPECS["firered-1.1"].family == "firered"


def test_manager_selects_qwen_aio_adapter() -> None:
    from photo_edit_studio.models.diffusers_adapters import QwenAioAdapter

    adapter = ModelManager().get("qwen-2511-aio")
    assert isinstance(adapter, QwenAioAdapter)
    assert MODEL_SPECS["qwen-2511-aio"].default_steps == 4


def test_manager_selects_qwen21_turbo_adapter() -> None:
    from photo_edit_studio.models.comfy_swap import ComfyQwen21Adapter

    adapter = ModelManager().get("qwen-2.1-turbo")
    assert isinstance(adapter, ComfyQwen21Adapter)
    assert MODEL_SPECS["qwen-2.1-turbo"].default_steps == 6


def test_manager_selects_krea_text_to_image_adapter() -> None:
    from photo_edit_studio.models.diffusers_adapters import Krea2Adapter

    adapter = ModelManager().get("krea-2-turbo")
    assert isinstance(adapter, Krea2Adapter)
    assert MODEL_SPECS["krea-2-turbo"].task == "text-to-image"
    assert MODEL_SPECS["krea-2-turbo"].default_steps == 8


def test_flux_custom_encoder_resolves_actual_abl_filename(tmp_path: Path) -> None:
    from photo_edit_studio.models.diffusers_adapters import resolve_flux_klein_text_encoder

    actual = tmp_path / "qwen3-4b-abl-q4_0.gguf"
    actual.write_bytes(b"GGUF")
    configured = tmp_path / "qwen3-4b-alb-q4_0.gguf"
    assert resolve_flux_klein_text_encoder(configured) == actual


def test_flux_custom_encoder_rejects_missing_file(tmp_path: Path) -> None:
    from photo_edit_studio.models.diffusers_adapters import resolve_flux_klein_text_encoder

    with pytest.raises(FileNotFoundError, match="custom text encoder"):
        resolve_flux_klein_text_encoder(tmp_path / "qwen3-4b-alb-q4_0.gguf")


def test_manager_starts_unloaded() -> None:
    assert ModelManager().status == "No model loaded"


def test_normalize_image_converts_and_resizes() -> None:
    image = Image.new("RGBA", (3000, 1000), "red")
    result = normalize_image(image, max_side=1000)
    assert result.mode == "RGB"
    assert result.size == (1000, 333)


def test_mask_composite() -> None:
    source = Image.new("RGB", (8, 8), "red")
    generated = Image.new("RGB", (8, 8), "blue")
    white = Image.new("L", (8, 8), 255)
    assert composite_with_mask(source, generated, white, feather=0).getpixel((0, 0)) == (0, 0, 255)


def test_empty_prompt_rejected() -> None:
    with pytest.raises(ValueError, match="instruction"):
        _run(
            mode=EDIT_MODE,
            source=Image.new("RGB", (8, 8), "red"),
            references=None,
            combine_1=None,
            combine_2=None,
            combine_3=None,
            mask=None,
            model_key="qwen-2511",
            prompt=" ",
            negative_prompt="",
            size_multiplier=1,
            output_resolution="1K",
            steps=4,
            guidance=1.0,
            true_cfg=4.0,
            strength=0.8,
            seed=0,
            count=1,
            preserve_identity=True,
            restoration="Off",
            restore_weight=0.35,
        )


def test_combine_mode_collects_up_to_three_images() -> None:
    one = Image.new("RGB", (64, 32), "red")
    two = Image.new("RGB", (32, 64), "blue")
    three = Image.new("RGB", (48, 48), "green")
    images = _collect_images(COMBINE_MODE, None, [one], one, two, three)
    assert images == [one, two, three]


def test_edit_mode_uses_source_and_references() -> None:
    source = Image.new("RGB", (64, 32), "red")
    reference = Image.new("RGB", (32, 32), "blue")
    images = _collect_images(EDIT_MODE, source, [reference], None, None, None)
    assert images == [source, reference]


def test_text_mode_does_not_collect_images() -> None:
    source = Image.new("RGB", (64, 32), "red")
    assert _collect_images(TEXT_MODE, source, [source], source, source, source) == []


def test_size_preview_uses_image_one_aspect() -> None:
    source = Image.new("RGB", (1024, 768), "red")
    text = _size_preview(EDIT_MODE, 2, "1K", source, None, None, None)
    assert "2048 × 1536" in text


def test_square_upload_keeps_square_output_canvas(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from photo_edit_studio import engine
    from photo_edit_studio.config import settings

    seen: list[tuple[tuple[int, int], tuple[int, int]]] = []

    class Adapter:
        def generate(self, request: GenerationRequest, progress: object = None) -> list[Image.Image]:
            seen.append((request.images[0].size, (request.width, request.height)))
            return [Image.new("RGB", (request.width, request.height))]

    monkeypatch.setattr(engine.model_manager, "get", lambda _key: Adapter())
    monkeypatch.setattr(settings, "output_dir", tmp_path)
    request = GenerationRequest(
        model_key="qwen-2511-aio", prompt="Recolor clothes", negative_prompt="",
        images=[Image.new("RGB", (526, 526))], mask=None,
        width=1024, height=1024, steps=4, guidance=1.0, true_cfg=2.0,
        strength=0.8, seed=17,
    )
    result = generate(request, "Off", 0.35)
    assert seen == [((512, 512), (512, 512))]
    assert result.images[0].size == (512, 512)


def test_qwen_edits_preserve_frame_instruction_but_text_creation_does_not(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    from photo_edit_studio.models.diffusers_adapters import QwenAioAdapter

    class Pipe:
        def __init__(self) -> None:
            self.prompts: list[str] = []

        def __call__(self, *, prompt: str, image: Image.Image | None = None) -> SimpleNamespace:
            self.prompts.append(prompt)
            return SimpleNamespace(images=[Image.new("RGB", (512, 512))])

    adapter = QwenAioAdapter(MODEL_SPECS["qwen-2511-aio"])
    pipe = Pipe()
    adapter.pipe = pipe
    monkeypatch.setattr(adapter, "apply_loras", lambda _request: None)
    req = GenerationRequest(
        model_key="qwen-2511-aio", prompt="Change clothing color", negative_prompt="",
        images=[Image.new("RGB", (512, 512))], mask=None,
        width=512, height=512, steps=4, guidance=1.0, true_cfg=2.0,
        strength=0.8, seed=17,
    )
    adapter.generate(req)
    assert "Do not crop, zoom in, or reframe" in pipe.prompts[0]
    req.preserve_identity = False
    adapter.generate(req)
    assert "Do not crop, zoom in, or reframe" in pipe.prompts[1]
    assert "facial geometry" not in pipe.prompts[1]


def test_size_preview_combine_uses_resolution_canvas() -> None:
    text = _combine_size_preview("2K", None, None, None)
    assert "2048 × 2048" in text
    assert "2K" in text


def test_size_preview_combine_uses_aspect_ratio() -> None:
    text = _combine_size_preview("2K", None, None, None, "9:16")
    assert "1152 × 2048" in text
    assert "9:16" in text


def test_text_canvas_preview_uses_selected_resolution_and_aspect() -> None:
    text = _canvas_size_preview(TEXT_MODE, "2K", None, None, None, "16:9")
    assert "2048 × 1152" in text
    assert "2K · 16:9" in text


def test_text_mode_limits_model_dropdown_and_keeps_flux_selected() -> None:
    choices = _model_choices_for_mode(TEXT_MODE, "flux-klein-4b")
    assert [key for _, key in choices["choices"]] == ["krea-2-turbo", "flux-klein-4b", "qwen-2.1-turbo", "qwen-2.1-turbo-r128"]
    assert choices["value"] == "flux-klein-4b"
    assert _model_choices_for_mode(TEXT_MODE, "qwen-2511")["value"] == "krea-2-turbo"
    assert _standard_model_changed("flux-klein-4b", TEXT_MODE)[5]["value"] == TEXT_MODE


@pytest.mark.parametrize("mode", [EDIT_MODE, COMBINE_MODE])
def test_image_edit_modes_exclude_text_only_krea(mode: str) -> None:
    choices = _model_choices_for_mode(mode, "krea-2-turbo")
    keys = [key for _, key in choices["choices"]]
    assert keys == ["qwen-2511", "qwen-2511-aio", "qwen-2.1-turbo", "qwen-2.1-turbo-r128", "firered-1.1", "flux-klein-4b"]
    assert choices["value"] == "qwen-2511"
    assert _model_choices_for_mode(mode, "firered-1.1")["value"] == "firered-1.1"


def test_krea_reference_mode_hides_combine_uploads_and_duplicate_model_label() -> None:
    changes = _mode_changed(KREA_EDIT_MODE)
    assert "mode-hidden" in changes[1]["elem_classes"]
    assert "mode-hidden" not in changes[3]["elem_classes"]
    assert "mode-hidden" in changes[10]["elem_classes"]  # standard model dropdown
    assert "mode-hidden" not in changes[13]["elem_classes"]  # dedicated Krea label
    defaults = _mode_model_defaults(KREA_EDIT_MODE, "qwen-2511-aio", "qwen-2511", "Head")
    assert defaults[4]["value"] == ""
    assert "mode-hidden" in defaults[4]["elem_classes"]
    assert "mode-hidden" in _standard_model_changed("krea-2-turbo", KREA_EDIT_MODE)[3]["elem_classes"]


def test_switching_back_to_combine_restores_uploads_and_model_info() -> None:
    changes = _mode_changed(COMBINE_MODE)
    assert "mode-hidden" not in changes[1]["elem_classes"]
    assert "mode-hidden" in changes[3]["elem_classes"]
    assert changes[10]["elem_classes"] == ["mode-hidden"]
    assert changes[11]["elem_classes"] == ["mode-hidden"]
    defaults = _mode_model_defaults(COMBINE_MODE, "qwen-2511-aio", "qwen-2511", "Head")
    assert "mode-hidden" not in defaults[4]["elem_classes"]
    assert "Qwen Rapid AIO" in defaults[4]["value"]


def test_text_mode_rejects_qwen_edit_model() -> None:
    with pytest.raises(ValueError, match="text-capable model"):
        _run(
            mode=TEXT_MODE, source=None, references=None, combine_1=None,
            combine_2=None, combine_3=None, mask=None, model_key="qwen-2511",
            prompt="a fox", negative_prompt="", size_multiplier=1,
            output_resolution="1K", steps=4, guidance=1.0, true_cfg=1.0,
            strength=0.8, seed=0, count=1, preserve_identity=False,
            restoration="Off", restore_weight=0.35,
            progress=lambda *_args, **_kwargs: None,
        )


def test_compose_without_images_is_rejected() -> None:
    request = GenerationRequest(
        model_key="qwen-2511",
        prompt="combine them",
        negative_prompt="",
        images=[],
        mask=None,
        width=1024,
        height=1024,
        steps=4,
        guidance=1.0,
        true_cfg=4.0,
        strength=0.8,
        seed=0,
        compose=True,
    )
    with pytest.raises(ValueError, match="1"):
        generate(request, "Off", 0.35)


def test_combine_2k_aspect_reaches_adapter_and_metadata(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from photo_edit_studio import engine
    from photo_edit_studio.config import settings

    received: list[tuple[int, int]] = []

    class Adapter:
        def generate(self, request: GenerationRequest, progress: object = None) -> list[Image.Image]:
            received.append((request.width, request.height))
            return [Image.new("RGB", (request.width, request.height))]

    monkeypatch.setattr(engine.model_manager, "get", lambda _key: Adapter())
    monkeypatch.setattr(settings, "output_dir", tmp_path)
    request = GenerationRequest(
        model_key="qwen-2511", prompt="combine", negative_prompt="",
        images=[Image.new("RGB", (800, 600))], mask=None,
        width=1024, height=1024, steps=4, guidance=1.0,
        true_cfg=4.0, strength=0.8, seed=123, compose=True,
        output_resolution="2K", output_aspect_ratio="9:16",
    )
    result = generate(request, "Off", 0.35)
    assert received == [(1152, 2048)]
    assert result.images[0].size == (1152, 2048)
    metadata = json.loads((result.saved_paths[0].parent / "metadata.json").read_text())
    assert (metadata["width"], metadata["height"]) == (1152, 2048)
    assert metadata["output_aspect_ratio"] == "9:16"


def test_krea_text_2k_aspect_reaches_backend_and_metadata(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from photo_edit_studio import engine
    from photo_edit_studio.config import settings

    received: list[tuple[int, int, str]] = []

    def fake_generate(self: object, request: GenerationRequest, progress: object = None) -> list[Image.Image]:
        received.append((request.width, request.height, request.workflow))
        return [Image.new("RGB", (64, 64))]

    monkeypatch.setattr(
        "photo_edit_studio.models.comfy_swap.ComfyKreaSwapAdapter.generate", fake_generate
    )
    monkeypatch.setattr(settings, "output_dir", tmp_path)
    request = GenerationRequest(
        model_key="krea-2-turbo", prompt="a fox", negative_prompt="", images=[],
        mask=None, width=1024, height=1024, steps=8, guidance=1.0,
        true_cfg=0.0, strength=1.0, seed=456,
        output_resolution="2K", output_aspect_ratio="9:16",
    )
    result = engine.generate(request, "Off", 0.35)
    assert received == [(1152, 2048, "krea-text")]
    metadata = json.loads((result.saved_paths[0].parent / "metadata.json").read_text())
    assert (metadata["width"], metadata["height"]) == (1152, 2048)
    assert (metadata["output_resolution"], metadata["output_aspect_ratio"]) == ("2K", "9:16")


def test_flux_text_2k_aspect_reaches_selected_adapter(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from photo_edit_studio import engine
    from photo_edit_studio.config import settings

    seen: list[tuple[str, str, int, int]] = []

    class FakeFluxAdapter:
        def generate(self, request: GenerationRequest, progress: object = None) -> list[Image.Image]:
            seen.append((request.model_key, request.workflow, request.width, request.height))
            return [Image.new("RGB", (64, 64))]

    monkeypatch.setattr(engine.model_manager, "get", lambda _key: FakeFluxAdapter())
    monkeypatch.setattr(settings, "output_dir", tmp_path)
    request = GenerationRequest(
        model_key="flux-klein-4b", prompt="a fox", negative_prompt="", images=[],
        mask=None, width=1024, height=1024, steps=4, guidance=1.0,
        true_cfg=1.0, strength=0.8, seed=987, compose=True,
        workflow="text", output_resolution="2K", output_aspect_ratio="16:9",
    )
    result = engine.generate(request, "Off", 0.35)
    assert seen == [("flux-klein-4b", "text", 2048, 1152)]
    metadata = json.loads((result.saved_paths[0].parent / "metadata.json").read_text())
    assert metadata["model"] == MODEL_SPECS["flux-klein-4b"].repo_id
    assert metadata["output_aspect_ratio"] == "16:9"
