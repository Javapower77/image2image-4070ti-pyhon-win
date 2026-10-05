from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from photo_edit_studio.models.diffusers_adapters import FluxKleinAdapter
from photo_edit_studio.models.registry import MODEL_SPECS
from photo_edit_studio.types import GenerationRequest


def _request(workflow: str, images: list[Image.Image]) -> GenerationRequest:
    return GenerationRequest(
        model_key="flux-klein-4b", prompt="a red fox", negative_prompt="",
        images=images, mask=None, width=1024, height=576, steps=4,
        guidance=1.0, true_cfg=1.0, strength=0.8, seed=2,
        workflow=workflow,
    )


def test_flux_text_does_not_pass_any_image_to_pipeline(monkeypatch) -> None:
    from photo_edit_studio.models import diffusers_adapters

    calls: list[dict] = []

    class FakePipe:
        def __call__(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(images=[Image.new("RGB", (64, 64))])

    class FakeGenerator:
        def __init__(self, device: str) -> None:
            self.device = device

        def manual_seed(self, seed: int):
            return self

    monkeypatch.setattr(
        diffusers_adapters, "_runtime",
        lambda: (None, SimpleNamespace(Generator=FakeGenerator, no_grad=lambda: __import__("contextlib").nullcontext())),
    )
    adapter = FluxKleinAdapter(MODEL_SPECS["flux-klein-4b"])
    adapter.pipe = FakePipe()
    adapter.generate(_request("text", []))
    assert "image" not in calls[0] and "images" not in calls[0]
    assert (calls[0]["width"], calls[0]["height"]) == (1024, 576)
    adapter.generate(_request("standard", [Image.new("RGB", (64, 64))]))
    assert calls[1]["image"].size == (64, 64)


def test_flux_gguf_load_uses_explicit_local_config_and_absolute_weights(monkeypatch, tmp_path):
    from photo_edit_studio.models import diffusers_adapters

    monkeypatch.chdir(tmp_path)
    encoder = Path("models/qwen3-4b-abl-q4_0.gguf")
    encoder.parent.mkdir()
    encoder.write_bytes(b"GGUF")
    snapshot = Path("models/repos/klein")
    calls = {}
    config = object()
    text_encoder = object()
    pipe = object()

    def load_config(path, **kwargs):
        calls["config"] = (path, kwargs)
        return config

    def load_encoder(path, **kwargs):
        calls["encoder"] = (path, kwargs)
        assert kwargs["config"] is config
        assert Path(kwargs["gguf_file"]).is_absolute()
        assert Path(kwargs["gguf_file"]).is_file()
        assert "safety_checker" not in kwargs
        return text_encoder

    def load_pipe(path, **kwargs):
        calls["pipe"] = (path, kwargs)
        return pipe

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        Qwen3Config=SimpleNamespace(from_pretrained=load_config),
        Qwen3ForCausalLM=SimpleNamespace(from_pretrained=load_encoder),
    ))
    monkeypatch.setattr(diffusers_adapters.settings, "flux_klein_4b_text_encoder",
                        Path("models/qwen3-4b-alb-q4_0.gguf"))
    monkeypatch.setattr(diffusers_adapters, "_runtime", lambda: (
        SimpleNamespace(Flux2KleinPipeline=SimpleNamespace(from_pretrained=load_pipe)),
        SimpleNamespace(bfloat16="bf16"),
    ))
    monkeypatch.setattr(diffusers_adapters, "_local_path", lambda path, label: str(path))
    adapter = FluxKleinAdapter(SimpleNamespace(local_path=snapshot, label="Klein"))
    monkeypatch.setattr(adapter, "_validate_hardware", lambda: None)
    monkeypatch.setattr(adapter, "configure_memory", lambda: None)
    adapter.load()
    assert calls["config"] == (str((snapshot / "text_encoder").resolve()),
                               {"local_files_only": True})
    assert calls["encoder"][1]["gguf_file"] == str(encoder.resolve())
    assert calls["pipe"][1]["text_encoder"] is text_encoder
    assert adapter.pipe is pipe