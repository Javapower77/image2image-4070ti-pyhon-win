from __future__ import annotations

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