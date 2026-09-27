from __future__ import annotations

from types import SimpleNamespace

import pytest

from photo_edit_studio.models.diffusers_adapters import _progress_kwargs
from photo_edit_studio.progress import GenerationProgress


def test_generation_progress_scales_steps_across_images() -> None:
    events: list[tuple[float | None, str]] = []
    progress = GenerationProgress(lambda fraction, message: events.append((fraction, message)))
    progress.inference(0, 2, 2, 4)
    progress.inference(1, 2, 4, 4)
    assert [fraction for fraction, _ in events] == pytest.approx([0.325, 0.85])
    assert [message for _, message in events] == [
        "Generating image 1/2 · step 2/4",
        "Generating image 2/2 · step 4/4",
    ]


def test_step_callback_does_not_request_latent_tensors() -> None:
    events: list[tuple[float | None, str]] = []
    progress = GenerationProgress(lambda fraction, message: events.append((fraction, message)))

    class Pipeline:
        def __call__(self, callback_on_step_end=None, callback_on_step_end_tensor_inputs=None):
            return None

    kwargs = _progress_kwargs(Pipeline(), progress, 0, 1, 4)
    assert kwargs["callback_on_step_end_tensor_inputs"] == []
    tensors = {"latents": SimpleNamespace()}
    assert kwargs["callback_on_step_end"](None, 0, None, tensors) is tensors
    assert events[0][0] == pytest.approx(0.325)
    assert events[0][1] == "Generating image 1/1 · step 1/4"


def test_callback_is_optional_for_older_pipeline() -> None:
    class Pipeline:
        def __call__(self, prompt: str):
            return prompt

    assert _progress_kwargs(Pipeline(), GenerationProgress(), 0, 1, 4) == {}