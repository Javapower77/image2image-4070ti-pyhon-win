from __future__ import annotations

import inspect
import json
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest
from PIL import Image

from photo_edit_studio.models import diffusers_adapters as adapters
from photo_edit_studio.models.registry import MODEL_SPECS
from photo_edit_studio.progress import GenerationProgress
from photo_edit_studio.types import GenerationRequest, LoraSpec

KEY = "qwen-2.1-turbo-official"
SIGMAS = [1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568]


def request(**changes: object) -> GenerationRequest:
    req = GenerationRequest(
        model_key=KEY, prompt="Keep this exact prompt", negative_prompt="",
        images=[Image.new("RGB", (64, 64))], mask=None, width=512, height=768,
        steps=8, guidance=1, true_cfg=1, strength=0.8, seed=37,
    )
    return replace(req, **changes)


class Generator:
    def __init__(self, *, device: str) -> None:
        self.device = device

    def manual_seed(self, seed: int) -> Generator:
        self.seed = seed
        return self


class Pipe:
    """Explicit upstream contract; no permissive **kwargs can mask missing APIs."""

    loads: ClassVar[list[tuple[str, dict]]] = []

    def __init__(self, sample_sigmas: list[float]) -> None:
        self.config = SimpleNamespace(sample_sigmas=list(sample_sigmas))
        self.transformer = SimpleNamespace(config=SimpleNamespace(causal_condition=True))
        self.calls: list[dict] = []

    @classmethod
    def from_pretrained(cls, path: str, **kwargs: object) -> Pipe:
        cls.loads.append((path, kwargs))
        return cls(SIGMAS)

    def __call__(
        self, *, prompt, width, height, num_inference_steps, true_cfg_scale,
        use_kv_cache, generator, num_images_per_prompt, image=None,
        callback_on_step_end=None, callback_on_step_end_tensor_inputs=None,
    ) -> SimpleNamespace:
        self.calls.append({
            "prompt": prompt, "width": width, "height": height,
            "num_inference_steps": num_inference_steps, "true_cfg_scale": true_cfg_scale,
            "use_kv_cache": use_kv_cache, "generator": generator,
            "num_images_per_prompt": num_images_per_prompt, "image": image,
        })
        if callback_on_step_end is not None:
            assert callback_on_step_end_tensor_inputs == []
            tensors = {"sentinel": object()}
            for step in range(num_inference_steps):
                assert callback_on_step_end(self, step, 100 - step, tensors) is tensors
        return SimpleNamespace(images=[Image.new("RGB", (width, height))])


@pytest.fixture
def adapter(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> adapters.Qwen21OfficialAdapter:
    monkeypatch.setattr(adapters.settings, "model_dir", tmp_path)
    tmp_path = MODEL_SPECS[KEY].local_path
    (tmp_path / "transformer").mkdir(parents=True)
    (tmp_path / "model_index.json").write_text(json.dumps({
        "_class_name": "QwenImage21Pipeline", "sample_sigmas": SIGMAS,
    }), encoding="utf-8")
    (tmp_path / "transformer" / "config.json").write_text(
        json.dumps({"causal_condition": True}), encoding="utf-8",
    )
    Pipe.loads = []
    monkeypatch.setattr(adapters, "_runtime", lambda: (
        SimpleNamespace(QwenImage21Pipeline=Pipe),
        SimpleNamespace(bfloat16="bf16", Generator=Generator, no_grad=nullcontext),
    ))
    instance = adapters.Qwen21OfficialAdapter(MODEL_SPECS[KEY])
    monkeypatch.setattr(instance, "_validate_hardware", lambda: None)
    monkeypatch.setattr(instance, "configure_memory", lambda: None)
    monkeypatch.setattr(instance, "apply_loras", lambda *_args: pytest.fail("No LoRA application"))
    monkeypatch.setattr(adapters, "_supported_call", lambda *_args: pytest.fail("No kwarg filtering"))
    return instance


def test_load_uses_saved_config_bf16_local_only_without_scheduler_overrides(adapter) -> None:
    before = (adapter.spec.local_path / "model_index.json").read_bytes()
    adapter.load()
    assert Pipe.loads == [(str(adapter.spec.local_path), {
        "torch_dtype": "bf16", "local_files_only": True, "low_cpu_mem_usage": True,
    })]
    assert adapter.pipe.config.sample_sigmas == SIGMAS
    assert adapter.pipe.transformer.config.causal_condition is True
    assert (adapter.spec.local_path / "model_index.json").read_bytes() == before


def test_missing_pipeline_fails_with_upgrade_message_before_hardware_or_weights(
    adapter, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(adapters, "_runtime", lambda: (SimpleNamespace(), None))
    monkeypatch.setattr(adapter, "_validate_hardware", lambda: pytest.fail("Hardware touched"))
    with pytest.raises(RuntimeError, match=r"PR #14950.*Transformers >=5.17,<6"):
        adapter.load()
    assert Pipe.loads == [] and adapter.pipe is None


def test_constructor_without_saved_sigmas_rejected_before_loading(adapter, monkeypatch) -> None:
    class OldPipe(Pipe):
        def __init__(self, **kwargs):
            pytest.fail("Constructor must not run")

    monkeypatch.setattr(adapters, "_runtime", lambda: (
        SimpleNamespace(QwenImage21Pipeline=OldPipe), None,
    ))
    with pytest.raises(RuntimeError, match="PR #14950"):
        adapter.load()
    assert Pipe.loads == []


@pytest.mark.parametrize("missing", [
    "prompt", "image", "width", "height", "num_inference_steps", "true_cfg_scale",
    "use_kv_cache", "generator", "num_images_per_prompt", "callback_on_step_end",
    "callback_on_step_end_tensor_inputs",
])
def test_every_essential_call_parameter_is_required_before_loading(adapter, monkeypatch, missing):
    class IncompatiblePipe(Pipe):
        def __call__(self, **kwargs):
            pytest.fail("Incompatible pipeline must not run")

    signature = inspect.signature(Pipe.__call__)
    # Expose an API missing exactly one essential parameter, even with **kwargs.
    IncompatiblePipe.__call__.__signature__ = signature.replace(parameters=[
        parameter for name, parameter in signature.parameters.items() if name != missing
    ] + [inspect.Parameter("kwargs", inspect.Parameter.VAR_KEYWORD)])
    monkeypatch.setattr(adapters, "_runtime", lambda: (
        SimpleNamespace(QwenImage21Pipeline=IncompatiblePipe), None,
    ))
    with pytest.raises(RuntimeError, match="PR #14950"):
        adapter.load()
    assert Pipe.loads == []


@pytest.mark.parametrize("filename,payload,match", [
    ("model_index.json", {"_class_name": "QwenImage21Pipeline", "sample_sigmas": SIGMAS[:6]}, "exact publisher eight"),
    ("model_index.json", {"_class_name": "QwenImage21Pipeline", "sample_sigmas": SIGMAS[:-1] + [0.414569]}, "exact publisher eight"),
    ("model_index.json", {"_class_name": "QwenImageEditPipeline", "sample_sigmas": SIGMAS}, "exact publisher eight"),
    ("model_index.json", {"_class_name": "QwenImage21Pipeline"}, "exact publisher eight"),
    ("transformer/config.json", {"causal_condition": False}, "causal_condition=true"),
    ("transformer/config.json", {"causal_condition": 1}, "causal_condition=true"),
])
def test_exact_eight_sigma_and_causal_config_fail_before_hardware_and_weights(
    adapter, monkeypatch, filename, payload, match,
) -> None:
    (adapter.spec.local_path / filename).write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(adapter, "_validate_hardware", lambda: pytest.fail("Hardware touched"))
    with pytest.raises(RuntimeError, match=match):
        adapter.load()
    assert Pipe.loads == []


@pytest.mark.parametrize("filename", ["model_index.json", "transformer/config.json"])
def test_missing_config_fails_before_loading(adapter, filename) -> None:
    (adapter.spec.local_path / filename).unlink()
    with pytest.raises(RuntimeError, match="requires model_index.json"):
        adapter.load()
    assert Pipe.loads == []


@pytest.mark.parametrize("field,value,match", [
    ("steps", 6, "eight"), ("steps", 8.0, "eight"), ("steps", True, "eight"),
    ("guidance", 2, "guidance 1"), ("true_cfg", 2, "True CFG 1"),
    ("negative_prompt", "blur", "empty negative"),
    ("loras", [LoraSpec("style", Path("unused.safetensors"), 1, "style")], "LoRAs"),
    ("dlss", {"enabled": True}, "DLSS"), ("dlss", {"enabled": False}, "DLSS"),
    ("workflow", "swap", "exactly two ordered images"), ("swap_kind", "Head", "no swaps"),
    ("images", [], "one to three"), ("images", [Image.new("RGB", (8, 8))] * 4, "one to three"),
])
def test_unsupported_request_fails_before_load(adapter, monkeypatch, field, value, match):
    monkeypatch.setattr(adapter, "load", lambda: pytest.fail("Invalid request loaded weights"))
    with pytest.raises(ValueError, match=match):
        adapter.generate(request(**{field: value}))


def test_text_with_images_rejected_before_load(adapter, monkeypatch) -> None:
    monkeypatch.setattr(adapter, "load", lambda: pytest.fail("Invalid text loaded weights"))
    with pytest.raises(ValueError, match="does not accept images"):
        adapter.generate(request(workflow="text"))


@pytest.mark.parametrize("workflow,count_images", [("text", 0), ("standard", 1), ("standard", 3)])
def test_explicit_call_contract_cpu_seeds_ordered_edits_and_batch_progress(
    adapter, workflow, count_images,
) -> None:
    inputs = [Image.new("RGB", (64, 64), color) for color in ("red", "green", "blue")][:count_images]
    req = request(workflow=workflow, images=inputs, compose=count_images == 3, count=2)
    events: list[tuple[float | None, str]] = []
    result = adapter.generate(req, GenerationProgress(lambda fraction, message: events.append((fraction, message))))
    assert [image.size for image in result] == [(512, 768)] * 2
    assert len(adapter.pipe.calls) == 2
    for index, call in enumerate(adapter.pipe.calls):
        assert call["prompt"] == req.prompt
        assert (call["width"], call["height"], call["num_inference_steps"]) == (512, 768, 8)
        assert call["true_cfg_scale"] == 1.0 and call["use_kv_cache"] is True
        assert call["num_images_per_prompt"] == 1
        assert (call["generator"].device, call["generator"].seed) == ("cpu", 37 + index)
        assert call["image"] is (None if workflow == "text" else req.images)
    assert events[0][0] == 0.05 and "Loading" in events[0][1]
    assert [message for _, message in events[1:]] == [
        f"Generating image {index + 1}/2 · step {step}/8"
        for index in range(2) for step in range(9)
    ]
    assert events[-1][0] == pytest.approx(0.85)


@pytest.mark.parametrize("target,value,match", [
    ("sigmas", SIGMAS[:6], "sample_sigmas changed"),
    ("causal", False, "configuration changed"),
])
def test_loaded_configuration_drift_refuses_inference(adapter, target, value, match):
    adapter.load()
    if target == "sigmas":
        adapter.pipe.config.sample_sigmas = value
    else:
        adapter.pipe.transformer.config.causal_condition = value
    with pytest.raises(RuntimeError, match=match):
        adapter.generate(request())
    assert adapter.pipe.calls == []


def test_memory_configuration_failure_unloads_adapter(adapter, monkeypatch) -> None:
    def fail() -> None:
        raise RuntimeError("offload failed")

    monkeypatch.setattr(adapter, "configure_memory", fail)
    with pytest.raises(RuntimeError, match="offload failed"):
        adapter.load()
    assert adapter.pipe is None