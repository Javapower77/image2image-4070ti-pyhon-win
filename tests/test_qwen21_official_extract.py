from __future__ import annotations

import hashlib
import json
from contextlib import nullcontext
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar
from unittest.mock import Mock

import numpy as np
import pytest
from safetensors.numpy import save
from test_qwen21_official_adapter import SIGMAS, Generator, Pipe, request

from photo_edit_studio.models import diffusers_adapters as adapters
from photo_edit_studio.models import qwen21_official_extract as assets
from photo_edit_studio.models.base import ModelAdapter
from photo_edit_studio.models.registry import MODEL_SPECS
from photo_edit_studio.types import LoraSpec

KEY = "qwen-2.1-turbo-official-extract"
FILENAME = "qwen_image_2.1_turbo_lora_avg_rank_178_bf16.safetensors"
SHA256 = "208DD43250E1E01467BA190572AE2E7107A7870EC1FF026BC65F791FA1E80E95"


def test_independent_exact_asset_pin_and_shared_turbo_not_original_base(monkeypatch, tmp_path):
    assert assets.QWEN21_OFFICIAL_EXTRACT_KEY == KEY
    assert assets.QWEN21_OFFICIAL_EXTRACT_LORA == {
        "version_id": 3394831, "file_id": 3284648, "filename": FILENAME,
        "url": "https://civitai.com/api/download/models/3394831?fileId=3284648",
        "size": 913314512, "sha256": SHA256.lower(),
    }
    monkeypatch.setattr(assets.settings, "lora_dir", tmp_path)
    assert assets.qwen21_official_extract_path() == tmp_path / "qwen21-official" / FILENAME
    spec = MODEL_SPECS[KEY]
    assert spec.repo_id == "Qwen/Qwen-Image-2.1-Turbo"
    assert spec.local_path == MODEL_SPECS["qwen-2.1-turbo-official"].local_path
    assert spec.loader == "qwen21_official_extract" and spec.family == "qwen21-official"
    assert issubclass(adapters.Qwen21OfficialExtractAdapter, adapters.Qwen21OfficialAdapter)


@pytest.fixture
def tiny_pin(monkeypatch):
    # Real size/hash/header checks on tiny local bytes; never allocate a 913 MB fixture.
    payload = save({"test": np.zeros(1, dtype=np.float32)})
    pin = dict(assets.QWEN21_OFFICIAL_EXTRACT_LORA,
               size=len(payload), sha256=hashlib.sha256(payload).hexdigest())
    monkeypatch.setattr(assets, "QWEN21_OFFICIAL_EXTRACT_LORA", pin)
    return payload, pin


@pytest.fixture
def extracted(monkeypatch, tmp_path):
    monkeypatch.setattr(assets.settings, "model_dir", tmp_path / "models")
    monkeypatch.setattr(assets.settings, "lora_dir", tmp_path / "loras")
    path = MODEL_SPECS[KEY].local_path
    (path / "transformer").mkdir(parents=True)
    (path / "model_index.json").write_text(json.dumps({
        "_class_name": "QwenImage21Pipeline", "sample_sigmas": SIGMAS,
    }), encoding="utf-8")
    (path / "transformer/config.json").write_text('{"causal_condition": true}', encoding="utf-8")
    events = []

    class LoraPipe(Pipe):
        loads: ClassVar[list] = []

        @classmethod
        def from_pretrained(cls, path, **kwargs):
            events.append("allocate full pipeline")
            return super().from_pretrained(path, **kwargs)

        def load_lora_weights(self, state_dict, *, adapter_name, local_files_only):
            assert isinstance(state_dict, dict)
            events.append(("load mandatory", state_dict, adapter_name, local_files_only))

        def set_adapters(self, names, *, adapter_weights):
            self.active = (names, adapter_weights)
            events.append(("activate", names, adapter_weights))

        def unload_lora_weights(self):
            pytest.fail("Mandatory adapter must not be unloaded by the optional-LoRA path")

    runtime = Mock(return_value=(SimpleNamespace(QwenImage21Pipeline=LoraPipe),
                               SimpleNamespace(bfloat16="bf16", Generator=Generator, no_grad=nullcontext)))
    monkeypatch.setattr(adapters, "_runtime", runtime)
    monkeypatch.setattr(adapters, "_supported_call", lambda *_args: pytest.fail("No kwarg filtering"))
    # These high-level tests isolate the hook; real tensors/conversion are tested
    # independently in test_qwen21_extracted_weights.py, not with header-only bytes.
    converted = {"transformer.img_in.lora_A.weight": object(),
                 "transformer.img_in.lora_B.weight": object()}

    def apply_mixed(pipe, path):
        assert path == assets.qwen21_official_extract_path()
        assert "memory" not in events
        events.append("prepare mixed weights")
        pipe.load_lora_weights(converted, adapter_name="official_extract", local_files_only=True)
        pipe.set_adapters(["official_extract"], adapter_weights=[1.0])

    monkeypatch.setattr(adapters, "apply_qwen21_extracted_weights", apply_mixed)
    instance = adapters.Qwen21OfficialExtractAdapter(MODEL_SPECS[KEY])
    monkeypatch.setattr(instance, "_validate_hardware", lambda: events.append("hardware"))
    monkeypatch.setattr(instance, "configure_memory", lambda: events.append("memory"))
    return instance, LoraPipe, runtime, events


@pytest.mark.parametrize("failure,match", [
    ("missing", "missing"), ("size", "size mismatch"), ("hash", "SHA256 mismatch"),
    ("header", ".*"), ("empty", "empty Safetensors header"),
])
def test_real_mandatory_validation_fails_before_runtime_hardware_and_allocation(
    extracted, tiny_pin, failure, match,
):
    instance, pipe, runtime, events = extracted
    payload, pin = tiny_pin
    path = assets.qwen21_official_extract_path()
    path.parent.mkdir(parents=True)
    if failure != "missing":
        content = {"size": payload[:-1], "hash": bytes([payload[0] ^ 1]) + payload[1:],
                   "header": b"not safetensors", "empty": save({})}.get(failure, payload)
        path.write_bytes(content)
        if failure in {"header", "empty"}:
            pin.update(size=len(content), sha256=hashlib.sha256(content).hexdigest())
    with pytest.raises(Exception, match=match):
        instance.load()
    runtime.assert_not_called()
    assert events == [] and pipe.loads == [] and instance.pipe is None


def test_header_check_is_cpu_keys_only_no_tensor_materialization(monkeypatch, tmp_path, tiny_pin):
    import safetensors

    payload, _pin = tiny_pin
    path = tmp_path / FILENAME
    path.write_bytes(payload)
    handle = Mock()
    handle.keys.return_value = ["test"]
    context = Mock()
    context.__enter__ = Mock(return_value=handle)
    context.__exit__ = Mock(return_value=False)
    opened = Mock(return_value=context)
    monkeypatch.setattr(safetensors, "safe_open", opened)
    assets.validate_qwen21_official_extract_lora(path)
    opened.assert_called_once_with(path, framework="pt", device="cpu")
    handle.keys.assert_called_once_with()
    handle.get_tensor.assert_not_called()
    handle.get_slice.assert_not_called()


@pytest.mark.parametrize("workflow,image_count", [("standard", 1), ("standard", 3), ("text", 0)])
def test_full_pipeline_mandatory_weight_one_retained_across_calls_without_schedule_override(
    extracted, tiny_pin, monkeypatch, workflow, image_count,
):
    instance, pipe_class, _runtime, events = extracted
    path = assets.qwen21_official_extract_path()
    path.parent.mkdir(parents=True)
    path.write_bytes(tiny_pin[0])
    validate = Mock(side_effect=assets.validate_qwen21_official_extract_lora)
    monkeypatch.setattr(adapters, "validate_qwen21_official_extract_lora", validate)
    base_apply = Mock(side_effect=AssertionError("Generic optional path must not run"))
    monkeypatch.setattr(ModelAdapter, "apply_loras", base_apply)
    req = request(model_key=KEY, workflow=workflow)
    req.images = req.images * image_count
    before = (instance.spec.local_path / "model_index.json").read_bytes()
    instance.generate(req)
    pipe = instance.pipe
    instance._lora_signature = (("stale optional adapter", 0.5),)
    instance.apply_loras(req)
    instance.generate(req)
    assert instance.pipe is pipe and pipe.active == (["official_extract"], [1.0])
    validate.assert_called_once_with(path)
    base_apply.assert_not_called()
    assert events[:3] == ["hardware", "allocate full pipeline", "prepare mixed weights"]
    assert events[3][0] == "load mandatory"
    assert set(events[3][1]) == {"transformer.img_in.lora_A.weight",
                                "transformer.img_in.lora_B.weight"}
    assert events[3][2:] == ("official_extract", True)
    assert events[4:] == [("activate", ["official_extract"], [1.0]), "memory"]
    assert pipe_class.loads == [(str(instance.spec.local_path), {
        "torch_dtype": "bf16", "local_files_only": True, "low_cpu_mem_usage": True,
    })]
    assert pipe.config.sample_sigmas == SIGMAS
    assert (instance.spec.local_path / "model_index.json").read_bytes() == before
    assert len(pipe.calls) == 2
    for call in pipe.calls:
        assert call["num_inference_steps"] == 8 and call["true_cfg_scale"] == 1
        assert call["use_kv_cache"] is True and call["prompt"] == req.prompt
        assert call["image"] is (None if workflow == "text" else req.images)
        assert not {"sigmas", "sample_sigmas", "timesteps", "guidance_scale", "negative_prompt", "strength"}.intersection(call)


@pytest.mark.parametrize("field,value,match", [
    ("loras", [LoraSpec("style", Path("never-read.safetensors"), 1, "style")], "LoRAs"),
    ("negative_prompt", "blur", "empty negative"), ("guidance", -1, "guidance 1"),
    ("true_cfg", -1, "True CFG 1"), ("true_cfg", 2, "True CFG 1"),
    ("steps", 6, "eight"), ("workflow", "swap", "no swaps"),
    ("swap_kind", "Body", "no swaps"), ("dlss", {"enabled": True}, "DLSS"),
    ("dlss", {"enabled": False}, "DLSS"),
    ("model_key", "qwen-2.1-turbo-official", "key"),
])
@pytest.mark.parametrize("method", ["generate", "apply_loras"])
def test_illegal_requests_rejected_unchanged_before_load(extracted, monkeypatch, field, value, match, method):
    instance, _pipe, runtime, events = extracted
    req = request(model_key=KEY, **{field: value}) if field != "model_key" else request(model_key=value)
    before = {field.name: getattr(req, field.name) for field in fields(req)}
    monkeypatch.setattr(instance, "load", lambda: pytest.fail("Invalid request loaded weights"))
    with pytest.raises(ValueError, match=match):
        getattr(instance, method)(req)
    assert {field.name: getattr(req, field.name) for field in fields(req)} == before
    assert instance.pipe is None and events == []
    runtime.assert_not_called()


@pytest.mark.parametrize("api", ["load_lora_weights", "set_adapters"])
def test_missing_lora_api_rejected_before_checkpoint_allocation(extracted, monkeypatch, api):
    instance, pipe, _runtime, events = extracted
    monkeypatch.setattr(adapters, "validate_qwen21_official_extract_lora", lambda _path: None)
    monkeypatch.setattr(pipe, api, None)
    with pytest.raises(RuntimeError, match="LoRA-capable"):
        instance.load()
    assert pipe.loads == [] and events == [] and instance.pipe is None


@pytest.mark.parametrize("stage", ["configure_memory", "load_lora_weights", "set_adapters"])
def test_mandatory_load_or_activation_failure_releases_entire_pipeline(extracted, monkeypatch, stage):
    instance, pipe, _runtime, _events = extracted
    monkeypatch.setattr(adapters, "validate_qwen21_official_extract_lora", lambda _path: None)

    def fail(*_args, **_kwargs):
        raise RuntimeError("mock stage failed")

    monkeypatch.setattr(instance if stage == "configure_memory" else pipe, stage, fail)
    with pytest.raises(RuntimeError, match="mock stage failed"):
        instance.load()
    assert instance.pipe is None


def test_loaded_sigma_drift_rejects_inference_without_changing_mandatory(extracted, monkeypatch):
    instance, _pipe, _runtime, _events = extracted
    monkeypatch.setattr(adapters, "validate_qwen21_official_extract_lora", lambda _path: None)
    instance.load()
    instance.pipe.config.sample_sigmas = SIGMAS[:-1]
    with pytest.raises(RuntimeError, match="sample_sigmas changed"):
        instance.generate(request(model_key=KEY))
    assert instance.pipe.calls == [] and instance.pipe.active == (["official_extract"], [1.0])