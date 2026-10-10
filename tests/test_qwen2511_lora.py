"""Tiny real-Torch CPU contracts; never read installed checkpoint payloads."""
from __future__ import annotations

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from photo_edit_studio.models.diffusers_adapters import QwenAdapter
from photo_edit_studio.models.qwen2511_lora import (
    QwenDirectState,
    parse_qwen_lora,
    qwen_parameters,
    validate_qwen_lora,
)
from photo_edit_studio.models.registry import MODEL_SPECS
from photo_edit_studio.swap import swap_profile
from photo_edit_studio.types import GenerationRequest, LoraSpec


class TinyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 2, dtype=torch.float64)
        self.norm = nn.LayerNorm(2, dtype=torch.float64)
        with torch.no_grad():
            self.linear.weight.copy_(torch.arange(6).reshape(2, 3) / 8)
            self.linear.bias.copy_(torch.tensor([0.25, -0.5]))
            self.norm.weight.copy_(torch.tensor([1.25, 0.75]))


def payload(*, alpha=None):
    tensors = {
        "diffusion_model.linear.lora_down.weight": torch.tensor(
            [[1., 2., 3.], [-1., 0., 2.]], dtype=torch.float64,
        ),
        "diffusion_model.linear.lora_up.weight": torch.tensor(
            [[2., -1.], [0.5, 3.]], dtype=torch.float64,
        ),
        "diffusion_model.linear.diff_b": torch.tensor([0.5, -0.25], dtype=torch.float64),
        "diffusion_model.norm.diff": torch.tensor([0.125, -0.5], dtype=torch.float64),
    }
    if alpha is not None:
        tensors["diffusion_model.linear.alpha"] = torch.tensor(alpha)
    return tensors


def write_weights(tmp_path, tensors, name="adapter"):
    path = tmp_path / f"{name}.safetensors"
    save_file(tensors, str(path))
    return path


def request(loras=()):
    return GenerationRequest(
        model_key="qwen-2511", prompt="", negative_prompt="", images=[], mask=None,
        width=64, height=64, steps=1, guidance=1, true_cfg=1, strength=1, seed=0,
        loras=list(loras),
    )


def snapshot(transformer):
    return {key: value.detach().clone() for key, value in qwen_parameters(transformer).items()}


def assert_pristine(transformer, pristine):
    for key, value in qwen_parameters(transformer).items():
        assert torch.equal(value, pristine[key]), key


class WrappedLinear(nn.Module):
    """Model only PEFT's canonical base_layer naming, not PEFT implementation."""
    def __init__(self, base):
        super().__init__()
        self.base_layer = base


class Pipe:
    """Explicit fake pipeline boundary with real, small CPU parameters."""
    def __init__(self):
        self.transformer = TinyTransformer()
        self.events = []
        self.loaded = {}
        self.active = []

    def remove_all_hooks(self):
        self.events.append("remove_hooks")

    def unload_lora_weights(self):
        self.events.append("unload")
        if isinstance(self.transformer.linear, WrappedLinear):
            self.transformer.linear = self.transformer.linear.base_layer
        self.loaded.clear()
        self.active.clear()

    def load_lora_weights(self, weights, *, adapter_name, local_files_only):
        assert local_files_only is True
        assert isinstance(weights, dict)  # Never hand the mixed file to Diffusers.
        assert all(".lora_" in key for key in weights)
        self.events.append(f"load:{adapter_name}")
        self.loaded[adapter_name] = weights
        if not isinstance(self.transformer.linear, WrappedLinear):
            self.transformer.linear = WrappedLinear(self.transformer.linear)

    def set_adapters(self, names, *, adapter_weights):
        self.events.append("set_adapters")
        self.active = list(zip(names, adapter_weights, strict=True))


def make_adapter(monkeypatch, pipe=None):
    adapter = QwenAdapter(MODEL_SPECS["qwen-2511"])
    adapter.pipe = pipe or Pipe()
    monkeypatch.setattr(adapter, "configure_memory", lambda: adapter.pipe.events.append("memory"))
    return adapter


@pytest.mark.parametrize("alpha,scale", [(None, 1.), (2, 1.), (8, 4.), (0, 0.), (-2, -1.)])
@pytest.mark.parametrize("normalized", [False, True])
def test_actual_rank_alpha_algebra_and_source_immutability(tmp_path, alpha, scale, normalized):
    tensors = payload(alpha=alpha)
    if normalized:
        tensors = {key.replace("lora_down", "lora_A").replace("lora_up", "lora_B"): value
                   for key, value in tensors.items()}
    path = write_weights(tmp_path, tensors)
    before = path.read_bytes()  # Synthetic file only.
    weights = parse_qwen_lora(path)
    a = tensors[next(key for key in tensors if key.endswith(("down.weight", "A.weight")))]
    b = tensors[next(key for key in tensors if key.endswith(("up.weight", "B.weight")))]
    assert set(weights.matrices) == {
        "transformer.linear.lora_A.weight", "transformer.linear.lora_B.weight",
    }
    assert torch.equal(weights.matrices["transformer.linear.lora_A.weight"], a)
    assert torch.equal(weights.matrices["transformer.linear.lora_B.weight"] @ a, (b @ a) * scale)
    assert torch.equal(weights.deltas["linear.bias"], tensors["diffusion_model.linear.diff_b"])
    assert torch.equal(weights.deltas["norm.weight"], tensors["diffusion_model.norm.diff"])
    assert path.read_bytes() == before


@pytest.mark.parametrize("strength", [-2., -0.5, 0., 1., 2.])
def test_signed_strength_applied_once_matrices_and_direct_bias_weight(tmp_path, monkeypatch, strength):
    source = payload(alpha=8)
    path = write_weights(tmp_path, source)
    adapter = make_adapter(monkeypatch)
    pipe = adapter.pipe
    pristine = snapshot(pipe.transformer)
    adapter.apply_loras(request([LoraSpec("mixed", path, strength, "mixed")]))
    assert pipe.active == [("mixed", strength)]
    weights = pipe.loaded["mixed"]
    # Loader receives intrinsic alpha/rank only; request strength stays separate.
    a = source["diffusion_model.linear.lora_down.weight"]
    b = source["diffusion_model.linear.lora_up.weight"]
    assert torch.equal(weights["transformer.linear.lora_B.weight"] @ a, 4 * (b @ a))
    parameters = qwen_parameters(pipe.transformer)
    assert torch.equal(parameters["linear.weight"], pristine["linear.weight"])
    assert torch.equal(parameters["linear.bias"],
                       pristine["linear.bias"] + strength * source["diffusion_model.linear.diff_b"])
    assert torch.equal(parameters["norm.weight"],
                       pristine["norm.weight"] + strength * source["diffusion_model.norm.diff"])
    assert torch.equal(parameters["norm.bias"], pristine["norm.bias"])
    assert pipe.events == ["remove_hooks", "unload", "load:mixed", "set_adapters", "memory"]


def test_overlapping_deltas_switch_weights_files_and_deselect_restore_exactly(tmp_path, monkeypatch):
    first = write_weights(tmp_path, payload(), "first")
    second_source = {
        "linear.diff_b": torch.tensor([-0.125, 0.5], dtype=torch.float64),
        "norm.diff": torch.tensor([0.25, 0.125], dtype=torch.float64),
    }
    second = write_weights(tmp_path, second_source, "second")
    adapter = make_adapter(monkeypatch)
    pipe = adapter.pipe
    pristine = snapshot(pipe.transformer)
    for strengths in [(1., -0.5), (-2., 2.), (0.25, -1.), (1., -0.5)]:
        req = request([LoraSpec("first", first, strengths[0], "first"),
                       LoraSpec("second", second, strengths[1], "second")])
        adapter.apply_loras(req)
        expected_bias = (pristine["linear.bias"]
                         + strengths[0] * payload()["diffusion_model.linear.diff_b"]
                         + strengths[1] * second_source["linear.diff_b"])
        expected_norm = (pristine["norm.weight"]
                         + strengths[0] * payload()["diffusion_model.norm.diff"]
                         + strengths[1] * second_source["norm.diff"])
        assert torch.equal(qwen_parameters(pipe.transformer)["linear.bias"], expected_bias)
        assert torch.equal(qwen_parameters(pipe.transformer)["norm.weight"], expected_norm)
        events = list(pipe.events)
        before = snapshot(pipe.transformer)
        adapter.apply_loras(req)
        assert pipe.events == events
        assert_pristine(pipe.transformer, before)  # Signature reuse cannot compound.
    adapter.apply_loras(request([LoraSpec("second", second, 0.5, "second")]))
    assert not pipe.loaded and not pipe.active  # Direct-only never activates PEFT.
    assert torch.equal(pipe.transformer.linear.weight, pristine["linear.weight"])
    assert torch.equal(pipe.transformer.linear.bias,
                       pristine["linear.bias"] + 0.5 * second_source["linear.diff_b"])
    adapter.apply_loras(request())
    assert_pristine(pipe.transformer, pristine)
    assert adapter._direct_state.originals == {}
    assert adapter._direct_state.parameters == {}


@pytest.mark.parametrize("bad,match", [
    ({"unknown.weight": torch.ones(1)}, "Unknown"),
    ({"linear.alpha": torch.tensor(2)}, "Incomplete"),
    ({"linear.lora_down.weight": torch.ones(2, 3)}, "Incomplete"),
    ({"linear.lora_down.weight": torch.ones(2, 3),
      "linear.lora_up.weight": torch.ones(2, 3)}, "rank/shapes"),
    ({"missing.diff": torch.ones(2)}, "Unknown"),
    ({"linear.diff_b": torch.ones(3)}, "shape/dtype"),
    ({"linear.lora_down.weight": torch.ones(2, 4),
      "linear.lora_up.weight": torch.ones(2, 2)}, "shape/dtype"),
    ({"norm.lora_down.weight": torch.ones(1, 2),
      "norm.lora_up.weight": torch.ones(2, 1)}, "shape/dtype"),
    ({"linear.diff_b": torch.tensor([float("nan"), 0.])}, "Nonfinite"),
])
def test_all_files_and_targets_validate_before_any_mutation(tmp_path, monkeypatch, bad, match):
    valid = write_weights(tmp_path, payload(), "valid")
    invalid = write_weights(tmp_path, bad, "invalid")
    adapter = make_adapter(monkeypatch)
    pipe = adapter.pipe
    pristine = snapshot(pipe.transformer)
    with pytest.raises(ValueError, match=match):
        adapter.apply_loras(request([LoraSpec("valid", valid, 1, "valid"),
                                    LoraSpec("invalid", invalid, 1, "invalid")]))
    assert_pristine(pipe.transformer, pristine)
    assert pipe.events == ["remove_hooks"]  # Not even unload/install/capture occurred.
    assert adapter.pipe is None  # Fail closed, do not expose the failed pipeline.
    assert adapter._lora_signature == ()
    assert adapter._direct_state.originals == {}


def test_failed_replacement_does_not_mutate_previous_selection(tmp_path, monkeypatch):
    good = write_weights(tmp_path, payload(), "good")
    bad = write_weights(tmp_path, {"missing.diff": torch.ones(2)}, "bad")
    adapter = make_adapter(monkeypatch)
    pipe = adapter.pipe
    adapter.apply_loras(request([LoraSpec("good", good, 1, "good")]))
    previous = snapshot(pipe.transformer)
    pipe.events.clear()
    with pytest.raises(ValueError, match="Unknown"):
        adapter.apply_loras(request([LoraSpec("bad", bad, 1, "bad")]))
    assert_pristine(pipe.transformer, previous)
    assert pipe.events == ["remove_hooks"]
    assert adapter.pipe is None


def test_prepare_checks_matrix_target_after_valid_direct_target_without_mutation(tmp_path):
    tensors = payload()
    tensors["diffusion_model.linear.lora_down.weight"] = torch.ones(2, 4)
    weights = parse_qwen_lora(write_weights(tmp_path, tensors))
    transformer = TinyTransformer()
    pristine = snapshot(transformer)
    state = QwenDirectState()
    with pytest.raises(ValueError, match="shape/dtype"):
        state.prepare(transformer, [(weights, 1.)])
    assert_pristine(transformer, pristine)
    assert state.originals == state.parameters == {}


@pytest.mark.parametrize("strength", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_strength_fails_before_capture(tmp_path, strength):
    weights = parse_qwen_lora(write_weights(tmp_path, payload()))
    transformer = TinyTransformer()
    before = snapshot(transformer)
    with pytest.raises(ValueError, match="finite"):
        QwenDirectState().prepare(transformer, [(weights, strength)])
    assert_pristine(transformer, before)


def test_restore_rebinds_replaced_parameters(tmp_path):
    transformer = TinyTransformer()
    pristine = snapshot(transformer)
    state = QwenDirectState()
    weights = parse_qwen_lora(write_weights(tmp_path, payload()))
    updates = state.prepare(transformer, [(weights, 1.)])
    state.capture(transformer, updates)
    state.apply(updates)
    stale = transformer.linear.bias
    transformer.linear.bias = nn.Parameter(stale.detach().clone())
    state.restore(transformer)
    assert_pristine(transformer, pristine)
    assert not torch.equal(stale, pristine["linear.bias"])
    assert state.originals == state.parameters == {}


def test_real_accelerate_meta_hooks_detach_before_validation_and_reoffload(tmp_path, monkeypatch):
    hooks = pytest.importorskip("accelerate.hooks")

    class OffloadPipe(Pipe):
        def remove_all_hooks(self):
            super().remove_all_hooks()
            hooks.remove_hook_from_module(self.transformer, recurse=True)
            assert not any(parameter.is_meta for parameter in self.transformer.parameters())

        def offload(self):
            self.events.append("memory")
            for module in (self.transformer.linear, self.transformer.norm):
                hooks.add_hook_to_module(module, hooks.AlignDevicesHook(
                    execution_device="cpu", offload=True,
                ))
            assert all(parameter.is_meta for parameter in self.transformer.parameters())

    pipe = OffloadPipe()
    adapter = make_adapter(monkeypatch, pipe)
    monkeypatch.setattr(adapter, "configure_memory", pipe.offload)
    pristine = snapshot(pipe.transformer)
    pipe.offload()
    path = write_weights(tmp_path, {"linear.diff_b": torch.tensor([0.5, -0.25],
                                                                 dtype=torch.float64)})
    weights = parse_qwen_lora(path)
    with pytest.raises(ValueError, match="meta"):
        validate_qwen_lora(pipe.transformer, weights)
    for strength in (1., -0.5, 2.):
        pipe.events.clear()
        adapter.apply_loras(request([LoraSpec("bias", path, strength, "bias")]))
        assert pipe.events == ["remove_hooks", "unload", "memory"]
        stored = pipe.transformer.linear._hf_hook.weights_map["bias"]
        assert torch.equal(stored, pristine["linear.bias"] + strength * weights.deltas["linear.bias"])
    adapter.apply_loras(request())
    pipe.remove_all_hooks()
    assert_pristine(pipe.transformer, pristine)


def test_qwen2511_bfs_is_head_only():
    assert swap_profile("qwen-2511", "Head").filename == (
        "bfs_head_v5_2511_merged_version_rank_16_fp16.safetensors"
    )
    with pytest.raises(ValueError, match="does not support body"):
        swap_profile("qwen-2511", "Body")


@pytest.mark.parametrize("alpha", [[1., 2.], float("nan"), float("inf")])
def test_invalid_alpha_rejected_without_source_changes(tmp_path, alpha):
    path = write_weights(tmp_path, payload(alpha=alpha))
    before = path.read_bytes()
    with pytest.raises(ValueError, match="scalar|Nonfinite"):
        parse_qwen_lora(path)
    assert path.read_bytes() == before


def test_duplicate_canonical_key_and_mixed_pair_fail(tmp_path):
    tensors = payload()
    tensors["transformer.linear.lora_down.weight"] = torch.ones(2, 3)
    with pytest.raises(ValueError, match="duplicate"):
        parse_qwen_lora(write_weights(tmp_path, tensors, "duplicate"))
    tensors = payload()
    tensors["linear.lora_A.weight"] = torch.ones(2, 3)
    with pytest.raises(ValueError, match="mixed"):
        parse_qwen_lora(write_weights(tmp_path, tensors, "mixed"))


def test_matrix_requires_linear_even_when_parameter_shape_matches(tmp_path):
    transformer = TinyTransformer()
    transformer.linear = nn.Module()
    transformer.linear.weight = nn.Parameter(torch.ones(2, 3))
    tensors = {key: value for key, value in payload().items() if ".lora_" in key}
    weights = parse_qwen_lora(write_weights(tmp_path, tensors))
    with pytest.raises(ValueError, match="unquantized Linear"):
        validate_qwen_lora(transformer, weights)


@pytest.mark.parametrize("stage", ["load_lora_weights", "set_adapters", "configure_memory"])
def test_partial_install_or_memory_failure_disposes_pipeline(tmp_path, monkeypatch, stage):
    adapter = make_adapter(monkeypatch)
    pipe = adapter.pipe
    path = write_weights(tmp_path, payload())

    def fail(*args, **kwargs):
        raise RuntimeError("synthetic install failure")

    monkeypatch.setattr(adapter if stage == "configure_memory" else pipe, stage, fail)
    with pytest.raises(RuntimeError, match="synthetic install failure"):
        adapter.apply_loras(request([LoraSpec("mixed", path, 1, "mixed")]))
    assert adapter.pipe is None
    assert adapter._lora_signature == ()
    assert adapter._direct_state.originals == adapter._direct_state.parameters == {}


def test_pipeline_requires_explicit_hook_removal_before_mutation(tmp_path, monkeypatch):
    adapter = make_adapter(monkeypatch)
    pipe = adapter.pipe
    pristine = snapshot(pipe.transformer)
    monkeypatch.setattr(pipe, "remove_all_hooks", None)
    path = write_weights(tmp_path, payload())
    with pytest.raises(TypeError, match="remove_all_hooks"):
        adapter.apply_loras(request([LoraSpec("mixed", path, 1, "mixed")]))
    assert_pristine(pipe.transformer, pristine)
    assert pipe.events == []
    assert adapter.pipe is None