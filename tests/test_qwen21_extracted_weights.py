"""Offline mixed-format regression tests: real CPU parameters, no model inference."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import safetensors
import test_qwen21_official_extract as extract_tests
import torch
from diffusers.loaders import lora_conversion_utils as conversion
from safetensors.torch import save_file

from photo_edit_studio.models import diffusers_adapters as adapters
from photo_edit_studio.models import qwen21_official_extract as assets

# Share the pipeline fixture only; the real hook test replaces its helper mock.
extracted = extract_tests.extracted


class TinyTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(causal_condition=True)
        self.img_in = torch.nn.Linear(3, 2, bias=False)
        self.txt_in = torch.nn.Module()
        self.txt_in.text_norm = torch.nn.LayerNorm(4)
        self.transformer_blocks = torch.nn.ModuleList()
        for _ in range(32):
            block = torch.nn.Module()
            block.attn = torch.nn.Module()
            for name in ("norm_q", "norm_k"):
                setattr(block.attn, name, torch.nn.LayerNorm(2))
            for name in ("to_q", "to_k", "to_v", "add_q_proj", "add_k_proj", "add_v_proj"):
                setattr(block.attn, name, torch.nn.Linear(3, 2, bias=False))
            block.img_mlp = torch.nn.Module()
            block.img_mlp.gate_layer = torch.nn.Linear(3, 2, bias=False)
            block.img_mlp.proj = torch.nn.Linear(3, 2, bias=False)
            self.transformer_blocks.append(block)
        self.extras = torch.nn.ModuleList([torch.nn.Linear(3, 2, bias=False) for _ in range(7)])


class TinyPipe:
    def __init__(self):
        self.transformer = TinyTransformer()
        self.events = []
        self.loaded = None
        self.active = None
        self.before_load = lambda: None

    def load_lora_weights(self, state_dict, *, adapter_name, local_files_only):
        self.before_load()
        assert adapter_name == "official_extract" and local_files_only is True
        assert isinstance(state_dict, dict) and not any(".diff" in key for key in state_dict)
        self.loaded = state_dict
        self.events.append("load converted dictionary")

    def set_adapters(self, names, *, adapter_weights):
        self.active = (names, adapter_weights)
        self.events.append("activate")


def triple(base="img_in", alpha=1.0):
    prefix = "diffusion_model." + base
    return {
        prefix + ".lora_down.weight": torch.tensor([[1., 2., -3.], [4., -2., 1.]]),
        prefix + ".lora_up.weight": torch.tensor([[2., -1.], [3., 4.]]),
        prefix + ".alpha": torch.tensor(alpha, dtype=torch.float64),
    }


def direct_deltas():
    result = {"diffusion_model.txt_in.text_norm.diff": torch.tensor([.25, -.5, .75, 1.])}
    for index in range(32):
        for name in ("q", "k"):
            result[f"diffusion_model.transformer_blocks.{index}.attn.norm_{name}.diff"] = (
                torch.tensor([index + .125, -index - .25])
            )
    assert len(result) == 65
    return result


def snapshot(module):
    return {name: value.detach().clone() for name, value in module.named_parameters()}


def assert_unchanged(module, before):
    parameters = dict(module.named_parameters())
    assert set(parameters) == set(before)
    for name, value in before.items():
        if value.is_meta:
            assert parameters[name].is_meta and parameters[name].shape == value.shape
        else:
            torch.testing.assert_close(parameters[name], value, rtol=0, atol=0)


@pytest.mark.parametrize("pair_count", [1, 200])
def test_all_65_deltas_and_source_pairs_preserved_with_real_converter(tmp_path, pair_count):
    pipe = TinyPipe()
    bases = ["img_in"]
    if pair_count == 200:
        bases += [f"transformer_blocks.{index}.attn.{name}" for index in range(32)
                  for name in ("to_q", "to_k", "to_v", "add_q_proj", "add_k_proj", "add_v_proj")]
        bases += [f"extras.{index}" for index in range(7)]
    assert len(bases) == pair_count
    state = direct_deltas()
    for index, base in enumerate(bases):
        state.update(triple(base, alpha=(index % 4) / 2))
    source_before = {key: value.clone() for key, value in state.items()}
    before = snapshot(pipe.transformer)
    path = tmp_path / "mixed.safetensors"
    save_file(state, path)
    file_before = path.read_bytes()

    def assert_direct_before_load():
        parameters = dict(pipe.transformer.named_parameters())
        assert all(parameter.device.type == "cpu" and not parameter.is_meta
                   for parameter in parameters.values())
        for key, delta in direct_deltas().items():
            name = key.removeprefix("diffusion_model.").removesuffix(".diff") + ".weight"
            torch.testing.assert_close(parameters[name], before[name] + delta, rtol=0, atol=0)

    pipe.before_load = assert_direct_before_load
    assets.apply_qwen21_extracted_weights(pipe, path)
    assert pipe.events == ["load converted dictionary", "activate"]
    assert pipe.active == (["official_extract"], [1.0])
    assert len(pipe.loaded) == pair_count * 2
    for base in bases:
        prefix = "diffusion_model." + base
        expected = (state[prefix + ".alpha"].item() / 2) * (
            state[prefix + ".lora_up.weight"] @ state[prefix + ".lora_down.weight"])
        actual = pipe.loaded[f"transformer.{base}.lora_B.weight"] @ pipe.loaded[f"transformer.{base}.lora_A.weight"]
        torch.testing.assert_close(actual, expected)
    delta_names = {key.removeprefix("diffusion_model.").removesuffix(".diff") + ".weight"
                   for key in direct_deltas()}
    for name, parameter in pipe.transformer.named_parameters():
        if name not in delta_names:
            torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)
    assert path.read_bytes() == file_before
    for key in state:
        torch.testing.assert_close(state[key], source_before[key], rtol=0, atol=0)


@pytest.mark.parametrize("alpha", [0., .125, 1., 2., 8.])
def test_fused_swiglu_gate_first_split_preserves_actual_algebra_and_inputs(monkeypatch, alpha):
    base = "transformer_blocks.0.img_mlp.gate_up"
    state = triple(base, alpha)
    state[f"diffusion_model.{base}.lora_up.weight"] = torch.tensor(
        [[2., -1.], [3., 4.], [-5., 6.], [7., -8.]])
    state.update(direct_deltas())
    originals = dict(state)
    before = {key: value.clone() for key, value in state.items()}

    class Handle:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def keys(self):
            return state.keys()

        def get_tensor(self, key):
            return state[key]  # Shared real tensors expose in-place mutation bugs.

    opened = Mock(return_value=Handle())
    monkeypatch.setattr(safetensors, "safe_open", opened)
    real_converter = conversion._convert_non_diffusers_qwen_lora_to_diffusers

    def checked_converter(source):
        assert set(source) == {base + suffix for suffix in
                               (".lora_down.weight", ".lora_up.weight", ".alpha")}
        return real_converter(source)

    monkeypatch.setattr(conversion, "_convert_non_diffusers_qwen_lora_to_diffusers", checked_converter)
    pipe = TinyPipe()
    assets.apply_qwen21_extracted_weights(pipe, "synthetic.safetensors")
    opened.assert_called_once_with("synthetic.safetensors", framework="pt", device="cpu")
    expected = alpha / 2 * (before[f"diffusion_model.{base}.lora_up.weight"] @
                            before[f"diffusion_model.{base}.lora_down.weight"])
    assert len(pipe.loaded) == 4
    for index, name in enumerate(("gate_layer", "proj")):
        prefix = "transformer." + base.replace("gate_up", name)
        actual = pipe.loaded[prefix + ".lora_B.weight"] @ pipe.loaded[prefix + ".lora_A.weight"]
        torch.testing.assert_close(actual, expected[index * 2:(index + 1) * 2])
    # Check the gate-first SwiGLU function, not merely key names or matrix shapes.
    x = torch.tensor([[.25, -.5, 1.]])
    gate, up = (x @ expected.T).chunk(2, dim=-1)
    outputs = []
    for name in ("gate_layer", "proj"):
        prefix = "transformer." + base.replace("gate_up", name)
        outputs.append(x @ pipe.loaded[prefix + ".lora_A.weight"].T @
                       pipe.loaded[prefix + ".lora_B.weight"].T)
    torch.testing.assert_close(torch.nn.functional.silu(outputs[0]) * outputs[1],
                               torch.nn.functional.silu(gate) * up)
    assert pipe.active == (["official_extract"], [1.0])
    assert set(state) == set(before)
    for key in state:
        assert state[key] is originals[key]
        torch.testing.assert_close(state[key], before[key], rtol=0, atol=0)


@pytest.mark.parametrize("failure,match", [
    ("missing_lora_target", "unknown transformer parameter"),
    ("missing_direct_target", "unknown transformer parameter"),
    ("meta_lora", "not meta"), ("meta_direct", "not meta"),
    ("wrong_lora_shape", "shape/dtype mismatch"),
    ("wrong_direct_shape", "shape/dtype mismatch"),
    ("unknown_delta", "Unsupported"), ("unknown_key", "Unsupported"),
    ("orphan_alpha", "Incomplete"), ("missing_alpha", "Incomplete"),
    ("missing_up", "Incomplete"), ("missing_down", "Incomplete"),
    ("negative_alpha", "nonnegative"), ("nan_alpha", "Nonfinite"),
    ("vector_alpha", "scalar"), ("nonfinite_matrix", "Nonfinite"),
    ("nonfloating_delta", "nonfloating"), ("wrong_rank", "matrix pair"),
    ("odd_fused", "fused extracted"), ("duplicate", "Duplicate"),
    ("missing_api", "LoRA-capable"),
])
def test_invalid_mixed_input_fails_before_any_parameter_or_adapter_mutation(tmp_path, failure, match):
    pipe = TinyPipe()
    state = {**triple(), **direct_deltas()}
    prefix = "diffusion_model.img_in"
    if failure == "missing_lora_target":
        del pipe.transformer.img_in
    elif failure == "missing_direct_target":
        del pipe.transformer.txt_in.text_norm
    elif failure == "meta_lora":
        pipe.transformer.img_in.to("meta")
    elif failure == "meta_direct":
        pipe.transformer.txt_in.text_norm.to("meta")
    elif failure == "wrong_lora_shape":
        state[prefix + ".lora_up.weight"] = torch.ones(5, 2)
    elif failure == "wrong_direct_shape":
        state["diffusion_model.txt_in.text_norm.diff"] = torch.ones(5)
    elif failure == "unknown_delta":
        state["diffusion_model.transformer_blocks.32.attn.norm_q.diff"] = torch.ones(2)
    elif failure == "unknown_key":
        state["diffusion_model.img_in.weight"] = torch.ones(2, 3)
    elif failure == "orphan_alpha":
        state["diffusion_model.orphan.alpha"] = torch.tensor(1.)
    elif failure.startswith("missing_") and failure != "missing_api":
        del state[prefix + {"missing_alpha": ".alpha", "missing_up": ".lora_up.weight",
                           "missing_down": ".lora_down.weight"}[failure]]
    elif failure in {"negative_alpha", "nan_alpha", "vector_alpha"}:
        state[prefix + ".alpha"] = {"negative_alpha": torch.tensor(-1.),
                                    "nan_alpha": torch.tensor(float("nan")),
                                    "vector_alpha": torch.ones(2)}[failure]
    elif failure == "nonfinite_matrix":
        state[prefix + ".lora_up.weight"][0, 0] = float("inf")
    elif failure == "nonfloating_delta":
        state["diffusion_model.txt_in.text_norm.diff"] = torch.ones(4, dtype=torch.int64)
    elif failure == "wrong_rank":
        state[prefix + ".lora_up.weight"] = torch.ones(2, 3)
    elif failure == "odd_fused":
        base = "transformer_blocks.0.img_mlp.gate_up"
        state.update(triple(base))
        state[f"diffusion_model.{base}.lora_up.weight"] = torch.ones(3, 2)
    elif failure == "duplicate":
        state["img_in.alpha"] = state[prefix + ".alpha"].clone()
    elif failure == "missing_api":
        pipe.set_adapters = None
    before = snapshot(pipe.transformer)
    source_before = {key: value.clone() for key, value in state.items()}
    path = tmp_path / "invalid.safetensors"
    save_file(state, path)
    bytes_before = path.read_bytes()
    with pytest.raises(ValueError, match=match):
        assets.apply_qwen21_extracted_weights(pipe, path)
    assert_unchanged(pipe.transformer, before)
    assert pipe.events == [] and pipe.loaded is None and pipe.active is None
    assert path.read_bytes() == bytes_before
    for key, value in state.items():
        torch.testing.assert_close(value, source_before[key], rtol=0, atol=0, equal_nan=True)


def test_real_cpu_mixed_helper_hook_completes_before_offload(extracted, monkeypatch, tmp_path):
    instance, pipe_class, _runtime, events = extracted
    state = {**triple(), **direct_deltas()}
    path = tmp_path / "hook.safetensors"
    save_file(state, path)
    monkeypatch.setattr(adapters, "qwen21_official_extract_path", lambda: path)
    monkeypatch.setattr(adapters, "validate_qwen21_official_extract_lora", lambda _path: None)
    monkeypatch.setattr(adapters, "apply_qwen21_extracted_weights", assets.apply_qwen21_extracted_weights)
    real_init = pipe_class.__init__

    def init(pipe, sample_sigmas):
        real_init(pipe, sample_sigmas)
        pipe.transformer = TinyTransformer()
        pipe.before = snapshot(pipe.transformer)

    monkeypatch.setattr(pipe_class, "__init__", init)

    def offload():
        pipe = instance.pipe
        assert pipe.active == (["official_extract"], [1.0])
        assert events[-1] == ("activate", ["official_extract"], [1.0])
        parameters = dict(pipe.transformer.named_parameters())
        for key, delta in direct_deltas().items():
            name = key.removeprefix("diffusion_model.").removesuffix(".diff") + ".weight"
            assert parameters[name].device.type == "cpu" and not parameters[name].is_meta
            torch.testing.assert_close(parameters[name], pipe.before[name] + delta, rtol=0, atol=0)
        events.append("offload after real mixed preparation")

    monkeypatch.setattr(instance, "configure_memory", offload)
    instance.load()
    assert events[-1] == "offload after real mixed preparation"
    assert len(events[2][1]) == 2
    assert instance.pipe.calls == []  # Loading/hook only, never inference.

