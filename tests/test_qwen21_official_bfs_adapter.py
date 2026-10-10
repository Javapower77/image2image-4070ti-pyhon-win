"""Real tiny CPU safetensors; no publisher assets, checkpoint or inference runtime."""
from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest
import torch
from PIL import Image
from safetensors.torch import save_file
from test_qwen21_official_adapter import KEY, SIGMAS, Pipe, request

from photo_edit_studio import swap
from photo_edit_studio.models import diffusers_adapters as adapters
from photo_edit_studio.models import qwen2511_lora as lora_runtime
from photo_edit_studio.models.registry import MODEL_SPECS
from photo_edit_studio.types import LoraSpec

FUSED = "diffusion_model.transformer_blocks.0.img_mlp.gate_up"
LINEAR = "transformer.transformer_blocks.0.attn.to_q"


def pair(prefix: str, *, outputs: int = 4, inputs: int = 3, rank: int = 2):
    # Distinct rows make gate-first (rather than up-first) splitting observable.
    return {
        prefix + ".lora_A.weight": torch.arange(1, rank * inputs + 1).reshape(
            rank, inputs,
        ).float(),
        prefix + ".lora_B.weight": torch.arange(1, outputs * rank + 1).reshape(
            outputs, rank,
        ).float(),
    }


class TinyTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(causal_condition=True)
        block = torch.nn.Module()
        block.img_mlp = torch.nn.Module()
        block.img_mlp.gate_layer = torch.nn.Linear(3, 2, bias=False)
        block.img_mlp.proj = torch.nn.Linear(3, 2, bias=False)
        block.attn = torch.nn.Module()
        block.attn.to_q = torch.nn.Linear(3, 4, bias=False)
        self.transformer_blocks = torch.nn.ModuleList([block])


@pytest.fixture
def rig(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    events = []
    monkeypatch.setattr(adapters.settings, "model_dir", tmp_path / "models")
    monkeypatch.setattr(adapters.settings, "lora_dir", tmp_path / "loras")
    snapshot = MODEL_SPECS[KEY].local_path
    (snapshot / "transformer").mkdir(parents=True)
    (snapshot / "model_index.json").write_text(json.dumps({
        "_class_name": "QwenImage21Pipeline", "sample_sigmas": SIGMAS,
    }), encoding="utf-8")
    (snapshot / "transformer" / "config.json").write_text(
        json.dumps({"causal_condition": True}), encoding="utf-8",
    )
    # Replace the mapping locally, never mutate the real publisher pin dictionaries.
    pins = {kind: dict(pin) for kind, pin in swap.QWEN21_BFS_PINS.items()}
    monkeypatch.setattr(swap, "QWEN21_BFS_PINS", pins)
    monkeypatch.setattr(adapters, "QWEN21_BFS_PINS", pins)

    class BFSPipe(Pipe):
        loads: ClassVar[list[tuple[str, dict]]] = []

        def __init__(self, sample_sigmas):
            super().__init__(sample_sigmas)
            self.transformer = TinyTransformer()
            self.base_weights = {
                name: value.detach().clone()
                for name, value in self.transformer.named_parameters()
            }
            self.installs = []
            self.activations = []
            self.active = None

        @classmethod
        def from_pretrained(cls, path, **kwargs):
            events.append("allocate")
            cls.loads.append((path, kwargs))
            instance = cls(SIGMAS)
            state.pipes.append(instance)
            return instance

        def remove_all_hooks(self):
            events.append("removehooks")

        def unload_lora_weights(self):
            events.append("unloadBFS")
            self.active = None

        def load_lora_weights(self, matrices, *, adapter_name, local_files_only):
            events.append("install")
            assert adapter_name == "bfs_swap" and local_files_only is True
            assert all(value.device.type == "cpu" for value in matrices.values())
            assert all(key.endswith((".lora_A.weight", ".lora_B.weight")) for key in matrices)
            self.installs.append(matrices)
            self.active = adapter_name

        def set_adapters(self, names, *, adapter_weights):
            events.append("activate")
            assert names == ["bfs_swap"]
            self.activations.append((names, adapter_weights))

    instance = adapters.Qwen21OfficialAdapter(MODEL_SPECS[KEY])
    state = SimpleNamespace(
        adapter=instance, pipe_class=BFSPipe, pipes=[], events=events, pins=pins,
    )

    def write(kind="Head", tensors=None, *, pairs=2):
        path = adapters.settings.lora_dir / "qwen21" / swap.swap_profile(KEY, kind).filename
        path.parent.mkdir(parents=True, exist_ok=True)
        source = {**pair(FUSED), **pair(LINEAR)} if tensors is None else tensors
        save_file(source, str(path))
        pins[kind].update(
            size=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            pairs=pairs,
        )
        return path

    def swap_request(kind="Head", weight=0.65):
        profile = swap.swap_profile(KEY, kind)
        return request(
            workflow="swap", swap_kind=kind,
            images=[Image.new("RGB", (12, 16), "red"), Image.new("RGB", (16, 12), "blue")],
            loras=[LoraSpec(profile.filename, adapters.settings.lora_dir / "qwen21" /
                            profile.filename, weight, "bfs_swap")],
        )

    real_hash = adapters.validate_qwen21_bfs_file

    def validate_hash(path, kind):
        real_hash(path, kind)
        events.append("hash")

    real_parse = lora_runtime.parse_qwen_lora
    real_split = adapters.split_qwen21_fused_lora
    real_shape = lora_runtime.validate_qwen_lora

    def parse(path):
        events.append("parse")
        return real_parse(path)

    def split(matrices):
        events.append("split")
        return real_split(matrices)

    def shape(transformer, weights):
        events.append("shape")
        return real_shape(transformer, weights)

    def runtime():
        events.append("runtime")
        return SimpleNamespace(QwenImage21Pipeline=BFSPipe), torch

    monkeypatch.setattr(adapters, "validate_qwen21_bfs_file", validate_hash)
    monkeypatch.setattr(lora_runtime, "parse_qwen_lora", parse)
    monkeypatch.setattr(adapters, "split_qwen21_fused_lora", split)
    monkeypatch.setattr(lora_runtime, "validate_qwen_lora", shape)
    monkeypatch.setattr(adapters, "_runtime", runtime)
    monkeypatch.setattr(instance, "_validate_hardware", lambda: events.append("hardware"))
    monkeypatch.setattr(instance, "configure_memory", lambda: events.append("offload"))
    monkeypatch.setattr(adapters, "_supported_call", lambda *_: pytest.fail("Filtered call"))
    state.write = write
    state.request = swap_request
    write("Head")
    write("Body")
    return state


@pytest.mark.parametrize("kind", ["Head", "Body"])
@pytest.mark.parametrize("weight", [0.1, 0.65, 1.5])
def test_fresh_swap_real_cpu_pairs_gate_first_unit_scale_and_unchanged_inputs(rig, kind, weight):
    req = rig.request(kind, weight)
    req.count = 2
    before = [image.tobytes() for image in req.images]
    source_bytes = req.loras[0].path.read_bytes()
    result = rig.adapter.generate(req)
    assert rig.events == [
        "hash", "runtime", "hardware", "allocate", "parse", "split", "shape",
        "install", "activate", "offload", "runtime",
    ]
    pipe = rig.adapter.pipe
    assert len(rig.pipe_class.loads) == 1
    assert rig.pipe_class.loads[0][1] == {
        "torch_dtype": torch.bfloat16, "local_files_only": True, "low_cpu_mem_usage": True,
    }
    assert pipe.activations == [(["bfs_swap"], [weight])]
    assert len(pipe.installs) == 1
    matrices = pipe.installs[0]
    assert len(matrices) == 6
    assert not any("alpha" in key or "gate_up" in key for key in matrices)
    original_a = pair(FUSED)[FUSED + ".lora_A.weight"]
    original_b = pair(FUSED)[FUSED + ".lora_B.weight"]
    x = torch.tensor([0.25, -0.5, 1.0])
    for index, module in enumerate(("gate_layer", "proj")):
        prefix = f"transformer.transformer_blocks.0.img_mlp.{module}"
        a, b = (matrices[prefix + suffix] for suffix in (".lora_A.weight", ".lora_B.weight"))
        torch.testing.assert_close(a, original_a)
        torch.testing.assert_close(b, original_b[index * 2:(index + 1) * 2])
        # No implicit 1/rank or request-weight multiplication in serialized B.
        torch.testing.assert_close(weight * (b @ a @ x),
                                   (weight * (original_b @ original_a @ x))[index * 2:(index + 1) * 2])
    for key, value in pair(LINEAR).items():
        torch.testing.assert_close(matrices[key], value)
    for name, value in pipe.transformer.named_parameters():
        torch.testing.assert_close(value, pipe.base_weights[name])
    assert req.loras[0].path.read_bytes() == source_bytes
    assert [image.tobytes() for image in req.images] == before
    assert [image.size for image in result] == [(512, 768)] * 2
    for index, call in enumerate(pipe.calls):
        assert call["image"] is req.images
        assert call["prompt"] == "Keep this exact prompt"
        assert (call["width"], call["height"], call["num_inference_steps"]) == (512, 768, 8)
        assert call["true_cfg_scale"] == 1 and call["use_kv_cache"] is True
        assert call["generator"].device.type == "cpu"
        assert call["generator"].initial_seed() == req.seed + index
        assert call["num_images_per_prompt"] == 1


@pytest.mark.parametrize("destination", ["standard", "compound", "text"])
def test_lifecycle_plain_head_noop_weight_body_plain_without_checkpoint_reload(rig, destination):
    rig.adapter.generate(request())
    pipe = rig.adapter.pipe
    assert rig.events == ["runtime", "hardware", "allocate", "offload", "runtime"]
    head = rig.request()

    def transition(req, expected):
        rig.events.clear()
        rig.adapter.generate(req)
        assert rig.events == expected
        assert rig.adapter.pipe is pipe
        assert len(rig.pipe_class.loads) == 1

    install = ["hash", "removehooks", "unloadBFS", "parse", "split", "shape",
               "install", "activate", "offload", "runtime"]
    transition(head, install)
    transition(head, ["runtime"])
    transition(rig.request(weight=0.9), install)
    transition(rig.request("Body", 1.25), install)
    plain = request()
    if destination == "compound":
        plain = request(images=[Image.new("RGB", (8, 8)) for _ in range(3)], compose=True)
    elif destination == "text":
        plain = request(workflow="text", images=[])
    transition(plain, ["removehooks", "unloadBFS", "offload", "runtime"])
    assert pipe.active is None and rig.adapter._lora_signature == ()
    assert pipe.activations == [(["bfs_swap"], [0.65]), (["bfs_swap"], [0.9]),
                                (["bfs_swap"], [1.25])]
    assert len(pipe.installs) == 3  # No compound/text adapter or unrelated Turbo stack.
    transition(plain, ["runtime"])


@pytest.mark.parametrize("defect,match", [
    ("missing", "missing"), ("size", "size mismatch"), ("hash", "SHA256 mismatch"),
])
def test_file_integrity_rejected_before_runtime_hardware_or_allocation(rig, defect, match):
    req = rig.request()
    if defect == "missing":
        req.loras[0].path.unlink()
    elif defect == "size":
        rig.pins["Head"]["size"] += 1
    else:
        rig.pins["Head"]["sha256"] = "0" * 64
    with pytest.raises((ValueError, FileNotFoundError), match=match):
        rig.adapter.generate(req)
    assert rig.events == [] and rig.pipe_class.loads == []
    assert rig.adapter.pipe is None and rig.adapter._pending_request is None


@pytest.mark.parametrize("defect,match", [
    ("pair-count", "A/B-only"), ("missing-pair", "Incomplete or mixed"),
    ("old-format", "A/B-only"), ("alpha", "A/B-only"), ("delta", "A/B-only"),
    ("rank", "rank/shapes"), ("ndim", "rank/shapes"), ("nonfinite", "Nonfinite"),
    ("integer", "nonfloating"), ("odd-gate", "Invalid fused"),
    ("shape", "shape/dtype mismatch"), ("unknown", "Unknown Qwen"),
])
def test_malformed_valid_hash_files_reject_before_loader_and_offload(rig, defect, match):
    tensors = pair(FUSED)
    pairs = 1
    a_key, b_key = FUSED + ".lora_A.weight", FUSED + ".lora_B.weight"
    if defect == "pair-count":
        pairs = 2
    elif defect == "missing-pair":
        tensors[LINEAR + ".lora_A.weight"] = tensors.pop(b_key)
    elif defect == "old-format":
        tensors = {key.replace("lora_A", "lora_down").replace("lora_B", "lora_up"): value
                   for key, value in tensors.items()}
    elif defect in {"alpha", "delta"}:
        tensors[FUSED + (".alpha" if defect == "alpha" else ".diff")] = torch.tensor(2.0)
    elif defect == "rank":
        tensors[b_key] = torch.ones(4, 3)
    elif defect == "ndim":
        tensors[a_key] = torch.ones(2, 3, 1)
    elif defect == "nonfinite":
        tensors[a_key][0, 0] = float("nan")
    elif defect == "integer":
        tensors[a_key] = tensors[a_key].long()
    elif defect == "odd-gate":
        tensors = pair(FUSED, outputs=3)
    elif defect == "shape":
        tensors = pair(FUSED, inputs=5)
    elif defect == "unknown":
        tensors = pair("transformer.missing")
    rig.write(tensors=tensors, pairs=pairs)
    with pytest.raises(ValueError, match=match):
        rig.adapter.generate(rig.request())
    assert "install" not in rig.events and "activate" not in rig.events
    assert "offload" not in rig.events
    assert rig.adapter.pipe is None and rig.adapter._lora_signature == ()
    assert rig.adapter._pending_request is None
    assert all(not pipe.calls for pipe in rig.pipes)


def test_malformed_valid_hash_shape_rejected_before_checkpoint_allocation(rig):
    """Deliberate non-xfail regression: file shape preflight must precede allocation."""
    rig.write(tensors=pair(FUSED, inputs=5), pairs=1)
    with pytest.raises(ValueError, match="shape/dtype mismatch"):
        rig.adapter.generate(rig.request())
    assert rig.pipe_class.loads == [], (
        "Production gap: valid-hash malformed BFS shapes reached from_pretrained; "
        "shape preflight must run BEFORE checkpoint allocation"
    )


@pytest.mark.parametrize("defect,match", [("meta", "offloaded to meta"),
                                        ("nonlinear", "unquantized Linear")])
def test_actual_transformer_target_must_be_materialized_linear_before_install(rig, defect, match):
    rig.adapter.generate(request())
    pipe = rig.adapter.pipe
    target = pipe.transformer.transformer_blocks[0].img_mlp
    if defect == "meta":
        target.gate_layer = torch.nn.Linear(3, 2, bias=False, device="meta")
    else:
        target.gate_layer = torch.nn.Module()
        target.gate_layer.register_parameter("weight", torch.nn.Parameter(torch.ones(2, 3)))
    rig.events.clear()
    with pytest.raises(ValueError, match=match):
        rig.adapter.generate(rig.request())
    assert rig.events == ["hash", "removehooks", "unloadBFS", "parse", "split", "shape"]
    assert not pipe.installs
    assert rig.adapter.pipe is None and rig.adapter._lora_signature == ()


@pytest.mark.parametrize("method", ["remove_all_hooks", "unload_lora_weights",
                                   "load_lora_weights", "set_adapters"])
@pytest.mark.parametrize("loaded", [False, True])
def test_missing_lora_api_fails_closed_before_allocation_or_mutation(rig, monkeypatch, method, loaded):
    if loaded:
        rig.adapter.generate(request())
        target = rig.adapter.pipe
    else:
        target = rig.pipe_class
    monkeypatch.setattr(target, method, None)
    rig.events.clear()
    with pytest.raises((TypeError, RuntimeError), match="LoRA"):
        rig.adapter.generate(rig.request())
    assert rig.events == (["hash"] if loaded else ["hash", "runtime"])
    assert len(rig.pipe_class.loads) == int(loaded)
    assert rig.adapter.pipe is None and rig.adapter._lora_signature == ()
    assert rig.adapter._pending_request is None


@pytest.mark.parametrize("loaded,failure", [
    (False, "install"), (False, "activate"), (False, "offload"),
    (True, "removehooks"), (True, "unloadBFS"), (True, "install"),
    (True, "activate"), (True, "offload"),
])
def test_partial_lifecycle_failure_discards_pipeline_and_signature(rig, monkeypatch, failure, loaded):
    if loaded:
        rig.adapter.generate(rig.request())
    pipe = rig.adapter.pipe
    method = {"removehooks": "remove_all_hooks", "unloadBFS": "unload_lora_weights",
              "install": "load_lora_weights", "activate": "set_adapters"}.get(failure)

    def fail(*args, **kwargs):
        rig.events.append(failure)
        raise RuntimeError(f"synthetic {failure} failure")

    if method:
        monkeypatch.setattr(pipe if loaded else rig.pipe_class, method, fail)
    else:
        monkeypatch.setattr(rig.adapter, "configure_memory", fail)
    rig.events.clear()
    with pytest.raises(RuntimeError, match=f"synthetic {failure} failure"):
        rig.adapter.generate(rig.request("Body"))
    assert rig.events[-1] == failure
    assert rig.adapter.pipe is None and rig.adapter._lora_signature == ()
    assert rig.adapter._pending_request is None
    assert len(rig.pipe_class.loads) == 1
    assert all(len(item.calls) == int(loaded) for item in rig.pipes)


def test_corrupt_replacement_hash_discards_loaded_pipeline_without_hook_mutation(rig):
    rig.adapter.generate(rig.request())
    pipe = rig.adapter.pipe
    rig.pins["Body"]["sha256"] = "0" * 64
    rig.events.clear()
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        rig.adapter.generate(rig.request("Body"))
    assert rig.events == []
    assert rig.adapter.pipe is None and rig.adapter._lora_signature == ()
    assert rig.adapter._pending_request is None
    assert len(pipe.installs) == 1 and len(pipe.calls) == 1
    assert len(rig.pipe_class.loads) == 1


@pytest.mark.parametrize("defect", ["optional", "extract-stack"])
def test_optional_or_extracted_stack_rejected_without_reading_files(rig, defect):
    req = rig.request()
    if defect == "optional":
        req.loras.append(LoraSpec("style", Path("never-read.safetensors"), 1.0, "style"))
        match = "no optional LoRAs"
    else:
        req = replace(req, model_key="qwen-2.1-turbo-official-extract")
        match = "no swaps"
    with pytest.raises(ValueError, match=match):
        rig.adapter.generate(req)
    assert rig.events == [] and rig.pipe_class.loads == []


@pytest.mark.parametrize("field,value,match", [
    ("swap_kind", "Face", "Head or Body"),
    ("images", [], "two ordered"), ("images", [None], "two ordered"),
    ("images", [None] * 3, "two ordered"), ("loras", [], "exactly one"),
    ("steps", 6, "eight"), ("steps", 8.0, "eight"), ("steps", True, "eight"),
    ("guidance", 0, "guidance 1"), ("guidance", 8, "guidance 1"),
    ("true_cfg", 0, "True CFG 1"), ("true_cfg", 8, "True CFG 1"),
    ("negative_prompt", "blur", "empty negative"),
    ("dlss", {"enabled": True}, "DLSS"), ("dlss", {"enabled": False}, "DLSS"),
])
def test_swap_contract_rejects_before_any_runtime_or_file_read(rig, field, value, match):
    req = replace(rig.request(), **{field: value})
    with pytest.raises(ValueError, match=match):
        rig.adapter.generate(req)
    assert rig.events == [] and rig.pipe_class.loads == []


@pytest.mark.parametrize("weight", [0, -1, 0.099, 1.501, float("nan"),
                                    float("inf"), -float("inf")])
def test_bfs_weight_requires_positive_finite_bounded_request_weight(rig, weight):
    with pytest.raises(ValueError, match="finite.*0.1"):
        rig.adapter.generate(rig.request(weight=weight))
    assert rig.events == [] and rig.pipe_class.loads == []


@pytest.mark.parametrize("defect", ["name", "path", "adapter", "wrong-kind", "legacy"])
def test_exact_matching_canonical_profile_required_before_file_read(rig, defect):
    req = rig.request()
    item = req.loras[0]
    if defect == "name":
        item = replace(item, name="different.safetensors")
    elif defect == "path":
        item = replace(item, path=item.path.parent.parent / item.path.name)
    elif defect == "adapter":
        item = replace(item, adapter_name="official_extract")
    elif defect == "wrong-kind":
        item = rig.request("Body").loras[0]
    else:
        name = "Qwen21-BFS_Head_v1.1.safetensors"
        item = replace(item, name=name, path=item.path.with_name(name))
    req.loras = [item]
    with pytest.raises(ValueError, match="exact matching"):
        rig.adapter.generate(req)
    assert rig.events == [] and rig.pipe_class.loads == []