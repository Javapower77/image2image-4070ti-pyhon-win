from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from photo_edit_studio.models.qwen_aio import (
    is_safetensors_file,
    materialize_qwen_rope,
    resolve_qwen_aio_checkpoint,
    transformer_state_from_comfy_aio,
)


def test_rejects_html_masquerading_as_safetensors(tmp_path: Path) -> None:
    fake = tmp_path / "Qwen-Rapid-AIO.safetensors"
    fake.write_text("<!doctype html><title>huggingface</title>", encoding="utf-8")
    assert is_safetensors_file(fake) is False
    with pytest.raises(FileNotFoundError, match="HTML"):
        resolve_qwen_aio_checkpoint(tmp_path)


def test_resolves_preferred_aio_filename(tmp_path: Path) -> None:
    other = tmp_path / "other.safetensors"
    preferred = tmp_path / "Qwen-Rapid-AIO.safetensors"
    save_file({"model.diffusion_model.img_in.weight": torch.zeros(2, 2)}, other)
    save_file({"model.diffusion_model.img_in.weight": torch.ones(2, 2)}, preferred)
    assert resolve_qwen_aio_checkpoint(tmp_path) == preferred


def test_extracts_transformer_keys_from_comfy_aio(tmp_path: Path, caplog) -> None:
    checkpoint = tmp_path / "Qwen-Rapid-AIO.safetensors"
    save_file(
        {
            "model.diffusion_model.img_in.weight": torch.ones(2, 2, dtype=torch.float32),
            "model.diffusion_model.__index_timestep_zero__": torch.zeros(1),
            "vae.conv1.weight": torch.zeros(3, 3),
            "text_encoders.qwen25_7b.weight": torch.zeros(4),
        },
        checkpoint,
    )
    with caplog.at_level(logging.INFO, logger="photo_edit_studio.models.qwen_aio"):
        state = transformer_state_from_comfy_aio(checkpoint, torch.bfloat16)
    assert set(state) == {"img_in.weight"}
    assert state["img_in.weight"].dtype == torch.bfloat16
    assert f"checkpoint_bytes={checkpoint.stat().st_size}" in caplog.text
    assert "source_keys=4" in caplog.text
    assert "transformer_keys=1 skipped_keys=3" in caplog.text
    assert "img_in.weight" not in caplog.text
    assert str(checkpoint) not in caplog.text


def test_materialize_qwen_rope_rebuilds_meta_freqs(caplog) -> None:
    class FakeRope:
        def __init__(self, theta: int, axes_dim: list[int], scale_rope: bool = False) -> None:
            self.theta = theta
            self.axes_dim = axes_dim
            self.scale_rope = scale_rope
            self.pos_freqs = torch.ones(2, 2)
            self.neg_freqs = torch.ones(2, 2)

    class FakeTransformer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.pos_embed = FakeRope(10000, [16, 56, 56], True)
            self.pos_embed.pos_freqs = torch.empty(2, 2, device="meta")
            self.pos_embed.neg_freqs = torch.empty(2, 2, device="meta")

    module = FakeTransformer()
    with caplog.at_level(logging.INFO, logger="photo_edit_studio.models.qwen_aio"):
        materialize_qwen_rope(module)
    assert not module.pos_embed.pos_freqs.is_meta
    assert not module.pos_embed.neg_freqs.is_meta
    assert module.pos_embed.pos_freqs.device.type == "cpu"
    assert "Qwen rope buffer rebuild begin" in caplog.text
    assert "Qwen rope buffer rebuild end" in caplog.text


def test_aio_load_logs_ordered_stages_without_loading_models(monkeypatch, caplog) -> None:
    from photo_edit_studio.models import diffusers_adapters as adapters

    calls = []
    state = {"private-checkpoint-key": object()}

    def apply_state(actual, *, strict):
        assert actual is state
        assert strict is False
        calls.append("apply")
        return SimpleNamespace(missing_keys=["private-missing-key"], unexpected_keys=[])

    transformer = SimpleNamespace(load_state_dict=apply_state)
    pipe = object()

    def load_transformer(*args, **kwargs):
        calls.append("transformer")
        assert kwargs["subfolder"] == "transformer"
        assert kwargs["torch_dtype"] is torch.bfloat16
        assert kwargs["local_files_only"] is True
        assert kwargs["low_cpu_mem_usage"] is True
        return transformer

    def load_state(*args):
        calls.append("state")
        return state

    def load_pipeline(*args, **kwargs):
        calls.append("pipeline")
        assert kwargs["transformer"] is transformer
        assert kwargs["torch_dtype"] is torch.bfloat16
        assert kwargs["local_files_only"] is True
        assert kwargs["low_cpu_mem_usage"] is True
        return pipe

    diffusers = SimpleNamespace(
        QwenImageTransformer2DModel=SimpleNamespace(from_pretrained=load_transformer),
        QwenImageEditPlusPipeline=SimpleNamespace(from_pretrained=load_pipeline),
    )
    monkeypatch.setattr(adapters, "_runtime", lambda: (diffusers, torch))
    monkeypatch.setattr(adapters, "_local_path", lambda *args: "private-base-path")
    monkeypatch.setattr(adapters, "resolve_qwen_aio_checkpoint", lambda *args: Path("private"))
    monkeypatch.setattr(adapters, "transformer_state_from_comfy_aio", load_state)
    monkeypatch.setattr(adapters, "materialize_qwen_rope", lambda value: calls.append("rope"))
    adapter = adapters.QwenAioAdapter(SimpleNamespace(local_path=Path("private")))
    monkeypatch.setattr(adapter, "_validate_hardware", lambda: calls.append("validate"))
    monkeypatch.setattr(adapter, "configure_memory", lambda: calls.append("offload"))
    with caplog.at_level(logging.INFO, logger=adapters.__name__):
        adapter.load()
    assert adapter.pipe is pipe
    assert calls == ["validate", "transformer", "state", "apply", "rope", "pipeline", "offload"]
    messages = [record.getMessage() for record in caplog.records]
    assert messages == [
        "Qwen AIO load begin",
        "Qwen AIO official transformer load begin",
        "Qwen AIO official transformer load end",
        "Qwen AIO state preparation begin",
        "Qwen AIO state preparation end: keys=1",
        "Qwen AIO state apply begin",
        "Qwen AIO state apply end: missing_keys=1 unexpected_keys=0",
        "Qwen AIO rope materialization begin",
        "Qwen AIO rope materialization end",
        "Qwen AIO pipeline load begin",
        "Qwen AIO pipeline load end",
        "Qwen AIO pipeline offload configuration begin",
        "Qwen AIO pipeline offload configuration end",
        "Qwen AIO load end",
    ]
    assert "private" not in caplog.text


def test_checkpoint_metadata_failure_does_not_change_loading(monkeypatch, caplog) -> None:
    import safetensors.torch

    class Checkpoint:
        def stat(self):
            raise OSError("metadata unavailable")

        def __str__(self):
            return "private-checkpoint-path"

    monkeypatch.setattr(
        safetensors.torch,
        "load_file",
        lambda path, device: {"model.diffusion_model.weight": torch.ones(1)},
    )
    with caplog.at_level(logging.INFO, logger="photo_edit_studio.models.qwen_aio"):
        state = transformer_state_from_comfy_aio(Checkpoint(), torch.bfloat16)
    assert state["weight"].dtype == torch.bfloat16
    assert "checkpoint_bytes=None" in caplog.text
    assert "private" not in caplog.text
