from __future__ import annotations

from pathlib import Path

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


def test_extracts_transformer_keys_from_comfy_aio(tmp_path: Path) -> None:
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
    state = transformer_state_from_comfy_aio(checkpoint, torch.bfloat16)
    assert set(state) == {"img_in.weight"}
    assert state["img_in.weight"].dtype == torch.bfloat16


def test_materialize_qwen_rope_rebuilds_meta_freqs() -> None:
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
    materialize_qwen_rope(module)
    assert not module.pos_embed.pos_freqs.is_meta
    assert not module.pos_embed.neg_freqs.is_meta
    assert module.pos_embed.pos_freqs.device.type == "cpu"
