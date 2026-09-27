from __future__ import annotations

import os
from types import SimpleNamespace

from photo_edit_studio.models.memory import (
    ALLOC_CONF,
    apply_inference_memory_settings,
    configure_cuda_allocator,
)


def test_configure_cuda_allocator_sets_default(monkeypatch) -> None:
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    configure_cuda_allocator()
    assert os.environ["PYTORCH_CUDA_ALLOC_CONF"] == ALLOC_CONF


def test_apply_inference_memory_settings_prefers_offload() -> None:
    vae = SimpleNamespace(sliced=False, tiled=False)
    vae.enable_slicing = lambda: setattr(vae, "sliced", True)
    vae.enable_tiling = lambda: setattr(vae, "tiled", True)
    moved = {"device": None}

    def enable_sequential_cpu_offload() -> None:
        moved["device"] = "offload"

    def to(device: str) -> None:
        moved["device"] = device

    pipe = SimpleNamespace(
        vae=vae, enable_sequential_cpu_offload=enable_sequential_cpu_offload, to=to
    )
    apply_inference_memory_settings(pipe)
    assert vae.sliced is True
    assert vae.tiled is True
    assert moved["device"] == "offload"


def test_apply_inference_memory_settings_falls_back_when_offload_hits_meta() -> None:
    moved = {"device": None}

    def enable_sequential_cpu_offload() -> None:
        raise NotImplementedError("Cannot copy out of meta tensor; no data!")

    def to(device: str) -> None:
        moved["device"] = device

    pipe = SimpleNamespace(enable_sequential_cpu_offload=enable_sequential_cpu_offload, to=to)
    apply_inference_memory_settings(pipe)
    assert moved["device"] == "cuda"
