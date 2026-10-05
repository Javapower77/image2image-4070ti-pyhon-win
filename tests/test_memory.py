from __future__ import annotations

import logging
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


def test_apply_inference_memory_settings_falls_back_when_offload_hits_meta(caplog) -> None:
    moved = {"device": None}

    def enable_sequential_cpu_offload() -> None:
        raise NotImplementedError("Cannot copy out of meta tensor; no data!")

    def to(device: str) -> None:
        moved["device"] = device

    pipe = SimpleNamespace(enable_sequential_cpu_offload=enable_sequential_cpu_offload, to=to)
    with caplog.at_level(logging.INFO, logger="photo_edit_studio.models.memory"):
        apply_inference_memory_settings(pipe)
    assert moved["device"] == "cuda"
    warnings = [record for record in caplog.records if record.levelno == logging.WARNING]
    assert warnings[0].exc_info is not None
    assert warnings[0].exc_info[0] is NotImplementedError
    assert "falling back to CUDA" in warnings[1].getMessage()
    assert "Pipeline CUDA placement begin" in caplog.text
    assert "Pipeline CUDA placement end" in caplog.text


def test_offload_failure_logs_traceback_and_preserves_fallback_order(caplog) -> None:
    calls = []

    def sequential() -> None:
        calls.append("sequential")
        raise RuntimeError("offload unavailable")

    def model() -> None:
        calls.append("model")

    pipe = SimpleNamespace(
        enable_sequential_cpu_offload=sequential,
        enable_model_cpu_offload=model,
        to=lambda device: calls.append(device),
    )
    with caplog.at_level(logging.INFO, logger="photo_edit_studio.models.memory"):
        apply_inference_memory_settings(pipe)
    assert calls == ["sequential", "model"]
    assert "Pipeline offload end: method=enable_model_cpu_offload" in caplog.text
    assert "falling back to CUDA" not in caplog.text
    assert next(record for record in caplog.records if record.exc_info).exc_info[0] is RuntimeError
