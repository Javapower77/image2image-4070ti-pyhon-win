from __future__ import annotations

import gc
import os
from typing import Any

ALLOC_CONF = "expandable_segments:True,max_split_size_mb:128"


def configure_cuda_allocator() -> None:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", ALLOC_CONF)


def _torch() -> Any:
    import torch

    return torch


def cuda_available() -> bool:
    try:
        return bool(_torch().cuda.is_available())
    except (ImportError, RuntimeError):
        return False


def cuda_vram_gb() -> float:
    if not cuda_available():
        return 0.0
    properties = _torch().cuda.get_device_properties(0)
    return properties.total_memory / (1024**3)


def release_cuda() -> None:
    gc.collect()
    if cuda_available():
        torch = _torch()
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()


def apply_inference_memory_settings(pipe: Any, mode: str = "sequential") -> None:
    """Configure a Diffusers pipeline for a 12 GB Windows GPU."""
    vae = getattr(pipe, "vae", None)
    if vae is not None:
        for method_name in ("enable_slicing", "enable_tiling"):
            method = getattr(vae, method_name, None)
            if callable(method):
                method()

    enable_attention_slicing = getattr(pipe, "enable_attention_slicing", None)
    if callable(enable_attention_slicing):
        enable_attention_slicing("max")

    offload_methods = (
        ("enable_sequential_cpu_offload", "enable_model_cpu_offload")
        if mode == "sequential"
        else ("enable_model_cpu_offload", "enable_sequential_cpu_offload")
    )
    for method_name in offload_methods:
        method = getattr(pipe, method_name, None)
        if not callable(method):
            continue
        try:
            method()
            return
        except (NotImplementedError, RuntimeError):
            continue

    to = getattr(pipe, "to", None)
    if callable(to):
        to("cuda")


configure_cuda_allocator()
