from __future__ import annotations

import gc
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

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
    logger.info("Memory release begin")
    gc.collect()
    if cuda_available():
        torch = _torch()
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()
        logger.info("CUDA cache release end")
    logger.info("Memory release end")


def apply_inference_memory_settings(pipe: Any, mode: str = "sequential") -> None:
    """Configure a Diffusers pipeline for a 12 GB Windows GPU."""
    logger.info("Pipeline memory configuration begin")
    vae = getattr(pipe, "vae", None)
    if vae is not None:
        for method_name in ("enable_slicing", "enable_tiling"):
            method = getattr(vae, method_name, None)
            if callable(method):
                logger.info("Pipeline VAE memory stage begin: method=%s", method_name)
                method()
                logger.info("Pipeline VAE memory stage end: method=%s", method_name)

    enable_attention_slicing = getattr(pipe, "enable_attention_slicing", None)
    if callable(enable_attention_slicing):
        logger.info("Pipeline attention slicing begin")
        enable_attention_slicing("max")
        logger.info("Pipeline attention slicing end")

    offload_methods = (
        ("enable_sequential_cpu_offload", "enable_model_cpu_offload")
        if mode == "sequential"
        else ("enable_model_cpu_offload", "enable_sequential_cpu_offload")
    )
    for method_name in offload_methods:
        method = getattr(pipe, method_name, None)
        if not callable(method):
            logger.info("Pipeline offload unavailable: method=%s", method_name)
            continue
        try:
            logger.info("Pipeline offload begin: method=%s", method_name)
            method()
            logger.info("Pipeline offload end: method=%s", method_name)
            logger.info("Pipeline memory configuration end")
            return
        except (NotImplementedError, RuntimeError):
            logger.warning(
                "Pipeline offload failed: method=%s; trying the next fallback",
                method_name,
                exc_info=True,
            )
            continue

    to = getattr(pipe, "to", None)
    if callable(to):
        logger.warning("Pipeline CPU offload unavailable or failed; falling back to CUDA")
        logger.info("Pipeline CUDA placement begin")
        to("cuda")
        logger.info("Pipeline CUDA placement end")
    logger.info("Pipeline memory configuration end")


configure_cuda_allocator()
