"""Local diagnostics without prompt, image, credential or tensor logging."""
from __future__ import annotations

import atexit
import faulthandler
import logging
import os
import sys
import threading
from logging.handlers import RotatingFileHandler
from pathlib import Path

_fault_file = None
_configured = False


def memory_snapshot() -> str:
    parts = []
    try:
        import psutil

        process = psutil.Process()
        memory = process.memory_info()
        system = psutil.virtual_memory()
        parts.extend((f"rss_mib={memory.rss / 2**20:.0f}",
                      f"available_ram_mib={system.available / 2**20:.0f}"))
        if hasattr(memory, "private"):
            parts.append(f"private_mib={memory.private / 2**20:.0f}")
    except Exception:  # noqa: BLE001 - diagnostics must not interrupt inference
        parts.append("cpu_memory=unavailable")
    # Never import torch or initialize CUDA just to collect diagnostics.
    torch = sys.modules.get("torch")
    try:
        if torch is not None and torch.cuda.is_initialized():
            parts.extend((f"cuda_allocated_mib={torch.cuda.memory_allocated() / 2**20:.0f}",
                          f"cuda_reserved_mib={torch.cuda.memory_reserved() / 2**20:.0f}"))
    except Exception:  # noqa: BLE001 - CUDA diagnostics are best effort
        parts.append("cuda_memory=unavailable")
    return " ".join(parts)


class _MemoryFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.memory = memory_snapshot()
        return True


def configure_logging(log_dir: Path | None = None) -> Path:
    global _configured, _fault_file
    directory = log_dir or Path(__file__).resolve().parents[1] / "logs"
    directory.mkdir(parents=True, exist_ok=True)
    if _configured:
        return directory
    handler = RotatingFileHandler(directory / "studio.log", maxBytes=10 * 1024 * 1024,
                                  backupCount=5, encoding="utf-8")
    handler.addFilter(_MemoryFilter())
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s pid=%(process)d thread=%(threadName)s "
        "%(name)s %(message)s | %(memory)s"
    ))
    # Scope detailed logs to application code; third-party HTTP logs can contain inputs.
    logger = logging.getLogger("photo_edit_studio")
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    _fault_file = (directory / f"fault-{os.getpid()}.log").open("a", encoding="utf-8")
    faulthandler.enable(file=_fault_file, all_threads=True)
    previous_hook = sys.excepthook

    def uncaught(kind, value, traceback):
        logger.critical("Uncaught main-thread exception", exc_info=(kind, value, traceback))
        previous_hook(kind, value, traceback)

    sys.excepthook = uncaught
    previous_thread_hook = threading.excepthook

    def thread_uncaught(args):
        logger.critical("Uncaught thread exception: %s", args.thread.name,
                        exc_info=(args.exc_type, args.exc_value, args.exc_traceback))
        previous_thread_hook(args)

    threading.excepthook = thread_uncaught
    atexit.register(lambda: logger.info("Python interpreter shutdown (atexit)"))
    _configured = True
    logger.info("Diagnostics enabled: Python %s", sys.version.split()[0])
    return directory