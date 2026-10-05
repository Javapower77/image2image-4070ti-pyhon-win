from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SAFETENSORS_MAGIC = b"{"
PREFERRED_FILENAME = "Qwen-Rapid-AIO.safetensors"


def is_safetensors_file(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size < 16:
        return False
    try:
        with path.open("rb") as handle:
            header_size = int.from_bytes(handle.read(8), "little")
            header = handle.read(1)
        return 2 <= header_size <= path.stat().st_size - 8 and header == SAFETENSORS_MAGIC
    except (OSError, ValueError):
        return False


def resolve_qwen_aio_checkpoint(directory: Path) -> Path:
    preferred = directory / PREFERRED_FILENAME
    candidates = [preferred, *sorted(directory.glob("*.safetensors"))]
    seen: set[Path] = set()
    invalid: list[Path] = []
    for candidate in candidates:
        if candidate in seen or not candidate.exists():
            continue
        seen.add(candidate)
        if is_safetensors_file(candidate):
            return candidate
        invalid.append(candidate)
    if invalid:
        raise FileNotFoundError(
            f"{invalid[0]} is not a Safetensors checkpoint (it may be an HTML download)."
        )
    raise FileNotFoundError(f"No Qwen Rapid AIO checkpoint found in {directory}.")


def transformer_state_from_comfy_aio(checkpoint: Path, dtype: Any) -> dict[str, Any]:
    from safetensors.torch import load_file

    prefix = "model.diffusion_model."
    ignored = {"__index_timestep_zero__"}

    # File metadata is diagnostic only: a failed stat must not prevent loading.
    try:
        checkpoint_bytes = checkpoint.stat().st_size
    except OSError:
        checkpoint_bytes = None
    logger.info("Qwen AIO checkpoint state load begin: checkpoint_bytes=%s", checkpoint_bytes)
    source = load_file(str(checkpoint), device="cpu")
    logger.info("Qwen AIO checkpoint state load end: source_keys=%d", len(source))
    logger.info("Qwen AIO state convert begin")
    state = {
        key.removeprefix(prefix): value.to(dtype=dtype)
        for key, value in source.items()
        if key.startswith(prefix) and key.removeprefix(prefix) not in ignored
    }
    logger.info(
        "Qwen AIO state convert end: transformer_keys=%d skipped_keys=%d",
        len(state),
        len(source) - len(state),
    )
    return state


def materialize_qwen_rope(transformer: Any) -> None:
    rope = getattr(transformer, "pos_embed", None)
    if rope is None:
        logger.info("Qwen rope materialization skipped: no positional embedding")
        return
    buffers = (getattr(rope, "pos_freqs", None), getattr(rope, "neg_freqs", None))
    if not any(getattr(value, "is_meta", False) for value in buffers):
        logger.info("Qwen rope materialization skipped: buffers already materialized")
        return
    rope_type = type(rope)
    logger.info("Qwen rope buffer rebuild begin")
    replacement = rope_type(
        rope.theta,
        rope.axes_dim,
        scale_rope=getattr(rope, "scale_rope", False),
    )
    rope.pos_freqs = replacement.pos_freqs
    rope.neg_freqs = replacement.neg_freqs
    logger.info("Qwen rope buffer rebuild end")
