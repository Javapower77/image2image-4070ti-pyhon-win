from __future__ import annotations

from pathlib import Path
from typing import Any

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
    source = load_file(str(checkpoint), device="cpu")
    return {
        key.removeprefix(prefix): value.to(dtype=dtype)
        for key, value in source.items()
        if key.startswith(prefix) and key.removeprefix(prefix) not in ignored
    }


def materialize_qwen_rope(transformer: Any) -> None:
    rope = getattr(transformer, "pos_embed", None)
    if rope is None:
        return
    buffers = (getattr(rope, "pos_freqs", None), getattr(rope, "neg_freqs", None))
    if not any(getattr(value, "is_meta", False) for value in buffers):
        return
    rope_type = type(rope)
    replacement = rope_type(
        rope.theta,
        rope.axes_dim,
        scale_rope=getattr(rope, "scale_rope", False),
    )
    rope.pos_freqs = replacement.pos_freqs
    rope.neg_freqs = replacement.neg_freqs
