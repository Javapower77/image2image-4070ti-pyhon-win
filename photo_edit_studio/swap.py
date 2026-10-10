from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from photo_edit_studio.config import settings
from photo_edit_studio.types import LoraSpec

BFS_REPO = "Alissonerdx/BFS-Best-Face-Swap"
QWEN21_BFS_HEAD_FILE = "bfs_head_v1.1_qwen_2.1.safetensors"
QWEN21_BFS_BODY_FILE = "bfs_body_swap_v1.0_qwen_2.1.safetensors"
QWEN21_BFS_PINS = {
    "Head": {"version": "1.1", "size": 260096144, "pairs": 176,
             "sha256": "d1d748d5601077f3b6d05766f6823510e901970d916afa404a97e85dc92fa88e"},
    "Body": {"version": "1.0", "size": 209753576, "pairs": 136,
             "sha256": "7dc0a53aba4dbc70c204f498936a619d226559da11011f748c389189af46b664"},
}


@dataclass(frozen=True)
class SwapProfile:
    filename: str
    family: str
    trigger: str


SWAP_PROFILES: dict[tuple[str, str], SwapProfile] = {
    ("qwen-2.1-turbo", "Head"): SwapProfile(
        QWEN21_BFS_HEAD_FILE,
        "qwen21",
        "head_swap: start with <image1> as the base image, keeping its lighting, "
        "environment, and background. remove the head from <image1> completely and "
        "replace it with the head from <image2>, strictly preserving the hair, eye "
        "color, nose structure from <image2>. copy the direction of the eye, head "
        "rotation, micro expressions from <image1>, high quality, sharp details, 4k",
    ),
    ("qwen-2.1-turbo", "Body"): SwapProfile(
        QWEN21_BFS_BODY_FILE,
        "qwen21",
        "body_swap: start with <image1> as the base image, keeping its lighting, "
        "environment, and background. replace the body from <image1> with the body "
        "from <image2>, strictly preserving pose, background, lightning and structure "
        "from <image2>. copy the pose, direction of the eye, head rotation, micro "
        "expressions from <image1>",
    ),
    ("qwen-2511", "Head"): SwapProfile(
        "bfs_head_v5_2511_merged_version_rank_16_fp16.safetensors",
        "qwen",
        "head_swap: Replace the head in Picture 1 (body) with the head from Picture 2 "
        "(face). Keep Picture 1's body, pose, lighting and background.",
    ),
    ("flux-klein-4b", "Head"): SwapProfile(
        "bfs_head_v1_flux-klein_4b.safetensors",
        "flux",
        "head_swap: Replace the head in Picture 1 (body) with the head from Picture 2 "
        "(face). Keep Picture 1's body, pose, lighting and background.",
    ),
    ("krea-2-turbo", "Head"): SwapProfile(
        "bfs_head_swap_v1.1_krea2.safetensors",
        "krea2",
        "head_swap: replace the head with the reference head.",
    ),
    ("krea-2-turbo", "Body"): SwapProfile(
        "bfs_body_swap_v1_krea2.safetensors",
        "krea2",
        "body_swap: replace the person with the reference person.",
    ),
}

# Comfy and unstacked official Turbo share BFS assets, not sampling schedules.
SWAP_PROFILES.update({
    (key, kind): SWAP_PROFILES[("qwen-2.1-turbo", kind)]
    for key in ("qwen-2.1-turbo-r128", "qwen-2.1-turbo-official")
    for kind in ("Head", "Body")
})


def validate_qwen21_bfs_file(path: Path, kind: str) -> None:
    """Check exact publisher bytes before allocating/loading any model weights."""
    pin = QWEN21_BFS_PINS[kind]
    if not path.is_file():
        raise FileNotFoundError(f"Required Qwen 2.1 BFS LoRA missing: {path}.")
    if path.stat().st_size != pin["size"]:
        raise ValueError("Qwen 2.1 BFS LoRA size mismatch.")
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != pin["sha256"]:
        raise ValueError("Qwen 2.1 BFS LoRA SHA256 mismatch.")


def swap_profile(model_key: str, kind: str) -> SwapProfile:
    try:
        return SWAP_PROFILES[(model_key, kind)]
    except KeyError as exc:
        raise ValueError(f"{model_key} does not support {kind.lower()} swap.") from exc


def swap_lora(model_key: str, kind: str, weight: float = 1.0) -> LoraSpec:
    profile = swap_profile(model_key, kind)
    path: Path = settings.lora_dir / profile.family / profile.filename
    if not path.is_file():
        hint = (
            f"Place the Qwen 2.1 {kind.lower()}-swap LoRA at {path}."
            if profile.family == "qwen21" else
            "Run scripts/download_models.py --bfs-swap."
        )
        raise FileNotFoundError(
            f"Required BFS LoRA missing: {path}. {hint}"
        )
    return LoraSpec(profile.filename, path, weight, "bfs_swap")