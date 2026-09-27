from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from photo_edit_studio.config import settings
from photo_edit_studio.types import LoraSpec

BFS_REPO = "Alissonerdx/BFS-Best-Face-Swap"
QWEN21_BFS_HEAD_FILE = "Qwen21-BFS_Head_v1.1.safetensors"


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
        "environment and background. Replace the head from <image1> with the head "
        "from <image2>, preserving hair, eye color, and facial features from <image2>. "
        "Keep the head angle and expression from <image1>.",
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
            f"Place the Qwen 2.1 head-swap LoRA at {path}."
            if model_key == "qwen-2.1-turbo" else
            "Run scripts/download_models.py --bfs-swap."
        )
        raise FileNotFoundError(
            f"Required BFS LoRA missing: {path}. {hint}"
        )
    return LoraSpec(profile.filename, path, weight, "bfs_swap")