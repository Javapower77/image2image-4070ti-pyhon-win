from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from photo_edit_studio.config import settings


@dataclass(frozen=True, slots=True)
class ModelSpec:
    key: str
    label: str
    repo_id: str
    loader: str
    family: str
    default_steps: int
    default_guidance: float
    default_true_cfg: float
    max_images: int
    minimum_vram_gb: int
    recommended: bool
    license_note: str
    task: str = "image-to-image"

    @property
    def local_path(self) -> Path:
        return settings.model_dir / "repos" / self.repo_id.replace("/", "--")


MODEL_SPECS: dict[str, ModelSpec] = {
    "krea-2-turbo": ModelSpec(
        key="krea-2-turbo",
        label="Krea 2 Turbo (text-to-image · 12 GB offload)",
        repo_id="krea/Krea-2-Turbo",
        loader="krea2",
        family="krea2",
        default_steps=8,
        default_guidance=1.0,
        default_true_cfg=0.0,
        max_images=0,
        minimum_vram_gb=12,
        recommended=True,
        license_note="Gated Krea 2 Community License; filtering/review required",
        task="text-to-image",
    ),
    "qwen-2511": ModelSpec(
        key="qwen-2511",
        label="Qwen Image Edit 2511 (advanced / slow)",
        repo_id="Qwen/Qwen-Image-Edit-2511",
        loader="qwen",
        family="qwen",
        default_steps=40,
        default_guidance=1.0,
        default_true_cfg=4.0,
        max_images=3,
        minimum_vram_gb=12,
        recommended=False,
        license_note="Apache-2.0; sequential CPU offload required on 12 GB",
    ),
    "qwen-2511-aio": ModelSpec(
        key="qwen-2511-aio",
        label="Qwen Rapid AIO (recommended for 12 GB)",
        repo_id="Phr00t/Qwen-Image-Edit-Rapid-AIO",
        loader="qwen_aio",
        family="qwen",
        default_steps=4,
        default_guidance=1.0,
        default_true_cfg=1.0,
        max_images=3,
        minimum_vram_gb=12,
        recommended=True,
        license_note="Uses the Qwen 2511 components plus distilled AIO weights",
    ),
    "qwen-2.1-turbo": ModelSpec(
        key="qwen-2.1-turbo",
        label="Qwen Image 2.1 + Viggle Turbo (6-step · 12 GB offload)",
        repo_id="Qwen/Qwen-Image-2.1",
        loader="qwen21_comfy",
        family="qwen21",
        default_steps=6,
        default_guidance=1.0,
        default_true_cfg=1.0,
        max_images=3,
        minimum_vram_gb=12,
        recommended=False,
        license_note="Qwen Research License: non-commercial; INT8 + unmerged Viggle r256 LoRA",
    ),
    "qwen-2.1-turbo-r128": ModelSpec(
        key="qwen-2.1-turbo-r128",
        label="Qwen Image 2.1 + Turbo r128 (Civitai)",
        repo_id="Qwen/Qwen-Image-2.1",
        loader="qwen21_comfy",
        family="qwen21",
        default_steps=6,
        default_guidance=1.0,
        default_true_cfg=1.0,
        max_images=3,
        minimum_vram_gb=12,
        recommended=False,
        license_note="Qwen Research License: non-commercial; isHeSatoshi r128, tsolful compatibility modification (Civitai)",
    ),
    "firered-1.1": ModelSpec(
        key="firered-1.1",
        label="FireRed Image Edit 1.1 (GGUF Q4_K_M + Lightning)",
        repo_id="FireRedTeam/FireRed-Image-Edit-1.1-ComfyUI",
        loader="firered_comfy",
        family="firered",
        default_steps=8,
        default_guidance=1.0,
        default_true_cfg=1.0,
        max_images=3,
        minimum_vram_gb=12,
        recommended=False,
        license_note="Apache-2.0; GGUF Q4_K_M, FP8 vision encoder + Lightning v1.2; CPU offload",
    ),
    "flux-klein-4b": ModelSpec(
        key="flux-klein-4b",
        label="FLUX.2 Klein 4B (recommended for 12 GB)",
        repo_id="black-forest-labs/FLUX.2-klein-4B",
        loader="flux",
        family="flux",
        default_steps=4,
        default_guidance=1.0,
        default_true_cfg=1.0,
        max_images=3,
        minimum_vram_gb=12,
        recommended=True,
        license_note="Apache-2.0",
    ),
}

TEXT_TO_IMAGE_KEYS = ("krea-2-turbo", "flux-klein-4b", "qwen-2.1-turbo", "qwen-2.1-turbo-r128")
