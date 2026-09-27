from __future__ import annotations

import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PHOTO_EDIT_", env_file=".env", extra="ignore")

    host: str = "127.0.0.1"
    port: int = 7860
    share: bool = False
    model_dir: Path = ROOT / "models"
    output_dir: Path = ROOT / "outputs"
    lora_dir: Path = ROOT / "models" / "loras"
    hf_home: Path = ROOT / "models" / "huggingface"
    max_concurrency: int = 1
    memory_mode: str = "sequential"
    max_output_side: int = 1024
    max_output_pixels: int = 1024 * 1024
    combine_max_output_side: int = 2048
    combine_max_output_pixels: int = 2048 * 2048
    max_batch_count: int = 1
    allow_unsupported_models: bool = False
    flux_klein_4b_text_encoder: Path = ROOT / "models" / "qwen3-4b-alb-q4_0.gguf"
    comfy_url: str = "http://127.0.0.1:8188"
    comfy_dir: Path = ROOT / "vendor" / "ComfyUI"
    comfy_autostart: bool = False
    comfy_start_timeout_seconds: int = 180
    comfy_head_workflow: Path = ROOT / "models" / "workflows" / "bfs_krea_head_api.json"
    comfy_body_workflow: Path = ROOT / "models" / "workflows" / "bfs_krea_body_api.json"
    comfy_reference_workflow: Path = ROOT / "models" / "workflows" / "krea_reference_api.json"
    comfy_reference_lora: str = "krea2_identity_edit_v1_2_r128.safetensors"
    comfy_timeout_seconds: int = 900
    auth_user: str | None = None
    auth_password: str | None = None

    def prepare(self) -> None:
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.lora_dir.mkdir(parents=True, exist_ok=True)
        self.hf_home.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("HF_HOME", str(self.hf_home.resolve()))
        os.environ.setdefault("HF_HUB_CACHE", str((self.hf_home / "hub").resolve()))
        os.environ.setdefault("TRANSFORMERS_CACHE", str((self.hf_home / "transformers").resolve()))
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        os.environ.setdefault(
            "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:128"
        )
        os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")
        os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")


settings = Settings()
settings.prepare()
