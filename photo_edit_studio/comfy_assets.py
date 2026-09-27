from __future__ import annotations

COMFY_KREA_REPO = "Comfy-Org/Krea-2"
COMFY_KREA_FILES = (
    "diffusion_models/krea2_turbo_fp8_scaled.safetensors",
    "text_encoders/qwen3vl_4b_fp8_scaled.safetensors",
    "vae/qwen_image_vae.safetensors",
)
KREA_EDIT_REPO = "conradlocke/krea2-identity-edit"
KREA_EDIT_FILE = "krea2_identity_edit_v1_2_r128.safetensors"
KREA_FIRST_LORA_FILE = "Krea2_ALWAYS_LOAD_FIRST.safetensors"
KREA_FIRST_LORA_REPO = "INFOMSG/Krea2_TextFusion"
KREA_FIRST_LORA_REMOTE = "Krea2_TextFusion_Refusal_Reduction.safetensors"
KREA_TURBO_LORA_FILE = "krea2_turbo_lora_rank_64_bf16.safetensors"
KREA_TURBO_LORA_REMOTE = f"loras/{KREA_TURBO_LORA_FILE}"
FIRERED_REPO = "FireRedTeam/FireRed-Image-Edit-1.1-ComfyUI"
FIRERED_TRANSFORMER = "FireRed-Image-Edit-1.1-transformer-q4_k_m.gguf"
FIRERED_ENCODER_REPO = "Comfy-Org/Qwen-Image_ComfyUI"
FIRERED_ENCODER_REMOTE = "split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors"
FIRERED_ENCODER = "qwen_2.5_vl_7b_fp8_scaled.safetensors"
FIRERED_VAE = "qwen_image_vae.safetensors"
FIRERED_LIGHTNING = "FireRed-Image-Edit-1.1-Lightning-8steps-v1.2.safetensors"
FIRERED_FILES = (
    f"diffusion_models/{FIRERED_TRANSFORMER}",
    f"text_encoders/{FIRERED_ENCODER}",
    f"vae/{FIRERED_VAE}",
    f"loras/{FIRERED_LIGHTNING}",
)
QWEN21_COMFY_REPO = "Comfy-Org/Qwen-Image-2.1"
QWEN21_VIGGLE_REPO = "Viggle/Qwen-Image-2.1-viggle-turbo"
QWEN21_TRANSFORMER = "qwen_image_2.1_int8_convrot.safetensors"
QWEN21_ENCODER = "qwen3vl_8b_int8_convrot.safetensors"
QWEN21_VAE = "qwen_image_2.1_vae_bf16.safetensors"
QWEN21_TURBO_LORA = "Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r256.safetensors"
QWEN21_TURBO_NODE = "viggle_turbo.py"
QWEN21_FILES = (
    f"diffusion_models/{QWEN21_TRANSFORMER}",
    f"text_encoders/{QWEN21_ENCODER}",
    f"vae/{QWEN21_VAE}",
    f"loras/{QWEN21_TURBO_LORA}",
)


def missing_assets(root: object) -> list[str]:
    from pathlib import Path

    base = Path(root) / "models"
    paths = [base / file for file in COMFY_KREA_FILES]
    paths.append(base / "loras" / KREA_EDIT_FILE)
    paths.append(base / "loras" / KREA_FIRST_LORA_FILE)
    return [str(path) for path in paths if not path.is_file()]


def missing_firered_assets(root: object) -> list[str]:
    from pathlib import Path

    base = Path(root)
    paths = [base / "models" / file for file in FIRERED_FILES]
    paths.append(base / "custom_nodes" / "ComfyUI-GGUF" / "__init__.py")
    return [str(path) for path in paths if not path.is_file()]


def missing_qwen21_assets(root: object) -> list[str]:
    from pathlib import Path

    base = Path(root)
    paths = [base / "models" / file for file in QWEN21_FILES]
    paths.append(base / "custom_nodes" / QWEN21_TURBO_NODE)
    return [str(path) for path in paths if not path.is_file()]