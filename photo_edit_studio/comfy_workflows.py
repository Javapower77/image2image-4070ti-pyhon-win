from __future__ import annotations

from typing import Any


def _node(class_type: str, **inputs: Any) -> dict[str, Any]:
    return {"class_type": class_type, "inputs": inputs}


QWEN21_TURBO_SIGMAS = "1.0, 0.9375, 0.875, 0.75, 0.5, 0.25"


def qwen21_turbo_template() -> dict[str, dict[str, Any]]:
    """Six-step Viggle Qwen-Image-2.1 sampling with unmerged rank-256 LoRA."""
    from photo_edit_studio.comfy_assets import (
        QWEN21_ENCODER,
        QWEN21_TRANSFORMER,
        QWEN21_TURBO_LORA,
        QWEN21_VAE,
    )

    return {
        "1": _node("UNETLoader", unet_name=QWEN21_TRANSFORMER, weight_dtype="default"),
        "2": _node("ViggleTurboLora", model=["1", 0], lora_name=QWEN21_TURBO_LORA, strength=1.0),
        "3": _node("CLIPLoader", clip_name=QWEN21_ENCODER, type="qwen_image", device="default"),
        "4": _node("VAELoader", vae_name=QWEN21_VAE),
        "5": _node("TextEncodeQwenImage21", clip=["3", 0], vae=["4", 0],
                   prompt="", negative_prompt="", resolution=1024),
        "6": _node("EmptyLatentImage", width=1024, height=1024, batch_size=1),
        "7": _node("RandomNoise", noise_seed=0),
        "8": _node("BasicGuider", model=["2", 0], conditioning=["5", 0]),
        "9": _node("KSamplerSelect", sampler_name="euler"),
        "10": _node("ViggleTurboSigmas", latent=["6", 0], nodes=QWEN21_TURBO_SIGMAS),
        "11": _node("SamplerCustomAdvanced", noise=["7", 0], guider=["8", 0],
                    sampler=["9", 0], sigmas=["10", 0], latent_image=["6", 0]),
        "12": _node("VAEDecode", samples=["11", 0], vae=["4", 0]),
        "29": _node("SaveImage", images=["12", 0], filename_prefix="photo_edit_qwen21"),
    }


def firered_edit_template() -> dict[str, dict[str, Any]]:
    """FireRed 1.1 quantized edit; Lightning v1.2 follows the official 8-step path."""
    from photo_edit_studio.comfy_assets import (
        FIRERED_ENCODER,
        FIRERED_LIGHTNING,
        FIRERED_TRANSFORMER,
        FIRERED_VAE,
    )

    return {
        "1": _node("UnetLoaderGGUF", unet_name=FIRERED_TRANSFORMER),
        "2": _node("CLIPLoader", clip_name=FIRERED_ENCODER, type="qwen_image", device="default"),
        "3": _node("VAELoader", vae_name=FIRERED_VAE),
        "4": _node("LoraLoaderModelOnly", model=["1", 0], lora_name=FIRERED_LIGHTNING, strength_model=1.0),
        "5": _node("ModelSamplingAuraFlow", model=["4", 0], shift=3.1),
        "6": _node("CFGNorm", model=["5", 0], strength=1.0),
        "7": _node("LoadImage", image="source.png"),
        "8": _node("FluxKontextImageScale", image=["7", 0]),
        "14": _node("ImageScale", image=["7", 0], upscale_method="lanczos",
                width=1024, height=1024, crop="disabled"),
        "9": _node("VAEEncode", pixels=["14", 0], vae=["3", 0]),
        "10": _node("TextEncodeQwenImageEditPlus", clip=["2", 0], vae=["3", 0], image1=["8", 0], prompt=""),
        "11": _node("TextEncodeQwenImageEditPlus", clip=["2", 0], vae=["3", 0], image1=["8", 0], prompt=""),
        "12": _node("KSampler", model=["6", 0], positive=["10", 0], negative=["11", 0],
                    latent_image=["9", 0], seed=0, steps=8, cfg=1.0,
                    sampler_name="euler", scheduler="simple", denoise=1.0),
        "13": _node("VAEDecode", samples=["12", 0], vae=["3", 0]),
        "29": _node("SaveImage", images=["13", 0], filename_prefix="photo_edit_firered"),
    }


def krea_reference_template() -> dict[str, dict[str, Any]]:
    """Krea2Edit v1.2 two-reference graph (scene A, subject B), API format."""
    return {
        "55": _node("UNETLoader", unet_name="krea2_turbo_fp8_scaled.safetensors", weight_dtype="default"),
        "56": _node("CLIPLoader", clip_name="qwen3vl_4b_fp8_scaled.safetensors", type="krea2", device="default"),
        "57": _node("VAELoader", vae_name="qwen_image_vae.safetensors"),
        "71": _node("LoraLoaderModelOnly", model=["55", 0], lora_name="krea2_identity_edit_v1_2_r128.safetensors", strength_model=1.0),
        "72": _node("LoadImage", image="source.png"),
        "73": _node("VAEEncode", pixels=["72", 0], vae=["57", 0]),
        "90": _node("LoadImage", image="reference.png"),
        "92": _node("VAEEncode", pixels=["90", 0], vae=["57", 0]),
        "82": _node("EmptySD3LatentImage", width=1024, height=1024, batch_size=1),
        "79": _node(
            "Krea2EditModelPatch", model=["71", 0], source_latent=["73", 0],
            source_latent_b=["92", 0], vae=["57", 0], source_image=["72", 0],
            source_image_b=["90", 0], target_latent=["82", 0],
            ref_boost=4.0, ref_boost_a=1.0, fit_mode="fit",
        ),
        "84": _node(
            "Krea2EditGroundedEncode", clip=["56", 0], image=["72", 0],
            image_b=["90", 0], prompt="Edit the source using the second reference.",
            grounding_px=768,
        ),
        "85": _node(
            "Krea2EditGroundedEncode", clip=["56", 0], image=["72", 0],
            image_b=["90", 0], prompt="", grounding_px=768,
        ),
        "53": _node(
            "KSampler", model=["79", 0], positive=["84", 0], negative=["85", 0],
            latent_image=["82", 0], seed=0, steps=10, cfg=1.0,
            sampler_name="euler", scheduler="simple", denoise=1.0,
        ),
        "54": _node("VAEDecode", samples=["53", 0], vae=["57", 0]),
        "29": _node("SaveImage", images=["54", 0], filename_prefix="photo_edit_krea"),
    }


def krea_text_template() -> dict[str, dict[str, Any]]:
    """Text-only Krea Turbo graph; adapter chain is added by the Comfy bridge."""
    return {
        "55": _node("UNETLoader", unet_name="krea2_turbo_fp8_scaled.safetensors", weight_dtype="default"),
        "56": _node("CLIPLoader", clip_name="qwen3vl_4b_fp8_scaled.safetensors", type="krea2", device="default"),
        "57": _node("VAELoader", vae_name="qwen_image_vae.safetensors"),
        "82": _node("EmptySD3LatentImage", width=1024, height=1024, batch_size=1),
        "84": _node("CLIPTextEncode", clip=["56", 0], text=""),
        "85": _node("CLIPTextEncode", clip=["56", 0], text=""),
        "53": _node(
            "KSampler", model=["55", 0], positive=["84", 0], negative=["85", 0],
            latent_image=["82", 0], seed=0, steps=8, cfg=1.0,
            sampler_name="euler", scheduler="simple", denoise=1.0,
        ),
        "54": _node("VAEDecode", samples=["53", 0], vae=["57", 0]),
        "29": _node("SaveImage", images=["54", 0], filename_prefix="photo_edit_krea_text"),
    }