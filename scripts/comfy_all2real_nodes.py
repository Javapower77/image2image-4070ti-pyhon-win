"""Project-owned legacy ComfyUI nodes; copy to custom_nodes/photo_edit_all2real.py.

Importable without torch or ComfyUI. No extension dependencies beyond ComfyUI's
own upscale-model implementation. This module does not install or load assets.
"""

from __future__ import annotations

import math
from copy import copy
from types import MethodType
from typing import Any


def _decode_tiled_3d(vae: Any, samples: Any, tile_t: int = 999,
                     tile_x: int = 32, tile_y: int = 32,
                     overlap: tuple[int, int, int] = (1, 8, 8)) -> Any:
    """Local sd.py VAE.decode_tiled_3d, allocating packed decoder channels.

    Behavior reference: spacepxl/ComfyUI-VAE-Utils, vae_patch.py
    https://github.com/spacepxl/ComfyUI-VAE-Utils/blob/main/vae_patch.py
    Only the narrow tiled allocation fix is applied; no broad VAE patching.
    """
    from comfy.utils import tiled_scale_multidim

    def decode_fn(value: Any) -> Any:
        return vae.first_stage_model.decode(
            value.to(vae.vae_dtype).to(vae.device)
        ).to(dtype=vae.vae_output_dtype())

    return vae.process_output(tiled_scale_multidim(
        samples, decode_fn, tile=(tile_t, tile_x, tile_y), overlap=overlap,
        upscale_amount=vae.upscale_ratio, out_channels=vae.conv_out_channels,
        index_formulas=vae.upscale_index_formula, output_device=vae.output_device,
    ))


class StudioAll2RealVAEDecode:
    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {"required": {"samples": ("LATENT",), "vae": ("VAE",)}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "decode"
    CATEGORY = "photo_edit/all2real"

    def decode(self, samples: Any, vae: Any) -> tuple[Any]:
        """Decode on a shallow copy, then unpack normalized packed RGB.

        Confirmed Wan weights: encoder.conv1.weight [96,3,3,3,3],
        decoder.head.2.weight [12,96,3,3,3]. Core returns BHWC12,
        not native RGB upscale. Finishing follows the vae_patch.py reference.
        """
        import torch.nn.functional as F

        if getattr(vae, "output_channels", 3) != 3:
            raise ValueError("All2Real VAE decode requires RGB output_channels=3.")
        copied = copy(vae)
        if (getattr(copied, "latent_dim", None) == 3
                and getattr(copied, "output_channels", None) == 3
                and getattr(copied, "conv_out_channels", None) == 12):
            # Core decode's automatic OOM fallback dispatches on this copy.
            copied.decode_tiled_3d = MethodType(_decode_tiled_3d, copied)
        images = copied.decode(samples["samples"])
        if images.ndim == 5:
            images = images.reshape(-1, *images.shape[-3:])
        if images.ndim != 4:
            raise ValueError("All2Real VAE decode requires BHWC or BTHWC output.")
        channels = images.shape[-1]
        ratio_squared, remainder = divmod(channels, 3)
        ratio = math.isqrt(ratio_squared)
        if remainder or ratio < 1 or 3 * ratio * ratio != channels:
            raise ValueError(
                f"Cannot unpack {channels} All2Real decoder channels into RGB: "
                "expected 3*r*r channels for an integer pixel-shuffle ratio."
            )
        # Core normally already normalizes. Never apply that conversion twice.
        if images.numel() > 0 and images.min() < -0.1:
            images = ((images.float() + 1.0) / 2.0).clamp(0.0, 1.0)
        if channels != 3:
            images = F.pixel_shuffle(images.movedim(-1, 1), ratio).movedim(1, -1)
        return (images,)


class StudioAll2RealRemoveReferences:
    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {"required": {"conditioning": ("CONDITIONING",)}}

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "remove_references"
    CATEGORY = "photo_edit/all2real"

    def remove_references(self, conditioning: Any) -> tuple[Any]:
        """Copy entries/metadata, preserving embeddings and unrelated fields."""
        result = []
        for embedding, metadata in conditioning:
            copied = metadata.copy()
            copied.pop("reference_latents", None)
            copied.pop("reference_latents_method", None)
            result.append([embedding, copied])
        return (result,)


class StudioAll2RealLatentNoise:
    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {"required": {
            "samples": ("LATENT",),
            "noise_std": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 2.0, "step": 0.01}),
            "seed": ("INT", {"default": 10, "min": 0, "max": 2**64 - 1}),
        }}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "inject_noise"
    CATEGORY = "photo_edit/all2real"

    def inject_noise(self, samples: Any, noise_std: float, seed: int) -> tuple[Any]:
        import torch

        if not math.isfinite(noise_std) or not 0 <= noise_std <= 2:
            raise ValueError("Latent noise standard deviation must be finite and between 0 and 2.")
        if type(seed) is not int or not 0 <= seed <= 2**64 - 1:
            raise ValueError("Latent noise seed must be an unsigned 64-bit integer.")
        original = samples["samples"]
        if not original.is_floating_point():
            raise ValueError("All2Real requires floating-point latent samples.")
        result = samples.copy()
        latent = original.clone()
        generator = torch.Generator(device="cpu").manual_seed(seed)
        # Sample in float32 on CPU before converting to the latent's dtype/device.
        # Neither the global RNG nor the caller's tensor/metadata is modified.
        noise = torch.randn(latent.shape, generator=generator, device="cpu", dtype=torch.float32)
        result["samples"] = latent + noise.to(device=latent.device, dtype=latent.dtype) * noise_std
        return (result,)


class StudioAll2RealSkinDetail:
    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {"required": {"image": ("IMAGE",), "upscale_model": ("UPSCALE_MODEL",)}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "enhance"
    CATEGORY = "photo_edit/all2real"

    def enhance(self, image: Any, upscale_model: Any) -> tuple[Any]:
        """Enhance TL/TR/BL/BR quadrants, then concatenate right, right, down.

        Keep the source batch dimension and all pixels, including odd edges.
        Each quadrant is processed independently using ComfyUI's native tiling.
        The required SkinDiffDetail asset is 1x, not a further resolution upscale.
        """
        import torch
        from comfy_extras.nodes_upscale_model import ImageUpscaleWithModel

        if image.ndim != 4 or image.shape[0] < 1 or image.shape[1] < 2 or image.shape[2] < 2:
            raise ValueError("Skin detail requires a nonempty BHWC image at least 2x2.")
        if image.shape[-1] != 3:
            raise ValueError(
                f"Skin detail requires final RGB3, got {image.shape[-1]} channels. "
                "Use StudioAll2RealVAEDecode to unpack the Wan VAE output first."
            )
        if float(upscale_model.scale) != 1.0:
            raise ValueError("All2Real skin detail requires the original 1x SkinDiffDetail model.")
        height, width = image.shape[1:3]
        mid_h, mid_w = height // 2, width // 2
        tiles = (
            image[:, :mid_h, :mid_w, :], image[:, :mid_h, mid_w:, :],
            image[:, mid_h:, :mid_w, :], image[:, mid_h:, mid_w:, :],
        )
        upscaler = ImageUpscaleWithModel()
        enhanced = []
        for tile in tiles:
            # Verified local API: upscale(upscale_model, image) aliases execute
            # and returns an indexable io.NodeOutput (legacy returns a tuple).
            output = upscaler.upscale(upscale_model=upscale_model, image=tile.contiguous())[0]
            if tuple(output.shape) != tuple(tile.shape):
                raise ValueError("The required 1x skin model must preserve each tile's shape.")
            enhanced.append(output)
        top = torch.cat((enhanced[0], enhanced[1]), dim=2)
        bottom = torch.cat((enhanced[2], enhanced[3]), dim=2)
        return (torch.cat((top, bottom), dim=1),)


NODE_CLASS_MAPPINGS = {
    "StudioAll2RealVAEDecode": StudioAll2RealVAEDecode,
    "StudioAll2RealRemoveReferences": StudioAll2RealRemoveReferences,
    "StudioAll2RealLatentNoise": StudioAll2RealLatentNoise,
    "StudioAll2RealSkinDetail": StudioAll2RealSkinDetail,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StudioAll2RealVAEDecode": "Studio All2Real Packed RGB VAE Decode",
    "StudioAll2RealRemoveReferences": "Studio All2Real Remove References",
    "StudioAll2RealLatentNoise": "Studio All2Real Seeded Latent Noise",
    "StudioAll2RealSkinDetail": "Studio All2Real Four-Tile Skin Detail",
}