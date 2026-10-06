from __future__ import annotations

from PIL import Image, ImageFilter, ImageOps

MIN_OUTPUT_SIDE = 512
MAX_OUTPUT_SIDE = 2048
OUTPUT_ALIGN = 64
SIZE_MULTIPLIERS = (1, 2, 3)
COMBINE_RESOLUTIONS = {
    "1K": (1024, 1024),
    "2K": (2048, 2048),
    "4K": (4096, 4096),
}
COMBINE_ASPECT_RATIOS = {
    "1:1": (1, 1),
    "9:16": (9, 16),
    "16:9": (16, 9),
}


def combine_output_size(resolution: str) -> tuple[int, int]:
    try:
        return COMBINE_RESOLUTIONS[resolution]
    except KeyError as exc:
        raise ValueError(
            f"Combine resolution must be one of {tuple(COMBINE_RESOLUTIONS)}."
        ) from exc


def combine_canvas_size(resolution: str, aspect_ratio: str) -> tuple[int, int]:
    """Fit a combine canvas within the selected longest side, aligned to 64 pixels."""
    if resolution not in {"1K", "2K"}:
        raise ValueError("Combine resolution must be 1K or 2K.")
    try:
        ratio_width, ratio_height = COMBINE_ASPECT_RATIOS[aspect_ratio]
    except KeyError as exc:
        raise ValueError("Combine aspect ratio must be 1:1, 9:16, or 16:9.") from exc
    side = combine_output_size(resolution)[0]
    if ratio_width >= ratio_height:
        width, height = side, side * ratio_height / ratio_width
    else:
        width, height = side * ratio_width / ratio_height, side
    return (
        max(OUTPUT_ALIGN, round(width / OUTPUT_ALIGN) * OUTPUT_ALIGN),
        max(OUTPUT_ALIGN, round(height / OUTPUT_ALIGN) * OUTPUT_ALIGN),
    )


def constrain_output_size(
    size: tuple[int, int],
    *,
    max_side: int,
    max_pixels: int,
    align: int = OUTPUT_ALIGN,
) -> tuple[int, int]:
    """Reduce a requested size to fit a VRAM-oriented side and pixel budget."""
    width, height = size
    if width <= 0 or height <= 0 or max_side <= 0 or max_pixels <= 0:
        raise ValueError("Output dimensions and limits must be positive.")
    scale = min(1.0, max_side / max(width, height), (max_pixels / (width * height)) ** 0.5)
    width = max(align, int(width * scale) // align * align)
    height = max(align, int(height * scale) // align * align)
    return width, height


def scaled_output_size(
    size: tuple[int, int],
    multiplier: int,
    *,
    min_side: int = MIN_OUTPUT_SIDE,
    max_side: int = MAX_OUTPUT_SIDE,
    align: int = OUTPUT_ALIGN,
) -> tuple[int, int]:
    """Scale an image size by ×1/×2/×3 while keeping aspect ratio within model limits."""
    if multiplier not in SIZE_MULTIPLIERS:
        raise ValueError(f"Size multiplier must be one of {SIZE_MULTIPLIERS}.")
    width, height = size
    if width <= 0 or height <= 0:
        raise ValueError("Image has invalid dimensions.")
    width *= multiplier
    height *= multiplier
    longest = max(width, height)
    if longest > max_side:
        scale = max_side / longest
        width *= scale
        height *= scale
    longest = max(width, height)
    if longest < min_side:
        scale = min_side / longest
        width *= scale
        height *= scale
    longest = max(width, height)
    if longest > max_side:
        scale = max_side / longest
        width *= scale
        height *= scale
    width = int(max(align, round(width / align) * align))
    height = int(max(align, round(height / align) * align))
    width = min(max(align, (width // align) * align), max_side)
    height = min(max(align, (height // align) * align), max_side)
    return width, height


def diffusion_output_size(
    size: tuple[int, int],
    multiplier: int = 1,
    *,
    family: str = "",
    workflow: str = "standard",
    canvas: bool = False,
    pre_sized: bool = False,
) -> tuple[int, int]:
    """Apply the shared diffusion budget, independently of DLSS enhancement.

    Source sizes are original upload dimensions: scale before normalization.
    The 2048 ceiling retains the 1024 working-side ×2 sizing semantic.
    Canvas dimensions are already selected/aligned and must not be multiplied.
    """
    from photo_edit_studio.config import settings

    requested = size if canvas or pre_sized else scaled_output_size(size, multiplier)
    if canvas:
        max_side, max_pixels = settings.combine_max_output_side, settings.combine_max_output_pixels
    elif family == "qwen21" and workflow in {"standard", "swap"} and multiplier == 2:
        max_side, max_pixels = settings.qwen21_x2_max_output_side, settings.qwen21_x2_max_output_pixels
    else:
        max_side, max_pixels = settings.max_output_side, settings.max_output_pixels
    return constrain_output_size(requested, max_side=max_side, max_pixels=max_pixels)


def normalize_image(image: Image.Image, max_side: int = 2048) -> Image.Image:
    image = ImageOps.exif_transpose(image).convert("RGB")
    if max(image.size) > max_side:
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return image


def fit_reference(image: Image.Image, width: int, height: int) -> Image.Image:
    return ImageOps.contain(normalize_image(image), (width, height), Image.Resampling.LANCZOS)


def composite_with_mask(
    source: Image.Image, generated: Image.Image, mask: Image.Image, feather: int = 16
) -> Image.Image:
    target_size = generated.size
    source = ImageOps.fit(normalize_image(source), target_size, Image.Resampling.LANCZOS)
    mask = ImageOps.fit(mask.convert("L"), target_size, Image.Resampling.LANCZOS)
    if feather:
        mask = mask.filter(ImageFilter.GaussianBlur(feather))
    return Image.composite(generated.convert("RGB"), source, mask)
