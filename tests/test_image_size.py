from __future__ import annotations

import pytest

from photo_edit_studio.image_utils import (
    OUTPUT_ALIGN,
    combine_canvas_size,
    combine_output_size,
    constrain_output_size,
    scaled_output_size,
)


def test_exact_multiple_keeps_aspect() -> None:
    assert scaled_output_size((1024, 768), 1) == (1024, 768)
    assert scaled_output_size((1024, 768), 2) == (2048, 1536)


def test_multiplier_clamps_to_max_side() -> None:
    width, height = scaled_output_size((1024, 768), 3)
    assert max(width, height) == 2048
    assert width % OUTPUT_ALIGN == 0
    assert height % OUTPUT_ALIGN == 0
    assert abs((width / height) - (1024 / 768)) < 0.02


def test_portrait_and_tiny_sources() -> None:
    width, height = scaled_output_size((600, 800), 1)
    assert height >= width
    assert abs((width / height) - (600 / 800)) < 0.05
    tiny_w, tiny_h = scaled_output_size((200, 100), 1)
    assert max(tiny_w, tiny_h) >= 512
    assert tiny_w % OUTPUT_ALIGN == 0
    assert tiny_h % OUTPUT_ALIGN == 0


def test_invalid_multiplier_rejected() -> None:
    with pytest.raises(ValueError, match="multiplier"):
        scaled_output_size((1024, 1024), 4)


def test_combine_resolutions() -> None:
    assert combine_output_size("1K") == (1024, 1024)
    assert combine_output_size("2K") == (2048, 2048)
    assert combine_output_size("4K") == (4096, 4096)
    with pytest.raises(ValueError, match="resolution"):
        combine_output_size("8K")


@pytest.mark.parametrize(
    ("resolution", "aspect", "expected"),
    [
        ("1K", "1:1", (1024, 1024)),
        ("1K", "9:16", (576, 1024)),
        ("1K", "16:9", (1024, 576)),
        ("2K", "1:1", (2048, 2048)),
        ("2K", "9:16", (1152, 2048)),
        ("2K", "16:9", (2048, 1152)),
    ],
)
def test_combine_canvas_dimensions(
    resolution: str, aspect: str, expected: tuple[int, int]
) -> None:
    assert combine_canvas_size(resolution, aspect) == expected


@pytest.mark.parametrize(("resolution", "aspect"), [("4K", "1:1"), ("1K", "3:2")])
def test_combine_canvas_rejects_unsupported_choices(resolution: str, aspect: str) -> None:
    with pytest.raises(ValueError, match="Combine"):
        combine_canvas_size(resolution, aspect)


def test_low_vram_budget_caps_side_and_pixels() -> None:
    width, height = constrain_output_size(
        (2048, 1536), max_side=1024, max_pixels=1024 * 1024
    )
    assert (width, height) == (1024, 768)
    assert width * height <= 1024 * 1024


def test_low_vram_budget_preserves_safe_size() -> None:
    assert constrain_output_size(
        (768, 1024), max_side=1024, max_pixels=1024 * 1024
    ) == (768, 1024)
