from __future__ import annotations

import importlib
from typing import Any

import numpy as np
from PIL import Image

from photo_edit_studio.config import settings

_RESTORERS: dict[str, Any] = {}


def restore_faces(image: Image.Image, method: str, weight: float) -> Image.Image:
    if method == "Off":
        return image
    if method != "GFPGAN":
        raise RuntimeError(
            f"{method} is not installed. The UI exposes only verified local restorers."
        )
    restorer = _load_gfpgan()
    cv_image = np.asarray(image.convert("RGB"))[:, :, ::-1]
    _, _, restored = restorer.enhance(
        cv_image, has_aligned=False, only_center_face=False, paste_back=True, weight=weight
    )
    return Image.fromarray(restored[:, :, ::-1])


def _load_gfpgan() -> Any:
    if "GFPGAN" in _RESTORERS:
        return _RESTORERS["GFPGAN"]
    try:
        module = importlib.import_module("gfpgan")
    except ImportError as exc:
        raise RuntimeError("Install restoration extras: pip install -e '.[restore]'") from exc
    checkpoint = settings.model_dir / "restorers" / "GFPGANv1.4.pth"
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"GFPGAN checkpoint missing at {checkpoint}. Run download_models.py --restorers."
        )
    restorer = module.GFPGANer(
        model_path=str(checkpoint),
        upscale=1,
        arch="clean",
        channel_multiplier=2,
        bg_upsampler=None,
    )
    _RESTORERS["GFPGAN"] = restorer
    return restorer
