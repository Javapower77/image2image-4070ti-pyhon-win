from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from pathlib import Path
from typing import Any

from PIL import Image


def validate_optional_lora_weight(weight: float) -> float:
    value = float(weight)
    if not isfinite(value) or not -2 <= value <= 2:
        raise ValueError("Optional LoRA weight must be finite and between -2 and 2.")
    return value


@dataclass(frozen=True, slots=True)
class LoraSpec:
    name: str
    path: Path
    weight: float
    adapter_name: str


@dataclass(slots=True)
class GenerationRequest:
    model_key: str
    prompt: str
    negative_prompt: str
    images: list[Image.Image]
    mask: Image.Image | None
    width: int
    height: int
    steps: int
    guidance: float
    true_cfg: float
    strength: float
    seed: int
    count: int = 1
    preserve_identity: bool = True
    size_multiplier: int = 1
    compose: bool = False
    output_resolution: str = "1K"
    output_aspect_ratio: str = "1:1"
    loras: list[LoraSpec] = field(default_factory=list)
    workflow: str = "standard"
    swap_kind: str | None = None
    krea_first_lora_weight: float = 1.0
    dlss: dict[str, Any] | None = None


@dataclass(slots=True)
class GenerationResult:
    images: list[Image.Image]
    seed: int
    elapsed_seconds: float
    model_key: str
    saved_paths: list[Path] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    raw: Any = None
