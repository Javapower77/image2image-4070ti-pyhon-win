from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any

from photo_edit_studio.comfy_assets import QWEN21_R128_TURBO_LORA, QWEN21_TURBO_LORA
from photo_edit_studio.config import settings
from photo_edit_studio.models.registry import MODEL_SPECS
from photo_edit_studio.types import LoraSpec, validate_optional_lora_weight

MAX_LORAS = 5
NONE_CHOICE = "(none)"
LORA_SUFFIXES = {".safetensors", ".pt", ".bin"}
QWEN21_REQUIRED_LORAS = {
    QWEN21_TURBO_LORA.casefold(), QWEN21_R128_TURBO_LORA.casefold(),
}


def _qwen21_turbo_label(model_key: str) -> str:
    return "Civitai Turbo r128" if model_key == "qwen-2.1-turbo-r128" else "Viggle Turbo"


def family_for(model_key: str) -> str:
    if model_key not in MODEL_SPECS:
        raise KeyError(f"Unknown model: {model_key}")
    return MODEL_SPECS[model_key].family


def library_dir(model_key: str, root: Path | None = None) -> Path:
    path = (root or settings.lora_dir) / family_for(model_key)
    path.mkdir(parents=True, exist_ok=True)
    return path


def list_lora_files(model_key: str, root: Path | None = None) -> list[str]:
    if model_key == "firered-1.1":
        return []  # Lightning is loaded automatically.
    directory = library_dir(model_key, root)
    qwen21 = family_for(model_key) == "qwen21"
    suffixes = {".safetensors"} if qwen21 else LORA_SUFFIXES
    names = [
        path.name
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in suffixes
        and (not qwen21 or path.name.casefold() not in QWEN21_REQUIRED_LORAS)
    ]
    return sorted(names, key=str.lower)


def lora_choices(model_key: str, root: Path | None = None) -> list[str]:
    return [NONE_CHOICE, *list_lora_files(model_key, root)]


def library_status(model_key: str, saved: list[str] | None = None, root: Path | None = None) -> str:
    family = family_for(model_key)
    names = list_lora_files(model_key, root)
    parts = [f"**{family}** LoRA library: {len(names)} file(s) in `{library_dir(model_key, root)}`."]
    if saved:
        parts.append("Saved: " + ", ".join(saved) + ".")
    parts.append(
        "Qwen 2511, Qwen 2.1, FLUX.2 Klein, Krea 2 and FireRed have separate libraries. "
        "FireRed Lightning loads automatically."
    )
    if family == "qwen21":
        parts.append(
            f"{MODEL_SPECS[model_key].label} always loads {_qwen21_turbo_label(model_key)} "
            "first, then selected Qwen 2.1 .safetensors adapters from this shared library. "
            "Both mandatory Turbo files are excluded from optional slots. "
            "Head/Body swaps require the matching BFS adapter, plus at most four optional LoRAs."
        )
    return " ".join(parts)


def import_lora_files(
    model_key: str,
    files: Any,
    root: Path | None = None,
) -> list[str]:
    if model_key == "firered-1.1":
        raise ValueError("This model loads its required LoRA automatically; additional LoRAs are unsupported.")
    saved: list[str] = []
    target = library_dir(model_key, root)
    for source in _as_paths(files):
        if family_for(model_key) == "qwen21" and (
            source.suffix.lower() != ".safetensors"
            or source.name.casefold() in QWEN21_REQUIRED_LORAS
        ):
            raise ValueError(
                "Qwen 2.1 optional LoRAs must be .safetensors and cannot duplicate "
                "Viggle Turbo or Civitai Turbo r128."
            )
        if source.suffix.lower() not in LORA_SUFFIXES:
            raise ValueError(
                f"Unsupported LoRA file '{source.name}'. Use .safetensors, .pt, or .bin."
            )
        destination = target / _safe_filename(source.name)
        shutil.copy2(source, destination)
        saved.append(destination.name)
    return saved


def selected_loras(
    model_key: str,
    names: list[str | None],
    weights: list[float],
    root: Path | None = None,
) -> list[LoraSpec]:
    if len(names) != len(weights):
        raise ValueError("Each LoRA slot needs a matching weight.")
    if model_key == "firered-1.1" and any(
        name and name != NONE_CHOICE and float(weight) != 0
        for name, weight in zip(names, weights, strict=True)
    ):
        raise ValueError("This model loads its required LoRA automatically; additional LoRAs are unsupported.")
    selected: list[LoraSpec] = []
    seen: set[str] = set()
    for index, (name, weight) in enumerate(zip(names, weights, strict=True), start=1):
        label = (name or NONE_CHOICE).strip()
        if not label or label == NONE_CHOICE or float(weight) == 0:
            continue
        weight = validate_optional_lora_weight(weight)
        if label in seen:
            continue
        if family_for(model_key) == "qwen21" and (
            Path(label).suffix.lower() != ".safetensors"
            or Path(label).name.casefold() in QWEN21_REQUIRED_LORAS
        ):
            raise ValueError(
                "Only optional Qwen 2.1 .safetensors LoRAs may follow "
                f"{_qwen21_turbo_label(model_key)}; neither mandatory Turbo file is optional."
            )
        path = library_dir(model_key, root) / Path(label).name
        if not path.exists():
            raise ValueError(f"LoRA '{label}' is not in the {family_for(model_key)} library.")
        selected.append(
            LoraSpec(
                name=path.name,
                path=path,
                weight=float(weight),
                adapter_name=_adapter_name(path.stem, index),
            )
        )
        seen.add(label)
        if len(selected) >= MAX_LORAS:
            break
    return selected


def assign_new_slots(current: list[str], uploaded: list[str]) -> list[str]:
    names = [(name or NONE_CHOICE) for name in current]
    while len(names) < MAX_LORAS:
        names.append(NONE_CHOICE)
    names = names[:MAX_LORAS]
    for filename in uploaded:
        if filename in names:
            continue
        try:
            empty = names.index(NONE_CHOICE)
        except ValueError:
            break
        names[empty] = filename
    return names


def _adapter_name(stem: str, index: int) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", stem).strip("_")
    if not cleaned:
        cleaned = f"lora_{index}"
    if cleaned[0].isdigit():
        cleaned = f"lora_{cleaned}"
    return f"{cleaned}_{index}"


def _safe_filename(name: str) -> str:
    filename = Path(name).name
    if filename in {"", ".", ".."} or filename != Path(filename).name:
        raise ValueError("Invalid LoRA filename.")
    return filename


def _as_paths(files: Any) -> list[Path]:
    if not files:
        return []
    items = files if isinstance(files, list) else [files]
    paths: list[Path] = []
    for item in items:
        if item is None:
            continue
        raw: Any = item
        if isinstance(item, dict):
            raw = item.get("path") or item.get("name")
        elif hasattr(item, "name") and not isinstance(item, (str, Path)):
            raw = item.name
        path = Path(str(raw))
        if not path.exists() or not path.is_file():
            raise ValueError(f"Uploaded LoRA file is missing: {path}")
        paths.append(path)
    return paths
