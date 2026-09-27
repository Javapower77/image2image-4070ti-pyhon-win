from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from photo_edit_studio.loras import (
    NONE_CHOICE,
    assign_new_slots,
    import_lora_files,
    list_lora_files,
    selected_loras,
)


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"lora")
    return path


def test_import_and_list_loras(tmp_path: Path) -> None:
    source = _touch(tmp_path / "uploads" / "style.safetensors")
    saved = import_lora_files("qwen-2511", [source], root=tmp_path)
    assert saved == ["style.safetensors"]
    assert list_lora_files("qwen-2511", root=tmp_path) == ["style.safetensors"]
    assert list_lora_files("firered-1.1", root=tmp_path) == []
    assert list_lora_files("flux-klein-4b", root=tmp_path) == []
    assert list_lora_files("krea-2-turbo", root=tmp_path) == []


def test_firered_rejects_additional_loras(tmp_path: Path) -> None:
    source = _touch(tmp_path / "uploads" / "style.safetensors")
    with pytest.raises(ValueError, match="automatically"):
        import_lora_files("firered-1.1", [source], root=tmp_path)
    with pytest.raises(ValueError, match="automatically"):
        selected_loras("firered-1.1", ["style.safetensors"], [1.0], root=tmp_path)


def test_selected_loras_skips_empty_zero_and_duplicates(tmp_path: Path) -> None:
    _touch(tmp_path / "qwen" / "a.safetensors")
    _touch(tmp_path / "qwen" / "b.safetensors")
    specs = selected_loras(
        "qwen-2511",
        ["a.safetensors", NONE_CHOICE, "a.safetensors", "b.safetensors", "b.safetensors"],
        [1.0, 1.0, 0.8, 0.0, 0.5],
        root=tmp_path,
    )
    assert [spec.name for spec in specs] == ["a.safetensors", "b.safetensors"]
    assert specs[0].weight == 1.0
    assert specs[1].weight == 0.5
    assert specs[0].adapter_name == "a_1"
    assert specs[1].adapter_name == "b_5"


def test_selected_loras_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not in the qwen library"):
        selected_loras("qwen-2511", ["missing.safetensors"], [1.0], root=tmp_path)


def test_import_rejects_unsupported_suffix(tmp_path: Path) -> None:
    source = _touch(tmp_path / "uploads" / "notes.txt")
    with pytest.raises(ValueError, match="Unsupported LoRA"):
        import_lora_files("qwen-2511", [source], root=tmp_path)


def test_assign_new_slots_fills_empty() -> None:
    current = [NONE_CHOICE, "kept.safetensors", NONE_CHOICE, NONE_CHOICE, NONE_CHOICE]
    assert assign_new_slots(current, ["kept.safetensors", "new.safetensors"]) == [
        "new.safetensors",
        "kept.safetensors",
        NONE_CHOICE,
        NONE_CHOICE,
        NONE_CHOICE,
    ]


def test_clone_inference_lora_parameters() -> None:
    from photo_edit_studio.models.lora_runtime import clone_inference_parameters

    module = nn.Linear(2, 2)
    with torch.inference_mode():
        module.lora_A = nn.Parameter(torch.ones(2, 2))
    assert module.lora_A.is_inference()
    assert clone_inference_parameters(module) == 1
    assert not module.lora_A.is_inference()
    module.lora_A.requires_grad_(True)


def test_activate_peft_adapters_uses_inference_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    from photo_edit_studio.models import lora_runtime

    class FakeLayer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.calls: list[tuple[str, object]] = []

        def set_adapter(self, names: list[str], inference_mode: bool = False) -> None:
            self.calls.append(("set_adapter", names, inference_mode))

        def set_scale(self, name: str, weight: float) -> None:
            self.calls.append(("set_scale", name, weight))

    layer = FakeLayer()

    class FakeRoot(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.child = layer

    pipe = SimpleNamespace(transformer=FakeRoot())
    monkeypatch.setattr(lora_runtime, "_peft_layers", lambda _pipe: [layer])
    lora_runtime.activate_peft_adapters(pipe, ["style_1"], [0.4])
    assert ("set_adapter", ["style_1"], True) in layer.calls
    assert ("set_scale", "style_1", 0.4) in layer.calls
    layer.calls.clear()
    lora_runtime.set_peft_adapter_scales(pipe, ["style_1"], [1.25])
    assert layer.calls == [("set_scale", "style_1", 1.25)]
