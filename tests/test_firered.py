from __future__ import annotations

import sys
from pathlib import Path

import pytest
from PIL import Image

from photo_edit_studio.comfy_assets import (
    FIRERED_ENCODER,
    FIRERED_ENCODER_REMOTE,
    FIRERED_ENCODER_REPO,
    FIRERED_FILES,
    FIRERED_LIGHTNING,
    FIRERED_TRANSFORMER,
    KREA_FIRST_LORA_FILE,
    KREA_FIRST_LORA_REMOTE,
    KREA_FIRST_LORA_REPO,
    KREA_TURBO_LORA_FILE,
    KREA_TURBO_LORA_REMOTE,
    missing_assets,
)
from photo_edit_studio.comfy_workflows import firered_edit_template
from photo_edit_studio.models.comfy_swap import configure_firered_graph
from photo_edit_studio.types import GenerationRequest
from scripts import download_models


def request(images: int = 1, *, compose: bool = False) -> GenerationRequest:
    return GenerationRequest(
        model_key="firered-1.1", prompt="Make it blue", negative_prompt="",
        images=[Image.new("RGB", (32, 32)) for _ in range(images)], mask=None,
        width=1024, height=768, steps=8, guidance=1.0, true_cfg=1.0,
        strength=0.8, seed=42, compose=compose,
    )


def test_fire_red_template_uses_exact_lightning_and_gguf_assets() -> None:
    graph = firered_edit_template()
    assert graph["1"]["class_type"] == "UnetLoaderGGUF"
    assert graph["1"]["inputs"]["unet_name"] == FIRERED_TRANSFORMER
    assert graph["2"]["class_type"] == "CLIPLoader"
    assert graph["2"]["inputs"]["clip_name"] == FIRERED_ENCODER
    assert graph["2"]["inputs"]["type"] == "qwen_image"
    assert graph["4"]["inputs"]["lora_name"] == FIRERED_LIGHTNING
    assert graph["12"]["inputs"]["steps"] == 8
    assert graph["12"]["inputs"]["model"] == ["6", 0]


@pytest.mark.parametrize("count", [1, 2, 3])
def test_fire_red_connects_every_reference(count: int) -> None:
    graph = configure_firered_graph(
        firered_edit_template(), request(count), tuple(f"ref{i}.png" for i in range(count))
    )
    assert graph["7"]["inputs"]["image"] == "ref0.png"
    assert graph["14"]["inputs"]["width"] == 1024
    for i in range(2, count + 1):
        node = str(18 + i)
        assert graph[node]["inputs"]["image"] == f"ref{i - 1}.png"
        assert graph["10"]["inputs"][f"image{i}"] == [node, 0]
        assert graph["11"]["inputs"][f"image{i}"] == [node, 0]
    assert graph["12"]["inputs"]["latent_image"] == ["9", 0]
    assert "Do not crop, zoom in, or reframe" in graph["10"]["inputs"]["prompt"]
    assert graph["11"]["inputs"]["prompt"] == ""


def test_fire_red_combine_uses_empty_canvas() -> None:
    graph = configure_firered_graph(firered_edit_template(), request(2, compose=True), ("a.png", "b.png"))
    assert graph["14"]["class_type"] == "EmptySD3LatentImage"
    assert graph["12"]["inputs"]["latent_image"] == ["14", 0]
    assert graph["14"]["inputs"]["height"] == 768


def test_fire_red_rejects_unsupported_extra_lora() -> None:
    req = request()
    from photo_edit_studio.types import LoraSpec

    req.loras = [LoraSpec("extra.safetensors", Path("extra.safetensors"), 1.0, "extra")]
    with pytest.raises(ValueError, match="additional LoRAs"):
        configure_firered_graph(firered_edit_template(), req, ("a.png",))


def test_fire_red_downloads_only_target_files(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # The downloader receives only four exact filenames (never a full snapshot).
    monkeypatch.setattr(download_models.settings, "comfy_dir", tmp_path)
    (tmp_path / "main.py").touch()
    calls: list[str] = []

    def fake_download(*, repo_id: str, filename: str, local_dir: Path) -> str:
        calls.append(filename)
        target = local_dir / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.touch()
        return str(target)

    monkeypatch.setattr(download_models, "hf_hub_download", fake_download)
    download_models.download_comfy_firered()
    assert sorted(calls) == sorted(
        FIRERED_ENCODER_REMOTE if Path(file).name == FIRERED_ENCODER else Path(file).name
        for file in FIRERED_FILES
    )
    assert (tmp_path / "models" / "text_encoders" / FIRERED_ENCODER).is_file()
    assert FIRERED_ENCODER_REPO == "Comfy-Org/Qwen-Image_ComfyUI"


def test_krea_preset_downloads_mandatory_adapter(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(download_models.settings, "comfy_dir", tmp_path)
    (tmp_path / "main.py").touch()
    calls: list[tuple[str, str]] = []

    def fake_download(*, repo_id: str, filename: str, local_dir: Path) -> str:
        calls.append((repo_id, filename))
        target = local_dir / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.touch()
        return str(target)

    monkeypatch.setattr(download_models, "hf_hub_download", fake_download)
    download_models.download_comfy_krea()
    assert (KREA_FIRST_LORA_REPO, KREA_FIRST_LORA_REMOTE) in calls
    assert (tmp_path / "models" / "loras" / KREA_FIRST_LORA_FILE).is_file()
    assert missing_assets(tmp_path) == []
    download_models.download_comfy_krea()
    assert calls.count((KREA_FIRST_LORA_REPO, KREA_FIRST_LORA_REMOTE)) == 1


def test_optional_turbo_lora_reaches_app_library(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(download_models.settings, "lora_dir", tmp_path)
    calls: list[tuple[str, str]] = []

    def fake_download(*, repo_id: str, filename: str, local_dir: Path) -> str:
        calls.append((repo_id, filename))
        target = local_dir / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.touch()
        return str(target)

    monkeypatch.setattr(download_models, "hf_hub_download", fake_download)
    download_models.download_krea_turbo_lora()
    assert calls == [("Comfy-Org/Krea-2", KREA_TURBO_LORA_REMOTE)]
    assert (tmp_path / "krea2" / KREA_TURBO_LORA_FILE).is_file()
    download_models.download_krea_turbo_lora()
    assert len(calls) == 1


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        (["krea-2-turbo"], ["krea", "turbo"]),
        (["firered-1.1"], ["firered"]),
        (["--comfy-krea"], ["krea", "turbo"]),
        (["--comfy-firered"], ["firered"]),
        (["--turbo-lora"], ["turbo"]),
        (["krea-2-turbo", "--comfy-krea", "--turbo-lora"], ["krea", "turbo"]),
        (["firered-1.1", "--comfy-firered"], ["firered"]),
    ],
)
def test_download_cli_routes_preset_assets_once(
    monkeypatch: pytest.MonkeyPatch, args: list[str], expected: list[str]
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(sys, "argv", ["download_models.py", *args])
    monkeypatch.setattr(download_models, "download_comfy_krea", lambda: calls.append("krea"))
    monkeypatch.setattr(download_models, "download_krea_turbo_lora", lambda: calls.append("turbo"))
    monkeypatch.setattr(download_models, "download_comfy_firered", lambda: calls.append("firered"))
    download_models.main()
    assert calls == expected
