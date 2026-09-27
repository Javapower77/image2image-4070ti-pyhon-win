from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
from PIL import Image

from photo_edit_studio.comfy_assets import (
    QWEN21_COMFY_REPO,
    QWEN21_FILES,
    QWEN21_TURBO_LORA,
    QWEN21_TURBO_NODE,
    QWEN21_VIGGLE_REPO,
    missing_qwen21_assets,
)
from photo_edit_studio.comfy_workflows import QWEN21_TURBO_SIGMAS, qwen21_turbo_template
from photo_edit_studio.loras import (
    NONE_CHOICE,
    import_lora_files,
    list_lora_files,
    selected_loras,
)
from photo_edit_studio.models.comfy_swap import configure_qwen21_graph, resolve_qwen21_lora_names
from photo_edit_studio.swap import QWEN21_BFS_HEAD_FILE, swap_lora
from photo_edit_studio.types import GenerationRequest, LoraSpec
from photo_edit_studio.ui import (
    TEXT_MODE,
    _model_choices_for_mode,
    _run_swap,
    _swap_model_changed,
    _turbo_controls_for_model,
)
from scripts import download_models


def request(images: int = 0, *, compose: bool = False, workflow: str = "standard", kind: str | None = None) -> GenerationRequest:
    return GenerationRequest(
        model_key="qwen-2.1-turbo", prompt="Change the colors", negative_prompt="",
        images=[Image.new("RGB", (800, 600)) for _ in range(images)], mask=None,
        width=1024, height=768, steps=6, guidance=1.0, true_cfg=1.0,
        strength=0.8, seed=17, compose=compose, workflow=workflow, swap_kind=kind,
    )


def test_qwen21_template_has_unmerged_r256_and_matching_sampler_sigmas() -> None:
    graph = qwen21_turbo_template()
    assert graph["1"]["inputs"]["unet_name"] == "qwen_image_2.1_int8_convrot.safetensors"
    assert graph["2"]["class_type"] == "ViggleTurboLora"
    assert graph["2"]["inputs"]["lora_name"] == QWEN21_TURBO_LORA
    assert graph["2"]["inputs"]["strength"] == 1.0
    assert graph["10"]["inputs"]["nodes"] == QWEN21_TURBO_SIGMAS
    assert graph["10"]["inputs"]["latent"] == graph["11"]["inputs"]["latent_image"]
    assert graph["11"]["inputs"]["guider"] == ["8", 0]
    assert graph["5"]["class_type"] == "TextEncodeQwenImage21"


@pytest.mark.parametrize("images,compose,workflow,kind", [
    (0, True, "text", None), (1, False, "standard", None),
    (3, True, "standard", None), (2, False, "swap", "Head"), (2, False, "swap", "Body"),
])
def test_qwen21_wires_all_four_workflows(
    images: int, compose: bool, workflow: str, kind: str | None
) -> None:
    req = request(images, compose=compose, workflow=workflow, kind=kind)
    graph = configure_qwen21_graph(qwen21_turbo_template(), req, tuple(f"image{i}.png" for i in range(images)))
    assert graph["6"]["inputs"]["width"] == req.width
    assert graph["6"]["inputs"]["height"] == req.height
    assert graph["10"]["inputs"]["latent"] == graph["11"]["inputs"]["latent_image"]
    for i in range(1, images + 1):
        assert graph["5"]["inputs"][f"images.image_{i}"] == [str(30 + i), 0]
    assert not any(node["class_type"] == "LoraLoaderModelOnly" for node in graph.values())


def test_qwen21_rejects_incompatible_student_controls() -> None:
    for field, value in (("steps", 4), ("guidance", 4), ("true_cfg", 4), ("negative_prompt", "bad")):
        req = request(1)
        setattr(req, field, value)
        with pytest.raises(ValueError):
            configure_qwen21_graph(qwen21_turbo_template(), req, ("a.png",))
    with pytest.raises(ValueError, match="requires 1–3"):
        configure_qwen21_graph(qwen21_turbo_template(), request(0), ())


def test_qwen21_text_and_swap_model_options(tmp_path: Path) -> None:
    assert "qwen-2.1-turbo" in [key for _, key in _model_choices_for_mode(TEXT_MODE, "qwen-2.1-turbo")["choices"]]
    head = _swap_model_changed("qwen-2.1-turbo", "Head")
    assert head[0]["choices"] == ["Head", "Body"]
    assert QWEN21_BFS_HEAD_FILE in head[4]
    assert list_lora_files("qwen-2.1-turbo", root=tmp_path) == []
    with pytest.raises(ValueError, match="qwen21 library"):
        selected_loras("qwen-2.1-turbo", ["2511-lora.safetensors"], [1.0], root=tmp_path)
    assert selected_loras("qwen-2.1-turbo", [NONE_CHOICE], [1.0], root=tmp_path) == []


def test_qwen21_uploaded_loras_appear_in_five_slots_and_keep_turbo_first(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from photo_edit_studio.config import settings

    monkeypatch.setattr(settings, "lora_dir", tmp_path)
    source = tmp_path / "uploads" / "style.safetensors"
    source.parent.mkdir()
    source.write_bytes(b"test")
    assert import_lora_files("qwen-2.1-turbo", [source], root=tmp_path) == ["style.safetensors"]
    assert list_lora_files("qwen-2.1-turbo", root=tmp_path) == ["style.safetensors"]
    adapter = selected_loras("qwen-2.1-turbo", ["style.safetensors"], [0.7], root=tmp_path)
    assert adapter[0].path == tmp_path / "qwen21" / "style.safetensors"
    req = request(1)
    req.loras = adapter
    graph = configure_qwen21_graph(qwen21_turbo_template(), req, ("source.png",))
    assert graph["2"]["class_type"] == "ViggleTurboLora"
    assert graph["40"] == {"class_type": "LoraLoaderModelOnly", "inputs": {
        "model": ["2", 0], "lora_name": os.path.join("qwen21", "style.safetensors"), "strength_model": 0.7,
    }}
    assert graph["8"]["inputs"]["model"] == ["40", 0]

    names = []
    for i in range(5):
        path = tmp_path / "qwen21" / f"adapter_{i}.safetensors"
        path.write_bytes(b"test")
        names.append(path.name)
    req.loras = selected_loras("qwen-2.1-turbo", names, [0.2, 0.4, 0.6, 0.8, 1.0], root=tmp_path)
    graph = configure_qwen21_graph(qwen21_turbo_template(), req, ("source.png",))
    assert graph["40"]["inputs"]["model"] == ["2", 0]
    for i in range(1, 5):
        assert graph[str(40 + i)]["inputs"]["model"] == [str(39 + i), 0]
    assert graph["8"]["inputs"]["model"] == ["44", 0]


def test_qwen21_optional_chain_is_used_in_text_and_swap(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from photo_edit_studio.config import settings

    monkeypatch.setattr(settings, "lora_dir", tmp_path)
    library = tmp_path / "qwen21"
    library.mkdir()
    (library / "style.safetensors").write_bytes(b"test")
    for images, workflow, kind in ((0, "text", None), (2, "swap", "Head")):
        req = request(images, workflow=workflow, kind=kind)
        if workflow == "text":
            req.compose = True
        req.loras = selected_loras("qwen-2.1-turbo", ["style.safetensors"], [0.8], root=tmp_path)
        graph = configure_qwen21_graph(
            qwen21_turbo_template(), req, tuple(f"source{i}.png" for i in range(images))
        )
        assert graph["40"]["inputs"]["model"] == ["2", 0]
        assert graph["8"]["inputs"]["model"] == ["40", 0]


def test_qwen21_resolves_exact_comfyui_lora_names() -> None:
    graph = qwen21_turbo_template()
    graph["40"] = {"class_type": "LoraLoaderModelOnly", "inputs": {
        "lora_name": "qwen21/style.safetensors", "model": ["2", 0], "strength_model": 1.0,
    }}
    resolve_qwen21_lora_names(graph, ["qwen21\\style.safetensors"])
    assert graph["40"]["inputs"]["lora_name"] == "qwen21\\style.safetensors"
    resolve_qwen21_lora_names(graph, ["qwen21/style.safetensors"])
    assert graph["40"]["inputs"]["lora_name"] == "qwen21/style.safetensors"
    with pytest.raises(RuntimeError, match="Restart the project-managed ComfyUI backend"):
        resolve_qwen21_lora_names(graph, ["other.safetensors"])


def test_qwen21_rejects_unsafe_optional_adapters(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from photo_edit_studio.config import settings

    monkeypatch.setattr(settings, "lora_dir", tmp_path)
    library = tmp_path / "qwen21"
    library.mkdir()
    wrong = library / "wrong.pt"
    wrong.write_bytes(b"test")
    with pytest.raises(ValueError, match=".safetensors"):
        import_lora_files("qwen-2.1-turbo", [wrong], root=tmp_path)
    assert list_lora_files("qwen-2.1-turbo", root=tmp_path) == []
    with pytest.raises(ValueError, match="Viggle Turbo"):
        selected_loras("qwen-2.1-turbo", [QWEN21_TURBO_LORA], [1.0], root=tmp_path)
    with pytest.raises(ValueError, match="models/loras/qwen21"):
        req = request(1)
        req.loras = [LoraSpec("outside.safetensors", tmp_path / "outside.safetensors", 1.0, "outside")]
        configure_qwen21_graph(qwen21_turbo_template(), req, ("source.png",))


def test_qwen21_controls_lock_viggle_schedule() -> None:
    settings = _turbo_controls_for_model(TEXT_MODE, "qwen-2.1-turbo", "qwen-2511")
    assert [control["value"] for control in settings] == [6, 1.0, 1.0, ""]
    assert all(control["interactive"] is False for control in settings)
    normal = _turbo_controls_for_model(TEXT_MODE, "flux-klein-4b", "qwen-2511")
    assert all(control["interactive"] is True for control in normal)


def test_qwen21_swap_uses_two_images_without_bfs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from photo_edit_studio import ui
    from photo_edit_studio.types import GenerationResult

    seen: list[GenerationRequest] = []

    def fake_generate(request: GenerationRequest, *_args: object, **_kwargs: object) -> GenerationResult:
        seen.append(request)
        return GenerationResult([Image.new("RGB", (64, 64))], request.seed, 0.1, request.model_key)

    monkeypatch.setattr(ui, "generate", fake_generate)
    monkeypatch.setattr(ui, "metadata_text", lambda _result: "{}")
    values = [NONE_CHOICE] * 5 + [1.0] * 5
    _run_swap(
        Image.new("RGB", (64, 64)), Image.new("RGB", (64, 64)),
        "qwen-2.1-turbo", "Body", "Keep the shirt", "", 1, 6, 1.0, 1.0,
        0.8, 19, True, "Off", 0.35, 1.0, 1.0, *values,
        progress=lambda *_args, **_kwargs: None,
    )
    assert len(seen) == 1
    assert seen[0].workflow == "swap" and seen[0].swap_kind == "Body"
    assert len(seen[0].images) == 2 and seen[0].loras == []
    assert "Picture 2" in seen[0].prompt


def test_qwen21_head_swap_loads_bfs_after_viggle_and_before_optional_loras(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from photo_edit_studio import ui
    from photo_edit_studio.config import settings
    from photo_edit_studio.types import GenerationResult

    monkeypatch.setattr(settings, "lora_dir", tmp_path)
    family = tmp_path / "qwen21"
    family.mkdir()
    (family / QWEN21_BFS_HEAD_FILE).write_bytes(b"test")
    (family / "style.safetensors").write_bytes(b"test")
    seen: list[GenerationRequest] = []

    def fake_generate(req: GenerationRequest, *_args: object, **_kwargs: object) -> GenerationResult:
        seen.append(req)
        return GenerationResult([Image.new("RGB", (64, 64))], req.seed, 0.1, req.model_key)

    monkeypatch.setattr(ui, "generate", fake_generate)
    monkeypatch.setattr(ui, "metadata_text", lambda _result: "{}")
    _run_swap(
        Image.new("RGB", (64, 64)), Image.new("RGB", (64, 64)),
        "qwen-2.1-turbo", "Head", "", "", 1, 6, 1.0, 1.0,
        0.8, 19, True, "Off", 0.35, 0.7, 1.0,
        QWEN21_BFS_HEAD_FILE, "style.safetensors", NONE_CHOICE, NONE_CHOICE, NONE_CHOICE,
        1.0, 0.5, 1.0, 1.0, 1.0,
        progress=lambda *_args, **_kwargs: None,
    )
    assert len(seen) == 1
    assert [lora.name for lora in seen[0].loras] == [QWEN21_BFS_HEAD_FILE, "style.safetensors"]
    assert seen[0].loras[0].weight == 0.7
    assert "head_swap:" in seen[0].prompt
    graph = configure_qwen21_graph(qwen21_turbo_template(), seen[0], ("body.png", "head.png"))
    assert graph["2"]["class_type"] == "ViggleTurboLora"
    assert graph["40"]["inputs"] == {
        "model": ["2", 0], "lora_name": os.path.join("qwen21", QWEN21_BFS_HEAD_FILE), "strength_model": 0.7,
    }
    assert graph["41"]["inputs"]["model"] == ["40", 0]
    assert graph["8"]["inputs"]["model"] == ["41", 0]


def test_qwen21_missing_head_file_reports_local_install_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from photo_edit_studio.config import settings

    monkeypatch.setattr(settings, "lora_dir", tmp_path)
    with pytest.raises(FileNotFoundError, match="Place the Qwen 2.1 head-swap LoRA"):
        swap_lora("qwen-2.1-turbo", "Head")


def test_qwen21_targeted_download_and_idempotence(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
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
    download_models.download_comfy_qwen21()
    assert len(calls) == 5
    assert (QWEN21_VIGGLE_REPO, QWEN21_TURBO_LORA) in calls
    assert (QWEN21_VIGGLE_REPO, f"comfyui/{QWEN21_TURBO_NODE}") in calls
    assert all((QWEN21_COMFY_REPO, path) in calls for path in QWEN21_FILES if not path.endswith(QWEN21_TURBO_LORA))
    assert missing_qwen21_assets(tmp_path) == []
    download_models.download_comfy_qwen21()
    assert len(calls) == 5


def test_qwen21_cli_does_not_download_full_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(sys, "argv", ["download_models.py", "qwen-2.1-turbo"])
    monkeypatch.setattr(download_models, "download_comfy_qwen21", lambda: calls.append("targeted"))
    monkeypatch.setattr(download_models, "snapshot_download", lambda **_kwargs: calls.append("snapshot"))
    download_models.main()
    assert calls == ["targeted"]