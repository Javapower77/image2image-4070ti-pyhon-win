from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path

import pytest
from PIL import Image

from photo_edit_studio import ui
from photo_edit_studio.comfy_assets import QWEN21_R128_TURBO_LORA, QWEN21_TURBO_LORA
from photo_edit_studio.comfy_workflows import qwen21_turbo_template
from photo_edit_studio.config import settings
from photo_edit_studio.dlss import DLSS_DEFAULTS, add_dlss_nodes, validate_dlss_request
from photo_edit_studio.loras import (
    NONE_CHOICE,
    import_lora_files,
    library_dir,
    list_lora_files,
    selected_loras,
)
from photo_edit_studio.models.comfy_swap import configure_qwen21_graph
from photo_edit_studio.models.registry import MODEL_SPECS
from photo_edit_studio.swap import QWEN21_BFS_BODY_FILE, QWEN21_BFS_HEAD_FILE
from photo_edit_studio.types import GenerationRequest, GenerationResult

PROFILES = ("qwen-2.1-turbo", "qwen-2.1-turbo-r128")
MODES = (ui.TEXT_MODE, ui.EDIT_MODE, ui.COMBINE_MODE)
MANDATORY = (QWEN21_TURBO_LORA, QWEN21_R128_TURBO_LORA)
SWAPS = (("Head", QWEN21_BFS_HEAD_FILE), ("Body", QWEN21_BFS_BODY_FILE))


@pytest.fixture
def lora_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.setattr(settings, "lora_dir", tmp_path)
    (tmp_path / "qwen21").mkdir()
    return tmp_path


def request(model_key: str, workflow: str) -> GenerationRequest:
    count = {"standard": 1, "text": 0, "swap": 2}.get(workflow, 1)
    return GenerationRequest(
        model_key=model_key, prompt="Change the colors", negative_prompt="",
        images=[Image.new("RGB", (64, 64)) for _ in range(count)], mask=None,
        width=1024, height=1024, steps=6, guidance=1.0, true_cfg=1.0,
        strength=0.8, seed=17, compose=workflow == "text", workflow=workflow,
        swap_kind="Head" if workflow == "swap" else None,
    )


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("model_key", PROFILES)
def test_profiles_are_selectable_with_distinct_labels_and_icons(
    mode: str, model_key: str,
) -> None:
    update = ui._model_choices_for_mode(mode, model_key)
    choices = {key: label for label, key in update["choices"]}
    assert update["value"] == model_key
    assert choices[PROFILES[0]] == MODEL_SPECS[PROFILES[0]].label
    assert choices[PROFILES[1]] == "Qwen Image 2.1 + Turbo r128 (Civitai)"
    assert choices[PROFILES[0]] != choices[PROFILES[1]]
    assert "Viggle Turbo" in choices[PROFILES[0]]
    assert ui._model_icon_file(model_key) == "qwen.svg"
    assert ui._model_icon_for_mode(mode, model_key, "flux-klein-4b")["icon"].name == "qwen.svg"
    visibility = ui._model_picker_visibility(mode, model_key, "flux-klein-4b")
    for key in PROFILES:
        row = visibility[list(MODEL_SPECS).index(key)]
        assert "mode-hidden" not in row["elem_classes"]
        assert row["variant"] == ("primary" if key == model_key else "secondary")


@pytest.mark.parametrize("mode,default", [
    (ui.TEXT_MODE, "krea-2-turbo"),
    (ui.EDIT_MODE, "qwen-2511"),
    (ui.COMBINE_MODE, "qwen-2511"),
])
def test_new_profile_does_not_replace_mode_defaults(mode: str, default: str) -> None:
    assert ui._model_choices_for_mode(mode, "unknown-model")["value"] == default
    assert ui._model_choices_for_mode(mode, default)["value"] == default
    assert qwen21_turbo_template()["2"]["inputs"]["lora_name"] == QWEN21_TURBO_LORA


@pytest.mark.parametrize("mode", (*MODES, ui.SWAP_MODE))
@pytest.mark.parametrize("model_key", PROFILES)
def test_both_profiles_lock_turbo_controls(mode: str, model_key: str) -> None:
    controls = ui._turbo_controls_for_model(
        mode, "flux-klein-4b" if mode == ui.SWAP_MODE else model_key, model_key,
    )
    assert [control["value"] for control in controls] == [6, 1.0, 1.0, ""]
    assert all(control["interactive"] is False for control in controls)
    unlocked = ui._turbo_controls_for_model(mode, "flux-klein-4b", "flux-klein-4b")
    assert all(control["interactive"] is True for control in unlocked)


def test_real_picker_buttons_keep_distinct_labels_and_qwen_icons(lora_root: Path) -> None:
    components = ui.build_app().config["components"]
    for model_key in PROFILES:
        buttons = [component for component in components if component["type"] == "button"
                   and component["props"].get("value") == MODEL_SPECS[model_key].label]
        assert len(buttons) == 2  # Standard picker and swap picker.
        for button in buttons:
            assert "model-choice" in button["props"]["elem_classes"]
            assert button["props"]["icon"]["path"].endswith("qwen.svg")


@pytest.mark.parametrize("model_key", PROFILES)
def test_shared_library_filters_both_mandatory_weights_and_unsafe_formats(
    model_key: str, lora_root: Path,
) -> None:
    family = lora_root / "qwen21"
    for name in (*MANDATORY, *(name.upper() for name in MANDATORY),
                 "style.safetensors", "other.SAFETENSORS", "unsafe.pt", "unsafe.bin"):
        (family / name).write_bytes(b"test")
    (family / "directory.safetensors").mkdir()
    assert library_dir(model_key) == library_dir(PROFILES[0]) == family
    assert list_lora_files(model_key) == ["other.SAFETENSORS", "style.safetensors"]
    assert selected_loras(model_key, [NONE_CHOICE], [1.0]) == []
    for name in (*MANDATORY, *(name.upper() for name in MANDATORY), "unsafe.pt", "unsafe.bin"):
        with pytest.raises(ValueError, match="optional Qwen 2.1"):
            selected_loras(model_key, [name], [1.0])


@pytest.mark.parametrize("model_key", PROFILES)
@pytest.mark.parametrize("filename", [*MANDATORY, *(name.upper() for name in MANDATORY),
                                      "unsafe.pt", "unsafe.bin", "unsafe.txt"])
def test_import_rejects_mandatory_weights_and_non_safetensors(
    model_key: str, filename: str, lora_root: Path,
) -> None:
    uploads = lora_root / "uploads"
    uploads.mkdir()
    source = uploads / filename
    source.write_bytes(b"test")
    with pytest.raises(ValueError, match=r"optional LoRAs must be \.safetensors"):
        import_lora_files(model_key, [source])
    assert list((lora_root / "qwen21").iterdir()) == []


@pytest.mark.parametrize("model_key", PROFILES)
def test_imported_adapters_are_shared_across_profiles_and_all_five_ui_slots(
    model_key: str, lora_root: Path,
) -> None:
    source = lora_root / "style.safetensors"
    source.write_bytes(b"test")
    updates = ui._upload_loras(model_key, [source], *([NONE_CHOICE] * 5))
    assert len(updates) == 6
    assert [slot["value"] for slot in updates[1:]] == ["style.safetensors", *([NONE_CHOICE] * 4)]
    assert all(slot["choices"] == [NONE_CHOICE, "style.safetensors"] for slot in updates[1:])
    for key in PROFILES:
        assert list_lora_files(key) == ["style.safetensors"]
        selected = selected_loras(key, ["style.safetensors"], [0.5])
        assert selected[0].path == lora_root / "qwen21" / "style.safetensors"
        assert selected[0].weight == 0.5


@pytest.mark.parametrize("model_key", PROFILES)
@pytest.mark.parametrize("kind,bfs_file", SWAPS)
def test_swap_ui_offers_head_body_and_matching_shared_bfs(
    model_key: str, kind: str, bfs_file: str, lora_root: Path,
) -> None:
    assert model_key in ui.SWAP_MODEL_KEYS
    updates = ui._swap_model_changed(model_key, kind)
    assert updates[0]["choices"] == ["Head", "Body"]
    assert updates[0]["value"] == kind
    assert [control["value"] for control in updates[1:4]] == [6, 1.0, 1.0]
    assert bfs_file in updates[4]
    assert f"{kind} BFS LoRA" in updates[4]
    assert "instruction-only" not in updates[4]
    assert ("Civitai Turbo r128" if model_key == PROFILES[1] else "Viggle Turbo r256") in updates[4]
    assert ui._model_icon_for_mode(ui.SWAP_MODE, "flux-klein-4b", model_key)["icon"].name == "qwen.svg"


@pytest.mark.parametrize("model_key", PROFILES)
@pytest.mark.parametrize("kind,bfs_file", SWAPS)
def test_run_swap_keeps_mandatory_bfs_optional_order_and_deduplicates(
    monkeypatch: pytest.MonkeyPatch, model_key: str, kind: str,
    bfs_file: str, lora_root: Path,
) -> None:
    for name in (bfs_file, "style.safetensors", "detail.safetensors"):
        (lora_root / "qwen21" / name).write_bytes(b"test")
    seen: list[GenerationRequest] = []

    def fake_generate(req: GenerationRequest, *_args: object, **_kwargs: object) -> GenerationResult:
        seen.append(req)
        return GenerationResult([Image.new("RGB", (64, 64))], req.seed, 0.1, req.model_key)

    monkeypatch.setattr(ui, "generate", fake_generate)
    monkeypatch.setattr(ui, "metadata_text", lambda _result: "{}")
    body, donor = Image.new("RGB", (64, 64)), Image.new("RGB", (64, 64))
    ui._run_swap(
        body, donor, model_key, kind, "Keep the lighting", "", 1, 6, 1.0, 1.0,
        0.8, 19, True, "Off", 0.35, 0.7, 1.0,
        "style.safetensors", bfs_file, "style.safetensors", "detail.safetensors", NONE_CHOICE,
        -2.0, 1.8, 0.9, -0.5, 1.0, progress=lambda *_args, **_kwargs: None,
    )
    assert len(seen) == 1
    req = seen[0]
    assert req.model_key == model_key
    assert (req.workflow, req.swap_kind) == ("swap", kind)
    assert req.images == [body, donor]
    assert req.seed == 19
    assert req.prompt.startswith(f"{kind.lower()}_swap:")
    assert "Keep the lighting" in req.prompt
    assert [lora.name for lora in req.loras] == [bfs_file, "style.safetensors", "detail.safetensors"]
    assert [lora.weight for lora in req.loras] == [0.7, -2.0, -0.5]
    template = qwen21_turbo_template(model_key)
    mandatory = deepcopy(template["2"])
    graph = configure_qwen21_graph(template, req, ("body.png", "donor.png"))
    assert graph["2"] == mandatory
    assert graph["2"]["inputs"]["lora_name"] == MANDATORY[PROFILES.index(model_key)]
    for index, (name, weight) in enumerate(zip(
        (bfs_file, "style.safetensors", "detail.safetensors"), (0.7, -2.0, -0.5), strict=True,
    )):
        assert graph[str(40 + index)] == {"class_type": "LoraLoaderModelOnly", "inputs": {
            "model": ["2" if index == 0 else str(39 + index), 0],
            "lora_name": os.path.join("qwen21", name), "strength_model": weight,
        }}
    assert graph["8"]["inputs"]["model"] == ["42", 0]
    assert graph["5"]["inputs"]["images.image_1"] == ["31", 0]
    assert graph["5"]["inputs"]["images.image_2"] == ["32", 0]


@pytest.mark.parametrize("model_key", PROFILES)
@pytest.mark.parametrize("workflow", ["standard", "text", "swap"])
def test_dlss_is_eligible_and_splices_after_decode_for_both_profiles(
    model_key: str, workflow: str,
    lora_root: Path,
) -> None:
    req = request(model_key, workflow)
    if workflow == "swap":
        (lora_root / "qwen21" / QWEN21_BFS_HEAD_FILE).touch()
        req.loras = selected_loras(model_key, [QWEN21_BFS_HEAD_FILE], [1.0], root=lora_root)
    graph = configure_qwen21_graph(
        qwen21_turbo_template(model_key), req,
        tuple(f"source{i}.png" for i in range(len(req.images))),
    )
    original = deepcopy(graph)
    assert validate_dlss_request(req) is None
    assert add_dlss_nodes(graph, req) is graph
    assert graph == original
    req.dlss = {"enabled": False}
    assert validate_dlss_request(req) is None
    assert add_dlss_nodes(graph, req) is graph
    assert graph == original
    req.dlss = {"upscaling_mode": "2x (Performance)"}
    assert validate_dlss_request(req) == {**DLSS_DEFAULTS, "upscaling_mode": "2x (Performance)"}
    assert add_dlss_nodes(graph, req) is graph
    enhancer = graph[graph["29"]["inputs"]["images"][0]]
    assert enhancer["class_type"] == "DLSS5EnhanceImages"
    assert enhancer["inputs"]["images"] == original["29"]["inputs"]["images"]
    assert enhancer["inputs"]["verify_neural_rendering"] is True
    config = graph[enhancer["inputs"]["settings"][0]]
    assert config["class_type"] == "DLSS5Settings"
    assert config["inputs"]["upscaling_mode"] == "2x (Performance)"
    assert len(graph) == len(original) + 2
    assert all(graph[key] == original[key] for key in original if key != "29")


@pytest.mark.parametrize("model_key", PROFILES)
def test_dlss_rejects_unrelated_workflow_for_both_profiles(model_key: str) -> None:
    req = request(model_key, "krea-remix")
    req.dlss = {}
    with pytest.raises(ValueError, match="supports only current ComfyUI"):
        validate_dlss_request(req)