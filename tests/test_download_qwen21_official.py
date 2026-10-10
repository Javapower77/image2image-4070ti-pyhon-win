from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

from photo_edit_studio.models import diffusers_adapters
from scripts import download_models as downloader

OFFICIAL_KEY = "qwen-2.1-turbo-official"
OFFICIAL_REPO = "Qwen/Qwen-Image-2.1-Turbo"
# Independent contract literals: do not derive expected sigmas from production constants.
MODEL_INDEX = b'''{
  "_class_name": "QwenImage21Pipeline",
  "sample_sigmas": [1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568],
  "publisher_extra": {"keep": [null, "unchanged"]}
}
'''
TRANSFORMER_CONFIG = b'''{
  "causal_condition": true,
  "publisher_extra": {"keep": [3, 2, 1]},
  "_class_name": "QwenImageTransformer2DModel"
}
'''


@pytest.fixture(autouse=True)
def isolated_downloads(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(downloader.settings, "model_dir", tmp_path)
    monkeypatch.setattr(sys, "argv", ["download_models.py", OFFICIAL_KEY])
    monkeypatch.setattr(downloader, "_token", lambda _provider: None)
    monkeypatch.setattr(downloader, "_TOKENS", {})

    def forbidden(*_args, **_kwargs):
        pytest.fail("Unexpected network, targeted asset download, or inference runtime")

    for name in (
        "snapshot_download", "hf_hub_download", "urlretrieve", "_download_qwen_aio",
        "download_comfy_qwen21", "download_comfy_qwen21_r128",
        "download_comfy_qwen_character_sheet", "download_comfy_krea",
        "download_krea_turbo_lora", "download_comfy_firered",
    ):
        monkeypatch.setattr(downloader, name, forbidden)
    monkeypatch.setattr(downloader.httpx, "Client", forbidden)
    monkeypatch.setattr(diffusers_adapters, "_runtime", forbidden)


def write_snapshot(destination: Path, files: dict[str, bytes]) -> None:
    """Write configuration only; no weights or model loading are needed."""
    for filename, content in files.items():
        target = destination / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)


def test_official_registry_selects_publisher_repo_not_comfy_or_viggle(tmp_path: Path) -> None:
    spec = downloader.MODEL_SPECS[OFFICIAL_KEY]
    assert spec.key == OFFICIAL_KEY
    assert spec.repo_id == OFFICIAL_REPO
    assert spec.loader == "qwen21_official"
    assert spec.local_path == tmp_path / "repos" / "Qwen--Qwen-Image-2.1-Turbo"
    assert OFFICIAL_KEY not in downloader.RECOMMENDED_MODELS


@pytest.mark.parametrize("token", [None, "test-only-token"])
def test_official_full_snapshot_preserves_configs_then_validates(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, token: str | None,
) -> None:
    destination = tmp_path / "repos" / "Qwen--Qwen-Image-2.1-Turbo"
    files = {
        "model_index.json": MODEL_INDEX,
        "transformer/config.json": TRANSFORMER_CONFIG,
        "scheduler/scheduler_config.json": b'{"publisher_extra": "keep scheduler"}\n',
        "processor/preprocessor_config.json": b'{"publisher_extra": "keep processor"}\n',
    }
    calls = []
    events = []
    monkeypatch.setattr(downloader, "_token", lambda _provider: token)

    def snapshot(**kwargs):
        calls.append(kwargs)
        # Absence, rather than empty values, is the full upstream snapshot contract.
        assert "allow_patterns" not in kwargs
        assert "ignore_patterns" not in kwargs
        assert kwargs["repo_id"] == OFFICIAL_REPO
        assert kwargs["local_dir"] == destination
        write_snapshot(destination, files)
        events.append("snapshot complete")
        return str(destination)

    real_validate = diffusers_adapters.validate_qwen21_official_config

    def validate(path: Path) -> None:
        assert path == destination
        assert events == ["snapshot complete"]
        for filename, content in files.items():
            assert (path / filename).read_bytes() == content
        real_validate(path)
        events.append("validated")

    monkeypatch.setattr(downloader, "snapshot_download", snapshot)
    monkeypatch.setattr(diffusers_adapters, "validate_qwen21_official_config", validate)
    downloader.main()

    assert calls == [{
        "repo_id": OFFICIAL_REPO,
        "local_dir": destination,
        "local_dir_use_symlinks": False,
        "token": token or False,
    }]
    assert events == ["snapshot complete", "validated"]
    for filename, content in files.items():
        assert (destination / filename).read_bytes() == content


@pytest.mark.parametrize("invalid, message", [
    ("missing-index", "requires model_index.json and transformer/config.json"),
    ("missing-transformer", "requires model_index.json and transformer/config.json"),
    ("malformed-index", "requires model_index.json and transformer/config.json"),
    ("wrong-pipeline", "exact publisher eight sample_sigmas"),
    ("missing-sigmas", "exact publisher eight sample_sigmas"),
    ("changed-sigma", "exact publisher eight sample_sigmas"),
    ("extra-sigma", "exact publisher eight sample_sigmas"),
    ("noncausal-transformer", "causal_condition=true"),
])
def test_official_download_rejects_invalid_saved_config_without_rewriting(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, invalid: str, message: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    destination = tmp_path / "repos" / "Qwen--Qwen-Image-2.1-Turbo"
    files = {"model_index.json": MODEL_INDEX, "transformer/config.json": TRANSFORMER_CONFIG}
    if invalid == "missing-index":
        del files["model_index.json"]
    elif invalid == "missing-transformer":
        del files["transformer/config.json"]
    elif invalid == "malformed-index":
        files["model_index.json"] = b"{broken json"
    elif invalid == "noncausal-transformer":
        files["transformer/config.json"] = TRANSFORMER_CONFIG.replace(b"true", b"false")
    else:
        index = json.loads(MODEL_INDEX)
        if invalid == "wrong-pipeline":
            index["_class_name"] = "QwenImagePipeline"
        elif invalid == "missing-sigmas":
            del index["sample_sigmas"]
        elif invalid == "changed-sigma":
            index["sample_sigmas"][1] = 0.978454
        elif invalid == "extra-sigma":
            index["sample_sigmas"].append(0.0)
        files["model_index.json"] = json.dumps(index).encode("utf-8")
    calls = []

    def snapshot(**kwargs):
        calls.append(kwargs)
        assert kwargs["repo_id"] == OFFICIAL_REPO
        assert kwargs["local_dir"] == destination
        write_snapshot(destination, files)
        return str(destination)

    monkeypatch.setattr(downloader, "snapshot_download", snapshot)
    with pytest.raises(RuntimeError, match=re.escape(message)):
        downloader.main()

    assert len(calls) == 1
    assert "Downloads complete." not in capsys.readouterr().out
    for filename in ("model_index.json", "transformer/config.json"):
        target = destination / filename
        if filename in files:
            assert target.read_bytes() == files[filename]
        else:
            assert not target.exists()


def test_powershell_official_preset_is_exact_singleton() -> None:
    script = (Path(__file__).parents[1] / "scripts" / "download-models.ps1").read_text(
        encoding="utf-8",
    )
    validate_set = re.search(r"\[ValidateSet\((.*?)\)\]", script, flags=re.DOTALL)
    assert validate_set is not None
    assert re.findall(r'"([^"]+)"', validate_set.group(1)).count("qwen-2.1-official") == 1
    bodies = re.findall(r'"qwen-2\.1-official"\s*\{([^}]*)\}', script)
    assert len(bodies) == 1
    assert re.fullmatch(r'\s*\$models\s*=\s*@\("qwen-2\.1-turbo-official"\)\s*', bodies[0])