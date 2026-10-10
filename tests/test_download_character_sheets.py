from __future__ import annotations

import hashlib
import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save

from photo_edit_studio import krea_character_sheet as backend
from scripts import download_models as downloader


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    monkeypatch.setattr(downloader.settings, "comfy_dir", tmp_path / "comfy")
    monkeypatch.setattr(downloader.settings, "model_dir", tmp_path / "models")
    monkeypatch.setattr(downloader.settings, "lora_dir", tmp_path / "separate-loras")
    downloader.settings.comfy_dir.mkdir()
    (downloader.settings.comfy_dir / "main.py").touch()
    monkeypatch.setattr(downloader, "_token", lambda _: "private-secret")
    for name in (
        "hf_hub_download", "snapshot_download", "urlretrieve", "_download_qwen_aio",
        "download_krea_turbo_lora", "download_comfy_krea_originals",
        "download_comfy_firered", "download_comfy_qwen21", "download_comfy_qwen21_r128",
    ):
        monkeypatch.setattr(downloader, name, lambda *a, **kw: pytest.fail("unexpected download"))
    downloader._TOKENS.clear()
    yield
    downloader._TOKENS.clear()


def test_exact_manifest_and_backend_paths():
    expected = (
        ("QuadView_krea2_v1.safetensors", "krea2/QuadView_krea2_v1.safetensors", 914160176,
         "9435005af21cadaed16f79ef1eca9b76b8497789644eeb7f2475b826bb480fbc"),
        ("DynamicCharacterSheet_krea2_v1.safetensors", "krea2/DynamicCharacterSheet_krea2_v1.safetensors", 914160184,
         "bff2dd8003b5d4e50ef6f2f2793a02a3d1657aeb9176bfec7941be85cb646990"),
        ("workflows/QuadView_krea2_v1.json", "workflows/charactersheet/QuadView_krea2_v1.json", 30378,
         "5f6bdeac5b81aa6f0d331fbaa45b5aca31660de9d943bd9d63dc13c108ba304b"),
        ("workflows/DynamicCharacterSheet_krea2_v1.json", "workflows/charactersheet/DynamicCharacterSheet_krea2_v1.json", 57217,
         "d144cda8af10c5c6e3fe8472d541f619a751f597cba298d581e23dbe5901b706"),
    )
    assert downloader.KREA_CHARACTER_SHEET_REPO == "Alissonerdx/CharacterSheet"
    assert downloader.KREA_CHARACTER_SHEET_REVISION == backend.KREA_CHARACTER_SHEET_REVISION
    assert downloader.KREA_CHARACTER_SHEET_REVISION == "3dc4295163dacc924d213168d67bf16850fd954f"
    assert downloader.KREA_CHARACTER_SHEET_ASSETS == tuple({
        "repo": downloader.KREA_CHARACTER_SHEET_REPO,
        "revision": downloader.KREA_CHARACTER_SHEET_REVISION,
        "remote": remote, "destination": destination, "size": size, "sha256": digest,
    } for remote, destination, size, digest in expected)
    assert {a["remote"] for a in downloader.KREA_CHARACTER_SHEET_ASSETS[:2]} == set(
        backend.KREA_CHARACTER_SHEET_PROFILES.values()
    )
    dynamic = downloader.KREA_CHARACTER_SHEET_ASSETS[-1]
    assert downloader.settings.model_dir / dynamic["destination"] == backend.dynamic_template_path()
    assert dynamic["sha256"] == backend.KREA_DYNAMIC_TEMPLATE_SHA256


@pytest.fixture
def tiny_manifest(monkeypatch):
    weights = save({"test": np.zeros(1, dtype=np.float32)})
    graph = json.dumps({"nodes": [{"id": 184, "type": "PrimitiveStringMultiline",
                                   "widgets_values": ["caption template"]}]}).encode()
    payloads = {a["remote"]: graph if a["remote"].endswith(".json") else weights
                for a in downloader.KREA_CHARACTER_SHEET_ASSETS}
    manifest = tuple(dict(a, size=len(payloads[a["remote"]]),
                          sha256=hashlib.sha256(payloads[a["remote"]]).hexdigest())
                     for a in downloader.KREA_CHARACTER_SHEET_ASSETS)
    monkeypatch.setattr(downloader, "KREA_CHARACTER_SHEET_ASSETS", manifest)
    return manifest, payloads


def target_for(asset):
    return downloader.settings.model_dir if asset["remote"].endswith(".json") else downloader.settings.lora_dir


def mock_hub(monkeypatch, tiny_manifest, *, content=None, race=False):
    manifest, payloads = tiny_manifest
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        assert kwargs["token"] == "private-secret"
        remote = kwargs["filename"]
        path = Path(kwargs["local_dir"]) / remote
        path.parent.mkdir(parents=True, exist_ok=True)
        if kwargs["repo_id"] == downloader.KREA_CHARACTER_SHEET_REPO:
            asset = next(a for a in manifest if a["remote"] == remote)
            assert kwargs["revision"] == downloader.KREA_CHARACTER_SHEET_REVISION
            destination = target_for(asset) / asset["destination"]
            assert not destination.exists()
            assert Path(kwargs["local_dir"]).parent == destination.parent
            path.write_bytes(payloads[remote] if content is None else content)
            if race:
                destination.write_bytes(b"concurrent file")
        else:
            assert "revision" not in kwargs
            allowed = (
                kwargs["repo_id"] == downloader.COMFY_KREA_REPO
                and remote in downloader.COMFY_KREA_FILES
            ) or (
                kwargs["repo_id"] == downloader.KREA_FIRST_LORA_REPO
                and remote == downloader.KREA_FIRST_LORA_REMOTE
            )
            assert allowed, "identity, Turbo, Klein and extra assets are forbidden"
            path.write_bytes(b"mock shared weights")
        return str(path)

    monkeypatch.setattr(downloader, "hf_hub_download", download)
    return calls


def test_actual_cli_dispatch_installs_only_required_files(monkeypatch, tiny_manifest, capsys):
    manifest, payloads = tiny_manifest
    calls = mock_hub(monkeypatch, tiny_manifest)
    monkeypatch.setattr(sys, "argv", ["download_models.py", "--comfy-krea-character-sheets"])
    downloader.main()
    assert len(calls) == 8  # three shared weights, mandatory adapter, two LoRAs, two JSONs
    assert [c["filename"] for c in calls[:4]] == [
        *downloader.COMFY_KREA_FILES, downloader.KREA_FIRST_LORA_REMOTE,
    ]
    for asset in manifest:
        assert (target_for(asset) / asset["destination"]).read_bytes() == payloads[asset["remote"]]
    for profile in backend.KREA_CHARACTER_SHEET_PROFILES:
        assert backend.missing_krea_character_sheet_assets(downloader.settings.comfy_dir, profile) == []
    assert not list(downloader.settings.model_dir.parent.rglob("*.staging-*"))
    output = capsys.readouterr().out
    assert "private-secret" not in output
    assert "https://civitai.com/models/2764727" in output
    assert "redistribution license grant" in output
    assert "krea-character-sheets" not in downloader.MODEL_SPECS
    assert downloader.RECOMMENDED_MODELS == ["qwen-2511", "qwen-2511-aio", "flux-klein-4b"]


def test_verified_cache_skips_all_network_and_auth(monkeypatch, tiny_manifest, capsys):
    calls = mock_hub(monkeypatch, tiny_manifest)
    downloader.download_comfy_krea_character_sheets()
    calls.clear()
    monkeypatch.setattr(downloader, "_token", lambda _: pytest.fail("cached file requested auth"))
    downloader.download_comfy_krea_character_sheets()
    assert calls == []
    assert capsys.readouterr().out.count("Already present (size/SHA256 verified)") == 4


@pytest.mark.parametrize("index", [0, 2])
@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("failure", ["size", "sha256", "format"])
def test_corruption_preserved_or_not_installed(monkeypatch, tiny_manifest, index, existing, failure):
    manifest, payloads = tiny_manifest
    asset = dict(manifest[index])
    payload = payloads[asset["remote"]]
    corrupted = payload[:-1] if failure == "size" else b"X" * len(payload)
    if failure == "format":
        # Matching size/hash must still be rejected by real Safetensors/JSON parsing.
        asset["sha256"] = hashlib.sha256(corrupted).hexdigest()
    target = target_for(asset)
    destination = target / asset["destination"]
    if existing:
        destination.parent.mkdir(parents=True)
        destination.write_bytes(corrupted)
    else:
        mock_hub(monkeypatch, tiny_manifest, content=corrupted)
    message = {"size": "size mismatch", "sha256": "SHA256 mismatch", "format": "invalid"}[failure]
    with pytest.raises(RuntimeError, match=message):
        downloader._download_pinned_krea_asset(asset, target, label="Krea character sheets")
    assert destination.exists() == existing
    if existing:
        assert destination.read_bytes() == corrupted
    assert not list(target.parent.rglob("*.staging-*"))


@pytest.mark.parametrize("workflow", [
    [], {}, {"nodes": []}, {"nodes": {}}, {"nodes": [None]},
    {"nodes": [{"id": 1}]}, {"nodes": [{"id": True, "type": "Node"}]},
    {"nodes": [{"id": 1, "type": " "}]},
])
def test_hash_verified_json_requires_valid_nodes(tmp_path, workflow):
    path = tmp_path / "workflow.json"
    raw = json.dumps(workflow).encode()
    path.write_bytes(raw)
    asset = {"size": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    assert downloader._pinned_asset_error(path, asset) == "invalid workflow nodes"


def test_json_never_parsed_before_sha256(monkeypatch, tmp_path):
    path = tmp_path / "workflow.json"
    path.write_bytes(b"not JSON")
    monkeypatch.setattr(downloader.json, "loads", lambda _: pytest.fail("parsed before hash"))
    assert downloader._pinned_asset_error(path, {"size": 8, "sha256": "0" * 64}) == "SHA256 mismatch"


def test_concurrent_destination_left_unchanged(monkeypatch, tiny_manifest):
    asset = tiny_manifest[0][0]
    mock_hub(monkeypatch, tiny_manifest, race=True)
    target = target_for(asset)
    with pytest.raises(RuntimeError, match="Destination appeared"):
        downloader._download_pinned_krea_asset(asset, target, label="Krea character sheets")
    assert (target / asset["destination"]).read_bytes() == b"concurrent file"
    assert not list(target.rglob("*.staging-*"))


def test_transport_failure_redacted(monkeypatch, tiny_manifest, capsys, caplog):
    asset = tiny_manifest[0][0]

    def failure(**kwargs):
        logging.getLogger("httpx").critical("Authorization: private-secret")
        raise RuntimeError("https://example.org/?token=private-secret")

    monkeypatch.setattr(downloader, "hf_hub_download", failure)
    target = target_for(asset)
    with pytest.raises(RuntimeError, match="Krea character sheets asset failed") as error:
        downloader._download_pinned_krea_asset(asset, target, label="Krea character sheets")
    captured = capsys.readouterr()
    assert "private-secret" not in str(error.value) + captured.out + captured.err + caplog.text
    assert error.value.__suppress_context__
    assert not (target / asset["destination"]).exists()
    assert not list(target.rglob("*.staging-*"))


def test_missing_comfy_fails_before_download():
    (downloader.settings.comfy_dir / "main.py").unlink()
    with pytest.raises(FileNotFoundError, match="setup-comfy"):
        downloader.download_comfy_krea_character_sheets()


def test_powershell_actual_preset_only_dispatches_flag():
    script = Path(__file__).parents[1] / "scripts" / "download-models.ps1"
    text = script.read_text()
    assert '"krea-character-sheets"' in text.split("[string]$Preset")[0]
    assert "--token" not in text and "Read-Host" not in text
    pwsh = shutil.which("pwsh")
    if pwsh is None:
        pytest.skip("PowerShell unavailable")
    command = r"""
function Set-Location {}
function Test-Path { return $true }
function Write-Host {}
function Join-Path { return 'MockPython' }
function MockPython { $script:captured = @($args); $global:LASTEXITCODE = 0 }
$wrapper = [scriptblock]::Create((Get-Content -Raw -LiteralPath SCRIPT_PATH))
& $wrapper -Preset krea-character-sheets
if (($script:captured -join '|') -ne 'scripts\download_models.py|--comfy-krea-character-sheets') {
    throw 'Unexpected wrapper arguments'
}
"""
    command = command.replace("SCRIPT_PATH", "'" + str(script).replace("'", "''") + "'")
    result = subprocess.run([pwsh, "-NoProfile", "-NonInteractive", "-Command", command],
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr