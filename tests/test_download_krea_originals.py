from __future__ import annotations

import hashlib
import logging
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import download_models as downloader


def test_manifest_pins_exact_approved_sources_and_destinations():
    assert downloader.KREA_ORIGINAL_ASSETS == (
        {
            "repo": "Comfy-Org/Krea-2",
            "revision": "6b1d7191d84d5ded74d83a1a98211dad0ac8ae25",
            "remote": "diffusion_models/krea2_turbo_int8_convrot.safetensors",
            "destination": "diffusion_models/krea2_turbo_int8_convrot-b19a4f0be264.safetensors",
            "sha256": "8e4eeda70dd5037ab1ba2bef6b417f9f901e26093117cf397f741fc1fdaaf3f1",
            "size": 13492686496,
        },
        {
            "repo": "spacepxl/Wan2.1-VAE-upscale2x",
            "revision": "384fb7de682e60bd54b59d6eea810ca9d9993497",
            "remote": "Wan2.1_VAE_upscale2x_imageonly_real_v1.safetensors",
            "destination": "vae/Wan2.1_VAE_upscale2x_imageonly_real_v1.safetensors",
            "sha256": "2413554bbec24215185662d009893cf4666b8e777efece2d895e03e1a6b63e06",
            "size": 507684560,
        },
        {
            "repo": "timothy692/1x-ITF-SkinDiffDetail-Lite-v1",
            "revision": "c5b4f4c21c62eb0819c9e3fd0f2ea15c343c3754",
            "remote": "1x-ITF-SkinDiffDetail-Lite-v1.pth",
            "destination": "upscale_models/1x-ITF-SkinDiffDetail-Lite-v1.pth",
            "sha256": "94d368b633614958f84f335b129fd85abd30200e8fbc575b859ba6762116222b",
            "size": 20099337,
        },
    )
    assert sum(a["size"] for a in downloader.KREA_ORIGINAL_ASSETS) / 1024**3 == pytest.approx(13.0576, abs=0.001)


@pytest.fixture
def assets(monkeypatch, tmp_path):
    monkeypatch.setattr(downloader.settings, "comfy_dir", tmp_path)
    (tmp_path / "main.py").touch()
    payload = b"mock weights, never unpickled"
    assets = tuple(dict(asset, size=len(payload), sha256=hashlib.sha256(payload).hexdigest())
                   for asset in downloader.KREA_ORIGINAL_ASSETS)
    monkeypatch.setattr(downloader, "KREA_ORIGINAL_ASSETS", assets)
    # Safetensors parsing is independently exercised below using real safe_open.
    monkeypatch.setattr(downloader, "_valid_safetensors", lambda _: True)
    monkeypatch.setattr(downloader, "_token", lambda provider: "private-secret")
    monkeypatch.setattr(downloader, "snapshot_download", lambda **_: pytest.fail("snapshot download"))
    monkeypatch.setattr(downloader, "_download_named_file", lambda *_: pytest.fail("extra asset"))
    return assets, payload


def install_mock(monkeypatch, tmp_path, assets, payload, *, content=None):
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        asset = next(a for a in assets if a["repo"] == kwargs["repo_id"])
        assert kwargs["revision"] == asset["revision"]
        assert kwargs["filename"] == asset["remote"]
        assert kwargs["token"] == "private-secret"
        destination = tmp_path / "models" / asset["destination"]
        assert not destination.exists(), "must stage before installation"
        staging = Path(kwargs["local_dir"])
        assert staging.parent == destination.parent
        path = staging / asset["remote"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload if content is None else content)
        return str(path)

    monkeypatch.setattr(downloader, "hf_hub_download", download)
    return calls


def test_three_assets_staged_verified_renamed_and_provenance_printed(monkeypatch, tmp_path, assets, capsys):
    manifest, payload = assets
    calls = install_mock(monkeypatch, tmp_path, manifest, payload)
    downloader.download_comfy_krea_originals()
    assert len(calls) == 3
    for asset in manifest:
        assert (tmp_path / "models" / asset["destination"]).read_bytes() == payload
    assert not list(tmp_path.rglob("*.staging-*"))
    output = capsys.readouterr().out
    for text in ("current publisher INT8 alias", "NOT proof", "historical suffix", "gemasai", "License metadata is None", "never pickle-loaded"):
        assert text in output
    assert "private-secret" not in output


def test_verified_existing_files_skip_network_and_credentials(monkeypatch, tmp_path, assets, capsys):
    manifest, payload = assets
    for asset in manifest:
        path = tmp_path / "models" / asset["destination"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    monkeypatch.setattr(downloader, "hf_hub_download", lambda **_: pytest.fail("network"))
    monkeypatch.setattr(downloader, "_token", lambda _: pytest.fail("credentials"))
    downloader.download_comfy_krea_originals()
    assert capsys.readouterr().out.count("Already present (size/SHA256 verified)") == 3


@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("failure", ["size", "sha256", "header"])
def test_corruption_rejected_without_install_or_overwrite(monkeypatch, tmp_path, assets, existing, failure, capsys):
    manifest, payload = assets
    asset = manifest[0]
    corrupted = payload[:-1] if failure == "size" else b"X" * len(payload)
    if failure == "header":
        corrupted = payload
        monkeypatch.setattr(downloader, "_valid_safetensors", lambda _: False)
    destination = tmp_path / "models" / asset["destination"]
    if existing:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(corrupted)
        monkeypatch.setattr(downloader, "hf_hub_download", lambda **_: pytest.fail("network"))
    else:
        install_mock(monkeypatch, tmp_path, manifest, payload, content=corrupted)
    with pytest.raises(RuntimeError, match={"size": "size mismatch", "sha256": "SHA256 mismatch", "header": "Safetensors header"}[failure]):
        downloader._download_pinned_krea_asset(asset, tmp_path / "models")
    assert destination.exists() == existing
    if existing:
        assert destination.read_bytes() == corrupted
        assert "move/remove it and rerun" in capsys.readouterr().err
    assert not list(tmp_path.rglob("*.staging-*"))


def test_explicit_remove_then_redownload(monkeypatch, tmp_path, assets):
    manifest, payload = assets
    destination = tmp_path / "models" / manifest[0]["destination"]
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"stale")
    with pytest.raises(RuntimeError, match="Left unchanged"):
        downloader._download_pinned_krea_asset(manifest[0], tmp_path / "models")
    destination.unlink()
    install_mock(monkeypatch, tmp_path, manifest, payload)
    downloader._download_pinned_krea_asset(manifest[0], tmp_path / "models")
    assert destination.read_bytes() == payload


def test_transport_errors_redacted_and_staging_removed(monkeypatch, tmp_path, assets, capsys, caplog):
    manifest, _ = assets

    def failure(**kwargs):
        logging.getLogger("httpx").critical("Authorization: private-secret")
        raise RuntimeError("https://example.org/?token=private-secret")

    monkeypatch.setattr(downloader, "hf_hub_download", failure)
    with pytest.raises(RuntimeError) as error:
        downloader.download_comfy_krea_originals()
    captured = capsys.readouterr()
    assert "private-secret" not in str(error.value) + captured.out + captured.err + caplog.text
    assert error.value.__suppress_context__
    assert not list(tmp_path.rglob("*.staging-*"))
    assert not (tmp_path / "models" / manifest[0]["destination"]).exists()


def test_safetensors_header_is_opened_but_pth_is_not(monkeypatch, tmp_path):
    # Valid empty Safetensors header: no tensor loading is necessary to reject it.
    path = tmp_path / "empty.safetensors"
    payload = (2).to_bytes(8, "little") + b"{}"
    path.write_bytes(payload)
    asset = {"size": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
    assert downloader._pinned_asset_error(path, asset) == "invalid Safetensors header"
    pth = tmp_path / "opaque.pth"
    pth.write_bytes(payload)
    monkeypatch.setattr(downloader, "safe_open", lambda *_args, **_kwargs: pytest.fail("parsed .pth"))
    assert downloader._pinned_asset_error(pth, asset) is None


def test_missing_comfy_install_fails_before_network(monkeypatch, tmp_path):
    monkeypatch.setattr(downloader.settings, "comfy_dir", tmp_path)
    monkeypatch.setattr(downloader, "hf_hub_download", lambda **_: pytest.fail("network"))
    with pytest.raises(FileNotFoundError, match="setup-comfy"):
        downloader.download_comfy_krea_originals()


def test_cli_flag_is_standalone_and_recommended_is_unchanged(monkeypatch):
    calls = []
    monkeypatch.setattr(downloader, "download_comfy_krea_originals", lambda: calls.append("originals"))
    monkeypatch.setattr(downloader, "_hf_download", lambda *_args, **_kwargs: pytest.fail("other downloads"))
    monkeypatch.setattr(downloader, "download_comfy_krea", lambda: pytest.fail("normal Krea"))
    monkeypatch.setattr(downloader, "download_krea_turbo_lora", lambda: pytest.fail("LoRA"))
    monkeypatch.setattr(sys, "argv", ["download_models.py", "--comfy-krea-originals"])
    downloader.main()
    assert calls == ["originals"]
    assert downloader.RECOMMENDED_MODELS == ["qwen-2511", "qwen-2511-aio", "flux-klein-4b"]
    assert "krea-originals" not in downloader.MODEL_SPECS


def test_recommended_cli_does_not_invoke_originals(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["download_models.py"])
    monkeypatch.setattr(downloader, "download_comfy_krea_originals", lambda: pytest.fail("originals"))
    monkeypatch.setattr(downloader, "_hf_download", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(downloader, "_download_qwen_aio", lambda _: None)
    downloader.main()


def test_powershell_preset_dispatches_only_flag(tmp_path):
    script = Path(__file__).parents[1] / "scripts" / "download-models.ps1"
    text = script.read_text()
    assert '"krea-originals" { $models = @("--comfy-krea-originals") }' in text
    assert '"krea-originals"' in text.split("[string]$Preset")[0]
    assert "--token" not in text and "Read-Host" not in text
    pwsh = shutil.which("pwsh")
    if pwsh is None:
        pytest.skip("PowerShell unavailable")
    # Execute a scriptblock with commands shadowed: no Python process/model download.
    # Get-Content reads the real wrapper and its actual parameter binding/switch.
    command = """
function Set-Location {}
function Test-Path { return $true }
function Write-Host {}
function Join-Path { return 'MockPython' }
function MockPython { $script:captured = @($args); $global:LASTEXITCODE = 0 }
$wrapper = [scriptblock]::Create((Get-Content -Raw -LiteralPath $args[0]))
& $wrapper -Preset krea-originals
if (($script:captured -join '|') -ne 'scripts\\download_models.py|--comfy-krea-originals') {
    throw 'Unexpected wrapper arguments'
}
"""
    # Pass the path through a literal in the command, not -Command positional args.
    command = command.replace("$args[0]", "'" + str(script).replace("'", "''") + "'")
    result = subprocess.run([pwsh, "-NoProfile", "-NonInteractive", "-Command", command],
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr