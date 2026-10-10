from __future__ import annotations

import hashlib
import io
import json
import logging
import shutil
import subprocess
import sys
import warnings
import zipfile
from pathlib import Path

import httpx
import numpy as np
import pytest
from safetensors.numpy import save

from photo_edit_studio import qwen_character_sheet as backend
from scripts import download_models as downloader


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    monkeypatch.setattr(downloader.settings, "comfy_dir", tmp_path / "comfy")
    monkeypatch.setattr(downloader.settings, "model_dir", tmp_path / "models")
    downloader.settings.comfy_dir.mkdir()
    (downloader.settings.comfy_dir / "main.py").touch()
    monkeypatch.setattr(downloader, "_token", lambda _: "private-secret")
    for name in ("hf_hub_download", "snapshot_download", "urlretrieve"):
        monkeypatch.setattr(downloader, name, lambda *a, **kw: pytest.fail("live download forbidden"))
    real_client = httpx.Client
    monkeypatch.setattr(downloader.httpx, "Client", lambda **kw: real_client(
        transport=httpx.MockTransport(lambda _: pytest.fail("live HTTP forbidden")), **kw,
    ))
    downloader._TOKENS.clear()
    yield
    downloader._TOKENS.clear()


def document(profile):
    _, group, static = backend.QWEN_CHARACTER_SHEET_PROFILES[profile]
    return {"definitions": {"subgraphs": [{"id": group, "nodes": [
        {"id": 478, "widgets_values_named": {"value": "system prompt"}},
        {"id": static, "widgets_values": ["static prompt"]},
    ]}]}}


def archive_bytes(*, extra=(), omit=None, content=None, compression=zipfile.ZIP_STORED):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=compression) as archive:
        for profile, (filename, _, _) in backend.QWEN_CHARACTER_SHEET_PROFILES.items():
            if profile != omit:
                archive.writestr(filename, json.dumps(document(profile)) if content is None else content)
        for filename, payload in extra:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                # ZipInfo normalizes Windows separators on construction. Set
                # the raw ZIP name afterwards to test malicious archive bytes.
                info = zipfile.ZipInfo("placeholder")
                info.filename = filename
                archive.writestr(info, payload)
    return buffer.getvalue()


def pin_archive(monkeypatch, raw):
    pin = dict(downloader.QWEN_CHARACTER_SHEET_ARCHIVE,
               size=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    monkeypatch.setattr(downloader, "QWEN_CHARACTER_SHEET_ARCHIVE", pin)
    monkeypatch.setattr(backend, "QWEN_CHARACTER_SHEET_SHA256", pin["sha256"])
    return pin


def mock_http(monkeypatch, handler):
    # Use the real class despite the fixture's monkeypatch of the module attribute.
    from httpx._client import Client

    monkeypatch.setattr(downloader.httpx, "Client", lambda **kw: Client(
        transport=httpx.MockTransport(handler), **kw,
    ))


def test_exact_pins_and_backend_contract():
    expected = (
        ("diffusion_models/qwen_image_2.1_bf16.safetensors", 14230280616,
         "89f4158d066cc33906a199fca85634f766892dd78f49b6698dabf187ac86c4bc"),
        ("text_encoders/qwen3vl_8b_bf16.safetensors", 17534334616,
         "68bdc82bc1b66851162ae656225e7e2068166b603db19bd5d5a3b90eb12669a9"),
        ("vae/qwen_image_2.1_vae_bf16.safetensors", 675509688,
         "bb21f7473051e1ac368515dd3f2e15cd44d7a11748ee8823e1ddca3e4876b7c9"),
        ("text_encoders/qwen3.5_4b_int8_convrot.safetensors", 5588607110,
         "088495aba6219cb8933339e0e433fc326b86787789025c62af4d855864b24455"),
    )
    assets = downloader.QWEN_CHARACTER_SHEET_ASSETS
    assert len(assets) == 4
    for index, (asset, (destination, size, digest)) in enumerate(zip(assets, expected)):
        assert asset == {
            "repo": ("Comfy-Org/Qwen-Image-2.1" if index < 3
                     else "Winnougan/Qwen-3.5-INT8-Convrot-Comfy"),
            "revision": ("cb504a4090723e43f17ad01cec0359490e2de613" if index < 3
                         else "e019237a5e495acfa039a1829822ab75dc642e71"),
            "remote": destination if index < 3 else Path(destination).name,
            "destination": destination, "size": size, "sha256": digest,
        }
    assert tuple(a["destination"] for a in assets[:3]) == backend.QWEN_CHARACTER_SHEET_FILES
    assert assets[3]["destination"] == backend.QWEN_CHARACTER_SHEET_PE
    assert downloader.QWEN_CHARACTER_SHEET_ARCHIVE == {
        "version_id": 3377047, "file_id": 3265393,
        "filename": "QwenImage21Character_qwenImage21V30.zip",
        "url": "https://civitai.com/api/download/models/3377047?fileId=3265393",
        "size": 33591,
        "sha256": "c9a760f237ee6044aa527949cca4ed0986539cbebda94002a7631710e06f2347",
    }
    assert downloader.qwen_character_sheet_archive_path() == (
        downloader.settings.model_dir / "workflows/qwen-character-sheet/Character_Sheet.zip"
    )
    assert downloader.RECOMMENDED_MODELS == ["qwen-2511", "qwen-2511-aio", "flux-klein-4b"]


@pytest.mark.parametrize("arguments", [
    ["--comfy-qwen-character-sheet"], ["qwen-2.1-sheet"],
    ["qwen-2.1-sheet", "--comfy-qwen-character-sheet"],
])
def test_cli_only_four_pinned_weights_and_archive(monkeypatch, arguments, capsys):
    raw = archive_bytes()
    pin_archive(monkeypatch, raw)
    weights = save({"test": np.zeros(1, dtype=np.float32)})
    assets = tuple(dict(a, size=len(weights), sha256=hashlib.sha256(weights).hexdigest())
                   for a in downloader.QWEN_CHARACTER_SHEET_ASSETS)
    monkeypatch.setattr(downloader, "QWEN_CHARACTER_SHEET_ASSETS", assets)
    calls = []

    def hub(**kwargs):
        calls.append(kwargs)
        asset = next(a for a in assets if a["remote"] == kwargs["filename"])
        assert kwargs["repo_id"] == asset["repo"] and kwargs["revision"] == asset["revision"]
        assert kwargs["token"] == "private-secret"
        destination = downloader.settings.comfy_dir / "models" / asset["destination"]
        assert not destination.exists()
        assert Path(kwargs["local_dir"]).parent == destination.parent
        output = Path(kwargs["local_dir"]) / kwargs["filename"]
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(weights)
        return str(output)

    requests = []

    def handler(request):
        requests.append(request)
        assert str(request.url) == downloader.QWEN_CHARACTER_SHEET_ARCHIVE["url"]
        assert request.headers["authorization"] == "Bearer private-secret"
        return httpx.Response(200, content=raw)

    monkeypatch.setattr(downloader, "hf_hub_download", hub)
    mock_http(monkeypatch, handler)
    monkeypatch.setattr(sys, "argv", ["download_models.py", *arguments])
    downloader.main()
    assert len(calls) == 4 and len(requests) == 1
    assert backend.missing_qwen_character_sheet_assets(downloader.settings.comfy_dir, "Auto") == []
    assert backend.load_qwen_character_sheet_prompts("Simple") == ("system prompt", "static prompt")
    assert backend.load_qwen_character_sheet_prompts("Production") == ("system prompt", "static prompt")
    monkeypatch.setattr(downloader, "_token", lambda _: pytest.fail("cache requested auth"))
    downloader.download_comfy_qwen_character_sheet()
    assert len(calls) == 4 and len(requests) == 1
    assert "private-secret" not in capsys.readouterr().out
    assert not list(downloader.settings.comfy_dir.rglob("*.staging-*"))
    assert not list(downloader.settings.model_dir.rglob("*.part"))


def test_missing_comfy_before_network():
    (downloader.settings.comfy_dir / "main.py").unlink()
    with pytest.raises(FileNotFoundError, match="setup-comfy"):
        downloader.download_comfy_qwen_character_sheet()


@pytest.mark.parametrize("failure", ["size", "hash", "header"])
@pytest.mark.parametrize("existing", [False, True])
def test_weights_fail_closed(monkeypatch, failure, existing):
    raw = b"not a tensor"
    asset = dict(downloader.QWEN_CHARACTER_SHEET_ASSETS[0], size=len(raw),
                 sha256=hashlib.sha256(raw).hexdigest())
    if failure == "size":
        asset["size"] += 1
    if failure == "hash":
        asset["sha256"] = "0" * 64
    target = downloader.settings.comfy_dir / "models"
    destination = target / asset["destination"]
    if existing:
        destination.parent.mkdir(parents=True)
        destination.write_bytes(raw)
    else:
        def hub(**kwargs):
            path = Path(kwargs["local_dir"]) / "test.safetensors"
            path.write_bytes(raw)
            return str(path)
        monkeypatch.setattr(downloader, "hf_hub_download", hub)
    with pytest.raises(RuntimeError, match="Qwen character sheet asset failed"):
        downloader._download_pinned_krea_asset(asset, target, label="Qwen character sheet")
    assert destination.exists() == existing
    if existing:
        assert destination.read_bytes() == raw
    assert not list(target.rglob("*.staging-*"))


@pytest.mark.parametrize("kind", ["size", "hash", "html", "truncated", "length", "transport", "race"])
def test_archive_failure_atomic_redacted(monkeypatch, tmp_path, kind, capsys, caplog):
    raw = archive_bytes()
    pin = pin_archive(monkeypatch, raw)
    destination = tmp_path / "Character_Sheet.zip"
    if kind == "hash":
        monkeypatch.setattr(downloader, "QWEN_CHARACTER_SHEET_ARCHIVE", dict(pin, sha256="0" * 64))

    def handler(request):
        logging.getLogger("httpx").critical("private-secret %s", request.url)
        if kind == "transport":
            raise httpx.ReadError("private-secret signed URL")
        if kind == "race":
            destination.write_bytes(b"concurrent")
        body = {"size": raw + b"x", "html": b"<html>private-secret</html>",
                "truncated": raw[:-1]}.get(kind, raw)
        headers = {"content-length": "99999999"} if kind == "length" else {}
        return httpx.Response(200, content=body, headers=headers)

    mock_http(monkeypatch, handler)
    with pytest.raises(RuntimeError) as error:
        downloader._download_qwen_sheet_archive(destination)
    assert error.value.__suppress_context__
    captured = capsys.readouterr()
    assert "private-secret" not in str(error.value) + captured.out + captured.err + caplog.text
    assert not list(tmp_path.glob("*.part"))
    assert destination.exists() == (kind == "race")
    if kind == "race":
        assert destination.read_bytes() == b"concurrent"


def test_invalid_existing_archive_preserved_without_auth(monkeypatch, tmp_path):
    path = tmp_path / "Character_Sheet.zip"
    path.write_bytes(b"old archive")
    monkeypatch.setattr(downloader, "_token", lambda _: pytest.fail("invalid cache requested auth"))
    with pytest.raises(RuntimeError):
        downloader._download_qwen_sheet_archive(path)
    assert path.read_bytes() == b"old archive"


@pytest.mark.parametrize("raw", [
    archive_bytes(omit="Simple"),
    archive_bytes(extra=[("Character_Sheet_Simple.json", "{}")]),
    archive_bytes(extra=[("../escape.json", "{}")]),
    archive_bytes(extra=[("C:/escape.json", "{}")]),
    archive_bytes(extra=[("/escape.json", "{}")]),
    archive_bytes(extra=[("dir\\escape.json", "{}")]),
    archive_bytes(extra=[(f"extra-{i}", "") for i in range(127)]),
    archive_bytes(content="not JSON"), archive_bytes(content="{}"),
    archive_bytes(content=" " * 100000, compression=zipfile.ZIP_DEFLATED),
], ids=["missing", "duplicate", "traversal", "drive", "absolute", "backslash",
        "member-count", "invalid-json", "missing-subgraph", "compression-bomb"])
def test_bad_hash_pinned_zip_not_installed(monkeypatch, tmp_path, raw):
    pin_archive(monkeypatch, raw)
    mock_http(monkeypatch, lambda _: httpx.Response(200, content=raw))
    path = tmp_path / "Character_Sheet.zip"
    with pytest.raises(RuntimeError):
        downloader._download_qwen_sheet_archive(path)
    assert not path.exists() and not list(tmp_path.glob("*.part"))


def test_member_limit_and_no_extract(monkeypatch, tmp_path):
    raw = archive_bytes()
    pin_archive(monkeypatch, raw)
    path = tmp_path / "Character_Sheet.zip"
    path.write_bytes(raw)
    monkeypatch.setattr(zipfile.ZipFile, "extractall", lambda *a, **kw: pytest.fail("extractall"))
    monkeypatch.setattr(zipfile.ZipFile, "extract", lambda *a, **kw: pytest.fail("extract"))
    downloader._validate_qwen_sheet_archive(path)
    monkeypatch.setattr(downloader, "_SHEET_MEMBER_LIMIT", 10)
    with pytest.raises(ValueError, match="Unsafe"):
        downloader._validate_qwen_sheet_archive(path)


def test_hash_before_zip_parse(monkeypatch, tmp_path):
    raw = archive_bytes()
    pin = pin_archive(monkeypatch, raw)
    path = tmp_path / "Character_Sheet.zip"
    path.write_bytes(b"x" * pin["size"])
    monkeypatch.setattr(zipfile, "ZipFile", lambda *a, **kw: pytest.fail("ZIP parsed before hash"))
    with pytest.raises(ValueError, match="SHA256"):
        downloader._validate_qwen_sheet_archive(path)


def test_bounded_unknown_length_stream(monkeypatch, tmp_path):
    raw = archive_bytes()
    pin_archive(monkeypatch, raw)
    consumed = []

    class Stream(httpx.SyncByteStream):
        def __iter__(self):
            for i in range(10):
                consumed.append(i)
                yield b"x" * 65536

    mock_http(monkeypatch, lambda _: httpx.Response(200, stream=Stream()))
    with pytest.raises(RuntimeError):
        downloader._download_qwen_sheet_archive(tmp_path / "Character_Sheet.zip")
    assert consumed == [0]
    assert not list(tmp_path.glob("*.part"))


def test_canonical_query_fallback_cross_origin_no_secrets(monkeypatch, tmp_path, caplog):
    raw = archive_bytes()
    pin_archive(monkeypatch, raw)
    requests = []

    def handler(request):
        requests.append(request)
        logging.getLogger("httpx").critical("private-secret %s", request.url)
        if request.url.host == "civitai.com":
            assert request.url.params["fileId"] == "3265393"
            assert request.headers["authorization"] == "Bearer private-secret"
            if "token" not in request.url.params:
                return httpx.Response(403)
            return httpx.Response(302, headers={"location": (
                "https://civitai.red/archive?token=private-secret&signature=public"
            )})
        assert "authorization" not in request.headers and "token" not in request.url.params
        assert request.url.params["signature"] == "public"
        return httpx.Response(200, content=raw)

    mock_http(monkeypatch, handler)
    destination = tmp_path / "Character_Sheet.zip"
    downloader._download_qwen_sheet_archive(destination)
    assert destination.read_bytes() == raw and len(requests) == 3
    assert "private-secret" not in caplog.text


@pytest.mark.parametrize("location", [
    "http://civitai.com/archive", "https://user:pass@civitai.com/archive",
    "https://evil.example/archive", "https://civitai.com.evil.example/archive",
    "https://civitai.com:8443/archive", "https://civitai.red/?leak=private-secret",
])
def test_unsafe_redirect_never_requested(monkeypatch, tmp_path, location):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(302, headers={"location": location})

    mock_http(monkeypatch, handler)
    with pytest.raises(RuntimeError):
        downloader._download_qwen_sheet_archive(tmp_path / "Character_Sheet.zip")
    assert len(requests) == 1


def test_powershell_syntax_preset_and_node_pins():
    root = Path(__file__).parents[1]
    download = root / "scripts/download-models.ps1"
    setup = root / "scripts/setup-comfy.ps1"
    text = setup.read_text()
    for name, repo, revision in (
        ("ComfyUI-KJNodes", "kijai", "d3cfe21625e5170126ce06fbfcfe1d88108688c3"),
        ("ComfyUI-DeGrid", "lunaaispace-eng", "5699bc33f71e1be12523fdea105cc9c2abfe1cd0"),
    ):
        assert f'https://github.com/{repo}/{name}.git' in text
        assert revision in text
    assert "checkout --detach $pin.Revision" in text
    assert "Keeping existing" in text and "git reset" not in text and "git pull" not in text
    assert "QwenImage21Cache" in text
    assert '--comfy-qwen-character-sheet' not in text  # setup never downloads weights
    assert "--token" not in download.read_text() and "Read-Host" not in download.read_text()
    pwsh = shutil.which("pwsh")
    if pwsh is None:
        pytest.skip("PowerShell unavailable")
    paths = ["'" + str(path).replace("'", "''") + "'" for path in (download, setup)]
    command = r"""
foreach ($path in @(DOWNLOAD_PATH, SETUP_PATH)) {
    $parseErrors = $null; $tokens = $null
    $null = [System.Management.Automation.Language.Parser]::ParseFile($path, [ref]$tokens, [ref]$parseErrors)
    if ($parseErrors.Count) { throw ($parseErrors | Out-String) }
}
function Set-Location {}
function Test-Path { return $true }
function Write-Host {}
function Join-Path { return 'MockPython' }
function MockPython { $script:captured = @($args); $global:LASTEXITCODE = 0 }
$wrapper = [scriptblock]::Create((Get-Content -Raw -LiteralPath DOWNLOAD_PATH))
& $wrapper -Preset qwen-character-sheet
if (($script:captured -join '|') -ne 'scripts\download_models.py|--comfy-qwen-character-sheet') {
    throw 'Unexpected wrapper arguments'
}
""".replace("DOWNLOAD_PATH", paths[0]).replace("SETUP_PATH", paths[1])
    result = subprocess.run([pwsh, "-NoProfile", "-NonInteractive", "-Command", command],
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr