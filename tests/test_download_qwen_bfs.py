from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

import httpx
import numpy as np
import pytest
from safetensors.numpy import save

from photo_edit_studio.swap import QWEN21_BFS_PINS, swap_profile
from scripts import download_models as downloader


@pytest.fixture(autouse=True)
def offline(monkeypatch, tmp_path):
    monkeypatch.setattr(downloader.settings, "lora_dir", tmp_path / "loras")
    monkeypatch.setattr(downloader, "_TOKENS", {})

    def forbidden(*args, **kwargs):
        pytest.fail("Unexpected network, credentials or base model download")

    for name in ("_token", "hf_hub_download", "snapshot_download", "urlretrieve"):
        monkeypatch.setattr(downloader, name, forbidden)
    monkeypatch.setattr(downloader.httpx, "Client", forbidden)


@pytest.fixture
def tiny_assets(monkeypatch):
    payload = save({"test": np.zeros(1, dtype=np.float32)})
    pins = tuple(dict(pin, size=len(payload), sha256=hashlib.sha256(payload).hexdigest())
                 for pin in downloader.QWEN21_BFS_ASSETS)
    monkeypatch.setattr(downloader, "QWEN21_BFS_ASSETS", pins)
    return payload, pins


def transport(monkeypatch, handler):
    from httpx._client import Client

    monkeypatch.setattr(downloader.httpx, "Client", lambda **kwargs: Client(
        transport=httpx.MockTransport(handler), **kwargs,
    ))


def destination(pin):
    return downloader.settings.lora_dir / "qwen21" / pin["filename"]


def test_exact_publisher_pins_and_legacy_names():
    expected = (
        (3356102, 3248321, "bfs_head_v1.1_qwen_2.1.safetensors",
         "Qwen21-BFS_Head_v1.1.safetensors", 260096144,
         "d1d748d5601077f3b6d05766f6823510e901970d916afa404a97e85dc92fa88e"),
        (3363725, 3251548, "bfs_body_swap_v1.0_qwen_2.1.safetensors",
         "Qwen21-BFS_Body_v1.1.safetensors", 209753576,
         "7dc0a53aba4dbc70c204f498936a619d226559da11011f748c389189af46b664"),
    )
    assert downloader.QWEN21_BFS_ASSETS == tuple({
        "version_id": version, "file_id": file_id, "filename": filename,
        "legacy_filename": legacy, "size": size, "sha256": digest,
        "url": f"https://civitai.com/api/download/models/{version}?fileId={file_id}",
    } for version, file_id, filename, legacy, size, digest in expected)
    for profile in downloader.SWAP_PROFILES.values():
        if profile.family == "qwen21":
            assert profile.filename in {row[2] for row in expected}
    for kind, row, version, pairs in zip(("Head", "Body"), expected, ("1.1", "1.0"),
                                         (176, 136), strict=True):
        assert QWEN21_BFS_PINS[kind] == {
            "version": version, "pairs": pairs, "size": row[4], "sha256": row[5],
        }
        assert swap_profile("qwen-2.1-turbo-official", kind).filename == row[2]


def test_powershell_singleton_preset_and_bfs_switch():
    script = (Path(__file__).parents[1] / "scripts/download-models.ps1").read_text(encoding="utf-8")
    choices = re.search(r"\[ValidateSet\((.*?)\)\]", script, re.DOTALL)
    assert choices is not None
    assert re.findall(r'"([^"]+)"', choices.group(1)).count("qwen-2.1-bfs") == 1
    bodies = re.findall(r'"qwen-2\.1-bfs"\s*\{([^}]*)\}', script)
    assert len(bodies) == 1
    assert re.fullmatch(r'\s*\$models\s*=\s*@\("--qwen21-bfs"\)\s*', bodies[0])
    assert re.search(r'if \(\$BfsSwap\)\s*\{\s*\$arguments \+= "--bfs-swap"\s*\}', script)


def test_verified_legacy_copies_atomically_keeps_originals_without_auth_or_network(
    monkeypatch, tiny_assets,
):
    payload, pins = tiny_assets
    copies = []
    real_copy = downloader.shutil.copyfile
    for pin in pins:
        path = destination(pin)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.with_name(pin["legacy_filename"]).write_bytes(payload)

    def copy(source, target):
        pin = next(pin for pin in pins if source.name == pin["legacy_filename"])
        assert target.parent == destination(pin).parent
        assert target.suffix == ".part" and not destination(pin).exists()
        copies.append((source, target))
        return real_copy(source, target)

    monkeypatch.setattr(downloader.shutil, "copyfile", copy)
    monkeypatch.setattr(sys, "argv", ["download_models.py", "--qwen21-bfs"])
    downloader.main()
    assert len(copies) == 2
    for pin in pins:
        assert destination(pin).read_bytes() == payload
        assert destination(pin).with_name(pin["legacy_filename"]).read_bytes() == payload
        assert not list(destination(pin).parent.glob("*.part"))
    downloader.main()  # verified canonical files skip copying/auth/network too
    assert len(copies) == 2


@pytest.mark.parametrize("failure", ["size", "hash", "header"])
def test_invalid_legacy_is_not_substituted_fetches_canonical(
    monkeypatch, tiny_assets, failure, capsys,
):
    payload, pins = tiny_assets
    content = {"size": payload[:-1], "hash": bytes([payload[0] ^ 1]) + payload[1:],
               "header": b"not safetensors"}[failure]
    if failure == "header":
        for pin in pins:
            pin.update(size=len(content), sha256=hashlib.sha256(content).hexdigest())
    calls = []
    for pin in pins:
        path = destination(pin)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.with_name(pin["legacy_filename"]).write_bytes(content)

    def fetch(path, pin, validate, *, label):
        assert path == destination(pin) and not path.exists()
        calls.append(pin)

    monkeypatch.setattr(downloader, "_download_pinned_civitai", fetch)
    downloader.download_qwen21_bfs()
    assert calls == list(pins)
    for pin in pins:
        assert destination(pin).with_name(pin["legacy_filename"]).read_bytes() == content
        assert not destination(pin).exists()
    assert capsys.readouterr().out.count("Fetching pinned canonical file") == 2


@pytest.mark.parametrize("failure", ["copy", "changed-source", "race"])
def test_migration_failure_preserves_legacy_cleans_staging_and_never_fetches(
    monkeypatch, tiny_assets, failure,
):
    payload, pins = tiny_assets
    pin = pins[0]
    path = destination(pin)
    path.parent.mkdir(parents=True)
    legacy = path.with_name(pin["legacy_filename"])
    legacy.write_bytes(payload)
    real_copy = downloader.shutil.copyfile

    def copy(source, target):
        if failure == "copy":
            raise OSError("private-secret")
        real_copy(source, target)
        if failure == "changed-source":
            target.write_bytes(b"changed after validation")
        else:
            path.write_bytes(b"concurrent destination")

    monkeypatch.setattr(downloader.shutil, "copyfile", copy)
    with pytest.raises(RuntimeError, match="local migration failed") as error:
        downloader.download_qwen21_bfs()
    assert "private-secret" not in str(error.value)
    assert legacy.read_bytes() == payload and not list(path.parent.glob("*.part"))
    assert path.read_bytes() == b"concurrent destination" if failure == "race" else not path.exists()


@pytest.mark.parametrize("token", [None, "private-secret"])
def test_cli_only_two_weights_canonical_delivery_and_auth_fallback(
    monkeypatch, tiny_assets, token, capsys, caplog,
):
    payload, pins = tiny_assets
    monkeypatch.setattr(downloader, "_token", lambda provider: token)
    requests = []

    def handler(request):
        requests.append(request)
        if request.url.host == "civitai.com":
            pin = next(pin for pin in pins if request.url.path.endswith(str(pin["version_id"])))
            assert request.url.params["fileId"] == str(pin["file_id"])
            assert request.headers.get("authorization") == (f"Bearer {token}" if token else None)
            if token and "token" not in request.url.params:
                assert str(request.url) == pin["url"]
                return httpx.Response(403)
            assert request.url.params.get("token") == token
            return httpx.Response(307, headers={"location": (
                f"https://{downloader.CIVITAI_DELIVERY_HOST}/{pin['file_id']}?signature=public"
                + (f"&token={token}&access_token={token}&api_key={token}" if token else "")
            )})
        assert request.url.host == downloader.CIVITAI_DELIVERY_HOST
        assert "authorization" not in request.headers
        assert not {"token", "access_token", "api_key"}.intersection(request.url.params)
        pin = next(pin for pin in pins if request.url.path == f"/{pin['file_id']}")
        assert not destination(pin).exists()
        return httpx.Response(200, content=payload)

    transport(monkeypatch, handler)
    monkeypatch.setattr(sys, "argv", ["download_models.py", "--qwen21-bfs"])
    downloader.main()
    assert len(requests) == (6 if token else 4)
    for pin in pins:
        assert destination(pin).read_bytes() == payload
    assert not list(downloader.settings.lora_dir.rglob("*.part"))
    captured = capsys.readouterr()
    assert "private-secret" not in captured.out + captured.err + caplog.text


@pytest.mark.parametrize("failure", [
    "hash", "header", "empty", "truncated", "oversize", "length", "transport",
    "host", "http", "userinfo", "credential-url", "race",
])
def test_failed_download_is_bounded_atomic_redacted_and_cleans_parts(
    monkeypatch, tiny_assets, failure, capsys, caplog,
):
    payload, pins = tiny_assets
    pin = pins[0]
    path = destination(pin)
    content = {"hash": bytes([payload[0] ^ 1]) + payload[1:], "header": b"invalid",
               "empty": save({}), "truncated": payload[:-1], "oversize": payload + b"x"}.get(failure, payload)
    if failure in {"header", "empty"}:
        pin.update(size=len(content), sha256=hashlib.sha256(content).hexdigest())
    monkeypatch.setattr(downloader, "_token", lambda provider: "private-secret")
    requests = []

    class Stream(httpx.SyncByteStream):
        def __iter__(self):
            yield content

    def handler(request):
        requests.append(request)
        assert str(request.url) == pin["url"]
        if failure == "transport":
            raise httpx.ReadError("private-secret")
        redirects = {
            "host": "https://unapproved.example/file", "http": "http://civitai.com/file",
            "userinfo": "https://user:password@civitai.com/file",
            "credential-url": f"https://{downloader.CIVITAI_DELIVERY_HOST}/file?leak=private-secret",
        }
        if failure in redirects:
            return httpx.Response(302, headers={"location": redirects[failure]})
        if failure == "race":
            path.write_bytes(b"concurrent destination")
        headers = {"content-length": str(len(payload) + 1)} if failure == "length" else {}
        return httpx.Response(200, stream=Stream(), headers=headers)

    transport(monkeypatch, handler)
    with pytest.raises(RuntimeError, match="Safetensors") as error:
        downloader.download_qwen21_bfs()
    assert len(requests) == 1 and error.value.__suppress_context__
    assert not list(path.parent.glob("*.part"))
    assert path.read_bytes() == b"concurrent destination" if failure == "race" else not path.exists()
    captured = capsys.readouterr()
    assert "private-secret" not in str(error.value) + captured.out + captured.err + caplog.text


def test_invalid_canonical_fails_closed_without_auth_or_legacy_substitution(tiny_assets):
    payload, pins = tiny_assets
    path = destination(pins[0])
    path.parent.mkdir(parents=True)
    path.write_bytes(b"invalid canonical")
    legacy = path.with_name(pins[0]["legacy_filename"])
    legacy.write_bytes(payload)
    with pytest.raises(RuntimeError, match="left unchanged"):
        downloader.download_qwen21_bfs()
    assert path.read_bytes() == b"invalid canonical" and legacy.read_bytes() == payload


def test_bfs_cli_includes_pinned_qwen_once_preserves_other_downloads(monkeypatch):
    qwen_calls = []
    hf_calls = []
    monkeypatch.setattr(downloader, "download_qwen21_bfs", lambda: qwen_calls.append(True))
    monkeypatch.setattr(downloader, "_hf_download", lambda function, **kwargs: hf_calls.append(kwargs))
    monkeypatch.setattr(sys, "argv", ["download_models.py", "--bfs-swap", "--qwen21-bfs"])
    downloader.main()
    assert qwen_calls == [True]
    expected = [profile for profile in downloader.SWAP_PROFILES.values() if profile.family != "qwen21"]
    assert len(hf_calls) == len(expected) == 4
    assert hf_calls == [{"repo_id": downloader.BFS_REPO, "filename": profile.filename,
                         "local_dir": downloader.settings.lora_dir / profile.family}
                        for profile in expected]