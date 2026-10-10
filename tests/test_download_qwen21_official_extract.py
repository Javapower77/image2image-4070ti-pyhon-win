from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path
from unittest.mock import Mock

import httpx
import numpy as np
import pytest
from safetensors.numpy import save
from test_download_qwen21_official import MODEL_INDEX, TRANSFORMER_CONFIG, write_snapshot

from photo_edit_studio.models import diffusers_adapters as adapters
from photo_edit_studio.models import qwen21_official_extract as assets
from scripts import download_models as downloader

KEY = "qwen-2.1-turbo-official-extract"
URL = "https://civitai.com/api/download/models/3394831?fileId=3284648"
FILENAME = "qwen_image_2.1_turbo_lora_avg_rank_178_bf16.safetensors"


@pytest.fixture(autouse=True)
def offline(monkeypatch, tmp_path):
    monkeypatch.setattr(downloader.settings, "model_dir", tmp_path / "models")
    monkeypatch.setattr(downloader.settings, "lora_dir", tmp_path / "loras")
    monkeypatch.setattr(downloader, "_TOKENS", {})
    monkeypatch.setattr(downloader, "_token", lambda _provider: "test-only-secret")

    def forbidden(*_args, **_kwargs):
        pytest.fail("Unexpected live download or inference")

    for name in ("snapshot_download", "hf_hub_download", "urlretrieve",
                 "download_comfy_qwen21", "download_comfy_qwen21_r128"):
        monkeypatch.setattr(downloader, name, forbidden)
    monkeypatch.setattr(downloader.httpx, "Client", forbidden)
    monkeypatch.setattr(adapters, "_runtime", forbidden)


@pytest.fixture
def tiny_asset(monkeypatch):
    payload = save({"test": np.zeros(1, dtype=np.float32)})
    pin = dict(assets.QWEN21_OFFICIAL_EXTRACT_LORA,
               size=len(payload), sha256=hashlib.sha256(payload).hexdigest())
    monkeypatch.setattr(assets, "QWEN21_OFFICIAL_EXTRACT_LORA", pin)
    monkeypatch.setattr(downloader, "QWEN21_OFFICIAL_EXTRACT_LORA", pin)
    return payload, pin


def transport(monkeypatch, handler):
    # The autouse guard replaces the module attribute; recover the real class
    # from its defining module, not from the guarded httpx.Client alias.
    from httpx._client import Client

    monkeypatch.setattr(downloader.httpx, "Client", lambda **kwargs: Client(
        transport=httpx.MockTransport(handler), **kwargs,
    ))


@pytest.mark.parametrize("valid_snapshot", [True, False])
def test_cli_full_official_snapshot_then_config_validation_then_mandatory_asset(
    monkeypatch, valid_snapshot,
):
    monkeypatch.setattr(sys, "argv", ["download_models.py", KEY])
    events = []
    destination = downloader.MODEL_SPECS[KEY].local_path
    files = {"model_index.json": MODEL_INDEX, "transformer/config.json": TRANSFORMER_CONFIG}
    if not valid_snapshot:
        files["transformer/config.json"] = b'{"causal_condition": false}'

    def snapshot(**kwargs):
        assert kwargs == {
            "repo_id": "Qwen/Qwen-Image-2.1-Turbo", "local_dir": destination,
            "local_dir_use_symlinks": False, "token": "test-only-secret",
        }
        write_snapshot(destination, files)
        events.append("full Turbo snapshot")

    real_validate = adapters.validate_qwen21_official_config

    def validate(path):
        assert path == destination and events == ["full Turbo snapshot"]
        real_validate(path)
        events.append("saved config validated")

    monkeypatch.setattr(downloader, "snapshot_download", snapshot)
    monkeypatch.setattr(adapters, "validate_qwen21_official_config", validate)
    mandatory = Mock(side_effect=lambda path: events.append(("mandatory", path)))
    monkeypatch.setattr(downloader, "_download_qwen21_official_extract", mandatory)
    if valid_snapshot:
        downloader.main()
        assert events == ["full Turbo snapshot", "saved config validated",
                          ("mandatory", assets.qwen21_official_extract_path())]
    else:
        with pytest.raises(RuntimeError, match="causal_condition"):
            downloader.main()
        mandatory.assert_not_called()
        assert events == ["full Turbo snapshot"]
    for filename, content in files.items():
        assert (destination / filename).read_bytes() == content
    assert KEY not in downloader.RECOMMENDED_MODELS


def test_powershell_extract_preset_exact_singleton_preserves_unstacked_preset():
    script = (Path(__file__).parents[1] / "scripts/download-models.ps1").read_text(encoding="utf-8")
    validate_set = re.search(r"\[ValidateSet\((.*?)\)\]", script, re.DOTALL)
    assert validate_set is not None
    for preset, key in [("qwen-2.1-official-extract", KEY),
                        ("qwen-2.1-official", "qwen-2.1-turbo-official")]:
        assert re.findall(r'"([^"]+)"', validate_set.group(1)).count(preset) == 1
        bodies = re.findall(r'"' + re.escape(preset) + r'"\s*\{([^}]*)\}', script)
        assert len(bodies) == 1
        assert re.fullmatch(r'\s*\$models\s*=\s*@\("' + re.escape(key) + r'"\)\s*', bodies[0])


@pytest.mark.parametrize("token", [None, "test-only-secret"])
def test_canonical_exact_file_delivery_redirect_strips_bearer_cross_origin(
    monkeypatch, tmp_path, tiny_asset, token,
):
    payload, _pin = tiny_asset
    monkeypatch.setattr(downloader, "_token", lambda _provider: token)
    destination = tmp_path / FILENAME
    requests = []

    def handler(request):
        requests.append(request)
        if len(requests) == 1:
            assert str(request.url) == URL
            assert request.headers.get("authorization") == (f"Bearer {token}" if token else None)
            return httpx.Response(307, headers={"location": (
                f"https://{downloader.CIVITAI_DELIVERY_HOST}/file?signature=public"
                + (f"&token={token}&access_token={token}&api_key={token}" if token else "")
            )})
        assert request.url.host == downloader.CIVITAI_DELIVERY_HOST
        assert "authorization" not in request.headers
        assert not {"token", "access_token", "api_key"}.intersection(request.url.params)
        assert request.url.params["signature"] == "public"
        assert not destination.exists()
        return httpx.Response(200, content=payload)

    transport(monkeypatch, handler)
    downloader._download_qwen21_official_extract(destination)
    assert destination.read_bytes() == payload and len(requests) == 2
    assert not list(tmp_path.glob("*.part"))


def test_query_auth_fallback_then_cross_origin_return_never_restores_bearer(
    monkeypatch, tmp_path, tiny_asset, capsys, caplog,
):
    payload, _pin = tiny_asset
    requests = []

    def handler(request):
        requests.append(request)
        if len(requests) == 1:
            assert str(request.url) == URL
            return httpx.Response(401)
        if len(requests) == 2:
            assert request.url.params["fileId"] == "3284648"
            assert request.url.params["token"] == "test-only-secret"
            assert request.headers["authorization"] == "Bearer test-only-secret"
            return httpx.Response(302, headers={"location": f"https://{downloader.CIVITAI_DELIVERY_HOST}/file"})
        assert "authorization" not in request.headers and "token" not in request.url.params
        if len(requests) == 3:
            return httpx.Response(302, headers={"location": URL})
        assert request.url.host == "civitai.com"
        return httpx.Response(200, content=payload)

    transport(monkeypatch, handler)
    downloader._download_qwen21_official_extract(tmp_path / FILENAME)
    assert len(requests) == 4
    captured = capsys.readouterr()
    assert "test-only-secret" not in captured.out + captured.err + caplog.text


def test_existing_verified_hash_and_header_skips_credentials_and_network(monkeypatch, tmp_path, tiny_asset):
    destination = tmp_path / FILENAME
    destination.write_bytes(tiny_asset[0])
    monkeypatch.setattr(downloader, "_token", lambda _provider: pytest.fail("Token requested"))
    downloader._download_qwen21_official_extract(destination)
    assert destination.read_bytes() == tiny_asset[0] and not list(tmp_path.glob("*.part"))


@pytest.mark.parametrize("failure", ["size", "hash", "header", "empty"])
def test_existing_corrupt_file_fails_closed_without_replacement_or_network(
    monkeypatch, tmp_path, tiny_asset, failure,
):
    payload, pin = tiny_asset
    content = {"size": payload[:-1], "hash": bytes([payload[0] ^ 1]) + payload[1:],
               "header": b"invalid header", "empty": save({})}[failure]
    if failure in {"header", "empty"}:
        pin.update(size=len(content), sha256=hashlib.sha256(content).hexdigest())
    destination = tmp_path / FILENAME
    destination.write_bytes(content)
    monkeypatch.setattr(downloader, "_token", lambda _provider: pytest.fail("Token requested"))
    with pytest.raises(RuntimeError, match="Existing archive/file left unchanged"):
        downloader._download_qwen21_official_extract(destination)
    assert destination.read_bytes() == content and not list(tmp_path.glob("*.part"))


@pytest.mark.parametrize("failure", [
    "hash", "header", "empty", "truncated", "oversize", "length", "transport",
    "auth", "unapproved-host", "insecure", "userinfo", "credential-url", "race",
])
def test_failed_staged_download_is_atomic_cleans_parts_and_redacts(
    monkeypatch, tmp_path, tiny_asset, failure, capsys, caplog,
):
    payload, pin = tiny_asset
    destination = tmp_path / FILENAME
    content = {"hash": bytes([payload[0] ^ 1]) + payload[1:], "header": b"invalid header",
               "empty": save({}), "truncated": payload[:-1], "oversize": payload + b"x"}.get(failure, payload)
    if failure in {"header", "empty"}:
        pin.update(size=len(content), sha256=hashlib.sha256(content).hexdigest())
    requests = []

    def handler(request):
        requests.append(request)
        assert str(request.url) == URL or (failure == "auth" and request.url.params.get("token"))
        if failure == "transport":
            raise httpx.ReadError("test-only-secret")
        if failure == "auth":
            return httpx.Response(403, text="test-only-secret")
        redirects = {
            "unapproved-host": "https://unapproved.example/file",
            "insecure": "http://civitai.com/file",
            "userinfo": "https://user:password@civitai.com/file",
            "credential-url": f"https://{downloader.CIVITAI_DELIVERY_HOST}/file?leak=test-only-secret",
        }
        if failure in redirects:
            return httpx.Response(302, headers={"location": redirects[failure]})
        if failure == "race":
            destination.write_bytes(b"concurrent destination")
        headers = {"content-length": str(len(payload) + 1)} if failure == "length" else {}
        return httpx.Response(200, content=content, headers=headers)

    transport(monkeypatch, handler)
    with pytest.raises(RuntimeError, match="CIVITAI_API_TOKEN") as error:
        downloader._download_qwen21_official_extract(destination)
    assert error.value.__suppress_context__
    assert not list(tmp_path.glob("*.part"))
    if failure == "race":
        assert destination.read_bytes() == b"concurrent destination"
    else:
        assert not destination.exists()
    assert len(requests) == (2 if failure == "auth" else 1)
    captured = capsys.readouterr()
    assert "test-only-secret" not in str(error.value) + captured.out + captured.err + caplog.text