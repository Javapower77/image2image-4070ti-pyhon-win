from __future__ import annotations

import hashlib
import logging
import sys
from pathlib import Path

import httpx
import numpy as np
import pytest
from safetensors.numpy import save

from scripts import download_models as downloader


@pytest.fixture(autouse=True)
def isolated_auth(monkeypatch):
    downloader._TOKENS.clear()
    for name in ("HF_TOKEN", "CIVITAI_API_TOKEN", "CIVITAI_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(downloader, "get_token", lambda: None)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    yield
    downloader._TOKENS.clear()


@pytest.mark.parametrize("provider,variable", [
    ("HF", "HF_TOKEN"), ("Civitai", "CIVITAI_API_TOKEN"), ("Civitai", "CIVITAI_TOKEN"),
])
def test_environment_tokens_never_prompt_or_print(monkeypatch, capsys, provider, variable):
    monkeypatch.setenv(variable, "private-secret")
    monkeypatch.setattr(downloader.getpass, "getpass", lambda _: pytest.fail("prompt"))
    assert downloader._token(provider) == "private-secret"
    assert "private-secret" not in capsys.readouterr().out


def test_cached_hf_token_and_environment_precedence(monkeypatch):
    monkeypatch.setattr(downloader, "get_token", lambda: "cached-secret")
    assert downloader._token("HF") == "cached-secret"
    downloader._TOKENS.clear()
    monkeypatch.setenv("HF_TOKEN", "env-secret")
    assert downloader._token("HF") == "env-secret"
    monkeypatch.setenv("CIVITAI_API_TOKEN", "primary-secret")
    monkeypatch.setenv("CIVITAI_TOKEN", "alias-secret")
    assert downloader._token("Civitai") == "primary-secret"


@pytest.mark.parametrize("provider", ["HF", "Civitai"])
@pytest.mark.parametrize("answer", ["", "hidden-secret"])
def test_hidden_prompt_once_blank_allowed(monkeypatch, provider, answer, capsys):
    calls = []
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(downloader.getpass, "getpass", lambda prompt: calls.append(prompt) or answer)
    assert downloader._token(provider) == (answer or None)
    assert downloader._token(provider) == (answer or None)
    assert len(calls) == 1 and "blank for public" in calls[0]
    assert "hidden-secret" not in capsys.readouterr().out


@pytest.mark.parametrize("provider", ["HF", "Civitai"])
def test_noninteractive_public_has_env_help_no_prompt(monkeypatch, provider, capsys):
    monkeypatch.setattr(downloader.getpass, "getpass", lambda _: pytest.fail("prompt"))
    assert downloader._token(provider) is None
    assert downloader.AUTH_HELP[provider] in capsys.readouterr().out


def test_getpass_echo_fallback_forbidden(monkeypatch):
    import warnings

    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)

    def insecure(_):
        warnings.warn("No terminal", downloader.getpass.GetPassWarning)
        pytest.fail("Would read echoed input")

    monkeypatch.setattr(downloader.getpass, "getpass", insecure)
    with pytest.raises(RuntimeError, match="Secure token prompt unavailable"):
        downloader._token("HF")


def test_hf_token_forwarding_and_sanitized_failure(monkeypatch, caplog):
    monkeypatch.setenv("HF_TOKEN", "private-secret")
    calls = []

    def success(**kwargs):
        calls.append(kwargs)
        return "ok"

    assert downloader._hf_download(success, repo_id="public/repo") == "ok"
    assert calls == [{"repo_id": "public/repo", "token": "private-secret"}]

    def failure(**kwargs):
        logging.getLogger("httpx").critical("https://example.org/?token=private-secret")
        raise RuntimeError("Authorization: Bearer private-secret")

    with pytest.raises(RuntimeError, match="HF_TOKEN") as error:
        downloader._hf_download(failure)
    assert "private-secret" not in str(error.value) + caplog.text
    assert error.value.__suppress_context__


def test_public_hf_explicitly_disables_implicit_token():
    assert downloader._hf_download(lambda **kwargs: kwargs["token"]) is False


@pytest.fixture
def weights():
    return save({"test": np.zeros(1, dtype=np.float32)})


def mock_client(monkeypatch, handler):
    real_client = httpx.Client
    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(downloader.httpx, "Client", lambda **kwargs: real_client(
        transport=transport, **kwargs,
    ))


def metadata(weights, *, sha=True, file_id=3273779, name=None):
    return {"id": 3384956, "files": [{
        "id": file_id, "name": name or downloader.QWEN21_R128_FILENAME,
        "hashes": {"SHA256": hashlib.sha256(weights).hexdigest().upper()} if sha else {},
    }]}


@pytest.mark.parametrize("sha", [True, False])
def test_exact_stream_atomic_download(monkeypatch, tmp_path, weights, sha):
    destination = tmp_path / downloader.QWEN21_R128_FILENAME
    destination.write_bytes(b"old-invalid-html")
    requests = []
    monkeypatch.setenv("CIVITAI_API_TOKEN", "private-secret")

    def handler(request):
        requests.append(request)
        assert request.headers["Authorization"] == "Bearer private-secret"
        if "/model-versions/" in request.url.path:
            return httpx.Response(200, json=metadata(weights, sha=sha))
        assert str(request.url) == downloader.QWEN21_R128_URL
        assert destination.read_bytes() == b"old-invalid-html"
        return httpx.Response(200, content=weights)

    mock_client(monkeypatch, handler)
    downloader._download_civitai_r128(destination)
    assert destination.read_bytes() == weights
    assert len(requests) == 2
    assert not list(tmp_path.glob("*.part"))


def test_existing_valid_file_skips_network_and_prompt(monkeypatch, tmp_path, weights):
    destination = tmp_path / downloader.QWEN21_R128_FILENAME
    destination.write_bytes(weights)
    monkeypatch.setattr(downloader, "_token", lambda _: pytest.fail("token requested"))
    monkeypatch.setattr(downloader.httpx, "Client", lambda **_: pytest.fail("network"))
    downloader._download_civitai_r128(destination)


@pytest.mark.parametrize("failure", ["sha", "html", "truncated", "file", "name", "version", "bad-hash", "network", "auth"])
def test_failed_download_preserves_destination_cleans_temp_and_redacts(
    monkeypatch, tmp_path, weights, failure, capsys, caplog,
):
    destination = tmp_path / downloader.QWEN21_R128_FILENAME
    destination.write_bytes(b"old")
    monkeypatch.setenv("CIVITAI_API_TOKEN", "private-secret")
    data = metadata(weights)
    if failure == "file":
        data["files"][0]["id"] = 1
    if failure == "name":
        data["files"][0]["name"] = "other.safetensors"
    if failure == "version":
        data["id"] = 1
    if failure == "bad-hash":
        data["files"][0]["hashes"]["SHA256"] = "private-secret"
    if failure in {"html", "truncated"}:
        data["files"][0]["hashes"] = {}

    def handler(request):
        if "/model-versions/" in request.url.path:
            return httpx.Response(200, json=data)
        if failure == "network":
            raise httpx.ReadError("https://example.org/?token=private-secret")
        if failure == "auth":
            return httpx.Response(401, text="private-secret")
        content = {
            "sha": weights + b"wrong", "html": b"<html>private-secret</html>",
            "truncated": weights[:20],
        }.get(failure, weights)
        return httpx.Response(200, content=content)

    mock_client(monkeypatch, handler)
    with pytest.raises(RuntimeError, match="CIVITAI_API_TOKEN") as error:
        downloader._download_civitai_r128(destination)
    assert error.value.__suppress_context__
    assert destination.read_bytes() == b"old"
    assert not list(tmp_path.glob("*.part"))
    captured = capsys.readouterr()
    assert "private-secret" not in str(error.value) + captured.out + captured.err + caplog.text


def test_query_fallback_redirect_strips_secrets_and_logs(monkeypatch, tmp_path, weights, caplog):
    monkeypatch.setenv("CIVITAI_API_TOKEN", "private-secret")
    requests = []

    def handler(request):
        requests.append(request)
        logging.getLogger("httpx").critical("request %s", request.url)
        if "/model-versions/" in request.url.path:
            return httpx.Response(200, json=metadata(weights))
        if request.url.host == "civitai.red":
            assert request.headers["Authorization"] == "Bearer private-secret"
            if "token" not in request.url.params:
                return httpx.Response(401)
            assert request.url.params["fileId"] == "3273779"
            assert request.url.params["token"] == "private-secret"
            return httpx.Response(302, headers={"location": (
                "https://cdn.example.org/model?token=private-secret&signature=public-signature"
            )})
        assert "authorization" not in request.headers
        assert "token" not in request.url.params
        assert request.url.params["signature"] == "public-signature"
        return httpx.Response(200, content=weights)

    mock_client(monkeypatch, handler)
    downloader._download_civitai_r128(tmp_path / downloader.QWEN21_R128_FILENAME)
    assert len(requests) == 4
    assert "private-secret" not in caplog.text


@pytest.mark.parametrize("location", ["http://cdn.example.org/model", "https://user:password@cdn.example.org/model", "https://cdn.example.org/model?leak=private-secret"])
def test_unsafe_redirect_not_followed(monkeypatch, tmp_path, weights, location):
    monkeypatch.setenv("CIVITAI_API_TOKEN", "private-secret")

    def handler(request):
        assert request.url.host == "civitai.red"
        if "/model-versions/" in request.url.path:
            return httpx.Response(200, json=metadata(weights))
        return httpx.Response(302, headers={"location": location})

    mock_client(monkeypatch, handler)
    with pytest.raises(RuntimeError):
        downloader._download_civitai_r128(tmp_path / downloader.QWEN21_R128_FILENAME)
    assert not list(tmp_path.glob("*.part"))


def test_r128_assets_shared_no_viggle_or_bf16_snapshot(monkeypatch, tmp_path):
    monkeypatch.setattr(downloader.settings, "comfy_dir", tmp_path)
    (tmp_path / "main.py").touch()
    calls = []
    monkeypatch.setattr(downloader, "_download_named_file", lambda *args: calls.append(args))
    monkeypatch.setattr(downloader, "_download_civitai_r128", lambda path: calls.append(path))
    downloader.download_comfy_qwen21_r128()
    assert calls[:3] == [(downloader.QWEN21_COMFY_REPO, filename, tmp_path / "models" / filename)
                         for filename in downloader.QWEN21_FILES[:3]]
    assert calls[3] == tmp_path / "models" / "loras" / downloader.QWEN21_R128_FILENAME
    assert len(calls) == 4


@pytest.mark.parametrize("key,expected", [(downloader.QWEN21_R128_KEY, "r128"), ("qwen-2.1-turbo", "r256")])
def test_registry_dispatch_preserves_old_cli(monkeypatch, key, expected):
    calls = []
    monkeypatch.setattr(sys, "argv", ["download_models.py", key])
    monkeypatch.setattr(downloader, "download_comfy_qwen21_r128", lambda: calls.append("r128"))
    monkeypatch.setattr(downloader, "download_comfy_qwen21", lambda: calls.append("r256"))
    monkeypatch.setattr(downloader, "snapshot_download", lambda **_: pytest.fail("full snapshot"))
    downloader.main()
    assert calls == [expected]


def test_powershell_presets_and_no_token_arguments():
    script = (Path(__file__).parents[1] / "scripts" / "download-models.ps1").read_text()
    for preset in ("recommended", "qwen-aio", "qwen-2.1", "qwen-2.1-r128", "flux-4b", "krea-2", "firered", "all"):
        assert f'"{preset}"' in script
    assert '"qwen-2.1-r128" { $models = @("qwen-2.1-turbo-r128") }' in script
    assert "--token" not in script and "Read-Host" not in script