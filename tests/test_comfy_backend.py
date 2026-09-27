from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from photo_edit_studio import comfy_backend
from photo_edit_studio.config import settings


def test_endpoint_rejects_remote_host() -> None:
    assert comfy_backend._endpoint("http://127.0.0.1:8188") == ("127.0.0.1", 8188)
    with pytest.raises(ValueError, match="loopback"):
        comfy_backend._endpoint("http://example.com:8188")


def test_missing_embedded_install_is_actionable(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(settings, "comfy_dir", tmp_path)
    monkeypatch.setattr(comfy_backend, "is_ready", lambda: False)
    with pytest.raises(FileNotFoundError, match="setup-comfy.ps1"):
        comfy_backend.ensure_backend(enabled=True)


def test_existing_server_prevents_implicit_ownership(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(comfy_backend, "is_ready", lambda: True)
    with pytest.raises(RuntimeError, match="Port already occupied"):
        comfy_backend.ensure_backend(enabled=True)
    assert comfy_backend._process is None


def test_embedded_backend_starts_and_stops_own_process(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    root = tmp_path / "ComfyUI"
    root.mkdir()
    (root / "main.py").write_text("", encoding="utf-8")
    monkeypatch.setattr(settings, "comfy_dir", root)
    monkeypatch.setattr(settings, "model_dir", tmp_path / "models")
    monkeypatch.setattr(settings, "lora_dir", tmp_path / "models" / "loras")
    monkeypatch.setattr(settings, "output_dir", tmp_path / "outputs")
    monkeypatch.setattr(comfy_backend, "_process", None)
    monkeypatch.setattr(comfy_backend, "_log_handle", None)
    ready = iter([False, True])
    monkeypatch.setattr(comfy_backend, "is_ready", lambda: next(ready))
    monkeypatch.setattr(comfy_backend, "missing_assets", lambda _root: [])

    class FakeProcess:
        def __init__(self, args: list[str], **kwargs: object) -> None:
            self.args = args
            self.terminated = False

        def poll(self) -> None:
            return None

        def terminate(self) -> None:
            self.terminated = True

        def wait(self, timeout: int) -> int:
            return 0

    fake = SimpleNamespace(process=None)

    def start(args: list[str], **kwargs: object) -> FakeProcess:
        fake.process = FakeProcess(args, **kwargs)
        return fake.process

    monkeypatch.setattr(comfy_backend.subprocess, "Popen", start)
    try:
        assert comfy_backend.ensure_backend(enabled=True)
        assert Path(fake.process.args[1]) == (root / "main.py").resolve()
        assert Path(fake.process.args[1]).is_absolute()
        assert fake.process.args[fake.process.args.index("--extra-model-paths-config") + 1] == str(
            (root / "photo_edit_extra_paths.yaml").resolve()
        )
        assert "--disable-api-nodes" in fake.process.args
        assert "--lowvram" in fake.process.args
        extra_paths = root / "photo_edit_extra_paths.yaml"
        assert extra_paths.is_file()
        assert f"  loras: {json.dumps(str(settings.lora_dir.resolve()))}" in extra_paths.read_text()
    finally:
        comfy_backend.stop_backend()
    assert fake.process.terminated