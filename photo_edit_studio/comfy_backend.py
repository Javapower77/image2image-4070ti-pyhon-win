from __future__ import annotations

import atexit
import json
import subprocess
import sys
import time
from urllib.parse import urlsplit

import httpx

from photo_edit_studio.comfy_assets import (
    missing_assets,
    missing_firered_assets,
    missing_krea_remix_assets,
    missing_qwen21_assets,
)
from photo_edit_studio.config import settings

_process: subprocess.Popen[bytes] | None = None
_log_handle = None


def _endpoint(url: str) -> tuple[str, int]:
    parsed = urlsplit(url)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("Embedded ComfyUI must bind to an HTTP loopback address.")
    if parsed.path not in {"", "/"} or parsed.query or parsed.fragment or parsed.username:
        raise ValueError("ComfyUI URL must contain only a loopback host and port.")
    return parsed.hostname, parsed.port or 80


def is_ready() -> bool:
    try:
        with httpx.Client(timeout=2.0, trust_env=False) as client:
            return client.get(settings.comfy_url.rstrip("/") + "/system_stats").status_code == 200
    except httpx.HTTPError:
        return False


def stop_backend() -> None:
    global _process, _log_handle
    if _process is not None:
        if _process.poll() is None:
            _process.terminate()
            try:
                _process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                _process.kill()
                _process.wait(timeout=5)
        _process = None
    if _log_handle is not None:
        _log_handle.close()
        _log_handle = None


def ensure_backend(
    *, enabled: bool | None = None, model_key: str = "krea-2-turbo", workflow: str = "standard"
) -> bool:
    """Start only our own local ComfyUI; never claim an unrelated server is embedded."""
    global _process, _log_handle
    if enabled is None:
        enabled = settings.comfy_autostart
    if not enabled:
        return False
    host, port = _endpoint(settings.comfy_url)
    if _process is not None and _process.poll() is None:
        return True
    if is_ready():
        raise RuntimeError(
            f"Port already occupied at {settings.comfy_url}. Stop that server or disable "
            "PHOTO_EDIT_COMFY_AUTOSTART before starting the project-managed backend."
        )
    comfy_dir = settings.comfy_dir.resolve()
    main = comfy_dir / "main.py"
    if not main.is_file():
        raise FileNotFoundError(
            f"Embedded ComfyUI not installed at {main}. Run scripts/setup-comfy.ps1."
        )
    if model_key == "krea-2-turbo" and workflow == "krea-remix":
        missing = missing_krea_remix_assets(settings.comfy_dir)
        hint = "--comfy-krea (place Krea2-Remix_Patreon.safetensors in vendor/ComfyUI/models/loras manually)"
    elif model_key == "firered-1.1":
        missing = missing_firered_assets(settings.comfy_dir)
        hint = "firered-1.1"
    elif model_key == "qwen-2.1-turbo":
        missing = missing_qwen21_assets(settings.comfy_dir)
        hint = "qwen-2.1-turbo"
    else:
        missing = missing_assets(settings.comfy_dir)
        hint = "--comfy-krea"
    if missing:
        raise FileNotFoundError(
            "Embedded ComfyUI is missing model weights or custom nodes. Run "
            "scripts/setup-comfy.ps1 and scripts/download_models.py "
            f"{hint}. Missing: {', '.join(missing)}"
        )
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    extra_paths = comfy_dir / "photo_edit_extra_paths.yaml"
    extra_paths.write_text(
        "photo_edit_studio:\n"
        f"  base_path: {json.dumps(str(settings.model_dir.resolve()))}\n"
        f"  loras: {json.dumps(str((settings.lora_dir / 'krea2').resolve()))}\n"
        "photo_edit_qwen21:\n"
        f"  base_path: {json.dumps(str(settings.model_dir.resolve()))}\n"
        f"  loras: {json.dumps(str(settings.lora_dir.resolve()))}\n",
        encoding="utf-8",
    )
    _log_handle = (settings.output_dir / "comfyui.log").open("ab")
    args = [
        sys.executable, str(main), "--listen", host, "--port", str(port),
        "--disable-auto-launch", "--disable-api-nodes", "--lowvram",
        "--extra-model-paths-config", str(extra_paths),
    ]
    try:
        _process = subprocess.Popen(
            args, cwd=comfy_dir, stdout=_log_handle, stderr=subprocess.STDOUT
        )
        deadline = time.monotonic() + settings.comfy_start_timeout_seconds
        while time.monotonic() < deadline:
            if is_ready():
                return True
            if _process.poll() is not None:
                raise RuntimeError(f"Embedded ComfyUI exited during startup. See {settings.output_dir / 'comfyui.log'}.")
            time.sleep(1)
        raise TimeoutError(f"Embedded ComfyUI did not become ready. See {settings.output_dir / 'comfyui.log'}.")
    except BaseException:
        stop_backend()
        raise


atexit.register(stop_backend)