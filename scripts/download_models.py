from __future__ import annotations

import argparse
import getpass
import hashlib
import inspect
import json
import logging
import os
import re
import sys
import tempfile
import warnings
from contextlib import contextmanager
from pathlib import Path
from urllib.request import urlretrieve

import httpx
from huggingface_hub import get_token, hf_hub_download, snapshot_download
from safetensors import SafetensorError, safe_open

from photo_edit_studio.comfy_assets import (
    COMFY_KREA_FILES,
    COMFY_KREA_REPO,
    FIRERED_ENCODER,
    FIRERED_ENCODER_REMOTE,
    FIRERED_ENCODER_REPO,
    FIRERED_FILES,
    FIRERED_REPO,
    KREA_EDIT_FILE,
    KREA_EDIT_REPO,
    KREA_FIRST_LORA_FILE,
    KREA_FIRST_LORA_REMOTE,
    KREA_FIRST_LORA_REPO,
    KREA_TURBO_LORA_FILE,
    KREA_TURBO_LORA_REMOTE,
    QWEN21_COMFY_REPO,
    QWEN21_FILES,
    QWEN21_TURBO_LORA,
    QWEN21_TURBO_NODE,
    QWEN21_VIGGLE_REPO,
)
from photo_edit_studio.config import settings
from photo_edit_studio.models import MODEL_SPECS
from photo_edit_studio.models.qwen_aio import is_safetensors_file
from photo_edit_studio.swap import BFS_REPO, SWAP_PROFILES

GFPGAN_URL = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth"
QWEN_AIO_REPO = "Phr00t/Qwen-Image-Edit-Rapid-AIO"
QWEN_AIO_REMOTE_FILE = "v23/Qwen-Rapid-AIO-NSFW-v23.safetensors"
QWEN_AIO_LOCAL_NAME = "Qwen-Rapid-AIO.safetensors"
RECOMMENDED_MODELS = ["qwen-2511", "qwen-2511-aio", "flux-klein-4b"]
QWEN21_R128_KEY = "qwen-2.1-turbo-r128"
QWEN21_R128_FILENAME = "Qwen-Image-2.1-turbo-v0.2.1-6step-lora-r128.safetensors"
QWEN21_R128_URL = "https://civitai.red/api/download/models/3384956?fileId=3273779"
# These are pinned bytes, not evidence that the historical transformer was identical.
# Destinations are relative to the embedded ComfyUI models directory.
KREA_ORIGINAL_ASSETS = (
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
_TOKENS: dict[str, str | None] = {}
AUTH_HELP = {
    "HF": "Set HF_TOKEN or sign in with the Hugging Face CLI for gated assets.",
    "Civitai": "Set CIVITAI_API_TOKEN (or CIVITAI_TOKEN) for authenticated downloads.",
}


def _token(provider: str) -> str | None:
    if provider in _TOKENS:
        return _TOKENS[provider]
    value = (os.environ.get("HF_TOKEN") or get_token()) if provider == "HF" else (
        os.environ.get("CIVITAI_API_TOKEN") or os.environ.get("CIVITAI_TOKEN")
    )
    if not value:
        if sys.stdin.isatty():
            try:
                # Never permit getpass's echoing fallback when no secure terminal exists.
                with warnings.catch_warnings():
                    warnings.simplefilter("error", getpass.GetPassWarning)
                    value = getpass.getpass(f"{provider} token (hidden; blank for public): ").strip()
            except (EOFError, getpass.GetPassWarning):
                raise RuntimeError(f"Secure token prompt unavailable. {AUTH_HELP[provider]}") from None
        else:
            print(f"Noninteractive: trying public {provider} downloads. {AUTH_HELP[provider]}")
    _TOKENS[provider] = value or None
    return _TOKENS[provider]


@contextmanager
def _quiet_network():
    # HTTP debug logs and exception URLs can contain bearer tokens or signed queries.
    previous = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    try:
        yield
    finally:
        logging.disable(previous)


def _hf_download(function, **kwargs):
    token = _token("HF")
    try:
        # Keep compatible with injected download callables that expose only the
        # original repo/filename/local_dir contract. Hub functions support token.
        parameters = inspect.signature(function).parameters
        if "token" in parameters or any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
        ):
            kwargs["token"] = token or False
        with _quiet_network():
            return function(**kwargs)
    except Exception:  # noqa: BLE001 - SDK errors may include credential-bearing URLs
        raise RuntimeError(f"Hugging Face download failed. {AUTH_HELP['HF']}") from None


@contextmanager
def _civitai_stream(client, url: str, token: str | None, *, query_fallback=False):
    current = httpx.URL(url)
    origin = current.copy_with(path="/", query=None)
    credentials_allowed = True
    used_query = False
    for _ in range(10):
        if current.scheme != "https" or current.userinfo:
            raise RuntimeError("Unsafe Civitai download redirect rejected.")
        headers = {"Authorization": f"Bearer {token}"} if token and credentials_allowed else {}
        with client.stream("GET", current, headers=headers) as response:
            if (response.status_code in (401, 403) and token and query_fallback
                    and not used_query and credentials_allowed):
                current = current.copy_add_param("token", token)
                used_query = True
                continue
            if response.is_redirect:
                location = response.headers.get("location")
                if not location:
                    raise RuntimeError("Civitai redirect has no destination.")
                current = current.join(location)
                if current.copy_with(path="/", query=None) != origin:
                    credentials_allowed = False
                if not credentials_allowed:
                    for name in ("token", "access_token", "api_key"):
                        current = current.copy_remove_param(name)
                    if token and (token in str(current) or token in str(current.copy_with(query=None))):
                        raise RuntimeError("Credential-bearing cross-host redirect rejected.")
                continue
            if response.status_code != 200:
                raise RuntimeError(
                    f"Civitai request failed (HTTP {response.status_code}). {AUTH_HELP['Civitai']}"
                )
            yield response
            return
    raise RuntimeError("Too many Civitai download redirects.")


def _valid_safetensors(path: Path) -> bool:
    try:
        with safe_open(path, framework="pt", device="cpu") as handle:
            return bool(handle.keys())
    except (SafetensorError, OSError, ValueError):
        return False


def _download_civitai_r128(destination: Path) -> None:
    if _valid_safetensors(destination):
        print(f"Already present: {destination}")
        return
    token = _token("Civitai")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with _quiet_network(), httpx.Client(follow_redirects=False, timeout=120) as client:
            with _civitai_stream(
                client, "https://civitai.red/api/v1/model-versions/3384956", token
            ) as response:
                metadata = response.read()
                version = json.loads(metadata)
            if version.get("id") != 3384956:
                raise RuntimeError("Unexpected Civitai model version.")
            file = next((item for item in version.get("files", [])
                         if item.get("id") == 3273779), None)
            if file is None or file.get("name") != QWEN21_R128_FILENAME:
                raise RuntimeError("Exact Civitai r128 file is unavailable or has changed.")
            expected = file.get("hashes", {}).get("SHA256")
            if expected and not re.fullmatch(r"[0-9a-fA-F]{64}", expected):
                raise RuntimeError("Invalid Civitai SHA256 metadata.")
            print(f"Downloading Civitai version 3384956 / file 3273779 -> {destination}")
            digest = hashlib.sha256()
            with tempfile.NamedTemporaryFile(
                dir=destination.parent, prefix=destination.name + ".", suffix=".part", delete=False
            ) as output:
                temporary = Path(output.name)
                with _civitai_stream(client, QWEN21_R128_URL, token, query_fallback=True) as response:
                    for chunk in response.iter_bytes(chunk_size=1024 * 1024):
                        output.write(chunk)
                        digest.update(chunk)
                output.flush()
                os.fsync(output.fileno())
            if expected and digest.hexdigest().casefold() != expected.casefold():
                raise RuntimeError("Civitai file SHA256 mismatch.")
            if not _valid_safetensors(temporary):
                raise RuntimeError("Civitai response is not a valid Safetensors model (possibly HTML).")
            temporary.replace(destination)
    except Exception:  # noqa: BLE001 - sanitize all transport/server/validation failures
        # Do not expose response bodies, URLs, exception chains, or server-controlled text.
        raise RuntimeError(
            "Civitai r128 download failed: check connectivity, exact file availability, "
            f"SHA256 and Safetensors validity. {AUTH_HELP['Civitai']}"
        ) from None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _download_named_file(repo: str, remote: str, destination: Path) -> None:
    if destination.is_file():
        print(f"Already present: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {repo}/{remote} -> {destination}")
    downloaded = Path(_hf_download(hf_hub_download, repo_id=repo, filename=remote, local_dir=destination.parent))
    if downloaded != destination:
        downloaded.replace(destination)


def _pinned_asset_error(path: Path, asset: dict) -> str | None:
    """Verify bytes without tensor materialization or deserializing pickle checkpoints."""
    if path.stat().st_size != asset["size"]:
        return f"size mismatch (expected {asset['size']} bytes)"
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != asset["sha256"]:
        return "SHA256 mismatch"
    if path.suffix == ".safetensors" and not _valid_safetensors(path):
        return "invalid Safetensors header"
    return None


def _download_pinned_krea_asset(asset: dict, target: Path) -> None:
    destination = target / asset["destination"]
    # Never silently replace a stale file. Move/remove it explicitly and rerun
    # to download a replacement into same-filesystem staging before installation.
    detail = "local file verification failed"
    try:
        if destination.exists():
            error = _pinned_asset_error(destination, asset)
            if error:
                detail = (
                    f"Existing file: {error}. Left unchanged; move/remove it and rerun."
                )
                raise RuntimeError(detail)
            print(f"Already present (size/SHA256 verified): {destination}")
            return
        destination.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {asset['repo']}@{asset['revision']}/{asset['remote']} -> {destination}")
        detail = f"pinned Hugging Face download failed. {AUTH_HELP['HF']}"
        with tempfile.TemporaryDirectory(
            dir=destination.parent, prefix=destination.name + ".staging-"
        ) as staging:
            downloaded = Path(_hf_download(
                hf_hub_download,
                repo_id=asset["repo"],
                revision=asset["revision"],
                filename=asset["remote"],
                local_dir=Path(staging),
            ))
            detail = "downloaded file verification failed"
            error = _pinned_asset_error(downloaded, asset)
            if error:
                detail = f"Downloaded file: {error}; not installed."
                raise RuntimeError(detail)
            # A concurrently installed file must also pass verification; do not
            # overwrite it just because it appeared while the download was running.
            if destination.exists():
                detail = "Destination appeared during download; left unchanged. Rerun to verify it."
                raise RuntimeError(detail)
            detail = "atomic installation failed"
            downloaded.replace(destination)
    except Exception:  # noqa: BLE001 - never expose SDK URLs, bodies or credentials
        message = f"Krea originals asset failed ({destination.name}): {detail}"
        print(message, file=sys.stderr)
        raise RuntimeError(message) from None


def download_comfy_krea_originals() -> None:
    """Install only the three approved workflow assets, separately from recommendations."""
    if not (settings.comfy_dir / "main.py").is_file():
        raise FileNotFoundError("Install embedded ComfyUI first: scripts/setup-comfy.ps1")
    total = sum(asset["size"] for asset in KREA_ORIGINAL_ASSETS)
    print(f"Krea originals: three pinned assets, {total / 1024**3:.2f} GiB total.")
    print(
        "Transformer substitution: approved current publisher INT8 alias installed as "
        "krea2_turbo_int8_convrot-b19a4f0be264.safetensors. The exact historical suffix "
        "was not found; this is NOT proof of identical original bytes."
    )
    print(
        "Skin checkpoint provenance: timothy692 mirror; the same SHA256 was verified "
        "in the gemasai mirror. License metadata is None (not a license grant); "
        "review upstream provenance/license before use. The .pth file is verified "
        "as bytes only, never pickle-loaded by this downloader."
    )
    for asset in KREA_ORIGINAL_ASSETS:
        _download_pinned_krea_asset(asset, settings.comfy_dir / "models")


def download_krea_turbo_lora() -> None:
    """Optional official Turbo adapter; it is not required by the Turbo checkpoint."""
    _download_named_file(
        COMFY_KREA_REPO,
        KREA_TURBO_LORA_REMOTE,
        settings.lora_dir / "krea2" / KREA_TURBO_LORA_FILE,
    )


def download_comfy_qwen21() -> None:
    if not (settings.comfy_dir / "main.py").is_file():
        raise FileNotFoundError("Install embedded ComfyUI first: scripts/setup-comfy.ps1")
    target = settings.comfy_dir / "models"
    for filename in QWEN21_FILES:
        destination = target / filename
        if destination.name == QWEN21_TURBO_LORA:
            _download_named_file(QWEN21_VIGGLE_REPO, destination.name, destination)
        else:
            _download_named_file(QWEN21_COMFY_REPO, filename, destination)
    _download_named_file(
        QWEN21_VIGGLE_REPO,
        f"comfyui/{QWEN21_TURBO_NODE}",
        settings.comfy_dir / "custom_nodes" / QWEN21_TURBO_NODE,
    )


def download_comfy_qwen21_r128() -> None:
    if not (settings.comfy_dir / "main.py").is_file():
        raise FileNotFoundError("Install embedded ComfyUI first: scripts/setup-comfy.ps1")
    target = settings.comfy_dir / "models"
    for filename in QWEN21_FILES:
        if Path(filename).name != QWEN21_TURBO_LORA:
            _download_named_file(QWEN21_COMFY_REPO, filename, target / filename)
    _download_civitai_r128(target / "loras" / QWEN21_R128_FILENAME)


def download_comfy_firered() -> None:
    if not (settings.comfy_dir / "main.py").is_file():
        raise FileNotFoundError("Install embedded ComfyUI first: scripts/setup-comfy.ps1")
    target = settings.comfy_dir / "models"
    for filename in FIRERED_FILES:
        destination = target / filename
        if destination.name == FIRERED_ENCODER:
            _download_named_file(FIRERED_ENCODER_REPO, FIRERED_ENCODER_REMOTE, destination)
        else:
            _download_named_file(FIRERED_REPO, destination.name, destination)


def download_comfy_krea() -> None:
    if not (settings.comfy_dir / "main.py").is_file():
        raise FileNotFoundError("Install embedded ComfyUI first: scripts/setup-comfy.ps1")
    target = settings.comfy_dir / "models"
    for filename in COMFY_KREA_FILES:
        path = target / filename
        if path.is_file():
            print(f"Already present: {path}")
            continue
        print(f"Downloading {COMFY_KREA_REPO}/{filename} -> {target}")
        _hf_download(hf_hub_download, repo_id=COMFY_KREA_REPO, filename=filename, local_dir=target)
    adapter = target / "loras" / KREA_EDIT_FILE
    if not adapter.is_file():
        adapter.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {KREA_EDIT_REPO}/{KREA_EDIT_FILE} -> {adapter}")
        _hf_download(hf_hub_download, repo_id=KREA_EDIT_REPO, filename=KREA_EDIT_FILE, local_dir=adapter.parent)
    _download_named_file(
        KREA_FIRST_LORA_REPO,
        KREA_FIRST_LORA_REMOTE,
        target / "loras" / KREA_FIRST_LORA_FILE,
    )



def _download_qwen_aio(local_dir: Path) -> None:
    local_dir.mkdir(parents=True, exist_ok=True)
    target = local_dir / QWEN_AIO_LOCAL_NAME
    if is_safetensors_file(target):
        print(f"Already present: {target}")
        return
    if target.exists():
        target.unlink()
    print(f"Downloading {QWEN_AIO_REPO}/{QWEN_AIO_REMOTE_FILE} -> {target}")
    downloaded = Path(
        _hf_download(hf_hub_download,
            repo_id=QWEN_AIO_REPO,
            filename=QWEN_AIO_REMOTE_FILE,
            local_dir=local_dir,
        )
    )
    if downloaded != target:
        downloaded.replace(target)
    nested = local_dir / "v23"
    if nested.is_dir() and not any(nested.iterdir()):
        nested.rmdir()


def main() -> None:
    _TOKENS.clear()
    parser = argparse.ArgumentParser(description="Download model snapshots into the local project.")
    parser.add_argument(
        "models",
        nargs="*",
        help=f"Model keys ({', '.join(MODEL_SPECS)}); defaults to the RTX 4070 Ti recommended set",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download every registered model, including models unsuitable for 12 GB VRAM",
    )
    parser.add_argument("--restorers", action="store_true", help="Download GFPGAN checkpoint")
    parser.add_argument(
        "--bfs-swap", action="store_true", help="Download only the supported BFS swap LoRAs"
    )
    parser.add_argument(
        "--comfy-krea", action="store_true", help="Download Krea Turbo ComfyUI checkpoints and identity-edit LoRA"
    )
    parser.add_argument(
        "--comfy-krea-originals", action="store_true",
        help="Download only three pinned Krea workflow assets (~13.06 GiB), with the approved current INT8 alias substitution",
    )
    parser.add_argument(
        "--comfy-firered", action="store_true", help="Download only FireRed GGUF Q4_K_M, FP8 vision encoder, VAE and Lightning v1.2"
    )
    parser.add_argument(
        "--comfy-qwen21", action="store_true", help="Download only Qwen Image 2.1 INT8 Comfy weights and Viggle Turbo r256 LoRA/node"
    )
    parser.add_argument(
        "--turbo-lora", action="store_true", help="Download the optional official Krea 2 Turbo LoRA into the app's Krea library"
    )
    args = parser.parse_args()
    if args.all and args.models:
        parser.error("Use model names or --all, not both.")
    unknown = [key for key in args.models if key not in MODEL_SPECS]
    if unknown:
        parser.error(f"Unknown model(s): {', '.join(unknown)}")
    selected = list(MODEL_SPECS) if args.all else (args.models or ([] if args.bfs_swap or args.comfy_krea or args.comfy_krea_originals or args.comfy_firered or args.comfy_qwen21 or args.turbo_lora else RECOMMENDED_MODELS))
    for key in selected:
        spec = MODEL_SPECS[key]
        if spec.loader == "firered_comfy":
            download_comfy_firered()
            continue
        if spec.loader == "qwen21_comfy":
            if key == QWEN21_R128_KEY:
                download_comfy_qwen21_r128()
            else:
                download_comfy_qwen21()
            continue
        if spec.loader == "krea2":
            download_comfy_krea()
            download_krea_turbo_lora()
            continue
        if spec.loader == "qwen_aio":
            _download_qwen_aio(spec.local_path)
            continue
        print(f"Downloading {spec.repo_id} -> {spec.local_path}")
        _hf_download(snapshot_download,
            repo_id=spec.repo_id,
            local_dir=spec.local_path,
            local_dir_use_symlinks=False,
        )
    if args.bfs_swap:
        for (model_key, _kind), profile in SWAP_PROFILES.items():
            if model_key in {"qwen-2.1-turbo", QWEN21_R128_KEY}:
                # This separately released file is installed by the user; it is
                # not part of the legacy BFS repository used below.
                continue
            target = settings.lora_dir / profile.family / profile.filename
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.is_file():
                print(f"Already present: {target}")
                continue
            print(f"Downloading {BFS_REPO}/{profile.filename} -> {target}")
            _hf_download(hf_hub_download, repo_id=BFS_REPO, filename=profile.filename, local_dir=target.parent)
    if args.comfy_krea and "krea-2-turbo" not in selected:
        download_comfy_krea()
        download_krea_turbo_lora()
    if args.comfy_krea_originals:
        download_comfy_krea_originals()
    if args.comfy_firered and "firered-1.1" not in selected:
        download_comfy_firered()
    if args.comfy_qwen21 and "qwen-2.1-turbo" not in selected:
        download_comfy_qwen21()
    if args.turbo_lora and "krea-2-turbo" not in selected and not args.comfy_krea:
        download_krea_turbo_lora()
    if args.restorers:
        target = settings.model_dir / "restorers" / "GFPGANv1.4.pth"
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            print(f"Downloading GFPGAN -> {target}")
            urlretrieve(GFPGAN_URL, target)
    print("Downloads complete. Set HF_HUB_OFFLINE=1 for strictly offline launches.")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, KeyboardInterrupt):
        # Even traceback source/context can reveal credential-bearing request URLs.
        print("Download failed or cancelled. " + " ".join(AUTH_HELP.values()), file=sys.stderr)
        sys.exit(1)
