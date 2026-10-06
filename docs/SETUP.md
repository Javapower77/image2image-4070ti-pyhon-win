# Setup

## Windows 11 (primary target)

Use Python 3.11, a current NVIDIA driver, and PowerShell. Run `.\scripts\setup.ps1 -Download recommended`, then launch `.\scripts\run.ps1`. To keep installation and model downloads separate, run `.\scripts\setup.ps1` followed by `.\scripts\download-models.ps1 -Preset recommended`. Full instructions are in `docs/WINDOWS.md`.

The setup script installs the CUDA 12.8 PyTorch wheel. Do not install a separate CUDA Toolkit solely for this project. Verify the result with `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name())"` from the activated environment.

## Linux

Run `bash scripts/setup.sh`, activate with `source .venv/bin/activate`, download models, and start with `bash scripts/run.sh`. Linux now uses the same 12 GB-safe defaults. Override them in `.env` only when the GPU has more memory.

## Storage and authentication

Snapshots live in `models/repos`, Hugging Face cache data in `models/huggingface`, LoRAs in `models/loras`, restorer weights in `models/restorers`, and results in `outputs`.

For gated repositories, accept the repository terms and run `hf auth login`. Never commit tokens. An environment-provided `HF_TOKEN` is also supported.

## Recommended downloads

- `.\scripts\download-models.ps1 -Preset qwen-aio` for Rapid AIO.
- `.\scripts\setup-comfy.ps1` then `.\scripts\download-models.ps1 -Preset qwen-2.1` for Qwen Image 2.1 INT8 with Viggle Turbo r256; launch with `.\scripts\run.ps1 -ComfyUI` (Qwen Research License; non-commercial).
- `.\scripts\download-models.ps1 -Preset flux-4b` for Klein 4B.
- `.\scripts\download-models.ps1 -Preset krea-2` for gated Krea 2 Turbo text-to-image.
- `.\scripts\setup-comfy.ps1` then `.\scripts\download-models.ps1 -Preset firered` for FireRed GGUF Q4_K_M + automatic Lightning 8-step editing; launch with `.\scripts\run.ps1 -ComfyUI`.
- Add `--restorers` to download GFPGAN weights after installing the restoration extra.

Downloading every registered model is not recommended for the 12 GB target. FireRed requires CPU offload because its GGUF checkpoint alone exceeds 12 GB.

## Network and privacy

The default bind address is `127.0.0.1`. If binding to `0.0.0.0`, configure `PHOTO_EDIT_AUTH_USER` and `PHOTO_EDIT_AUTH_PASSWORD`, restrict the firewall, and use TLS at a reverse proxy. Do not enable Gradio share links for private images.

After downloads complete, set `HF_HUB_OFFLINE=1` for strict offline launches.

## Optional BFS head/body swap

Download compatible BFS LoRAs with `python scripts/download_models.py --bfs-swap`. Krea Head/Body swap uses the optional **project-managed ComfyUI** backend and built-in Krea2Edit graph. Install it with `.\scripts\setup-comfy.ps1`, download the Krea ComfyUI checkpoints with `python scripts/download_models.py --comfy-krea`, and launch with `.\scripts\run.ps1 -ComfyUI`. See `docs/SWAP.md` and `docs/KREA_REFERENCES.md`.

## Optional Krea reference editing

The Krea 2 Diffusers model cannot accept reference images. For a source image plus optional second reference use **Krea reference edit** and the optional project-managed ComfyUI Krea2Edit backend described in `docs/KREA_REFERENCES.md`.

## Optional Krea All2Real assets

After `.\scripts\setup-comfy.ps1`, explicitly run
`.\scripts\download-models.ps1 -Preset krea-originals`, or, in the activated
environment, `python scripts/download_models.py --comfy-krea-originals`.
Used alone, this downloads only three pinned assets (~13.06 GiB): the approved
current Comfy-Org INT8 Turbo alias, exact spacepxl Wan upscale VAE, and exact skin
filename from the timothy692 mirror (same SHA256 as gemasai). MoreReal, the shared
encoder and mandatory first adapter are **not included**; supply them separately.

**Changed decision:** the historical suffixed transformer was not found; the user
approved the current checkpoint saved as `krea2_turbo_int8_convrot-b19a4f0be264.safetensors`.
This is not proof of identical upstream-original bytes or visual parity. Runtime
still never downloads or silently substitutes/falls back. Staged files are checked
for size/SHA256 and Safetensors headers; existing mismatches cause an error and
remain unchanged. The downloader never pickle-loads the skin `.pth`; its license
metadata is unavailable, so review source terms/provenance before use. Allow extra
staging/cache space. See [All2Real asset pins and limits](KREA_ALL2REAL.md).
