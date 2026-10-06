# Windows 11 and RTX 4070 Ti setup

## Target machine

- Windows 11, 64-bit
- Intel Core i9 and 64 GB system RAM
- NVIDIA RTX 4070 Ti with 12 GB VRAM
- Python 3.11 and Git
- Current NVIDIA Studio or Game Ready driver
- At least 100 GB free SSD space for the recommended models; substantially more for all models

A separate CUDA Toolkit is not required. PyTorch supplies its CUDA runtime; the NVIDIA display driver must be recent enough for the selected wheel.

## Install

Open PowerShell in the repository root and run `.\scripts\setup.ps1`. The script creates `.venv`, installs the CUDA 12.8 PyTorch wheel, installs the project, and verifies that CUDA is visible. It does not download multi-gigabyte models unless requested. Use `.\scripts\setup.ps1 -Download recommended` for a combined setup and model download. To include the optional GFPGAN packages, add `-Restore`.

If script execution is blocked, use `Set-ExecutionPolicy -Scope Process Bypass` in that PowerShell window and rerun the setup script. This temporary setting does not change the machine-wide policy.

## Download models

Use the Windows downloader from the repository root:

- Recommended set: `.\scripts\download-models.ps1 -Preset recommended`
- Qwen Rapid AIO: `.\scripts\download-models.ps1 -Preset qwen-aio`
- Qwen Image 2.1 + Viggle Turbo r256: first `.\scripts\setup-comfy.ps1`, then `.\scripts\download-models.ps1 -Preset qwen-2.1`; start with `.\scripts\run.ps1 -ComfyUI` (non-commercial research license).
- FLUX.2 Klein 4B: `.\scripts\download-models.ps1 -Preset flux-4b`
- Krea 2 Turbo: `.\scripts\download-models.ps1 -Preset krea-2`
- Optional Krea All2Real three-asset set: after `.\scripts\setup-comfy.ps1`, run `.\scripts\download-models.ps1 -Preset krea-originals` (~13.06 GiB plus staging/cache space; not a complete All2Real setup).
- FireRed 1.1 GGUF Q4_K_M + Lightning v1.2: first `.\scripts\setup-comfy.ps1`, then `.\scripts\download-models.ps1 -Preset firered`; start with `.\scripts\run.ps1 -ComfyUI`.
- Every registered model: `.\scripts\download-models.ps1 -Preset all` (not recommended for 12 GB)

Rapid AIO requires the official Qwen 2511 components as well as its distilled transformer. Model downloads are large and may require accepting upstream terms and authenticating with `hf auth login`.

Krea 2 Turbo is gated. Visit `https://huggingface.co/krea/Krea-2-Turbo`, accept the Krea 2 Community License, run `hf auth login`, and then use the Krea 2 preset. It is a 12B model and relies on 64 GB system RAM plus sequential CPU offload on the RTX 4070 Ti.

For **Krea reference edit**, install the optional project-managed ComfyUI in the existing venv with `.\scripts\setup-comfy.ps1`, download Krea ComfyUI checkpoints with `python scripts/download_models.py --comfy-krea`, then launch `.\scripts\run.ps1 -ComfyUI`. The Krea Diffusers snapshot is text-only and is not a ComfyUI checkpoint. See `docs/KREA_REFERENCES.md` for model and privacy details.

FireRed's quantized transformer alone exceeds 12 GB VRAM: it relies on ComfyUI's low-VRAM CPU offload, not full GPU residency. See `docs/MODELS.md`.

### Optional All2Real download and changed asset decision

Python equivalent: `.\.venv\Scripts\python.exe scripts\download_models.py --comfy-krea-originals`.
Used alone, the option downloads only the approved INT8 alias, exact spacepxl Wan
upscale VAE and exact skin filename from the timothy692 mirror (same SHA256 as
gemasai). It does not obtain MoreReal, the shared encoder or mandatory first
adapter, and is not automatically included in recommended/all-model downloads.

The exact historical suffixed transformer was not found. The user explicitly
approved current Comfy-Org/Krea-2 INT8 Turbo saved under
`krea2_turbo_int8_convrot-b19a4f0be264.safetensors`; the alias is **not proof of
identical upstream-original bytes**. Runtime never downloads or silently
substitutes/falls back. Staged downloads are size/SHA256 verified and Safetensors
header validated before installation. Existing mismatches are retained with an
error; move/remove them explicitly before rerunning to replace them. The
downloader never pickle-loads the skin `.pth`. Its license metadata is unavailable,
not permission to use it; review source provenance and terms. See
[All2Real pinned sources, hashes and remaining prerequisites](KREA_ALL2REAL.md).
Neither the 12 GB GPU nor 64 GB RAM guarantees this workflow fits memory.

## Run

Start with `.\scripts\run.ps1`, then open `http://127.0.0.1:7860`.

The Windows profile defaults to:

- sequential CPU offload;
- VAE slicing and tiling;
- attention slicing when the pipeline supports it;
- one queued generation at a time;
- one output per request;
- a 1024-pixel longest side and one-megapixel output budget;
- lazy CUDA module loading and the expandable CUDA allocator.

FLUX.2 Klein 4B additionally loads `models/qwen3-4b-alb-q4_0.gguf` as its Qwen3 text encoder. The existing `qwen3-4b-abl-q4_0.gguf` spelling is accepted. Override the location with `PHOTO_EDIT_FLUX_KLEIN_4B_TEXT_ENCODER` in `.env`. Other models never use this file.

CPU offload uses system RAM and PCIe transfers, so it is intentionally slower than H100 execution. First load can take several minutes. An NVMe SSD is strongly recommended.

## Configuration

Copy `.env.example` to `.env` to override settings. Safe RTX 4070 Ti values are:

- `PHOTO_EDIT_MEMORY_MODE=sequential`
- `PHOTO_EDIT_MAX_OUTPUT_SIDE=1024`
- `PHOTO_EDIT_MAX_OUTPUT_PIXELS=1048576`
- `PHOTO_EDIT_COMBINE_MAX_OUTPUT_SIDE=2048` and `PHOTO_EDIT_COMBINE_MAX_OUTPUT_PIXELS=4194304` (Combine images and Create from text only; 2K may OOM on 12 GB)
- `PHOTO_EDIT_MAX_BATCH_COUNT=1`
- `PHOTO_EDIT_MAX_CONCURRENCY=1`

Raising these limits can exhaust VRAM. Change one value at a time and restart the app after allocator changes.
