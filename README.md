# Local Photo Edit Studio

A private Gradio studio for local text-to-image creation, instruction-based image editing, and image composition. This version is optimized for Windows 11, Python 3.11, an RTX 4070 Ti with 12 GB VRAM, and 64 GB system RAM. It uses aggressive CPU offload so large models can operate within a much smaller GPU-memory budget.

## Low-VRAM profile

- Qwen Rapid AIO and FLUX.2 Klein 4B are the recommended model choices.
- Krea 2 Turbo is available for text-to-image generation at 8 steps with sequential CPU offload.
- All Krea 2 modes use the project-managed ComfyUI backend and require `vendor/ComfyUI/models/loras/Krea2_ALWAYS_LOAD_FIRST.safetensors` as the first LoRA. Its nonzero weight is adjustable in the UI.
- FLUX.2 Klein 4B replaces its bundled Qwen3 text encoder with `models/qwen3-4b-alb-q4_0.gguf` when loaded. The current file spelling `qwen3-4b-abl-q4_0.gguf` is also recognized.
- Sequential CPU offload, VAE slicing/tiling, attention slicing, and lazy one-model loading are enabled.
- Output defaults are capped at a 1024-pixel longest side and one megapixel; **Combine images** and **Create from text** may optionally use a 2K canvas at 1:1, 9:16, or 16:9.
- Batch count and Gradio concurrency are fixed at one.
- Qwen 2511 remains an advanced, slower choice. FireRed 1.1 uses the project-managed ComfyUI GGUF Q4_K_M transformer with an automatic Lightning v1.2 8-step LoRA and CPU offload.
- Qwen Image 2.1 + Viggle Turbo uses targeted INT8 ComfyUI weights and the mandatory unmerged rank-256 LoRA with a fixed six-step schedule for text, edit, combine and experimental Head/Body BFS swaps. The Qwen Research License permits non-commercial research/evaluation only.
- Unstacked `qwen-2.1-turbo-official` uses full BF16 Diffusers and eight saved-sigma steps, including experimental Head/Body with one mandatory pinned BFS adapter, CFG/True CFG 1 and no optional stack or DLSS. `official-extract` still rejects swaps. Live BFS inference, quality and VRAM fit are unconfirmed; see [official setup](docs/SETUP.md#official-qwen-21-turbo-bf16-opt-in).
- Models, inputs, and outputs stay local; Gradio analytics and public sharing are disabled by default.

## Windows quick start

1. Install a current NVIDIA driver and confirm `nvidia-smi` works.
2. In PowerShell, run `.\scripts\setup.ps1`.
3. Activate the environment with `.\.venv\Scripts\Activate.ps1`.
4. Download models with `.\scripts\download-models.ps1 -Preset recommended`.
   - Rapid AIO only: `.\scripts\download-models.ps1 -Preset qwen-aio`
   - Qwen Image 2.1: run `.\scripts\setup-comfy.ps1` first, then `.\scripts\download-models.ps1 -Preset qwen-2.1` and launch with `.\scripts\run.ps1 -ComfyUI`.
   - Official Qwen 2.1 BF16: follow [dependency requirements](docs/SETUP.md#official-qwen-21-turbo-bf16-opt-in), download `-Preset qwen-2.1-official`, then separately `-Preset qwen-2.1-bfs` for swaps; no ComfyUI required.
   - Klein 4B only: `.\scripts\download-models.ps1 -Preset flux-4b`
   - Krea 2 Turbo only: `.\scripts\download-models.ps1 -Preset krea-2`
   - Krea QuadView/Dynamic sheets: `.\scripts\download-models.ps1 -Preset krea-character-sheets` after ComfyUI setup; see [character-sheet prerequisites](docs/KREA_CHARACTER_SHEETS.md).
5. Start with `.\scripts\run.ps1`.
6. Open `http://127.0.0.1:7860`.

See `docs/WINDOWS.md` for driver, PowerShell, storage, authentication, and configuration details.

Setup can also install and download in one pass with `.\scripts\setup.ps1 -Download recommended`.

## Features

- Sixth notebook workflow: **Character Sheet Creator**, with user-chosen full-BF16 Qwen 2.1 assets, Simple/Production layouts, Static/Auto prompts and native 1/3.4/6 binary-MP canvases. No mandatory Turbo. See [usage, pinned assets and rights](docs/QWEN_CHARACTER_SHEET.md); performance/parity are unverified.
- Mockup-inspired Beta studio UI: a 70% left creative canvas with workflow icons beside elapsed-time progress, a card for each workflow description, and a Run Details icon opening a popup; the 30% right rail starts with LoRA/restoration/other options. Prompt-toolbar popovers, circular Generate, the one-output GPU cap and backend routes remain unchanged. See `docs/USAGE.md`.
- Edit a source image or combine up to three reference images with 1K/2K and square/portrait/landscape output options.
- Create new images from text using Krea 2 Turbo, FLUX.2 Klein 4B or Qwen Image 2.1 Turbo at 1K/2K and square/portrait/landscape aspect ratios.
- Edit a Krea 2 source image using one optional reference through project-managed ComfyUI/Krea2Edit (opt-in setup in the same `.venv`). See `docs/KREA_REFERENCES.md`.
- Create a fixed 1536×1024 **QuadView — Krea 2** or experimental **DynamicCharacterSheet** from one aspect-preserved source under Krea reference edit (10 steps / CFG 1). See [character-sheet workflows and limits](docs/KREA_CHARACTER_SHEETS.md) and [Krea asset inventory](docs/KREA_MODELS.md); memory and visual results are unbenchmarked.
- Swap a head using Qwen 2511, FLUX.2 Klein 4B, Krea 2, or Qwen Image 2.1 Turbo; swap a person/body with Krea 2 or Qwen 2.1 Turbo. Both ComfyUI profiles and unstacked official BF16 use canonical `bfs_head_v1.1_qwen_2.1.safetensors` / `bfs_body_swap_v1.0_qwen_2.1.safetensors` in `models/loras/qwen21/`. `--qwen21-bfs` / preset `qwen-2.1-bfs` installs only these two pinned files; `--bfs-swap` now includes them too. Verified migration copies old aliases and keeps originals: same bytes, not new weights; Body is published v1.0. Official BFS requires two ordered source/reference images and sole BFS weight 0.1–1.5. See `docs/SWAP.md` for pins and limitations.
- Identity-preservation prompt mode, seeds, CFG, true CFG, steps, strength, and masks.
- Local, family-scoped LoRA library with up to five selected adapters.
- Optional local GFPGAN post-processing.
- Explicit model unload and CUDA cache release.
- Optional project-managed Krea reference/swap backend: `.\scripts\setup-comfy.ps1`, `python scripts\download_models.py --comfy-krea`, then `.\scripts\run.ps1 -ComfyUI`.
- Local PNG and metadata output; prompts and source images are not persisted in metadata.

## Practical limitations

- Qwen sheets bypass ordinary output caps: 1248×832, 2304×1536 or 3072×2048. Mandatory BF16 files alone occupy 32.44 GB / 30.21 GiB; Auto adds 5.59 GB of PE weights. The 12 GB threshold is not a fit guarantee. Qwen non-commercial research restrictions and separate workflow/PE rights review apply.
- The RTX 4070 Ti profile trades speed for memory. CPU offload can make generation take minutes.
- Edit and swap workflows default to 1K; the two ComfyUI Qwen 2.1 profiles and unstacked official BF16 support native ×2 with a separate 2048-side/4-megapixel budget, including swaps. **Combine images** and **Create from text** offer 2K as an advanced option that may run out of VRAM on a 12 GB GPU. 4K is not available in the UI.
- Generic image models cannot guarantee identity. Use a clean face reference and narrowly scoped prompts.
- The mask is a post-generation composite, not native latent inpainting.
- LoRAs must match the selected Qwen, FLUX.2 Klein, or Krea architecture. FireRed Lightning v1.2 is applied automatically; extra FireRed LoRAs are not supported.
- Krea 2 is gated and uses the Krea 2 Community License. Accept its Hugging Face terms before downloading and follow its required filtering or review obligations.
- Upstream model licenses and gated repository requirements still apply.

## Linux

The existing `scripts/setup.sh` and `scripts/run.sh` remain available. The same low-VRAM application defaults apply unless overridden in `.env`.

Additional documentation: `docs/SETUP.md`, `docs/MODELS.md`, `docs/ARCHITECTURE.md`, `docs/USAGE.md`, `docs/KREA_REFERENCES.md`, `docs/SWAP.md`, `docs/TROUBLESHOOTING.md`, and `docs/WINDOWS.md`.
