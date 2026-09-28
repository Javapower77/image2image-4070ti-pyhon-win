# Changelog

All notable changes are documented here.

## [0.9.0] - 2026-09-28

### Added

- Qwen Image 2.1 **Head** and **Body** swaps now automatically apply the locally supplied `Qwen21-BFS_Head_v1.1.safetensors` or `Qwen21-BFS_Body_v1.1.safetensors` after mandatory Viggle Turbo and before optional adapters.
- Qwen Image 2.1 now supports up to five optional family-isolated `.safetensors` LoRAs at individual UI weights, chained after the mandatory unmerged Viggle Turbo LoRA for text, edit, combine and swap workflows.
- Qwen Image 2.1 + Viggle Turbo r256 as a six-step, INT8/low-VRAM ComfyUI option for Edit source, Combine images, Create from text and experimental two-image Head/Body instruction swaps; targeted ComfyUI assets and Windows download preset (`qwen-2.1`). Requires non-commercial Qwen Research License compliance.
- FireRed 1.1 ComfyUI GGUF Q4_K_M plus automatic Lightning 8-step v1.2 support with targeted downloads and a separate LoRA library.
- **Combine images** now offers 1K/2K output and 1:1, 9:16, or 16:9 aspect ratios, with a live size preview, separate combine-only memory budget, and saved aspect metadata.
- **Create from text** now shares the 1K/2K and 1:1/9:16/16:9 canvas controls, live preview, output budget, and saved canvas metadata; its selected dimensions reach the Krea ComfyUI graph.

### Changed

- Replaced the native model dropdown with per-row brand icons from the mockup Icons-MODELS set, hid the leftover Gradio combobox in the model popover, and matched prompt-toolbar hover to the top workflow toolbar gray (`#f0efec`). Inference callbacks are unchanged.
- Split the prompt composer into a white prompt-and-toolbar card and a separate white model-description card, replaced the right-arrow submit control with the mockup up-arrow, and removed the gray toolbar band. Inference callbacks are unchanged.
- Placed the workflow toolbar and progress card together atop the left canvas, wrapped each workflow description in its own card, and moved LoRAs/restoration/other options to the top of the right rail. The far-right progress icon now opens Run Details as a dismissible popup without changing generation callbacks.
- Matched the mockup's left/right proportions with a fixed 70% creative canvas and 30% options rail; normalized card widths and replaced approximate workflow/model/toolbar graphics with SVG path geometry from `docs/mockup-ui/Icons.html`.
- Reworked the mockup UI into a real desktop left-rail/right-canvas grid with compact icon-only workflow and generation controls, SVG model indicators, and responsive single-column mobile layout. Kept the same Gradio model and inference handlers.
- Refined the mockup-based UI with anchored, shadowed prompt-tool popovers, synchronized model-family SVG icons, and Escape/outside-click dismissal using a browser-side visual layer; the underlying Gradio controls and inference callbacks remain unchanged.
- Rebranded the Gradio UI into a single-column Local Photo Edit Studio Beta layout following `docs/mockup-ui/UI-MOCKUP.md`. The five workflows use icon buttons; model, sizing, output variants, advanced model controls, swap type and BFS weight move into the prompt toolbar with mode-aware panels. Run details, image results, uploads and three shared option accordions are regrouped without changing model inference or the one-output GPU cap.
- Added visible progress with overall completion and elapsed time, model-family icons, an active-workflow description, and a Beta release badge driven by the changelog. Krea reference edit keeps its dedicated model path without a model picker.
- Removed the unsupported FLUX.2 Klein 9B entry from application model selection and downloads.
- All Krea 2 modes now require `Krea2_ALWAYS_LOAD_FIRST.safetensors` as the first ComfyUI LoRA, with a dedicated nonzero weight control in the UI. Krea text-to-image now runs through ComfyUI to guarantee this ordering.

### Fixed

- **Krea reference edit** now hides the unrelated three-image **Combine images** upload section and the duplicate Krea text-to-image model description; its dedicated two-image controls and workflow label remain visible.
- **Create from text** now renders with the text-capable model selected in **02 / Generation settings**: Krea 2 Turbo through ComfyUI or FLUX.2 Klein 4B through Diffusers. The text-mode dropdown excludes image-edit-only and unsupported models, and FLUX text generation no longer expects an uploaded image.
- Krea reference edit and Krea head/body swap now chain selected Krea LoRAs into the ComfyUI model path at their UI weights, after the required identity-edit or BFS LoRA. Previously the extra UI slots had no effect.
- Reject missing, duplicate, wrong-family or unsupported optional Krea adapter files instead of silently ignoring them.
- Selecting the required identity-edit LoRA in Krea reference edit now adjusts its existing graph-node weight rather than raising a duplicate-base error; the BFS swap weight remains controlled by its dedicated slider.

## [0.8.0] - 2026-09-23

### Project-managed Krea backend

- Added opt-in project-local ComfyUI and Krea2Edit setup in the Python 3.11 `.venv`.
- Added on-demand loopback subprocess management with low-VRAM startup, local logging and owned-process shutdown.
- Bundled a validated 1–2 image Krea2Edit API graph usable for reference and BFS head/body swaps without manual workflow export.
- Added an explicit `--comfy-krea` download option for the four required ComfyUI-format weights; no automatic multi-gigabyte downloads.

## [0.7.1] - 2026-09-23

### Krea reference editing

- Added a dedicated **Krea reference edit** UI workflow with source image and optional second reference.
- Added a separate ComfyUI Krea2Edit API graph adapter that verifies second-image grounding and latent wiring before inference.
- Retained Krea text-to-image through Diffusers and the existing BFS head/body swap workflows.
- Restricted the ComfyUI bridge to local loopback endpoints and documented required external setup.

## [0.7.0] - 2026-09-23

### Added

- Separate two-image **Swap head / body** section preserving the other generation workflows and controls.
- Qwen 2511 and FLUX.2 Klein 4B BFS head swap adapters; Krea 2 BFS head/body swap through a local ComfyUI editing backend and validated API-format workflows.
- Explicit `--bfs-swap` downloader for four matching BFS LoRAs; no implicit download of the full upstream collection.
- `docs/SWAP.md` with model compatibility, required ComfyUI setup, consent, and quality limitations.

## [0.6.1] - 2026-09-17

### FLUX.2 Klein 4B custom encoder

- Added a FLUX.2 Klein 4B-only Qwen3 GGUF text encoder loaded from `models/qwen3-4b-alb-q4_0.gguf`.
- Added compatibility for the existing `qwen3-4b-abl-q4_0.gguf` filename and an environment override.
- Added the `gguf` dependency and documented CUDA dequantization/system-memory behavior.

## [0.6.0] - 2026-09-17

### Krea 2 text-to-image

- Added gated `krea/Krea-2-Turbo` through Diffusers `Krea2Pipeline`.
- Added a **Create from text** workflow with 8-step, CFG-0, 1K low-VRAM defaults.
- Added Krea 2-specific LoRA storage and multi-adapter loading through Diffusers/PEFT.
- Added a `krea-2` Windows download preset and license/access documentation.

## [0.5.0] - 2026-09-15

### Windows and low-VRAM refactor

- Rebuilt the missing model package with local Diffusers adapters, registry, Rapid AIO loader, and CUDA memory management.
- Added Windows PowerShell setup and launch scripts for Python 3.11 and CUDA PyTorch.
- Added a Windows model downloader with recommended, Qwen AIO, Klein 4B, and explicit all-model presets.
- Added sequential CPU offload, VAE tiling/slicing, attention slicing, lazy CUDA loading, and explicit cache release.
- Enforced one-output, one-megapixel/1024-side generation defaults for an RTX 4070 Ti 12 GB.
- Made Rapid AIO and Klein 4B the recommended UI choices and marked Klein 9B unsupported on 12 GB.
- Updated all setup, architecture, model, usage, and troubleshooting documentation.

## [0.4.3] - 2026-09-14

### Fixed

- Changing a LoRA weight no longer crashes with `Setting requires_grad=True on inference tensor`. Adapter weights are applied in inference mode, and generation uses `torch.no_grad()` instead of `torch.inference_mode()`.

## [0.4.2] - 2026-09-14

### Fixed

- Combine-images size preview no longer crashes with `needed: 7, got: 6` when some image slots are empty. Hidden Gradio widgets are not used as event inputs.

## [0.4.1] - 2026-09-14

### Changed

- Combine-images output size is now a 1K / 2K / 4K canvas (1024, 2048, or 4096 square). Edit source still uses ×1 / ×2 / ×3 from the source aspect ratio.

## [0.4.0] - 2026-09-14

### Added

- Output size now follows the first uploaded image's aspect ratio. Choose ×1, ×2, or ×3 instead of independent width/height sliders (clamped to 512–2048 and aligned to 64 px).
- Combine-images workflow: upload up to three references and a prompt to generate one new picture. Image 1 still sets the aspect ratio.

## [0.3.2] - 2026-09-12

### Fixed

- Rapid AIO load no longer builds the transformer on the meta device, which caused `Cannot copy out of meta tensor; no data!` when moving RoPE caches or enabling CPU offload.

## [0.3.1] - 2026-09-12

### Fixed

- CUDA out-of-memory on Qwen 2511 / Rapid AIO: model CPU offload, VAE tiling, sequential batch images, allocator `expandable_segments`, and Rapid AIO no longer loads two transformers.

## [0.3.0] - 2026-09-12

### Added

- Qwen Image Edit 2511 Rapid AIO as a selectable model. It loads the distilled ComfyUI transformer from `models/repos/Qwen--Qwen-Image-Edit-2511-AIO` onto the official 2511 pipeline.

## [0.2.0] - 2026-09-12

### Added

- Local LoRA library with upload, up to five selected adapters per run, and per-LoRA weights.
- Qwen/FireRed and FLUX.2 Klein family LoRA folders under `models/loras/`.

## [0.1.1] - 2026-09-12

### Changed

- Removed extra application-level content filters and the permission checkbox.
- Left model-native and license-required safeguards enabled.

## [0.1.0] - 2026-09-12

### Added

- Local Gradio photo editing studio with a modern two-panel interface.
- Lazy, one-at-a-time adapters for Qwen Image Edit 2511, FireRed Image Edit 1.1, and FLUX.2 [klein] 4B/9B.
- Multi-reference inputs, identity-preservation prompting, seed, dimensions, steps, CFG, true CFG, strength, batching, and optional mask compositing.
- Optional lazy-loaded GFPGAN post-processing.
- Project-local snapshot downloader and strictly local runtime loading.
- Python 3.11/H100 setup and run scripts.
- Privacy-conscious metadata without prompt or input persistence.
- Setup, model, architecture, usage, license/safety, and troubleshooting documentation.
