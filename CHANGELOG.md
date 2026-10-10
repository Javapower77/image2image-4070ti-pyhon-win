# Changelog

<!-- markdownlint-configure-file { "MD024": { "siblings_only": true } } -->

All notable changes are grouped by day, newest first. Release versions are retained
within each day; new entries belong under the date they were made, grouped as
Added, Changed, Fixed, or Removed as appropriate.

## 2026-10-10

### Fixed

- Added Qwen 2511 mixed-format LoRA regression tests using tiny real-Torch CPU modules: actual-rank omitted-alpha scale 1, explicit alpha/rank, signed matrix versus direct bias/weight scaling, all-target validation before mutation, overlapping deltas, exact restoration on switches/deselection and signature reuse without compounding. Covers fake pipeline hook/memory ordering and real Accelerate meta/offload hooks, with legitimate high-level mock updates. Documents Head-only BFS, measured 846 pairs/846 biases/241 direct weights and cache-pinned header provenance without silent drops or runtime pin claims. Tests/docs only; no production edits, full-weight reads or inference.
- Corrected extracted-Qwen regression fixtures and mixed-format documentation: 200 source LoRA pairs plus all 65 additive normalization deltas, converted state-dictionary loading before offload, alpha/rank scaling at adapter weight 1, and gate-first fused SwiGLU splitting without discarded updates. Added independent tiny real-Torch CPU tests for preservation, input immutability, fail-before-mutation validation and real adapter-hook ordering; updated stale high-level mocks without relaxing assertions. Records user-reported success of the repaired **unstacked** official setup separately; no extracted-stack live success is claimed. Tests/docs only, no production edits or real inference.
- Corrected official Qwen Turbo installation guidance: force-reinstall the pinned Diffusers revision when identical development-version labels cause `--upgrade` to retain an older commit. Verified pipeline import, saved-sigma support and KV-cache signature after repairing the local environment.

### Added

- Tests/docs-only official BF16 BFS regression coverage: tiny real CPU normalized A/B tensors, unit intrinsic scaling and gate-first gate/proj splitting, strict mandatory-only Head/Body requests, canonical runtime/downloader pins, ordered images, loaded-pipeline adapter/weight changes and return to plain Edit/Combine/Text, hook/offload ordering, UI locks and JSON/PNG provenance. Corrected stale official-swap assertions while retaining official-extract exclusions and all other profiles. Full pytest: **1383 passed, 1 failed**; the non-xfail preallocation regression exposes valid-hash malformed shapes reaching checkpoint allocation before rejection. Repository Ruff: **one existing B023** at `scripts/download_models.py:684` (late-bound `pin` callback); test import lint issues corrected. `git diff --check` passed. No implementation edits, live weights, downloads or inference.
- Offline regression coverage and MODELS/SETUP/USAGE documentation for the dedicated `qwen-2.1-turbo-official-extract` subclass: explicitly user-chosen full official Turbo snapshot + mandatory extracted LoRA, not original base. Covers preallocation size/hash/header validation, pipeline LoRA activation at weight 1 and retention, saved sigmas without override, unsupported requests, shared UI/sizing/metadata, unchanged fallbacks and swap exclusion, exact Civitai version 3394831/file 3284648, credential-safe canonical redirects and atomic corruption failures. Documents `qwen-2.1-official-extract` snapshot-then-LoRA preset. This stack on an already-trained/distilled checkpoint is experimental and NOT recommended; no version-specific Civitai recommendations or proven optimal weight/schedule. Tests/docs only; no production changes, live downloads or inference.
- Initial documentation for the separate `qwen-2.1-turbo-official` full-BF16, eight-step Diffusers profile (`Qwen/Qwen-Image-2.1-Turbo`), exact PR #14950 API/config requirements, saved publisher sigmas and causal KV cache, existing full-snapshot `qwen-2.1-official` download preset, and explicit dependency-upgrade instructions. Initially described swaps/optional LoRAs/DLSS as unsupported and schedule metadata as absent; the **Changed** entries below supersede the swap/metadata claims while optional LoRAs/DLSS remain unsupported. Initial source-only inspection found a venv lacking `QwenImage21Pipeline`, subsequently repaired as recorded above. No production/test edits, installs, downloads or inference in that initial documentation update.

### Changed

- Documentation now reflects experimental Head/Body BFS on **unstacked `qwen-2.1-turbo-official` BF16**, not `official-extract`: exactly two body/source-then-reference images, eight saved-sigma steps, CFG/True CFG 1, sole mandatory BFS at finite weight 0.1–1.5, no optional stack or DLSS. Records canonical Head `bfs_head_v1.1_qwen_2.1.safetensors` (260096144 bytes, Civitai 3356102/file 3248321, SHA256 `d1d748d5601077f3b6d05766f6823510e901970d916afa404a97e85dc92fa88e`) and Body `bfs_body_swap_v1.0_qwen_2.1.safetensors` (published **1.0**, 209753576 bytes, Civitai 3363725/file 3251548, SHA256 `7dc0a53aba4dbc70c204f498936a619d226559da11011f748c389189af46b664`). These are the same verified bytes as legacy aliases, not new weights. Documents BFS-only `--qwen21-bfs` / PowerShell `qwen-2.1-bfs`, broader `--bfs-swap` inclusion, verified migration copies retaining originals, preallocation byte checks versus **post-allocation** target-shape validation before installation, CPU gate/proj normalization at intrinsic scale 1 with separate request weight, and saved sigma/BFS provenance metadata. Supersedes earlier absolute no-official-swaps claims while retaining extracted/other-profile restrictions. Documentation only; no implementation/test edits, downloads, inference, quality or VRAM verification.
- Official-profile reproducibility documentation now reflects recorded saved `sample_sigmas`, KV-cache flag and actual output dimensions, plus extracted-adapter provenance; actual scheduler timesteps, explicit precision and snapshot revision remain unrecorded. This supersedes the earlier metadata limitation above.
- Earlier regression tests exposed an engine-boundary bug where shared DLSS validation replaced enabled/disabled `request.dlss` dictionaries before rejecting either official profile. Current source inspection shows official request validation now precedes shared DLSS normalization, superseding the unfixed-bug claim. No tests were run or production code changed in this documentation update.

## 2026-10-07

### Fixed

- Qwen character-sheet archive downloads now permit Civitai's verified storage-delivery redirect. API credentials remain stripped across origins; pinned archive validation is unchanged. Previously this redirect was rejected and surfaced as a generic token/transport error.

### Added

- Qwen Character Sheet Creator offline regression tests and `docs/QWEN_CHARACTER_SHEET.md`: two-profile scoped prompt parsing, full-BF16 graph, DeGrid output 0/SaveImage 29, guards, actual preflight and mocked HTTP, native unclamped canvases and private metadata; sixth notebook UI and legitimate registry/count updates. Documents pinned large assets/archive, aspect-preserved source1536 and greedy Auto deviations, license review and unverified performance. Tests/docs only; no model downloads, inference or implementation edits.
- QuadView — Krea 2 and experimental DynamicCharacterSheet regression tests and `docs/KREA_CHARACTER_SHEETS.md`: active adapter chain, fixed 1536×1024 canvas/source aspect, greedy publisher-template captioning or manual headers, pre-upload validation, actual HTTP payload/SaveImage 29 filtering, UI defaults and engine metadata. Documented the explicit `krea-character-sheets` / `--comfy-krea-character-sheets` preset, two ~0.85 GiB adapters and two JSONs pinned to CharacterSheet revision `3dc4295163dacc924d213168d67bf16850fd954f`, shared FP8 adaptation and memory/text-accuracy limits. Includes strict expected-failure tests for fractional-CFG negative-conditioning metadata and the UI's dropped zero-weight required-sheet override. Tests/docs only; no implementation/downloader-test edits, live inference or downloads.

## 2026-10-06

### Added

- Explicit Python `--comfy-krea-originals` and PowerShell `krea-originals` preset for only three revision/size/SHA256-pinned assets (~13.06 GiB): INT8 Turbo alias, exact spacepxl Wan upscale VAE and exact skin filename from the timothy692 mirror (same SHA256 as gemasai). Excludes MoreReal, shared encoder and mandatory first adapter. Staged downloads validate size/hash and Safetensors headers; existing mismatches remain unchanged with an error. Skin license metadata is unavailable; review provenance/terms. Downloader never pickle-loads the `.pth`; runtime never downloads. No actual downloads performed for this documentation update.
- All2Real regression tests and `docs/KREA_ALL2REAL.md` covering required loader filenames and user-supplied prerequisites, shared two-pass adapters, seeded noise, pre-upload validation, UI/engine routing and four-tile skin detail. Documented native Wan upscale/memory uncertainty and unverified NO8D/WAS/VAEUtils parity; no live inference or downloads performed.
- Krea workflow asset inventory in `docs/KREA_MODELS.md`, with verified installed filenames/file sizes, mandatory and optional adapter chains, RTX 4070 Ti 12 GB hardware trade-offs, workflow overrides and the existing CFG-default inconsistency. No inference settings changed.

### Changed

- User-approved All2Real transformer substitution supersedes the original-exact decision: the historical suffixed checkpoint was not found. The explicit downloader pins current `Comfy-Org/Krea-2` remote `diffusion_models/krea2_turbo_int8_convrot.safetensors` at revision `6b1d7191d84d5ded74d83a1a98211dad0ac8ae25`, SHA256 `8e4eeda70dd5037ab1ba2bef6b417f9f901e26093117cf397f741fc1fdaaf3f1`, saving alias `krea2_turbo_int8_convrot-b19a4f0be264.safetensors`. This is NOT proof of identical upstream-original bytes; runtime still has no silent fallback. Updated All2Real/model/setup/Windows documentation with explicit usage and verification/provenance limits only; no implementation edits in this documentation pass.

### Fixed

- All2Real now pixel-shuffles the Wan VAE's 12 packed decoder channels into RGB after both decodes. Fixes the skin-detail RGB channel error and the intermediate re-encode silently discarding channels; tiled fallback allocates all 12 channels. Added exact channel-layout regression tests. Preserving original scaling enlarges the second pass and final output; runtime memory remains unbenchmarked.

- Krea UI defaults now use ComfyUI CFG 1 instead of the historical Diffusers-style 0, which discarded the positive prompt in standard KSampler guidance. Text generation rejects zero, negative and nonfinite CFG values with an explanatory error; positive fractional settings remain supported. Added prompt-routing and UI-default regression tests.
- Krea reference edit now also rejects nonpositive/nonfinite CFG. Added HTTP-boundary regression coverage confirming prompts, negative conditioning, dimensions, steps, seed, CFG and signed optional LoRA weights reach the active saved-output graph. Documented the existing reference-attention boost without changing its behavior.

## 2026-10-05

### Changed

- All five optional right-rail LoRA slots now accept finite weights from -2 to 2, including negative contributions; zero skips a slot. Krea reference/remix and both Qwen 2.1 profiles preserve signed adapter ordering, and Diffusers receives signed weights unchanged. Mandatory Krea-first, Remix, Turbo and dedicated BFS weights remain unchanged.

### Fixed

- Both Qwen 2.1 Turbo profiles now honor source-edit/swap ×2 as native diffusion at the previewed dimensions, using a separate configurable 2048-side/4-megapixel budget. Preview and engine sizing agree; other models and reference-conditioning limits remain unchanged.
- Qwen 2.1 r128 sampler preflight now recognizes ComfyUI V3 `COMBO` options, avoiding a false missing-RES4LYF error when `res_2s_ode` is already registered.

### Added

- Separate selectable **Qwen Image 2.1 + Turbo r128 (Civitai)** profile using exact version 3384956/file 3273779, shared INT8 assets, six manual sigmas, and `res_2s_ode`. Existing Viggle r256 remains unchanged. Missing sampler dependencies fail preflight rather than silently falling back.
- `qwen-2.1-r128` Windows download preset and validated Civitai download, with secure hidden Hugging Face/Civitai token prompts, environment/cached credentials and noninteractive authentication guidance.

## 2026-10-04

### Added

- Optional DLSS 5 enhancement for existing ComfyUI Qwen 2.1, Krea and FireRed workflows, with disabled-by-default UI controls, upstream parameters, node preflight and enhanced-resolution metadata. Node setup is included; the separately licensed native runtime remains a manual install. See `docs/DLSS.md`.
- Persistent `logs/` diagnostics: rotating application logs, memory snapshots, model-load stages, Python exception/fault handlers, and per-run Windows console capture with actual Python exit codes. Rapid AIO loading and offload failures now have explicit diagnostic stages.

### Changed

- Project-managed ComfyUI console output now goes to `logs/comfyui.log` rather than `outputs/comfyui.log`.

### Fixed

- Krea Composition remix now applies optional LoRA slots 1–5 after `Krea2_ALWAYS_LOAD_FIRST` and `Krea2-Remix_Patreon`, in slot order at selected weights, for the first pass only. Refinement remains mandatory-first-only; optional selections are validated rather than ignored.
- FLUX.2 Klein now loads its bundled Qwen3 encoder configuration explicitly and resolves the external GGUF weights to an absolute path, avoiding Transformers searching for the GGUF inside the snapshot's `text_encoder/` folder across all FLUX workflows.
- Generation worker exits such as `SystemExit` now report a failure rather than leaving progress streaming indefinitely. Windows launch failures now propagate Python's exit status.

## 2026-10-02

### Added

- Krea-only **Composition remix** operation for one uploaded composed canvas, using the supplied Remix adapter on the first pass and base-model refinement. Reuses existing FP8 assets, caps output at 1024, and omits upstream upscaling/DetailDaemon; see `docs/KREA_REMIX.md` for differences and prerequisites.

### Changed

- Windows ComfyUI setup now installs the Ostris Krea2 edit extension required by Composition remix.

### Fixed

- Generate buttons now stream four Gradio outputs (gallery, run details, status, progress) instead of returning the `_stream_run` generator object. This unblocks Krea reference edit (and Edit/Swap) on Gradio 6.
- Restricted the model-download ignore rule to the root `models/` folder so application source under `photo_edit_studio/models/` is no longer hidden from Git.

## 2026-09-28

### [0.9.0]

#### Added

- Qwen Image 2.1 **Head** and **Body** swaps now automatically apply the locally supplied `Qwen21-BFS_Head_v1.1.safetensors` or `Qwen21-BFS_Body_v1.1.safetensors` after mandatory Viggle Turbo and before optional adapters.
- Qwen Image 2.1 now supports up to five optional family-isolated `.safetensors` LoRAs at individual UI weights, chained after the mandatory unmerged Viggle Turbo LoRA for text, edit, combine and swap workflows.
- Qwen Image 2.1 + Viggle Turbo r256 as a six-step, INT8/low-VRAM ComfyUI option for Edit source, Combine images, Create from text and experimental two-image Head/Body instruction swaps; targeted ComfyUI assets and Windows download preset (`qwen-2.1`). Requires non-commercial Qwen Research License compliance.
- FireRed 1.1 ComfyUI GGUF Q4_K_M plus automatic Lightning 8-step v1.2 support with targeted downloads and a separate LoRA library.
- **Combine images** now offers 1K/2K output and 1:1, 9:16, or 16:9 aspect ratios, with a live size preview, separate combine-only memory budget, and saved aspect metadata.
- **Create from text** now shares the 1K/2K and 1:1/9:16/16:9 canvas controls, live preview, output budget, and saved canvas metadata; its selected dimensions reach the Krea ComfyUI graph.

#### Changed

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

#### Fixed

- **Krea reference edit** now hides the unrelated three-image **Combine images** upload section and the duplicate Krea text-to-image model description; its dedicated two-image controls and workflow label remain visible.
- **Create from text** now renders with the text-capable model selected in **02 / Generation settings**: Krea 2 Turbo through ComfyUI or FLUX.2 Klein 4B through Diffusers. The text-mode dropdown excludes image-edit-only and unsupported models, and FLUX text generation no longer expects an uploaded image.
- Krea reference edit and Krea head/body swap now chain selected Krea LoRAs into the ComfyUI model path at their UI weights, after the required identity-edit or BFS LoRA. Previously the extra UI slots had no effect.
- Reject missing, duplicate, wrong-family or unsupported optional Krea adapter files instead of silently ignoring them.
- Selecting the required identity-edit LoRA in Krea reference edit now adjusts its existing graph-node weight rather than raising a duplicate-base error; the BFS swap weight remains controlled by its dedicated slider.

## 2026-09-23

### [0.8.0]

#### Added · Project-managed Krea backend

- Added opt-in project-local ComfyUI and Krea2Edit setup in the Python 3.11 `.venv`.
- Added on-demand loopback subprocess management with low-VRAM startup, local logging and owned-process shutdown.
- Bundled a validated 1–2 image Krea2Edit API graph usable for reference and BFS head/body swaps without manual workflow export.
- Added an explicit `--comfy-krea` download option for the four required ComfyUI-format weights; no automatic multi-gigabyte downloads.

### [0.7.1]

#### Added · Krea reference editing

- Added a dedicated **Krea reference edit** UI workflow with source image and optional second reference.
- Added a separate ComfyUI Krea2Edit API graph adapter that verifies second-image grounding and latent wiring before inference.
- Retained Krea text-to-image through Diffusers and the existing BFS head/body swap workflows.
- Restricted the ComfyUI bridge to local loopback endpoints and documented required external setup.

### [0.7.0]

#### Added

- Separate two-image **Swap head / body** section preserving the other generation workflows and controls.
- Qwen 2511 and FLUX.2 Klein 4B BFS head swap adapters; Krea 2 BFS head/body swap through a local ComfyUI editing backend and validated API-format workflows.
- Explicit `--bfs-swap` downloader for four matching BFS LoRAs; no implicit download of the full upstream collection.
- `docs/SWAP.md` with model compatibility, required ComfyUI setup, consent, and quality limitations.

## 2026-09-17

### [0.6.1]

#### Added · FLUX.2 Klein 4B custom encoder

- Added a FLUX.2 Klein 4B-only Qwen3 GGUF text encoder loaded from `models/qwen3-4b-alb-q4_0.gguf`.
- Added compatibility for the existing `qwen3-4b-abl-q4_0.gguf` filename and an environment override.
- Added the `gguf` dependency and documented CUDA dequantization/system-memory behavior.

### [0.6.0]

#### Added · Krea 2 text-to-image

- Added gated `krea/Krea-2-Turbo` through Diffusers `Krea2Pipeline`.
- Added a **Create from text** workflow with 8-step, CFG-0, 1K low-VRAM defaults.
- Added Krea 2-specific LoRA storage and multi-adapter loading through Diffusers/PEFT.
- Added a `krea-2` Windows download preset and license/access documentation.

## 2026-09-15

### [0.5.0]

#### Changed · Windows and low-VRAM refactor

- Rebuilt the missing model package with local Diffusers adapters, registry, Rapid AIO loader, and CUDA memory management.
- Added Windows PowerShell setup and launch scripts for Python 3.11 and CUDA PyTorch.
- Added a Windows model downloader with recommended, Qwen AIO, Klein 4B, and explicit all-model presets.
- Added sequential CPU offload, VAE tiling/slicing, attention slicing, lazy CUDA loading, and explicit cache release.
- Enforced one-output, one-megapixel/1024-side generation defaults for an RTX 4070 Ti 12 GB.
- Made Rapid AIO and Klein 4B the recommended UI choices and marked Klein 9B unsupported on 12 GB.
- Updated all setup, architecture, model, usage, and troubleshooting documentation.

## 2026-09-14

### [0.4.3]

#### Fixed

- Changing a LoRA weight no longer crashes with `Setting requires_grad=True on inference tensor`. Adapter weights are applied in inference mode, and generation uses `torch.no_grad()` instead of `torch.inference_mode()`.

### [0.4.2]

#### Fixed

- Combine-images size preview no longer crashes with `needed: 7, got: 6` when some image slots are empty. Hidden Gradio widgets are not used as event inputs.

### [0.4.1]

#### Changed

- Combine-images output size is now a 1K / 2K / 4K canvas (1024, 2048, or 4096 square). Edit source still uses ×1 / ×2 / ×3 from the source aspect ratio.

### [0.4.0]

#### Added

- Output size now follows the first uploaded image's aspect ratio. Choose ×1, ×2, or ×3 instead of independent width/height sliders (clamped to 512–2048 and aligned to 64 px).
- Combine-images workflow: upload up to three references and a prompt to generate one new picture. Image 1 still sets the aspect ratio.

## 2026-09-12

### [0.3.2]

#### Fixed

- Rapid AIO load no longer builds the transformer on the meta device, which caused `Cannot copy out of meta tensor; no data!` when moving RoPE caches or enabling CPU offload.

### [0.3.1]

#### Fixed

- CUDA out-of-memory on Qwen 2511 / Rapid AIO: model CPU offload, VAE tiling, sequential batch images, allocator `expandable_segments`, and Rapid AIO no longer loads two transformers.

### [0.3.0]

#### Added

- Qwen Image Edit 2511 Rapid AIO as a selectable model. It loads the distilled ComfyUI transformer from `models/repos/Qwen--Qwen-Image-Edit-2511-AIO` onto the official 2511 pipeline.

### [0.2.0]

#### Added

- Local LoRA library with upload, up to five selected adapters per run, and per-LoRA weights.
- Qwen/FireRed and FLUX.2 Klein family LoRA folders under `models/loras/`.

### [0.1.1]

#### Changed

- Removed extra application-level content filters and the permission checkbox.
- Left model-native and license-required safeguards enabled.

### [0.1.0]

#### Added

- Local Gradio photo editing studio with a modern two-panel interface.
- Lazy, one-at-a-time adapters for Qwen Image Edit 2511, FireRed Image Edit 1.1, and FLUX.2 [klein] 4B/9B.
- Multi-reference inputs, identity-preservation prompting, seed, dimensions, steps, CFG, true CFG, strength, batching, and optional mask compositing.
- Optional lazy-loaded GFPGAN post-processing.
- Project-local snapshot downloader and strictly local runtime loading.
- Python 3.11/H100 setup and run scripts.
- Privacy-conscious metadata without prompt or input persistence.
- Setup, model, architecture, usage, license/safety, and troubleshooting documentation.
