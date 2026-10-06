# Krea 2 workflow assets and hardware assessment

Inventory verified against application source and installed file sizes on **2026-10-06**.
Target: **RTX 4070 Ti, 12 GB VRAM, 64 GB system RAM, Windows, Python 3.11**.
Paths below are relative to the project root. Sizes are installed file sizes in
GiB, not measured peak RAM/VRAM or model quality scores.

## Shared files used by standard Krea workflows

Text, Reference edit, Composition remix and head/body swaps use the set below.
**All2Real is an exception:** it shares the encoder and mandatory first adapter,
but requires the approved INT8 transformer alias, Wan upscale VAE and skin-detail model
instead of the standard transformer/VAE. Its assets are not covered by the
installed-size claims in this table; see the All2Real subsection below.

| Role | Filename | Folder | Installed size |
| --- | --- | --- | ---: |
| Diffusion transformer | `krea2_turbo_fp8_scaled.safetensors` | `vendor/ComfyUI/models/diffusion_models/` | 12.239 GiB |
| Text/vision encoder | `qwen3vl_4b_fp8_scaled.safetensors` | `vendor/ComfyUI/models/text_encoders/` | 4.882 GiB |
| Image VAE | `qwen_image_vae.safetensors` | `vendor/ComfyUI/models/vae/` | 0.236 GiB |
| Mandatory first adapter | `Krea2_ALWAYS_LOAD_FIRST.safetensors` | `vendor/ComfyUI/models/loras/` | 0.026 GiB |

All files above were present. All active Krea workflows use the loopback ComfyUI
bridge, not the legacy Diffusers Krea adapter. The managed server uses `--lowvram`
CPU offload; the application’s Diffusers sequential-offload setting is separate.

The first adapter is a project-local alias: the downloader obtains
`INFOMSG/Krea2_TextFusion/Krea2_TextFusion_Refusal_Reduction.safetensors` and saves
it as `Krea2_ALWAYS_LOAD_FIRST.safetensors`. It is a project requirement, not proof
that the upstream base model needs this adapter. Its default weight is 1; the
dedicated UI control allows 0.05–2 and backend validation requires a positive weight.
Check each source’s license and provenance before use.

## Text-to-image

UI: **Create from text → Krea 2 Turbo**. Internal workflow: `krea-text`.

Model chain:

Chain: **FP8 transformer → mandatory first adapter → selected optional Krea LoRAs → sampler**.

The shared encoder encodes text without reference images. The graph creates an
empty latent canvas and decodes using the shared VAE. No identity-edit, Remix or
BFS adapter is automatically added.

`krea2_turbo_lora_rank_64_bf16.safetensors` is installed in `models/loras/krea2/`
(0.437 GiB), but remains an **optional selection**, not an automatic requirement.
Do not assume applying it to an already-Turbo checkpoint improves speed or quality;
that combination needs comparison testing.

Text supports the existing 1K/2K canvas controls. Start at 1K on this GPU.

## Image-to-image: reference edit

UI: **Krea reference edit → Reference edit**. Workflow: `krea-reference`.

| Required workflow adapter | Folder | Installed size |
| --- | --- | ---: |
| `krea2_identity_edit_v1_2_r128.safetensors` | `vendor/ComfyUI/models/loras/` | 0.851 GiB |

Chain: **FP8 transformer → mandatory first → identity-edit → optional LoRA 1–5 → Krea2Edit model patch → sampler**.

Accepts a source scene and optionally a second reference. Images feed both
`Krea2EditGroundedEncode` and the model patch’s pixel/VAE-latent paths.
The required extension is `comfyui-krea2edit`. Default reference adapter can be
overridden using `PHOTO_EDIT_COMFY_REFERENCE_LORA`.

Selecting the identity-edit filename in an optional slot adjusts the existing
adapter’s weight rather than creating another copy. The mandatory-first adapter
has its own control and must not be selected in an optional slot.

Reference editing retains a 1024-side/1,048,576-pixel diffusion budget. The Qwen
2.1 native ×2 exception does **not** apply to Krea.

The built-in patch uses `ref_boost=4.0`: with one source image this amplifies
attention to that source; with two it applies to the last reference. This favors
reference fidelity and can make style edits subtle. It is currently a graph
setting, not an exposed UI dial. Identity-edit remains active at weight 1 unless
its filename is selected in an optional slot to adjust that existing weight.
CFG 1 uses the positive instruction only; the negative prompt influences standard
guidance only when CFG differs from 1. Zero/nonfinite CFG is rejected.

For an adapter comparison, use the same source, prompt, seed and size, then change
only one optional weight between zero and its recommended value. Include the
adapter's documented trigger words. A connected loader does not prove its tensor
keys match the model or that it will overcome reference preservation.

## Image-to-image: composition remix

UI: **Krea reference edit → Composition remix**. Workflow: `krea-remix`.

| Required workflow adapter | Folder | Installed size |
| --- | --- | ---: |
| `Krea2-Remix_Patreon.safetensors` | `vendor/ComfyUI/models/loras/` | 0.213 GiB |

Accepts one already composed canvas. The application uses two sampling passes:

- **First pass:** mandatory first → Remix (fixed weight 1) → optional LoRA 1–5.
- **Refinement:** mandatory first only; no Remix or optional adapters.

Both use the same FP8 transformer and the Ostris model patch. The required extension
is `ComfyUI-Krea2-Ostris-Edit`; the identity-edit patch is not interchangeable with it.
Default: 9 total steps, CFG 1, Euler `kl_optimal` then `simple`, source-aspect output
within alignment, longest side capped at 1024. Reference KV caching remains enabled.

This is the application’s adapted pipeline, not the full `workflows/remix.json`.
That editor workflow references `krea2_turbo_int8_convrot-b19a4f0be264.safetensors`,
`qwenImageVAESharpKrea2_sharpFp32.safetensors`,
`Wan2.1_VAE_upscale2x_imageonly_real_v1.safetensors` and `4xFFHQDAT.safetensors`.
Those files are **not used by the built-in application Remix graph**. Internal
upscaling and DetailDaemon were omitted. See [Remix details](KREA_REMIX.md).

## Image-to-image: All2Real

UI: **Krea reference edit → All2Real**. Workflow: `krea-all2real`.

This is an explicit user choice of the All2Real asset set. **Changed decision:**
the exact historical suffixed transformer was not found, so the user approved
the current Comfy-Org/Krea-2 INT8 Turbo checkpoint saved under the historical
loader alias. This is not proof of identical upstream-original bytes. Required files:

- `vendor/ComfyUI/models/diffusion_models/krea2_turbo_int8_convrot-b19a4f0be264.safetensors`
- `vendor/ComfyUI/models/text_encoders/qwen3vl_4b_fp8_scaled.safetensors`
- `vendor/ComfyUI/models/vae/Wan2.1_VAE_upscale2x_imageonly_real_v1.safetensors`
- `vendor/ComfyUI/models/upscale_models/1x-ITF-SkinDiffDetail-Lite-v1.pth`
- `vendor/ComfyUI/models/loras/Krea2_ALWAYS_LOAD_FIRST.safetensors`
- `models/loras/krea2/Krea2-MoreReal.safetensors`
- `vendor/ComfyUI/custom_nodes/photo_edit_all2real.py` from the project setup,
  plus Ostris and compatible ComfyUI core nodes.

Explicit download: `.\scripts\download-models.ps1 -Preset krea-originals`,
or `python scripts/download_models.py --comfy-krea-originals`
in the activated project environment. Used alone, it obtains only three pinned
assets (~13.06 GiB): the approved INT8 alias, exact spacepxl Wan VAE, and exact
skin filename from the timothy692 mirror (same SHA256 as gemasai). It does **not**
supply MoreReal, the shared encoder or mandatory first adapter. These are pinned
download sizes, not installed-size or inference measurements.

Staged downloads and existing files are verified by size/SHA256, with Safetensors
header validation; existing mismatches are retained and cause an error. The
downloader never pickle-loads the skin `.pth`. Skin license metadata is unavailable;
review sources/terms before use. See [pinned revisions, hashes and provenance](KREA_ALL2REAL.md#explicit-three-asset-download).

Runtime never downloads or silently falls back to the standard FP8/Qwen VAE set.
The approved alias is an explicit installation decision, not a runtime fallback.
Both passes share
**mandatory first → Ostris → MoreReal (default 0.9) → optional adapters**.
Library loaders use qualified `krea2/` names. Defaults: 11 total steps, CFG 1,
`er_sde` with `kl_optimal` (1–8), then `simple` (9–11); the editor's exposed 9 is
a seed, not total steps. One source only; no identity-edit or Remix adapter.

Source upload is capped at 1024, then the graph scales to 1MP and FluxKontext
buckets. Both Wan decodes unpack 12 subpixel channels into RGB using 2× pixel
shuffle. The second pass therefore has 4× latent spatial area, and final dimensions
can be 4× the source bucket before DLSS. Memory impact remains unbenchmarked.
Four-tile skin detail follows final RGB decode.
The project's NO8D/WAS replacements and core decode are **not verified exact
upstream/VAEUtils parity**, and no 12 GB memory or visual-quality claim is made.
See [All2Real prerequisites and limits](KREA_ALL2REAL.md).

## Image-to-image: head/body swaps

UI: **Swap head / body → Krea 2 Turbo**. Workflow: `swap`.

| Selection | Required adapter in `models/loras/krea2/` | Installed size |
| --- | --- | ---: |
| Head | `bfs_head_swap_v1.1_krea2.safetensors` | 0.851 GiB |
| Body | `bfs_body_swap_v1_krea2.safetensors` | 0.851 GiB |

Chain: **FP8 transformer → mandatory first → selected BFS adapter → optional adapters → Krea2Edit model patch → sampler**.

The BFS adapter replaces the reference graph’s identity-edit adapter; identity-edit
and BFS are not automatically stacked. Picture 1 is the target scene/body and
Picture 2 the donor. The dedicated BFS weight control is separate from optional
LoRA weights. These workflows retain the 1024-side/one-megapixel budget.

## Optional adapters and final enhancement

Optional Krea `.safetensors` files belong directly in `models/loras/krea2/`.
Weights −2 to 2 preserve slot order; zero skips the slot. Swaps also consume an
adapter entry for BFS, limiting further adapters according to swap validation.
More adapters increase loading/patching overhead and can require additional memory.
Folder placement alone does not establish architectural or training compatibility.

DLSS, when enabled, processes the decoded generated IMAGE before saving. Its native
runtime is separate from Krea weights. DLSS scaling affects final dimensions, not
the diffusion canvas, and can increase memory use. Start with 1×. See [DLSS](DLSS.md).

## Are these the best files for the target hardware?

**They are a practical low-memory starting point, not a proven optimal setup.**

| Choice | Assessment for this application |
| --- | --- |
| FP8 transformer | Smaller than full BF16, but the installed file alone is slightly larger than GPU VRAM. CPU offload is essential; activations, adapter patches and the encoder need additional space. |
| FP8 4B vision encoder | Sensible memory-saving choice for reference workflows. The 4.882 GiB file cannot remain on the GPU alongside the whole transformer. |
| Shared image VAE | Avoids the original Remix workflow’s extra sharp/upscale VAE dependencies. Small file size does not eliminate high-resolution decode peaks. |
| Approved INT8 transformer alias for All2Real | Current publisher checkpoint substituted explicitly with user approval; historical-byte identity is unverified. Not a standard-workflow fallback. Quality, speed, adapter compatibility and decode/re-encode memory are unbenchmarked; the filename alone does not establish lower peak memory. |
| Full BF16 transformer/encoder | Not recommended as the default for 12 GB: more CPU-memory/storage pressure and offload work without demonstrated benefit here. |
| Five optional adapters | Supported functionally, not a recommendation to use all five. Begin with none, then add one at a time. Negative weights may destabilize output. |
| 2K text or DLSS upscaling | Available, but not guaranteed to fit 12 GB. They have different memory costs and should be tested separately. |

Existing ComfyUI logs show Krea Remix-style two-pass runs completing, including
DLSS feature-18 verification. They do not establish peak memory, optimal visual
quality, or that every workflow/optional-adapter combination is validated.

### ComfyUI CFG defaults

The Krea registry and shared UI callbacks now default to CFG 1, matching the
built-in text/reference templates and reference/remix operation defaults.
ComfyUI's standard guidance formula is negative prediction plus CFG times the
difference between positive and negative predictions: CFG 0 ignores the positive
prompt, whereas CFG 1 uses positive-only conditioning. Text generation rejects
zero, negative and nonfinite CFG values with an explanatory error; positive
fractional values remain available for deliberate experimentation.

The official Diffusers `guidance_scale=0` recommendation uses different semantics
and must not be copied to ComfyUI KSampler. Restart the studio and inspect Run
Details when comparing results; existing browser controls may retain old values.

### Suggested comparison procedure

1. Keep a fixed seed, prompt, source and output size; disable optional adapters,
   restoration and DLSS initially.
2. Test text and reference edit at 1K, then Remix and each swap separately.
3. Record time, peak dedicated GPU memory, CPU memory, visual fidelity and errors.
4. Add one optional adapter at a time, preserving seed/settings.
5. Compare quantized base alternatives only with verified compatible encoder,
   patch and adapter files; retain the working set until the alternative is proven.

Use a fast local SSD, substantial free RAM/pagefile capacity, and one job at a time.
The target’s 64 GB RAM helps offload but does not guarantee every load-time or
sampling allocation succeeds. Diagnostics are in `logs/studio.log` and
`logs/comfyui.log`.

## Overrides, downloads and source of truth

- The Krea download preset supplies standard shared assets, mandatory first,
  identity-edit and the optional official Turbo adapter. Remix and MoreReal remain
  user supplied. Separate `--comfy-krea-originals` / `krea-originals` downloads
  only the three pinned All2Real assets with the approved INT8 alias substitution;
  it is not included automatically in recommended or all-model presets. All2Real
  runtime never substitutes the standard preset's transformer or VAE.
- BFS download support obtains the matching head/body files separately.
- Reference/head/body workflow overrides must be **API-format** files. If configured
  files do not exist, the bridge falls back to built-in graphs. Text/Remix/All2Real
  use built-in templates directly; All2Real fails closed on missing required
  loader filenames/nodes. Runtime preflight is not proof of historical-byte identity.
- Custom API exports may change checkpoint filenames; this inventory describes
  the built-in graphs and the installed defaults, not every possible override.
- Source: `photo_edit_studio/comfy_assets.py`, `comfy_workflows.py`, `krea_all2real.py`,
  `models/comfy_swap.py`, `swap.py`, `models/registry.py`, `ui.py`,
  `scripts/download_models.py` and `config.py`.
