# Editing guide

## Studio layout (Beta)

The two-column studio follows `docs/mockup-ui/UI-MOCKUP.md`: a branded header, a **70% left creative canvas** with six icon-only workflow buttons beside the progress card, a separate card describing the active workflow, results, uploads and prompt bar. The **30% right utility rail** starts at the top with Advanced LoRAs, Face restoration and Other options. The workflow and prompt controls use reference SVG assets; the sixth notebook opens Character Sheet Creator. Hover or focus a workflow icon for its accessible name. **Settings** opens steps/CFG/strength, **Variants** shows the configured one-output limit, **Size** opens either source multiplier or canvas resolution/aspect, and the model-family icon opens the mode-compatible model picker. **Swap head / body** also shows **Replace** and **BFS weight**; Krea reference edit has a fixed model and no picker. Sheets hide model, Variants and Size triggers in favor of dedicated controls. Click an icon again to close its panel, or open another to switch panels. The circular **Generate** action remains at the right end of the prompt bar. On a narrow viewport, the rail moves beneath the canvas and the top cards stack when space is limited.

Below the prompt, the negative prompt is model-dependent. **Other options** contains seed, identity-preservation prompt, unload and clear. The progress card displays overall completion and elapsed time; click its far-right **Run details** icon to open saved output metadata and model status in a popup. Prompts are not stored there. Click the icon again, press **Escape**, or click outside to close it. The one-output hardware cap and five existing inference routes remain alongside the dedicated sheet route. A small browser-side visual layer anchors the actual Gradio control panels over the prompt toolbar as popovers; **Escape** or clicking outside dismisses an open panel, and Gradio still owns every input and inference event. If scripting is unavailable, the same controls remain reachable as inline panels.

## Character Sheet Creator

The toolbar now has **six** icons; the sixth notebook generates one Qwen 2.1
full-BF16 sheet from one reference. The five previous routes are preserved.
Choose Simple/Production, Static/Auto, and dedicated 1/3.4/6 binary-MP controls
(1248×832, 2304×1536, 3072×2048). Static uses archived text plus optional
customization; Auto uses the name for Production or description for Simple and
local greedy thinking captioning. Settings are 25 steps, CFG/true CFG 1, no mask,
negative conditioning, multiplier, swap or restoration. Model picker, Variants
and Size triggers are hidden. Compatible adapters share the qwen21 library
without mandatory Turbo. See [complete usage and exact assets](QWEN_CHARACTER_SHEET.md)
for source1536 aspect adaptation, archive pins, unverified performance and rights.

## Head / body swap

In **Swap head / body**, upload Picture 1 (target body/source scene) then Picture 2 (replacement face, head, or person reference), and choose a supported model and swap type. Qwen 2511 and FLUX.2 Klein 4B are Head-only; Krea supports Head/Body through local ComfyUI. The two ComfyUI Qwen 2.1 profiles and unstacked official BF16 support experimental Head/Body with canonical `bfs_head_v1.1_qwen_2.1.safetensors` or `bfs_body_swap_v1.0_qwen_2.1.safetensors` in `models/loras/qwen21/`. Use `qwen-2.1-bfs` / `--qwen21-bfs`; verified migration copies legacy aliases without deleting them (same bytes, not new weights; Body is v1.0). ComfyUI uses six steps and BFS after its matching Viggle r256 or r128 Turbo, then optional compatible LoRAs. Official uses eight steps, CFG/True CFG 1, an empty negative prompt, exactly two ordered images and **only** mandatory BFS at finite weight **0.1–1.5**, with no optional stack or DLSS. Official-extract is not offered for swaps. Complex swaps may ghost or change identity; official BFS inference, quality and VRAM fit are unconfirmed. See `docs/SWAP.md`.

## Create from text: choose the model

Choose **Create from text**, then open the **model icon in the prompt toolbar** to select **Krea 2 Turbo**, either six-step ComfyUI Qwen 2.1 Turbo profile (Viggle r256 or Civitai r128), **Qwen Image 2.1 Turbo (Official BF16)**, or **FLUX.2 Klein 4B**. Krea and the six-step Qwen profiles use project-managed ComfyUI; official BF16 Qwen and FLUX use Diffusers. Qwen 2511, Rapid AIO and FireRed remain editing-only. The Comfy Qwen profiles use six steps, CFG 1 and no negative prompt; compatible optional LoRAs follow their mandatory Turbo adapter. Official Qwen uses eight saved-sigma steps and rejects optional LoRAs. See `docs/MODELS.md` for profile-specific setup and license caveats.

Enter a detailed visual prompt. In the **Size** toolbar panel, set **Output resolution** to **1K** or **2K**, and **Aspect ratio** to **1:1**, **9:16** (portrait), or **16:9** (landscape). The preview shows the requested dimensions; they are sent to the selected model. Defaults are 1K, 1:1 and the selected model's step/guidance settings. 2K on a 12 GB GPU may be slow or run out of memory. Source images, masks, identity preservation, and edit strength do not apply in this workflow.

Krea 2 LoRAs can be uploaded in the same LoRA panel and are stored under `models/loras/krea2/`. Use only adapters trained specifically for Krea 2. Start with one LoRA near its author's recommended weight.

## Official Qwen 2.1 Turbo BF16

Select **Qwen Image 2.1 Turbo Official (BF16 · 12 GB offload unverified)**
(`qwen-2.1-turbo-official`) in
Edit source, Combine images, Create from text or experimental Head/Body swap. This loads the full
`Qwen/Qwen-Image-2.1-Turbo` Diffusers snapshot, not the Viggle/Civitai Comfy
profiles. [Upgrade requirements and opt-in download](SETUP.md#official-qwen-21-turbo-bf16-opt-in)
must be satisfied first; the earlier missing-pipeline installation was repaired
as recorded in SETUP. User-reported unstacked official generation does not verify
the BFS route. Live official BFS inference, quality and 12 GB fit are unconfirmed.

Use 1–3 ordered references for Edit/Combine and none for text. Settings are
locked to eight steps, guidance/True CFG 1 and an empty negative prompt.
The saved publisher schedule is
`1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568`,
with causal KV caching enabled; do not replace it with six-step Comfy sigmas.
For Head/Body, use exactly two images in body/source-then-reference order and
the sole matching canonical BFS adapter at positive weight **0.1–1.5**; no
six-step Turbo or extracted adapter is stacked. Optional LoRAs and DLSS remain
unavailable and rejected. Direct
requests must omit DLSS entirely (`dlss=None`), not send a disabled settings
dictionary. Edit strength and identity-preservation suffixes are not forwarded
by this adapter; phrase preservation requirements explicitly in the prompt.
Any mask compositing or face restoration is application post-processing, not
official pipeline conditioning.

## Official Turbo + extracted LoRA (experimental)

Choose the additional **Official + Extracted LoRA** model
(`qwen-2.1-turbo-official-extract`) in Edit source, Combine images or Create
from text after the opt-in `qwen-2.1-official-extract` download. Existing initial
choices/fallbacks are unchanged, and this is not a swap-picker model.
It uses **the full official Turbo snapshot plus the mandatory extracted LoRA**,
not the original base. This user-chosen stack on an already-trained/distilled
Turbo checkpoint is **experimental and NOT recommended**.

The mandatory `official_extract` adapter stays active at **weight 1.0** across
generation calls; it is not an optional slot or user-adjustable weight. Eight
steps, unit guidance/True CFG, empty negative prompt and the saved official
sigmas above remain locked, without schedule overrides. These fixed choices
are **not proven optimal**; no version-specific Civitai recommendations have
been established. Optional LoRAs/uploads, swaps and DLSS are rejected, including
`dlss={"enabled": False}`; omit DLSS entirely. Use 1–3 ordered edit/combine
references or no images for text. Express preservation requirements directly
in the prompt. Inference quality, compatibility, speed and 12 GB fit are unverified.

**Current engine boundary (2026-10-10, source inspection):** both official
profiles are validated before shared DLSS normalization, model selection or
sizing. Enabled and disabled DLSS dictionaries are rejected at that boundary,
superseding the earlier documented mutation bug. No tests were run or production
code changed in this documentation-only update.

## Krea 2 reference editing

Choose **Krea reference edit** to upload a source scene and an optional second reference (a person, object or style cue). Enter an editing instruction; this uses the project-managed ComfyUI Krea2Edit graph, not Krea's Diffusers text-to-image pipeline. Only 1–2 images are supported. The configured identity-edit LoRA is always used, followed by compatible Krea LoRAs selected in the UI at their chosen weights. Install the optional backend as described in `docs/KREA_REFERENCES.md`.

All **Krea** modes, including Krea **Create from text**, require `Krea2_ALWAYS_LOAD_FIRST.safetensors` in ComfyUI's `models/loras/`. Set its mandatory weight in the **Advanced · LoRAs** panel. It is applied before identity-edit, BFS, or optional LoRAs. FLUX text creation does not load this adapter.

## Output size

In **Edit source**, width and height follow the source photo. The Windows low-VRAM profile normally reduces the request to a maximum 1024-pixel side and one megapixel while preserving aspect ratio. The two six-step Comfy Qwen 2.1 profiles and official BF16 Qwen share a separate **×2 native-diffusion** budget: 2048 longest side / 4,194,304 pixels, with the same sizing helper for preview and generation. For example, a 3784×4852 source requests 1600×2048 at ×2, not twice the literal original dimensions. ×1/×3 retain the global budget. Configure the ×2 ceiling with `PHOTO_EDIT_QWEN21_X2_MAX_OUTPUT_SIDE` and `PHOTO_EDIT_QWEN21_X2_MAX_OUTPUT_PIXELS`. This also applies to Head/Body swaps on the two Comfy profiles and unstacked official BF16, not official-extract. This is not DLSS or a memory-fit guarantee; official Qwen rejects DLSS.

In **Combine images** and **Create from text**, open the **Size** toolbar panel to select **1K** or **2K**, then **Aspect ratio**: **1:1**, **9:16** (portrait), or **16:9** (landscape). These set the output canvas independently of input images. 1K produces 1024×1024 or 1024×576/576×1024; 2K produces 2048×2048 or 2048×1152/1152×2048. The longer side defines 1K/2K, and dimensions are aligned to 64 pixels. Canvas sizing takes precedence over source multipliers and uses its separate budget. Other edit/reference/swap requests retain the 1K low-VRAM cap, except the Qwen 2.1 ×2 policy above. 2K is more likely to exhaust the RTX 4070 Ti's 12 GB VRAM and takes longer.

## Combine images

Switch the workflow to **Combine images**, upload one to three pictures, and write a prompt that names them (`image 1`, `image 2`, `image 3`). Choose 1K or 2K and the desired aspect ratio; the size preview shows the actual canvas before generating. This generates a new picture from those references rather than editing a single source in place. Masks are ignored in this mode.

## Identity-preserving edits

Use a high-resolution source without heavy compression. Add a sharp, front-facing face crop as image 2 when the source face is small. Keep identity preservation enabled and phrase the request as a constrained delta: state the desired change, then list face, pose, body proportions, camera, lighting, and background elements that must stay unchanged.

Avoid asking for many unrelated changes in one run. Make one structural edit first, reuse its output as a new source, and then make a smaller refinement.

## Clothing changes

Use the person as image 1 and a clean product or worn-garment photo as image 2. Mention fabric, cut, sleeve length, closures, logos, and which image supplies the garment. Ask to preserve body proportions, pose, hands, face, and scene. Clothing replacement is generative, not measurement-accurate virtual fitting.

## Background and scene edits

Describe perspective, time of day, light direction, depth of field, and desired contact shadows. Explicitly preserve the subject, foreground edges, and camera framing. A white mask can constrain final compositing, but it does not currently condition denoising.

## LoRAs

Upload `.safetensors`, `.pt`, or `.bin` adapters in the **LoRAs** panel for Qwen 2511/Rapid AIO, FLUX.2 Klein 4B or Krea 2. For either six-step ComfyUI Qwen Image 2.1 profile use **only compatible `.safetensors` files**, stored in `models/loras/qwen21/`, and select up to five. The profile's mandatory Viggle Turbo r256 or Civitai Turbo r128 LoRA always loads first; chosen extras follow in slot order at their selected weights. These ComfyUI model-only adapter patches may increase VRAM/RAM and have not been quality-tested together. Do not reuse Qwen 2511 files or select either mandatory Turbo again. Official Qwen 2.1 Turbo BF16 rejects optional LoRAs; FireRed's automatic Lightning v1.2 LoRA also rejects additional adapters.

Choose up to five existing files and set each optional weight from **-2 to 2** (typically 0.6–1.2; 0 skips the slot). Negative weights reverse the adapter contribution and are passed unchanged to ComfyUI or Diffusers; visual quality is model-dependent. Nonfinite and out-of-range weights are rejected. Mandatory Krea-first remains greater than 0 and at most 2; required Remix/Turbo strength stays 1, and the dedicated BFS slider is unchanged. LoRAs must match the selected family: a FLUX.2 Klein adapter will not load on Qwen, and a Qwen Image Edit adapter will not load on Klein. Changing the base model reloads that family’s library.

LoRA names and weights are stored in run metadata. Prompt text is still not stored.

## Restoration

Generate without restoration first. If eyes, mouth, or pores are weak, enable GFPGAN at about 0.2–0.4. Higher values can produce smooth skin or identity drift. A future FaceDetailer-style implementation should use local detection, a padded face crop, low-denoise regeneration, and feathered paste-back; it is not falsely represented as available in this release.

## Reproducibility

Set a fixed non-negative seed. Metadata is saved beside outputs. Prompt text is deliberately excluded for privacy, so retain it separately if exact reruns matter.

For both official Qwen 2.1 Turbo profiles, PNG/`metadata.json` records the
validated saved `sample_sigmas`, `scheduler_source="checkpoint"`,
`use_kv_cache=true` and actual returned image sizes separately from requested
dimensions. These are not a trace of actual scheduler timesteps. Snapshot
revision and explicit BF16 precision are not recorded; a saved `strength`
field does not prove this adapter consumed it.
The extracted profile additionally records `experimental_stack`,
`checkpoint_source="official Turbo (not original base)"` and a separate
`mandatory_adapter` with filename/version/file/hash/size and weight 1.0;
`loras=[]` describes optional adapters only, not absence of the mandatory one.
Unstacked official BFS records `mandatory_adapter.name="bfs_swap"`, canonical
filename, published version (Head 1.1 / Body **1.0**), SHA256, size and requested
weight, plus `ordered_image_sources` and the fixed `swap_trigger`. Civitai
version/file IDs are not saved. User instructions and source images remain
excluded, but the fixed training trigger is stored. Byte checks happen before
cold checkpoint allocation; CPU normalization/real target-shape checks happen
after allocation, before BFS installation/offload. See [SWAP](SWAP.md) for this
validation gap and gate/proj intrinsic-scale-1 normalization details.
Retain exact snapshot configs/revision, dependency commits and prompt separately.
Neither reproducibility nor optimal stacking settings are proven by these
offline mocked tests.
