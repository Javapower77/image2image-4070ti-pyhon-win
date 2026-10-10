# Krea 2 character sheets

Two operations are available under **Krea reference edit**:

| UI operation | Internal workflow | Required sheet adapter | Default sampling |
| --- | --- | --- | --- |
| QuadView — Krea 2 | `krea-quadview` | `QuadView_krea2_v1.safetensors` | 10 steps, CFG 1, Euler / simple |
| DynamicCharacterSheet (experimental) | `krea-dynamic-sheet` | `DynamicCharacterSheet_krea2_v1.safetensors` | 10 steps, CFG 1, LCM / simple |

These are adapted Krea 2 workflows, not identity swaps. Upload **Picture 1** only;
Picture 2 is hidden and ignored. Count and size multiplier are fixed to 1, and
face restoration is forced Off. No other character-sheet family is included here.

## Explicit installation

Run `.\scripts\setup-comfy.ps1` to install/update the local backend, then restart
ComfyUI after node changes.
Use the PowerShell command `.\scripts\download-models.ps1 -Preset krea-character-sheets`
or, in the activated project
environment, `python scripts/download_models.py --comfy-krea-character-sheets`.
The standalone preset does not request recommended models or identity/MoreReal
extras. Accept gated base-model terms and review all source licenses before use.

The four CharacterSheet downloads come from
[Alissonerdx/CharacterSheet](https://huggingface.co/Alissonerdx/CharacterSheet/tree/3dc4295163dacc924d213168d67bf16850fd954f),
pinned at **`3dc4295163dacc924d213168d67bf16850fd954f`**:

| File | Project destination | Pinned size |
| --- | --- | ---: |
| `QuadView_krea2_v1.safetensors` | `models/loras/krea2/` | 914,160,176 bytes (~0.85 GiB) |
| `DynamicCharacterSheet_krea2_v1.safetensors` | `models/loras/krea2/` | 914,160,184 bytes (~0.85 GiB) |
| `workflows/QuadView_krea2_v1.json` | `models/workflows/charactersheet/` | 30,378 bytes |
| `workflows/DynamicCharacterSheet_krea2_v1.json` | `models/workflows/charactersheet/` | 57,217 bytes |

The downloader verifies size/SHA256 and format before installing these four files.
These are pinned download sizes, not a claim that the files are installed locally.
Both upstream JSONs are downloaded; runtime constructs its own API graph rather
than queueing the editor JSON. QuadView JSON is a reference artifact. Dynamic JSON
is required at runtime, including manual-caption mode, because its full caption
template is extracted verbatim from **PrimitiveStringMultiline node 184**.
Its raw-byte SHA256 must be
`d144cda8af10c5c6e3fe8472d541f619a751f597cba298d581e23dbe5901b706`.

The full preset also supplies the shared runtime files:

- `vendor/ComfyUI/models/diffusion_models/krea2_turbo_fp8_scaled.safetensors`
- `vendor/ComfyUI/models/text_encoders/qwen3vl_4b_fp8_scaled.safetensors`
- `vendor/ComfyUI/models/vae/qwen_image_vae.safetensors`
- `vendor/ComfyUI/models/loras/Krea2_ALWAYS_LOAD_FIRST.safetensors`

Shared weights and the mandatory-first adapter are not covered by the above
CharacterSheet revision pin. The mandatory adapter is the project's alias for
`INFOMSG/Krea2_TextFusion/Krea2_TextFusion_Refusal_Reduction.safetensors`.
The shared **FP8 runtime adaptation is intentional and allowed here**, rather than
a claim of byte-identical execution of the publisher's original loader choices.
No original INT8, identity-edit, Remix or MoreReal bundle is required or stacked.
Missing files/nodes fail before source upload; runtime never downloads or
substitutes another profile. Use compatible core ComfyUI and `comfyui-krea2edit`;
Ostris nodes do not substitute for `Krea2EditModelPatch`/`Krea2EditGroundedEncode`.
Automatic Dynamic captioning additionally requires core `TextGenerate` with its
registered empty `sampling_mode=off` branch.

## Active graph and sizing

**Shared FP8 transformer → mandatory first adapter → selected sheet adapter →
optional Krea LoRAs (slot order) → Krea2Edit model patch → sampler → shared VAE
decode → SaveImage 29.**

The mandatory-first weight must be finite and in `(0, 2]`; default 1. The sheet
adapter defaults to 1. Up to five optional Krea `.safetensors` selections are
supported at finite weights −2 to 2; zero skips an optional adapter. A direct
request selecting the active sheet filename adjusts its existing loader instead
of duplicating it, including weight zero. The UI's generic selection filter
currently drops zero-weight slots, so that direct zero override is not available
through the UI. The other sheet profile, identity-edit, Remix, MoreReal and a
duplicate mandatory-first selection are rejected.

The source is EXIF-normalized to RGB and resized proportionally to **longest side
at most 1024**. It is **not square-cropped or stretched to the sheet aspect**;
smaller sources are not enlarged. Both grounded conditioners use this retained
source with **`grounding_px=0`**. The model patch uses source pixels and VAE latent,
`fit_mode=fit`, **`ref_boost=1` / `ref_boost_a=1`**, and the target latent.

The diffusion sheet is fixed at **1536×1024 landscape (1,572,864 pixels)**,
independent of source aspect and the ordinary 1024-side/1MP edit clamp. It is
higher than 1MP and may need additional memory; suitability for 12 GB VRAM is
unbenchmarked. There is no automatic smaller-sheet fallback. Optional DLSS acts
after decode and can change final saved dimensions, not the diffusion canvas.

## Prompt policies

### QuadView

The positive prompt always starts with the profile trigger:
“Convert the character in the image to a Character Sheet showing a face close-up,
front full body, side full body and back full body views”. Nonblank custom prompt
text is appended on a new line; it does not replace the trigger. No VLM caption
node is used. View completeness or consistency is not guaranteed.

### Dynamic (experimental)

Leave the prompt blank for automatic captioning. The full pinned publisher
template from node 184 feeds `TextGenerate` node 160 with the uploaded source and
shared encoder. Its string output **`["160", 0]`** feeds grounded positive
conditioner **84**. Runtime uses **greedy `sampling_mode=off`**, max length 2048,
thinking disabled and the default chat template enabled. This deliberately
differs from the publisher's **sampled/on** caption-generation setting; it is not
a shortened or reconstructed caption template.

A nonblank manual caption bypasses TextGenerate and must include:
`[TASK: ENTITY_SHEET_GENERATION]`, `[TEMPLATE: MULTI_ANGLE_ENTITY_SHEET_V1]`,
`[ENTITY_TYPE: ...]` and `[ENTITY_ID: ...]`, with nonempty entity values.
Header validation is **not a guarantee of a complete publisher-format caption**,
semantic correctness or a successful sheet. A stale ordinary edit prompt will
be rejected; clear it when selecting Dynamic automatic mode.

Dynamic defaults the negative prompt to `imperfect text, bad text, blur` unless
custom text is supplied. CFG must be finite and positive; **CFG 0 is rejected**
because standard ComfyUI KSampler would discard positive conditioning. CFG 1
uses positive-only conditioning; a negative prompt affects ordinary guidance
when CFG differs from 1. No on-sheet text accuracy claim is made.

## Validation, saved results and limits

Steps must be integers 4–40, default 10. Direct backend seeds must be integers
0 through `2**64-1`; the engine resolves negative integer seeds before validation.
Wrong model, missing/extra sources, mask, count other than 1, swap kind and size
multiplier other than 1 are rejected before upload. Loader/node/input/enum
registrations are checked against the configured graph before uploading pixels.

Only the single generated image from **SaveImage 29** is returned. Source and
preview outputs are not gallery results. Without DLSS, the engine rejects wrong
output count or dimensions instead of silently resizing. Saved PNG/run metadata
records resolved seed, actual uploaded-source and output dimensions, fixed canvas,
profile/revision, sampler/scheduler, adapter weights and caption policy; prompts
are not stored. Known metadata limitation: `negative_conditioning_active` is
currently false for fractional CFG below 1 even though negative conditioning
contributes there; a strict expected-failure regression test tracks it.

`tests/test_krea_character_sheet.py` uses temporary fake assets, a fake pinned
template/hash and mocked HTTP/engine outputs. Downloader tests remain separate.
**No live inference, real asset download, visual fidelity, text accuracy, peak
memory or performance validation was performed for this tests/documentation pass.**
