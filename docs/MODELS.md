# Models and 12 GB guidance

| UI model | Default steps | 12 GB status | Notes |
| --- | ---: | --- | --- |
| Krea 2 Turbo | 8 | Supported with offload | Gated 12B text-to-image model; ComfyUI CFG 1; Krea 2 Community License. |
| Qwen Rapid AIO | 4 | Recommended | Best first choice for Qwen edits; requires official Qwen 2511 components. |
| FLUX.2 Klein 4B | 4 | Recommended | Fast distilled option; use 1K output and one image. |
| Qwen Image Edit 2511 | 40 | Advanced | Runs through sequential CPU offload and is substantially slower. |
| Qwen Image 2.1 Turbo (Official BF16) | 8 | Unverified; no 12 GB fit guarantee | Separate `qwen-2.1-turbo-official` Diffusers profile, publisher snapshot and saved schedule; experimental Head/Body with sole mandatory BFS, no optional LoRAs or DLSS. |
| Qwen Image 2.1 Turbo Official + Extracted LoRA | 8 | Experimental; NOT recommended | `qwen-2.1-turbo-official-extract` stacks a mandatory extracted adapter on the already-distilled full official Turbo checkpoint, not the original base. No 12 GB fit guarantee. |
| Qwen Image 2.1 + Viggle Turbo r256 | 6 | Experimental / low-VRAM offload | INT8 base, unmerged six-step LoRA, 1–3 reference edits, text-to-image and instruction swaps. Non-commercial research license. |
| Qwen Image 2.1 + Turbo r128 (Civitai) | 6 | Experimental / low-VRAM offload | Separate r128 compatibility adapter; manual sigmas and `res_2s_ode`; requires a registered compatible sampler extension. |
| FireRed Image Edit 1.1 | 8 | Advanced/offload | Q4_K_M GGUF, FP8 Qwen2.5-VL vision encoder and automatic Lightning v1.2 via ComfyUI; substantial RAM/disk required. |
| Qwen Image 2.1 Character Sheet Creator | 25 | Unverified full-BF16/offload | Dedicated sixth mode, not a normal model-picker choice; CFG 1, res_multistep/beta, no mandatory Turbo. |

Diffusers models use local snapshots; FireRed instead loads only targeted ComfyUI GGUF/LoRA weights. CPU offload is mandatory for the large models on the target card. A current Diffusers source build is required for the Diffusers pipelines.

## Qwen Image Edit 2511 mixed-format LoRAs

`qwen-2511` uses its local Diffusers edit pipeline. BFS supports **Head only,
not Body**; Qwen 2.1 Head/Body assets are a different family. Its BFS file is
`models/loras/qwen/bfs_head_v5_2511_merged_version_rank_16_fp16.safetensors`.

**Measured header, 2026-10-10:** the local file is **307464216 bytes**, with a
375696-byte JSON header and 2779 tensors: **846 down/up matrix pairs**, **846
`.diff_b` bias deltas**, **241 `.diff` direct weight deltas**, and **no `.alpha`
tensors**. This provenance reference is pinned to the local Hugging Face cache's
recorded `Alissonerdx/BFS-Best-Face-Swap` revision
`c758d55502463bc5eb281326c107d0850d7a8b4a` and LFS ETag
`1315a08947e5d6d7c53ea4fc59f272e0d54efd7f2da35999b7d873a7cf4fa89b`.
Only the header/cache metadata were read; no full-file hash or payload validation
was performed here. These counts are a measured asset description, **not runtime
count requirements**. The existing BFS downloader does not enforce this pin.

`QwenAdapter.apply_loras` parses all tensors on CPU rather than passing this
mixed file into Diffusers' legacy converter. Complete old down/up or normalized
A/B pairs retain their actual rank; omitted alpha means **alpha = rank**, hence
intrinsic scale **1**, not an invented rank16 or zero update. Explicit finite
scalar alpha is retained algebraically by folding **alpha/rank into B**. Signed
request strength is passed separately to `set_adapters`, exactly once. Direct
bias and weight deltas are added to pristine base parameters at that same signed
strength; alpha does not scale those direct deltas. No tensor is silently dropped.

All files, paired shapes, materialized targets and aggregate direct updates are
validated before replacement/mutation. Unknown keys/targets, orphan or mixed
pairs/alpha and bad shapes fail closed. Offload hooks are removed before
validation, capture or restoration and memory configuration is reinstated after
installation. Weight/file changes and deselection restore exact CPU originals;
overlapping direct deltas are aggregated from the pristine base, and unchanged
signatures do not compound updates. Installation/offload failures discard the
pipeline rather than claiming rollback. Synthetic CPU tests cover these
contracts, including real Accelerate meta/offload hooks when available; no full
model weights, inference, visual quality or hardware fit are validated here.

## Qwen Image 2.1 Turbo (Official BF16)

The registry key **`qwen-2.1-turbo-official`** selects
`Qwen/Qwen-Image-2.1-Turbo` through `Qwen21OfficialAdapter`, family
`qwen21-official`. This is a separate full-BF16 Diffusers profile, **not** the
six-step INT8 ComfyUI profiles `qwen-2.1-turbo` (Viggle r256) or
`qwen-2.1-turbo-r128` (Civitai r128). It does not attach either six-step LoRA.
Edit source and Combine images accept 1–3 ordered references; Create from text
accepts none. Eight steps, guidance/True CFG 1 and an empty negative prompt are
required. Experimental Head/Body swaps accept exactly two ordered images
(body/source first, replacement reference second) and the sole matching mandatory
BFS adapter at finite weight **0.1–1.5**. No Viggle/r128 or extracted adapter is
stacked on this route. Optional LoRAs and DLSS are rejected, not ignored;
`qwen-2.1-turbo-official-extract` still rejects swaps.

Use `qwen-2.1-bfs` / `--qwen21-bfs` for the two pinned canonical assets in
`models/loras/qwen21/`: Head `bfs_head_v1.1_qwen_2.1.safetensors` (**1.1**,
260096144 bytes, Civitai 3356102/file 3248321), Body
`bfs_body_swap_v1.0_qwen_2.1.safetensors` (**1.0**, 209753576 bytes,
Civitai 3363725/file 3251548). Head SHA256 is
`d1d748d5601077f3b6d05766f6823510e901970d916afa404a97e85dc92fa88e`;
Body SHA256 is `7dc0a53aba4dbc70c204f498936a619d226559da11011f748c389189af46b664`.
These are the same verified bytes formerly named `Qwen21-BFS_Head_v1.1.safetensors`
and `Qwen21-BFS_Body_v1.1.safetensors`, **not new weights**; the Body alias does
not change its published version. Verified migration copies keep legacy originals.
See [pins, delivery and migration](SWAP.md#canonical-qwen-21-bfs-assets-and-migration).

On a cold official BFS load, byte size/SHA256 checks precede checkpoint allocation.
CPU A/B parsing and **actual target-shape validation follow checkpoint allocation**,
before installation/offload: shape failures can still consume full-checkpoint RAM.
Pinned structure is 176 Head / 136 Body A/B pairs, no direct deltas. Fused `gate_up`
splits gate-first into `gate_layer`/`proj`, reusing A; missing alpha means intrinsic
scale **1**, with request weight applied separately once via `set_adapters`.
No full pre-allocation shape gate or live BF16 BFS compatibility is claimed.
Saved PNG/JSON metadata includes sigmas, KV cache, output sizes, ordered sources,
fixed trigger and BFS filename/version/hash/size/weight, not Civitai IDs or actual
scheduler timesteps. [Official swap contract and limits](SWAP.md#official-bf16-headbody-bfs-experimental)
remain experimental: no live inference, quality, performance or VRAM fit confirmed.

The authoritative `model_index.json` must declare
`_class_name: "QwenImage21Pipeline"` and exactly these eight `sample_sigmas`:
`1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568`.
Do not append zero or replace the scheduler/grid with either Comfy schedule.
`transformer/config.json` must retain `causal_condition: true`. The adapter loads
locally with `torch.bfloat16`, rechecks the loaded configs and calls with
`num_inference_steps=8`, `true_cfg_scale=1.0` and `use_kv_cache=True`; it relies on
the saved schedule rather than passing replacement sigmas.

Required Diffusers capabilities are those of upstream
[PR #14950](https://github.com/huggingface/diffusers/pull/14950), merge commit
`da1d3829cf08d4f329b526d89e17cc035c049d8d` or a compatible descendant, plus
Transformers `>=5.17,<6`. The adapter checks `QwenImage21Pipeline`, an explicit
constructor parameter `sample_sigmas`, and explicit call parameters `prompt`,
`image`, `width`, `height`, `num_inference_steps`, `true_cfg_scale`, `use_kv_cache`,
`generator`, `num_images_per_prompt`, `callback_on_step_end` and
`callback_on_step_end_tensor_inputs`. A dev version label or generic `**kwargs`
is not sufficient; missing capabilities/configs fail rather than falling back
to an older Qwen edit pipeline.

**Historical source-only check on 2026-10-10:** the workspace `.venv` contained Diffusers
`0.41.0.dev0` at commit `d77d53044518ed45583ce3ff680f9f87f816aeb3`, but its
Qwen pipeline files/exports lack `QwenImage21Pipeline`; its transformer source
also lacks `causal_condition`. This installation cannot meet the official
adapter contract at that time. No imports, model loading or inference were used
for that initial check. The subsequent pinned reinstall and API verification are
recorded in SETUP. The user reports successful testing of the **unstacked official
profile** in the repaired setup; this is user-reported, not independent live
verification in this tests/docs update, and does not validate the extracted stack.
See [explicit dependency upgrade and full-snapshot download](SETUP.md#official-qwen-21-turbo-bf16-opt-in).

The registry's 12 GB minimum is **not a fit guarantee**. BF16 weights, reference
conditioning, KV cache and larger canvases require substantial RAM/disk and
offload; actual RAM/VRAM use, performance and visual parity are unverified.
Configuration checks do not prove checkpoint completeness or stored precision.
The registry marks this profile as Qwen Research License/non-commercial.
Review the publisher snapshot's current license and NOTICE separately; live
terms were not verified here and should not be inferred from the Viggle/Civitai
adapters.

## Qwen Image 2.1 Turbo Official + Extracted LoRA (experimental)

The separate key **`qwen-2.1-turbo-official-extract`** uses the dedicated
`Qwen21OfficialExtractAdapter` subclass and the **full official
`Qwen/Qwen-Image-2.1-Turbo` snapshot plus an extracted LoRA**. This is the
explicitly user-chosen stack, **not the original non-Turbo base plus LoRA**,
not a replacement transformer and not either six-step ComfyUI profile.
The official Turbo checkpoint is already trained/distilled: stacking its
extracted adapter is **experimental and NOT recommended**. Successful byte
validation does not prove compatibility, benefit, quality or optimal settings.

The mandatory asset is Civitai version **3394831**, file **3284648**:
`qwen_image_2.1_turbo_lora_avg_rank_178_bf16.safetensors`, **913314512 bytes**,
SHA256 `208DD43250E1E01467BA190572AE2E7107A7870EC1FF026BC65F791FA1E80E95`.
Its canonical download URL is
<https://civitai.com/api/download/models/3394831?fileId=3284648>.
There are **no version-specific Civitai recommendations established here**;
do not interpret weight **1.0** or the official saved eight-sigma schedule as
author-endorsed or proven optimal for this stack. Review source/license terms
separately; no live Civitai page or inference was used for this update.

The loader checks exact size/SHA256 and a nonempty Safetensors header on CPU
before runtime/hardware/checkpoint allocation. The pinned asset is **mixed-format**:
**200 source down/up/alpha triples (200 source LoRA pairs)** plus **65 direct
`.diff` deltas**: Q/K attention normalization weights in all 32 blocks (64)
and `txt_in.text_norm.weight` (1, not `txt_norm`). The helper accepts synthetic
subsets; it does not require a 200-pair or 65-delta count at runtime. Exact asset
identity is enforced separately by the dedicated adapter's byte pin.

After loading the full pipeline and **before CPU/sequential offload hooks**, the
helper reads real tensors on CPU, checks source keys, complete triples, scalar
finite nonnegative alpha, shapes and every materialized transformer target.
Unknown keys/deltas, orphan alpha, missing targets, meta parameters and wrong
shapes fail before any direct parameter update or adapter installation. Each
normalization delta is **added to the existing weight**, not substituted,
discarded or sent through the legacy LoRA converter. Biases and other base
parameters remain untouched by this preparation.

Only the LoRA portion goes through the installed Qwen converter. It incorporates
`alpha/rank` into A/B; alpha is not ignored. Zero alpha produces an exact zero
update. Fused image-MLP `gate_up` pairs are split with **gate first, up second**:
shared A and the corresponding B row halves target `gate_layer` and `proj`.
Thus the resulting target-pair count can exceed the source-pair count; neither
half is discarded. The helper passes the **converted state dictionary**, not
the mixed file path, to `load_lora_weights` with adapter name `official_extract`,
then activates `set_adapters(["official_extract"], adapter_weights=[1.0])`.
Weight 1 preserves the already-converted alpha/rank scaling; it does not undo
or rescale the additive normalization deltas. Source tensors/file bytes are not
mutated. Validation failures are pre-mutation; if installation/activation or
offload subsequently fails, the adapter disposes of the entire pipeline rather
than claiming rollback of already-applied deltas. Generic optional
LoRA application is bypassed so it cannot unload the mandatory adapter.
The saved publisher sigmas and causal KV-cache config remain authoritative;
no scheduler/sigma override is supplied. Weight 1 and saved sigmas are fixed
profile choices, not evidence of optimality.

Edit source, Combine images and Create from text expose this additional choice;
existing initial selections/fallbacks remain unchanged. Swaps, optional LoRAs,
nonempty negative prompts, nonunit guidance/True CFG and DLSS (even a disabled
settings dictionary) are rejected. Installation uses the opt-in
[`qwen-2.1-official-extract` preset](SETUP.md#official-turbo-extracted-lora-opt-in).
Actual inference, performance, visual parity and 12 GB fit remain unverified.
Offline tests use tiny real-Torch CPU modules, all 65 allowed deltas and synthetic
1/200-pair payloads, the installed converter, alpha/algebra assertions and the
real pre-offload hook. They are not a full-checkpoint or live-inference result.

## Qwen Image 2.1 Character Sheet Creator

The dedicated `qwen-2.1-sheet` model retains the user-chosen author full-BF16
transformer, encoder and VAE (32.44 GB / 30.21 GiB total); Auto requires a separate
5.59 GB INT8 PE. Native 1/3.4/6 binary-MP sheet canvases bypass ordinary edit caps.
The explicit `qwen-character-sheet` download preset fetches all four pinned
weights and archive, but runtime Static requires no PE. No mandatory Turbo or
precision substitution. Twelve GB is not a fit guarantee; performance and exact
publisher parity are unverified. See [exact assets, archive and usage](QWEN_CHARACTER_SHEET.md).
Qwen non-commercial research restrictions and separate workflow/PE rights review apply.

## Qwen Image 2.1 + Viggle Turbo (12 GB experimental)

Install the project-managed ComfyUI 0.37.0+ backend (`.\scripts\setup-comfy.ps1`) and run `.\scripts\download-models.ps1 -Preset qwen-2.1`, then `.\scripts\run.ps1 -ComfyUI`. The targeted download gets `qwen_image_2.1_int8_convrot.safetensors` (~7.3 GB), `qwen3vl_8b_int8_convrot.safetensors` (~9.4 GB) and `qwen_image_2.1_vae_bf16.safetensors` (~0.7 GB) from `Comfy-Org/Qwen-Image-2.1`, plus the **exact** `Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r256.safetensors` (~1.4 GB) and upstream `comfyui/viggle_turbo.py` node from `Viggle/Qwen-Image-2.1-viggle-turbo`. No full BF16 Qwen snapshot or alternative r128 LoRA is downloaded. Restart ComfyUI after adding the node. Leave substantial disk and free system RAM; INT8 files cannot both fit in 12 GB VRAM at once, so `--lowvram` CPU offload is essential and may be slow or run out of memory at 2K. Actual inference on the target GPU remains to be verified.

The app uses Viggle's **unmerged** LoRA at weight 1, fixed six-step raw sigmas `1.0, 0.9375, 0.875, 0.75, 0.5, 0.25`, Euler, CFG 1, and no negative prompt. Those settings are locked in the UI. Up to five *optional Qwen Image 2.1-compatible* `.safetensors` LoRAs can be uploaded to `models/loras/qwen21/` and selected in the five slots at individual weights; ComfyUI applies them **after** the mandatory Viggle adapter. Viggle itself stays in `vendor/ComfyUI/models/loras/` and must not be selected a second time. The additional LoRAs use ComfyUI's model-only loader (merged patches), unlike Viggle's unmerged node; combinations are not validated for image quality, may require extra VRAM/RAM, and must not be Qwen 2511 adapters. Edit source and Combine images accept 1–3 references and Create from text accepts none. **Head swap** and **Body swap** automatically load canonical `bfs_head_v1.1_qwen_2.1.safetensors` or `bfs_body_swap_v1.0_qwen_2.1.safetensors` from `models/loras/qwen21/` after Viggle; the BFS weight slider applies and up to four further optional adapters can follow. Use `qwen-2.1-bfs` / `--qwen21-bfs` for these pinned files and verified legacy-alias copying; they are the same bytes, not new weights. Complex swaps may ghost or alter identity; these are generative edits, not reliable exact face transfers. The source and Viggle adapter carry the **Qwen Research License Agreement (non-commercial research/evaluation only)**. Check upstream license and NOTICE before use.

## Qwen Image 2.1 + Turbo r128 (Civitai)

Select **Qwen Image 2.1 + Turbo r128 (Civitai)** beside the existing Viggle r256
option. Its model key is `qwen-2.1-turbo-r128`. It reuses the same INT8 transformer,
encoder and VAE but loads **only** `Qwen-Image-2.1-turbo-v0.2.1-6step-lora-r128.safetensors`
as its mandatory turbo adapter. It does not stack r128 on Viggle r256.

The exact Civitai version is **3384956**, file **3273779** (~648 MB):
<https://civitai.red/models/2958738/qwen-21-turbo-loras?modelVersionId=3384956>.
The page attributes the original weights to isHeSatoshi and describes its file as
a compatibility modification for ComfyUI's built-in LoRA loader. The Qwen Research
License applies; review the linked source/license before use.

The profile locks adapter strength 1, CFG 1, six steps and manual sigmas
`1.0, 0.9375, 0.875, 0.75, 0.5, 0.25, 0.0` with `res_2s_ode`.
The sampler must be registered in ComfyUI (upstream extension:
<https://github.com/ClownsharkBatwing/RES4LYF>). Install a compatible extension
following its instructions and restart ComfyUI; the app fails preflight if the
sampler is missing rather than switching to Euler. The extension is not installed
automatically by the model downloader.

Use Windows download preset `qwen-2.1-r128`, or select `qwen-2.1-turbo-r128` in
`scripts/download_models.py`. Downloads reuse shared base assets and fetch the
exact Civitai file, validating Safetensors format and SHA256 when supplied by the API.
This profile supports text, edit, combine, Head/Body swaps and optional DLSS.
Optional Qwen 2.1 LoRAs follow the mandatory r128 adapter; BFS adapters follow it
for swaps. Do not select either mandatory turbo file in an optional slot.
Both six-step ComfyUI Qwen 2.1 profiles accept finite optional LoRA weights from **-2 to 2** in
slot order; negative weights are passed through and zero skips the slot.
Mandatory Turbo strength remains 1 and dedicated BFS weights are unchanged.
Actual inference and visual quality of this profile remain unverified.

## Download authentication

`scripts/download_models.py` requests Hugging Face/Civitai tokens using hidden
terminal input when credentials are absent. A blank response tries public access.
Existing `HF_TOKEN` or cached Hugging Face login credentials are reused; Civitai
uses `CIVITAI_API_TOKEN` or `CIVITAI_TOKEN`. Noninteractive runs do not wait for input:
provide credentials through these environment variables for gated downloads.

Enter tokens directly in the terminal, never in chat. The downloader does not print
tokens or authenticated URLs and does not save typed credentials. It strips
credentials from cross-host redirects and sanitizes request failures.

## FireRed GGUF + Lightning (Windows 12 GB)

Install the project-managed backend with `.\scripts\setup-comfy.ps1`, then run `.\scripts\download-models.ps1 -Preset firered` or `python scripts/download_models.py firered-1.1`. Start the application with `.\scripts\run.ps1 -ComfyUI`. This downloads exactly `FireRed-Image-Edit-1.1-transformer-q4_k_m.gguf`, `qwen_image_vae.safetensors`, and `FireRed-Image-Edit-1.1-Lightning-8steps-v1.2.safetensors` from the official FireRed ComfyUI repository, plus the complete `qwen_2.5_vl_7b_fp8_scaled.safetensors` vision/text encoder from `Comfy-Org/Qwen-Image_ComfyUI`. The FireRed repository's Q8 GGUF encoder is deliberately not selected: ComfyUI-GGUF expects a separate multimodal projector (`mmproj`) that the repository does not include. The transformer GGUF alone is about 12.17 GiB and **does not fit entirely into 12 GB VRAM**; ComfyUI's `--lowvram` CPU offload is required and runs may be slow. Allow substantial free disk and system RAM. Lightning is applied automatically at strength 1 with 8 default steps, Euler/simple sampling; optional user FireRed LoRAs are rejected because GGUF LoRA compatibility is experimental. Live output on the target GPU must be verified after downloading the weights.

**Create from text** offers Krea 2 Turbo and the two six-step Qwen 2.1 Turbo profiles through project-managed ComfyUI, or Official Qwen 2.1 Turbo BF16 and FLUX.2 Klein 4B through Diffusers. Selecting a model changes the inference backend; text-only requests do not provide image-conditioning arguments. These models accept 1K/2K canvases and 1:1, 9:16 or 16:9 aspect controls. The editing-only Qwen 2511, Rapid AIO and FireRed models are excluded.

## FLUX.2 Klein 4B custom text encoder

Only FLUX.2 Klein 4B loads the local Qwen3 GGUF text encoder configured by `PHOTO_EDIT_FLUX_KLEIN_4B_TEXT_ENCODER`. The default path is `models/qwen3-4b-alb-q4_0.gguf`. For compatibility with the file currently in this workspace, `qwen3-4b-abl-q4_0.gguf` is also detected automatically.

The GGUF must represent the same Qwen3 4B architecture expected by the downloaded Klein 4B `text_encoder/config.json` (36 layers, hidden size 2560). The pipeline continues to use Klein's bundled tokenizer and uses hidden-state layers 9, 18, and 27. A differently shaped or incompatible GGUF fails during loading instead of silently falling back to the original encoder.

The loader reads Klein's bundled encoder configuration explicitly and passes the external
GGUF as an absolute local file path. The GGUF does not need to be copied into the snapshot's
`text_encoder/` folder. This loading path is shared by FLUX text, edit, combine and swap workflows.

On CUDA, Transformers currently dequantizes this Qwen3 GGUF at load time rather than retaining Q4 blocks. Sequential CPU offload still limits GPU residency, but expect higher system-RAM usage than the file size suggests. The `gguf` package is installed as a project dependency.

## Krea 2 Turbo

See [Krea workflow filenames and hardware assessment](KREA_MODELS.md) for the exact
installed transformer, encoder, VAE and adapter chains for text, reference edit,
Remix and head/body swaps, including effective-default caveats.

`krea/Krea-2-Turbo` is a gated 12-billion-parameter text-to-image model. Accept the Krea 2 Community License on Hugging Face and authenticate with `hf auth login` when needed. All Krea 2 modes, including **Create from text**, run through the project's optional ComfyUI backend so `Krea2_ALWAYS_LOAD_FIRST.safetensors` is always applied first. Install the ComfyUI-format Krea weights, mandatory TextFusion adapter, and optional official Krea Turbo LoRA with `.\scripts\download-models.ps1 -Preset krea-2`. The optional Turbo LoRA is placed in `models/loras/krea2/`; select it explicitly in the UI if wanted. See `docs/KREA_REFERENCES.md`.

For reference-conditioned editing select **Krea reference edit** instead. That mode uses `comfyui-krea2edit` and a built-in image-conditioned API-format workflow, adding the identity-edit LoRA after the mandatory adapter. See `docs/KREA_REFERENCES.md`.

Krea 2 provides Diffusers LoRA methods. Compatible Krea 2 LoRAs are stored in `models/loras/krea2/`; FLUX, Qwen, SDXL, and other architecture LoRAs are not interchangeable. The base model is large for 12 GB VRAM, so LoRAs add load time and system-memory pressure. Start with one adapter.

The Krea 2 Community License requires appropriate content filtering or an equivalent review process. Review the current license and acceptable-use policy before use or deployment.

## Rapid AIO

Rapid AIO is a single ComfyUI Safetensors transformer. Download both components with `python scripts/download_models.py qwen-2511 qwen-2511-aio`. The application keeps the official Qwen tokenizer, text encoder, VAE, and scheduler, then replaces the transformer. A small HTML file renamed `.safetensors` is not a valid checkpoint and is rejected.

## Output policy

For the two six-step ComfyUI Qwen Image 2.1 Turbo profiles and the official
BF16 profile, source editing honors **×2 as native diffusion**, not
post-generation upscaling. The Comfy profiles and unstacked official BF16 profile
also apply this policy to Head/Body swaps; official-extract remains excluded.
The working source canvas is
scaled using the multiplier, preserving aspect within model alignment. For example,
a 3784 × 4852 source requests 1600 × 2048 at ×2; it does not request twice the
literal original pixel dimensions. The preview and generation share the same sizing policy.

×1 retains the existing 1024-side/one-megapixel budget. Qwen 2.1 ×2 has a separate
2048-side/4,194,304-pixel ceiling, configurable with
`PHOTO_EDIT_QWEN21_X2_MAX_OUTPUT_SIDE` and `PHOTO_EDIT_QWEN21_X2_MAX_OUTPUT_PIXELS`.
Reference conditioning remains at its existing resolution. This higher native
diffusion resolution can exhaust 12 GB VRAM, especially with extra adapters; use
×1 if it fails. For the Comfy profiles only, DLSS scaling is independent and can
enlarge the final output again; the official profile rejects DLSS.
Other models' edit limits and text/combine canvas policies are unchanged.

Use 1K, one output, and four steps for distilled models. Larger source images are normalized and requested outputs are reduced to the configured pixel budget. For a larger final asset, generate at 1K and use a dedicated tiled upscaler afterward rather than increasing diffusion dimensions.

## LoRAs

Qwen Image Edit 2511 and Rapid AIO use `models/loras/qwen/`; the two six-step ComfyUI Qwen 2.1 profiles use a separate `models/loras/qwen21/` library for up to five optional `.safetensors` adapters. Official Qwen 2.1 Turbo BF16 rejects optional LoRAs. Klein 4B uses `models/loras/flux/`; Krea 2 uses `models/loras/krea2/`. FireRed Lightning and Qwen 2.1 Viggle r256 are installed separately in `vendor/ComfyUI/models/loras/` and applied automatically. FireRed does not support extras. Additional LoRAs increase system-memory and loading overhead.

Review the license files in every downloaded snapshot. Model terms can change independently of this project.

## BFS swap model support

The two-image BFS workflow uses model-matched LoRAs for Qwen 2511, FLUX.2 Klein 4B, Krea 2 and Qwen 2.1 **Head** and **Body**. Qwen 2.1 Body remains experimental. Rapid AIO and FireRed are not offered because BFS files target different base models. See `docs/SWAP.md`.
Unstacked `qwen-2.1-turbo-official` supports experimental Head/Body with the sole
mandatory pinned BFS adapter. `qwen-2.1-turbo-official-extract` is not a swap model.
