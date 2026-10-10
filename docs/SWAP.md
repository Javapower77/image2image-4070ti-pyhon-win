# Head and body swap workflow

Use **Swap head / body** with two ordered images: **Picture 1** is the target body/source scene; **Picture 2** is the replacement face/head or person reference. The output follows Picture 1's aspect ratio and the configured resolution budget. The workflow retains prompts, size multiplier, seed, local outputs, progress, and optional GFPGAN post-processing. Qwen 2.1 Head/Body support includes both six-step ComfyUI profiles and the unstacked eight-step official BF16 profile; their adapter chains and schedules differ. Only use images you have permission to edit; do not publish misleading impersonations.

## Supported combinations

| Base model | Swap | BFS LoRA | Backend |
| --- | --- | --- | --- |
| Qwen Image Edit 2511 | Head | `bfs_head_v5_2511_merged_version_rank_16_fp16.safetensors` | Existing local Diffusers pipeline |
| FLUX.2 Klein 4B | Head | `bfs_head_v1_flux-klein_4b.safetensors` | Existing local Diffusers pipeline (including configured Qwen3 GGUF text encoder) |
| Krea 2 Turbo | Head | `bfs_head_swap_v1.1_krea2.safetensors` | Project-managed ComfyUI with Krea2Edit custom nodes |
| Krea 2 Turbo | Body | `bfs_body_swap_v1_krea2.safetensors` | Project-managed ComfyUI with Krea2Edit custom nodes |
| Qwen Image 2.1 + Viggle Turbo r256 | Head / Body (experimental) | Canonical Qwen 2.1 BFS below, after mandatory unmerged Viggle r256 | Project-managed ComfyUI, six steps |
| Qwen Image 2.1 + Turbo r128 (Civitai) | Head / Body (experimental) | Canonical Qwen 2.1 BFS below, after mandatory r128 | Project-managed ComfyUI, six steps / `res_2s_ode` |
| Qwen Image 2.1 Turbo (Official BF16) | Head / Body (experimental) | Canonical Qwen 2.1 BFS below as the **sole mandatory adapter** | Local Diffusers, eight saved-sigma steps |

Rapid AIO and FireRed are not offered because these BFS LoRAs target different architectures. `qwen-2.1-turbo-official-extract` remains excluded: official BFS does not stack the extracted Turbo adapter. LoRA weights are not interchangeable. Krea body-swap and Qwen 2.1 Head/Body swaps are experimental; the Viggle release explicitly warns of possible ghosting and identity drift on complicated swaps.

For only the two pinned Qwen 2.1 BFS assets, use `.\scripts\download-models.ps1 -Preset qwen-2.1-bfs`, equivalent to `.\.venv\Scripts\python.exe scripts\download_models.py --qwen21-bfs`. This preset downloads no base model. The broader `--bfs-swap` now includes these two files **and** the existing four Qwen 2511/FLUX/Krea BFS files. Base models are downloaded separately using their normal presets; additional compatible LoRAs remain available on the ComfyUI profiles, **not official BF16**.

## Canonical Qwen 2.1 BFS assets and migration

All three supported Qwen 2.1 swap profiles use these files under
`models/loras/qwen21/` (`PHOTO_EDIT_LORA_DIR` changes the LoRA root):

| Swap | Canonical filename | Published version | Exact bytes | Civitai version / file | Legacy alias |
| --- | --- | --- | ---: | --- | --- |
| Head | `bfs_head_v1.1_qwen_2.1.safetensors` | 1.1 | 260096144 | 3356102 / 3248321 | `Qwen21-BFS_Head_v1.1.safetensors` |
| Body | `bfs_body_swap_v1.0_qwen_2.1.safetensors` | **1.0** | 209753576 | 3363725 / 3251548 | `Qwen21-BFS_Body_v1.1.safetensors` |

- Head SHA256: `d1d748d5601077f3b6d05766f6823510e901970d916afa404a97e85dc92fa88e`.
- Body SHA256: `7dc0a53aba4dbc70c204f498936a619d226559da11011f748c389189af46b664`.
- Pinned delivery: <https://civitai.com/api/download/models/3356102?fileId=3248321> and <https://civitai.com/api/download/models/3363725?fileId=3251548>.

These are the **same verified bytes as the old aliases, not new weights**.
The Body alias's `v1.1` is historical naming, not its published version.
If the canonical destination is absent, the downloader verifies the sibling
legacy file's size, SHA256 and nonempty Safetensors header, copies it to local
staging, reverifies it, then atomically installs the canonical name. The legacy
original is kept. Invalid legacy files are kept and a pinned download is attempted;
invalid existing canonical files fail closed and are not overwritten. New downloads
are bounded, staged and verified before installation; failed staging is removed.
Runtime expects canonical names, not legacy aliases. No bytes were downloaded or
rehash-verified by this documentation update; the values above are implementation pins.

## Official BF16 Head/Body BFS (experimental)

Select **`qwen-2.1-turbo-official`**, not `qwen-2.1-turbo-official-extract`.
Install the full official snapshot/dependencies via
[SETUP](SETUP.md#official-qwen-21-turbo-bf16-opt-in), then the separate
`qwen-2.1-bfs` preset. ComfyUI is not required for this route.

The request must contain exactly two images in body/source-then-reference order,
Head or Body, and exactly one matching canonical BFS LoRA named `bfs_swap` at
its configured profile path. Its finite positive weight is **0.1–1.5 inclusive**
(UI default 1.0), separate from optional LoRA controls. Eight steps, CFG **1**,
True CFG **1** and an empty negative prompt are mandatory. The saved publisher
sigmas and causal KV cache remain authoritative; no six-step Turbo, extracted
adapter, optional stack or DLSS is permitted. Direct requests must use
`dlss=None`, not a disabled settings dictionary.

**Validation order and gap:** on a cold swap load, exact byte size/SHA256 are
checked **before checkpoint allocation**. Safetensors A/B structure, tensor
normalization and real target-shape checks currently occur **after the full
checkpoint is allocated**, but before BFS installation and offload. A shape
failure can therefore still incur BF16 checkpoint RAM/allocation costs; this is
not complete pre-allocation compatibility validation.

The CPU loader requires the pinned A/B-only structure (176 Head pairs / 136 Body
pairs), rejects direct deltas, and splits fused image-MLP `gate_up` B rows
**gate first → `gate_layer`, second half → `proj`**, reusing A. Missing alpha
means actual-rank intrinsic scale **1**; the requested BFS weight is applied
separately, once, by `set_adapters`. All materialized linear targets and shapes
are validated before installation; no half is discarded. Switching kind/weight
or leaving swap removes hooks and the old adapter, then reinstates memory
configuration; an unchanged path/weight signature skips reinstalling/rehashing.
Installation/change failures dispose of the pipeline rather than claiming rollback.

PNG `generation` metadata and `metadata.json` save the checkpoint `sample_sigmas`,
`scheduler_source`, KV-cache flag, actual output sizes, swap kind, ordered sources,
fixed `swap_trigger`, and `mandatory_adapter` filename/version/SHA256/size/request
weight. Body is recorded as **1.0**. Civitai IDs, snapshot revision, explicit
precision and actual scheduler timesteps are not recorded; keep those separately.
User prompt text and source images are not stored, but the fixed training trigger is.

Compatibility is **experimental**: no live official BFS inference, visual quality,
identity fidelity, performance or VRAM fit has been confirmed. Earlier user-reported
unstacked official generation is not verification of this BFS path. Native ×2 uses
the shared Qwen 2.1 2048-side/4,194,304-pixel budget; ×1/×3 retain the global cap.
CPU normalization/offload does not mean CPU-only inference support or a 12 GB fit guarantee.

## Qwen 2511 BFS mixed-format compatibility

Qwen Image Edit 2511 supports **Head only; no Body profile exists**. Its
`bfs_head_v5_2511_merged_version_rank_16_fp16.safetensors` is not a matrix-only
adapter: the measured 2026-10-10 header has **846 down/up pairs, 846 direct bias
deltas and 241 direct weight deltas**, with no alpha tensors. See
[the pinned measured header and cache provenance](MODELS.md#qwen-image-edit-2511-mixed-format-loras).
The downloader currently does not enforce that provenance pin.

Missing alpha is interpreted as the actual pair rank (intrinsic scale 1).
Explicit alpha/rank is preserved; BFS/request strength scales both the active
matrix contribution and additive direct deltas exactly once. All targets are
validated before mutation, with no silent dropping of bias/weight updates.
Changing strength or selected files restores pristine parameters before applying
the new aggregate; deselecting restores originals and repeating an unchanged
selection cannot compound deltas. Hooks are detached before parameter access
and memory/offload settings are restored afterward. These are synthetic CPU
regression guarantees, not evidence of full-checkpoint inference or exact swaps.

## Krea editing backend: required setup

The installed Diffusers `Krea2Pipeline` supports **text-to-image only**; it cannot consume two reference images. The Krea swap workflows use `Krea2EditGroundedEncode` and `Krea2EditModelPatch` from `comfyui-krea2edit`. To use Krea Head/Body swap:

1. Run `.\scripts\setup-comfy.ps1` to install ComfyUI and Krea2Edit nodes inside this project's `.venv` and ignored `vendor/ComfyUI/` directory.
2. Run `python scripts/download_models.py --comfy-krea` to obtain matching ComfyUI-format Krea Turbo/CLIP/VAE weights and the identity-edit LoRA (not supplied by the Diffusers snapshot). Run `python scripts/download_models.py --bfs-swap` to obtain BFS LoRAs. The embedded process searches `models/loras/krea2/` directly without duplicate files.
3. Launch `.\scripts\run.ps1 -ComfyUI`. The app starts ComfyUI on demand, locally at `127.0.0.1:8188`. The built-in 2-image Krea API graph is used for Head/Body swap and selects the correct BFS LoRA. No manual workflow export is needed. Advanced users can override `PHOTO_EDIT_COMFY_HEAD_WORKFLOW` or `PHOTO_EDIT_COMFY_BODY_WORKFLOW` with a tested API-format graph.

The bridge uploads the two reference images to the loopback ComfyUI API, substitutes prompt, seed, steps, CFG, BFS weight, and output prefix, waits for the saved output, then saves a copy in this app's `outputs/`. For Krea swaps, the mandatory `Krea2_ALWAYS_LOAD_FIRST.safetensors` (with its dedicated UI weight) is loaded **first**, then the BFS LoRA, then any additional selected Krea LoRAs at their chosen weights. Qwen and Flux swap order is unchanged. The bundled Krea graph uses the upstream Krea2Edit node paths and a 1K output. Customize ref_boost, fit mode, sampler or masking with a tested API graph if necessary. For Krea swap, leave the negative prompt empty; configure it in a custom workflow if needed. Gradio's progress remains indeterminate while ComfyUI works because this bridge does not receive its internal sampler steps.

## Limitations

- The six-step ComfyUI Qwen 2.1 profiles require their INT8 assets and matching mandatory Turbo adapter: `qwen-2.1` for Viggle r256/custom node or `qwen-2.1-r128` for r128/compatible `res_2s_ode` extension. Keep six steps, CFG 1 and an empty negative prompt. Canonical BFS loads after that Turbo and before up to four optional Qwen 2.1 adapters. Selecting the same BFS file in an optional slot does not load it twice. The separate official route above uses eight steps and no optional stack. No pixel-exact swap is guaranteed.
- These are **head** (Qwen/Flux/Krea) or **person/body** (Krea/Qwen 2.1) replacements, not pixel-perfect face-only swaps. Expression, hair, pose, or background can drift. The stored Body training trigger includes preservation language for both images; it is not a guarantee of scene/pose fidelity.
- No segmentation, masked inpainting, or automatic crop/stitch is claimed. The original edit workflow's optional mask is still post-generation compositing; the swap section does not claim native masks.
- Diffusers and ComfyUI use different pipelines. Exact outputs from the supplied ComfyUI workflow are not guaranteed by the Qwen/Flux Diffusers approximations.
- Krea 2 is gated and subject to the Krea 2 Community License and its filtering/review requirements. Review upstream LoRA terms too.

Sources: [BFS model files/workflows](https://huggingface.co/Alissonerdx/BFS-Best-Face-Swap/tree/main), [Krea body swap](https://civitaiarchive.com/models/2027766?modelVersionId=3188610), [Flux Klein 4B head swap](https://civitaiarchive.com/models/2027766?modelVersionId=2609209), [Qwen 2511 head swap](https://civitaiarchive.com/models/2027766?modelVersionId=2556739).
