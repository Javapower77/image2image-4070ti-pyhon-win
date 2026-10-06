# Krea All2Real

Select **Krea reference edit → All2Real** to convert one source image toward a
photorealistic appearance. This is a separate `krea-all2real` operation, not
identity editing, a head/body swap or Composition remix. It does not load the
identity-edit or Remix adapter automatically.

## Choose the All2Real asset set explicitly

All2Real requires the transformer loader alias, encoder, Wan upscale VAE and
skin-detail model referenced by `workflows/Krea2-all2real.json`, plus the project's
mandatory first adapter and MoreReal library file. These are **not** the standard
FP8-transformer/Qwen-VAE Krea preset. Use All2Real only if you explicitly choose
this separate asset set and accept its unmeasured memory costs. Otherwise select
Reference edit or Composition remix. Runtime still **never downloads** these assets
or silently substitutes/falls back to another transformer or VAE.

**Changed decision: user-approved transformer substitution.** The exact historical
checkpoint `krea2_turbo_int8_convrot-b19a4f0be264.safetensors` was not found. The
explicit downloader now obtains the current Comfy-Org/Krea-2 INT8 Turbo checkpoint
`diffusion_models/krea2_turbo_int8_convrot.safetensors` at the pinned revision below
and saves it under that historical loader alias. This is **NOT proof of identical
bytes to the upstream original**; it supersedes the earlier original-exact
requirement for this checkpoint only, not the runtime's fail-closed behavior.
Matching filenames alone do not verify content or upstream workflow parity.

Paths below are relative to the project root, using the default ComfyUI/library
directories:

| Required file | Location |
| --- | --- |
| `krea2_turbo_int8_convrot-b19a4f0be264.safetensors` | `vendor/ComfyUI/models/diffusion_models/` |
| `qwen3vl_4b_fp8_scaled.safetensors` | `vendor/ComfyUI/models/text_encoders/` |
| `Wan2.1_VAE_upscale2x_imageonly_real_v1.safetensors` | `vendor/ComfyUI/models/vae/` |
| `1x-ITF-SkinDiffDetail-Lite-v1.pth` | `vendor/ComfyUI/models/upscale_models/` |
| `Krea2_ALWAYS_LOAD_FIRST.safetensors` | `vendor/ComfyUI/models/loras/` |
| `Krea2-MoreReal.safetensors` | `models/loras/krea2/` |
| Project custom-node module | `vendor/ComfyUI/custom_nodes/photo_edit_all2real.py` |

### Explicit three-asset download

Install embedded ComfyUI first with `.\scripts\setup-comfy.ps1`, then run either
command from the repository root:

```powershell
.\scripts\download-models.ps1 -Preset krea-originals
```

Python equivalent using the project environment:

```powershell
.\.venv\Scripts\python.exe scripts\download_models.py --comfy-krea-originals
```

Used alone, either option downloads **only these three pinned assets**, totaling
14,020,470,393 bytes (**~13.06 GiB**); allow additional staging/cache space. It does
not download MoreReal, the shared encoder, the mandatory first adapter, or the
recommended model set. Supply those other prerequisites separately; the standard
`--comfy-krea` option supplies shared assets/mandatory first but is a separate,
larger download. The `krea-originals` name is a preset label, not an authenticity
claim. No actual model download was performed for this documentation update.

| Asset / local destination under `vendor/ComfyUI/models/` | Pinned Hugging Face source / remote file | Revision | Bytes |
| --- | --- | --- | ---: |
| Approved INT8 alias: `diffusion_models/krea2_turbo_int8_convrot-b19a4f0be264.safetensors` | [Comfy-Org/Krea-2](https://huggingface.co/Comfy-Org/Krea-2), `diffusion_models/krea2_turbo_int8_convrot.safetensors` | `6b1d7191d84d5ded74d83a1a98211dad0ac8ae25` | 13,492,686,496 |
| Exact VAE: `vae/Wan2.1_VAE_upscale2x_imageonly_real_v1.safetensors` | [spacepxl/Wan2.1-VAE-upscale2x](https://huggingface.co/spacepxl/Wan2.1-VAE-upscale2x), `Wan2.1_VAE_upscale2x_imageonly_real_v1.safetensors` | `384fb7de682e60bd54b59d6eea810ca9d9993497` | 507,684,560 |
| Exact skin filename: `upscale_models/1x-ITF-SkinDiffDetail-Lite-v1.pth` | [timothy692/1x-ITF-SkinDiffDetail-Lite-v1](https://huggingface.co/timothy692/1x-ITF-SkinDiffDetail-Lite-v1), `1x-ITF-SkinDiffDetail-Lite-v1.pth` | `c5b4f4c21c62eb0819c9e3fd0f2ea15c343c3754` | 20,099,337 |

Pinned SHA256 values:

- INT8: `8e4eeda70dd5037ab1ba2bef6b417f9f901e26093117cf397f741fc1fdaaf3f1`
- VAE: `2413554bbec24215185662d009893cf4666b8e777efece2d895e03e1a6b63e06`
- Skin: `94d368b633614958f84f335b129fd85abd30200e8fbc575b859ba6762116222b`

The skin file is from the timothy692 mirror; the same SHA256 was verified in the
gemasai mirror. License metadata is unavailable (`None`), **not a license grant**.
Review source provenance, access terms and licenses before downloading or using
any of these files. A matching mirror hash does not establish permission to use it.

Downloads are staged on the destination filesystem, verified for exact byte size
and SHA256, and atomically installed only after validation. Safetensors headers
are validated without materializing tensors. The downloader verifies the `.pth`
as bytes only and **never pickle-loads it**; this is not a runtime safety guarantee.
Existing files are also verified: a mismatch causes an error and leaves the file
unchanged. Explicitly move/remove it and rerun if replacement is intended. Failed
download verification never installs the staged file.

The custom-node module is installed from `scripts/comfy_all2real_nodes.py` by the ComfyUI
setup script. Install the Ostris extension `ComfyUI-Krea2-Ostris-Edit` and restart
ComfyUI after node installation. Compatible core nodes must register `er_sde`,
`kl_optimal`, `simple`, FluxKontext scaling/reference conditioning and the exact
loader filenames. The application checks assets and `/object_info` before image
upload or queueing; identity-edit nodes cannot replace missing Ostris nodes.

MoreReal uses the project library filename `Krea2-MoreReal.safetensors`, not the
editor export's `Krea2-temp` training-checkpoint name. Supply the intended weight
under that library filename; renaming an unrelated adapter does not establish
compatibility. Queued library loaders use **qualified `krea2/` names**, including
optional adapters, to avoid binding a same-named vendor LoRA. ComfyUI must register
these qualified choices; Windows backslash choices are accepted. An unqualified
basename alone is not a safe fallback.

## Controls and graph

- Upload **Picture 1 only**; the second-reference control is hidden.
- Prompt defaults to `photorealistic`; a supplied prompt is sent to the Ostris
  vision/text encoder. Refinement removes reference metadata from that same
  conditioning, preserving embeddings rather than re-encoding text.
- Default **11 total steps**, CFG **1**, one output, no negative prompt or mask.
  Total steps may be 5–40. The editor's exposed integer **9 is a seed**, not a
  nine-step schedule. A negative UI seed requests a concrete random engine seed.
- Both samplers use that concrete request seed; latent noise injection uses
  **seed + 1**, standard deviation **0.3**, without wrapping. Direct adapter
  requests require an integer seed from 0 through `2**64 - 2`.
- Default first pass: `er_sde`, `kl_optimal`, start **1**, end **8**, total **11**.
  Default second pass: `er_sde`, `simple`, start **9**, end **11**, total **11**.
  With custom total steps N, the first ends at N−3 and the second starts at N−2.
  Both enable sampler noise and disable leftover-noise return.
- Both passes share **INT8 transformer → mandatory first LoRA → Ostris patch
  (KV cache enabled) → MoreReal → selected optional LoRAs in slot order**.
  MoreReal defaults to **0.9**. Selecting its library filename at a nonzero
  optional weight adjusts its existing node; zero leaves the 0.9 default.
  Up to five optional Krea library slots support finite −2 to 2 weights; zero
  skips a slot. The mandatory first weight uses its own positive control.
- The source drives VAE encoding and image-aware conditioning. Between passes,
  the graph decodes with the Wan VAE, re-encodes the image and injects latent noise.
  The second pass decodes again, then applies skin detail before SaveImage **29**.
  Only that node's single generated image is returned, never a source/preview.
- The UI fixes size multiplier/count to 1, disables identity preservation and
  composition flags, and forces face restoration Off. Optional DLSS remains a
  separate post-decode/skin-detail stage, not a diffusion-size control.

## Upstream adaptation, not verified parity

The upstream skin-detail subgraph uses **four tiles**, the required 1×
SkinDiffDetail model and concatenation right/right/down. The project implementation
splits into TL/TR/BL/BR quadrants, processes each through ComfyUI's native model
upscaler, then rejoins them, preserving batches and odd edges.

`StudioAll2RealRemoveReferences`, `StudioAll2RealLatentNoise` and
`StudioAll2RealSkinDetail` are **project-owned replacements**, not the upstream
NO8D/WAS node implementations. Reference metadata is copied with embeddings
retained; noise uses a local seeded CPU float32 generator before conversion to
the latent dtype/device. These deterministic and shape-preserving contracts can
be tested without loading models, but **NO8D/WAS/VAEUtils parity is NOT exactly
verified**. Do not infer matching visual output, tile/seam behavior or noise
distribution details from graph similarity alone.

`StudioAll2RealVAEDecode` now unpacks both intermediate and final Wan outputs.
The installed checkpoint encodes RGB3 but decodes 12 packed subpixel channels.
The wrapper applies a 2× pixel shuffle, matching the documented VAEUtils
`upscale=-1` conversion, and corrects packed-channel allocation for automatic
tiled fallback without mutating the shared VAE. Already-normalized pixels are
not normalized again. Whole-extension/end-to-end parity remains unverified.

The initial implementation omitted this conversion: intermediate encode silently
kept only three packed channels and final skin detail failed with “expected 3
channels, but got 12”. Both boundaries are corrected; restart ComfyUI to load
the updated project nodes. Do not truncate packed output to RGB channels.

## Dimensions, memory and verification limits

The uploaded source is capped at **1024** on its longest side. The graph then
rescales to **1 megapixel**, blurs, and applies **FluxKontext aspect buckets**.
That source cap is **not** a diffusion/output cap: each packed-output decode
doubles RGB width and height. After intermediate decode/re-encode, the second
diffusion latent has four times the first pass's spatial area; two decodes can
produce **4× the source-bucket dimensions** before optional DLSS. This preserves
the supplied workflow rather than silently resizing the intermediate image.
Metadata records the source budget, uploaded size and actual returned sizes
separately. Optional DLSS uses the decoded skin-detail image, not the source size.

Peak VRAM/RAM, native upscale behavior, visual quality and end-to-end upstream
parity have not been verified by the mocked tests. In particular, decode/re-encode
and native upscale have **unknown memory impact** on a 12 GB GPU; `--lowvram` and
the 1024 source cap do not guarantee a successful run. Start without optional
LoRAs or DLSS and monitor local logs if you elect to run your supplied weights.
See [Krea asset inventory](KREA_MODELS.md) and [troubleshooting](TROUBLESHOOTING.md).
