# Qwen Image 2.1 Character Sheet Creator

## Scope and usage

Select the **sixth workflow icon (notebook), Character Sheet Creator**. This is
the isolated `qwen-2.1-sheet` / `qwen-character-sheet` route, not a normal edit
model or a Krea sheet operation. The author-workflow full-precision model choice
is retained at the user's request: BF16 transformer, BF16 vision/text encoder
and BF16 VAE, without mandatory Turbo adapters or FP8/INT8 diffusion substitution.
Auto's separate prompt engine (PE) is INT8.

1. Set up compatible project-managed ComfyUI and nodes, review asset rights, and
   obtain the exact files below. Launch with the existing ComfyUI-enabled launcher.
2. Select the notebook icon and upload **one** clear reference image.
3. Choose **Simple** (default) or **Production**, and **Static** (default) or **Auto**.
4. Set the applicable optional name/description and shared prompt customization.
5. Choose **1**, **3.4** (default) or **6** sheet megapixels. Optionally select
   compatible Qwen 2.1 adapters, then generate one sheet.

The model, steps **25**, CFG **1**, true CFG **1**, multiplier **1**, and count
**1** are fixed. Masks and swaps are rejected; restoration must be **Off**.
The UI ignores negative text and identity preservation; direct backend requests
with nonempty negatives or incompatible settings fail validation. The normal
model picker, Variants and Size triggers are hidden, not removed from Gradio.
Dedicated sheet megapixels replace ordinary size controls. Switching modes
preserves the hidden standard model selection. The unused generic True CFG
slider shows 0; the sheet request explicitly sets true CFG 1.

### Prompt modes

| Selection | Prompt inputs actually used |
| --- | --- |
| Static, either layout | Archived static text; example identity replaced by entity name or neutral reference entity; optional shared customization appended. |
| Production Auto | Entity name, shared customization, reference image, archived system prompt; maximum length 2560. |
| Simple Auto | Character description, shared customization, reference image, archived system prompt; maximum length 2048. |

Character description is unused in Static and Production Auto; entity name is
unused in Simple Auto. Auto-generated text replaces the diffusion prompt instead
of appending the static prompt. Empty optional fields are allowed.

Auto uses `TextGenerate`, `thinking=True`, `use_default_template=True` and
**`sampling_mode="off"` (greedy)**. This is a deliberate adaptation of the
author's sampled-on configuration, **not verified equivalence**. Captioning,
identity, labels and consistency across views are not guaranteed.

## Exact assets and substantial storage

Paths below are relative to configured ComfyUI `models/`, normally
`vendor/ComfyUI/models/`. GB is decimal and GiB binary.

| Asset | Exact bytes | GB / GiB |
| --- | ---: | ---: |
| `diffusion_models/qwen_image_2.1_bf16.safetensors` | 14,230,280,616 | 14.23 / 13.25 |
| `text_encoders/qwen3vl_8b_bf16.safetensors` | 17,534,334,616 | 17.53 / 16.33 |
| `vae/qwen_image_2.1_vae_bf16.safetensors` | 675,509,688 | 0.68 / 0.63 |
| Auto only: `text_encoders/qwen3.5_4b_int8_convrot.safetensors` | 5,588,607,110 | 5.59 / 5.20 |

Mandatory BF16 files total **32,440,124,920 bytes (32.44 GB / 30.21 GiB)**;
with Auto PE, **38,028,732,030 bytes (38.03 GB / 35.42 GiB)**. Allow extra space
for staging, caches, dependencies and outputs. File size is not runtime memory.

First three assets: [Comfy-Org/Qwen-Image-2.1](https://huggingface.co/Comfy-Org/Qwen-Image-2.1),
revision `cb504a4090723e43f17ad01cec0359490e2de613`, remote paths as listed.
PE: [Winnougan/Qwen-3.5-INT8-Convrot-Comfy](https://huggingface.co/Winnougan/Qwen-3.5-INT8-Convrot-Comfy),
revision `e019237a5e495acfa039a1829822ab75dc642e71`, remote basename as listed.

| File | SHA256 |
| --- | --- |
| Transformer | `89f4158d066cc33906a199fca85634f766892dd78f49b6698dabf187ac86c4bc` |
| Vision/text encoder | `68bdc82bc1b66851162ae656225e7e2068166b603db19bd5d5a3b90eb12669a9` |
| VAE | `bb21f7473051e1ac368515dd3f2e15cd44d7a11748ee8823e1ddca3e4876b7c9` |
| Auto PE | `088495aba6219cb8933339e0e433fc326b86787789025c62af4d855864b24455` |

The existing explicit PowerShell preset `qwen-character-sheet` maps to Python
`--comfy-qwen-character-sheet`; the Python model key `qwen-2.1-sheet` is another
entry point. These download **all four** pinned weights and the archive, even
for Static users; Static runtime itself requires only the three BF16 files and
archive. No Static-only preset exists. Runtime never downloads missing assets.
The downloader validates size/hash/Safetensors headers with staged installation;
invalid existing files remain unchanged with an error. Its recommendations are
unchanged, but `--all` includes these large assets. No download was run for this
documentation or regression-test work.

## Pinned archive and plain prompt extraction

Supply the unchanged archive at
`models/workflows/qwen-character-sheet/Character_Sheet.zip`, relative to the
workspace; the actual root is `settings.model_dir` (default
`D:\VSCode\image2image\models`).

- Civitai version **3377047**, file **3265393**.
- Remote filename: `QwenImage21Character_qwenImage21V30.zip`.
- Endpoint: <https://civitai.com/api/download/models/3377047?fileId=3265393>.
- Exact size **33,591 bytes**; SHA256
  `c9a760f237ee6044aa527949cca4ed0986539cbebda94002a7631710e06f2347`.

| Layout | Root JSON member | Subgraph | System/static node IDs |
| --- | --- | --- | --- |
| Production | `Character_Sheet_Production.json` | `33b66f45-313c-482b-9520-f63fa441a34b` | 478 / 592 |
| Simple | `Character_Sheet_Simple.json` | `4c425bd9-fbac-47a7-be3b-5041bb1b186f` | 478 / 600 |

Both named documents are read; only the selected subgraph's unique system and
static nodes supply plain strings. `widgets_values_named.value` takes precedence
over the first `widgets_values` entry. Empty or nonstring values, ambiguous
subgraphs/nodes, changed hashes and duplicate/missing root members are rejected.
The publisher graph is **not executed or extracted**. Publisher static/system
prompt text is read locally by the parser and **not reproduced in these docs**.

Runtime verifies SHA256 before parsing, caps archive reads at 32 MiB and members
at 8 MiB, permits at most 128 entries and rejects encrypted named members and
expansion ratios above 200:1. Downloader validation additionally enforces exact
archive size and rejects unsafe paths/symlinks. Do not edit or repack the archive.

## Source and native canvas adaptations

The source is EXIF-oriented, RGB-converted and aspect-preserved, downscaled to a
**1536 longest-side cap**, never stretched into the output canvas. The same upload
feeds diffusion conditioning and Auto PE. This replaces the author's
`ImageResizeKJv2` **total_pixels (area-based)** preprocessing; it is not identical
for elongated references. KJ resizing is not a runtime dependency. Conditioning
still uses `TextEncodeQwenImage21(resolution=1536)`: its area target can produce
internal dimensions exceeding 1536 on the longest side.

Native 3:2 ResolutionSelector dimensions use **binary MP × 1024 × 1024**, rounded
to multiples of 32:

| Preset | Native output |
| ---: | --- |
| 1 | **1248×832** |
| 3.4 | **2304×1536** |
| 6 | **3072×2048** |

The engine bypasses global edit/text/combine clamps for these canvases. They do
not follow source aspect or the generic 1K/2K controls. Unexpected native output
sizes or counts are rejected, never silently resized. Optional [DLSS](DLSS.md)
is separate post-diffusion enhancement and may change final dimensions.

## Active graph and dependencies

- BF16 loaders with default dtype; optional `LoraLoaderModelOnly` chain first,
  then `ModelAttentionBackend` with `comfy kitchen attention`, then
  `QwenImage21Cache(device="auto", dtype="default")`.
- Reference-conditioned `TextEncodeQwenImage21`, empty native latent, **25-step
  KSampler CFG 1, `res_multistep` / `beta`, denoise 1**. No mandatory Turbo.
- BF16 VAE decode, then `VAEDeGrid`: enabled, auto, limit 0.02,
  skip_when_clean true, grid_gain 10, grid_view `4x zoom`.
- **DeGrid output 0**, not diagnostic output 1, feeds **SaveImage 29**. Only that
  saver is retrieved from history; exactly one image must be returned.

Use compatible current ComfyUI Qwen/cache/attention nodes, `TextGenerate` for
Auto, registered `res_multistep` support (such as compatible RES4LYF), and the
publisher's VAEDeGrid pack; restart after installation. Node setup pins new
DeGrid clones to `5699bc33f71e1be12523fdea105cc9c2abfe1cd0`, and KJNodes to
`d3cfe21625e5170126ce06fbfcfe1d88108688c3`; existing installations are preserved.
Setup does not download sheet weights. Missing/incompatible nodes, registry
inputs or sampler choices fail actual API preflight **before image upload**.
Missing local assets fail even before backend startup. No fallback substitution.

Up to five compatible `.safetensors` adapters share `models/loras/qwen21/`.
Finite weights -2 to 2 pass through in slot order; zero skips. Missing, duplicate,
wrong-path or invalid-weight files are rejected. No BFS/Krea/Turbo adapter is
automatically injected. Sharing the library is not proof that every adapter is
suitable for full-BF16 sheets; leave unrelated adapters unselected.

## Metadata, limits and rights

PNG generation metadata and adjacent `metadata.json` record layout/mode/MP,
archive hash, native/actual/source sizes, source adaptation, BF16 precision,
sampler/scheduler, attention/cache/DeGrid, Auto greedy/thinking/length settings,
and optional LoRA names/weights. Prompt/name/description strings are not stored;
keep private prompt notes separately when reruns matter.

**Actual inference performance, peak memory, visual quality and exact author
workflow parity are not verified.** The 12 GB CUDA threshold is eligibility,
not a fit guarantee, especially at 3.4–6 MP. CPU offload and ample RAM are needed.
For errors, check filenames/roots, unchanged archive hash, compatible nodes and
`logs/comfyui.log`. Trying 1 MP, Static and no optional adapters/DLSS is a
troubleshooting suggestion, not a benchmarked recipe.

The registry identifies **Qwen Research License Agreement: non-commercial
research/evaluation**. Review upstream license/NOTICE and restrictions before
use. Separately review Civitai workflow and embedded prompt rights, PE/source
conversion rights, node licenses and optional LoRA terms. Availability, ownership
or hashes do not grant redistribution or commercial permission; the project
license does not override upstream restrictions. Remote rights/availability and
publisher performance claims were not independently validated in this pass.