# Setup

## Windows 11 (primary target)

Use Python 3.11, a current NVIDIA driver, and PowerShell. Run `.\scripts\setup.ps1 -Download recommended`, then launch `.\scripts\run.ps1`. To keep installation and model downloads separate, run `.\scripts\setup.ps1` followed by `.\scripts\download-models.ps1 -Preset recommended`. Full instructions are in `docs/WINDOWS.md`.

The setup script installs the CUDA 12.8 PyTorch wheel. Do not install a separate CUDA Toolkit solely for this project. Verify the result with `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name())"` from the activated environment.

## Linux

Run `bash scripts/setup.sh`, activate with `source .venv/bin/activate`, download models, and start with `bash scripts/run.sh`. Linux now uses the same 12 GB-safe defaults. Override them in `.env` only when the GPU has more memory.

## Storage and authentication

Snapshots live in `models/repos`, Hugging Face cache data in `models/huggingface`, LoRAs in `models/loras`, restorer weights in `models/restorers`, and results in `outputs`.

For gated repositories, accept the repository terms and run `hf auth login`. Never commit tokens. An environment-provided `HF_TOKEN` is also supported.

## Recommended downloads

- `.\scripts\download-models.ps1 -Preset qwen-aio` for Rapid AIO.
- `.\scripts\setup-comfy.ps1` then `.\scripts\download-models.ps1 -Preset qwen-2.1` for Qwen Image 2.1 INT8 with Viggle Turbo r256; launch with `.\scripts\run.ps1 -ComfyUI` (Qwen Research License; non-commercial).
- `.\scripts\download-models.ps1 -Preset flux-4b` for Klein 4B.
- `.\scripts\download-models.ps1 -Preset krea-2` for gated Krea 2 Turbo text-to-image.
- `.\scripts\setup-comfy.ps1` then `.\scripts\download-models.ps1 -Preset firered` for FireRed GGUF Q4_K_M + automatic Lightning 8-step editing; launch with `.\scripts\run.ps1 -ComfyUI`.
- Add `--restorers` to download GFPGAN weights after installing the restoration extra.

Downloading every registered model is not recommended for the 12 GB target. FireRed requires CPU offload because its GGUF checkpoint alone exceeds 12 GB.

## Official Qwen 2.1 Turbo BF16 (opt-in)

This separate eight-step Diffusers profile is **not** included in `recommended`
and does not need ComfyUI. Use Python 3.11 in the project `.venv`. Before opting
in, review compatibility with other installed models: the official adapter
requires Diffusers [PR #14950](https://github.com/huggingface/diffusers/pull/14950),
merge commit `da1d3829cf08d4f329b526d89e17cc035c049d8d` or a compatible descendant,
and Transformers `>=5.17,<6`. A `0.41.0.dev0` label alone is insufficient.
The exact pipeline signatures and saved-config requirements are listed in
[Models](MODELS.md#qwen-image-21-turbo-official-bf16).

An explicit upgrade instruction, **not executed by this documentation update**,
is `.\.venv\Scripts\python.exe -m pip install --force-reinstall --no-deps "diffusers @ git+https://github.com/huggingface/diffusers.git@da1d3829cf08d4f329b526d89e17cc035c049d8d"`.
Install the required Transformers version separately with
`.\.venv\Scripts\python.exe -m pip install --upgrade "transformers>=5.17,<6"`.
`--upgrade` alone can leave an older Git commit installed when both commits
report `0.41.0.dev0`. Force-reinstall changes the source revision; `--no-deps`
avoids replacing unrelated dependencies. Restart the studio after installation.
A reviewed compatible descendant may be used instead.
Installing/upgrading can affect other pipelines; the normal setup's upstream-Git
dependency does not prove that an existing venv has the needed APIs.
Static inspection on 2026-10-10 found installed Diffusers commit
`d77d53044518ed45583ce3ff680f9f87f816aeb3` without `QwenImage21Pipeline`.
On 2026-10-10, force-reinstallation corrected that environment to commit
`da1d3829cf08d4f329b526d89e17cc035c049d8d`. The pipeline import and its
`sample_sigmas`/`use_kv_cache` signatures passed verification with Transformers
5.19.0. No model inference was performed in that verification.
The user subsequently reports successful testing of the **unstacked official
Turbo profile** with this repaired setup. This report is not an independent
live-inference result from the current tests/docs update and does not establish
that the experimental official-plus-extracted stack works or fits in 12 GB.

For the **full publisher snapshot**, use the existing standalone downloader:
`.\scripts\download-models.ps1 -Preset qwen-2.1-official`, or equivalently
`.\.venv\Scripts\python.exe scripts\download_models.py qwen-2.1-turbo-official`.
The preset maps to `Qwen/Qwen-Image-2.1-Turbo`, downloaded with unrestricted
`snapshot_download` to `models/repos/Qwen--Qwen-Image-2.1-Turbo` by default
(`PHOTO_EDIT_MODEL_DIR` can change the root). No file allow/ignore filters or
revision pin are supplied. Post-download validation requires the exact eight
saved `sample_sigmas` and `causal_condition: true`; configs are not rewritten.
Retain the resolved publisher revision/configs separately for reproducibility.

For the official base, `qwen-2.1` and `qwen-2.1-r128` are not substitutes: they
fetch different, targeted ComfyUI INT8/six-step assets. Do not add
`-ComfyUIQwen21` or `-TurboLora` for the official profile. For Head/Body swaps,
download the separate `qwen-2.1-bfs` preset below; `-BfsSwap` is also supported
but fetches the broader six-file BFS set. The setup script's `-Download` choices
do not include `qwen-2.1-official`; use the standalone downloader afterward.
Authenticate/accept publisher terms if required, then launch with
`.\scripts\run.ps1` and select
**Qwen Image 2.1 Turbo Official (BF16 · 12 GB offload unverified)**.

The default official snapshot directory was absent during source inspection.
No download was performed and live snapshot contents/license were not checked.
Allow substantial BF16 storage, cache/staging space and system RAM; offload is
not a guaranteed 12 GB fit, especially with KV cache, multiple references or 2K.
Unstacked official BF16 supports experimental Head/Body with one mandatory BFS;
optional LoRAs and DLSS remain unsupported. Official-extract still rejects swaps.

### Official Head/Body BFS assets (separate opt-in)

After the official snapshot, run `.\scripts\download-models.ps1 -Preset qwen-2.1-bfs`,
equivalent to `.\.venv\Scripts\python.exe scripts\download_models.py --qwen21-bfs`.
This downloads only the two pinned BFS assets, no checkpoint or extracted Turbo
LoRA. It is not in `recommended` or the setup script's `-Download` choices.
ComfyUI is unnecessary for official BF16 BFS; launch `.\scripts\run.ps1` and
choose the unstacked official model in **Swap head / body**.

Default destination: `models/loras/qwen21/` (`PHOTO_EDIT_LORA_DIR` changes the root).

| Swap | Canonical filename | Published version | Exact bytes | Civitai version / file |
| --- | --- | --- | ---: | --- |
| Head | `bfs_head_v1.1_qwen_2.1.safetensors` | 1.1 | 260096144 | 3356102 / 3248321 |
| Body | `bfs_body_swap_v1.0_qwen_2.1.safetensors` | **1.0** | 209753576 | 3363725 / 3251548 |

Head SHA256: `d1d748d5601077f3b6d05766f6823510e901970d916afa404a97e85dc92fa88e`.
Body SHA256: `7dc0a53aba4dbc70c204f498936a619d226559da11011f748c389189af46b664`.
Delivery is pinned to
<https://civitai.com/api/download/models/3356102?fileId=3248321> and
<https://civitai.com/api/download/models/3363725?fileId=3251548>.

Legacy aliases `Qwen21-BFS_Head_v1.1.safetensors` and
`Qwen21-BFS_Body_v1.1.safetensors` contain the **same verified bytes, not new
weights**; Body's published version is 1.0 despite the alias. With a missing
canonical destination, the downloader checks legacy size/hash/header, stages a
copy, reverifies it and atomically installs the canonical filename **without
removing the original**. Invalid legacy files are left unchanged and a pinned
download is attempted. Invalid existing canonical files fail closed and remain
unchanged; deliberately move/remove them before retrying. Valid canonical files
are verified and skipped; downloads are bounded and staged before installation.

Exactly two images (body/source then reference), eight steps, CFG/True CFG 1,
empty negative prompt and sole BFS at finite weight **0.1–1.5** are required.
No optional stack, extracted Turbo or DLSS (including disabled dictionaries).
Runtime size/hash checks precede cold checkpoint allocation, but CPU A/B
normalization and real target-shape validation currently occur **after allocation,
before installation/offload**; malformed shapes can still incur full BF16 RAM costs.
Intrinsic scale is 1, with request weight applied separately; gate/proj splitting
and saved BFS/sigma metadata are detailed in [SWAP](SWAP.md#official-bf16-headbody-bfs-experimental).
Compatibility, live inference, quality, speed and VRAM fit remain unconfirmed.
This documentation update performed no downloads or inference.

## Official Turbo + extracted LoRA (opt-in)

For the explicitly chosen **official Turbo + extracted LoRA**, use
`.\scripts\download-models.ps1 -Preset qwen-2.1-official-extract`, or
`.\.venv\Scripts\python.exe scripts\download_models.py qwen-2.1-turbo-official-extract`.
The preset maps only to the dedicated extract model key. It first downloads the
**full `Qwen/Qwen-Image-2.1-Turbo` snapshot**, validates its saved configs without
rewriting them, then downloads the mandatory adapter. It does **not** download
the original non-Turbo base. The unstacked `qwen-2.1-official` preset remains
unchanged; neither opt-in profile replaces recommended defaults.

The mandatory file goes to
`models/loras/qwen21-official/qwen_image_2.1_turbo_lora_avg_rank_178_bf16.safetensors`
by default (`PHOTO_EDIT_LORA_DIR` changes the LoRA root). Exact Civitai version
**3394831** / file **3284648**, size **913314512 bytes**, SHA256
`208DD43250E1E01467BA190572AE2E7107A7870EC1FF026BC65F791FA1E80E95` are pinned.
Delivery starts at `https://civitai.com/api/download/models/3394831?fileId=3284648`;
approved HTTPS Civitai/storage redirects do not receive bearer credentials
across origins. No mutable metadata, alternate file or mirror is substituted.

An existing file is skipped only after size/hash/nonempty Safetensors-header
validation. A corrupt existing file fails closed and remains unchanged: move
or remove it deliberately before rerunning. New bytes are staged on the same
filesystem, bounded, validated and atomically installed; failures remove the
temporary `.part` file and preserve any existing destination. Skipping the
LoRA does not mean the CLI skips its preceding Hugging Face snapshot operation.

The same official API/dependency requirements apply, plus callable pipeline
`load_lora_weights` and `set_adapters`. No install, download or inference was
performed here. This is **experimental and NOT recommended stacking on an
already-trained/distilled checkpoint**; no version-specific Civitai advice or
optimal weight/schedule is established. Weight 1.0 and the saved official
schedule are fixed experimental choices, with no guaranteed 12 GB fit.
The mandatory file contains LoRA pairs **and additive normalization deltas**;
the mixed helper validates/converts them before offload rather than loading the
file wholesale through the generic LoRA API. See
[mixed-format loading details](MODELS.md#qwen-image-21-turbo-official--extracted-lora-experimental).

## Network and privacy

The default bind address is `127.0.0.1`. If binding to `0.0.0.0`, configure `PHOTO_EDIT_AUTH_USER` and `PHOTO_EDIT_AUTH_PASSWORD`, restrict the firewall, and use TLS at a reverse proxy. Do not enable Gradio share links for private images.

After downloads complete, set `HF_HUB_OFFLINE=1` for strict offline launches.

## Optional BFS head/body swap

Download compatible BFS LoRAs with `python scripts/download_models.py --bfs-swap`: the four existing Qwen 2511/FLUX/Krea files plus the two pinned Qwen 2.1 files. For Qwen 2.1 only, use `--qwen21-bfs` / PowerShell preset `qwen-2.1-bfs`, as above. Krea Head/Body swap uses the optional **project-managed ComfyUI** backend and built-in Krea2Edit graph. Install it with `.\scripts\setup-comfy.ps1`, download the Krea ComfyUI checkpoints with `python scripts/download_models.py --comfy-krea`, and launch with `.\scripts\run.ps1 -ComfyUI`. See `docs/SWAP.md` and `docs/KREA_REFERENCES.md`.

## Optional Krea reference editing

The Krea 2 Diffusers model cannot accept reference images. For a source image plus optional second reference use **Krea reference edit** and the optional project-managed ComfyUI Krea2Edit backend described in `docs/KREA_REFERENCES.md`.

## Optional Krea All2Real assets

After `.\scripts\setup-comfy.ps1`, explicitly run
`.\scripts\download-models.ps1 -Preset krea-originals`, or, in the activated
environment, `python scripts/download_models.py --comfy-krea-originals`.
Used alone, this downloads only three pinned assets (~13.06 GiB): the approved
current Comfy-Org INT8 Turbo alias, exact spacepxl Wan upscale VAE, and exact skin
filename from the timothy692 mirror (same SHA256 as gemasai). MoreReal, the shared
encoder and mandatory first adapter are **not included**; supply them separately.

**Changed decision:** the historical suffixed transformer was not found; the user
approved the current checkpoint saved as `krea2_turbo_int8_convrot-b19a4f0be264.safetensors`.
This is not proof of identical upstream-original bytes or visual parity. Runtime
still never downloads or silently substitutes/falls back. Staged files are checked
for size/SHA256 and Safetensors headers; existing mismatches cause an error and
remain unchanged. The downloader never pickle-loads the skin `.pth`; its license
metadata is unavailable, so review source terms/provenance before use. Allow extra
staging/cache space. See [All2Real asset pins and limits](KREA_ALL2REAL.md).
