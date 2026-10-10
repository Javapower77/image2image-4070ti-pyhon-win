# Troubleshooting

## CUDA is unavailable on Windows

Activate `.venv`, run `python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`, and confirm `nvidia-smi` works. If PyTorch reports a CPU-only build, rerun `.\scripts\setup.ps1`. Update the NVIDIA driver if the CUDA wheel cannot initialize.

## PowerShell will not run scripts

Run `Set-ExecutionPolicy -Scope Process Bypass` in the current PowerShell window, then rerun the script. Avoid changing machine-wide policy unless required by the administrator.

## Pipeline class unavailable

From the activated environment, reinstall the editable project: `python -m pip install --upgrade --force-reinstall -e .`. The project depends on a current Diffusers source checkout.

## Local model not found

Run `python scripts\download_models.py MODEL_KEY`. Runtime loading is local-only and does not silently download weights.

## Krea 2 returns 401 or gated-repository errors

Open `https://huggingface.co/krea/Krea-2-Turbo`, sign in, accept the Krea 2 Community License, run `hf auth login` from the activated environment, and retry `.\scripts\download-models.ps1 -Preset krea-2`.

## Krea reference edit cannot find a workflow or connect to ComfyUI

The **Krea reference edit** workflow uses an optional project-managed ComfyUI backend. Run `.\scripts\setup-comfy.ps1`, download weights with `python scripts/download_models.py --comfy-krea`, and start `.\scripts\run.ps1 -ComfyUI`. The default API graph is bundled; no manual export is required. For startup errors inspect `logs/comfyui.log` and follow `docs/KREA_REFERENCES.md`.

## Krea text output ignores the positive prompt

Check CFG in Run Details. The previous UI default of 0 discarded positive-prompt
conditioning in ComfyUI's standard sampler. The corrected default is **CFG 1**;
text generation now rejects CFG 0 instead of silently producing empty/negative
prompt output. The official Diffusers `guidance_scale=0` is not equivalent.
Restart the studio, refresh the browser, select Krea again and verify CFG 1.
Retry the same prompt/seed at 1K with optional LoRAs and DLSS disabled before
comparing higher resolutions or adapters. This does not guarantee exact prompt
adherence; model limitations and adapters can still affect the result.

## Application disappears or loses its server connection

Restart using the Windows launcher to enable console capture. Diagnostics are stored in:

- `logs/studio.log`: rotating application log (10 MiB, five backups), with timestamps,
	process/thread IDs, model-load stages, exceptions, RAM availability, process RSS/private
	memory, and allocated/reserved CUDA memory when CUDA is already initialized.
- `logs/console-<timestamp>-<launcher-pid>.log`: unbuffered Python stdout/stderr, start/end
	markers and the actual Python exit code. The launcher propagates nonzero exit codes.
- `logs/fault-<python-pid>.log`: Python faulthandler stack dumps for supported fatal faults.
- `logs/comfyui.log`: stdout/stderr from the project-managed ComfyUI subprocess.

Rapid AIO logs bracket the official transformer load, AIO checkpoint extraction/conversion,
state application, RoPE initialization, remaining pipeline components and memory offload.
The final stage before termination helps localize the problem. Completing the checkpoint
shard progress bar does not mean the model is ready: additional CPU allocations and offload
configuration follow. Large RAM/pagefile pressure is a possibility, not a confirmed diagnosis.

If `Gradio launch returned` and `Python interpreter shutdown` appear, the server exited
through normal Python shutdown. If they are missing, inspect the launcher exit code and
fault file. Windows process termination or a system-level resource failure can bypass Python
handlers entirely; an empty fault file does not rule those out. Check Windows Event Viewer
Application/System logs for a matching timestamp. Share the final load stages and exit code.

Application logs do not intentionally record prompts, images, credentials or tensor contents.
Exception tracebacks and raw third-party console output may contain paths or input details;
review/redact logs before sharing. Console and fault files are per run/process; remove old
files periodically. Logging does not change the model loading or inference algorithm.

## Second Krea reference is disconnected

The bundled graph connects both `Krea2EditGroundedEncode` nodes and the `Krea2EditModelPatch` pixel/latent paths to the second LoadImage (node 90). If using a custom exported API graph, enable the second image group before export; the app rejects disconnected inputs instead of pretending the second image influenced the result.

## BFS LoRA missing / swap model mismatch

Run `python scripts/download_models.py --bfs-swap` for the existing four
Qwen 2511/FLUX/Krea files **plus** the two pinned Qwen 2.1 files. For Qwen 2.1
only, use `.\scripts\download-models.ps1 -Preset qwen-2.1-bfs` or
`.\.venv\Scripts\python.exe scripts\download_models.py --qwen21-bfs`.
Expected canonical files in `models/loras/qwen21/` are
`bfs_head_v1.1_qwen_2.1.safetensors` (260096144 bytes) and
`bfs_body_swap_v1.0_qwen_2.1.safetensors` (209753576 bytes).
See [exact SHA256/Civitai pins](SWAP.md#canonical-qwen-21-bfs-assets-and-migration).
Verified legacy aliases are copied to canonical names with originals kept;
these are the same bytes, not new weights. Body is published **v1.0**, despite
the legacy `Qwen21-BFS_Body_v1.1.safetensors` alias. Invalid legacy files are kept
and a pinned download is attempted; invalid canonical files fail closed without
overwrite. Move/remove an invalid canonical file deliberately before retrying.
Other models remain intentionally excluded; cross-family LoRAs are unsafe.

### Official BF16 BFS request or shape failure

Use `qwen-2.1-turbo-official`, not the swap-ineligible `official-extract` profile.
Supply exactly two images (body/source then replacement reference), Head/Body,
eight steps, CFG/True CFG 1, empty negative prompt and sole matching canonical
`bfs_swap` adapter at finite weight **0.1–1.5**. Omit optional LoRAs and set
`dlss=None`; a disabled DLSS dictionary is still rejected. The full official
snapshot/dependencies are required, but ComfyUI/Viggle/r128 are not used here.

Size/hash failures are checked before a cold checkpoint allocation. **Actual
target shapes are checked after the full BF16 checkpoint is allocated**, before
installation/offload, so a shape failure can still consume substantial RAM.
Do not bypass pins or discard incompatible keys: CPU A/B normalization splits
gate-first into `gate_layer`/`proj`, with intrinsic scale 1 and request weight
separate. Reload after an installation/change failure rather than assuming
rollback. Compatibility, live inference, visual quality and VRAM fit remain
unconfirmed; this is not a validated 12 GB inference recipe.

## Krea swap workflow missing or ComfyUI rejects the prompt

Download the BFS **UI** JSON from upstream, open in ComfyUI with `comfyui-krea2edit` and required custom nodes installed, select the correct local base model, VAE, text encoder, and LoRA, test the workflow, then export **API Format** to `models/workflows/`. Normal UI JSON is not accepted by the ComfyUI API. The workflow must retain the expected upstream node IDs. See `docs/SWAP.md`.

## Krea 2 is slow or runs out of memory

Keep 1024×1024, one output, 8 steps, guidance 0, and sequential offload. Close other GPU applications and avoid multiple LoRAs. Krea 2 has about 12B parameters, so loading and layer transfers are significantly slower than smaller distilled models on a 12 GB GPU.

## FLUX.2 Klein custom text encoder is missing

Place the GGUF at `models/qwen3-4b-alb-q4_0.gguf` or set `PHOTO_EDIT_FLUX_KLEIN_4B_TEXT_ENCODER` to its full path. The application also accepts the current `qwen3-4b-abl-q4_0.gguf` filename. Restart after changing `.env`.

## FLUX.2 Klein GGUF fails to load

Confirm it is a Qwen3 4B GGUF compatible with Klein's text-encoder configuration, not a diffusion transformer or another Qwen size. Rerun `.\scripts\setup.ps1` to install the `gguf` loader. On Windows/CUDA the encoder is dequantized while loading, so ensure substantial free system RAM and close memory-heavy applications.

## Rapid AIO file is HTML or tiny

Delete the invalid file and run `python scripts\download_models.py qwen-2511 qwen-2511-aio`. A valid checkpoint is many gigabytes and has a Safetensors header.

## CUDA out of memory

1. Click **Unload model / release VRAM**.
2. Keep one output, 1K resolution, and the default four steps for Rapid AIO/Klein 4B.
3. Close browsers, games, video tools, and other GPU applications; inspect usage with `nvidia-smi`.
4. Restart the application after an OOM because CUDA fragmentation can persist.
5. Use Rapid AIO or Klein 4B instead of a full 40-step model.
6. Keep `PHOTO_EDIT_MEMORY_MODE=sequential`, `PHOTO_EDIT_MAX_OUTPUT_SIDE=1024`, and `PHOTO_EDIT_MAX_OUTPUT_PIXELS=1048576`.
7. For FireRed, start project-managed ComfyUI with `-ComfyUI`, use 1K, and ensure ample free system RAM for GGUF CPU offload.

If OOM continues, lower `PHOTO_EDIT_MAX_OUTPUT_SIDE` to 896 and `PHOTO_EDIT_MAX_OUTPUT_PIXELS` to 802816, restart, and retry.

## Generation is slow

Sequential CPU offload continuously transfers layers over PCIe. This is expected on 12 GB hardware. Keep models and the Hugging Face cache on NVMe, use distilled four-step models, avoid multiple LoRAs, and leave enough free system RAM to prevent paging.

## LoRA failed to load

Confirm that the file is an adapter for the selected architecture, not a complete checkpoint. Start with one LoRA, unload the model after a failed attempt, and keep its weight near 1.0.

## Qwen 2511 BFS: missing alpha or silently missing bias updates

An older legacy-converter failure such as `KeyError: 'img_in.alpha'` does not
mean this BFS file is corrupt. The measured Qwen 2511 Head file has **846 LoRA
matrix pairs, 846 `.diff_b` bias deltas and 241 `.diff` direct weight deltas**,
and **no alpha tensors**. See [pinned measured header/cache provenance](MODELS.md#qwen-image-edit-2511-mixed-format-loras)
for revision, LFS ETag, size and verification limits. Qwen 2511 is **Head only,
not Body**; do not substitute Qwen 2.1 adapters.

The Qwen-specific loader uses omitted alpha = actual rank (intrinsic scale 1),
preserves explicit alpha/rank and applies signed strength separately, once.
Direct biases/weights are additive at request strength, not alpha-scaled or
discarded. Unknown keys/targets, orphan pairs/alpha and incompatible shapes
fail before updates; **no silent drop or architecture fallback** is supported.
After an installation/offload failure the pipeline is discarded: reload it,
rather than assuming rollback. Weight changes and deselection normally restore
pristine originals; unchanged signatures do not accumulate direct deltas.
Hook detachment must precede validation/restoration, with memory configuration
after installation. CPU regression tests cover synthetic modules and real
Accelerate hooks when installed, not full weights or inference.

## Identity changes

Use a sharp face crop as the first reference, narrow the edit request, explicitly preserve facial geometry and skin texture, reuse the seed, and disable restoration while diagnosing. No generic model guarantees biometric identity.

## GFPGAN errors

Install with `.\scripts\setup.ps1 -Restore` and download weights with `python scripts\download_models.py --restorers`. Restoration remains optional because its dependency stack can lag current PyTorch releases.
