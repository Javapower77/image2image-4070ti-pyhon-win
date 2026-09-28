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

The **Krea reference edit** workflow uses an optional project-managed ComfyUI backend. Run `.\scripts\setup-comfy.ps1`, download weights with `python scripts/download_models.py --comfy-krea`, and start `.\scripts\run.ps1 -ComfyUI`. The default API graph is bundled; no manual export is required. For startup errors inspect `outputs/comfyui.log` and follow `docs/KREA_REFERENCES.md`. Text-to-image does not require ComfyUI.

## Second Krea reference is disconnected

The bundled graph connects both `Krea2EditGroundedEncode` nodes and the `Krea2EditModelPatch` pixel/latent paths to the second LoadImage (node 90). If using a custom exported API graph, enable the second image group before export; the app rejects disconnected inputs instead of pretending the second image influenced the result.

## BFS LoRA missing / swap model mismatch

For Qwen 2511, FLUX.2 Klein 4B, and Krea 2, run `python scripts/download_models.py --bfs-swap`. Qwen Image 2.1 Head/Body LoRAs are not in that download: copy `Qwen21-BFS_Head_v1.1.safetensors` and `Qwen21-BFS_Body_v1.1.safetensors` into `models/loras/qwen21/`. Other models are intentionally excluded; cross-family LoRAs cannot be loaded safely.

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

## Identity changes

Use a sharp face crop as the first reference, narrow the edit request, explicitly preserve facial geometry and skin texture, reuse the seed, and disable restoration while diagnosing. No generic model guarantees biometric identity.

## GFPGAN errors

Install with `.\scripts\setup.ps1 -Restore` and download weights with `python scripts\download_models.py --restorers`. Restoration remains optional because its dependency stack can lag current PyTorch releases.
