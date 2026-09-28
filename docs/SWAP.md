# Head and body swap workflow

Use **Swap head / body** with two images: **Picture 1** is the target body/scene to keep; **Picture 2** is the replacement face/head or person. The output follows Picture 1's aspect ratio and the 12 GB resolution cap. The workflow retains prompts, size multiplier, seed, local outputs, progress, and optional GFPGAN. Qwen 2.1 always uses Viggle Turbo first and now loads a model-matched BFS LoRA for **Head** and **Body**. Only use images you have permission to edit; do not publish misleading impersonations.

## Supported combinations

| Base model | Swap | BFS LoRA | Backend |
| --- | --- | --- | --- |
| Qwen Image Edit 2511 | Head | `bfs_head_v5_2511_merged_version_rank_16_fp16.safetensors` | Existing local Diffusers pipeline |
| FLUX.2 Klein 4B | Head | `bfs_head_v1_flux-klein_4b.safetensors` | Existing local Diffusers pipeline (including configured Qwen3 GGUF text encoder) |
| Krea 2 Turbo | Head | `bfs_head_swap_v1.1_krea2.safetensors` | Project-managed ComfyUI with Krea2Edit custom nodes |
| Krea 2 Turbo | Body | `bfs_body_swap_v1_krea2.safetensors` | Project-managed ComfyUI with Krea2Edit custom nodes |
| Qwen Image 2.1 + Viggle Turbo | Head | `Qwen21-BFS_Head_v1.1.safetensors` after mandatory unmerged Viggle r256 | Project-managed ComfyUI, two-image head swap |
| Qwen Image 2.1 + Viggle Turbo | Body (experimental) | `Qwen21-BFS_Body_v1.1.safetensors` after mandatory unmerged Viggle r256 | Project-managed ComfyUI, two-image body swap |

Rapid AIO and FireRed are not offered because these BFS LoRAs target different architectures. LoRA weights are not interchangeable. Krea body-swap and Qwen 2.1 Head/Body swaps are experimental; the Viggle release explicitly warns of possible ghosting and identity drift on complicated swaps.

Download only the four legacy LoRAs from `Alissonerdx/BFS-Best-Face-Swap` with `.\scripts\download-models.ps1 -BfsSwap -Preset flux-4b` (the preset also checks the already-downloaded Flux base model). Alternatively run `python scripts/download_models.py --bfs-swap`. Qwen 2.1's **separately released** BFS Head/Body v1.1 files are not in that download set; place your copies at `models/loras/qwen21/Qwen21-BFS_Head_v1.1.safetensors` and `models/loras/qwen21/Qwen21-BFS_Body_v1.1.safetensors`. Base model snapshots are downloaded separately using the normal presets. Uploading additional compatible LoRAs is supported, but start with only the BFS LoRA on 12 GB.

## Krea editing backend: required setup

The installed Diffusers `Krea2Pipeline` supports **text-to-image only**; it cannot consume two reference images. The Krea swap workflows use `Krea2EditGroundedEncode` and `Krea2EditModelPatch` from `comfyui-krea2edit`. To use Krea Head/Body swap:

1. Run `.\scripts\setup-comfy.ps1` to install ComfyUI and Krea2Edit nodes inside this project's `.venv` and ignored `vendor/ComfyUI/` directory.
2. Run `python scripts/download_models.py --comfy-krea` to obtain matching ComfyUI-format Krea Turbo/CLIP/VAE weights and the identity-edit LoRA (not supplied by the Diffusers snapshot). Run `python scripts/download_models.py --bfs-swap` to obtain BFS LoRAs. The embedded process searches `models/loras/krea2/` directly without duplicate files.
3. Launch `.\scripts\run.ps1 -ComfyUI`. The app starts ComfyUI on demand, locally at `127.0.0.1:8188`. The built-in 2-image Krea API graph is used for Head/Body swap and selects the correct BFS LoRA. No manual workflow export is needed. Advanced users can override `PHOTO_EDIT_COMFY_HEAD_WORKFLOW` or `PHOTO_EDIT_COMFY_BODY_WORKFLOW` with a tested API-format graph.

The bridge uploads the two reference images to the loopback ComfyUI API, substitutes prompt, seed, steps, CFG, BFS weight, and output prefix, waits for the saved output, then saves a copy in this app's `outputs/`. For Krea swaps, the mandatory `Krea2_ALWAYS_LOAD_FIRST.safetensors` (with its dedicated UI weight) is loaded **first**, then the BFS LoRA, then any additional selected Krea LoRAs at their chosen weights. Qwen and Flux swap order is unchanged. The bundled Krea graph uses the upstream Krea2Edit node paths and a 1K output. Customize ref_boost, fit mode, sampler or masking with a tested API graph if necessary. For Krea swap, leave the negative prompt empty; configure it in a custom workflow if needed. Gradio's progress remains indeterminate while ComfyUI works because this bridge does not receive its internal sampler steps.

## Limitations

- Qwen 2.1 swap requires its INT8 ComfyUI model, Viggle r256 LoRA and custom node; use `.\scripts\download-models.ps1 -Preset qwen-2.1` after `.\scripts\setup-comfy.ps1`. Keep six steps, CFG 1, and the negative prompt empty. **Head** and **Body** load the separately provided `Qwen21-BFS_Head_v1.1.safetensors` or `Qwen21-BFS_Body_v1.1.safetensors` at the BFS weight slider setting, after Viggle and before up to four optional Qwen 2.1 adapters. Selecting that same BFS file in an optional slot does not load it twice. No pixel-exact swap is guaranteed.
- These are **head** (Qwen/Flux/Krea) or **person/body** (Krea) replacements, not pixel-perfect face-only swaps. Expression, hair, pose, or background can drift.
- No segmentation, masked inpainting, or automatic crop/stitch is claimed. The original edit workflow's optional mask is still post-generation compositing; the swap section does not claim native masks.
- Diffusers and ComfyUI use different pipelines. Exact outputs from the supplied ComfyUI workflow are not guaranteed by the Qwen/Flux Diffusers approximations.
- Krea 2 is gated and subject to the Krea 2 Community License and its filtering/review requirements. Review upstream LoRA terms too.

Sources: [BFS model files/workflows](https://huggingface.co/Alissonerdx/BFS-Best-Face-Swap/tree/main), [Krea body swap](https://civitaiarchive.com/models/2027766?modelVersionId=3188610), [Flux Klein 4B head swap](https://civitaiarchive.com/models/2027766?modelVersionId=2609209), [Qwen 2511 head swap](https://civitaiarchive.com/models/2027766?modelVersionId=2556739).
