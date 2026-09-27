# Models and 12 GB guidance

| UI model | Default steps | 12 GB status | Notes |
| --- | ---: | --- | --- |
| Krea 2 Turbo | 8 | Supported with offload | Gated 12B text-to-image model; CFG 0; Krea 2 Community License. |
| Qwen Rapid AIO | 4 | Recommended | Best first choice for Qwen edits; requires official Qwen 2511 components. |
| FLUX.2 Klein 4B | 4 | Recommended | Fast distilled option; use 1K output and one image. |
| Qwen Image Edit 2511 | 40 | Advanced | Runs through sequential CPU offload and is substantially slower. |
| Qwen Image 2.1 + Viggle Turbo r256 | 6 | Experimental / low-VRAM offload | INT8 base, unmerged six-step LoRA, 1–3 reference edits, text-to-image and instruction swaps. Non-commercial research license. |
| FireRed Image Edit 1.1 | 8 | Advanced/offload | Q4_K_M GGUF, FP8 Qwen2.5-VL vision encoder and automatic Lightning v1.2 via ComfyUI; substantial RAM/disk required. |

Diffusers models use local snapshots; FireRed instead loads only targeted ComfyUI GGUF/LoRA weights. CPU offload is mandatory for the large models on the target card. A current Diffusers source build is required for the Diffusers pipelines.

## Qwen Image 2.1 + Viggle Turbo (12 GB experimental)

Install the project-managed ComfyUI 0.37.0+ backend (`.\scripts\setup-comfy.ps1`) and run `.\scripts\download-models.ps1 -Preset qwen-2.1`, then `.\scripts\run.ps1 -ComfyUI`. The targeted download gets `qwen_image_2.1_int8_convrot.safetensors` (~7.3 GB), `qwen3vl_8b_int8_convrot.safetensors` (~9.4 GB) and `qwen_image_2.1_vae_bf16.safetensors` (~0.7 GB) from `Comfy-Org/Qwen-Image-2.1`, plus the **exact** `Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r256.safetensors` (~1.4 GB) and upstream `comfyui/viggle_turbo.py` node from `Viggle/Qwen-Image-2.1-viggle-turbo`. No full BF16 Qwen snapshot or alternative r128 LoRA is downloaded. Restart ComfyUI after adding the node. Leave substantial disk and free system RAM; INT8 files cannot both fit in 12 GB VRAM at once, so `--lowvram` CPU offload is essential and may be slow or run out of memory at 2K. Actual inference on the target GPU remains to be verified.

The app uses Viggle's **unmerged** LoRA at weight 1, fixed six-step raw sigmas `1.0, 0.9375, 0.875, 0.75, 0.5, 0.25`, Euler, CFG 1, and no negative prompt. Those settings are locked in the UI. Up to five *optional Qwen Image 2.1-compatible* `.safetensors` LoRAs can be uploaded to `models/loras/qwen21/` and selected in the five slots at individual weights; ComfyUI applies them **after** the mandatory Viggle adapter. Viggle itself stays in `vendor/ComfyUI/models/loras/` and must not be selected a second time. The additional LoRAs use ComfyUI's model-only loader (merged patches), unlike Viggle's unmerged node; combinations are not validated for image quality, may require extra VRAM/RAM, and must not be Qwen 2511 adapters. Edit source and Combine images accept 1–3 references and Create from text accepts none. **Head swap** automatically loads `Qwen21-BFS_Head_v1.1.safetensors` from `models/loras/qwen21/` after Viggle; its BFS weight slider applies and up to four further optional adapters can follow. **Body swap** remains two-image instruction-only (no BFS Body adapter). Complex swaps may ghost or alter identity; these are generative edits, not reliable exact face transfers. The source and Viggle adapter carry the **Qwen Research License Agreement (non-commercial research/evaluation only)**. Check upstream license and NOTICE before use.

## FireRed GGUF + Lightning (Windows 12 GB)

Install the project-managed backend with `.\scripts\setup-comfy.ps1`, then run `.\scripts\download-models.ps1 -Preset firered` or `python scripts/download_models.py firered-1.1`. Start the application with `.\scripts\run.ps1 -ComfyUI`. This downloads exactly `FireRed-Image-Edit-1.1-transformer-q4_k_m.gguf`, `qwen_image_vae.safetensors`, and `FireRed-Image-Edit-1.1-Lightning-8steps-v1.2.safetensors` from the official FireRed ComfyUI repository, plus the complete `qwen_2.5_vl_7b_fp8_scaled.safetensors` vision/text encoder from `Comfy-Org/Qwen-Image_ComfyUI`. The FireRed repository's Q8 GGUF encoder is deliberately not selected: ComfyUI-GGUF expects a separate multimodal projector (`mmproj`) that the repository does not include. The transformer GGUF alone is about 12.17 GiB and **does not fit entirely into 12 GB VRAM**; ComfyUI's `--lowvram` CPU offload is required and runs may be slow. Allow substantial free disk and system RAM. Lightning is applied automatically at strength 1 with 8 default steps, Euler/simple sampling; optional user FireRed LoRAs are rejected because GGUF LoRA compatibility is experimental. Live output on the target GPU must be verified after downloading the weights.

**Create from text** offers Krea 2 Turbo and Qwen 2.1 Turbo through project-managed ComfyUI, or FLUX.2 Klein 4B through Diffusers. Selecting a model in section **02 / Generation settings** changes the inference backend; text-only requests do not provide image-conditioning arguments. These models accept 1K/2K canvases and 1:1, 9:16 or 16:9 aspect controls. The editing-only Qwen 2511, Rapid AIO and FireRed models are excluded.

## FLUX.2 Klein 4B custom text encoder

Only FLUX.2 Klein 4B loads the local Qwen3 GGUF text encoder configured by `PHOTO_EDIT_FLUX_KLEIN_4B_TEXT_ENCODER`. The default path is `models/qwen3-4b-alb-q4_0.gguf`. For compatibility with the file currently in this workspace, `qwen3-4b-abl-q4_0.gguf` is also detected automatically.

The GGUF must represent the same Qwen3 4B architecture expected by the downloaded Klein 4B `text_encoder/config.json` (36 layers, hidden size 2560). The pipeline continues to use Klein's bundled tokenizer and uses hidden-state layers 9, 18, and 27. A differently shaped or incompatible GGUF fails during loading instead of silently falling back to the original encoder.

On CUDA, Transformers currently dequantizes this Qwen3 GGUF at load time rather than retaining Q4 blocks. Sequential CPU offload still limits GPU residency, but expect higher system-RAM usage than the file size suggests. The `gguf` package is installed as a project dependency.

## Krea 2 Turbo

`krea/Krea-2-Turbo` is a gated 12-billion-parameter text-to-image model. Accept the Krea 2 Community License on Hugging Face and authenticate with `hf auth login` when needed. All Krea 2 modes, including **Create from text**, run through the project's optional ComfyUI backend so `Krea2_ALWAYS_LOAD_FIRST.safetensors` is always applied first. Install the ComfyUI-format Krea weights, mandatory TextFusion adapter, and optional official Krea Turbo LoRA with `.\scripts\download-models.ps1 -Preset krea-2`. The optional Turbo LoRA is placed in `models/loras/krea2/`; select it explicitly in the UI if wanted. See `docs/KREA_REFERENCES.md`.

For reference-conditioned editing select **Krea reference edit** instead. That mode uses `comfyui-krea2edit` and a built-in image-conditioned API-format workflow, adding the identity-edit LoRA after the mandatory adapter. See `docs/KREA_REFERENCES.md`.

Krea 2 provides Diffusers LoRA methods. Compatible Krea 2 LoRAs are stored in `models/loras/krea2/`; FLUX, Qwen, SDXL, and other architecture LoRAs are not interchangeable. The base model is large for 12 GB VRAM, so LoRAs add load time and system-memory pressure. Start with one adapter.

The Krea 2 Community License requires appropriate content filtering or an equivalent review process. Review the current license and acceptable-use policy before use or deployment.

## Rapid AIO

Rapid AIO is a single ComfyUI Safetensors transformer. Download both components with `python scripts/download_models.py qwen-2511 qwen-2511-aio`. The application keeps the official Qwen tokenizer, text encoder, VAE, and scheduler, then replaces the transformer. A small HTML file renamed `.safetensors` is not a valid checkpoint and is rejected.

## Output policy

Use 1K, one output, and four steps for distilled models. Larger source images are normalized and requested outputs are reduced to the configured pixel budget. For a larger final asset, generate at 1K and use a dedicated tiled upscaler afterward rather than increasing diffusion dimensions.

## LoRAs

Qwen Image Edit 2511 and Rapid AIO use `models/loras/qwen/`; Qwen 2.1 uses a separate `models/loras/qwen21/` library for up to five optional `.safetensors` adapters. Klein 4B uses `models/loras/flux/`; Krea 2 uses `models/loras/krea2/`. FireRed Lightning and Qwen 2.1 Viggle r256 are installed separately in `vendor/ComfyUI/models/loras/` and applied automatically. FireRed does not support extras. Additional LoRAs increase system-memory and loading overhead.

Review the license files in every downloaded snapshot. Model terms can change independently of this project.

## BFS swap model support

The two-image BFS workflow uses model-matched LoRAs for Qwen 2511, FLUX.2 Klein 4B, Krea 2 and now Qwen 2.1 **Head**. Qwen 2.1 **Body** remains two-image instruction-only and experimental. Rapid AIO and FireRed are not offered because BFS files target different base models. See `docs/SWAP.md`.
