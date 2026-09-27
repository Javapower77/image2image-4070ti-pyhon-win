# Editing guide

## Head / body swap

In **Swap head / body**, upload Picture 1 (target body and scene) and Picture 2 (replacement face, head, or person), then choose a supported model and swap type. Qwen 2511 and FLUX.2 Klein 4B use their matching BFS LoRA; Krea 2 supports head and body swap with a BFS LoRA through local ComfyUI. Qwen Image 2.1 **Head** uses its `Qwen21-BFS_Head_v1.1.safetensors` at the BFS weight setting, loaded after its mandatory Viggle Turbo r256 adapter. Place the file in `models/loras/qwen21/`. Qwen 2.1 **Body** remains instruction-only and does not use the Head LoRA. Qwen 2.1 keeps the fixed six steps, CFG 1 and empty negative prompt; optional compatible LoRAs load last. Complex swaps may ghost or change identity. See `docs/SWAP.md`.

## Create from text: choose the model

Choose **Create from text**, then select **Krea 2 Turbo**, **Qwen Image 2.1 + Viggle Turbo**, or **FLUX.2 Klein 4B** in **02 / Generation settings**. Krea and Qwen 2.1 use project-managed ComfyUI; FLUX uses Diffusers. Qwen 2511, Rapid AIO and FireRed remain editing-only. Qwen 2.1 uses a fixed six-step schedule, CFG 1 and no negative prompt; optional Qwen 2.1 LoRAs follow the mandatory Viggle adapter. See `docs/MODELS.md` for the non-commercial research license and setup.

Enter a detailed visual prompt. Set **Output resolution** to **1K** or **2K**, and **Aspect ratio** to **1:1**, **9:16** (portrait), or **16:9** (landscape). The preview shows the requested dimensions; they are sent to the selected model. Defaults are 1K, 1:1 and the selected model's step/guidance settings. 2K on a 12 GB GPU may be slow or run out of memory. Source images, masks, identity preservation, and edit strength do not apply in this workflow.

Krea 2 LoRAs can be uploaded in the same LoRA panel and are stored under `models/loras/krea2/`. Use only adapters trained specifically for Krea 2. Start with one LoRA near its author's recommended weight.

## Krea 2 reference editing

Choose **Krea reference edit** to upload a source scene and an optional second reference (a person, object or style cue). Enter an editing instruction; this uses the project-managed ComfyUI Krea2Edit graph, not Krea's Diffusers text-to-image pipeline. Only 1–2 images are supported. The configured identity-edit LoRA is always used, followed by compatible Krea LoRAs selected in the UI at their chosen weights. Install the optional backend as described in `docs/KREA_REFERENCES.md`.

All **Krea** modes, including Krea **Create from text**, require `Krea2_ALWAYS_LOAD_FIRST.safetensors` in ComfyUI's `models/loras/`. Set its mandatory weight in the **Advanced · LoRAs** panel. It is applied before identity-edit, BFS, or optional LoRAs. FLUX text creation does not load this adapter.

## Output size

In **Edit source**, width and height follow the source photo. The Windows low-VRAM profile then reduces the request to a maximum 1024-pixel side and one megapixel while preserving aspect ratio.

In **Combine images** and **Create from text**, select **1K** or **2K**, then **Aspect ratio**: **1:1**, **9:16** (portrait), or **16:9** (landscape). These set the output canvas independently of input images. 1K produces 1024×1024 or 1024×576/576×1024; 2K produces 2048×2048 or 2048×1152/1152×2048. The longer side defines 1K/2K, and dimensions are aligned to 64 pixels. The app reserves the higher 2K budget for these two canvas workflows; edit, reference edit, and swap retain the 1K low-VRAM cap. 2K is more likely to exhaust the RTX 4070 Ti's 12 GB VRAM and takes longer.

## Combine images

Switch the workflow to **Combine images**, upload one to three pictures, and write a prompt that names them (`image 1`, `image 2`, `image 3`). Choose 1K or 2K and the desired aspect ratio; the size preview shows the actual canvas before generating. This generates a new picture from those references rather than editing a single source in place. Masks are ignored in this mode.

## Identity-preserving edits

Use a high-resolution source without heavy compression. Add a sharp, front-facing face crop as image 2 when the source face is small. Keep identity preservation enabled and phrase the request as a constrained delta: state the desired change, then list face, pose, body proportions, camera, lighting, and background elements that must stay unchanged.

Avoid asking for many unrelated changes in one run. Make one structural edit first, reuse its output as a new source, and then make a smaller refinement.

## Clothing changes

Use the person as image 1 and a clean product or worn-garment photo as image 2. Mention fabric, cut, sleeve length, closures, logos, and which image supplies the garment. Ask to preserve body proportions, pose, hands, face, and scene. Clothing replacement is generative, not measurement-accurate virtual fitting.

## Background and scene edits

Describe perspective, time of day, light direction, depth of field, and desired contact shadows. Explicitly preserve the subject, foreground edges, and camera framing. A white mask can constrain final compositing, but it does not currently condition denoising.

## LoRAs

Upload `.safetensors`, `.pt`, or `.bin` adapters in the **LoRAs** panel for Qwen 2511/Rapid AIO, FLUX.2 Klein 4B or Krea 2. For Qwen Image 2.1 use **only compatible `.safetensors` files**, stored in `models/loras/qwen21/`, and select up to five. The mandatory Viggle Turbo r256 LoRA always loads first; chosen extras follow in slot order at their selected weights. These ComfyUI model-only adapter patches may increase VRAM/RAM and have not been quality-tested together. Do not reuse Qwen 2511 files or select Viggle again. FireRed's automatic Lightning v1.2 LoRA still rejects additional adapters.

Choose up to five existing files and set each weight (typically 0.6–1.2; 0 skips the slot). LoRAs must match the selected family: a FLUX.2 Klein adapter will not load on Qwen, and a Qwen Image Edit adapter will not load on Klein. Changing the base model reloads that family’s library.

LoRA names and weights are stored in run metadata. Prompt text is still not stored.

## Restoration

Generate without restoration first. If eyes, mouth, or pores are weak, enable GFPGAN at about 0.2–0.4. Higher values can produce smooth skin or identity drift. A future FaceDetailer-style implementation should use local detection, a padded face crop, low-denoise regeneration, and feathered paste-back; it is not falsely represented as available in this release.

## Reproducibility

Set a fixed non-negative seed. Metadata is saved beside outputs. Prompt text is deliberately excluded for privacy, so retain it separately if exact reruns matter.
