# Krea 2 composition remix

## Use

Select **Krea reference edit**, then **Composition remix**. Upload an already
composed canvas as Picture 1. Enter `remix` or describe the intended realistic
result. Defaults are 9 steps and CFG 1; the seed controls reproducibility.
Only Krea 2 Turbo is used. Remix returns one generated image, preserves the
source aspect within model alignment, and caps the longest side at 1024.

Picture 2, negative prompt, optional LoRAs, size multiplier, identity-preservation
prompting and face restoration are ignored for this operation. The mandatory
`Krea2_ALWAYS_LOAD_FIRST` weight still applies. Ordinary reference edit is unchanged.

## Required assets

- Existing Krea FP8 transformer, Qwen3-VL encoder and Qwen image VAE.
- `vendor/ComfyUI/models/loras/Krea2_ALWAYS_LOAD_FIRST.safetensors`.
- `vendor/ComfyUI/models/loras/Krea2-Remix_Patreon.safetensors` (user supplied).
- `ComfyUI-Krea2-Ostris-Edit` in ComfyUI's `custom_nodes` directory. The Windows
  ComfyUI setup script now installs this extension. Restart ComfyUI after installation.

The app checks registered node types and filenames before queueing a remix.
No paid adapter or replacement model weights are downloaded automatically.

## Supplied workflow analysis

`workflows/remix.json` is an editor workflow with embedded subgraphs, not an API
prompt. The subgraph instance overrides the prompt to `remix`, seed to 666 and
adapter to `Krea2-Remix_Patreon.safetensors`. Connected integer controls resolve
the main schedule to 9 steps (8 + 1), overriding stale saved sampler widgets.

Its model path is Krea → Ostris patch → Remix adapter for the first sampler.
Later samplers use the patched base model without Remix. Reference latents
are removed for refinement. The graph includes image upscaling, a sharp VAE,
an additional upscale VAE, and DetailDaemon refinement. Although the *outer*
skin-detail upscale subgraph is bypassed, internal upscaling remains active.
Its first exposed output is the processed source, not a generated result;
the second is the final generated image.

## Adaptation selected for this application

The application deliberately does **not** reproduce the full upstream pipeline:

1. Reuse existing FP8 Krea weights, encoder and VAE; encode the uploaded canvas.
2. Apply the mandatory first adapter before the model branches.
3. Use the Ostris patch with cached reference attention and the Remix adapter
   at weight 1 for Euler / `kl_optimal`, steps 1 through `steps - 1`.
4. Refine with the patched base model, without Remix, using Euler / `simple`
   from `steps - 1` through `steps`.
5. Decode and return the generated image only. No upscale or DetailDaemon.

The refinement re-encodes the same prompt without references. Unlike the original
reference-latent-removal node, this also discards vision-derived text context.
The original workflow's KV-cache setting is retained; compatibility and visual
quality with the installed FP8 base need a real inference run. A 1024 cap reduces
memory pressure but does not guarantee every image fits 12 GB VRAM.

Graph, routing and UI tests run without GPU inference. Exact visual parity,
runtime custom-node registration and peak VRAM require testing with ComfyUI.