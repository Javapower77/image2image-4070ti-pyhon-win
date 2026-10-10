# Optional DLSS 5 enhancement

The **DLSS 5 enhancement · ComfyUI only** accordion enables the upstream
[Blueforcer node pack](https://github.com/Blueforcer/ComfyUI-DLSS5-Enhancer).
It is off by default. When enabled, `DLSS5Settings` feeds `DLSS5EnhanceImages`
between the generated IMAGE and SaveImage node 29.

Supported paths: the two six-step **ComfyUI** Qwen Image 2.1 profiles' editing,
combination, text and swaps; Krea
reference edit, Composition remix, text and swaps; FireRed editing. Disable
enhancement when selecting Diffusers models (Rapid AIO, Qwen 2511, FLUX or either
official Qwen 2.1 profile). Official BF16 BFS also rejects DLSS; direct official
requests must use `dlss=None`, not a disabled settings dictionary.
Unsupported requests fail before model loading rather than silently skipping it.

## Setup

The Windows ComfyUI setup script installs the MIT-licensed node pack and Python
dependencies. The pack has also been installed in this workspace. It requires
ComfyUI's V3 API and Windows with a supported NVIDIA RTX GPU and current driver.
Restart ComfyUI after adding the pack.

**The proprietary native runtime is separate and has not been downloaded.**
Review upstream license notices and run its `install_runtime.py` yourself to
download the supported runtime or register an existing installation. Alternatively,
set the Runtime dir control to an authorized runtime folder containing `nvngx.dll`.
An empty field uses upstream environment/configuration discovery. The application
does not run the runtime installer or alter antivirus settings automatically.

## Parameters

| Control | Default | Options/range |
| --- | --- | --- |
| Enabled | Off | Opt in per generation |
| Upscaling mode | 1x DLAA/native | 1x, 1.5x Quality, 1.724x Balanced, 2x Performance, 3x Ultra Performance |
| NR preset | Default | Default, presets 1–3 |
| NR style | Default | Default, Natural, Cinematic |
| NR intensity | 1 | 0–2 |
| Local tone strength | 1 | 0–2 |
| Local structure strength | 1.5 | 0–2 |
| Skin structure strength | 2 | −1–2; requires automatic mask |
| Automatic mask | On | Semantic skin masking |
| DLSS model preset | M | Default, J, K, L, M |
| Motion | auto | auto, optical_flow, none |
| Scene change threshold | 0.24 | 0.01–1 |
| Warmup frames | 0 | 0–16 |
| Runtime dir | Empty | Optional local ComfyUI runtime override |
| Verify neural rendering | On | Require upstream feature-18 verification |

Upstream reports NR preset is currently inert and intensity above 1 has no further
effect. Natural is more conservative; Cinematic changes grading. For independent
still images use the default auto motion mode. Verification stays on by default
so a plain-upscale fallback is not silently reported as neural enhancement.

The diffusion resolution budget is unchanged. Upscaling increases the *final*
resolution and memory usage independently; use 1x initially on 12 GB. Only the
enhanced output is saved. Enhancement settings and output sizes are recorded in
metadata. Native/runtime errors are reported by ComfyUI; inspect `logs/comfyui.log`
for the managed server. The native worker also has its own upstream diagnostics.

Node schema, graph placement, parameters, routing and UI are regression tested.
Native runtime execution, compatibility with the installed driver and peak GPU
memory have not been verified. Do not assume the community runtime is endorsed
by NVIDIA or that reconstruction preserves facial identity exactly.
