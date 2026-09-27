[CmdletBinding()]
param([switch]$ComfyUI)

$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

$venvPython = Join-Path $PWD ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    throw "Virtual environment not found. Run .\scripts\setup.ps1 first."
}

if (-not $env:HF_HOME) { $env:HF_HOME = Join-Path $PWD "models\huggingface" }
if (-not $env:HF_HUB_CACHE) { $env:HF_HUB_CACHE = Join-Path $env:HF_HOME "hub" }
$env:TOKENIZERS_PARALLELISM = "false"
if (-not $env:PYTORCH_CUDA_ALLOC_CONF) {
    $env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True,max_split_size_mb:128"
}
$env:CUDA_MODULE_LOADING = "LAZY"
$env:GRADIO_ANALYTICS_ENABLED = "False"
if ($ComfyUI) { $env:PHOTO_EDIT_COMFY_AUTOSTART = "true" }

& $venvPython app.py
