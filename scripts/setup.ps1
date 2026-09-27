[CmdletBinding()]
param(
    [switch]$Restore,
    [switch]$ComfyUI,
    [ValidateSet("none", "recommended", "qwen-aio", "qwen-2.1", "flux-4b", "krea-2", "firered", "all")]
    [string]$Download = "none"
)

$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

$python = (Get-Command py -ErrorAction Stop).Source
& $python -3.11 -m venv .venv
$venvPython = Join-Path $PWD ".venv\Scripts\python.exe"

& $venvPython -m pip install --upgrade pip setuptools wheel
& $venvPython -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
& $venvPython -m pip install -e ".[dev]"
if ($Restore) {
    & $venvPython -m pip install -e ".[restore]"
}
if ($ComfyUI) {
    & (Join-Path $PSScriptRoot "setup-comfy.ps1")
    if ($LASTEXITCODE -ne 0) { throw "Embedded ComfyUI setup failed." }
}

New-Item -ItemType Directory -Force -Path models, outputs | Out-Null
& $venvPython -c "import torch; print(f'PyTorch {torch.__version__}; CUDA available: {torch.cuda.is_available()}'); assert torch.cuda.is_available(), 'CUDA is unavailable. Update the NVIDIA driver and rerun setup.'"

if ($Download -ne "none") {
    & (Join-Path $PSScriptRoot "download-models.ps1") -Preset $Download
}

Write-Host "Setup complete."
if ($Download -eq "none") {
    Write-Host "Models are not bundled. Download one with:"
    Write-Host "  .\scripts\download-models.ps1 -Preset recommended"
    Write-Host "  .\scripts\download-models.ps1 -Preset qwen-aio"
    Write-Host "  .\scripts\download-models.ps1 -Preset qwen-2.1"
    Write-Host "  .\scripts\download-models.ps1 -Preset flux-4b"
    Write-Host "  .\scripts\download-models.ps1 -Preset krea-2"
    Write-Host "  .\scripts\download-models.ps1 -Preset firered"
}
Write-Host "Start with: .\scripts\run.ps1"
if ($ComfyUI) {
    Write-Host "For embedded ComfyUI: .\scripts\run.ps1 -ComfyUI"
}
