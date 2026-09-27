[CmdletBinding()]
param(
    [ValidateSet("recommended", "qwen-aio", "qwen-2.1", "flux-4b", "krea-2", "firered", "all")]
    [string]$Preset = "recommended",
    [switch]$Restorers,
    [switch]$BfsSwap,
    [switch]$ComfyUIKrea,
    [switch]$ComfyUIFireRed,
    [switch]$ComfyUIQwen21,
    [switch]$TurboLora
)

$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

$venvPython = Join-Path $PWD ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    throw "Virtual environment not found. Run .\scripts\setup.ps1 first."
}

switch ($Preset) {
    "recommended" { $models = @("qwen-2511", "qwen-2511-aio", "flux-klein-4b") }
    "qwen-aio" { $models = @("qwen-2511", "qwen-2511-aio") }
    "qwen-2.1" { $models = @("qwen-2.1-turbo") }
    "flux-4b" { $models = @("flux-klein-4b") }
    "krea-2" { $models = @("krea-2-turbo") }
    "firered" { $models = @("firered-1.1") }
    "all" { $models = @("--all") }
}

$arguments = @("scripts\download_models.py") + $models
if ($Restorers) {
    $arguments += "--restorers"
}
if ($BfsSwap) {
    $arguments += "--bfs-swap"
}
if ($ComfyUIKrea) {
    $arguments += "--comfy-krea"
}
if ($ComfyUIFireRed) {
    $arguments += "--comfy-firered"
}
if ($ComfyUIQwen21) {
    $arguments += "--comfy-qwen21"
}
if ($TurboLora) {
    $arguments += "--turbo-lora"
}

Write-Host "Downloading model preset '$Preset'. Downloads can be tens of gigabytes."
& $venvPython @arguments
if ($LASTEXITCODE -ne 0) {
    throw "Model download failed with exit code $LASTEXITCODE. See docs\TROUBLESHOOTING.md."
}
Write-Host "Model download complete. Start with: .\scripts\run.ps1"
