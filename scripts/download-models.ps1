[CmdletBinding()]
param(
    [ValidateSet("recommended", "qwen-aio", "qwen-2.1", "qwen-2.1-r128", "qwen-2.1-bfs", "qwen-2.1-official", "qwen-2.1-official-extract", "qwen-character-sheet", "flux-4b", "krea-2", "krea-originals", "krea-character-sheets", "firered", "all")]
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
    "qwen-2.1-r128" { $models = @("qwen-2.1-turbo-r128") }
    "qwen-2.1-bfs" { $models = @("--qwen21-bfs") }
    "qwen-2.1-official" { $models = @("qwen-2.1-turbo-official") }
    "qwen-2.1-official-extract" { $models = @("qwen-2.1-turbo-official-extract") }
    "qwen-character-sheet" { $models = @("--comfy-qwen-character-sheet") }
    "flux-4b" { $models = @("flux-klein-4b") }
    "krea-2" { $models = @("krea-2-turbo") }
    "krea-originals" { $models = @("--comfy-krea-originals") }
    "krea-character-sheets" { $models = @("--comfy-krea-character-sheets") }
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
Write-Host "Missing tokens are requested securely by Python in an interactive terminal (blank for public)."
Write-Host "For automation use HF_TOKEN/cached HF login and CIVITAI_API_TOKEN (or CIVITAI_TOKEN); redirected input never prompts."
& $venvPython @arguments
if ($LASTEXITCODE -ne 0) {
    throw "Model download failed with exit code $LASTEXITCODE. See docs\TROUBLESHOOTING.md."
}
Write-Host "Model download complete. Start with: .\scripts\run.ps1"
