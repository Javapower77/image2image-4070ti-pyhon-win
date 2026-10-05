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

$logDir = Join-Path $PWD "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$consoleLog = Join-Path $logDir ("console-{0}-{1}.log" -f (Get-Date -Format "yyyyMMdd-HHmmss"), $PID)
$start = Get-Date
"[$($start.ToString('o'))] Starting studio. Launcher PID=$PID" | Out-File $consoleLog -Encoding utf8
Write-Host "Diagnostics: $logDir (console: $consoleLog)"
# Native stderr must be captured rather than turned into terminating PS errors.
$previousPreference = $ErrorActionPreference
$nativePreference = $PSNativeCommandUseErrorActionPreference
try {
    $ErrorActionPreference = "Continue"
    $PSNativeCommandUseErrorActionPreference = $false
    & $venvPython -u -X faulthandler app.py 2>&1 | Tee-Object -FilePath $consoleLog -Append
    $pythonExit = $LASTEXITCODE
} finally {
    $ErrorActionPreference = $previousPreference
    $PSNativeCommandUseErrorActionPreference = $nativePreference
}
if ($null -eq $pythonExit) { $pythonExit = 1 }
$message = "[$((Get-Date).ToString('o'))] Python exited: code=$pythonExit; elapsed=$((Get-Date) - $start). See studio.log and fault-*.log."
$message | Tee-Object -FilePath $consoleLog -Append
if ($pythonExit -ne 0) { Write-Warning "Studio terminated unexpectedly. $message" }
exit $pythonExit
