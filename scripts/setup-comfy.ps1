[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
$python = Join-Path $PWD ".venv\Scripts\python.exe"
$target = Join-Path $PWD "vendor\ComfyUI"
if (-not (Test-Path $python)) { throw "Run .\scripts\setup.ps1 first to create Python 3.11 .venv." }
& $python -c "import sys; assert sys.version_info[:2] == (3, 11), 'Use the project Python 3.11 venv'"
if ($LASTEXITCODE -ne 0) { throw "Python 3.11 environment is required." }
if (-not (Test-Path (Join-Path $target "main.py"))) {
    if (Test-Path $target) { throw "ComfyUI target exists but lacks main.py: $target" }
    & git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git $target
    if ($LASTEXITCODE -ne 0) { throw "Failed to clone ComfyUI." }
}
$nodes = Join-Path $target "custom_nodes\comfyui-krea2edit"
if (-not (Test-Path (Join-Path $nodes "__init__.py"))) {
    if (Test-Path $nodes) { throw "Krea2Edit node folder exists but is incomplete: $nodes" }
    & git clone --depth 1 https://github.com/lbouaraba/comfyui-krea2edit.git $nodes
    if ($LASTEXITCODE -ne 0) { throw "Failed to clone Krea2Edit nodes." }
}
$ostrisNodes = Join-Path $target "custom_nodes\ComfyUI-Krea2-Ostris-Edit"
if (-not (Test-Path (Join-Path $ostrisNodes "__init__.py"))) {
    if (Test-Path $ostrisNodes) { throw "Ostris Krea2 edit node folder exists but is incomplete: $ostrisNodes" }
    & git clone --depth 1 https://github.com/ostris/ComfyUI-Krea2-Ostris-Edit.git $ostrisNodes
    if ($LASTEXITCODE -ne 0) { throw "Failed to clone Ostris Krea2 edit nodes." }
}
$ggufNodes = Join-Path $target "custom_nodes\ComfyUI-GGUF"
if (-not (Test-Path (Join-Path $ggufNodes "__init__.py"))) {
    if (Test-Path $ggufNodes) { throw "ComfyUI-GGUF node folder exists but is incomplete: $ggufNodes" }
    & git clone --depth 1 https://github.com/city96/ComfyUI-GGUF.git $ggufNodes
    if ($LASTEXITCODE -ne 0) { throw "Failed to clone ComfyUI-GGUF nodes." }
}
& $python -m pip install -r (Join-Path $target "requirements.txt")
if ($LASTEXITCODE -ne 0) { throw "ComfyUI requirements failed to install in the project venv." }
$ostrisRequirements = Join-Path $ostrisNodes "requirements.txt"
if (Test-Path $ostrisRequirements) {
    & $python -m pip install -r $ostrisRequirements
    if ($LASTEXITCODE -ne 0) { throw "Ostris Krea2 edit requirements failed to install in the project venv." }
}
& $python -m pip install -r (Join-Path $ggufNodes "requirements.txt")
if ($LASTEXITCODE -ne 0) { throw "ComfyUI-GGUF requirements failed to install in the project venv." }
& $python -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable after ComfyUI dependency installation'; print('ComfyUI shares CUDA PyTorch:', torch.__version__)"
if ($LASTEXITCODE -ne 0) { throw "The shared Python environment lost CUDA support." }
Write-Host "ComfyUI installed into the project venv at $target."
Write-Host "Next: python scripts\download_models.py --comfy-krea"
Write-Host "For remix, manually place Krea2-Remix_Patreon.safetensors in vendor\ComfyUI\models\loras; no Remix weights are downloaded by setup."
Write-Host "Then: .\scripts\run.ps1 -ComfyUI"
