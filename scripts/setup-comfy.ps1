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
Copy-Item -LiteralPath (Join-Path $PSScriptRoot "comfy_all2real_nodes.py") -Destination (Join-Path $target "custom_nodes\photo_edit_all2real.py") -Force
if (-not (Test-Path (Join-Path $ggufNodes "__init__.py"))) {
    if (Test-Path $ggufNodes) { throw "ComfyUI-GGUF node folder exists but is incomplete: $ggufNodes" }
    & git clone --depth 1 https://github.com/city96/ComfyUI-GGUF.git $ggufNodes
    if ($LASTEXITCODE -ne 0) { throw "Failed to clone ComfyUI-GGUF nodes." }
}
$dlssNodes = Join-Path $target "custom_nodes\ComfyUI-DLSS5-Enhancer"
if (-not (Test-Path (Join-Path $dlssNodes "__init__.py"))) {
    if (Test-Path $dlssNodes) { throw "DLSS5 node folder exists but is incomplete: $dlssNodes" }
    & git clone --depth 1 https://github.com/Blueforcer/ComfyUI-DLSS5-Enhancer.git $dlssNodes
    if ($LASTEXITCODE -ne 0) { throw "Failed to clone DLSS5 nodes." }
}
# The supplied character-sheet graph records these revisions. Only new clones
# are checked out: never reset/pull an existing user-managed node installation.
$sheetNodePins = @(
    @{ Name = "ComfyUI-KJNodes"; Repository = "https://github.com/kijai/ComfyUI-KJNodes.git"; Revision = "d3cfe21625e5170126ce06fbfcfe1d88108688c3" },
    @{ Name = "ComfyUI-DeGrid"; Repository = "https://github.com/lunaaispace-eng/ComfyUI-DeGrid.git"; Revision = "5699bc33f71e1be12523fdea105cc9c2abfe1cd0" }
)
foreach ($pin in $sheetNodePins) {
    $nodeTarget = Join-Path $target ("custom_nodes\" + $pin.Name)
    if (-not (Test-Path (Join-Path $nodeTarget "__init__.py"))) {
        if (Test-Path $nodeTarget) { throw "Character-sheet node folder exists but is incomplete: $nodeTarget" }
        & git clone --no-checkout $pin.Repository $nodeTarget
        if ($LASTEXITCODE -ne 0) { throw "Failed to clone $($pin.Name)." }
        & git -C $nodeTarget checkout --detach $pin.Revision
        if ($LASTEXITCODE -ne 0) { throw "Failed to check out pinned $($pin.Name) revision." }
        if (-not (Test-Path (Join-Path $nodeTarget "__init__.py"))) { throw "Pinned $($pin.Name) lacks __init__.py." }
    } else {
        Write-Host "Keeping existing $($pin.Name) unchanged; graph reference revision: $($pin.Revision)."
    }
}
# QwenImage21Cache/TextEncodeQwenImage21/TextGenerate are ComfyUI core nodes,
# verified in local /object_info. No third cache-node dependency or model download.
& $python -m pip install -r (Join-Path $target "requirements.txt")
if ($LASTEXITCODE -ne 0) { throw "ComfyUI requirements failed to install in the project venv." }
$ostrisRequirements = Join-Path $ostrisNodes "requirements.txt"
if (Test-Path $ostrisRequirements) {
    & $python -m pip install -r $ostrisRequirements
    if ($LASTEXITCODE -ne 0) { throw "Ostris Krea2 edit requirements failed to install in the project venv." }
}
& $python -m pip install -r (Join-Path $ggufNodes "requirements.txt")
if ($LASTEXITCODE -ne 0) { throw "ComfyUI-GGUF requirements failed to install in the project venv." }
& $python -m pip install -r (Join-Path $dlssNodes "requirements.txt")
if ($LASTEXITCODE -ne 0) { throw "DLSS5 Python requirements failed to install in the project venv." }
foreach ($pin in $sheetNodePins) {
    $nodeRequirements = Join-Path $target ("custom_nodes\" + $pin.Name + "\requirements.txt")
    if (Test-Path $nodeRequirements) {
        & $python -m pip install -r $nodeRequirements
        if ($LASTEXITCODE -ne 0) { throw "$($pin.Name) requirements failed to install in the project venv." }
    }
}
# Do not invoke install_runtime.py or download/run proprietary binaries here.
Write-Host "Optional DLSS5: Python nodes installed only. Restart ComfyUI to register the V3 nodes."
Write-Host "Runtime is explicit/manual: review upstream licenses and install_runtime.py instructions yourself."
Write-Host "Supply DLSS 5 Visual Enhancer v3.0 runtime via runtime_dir (folder containing nvngx.dll), DLSS5_RUNTIME_DIR or pack config.json."
& $python -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable after ComfyUI dependency installation'; print('ComfyUI shares CUDA PyTorch:', torch.__version__)"
if ($LASTEXITCODE -ne 0) { throw "The shared Python environment lost CUDA support." }
Write-Host "ComfyUI installed into the project venv at $target."
Write-Host "Next: python scripts\download_models.py --comfy-krea"
Write-Host "Optional full BF16 Qwen sheets: .\scripts\download-models.ps1 -Preset qwen-character-sheet"
Write-Host "Restart ComfyUI after installing nodes; Qwen cache/encoding/generation require current core nodes."
Write-Host "For remix, manually place Krea2-Remix_Patreon.safetensors in vendor\ComfyUI\models\loras; no Remix weights are downloaded by setup."
Write-Host "Then: .\scripts\run.ps1 -ComfyUI"
