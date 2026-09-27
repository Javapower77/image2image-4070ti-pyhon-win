#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
# CUDA 12.8 wheels support the Windows/Linux low-VRAM profile with a current NVIDIA driver.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e '.[dev]'
mkdir -p models outputs
printf '\nSetup complete. Activate with: source .venv/bin/activate\n'
