#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
export HF_HOME="${HF_HOME:-$PWD/models/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:128}"
export CUDA_MODULE_LOADING="${CUDA_MODULE_LOADING:-LAZY}"
export GRADIO_ANALYTICS_ENABLED=false
python app.py
