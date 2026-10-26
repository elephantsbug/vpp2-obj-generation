#!/usr/bin/env bash
# Source this file to select the isolated VPP2 training environment.
export VPP2_PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export VPP2_REPO_ROOT="$VPP2_PROJECT_ROOT/third_party/vpp2"
export VPP2_STAGE2_VIDEO="$VPP2_PROJECT_ROOT/weights/checkpoints_video/vpp2-video-stage2-17f.pth"
export VPP2_WAN_ROOT="$VPP2_PROJECT_ROOT/weights/checkpoints/Wan2.1-I2V-14B-480P"
export VIRTUAL_ENV="$VPP2_PROJECT_ROOT/.venv-train"
export PYTHON_BIN="$VIRTUAL_ENV/bin/python"
export CUDA_HOME=/data2/niurc/vpp2-obj-generation/.venv-train/lib/python3.10/site-packages/nvidia/cu13
export PATH="$VIRTUAL_ENV/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH=/data2/niurc/vpp2-obj-generation/.venv-train/lib/python3.10/site-packages/nvidia/cu13/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export PYTHONPATH="$VPP2_PROJECT_ROOT/src:$VPP2_REPO_ROOT/src"
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export TORCH_EXTENSIONS_DIR="$VPP2_PROJECT_ROOT/.cache/torch_extensions_train"
