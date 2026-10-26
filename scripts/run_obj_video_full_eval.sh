#!/usr/bin/env bash
set -euo pipefail
cd /data2/niurc/vpp2-obj-generation
source env_train.sh
export VPP2_TORCH_LOAD_MMAP=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
exec "$PYTHON_BIN" scripts/launch_obj_video_full_eval.py "$@"
