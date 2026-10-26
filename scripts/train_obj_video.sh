#!/usr/bin/env bash
# First baseline: project configuration, upstream distributed launcher/trainer.
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$project_dir/env_train.sh"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
export VPP2_TRAIN_MODULE=vpp2_obj.train
export VPP2_TORCH_LOAD_MMAP=1
export VPP2_ACCELERATE_CONFIG="$VPP2_REPO_ROOT/configs/robodojo/accelerate.yaml"
exec bash "$VPP2_REPO_ROOT/scripts/train.sh" \
  --config "$project_dir/configs/obj_video_baseline_v1.yaml" "$@"
