#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-/opt/anaconda3/envs/starvla_env/bin/python}"
if [[ ! -x "$python_bin" ]]; then
  python_bin=python3
fi
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

exec "$python_bin" "$project_root/scripts/convert_observer.py" \
  --vpp2-root "$project_root/third_party/vpp2" \
  --output-root "$project_root/data/robotwin_observer_vpp2" \
  "$@"
