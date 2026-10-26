#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONDONTWRITEBYTECODE=1
exec "${PYTHON_BIN:-/opt/anaconda3/envs/starvla_env/bin/python}" "$PROJECT_ROOT/scripts/enhance_prompts_two_step.py" "$@"
