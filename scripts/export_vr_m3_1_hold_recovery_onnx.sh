#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 /path/to/model_xxxxxx.pt" >&2
  exit 2
fi

cd /home/hantp/mjlab

uv run python scripts/export_actor_only_onnx_for_deploy.py \
  --checkpoint "$1" \
  --output /home/hantp/vinrobotics_mjlab/mjlab/config/policy/motion_tracking/keyboard/exported/policy.onnx \
  --input-layout full
