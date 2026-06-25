#!/usr/bin/env bash
set -euo pipefail

cd /home/hantp/mjlab

uv run tensorboard \
  --logdir logs/rsl_rl/vr_m3_1_pose \
  --host 0.0.0.0 \
  --port 6006
