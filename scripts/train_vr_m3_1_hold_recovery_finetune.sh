#!/usr/bin/env bash
set -euo pipefail

cd /home/hantp/mjlab

uv run train Mjlab-Tracking-Flat-VR-M3-1 \
  --env.commands.motion.motion-file data/vr_m3_1_hold_recovery_stand_motion_4096.txt \
  --env.commands.motion.motion-pool-mode grow \
  --env.commands.motion.initial-num-load-motions 4096 \
  --env.commands.motion.max-num-load-motions 4096 \
  --env.commands.motion.motion-resample-interval 1000000 \
  --env.commands.motion.motion-curriculum-min-stable-iterations 100 \
  --env.commands.motion.motion-curriculum-min-mean-episode-length 1400 \
  --env.commands.motion.motion-curriculum-max-fall-rate 0.03 \
  --env.commands.motion.motion-curriculum-max-body-pos-error 0.18 \
  --env.commands.motion.motion-curriculum-max-body-rot-error 0.40 \
  --env.commands.motion.motion-curriculum-max-anchor-pos-error 0.20 \
  --env.commands.motion.motion-curriculum-max-anchor-rot-error 0.30 \
  --env.commands.motion.motion-curriculum-max-joint-pos-error 1.2 \
  --agent.run-name vr_m3_1_hold_recovery_stand_motion_finetune_from_190000 \
  --agent.resume True \
  --agent.load-run 2026-06-01_09-57-37_vr_m3_1_offline_clean5824_no_crouch_grow256_resume_easy95000 \
  --agent.load-checkpoint model_190000.pt \
  --agent.max-iterations 50000 \
  --gpu-ids '[0]'
