#!/usr/bin/env bash
set -euo pipefail

cd /home/hantp/mjlab

uv run train Mjlab-Tracking-Flat-VR-M3-1-Pose \
  --env.commands.motion.motion-file data/vr_m3_1_duplicate_motion_paths_valid.txt \
  --env.commands.motion.smpl-motion-file /home/hantp/Groot-WholeBodyControl/data/smpl_filtered \
  --env.commands.motion.smpl-num-joints 24 \
  --env.commands.motion.smpl-strict-pairing True \
  --env.commands.motion.initial-num-load-motions 4096 \
  --env.commands.motion.max-num-load-motions 4096 \
  --env.commands.motion.num-new-motions-per-resample 1024 \
  --env.commands.motion.motion-pool-mode streaming \
  --env.commands.motion.motion-replay-fraction 0.75 \
  --env.commands.motion.motion-resample-interval 250 \
  --env.commands.motion.motion-curriculum-ordered-loading False \
  --env.commands.motion.motion-curriculum-gate False \
  --agent.logger tensorboard \
  --agent.run-name vr_m3_1_pose_paired_smpl_duplicate_tb \
  --agent.resume False \
  --agent.max-iterations 150000 \
  --agent.save-interval 1000 \
  --agent.algorithm.learning-rate 5e-4 \
  --agent.algorithm.entropy-coef 0.008 \
  --agent.algorithm.desired-kl 0.01 \
  --gpu-ids "[0]"
