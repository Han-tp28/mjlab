#!/usr/bin/env bash
set -euo pipefail

cd /home/hantp/mjlab

# Resume the VR M3.1 full-body teleop policy from model_9000 of the swing run,
# with the heading (yaw) fix added. We resume from 9000 (not later) because that
# is the last checkpoint where heading tracking was still healthy:
# error_anchor_rot held ~0.09 through iter 9000, then jumped to ~1.1 rad the
# instant turn-in-place motions entered the curriculum pool and never recovered
# (the policy started shuffling its feet in place instead of turning).
#
# Root cause: heading had no hard constraint. bad_anchor_ori uses projected
# gravity (pitch/roll only, yaw-invariant), so an upright robot facing the wrong
# way never tripped it, and motion_global_root_ori (std 0.4) saturated to a flat
# gradient past ~50 deg of error, so there was nothing to pull heading back.
#
# Fix now baked into the Pose config (env_cfgs.py), resume-compatible (only a new
# termination + a reward weight/std change, no obs/network dim change):
#  1. anchor_yaw termination (bad_anchor_yaw @ 1.0 rad ~57 deg): the hard
#     backstop on heading drift, mirroring anchor_xy for horizontal position.
#  2. motion_global_root_ori std 0.4 -> 0.7 and weight 0.5 -> 0.75: a recovery
#     gradient at large heading errors, so absolute yaw competes with the
#     locally-satisfiable joint/feet terms a shuffle-in-place gait maxes out.
#
# Watch after the pool re-grows past the rotate motions (~iter where pool first
# includes step_rotate clips):
#  - Metrics/motion/error_anchor_rot stays low (~0.1), does NOT jump to ~1.1.
#  - Episode_Termination/anchor_yaw fires a little then trends to ~0 as the
#    policy learns to turn (if it pins high, relax the threshold toward 1.2).
#  - A play rollout on a step_rotate_idle clip actually sweeps through the
#    reference's yaw range instead of stamping in place.

uv run train Mjlab-Tracking-Flat-VR-M3-1-Pose \
  --env.scene.num-envs 4096 \
  --env.commands.motion.motion-file data/vr_m3_1_h3_easy_short_3271.txt \
  --env.commands.motion.smpl-motion-file /home/hantp/Groot-WholeBodyControl/data/smpl_filtered \
  --env.commands.motion.smpl-num-joints 24 \
  --env.commands.motion.smpl-strict-pairing True \
  --env.commands.motion.smpl-y-up True \
  --env.commands.motion.smpl-num-future-frames 10 \
  --env.commands.motion.smpl-dt-future-ref-frames 0.02 \
  --env.commands.motion.motion-curriculum-ordered-loading True \
  --env.commands.motion.motion-pool-mode grow \
  --env.commands.motion.initial-num-load-motions 64 \
  --env.commands.motion.max-num-load-motions 3271 \
  --env.commands.motion.num-new-motions-per-resample 128 \
  --env.commands.motion.motion-replay-fraction 0.75 \
  --env.commands.motion.motion-resample-interval 250 \
  --env.commands.motion.motion-resample-start-iteration 250 \
  --env.commands.motion.motion-resample-unique-until-all-seen True \
  --env.commands.motion.motion-curriculum-gate True \
  --env.commands.motion.motion-curriculum-force-after-iterations 3000 \
  --env.commands.motion.motion-curriculum-min-stable-iterations 50 \
  --env.commands.motion.motion-curriculum-min-mean-episode-length 1200 \
  --env.commands.motion.motion-curriculum-max-fall-rate 0.08 \
  --env.commands.motion.motion-curriculum-max-body-pos-error 0.2 \
  --env.commands.motion.motion-curriculum-max-anchor-pos-error 0.18 \
  --env.commands.motion.motion-curriculum-stage-sizes "(64,96,128,160,192,224,256,320,384,512,768,1024,1536,2048,3072,3271)" \
  --agent.logger tensorboard \
  --agent.run-name vr_m3_1_pose_3271_yaw_resume9k \
  --agent.resume True \
  --agent.load-run 2026-06-23_16-22-00_vr_m3_1_pose_3271_translation_swing_resume5k \
  --agent.load-checkpoint model_9000.pt \
  --agent.max-iterations 250000 \
  --agent.save-interval 1000 \
  --agent.algorithm.learning-rate 5e-4 \
  --agent.algorithm.entropy-coef 0.008 \
  --agent.algorithm.desired-kl 0.01 \
  --gpu-ids "[0]"
