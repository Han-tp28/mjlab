#!/usr/bin/env bash
set -euo pipefail

cd /home/hantp/mjlab

# Resume from model_15000 of the yaw_resume9k run -- the last checkpoint before
# the anchor_yaw termination collapsed training. With std-0.4 root_ori the policy
# had abandoned turning; the fix is the WIDENED heading reward
# (motion_global_root_ori std 0.4 -> 0.7, weight 0.5 -> 0.75), which tracked
# heading well (root_ori reward ~0.72, error_anchor_rot ~0.09) through iter 15000.
#
# The hard yaw termination that was also added has been REMOVED: a robot
# legitimately lagging a fast 188 deg sweep transiently exceeds any threshold
# that also catches the shuffle failure (steady ~57 deg), so the two cannot be
# separated by instantaneous yaw error. At 1.0 rad it fired on nearly every env
# the moment larger turns entered the pool at iter 15000 and collapsed
# mean_episode_length from ~1485 to ~260 and reward from ~430 to ~58. Heading is
# now constrained by the reward alone (no observation or network change, so this
# is resume-compatible).
#
# Watch after the pool re-grows past the rotate clips:
#  - Train/mean_episode_length stays high (~1400+), does NOT crash to ~260.
#  - Metrics/motion/error_anchor_rot stays ~0.1, does not jump to ~1.1.
#  - A headless yaw-span diag on a step_rotate clip climbs toward the reference's
#    188 deg over successive checkpoints (teleop already reached ~115 deg at 13k).

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
  --agent.run-name vr_m3_1_pose_3271_yaw_resume15k \
  --agent.resume True \
  --agent.load-run 2026-06-23_20-25-45_vr_m3_1_pose_3271_yaw_resume9k \
  --agent.load-checkpoint model_15000.pt \
  --agent.max-iterations 250000 \
  --agent.save-interval 1000 \
  --agent.algorithm.learning-rate 5e-4 \
  --agent.algorithm.entropy-coef 0.008 \
  --agent.algorithm.desired-kl 0.01 \
  --gpu-ids "[0]"
