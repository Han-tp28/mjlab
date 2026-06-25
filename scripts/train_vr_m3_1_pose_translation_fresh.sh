#!/usr/bin/env bash
set -euo pipefail

cd /home/hantp/mjlab

# Fresh full-body teleop training for VR M3.1 with the COMPLETE fix set baked in
# from iter 0 (no resume). Resuming kept perturbing the converged policy into
# worse gaits, and the foot reward needed a redesign anyway, so this trains
# everything coherently from scratch.
#
# What is now in the Pose config (env_cfgs.py), all learned from the start:
#  1. Root-translation command fed to the teleop & smpl encoders
#     (anchor_pos_b_multi_future), so the deploy/teleop policy follows the
#     operator's root (walk where you walk) instead of marching in place.
#  2. anchor_xy termination @ 0.8 m + motion_global_root_pos std 0.3 -> 0.6, so
#     horizontal root drift is punished and there is a recovery gradient before
#     the drift compounds. (Previously anchor_pos only checked height -> the
#     policy drifted metres away once walk motions entered the pool.)
#  3. One-sided motion_feet_swing_clearance (w2.0/std0.03) + feet_pos 0.8/0.2,
#     to stop the shuffle: it penalises a swing foot only when it is below the
#     reference foot, concentrating the gradient on the under-lift instead of
#     being diluted by stance time like a symmetric height reward.
#
# Curriculum: ordered idle-first grow + gate, force_after 5000 (gentle safety
# net; not 2500 which forced the pool through a collapse, not None which stalled
# it at 64). The anchor enforcement above makes forced growth safe now.
#
# Watch after a few thousand iters past warmup:
#  - error_anchor_pos stays low (~0.1) even as the pool grows past 384 (walk).
#  - g1_smpl_latent stays ~0.1-0.2 (does not diverge).
#  - foot lift climbs toward the reference (/tmp/diag_feet.py: robot lift ->
#    ~0.059 m) once real walk motions are in the pool.

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
  --agent.run-name vr_m3_1_pose_3271_translation_swing_fresh \
  --agent.resume False \
  --agent.max-iterations 250000 \
  --agent.save-interval 1000 \
  --agent.algorithm.learning-rate 5e-4 \
  --agent.algorithm.entropy-coef 0.008 \
  --agent.algorithm.desired-kl 0.01 \
  --gpu-ids "[0]"
