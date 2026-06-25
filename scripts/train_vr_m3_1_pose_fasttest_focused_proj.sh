#!/usr/bin/env bash
set -euo pipefail

cd /home/hantp/mjlab

# FOCUSED DIAGNOSTIC TEST: does reference projection (+ the velocity reward)
# make sustained fast walking learnable? Small 196-motion set concentrated on
# the exact failing behaviour (136 fast forward-walks incl. all
# Neutral_walk_forward / Loop_Forward_Walk, + 60 idle/turn for a stability
# base), so we get an answer in hours, not days.
#
# Resume from model_33000 (the last healthy checkpoint of the latentcap run) so
# the policy already has balance + slow walking; projection only changes the
# VALUES of the anchor observations (not their dimension), so the weights load
# fine and just fine-tune to the projected target.
#
# The fix under test (config, env_cfgs.py):
#  - reference_projection_enabled (NEW for the pose recipe; mirrors
#    balance_finetune). Clamps the tracked reference anchor to ~0.18 m around
#    the robot, so the robot chases a near "carrot" it can always track
#    in-distribution and walk toward, instead of an absolute reference that
#    walks 7 m away -> unbounded drift -> death -> out-of-distribution freeze.
#  - motion_body_lin_vel weight 2.0 / std 0.6: non-saturating "match the
#    reference forward speed" gradient.
#  - anchor_xy 2.0 m: now just a safety net (projection keeps the anchor error
#    bounded, so it should rarely fire).
#
# DECISIVE (the whole point): /tmp/diag_translate.py on a Neutral_walk_forward
# clip over successive checkpoints. Robot forward speed must climb off
# ~0.1-0.2 m/s toward the reference's ~0.9 m/s and STOP freezing. Also:
#  - Episode_Termination/anchor_xy must stay low (projection bounds the error).
#  - Loss/g1_smpl_latent + Policy/mean_std must stay bounded (no blow-up).
#  - Train/mean_reward recovers/climbs after the brief projection adaptation.
# If the robot finally walks fast here -> diagnosis right, projection is the
# fix -> do a full fresh run with a speed-aware curriculum. If it still
# freezes -> projection is not the answer either.

uv run train Mjlab-Tracking-Flat-VR-M3-1-Pose \
  --env.scene.num-envs 4096 \
  --env.commands.motion.motion-file data/vr_m3_1_fasttest_focused.txt \
  --env.commands.motion.smpl-motion-file /home/hantp/Groot-WholeBodyControl/data/smpl_filtered \
  --env.commands.motion.smpl-num-joints 24 \
  --env.commands.motion.smpl-strict-pairing True \
  --env.commands.motion.smpl-y-up True \
  --env.commands.motion.smpl-num-future-frames 10 \
  --env.commands.motion.smpl-dt-future-ref-frames 0.02 \
  --env.commands.motion.motion-curriculum-ordered-loading False \
  --env.commands.motion.motion-pool-mode grow \
  --env.commands.motion.initial-num-load-motions 196 \
  --env.commands.motion.max-num-load-motions 196 \
  --env.commands.motion.motion-replay-fraction 0.75 \
  --env.commands.motion.motion-resample-interval 250 \
  --env.commands.motion.motion-resample-start-iteration 250 \
  --env.commands.motion.motion-curriculum-gate False \
  --agent.logger tensorboard \
  --agent.run-name vr_m3_1_pose_fasttest_focused_proj \
  --agent.resume True \
  --agent.load-run 2026-06-24_20-28-19_vr_m3_1_pose_3271_resume19k_latentcap \
  --agent.load-checkpoint model_33000.pt \
  --agent.max-iterations 250000 \
  --agent.save-interval 250 \
  --agent.algorithm.learning-rate 5e-4 \
  --agent.algorithm.entropy-coef 0.008 \
  --agent.algorithm.desired-kl 0.01 \
  --gpu-ids "[0]"
