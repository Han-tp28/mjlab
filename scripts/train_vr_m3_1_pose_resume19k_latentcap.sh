#!/usr/bin/env bash
set -euo pipefail

cd /home/hantp/mjlab

# Resume from model_19000 of the full_fresh_v2 run -- the last healthy
# checkpoint before the iter-20000 blow-up. This is the SAME config as
# full_fresh_v2 plus one algorithm fix (aux_loss_cap, now baked into
# rl_cfg.py): resume-compatible, no obs/network/weight change.
#
# What happened at iter 20000 (full_fresh_v2): a batch of 64 hard motions
# entered the curriculum at once -- Neutral_walk_forward + Loop_Forward_Walk
# (sustained translation) plus step_rotate_idle_135 (135 deg turns). The
# policy could not handle them immediately, the value function was surprised
# (value loss 0.17 -> 1.99), and crucially the cross-encoder latent-distillation
# losses EXPLODED ~100x: g1_smpl_latent 2.3 -> 220, teleop_smpl_latent
# 1.3 -> 185. With coef 1.0 and no cap, those ~220 terms dominated every
# gradient step and dragged the shared trunk into chasing a target computed on
# a flailing policy -> action std exploded 0.55 -> 0.97, reward collapsed
# 400 -> 25, and anchor_xy then fired on ~every env because the now-broken
# policy drifted on ALL motions (not just the new ones). It never recovered.
#
# The latent explosion was the AMPLIFIER that turned "a few hard motions"
# (which should just be locally high error, recoverable) into a global,
# unrecoverable collapse. Fix: aux_loss_cap=8.0 (rl_cfg.py) bounds each latent
# term's contribution to ~8 (healthy values are ~1-2.5, so normal training is
# untouched) -- a spike is scaled by min(1, 8/|loss|), keeping its gradient
# direction but capping its magnitude so it can no longer dominate.
#
# Watch when the pool re-grows past ~320-384 motions (where it blew up before):
#  - Loss/g1_smpl_latent and Loss/teleop_smpl_latent must NOT run away past
#    ~8-15; they should bounce and settle, not climb to 200+.
#  - Policy/mean_std must stay ~0.5, NOT climb toward ~1.0.
#  - Train/mean_reward dips on the hard batch are fine; it must RECOVER within
#    a few hundred iters, not flatline at ~25.
#  - If the latent losses still run away despite the cap, lower aux_loss_cap
#    (e.g. 4.0) or slow the curriculum (force-after 6000, smaller batch) so the
#    hard motions enter more gradually.

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
  --env.commands.motion.motion-curriculum-max-anchor-pos-error 0.30 \
  --env.commands.motion.motion-curriculum-fall-termination-keys "fell_over_height,fell_over_orientation,knee_ground_contact" \
  --env.commands.motion.motion-curriculum-stage-sizes "(64,96,128,160,192,224,256,320,384,512,768,1024,1536,2048,3072,3271)" \
  --agent.logger tensorboard \
  --agent.run-name vr_m3_1_pose_3271_resume19k_latentcap \
  --agent.resume True \
  --agent.load-run 2026-06-24_10-42-15_vr_m3_1_pose_3271_full_fresh_v2 \
  --agent.load-checkpoint model_19000.pt \
  --agent.max-iterations 250000 \
  --agent.save-interval 1000 \
  --agent.algorithm.learning-rate 5e-4 \
  --agent.algorithm.entropy-coef 0.008 \
  --agent.algorithm.desired-kl 0.01 \
  --gpu-ids "[0]"
