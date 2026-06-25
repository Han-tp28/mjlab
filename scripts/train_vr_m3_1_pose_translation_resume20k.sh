#!/usr/bin/env bash
set -euo pipefail

cd /home/hantp/mjlab

# Resume the VR M3.1 Pose translation-following run from model_20000.pt, the
# last checkpoint before the policy drifted off the root-translation command
# once the pool grew past ~384 motions (anchor error went from 0.10 at iter
# 20000 to 1.85+ by iter 21000 and never recovered, settling into an
# in-place/free-drift basin even though the teleop/smpl encoders can see where
# the root should go -- see commands.py anchor_pos_b_multi_future).
#
# Root cause: the anchor_pos termination only checks height
# (bad_anchor_pos_z_only), so horizontal drift was free, and
# motion_global_root_pos's std=0.3 reward gradient vanishes past ~0.6 m, so
# once the robot drifted that far there was no recovery signal. Both are now
# fixed in env_cfgs.py (anchor_xy termination @ 0.5m, std 0.3 -> 0.6). Neither
# change touches observations or the network, so resuming is safe -- no need
# to retrain the ~7h already spent reaching iter 20000.
#
# Sets --motion-curriculum-force-after-iterations to 6000 (was 2500). 2500 was
# too fast and forced the pool to 3271 by iter 37500 while root tracking was
# already collapsing; None (gate-only) was too slow and stalled the pool at the
# initial 64 because the anchor_xy termination kept the fall-rate above the
# gate threshold. 6000 is a gentle safety net: the gate grows the pool when the
# policy is healthy, and the now-working anchor enforcement (anchor_xy + wide
# root std) prevents the forced steps from collapsing tracking the way 2500 did.
#
# Bundled reward fix (also resume-compatible): foot tracking was too loose
# (motion_feet_pos w0.6/std0.3 and no swing-height term), so the policy learned
# a shuffle -- dragging the swing foot ~2 cm vs the reference ~6 cm -- to
# turn/step. This was confirmed identical across g1/teleop/smpl, i.e. a
# reward-shaping issue, not an encoder one. env_cfgs.py now uses
# motion_feet_pos 1.0/0.18 plus a tight motion_feet_height term (1.0/0.05) to
# force a clean swing. Watch the foot-lift gap with /tmp/diag_feet.py after a
# few thousand iters (ref lifts ~0.059 m; robot should climb from ~0.02 toward
# that).

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
  --env.commands.motion.motion-curriculum-force-after-iterations 5000 \
  --env.commands.motion.motion-curriculum-min-stable-iterations 100 \
  --env.commands.motion.motion-curriculum-min-mean-episode-length 1200 \
  --env.commands.motion.motion-curriculum-max-fall-rate 0.08 \
  --env.commands.motion.motion-curriculum-max-body-pos-error 0.2 \
  --env.commands.motion.motion-curriculum-max-anchor-pos-error 0.18 \
  --env.commands.motion.motion-curriculum-stage-sizes "(64,96,128,160,192,224,256,320,384,512,768,1024,1536,2048,3072,3271)" \
  --agent.logger tensorboard \
  --agent.run-name vr_m3_1_pose_3271_translation_resume20k \
  --agent.resume True \
  --agent.load-run 2026-06-22_18-02-08_vr_m3_1_pose_3271_translation_fresh \
  --agent.load-checkpoint model_20000.pt \
  --agent.max-iterations 250000 \
  --agent.save-interval 1000 \
  --agent.algorithm.learning-rate 5e-4 \
  --agent.algorithm.entropy-coef 0.008 \
  --agent.algorithm.desired-kl 0.01 \
  --gpu-ids "[0]"
