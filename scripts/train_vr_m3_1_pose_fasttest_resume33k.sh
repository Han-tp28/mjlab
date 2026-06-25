#!/usr/bin/env bash
set -euo pipefail

cd /home/hantp/mjlab

# DIAGNOSTIC TEST (not a production run): does the reward+termination fix cure
# the "robot can't walk fast" collapse? Resume from model_33000 of the
# latentcap run -- the last healthy checkpoint, right before the fast
# walk-forward batch entered the pool at ~384 and collapsed training.
#
# Instead of waiting ~36k iters for the curriculum to grow back to the danger
# zone, this START the pool AT 384 (initial = max = 384) so the robot faces the
# exact motions that killed it (Neutral_walk_forward / Loop_Forward_Walk +
# step_rotate_idle_135, sustained ~1.0-1.3 m/s) from iteration 0. Pool is fixed
# (no growth) to isolate the test.
#
# The fix being tested (config, env_cfgs.py -- resume-compatible, no
# obs/network/weight change):
#  1. motion_body_lin_vel weight 1.0 -> 2.0, std 1.0 -> 0.6. Velocity error is
#     informative at ANY position lag, so this is the non-saturating "move at
#     the reference's forward speed" signal. Diagnosis: the robot capped at
#     ~0.7 m/s and FROZE (speed -> 0.003) when it fell behind, because the only
#     forward pull (motion_global_root_pos) saturates past ~1.5 m. 59% of the
#     dataset needs sustained >0.8 m/s.
#  2. motion_global_root_pos std 0.6 -> 1.5: secondary catch-up gradient out to
#     ~3 m instead of vanishing at 1.5 m.
#  3. anchor_xy 0.8 -> 2.0 m: the tight bound killed the transient lag while the
#     robot was legitimately trying to keep up with a fast walk (rollout: it
#     crossed 0.8 m at step 66 and was killed before it could catch up). 0.8 m
#     is unwinnable for the 59% of walks needing >0.8 m/s.
#
# IS THE DIAGNOSIS RIGHT? Watch over the first few thousand iters:
#  - Episode_Termination/anchor_xy must NOT spike to thousands (it was 19000+ at
#    collapse). A few hundred is fine.
#  - Loss/g1_smpl_latent / Policy/mean_std must stay bounded (latent < ~15, std
#    ~0.5), NOT run away (latent 7000, std 0.97) -- that blow-up was the symptom.
#  - Train/mean_reward must stay up (~300+), NOT crash to ~25.
#  - DECISIVE: /tmp/diag_translate.py on Neutral_walk_forward over successive
#    checkpoints -- robot forward speed should climb off ~0.1-0.2 m/s toward the
#    reference's ~0.9 m/s, and stop freezing. If it still freezes, the reward
#    de-saturation was not enough and the diagnosis/fix needs rethink.
#
# If it holds, the production run is a FRESH run with these fixes + a
# speed-aware curriculum (don't dump all the fast walks in one batch).

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
  --env.commands.motion.initial-num-load-motions 384 \
  --env.commands.motion.max-num-load-motions 384 \
  --env.commands.motion.motion-replay-fraction 0.75 \
  --env.commands.motion.motion-resample-interval 250 \
  --env.commands.motion.motion-resample-start-iteration 250 \
  --env.commands.motion.motion-resample-unique-until-all-seen True \
  --env.commands.motion.motion-curriculum-gate False \
  --agent.logger tensorboard \
  --agent.run-name vr_m3_1_pose_3271_fasttest_resume33k \
  --agent.resume True \
  --agent.load-run 2026-06-24_20-28-19_vr_m3_1_pose_3271_resume19k_latentcap \
  --agent.load-checkpoint model_33000.pt \
  --agent.max-iterations 250000 \
  --agent.save-interval 500 \
  --agent.algorithm.learning-rate 5e-4 \
  --agent.algorithm.entropy-coef 0.008 \
  --agent.algorithm.desired-kl 0.01 \
  --gpu-ids "[0]"
