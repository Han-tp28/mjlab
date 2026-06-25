#!/usr/bin/env bash
set -euo pipefail

cd /home/hantp/mjlab

# Fresh VR M3.1 Pose teleop run with the COMPLETE fix set baked in from iter 0.
# This restarts from scratch on purpose: the resume lineage (yaw_resume* /
# walk_resume*) fixed heading, translation drift, and the curriculum gate, but
# could NOT fix the foot shuffle, because that is a converged-policy local
# optimum that resume cannot escape:
#
#  - At model_40000 the robot tracks heading perfectly (yaw span ~193 deg vs
#    reference 183 deg) but lifts its feet only ~0.015-0.03 m vs the reference
#    ~0.10-0.14 m -- it drags its feet to turn ("let chan").
#  - The swing-clearance rewards are already saturated (~0.985 / 1.0 coarse,
#    ~1.9 / 2.0 fine) WHILE shuffling, because turn-in-place motions plant the
#    foot ~90% of frames (reward ~1.0) and swing only ~10% (reward ~0.49 while
#    shuffling); averaged over the episode even a total shuffle scores 0.985,
#    so there is almost no incentive left to improve.
#  - Policy/mean_std sat at ~0.36 and flat: the policy has converged into the
#    shuffle basin and never explores lifting (early lift attempts risk a fall
#    -> -200 termination -> reinforces "do not lift"). Adding the fine, then
#    the coarse, swing term by resume moved foot lift 0% in 2000 iterations.
#
# Why fresh is different this time (the genuinely new ingredient):
# the ORIGINAL fresh run (translation_swing_fresh) had ONLY the fine swing
# term (std 0.03). At std 0.03 a 4 cm under-lift gives ~exp(-8) ~ 0 reward, so
# from scratch it provides ZERO gradient until the foot already lifts almost
# perfectly -- chicken-and-egg, so even that fresh run shuffled. The new
# motion_feet_swing_clearance_coarse term (std 0.10, weight 1.0) gives a usable
# gradient across the whole 0-10 cm range, shaping foot lift from the very
# start before the policy can lock into a shuffle. Config (env_cfgs.py) already
# has both swing terms plus the heading/translation reward fixes.
#
# Curriculum fixes carried over from the resume diagnosis (CLI flags below):
#  - motion-curriculum-fall-termination-keys restricts the gate's fall_rate to
#    LITERAL falls (fell_over_height/orientation, knee_ground_contact). The old
#    fall_rate counted every non-timeout termination (anchor_xy, ee_body_pos)
#    as a fall, so it hit 0.15-1.0 even while the robot was upright and the
#    gate almost never passed -- growth came only from the force override.
#    Literal-fall rate stayed <1.3% across the entire prior run.
#  - motion-curriculum-max-anchor-pos-error 0.18 -> 0.30: the prior run's
#    error_anchor_pos sat at ~0.21-0.28 in its healthy window (root_pos reward
#    std is 0.6), so 0.18 was stricter than the reward even targets.
#  - force-after-iterations 3000, min-stable-iterations 50: how long a new
#    motion class gets to be learned before more content is forced in. Now that
#    the gate fall_rate is fixed (literal falls only), the gate should pass on
#    its own, so the force override is a safety net rather than the main driver.
#
# Watch early (this is the whole point of restarting):
#  - /tmp/diag_feet_turn.py on a step_rotate clip at iter ~5-10k: robot foot
#    lift should already be CLIMBING off ~0.02 m toward the reference, not
#    pinned. If it is still pinned at ~0.02 m by iter ~15k, the swing reward
#    needs strengthening (raise coarse weight / add swing-phase gating) BEFORE
#    spending more GPU days -- catch it early, do not let it converge.
#  - Log lines should say "gate stable for N/50 iterations", not always
#    "Forcing motion curriculum update".

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
  --agent.run-name vr_m3_1_pose_3271_full_fresh_v2 \
  --agent.resume False \
  --agent.max-iterations 250000 \
  --agent.save-interval 1000 \
  --agent.algorithm.learning-rate 5e-4 \
  --agent.algorithm.entropy-coef 0.008 \
  --agent.algorithm.desired-kl 0.01 \
  --gpu-ids "[0]"
