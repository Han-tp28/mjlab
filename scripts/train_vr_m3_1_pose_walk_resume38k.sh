#!/usr/bin/env bash
set -euo pipefail

cd /home/hantp/mjlab

# Resume from model_38000 of the yaw_resume15k run -- the last checkpoint
# before the curriculum forced Neutral_walk_forward (sustained ~0.8 m/s, ~8 m
# straight walks) into the pool at iter ~39000. The policy had never had to
# sustain continuous long-distance translation before (prior motions were
# short/local), and anchor_xy (0.8 m hard drift termination) killed every
# single rollout on these clips instantly: fail_rate hit 1.000 with ZERO
# timeouts. mean_reward crashed 376 -> 29 and mean_episode_length 1448 -> 181,
# flat with no recovery for 2500+ iterations.
#
# This is the translation analog of the earlier yaw_resume9k collapse: a hard
# threshold + curriculum force-introducing a genuinely new motion class before
# the policy can learn it. It already happened once before in this lineage
# (see train_vr_m3_1_pose_translation_resume20k.sh, force_after 2500 -> 5000),
# but force_after silently reset to 3000 when training restarted fresh for the
# swing/feet fix, so the same collapse recurred.
#
# Fix #1 is pacing only, no reward/termination/observation change (resume-
# compatible): motion-curriculum-force-after-iterations 3000 -> 6000, and
# motion-curriculum-min-stable-iterations 50 -> 150, so the gate requires a
# longer stable window before growing and the force-override gives a new
# motion class roughly 2x longer to be learned before more content is forced
# in on top of it.
#
# Fix #2 (bundled, also resume-compatible -- new reward term, no obs/arch
# change): a headless rollout of model_38000 on a turn-in-place motion
# (smpl encoder) showed clean heading tracking (robot yaw span 166 deg vs
# reference 183 deg) but the feet barely lift (~0.02-0.03 m vs reference
# ~0.10-0.14 m) -- a shuffle. The existing motion_feet_swing_clearance term
# (std 0.03) was tuned against walking-scale under-lift (~0.04 m) and
# saturates flat at turn-scale under-lift: measured swing reward mean 0.85 but
# min 0.00016 on this clip, i.e. zero gradient exactly at the swing peaks.
# Added motion_feet_swing_clearance_coarse (same function, std 0.10, weight
# 1.0) as a second, wider-std copy -- mirrors the existing arm joint pos
# fine/coarse pattern -- so under-lifts in the turning-motion range keep a
# usable pull without loosening the tight term that fixed the walking
# shuffle.
#
# Fix #3 (bundled): the curriculum gate was NEVER passing naturally in this
# run -- every pool growth so far happened via force_after_iterations, not the
# gate. Root cause: the runner's fall_rate counts EVERY non-timeout
# termination (anchor_xy, ee_body_pos, ...) as a "fall", not just literal
# falls. Recomputed fall_rate from this run's own termination logs using only
# fell_over_height/fell_over_orientation/knee_ground_contact: it stays under
# 1.3% in EVERY bucket of the entire run, including the iter-39000 collapse --
# the robot was never actually falling, just drifting past anchor_xy while
# upright. The old (broad) fall_rate hit 0.15-0.21 in several "healthy"
# windows purely from that conflation, which is why a 150-iteration stable
# window (let alone a shorter one) was nearly unreachable. Added
# motion_curriculum_fall_termination_keys (commands.py) so fall_rate can be
# restricted to literal falls; wired here via CLI (tuple[str, ...] -> tyro
# wants space-separated values, not the "(a,b,c)" syntax used for the int
# stage-sizes tuple).
#
# Also raised motion-curriculum-max-anchor-pos-error 0.18 -> 0.30: this run's
# own error_anchor_pos sat at ~0.21-0.28 through the entire healthy
# iter-16000-38000 window (root_pos reward std is 0.6, so 0.18 was stricter
# than what the reward recipe even asks for) and was the other chronic
# blocker. Both changes only affect when/whether the curriculum grows the
# pool, not the policy/reward/observations -- resume-compatible.
#
# Watch after resume:
#  - Train/mean_episode_length recovers back toward ~1400+ instead of staying
#    flat near ~180.
#  - motion_failure_report_*.tsv: Neutral_walk_forward fail_rate should trend
#    down from 1.000 instead of staying pinned there.
#  - Log lines should start saying "gate stable for N/150 iterations" instead
#    of always "Forcing motion curriculum update".
#  - /tmp/diag_feet_turn.py on a step_rotate clip: robot foot lift climbs from
#    ~0.02-0.03 m toward the reference's ~0.10-0.14 m over successive
#    checkpoints.

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
  --env.commands.motion.motion-curriculum-force-after-iterations 6000 \
  --env.commands.motion.motion-curriculum-min-stable-iterations 150 \
  --env.commands.motion.motion-curriculum-min-mean-episode-length 1200 \
  --env.commands.motion.motion-curriculum-max-fall-rate 0.08 \
  --env.commands.motion.motion-curriculum-max-body-pos-error 0.2 \
  --env.commands.motion.motion-curriculum-max-anchor-pos-error 0.30 \
  --env.commands.motion.motion-curriculum-fall-termination-keys \
    fell_over_height fell_over_orientation knee_ground_contact \
  --env.commands.motion.motion-curriculum-stage-sizes "(64,96,128,160,192,224,256,320,384,512,768,1024,1536,2048,3072,3271)" \
  --agent.logger tensorboard \
  --agent.run-name vr_m3_1_pose_3271_walk_resume38k \
  --agent.resume True \
  --agent.load-run 2026-06-23_23-30-59_vr_m3_1_pose_3271_yaw_resume15k \
  --agent.load-checkpoint model_38000.pt \
  --agent.max-iterations 250000 \
  --agent.save-interval 1000 \
  --agent.algorithm.learning-rate 5e-4 \
  --agent.algorithm.entropy-coef 0.008 \
  --agent.algorithm.desired-kl 0.01 \
  --gpu-ids "[0]"
