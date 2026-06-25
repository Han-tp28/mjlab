#!/usr/bin/env bash
set -euo pipefail

cd /home/hantp/mjlab

# Full-data full-body teleop training for VR M3.1.
#
# Trains the Pose SMPL 3-encoder policy (g1/teleop/smpl -> FSQ token -> decoder)
# on the FULL 120K valid NPZ+SMPL pairs, with an idle-first ORDERED, gated
# STREAMING curriculum (clean motions only, no teleop recovery clips).
#
# Streaming = the working pool stays bounded at 4096 motions (fits the 32 GB
# RTX 5090), but cycles through ALL 120,587 ordered motions over training:
# grow 128 -> 4096 from the front (all idle), then each gated resample keeps
# 75% replay and swaps in the next 128 unseen ordered clips (walk, then hard)
# until every motion has been seen. (Plain "grow" would stop at 4096 and never
# reach the rest of the playlist -- that is why we use streaming here.)
#
# Why ordered idle-first (the "best of Groot" recipe):
#   - The pool starts with idle/stand/stop/recover motions so the policy masters
#     standing balance and holding a pose BEFORE harder motions are introduced.
#   - The gate only expands the pool when the policy is stable (fall_rate low,
#     mean episode length high), so hard motions never swamp a shaky base.
#   - This is what makes the robot stay upright and stop mid-motion without
#     falling, in addition to the stability rewards already in the Pose config
#     (alive, termination, flat_orientation, root_height_asymmetric) and the
#     push_robot + base_com mass domain randomization.
#
# The idle-first ordered playlist data/vr_m3_1_pose_full_curriculum.txt is built
# (clean-only, no teleop) by sorting data/vr_m3_1_duplicate_motion_paths_valid.txt
# with the curriculum prioritization (balance/idle -> walk -> other -> hard):
#
#   uv run python - <<'PY'
#   from pathlib import Path
#   from mjlab.scripts.build_vr_m3_1_teleop_curriculum import (
#       _read_clean_playlist, _prioritize_clean_entries, _validate_playlist_entries)
#   out = Path("data/vr_m3_1_pose_full_curriculum.txt")
#   ordered = _prioritize_clean_entries(
#       _read_clean_playlist(Path("data/vr_m3_1_duplicate_motion_paths_valid.txt")))
#   _validate_playlist_entries(ordered, out.parent)
#   out.write_text("\n".join(ordered) + "\n")
#   PY

uv run train Mjlab-Tracking-Flat-VR-M3-1-Pose \
  --env.commands.motion.motion-file data/vr_m3_1_pose_full_curriculum.txt \
  --env.commands.motion.smpl-motion-file /home/hantp/Groot-WholeBodyControl/data/smpl_filtered \
  --env.commands.motion.smpl-num-joints 24 \
  --env.commands.motion.smpl-strict-pairing True \
  --env.commands.motion.smpl-y-up True \
  --env.commands.motion.smpl-num-future-frames 10 \
  --env.commands.motion.smpl-dt-future-ref-frames 0.02 \
  --env.commands.motion.motion-curriculum-ordered-loading True \
  --env.commands.motion.motion-pool-mode streaming \
  --env.commands.motion.initial-num-load-motions 128 \
  --env.commands.motion.max-num-load-motions 4096 \
  --env.commands.motion.num-new-motions-per-resample 128 \
  --env.commands.motion.motion-replay-fraction 0.75 \
  --env.commands.motion.motion-replay-failure-weighted True \
  --env.commands.motion.motion-resample-interval 250 \
  --env.commands.motion.motion-resample-start-iteration 250 \
  --env.commands.motion.motion-resample-unique-until-all-seen True \
  --env.commands.motion.motion-curriculum-gate True \
  --env.commands.motion.motion-curriculum-min-stable-iterations 250 \
  --env.commands.motion.motion-curriculum-min-mean-episode-length 1200 \
  --env.commands.motion.motion-curriculum-max-fall-rate 0.05 \
  --env.commands.motion.motion-curriculum-max-body-pos-error 0.40 \
  --agent.logger tensorboard \
  --agent.run-name vr_m3_1_pose_full_curriculum_tb \
  --agent.resume False \
  --agent.max-iterations 150000 \
  --agent.save-interval 1000 \
  --agent.algorithm.learning-rate 5e-4 \
  --agent.algorithm.entropy-coef 0.008 \
  --agent.algorithm.desired-kl 0.01 \
  --gpu-ids "[0]"
