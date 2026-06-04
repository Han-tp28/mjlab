MUJOCO_GL=egl uv run play Mjlab-Tracking-Flat-VR-H3-1 \
  --motion-file /home/hantp/mjlab/data/some_motion.npz \
  --checkpoint-file /home/hantp/mjlab/logs/rsl_rl/vr_h3_1_tracking/<RUN_DIR>/model_XXXXX.pt \
  --viewer viser \
  --num-envs 1 \
  --no-terminations True \
  --device cuda:0




cd /home/hantp/mjlab

uv run wandb sync --include-online --sync-tensorboard \
  wandb/run-20260522_165324-5k26xszz




MUJOCO_GL=egl uv run play Mjlab-Tracking-Flat-VR-H3-1 \
  --agent zero \
  --motion-file /home/hantp/mjlab/data/vr_h3_1_easy_npz_height_normalized/230223__dancing_routine_V001_001__A215.npz \
  --viewer viser \
  --num-envs 1 \
  --no-terminations True \
  --device cuda:0
