from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner

from .env_cfgs import (
  vr_m3_1_flat_tracking_balance_finetune_env_cfg,
  vr_m3_1_flat_tracking_balance_transition_env_cfg,
  vr_m3_1_flat_tracking_env_cfg,
  vr_m3_1_flat_tracking_pose_env_cfg,
  vr_m3_1_flat_tracking_teleop_robust_env_cfg,
)
from .rl_cfg import (
  vr_m3_1_balance_finetune_ppo_runner_cfg,
  vr_m3_1_balance_transition_ppo_runner_cfg,
  vr_m3_1_pose_ppo_runner_cfg,
  vr_m3_1_teleop_robust_ppo_runner_cfg,
  vr_m3_1_tracking_ppo_runner_cfg,
)

register_mjlab_task(
  task_id="Mjlab-Tracking-Flat-VR-M3-1",
  env_cfg=vr_m3_1_flat_tracking_env_cfg(),
  play_env_cfg=vr_m3_1_flat_tracking_env_cfg(play=True),
  rl_cfg=vr_m3_1_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Tracking-Flat-VR-M3-1-No-State-Estimation",
  env_cfg=vr_m3_1_flat_tracking_env_cfg(has_state_estimation=False),
  play_env_cfg=vr_m3_1_flat_tracking_env_cfg(has_state_estimation=False, play=True),
  rl_cfg=vr_m3_1_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Tracking-Flat-VR-M3-1-Pose",
  env_cfg=vr_m3_1_flat_tracking_pose_env_cfg(),
  play_env_cfg=vr_m3_1_flat_tracking_pose_env_cfg(play=True),
  rl_cfg=vr_m3_1_pose_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Tracking-Flat-VR-M3-1-Teleop-Robust",
  env_cfg=vr_m3_1_flat_tracking_teleop_robust_env_cfg(),
  play_env_cfg=vr_m3_1_flat_tracking_teleop_robust_env_cfg(play=True),
  rl_cfg=vr_m3_1_teleop_robust_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Tracking-Flat-VR-M3-1-Balance-Finetune",
  env_cfg=vr_m3_1_flat_tracking_balance_finetune_env_cfg(),
  play_env_cfg=vr_m3_1_flat_tracking_balance_finetune_env_cfg(play=True),
  rl_cfg=vr_m3_1_balance_finetune_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)
register_mjlab_task(
  task_id="Mjlab-Tracking-Flat-VR-M3-1-Balance-Transition",
  env_cfg=vr_m3_1_flat_tracking_balance_transition_env_cfg(),
  play_env_cfg=vr_m3_1_flat_tracking_balance_transition_env_cfg(play=True),
  rl_cfg=vr_m3_1_balance_transition_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)
