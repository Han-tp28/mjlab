from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner

from .env_cfgs import vr_h3_1_flat_tracking_env_cfg
from .rl_cfg import vr_h3_1_tracking_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Tracking-Flat-VR-H3-1",
  env_cfg=vr_h3_1_flat_tracking_env_cfg(),
  play_env_cfg=vr_h3_1_flat_tracking_env_cfg(play=True),
  rl_cfg=vr_h3_1_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Tracking-Flat-VR-H3-1-No-State-Estimation",
  env_cfg=vr_h3_1_flat_tracking_env_cfg(has_state_estimation=False),
  play_env_cfg=vr_h3_1_flat_tracking_env_cfg(has_state_estimation=False, play=True),
  rl_cfg=vr_h3_1_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)
