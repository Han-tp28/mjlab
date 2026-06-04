"""VR H3.1 flat tracking environment configurations."""

from mjlab.asset_zoo.robots import (
  VR_H3_1_ACTION_SCALE,
  get_vr_h3_1_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.observation_manager import ObservationGroupCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg


def vr_h3_1_flat_tracking_env_cfg(
  has_state_estimation: bool = True,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create VR H3.1 flat terrain tracking configuration."""
  cfg = make_tracking_env_cfg()

  cfg.sim.njmax = 1024
  cfg.sim.nconmax = 512
  cfg.scene.num_envs = 4096
  cfg.scene.entities = {"robot": get_vr_h3_1_robot_cfg()}

  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (self_collision_cfg,)

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = VR_H3_1_ACTION_SCALE

  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  motion_cmd.anchor_body_name = "pelvis"
  motion_cmd.initial_num_load_motions = 10
  motion_cmd.num_new_motions_per_resample = 54
  motion_cmd.max_num_load_motions = 4_096
  motion_cmd.motion_pool_mode = "grow"
  motion_cmd.motion_resample_replacement = True
  motion_cmd.motion_resample_unique_until_all_seen = True
  motion_cmd.motion_resample_interval = 250
  motion_cmd.motion_resample_start_iteration = 250
  motion_cmd.motion_curriculum_gate = True
  motion_cmd.motion_curriculum_ordered_loading = True
  motion_cmd.motion_curriculum_stage_sizes = (10, 64, 128, 256, 512, 1024, 2048, 4096)
  motion_cmd.motion_curriculum_min_mean_episode_length = 400.0
  motion_cmd.motion_curriculum_max_fall_rate = 0.10
  motion_cmd.motion_curriculum_max_body_pos_error = 0.18
  motion_cmd.motion_curriculum_max_body_rot_error = 0.45
  motion_cmd.motion_curriculum_max_anchor_pos_error = 0.18
  motion_cmd.motion_curriculum_max_anchor_rot_error = 0.35
  motion_cmd.motion_curriculum_max_joint_pos_error = 1.0
  motion_cmd.num_future_frames = 10
  motion_cmd.dt_future_ref_frames = 0.1
  motion_cmd.body_names = (
    "pelvis",
    "left_hip_roll_link",
    "left_knee_pitch_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_pitch_link",
    "right_ankle_roll_link",
    "waist_roll_link",
    "left_shoulder_roll_link",
    "left_elbow_pitch_link",
    "left_wrist_pitch_link",
    "right_shoulder_roll_link",
    "right_elbow_pitch_link",
    "right_wrist_pitch_link",
  )

  # First get clean motion imitation working; add pushes back for robustness later.
  cfg.events.pop("push_robot", None)
  cfg.events["foot_friction"].params[
    "asset_cfg"
  ].geom_names = r"^(left|right)_ankle_roll_link_collision_[1-9][0-9]*$"
  cfg.events["base_com"].params["asset_cfg"].body_names = ("waist_roll_link",)

  cfg.rewards["motion_global_root_pos"].weight = 0.5
  cfg.rewards["motion_global_root_pos"].params["std"] = 0.3
  cfg.rewards["motion_global_root_ori"].weight = 0.5
  cfg.rewards["motion_global_root_ori"].params["std"] = 0.4
  cfg.rewards["action_rate_l2"].weight = -0.05

  cfg.terminations["anchor_pos"].params["threshold"] = 0.45
  cfg.terminations["anchor_ori"].params["threshold"] = 0.7
  cfg.terminations["ee_body_pos"].params["body_names"] = (
    "left_ankle_roll_link",
    "right_ankle_roll_link",
  )
  cfg.terminations["ee_body_pos"].params["threshold"] = 0.5

  cfg.viewer.body_name = "waist_roll_link"

  if not has_state_estimation:
    new_actor_terms = {
      k: v
      for k, v in cfg.observations["actor"].terms.items()
      if k not in ["motion_anchor_pos_b", "base_lin_vel"]
    }
    cfg.observations["actor"] = ObservationGroupCfg(
      terms=new_actor_terms,
      concatenate_terms=True,
      enable_corruption=True,
    )

  if play:
    cfg.episode_length_s = int(1e9)

    cfg.observations["actor"].enable_corruption = False
    motion_cmd.pose_range = {}
    motion_cmd.velocity_range = {}
    motion_cmd.sampling_mode = "start"
    motion_cmd.motion_resample_interval = None

  return cfg
