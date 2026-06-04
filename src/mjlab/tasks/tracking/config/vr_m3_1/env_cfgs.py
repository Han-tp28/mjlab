"""VR M3.1 flat tracking environment configurations."""

from mjlab.asset_zoo.robots import VR_M3_1_ACTION_SCALE, get_vr_m3_1_robot_cfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.observation_manager import ObservationGroupCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.tracking import mdp
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg

MOTION_BODY_NAMES = (
  "pelvis",
  "left_hip_roll_link",
  "left_hip_yaw_link",
  "left_knee_pitch_link",
  "left_ankle_roll_link",
  "right_hip_roll_link",
  "right_hip_yaw_link",
  "right_knee_pitch_link",
  "right_ankle_roll_link",
  "waist_yaw_link",
  "left_shoulder_roll_link",
  "left_elbow_pitch_link",
  "left_wrist_yaw_link",
  "right_shoulder_roll_link",
  "right_elbow_pitch_link",
  "right_wrist_yaw_link",
)

LEFT_ARM_JOINT_NAMES = (
  "left_shoulder_pitch_joint",
  "left_shoulder_roll_joint",
  "left_shoulder_yaw_joint",
  "left_elbow_pitch_joint",
  "left_wrist_yaw_joint",
  "left_wrist_roll_joint",
  "left_wrist_pitch_joint",
)

RIGHT_ARM_JOINT_NAMES = (
  "right_shoulder_pitch_joint",
  "right_shoulder_roll_joint",
  "right_shoulder_yaw_joint",
  "right_elbow_pitch_joint",
  "right_wrist_yaw_joint",
  "right_wrist_roll_joint",
  "right_wrist_pitch_joint",
)

LEG_JOINT_NAMES = (
  "left_hip_pitch_joint",
  "left_hip_roll_joint",
  "left_hip_yaw_joint",
  "left_knee_pitch_joint",
  "left_ankle_pitch_joint",
  "left_ankle_roll_joint",
  "right_hip_pitch_joint",
  "right_hip_roll_joint",
  "right_hip_yaw_joint",
  "right_knee_pitch_joint",
  "right_ankle_pitch_joint",
  "right_ankle_roll_joint",
)

LOWER_BODY_NAMES = (
  "pelvis",
  "left_hip_roll_link",
  "left_hip_yaw_link",
  "left_knee_pitch_link",
  "right_hip_roll_link",
  "right_hip_yaw_link",
  "right_knee_pitch_link",
  "waist_yaw_link",
)

FEET_BODY_NAMES = (
  "left_ankle_roll_link",
  "right_ankle_roll_link",
)

VR_M3_1_TRACKING_ACTION_SCALE = {
  **VR_M3_1_ACTION_SCALE,
  ".*shoulder_pitch_joint": 0.18,
  ".*shoulder_roll_joint": 0.16,
  ".*shoulder_yaw_joint": 0.16,
  ".*elbow_pitch_joint": 0.16,
  ".*wrist_yaw_joint": 0.12,
  ".*wrist_roll_joint": 0.10,
  ".*wrist_pitch_joint": 0.10,
}


def vr_m3_1_flat_tracking_env_cfg(
  has_state_estimation: bool = True,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create VR M3.1 flat terrain tracking configuration."""
  cfg = make_tracking_env_cfg()

  cfg.seed = 42
  cfg.episode_length_s = 30.0
  cfg.sim.njmax = 1024
  cfg.sim.nconmax = 512
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nan_guard.enabled = False
  cfg.scene.num_envs = 4096
  cfg.scene.entities = {"robot": get_vr_m3_1_robot_cfg()}

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
  joint_pos_action.scale = VR_M3_1_TRACKING_ACTION_SCALE

  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  motion_cmd.motion_file = "/home/hantp/mjlab/data/vr_m3_1_easy_train_4096.txt"
  motion_cmd.anchor_body_name = "pelvis"
  motion_cmd.body_names = MOTION_BODY_NAMES
  motion_cmd.initial_num_load_motions = 64
  motion_cmd.num_new_motions_per_resample = 128
  motion_cmd.max_num_load_motions = 4096
  motion_cmd.motion_pool_mode = "grow"
  motion_cmd.motion_resample_replacement = True
  motion_cmd.motion_resample_unique_until_all_seen = True
  motion_cmd.motion_resample_interval = 250
  motion_cmd.motion_resample_start_iteration = 250
  motion_cmd.motion_curriculum_gate = True
  motion_cmd.motion_curriculum_min_stable_iterations = 0
  motion_cmd.motion_curriculum_force_after_iterations = 5000
  motion_cmd.motion_curriculum_ordered_loading = True
  motion_cmd.motion_curriculum_stage_sizes = (
    64,
    96,
    128,
    160,
    192,
    224,
    256,
    320,
    384,
    512,
    768,
    1024,
    1536,
    2048,
    3072,
    4096,
  )
  motion_cmd.motion_curriculum_min_mean_episode_length = 470.0
  motion_cmd.motion_curriculum_max_fall_rate = 0.08
  motion_cmd.motion_curriculum_max_body_pos_error = 0.2
  motion_cmd.motion_curriculum_max_body_rot_error = 0.45
  motion_cmd.motion_curriculum_max_anchor_pos_error = 0.18
  motion_cmd.motion_curriculum_max_anchor_rot_error = 0.35
  motion_cmd.motion_curriculum_max_joint_pos_error = 1.5
  motion_cmd.num_future_frames = 10
  motion_cmd.dt_future_ref_frames = 0.1

  # Re-enabled push_robot for sim2real robustness
  cfg.events["foot_friction"].params[
    "asset_cfg"
  ].geom_names = r"^(left|right)_ankle_roll_link_collision_[1-9][0-9]*$"
  cfg.events["base_com"].params["asset_cfg"].body_names = ("waist_yaw_link",)

  cfg.rewards["motion_global_root_pos"].weight = 1.0
  cfg.rewards["motion_global_root_pos"].params["std"] = 0.3
  cfg.rewards["motion_global_root_ori"].weight = 0.5
  cfg.rewards["motion_global_root_ori"].params["std"] = 0.4
  cfg.rewards["motion_body_pos"].weight = 1.0
  cfg.rewards["motion_body_pos"].params["std"] = 0.3
  cfg.rewards["motion_body_ori"].weight = 1.0
  cfg.rewards["motion_body_ori"].params["std"] = 0.4
  cfg.rewards["motion_body_lin_vel"].weight = 1.0
  cfg.rewards["motion_body_lin_vel"].params["std"] = 1.0
  cfg.rewards["motion_body_lin_vel"].params.pop("body_names", None)
  cfg.rewards["motion_body_ang_vel"].weight = 1.0
  cfg.rewards["motion_body_ang_vel"].params["std"] = 3.14
  cfg.rewards["motion_body_ang_vel"].params.pop("body_names", None)
  cfg.rewards["motion_lower_body_pos"] = RewardTermCfg(
    func=mdp.motion_relative_body_position_error_exp,
    weight=2.0,
    params={
      "command_name": "motion",
      "std": 0.15,
      "body_names": LOWER_BODY_NAMES,
    },
  )
  cfg.rewards["motion_lower_body_ori"] = RewardTermCfg(
    func=mdp.motion_relative_body_orientation_error_exp,
    weight=1.0,
    params={
      "command_name": "motion",
      "std": 0.4,
      "body_names": LOWER_BODY_NAMES,
    },
  )
  cfg.rewards["motion_feet_pos"] = RewardTermCfg(
    func=mdp.motion_relative_body_position_error_exp,
    weight=0.6,
    params={
      "command_name": "motion",
      "std": 0.3,
      "body_names": FEET_BODY_NAMES,
    },
  )
  cfg.rewards["motion_feet_ori"] = RewardTermCfg(
    func=mdp.motion_relative_body_orientation_error_exp,
    weight=0.5,
    params={
      "command_name": "motion",
      "std": 0.4,
      "body_names": FEET_BODY_NAMES,
    },
  )
  cfg.rewards["motion_right_arm_joint_pos"] = RewardTermCfg(
    func=mdp.motion_joint_position_error_exp,
    weight=1.0,
    params={
      "command_name": "motion",
      "std": 0.15,
      "joint_names": RIGHT_ARM_JOINT_NAMES,
    },
  )
  cfg.rewards["motion_right_arm_joint_pos_coarse"] = RewardTermCfg(
    func=mdp.motion_joint_position_error_exp,
    weight=0.5,
    params={
      "command_name": "motion",
      "std": 0.35,
      "joint_names": RIGHT_ARM_JOINT_NAMES,
    },
  )
  cfg.rewards["motion_left_arm_joint_pos"] = RewardTermCfg(
    func=mdp.motion_joint_position_error_exp,
    weight=1.0,
    params={
      "command_name": "motion",
      "std": 0.15,
      "joint_names": LEFT_ARM_JOINT_NAMES,
    },
  )
  cfg.rewards["motion_left_arm_joint_pos_coarse"] = RewardTermCfg(
    func=mdp.motion_joint_position_error_exp,
    weight=0.5,
    params={
      "command_name": "motion",
      "std": 0.35,
      "joint_names": LEFT_ARM_JOINT_NAMES,
    },
  )
  cfg.rewards["motion_leg_joint_pos"] = RewardTermCfg(
    func=mdp.motion_joint_position_error_exp,
    weight=0.2,
    params={
      "command_name": "motion",
      "std": 0.5,
      "joint_names": LEG_JOINT_NAMES,
    },
  )
  cfg.rewards["feet_acc"] = RewardTermCfg(
    func=mdp.joint_acc_l2,
    weight=-5.0e-7,
    params={"asset_cfg": mdp.SceneEntityCfg("robot", joint_names=(".*ankle.*",))},
  )
  cfg.rewards["action_rate_l2"].weight = -0.05
  cfg.rewards["joint_limit"].weight = -10.0
  cfg.rewards["self_collisions"].weight = -10.0

  cfg.terminations["anchor_pos"].params["threshold"] = 0.45
  cfg.terminations["anchor_ori"].params["threshold"] = 0.7
  cfg.terminations["ee_body_pos"].params["body_names"] = (
    "left_ankle_roll_link",
    "right_ankle_roll_link",
  )
  cfg.terminations["ee_body_pos"].params["threshold"] = 0.5

  cfg.metrics = {}

  cfg.viewer.body_name = "waist_yaw_link"

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
