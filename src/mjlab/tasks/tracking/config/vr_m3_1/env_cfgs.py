"""VR M3.1 flat tracking environment configurations."""

from mjlab.asset_zoo.robots import VR_M3_1_ACTION_SCALE, get_vr_m3_1_robot_cfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.observation_manager import (
  ObservationGroupCfg,
  ObservationTermCfg,
)
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.tracking import mdp
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

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

POSE_BODY_NAMES = (
  "pelvis",
  "left_hip_pitch_link",
  "left_hip_roll_link",
  "left_hip_yaw_link",
  "left_knee_pitch_link",
  "left_ankle_pitch_link",
  "left_ankle_roll_link",
  "right_hip_pitch_link",
  "right_hip_roll_link",
  "right_hip_yaw_link",
  "right_knee_pitch_link",
  "right_ankle_pitch_link",
  "right_ankle_roll_link",
  "waist_yaw_link",
  "left_shoulder_pitch_link",
  "left_shoulder_roll_link",
  "left_shoulder_yaw_link",
  "left_elbow_pitch_link",
  "left_wrist_yaw_link",
  "left_wrist_roll_link",
  "left_wrist_pitch_link",
  "right_shoulder_pitch_link",
  "right_shoulder_roll_link",
  "right_shoulder_yaw_link",
  "right_elbow_pitch_link",
  "right_wrist_yaw_link",
  "right_wrist_roll_link",
  "right_wrist_pitch_link",
  "head_yaw_link",
  "head_pitch_link",
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

SMPL_WRIST_JOINT_NAMES = (
  "left_wrist_roll_joint",
  "right_wrist_roll_joint",
  "left_wrist_pitch_joint",
  "right_wrist_pitch_joint",
  "left_wrist_yaw_joint",
  "right_wrist_yaw_joint",
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

KNEE_BODY_NAMES = (
  "left_knee_pitch_link",
  "right_knee_pitch_link",
)

UPPER_BODY_NAMES = (
  "waist_yaw_link",
  "left_shoulder_roll_link",
  "left_elbow_pitch_link",
  "left_wrist_yaw_link",
  "right_shoulder_roll_link",
  "right_elbow_pitch_link",
  "right_wrist_yaw_link",
)

TELEOP_ROBUST_STAGE_SIZES = (
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

POSE_MOTION_FILE = "data/vr_m3_1_duplicate_motion_paths_valid.txt"
POSE_SMPL_MOTION_FILE = "/home/hantp/Groot-WholeBodyControl/data/smpl_filtered"


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


def _set_actor_observation_delay(
  cfg: ManagerBasedRlEnvCfg,
  term_name: str,
  max_lag: int,
  hold_prob: float = 0.0,
  update_period: int = 0,
) -> None:
  term = cfg.observations["actor"].terms[term_name]
  term.delay_min_lag = 0
  term.delay_max_lag = max_lag
  term.delay_hold_prob = hold_prob
  term.delay_update_period = update_period


def _configure_teleop_sensor_latency(cfg: ManagerBasedRlEnvCfg) -> None:
  _set_actor_observation_delay(
    cfg,
    "command",
    max_lag=3,
    hold_prob=0.8,
    update_period=5,
  )
  _set_actor_observation_delay(cfg, "motion_anchor_pos_b", max_lag=2)
  _set_actor_observation_delay(cfg, "motion_anchor_ori_b", max_lag=2)
  _set_actor_observation_delay(cfg, "base_lin_vel", max_lag=2)
  _set_actor_observation_delay(cfg, "base_ang_vel", max_lag=2)
  _set_actor_observation_delay(cfg, "joint_pos", max_lag=2)
  _set_actor_observation_delay(cfg, "joint_vel", max_lag=2)

  robot_cfg = cfg.scene.entities["robot"]
  assert robot_cfg.articulation is not None
  for actuator in robot_cfg.articulation.actuators:
    actuator.delay_min_lag = 0
    actuator.delay_max_lag = 2


def _configure_pose_only_observations(
  cfg: ManagerBasedRlEnvCfg,
  enable_actor_corruption: bool,
) -> None:
  target_pose_params = {
    "command_name": "motion",
    "body_names": POSE_BODY_NAMES,
  }
  actor_terms = {
    "target_body_pos_local": ObservationTermCfg(
      func=mdp.motion_body_pos_local,
      params=target_pose_params,
    ),
    "motion_anchor_ori_b": ObservationTermCfg(
      func=mdp.motion_anchor_ori_b,
      params={"command_name": "motion"},
      noise=Unoise(n_min=-0.05, n_max=0.05),
    ),
    "base_ang_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_ang_vel"},
      noise=Unoise(n_min=-0.2, n_max=0.2),
    ),
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel,
      noise=Unoise(n_min=-0.01, n_max=0.01),
      params={"biased": True},
    ),
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel,
      noise=Unoise(n_min=-0.5, n_max=0.5),
    ),
    "actions": ObservationTermCfg(func=mdp.last_action),
  }
  critic_terms = {
    "target_body_pos_local": ObservationTermCfg(
      func=mdp.motion_body_pos_local,
      params=target_pose_params,
    ),
    "motion_anchor_pos_b": ObservationTermCfg(
      func=mdp.motion_anchor_pos_b,
      params={"command_name": "motion"},
    ),
    "motion_anchor_ori_b": ObservationTermCfg(
      func=mdp.motion_anchor_ori_b,
      params={"command_name": "motion"},
    ),
    "body_pos": ObservationTermCfg(
      func=mdp.robot_body_pos_b,
      params={"command_name": "motion"},
    ),
    "body_ori": ObservationTermCfg(
      func=mdp.robot_body_ori_b,
      params={"command_name": "motion"},
    ),
    "base_lin_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_lin_vel"},
    ),
    "base_ang_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_ang_vel"},
    ),
    "joint_pos": ObservationTermCfg(func=mdp.joint_pos_rel),
    "joint_vel": ObservationTermCfg(func=mdp.joint_vel_rel),
    "actions": ObservationTermCfg(func=mdp.last_action),
  }
  cfg.observations["actor"] = ObservationGroupCfg(
    terms=actor_terms,
    concatenate_terms=True,
    enable_corruption=enable_actor_corruption,
  )
  proprioception_terms = {
    "base_ang_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_ang_vel"},
      noise=Unoise(n_min=-0.2, n_max=0.2),
    ),
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel,
      noise=Unoise(n_min=-0.01, n_max=0.01),
      params={"biased": True},
    ),
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel,
      noise=Unoise(n_min=-0.5, n_max=0.5),
    ),
    "actions": ObservationTermCfg(
      func=mdp.last_action_padded,
      params={"target_dim": 29},
    ),
    "projected_gravity": ObservationTermCfg(
      func=mdp.projected_gravity,
      noise=Unoise(n_min=-0.02, n_max=0.02),
    ),
  }
  tokenizer_terms = {
    "encoder_index": ObservationTermCfg(
      func=mdp.motion_encoder_index, params={"command_name": "motion"}
    ),
    "compliance": ObservationTermCfg(
      func=mdp.motion_compliance, params={"command_name": "motion"}
    ),
  }
  actor_g1_terms = {
    "command_multi_future": ObservationTermCfg(
      func=mdp.motion_command_multi_future,
      params={"command_name": "motion"},
      noise=Unoise(n_min=-0.03, n_max=0.03),
    ),
    "motion_anchor_ori_b_multi_future": ObservationTermCfg(
      func=mdp.motion_anchor_ori_b_multi_future,
      params={"command_name": "motion"},
      noise=Unoise(n_min=-0.03, n_max=0.03),
    ),
  }
  actor_teleop_terms = {
    "teleop_lower_body_joint_pos_multi_future": ObservationTermCfg(
      func=mdp.motion_joint_pos_multi_future_select,
      params={"command_name": "motion", "joint_names": LEG_JOINT_NAMES},
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "teleop_3point_local_target": ObservationTermCfg(
      func=mdp.motion_body_pos_local_select,
      params={
        "command_name": "motion",
        "body_names": ("left_wrist_yaw_link", "right_wrist_yaw_link", "head_yaw_link"),
      },
      noise=Unoise(n_min=-0.03, n_max=0.03),
    ),
    "teleop_3point_local_orn_target": ObservationTermCfg(
      func=mdp.motion_body_ori_local_select,
      params={
        "command_name": "motion",
        "body_names": ("left_wrist_yaw_link", "right_wrist_yaw_link", "head_yaw_link"),
      },
      noise=Unoise(n_min=-0.03, n_max=0.03),
    ),
    "motion_anchor_ori_b": ObservationTermCfg(
      func=mdp.motion_anchor_ori_b, params={"command_name": "motion"}
    ),
    # Translation command (future root pos in robot-anchor frame), the same
    # signal g1 gets inside command_multi_future. Without it the teleop encoder
    # only sees orientation and pose, so it marches in place; with it the robot
    # follows the operator's root (walk/jump).
    "motion_anchor_pos_b_multi_future": ObservationTermCfg(
      func=mdp.motion_anchor_pos_b_multi_future,
      params={"command_name": "motion"},
      noise=Unoise(n_min=-0.03, n_max=0.03),
    ),
  }
  actor_smpl_terms = {
    "smpl_joints_multi_future_local": ObservationTermCfg(
      func=mdp.motion_smpl_joints_multi_future_local,
      params={"command_name": "motion"},
      noise=Unoise(n_min=-0.05, n_max=0.05),
    ),
    "smpl_root_ori_b_multi_future": ObservationTermCfg(
      func=mdp.motion_smpl_root_ori_b_multi_future,
      params={"command_name": "motion"},
      noise=Unoise(n_min=-0.05, n_max=0.05),
    ),
    "joint_pos_multi_future_wrist_for_smpl": ObservationTermCfg(
      func=mdp.motion_joint_pos_multi_future_for_smpl,
      params={
        "command_name": "motion",
        "joint_names": SMPL_WRIST_JOINT_NAMES,
      },
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    # Translation command (future root pos in robot-anchor frame): the SMPL
    # joints are root-local (orientation/translation stripped), so without this
    # the smpl encoder cannot know where to move. This makes the deploy encoder
    # follow the operator's root translation (the core of full-body teleop).
    "motion_anchor_pos_b_multi_future": ObservationTermCfg(
      func=mdp.motion_anchor_pos_b_multi_future,
      params={"command_name": "motion"},
      noise=Unoise(n_min=-0.05, n_max=0.05),
    ),
  }
  cfg.observations["critic"] = ObservationGroupCfg(
    terms=critic_terms,
    concatenate_terms=True,
    enable_corruption=False,
  )
  cfg.observations["tokenizer"] = ObservationGroupCfg(
    terms=tokenizer_terms,
    concatenate_terms=True,
    enable_corruption=False,
  )
  cfg.observations["proprioception"] = ObservationGroupCfg(
    terms=proprioception_terms,
    concatenate_terms=True,
    enable_corruption=enable_actor_corruption,
    history_length=10,
    flatten_history_dim=True,
  )
  cfg.observations["actor_g1"] = ObservationGroupCfg(
    terms=actor_g1_terms,
    concatenate_terms=True,
    enable_corruption=enable_actor_corruption,
  )
  cfg.observations["actor_teleop"] = ObservationGroupCfg(
    terms=actor_teleop_terms,
    concatenate_terms=True,
    enable_corruption=enable_actor_corruption,
  )
  cfg.observations["actor_smpl"] = ObservationGroupCfg(
    terms=actor_smpl_terms,
    concatenate_terms=True,
    enable_corruption=enable_actor_corruption,
  )


def _add_pose_contact_sensors(cfg: ManagerBasedRlEnvCfg) -> None:
  knee_ground_cfg = ContactSensorCfg(
    name="knee_ground_contact",
    primary=ContactMatch(
      mode="body",
      pattern=r"^(left_knee_pitch_link|right_knee_pitch_link)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    history_length=2,
  )
  cfg.scene.sensors = (*(cfg.scene.sensors or ()), knee_ground_cfg)


def _configure_pose_rewards_and_terminations(cfg: ManagerBasedRlEnvCfg) -> None:
  # Tracking reward recipe copied from the proven easy-short 3295
  # vr_m3_1_tracking run. Only terms shared with the Pose task are copied;
  # Pose-only rewards and terminations remain configured below. The previous Pose
  # recipe tracked loosely across many motions
  # because its common tracking terms were substantially softer.
  # Re-enabling the full set restores crisp many-motion tracking while keeping
  # the teleop 3-encoder (g1/teleop/smpl) architecture unchanged.
  pose_tracking_reward_recipe = {
    # Widen std 0.6 -> 1.5 so a multi-metre position lag still has a catch-up
    # gradient. At std 0.6 a >1.5 m lag gives ~0 reward (flat), so once the robot
    # falls behind on a sustained fast walk it gets no signal to catch up and
    # freezes (marches in place). 1.5 keeps a usable pull out to ~3 m. The
    # non-saturating "move at the reference speed" signal is motion_body_lin_vel
    # below; this is the secondary "catch up to the reference location" pull.
    "motion_global_root_pos": (1.0, 1.5),
    # Heading needs a recovery gradient, not just a hard termination: at std 0.4
    # a ~57 deg heading error gives ~0.002 reward (flat gradient) so the policy
    # never gets pulled back toward the reference yaw. Widen std 0.4 -> 0.7 (a
    # 57 deg error now gives ~0.13, a clear pull) and bump weight 0.5 -> 0.75 so
    # absolute heading competes with the locally-satisfiable joint/feet terms a
    # shuffle-in-place gait can max out.
    "motion_global_root_ori": (0.75, 0.7),
    "motion_body_pos": (1.0, 0.3),
    "motion_body_ori": (1.0, 0.4),
    # The key fix for sustained fast walking. Unlike position tracking (which
    # saturates once the robot is metres behind), velocity error is informative
    # at ANY lag: a robot standing still while the reference walks at ~1 m/s has
    # a ~1 m/s velocity error every step, so this term keeps pulling it to move
    # at the reference's forward speed even when it has fallen far behind. 59% of
    # the dataset needs sustained >0.8 m/s but the policy capped at ~0.7 m/s and
    # froze when behind; strengthen weight 1.0 -> 2.0 and tighten std 1.0 -> 0.6
    # so matching the reference speed strongly beats the balance-comfort of
    # standing/marching in place.
    "motion_body_lin_vel": (2.0, 0.6),
    "motion_body_ang_vel": (1.0, 3.14),
    "motion_lower_body_pos": (2.0, 0.15),
    "motion_lower_body_ori": (1.0, 0.4),
    # Feet were only loosely tracked (w 0.6 / std 0.3): a 4 cm under-lift gave
    # ~0.96 reward, so the policy learned a minimal-effort shuffle (dragging the
    # swing foot ~2 cm instead of the reference ~6 cm). Mildly tighten XY/3D foot
    # placement; the swing phase itself is enforced by the one-sided
    # motion_feet_swing_clearance term added below (a symmetric height reward is
    # diluted by stance time and does not fix the shuffle).
    "motion_feet_pos": (0.8, 0.2),
    "motion_feet_ori": (0.5, 0.4),
    "motion_right_arm_joint_pos": (1.0, 0.15),
    "motion_right_arm_joint_pos_coarse": (0.5, 0.35),
    "motion_left_arm_joint_pos": (1.0, 0.15),
    "motion_left_arm_joint_pos_coarse": (0.5, 0.35),
    "motion_leg_joint_pos": (0.2, 0.5),
  }
  for reward_name, (weight, std) in pose_tracking_reward_recipe.items():
    if reward_name in cfg.rewards:
      cfg.rewards[reward_name].weight = weight
      cfg.rewards[reward_name].params["std"] = std

  # One-sided swing-clearance: penalise a foot only when it is below the
  # reference foot height. Unlike a symmetric height reward (which averages to
  # ~0.9 even while shuffling because stance frames dominate), this is ~0 during
  # stance and only fires on the under-lift that causes the shuffle, with a
  # tight std for a strong "lift the swing foot" gradient.
  cfg.rewards["motion_feet_swing_clearance"] = RewardTermCfg(
    func=mdp.motion_feet_swing_clearance_error_exp,
    weight=2.0,
    params={
      "command_name": "motion",
      "std": 0.03,
      "body_names": FEET_BODY_NAMES,
    },
  )
  # Turn-in-place motions lift the feet much higher than walking (~0.10-0.14 m
  # vs ~0.06 m), so once the robot under-lifts by more than a few std (0.03) the
  # fine term above saturates flat (observed min reward ~0.0002 on a rotate
  # clip even though the mean stays ~0.85, the swing frames are too rare to show
  # up there) and gives no pull toward lifting higher -- the robot shuffles its
  # feet to turn instead of stepping. Mirrors the arm joint pos
  # fine/coarse pattern: a second, wider-std copy of the same term keeps a
  # usable gradient at turn-scale under-lifts without loosening the tight term
  # that fixed the walking shuffle.
  cfg.rewards["motion_feet_swing_clearance_coarse"] = RewardTermCfg(
    func=mdp.motion_feet_swing_clearance_error_exp,
    weight=1.0,
    params={
      "command_name": "motion",
      "std": 0.10,
      "body_names": FEET_BODY_NAMES,
    },
  )

  cfg.rewards["alive"] = RewardTermCfg(func=mdp.is_alive, weight=1.0)
  cfg.rewards["termination"] = RewardTermCfg(func=mdp.is_terminated, weight=-200.0)
  cfg.rewards["flat_orientation"] = RewardTermCfg(
    func=mdp.flat_orientation_l2,
    weight=-2.0,
    params={"asset_cfg": mdp.SceneEntityCfg("robot")},
  )
  # Plan A: drop the single dominant local-body-position term in favor of the
  # dense recipe above (motion_body_pos + motion_lower_body_pos already
  # constrain position). Kept defined at weight 0 so it can be re-enabled for a
  # hybrid teleop recipe later. Root pos/ori, body_lin_vel and feet_pos are now
  # set by the recipe loop, so the old per-term overrides here were removed.
  cfg.rewards["motion_pose_body_pos"] = RewardTermCfg(
    func=mdp.motion_local_body_position_error_exp,
    weight=0.0,
    params={
      "command_name": "motion",
      "std": 0.12,
      "body_names": POSE_BODY_NAMES,
    },
  )
  cfg.rewards["root_height_asymmetric"] = RewardTermCfg(
    func=mdp.root_height_asymmetric_error_exp,
    weight=0.75,
    params={
      "command_name": "motion",
      "std": 0.10,
      "under_track_scale": 0.30,
      "over_drop_scale": 2.00,
    },
  )
  cfg.rewards["joint_acc_smoothness"] = RewardTermCfg(
    func=mdp.joint_acc_l2,
    weight=-1.0e-6,
    params={"asset_cfg": mdp.SceneEntityCfg("robot", joint_names=(".*",))},
  )
  cfg.rewards["action_acc_l2"] = RewardTermCfg(
    func=mdp.action_acc_l2,
    weight=-0.02,
  )
  cfg.rewards["anti_shake_upper_body_ang_vel"] = RewardTermCfg(
    func=mdp.body_angular_velocity_l2,
    weight=-0.03,
    params={
      "asset_cfg": mdp.SceneEntityCfg("robot", body_names=UPPER_BODY_NAMES),
      "axes": ("x", "y"),
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
  cfg.terminations["anchor_ori"].params["threshold"] = 1.00
  # The base "anchor_pos" termination only checks height (bad_anchor_pos_z_only),
  # so horizontal root drift is free as long as the robot stays upright. That is
  # fine for locomotion tasks where the robot is supposed to walk away from the
  # reference, but for full-body teleop the robot must follow the operator's
  # root translation, so drifting away has to be punished. Without this the
  # policy learns to ignore the (now present) root-translation command and
  # settles into marching in place once enough motions are in the pool.
  # 2.0 m, raised from 0.8 m: the tight threshold was killing the *transient*
  # lag that occurs while the robot is legitimately trying to keep up with a
  # sustained fast walk. Rollout of a healthy checkpoint on a Neutral_walk_forward
  # clip (reference ~1.0-1.2 m/s) showed the robot tracking but lagging during
  # the speed-up, crossing 0.8 m at step 66 and being killed before it could
  # catch up -- the same "hard threshold cannot tell transient lag from failure"
  # problem as the reverted anchor_yaw termination. 59% of the dataset needs
  # sustained >0.8 m/s, so a 0.8 m anchor bound is unwinnable for most walks. 2.0
  # m still catches a true drift/march-in-place failure (old runs drifted 2-6 m)
  # while giving the policy runway to learn fast walking; the forward pull is now
  # carried by the strengthened velocity reward, not this termination.
  cfg.terminations["anchor_xy"] = TerminationTermCfg(
    func=mdp.bad_anchor_xy,
    params={"command_name": "motion", "threshold": 2.0},
  )
  # Heading (yaw) had no recovery gradient: bad_anchor_ori uses projected gravity
  # (pitch/roll only, yaw-invariant), so an upright robot facing the wrong way
  # never tripped it, and motion_global_root_ori at std 0.4 saturates to a flat
  # gradient past ~50 deg. Once turn-in-place motions entered the pool the policy
  # abandoned rotation (error_anchor_rot jumped ~0.09 -> ~1.1 rad) and shuffled
  # in place. The fix is the widened root_ori reward below (std 0.4 -> 0.7,
  # weight 0.5 -> 0.75). A hard yaw termination was tried as a backstop and
  # removed: a robot legitimately lagging a fast 188 deg sweep transiently
  # exceeds any threshold that also catches the shuffle failure (steady ~57 deg),
  # so the two cannot be separated by instantaneous yaw error. At 1.0 rad it
  # fired on ~every env once larger turns entered the pool, collapsing
  # mean_episode_length from ~1485 to ~260 and reward from ~430 to ~58. The
  # reward alone tracked heading well (root_ori reward ~0.72, error_anchor_rot
  # ~0.09) in the window where the termination was not yet firing.
  cfg.terminations["ee_body_pos"].params["body_names"] = FEET_BODY_NAMES
  # Foot-height tracking termination. Raised from 0.50 -> 0.80 so the policy is
  # not killed for under-tracking high foot lifts (kicks, single-leg, stoop)
  # while it is actually still upright -- this termination was ending ~75% of
  # episodes even though the robot was not falling. Adaptive: relax further to
  # 1.20 when the reference root is low (deep squat / stoop). Real falls are
  # still caught by fell_over_height (0.25) and fell_over_orientation (1.25).
  cfg.terminations["ee_body_pos"].params["threshold"] = 0.80
  cfg.terminations["ee_body_pos"].params["threshold_adaptive"] = True
  cfg.terminations["ee_body_pos"].params["down_threshold"] = 1.20
  cfg.terminations["ee_body_pos"].params["root_height_threshold"] = 0.55
  cfg.terminations["pose_feet_body_pos"] = TerminationTermCfg(
    func=mdp.bad_motion_body_pos,
    params={
      "command_name": "motion",
      "threshold": 1.00,
      "body_names": FEET_BODY_NAMES,
      # Adaptive (Groot-style): when the reference root is low (deep squat /
      # crouch), relax the feet tracking tolerance so the policy is not killed
      # for under-tracking a pose it physically cannot reach -- it learns to go
      # as low as it safely can instead of falling.
      "threshold_adaptive": True,
      "down_threshold": 1.40,
      "root_height_threshold": 0.55,
    },
  )
  cfg.terminations["knee_ground_contact"] = TerminationTermCfg(
    func=mdp.illegal_contact,
    params={
      "sensor_name": "knee_ground_contact",
      "force_threshold": 5.0,
    },
  )
  cfg.terminations["fell_over_height"] = TerminationTermCfg(
    func=mdp.root_height_below_minimum,
    params={
      "asset_cfg": mdp.SceneEntityCfg("robot"),
      "minimum_height": 0.25,
    },
  )
  cfg.terminations["fell_over_orientation"] = TerminationTermCfg(
    func=mdp.bad_orientation,
    params={
      "asset_cfg": mdp.SceneEntityCfg("robot"),
      "limit_angle": 1.25,
    },
  )


def _configure_teleop_robust_rewards_and_terminations(
  cfg: ManagerBasedRlEnvCfg,
) -> None:
  cfg.rewards["alive"] = RewardTermCfg(func=mdp.is_alive, weight=1.0)
  cfg.rewards["flat_orientation"] = RewardTermCfg(
    func=mdp.flat_orientation_l2,
    weight=-2.0,
    params={"asset_cfg": mdp.SceneEntityCfg("robot")},
  )

  cfg.rewards["motion_global_root_pos"].weight = 0.25
  cfg.rewards["motion_global_root_pos"].params["std"] = 0.8
  cfg.rewards["motion_global_root_ori"].weight = 0.2
  cfg.rewards["motion_global_root_ori"].params["std"] = 0.9
  cfg.rewards["motion_body_pos"].weight = 0.5
  cfg.rewards["motion_body_pos"].params["std"] = 0.5
  cfg.rewards["motion_body_ori"].weight = 0.5
  cfg.rewards["motion_body_ori"].params["std"] = 0.7
  cfg.rewards["motion_body_lin_vel"].weight = 0.5
  cfg.rewards["motion_body_lin_vel"].params["std"] = 1.3
  cfg.rewards["motion_body_ang_vel"].weight = 0.5
  cfg.rewards["motion_body_ang_vel"].params["std"] = 4.0
  cfg.rewards["motion_lower_body_pos"].weight = 0.25
  cfg.rewards["motion_lower_body_pos"].params["std"] = 0.28
  cfg.rewards["motion_lower_body_ori"].weight = 0.25
  cfg.rewards["motion_lower_body_ori"].params["std"] = 0.65
  cfg.rewards["motion_feet_pos"].weight = 0.08
  cfg.rewards["motion_feet_pos"].params["std"] = 0.45
  cfg.rewards["motion_feet_ori"].weight = 0.05
  cfg.rewards["motion_feet_ori"].params["std"] = 0.65
  cfg.rewards["motion_right_arm_joint_pos"].weight = 0.55
  cfg.rewards["motion_left_arm_joint_pos"].weight = 0.55
  cfg.rewards["motion_right_arm_joint_pos_coarse"].weight = 0.35
  cfg.rewards["motion_left_arm_joint_pos_coarse"].weight = 0.35
  cfg.rewards["motion_leg_joint_pos"].weight = 0.05
  cfg.rewards["motion_leg_joint_pos"].params["std"] = 0.75
  cfg.rewards["action_rate_l2"].weight = -0.12
  cfg.rewards["feet_acc"].weight = -1.0e-6
  cfg.rewards["joint_limit"].weight = -15.0
  cfg.rewards["self_collisions"].weight = -15.0

  cfg.terminations["anchor_pos"].params["threshold"] = 2.0
  cfg.terminations["anchor_ori"].params["threshold"] = 1.5
  cfg.terminations["ee_body_pos"].params["threshold"] = 2.0
  cfg.terminations["fell_over_height"] = TerminationTermCfg(
    func=mdp.root_height_below_minimum,
    params={
      "asset_cfg": mdp.SceneEntityCfg("robot"),
      "minimum_height": 0.25,
    },
  )
  cfg.terminations["fell_over_orientation"] = TerminationTermCfg(
    func=mdp.bad_orientation,
    params={
      "asset_cfg": mdp.SceneEntityCfg("robot"),
      "limit_angle": 1.35,
    },
  )


def _configure_balance_finetune_rewards_and_terminations(
  cfg: ManagerBasedRlEnvCfg,
) -> None:
  cfg.rewards["alive"].weight = 3.0
  cfg.rewards["termination"] = RewardTermCfg(func=mdp.is_terminated, weight=-250.0)
  cfg.rewards["flat_orientation"].weight = -10.0

  cfg.rewards["motion_anchor_xy"] = RewardTermCfg(
    func=mdp.motion_global_anchor_xy_error_exp,
    weight=0.50,
    params={"command_name": "motion", "std": 0.25},
  )
  cfg.rewards["root_height_asymmetric"] = RewardTermCfg(
    func=mdp.root_height_asymmetric_error_exp,
    weight=0.80,
    params={
      "command_name": "motion",
      "std": 0.08,
      "under_track_scale": 0.30,
      "over_drop_scale": 2.00,
    },
  )
  cfg.rewards["motion_anchor_height"] = RewardTermCfg(
    func=mdp.motion_global_anchor_height_error_exp,
    weight=0.0,
    params={"command_name": "motion", "std": 0.24},
  )
  cfg.rewards["motion_feet_height"] = RewardTermCfg(
    func=mdp.motion_relative_body_height_error_exp,
    weight=0.15,
    params={
      "command_name": "motion",
      "std": 0.40,
      "body_names": FEET_BODY_NAMES,
    },
  )
  cfg.rewards["pelvis_height_minimum"] = RewardTermCfg(
    func=mdp.pelvis_height_above_minimum_exp,
    weight=1.00,
    params={
      "asset_cfg": mdp.SceneEntityCfg("robot", body_names=("pelvis",)),
      "minimum_height": 0.60,
      "std": 0.05,
    },
  )
  cfg.rewards["feet_lateral_width"] = RewardTermCfg(
    func=mdp.feet_lateral_distance_error_exp,
    weight=0.60,
    params={
      "asset_cfg": mdp.SceneEntityCfg("robot", body_names=FEET_BODY_NAMES),
      "target_width": 0.20,
      "std": 0.06,
    },
  )

  cfg.rewards["motion_global_root_pos"].weight = 0.0
  cfg.rewards["motion_global_root_pos"].params["std"] = 1.80
  cfg.rewards["motion_global_root_ori"].weight = 0.10
  cfg.rewards["motion_global_root_ori"].params["std"] = 0.70
  cfg.rewards["motion_body_pos"].weight = 0.50
  cfg.rewards["motion_body_pos"].params["std"] = 0.40
  cfg.rewards["motion_body_ori"].weight = 0.40
  cfg.rewards["motion_body_ori"].params["std"] = 0.60
  cfg.rewards["motion_body_lin_vel"].weight = 0.10
  cfg.rewards["motion_body_lin_vel"].params["std"] = 1.50
  cfg.rewards["motion_body_ang_vel"].weight = 0.10
  cfg.rewards["motion_body_ang_vel"].params["std"] = 4.00

  cfg.rewards["motion_lower_body_pos"].weight = 0.30
  cfg.rewards["motion_lower_body_pos"].params["std"] = 0.35
  cfg.rewards["motion_lower_body_ori"].weight = 0.20
  cfg.rewards["motion_lower_body_ori"].params["std"] = 0.70
  cfg.rewards["motion_feet_pos"].weight = 0.10
  cfg.rewards["motion_feet_pos"].params["std"] = 0.50
  cfg.rewards["motion_feet_ori"].weight = 0.05
  cfg.rewards["motion_feet_ori"].params["std"] = 0.80
  cfg.rewards["motion_leg_joint_pos"].weight = 0.05
  cfg.rewards["motion_leg_joint_pos"].params["std"] = 0.80

  cfg.rewards["motion_right_arm_joint_pos"].weight = 1.20
  cfg.rewards["motion_right_arm_joint_pos"].params["std"] = 0.18
  cfg.rewards["motion_left_arm_joint_pos"].weight = 1.20
  cfg.rewards["motion_left_arm_joint_pos"].params["std"] = 0.18
  cfg.rewards["motion_right_arm_joint_pos_coarse"].weight = 0.40
  cfg.rewards["motion_right_arm_joint_pos_coarse"].params["std"] = 0.35
  cfg.rewards["motion_left_arm_joint_pos_coarse"].weight = 0.40
  cfg.rewards["motion_left_arm_joint_pos_coarse"].params["std"] = 0.35

  cfg.rewards["base_xy_vel_l2"] = RewardTermCfg(
    func=mdp.robot_anchor_xy_velocity_l2,
    weight=-0.05,
    params={"command_name": "motion"},
  )
  cfg.rewards["no_unwanted_backward_drift"] = RewardTermCfg(
    func=mdp.no_unwanted_backward_drift_l2,
    weight=-0.80,
    params={
      "command_name": "motion",
      "threshold": 0.02,
      "ref_backward_threshold": -0.05,
    },
  )
  cfg.rewards["feet_slip_l2"] = RewardTermCfg(
    func=mdp.feet_slip_l2,
    weight=-0.80,
    params={
      "sensor_name": "feet_ground_contact",
      "asset_cfg": mdp.SceneEntityCfg("robot", body_names=FEET_BODY_NAMES),
    },
  )
  cfg.rewards["joint_acc_smoothness"] = RewardTermCfg(
    func=mdp.joint_acc_l2,
    weight=-1.0e-6,
    params={"asset_cfg": mdp.SceneEntityCfg("robot", joint_names=(".*",))},
  )
  cfg.rewards["feet_acc"].weight = -1.0e-6
  cfg.rewards["action_rate_l2"].weight = -0.08
  cfg.rewards["action_acc_l2"] = RewardTermCfg(
    func=mdp.action_acc_l2,
    weight=-0.04,
  )
  cfg.rewards["anti_shake_upper_body_ang_vel"] = RewardTermCfg(
    func=mdp.body_angular_velocity_l2,
    weight=-0.04,
    params={
      "asset_cfg": mdp.SceneEntityCfg(
        "robot",
        body_names=UPPER_BODY_NAMES,
      ),
      "axes": ("x", "y"),
    },
  )

  cfg.terminations["anchor_pos"].params["threshold"] = 1.20
  cfg.terminations["anchor_ori"].params["threshold"] = 1.20
  cfg.terminations["ee_body_pos"].params["body_names"] = FEET_BODY_NAMES
  cfg.terminations["ee_body_pos"].params["threshold"] = 1.50
  cfg.terminations.pop("anchor_xy", None)
  cfg.terminations.pop("anchor_height", None)
  cfg.terminations.pop("body_pos_guard", None)
  cfg.terminations.pop("body_height_guard", None)
  cfg.terminations["fell_over_height"].params["minimum_height"] = 0.25
  cfg.terminations["fell_over_orientation"].params["limit_angle"] = 1.25


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

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (feet_ground_cfg, self_collision_cfg)

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
  motion_cmd.motion_curriculum_gate = False
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
      "soften_transition": False,
    },
  )
  cfg.rewards["motion_right_arm_joint_pos_coarse"] = RewardTermCfg(
    func=mdp.motion_joint_position_error_exp,
    weight=0.5,
    params={
      "command_name": "motion",
      "std": 0.35,
      "joint_names": RIGHT_ARM_JOINT_NAMES,
      "soften_transition": False,
    },
  )
  cfg.rewards["motion_left_arm_joint_pos"] = RewardTermCfg(
    func=mdp.motion_joint_position_error_exp,
    weight=1.0,
    params={
      "command_name": "motion",
      "std": 0.15,
      "joint_names": LEFT_ARM_JOINT_NAMES,
      "soften_transition": False,
    },
  )
  cfg.rewards["motion_left_arm_joint_pos_coarse"] = RewardTermCfg(
    func=mdp.motion_joint_position_error_exp,
    weight=0.5,
    params={
      "command_name": "motion",
      "std": 0.35,
      "joint_names": LEFT_ARM_JOINT_NAMES,
      "soften_transition": False,
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
    cfg.events.pop("push_robot", None)
    motion_cmd.pose_range = {}
    motion_cmd.velocity_range = {}
    motion_cmd.sampling_mode = "start"
    motion_cmd.motion_resample_interval = None

  return cfg


def vr_m3_1_flat_tracking_pose_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create the VR M3.1 pose-only full-body imitation configuration."""
  cfg = vr_m3_1_flat_tracking_env_cfg(has_state_estimation=False, play=play)
  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)

  motion_cmd.motion_file = POSE_MOTION_FILE
  motion_cmd.smpl_motion_file = POSE_SMPL_MOTION_FILE
  motion_cmd.smpl_num_joints = 24
  motion_cmd.smpl_strict_pairing = True
  motion_cmd.smpl_y_up = True
  motion_cmd.smpl_num_future_frames = 10
  motion_cmd.smpl_dt_future_ref_frames = 0.02
  motion_cmd.body_names = POSE_BODY_NAMES
  motion_cmd.num_future_frames = 10
  motion_cmd.dt_future_ref_frames = 0.1
  motion_cmd.encoder_sample_probs = {"g1": 1.0, "teleop": 1.0, "smpl": 1.0}
  motion_cmd.encoder_curriculum_initial_probs = {
    "g1": 1.0,
    "teleop": 0.0,
    "smpl": 0.0,
  }
  motion_cmd.encoder_curriculum_warmup_steps = 72_000
  motion_cmd.encoder_curriculum_ramp_steps = 144_000
  motion_cmd.initial_num_load_motions = 4096
  motion_cmd.max_num_load_motions = 4096
  motion_cmd.num_new_motions_per_resample = 1024
  motion_cmd.motion_pool_mode = "streaming"
  motion_cmd.motion_replay_fraction = 0.75
  motion_cmd.motion_resample_interval = 250
  motion_cmd.motion_resample_start_iteration = 250
  motion_cmd.motion_resample_unique_until_all_seen = True
  motion_cmd.motion_curriculum_gate = False
  motion_cmd.motion_curriculum_min_stable_iterations = 100
  motion_cmd.motion_curriculum_force_after_iterations = 5000
  motion_cmd.motion_curriculum_ordered_loading = False
  motion_cmd.motion_curriculum_stage_sizes = TELEOP_ROBUST_STAGE_SIZES
  motion_cmd.motion_curriculum_min_mean_episode_length = 800.0
  motion_cmd.motion_curriculum_max_fall_rate = 0.08
  motion_cmd.motion_curriculum_max_body_pos_error = 0.35
  motion_cmd.motion_curriculum_max_body_rot_error = None
  motion_cmd.motion_curriculum_max_anchor_pos_error = None
  motion_cmd.motion_curriculum_max_anchor_rot_error = None
  motion_cmd.motion_curriculum_max_joint_pos_error = None

  if not play:
    motion_cmd.pose_range = {
      "x": (-0.10, 0.10),
      "y": (-0.10, 0.10),
      "z": (-0.04, 0.06),
      "roll": (-0.20, 0.20),
      "pitch": (-0.20, 0.20),
      "yaw": (-0.35, 0.35),
    }
    motion_cmd.velocity_range = {
      "x": (-0.5, 0.5),
      "y": (-0.5, 0.5),
      "z": (-0.2, 0.2),
      "roll": (-0.7, 0.7),
      "pitch": (-0.7, 0.7),
      "yaw": (-0.9, 0.9),
    }
    motion_cmd.joint_position_range = (-0.12, 0.12)
    # Reference projection (mirrors the balance_finetune recipe, which already
    # uses it). Clamps the tracked reference anchor to stay within a bounded
    # window of the robot (~0.18 m forward) instead of tracking the true
    # absolute position. Without it the pose recipe tracks the absolute
    # reference: on a sustained fast walk the reference walks away faster than
    # the robot can follow, so the robot drifts unboundedly, dies on anchor_xy,
    # never learns to keep up, and FREEZES out-of-distribution when behind (it
    # never saw a >0.8 m lag in training because anchor_xy killed it first).
    # Projection turns the target into a near "carrot" the robot can always
    # track in-distribution and walk toward, making sustained forward walking
    # learnable. 59% of the dataset needs >0.8 m/s sustained.
    motion_cmd.reference_projection_enabled = True
    motion_cmd.reference_projection_future_frames = 3
    motion_cmd.reference_projection_max_forward = 0.18
    motion_cmd.reference_projection_max_backward = 0.0
    motion_cmd.reference_projection_max_lateral = 0.12
    motion_cmd.reference_projection_max_z_down = 0.08
    motion_cmd.reference_projection_max_z_up = 0.08
    motion_cmd.reference_projection_max_squat_depth = 0.16
    motion_cmd.reference_projection_min_anchor_height = 0.60
    motion_cmd.reference_projection_max_body_delta = 0.28
    motion_cmd.reference_projection_velocity_scale = 0.15
    motion_cmd.reference_projection_yaw_only_anchor = True
    motion_cmd.reference_projection_joint_delta = 0.18
    motion_cmd.reference_projection_joint_delta_by_pattern = {
      ".*hip.*": 0.16,
      ".*knee.*": 0.20,
      ".*ankle.*": 0.12,
      ".*waist.*": 0.10,
      ".*shoulder.*": 10.0,
      ".*elbow.*": 10.0,
      ".*wrist.*": 10.0,
    }
    motion_cmd.reference_projection_default_joint_delta = 0.45
    motion_cmd.reference_projection_default_joint_delta_by_pattern = {
      ".*hip.*": 0.35,
      ".*knee.*": 0.55,
      ".*ankle.*": 0.25,
      ".*waist.*": 0.14,
      ".*shoulder.*": 10.0,
      ".*elbow.*": 10.0,
      ".*wrist.*": 10.0,
    }
  else:
    motion_cmd.pose_range = {}
    motion_cmd.velocity_range = {}
    motion_cmd.sampling_mode = "start"
    motion_cmd.motion_resample_interval = None

  _configure_pose_only_observations(cfg, enable_actor_corruption=not play)
  _add_pose_contact_sensors(cfg)
  _configure_pose_rewards_and_terminations(cfg)
  return cfg


def vr_m3_1_flat_tracking_teleop_robust_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create the from-scratch VR M3.1 teleop-robust tracking configuration."""
  cfg = vr_m3_1_flat_tracking_env_cfg(play=play)
  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)

  motion_cmd.motion_file = "data/vr_m3_1_teleop_robust_curriculum.txt"

  if play:
    return cfg

  motion_cmd.initial_num_load_motions = 128
  motion_cmd.max_num_load_motions = None
  motion_cmd.num_new_motions_per_resample = 128
  motion_cmd.motion_pool_mode = "grow"
  motion_cmd.motion_resample_interval = 250
  motion_cmd.motion_resample_start_iteration = 250
  motion_cmd.motion_resample_unique_until_all_seen = True
  motion_cmd.motion_curriculum_gate = True
  motion_cmd.motion_curriculum_min_stable_iterations = 250
  motion_cmd.motion_curriculum_force_after_iterations = None
  motion_cmd.motion_curriculum_ordered_loading = True
  motion_cmd.motion_curriculum_stage_sizes = TELEOP_ROBUST_STAGE_SIZES
  motion_cmd.motion_curriculum_min_mean_episode_length = 1200.0
  motion_cmd.motion_curriculum_max_fall_rate = 0.05
  motion_cmd.motion_curriculum_max_body_pos_error = 0.40
  motion_cmd.motion_curriculum_max_body_rot_error = 1.00
  motion_cmd.motion_curriculum_max_anchor_pos_error = None
  motion_cmd.motion_curriculum_max_anchor_rot_error = None
  motion_cmd.motion_curriculum_max_joint_pos_error = 2.0
  motion_cmd.pose_range = {
    "x": (-0.15, 0.15),
    "y": (-0.15, 0.15),
    "z": (-0.06, 0.08),
    "roll": (-0.30, 0.30),
    "pitch": (-0.30, 0.30),
    "yaw": (-0.50, 0.50),
  }
  motion_cmd.velocity_range = {
    "x": (-0.8, 0.8),
    "y": (-0.8, 0.8),
    "z": (-0.3, 0.3),
    "roll": (-1.0, 1.0),
    "pitch": (-1.0, 1.0),
    "yaw": (-1.2, 1.2),
  }
  motion_cmd.joint_position_range = (-0.20, 0.20)

  _configure_teleop_sensor_latency(cfg)
  _configure_teleop_robust_rewards_and_terminations(cfg)
  return cfg


def vr_m3_1_flat_tracking_balance_finetune_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  cfg = vr_m3_1_flat_tracking_teleop_robust_env_cfg(play=play)
  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)

  if play:
    return cfg

  motion_cmd.initial_num_load_motions = None
  motion_cmd.max_num_load_motions = None
  motion_cmd.num_new_motions_per_resample = None
  motion_cmd.motion_resample_interval = None
  motion_cmd.motion_curriculum_gate = False
  motion_cmd.motion_curriculum_force_after_iterations = None
  motion_cmd.motion_curriculum_ordered_loading = False
  motion_cmd.motion_curriculum_stage_sizes = ()
  motion_cmd.sampling_mode = "uniform"

  motion_cmd.pose_range = {
    "x": (-0.08, 0.08),
    "y": (-0.08, 0.08),
    "z": (-0.04, 0.05),
    "roll": (-0.20, 0.20),
    "pitch": (-0.20, 0.20),
    "yaw": (-0.35, 0.35),
  }
  motion_cmd.velocity_range = {
    "x": (-0.45, 0.45),
    "y": (-0.45, 0.45),
    "z": (-0.20, 0.20),
    "roll": (-0.70, 0.70),
    "pitch": (-0.70, 0.70),
    "yaw": (-0.90, 0.90),
  }
  motion_cmd.joint_position_range = (-0.12, 0.12)
  motion_cmd.dt_future_ref_frames = 0.05
  motion_cmd.reference_projection_enabled = True
  motion_cmd.reference_projection_future_frames = 3
  motion_cmd.reference_projection_max_forward = 0.18
  motion_cmd.reference_projection_max_backward = 0.0
  motion_cmd.reference_projection_max_lateral = 0.12
  motion_cmd.reference_projection_max_z_down = 0.08
  motion_cmd.reference_projection_max_z_up = 0.08
  motion_cmd.reference_projection_max_squat_depth = 0.16
  motion_cmd.reference_projection_min_anchor_height = 0.60
  motion_cmd.reference_projection_max_body_delta = 0.28
  motion_cmd.reference_projection_velocity_scale = 0.15
  motion_cmd.reference_projection_yaw_only_anchor = True
  motion_cmd.reference_projection_joint_delta = 0.18
  motion_cmd.reference_projection_joint_delta_by_pattern = {
    ".*hip.*": 0.16,
    ".*knee.*": 0.20,
    ".*ankle.*": 0.12,
    ".*waist.*": 0.10,
    ".*shoulder.*": 10.0,
    ".*elbow.*": 10.0,
    ".*wrist.*": 10.0,
  }
  motion_cmd.reference_projection_default_joint_delta = 0.45
  motion_cmd.reference_projection_default_joint_delta_by_pattern = {
    ".*hip.*": 0.35,
    ".*knee.*": 0.55,
    ".*ankle.*": 0.25,
    ".*waist.*": 0.14,
    ".*shoulder.*": 10.0,
    ".*elbow.*": 10.0,
    ".*wrist.*": 10.0,
  }

  _configure_balance_finetune_rewards_and_terminations(cfg)
  return cfg


def vr_m3_1_flat_tracking_balance_transition_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  cfg = vr_m3_1_flat_tracking_balance_finetune_env_cfg(play=False)
  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)

  motion_cmd.motion_file = "data/vr_m3_1_full_unique_names.txt"
  motion_cmd.transition_enabled = True
  motion_cmd.transition_duration_s = 0.50
  motion_cmd.transition_append_to_command = True
  motion_cmd.transition_soft_reward_enabled = True
  motion_cmd.transition_reward_softness = 0.65

  if not play:
    motion_cmd.initial_num_load_motions = 128
    motion_cmd.max_num_load_motions = None
    motion_cmd.num_new_motions_per_resample = 128
    motion_cmd.motion_pool_mode = "grow"
    motion_cmd.motion_resample_interval = 250
    motion_cmd.motion_resample_start_iteration = 250
    motion_cmd.motion_resample_unique_until_all_seen = True
    motion_cmd.motion_curriculum_gate = True
    motion_cmd.motion_curriculum_min_stable_iterations = 250
    motion_cmd.motion_curriculum_force_after_iterations = 5000
    motion_cmd.motion_curriculum_ordered_loading = True
    motion_cmd.motion_curriculum_stage_sizes = TELEOP_ROBUST_STAGE_SIZES
    motion_cmd.motion_curriculum_min_mean_episode_length = 1200.0
    motion_cmd.motion_curriculum_max_fall_rate = 0.08
    motion_cmd.motion_curriculum_max_body_pos_error = 0.40
    motion_cmd.motion_curriculum_max_body_rot_error = 1.00
    motion_cmd.motion_curriculum_max_anchor_pos_error = None
    motion_cmd.motion_curriculum_max_anchor_rot_error = None
    motion_cmd.motion_curriculum_max_joint_pos_error = 2.0

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    motion_cmd.pose_range = {}
    motion_cmd.velocity_range = {}
    motion_cmd.sampling_mode = "start"
    motion_cmd.motion_resample_interval = None

  return cfg
