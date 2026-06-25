"""Tests specific to motion tracking tasks."""

from typing import cast

import pytest

from mjlab.asset_zoo.robots import (
  G1_ACTION_SCALE,
  VR_H3_1_ACTION_SCALE,
  VR_M3_1_ACTION_SCALE,
)
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.sensor import ContactSensorCfg
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg

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


@pytest.fixture(scope="module")
def tracking_task_ids() -> list[str]:
  """Get all tracking task IDs."""
  return [t for t in list_tasks() if "Tracking" in t]


@pytest.fixture(scope="module")
def g1_tracking_task_ids(tracking_task_ids: list[str]) -> list[str]:
  """Get all G1 tracking task IDs."""
  return [t for t in tracking_task_ids if "G1" in t]


@pytest.fixture(scope="module")
def vr_h3_1_tracking_task_ids(tracking_task_ids: list[str]) -> list[str]:
  """Get all VR H3.1 tracking task IDs."""
  return [t for t in tracking_task_ids if "VR-H3-1" in t]


@pytest.fixture(scope="module")
def vr_m3_1_tracking_task_ids(tracking_task_ids: list[str]) -> list[str]:
  """Get all VR M3.1 tracking task IDs."""
  return [t for t in tracking_task_ids if "VR-M3-1" in t]


def test_tracking_tasks_have_motion_command(tracking_task_ids: list[str]) -> None:
  """All tracking tasks should have a 'motion' command of type MotionCommandCfg."""
  for task_id in tracking_task_ids:
    cfg = load_env_cfg(task_id)

    assert "motion" in cfg.commands, f"Task {task_id} missing 'motion' command"

    motion_cmd = cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg), (
      f"Task {task_id} motion command is not MotionCommandCfg"
    )


def test_tracking_tasks_have_self_collision_sensor(
  tracking_task_ids: list[str],
) -> None:
  """All tracking tasks should have a self_collision sensor."""
  for task_id in tracking_task_ids:
    cfg = load_env_cfg(task_id)

    assert cfg.scene.sensors is not None, f"Task {task_id} has no sensors"

    sensor_names = {s.name for s in cfg.scene.sensors}
    assert "self_collision" in sensor_names, (
      f"Task {task_id} missing self_collision sensor"
    )


def test_tracking_no_state_estimation_observations() -> None:
  """No-state-estimation tasks remove observations that depend on state estimation."""
  task_id = "Mjlab-Tracking-Flat-Unitree-G1-No-State-Estimation"

  for play_mode in [False, True]:
    cfg = load_env_cfg(task_id, play=play_mode)
    mode_str = "play mode" if play_mode else "training mode"

    assert "actor" in cfg.observations, (
      f"Task {task_id} ({mode_str}) missing policy observations"
    )
    actor_terms = cfg.observations["actor"].terms

    assert "motion_anchor_pos_b" not in actor_terms, (
      f"Task {task_id} ({mode_str}) has motion_anchor_pos_b in policy, "
      "expected it to be removed for no-state-estimation variant"
    )
    assert "base_lin_vel" not in actor_terms, (
      f"Task {task_id} ({mode_str}) has base_lin_vel in policy, "
      "expected it to be removed for no-state-estimation variant"
    )


def test_tracking_play_disables_rsi_randomization() -> None:
  """Tracking play tasks should disable RSI randomization."""
  for task_id in list_tasks():
    if "Tracking" not in task_id:
      continue
    cfg = load_env_cfg(task_id, play=True)

    motion_cmd = cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg), (
      f"Task {task_id} (play mode) motion command is not MotionCommandCfg"
    )

    assert motion_cmd.pose_range == {}, (
      f"Task {task_id} (play mode) has non-empty pose_range={motion_cmd.pose_range}, "
      "expected empty dict for disabled RSI"
    )
    assert motion_cmd.velocity_range == {}, (
      f"Task {task_id} (play mode) has non-empty velocity_range={motion_cmd.velocity_range}, "
      "expected empty dict for disabled RSI"
    )


def test_tracking_play_uses_start_sampling_mode() -> None:
  """Tracking play tasks should use sampling_mode='start'."""
  for task_id in list_tasks():
    if "Tracking" not in task_id:
      continue
    cfg = load_env_cfg(task_id, play=True)

    motion_cmd = cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg), (
      f"Task {task_id} (play mode) motion command is not MotionCommandCfg"
    )

    assert motion_cmd.sampling_mode == "start", (
      f"Task {task_id} (play mode) sampling_mode={motion_cmd.sampling_mode}, "
      "expected 'start'"
    )


def test_g1_tracking_has_correct_action_scale(g1_tracking_task_ids: list[str]) -> None:
  """G1 tracking tasks should use G1_ACTION_SCALE."""
  for task_id in g1_tracking_task_ids:
    cfg = load_env_cfg(task_id)

    assert "joint_pos" in cfg.actions, f"Task {task_id} missing 'joint_pos' action"

    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg), (
      f"Task {task_id} joint_pos action is not JointPositionActionCfg"
    )

    assert joint_pos_action.scale == G1_ACTION_SCALE, (
      f"Task {task_id} action scale mismatch, expected G1_ACTION_SCALE"
    )


def test_vr_h3_1_tracking_has_correct_action_scale(
  vr_h3_1_tracking_task_ids: list[str],
) -> None:
  """VR H3.1 tracking tasks should use VR_H3_1_ACTION_SCALE."""
  for task_id in vr_h3_1_tracking_task_ids:
    cfg = load_env_cfg(task_id)

    assert "joint_pos" in cfg.actions, f"Task {task_id} missing 'joint_pos' action"

    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg), (
      f"Task {task_id} joint_pos action is not JointPositionActionCfg"
    )

    assert joint_pos_action.scale == VR_H3_1_ACTION_SCALE, (
      f"Task {task_id} action scale mismatch, expected VR_H3_1_ACTION_SCALE"
    )


def test_vr_m3_1_pose_tracking_uses_paired_smpl_terms() -> None:
  """VR M3.1 Pose should expose robot and SMPL observations for alignment."""
  cfg = load_env_cfg("Mjlab-Tracking-Flat-VR-M3-1-Pose")
  actor_terms = cfg.observations["actor"].terms
  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)

  assert "target_body_pos_local" in actor_terms
  assert "actor_smpl" in cfg.observations
  actor_smpl_terms = cfg.observations["actor_smpl"].terms
  assert "smpl_joints_multi_future_local" in actor_smpl_terms
  assert "smpl_root_ori_b_multi_future" in actor_smpl_terms
  assert "joint_pos_multi_future_wrist_for_smpl" in actor_smpl_terms
  assert "joint_pos" not in actor_smpl_terms
  assert "joint_vel" not in actor_smpl_terms
  assert "base_ang_vel" not in actor_smpl_terms
  assert "actions" not in actor_smpl_terms
  assert "compliance" not in actor_smpl_terms

  # The teleop/smpl encoders must receive the root-translation command
  # (future root pos in robot-anchor frame), the same signal g1 gets inside
  # command_multi_future. Without it they only see pose/orientation and the
  # robot marches in place instead of following the operator's root.
  assert "motion_anchor_pos_b_multi_future" in actor_smpl_terms
  assert "actor_teleop" in cfg.observations
  actor_teleop_terms = cfg.observations["actor_teleop"].terms
  assert "motion_anchor_pos_b_multi_future" in actor_teleop_terms

  proprioception_cfg = cfg.observations["proprioception"]
  proprioception_terms = proprioception_cfg.terms
  assert proprioception_cfg.history_length == 10
  assert proprioception_cfg.flatten_history_dim
  assert "base_ang_vel" in proprioception_terms
  assert "joint_pos" in proprioception_terms
  assert "joint_vel" in proprioception_terms
  assert "actions" in proprioception_terms
  assert "projected_gravity" in proprioception_terms

  assert "command" not in actor_terms
  assert "motion_anchor_pos_b" not in actor_terms
  assert "base_lin_vel" not in actor_terms
  # Horizontal root drift must be punished (the base anchor_pos termination only
  # checks height), otherwise the policy ignores the root-translation command
  # and marches in place once the motion pool grows.
  assert "anchor_xy" in cfg.terminations
  # Heading is constrained by the widened motion_global_root_ori reward only.
  # A hard yaw termination was tried and removed: a robot legitimately lagging a
  # fast turn transiently exceeds any threshold that also catches the shuffle
  # failure, so at 1.0 rad it fired on ~every env and collapsed training
  # (mean_episode_length ~1485 -> ~260). bad_anchor_ori is a tilt check only.
  assert "anchor_yaw" not in cfg.terminations
  assert len(motion_cmd.body_names) >= 24
  assert motion_cmd.num_future_frames == 10
  assert motion_cmd.smpl_y_up
  assert motion_cmd.smpl_num_future_frames == 10
  assert motion_cmd.smpl_dt_future_ref_frames == 0.02
  assert motion_cmd.motion_file == "data/vr_m3_1_duplicate_motion_paths_valid.txt"
  assert motion_cmd.smpl_motion_file == (
    "/home/hantp/Groot-WholeBodyControl/data/smpl_filtered"
  )
  assert motion_cmd.smpl_strict_pairing
  assert motion_cmd.initial_num_load_motions == 4096
  assert motion_cmd.max_num_load_motions == 4096
  assert motion_cmd.motion_pool_mode == "streaming"
  assert motion_cmd.num_new_motions_per_resample == 1024
  assert motion_cmd.motion_replay_fraction == 0.75
  assert not motion_cmd.motion_curriculum_gate
  assert not motion_cmd.motion_curriculum_ordered_loading

  # Reference projection clamps the tracked anchor to a bounded window around
  # the robot so sustained fast walking is learnable (the absolute reference
  # otherwise walks away faster than the robot can follow -> unbounded drift ->
  # death + out-of-distribution freeze). Mirrors the balance_finetune recipe.
  assert motion_cmd.reference_projection_enabled
  assert motion_cmd.reference_projection_max_forward == 0.18
  # global_root_pos std widened 0.6 -> 1.5 so a multi-metre position lag keeps a
  # catch-up gradient instead of saturating to zero once the robot falls behind
  # on a sustained fast walk (where it otherwise freezes / marches in place).
  assert cfg.rewards["motion_global_root_pos"].weight == 1.0
  assert cfg.rewards["motion_global_root_pos"].params["std"] == 1.5
  # Heading reward widened (std 0.4 -> 0.7) and strengthened (weight 0.5 ->
  # 0.75) so absolute yaw keeps a recovery gradient and competes with the
  # locally-satisfiable joint/feet terms a shuffle gait maxes out.
  assert cfg.rewards["motion_global_root_ori"].weight == 0.75
  assert cfg.rewards["motion_global_root_ori"].params["std"] == 0.7
  # Velocity tracking strengthened (weight 1.0 -> 2.0, std 1.0 -> 0.6): unlike
  # position, velocity error is informative at any lag, so this is the
  # non-saturating "move at the reference's forward speed" signal that stops the
  # robot freezing when behind. 59% of the dataset needs sustained >0.8 m/s.
  assert cfg.rewards["motion_body_lin_vel"].weight == 2.0
  assert cfg.rewards["motion_body_lin_vel"].params["std"] == 0.6
  assert cfg.rewards["motion_feet_pos"].weight == 0.8
  assert cfg.rewards["motion_feet_pos"].params["std"] == 0.2
  # One-sided swing-clearance term (tight std) to stop the shuffle gait; a
  # symmetric height reward is diluted by stance time and does not fix it.
  assert cfg.rewards["motion_feet_swing_clearance"].weight == 2.0
  assert cfg.rewards["motion_feet_swing_clearance"].params["std"] == 0.03
  # Wider-std twin so turn-in-place motions (~0.10-0.14 m foot lift, vs ~0.06 m
  # for walking) keep a recovery gradient instead of saturating the tight term
  # flat once the under-lift exceeds a few std.
  assert cfg.rewards["motion_feet_swing_clearance_coarse"].weight == 1.0
  assert cfg.rewards["motion_feet_swing_clearance_coarse"].params["std"] == 0.10
  assert cfg.rewards["motion_pose_body_pos"].weight == 0.0
  assert cfg.terminations["pose_feet_body_pos"].params["threshold"] == 1.00

  rl_cfg = load_rl_cfg("Mjlab-Tracking-Flat-VR-M3-1-Pose")
  assert isinstance(rl_cfg, RslRlOnPolicyRunnerCfg)
  assert rl_cfg.obs_groups["actor_g1"] == ("actor_g1",)
  assert rl_cfg.obs_groups["actor_teleop"] == ("actor_teleop",)
  assert rl_cfg.obs_groups["actor_smpl"] == ("actor_smpl",)
  assert rl_cfg.obs_groups["tokenizer"] == ("tokenizer",)
  assert rl_cfg.actor.class_name.endswith("UniversalTokenActor")
  assert rl_cfg.algorithm.class_name.endswith("UniversalTokenPPO")

  assert "motion_pose_body_pos" in cfg.rewards
  # Plan A: dense tracking terms drive the policy; the single local-pose term is
  # disabled (weight 0) in favor of body/velocity/joint terms.
  assert cfg.rewards["motion_body_pos"].weight > 0.0
  assert cfg.rewards["motion_body_ang_vel"].weight > 0.0
  assert cfg.rewards["motion_leg_joint_pos"].weight > 0.0
  assert "knee_ground_contact" in cfg.terminations
  knee_sensor = cast(
    ContactSensorCfg,
    next(s for s in cfg.scene.sensors or () if s.name == "knee_ground_contact"),
  )
  assert knee_sensor.primary.mode == "body"
  sensor_names = {s.name for s in cfg.scene.sensors or ()}
  assert "knee_ground_contact" in sensor_names


def test_vr_m3_1_tracking_has_correct_action_scale(
  vr_m3_1_tracking_task_ids: list[str],
) -> None:
  """VR M3.1 tracking tasks should use the tracking action scale."""
  for task_id in vr_m3_1_tracking_task_ids:
    cfg = load_env_cfg(task_id)

    assert "joint_pos" in cfg.actions, f"Task {task_id} missing 'joint_pos' action"

    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg), (
      f"Task {task_id} joint_pos action is not JointPositionActionCfg"
    )

    assert joint_pos_action.scale == VR_M3_1_TRACKING_ACTION_SCALE, (
      f"Task {task_id} action scale mismatch, expected tracking action scale"
    )
