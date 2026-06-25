from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import (
  quat_apply_inverse,
  quat_error_magnitude,
  yaw_quat,
)

from .commands import MotionCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def _get_body_indexes(
  command: MotionCommand, body_names: tuple[str, ...] | None
) -> list[int]:
  return [
    i
    for i, name in enumerate(command.cfg.body_names)
    if (body_names is None) or (name in body_names)
  ]


def _soften_transition_reward(
  command: MotionCommand, reward: torch.Tensor
) -> torch.Tensor:
  if not command.cfg.transition_soft_reward_enabled:
    return reward
  softness = command.cfg.transition_reward_softness
  transition_amount = (1.0 - command.transition_alpha) * command.is_transition.float()
  softened_target = torch.ones_like(reward)
  return reward * (1.0 - softness * transition_amount) + (
    softened_target * softness * transition_amount
  )


def motion_global_anchor_position_error_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  error = torch.sum(
    torch.square(command.anchor_pos_w - command.robot_anchor_pos_w), dim=-1
  )
  return _soften_transition_reward(command, torch.exp(-error / std**2))


def motion_global_anchor_orientation_error_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  error = quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w) ** 2
  return _soften_transition_reward(command, torch.exp(-error / std**2))


def motion_global_anchor_xy_error_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  error = torch.sum(
    torch.square(command.anchor_pos_w[:, :2] - command.robot_anchor_pos_w[:, :2]),
    dim=-1,
  )
  return _soften_transition_reward(command, torch.exp(-error / std**2))


def motion_global_anchor_height_error_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  error = torch.square(command.anchor_pos_w[:, 2] - command.robot_anchor_pos_w[:, 2])
  return _soften_transition_reward(command, torch.exp(-error / std**2))


def motion_relative_body_height_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)
  error = torch.square(
    command.body_pos_relative_w[:, body_indexes, 2]
    - command.robot_body_pos_w[:, body_indexes, 2]
  )
  return _soften_transition_reward(command, torch.exp(-error.mean(-1) / std**2))


def motion_feet_swing_clearance_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  """One-sided swing-clearance reward: penalise a foot only when it is *below*
  the reference foot height.

  A plain (symmetric) foot-height error reward is averaged over the whole
  episode, where stance frames (both feet on the ground, error ~0) dominate and
  dilute the few swing frames, so the policy can keep ~0.9 reward while dragging
  the swing foot (a shuffle). Here the error is ``relu(ref_z - robot_z)``: it is
  ~0 during stance (both feet at the same height) and only fires when the
  reference lifts a foot and the robot does not, so the gradient concentrates on
  exactly the under-lift that causes shuffling. It is one-sided, so lifting the
  foot is never discouraged.
  """
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)
  under_lift = torch.clamp(
    command.body_pos_relative_w[:, body_indexes, 2]
    - command.robot_body_pos_w[:, body_indexes, 2],
    min=0.0,
  )
  error = torch.square(under_lift).mean(-1)
  return _soften_transition_reward(command, torch.exp(-error / std**2))


def motion_relative_body_position_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)
  error = torch.sum(
    torch.square(
      command.body_pos_relative_w[:, body_indexes]
      - command.robot_body_pos_w[:, body_indexes]
    ),
    dim=-1,
  )
  return _soften_transition_reward(command, torch.exp(-error.mean(-1) / std**2))


def motion_local_body_position_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)

  ref_delta_w = command.body_pos_w[:, body_indexes] - command.anchor_pos_w[:, None, :]
  robot_delta_w = (
    command.robot_body_pos_w[:, body_indexes] - command.robot_anchor_pos_w[:, None, :]
  )
  num_bodies = len(body_indexes)
  ref_anchor_quat_w = command.anchor_quat_w[:, None, :].expand(
    -1,
    num_bodies,
    -1,
  )
  robot_anchor_quat_w = command.robot_anchor_quat_w[:, None, :].expand(
    -1,
    num_bodies,
    -1,
  )

  ref_pos_local = quat_apply_inverse(
    ref_anchor_quat_w.reshape(-1, 4),
    ref_delta_w.reshape(-1, 3),
  ).reshape(env.num_envs, num_bodies, 3)
  robot_pos_local = quat_apply_inverse(
    robot_anchor_quat_w.reshape(-1, 4),
    robot_delta_w.reshape(-1, 3),
  ).reshape(env.num_envs, num_bodies, 3)

  error = torch.sum(torch.square(ref_pos_local - robot_pos_local), dim=-1)
  env.extras["log"]["Metrics/motion_local_body_pos_error"] = torch.sqrt(
    error.mean(-1)
  ).mean()
  return _soften_transition_reward(command, torch.exp(-error.mean(-1) / std**2))


def motion_relative_body_orientation_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)
  error = (
    quat_error_magnitude(
      command.body_quat_relative_w[:, body_indexes],
      command.robot_body_quat_w[:, body_indexes],
    )
    ** 2
  )
  return _soften_transition_reward(command, torch.exp(-error.mean(-1) / std**2))


def motion_global_body_linear_velocity_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)
  error = torch.sum(
    torch.square(
      command.body_lin_vel_w[:, body_indexes]
      - command.robot_body_lin_vel_w[:, body_indexes]
    ),
    dim=-1,
  )
  return _soften_transition_reward(command, torch.exp(-error.mean(-1) / std**2))


def motion_global_body_angular_velocity_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)
  error = torch.sum(
    torch.square(
      command.body_ang_vel_w[:, body_indexes]
      - command.robot_body_ang_vel_w[:, body_indexes]
    ),
    dim=-1,
  )
  return _soften_transition_reward(command, torch.exp(-error.mean(-1) / std**2))


def motion_joint_position_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  joint_names: tuple[str, ...] | None = None,
  soften_transition: bool = True,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  if joint_names is None:
    reference_joint_pos = command.joint_pos
    robot_joint_pos = command.robot_joint_pos
  else:
    joint_ids = command.robot.find_joints(joint_names, preserve_order=True)[0]
    joint_ids_tensor = torch.tensor(joint_ids, dtype=torch.long, device=env.device)
    reference_joint_pos = command.joint_pos[:, joint_ids_tensor]
    robot_joint_pos = command.robot_joint_pos[:, joint_ids_tensor]
  error = torch.mean(torch.square(reference_joint_pos - robot_joint_pos), dim=-1)
  reward = torch.exp(-error / std**2)
  if not soften_transition:
    return reward
  return _soften_transition_reward(command, reward)


def self_collision_cost(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 10.0,
) -> torch.Tensor:
  """Penalize self-collisions.

  When the sensor provides force history (from ``history_length > 0``),
  counts substeps where any contact force exceeds *force_threshold*.
  Falls back to the instantaneous ``found`` count otherwise.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  if data.force_history is not None:
    # force_history: [B, N, H, 3]
    force_mag = torch.norm(data.force_history, dim=-1)  # [B, N, H]
    hit = (force_mag > force_threshold).any(dim=1)  # [B, H]
    return hit.sum(dim=-1).float()  # [B]
  assert data.found is not None
  return data.found.squeeze(-1)


def feet_slip_l2(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Penalize horizontal foot velocity while the foot is in ground contact."""
  asset = env.scene[asset_cfg.name]
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.found is not None
  in_contact = (sensor.data.found > 0).float()
  foot_vel_xy = asset.data.body_link_lin_vel_w[:, asset_cfg.body_ids, :2]
  cost = torch.sum(torch.square(foot_vel_xy).sum(dim=-1) * in_contact, dim=1)
  num_in_contact = torch.sum(in_contact)
  slip_speed = torch.norm(foot_vel_xy, dim=-1)
  env.extras["log"]["Metrics/tracking_feet_slip_mean"] = torch.sum(
    slip_speed * in_contact
  ) / torch.clamp(num_in_contact, min=1)
  return cost


def robot_anchor_xy_velocity_l2(
  env: ManagerBasedRlEnv,
  command_name: str,
) -> torch.Tensor:
  """Penalize horizontal drift of the tracked anchor body."""
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  return torch.sum(torch.square(command.robot_anchor_lin_vel_w[:, :2]), dim=-1)


def root_height_asymmetric_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  under_track_scale: float = 0.3,
  over_drop_scale: float = 2.0,
) -> torch.Tensor:
  """Reward root height while allowing shallow under-tracking of hard references."""
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  robot_z = command.robot_anchor_pos_w[:, 2]
  ref_z = command.anchor_pos_w[:, 2]
  under_track = torch.clamp(robot_z - ref_z, min=0.0)
  over_drop = torch.clamp(ref_z - robot_z, min=0.0)
  cost = under_track_scale * torch.square(under_track)
  cost += over_drop_scale * torch.square(over_drop)
  return _soften_transition_reward(command, torch.exp(-cost / std**2))


def no_unwanted_backward_drift_l2(
  env: ManagerBasedRlEnv,
  command_name: str,
  threshold: float = 0.02,
  ref_backward_threshold: float = -0.05,
) -> torch.Tensor:
  """Penalize backward root velocity unless the projected reference asks for it."""
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  robot_yaw = yaw_quat(command.robot_anchor_quat_w)
  robot_lin_vel_b = quat_apply_inverse(robot_yaw, command.robot_anchor_lin_vel_w)
  ref_lin_vel_b = quat_apply_inverse(robot_yaw, command.anchor_lin_vel_w)
  no_backward_cmd = ref_lin_vel_b[:, 0] > ref_backward_threshold
  backward_vel = torch.clamp(-robot_lin_vel_b[:, 0] - threshold, min=0.0)
  return no_backward_cmd.float() * torch.square(backward_vel)


def feet_lateral_distance_error_exp(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  target_width: float,
  std: float,
) -> torch.Tensor:
  """Reward keeping the two feet near a target lateral stance width."""
  asset = env.scene[asset_cfg.name]
  body_ids = asset_cfg.body_ids
  if not isinstance(body_ids, list) or len(body_ids) != 2:
    raise ValueError("feet_lateral_distance_error_exp expects exactly two bodies")
  feet_pos_w = asset.data.body_link_pos_w[:, body_ids, :]
  foot_delta_w = feet_pos_w[:, 1] - feet_pos_w[:, 0]
  root_yaw = yaw_quat(asset.data.root_link_quat_w)
  foot_delta_b = quat_apply_inverse(root_yaw, foot_delta_w)
  lateral_width = torch.abs(foot_delta_b[:, 1])
  env.extras["log"]["Metrics/feet_lateral_width"] = lateral_width.mean()
  error = torch.square(lateral_width - target_width)
  return torch.exp(-error / std**2)


def pelvis_height_above_minimum_exp(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  minimum_height: float,
  std: float,
) -> torch.Tensor:
  """Reward pelvis/root height above a minimum height without chasing raw reference."""
  asset = env.scene[asset_cfg.name]
  body_ids = asset_cfg.body_ids
  if not isinstance(body_ids, list) or len(body_ids) != 1:
    raise ValueError("pelvis_height_above_minimum_exp expects exactly one body")
  pelvis_z = asset.data.body_link_pos_w[:, body_ids[0], 2]
  floor_z = env.scene.env_origins[:, 2]
  height = pelvis_z - floor_z
  env.extras["log"]["Metrics/pelvis_height"] = height.mean()
  shortfall = torch.clamp(minimum_height - height, min=0.0)
  return torch.exp(-torch.square(shortfall) / std**2)
