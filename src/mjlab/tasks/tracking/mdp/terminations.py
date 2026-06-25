from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import (
  quat_apply_inverse,
  quat_error_magnitude,
  yaw_quat,
)

from .commands import MotionCommand
from .rewards import _get_body_indexes

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.scene_entity_config import SceneEntityCfg


def bad_anchor_pos(
  env: ManagerBasedRlEnv, command_name: str, threshold: float
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  return (
    torch.norm(command.anchor_pos_w - command.robot_anchor_pos_w, dim=1) > threshold
  )


def bad_anchor_pos_z_only(
  env: ManagerBasedRlEnv, command_name: str, threshold: float
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  return (
    torch.abs(command.anchor_pos_w[:, -1] - command.robot_anchor_pos_w[:, -1])
    > threshold
  )


def bad_anchor_xy(
  env: ManagerBasedRlEnv, command_name: str, threshold: float
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  return (
    torch.norm(command.anchor_pos_w[:, :2] - command.robot_anchor_pos_w[:, :2], dim=1)
    > threshold
  )


def bad_anchor_height(
  env: ManagerBasedRlEnv, command_name: str, threshold: float
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  return (
    torch.abs(command.anchor_pos_w[:, 2] - command.robot_anchor_pos_w[:, 2]) > threshold
  )


def illegal_contact(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 10.0,
) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  if data.force_history is not None:
    force_mag = torch.norm(data.force_history, dim=-1)
    return (force_mag > force_threshold).any(dim=-1).any(dim=-1)
  assert data.found is not None
  return torch.any(data.found, dim=-1)


def bad_anchor_ori(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg, command_name: str, threshold: float
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]

  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  motion_projected_gravity_b = quat_apply_inverse(
    command.anchor_quat_w, asset.data.gravity_vec_w
  )

  robot_projected_gravity_b = quat_apply_inverse(
    command.robot_anchor_quat_w, asset.data.gravity_vec_w
  )

  return (
    motion_projected_gravity_b[:, 2] - robot_projected_gravity_b[:, 2]
  ).abs() > threshold


def bad_anchor_yaw(
  env: ManagerBasedRlEnv, command_name: str, threshold: float
) -> torch.Tensor:
  """Terminate on large anchor heading (yaw) error.

  ``bad_anchor_ori`` compares the projected-gravity z-component of the motion
  and robot anchor frames, which captures pitch/roll (tilt) but is invariant to
  yaw: an upright robot facing the wrong way has near-zero tilt error and never
  trips it. Heading therefore had no hard constraint, and the only absolute
  heading signal (the ``motion_global_root_ori`` reward) saturates to a flat
  gradient at large errors, so once turn-in-place motions enter the pool the
  policy abandons rotation and shuffles in place. This isolates the yaw
  component of both anchors and terminates when they diverge past *threshold*
  radians, mirroring how ``bad_anchor_xy`` constrains horizontal drift.
  """
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  yaw_error = quat_error_magnitude(
    yaw_quat(command.anchor_quat_w), yaw_quat(command.robot_anchor_quat_w)
  )
  return yaw_error > threshold


def bad_motion_body_pos(
  env: ManagerBasedRlEnv,
  command_name: str,
  threshold: float,
  body_names: tuple[str, ...] | None = None,
  threshold_adaptive: bool = False,
  down_threshold: float | None = None,
  root_height_threshold: float = 0.5,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))

  body_indexes = _get_body_indexes(command, body_names)
  error = torch.norm(
    command.body_pos_relative_w[:, body_indexes]
    - command.robot_body_pos_w[:, body_indexes],
    dim=-1,
  )
  if threshold_adaptive and down_threshold is not None:
    # Relax the body-position tolerance for environments whose reference root is
    # low (e.g. a deep squat), where exact tracking is both harder and less
    # safety-critical than staying upright. Mirrors Groot's adaptive height
    # termination: the policy may under-track a deep pose instead of being
    # killed for it, so it learns "go low but don't fall."
    thresh = error.new_full((error.shape[0],), threshold)
    low_ref = command.anchor_pos_w[:, 2] < root_height_threshold
    thresh[low_ref] = down_threshold
    return torch.any(error > thresh[:, None], dim=-1)
  return torch.any(error > threshold, dim=-1)


def bad_motion_body_pos_z_only(
  env: ManagerBasedRlEnv,
  command_name: str,
  threshold: float,
  body_names: tuple[str, ...] | None = None,
  threshold_adaptive: bool = False,
  down_threshold: float | None = None,
  root_height_threshold: float = 0.5,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))

  body_indexes = _get_body_indexes(command, body_names)
  error = torch.abs(
    command.body_pos_relative_w[:, body_indexes, -1]
    - command.robot_body_pos_w[:, body_indexes, -1]
  )
  if threshold_adaptive and down_threshold is not None:
    # Relax the per-body height tolerance for environments whose reference root
    # is low (deep squat / stoop), where matching an extreme foot height is both
    # harder and less safety-critical than staying upright. Real falls are still
    # caught by the fell_over_height / fell_over_orientation terminations.
    thresh = error.new_full((error.shape[0],), threshold)
    low_ref = command.anchor_pos_w[:, 2] < root_height_threshold
    thresh[low_ref] = down_threshold
    return torch.any(error > thresh[:, None], dim=-1)
  return torch.any(error > threshold, dim=-1)
