from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  quat_apply,
  quat_apply_inverse,
  quat_inv,
  quat_mul,
  subtract_frame_transforms,
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


def motion_anchor_pos_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))

  pos, _ = subtract_frame_transforms(
    command.robot_anchor_pos_w,
    command.robot_anchor_quat_w,
    command.anchor_pos_w,
    command.anchor_quat_w,
  )

  return pos.view(env.num_envs, -1)


def motion_anchor_ori_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))

  if command.live_teleop_active:
    return command.live_anchor_ori_b

  _, ori = subtract_frame_transforms(
    command.robot_anchor_pos_w,
    command.robot_anchor_quat_w,
    command.anchor_pos_w,
    command.anchor_quat_w,
  )
  mat = matrix_from_quat(ori)
  return mat[..., :2].reshape(mat.shape[0], -1)


def motion_encoder_index(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  return command.encoder_index


def motion_compliance(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  return command.compliance


def last_action_padded(
  env: ManagerBasedRlEnv,
  target_dim: int,
  action_name: str | None = None,
) -> torch.Tensor:
  """Return last action padded/cropped to a deploy observation width."""
  if action_name is None:
    action = env.action_manager.action
  else:
    action = env.action_manager.get_term(action_name).raw_action

  if action.shape[-1] == target_dim:
    return action
  if action.shape[-1] > target_dim:
    return action[..., :target_dim]

  padding = action.new_zeros((*action.shape[:-1], target_dim - action.shape[-1]))
  return torch.cat([action, padding], dim=-1)


def motion_command_multi_future(
  env: ManagerBasedRlEnv, command_name: str
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  return command.command_multi_future


def motion_anchor_ori_b_multi_future(
  env: ManagerBasedRlEnv, command_name: str
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  return command.motion_anchor_ori_b_multi_future


def motion_anchor_pos_b_multi_future(
  env: ManagerBasedRlEnv, command_name: str
) -> torch.Tensor:
  """Future reference-root positions in the robot-anchor frame.

  The translation command for the teleop/smpl encoders: it tells the policy
  where the operator root is heading so the robot follows (walk/jump) instead of
  marching in place. Under live teleop the deploy bridge supplies this via
  ``set_live_teleop``; otherwise it comes from the replayed motion.
  """
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  if command.live_teleop_active:
    return command.live_anchor_pos_b_multi_future
  return command.anchor_pos_b_multi_future


def motion_joint_pos_multi_future_select(
  env: ManagerBasedRlEnv,
  command_name: str,
  joint_names: tuple[str, ...],
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  joint_ids = command.robot.find_joints(joint_names, preserve_order=True)[0]
  if command.live_teleop_active:
    joint_pos = command.live_joint_pos_future[..., joint_ids]
    return joint_pos.reshape(env.num_envs, -1)
  joint_pos = command.joint_pos_multi_future_for_smpl[..., joint_ids]
  return joint_pos.reshape(env.num_envs, -1)


def motion_body_pos_local_select(
  env: ManagerBasedRlEnv,
  command_name: str,
  body_names: tuple[str, ...],
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  if command.live_teleop_active:
    return command.live_3point_pos_local.reshape(env.num_envs, -1)
  body_indexes = _get_body_indexes(command, body_names)
  body_pos_w = command.body_pos_w[:, body_indexes]
  ref_delta_w = body_pos_w - command.anchor_pos_w[:, None, :]
  anchor_quat_w = command.anchor_quat_w[:, None, :].expand(-1, len(body_indexes), -1)
  pos_b = quat_apply_inverse(
    anchor_quat_w.reshape(-1, 4), ref_delta_w.reshape(-1, 3)
  ).reshape(env.num_envs, len(body_indexes), 3)
  return pos_b.reshape(env.num_envs, -1)


def motion_body_ori_local_select(
  env: ManagerBasedRlEnv,
  command_name: str,
  body_names: tuple[str, ...],
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  if command.live_teleop_active:
    n = command.live_3point_orn_quat.shape[1]
    mat = matrix_from_quat(command.live_3point_orn_quat.reshape(-1, 4)).reshape(
      env.num_envs, n, 3, 3
    )
    return mat[..., :2].reshape(env.num_envs, -1)
  body_indexes = _get_body_indexes(command, body_names)
  body_quat_w = command.body_quat_w[:, body_indexes]
  anchor_quat_inv = quat_inv(command.anchor_quat_w)[:, None, :].expand(
    -1, len(body_indexes), -1
  )
  body_quat_b = quat_mul(anchor_quat_inv.reshape(-1, 4), body_quat_w.reshape(-1, 4))
  mat = matrix_from_quat(body_quat_b).reshape(env.num_envs, len(body_indexes), 3, 3)
  return mat[..., :2].reshape(env.num_envs, -1)


def motion_body_pos_local(
  env: ManagerBasedRlEnv,
  command_name: str,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)
  if not body_indexes:
    return torch.empty(env.num_envs, 0, device=env.device)

  body_pos_w = command.body_pos_w[:, body_indexes]
  ref_delta_w = body_pos_w - command.anchor_pos_w[:, None, :]
  anchor_quat_w = command.anchor_quat_w[:, None, :].expand(
    -1,
    len(body_indexes),
    -1,
  )
  ref_pos_local = quat_apply_inverse(
    anchor_quat_w.reshape(-1, 4),
    ref_delta_w.reshape(-1, 3),
  ).reshape(env.num_envs, len(body_indexes), 3)
  return ref_pos_local.reshape(env.num_envs, -1)


def motion_smpl_joints_local(
  env: ManagerBasedRlEnv,
  command_name: str,
  root_index: int = 0,
) -> torch.Tensor:
  del root_index
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  smpl_joints = command.smpl_joints
  root_quat = command.smpl_root_quat_w[:, None, :].expand(-1, smpl_joints.shape[-2], -1)
  local_joints = quat_apply(
    quat_inv(root_quat.reshape(-1, 4)),
    smpl_joints.reshape(-1, 3),
  ).reshape_as(smpl_joints)
  return local_joints.reshape(env.num_envs, -1)


def motion_smpl_joints_multi_future_local(
  env: ManagerBasedRlEnv,
  command_name: str,
  joints_idx: tuple[int, ...] | None = None,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  smpl_joints = command.smpl_joints_multi_future
  root_quat = command.smpl_root_quat_w_multi_future.unsqueeze(-2).expand(
    -1, -1, smpl_joints.shape[-2], -1
  )
  local_joints = quat_apply(
    quat_inv(root_quat.reshape(-1, 4)),
    smpl_joints.reshape(-1, 3),
  ).reshape_as(smpl_joints)
  if joints_idx is not None:
    local_joints = local_joints[..., list(joints_idx), :]
  return local_joints.reshape(env.num_envs, -1)


def motion_smpl_root_ori_b_multi_future(
  env: ManagerBasedRlEnv, command_name: str
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  return command.smpl_root_quat_w_dif_l_multi_future.reshape(env.num_envs, -1)


def motion_joint_pos_multi_future_for_smpl(
  env: ManagerBasedRlEnv,
  command_name: str,
  joint_names: tuple[str, ...],
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  joint_ids = command.robot.find_joints(joint_names, preserve_order=True)[0]
  joint_pos = command.joint_pos_multi_future_for_smpl[..., joint_ids]
  return joint_pos.reshape(env.num_envs, -1)


def robot_body_pos_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))

  num_bodies = len(command.cfg.body_names)
  pos_b, _ = subtract_frame_transforms(
    command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
    command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
    command.robot_body_pos_w,
    command.robot_body_quat_w,
  )

  return pos_b.view(env.num_envs, -1)


def robot_body_ori_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))

  num_bodies = len(command.cfg.body_names)
  _, ori_b = subtract_frame_transforms(
    command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
    command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
    command.robot_body_pos_w,
    command.robot_body_quat_w,
  )
  mat = matrix_from_quat(ori_b)
  return mat[..., :2].reshape(mat.shape[0], -1)
