"""Retarget VR head/hand poses to VR-M3.1 teleop observation targets.

This module is pure (torch math only, no MuJoCo/env dependency) so it can be
unit-tested directly. It maps the three tracked VR poses (headset + two
controllers) onto the ``teleop_3point_*`` targets the trained policy expects:
positions and orientations of ``(left_wrist, right_wrist, head)`` expressed in
the robot root (pelvis) frame.

Mapping strategy: at calibration the user holds a neutral standing pose; the
robot is then assumed to be at its own neutral pose. During teleop, each VR
point's displacement from its calibration pose is scaled by the arm-span ratio
and added to the robot's neutral target. This avoids having to align the human
and robot skeletons in absolute terms and degrades gracefully across users.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  quat_from_matrix,
  quat_mul,
  yaw_quat,
)

if TYPE_CHECKING:
  from mjlab.tasks.tracking.teleop.pose_source import TeleopFrame

# Canonical body order for the teleop targets. Must match the order in which
# these bodies appear in ``POSE_BODY_NAMES`` (left wrist before right wrist
# before head), because the teleop observations index bodies in that order.
TELEOP_BODY_NAMES: tuple[str, str, str] = (
  "left_wrist_yaw_link",
  "right_wrist_yaw_link",
  "head_yaw_link",
)

# Axis remap from the OpenXR world frame (x=right, y=up, z=back) to the MuJoCo
# robot world frame (x=forward, y=left, z=up): x_r=-z_vr, y_r=-x_vr, z_r=y_vr.
_VR_TO_ROBOT = torch.tensor(
  [
    [0.0, 0.0, -1.0],
    [-1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
  ]
)


@dataclass
class TeleopTargets:
  """Retargeted targets ready for ``MotionCommand.set_live_teleop`` (unbatched).

  Shapes: ``pos_local (3, 3)``, ``orn_local_quat (3, 4)`` (wxyz),
  ``joint_pos (num_joints,)``, ``root_yaw`` scalar tensor (world-frame delta
  from the calibration heading).
  """

  pos_local: torch.Tensor
  orn_local_quat: torch.Tensor
  joint_pos: torch.Tensor
  root_yaw: torch.Tensor


def vr_pos_to_robot(pos_vr: torch.Tensor) -> torch.Tensor:
  """Remap a VR-frame position (..., 3) to the robot world frame."""
  remap = _VR_TO_ROBOT.to(pos_vr)
  return pos_vr @ remap.T


def vr_quat_to_robot(quat_vr: torch.Tensor) -> torch.Tensor:
  """Remap a VR-frame orientation quat (..., 4, wxyz) to the robot world frame."""
  remap = _VR_TO_ROBOT.to(quat_vr)
  mat_vr = matrix_from_quat(quat_vr)
  mat_robot = remap @ mat_vr @ remap.T
  return quat_from_matrix(mat_robot)


def yaw_angle(quat: torch.Tensor) -> torch.Tensor:
  """Yaw angle (radians) of a quaternion (..., 4, wxyz), robust to pitch/roll."""
  yq = yaw_quat(quat)
  return 2.0 * torch.atan2(yq[..., 3], yq[..., 0])


def _rotation_about_z(angle: torch.Tensor) -> torch.Tensor:
  """3x3 rotation matrix about +z for a scalar angle tensor."""
  c = torch.cos(angle)
  s = torch.sin(angle)
  zero = torch.zeros_like(angle)
  one = torch.ones_like(angle)
  return torch.stack(
    [
      torch.stack([c, -s, zero]),
      torch.stack([s, c, zero]),
      torch.stack([zero, zero, one]),
    ]
  )


@dataclass
class TeleopCalibration:
  """Per-user reference captured in the neutral standing pose."""

  # VR points remapped into the robot world frame, in [left, right, head] order.
  ref_pos_robot: torch.Tensor  # (3, 3)
  ref_quat_robot: torch.Tensor  # (3, 4)
  head_yaw: torch.Tensor  # scalar
  scale: torch.Tensor  # scalar arm-span ratio


class TeleopRetargeter:
  """Map ``TeleopFrame`` poses to teleop observation targets.

  Args:
    neutral_pos_local: ``(3, 3)`` robot neutral positions of
      ``(left_wrist, right_wrist, head)`` in the pelvis frame.
    neutral_orn_local: ``(3, 4)`` robot neutral orientations (wxyz), same order.
    default_joint_pos: ``(num_joints,)`` standing joint targets fed to the leg
      observation (the policy infers the lower body from the 3 points).
    scale_limits: clamp range for the auto-computed arm-span scale.
  """

  def __init__(
    self,
    neutral_pos_local: torch.Tensor,
    neutral_orn_local: torch.Tensor,
    default_joint_pos: torch.Tensor,
    scale_limits: tuple[float, float] = (0.5, 2.0),
  ) -> None:
    self.neutral_pos_local = neutral_pos_local
    self.neutral_orn_local = neutral_orn_local
    self.default_joint_pos = default_joint_pos
    self.scale_limits = scale_limits
    self.calibration: TeleopCalibration | None = None

  def _frame_to_robot(self, frame: "TeleopFrame") -> tuple[torch.Tensor, torch.Tensor]:
    """Stack [left, right, head] poses and remap into the robot world frame."""
    pos_vr = torch.stack([frame.left_pos, frame.right_pos, frame.head_pos]).to(
      self.neutral_pos_local
    )
    quat_vr = torch.stack([frame.left_quat, frame.right_quat, frame.head_quat]).to(
      self.neutral_orn_local
    )
    return vr_pos_to_robot(pos_vr), vr_quat_to_robot(quat_vr)

  def calibrate(self, frame: "TeleopFrame") -> TeleopCalibration:
    """Capture the neutral reference and arm-span scale from a frame."""
    ref_pos, ref_quat = self._frame_to_robot(frame)
    human_span = torch.norm(ref_pos[0] - ref_pos[1])
    robot_span = torch.norm(self.neutral_pos_local[0] - self.neutral_pos_local[1])
    scale = (robot_span / torch.clamp(human_span, min=1e-3)).clamp(*self.scale_limits)
    self.calibration = TeleopCalibration(
      ref_pos_robot=ref_pos,
      ref_quat_robot=ref_quat,
      head_yaw=yaw_angle(ref_quat[2]),
      scale=scale,
    )
    return self.calibration

  def retarget(self, frame: "TeleopFrame") -> TeleopTargets:
    """Map a live VR frame to teleop targets. Calibrates lazily on first call."""
    if self.calibration is None:
      self.calibrate(frame)
    calib = self.calibration
    assert calib is not None

    pos_robot, quat_robot = self._frame_to_robot(frame)
    head_yaw = yaw_angle(quat_robot[2])
    delta_yaw = head_yaw - calib.head_yaw

    # Displacement from the calibration pose, rotated into the current facing
    # (headset-yaw) frame, then scaled and added to the robot neutral pose.
    disp = (pos_robot - calib.ref_pos_robot) * calib.scale
    rot = _rotation_about_z(-delta_yaw).to(disp)  # (3, 3)
    disp_facing = disp @ rot.T
    pos_local = self.neutral_pos_local + disp_facing

    # Orientation: apply the VR orientation delta (in the facing frame) on top of
    # the robot neutral orientation for each point.
    quat_delta = quat_mul(quat_robot, _quat_conjugate(calib.ref_quat_robot))
    orn_local_quat = quat_mul(quat_delta, self.neutral_orn_local)

    return TeleopTargets(
      pos_local=pos_local,
      orn_local_quat=orn_local_quat,
      joint_pos=self.default_joint_pos,
      root_yaw=delta_yaw,
    )


def _quat_conjugate(q: torch.Tensor) -> torch.Tensor:
  return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)
