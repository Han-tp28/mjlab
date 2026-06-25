"""Tests for VR-M3.1 teleop retargeting, pose sources, and observation injection.

The OpenXR hardware path (``OpenXrPoseSource``) is intentionally not exercised
here; it needs a headset and an OpenXR runtime.
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from mjlab.tasks.tracking.mdp import observations as obs
from mjlab.tasks.tracking.teleop import (
  MockPoseSource,
  TeleopFrame,
  TeleopRetargeter,
)
from mjlab.tasks.tracking.teleop.retarget import (
  vr_pos_to_robot,
  vr_quat_to_robot,
  yaw_angle,
)

_IDENTITY = torch.tensor([1.0, 0.0, 0.0, 0.0])


def _neutral():
  pos = torch.tensor(
    [
      [0.10, 0.25, 0.10],  # left wrist
      [0.10, -0.25, 0.10],  # right wrist
      [0.00, 0.00, 0.60],  # head
    ]
  )
  orn = _IDENTITY.expand(3, 4).clone()
  joints = torch.zeros(12)
  return pos, orn, joints


def _calib_frame() -> TeleopFrame:
  # OpenXR world (x=right, y=up, z=back); hands 0.5 m apart, head at eye height.
  return TeleopFrame(
    head_pos=torch.tensor([0.0, 1.6, 0.0]),
    head_quat=_IDENTITY.clone(),
    left_pos=torch.tensor([-0.25, 1.2, -0.20]),
    left_quat=_IDENTITY.clone(),
    right_pos=torch.tensor([0.25, 1.2, -0.20]),
    right_quat=_IDENTITY.clone(),
  )


def test_vr_pos_to_robot_axes():
  # right -> -y, up -> +z, forward(-z) -> +x.
  assert torch.allclose(
    vr_pos_to_robot(torch.tensor([1.0, 0, 0])), torch.tensor([0.0, -1, 0])
  )
  assert torch.allclose(
    vr_pos_to_robot(torch.tensor([0.0, 1, 0])), torch.tensor([0.0, 0, 1])
  )
  assert torch.allclose(
    vr_pos_to_robot(torch.tensor([0.0, 0, -1])), torch.tensor([1.0, 0, 0])
  )


def test_vr_quat_identity_roundtrip():
  assert torch.allclose(vr_quat_to_robot(_IDENTITY), _IDENTITY, atol=1e-6)


def test_yaw_about_vr_up_maps_to_robot_yaw():
  theta = 0.5
  # Rotation about VR up-axis (y), wxyz.
  q = torch.tensor([math.cos(theta / 2), 0.0, math.sin(theta / 2), 0.0])
  q_robot = vr_quat_to_robot(q)
  assert yaw_angle(q_robot).item() == pytest.approx(theta, abs=1e-4)


def test_retarget_at_calibration_is_neutral():
  pos, orn, joints = _neutral()
  rt = TeleopRetargeter(pos, orn, joints)
  frame = _calib_frame()
  rt.calibrate(frame)
  assert rt.calibration is not None
  assert rt.calibration.scale.item() == pytest.approx(1.0, abs=1e-3)

  tgt = rt.retarget(frame)
  assert torch.allclose(tgt.pos_local, pos, atol=1e-5)
  assert tgt.root_yaw.item() == pytest.approx(0.0, abs=1e-5)
  assert torch.allclose(tgt.joint_pos, joints)


def test_retarget_hand_forward_moves_target_forward():
  pos, orn, joints = _neutral()
  rt = TeleopRetargeter(pos, orn, joints)
  rt.calibrate(_calib_frame())

  moved = _calib_frame()
  moved.left_pos = torch.tensor([-0.25, 1.2, -0.40])  # 0.2 m forward (-z)
  tgt = rt.retarget(moved)
  # Left wrist (index 0) should advance ~0.2 m along robot +x; others unchanged.
  assert tgt.pos_local[0, 0].item() == pytest.approx(pos[0, 0].item() + 0.2, abs=1e-3)
  assert torch.allclose(tgt.pos_local[1], pos[1], atol=1e-4)
  assert torch.allclose(tgt.pos_local[2], pos[2], atol=1e-4)


def test_retarget_output_shapes():
  pos, orn, joints = _neutral()
  rt = TeleopRetargeter(pos, orn, joints)
  tgt = rt.retarget(_calib_frame())
  assert tgt.pos_local.shape == (3, 3)
  assert tgt.orn_local_quat.shape == (3, 4)
  assert tgt.joint_pos.shape == (12,)
  assert tgt.root_yaw.shape == ()


def test_mock_source_shapes_and_motion():
  src = MockPoseSource(dt=0.02)
  f0 = src.poll()
  f1 = src.poll()
  for f in (f0, f1):
    assert f.head_pos.shape == (3,)
    assert f.left_quat.shape == (4,)
  # Hands should move between consecutive polls.
  assert not torch.allclose(f0.left_pos, f1.left_pos)


def test_replay_source(tmp_path):
  from mjlab.tasks.tracking.config.vr_m3_1.env_cfgs import POSE_BODY_NAMES
  from mjlab.tasks.tracking.teleop import ReplayPoseSource

  n = len(POSE_BODY_NAMES)
  path = tmp_path / "motion.npz"
  np.savez(
    path,
    body_pos_w=np.random.randn(5, n, 3).astype(np.float32),
    body_quat_w=np.tile([1.0, 0, 0, 0], (5, n, 1)).astype(np.float32),
  )
  src = ReplayPoseSource(path, POSE_BODY_NAMES)
  frame = src.poll()
  assert frame.head_pos.shape == (3,)
  assert frame.reset_calib is True  # first frame requests calibration


# --- Observation passthrough -------------------------------------------------


def _stub_env(command, num_envs=2) -> Any:
  return SimpleNamespace(
    num_envs=num_envs,
    command_manager=SimpleNamespace(get_term=lambda name: command),
  )


def _stub_command(num_envs=2, num_future=4, num_joints=12) -> Any:
  pos = torch.randn(num_envs, 3, 3)
  quat = _IDENTITY.expand(num_envs, 3, 4).clone()
  joints = torch.randn(num_envs, num_future, num_joints)
  anchor = torch.randn(num_envs, 6)
  anchor_pos = torch.randn(num_envs, num_future * 3)
  robot = SimpleNamespace(
    find_joints=lambda names, preserve_order=True: (list(range(len(names))), names)
  )
  return SimpleNamespace(
    live_teleop_active=True,
    live_3point_pos_local=pos,
    live_3point_orn_quat=quat,
    live_joint_pos_future=joints,
    live_anchor_ori_b=anchor,
    live_anchor_pos_b_multi_future=anchor_pos,
    robot=robot,
  )


def test_obs_body_pos_passthrough():
  cmd = _stub_command()
  env = _stub_env(cmd)
  out = obs.motion_body_pos_local_select(env, "motion", ("a", "b", "c"))
  assert torch.allclose(out, cmd.live_3point_pos_local.reshape(env.num_envs, -1))


def test_obs_body_ori_passthrough_format():
  cmd = _stub_command()
  env = _stub_env(cmd)
  out = obs.motion_body_ori_local_select(env, "motion", ("a", "b", "c"))
  # Identity quats -> first two columns of identity matrix per body (6 values).
  expected = torch.tensor([1.0, 0, 0, 1, 0, 0]).repeat(env.num_envs, 3)
  assert out.shape == (env.num_envs, 18)
  assert torch.allclose(out, expected, atol=1e-6)


def test_obs_anchor_ori_passthrough():
  cmd = _stub_command()
  env = _stub_env(cmd)
  out = obs.motion_anchor_ori_b(env, "motion")
  assert torch.allclose(out, cmd.live_anchor_ori_b)


def test_obs_anchor_pos_multi_future_passthrough():
  # Under live teleop the translation command must come from the deploy-supplied
  # buffer, not the (frozen) replayed motion.
  cmd = _stub_command()
  env = _stub_env(cmd)
  out = obs.motion_anchor_pos_b_multi_future(env, "motion")
  assert torch.allclose(out, cmd.live_anchor_pos_b_multi_future)


def test_obs_leg_joint_passthrough_selects_columns():
  cmd = _stub_command(num_future=4, num_joints=12)
  env = _stub_env(cmd)
  names = ("j0", "j1", "j2")  # find_joints stub -> ids [0, 1, 2]
  out = obs.motion_joint_pos_multi_future_select(env, "motion", names)
  expected = cmd.live_joint_pos_future[..., [0, 1, 2]].reshape(env.num_envs, -1)
  assert out.shape == (env.num_envs, 4 * 3)
  assert torch.allclose(out, expected)
