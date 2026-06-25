"""Sources of VR head/hand poses for VR-M3.1 teleoperation.

A ``PoseSource`` yields, each control step, a ``TeleopFrame`` holding the SE3
poses of the headset and the two controllers in the VR runtime's world frame
(OpenXR convention: x=right, y=up, z=back), plus a ``reset_calib`` flag.

Three implementations are provided:

* ``OpenXrPoseSource`` reads a real headset + controllers via ``pyopenxr``. It
  requires VR hardware and an OpenXR runtime and is therefore not exercised by
  the automated tests.
* ``MockPoseSource`` generates a scripted trajectory so the full pipeline can be
  run and tested without any hardware.
* ``ReplayPoseSource`` replays the head/wrist trajectory from a motion ``.npz``,
  useful for sanity-checking retargeting against known data.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import torch

from mjlab.tasks.tracking.teleop.retarget import (
  _VR_TO_ROBOT,
  TELEOP_BODY_NAMES,
)


@dataclass
class TeleopFrame:
  """One frame of VR poses, in the OpenXR world frame (positions in metres).

  Quaternions are wxyz. All tensors are shape ``(3,)`` / ``(4,)``.
  """

  head_pos: torch.Tensor
  head_quat: torch.Tensor
  left_pos: torch.Tensor
  left_quat: torch.Tensor
  right_pos: torch.Tensor
  right_quat: torch.Tensor
  reset_calib: bool = False


class PoseSource(Protocol):
  def poll(self) -> TeleopFrame: ...

  def close(self) -> None: ...


_IDENTITY_QUAT = torch.tensor([1.0, 0.0, 0.0, 0.0])


class MockPoseSource:
  """Scripted source: steady head, hands tracing horizontal circles.

  Useful to verify the end-to-end teleop loop (env + policy + viewer) without
  any VR hardware. Poses are emitted in the OpenXR world frame.
  """

  def __init__(self, dt: float = 0.02, radius: float = 0.12, period: float = 4.0):
    self._dt = dt
    self._radius = radius
    self._period = period
    self._t = 0.0
    # Neutral VR poses (y-up): head at eye height, hands forward of the chest.
    self._head0 = torch.tensor([0.0, 1.60, 0.0])
    self._left0 = torch.tensor([-0.25, 1.25, -0.20])
    self._right0 = torch.tensor([0.25, 1.25, -0.20])

  def poll(self) -> TeleopFrame:
    self._t += self._dt
    phase = 2.0 * np.pi * self._t / self._period
    # Circle in the horizontal (x, z) plane plus a small vertical bob.
    offset = torch.tensor(
      [
        self._radius * np.cos(phase),
        0.05 * np.sin(phase),
        self._radius * np.sin(phase),
      ],
      dtype=torch.float32,
    )
    return TeleopFrame(
      head_pos=self._head0.clone(),
      head_quat=_IDENTITY_QUAT.clone(),
      left_pos=self._left0 + offset,
      left_quat=_IDENTITY_QUAT.clone(),
      right_pos=self._right0 - offset,
      right_quat=_IDENTITY_QUAT.clone(),
      reset_calib=False,
    )

  def close(self) -> None:
    pass


class ReplayPoseSource:
  """Replay head/wrist world poses from a motion ``.npz`` as a VR stream.

  The motion's robot-frame world poses are mapped back into the OpenXR world
  frame so that the retargeter's VR->robot remap round-trips. Body order in the
  npz is assumed to match ``body_names`` (the env's ``POSE_BODY_NAMES``).
  """

  def __init__(self, motion_file: str | Path, body_names: tuple[str, ...]):
    data = np.load(Path(motion_file), allow_pickle=True)
    if "body_pos_w" not in data or "body_quat_w" not in data:
      raise ValueError(
        f"Motion file {motion_file} lacks 'body_pos_w'/'body_quat_w' arrays."
      )
    idx = [body_names.index(name) for name in TELEOP_BODY_NAMES]
    self._pos = torch.from_numpy(data["body_pos_w"][:, idx].astype(np.float32))
    self._quat = torch.from_numpy(data["body_quat_w"][:, idx].astype(np.float32))
    self._inv_remap = _VR_TO_ROBOT.T  # robot -> VR
    self._frame = 0

  def poll(self) -> TeleopFrame:
    t = min(self._frame, self._pos.shape[0] - 1)
    self._frame += 1
    pos_vr = self._pos[t] @ self._inv_remap.T  # (3, 3): [left, right, head]
    # Orientation round-trip back to VR frame.
    mat = _matrix_from_quat_np(self._quat[t])
    mat_vr = self._inv_remap @ mat @ self._inv_remap.T
    quat_vr = _quat_from_matrix_np(mat_vr)
    return TeleopFrame(
      head_pos=pos_vr[2],
      head_quat=quat_vr[2],
      left_pos=pos_vr[0],
      left_quat=quat_vr[0],
      right_pos=pos_vr[1],
      right_quat=quat_vr[1],
      reset_calib=(self._frame == 1),
    )

  def close(self) -> None:
    pass


def _matrix_from_quat_np(quat: torch.Tensor) -> torch.Tensor:
  from mjlab.utils.lab_api.math import matrix_from_quat

  return matrix_from_quat(quat)


def _quat_from_matrix_np(mat: torch.Tensor) -> torch.Tensor:
  from mjlab.utils.lab_api.math import quat_from_matrix

  return quat_from_matrix(mat)


class OpenXrPoseSource:  # pragma: no cover - requires VR hardware
  """Read headset + two controllers from a live OpenXR runtime via ``pyopenxr``.

  Requires VR hardware, a running OpenXR runtime (SteamVR, Monado, ALVR, ...) and
  the optional ``teleop`` dependency (``uv sync --extra teleop``). This path is
  not covered by automated tests.

  The headset pose comes from the ``VIEW`` reference space; controller poses come
  from the grip-pose action on each hand. The right-hand ``select`` button
  requests a recalibration (sets ``reset_calib``). The implementation follows the
  pyopenxr ``track_controllers`` example, driving the frame lifecycle one frame
  per :meth:`poll` via the context's frame-loop generator.
  """

  def __init__(self, reference_space: str = "STAGE"):
    try:
      import xr  # type: ignore
    except ImportError as exc:
      raise ImportError(
        "OpenXR teleop needs 'pyopenxr'. Install with: uv sync --extra teleop"
      ) from exc

    self._xr = xr
    self._ctx = xr.ContextObject(
      instance_create_info=xr.InstanceCreateInfo(
        enabled_extension_names=[xr.KHR_OPENGL_ENABLE_EXTENSION_NAME],
      ),
    )
    self._ctx.__enter__()
    self._closed = False
    instance = self._ctx.instance
    session = self._ctx.session

    hand_paths = [
      xr.string_to_path(instance, "/user/hand/left"),
      xr.string_to_path(instance, "/user/hand/right"),
    ]
    self._pose_action = xr.create_action(
      action_set=self._ctx.default_action_set,
      create_info=xr.ActionCreateInfo(
        action_type=xr.ActionType.POSE_INPUT,
        action_name="hand_pose",
        localized_action_name="Hand Pose",
        count_subaction_paths=len(hand_paths),
        subaction_paths=hand_paths,
      ),
    )
    self._select_action = xr.create_action(
      action_set=self._ctx.default_action_set,
      create_info=xr.ActionCreateInfo(
        action_type=xr.ActionType.BOOLEAN_INPUT,
        action_name="recalibrate",
        localized_action_name="Recalibrate",
        count_subaction_paths=len(hand_paths),
        subaction_paths=hand_paths,
      ),
    )
    bindings = [
      xr.ActionSuggestedBinding(
        action=self._pose_action,
        binding=xr.string_to_path(instance, "/user/hand/left/input/grip/pose"),
      ),
      xr.ActionSuggestedBinding(
        action=self._pose_action,
        binding=xr.string_to_path(instance, "/user/hand/right/input/grip/pose"),
      ),
      xr.ActionSuggestedBinding(
        action=self._select_action,
        binding=xr.string_to_path(instance, "/user/hand/right/input/select/click"),
      ),
    ]
    xr.suggest_interaction_profile_bindings(
      instance=instance,
      suggested_bindings=xr.InteractionProfileSuggestedBinding(
        interaction_profile=xr.string_to_path(
          instance, "/interaction_profiles/khr/simple_controller"
        ),
        count_suggested_bindings=len(bindings),
        suggested_bindings=(xr.ActionSuggestedBinding * len(bindings))(*bindings),
      ),
    )
    self._hand_spaces = [
      xr.create_action_space(
        session=session,
        create_info=xr.ActionSpaceCreateInfo(
          action=self._pose_action, subaction_path=hand_paths[i]
        ),
      )
      for i in range(2)
    ]
    self._view_space = xr.create_reference_space(
      session=session,
      create_info=xr.ReferenceSpaceCreateInfo(
        reference_space_type=xr.ReferenceSpaceType.VIEW,
      ),
    )
    self._frame_iter = self._ctx.frame_loop()
    self._last = TeleopFrame(
      head_pos=torch.zeros(3),
      head_quat=_IDENTITY_QUAT.clone(),
      left_pos=torch.zeros(3),
      left_quat=_IDENTITY_QUAT.clone(),
      right_pos=torch.zeros(3),
      right_quat=_IDENTITY_QUAT.clone(),
    )

  @staticmethod
  def _pose_to_tensors(pose) -> tuple[torch.Tensor, torch.Tensor]:
    p, q = pose.position, pose.orientation
    return (
      torch.tensor([p.x, p.y, p.z], dtype=torch.float32),
      torch.tensor([q.w, q.x, q.y, q.z], dtype=torch.float32),  # xyzw -> wxyz
    )

  def poll(self) -> TeleopFrame:
    import ctypes

    xr = self._xr
    frame_state = next(self._frame_iter)
    if self._ctx.session_state != xr.SessionState.FOCUSED:
      return self._last

    active = xr.ActiveActionSet(
      action_set=self._ctx.default_action_set, subaction_path=xr.NULL_PATH
    )
    xr.sync_actions(
      session=self._ctx.session,
      sync_info=xr.ActionsSyncInfo(
        count_active_action_sets=1, active_action_sets=ctypes.pointer(active)
      ),
    )
    t = frame_state.predicted_display_time
    poses = []
    for space in (*self._hand_spaces, self._view_space):
      loc = xr.locate_space(space=space, base_space=self._ctx.space, time=t)
      poses.append(self._pose_to_tensors(loc.pose))
    (left_p, left_q), (right_p, right_q), (head_p, head_q) = poses

    recalib = xr.get_action_state_boolean(
      session=self._ctx.session,
      get_info=xr.ActionStateGetInfo(action=self._select_action),
    )
    self._last = TeleopFrame(
      head_pos=head_p,
      head_quat=head_q,
      left_pos=left_p,
      left_quat=left_q,
      right_pos=right_p,
      right_quat=right_q,
      reset_calib=bool(recalib.current_state and recalib.changed_since_last_sync),
    )
    return self._last

  def close(self) -> None:
    if not self._closed:
      self._closed = True
      self._ctx.__exit__(None, None, None)
