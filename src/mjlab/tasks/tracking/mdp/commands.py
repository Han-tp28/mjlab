from __future__ import annotations

import copy
import math
import pickle
import re
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import torch

from mjlab.managers import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  quat_apply,
  quat_apply_inverse,
  quat_error_magnitude,
  quat_from_euler_xyz,
  quat_inv,
  quat_mul,
  sample_uniform,
  subtract_frame_transforms,
  yaw_quat,
)
from mjlab.viewer.debug_visualizer import DebugVisualizer

if TYPE_CHECKING:
  from collections.abc import Callable
  from typing import Any

  import viser

  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv

_DESIRED_FRAME_COLORS = ((1.0, 0.5, 0.5), (0.5, 1.0, 0.5), (0.5, 0.5, 1.0))
_MOTION_ARRAY_KEYS = (
  "joint_pos",
  "joint_vel",
  "body_pos_w",
  "body_quat_w",
  "body_lin_vel_w",
  "body_ang_vel_w",
)
_SMPL_JOINT_KEYS = (
  "smpl_joints_local",
  "smpl_joints",
  "soma_joints",
  "joints",
)
_SMPL_POSE_KEYS = ("smpl_pose", "pose_aa")
_SMPL_ROOT_QUAT_KEYS = ("smpl_root_quat_w", "root_quat_w", "root_quat")
_MOTION_PICKLE_SUFFIXES = {".pkl", ".pickle"}
_SHARD_FORMAT_KEY = "mjlab_motion_shard_format"
_SONIC_MOTION_KEYS = ("pose_aa", "fps")


@dataclass(frozen=True)
class _MotionEntry:
  path: Path
  index: int | None = None
  name: str | None = None
  source_file: str | None = None

  @property
  def display_name(self) -> str:
    if self.name is not None:
      return self.name
    if self.index is None:
      return self.path.stem
    return f"{self.path.stem}:{self.index}"

  @property
  def display_path(self) -> str:
    if self.index is None:
      return str(self.path)
    return f"{self.path}#{self.display_name}"

  @property
  def source_path(self) -> str:
    if self.source_file is not None:
      return self.source_file
    return str(self.path)


class MotionLoader:
  def __init__(
    self,
    motion_file: str,
    body_indexes: torch.Tensor,
    device: str = "cpu",
    max_num_motions: int | None = None,
    ordered_loading: bool = False,
    smpl_motion_file: str | None = None,
    smpl_num_joints: int = 24,
    smpl_strict_pairing: bool = False,
    smpl_y_up: bool = False,
    robot_default_joint_pos: np.ndarray | None = None,
    robot_default_body_pos_w: np.ndarray | None = None,
    robot_default_body_quat_w: np.ndarray | None = None,
    robot_default_root_pos_w: np.ndarray | None = None,
    robot_default_root_quat_w: np.ndarray | None = None,
  ) -> None:
    self.source_motion_file = motion_file
    self.source_smpl_motion_file = smpl_motion_file
    self.smpl_motion_path = (
      Path(smpl_motion_file).expanduser() if smpl_motion_file else None
    )
    self.smpl_num_joints = smpl_num_joints
    self.smpl_strict_pairing = smpl_strict_pairing
    self.smpl_y_up = smpl_y_up
    self.robot_default_joint_pos = (
      None
      if robot_default_joint_pos is None
      else np.asarray(robot_default_joint_pos, dtype=np.float32)
    )
    self.robot_default_body_pos_w = (
      None
      if robot_default_body_pos_w is None
      else np.asarray(robot_default_body_pos_w, dtype=np.float32)
    )
    self.robot_default_body_quat_w = (
      None
      if robot_default_body_quat_w is None
      else np.asarray(robot_default_body_quat_w, dtype=np.float32)
    )
    self.robot_default_root_pos_w = (
      None
      if robot_default_root_pos_w is None
      else np.asarray(robot_default_root_pos_w, dtype=np.float32)
    )
    self.robot_default_root_quat_w = (
      None
      if robot_default_root_quat_w is None
      else np.asarray(robot_default_root_quat_w, dtype=np.float32)
    )
    self.all_motion_entries = tuple(self._resolve_motion_entries(motion_file))
    self.all_motion_files = tuple(
      entry.display_path for entry in self.all_motion_entries
    )
    self.num_available_motions = len(self.all_motion_entries)
    self._body_indexes = body_indexes
    self.device = device
    self._loaded_motion_entries: tuple[_MotionEntry, ...] = ()
    self._seen_motion_entries: set[_MotionEntry] = set()
    self._pass_seen_motion_entries: set[_MotionEntry] = set()
    self._ordered_loading = ordered_loading
    # LRU cache of parsed (post-SMPL-pairing) motion arrays. Streaming resamples
    # keep most of the pool as replay; caching lets a resample touch disk only
    # for genuinely new motions instead of re-reading the entire pool.
    self._motion_array_cache: OrderedDict[_MotionEntry, tuple[np.ndarray, ...]] = (
      OrderedDict()
    )
    self._motion_array_cache_capacity = 0
    # Per-source-file failure rates used to bias which seen motions are replayed
    # (Groot-style hard-negative mining / Auto PMCP). Empty => uniform replay.
    self._replay_failure_weights: dict[str, float] = {}
    # Uniform floor added to every replay weight so well-mastered motions are
    # still revisited (anti-forgetting), mirroring Groot's uniform_sampling_rate.
    self._replay_uniform_floor = 0.05
    self.load_motions(max_num_motions=max_num_motions)

  def load_motions(
    self,
    max_num_motions: int | None = None,
    motion_files: list[_MotionEntry] | tuple[_MotionEntry, ...] | None = None,
  ) -> bool:
    if motion_files is None:
      motion_entries = self._select_motion_entries(max_num_motions)
    else:
      motion_entries = list(motion_files)
    motion_names: list[str] = []
    lengths: list[int] = []
    joint_pos_arrays: list[np.ndarray] = []
    joint_vel_arrays: list[np.ndarray] = []
    body_pos_arrays: list[np.ndarray] = []
    body_quat_arrays: list[np.ndarray] = []
    body_lin_vel_arrays: list[np.ndarray] = []
    body_ang_vel_arrays: list[np.ndarray] = []
    smpl_joints_arrays: list[np.ndarray] = []
    smpl_root_quat_arrays: list[np.ndarray] = []
    expected_joint_dim: int | None = None
    expected_body_shape: tuple[int, ...] | None = None
    expected_smpl_shape: tuple[int, ...] | None = None

    for entry, arrays in self._iter_motion_arrays(motion_entries):
      (
        joint_pos,
        joint_vel,
        body_pos,
        body_quat,
        body_lin_vel,
        body_ang_vel,
        smpl_joints,
        smpl_root_quat,
      ) = arrays

      if joint_pos.ndim != 2:
        raise ValueError(
          f"Motion file {entry.display_path} has invalid joint_pos shape "
          f"{joint_pos.shape}"
        )
      if joint_pos.shape[0] < 2:
        raise ValueError(
          f"Motion file {entry.display_path} must contain at least 2 frames"
        )
      if joint_vel.shape != joint_pos.shape:
        raise ValueError(
          f"Motion file {entry.display_path} joint_vel shape {joint_vel.shape} "
          f"does not match joint_pos shape {joint_pos.shape}"
        )
      body_shape = body_pos.shape[1:]
      if smpl_joints.ndim != 3 or smpl_joints.shape[-1] != 3:
        raise ValueError(
          f"Motion file {entry.display_path} has invalid smpl_joints shape "
          f"{smpl_joints.shape}"
        )
      if smpl_root_quat.ndim != 2 or smpl_root_quat.shape[-1] != 4:
        raise ValueError(
          f"Motion file {entry.display_path} has invalid smpl_root_quat shape "
          f"{smpl_root_quat.shape}"
        )
      if smpl_root_quat.shape[0] != smpl_joints.shape[0]:
        raise ValueError(
          f"Motion file {entry.display_path} smpl_root_quat shape "
          f"{smpl_root_quat.shape} does not match smpl_joints shape "
          f"{smpl_joints.shape}"
        )
      if smpl_joints.shape[0] != joint_pos.shape[0]:
        if not self.smpl_strict_pairing:
          raise ValueError(
            f"Motion file {entry.display_path} smpl_joints shape "
            f"{smpl_joints.shape} is not compatible with joint_pos shape "
            f"{joint_pos.shape}"
          )
        trimmed_frames = min(smpl_joints.shape[0], joint_pos.shape[0])
        if trimmed_frames < 2:
          raise ValueError(
            f"Motion file {entry.display_path} has too few paired frames after "
            f"trimming: robot={joint_pos.shape[0]}, smpl={smpl_joints.shape[0]}"
          )
        print(
          f"[WARN] Trimming paired motion {entry.display_path} to "
          f"{trimmed_frames} frames "
          f"(robot={joint_pos.shape[0]}, smpl={smpl_joints.shape[0]})"
        )
        joint_pos = joint_pos[:trimmed_frames]
        joint_vel = joint_vel[:trimmed_frames]
        body_pos = body_pos[:trimmed_frames]
        body_quat = body_quat[:trimmed_frames]
        body_lin_vel = body_lin_vel[:trimmed_frames]
        body_ang_vel = body_ang_vel[:trimmed_frames]
        smpl_joints = smpl_joints[:trimmed_frames]
        smpl_root_quat = smpl_root_quat[:trimmed_frames]
      smpl_shape = smpl_joints.shape[1:]
      for name, array in (
        ("body_quat_w", body_quat),
        ("body_lin_vel_w", body_lin_vel),
        ("body_ang_vel_w", body_ang_vel),
      ):
        if array.shape[0] != joint_pos.shape[0] or array.shape[1] != body_shape[0]:
          raise ValueError(
            f"Motion file {entry.display_path} {name} shape {array.shape} is not "
            f"compatible with body_pos_w shape {body_pos.shape}"
          )

      if expected_joint_dim is None:
        expected_joint_dim = joint_pos.shape[1]
        expected_body_shape = body_shape
        expected_smpl_shape = smpl_shape
      elif (
        joint_pos.shape[1] != expected_joint_dim
        or body_shape != expected_body_shape
        or smpl_shape != expected_smpl_shape
      ):
        raise ValueError(
          f"Motion file {entry.display_path} has incompatible shapes: "
          f"joint_pos={joint_pos.shape}, body_pos_w={body_pos.shape}, "
          f"smpl_joints={smpl_joints.shape}; expected joint dim "
          f"{expected_joint_dim}, body shape {expected_body_shape}, and SMPL "
          f"shape {expected_smpl_shape}"
        )

      motion_names.append(entry.display_name)
      lengths.append(joint_pos.shape[0])
      joint_pos_arrays.append(joint_pos)
      joint_vel_arrays.append(joint_vel)
      body_pos_arrays.append(body_pos)
      body_quat_arrays.append(body_quat)
      body_lin_vel_arrays.append(body_lin_vel)
      body_ang_vel_arrays.append(body_ang_vel)
      smpl_joints_arrays.append(smpl_joints)
      smpl_root_quat_arrays.append(smpl_root_quat)

    self.motion_files = tuple(entry.display_path for entry in motion_entries)
    self.motion_source_files = tuple(entry.source_path for entry in motion_entries)
    self._loaded_motion_entries = tuple(motion_entries)
    self._seen_motion_entries.update(motion_entries)
    self._pass_seen_motion_entries.update(motion_entries)
    self.motion_names = tuple(motion_names)
    self.num_motions = len(motion_entries)

    self.joint_pos = torch.tensor(
      np.concatenate(joint_pos_arrays, axis=0),
      dtype=torch.float32,
      device=self.device,
    )
    self.joint_vel = torch.tensor(
      np.concatenate(joint_vel_arrays, axis=0),
      dtype=torch.float32,
      device=self.device,
    )
    self._body_pos_w = torch.tensor(
      np.concatenate(body_pos_arrays, axis=0),
      dtype=torch.float32,
      device=self.device,
    )
    self._body_quat_w = torch.tensor(
      np.concatenate(body_quat_arrays, axis=0),
      dtype=torch.float32,
      device=self.device,
    )
    self._body_lin_vel_w = torch.tensor(
      np.concatenate(body_lin_vel_arrays, axis=0),
      dtype=torch.float32,
      device=self.device,
    )
    self._body_ang_vel_w = torch.tensor(
      np.concatenate(body_ang_vel_arrays, axis=0),
      dtype=torch.float32,
      device=self.device,
    )
    self.smpl_joints = torch.tensor(
      np.concatenate(smpl_joints_arrays, axis=0),
      dtype=torch.float32,
      device=self.device,
    )
    self.smpl_root_quat_w = torch.tensor(
      np.concatenate(smpl_root_quat_arrays, axis=0),
      dtype=torch.float32,
      device=self.device,
    )
    self.body_pos_w = self._body_pos_w[:, self._body_indexes]
    self.body_quat_w = self._body_quat_w[:, self._body_indexes]
    self.body_lin_vel_w = self._body_lin_vel_w[:, self._body_indexes]
    self.body_ang_vel_w = self._body_ang_vel_w[:, self._body_indexes]
    self.time_step_total = int(self.joint_pos.shape[0])

    motion_lengths = np.asarray(lengths, dtype=np.int64)
    motion_ends = np.cumsum(motion_lengths)
    motion_starts = motion_ends - motion_lengths
    self.motion_lengths = torch.tensor(
      motion_lengths, dtype=torch.long, device=self.device
    )
    self.motion_start_steps = torch.tensor(
      motion_starts, dtype=torch.long, device=self.device
    )
    self.motion_end_steps = torch.tensor(
      motion_ends, dtype=torch.long, device=self.device
    )

    file_word = "file" if self.num_motions == 1 else "files"
    print(
      f"[INFO] Loaded {self.num_motions}/{self.num_available_motions} "
      f"motion {file_word} ({self.time_step_total} frames total) "
      f"from {self.source_motion_file}"
    )
    return True

  @property
  def num_seen_motions(self) -> int:
    return len(self._seen_motion_entries)

  @staticmethod
  def _validate_required_keys(data: np.lib.npyio.NpzFile | dict, path: Path) -> None:
    missing = [key for key in _MOTION_ARRAY_KEYS if key not in data]
    if missing:
      raise KeyError(f"Motion file {path} is missing keys: {missing}")

  @staticmethod
  def _load_pickle_object(path: Path):
    try:
      import joblib
    except ImportError:
      with path.open("rb") as f:
        return pickle.load(f)
    return joblib.load(path)

  @staticmethod
  def _is_mjlab_motion_dict(data: object) -> bool:
    return isinstance(data, dict) and all(key in data for key in _MOTION_ARRAY_KEYS)

  @staticmethod
  def _is_sonic_motion_dict(data: object) -> bool:
    return isinstance(data, dict) and all(key in data for key in _SONIC_MOTION_KEYS)

  @staticmethod
  def _axis_angle_to_quat_wxyz(axis_angle: np.ndarray) -> np.ndarray:
    angle = np.linalg.norm(axis_angle, axis=-1, keepdims=True)
    half_angle = 0.5 * angle
    scale = np.divide(
      np.sin(half_angle),
      angle,
      out=np.full_like(angle, 0.5),
      where=angle > 1e-8,
    )
    quat = np.concatenate([np.cos(half_angle), axis_angle * scale], axis=-1)
    return quat.astype(np.float32)

  @staticmethod
  def _normalize_quat_wxyz(quat: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    return (quat / np.maximum(norm, 1e-8)).astype(np.float32)

  @classmethod
  def _finite_difference(cls, values: np.ndarray, fps: float) -> np.ndarray:
    if values.shape[0] < 2:
      return np.zeros_like(values, dtype=np.float32)
    return np.gradient(values.astype(np.float32), 1.0 / fps, axis=0).astype(np.float32)

  @staticmethod
  def _quat_conjugate_wxyz(quat: np.ndarray) -> np.ndarray:
    result = quat.copy()
    result[..., 1:] *= -1.0
    return result

  @staticmethod
  def _quat_mul_wxyz(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = np.moveaxis(lhs, -1, 0)
    rw, rx, ry, rz = np.moveaxis(rhs, -1, 0)
    return np.stack(
      (
        lw * rw - lx * rx - ly * ry - lz * rz,
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
      ),
      axis=-1,
    ).astype(np.float32)

  @staticmethod
  def _quat_to_axis_angle(quat_wxyz: np.ndarray) -> np.ndarray:
    quat = MotionLoader._normalize_quat_wxyz(quat_wxyz)
    w = np.clip(quat[..., :1], -1.0, 1.0)
    vector = quat[..., 1:]
    sin_half = np.linalg.norm(vector, axis=-1, keepdims=True)
    angle = 2.0 * np.arctan2(sin_half, w)
    scale = np.divide(
      angle,
      sin_half,
      out=np.full_like(angle, 2.0),
      where=sin_half > 1e-8,
    )
    return (vector * scale).astype(np.float32)

  @classmethod
  def _quat_angular_velocity(cls, quat_wxyz: np.ndarray, fps: float) -> np.ndarray:
    if quat_wxyz.shape[0] < 2:
      return np.zeros(quat_wxyz.shape[:-1] + (3,), dtype=np.float32)
    quat = cls._normalize_quat_wxyz(quat_wxyz)
    if quat.shape[0] == 2:
      next_quat = quat[[1, 1]]
      prev_quat = quat[[0, 0]]
    else:
      next_quat = np.concatenate([quat[1:], quat[-1:]], axis=0)
      prev_quat = np.concatenate([quat[:1], quat[:-1]], axis=0)
    delta = cls._quat_mul_wxyz(next_quat, cls._quat_conjugate_wxyz(prev_quat))
    delta = cls._normalize_quat_wxyz(delta)
    delta = np.where(delta[..., :1] < 0.0, -delta, delta)
    return (0.5 * fps * cls._quat_to_axis_angle(delta)).astype(np.float32)

  @staticmethod
  def _get_first_array(data: dict, names: tuple[str, ...]) -> np.ndarray | None:
    for name in names:
      if name in data:
        return np.asarray(data[name], dtype=np.float32)
    return None

  @staticmethod
  def _identity_quat_wxyz(frame_count: int) -> np.ndarray:
    quat = np.zeros((frame_count, 4), dtype=np.float32)
    quat[:, 0] = 1.0
    return quat

  @classmethod
  def _remove_smpl_base_rot_wxyz(cls, quat: np.ndarray) -> np.ndarray:
    base_conj = np.asarray([0.5, -0.5, -0.5, -0.5], dtype=np.float32)
    base_conj = np.broadcast_to(base_conj, quat.shape)
    return cls._normalize_quat_wxyz(cls._quat_mul_wxyz(quat, base_conj))

  @classmethod
  def _smpl_root_ytoz_up_wxyz(cls, quat_y_up: np.ndarray) -> np.ndarray:
    base_rot = cls._axis_angle_to_quat_wxyz(
      np.asarray([[np.pi / 2.0, 0.0, 0.0]], dtype=np.float32)
    )
    base_rot = np.broadcast_to(base_rot, quat_y_up.shape)
    return cls._normalize_quat_wxyz(cls._quat_mul_wxyz(base_rot, quat_y_up))

  @classmethod
  def _smpl_root_quat_from_pose(
    cls, pose: np.ndarray | None, smpl_y_up: bool
  ) -> np.ndarray | None:
    if pose is None:
      return None
    pose = np.asarray(pose, dtype=np.float32)
    if pose.ndim == 2 and pose.shape[-1] >= 3:
      root_axis_angle = pose[..., :3]
    elif pose.ndim >= 3 and pose.shape[-1] == 3:
      root_axis_angle = pose[..., 0, :]
    else:
      raise ValueError(f"Invalid SMPL pose shape {pose.shape}")
    quat = cls._axis_angle_to_quat_wxyz(root_axis_angle.reshape(-1, 3))
    if smpl_y_up:
      quat = cls._smpl_root_ytoz_up_wxyz(quat)
    return cls._remove_smpl_base_rot_wxyz(quat)

  @classmethod
  def _coerce_smpl_root_quat(
    cls,
    smpl_root_quat: np.ndarray | None,
    frame_count: int,
  ) -> np.ndarray:
    if smpl_root_quat is None:
      return cls._identity_quat_wxyz(frame_count)
    quat = np.asarray(smpl_root_quat, dtype=np.float32)
    if quat.ndim == 3:
      quat = quat[:, 0]
    if quat.ndim != 2 or quat.shape[-1] != 4:
      raise ValueError(f"Invalid SMPL root quaternion shape {quat.shape}")
    if quat.shape[0] > frame_count:
      quat = quat[:frame_count]
    elif quat.shape[0] < frame_count:
      if quat.shape[0] == 0:
        return cls._identity_quat_wxyz(frame_count)
      pad = np.repeat(quat[-1:], frame_count - quat.shape[0], axis=0)
      quat = np.concatenate([quat, pad], axis=0)
    return cls._normalize_quat_wxyz(quat)

  @classmethod
  def _coerce_smpl_joints(
    cls,
    smpl_joints: np.ndarray | None,
    frame_count: int,
    num_joints: int = 24,
    strict_frame_count: bool = False,
  ) -> np.ndarray:
    if smpl_joints is None:
      return np.zeros((frame_count, num_joints, 3), dtype=np.float32)
    smpl_joints = np.asarray(smpl_joints, dtype=np.float32)
    if smpl_joints.ndim == 2 and smpl_joints.shape[-1] % 3 == 0:
      smpl_joints = smpl_joints.reshape(smpl_joints.shape[0], -1, 3)
    if smpl_joints.ndim != 3 or smpl_joints.shape[-1] != 3:
      raise ValueError(f"Invalid SMPL joints shape {smpl_joints.shape}")
    if smpl_joints.shape[0] == frame_count:
      return smpl_joints.astype(np.float32)
    if strict_frame_count:
      return smpl_joints.astype(np.float32)
    if smpl_joints.shape[0] > frame_count:
      return smpl_joints[:frame_count].astype(np.float32)
    if smpl_joints.shape[0] == 0:
      return np.zeros((frame_count, num_joints, 3), dtype=np.float32)
    pad = np.repeat(smpl_joints[-1:], frame_count - smpl_joints.shape[0], axis=0)
    return np.concatenate([smpl_joints, pad], axis=0).astype(np.float32)

  @classmethod
  def _sonic_motion_to_arrays(
    cls,
    data: dict,
    path: Path,
  ) -> tuple[np.ndarray, ...]:
    fps = float(np.asarray(data.get("fps", 50.0)).reshape(-1)[0])
    pose_aa = np.asarray(data["pose_aa"], dtype=np.float32)
    if pose_aa.ndim != 3 or pose_aa.shape[-1] != 3:
      raise ValueError(f"SONIC motion file {path} has invalid pose_aa {pose_aa.shape}")

    if "dof" in data:
      joint_pos = np.asarray(data["dof"], dtype=np.float32)
    else:
      joint_pos = pose_aa[:, 1:, :].reshape(pose_aa.shape[0], -1)
    joint_vel = cls._finite_difference(joint_pos, fps)

    root_pos = cls._get_first_array(
      data, ("root_trans_offset", "root_trans", "trans", "transl", "smpl_transl")
    )
    if root_pos is None:
      root_pos = np.zeros((pose_aa.shape[0], 3), dtype=np.float32)
    if root_pos.ndim == 3:
      root_pos = root_pos[:, 0]

    smpl_joints = cls._get_first_array(data, _SMPL_JOINT_KEYS)
    body_pos = cls._get_first_array(data, ("body_pos_w", *_SMPL_JOINT_KEYS))
    if body_pos is None:
      body_pos = np.zeros((pose_aa.shape[0], pose_aa.shape[1], 3), dtype=np.float32)
    body_pos = np.asarray(body_pos, dtype=np.float32)
    if body_pos.ndim != 3 or body_pos.shape[-1] != 3:
      raise ValueError(f"SONIC motion file {path} has invalid body positions")
    body_pos = body_pos.copy()
    body_pos[:, 0, :] = root_pos

    if "body_quat_w" in data:
      body_quat = np.asarray(data["body_quat_w"], dtype=np.float32)
    else:
      body_quat = cls._axis_angle_to_quat_wxyz(pose_aa[:, : body_pos.shape[1], :])
      root_rot = cls._get_first_array(data, ("root_rot", "root_quat"))
      if root_rot is not None:
        if root_rot.ndim == 3:
          root_rot = root_rot[:, 0]
        body_quat[:, 0, :] = root_rot[:, [3, 0, 1, 2]]
    body_quat = cls._normalize_quat_wxyz(body_quat)

    arrays = (
      joint_pos.astype(np.float32),
      joint_vel,
      body_pos,
      body_quat,
      cls._finite_difference(body_pos, fps),
      cls._quat_angular_velocity(body_quat, fps),
    )
    smpl_root_quat = cls._smpl_root_quat_from_pose(pose_aa, smpl_y_up=False)
    smpl_root_quat = cls._coerce_smpl_root_quat(smpl_root_quat, joint_pos.shape[0])
    if smpl_joints is None:
      return (
        *arrays,
        cls._coerce_smpl_joints(None, joint_pos.shape[0]),
        smpl_root_quat,
      )
    return (
      *arrays,
      cls._coerce_smpl_joints(smpl_joints, joint_pos.shape[0]),
      smpl_root_quat,
    )

  @classmethod
  def _pickle_motion_to_arrays(
    cls,
    data: dict,
    path: Path,
  ) -> tuple[np.ndarray, ...]:
    if cls._is_mjlab_motion_dict(data):
      cls._validate_required_keys(data, path)
      arrays = tuple(
        np.asarray(data[key], dtype=np.float32) for key in _MOTION_ARRAY_KEYS
      )
      smpl_joints = cls._get_first_array(data, _SMPL_JOINT_KEYS)
      smpl_root_quat = cls._get_first_array(data, _SMPL_ROOT_QUAT_KEYS)
      if smpl_root_quat is None:
        smpl_root_quat = cls._smpl_root_quat_from_pose(
          cls._get_first_array(data, _SMPL_POSE_KEYS), smpl_y_up=False
        )
      if smpl_joints is None and smpl_root_quat is None:
        return arrays
      return (
        *arrays,
        cls._coerce_smpl_joints(smpl_joints, arrays[0].shape[0]),
        cls._coerce_smpl_root_quat(smpl_root_quat, arrays[0].shape[0]),
      )
    if cls._is_sonic_motion_dict(data):
      return cls._sonic_motion_to_arrays(data, path)
    keys = sorted(str(key) for key in data.keys())
    raise KeyError(
      f"Pickle motion file {path} is not a supported mjlab or SONIC motion dict. "
      f"Found keys: {keys}"
    )

  def _smpl_only_defaults_available(self) -> bool:
    return (
      self.robot_default_joint_pos is not None
      and self.robot_default_body_pos_w is not None
      and self.robot_default_body_quat_w is not None
    )

  @staticmethod
  def _nested_smpl_dict(data: object, entry: _MotionEntry) -> dict[Any, Any] | None:
    if not isinstance(data, dict):
      return None
    data_dict = cast(dict[object, object], data)
    nested = None
    if entry.name is not None and entry.name in data_dict:
      nested = data_dict[entry.name]
    if isinstance(nested, dict):
      return nested
    return data_dict

  @classmethod
  def _smpl_only_frame_count(cls, data: object, entry: _MotionEntry) -> int | None:
    data_dict = cls._nested_smpl_dict(data, entry)
    if data_dict is None:
      return None
    for keys in (_SMPL_JOINT_KEYS, _SMPL_POSE_KEYS, _SMPL_ROOT_QUAT_KEYS):
      array = cls._get_first_array(data_dict, keys)
      if array is not None and array.shape[0] > 0:
        return int(array.shape[0])
    return None

  def _smpl_only_motion_to_arrays(
    self,
    data: object,
    path: Path,
    entry: _MotionEntry,
  ) -> tuple[np.ndarray, ...] | None:
    if not self._smpl_only_defaults_available():
      return None
    frame_count = type(self)._smpl_only_frame_count(data, entry)
    if frame_count is None:
      return None

    smpl_joints, smpl_root_quat = type(self)._smpl_pair_from_object(
      data,
      entry,
      frame_count,
      self.smpl_num_joints,
      strict_frame_count=False,
      smpl_y_up=self.smpl_y_up,
    )
    if smpl_joints is None:
      return None
    if smpl_root_quat is None:
      smpl_root_quat = type(self)._identity_quat_wxyz(smpl_joints.shape[0])

    assert self.robot_default_joint_pos is not None
    assert self.robot_default_body_pos_w is not None
    assert self.robot_default_body_quat_w is not None
    assert self.robot_default_root_pos_w is not None
    assert self.robot_default_root_quat_w is not None
    frame_count = smpl_joints.shape[0]
    joint_pos = np.repeat(self.robot_default_joint_pos[None, :], frame_count, axis=0)
    joint_vel = np.zeros_like(joint_pos)
    body_pos_template = self.robot_default_body_pos_w.copy()
    body_quat_template = self.robot_default_body_quat_w.copy()
    root_shift = self.robot_default_root_pos_w - body_pos_template[0]
    body_pos_template = body_pos_template + root_shift[None, :]
    body_pos_template[0] = self.robot_default_root_pos_w
    body_quat_template[0] = self.robot_default_root_quat_w
    body_pos = np.repeat(body_pos_template[None, :, :], frame_count, axis=0)
    body_quat = np.repeat(body_quat_template[None, :, :], frame_count, axis=0)
    body_lin_vel = np.zeros_like(body_pos)
    body_ang_vel = np.zeros_like(body_pos)
    return (
      joint_pos.astype(np.float32),
      joint_vel.astype(np.float32),
      body_pos.astype(np.float32),
      body_quat.astype(np.float32),
      body_lin_vel.astype(np.float32),
      body_ang_vel.astype(np.float32),
      smpl_joints.astype(np.float32),
      smpl_root_quat.astype(np.float32),
    )

  def _load_pickle_motion_entry(
    self,
    path: Path,
    entry: _MotionEntry,
  ) -> tuple[np.ndarray, ...]:
    cls = type(self)
    data = cls._load_pickle_object(path)
    if cls._is_mjlab_motion_dict(data):
      return cls._pickle_motion_to_arrays(data, path)
    if (
      self._smpl_only_defaults_available()
      and isinstance(data, dict)
      and "dof" not in data
    ):
      smpl_only_arrays = self._smpl_only_motion_to_arrays(data, path, entry)
      if smpl_only_arrays is not None:
        return smpl_only_arrays
    if cls._is_sonic_motion_dict(data):
      return cls._pickle_motion_to_arrays(data, path)
    if (
      isinstance(data, dict)
      and entry.name in data
      and isinstance(data[entry.name], dict)
    ):
      nested = data[entry.name]
      if cls._is_mjlab_motion_dict(nested):
        return cls._pickle_motion_to_arrays(nested, path)
      if (
        self._smpl_only_defaults_available()
        and isinstance(nested, dict)
        and "dof" not in nested
      ):
        smpl_only_arrays = self._smpl_only_motion_to_arrays(data, path, entry)
        if smpl_only_arrays is not None:
          return smpl_only_arrays
      if cls._is_sonic_motion_dict(nested):
        return cls._pickle_motion_to_arrays(nested, path)
    raise KeyError(f"Pickle motion file {path} does not contain motion {entry.name!r}")

  @staticmethod
  def _slice_sharded_array(
    array: np.ndarray,
    starts: np.ndarray,
    lengths: np.ndarray,
    index: int,
  ) -> np.ndarray:
    start = int(starts[index])
    end = start + int(lengths[index])
    return np.asarray(array[start:end], dtype=np.float32)

  def _entry_name_candidates(self, entry: _MotionEntry) -> tuple[str, ...]:
    raw_names = [entry.display_name, Path(entry.source_path).stem, entry.path.stem]
    candidates: list[str] = []
    for raw_name in raw_names:
      name = raw_name.split(":", maxsplit=1)[0]
      for candidate in (name, re.sub(r"^\d+__", "", name)):
        if candidate and candidate not in candidates:
          candidates.append(candidate)
    return tuple(candidates)

  def _find_external_smpl_file(self, entry: _MotionEntry) -> Path | None:
    if self.smpl_motion_path is None or not self.smpl_motion_path.exists():
      return None
    if self.smpl_motion_path.is_file():
      return self.smpl_motion_path
    for name in self._entry_name_candidates(entry):
      for suffix in (*_MOTION_PICKLE_SUFFIXES, ".npz"):
        direct = self.smpl_motion_path / f"{name}{suffix}"
        if direct.is_file():
          return direct
        try:
          nested = next(self.smpl_motion_path.rglob(f"{name}{suffix}"))
        except StopIteration:
          nested = None
        if nested is not None and nested.is_file():
          return nested
    return None

  @classmethod
  def _smpl_pair_from_object(
    cls,
    data: object,
    entry: _MotionEntry,
    frame_count: int,
    num_joints: int,
    strict_frame_count: bool = False,
    smpl_y_up: bool = False,
  ) -> tuple[np.ndarray | None, np.ndarray | None]:
    if not isinstance(data, dict):
      return None, None
    data_dict: dict[Any, Any] = data
    nested = data_dict.get(entry.name) if entry.name is not None else None  # type: ignore[arg-type]
    if isinstance(nested, dict):
      data_dict = nested

    joints_raw = cls._get_first_array(data_dict, _SMPL_JOINT_KEYS)
    root_quat_raw = cls._get_first_array(data_dict, _SMPL_ROOT_QUAT_KEYS)
    if root_quat_raw is None:
      root_quat_raw = cls._smpl_root_quat_from_pose(
        cls._get_first_array(data_dict, _SMPL_POSE_KEYS), smpl_y_up=smpl_y_up
      )

    joints = None
    if joints_raw is not None:
      joints = cls._coerce_smpl_joints(
        joints_raw,
        frame_count,
        num_joints,
        strict_frame_count=strict_frame_count,
      )
    root_quat = None
    if root_quat_raw is not None:
      root_frame_count = joints.shape[0] if joints is not None else frame_count
      root_quat = cls._coerce_smpl_root_quat(root_quat_raw, root_frame_count)
    return joints, root_quat

  def _load_external_smpl_pair(
    self, entry: _MotionEntry, frame_count: int
  ) -> tuple[np.ndarray | None, np.ndarray | None]:
    smpl_file = self._find_external_smpl_file(entry)
    if smpl_file is None:
      if self.smpl_strict_pairing:
        candidates = ", ".join(self._entry_name_candidates(entry))
        raise FileNotFoundError(
          f"No paired SMPL motion found for {entry.display_path}. "
          f"Looked for candidates [{candidates}] under "
          f"{self.source_smpl_motion_file!r}."
        )
      return None, None
    if smpl_file.suffix.lower() in _MOTION_PICKLE_SUFFIXES:
      data = self._load_pickle_object(smpl_file)
      smpl_joints, smpl_root_quat = self._smpl_pair_from_object(
        data,
        entry,
        frame_count,
        self.smpl_num_joints,
        strict_frame_count=self.smpl_strict_pairing,
        smpl_y_up=self.smpl_y_up,
      )
      if smpl_joints is None and self.smpl_strict_pairing:
        raise KeyError(f"SMPL motion file {smpl_file} has no supported joints key")
      return smpl_joints, smpl_root_quat
    with np.load(smpl_file) as data:
      joints = self._get_first_array(data, _SMPL_JOINT_KEYS)
      root_quat = self._get_first_array(data, _SMPL_ROOT_QUAT_KEYS)
      if root_quat is None:
        root_quat = self._smpl_root_quat_from_pose(
          self._get_first_array(data, _SMPL_POSE_KEYS), smpl_y_up=self.smpl_y_up
        )
      if joints is None:
        if self.smpl_strict_pairing:
          raise KeyError(f"SMPL motion file {smpl_file} has no supported joints key")
        return None, None
      smpl_joints = self._coerce_smpl_joints(
        joints,
        frame_count,
        self.smpl_num_joints,
        strict_frame_count=self.smpl_strict_pairing,
      )
      return (
        smpl_joints,
        self._coerce_smpl_root_quat(root_quat, smpl_joints.shape[0]),
      )

  def _attach_external_or_zero_smpl(
    self, entry: _MotionEntry, arrays: tuple[np.ndarray, ...]
  ) -> tuple[np.ndarray, ...]:
    if len(arrays) == len(_MOTION_ARRAY_KEYS) + 2:
      return arrays
    frame_count = arrays[0].shape[0]
    external_joints = None
    external_root_quat = None
    if self.smpl_motion_path is not None or len(arrays) == len(_MOTION_ARRAY_KEYS):
      external_joints, external_root_quat = self._load_external_smpl_pair(
        entry, frame_count
      )

    if len(arrays) == len(_MOTION_ARRAY_KEYS) + 1:
      smpl_joints = arrays[-1]
      base_arrays = arrays[:-1]
    else:
      smpl_joints = external_joints
      base_arrays = arrays

    if smpl_joints is None:
      smpl_joints = np.zeros((frame_count, self.smpl_num_joints, 3), dtype=np.float32)
    if external_root_quat is None:
      external_root_quat = self._identity_quat_wxyz(smpl_joints.shape[0])
    return (*base_arrays, smpl_joints, external_root_quat)

  def _iter_motion_arrays(
    self, motion_entries: list[_MotionEntry]
  ) -> list[tuple[_MotionEntry, tuple[np.ndarray, ...]]]:
    cls = type(self)
    cache = self._motion_array_cache
    # Only read from disk for motions not already parsed (cache misses). Replay
    # motions kept across a streaming resample are served straight from memory.
    misses = [entry for entry in motion_entries if entry not in cache]
    by_path: dict[Path, list[_MotionEntry]] = {}
    for entry in misses:
      by_path.setdefault(entry.path, []).append(entry)

    loaded: dict[_MotionEntry, tuple[np.ndarray, ...]] = {}
    for path, entries in by_path.items():
      if path.suffix.lower() in _MOTION_PICKLE_SUFFIXES:
        for entry in entries:
          loaded[entry] = self._load_pickle_motion_entry(path, entry)
      else:
        with np.load(path) as data:
          cls._validate_required_keys(data, path)
          arrays = {
            key: np.asarray(data[key], dtype=np.float32) for key in _MOTION_ARRAY_KEYS
          }
          if _SHARD_FORMAT_KEY in data:
            lengths = np.asarray(data["motion_lengths"], dtype=np.int64)
            starts = np.concatenate(([0], np.cumsum(lengths)[:-1]))
            for entry in entries:
              if entry.index is None:
                raise ValueError(f"Shard entry {entry.display_path} is missing index")
              motion_arrays = tuple(
                cls._slice_sharded_array(arrays[key], starts, lengths, entry.index)
                for key in _MOTION_ARRAY_KEYS
              )
              smpl_joints = None
              for key in _SMPL_JOINT_KEYS:
                if key in data:
                  smpl_joints = cls._slice_sharded_array(
                    np.asarray(data[key], dtype=np.float32),
                    starts,
                    lengths,
                    entry.index,
                  )
                  break
              if smpl_joints is not None:
                smpl_joints = cls._coerce_smpl_joints(
                  smpl_joints, motion_arrays[0].shape[0], self.smpl_num_joints
                )
                loaded[entry] = (*motion_arrays, smpl_joints)
              else:
                loaded[entry] = self._attach_external_or_zero_smpl(entry, motion_arrays)
          else:
            if entries[0].index is not None:
              raise ValueError(f"Non-sharded motion file used as shard: {path}")
            motion_arrays = tuple(arrays[key] for key in _MOTION_ARRAY_KEYS)
            smpl_joints = None
            for key in _SMPL_JOINT_KEYS:
              if key in data:
                smpl_joints = np.asarray(data[key], dtype=np.float32)
                break
            if smpl_joints is not None:
              smpl_joints = cls._coerce_smpl_joints(
                smpl_joints, motion_arrays[0].shape[0], self.smpl_num_joints
              )
              loaded[entries[0]] = (*motion_arrays, smpl_joints)
            else:
              loaded[entries[0]] = self._attach_external_or_zero_smpl(
                entries[0], motion_arrays
              )

    # Attach SMPL pairing (reads the paired PKL/NPZ) for fresh motions only,
    # then cache the final arrays so subsequent resamples can reuse them.
    for entry in misses:
      cache[entry] = self._attach_external_or_zero_smpl(entry, loaded[entry])

    # Mark every requested motion as most-recently-used, then evict the oldest
    # beyond capacity. Capacity tracks the largest pool seen (x2) so a full pool
    # plus an incoming new batch always survive eviction.
    self._motion_array_cache_capacity = max(
      self._motion_array_cache_capacity, 2 * len(motion_entries)
    )
    for entry in motion_entries:
      cache.move_to_end(entry)
    while len(cache) > self._motion_array_cache_capacity:
      cache.popitem(last=False)

    return [(entry, cache[entry]) for entry in motion_entries]

  def _sample_unseen_entries(
    self,
    count: int,
    excluded_entries: set[_MotionEntry],
  ) -> list[_MotionEntry]:
    if count <= 0:
      return []

    candidates = [
      entry
      for entry in self.all_motion_entries
      if entry not in excluded_entries and entry not in self._pass_seen_motion_entries
    ]
    if len(candidates) < count:
      # Start a new pass over the dataset while preserving the active replay pool.
      self._pass_seen_motion_entries = set(excluded_entries)
      candidates = [
        entry for entry in self.all_motion_entries if entry not in excluded_entries
      ]
    if not candidates:
      return []

    count = min(count, len(candidates))
    if self._ordered_loading:
      return candidates[:count]
    sampled = torch.randperm(len(candidates))[:count].tolist()
    return [candidates[i] for i in sampled]

  def set_replay_failure_weights(self, weights: dict[str, float]) -> None:
    """Set per-source-file failure rates that bias replay sampling.

    Keys are motion source paths (``_MotionEntry.source_path``); values are
    failure rates in ``[0, 1]``. Higher-failing motions are replayed more often
    so the policy keeps practising (and stops forgetting) the hard ones. Pass an
    empty dict to fall back to uniform replay.
    """
    self._replay_failure_weights = dict(weights)

  def _sample_replay_entries(
    self,
    count: int,
    excluded_entries: set[_MotionEntry],
  ) -> list[_MotionEntry]:
    if count <= 0:
      return []
    candidates = [
      entry
      for entry in self.all_motion_entries
      if entry in self._seen_motion_entries and entry not in excluded_entries
    ]
    if not candidates:
      return []

    count = min(count, len(candidates))
    if not self._replay_failure_weights:
      sampled = torch.randperm(len(candidates))[:count].tolist()
      return [candidates[i] for i in sampled]

    # Failure-weighted replay: weight each candidate by its failure rate plus a
    # uniform floor, then sample without replacement so the replay set stays
    # distinct. Hard motions get revisited more; mastered ones still get a turn.
    weights = torch.tensor(
      [
        self._replay_failure_weights.get(entry.source_path, 0.0) for entry in candidates
      ],
      dtype=torch.float,
    )
    weights = weights + self._replay_uniform_floor
    sampled = torch.multinomial(weights, count, replacement=False).tolist()
    return [candidates[i] for i in sampled]

  def grow_motions(
    self,
    num_new_motions: int,
    max_num_motions: int | None,
  ) -> bool:
    """Append new motion files to the active pool without dropping old files."""
    if num_new_motions <= 0:
      raise ValueError("`num_new_motions` must be positive.")
    if self.num_motions >= self.num_available_motions:
      return False
    if max_num_motions is not None and self.num_motions >= max_num_motions:
      return False

    remaining_slots = (
      num_new_motions
      if max_num_motions is None
      else min(num_new_motions, max_num_motions - self.num_motions)
    )
    if remaining_slots <= 0:
      return False

    loaded = set(self._loaded_motion_entries)
    new_motion_files = self._sample_unseen_entries(remaining_slots, loaded)
    if not new_motion_files:
      return False
    return self.load_motions(
      motion_files=[*self._loaded_motion_entries, *new_motion_files]
    )

  def resample_motions(
    self,
    num_motions: int,
    replacement: bool = True,
    unique_until_all_seen: bool = False,
  ) -> bool:
    """Replace the active pool with a fresh sample from the full library."""
    if num_motions <= 0:
      raise ValueError("`num_motions` must be positive.")
    if not replacement and num_motions > self.num_available_motions:
      raise ValueError(
        "`num_motions` cannot exceed the available motions when `replacement=False`."
      )

    if unique_until_all_seen and self.num_seen_motions < self.num_available_motions:
      unseen_entries = [
        entry
        for entry in self.all_motion_entries
        if entry not in self._seen_motion_entries
      ]
      num_unseen_to_load = min(num_motions, len(unseen_entries))
      sampled_unseen = torch.randperm(len(unseen_entries))[:num_unseen_to_load].tolist()
      motion_entries = [unseen_entries[index] for index in sampled_unseen]

      num_remaining = num_motions - len(motion_entries)
      if num_remaining > 0:
        probabilities = torch.ones(self.num_available_motions, dtype=torch.float)
        sampled = torch.multinomial(
          probabilities,
          num_samples=num_remaining,
          replacement=True,
        ).tolist()
        motion_entries.extend(self.all_motion_entries[index] for index in sampled)
      return self.load_motions(motion_files=motion_entries)

    probabilities = torch.ones(self.num_available_motions, dtype=torch.float)
    sampled = torch.multinomial(
      probabilities,
      num_samples=num_motions,
      replacement=replacement,
    ).tolist()
    return self.load_motions(
      motion_files=[self.all_motion_entries[index] for index in sampled]
    )

  def stream_motions(
    self,
    num_new_motions: int,
    max_num_motions: int | None,
    replay_fraction: float,
  ) -> bool:
    """Refresh the active pool with new motions while keeping replay motions."""
    if max_num_motions is None or self.num_motions < max_num_motions:
      return self.grow_motions(num_new_motions, max_num_motions)
    if not 0.0 <= replay_fraction < 1.0:
      raise ValueError("`replay_fraction` must be in [0, 1).")

    pool_size = min(max_num_motions, self.num_available_motions)
    min_replay_count = int(round(pool_size * replay_fraction))
    new_count = min(num_new_motions, pool_size - min_replay_count)
    replay_count = pool_size - new_count

    new_entries = self._sample_unseen_entries(new_count, set())
    if not new_entries:
      return False
    replay_entries = self._sample_replay_entries(replay_count, set(new_entries))

    return self.load_motions(motion_files=[*replay_entries, *new_entries])

  def _select_motion_entries(self, max_num_motions: int | None) -> list[_MotionEntry]:
    if max_num_motions is None:
      return list(self.all_motion_entries)
    if max_num_motions <= 0:
      raise ValueError("`max_num_motions` must be positive when provided.")
    if self.num_available_motions <= max_num_motions:
      return list(self.all_motion_entries)

    if self._ordered_loading:
      return list(self.all_motion_entries[:max_num_motions])
    sampled = torch.randperm(self.num_available_motions)[:max_num_motions].tolist()
    return [self.all_motion_entries[i] for i in sampled]

  @staticmethod
  def _motion_files_in_dir(path: Path) -> list[Path]:
    return sorted(
      file
      for file in path.rglob("*")
      if file.suffix.lower() == ".npz" or file.suffix.lower() in _MOTION_PICKLE_SUFFIXES
    )

  @classmethod
  def _expand_motion_file(cls, path: Path) -> list[_MotionEntry]:
    if path.suffix.lower() in _MOTION_PICKLE_SUFFIXES:
      data = cls._load_pickle_object(path)
      if cls._is_mjlab_motion_dict(data) or cls._is_sonic_motion_dict(data):
        return [_MotionEntry(path)]
      if isinstance(data, dict):
        entries = [
          _MotionEntry(path=path, name=str(key))
          for key, value in data.items()
          if cls._is_mjlab_motion_dict(value) or cls._is_sonic_motion_dict(value)
        ]
        if entries:
          return entries
      raise KeyError(f"Pickle motion file {path} does not contain supported motions")

    try:
      data_context = np.load(path)
    except Exception as exc:
      raise ValueError(f"Failed to read motion file {path}: {exc}") from exc
    with data_context as data:
      if _SHARD_FORMAT_KEY not in data:
        return [_MotionEntry(path)]
      if "motion_lengths" not in data:
        raise KeyError(f"Motion shard {path} is missing 'motion_lengths'")
      lengths = np.asarray(data["motion_lengths"], dtype=np.int64)
      if "motion_names" in data:
        names = [str(name) for name in np.asarray(data["motion_names"])]
      else:
        names = [f"{path.stem}:{index}" for index in range(len(lengths))]
      if "source_files" in data:
        source_files = [str(source_file) for source_file in data["source_files"]]
      else:
        source_files = [str(path) for _ in range(len(lengths))]
      if len(names) != len(lengths):
        raise ValueError(
          f"Motion shard {path} has {len(names)} names but {len(lengths)} lengths"
        )
      if len(source_files) != len(lengths):
        raise ValueError(
          f"Motion shard {path} has {len(source_files)} source files but "
          f"{len(lengths)} lengths"
        )
      return [
        _MotionEntry(
          path=path,
          index=index,
          name=names[index],
          source_file=source_files[index],
        )
        for index in range(len(lengths))
      ]

  @classmethod
  def _resolve_motion_entries(cls, motion_file: str) -> list[_MotionEntry]:
    path = Path(motion_file).expanduser()
    is_playlist = path.is_file() and path.suffix.lower() in {".txt", ".lst", ".list"}
    if path.is_dir():
      files = cls._motion_files_in_dir(path)
    elif is_playlist:
      files = []
      base_dir = path.parent
      for raw_line in path.read_text().splitlines():
        line = raw_line.split("#", maxsplit=1)[0].strip()
        if not line:
          continue
        entry = Path(line).expanduser()
        if not entry.is_absolute():
          entry = base_dir / entry
        if entry.is_dir():
          files.extend(cls._motion_files_in_dir(entry))
        else:
          files.append(entry)
    elif path.is_file():
      files = [path]
    else:
      raise FileNotFoundError(f"Motion file, directory, or playlist not found: {path}")

    files = [file.resolve() for file in files]
    missing = [str(file) for file in files if not file.is_file()]
    if missing:
      raise FileNotFoundError(f"Motion playlist contains missing files: {missing}")
    if not files:
      raise FileNotFoundError(f"No .npz or .pkl motion files found in: {path}")

    entries: list[_MotionEntry] = []
    for file in files:
      if is_playlist and file.suffix.lower() == ".npz":
        entries.append(_MotionEntry(file))
      else:
        entries.extend(cls._expand_motion_file(file))
    if not entries:
      raise FileNotFoundError(f"No motions found in: {path}")
    return entries

  def motion_ids_from_steps(self, global_steps: torch.Tensor) -> torch.Tensor:
    steps = torch.clamp(global_steps, 0, self.time_step_total - 1)
    return torch.bucketize(steps, self.motion_end_steps, right=True)

  def clamp_steps_within_motion(
    self, global_steps: torch.Tensor, avoid_last: bool = False
  ) -> tuple[torch.Tensor, torch.Tensor]:
    motion_ids = self.motion_ids_from_steps(global_steps)
    starts = self.motion_start_steps[motion_ids]
    ends = self.motion_end_steps[motion_ids]
    end_limits = ends - (2 if avoid_last else 1)
    end_limits = torch.maximum(end_limits, starts)
    steps = torch.minimum(torch.maximum(global_steps, starts), end_limits)
    return steps, motion_ids

  def sample_global_steps(
    self, count: int, device: str, avoid_last: bool = True
  ) -> tuple[torch.Tensor, torch.Tensor]:
    motion_ids = torch.randint(0, self.num_motions, (count,), device=device)
    lengths = self.motion_lengths[motion_ids]
    sample_lengths = torch.clamp(lengths - (1 if avoid_last else 0), min=1)
    local_steps = (torch.rand(count, device=device) * sample_lengths.float()).long()
    global_steps = self.motion_start_steps[motion_ids] + local_steps
    return global_steps, motion_ids


class MotionCommand(CommandTerm):
  cfg: MotionCommandCfg
  _env: ManagerBasedRlEnv

  def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

    self.robot: Entity = env.scene[cfg.entity_name]
    self.robot_anchor_body_index = self.robot.body_names.index(
      self.cfg.anchor_body_name
    )
    self.motion_anchor_body_index = self.cfg.body_names.index(self.cfg.anchor_body_name)
    self.body_indexes = torch.tensor(
      self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0],
      dtype=torch.long,
      device=self.device,
    )
    self._projection_joint_delta = self._make_projection_joint_delta(
      self.cfg.reference_projection_joint_delta,
      self.cfg.reference_projection_joint_delta_by_pattern,
    )
    self._projection_default_joint_delta = self._make_projection_joint_delta(
      self.cfg.reference_projection_default_joint_delta,
      self.cfg.reference_projection_default_joint_delta_by_pattern,
    )

    self.motion = MotionLoader(
      self.cfg.motion_file,
      self.body_indexes,
      device=self.device,
      max_num_motions=self.cfg.initial_num_load_motions
      or self.cfg.max_num_load_motions,
      ordered_loading=self.cfg.motion_curriculum_ordered_loading,
      smpl_motion_file=self.cfg.smpl_motion_file,
      smpl_num_joints=self.cfg.smpl_num_joints,
      smpl_strict_pairing=self.cfg.smpl_strict_pairing,
      smpl_y_up=self.cfg.smpl_y_up,
      robot_default_joint_pos=self.robot.data.default_joint_pos[0]
      .detach()
      .cpu()
      .numpy(),
      robot_default_body_pos_w=self.robot.data.body_link_pos_w[0]
      .detach()
      .cpu()
      .numpy(),
      robot_default_body_quat_w=self.robot.data.body_link_quat_w[0]
      .detach()
      .cpu()
      .numpy(),
      robot_default_root_pos_w=self.robot.data.default_root_state[0, 0:3]
      .detach()
      .cpu()
      .numpy(),
      robot_default_root_quat_w=self.robot.data.default_root_state[0, 3:7]
      .detach()
      .cpu()
      .numpy(),
    )
    self.smpl_num_future_frames = self.cfg.smpl_num_future_frames
    self.encoder_names = tuple(self.cfg.encoder_sample_probs.keys())
    self.encoder_sample_probs = torch.tensor(
      tuple(self.cfg.encoder_sample_probs.values()),
      dtype=torch.float32,
      device=self.device,
    )
    self.encoder_sample_probs = self.encoder_sample_probs / torch.clamp(
      self.encoder_sample_probs.sum(), min=1e-6
    )
    self.encoder_curriculum_initial_probs = torch.tensor(
      tuple(
        self.cfg.encoder_curriculum_initial_probs.get(name, 0.0)
        for name in self.encoder_names
      ),
      dtype=torch.float32,
      device=self.device,
    )
    initial_sum = self.encoder_curriculum_initial_probs.sum()
    if initial_sum <= 0.0:
      self.encoder_curriculum_initial_probs = self.encoder_sample_probs.clone()
    else:
      self.encoder_curriculum_initial_probs = (
        self.encoder_curriculum_initial_probs / initial_sum
      )
    self.encoder_index = torch.zeros(
      (self.num_envs, len(self.encoder_names)), dtype=torch.float32, device=self.device
    )
    self.compliance = torch.zeros(
      self.num_envs, 3, dtype=torch.float32, device=self.device
    )
    self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self._resample_encoder_index(torch.arange(self.num_envs, device=self.device))
    self.motion_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self.body_pos_relative_w = torch.zeros(
      self.num_envs, len(cfg.body_names), 3, device=self.device
    )
    self.body_quat_relative_w = torch.zeros(
      self.num_envs, len(cfg.body_names), 4, device=self.device
    )
    self.body_quat_relative_w[:, :, 0] = 1.0

    self._transition_source_joint_pos = torch.zeros(
      self.num_envs, self.robot.num_joints, device=self.device
    )
    self._transition_source_joint_vel = torch.zeros(
      self.num_envs, self.robot.num_joints, device=self.device
    )
    self._transition_source_body_pos_w = torch.zeros(
      self.num_envs, len(cfg.body_names), 3, device=self.device
    )
    self._transition_source_body_quat_w = torch.zeros(
      self.num_envs, len(cfg.body_names), 4, device=self.device
    )
    self._transition_source_body_quat_w[..., 0] = 1.0
    self._transition_source_body_lin_vel_w = torch.zeros(
      self.num_envs, len(cfg.body_names), 3, device=self.device
    )
    self._transition_source_body_ang_vel_w = torch.zeros(
      self.num_envs, len(cfg.body_names), 3, device=self.device
    )
    self._transition_step = torch.full(
      (self.num_envs,),
      self.transition_duration_steps,
      dtype=torch.long,
      device=self.device,
    )
    self._transition_initialized = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )

    self._reset_adaptive_sampling_state()

    self.metrics["error_anchor_pos"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_anchor_rot"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_anchor_lin_vel"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["error_anchor_ang_vel"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["error_body_pos"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_body_rot"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_joint_vel"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["reference_projection_anchor_xy"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["reference_projection_joint"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["transition_alpha"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["is_transition"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["sampling_entropy"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["sampling_top1_prob"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["sampling_top1_bin"] = torch.zeros(self.num_envs, device=self.device)

    self._ghost_model = None
    self._ghost_color = np.array(cfg.viz.ghost_color, dtype=np.float32)

    # Live teleop override. When ``live_teleop_active`` is True, the teleop
    # observation terms read these externally-supplied targets instead of the
    # replayed motion (see ``set_live_teleop`` and ``mdp.observations``). Default
    # off, so training and ordinary playback are unaffected.
    self.live_teleop_active = False
    self.live_3point_pos_local = torch.zeros(self.num_envs, 3, 3, device=self.device)
    self.live_3point_orn_quat = torch.zeros(self.num_envs, 3, 4, device=self.device)
    self.live_3point_orn_quat[..., 0] = 1.0
    self.live_joint_pos_future = torch.zeros(
      self.num_envs,
      self.smpl_num_future_frames,
      self.robot.num_joints,
      device=self.device,
    )
    # Identity rotation as a flattened 6D (first two columns of the rotation
    # matrix), matching ``motion_anchor_ori_b``.
    identity_6d = torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 0.0], device=self.device)
    self.live_anchor_ori_b = identity_6d.expand(self.num_envs, 6).clone()
    # Future reference-root positions in the robot-anchor frame, matching
    # ``anchor_pos_b_multi_future``. This is the translation command that lets
    # the teleop/smpl encoders drive the robot to follow the operator's root
    # (walk/jump). Default zero = hold in place; the deploy bridge fills this
    # from the operator root tracker.
    self.live_anchor_pos_b_multi_future = torch.zeros(
      self.num_envs, self.cfg.num_future_frames * 3, device=self.device
    )

  def set_live_teleop(
    self,
    pos_local: torch.Tensor,
    orn_local_quat: torch.Tensor,
    joint_pos: torch.Tensor,
    root_yaw: torch.Tensor | None = None,
    anchor_pos_b_multi_future: torch.Tensor | None = None,
  ) -> None:
    """Inject externally-supplied teleop targets and freeze motion playback.

    Targets are expressed in the desired-root (e.g. headset-yaw) frame, matching
    the convention of the ``teleop_3point_local_*`` observations.

    Args:
      pos_local: ``(num_envs, 3, 3)`` positions of
        ``(left_wrist, right_wrist, head)`` in the desired-root frame.
      orn_local_quat: ``(num_envs, 3, 4)`` wxyz orientations, same body order.
      joint_pos: ``(num_envs, num_joints)`` full joint target vector; only the
        leg joints are read by the teleop observation (others are ignored).
      root_yaw: ``(num_envs,)`` desired world-frame root yaw. ``None`` keeps the
        robot's current heading (no turning command).
      anchor_pos_b_multi_future: ``(num_envs, num_future_frames * 3)`` future
        operator-root positions in the robot-anchor frame (the translation
        command). ``None`` keeps the previous value (default zero = hold).
    """
    self.live_3point_pos_local[:] = pos_local
    self.live_3point_orn_quat[:] = orn_local_quat
    self.live_joint_pos_future[:] = joint_pos[:, None, :]
    if anchor_pos_b_multi_future is not None:
      self.live_anchor_pos_b_multi_future[:] = anchor_pos_b_multi_future
    if root_yaw is None:
      desired_quat = yaw_quat(self.robot_anchor_quat_w)
    else:
      zeros = torch.zeros_like(root_yaw)
      desired_quat = quat_from_euler_xyz(zeros, zeros, root_yaw)
    _, ori = subtract_frame_transforms(
      self.robot_anchor_pos_w,
      self.robot_anchor_quat_w,
      self.robot_anchor_pos_w,
      desired_quat,
    )
    self.live_anchor_ori_b = matrix_from_quat(ori)[..., :2].reshape(self.num_envs, -1)
    self.live_teleop_active = True

  def _make_projection_joint_delta(
    self,
    default_delta: float,
    delta_by_pattern: dict[str, float],
  ) -> torch.Tensor:
    delta = torch.full(
      (self.robot.num_joints,),
      default_delta,
      dtype=torch.float32,
      device=self.device,
    )
    for pattern, value in delta_by_pattern.items():
      joint_ids = self.robot.find_joints(pattern)[0]
      if joint_ids:
        delta[torch.tensor(joint_ids, dtype=torch.long, device=self.device)] = value
    return delta

  @property
  def _project_reference(self) -> bool:
    return self.cfg.reference_projection_enabled

  def _expand_robot_tensor_for_reference(
    self, tensor: torch.Tensor, raw_reference: torch.Tensor
  ) -> torch.Tensor:
    if raw_reference.dim() == 3:
      return tensor[:, None, :].expand(-1, raw_reference.shape[1], -1)
    return tensor

  def _project_joint_pos(self, raw_joint_pos: torch.Tensor) -> torch.Tensor:
    if not self._project_reference:
      return raw_joint_pos

    current_joint_pos = self._expand_robot_tensor_for_reference(
      self.robot_joint_pos, raw_joint_pos
    )
    default_joint_pos = self._expand_robot_tensor_for_reference(
      self.robot.data.default_joint_pos, raw_joint_pos
    )
    step_delta = self._projection_joint_delta
    default_delta = self._projection_default_joint_delta
    if raw_joint_pos.dim() == 3:
      step_delta = step_delta[None, None, :]
      default_delta = default_delta[None, None, :]
    else:
      step_delta = step_delta[None, :]
      default_delta = default_delta[None, :]

    projected = torch.clamp(
      raw_joint_pos,
      current_joint_pos - step_delta,
      current_joint_pos + step_delta,
    )
    projected = torch.clamp(
      projected,
      default_joint_pos - default_delta,
      default_joint_pos + default_delta,
    )
    return projected

  def _project_anchor_pos_w(self, raw_anchor_pos_w: torch.Tensor) -> torch.Tensor:
    if not self._project_reference:
      return raw_anchor_pos_w

    robot_anchor_pos_w = self._expand_robot_tensor_for_reference(
      self.robot_anchor_pos_w, raw_anchor_pos_w
    )
    robot_anchor_quat_w = self._expand_robot_tensor_for_reference(
      self.robot_anchor_quat_w, raw_anchor_pos_w
    )
    robot_anchor_yaw_w = yaw_quat(robot_anchor_quat_w.reshape(-1, 4)).reshape(
      robot_anchor_quat_w.shape
    )
    delta_b = quat_apply_inverse(
      robot_anchor_yaw_w.reshape(-1, 4),
      (raw_anchor_pos_w - robot_anchor_pos_w).reshape(-1, 3),
    ).reshape(raw_anchor_pos_w.shape)
    delta_b = delta_b.clone()
    delta_b[..., 0] = torch.clamp(
      delta_b[..., 0],
      min=-self.cfg.reference_projection_max_backward,
      max=self.cfg.reference_projection_max_forward,
    )
    delta_b[..., 1] = torch.clamp(
      delta_b[..., 1],
      min=-self.cfg.reference_projection_max_lateral,
      max=self.cfg.reference_projection_max_lateral,
    )
    delta_b[..., 2] = torch.clamp(
      delta_b[..., 2],
      min=-self.cfg.reference_projection_max_z_down,
      max=self.cfg.reference_projection_max_z_up,
    )
    projected = robot_anchor_pos_w + quat_apply(
      robot_anchor_yaw_w.reshape(-1, 4), delta_b.reshape(-1, 3)
    ).reshape(raw_anchor_pos_w.shape)

    default_root_z = (
      self.robot.data.default_root_state[:, 2] + self._env.scene.env_origins[:, 2]
    )
    default_root_z = self._expand_robot_tensor_for_reference(
      default_root_z[:, None], projected
    ).squeeze(-1)
    min_z = default_root_z - self.cfg.reference_projection_max_squat_depth
    absolute_min_z = (
      self._env.scene.env_origins[:, 2]
      + self.cfg.reference_projection_min_anchor_height
    )
    absolute_min_z = self._expand_robot_tensor_for_reference(
      absolute_min_z[:, None], projected
    ).squeeze(-1)
    min_z = torch.maximum(min_z, absolute_min_z)
    projected[..., 2] = torch.maximum(projected[..., 2], min_z)
    return projected

  def _project_anchor_quat_w(self, raw_anchor_quat_w: torch.Tensor) -> torch.Tensor:
    if not self._project_reference or not self.cfg.reference_projection_yaw_only_anchor:
      return raw_anchor_quat_w
    return yaw_quat(raw_anchor_quat_w.reshape(-1, 4)).reshape(raw_anchor_quat_w.shape)

  def _project_body_pos_w(self, raw_body_pos_w: torch.Tensor) -> torch.Tensor:
    if not self._project_reference:
      return raw_body_pos_w

    raw_anchor_pos_w = raw_body_pos_w[..., self.motion_anchor_body_index, :]
    projected_anchor_pos_w = self._project_anchor_pos_w(raw_anchor_pos_w)
    anchor_shift = projected_anchor_pos_w - raw_anchor_pos_w
    shifted_body_pos_w = raw_body_pos_w + anchor_shift[..., None, :]
    robot_body_pos_w = self.robot_body_pos_w
    if raw_body_pos_w.dim() == 4:
      robot_body_pos_w = robot_body_pos_w[:, None, :, :]
    body_delta = torch.clamp(
      shifted_body_pos_w - robot_body_pos_w,
      min=-self.cfg.reference_projection_max_body_delta,
      max=self.cfg.reference_projection_max_body_delta,
    )
    return robot_body_pos_w + body_delta

  @property
  def transition_duration_steps(self) -> int:
    if not self.cfg.transition_enabled:
      return 0
    return max(round(self.cfg.transition_duration_s / self._env.step_dt), 1)

  @property
  def is_transition(self) -> torch.Tensor:
    if not self.cfg.transition_enabled or self.transition_duration_steps == 0:
      return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    return self._transition_step < self.transition_duration_steps

  @property
  def transition_alpha(self) -> torch.Tensor:
    duration = self.transition_duration_steps
    if not self.cfg.transition_enabled or duration == 0:
      return torch.ones(self.num_envs, device=self.device)
    return torch.clamp(self._transition_step.float() / float(duration), 0.0, 1.0)

  @property
  def transition_observation(self) -> torch.Tensor:
    return torch.stack([self.is_transition.float(), self.transition_alpha], dim=-1)

  def _capture_transition_source(self, env_ids: torch.Tensor) -> None:
    self._transition_source_joint_pos[env_ids] = self.joint_pos[env_ids].detach()
    self._transition_source_joint_vel[env_ids] = self.joint_vel[env_ids].detach()
    self._transition_source_body_pos_w[env_ids] = self.body_pos_w[env_ids].detach()
    self._transition_source_body_quat_w[env_ids] = self.body_quat_w[env_ids].detach()
    self._transition_source_body_lin_vel_w[env_ids] = self.body_lin_vel_w[
      env_ids
    ].detach()
    self._transition_source_body_ang_vel_w[env_ids] = self.body_ang_vel_w[
      env_ids
    ].detach()

  def _transition_blend(
    self, target: torch.Tensor, source: torch.Tensor
  ) -> torch.Tensor:
    if not self.cfg.transition_enabled or self.transition_duration_steps == 0:
      return target
    if target.dim() == source.dim() + 1:
      source = source[:, None].expand(-1, target.shape[1], *source.shape[1:])
    alpha = self.transition_alpha
    while alpha.dim() < target.dim():
      alpha = alpha.unsqueeze(-1)
    return source * (1.0 - alpha) + target * alpha

  def _transition_blend_quat(
    self, target: torch.Tensor, source: torch.Tensor
  ) -> torch.Tensor:
    if not self.cfg.transition_enabled or self.transition_duration_steps == 0:
      return target
    if target.dim() == source.dim() + 1:
      source = source[:, None].expand(-1, target.shape[1], *source.shape[1:])
    dot = torch.sum(source * target, dim=-1, keepdim=True)
    source = torch.where(dot < 0.0, -source, source)
    blended = self._transition_blend(target, source)
    return blended / torch.clamp(torch.norm(blended, dim=-1, keepdim=True), min=1e-6)

  def _reset_adaptive_sampling_state(self) -> None:
    self.bin_count = int(self.motion.time_step_total // (1 / self._env.step_dt)) + 1
    self.bin_failed_count = torch.zeros(
      self.bin_count, dtype=torch.float, device=self.device
    )
    self._current_bin_failed = torch.zeros(
      self.bin_count, dtype=torch.float, device=self.device
    )
    self.kernel = torch.tensor(
      [self.cfg.adaptive_lambda**i for i in range(self.cfg.adaptive_kernel_size)],
      device=self.device,
    )
    self.kernel = self.kernel / self.kernel.sum()

  def reload_motion_library(self) -> bool:
    if self.cfg.num_new_motions_per_resample is None:
      return False
    if self.cfg.motion_pool_mode == "grow" and (
      self.motion.num_available_motions <= self.motion.num_motions
    ):
      return False

    if self.cfg.motion_pool_mode == "resample":
      target_num_motions = (
        self.cfg.max_num_load_motions
        or self.cfg.initial_num_load_motions
        or self.motion.num_motions
      )
      did_reload = self.motion.resample_motions(
        num_motions=target_num_motions,
        replacement=self.cfg.motion_resample_replacement,
        unique_until_all_seen=self.cfg.motion_resample_unique_until_all_seen,
      )
    elif self.cfg.motion_pool_mode == "streaming":
      did_reload = self.motion.stream_motions(
        num_new_motions=self.cfg.num_new_motions_per_resample,
        max_num_motions=self.cfg.max_num_load_motions,
        replay_fraction=self.cfg.motion_replay_fraction,
      )
    else:
      did_reload = self.motion.grow_motions(
        num_new_motions=self.cfg.num_new_motions_per_resample,
        max_num_motions=self.cfg.max_num_load_motions,
      )
    if not did_reload:
      return False
    self.time_steps = torch.zeros_like(self.time_steps)
    self.motion_ids = torch.zeros_like(self.motion_ids)
    self._transition_step = torch.full_like(
      self._transition_step, self.transition_duration_steps
    )
    self._transition_initialized = torch.zeros_like(self._transition_initialized)
    self._reset_adaptive_sampling_state()
    return True

  @property
  def command(self) -> torch.Tensor:
    parts = [self.joint_pos, self.joint_vel]
    if self.cfg.num_future_frames > 0:
      parts.append(self._future_command())
    if self.cfg.transition_append_to_command:
      parts.append(self.transition_observation)
    return torch.cat(parts, dim=1)

  def _future_steps(self) -> torch.Tensor:
    frame_stride = max(round(self.cfg.dt_future_ref_frames / self._env.step_dt), 1)
    offsets = (
      torch.arange(
        1,
        self.cfg.num_future_frames + 1,
        dtype=torch.long,
        device=self.device,
      )
      * frame_stride
    )
    steps = self.time_steps[:, None] + offsets[None, :]
    motion_ends = self.motion.motion_end_steps[self.motion_ids][:, None]
    motion_starts = self.motion.motion_start_steps[self.motion_ids][:, None]
    steps = torch.minimum(torch.maximum(steps, motion_starts), motion_ends - 1)
    if (
      self._project_reference
      and self.cfg.reference_projection_future_frames is not None
      and self.cfg.reference_projection_future_frames < self.cfg.num_future_frames
    ):
      keep_frames = max(self.cfg.reference_projection_future_frames, 0)
      if keep_frames == 0:
        steps = self.time_steps[:, None].expand_as(steps)
      else:
        steps = steps.clone()
        steps[:, keep_frames:] = steps[:, keep_frames - 1 : keep_frames]
    return steps

  def _smpl_future_steps(self) -> torch.Tensor:
    if self.smpl_num_future_frames <= 0:
      return self.time_steps[:, None]
    frame_stride = max(round(self.cfg.smpl_dt_future_ref_frames / self._env.step_dt), 1)
    offsets = (
      torch.arange(
        self.smpl_num_future_frames,
        dtype=torch.long,
        device=self.device,
      )
      * frame_stride
    )
    steps = self.time_steps[:, None] + offsets[None, :]
    motion_ends = self.motion.motion_end_steps[self.motion_ids][:, None]
    motion_starts = self.motion.motion_start_steps[self.motion_ids][:, None]
    return torch.minimum(torch.maximum(steps, motion_starts), motion_ends - 1)

  def _future_reference(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    steps = self._future_steps()
    future_joint_pos = self._transition_blend(
      self._project_joint_pos(self._raw_joint_pos(steps)),
      self._transition_source_joint_pos,
    )

    future_anchor_pos_w = self._transition_blend(
      self._project_anchor_pos_w(self._raw_anchor_pos_w(steps)),
      self._transition_source_body_pos_w[:, self.motion_anchor_body_index],
    )
    future_anchor_quat_w = self._transition_blend_quat(
      self._project_anchor_quat_w(self._raw_anchor_quat_w(steps)),
      self._transition_source_body_quat_w[:, self.motion_anchor_body_index],
    )

    robot_anchor_quat_inv = quat_inv(self.robot_anchor_quat_w)[:, None, :].expand(
      -1, self.cfg.num_future_frames, -1
    )
    future_anchor_pos_b = quat_apply(
      robot_anchor_quat_inv,
      future_anchor_pos_w - self.robot_anchor_pos_w[:, None, :],
    )
    future_anchor_quat_b = quat_mul(robot_anchor_quat_inv, future_anchor_quat_w)
    future_anchor_ori_b = matrix_from_quat(future_anchor_quat_b)[..., :2]
    return future_joint_pos, future_anchor_pos_b, future_anchor_ori_b

  def _future_command(self) -> torch.Tensor:
    future_joint_pos, future_anchor_pos_b, future_anchor_ori_b = (
      self._future_reference()
    )
    return torch.cat(
      [
        future_joint_pos.reshape(self.num_envs, -1),
        future_anchor_pos_b.reshape(self.num_envs, -1),
        future_anchor_ori_b.reshape(self.num_envs, -1),
      ],
      dim=1,
    )

  @property
  def command_multi_future(self) -> torch.Tensor:
    future_joint_pos, future_anchor_pos_b, _ = self._future_reference()
    return torch.cat(
      [
        future_joint_pos.reshape(self.num_envs, -1),
        future_anchor_pos_b.reshape(self.num_envs, -1),
      ],
      dim=-1,
    )

  @property
  def motion_anchor_ori_b_multi_future(self) -> torch.Tensor:
    _, _, future_anchor_ori_b = self._future_reference()
    return future_anchor_ori_b.reshape(self.num_envs, -1)

  @property
  def anchor_pos_b_multi_future(self) -> torch.Tensor:
    """Future reference-root positions in the robot-anchor frame.

    This is the translation command shared by all encoders: ``g1`` already
    receives it inside ``command_multi_future``; exposing it on its own lets the
    ``teleop`` and ``smpl`` encoders see *where* the operator root is heading so
    the policy can follow translation (walk/jump) instead of marching in place.
    """
    _, future_anchor_pos_b, _ = self._future_reference()
    return future_anchor_pos_b.reshape(self.num_envs, -1)

  def _clamp_reference_steps(self, steps: torch.Tensor) -> torch.Tensor:
    return torch.clamp(steps, 0, self.motion.time_step_total - 1)

  def _raw_joint_pos(self, steps: torch.Tensor | None = None) -> torch.Tensor:
    if steps is None:
      steps = self.time_steps
    steps = self._clamp_reference_steps(steps)
    return self.motion.joint_pos[steps]

  def _raw_joint_vel(self, steps: torch.Tensor | None = None) -> torch.Tensor:
    if steps is None:
      steps = self.time_steps
    steps = self._clamp_reference_steps(steps)
    return self.motion.joint_vel[steps]

  def _raw_body_pos_w(self, steps: torch.Tensor | None = None) -> torch.Tensor:
    if steps is None:
      steps = self.time_steps
    steps = self._clamp_reference_steps(steps)
    origins = self._env.scene.env_origins
    if steps.dim() == 2:
      origins = origins[:, None, None, :]
    else:
      origins = origins[:, None, :]
    return self.motion.body_pos_w[steps] + origins

  def _raw_body_quat_w(self, steps: torch.Tensor | None = None) -> torch.Tensor:
    if steps is None:
      steps = self.time_steps
    steps = self._clamp_reference_steps(steps)
    return self.motion.body_quat_w[steps]

  def _raw_body_lin_vel_w(self, steps: torch.Tensor | None = None) -> torch.Tensor:
    if steps is None:
      steps = self.time_steps
    steps = self._clamp_reference_steps(steps)
    return self.motion.body_lin_vel_w[steps]

  def _raw_body_ang_vel_w(self, steps: torch.Tensor | None = None) -> torch.Tensor:
    if steps is None:
      steps = self.time_steps
    steps = self._clamp_reference_steps(steps)
    return self.motion.body_ang_vel_w[steps]

  def _raw_smpl_joints(self, steps: torch.Tensor | None = None) -> torch.Tensor:
    if steps is None:
      steps = self.time_steps
    steps = self._clamp_reference_steps(steps)
    return self.motion.smpl_joints[steps]

  def _raw_smpl_root_quat_w(self, steps: torch.Tensor | None = None) -> torch.Tensor:
    if steps is None:
      steps = self.time_steps
    steps = self._clamp_reference_steps(steps)
    return self.motion.smpl_root_quat_w[steps]

  def _raw_anchor_pos_w(self, steps: torch.Tensor | None = None) -> torch.Tensor:
    return self._raw_body_pos_w(steps)[..., self.motion_anchor_body_index, :]

  def _raw_anchor_quat_w(self, steps: torch.Tensor | None = None) -> torch.Tensor:
    return self._raw_body_quat_w(steps)[..., self.motion_anchor_body_index, :]

  def _raw_anchor_lin_vel_w(self, steps: torch.Tensor | None = None) -> torch.Tensor:
    return self._raw_body_lin_vel_w(steps)[..., self.motion_anchor_body_index, :]

  def _raw_anchor_ang_vel_w(self, steps: torch.Tensor | None = None) -> torch.Tensor:
    return self._raw_body_ang_vel_w(steps)[..., self.motion_anchor_body_index, :]

  @property
  def joint_pos(self) -> torch.Tensor:
    return self._transition_blend(
      self._project_joint_pos(self._raw_joint_pos()),
      self._transition_source_joint_pos,
    )

  @property
  def joint_vel(self) -> torch.Tensor:
    joint_vel = self._raw_joint_vel()
    if self._project_reference:
      joint_vel = joint_vel * self.cfg.reference_projection_velocity_scale
    return self._transition_blend(joint_vel, self._transition_source_joint_vel)

  @property
  def body_pos_w(self) -> torch.Tensor:
    return self._transition_blend(
      self._project_body_pos_w(self._raw_body_pos_w()),
      self._transition_source_body_pos_w,
    )

  @property
  def body_quat_w(self) -> torch.Tensor:
    body_quat = self._raw_body_quat_w()
    if self._project_reference:
      body_quat = body_quat.clone()
      body_quat[:, self.motion_anchor_body_index] = self.anchor_quat_w
    return self._transition_blend_quat(body_quat, self._transition_source_body_quat_w)

  @property
  def body_lin_vel_w(self) -> torch.Tensor:
    body_lin_vel = self._raw_body_lin_vel_w()
    if self._project_reference:
      body_lin_vel = body_lin_vel * self.cfg.reference_projection_velocity_scale
    return self._transition_blend(body_lin_vel, self._transition_source_body_lin_vel_w)

  @property
  def body_ang_vel_w(self) -> torch.Tensor:
    body_ang_vel = self._raw_body_ang_vel_w()
    if self._project_reference:
      body_ang_vel = body_ang_vel * self.cfg.reference_projection_velocity_scale
    return self._transition_blend(body_ang_vel, self._transition_source_body_ang_vel_w)

  @property
  def smpl_joints(self) -> torch.Tensor:
    return self._raw_smpl_joints()

  @property
  def smpl_joints_multi_future(self) -> torch.Tensor:
    return self._raw_smpl_joints(self._smpl_future_steps())

  @property
  def joint_pos_multi_future_for_smpl(self) -> torch.Tensor:
    return self._raw_joint_pos(self._smpl_future_steps())

  @property
  def smpl_root_quat_w(self) -> torch.Tensor:
    return self._raw_smpl_root_quat_w()

  @property
  def smpl_root_quat_w_multi_future(self) -> torch.Tensor:
    return self._raw_smpl_root_quat_w(self._smpl_future_steps())

  @property
  def smpl_root_quat_w_dif_l_multi_future(self) -> torch.Tensor:
    robot_anchor_quat_inv = quat_inv(self.robot_anchor_quat_w)[:, None, :].expand(
      -1, self.smpl_num_future_frames, -1
    )
    root_rot_dif = quat_mul(robot_anchor_quat_inv, self.smpl_root_quat_w_multi_future)
    mat = matrix_from_quat(root_rot_dif)
    return mat[..., :2].reshape(self.num_envs, -1)

  @property
  def anchor_pos_w(self) -> torch.Tensor:
    return self._transition_blend(
      self._project_anchor_pos_w(self._raw_anchor_pos_w()),
      self._transition_source_body_pos_w[:, self.motion_anchor_body_index],
    )

  @property
  def anchor_quat_w(self) -> torch.Tensor:
    return self._transition_blend_quat(
      self._project_anchor_quat_w(self._raw_anchor_quat_w()),
      self._transition_source_body_quat_w[:, self.motion_anchor_body_index],
    )

  @property
  def anchor_lin_vel_w(self) -> torch.Tensor:
    anchor_lin_vel = self._raw_anchor_lin_vel_w()
    if self._project_reference:
      anchor_lin_vel = anchor_lin_vel * self.cfg.reference_projection_velocity_scale
    return self._transition_blend(
      anchor_lin_vel,
      self._transition_source_body_lin_vel_w[:, self.motion_anchor_body_index],
    )

  @property
  def anchor_ang_vel_w(self) -> torch.Tensor:
    anchor_ang_vel = self._raw_anchor_ang_vel_w()
    if self._project_reference:
      anchor_ang_vel = anchor_ang_vel * self.cfg.reference_projection_velocity_scale
    return self._transition_blend(
      anchor_ang_vel,
      self._transition_source_body_ang_vel_w[:, self.motion_anchor_body_index],
    )

  @property
  def robot_joint_pos(self) -> torch.Tensor:
    return self.robot.data.joint_pos

  @property
  def robot_joint_vel(self) -> torch.Tensor:
    return self.robot.data.joint_vel

  @property
  def robot_body_pos_w(self) -> torch.Tensor:
    return self.robot.data.body_link_pos_w[:, self.body_indexes]

  @property
  def robot_body_quat_w(self) -> torch.Tensor:
    return self.robot.data.body_link_quat_w[:, self.body_indexes]

  @property
  def robot_body_lin_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_lin_vel_w[:, self.body_indexes]

  @property
  def robot_body_ang_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_ang_vel_w[:, self.body_indexes]

  @property
  def robot_anchor_pos_w(self) -> torch.Tensor:
    return self.robot.data.body_link_pos_w[:, self.robot_anchor_body_index]

  @property
  def robot_anchor_quat_w(self) -> torch.Tensor:
    return self.robot.data.body_link_quat_w[:, self.robot_anchor_body_index]

  @property
  def robot_anchor_lin_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_lin_vel_w[:, self.robot_anchor_body_index]

  @property
  def robot_anchor_ang_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_ang_vel_w[:, self.robot_anchor_body_index]

  def _update_metrics(self):
    self.metrics["error_anchor_pos"] = torch.norm(
      self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1
    )
    self.metrics["error_anchor_rot"] = quat_error_magnitude(
      self.anchor_quat_w, self.robot_anchor_quat_w
    )
    self.metrics["error_anchor_lin_vel"] = torch.norm(
      self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w, dim=-1
    )
    self.metrics["error_anchor_ang_vel"] = torch.norm(
      self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w, dim=-1
    )

    self.metrics["error_body_pos"] = torch.norm(
      self.body_pos_relative_w - self.robot_body_pos_w, dim=-1
    ).mean(dim=-1)
    self.metrics["error_body_rot"] = quat_error_magnitude(
      self.body_quat_relative_w, self.robot_body_quat_w
    ).mean(dim=-1)

    self.metrics["error_body_lin_vel"] = torch.norm(
      self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1
    ).mean(dim=-1)
    self.metrics["error_body_ang_vel"] = torch.norm(
      self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1
    ).mean(dim=-1)

    self.metrics["error_joint_pos"] = torch.norm(
      self.joint_pos - self.robot_joint_pos, dim=-1
    )
    self.metrics["error_joint_vel"] = torch.norm(
      self.joint_vel - self.robot_joint_vel, dim=-1
    )
    self.metrics["reference_projection_anchor_xy"] = torch.norm(
      self._raw_anchor_pos_w()[:, :2] - self.anchor_pos_w[:, :2], dim=-1
    )
    self.metrics["reference_projection_joint"] = torch.norm(
      self._raw_joint_pos() - self.joint_pos, dim=-1
    )
    self.metrics["transition_alpha"] = self.transition_alpha
    self.metrics["is_transition"] = self.is_transition.float()

  def _adaptive_sampling(self, env_ids: torch.Tensor):
    episode_failed = self._env.termination_manager.terminated[env_ids]
    if torch.any(episode_failed):
      current_bin_index = torch.clamp(
        (self.time_steps * self.bin_count) // max(self.motion.time_step_total, 1),
        0,
        self.bin_count - 1,
      )
      fail_bins = current_bin_index[env_ids][episode_failed]
      self._current_bin_failed[:] = torch.bincount(fail_bins, minlength=self.bin_count)

    # Sample.
    sampling_probabilities = (
      self.bin_failed_count + self.cfg.adaptive_uniform_ratio / float(self.bin_count)
    )
    sampling_probabilities = torch.nn.functional.pad(
      sampling_probabilities.unsqueeze(0).unsqueeze(0),
      (0, self.cfg.adaptive_kernel_size - 1),  # Non-causal kernel
      mode="replicate",
    )
    sampling_probabilities = torch.nn.functional.conv1d(
      sampling_probabilities, self.kernel.view(1, 1, -1)
    ).view(-1)

    sampling_probabilities = sampling_probabilities / sampling_probabilities.sum()

    sampled_bins = torch.multinomial(
      sampling_probabilities, len(env_ids), replacement=True
    )
    self.time_steps[env_ids] = (
      (sampled_bins + sample_uniform(0.0, 1.0, (len(env_ids),), device=self.device))
      / self.bin_count
      * (self.motion.time_step_total - 1)
    ).long()
    sampled_steps, sampled_motion_ids = self.motion.clamp_steps_within_motion(
      self.time_steps[env_ids], avoid_last=True
    )
    self.time_steps[env_ids] = sampled_steps
    self.motion_ids[env_ids] = sampled_motion_ids

    # Update metrics.
    H = -(sampling_probabilities * (sampling_probabilities + 1e-12).log()).sum()
    H_norm = H / math.log(self.bin_count) if self.bin_count > 1 else 1.0
    pmax, imax = sampling_probabilities.max(dim=0)
    self.metrics["sampling_entropy"][:] = H_norm
    self.metrics["sampling_top1_prob"][:] = pmax
    self.metrics["sampling_top1_bin"][:] = imax.float() / self.bin_count

  def _uniform_sampling(self, env_ids: torch.Tensor):
    sampled_steps, sampled_motion_ids = self.motion.sample_global_steps(
      len(env_ids), device=self.device
    )
    self.time_steps[env_ids] = sampled_steps
    self.motion_ids[env_ids] = sampled_motion_ids
    self.metrics["sampling_entropy"][:] = 1.0  # Maximum entropy for uniform.
    self.metrics["sampling_top1_prob"][:] = 1.0 / self.bin_count
    self.metrics["sampling_top1_bin"][:] = 0.5  # No specific bin preference.

  def _current_encoder_sample_probs(self) -> torch.Tensor:
    warmup_steps = self.cfg.encoder_curriculum_warmup_steps
    ramp_steps = self.cfg.encoder_curriculum_ramp_steps
    if warmup_steps <= 0 and ramp_steps <= 0:
      return self.encoder_sample_probs

    step = float(self._env.common_step_counter)
    if step <= warmup_steps:
      alpha = 0.0
    elif ramp_steps <= 0:
      alpha = 1.0
    else:
      alpha = min(max((step - warmup_steps) / ramp_steps, 0.0), 1.0)

    probs = (1.0 - alpha) * self.encoder_curriculum_initial_probs + alpha * (
      self.encoder_sample_probs
    )
    return probs / torch.clamp(probs.sum(), min=1e-6)

  def _resample_encoder_index(self, env_ids: torch.Tensor) -> None:
    if len(self.encoder_names) == 0:
      return
    if self.cfg.forced_encoder_source is not None:
      encoder_id = self.encoder_names.index(self.cfg.forced_encoder_source)
      sampled = torch.full(
        (len(env_ids),), encoder_id, dtype=torch.long, device=self.device
      )
    else:
      sampled = torch.multinomial(
        self._current_encoder_sample_probs(), len(env_ids), replacement=True
      ).to(self.device)
    self.encoder_index[env_ids] = 0.0
    self.encoder_index[env_ids, sampled] = 1.0

    if "smpl" in self.encoder_names and "g1" in self.encoder_names:
      smpl_idx = self.encoder_names.index("smpl")
      g1_idx = self.encoder_names.index("g1")
      smpl_env_ids = env_ids[sampled == smpl_idx]
      if smpl_env_ids.numel() > 0:
        self.encoder_index[smpl_env_ids, g1_idx] = 1.0
        if (
          "teleop" in self.encoder_names and self.cfg.teleop_sample_prob_when_smpl > 0.0
        ):
          teleop_idx = self.encoder_names.index("teleop")
          use_teleop = (
            torch.rand(smpl_env_ids.numel(), device=self.device)
            < self.cfg.teleop_sample_prob_when_smpl
          )
          self.encoder_index[smpl_env_ids[use_teleop], teleop_idx] = 1.0

    self.compliance[env_ids] = 0.0

  def force_encoder_source(self, encoder_source: str | None) -> None:
    if encoder_source is not None and encoder_source not in self.encoder_names:
      raise ValueError(
        f"Unknown encoder source {encoder_source!r}; expected one of {self.encoder_names}."
      )
    self.cfg.forced_encoder_source = cast(
      Literal["g1", "teleop", "smpl"] | None, encoder_source
    )
    self._resample_encoder_index(torch.arange(self.num_envs, device=self.device))

  def _write_reference_state_to_sim(
    self,
    env_ids: torch.Tensor,
    root_pos: torch.Tensor,
    root_ori: torch.Tensor,
    root_lin_vel: torch.Tensor,
    root_ang_vel: torch.Tensor,
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
  ) -> None:
    """Clip joint positions and write root + joint state to sim."""
    soft_limits = self.robot.data.soft_joint_pos_limits[env_ids]
    joint_pos = torch.clip(joint_pos, soft_limits[:, :, 0], soft_limits[:, :, 1])
    self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

    root_state = torch.cat([root_pos, root_ori, root_lin_vel, root_ang_vel], dim=-1)
    self.robot.write_root_state_to_sim(root_state, env_ids=env_ids)
    self.robot.reset(env_ids=env_ids)

  def _resample_command(
    self,
    env_ids: torch.Tensor,
    reset_sim: bool = True,
    capture_transition_source: bool = True,
  ):
    duration_steps = self.transition_duration_steps
    transition_env_ids = env_ids[
      self._transition_initialized[env_ids] & (self.command_counter[env_ids] > 0)
    ]
    can_transition = (
      self.cfg.transition_enabled
      and duration_steps > 0
      and transition_env_ids.numel() > 0
    )
    if can_transition and capture_transition_source:
      self._capture_transition_source(transition_env_ids)

    if self.cfg.sampling_mode == "start":
      self.time_steps[env_ids] = 0
      self.motion_ids[env_ids] = 0
    elif self.cfg.sampling_mode == "uniform":
      self._uniform_sampling(env_ids)
    else:
      assert self.cfg.sampling_mode == "adaptive"
      self._adaptive_sampling(env_ids)

    self._resample_encoder_index(env_ids)

    raw_body_pos_w = self._raw_body_pos_w()
    raw_body_quat_w = self._raw_body_quat_w()
    raw_body_lin_vel_w = self._raw_body_lin_vel_w()
    raw_body_ang_vel_w = self._raw_body_ang_vel_w()
    root_pos = raw_body_pos_w[env_ids, 0].clone()
    root_ori = raw_body_quat_w[env_ids, 0].clone()
    root_lin_vel = raw_body_lin_vel_w[env_ids, 0].clone()
    root_ang_vel = raw_body_ang_vel_w[env_ids, 0].clone()

    range_list = [
      self.cfg.pose_range.get(key, (0.0, 0.0))
      for key in ["x", "y", "z", "roll", "pitch", "yaw"]
    ]
    ranges = torch.tensor(range_list, device=self.device)
    rand_samples = sample_uniform(
      ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device
    )
    root_pos += rand_samples[:, 0:3]
    orientations_delta = quat_from_euler_xyz(
      rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5]
    )
    root_ori = quat_mul(orientations_delta, root_ori)
    range_list = [
      self.cfg.velocity_range.get(key, (0.0, 0.0))
      for key in ["x", "y", "z", "roll", "pitch", "yaw"]
    ]
    ranges = torch.tensor(range_list, device=self.device)
    rand_samples = sample_uniform(
      ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device
    )
    root_lin_vel += rand_samples[:, :3]
    root_ang_vel += rand_samples[:, 3:]

    joint_pos = self._raw_joint_pos()[env_ids].clone()
    joint_vel = self._raw_joint_vel()[env_ids]

    joint_pos += sample_uniform(
      lower=self.cfg.joint_position_range[0],
      upper=self.cfg.joint_position_range[1],
      size=joint_pos.shape,
      device=joint_pos.device,  # type: ignore
    )

    if reset_sim:
      self._write_reference_state_to_sim(
        env_ids,
        root_pos,
        root_ori,
        root_lin_vel,
        root_ang_vel,
        joint_pos,
        joint_vel,
      )

    self._transition_step[env_ids] = duration_steps
    if can_transition:
      self._transition_step[transition_env_ids] = 0
    self._transition_initialized[env_ids] = True

  def update_relative_body_poses(self) -> None:
    """Recompute ``body_pos_relative_w`` and ``body_quat_relative_w``.

    Called after ``reset_to_frame`` so that termination checks that
    compare relative body positions see the correct state.
    """
    anchor_pos_w_repeat = self.anchor_pos_w[:, None, :].repeat(
      1, len(self.cfg.body_names), 1
    )
    anchor_quat_w_repeat = self.anchor_quat_w[:, None, :].repeat(
      1, len(self.cfg.body_names), 1
    )
    robot_anchor_pos_w_repeat = self.robot_anchor_pos_w[:, None, :].repeat(
      1, len(self.cfg.body_names), 1
    )
    robot_anchor_quat_w_repeat = self.robot_anchor_quat_w[:, None, :].repeat(
      1, len(self.cfg.body_names), 1
    )

    delta_pos_w = robot_anchor_pos_w_repeat
    delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
    delta_ori_w = yaw_quat(
      quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat))
    )

    self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
    self.body_pos_relative_w = delta_pos_w + quat_apply(
      delta_ori_w, self.body_pos_w - anchor_pos_w_repeat
    )

  def _update_command(self):
    if self.live_teleop_active:
      # Live teleop drives observations from external targets, so freeze motion
      # playback: never advance time or resample (which would end the background
      # motion and reset/teleport the robot).
      self.update_relative_body_poses()
      return
    motion_end_steps = self.motion.motion_end_steps[self.motion_ids]
    next_time_steps = self.time_steps + 1
    env_ids = torch.where(next_time_steps >= motion_end_steps)[0]

    transition_env_ids = env_ids[
      self._transition_initialized[env_ids] & (self.command_counter[env_ids] > 0)
    ]
    if (
      self.cfg.transition_enabled
      and self.transition_duration_steps > 0
      and transition_env_ids.numel() > 0
    ):
      self._capture_transition_source(transition_env_ids)

    self.time_steps = next_time_steps
    if env_ids.numel() > 0:
      self._resample_command(
        env_ids,
        reset_sim=False,
        capture_transition_source=False,
      )

    if self.cfg.transition_enabled and self.transition_duration_steps > 0:
      self._transition_step = torch.minimum(
        self._transition_step + 1,
        torch.full_like(self._transition_step, self.transition_duration_steps),
      )

    self.update_relative_body_poses()

    if self.cfg.sampling_mode == "adaptive":
      self.bin_failed_count = (
        self.cfg.adaptive_alpha * self._current_bin_failed
        + (1 - self.cfg.adaptive_alpha) * self.bin_failed_count
      )
      self._current_bin_failed.zero_()

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    """Draw ghost robot or frames based on visualization mode."""
    env_indices = visualizer.get_env_indices(self.num_envs)
    if not env_indices:
      return

    if self.cfg.viz.mode == "ghost":
      if self._ghost_model is None:
        # Build a ghost model with only visual geoms visible. Collision geoms (nonzero
        # contype/conaffinity) get alpha=0 so the viewer's alpha filter excludes them.
        self._ghost_model = copy.deepcopy(self._env.sim.mj_model)
        for gi in range(self._ghost_model.ngeom):
          if (
            self._ghost_model.geom_contype[gi] != 0
            or self._ghost_model.geom_conaffinity[gi] != 0
          ):
            self._ghost_model.geom_rgba[gi, 3] = 0
          else:
            self._ghost_model.geom_rgba[gi] = self._ghost_color

      entity: Entity = self._env.scene[self.cfg.entity_name]
      indexing = entity.indexing
      free_joint_q_adr = indexing.free_joint_q_adr.cpu().numpy()
      joint_q_adr = indexing.joint_q_adr.cpu().numpy()

      for batch in env_indices:
        qpos = np.zeros(self._env.sim.mj_model.nq)
        qpos[free_joint_q_adr[0:3]] = self.body_pos_w[batch, 0].cpu().numpy()
        qpos[free_joint_q_adr[2]] += self.cfg.viz.z_offset
        qpos[free_joint_q_adr[3:7]] = self.body_quat_w[batch, 0].cpu().numpy()
        qpos[joint_q_adr] = self.joint_pos[batch].cpu().numpy()

        visualizer.add_ghost_mesh(
          qpos,
          model=self._ghost_model,
          label=f"ghost_{batch}",
        )

    elif self.cfg.viz.mode == "frames":
      for batch in env_indices:
        desired_body_pos = self.body_pos_w[batch].cpu().numpy()
        desired_body_pos[:, 2] += self.cfg.viz.z_offset
        desired_body_quat = self.body_quat_w[batch]
        desired_body_rotm = matrix_from_quat(desired_body_quat).cpu().numpy()

        current_body_pos = self.robot_body_pos_w[batch].cpu().numpy()
        current_body_quat = self.robot_body_quat_w[batch]
        current_body_rotm = matrix_from_quat(current_body_quat).cpu().numpy()

        for i, body_name in enumerate(self.cfg.body_names):
          visualizer.add_frame(
            position=desired_body_pos[i],
            rotation_matrix=desired_body_rotm[i],
            scale=0.08,
            label=f"desired_{body_name}_{batch}",
            axis_colors=_DESIRED_FRAME_COLORS,
          )
          visualizer.add_frame(
            position=current_body_pos[i],
            rotation_matrix=current_body_rotm[i],
            scale=0.12,
            label=f"current_{body_name}_{batch}",
          )

        desired_anchor_pos = self.anchor_pos_w[batch].cpu().numpy()
        desired_anchor_pos[2] += self.cfg.viz.z_offset
        desired_anchor_quat = self.anchor_quat_w[batch]
        desired_rotation_matrix = matrix_from_quat(desired_anchor_quat).cpu().numpy()
        visualizer.add_frame(
          position=desired_anchor_pos,
          rotation_matrix=desired_rotation_matrix,
          scale=0.1,
          label=f"desired_anchor_{batch}",
          axis_colors=_DESIRED_FRAME_COLORS,
        )

        current_anchor_pos = self.robot_anchor_pos_w[batch].cpu().numpy()
        current_anchor_quat = self.robot_anchor_quat_w[batch]
        current_rotation_matrix = matrix_from_quat(current_anchor_quat).cpu().numpy()
        visualizer.add_frame(
          position=current_anchor_pos,
          rotation_matrix=current_rotation_matrix,
          scale=0.15,
          label=f"current_anchor_{batch}",
        )

  def create_gui(
    self,
    name: str,
    server: viser.ViserServer,
    get_env_idx: Callable[[], int],
    on_change: Callable[[], None] | None = None,
    request_action: Callable[[str, Any], None] | None = None,
  ) -> None:
    """Create motion scrubber controls in the Viser viewer."""
    max_frame = int(self.motion.time_step_total) - 1

    with server.gui.add_folder(name.capitalize()):
      scrubber = server.gui.add_slider(
        "Frame",
        min=0,
        max=max_frame,
        step=1,
        initial_value=0,
      )

      @scrubber.on_update
      def _(_) -> None:
        idx = get_env_idx()
        self.time_steps[idx] = int(scrubber.value)
        if on_change is not None:
          on_change()

      all_envs_cb = server.gui.add_checkbox("All envs", initial_value=True)
      start_btn = server.gui.add_button("Start Here")

      @start_btn.on_click
      def _(_) -> None:
        if request_action is not None:
          request_action(
            "CUSTOM",
            {"type": "gui_reset", "all_envs": all_envs_cb.value},
          )

    self._scrubber_handles = (scrubber, all_envs_cb, start_btn)
    self._set_scrubber_disabled(True)

  def _set_scrubber_disabled(self, disabled: bool) -> None:
    """Enable or disable the motion scrubber GUI controls."""
    for handle in self._scrubber_handles:
      handle.disabled = disabled

  def on_viewer_pause(self, paused: bool) -> None:
    if hasattr(self, "_scrubber_handles"):
      self._set_scrubber_disabled(not paused)

  def apply_gui_reset(self, env_ids: torch.Tensor) -> bool:
    if not hasattr(self, "_scrubber_handles"):
      return False
    frame = int(self._scrubber_handles[0].value)
    self.reset_to_frame(env_ids, frame)
    self.update_relative_body_poses()
    return True

  def reset_to_frame(self, env_ids: torch.Tensor, frame: int) -> None:
    """Reset to exact reference state at a specific frame.

    Like ``_resample_command`` but deterministic: no random
    perturbations to pose, velocity, or joint positions.
    """
    requested_frame = torch.full(
      (len(env_ids),), frame, dtype=torch.long, device=self.device
    )
    steps, motion_ids = self.motion.clamp_steps_within_motion(requested_frame)
    self.time_steps[env_ids] = steps
    self.motion_ids[env_ids] = motion_ids
    self._write_reference_state_to_sim(
      env_ids,
      self._raw_body_pos_w()[env_ids, 0],
      self._raw_body_quat_w()[env_ids, 0],
      self._raw_body_lin_vel_w()[env_ids, 0],
      self._raw_body_ang_vel_w()[env_ids, 0],
      self._raw_joint_pos()[env_ids],
      self._raw_joint_vel()[env_ids],
    )
    self._transition_step[env_ids] = self.transition_duration_steps
    self._transition_initialized[env_ids] = True


@dataclass(kw_only=True)
class MotionCommandCfg(CommandTermCfg):
  motion_file: str
  smpl_motion_file: str | None = None
  smpl_num_joints: int = 24
  smpl_strict_pairing: bool = False
  smpl_y_up: bool = False
  smpl_num_future_frames: int = 0
  smpl_dt_future_ref_frames: float = 0.02
  anchor_body_name: str
  body_names: tuple[str, ...]
  entity_name: str
  pose_range: dict[str, tuple[float, float]] = field(default_factory=dict)
  velocity_range: dict[str, tuple[float, float]] = field(default_factory=dict)
  joint_position_range: tuple[float, float] = (-0.52, 0.52)
  adaptive_kernel_size: int = 1
  adaptive_lambda: float = 0.8
  adaptive_uniform_ratio: float = 0.1
  adaptive_alpha: float = 0.001
  sampling_mode: Literal["adaptive", "uniform", "start"] = "adaptive"
  max_num_load_motions: int | None = None
  initial_num_load_motions: int | None = None
  num_new_motions_per_resample: int | None = None
  motion_pool_mode: Literal["grow", "streaming", "resample"] = "grow"
  motion_resample_replacement: bool = True
  motion_resample_unique_until_all_seen: bool = False
  motion_replay_fraction: float = 0.25
  # When True, streaming replay biases toward motions with higher lifetime
  # failure rates (Groot-style hard-negative mining) instead of uniform replay.
  motion_replay_failure_weighted: bool = False
  motion_resample_interval: int | None = None
  motion_resample_start_iteration: int = 0
  motion_curriculum_gate: bool = False
  motion_curriculum_min_stable_iterations: int = 0
  motion_curriculum_force_after_iterations: int | None = None
  motion_curriculum_ordered_loading: bool = False
  motion_curriculum_stage_sizes: tuple[int, ...] = ()
  motion_curriculum_min_mean_episode_length: float | None = None
  motion_curriculum_max_fall_rate: float | None = None
  motion_curriculum_max_body_pos_error: float | None = None
  motion_curriculum_max_body_rot_error: float | None = None
  motion_curriculum_max_anchor_pos_error: float | None = None
  motion_curriculum_max_anchor_rot_error: float | None = None
  motion_curriculum_max_joint_pos_error: float | None = None
  # When empty (the default), the gate's fall_rate counts EVERY non-timeout
  # termination (anchor_xy, ee_body_pos, etc.) as a "fall", which conflates
  # literal instability with recoverable tracking misses already covered by the
  # error_body_pos/error_anchor_pos checks above. Set this to a comma-separated
  # list of literal-fall termination names (e.g.
  # "fell_over_height,fell_over_orientation") so fall_rate only measures actual
  # falls and the gate can pass on a policy that is stable but still imperfectly
  # tracking a hard motion. A plain string (not a tuple) so it takes a single
  # CLI token -- tyro parses a variadic tuple[str, ...] inconsistently through
  # the two-stage train parser (it grabs only the first value).
  motion_curriculum_fall_termination_keys: str = ""
  encoder_sample_probs: dict[str, float] = field(
    default_factory=lambda: {"g1": 1.0, "teleop": 1.0, "smpl": 1.0}
  )
  encoder_curriculum_initial_probs: dict[str, float] = field(
    default_factory=lambda: {"g1": 1.0, "teleop": 0.0, "smpl": 0.0}
  )
  encoder_curriculum_warmup_steps: int = 0
  encoder_curriculum_ramp_steps: int = 0
  teleop_sample_prob_when_smpl: float = 0.5
  forced_encoder_source: Literal["g1", "teleop", "smpl"] | None = None
  num_future_frames: int = 0
  dt_future_ref_frames: float = 0.1
  reference_projection_enabled: bool = False
  reference_projection_max_forward: float = 0.25
  reference_projection_max_backward: float = 0.02
  reference_projection_max_lateral: float = 0.18
  reference_projection_max_z_down: float = 0.12
  reference_projection_max_z_up: float = 0.10
  reference_projection_max_squat_depth: float = 0.18
  reference_projection_min_anchor_height: float = 0.0
  reference_projection_max_body_delta: float = 0.35
  reference_projection_velocity_scale: float = 0.25
  reference_projection_future_frames: int | None = None
  reference_projection_yaw_only_anchor: bool = True
  reference_projection_joint_delta: float = 0.25
  reference_projection_joint_delta_by_pattern: dict[str, float] = field(
    default_factory=dict
  )
  reference_projection_default_joint_delta: float = 10.0
  reference_projection_default_joint_delta_by_pattern: dict[str, float] = field(
    default_factory=dict
  )
  transition_enabled: bool = False
  transition_duration_s: float = 0.4
  transition_append_to_command: bool = False
  transition_soft_reward_enabled: bool = False
  transition_reward_softness: float = 0.6

  @dataclass
  class VizCfg:
    mode: Literal["ghost", "frames"] = "ghost"
    ghost_color: tuple[float, float, float, float] = (0.5, 0.7, 0.5, 0.5)
    z_offset: float = 0.0

  viz: VizCfg = field(default_factory=VizCfg)

  def build(self, env: ManagerBasedRlEnv) -> MotionCommand:
    return MotionCommand(self, env)
