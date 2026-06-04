from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import torch

from mjlab.managers import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  quat_apply,
  quat_error_magnitude,
  quat_from_euler_xyz,
  quat_inv,
  quat_mul,
  sample_uniform,
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
_SHARD_FORMAT_KEY = "mjlab_motion_shard_format"


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
  ) -> None:
    self.source_motion_file = motion_file
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
    expected_joint_dim: int | None = None
    expected_body_shape: tuple[int, ...] | None = None

    for entry, arrays in self._iter_motion_arrays(motion_entries):
      (
        joint_pos,
        joint_vel,
        body_pos,
        body_quat,
        body_lin_vel,
        body_ang_vel,
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
      elif (
        joint_pos.shape[1] != expected_joint_dim or body_shape != expected_body_shape
      ):
        raise ValueError(
          f"Motion file {entry.display_path} has incompatible shapes: "
          f"joint_pos={joint_pos.shape}, body_pos_w={body_pos.shape}; expected "
          f"joint dim {expected_joint_dim} and body shape {expected_body_shape}"
        )

      motion_names.append(entry.display_name)
      lengths.append(joint_pos.shape[0])
      joint_pos_arrays.append(joint_pos)
      joint_vel_arrays.append(joint_vel)
      body_pos_arrays.append(body_pos)
      body_quat_arrays.append(body_quat)
      body_lin_vel_arrays.append(body_lin_vel)
      body_ang_vel_arrays.append(body_ang_vel)

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
  def _validate_required_keys(data: np.lib.npyio.NpzFile, path: Path) -> None:
    missing = [key for key in _MOTION_ARRAY_KEYS if key not in data]
    if missing:
      raise KeyError(f"Motion file {path} is missing keys: {missing}")

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

  @classmethod
  def _iter_motion_arrays(
    cls, motion_entries: list[_MotionEntry]
  ) -> list[tuple[_MotionEntry, tuple[np.ndarray, ...]]]:
    by_path: dict[Path, list[_MotionEntry]] = {}
    for entry in motion_entries:
      by_path.setdefault(entry.path, []).append(entry)

    loaded: dict[_MotionEntry, tuple[np.ndarray, ...]] = {}
    for path, entries in by_path.items():
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
            loaded[entry] = tuple(
              cls._slice_sharded_array(arrays[key], starts, lengths, entry.index)
              for key in _MOTION_ARRAY_KEYS
            )
        else:
          if entries[0].index is not None:
            raise ValueError(f"Non-sharded motion file used as shard: {path}")
          loaded[entries[0]] = tuple(arrays[key] for key in _MOTION_ARRAY_KEYS)

    return [(entry, loaded[entry]) for entry in motion_entries]

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
    sampled = torch.randperm(len(candidates))[:count].tolist()
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
  def _expand_motion_file(path: Path) -> list[_MotionEntry]:
    with np.load(path) as data:
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
    if path.is_dir():
      files = sorted(path.rglob("*.npz"))
    elif path.is_file() and path.suffix.lower() in {".txt", ".lst", ".list"}:
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
          files.extend(sorted(entry.rglob("*.npz")))
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
      raise FileNotFoundError(f"No .npz motion files found in: {path}")

    entries: list[_MotionEntry] = []
    for file in files:
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

    self.motion = MotionLoader(
      self.cfg.motion_file,
      self.body_indexes,
      device=self.device,
      max_num_motions=self.cfg.initial_num_load_motions
      or self.cfg.max_num_load_motions,
      ordered_loading=self.cfg.motion_curriculum_ordered_loading,
    )
    self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self.motion_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self.body_pos_relative_w = torch.zeros(
      self.num_envs, len(cfg.body_names), 3, device=self.device
    )
    self.body_quat_relative_w = torch.zeros(
      self.num_envs, len(cfg.body_names), 4, device=self.device
    )
    self.body_quat_relative_w[:, :, 0] = 1.0

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
    self.metrics["sampling_entropy"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["sampling_top1_prob"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["sampling_top1_bin"] = torch.zeros(self.num_envs, device=self.device)

    self._ghost_model = None
    self._ghost_color = np.array(cfg.viz.ghost_color, dtype=np.float32)

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
    self.time_steps.zero_()
    self.motion_ids.zero_()
    self._reset_adaptive_sampling_state()
    return True

  @property
  def command(self) -> torch.Tensor:
    parts = [self.joint_pos, self.joint_vel]
    if self.cfg.num_future_frames > 0:
      parts.append(self._future_command())
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
    return torch.minimum(torch.maximum(steps, motion_starts), motion_ends - 1)

  def _future_command(self) -> torch.Tensor:
    steps = self._future_steps()
    future_joint_pos = self.motion.joint_pos[steps].reshape(self.num_envs, -1)

    future_anchor_pos_w = (
      self.motion.body_pos_w[steps, self.motion_anchor_body_index]
      + self._env.scene.env_origins[:, None, :]
    )
    future_anchor_quat_w = self.motion.body_quat_w[steps, self.motion_anchor_body_index]

    robot_anchor_quat_inv = quat_inv(self.robot_anchor_quat_w)[:, None, :].expand(
      -1, self.cfg.num_future_frames, -1
    )
    future_anchor_pos_b = quat_apply(
      robot_anchor_quat_inv,
      future_anchor_pos_w - self.robot_anchor_pos_w[:, None, :],
    ).reshape(self.num_envs, -1)
    future_anchor_quat_b = quat_mul(robot_anchor_quat_inv, future_anchor_quat_w)
    future_anchor_ori_b = matrix_from_quat(future_anchor_quat_b)[..., :2].reshape(
      self.num_envs, -1
    )

    return torch.cat(
      [future_joint_pos, future_anchor_pos_b, future_anchor_ori_b], dim=1
    )

  @property
  def joint_pos(self) -> torch.Tensor:
    return self.motion.joint_pos[self.time_steps]

  @property
  def joint_vel(self) -> torch.Tensor:
    return self.motion.joint_vel[self.time_steps]

  @property
  def body_pos_w(self) -> torch.Tensor:
    return (
      self.motion.body_pos_w[self.time_steps] + self._env.scene.env_origins[:, None, :]
    )

  @property
  def body_quat_w(self) -> torch.Tensor:
    return self.motion.body_quat_w[self.time_steps]

  @property
  def body_lin_vel_w(self) -> torch.Tensor:
    return self.motion.body_lin_vel_w[self.time_steps]

  @property
  def body_ang_vel_w(self) -> torch.Tensor:
    return self.motion.body_ang_vel_w[self.time_steps]

  @property
  def anchor_pos_w(self) -> torch.Tensor:
    return (
      self.motion.body_pos_w[self.time_steps, self.motion_anchor_body_index]
      + self._env.scene.env_origins
    )

  @property
  def anchor_quat_w(self) -> torch.Tensor:
    return self.motion.body_quat_w[self.time_steps, self.motion_anchor_body_index]

  @property
  def anchor_lin_vel_w(self) -> torch.Tensor:
    return self.motion.body_lin_vel_w[self.time_steps, self.motion_anchor_body_index]

  @property
  def anchor_ang_vel_w(self) -> torch.Tensor:
    return self.motion.body_ang_vel_w[self.time_steps, self.motion_anchor_body_index]

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

  def _resample_command(self, env_ids: torch.Tensor):
    if self.cfg.sampling_mode == "start":
      self.time_steps[env_ids] = 0
      self.motion_ids[env_ids] = 0
    elif self.cfg.sampling_mode == "uniform":
      self._uniform_sampling(env_ids)
    else:
      assert self.cfg.sampling_mode == "adaptive"
      self._adaptive_sampling(env_ids)

    root_pos = self.body_pos_w[env_ids, 0].clone()
    root_ori = self.body_quat_w[env_ids, 0].clone()
    root_lin_vel = self.body_lin_vel_w[env_ids, 0].clone()
    root_ang_vel = self.body_ang_vel_w[env_ids, 0].clone()

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

    joint_pos = self.joint_pos[env_ids].clone()
    joint_vel = self.joint_vel[env_ids]

    joint_pos += sample_uniform(
      lower=self.cfg.joint_position_range[0],
      upper=self.cfg.joint_position_range[1],
      size=joint_pos.shape,
      device=joint_pos.device,  # type: ignore
    )

    self._write_reference_state_to_sim(
      env_ids,
      root_pos,
      root_ori,
      root_lin_vel,
      root_ang_vel,
      joint_pos,
      joint_vel,
    )

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
    self.time_steps += 1
    motion_end_steps = self.motion.motion_end_steps[self.motion_ids]
    env_ids = torch.where(self.time_steps >= motion_end_steps)[0]
    if env_ids.numel() > 0:
      self._resample_command(env_ids)

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
      self.body_pos_w[env_ids, 0],
      self.body_quat_w[env_ids, 0],
      self.body_lin_vel_w[env_ids, 0],
      self.body_ang_vel_w[env_ids, 0],
      self.joint_pos[env_ids],
      self.joint_vel[env_ids],
    )


@dataclass(kw_only=True)
class MotionCommandCfg(CommandTermCfg):
  motion_file: str
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
  num_future_frames: int = 0
  dt_future_ref_frames: float = 0.1

  @dataclass
  class VizCfg:
    mode: Literal["ghost", "frames"] = "ghost"
    ghost_color: tuple[float, float, float, float] = (0.5, 0.7, 0.5, 0.5)
    z_offset: float = 0.0

  viz: VizCfg = field(default_factory=VizCfg)

  def build(self, env: ManagerBasedRlEnv) -> MotionCommand:
    return MotionCommand(self, env)
