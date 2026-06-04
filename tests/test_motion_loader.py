from pathlib import Path

import numpy as np
import torch

from mjlab.tasks.tracking.mdp.commands import MotionLoader


def _write_motion(path: Path, frames: int, offset: float) -> None:
  joints = np.arange(frames * 2, dtype=np.float32).reshape(frames, 2) + offset
  bodies = np.arange(frames * 3 * 3, dtype=np.float32).reshape(frames, 3, 3) + offset
  quats = np.zeros((frames, 3, 4), dtype=np.float32)
  quats[..., 0] = 1.0
  np.savez(
    path,
    joint_pos=joints,
    joint_vel=joints * 0.1,
    body_pos_w=bodies,
    body_quat_w=quats,
    body_lin_vel_w=bodies * 0.2,
    body_ang_vel_w=bodies * 0.3,
    fps=np.array([50.0], dtype=np.float32),
  )


def _write_motion_shard(path: Path, motions: list[tuple[str, int, float]]) -> None:
  joint_pos_arrays = []
  joint_vel_arrays = []
  body_pos_arrays = []
  body_quat_arrays = []
  body_lin_vel_arrays = []
  body_ang_vel_arrays = []
  lengths = []
  names = []
  for name, frames, offset in motions:
    joints = np.arange(frames * 2, dtype=np.float32).reshape(frames, 2) + offset
    bodies = np.arange(frames * 3 * 3, dtype=np.float32).reshape(frames, 3, 3) + offset
    quats = np.zeros((frames, 3, 4), dtype=np.float32)
    quats[..., 0] = 1.0
    joint_pos_arrays.append(joints)
    joint_vel_arrays.append(joints * 0.1)
    body_pos_arrays.append(bodies)
    body_quat_arrays.append(quats)
    body_lin_vel_arrays.append(bodies * 0.2)
    body_ang_vel_arrays.append(bodies * 0.3)
    lengths.append(frames)
    names.append(name)

  np.savez(
    path,
    mjlab_motion_shard_format=np.array([1], dtype=np.int32),
    motion_lengths=np.asarray(lengths, dtype=np.int64),
    motion_names=np.asarray(names),
    joint_pos=np.concatenate(joint_pos_arrays, axis=0),
    joint_vel=np.concatenate(joint_vel_arrays, axis=0),
    body_pos_w=np.concatenate(body_pos_arrays, axis=0),
    body_quat_w=np.concatenate(body_quat_arrays, axis=0),
    body_lin_vel_w=np.concatenate(body_lin_vel_arrays, axis=0),
    body_ang_vel_w=np.concatenate(body_ang_vel_arrays, axis=0),
  )


def test_motion_loader_loads_motion_directory(tmp_path: Path) -> None:
  library = tmp_path / "motions"
  library.mkdir()
  _write_motion(library / "a.npz", frames=3, offset=0.0)
  _write_motion(library / "b.npz", frames=4, offset=100.0)

  loader = MotionLoader(str(library), torch.tensor([0, 2]), device="cpu")

  assert loader.num_motions == 2
  assert loader.motion_names == ("a", "b")
  assert loader.time_step_total == 7
  assert loader.motion_start_steps.tolist() == [0, 3]
  assert loader.motion_end_steps.tolist() == [3, 7]
  assert tuple(loader.joint_pos.shape) == (7, 2)
  assert tuple(loader.body_pos_w.shape) == (7, 2, 3)
  assert loader.motion_ids_from_steps(torch.tensor([0, 2, 3, 6])).tolist() == [
    0,
    0,
    1,
    1,
  ]


def test_motion_loader_loads_relative_playlist(tmp_path: Path) -> None:
  _write_motion(tmp_path / "a.npz", frames=3, offset=0.0)
  _write_motion(tmp_path / "b.npz", frames=2, offset=10.0)
  playlist = tmp_path / "playlist.txt"
  playlist.write_text("# comment\n./a.npz\n\n./b.npz  # inline comment\n")

  loader = MotionLoader(str(playlist), torch.tensor([1]), device="cpu")

  assert loader.num_motions == 2
  assert loader.motion_names == ("a", "b")
  assert loader.time_step_total == 5
  assert tuple(loader.body_pos_w.shape) == (5, 1, 3)


def test_motion_loader_loads_lossless_motion_shard(tmp_path: Path) -> None:
  shard = tmp_path / "motion_shard_000000.npz"
  _write_motion_shard(
    shard,
    [
      ("walk", 3, 0.0),
      ("jump", 4, 100.0),
    ],
  )

  loader = MotionLoader(str(shard), torch.tensor([0, 2]), device="cpu")

  assert loader.num_available_motions == 2
  assert loader.num_motions == 2
  assert loader.motion_names == ("walk", "jump")
  assert loader.time_step_total == 7
  assert loader.motion_start_steps.tolist() == [0, 3]
  assert loader.motion_end_steps.tolist() == [3, 7]
  assert tuple(loader.joint_pos.shape) == (7, 2)
  assert tuple(loader.body_pos_w.shape) == (7, 2, 3)
  assert loader.joint_pos[3, 0].item() == 100.0


def test_motion_loader_can_load_motion_subset(tmp_path: Path) -> None:
  library = tmp_path / "motions"
  library.mkdir()
  for i in range(5):
    _write_motion(library / f"{i}.npz", frames=3, offset=float(i * 10))

  loader = MotionLoader(
    str(library),
    torch.tensor([0]),
    device="cpu",
    max_num_motions=2,
  )

  assert loader.num_available_motions == 5
  assert loader.num_motions == 2
  assert len(loader.motion_files) == 2
  assert tuple(loader.body_pos_w.shape) == (6, 1, 3)


def test_motion_loader_grows_motion_subset_without_dropping_old(
  tmp_path: Path,
) -> None:
  library = tmp_path / "motions"
  library.mkdir()
  for i in range(6):
    _write_motion(library / f"{i}.npz", frames=3, offset=float(i * 10))

  loader = MotionLoader(
    str(library),
    torch.tensor([0]),
    device="cpu",
    max_num_motions=2,
  )
  initial_files = set(loader.motion_files)

  assert loader.grow_motions(num_new_motions=2, max_num_motions=4)

  assert loader.num_available_motions == 6
  assert loader.num_motions == 4
  assert initial_files.issubset(set(loader.motion_files))


def test_motion_loader_streams_new_motions_after_pool_is_full(tmp_path: Path) -> None:
  library = tmp_path / "motions"
  library.mkdir()
  for i in range(10):
    _write_motion(library / f"{i}.npz", frames=3, offset=float(i * 10))

  loader = MotionLoader(
    str(library),
    torch.tensor([0]),
    device="cpu",
    max_num_motions=4,
  )
  initial_files = set(loader.motion_files)

  assert loader.stream_motions(
    num_new_motions=2,
    max_num_motions=4,
    replay_fraction=0.25,
  )

  assert loader.num_motions == 4
  assert len(set(loader.motion_files) - initial_files) == 2
  assert len(initial_files & set(loader.motion_files)) == 2
  assert loader.num_seen_motions == 6


def test_motion_loader_streaming_keeps_lifetime_seen_count_across_passes(
  tmp_path: Path,
) -> None:
  library = tmp_path / "motions"
  library.mkdir()
  for i in range(6):
    _write_motion(library / f"{i}.npz", frames=3, offset=float(i * 10))

  loader = MotionLoader(
    str(library),
    torch.tensor([0]),
    device="cpu",
    max_num_motions=4,
  )

  assert loader.stream_motions(
    num_new_motions=2,
    max_num_motions=4,
    replay_fraction=0.25,
  )
  assert loader.num_seen_motions == 6

  assert loader.stream_motions(
    num_new_motions=2,
    max_num_motions=4,
    replay_fraction=0.25,
  )
  assert loader.num_seen_motions == 6


def test_motion_loader_resamples_active_pool_from_full_library(tmp_path: Path) -> None:
  library = tmp_path / "motions"
  library.mkdir()
  for i in range(10):
    _write_motion(library / f"{i}.npz", frames=3, offset=float(i * 10))

  loader = MotionLoader(
    str(library),
    torch.tensor([0]),
    device="cpu",
    max_num_motions=4,
  )
  assert loader.num_motions == 4

  assert loader.resample_motions(num_motions=4, replacement=True)

  assert loader.num_available_motions == 10
  assert loader.num_motions == 4
  assert len(loader.motion_files) == 4
  assert 4 <= loader.num_seen_motions <= 8


def test_motion_loader_resample_can_prioritize_unseen_until_full_pass(
  tmp_path: Path,
) -> None:
  library = tmp_path / "motions"
  library.mkdir()
  for i in range(6):
    _write_motion(library / f"{i}.npz", frames=3, offset=float(i * 10))

  loader = MotionLoader(
    str(library),
    torch.tensor([0]),
    device="cpu",
    max_num_motions=4,
  )
  initial_files = set(loader.motion_files)

  assert loader.resample_motions(
    num_motions=2,
    replacement=True,
    unique_until_all_seen=True,
  )

  assert loader.num_motions == 2
  assert loader.num_seen_motions == 6
  assert initial_files.isdisjoint(set(loader.motion_files))


def test_motion_loader_clamps_steps_inside_clip_boundaries(tmp_path: Path) -> None:
  library = tmp_path / "motions"
  library.mkdir()
  _write_motion(library / "a.npz", frames=3, offset=0.0)
  _write_motion(library / "b.npz", frames=4, offset=100.0)
  loader = MotionLoader(str(library), torch.tensor([0]), device="cpu")

  steps, motion_ids = loader.clamp_steps_within_motion(
    torch.tensor([2, 3, 6]), avoid_last=True
  )

  assert steps.tolist() == [1, 3, 5]
  assert motion_ids.tolist() == [0, 1, 1]
