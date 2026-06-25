import pickle
from pathlib import Path

import numpy as np
import pytest
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


def _motion_dict(frames: int, offset: float) -> dict[str, np.ndarray]:
  joints = np.arange(frames * 2, dtype=np.float32).reshape(frames, 2) + offset
  bodies = np.arange(frames * 3 * 3, dtype=np.float32).reshape(frames, 3, 3) + offset
  quats = np.zeros((frames, 3, 4), dtype=np.float32)
  quats[..., 0] = 1.0
  return {
    "joint_pos": joints,
    "joint_vel": joints * 0.1,
    "body_pos_w": bodies,
    "body_quat_w": quats,
    "body_lin_vel_w": bodies * 0.2,
    "body_ang_vel_w": bodies * 0.3,
  }


def _write_motion_pickle(path: Path, frames: int, offset: float) -> None:
  with path.open("wb") as f:
    pickle.dump(_motion_dict(frames, offset), f)


def _write_sonic_motion_pickle(path: Path, frames: int) -> None:
  pose_aa = np.zeros((frames, 3, 3), dtype=np.float32)
  pose_aa[:, 1, 0] = np.linspace(0.0, 0.2, frames, dtype=np.float32)
  pose_aa[:, 2, 1] = np.linspace(0.0, 0.4, frames, dtype=np.float32)
  dof = np.stack(
    (
      np.linspace(0.0, 0.3, frames, dtype=np.float32),
      np.linspace(1.0, 1.3, frames, dtype=np.float32),
    ),
    axis=-1,
  )
  root_trans = np.zeros((frames, 3), dtype=np.float32)
  root_trans[:, 0] = np.arange(frames, dtype=np.float32)
  smpl_joints = np.zeros((frames, 3, 3), dtype=np.float32)
  smpl_joints[:, 1, 2] = 1.0
  root_rot_xyzw = np.zeros((frames, 4), dtype=np.float32)
  root_rot_xyzw[:, 3] = 1.0
  with path.open("wb") as f:
    pickle.dump(
      {
        "pose_aa": pose_aa,
        "dof": dof,
        "root_trans_offset": root_trans,
        "root_rot": root_rot_xyzw,
        "smpl_joints": smpl_joints,
        "fps": 30,
      },
      f,
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


def test_motion_loader_loads_motion_pickle(tmp_path: Path) -> None:
  motion = tmp_path / "motion.pkl"
  _write_motion_pickle(motion, frames=3, offset=10.0)

  loader = MotionLoader(str(motion), torch.tensor([0, 2]), device="cpu")

  assert loader.num_motions == 1
  assert loader.motion_names == ("motion",)
  assert tuple(loader.joint_pos.shape) == (3, 2)
  assert tuple(loader.body_pos_w.shape) == (3, 2, 3)
  assert loader.joint_pos[0, 0].item() == 10.0


def test_motion_loader_loads_sonic_motion_pickle(tmp_path: Path) -> None:
  motion = tmp_path / "sonic.pkl"
  _write_sonic_motion_pickle(motion, frames=4)

  loader = MotionLoader(str(motion), torch.tensor([0, 1]), device="cpu")

  assert loader.num_motions == 1
  assert loader.motion_names == ("sonic",)
  assert tuple(loader.joint_pos.shape) == (4, 2)
  assert tuple(loader.body_pos_w.shape) == (4, 2, 3)
  assert loader.body_pos_w[:, 0, 0].tolist() == [0.0, 1.0, 2.0, 3.0]
  assert torch.allclose(loader.body_quat_w[:, 0, 0], torch.ones(4))
  assert torch.all(loader.joint_vel[:, 0] > 0.0)


def test_motion_loader_loads_smpl_only_pickle_with_robot_defaults(
  tmp_path: Path,
) -> None:
  motion = tmp_path / "smpl_only.pkl"
  frames = 5
  smpl_joints = np.arange(frames * 24 * 3, dtype=np.float32).reshape(frames, 24, 3)
  pose_aa = np.zeros((frames, 24, 3), dtype=np.float32)
  pose_aa[:, 0, 2] = np.linspace(0.0, 0.4, frames, dtype=np.float32)
  with motion.open("wb") as f:
    pickle.dump({"smpl_joints": smpl_joints, "pose_aa": pose_aa, "fps": 50}, f)

  default_joint_pos = np.array([0.1, -0.2, 0.3], dtype=np.float32)
  default_body_pos = np.arange(4 * 3, dtype=np.float32).reshape(4, 3)
  default_body_quat = np.zeros((4, 4), dtype=np.float32)
  default_body_quat[:, 0] = 1.0

  loader = MotionLoader(
    str(motion),
    torch.tensor([0, 2]),
    device="cpu",
    robot_default_joint_pos=default_joint_pos,
    robot_default_body_pos_w=default_body_pos,
    robot_default_body_quat_w=default_body_quat,
    robot_default_root_pos_w=np.array([0.0, 0.0, 0.8], dtype=np.float32),
    robot_default_root_quat_w=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
  )

  assert loader.num_motions == 1
  assert loader.motion_names == ("smpl_only",)
  assert tuple(loader.joint_pos.shape) == (frames, 3)
  assert tuple(loader.body_pos_w.shape) == (frames, 2, 3)
  assert tuple(loader.smpl_joints.shape) == (frames, 24, 3)
  assert torch.allclose(loader.joint_pos[0], torch.tensor(default_joint_pos))
  assert torch.allclose(loader.body_pos_w[0, 0], torch.tensor([0.0, 0.0, 0.8]))
  assert torch.allclose(loader.body_pos_w[0, 1], torch.tensor([6.0, 6.0, 6.8]))
  assert torch.allclose(loader.smpl_joints, torch.tensor(smpl_joints))
  assert not torch.allclose(loader.smpl_root_quat_w, torch.eye(1, 4).repeat(frames, 1))


def test_motion_loader_pairs_external_smpl_pickle_by_stripped_name(
  tmp_path: Path,
) -> None:
  robot_motion = tmp_path / "230509__ab_bicycle_001__A359_M.npz"
  smpl_dir = tmp_path / "smpl"
  smpl_dir.mkdir()
  _write_motion(robot_motion, frames=3, offset=0.0)
  smpl_joints = np.arange(3 * 24 * 3, dtype=np.float32).reshape(3, 24, 3)
  pose_aa = np.zeros((3, 24, 3), dtype=np.float32)
  pose_aa[:, 0, 2] = np.linspace(0.0, 0.2, 3, dtype=np.float32)
  with (smpl_dir / "ab_bicycle_001__A359_M.pkl").open("wb") as f:
    pickle.dump({"smpl_joints": smpl_joints, "pose_aa": pose_aa, "fps": 50}, f)

  loader = MotionLoader(
    str(robot_motion),
    torch.tensor([0, 2]),
    device="cpu",
    smpl_motion_file=str(smpl_dir),
    smpl_strict_pairing=True,
  )

  assert tuple(loader.smpl_joints.shape) == (3, 24, 3)
  assert tuple(loader.smpl_root_quat_w.shape) == (3, 4)
  assert torch.allclose(loader.smpl_joints, torch.tensor(smpl_joints))
  assert not torch.allclose(loader.smpl_root_quat_w, torch.eye(1, 4).repeat(3, 1))


def test_motion_loader_strict_pairing_requires_external_smpl(
  tmp_path: Path,
) -> None:
  robot_motion = tmp_path / "230509__missing_pair.npz"
  smpl_dir = tmp_path / "smpl"
  smpl_dir.mkdir()
  _write_motion(robot_motion, frames=3, offset=0.0)

  with pytest.raises(FileNotFoundError, match="No paired SMPL motion"):
    MotionLoader(
      str(robot_motion),
      torch.tensor([0]),
      device="cpu",
      smpl_motion_file=str(smpl_dir),
      smpl_strict_pairing=True,
    )


def test_motion_loader_strict_pairing_trims_mismatched_frame_count(
  tmp_path: Path,
) -> None:
  robot_motion = tmp_path / "230509__short_pair.npz"
  smpl_dir = tmp_path / "smpl"
  smpl_dir.mkdir()
  _write_motion(robot_motion, frames=3, offset=0.0)
  smpl_joints = np.zeros((2, 24, 3), dtype=np.float32)
  with (smpl_dir / "short_pair.pkl").open("wb") as f:
    pickle.dump({"smpl_joints": smpl_joints, "fps": 50}, f)

  loader = MotionLoader(
    str(robot_motion),
    torch.tensor([0]),
    device="cpu",
    smpl_motion_file=str(smpl_dir),
    smpl_strict_pairing=True,
  )

  assert tuple(loader.joint_pos.shape) == (2, 2)
  assert tuple(loader.smpl_joints.shape) == (2, 24, 3)
  assert tuple(loader.smpl_root_quat_w.shape) == (2, 4)


def test_motion_loader_uses_zero_smpl_when_pair_is_missing(tmp_path: Path) -> None:
  robot_motion = tmp_path / "motion.npz"
  _write_motion(robot_motion, frames=3, offset=0.0)

  loader = MotionLoader(str(robot_motion), torch.tensor([0]), device="cpu")

  assert tuple(loader.smpl_joints.shape) == (3, 24, 3)
  assert tuple(loader.smpl_root_quat_w.shape) == (3, 4)
  assert torch.count_nonzero(loader.smpl_joints).item() == 0
  assert torch.allclose(loader.smpl_root_quat_w[:, 0], torch.ones(3))


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
