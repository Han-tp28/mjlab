from pathlib import Path

import numpy as np
import torch

from mjlab.scripts.shard_npz_motions import _write_shard
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


def test_shard_npz_motions_writes_loader_compatible_shard(tmp_path: Path) -> None:
  input_dir = tmp_path / "motions"
  input_dir.mkdir()
  first = input_dir / "first.npz"
  second = input_dir / "second.npz"
  _write_motion(first, frames=3, offset=0.0)
  _write_motion(second, frames=4, offset=100.0)

  shard = tmp_path / "motion_shard_000000.npz"
  _write_shard(shard, [first, second], input_dir, compressed=True)

  with np.load(shard) as data:
    assert data["fps"].tolist() == [50.0, 50.0]

  loader = MotionLoader(str(shard), torch.tensor([0, 2]), device="cpu")

  assert loader.motion_names == ("first", "second")
  assert loader.motion_start_steps.tolist() == [0, 3]
  assert loader.motion_end_steps.tolist() == [3, 7]
  assert loader.joint_pos[3, 0].item() == 100.0
  assert tuple(loader.body_pos_w.shape) == (7, 2, 3)
