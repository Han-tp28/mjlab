"""Pack per-motion NPZ files into lossless motion shards."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import tyro

import mjlab

MOTION_ARRAY_KEYS = (
  "joint_pos",
  "joint_vel",
  "body_pos_w",
  "body_quat_w",
  "body_lin_vel_w",
  "body_ang_vel_w",
)
SHARD_FORMAT_KEY = "mjlab_motion_shard_format"


def _safe_motion_name(path: Path, input_dir: Path) -> str:
  return "__".join(path.relative_to(input_dir).with_suffix("").parts)


def _find_input_files(input_dir: Path, glob: str) -> list[Path]:
  files = sorted(input_dir.rglob(glob))
  return [file for file in files if file.suffix == ".npz"]


def _read_motion(path: Path) -> dict[str, np.ndarray]:
  with np.load(path) as data:
    missing = [key for key in MOTION_ARRAY_KEYS if key not in data]
    if missing:
      raise KeyError(f"Motion file {path} is missing keys: {missing}")
    if SHARD_FORMAT_KEY in data:
      raise ValueError(
        f"Input file {path} is already a motion shard. Use raw per-motion NPZ files."
      )
    motion = {key: np.asarray(data[key]) for key in MOTION_ARRAY_KEYS}
    if "fps" in data:
      motion["fps"] = np.asarray(data["fps"])
    return motion


def _write_shard(
  shard_path: Path,
  motion_files: list[Path],
  input_dir: Path,
  compressed: bool,
) -> None:
  if not motion_files:
    raise ValueError("Cannot write an empty motion shard.")

  arrays: dict[str, list[np.ndarray]] = {key: [] for key in MOTION_ARRAY_KEYS}
  motion_lengths: list[int] = []
  motion_names: list[str] = []
  source_files: list[str] = []
  fps_values: list[float] = []
  expected_shapes: dict[str, tuple[int, ...]] = {}

  for path in motion_files:
    motion = _read_motion(path)
    length = int(motion["joint_pos"].shape[0])
    if length < 2:
      raise ValueError(f"Motion file {path} must contain at least 2 frames")

    for key in MOTION_ARRAY_KEYS:
      array = motion[key]
      if array.shape[0] != length:
        raise ValueError(
          f"Motion file {path} key {key} has {array.shape[0]} frames, expected {length}"
        )
      shape = array.shape[1:]
      if key not in expected_shapes:
        expected_shapes[key] = shape
      elif shape != expected_shapes[key]:
        raise ValueError(
          f"Motion file {path} key {key} shape {shape} does not match "
          f"expected {expected_shapes[key]}"
        )
      arrays[key].append(array)

    motion_lengths.append(length)
    motion_names.append(_safe_motion_name(path, input_dir))
    source_files.append(str(path))
    fps = np.asarray(motion.get("fps", np.array([np.nan], dtype=np.float32))).reshape(
      -1
    )
    fps_values.append(float(fps[0]))

  payload: dict[str, np.ndarray] = {
    SHARD_FORMAT_KEY: np.array([1], dtype=np.int32),
    "motion_lengths": np.asarray(motion_lengths, dtype=np.int64),
    "motion_names": np.asarray(motion_names),
    "source_files": np.asarray(source_files),
    "fps": np.asarray(fps_values, dtype=np.float32),
  }
  payload.update({key: np.concatenate(value, axis=0) for key, value in arrays.items()})

  shard_path.parent.mkdir(parents=True, exist_ok=True)
  if compressed:
    cast(Any, np.savez_compressed)(shard_path, **payload)
  else:
    cast(Any, np.savez)(shard_path, **payload)


def main(
  input_dir: str,
  output_dir: str,
  shard_size: int = 512,
  glob: str = "*.npz",
  compressed: bool = True,
  skip_existing: bool = True,
  limit: int | None = None,
) -> None:
  """Pack per-motion NPZ files into lossless sharded NPZ files.

  The shard keeps all current motion arrays exactly as arrays:
  joint_pos, joint_vel, body_pos_w, body_quat_w, body_lin_vel_w,
  body_ang_vel_w. It only changes storage layout from many small files to fewer
  larger files with motion_lengths metadata.
  """
  if shard_size <= 0:
    raise ValueError("`shard_size` must be positive.")

  input_path = Path(input_dir).expanduser().resolve()
  output_path = Path(output_dir).expanduser().resolve()
  if not input_path.is_dir():
    raise FileNotFoundError(f"Input directory not found: {input_path}")

  files = _find_input_files(input_path, glob)
  if limit is not None:
    files = files[:limit]
  if not files:
    raise FileNotFoundError(f"No files matching {glob!r} found in {input_path}")

  print(
    f"[INFO] Packing {len(files)} motions from {input_path} into shards at "
    f"{output_path}"
  )
  print(f"[INFO] shard_size={shard_size}, compressed={compressed}")

  written = 0
  skipped = 0
  for shard_index, start in enumerate(range(0, len(files), shard_size)):
    chunk = files[start : start + shard_size]
    shard_path = output_path / f"motion_shard_{shard_index:06d}.npz"
    if skip_existing and shard_path.exists():
      skipped += 1
      continue

    print(f"[{start + 1}-{start + len(chunk)}/{len(files)}] Write {shard_path.name}")
    _write_shard(shard_path, chunk, input_path, compressed=compressed)
    written += 1

  print(f"[INFO] Done. Written shards: {written}, skipped shards: {skipped}")


def cli() -> None:
  tyro.cli(main, config=mjlab.TYRO_FLAGS)


if __name__ == "__main__":
  cli()
