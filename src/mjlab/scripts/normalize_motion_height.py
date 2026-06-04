"""Normalize motion NPZ root height against the robot XML foot height."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import mujoco
import numpy as np
import tyro
from tqdm import tqdm

import mjlab
from mjlab.asset_zoo.robots.vr_h3_1.vr_h3_1_constants import (
  HOME_KEYFRAME,
  VR_H3_1_XML,
)

MOTION_ARRAY_KEYS = (
  "joint_pos",
  "joint_vel",
  "body_pos_w",
  "body_quat_w",
  "body_lin_vel_w",
  "body_ang_vel_w",
)


@dataclass(frozen=True)
class RobotHeightInfo:
  model: mujoco.MjModel
  target_foot_center_z: float
  non_free_joint_ids: tuple[int, ...]
  foot_geom_ids: tuple[int, ...]


def _iter_motion_files(input_path: Path, glob: str) -> list[Path]:
  if input_path.is_dir():
    return sorted(input_path.rglob(glob))
  if input_path.is_file() and input_path.suffix.lower() in {".txt", ".lst", ".list"}:
    files: list[Path] = []
    base_dir = input_path.parent
    for raw_line in input_path.read_text().splitlines():
      line = raw_line.split("#", maxsplit=1)[0].strip()
      if not line:
        continue
      entry = Path(line.split()[-1]).expanduser()
      if not entry.is_absolute():
        entry = base_dir / entry
      if entry.is_dir():
        files.extend(sorted(entry.rglob(glob)))
      else:
        files.append(entry)
    return files
  if input_path.is_file():
    return [input_path]
  raise FileNotFoundError(f"Motion input not found: {input_path}")


def _load_robot_height_info(robot: Literal["vr_h3_1"]) -> RobotHeightInfo:
  if robot != "vr_h3_1":
    raise ValueError(f"Unsupported robot for height normalization: {robot}")

  model = mujoco.MjModel.from_xml_path(str(VR_H3_1_XML))
  geom_names = [
    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) for i in range(model.ngeom)
  ]
  foot_geom_ids = tuple(
    i
    for i, name in enumerate(geom_names)
    if name is not None and "ankle_roll_link_collision" in name
  )
  if not foot_geom_ids:
    raise RuntimeError("Could not find VR H3.1 foot collision geoms in XML.")

  non_free_joint_ids = tuple(
    j for j in range(model.njnt) if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE
  )

  data = mujoco.MjData(model)
  data.qpos[:] = 0.0
  data.qvel[:] = 0.0
  data.qpos[0:3] = np.asarray(HOME_KEYFRAME.pos)
  data.qpos[3] = 1.0
  for j in non_free_joint_ids:
    joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
    if joint_name in HOME_KEYFRAME.joint_pos:
      data.qpos[model.jnt_qposadr[j]] = HOME_KEYFRAME.joint_pos[joint_name]
  mujoco.mj_forward(model, data)

  return RobotHeightInfo(
    model=model,
    target_foot_center_z=float(data.geom_xpos[list(foot_geom_ids), 2].min()),
    non_free_joint_ids=non_free_joint_ids,
    foot_geom_ids=foot_geom_ids,
  )


def _motion_foot_center_z(
  info: RobotHeightInfo,
  joint_pos: np.ndarray,
  body_pos_w: np.ndarray,
  body_quat_w: np.ndarray,
) -> float:
  if joint_pos.shape[1] != len(info.non_free_joint_ids):
    raise ValueError(
      f"Motion joint dim {joint_pos.shape[1]} does not match XML joint dim "
      f"{len(info.non_free_joint_ids)}."
    )

  data = mujoco.MjData(info.model)
  data.qpos[:] = 0.0
  data.qvel[:] = 0.0
  data.qpos[0:3] = body_pos_w[0, 0]
  root_quat = body_quat_w[0, 0].copy()
  root_quat /= np.linalg.norm(root_quat)
  data.qpos[3:7] = root_quat
  for k, j in enumerate(info.non_free_joint_ids):
    data.qpos[info.model.jnt_qposadr[j]] = joint_pos[0, k]
  mujoco.mj_forward(info.model, data)
  return float(data.geom_xpos[list(info.foot_geom_ids), 2].min())


def _normalize_motion(
  input_file: Path,
  output_file: Path,
  info: RobotHeightInfo,
  fixed_z_offset: float | None,
  dry_run: bool,
  overwrite: bool,
) -> float:
  if output_file.exists() and not overwrite:
    return float("nan")

  with np.load(input_file) as data:
    missing = [key for key in MOTION_ARRAY_KEYS if key not in data]
    if missing:
      raise KeyError(f"{input_file} is missing motion keys: {missing}")
    arrays = {key: np.asarray(data[key]) for key in data.files}

  joint_pos = np.asarray(arrays["joint_pos"], dtype=np.float32)
  body_pos_w = np.asarray(arrays["body_pos_w"], dtype=np.float32).copy()
  body_quat_w = np.asarray(arrays["body_quat_w"], dtype=np.float32)

  z_offset = (
    fixed_z_offset
    if fixed_z_offset is not None
    else _motion_foot_center_z(info, joint_pos, body_pos_w, body_quat_w)
    - info.target_foot_center_z
  )

  if dry_run:
    return float(z_offset)

  body_pos_w[:, :, 2] -= z_offset
  arrays["body_pos_w"] = body_pos_w
  arrays["height_normalization_z_offset"] = np.asarray([z_offset], dtype=np.float32)
  arrays["height_normalization_target_foot_center_z"] = np.asarray(
    [info.target_foot_center_z], dtype=np.float32
  )

  output_file.parent.mkdir(parents=True, exist_ok=True)
  np.savez(output_file, **arrays)
  return float(z_offset)


def main(
  input_path: str,
  output_dir: str,
  robot: Literal["vr_h3_1"] = "vr_h3_1",
  glob: str = "*.npz",
  fixed_z_offset: float | None = None,
  output_playlist: str | None = None,
  limit: int | None = None,
  dry_run: bool = False,
  overwrite: bool = False,
) -> None:
  """Normalize motion NPZ height without changing joint data or velocities.

  The script writes corrected copies. It only shifts ``body_pos_w[..., 2]`` by a
  constant per motion; joint positions, joint velocities, orientations, and
  linear/angular velocities are preserved.
  """

  source = Path(input_path).expanduser()
  output_root = Path(output_dir).expanduser()
  files = _iter_motion_files(source, glob)
  if limit is not None:
    files = files[:limit]
  if not files:
    raise FileNotFoundError(f"No motion NPZ files found in {source}")

  info = _load_robot_height_info(robot)
  print(
    f"[INFO] Target foot collision center z for {robot}: "
    f"{info.target_foot_center_z:.6f}"
  )
  print(f"[INFO] Normalizing {len(files)} motion files.")

  offsets: list[float] = []
  output_files: list[Path] = []
  for input_file in tqdm(files, desc="Normalizing motions", unit="motion"):
    if source.is_dir():
      relative = input_file.resolve().relative_to(source.resolve())
    else:
      relative = Path(input_file.name)
    output_file = output_root / relative
    offset = _normalize_motion(
      input_file=input_file,
      output_file=output_file,
      info=info,
      fixed_z_offset=fixed_z_offset,
      dry_run=dry_run,
      overwrite=overwrite,
    )
    if not np.isnan(offset):
      offsets.append(offset)
    output_files.append(output_file)

  if offsets:
    offset_array = np.asarray(offsets, dtype=np.float32)
    print(
      "[INFO] Applied z offsets: "
      f"min={offset_array.min():.6f}, mean={offset_array.mean():.6f}, "
      f"max={offset_array.max():.6f}"
    )

  if output_playlist is not None and not dry_run:
    playlist_path = Path(output_playlist).expanduser()
    playlist_path.parent.mkdir(parents=True, exist_ok=True)
    playlist_path.write_text(
      "\n".join(str(path.resolve()) for path in output_files) + "\n"
    )
    print(f"[INFO] Wrote normalized playlist: {playlist_path}")


def cli() -> None:
  tyro.cli(main, config=mjlab.TYRO_FLAGS)


if __name__ == "__main__":
  cli()
