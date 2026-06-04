#!/usr/bin/env python3
"""Offline heuristic scanner for hard VR-M3 motion clips.

This is intentionally conservative: it does not simulate the robot. It flags
motions that have repeatedly been hard for the current tracking curriculum:
lying/faint/death clips, flips, big jumps, large root height changes, and high
root angular/linear velocities.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


KEYWORD_WEIGHTS = {
  "50cm": 6.0,
  "all_fours": 5.0,
  "crawl": 5.0,
  "crouch": 3.0,
  "death": 8.0,
  "faint": 8.0,
  "fall": 8.0,
  "flip": 9.0,
  "high_jump": 7.0,
  "jump_off": 9.0,
  "jump_on": 9.0,
  "kick_back": 6.0,
  "lying": 8.0,
  "on_the_edge": 8.0,
  "roll": 6.0,
  "side_lift": 5.0,
  "stand_up_lying": 8.0,
  "toxic_gas": 8.0,
  "wall": 5.0,
}


def resolve_motion_path(root: Path, entry: str) -> Path:
  path = Path(entry.strip())
  if path.is_absolute():
    return path
  candidate = root / path
  if candidate.exists():
    return candidate
  return root / "data" / path


def percentile(values: np.ndarray, q: float) -> float:
  if values.size == 0:
    return 0.0
  return float(np.percentile(values, q))


def score_motion(path: Path) -> dict[str, object]:
  name = path.stem
  if name.endswith("_M"):
    name = name[:-2]

  score = 0.0
  reasons: list[str] = []
  lowered = name.lower()
  for keyword, weight in KEYWORD_WEIGHTS.items():
    if keyword in lowered:
      score += weight
      reasons.append(f"name:{keyword}")

  with np.load(path, allow_pickle=False) as data:
    fps = float(np.asarray(data["fps"]).reshape(-1)[0])
    body_pos = np.asarray(data["body_pos_w"], dtype=np.float32)
    body_quat = np.asarray(data["body_quat_w"], dtype=np.float32)
    body_lin_vel = np.asarray(data["body_lin_vel_w"], dtype=np.float32)
    body_ang_vel = np.asarray(data["body_ang_vel_w"], dtype=np.float32)
    joint_vel = np.asarray(data["joint_vel"], dtype=np.float32)

  frames = int(body_pos.shape[0])
  duration_s = frames / fps if fps > 0 else 0.0

  root_pos = body_pos[:, 0]
  root_quat = body_quat[:, 0]
  root_lin_vel = body_lin_vel[:, 0]
  root_ang_vel = body_ang_vel[:, 0]

  root_z = root_pos[:, 2]
  root_z_min = float(np.min(root_z))
  root_z_max = float(np.max(root_z))
  root_z_range = root_z_max - root_z_min
  body_z_min = float(np.min(body_pos[..., 2]))
  root_speed_xy_p95 = percentile(np.linalg.norm(root_lin_vel[:, :2], axis=1), 95)
  root_speed_z_p95 = percentile(np.abs(root_lin_vel[:, 2]), 95)
  root_ang_p95 = percentile(np.linalg.norm(root_ang_vel, axis=1), 95)
  joint_vel_p95 = percentile(np.linalg.norm(joint_vel, axis=1), 95)
  root_w_abs_min = float(np.min(np.abs(root_quat[:, 0])))

  if duration_s > 30.0:
    score += 2.0
    reasons.append(f"duration>{duration_s:.1f}s")
  if root_z_min < 0.45:
    score += 8.0
    reasons.append(f"low_root_z={root_z_min:.2f}")
  elif root_z_min < 0.65:
    score += 4.0
    reasons.append(f"low_root_z={root_z_min:.2f}")
  if root_z_range > 1.0:
    score += 8.0
    reasons.append(f"root_z_range={root_z_range:.2f}")
  elif root_z_range > 0.65:
    score += 4.0
    reasons.append(f"root_z_range={root_z_range:.2f}")
  if body_z_min < -0.10:
    score += 5.0
    reasons.append(f"body_below_ground={body_z_min:.2f}")
  if root_speed_xy_p95 > 4.0:
    score += 5.0
    reasons.append(f"root_xy_vel_p95={root_speed_xy_p95:.2f}")
  elif root_speed_xy_p95 > 3.0:
    score += 2.5
    reasons.append(f"root_xy_vel_p95={root_speed_xy_p95:.2f}")
  if root_speed_z_p95 > 3.0:
    score += 5.0
    reasons.append(f"root_z_vel_p95={root_speed_z_p95:.2f}")
  elif root_speed_z_p95 > 2.0:
    score += 2.5
    reasons.append(f"root_z_vel_p95={root_speed_z_p95:.2f}")
  if root_ang_p95 > 8.0:
    score += 5.0
    reasons.append(f"root_ang_vel_p95={root_ang_p95:.2f}")
  elif root_ang_p95 > 5.0:
    score += 2.5
    reasons.append(f"root_ang_vel_p95={root_ang_p95:.2f}")
  if joint_vel_p95 > 35.0:
    score += 3.0
    reasons.append(f"joint_vel_norm_p95={joint_vel_p95:.1f}")
  if root_w_abs_min < 0.35:
    score += 4.0
    reasons.append(f"large_root_rotation_w_min={root_w_abs_min:.2f}")

  return {
    "score": round(score, 3),
    "motion": name,
    "path": str(path),
    "duration_s": round(duration_s, 3),
    "root_z_min": round(root_z_min, 3),
    "root_z_range": round(root_z_range, 3),
    "body_z_min": round(body_z_min, 3),
    "root_speed_xy_p95": round(root_speed_xy_p95, 3),
    "root_speed_z_p95": round(root_speed_z_p95, 3),
    "root_ang_p95": round(root_ang_p95, 3),
    "joint_vel_p95": round(joint_vel_p95, 3),
    "reasons": ",".join(reasons),
  }


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--playlist",
    default="data/vr_m3_1_full_unique_names.txt",
    help="Playlist txt containing motion paths.",
  )
  parser.add_argument("--output", default="data/vr_m3_1_offline_hard_motion_scan.tsv")
  parser.add_argument("--top", type=int, default=200)
  parser.add_argument("--min-score", type=float, default=8.0)
  args = parser.parse_args()

  root = Path.cwd()
  playlist = Path(args.playlist)
  entries = [line.strip() for line in playlist.read_text().splitlines() if line.strip()]

  rows: list[dict[str, object]] = []
  for i, entry in enumerate(entries, start=1):
    path = resolve_motion_path(root, entry)
    if not path.exists():
      rows.append({
        "score": 999.0,
        "motion": Path(entry).stem,
        "path": str(path),
        "duration_s": 0.0,
        "root_z_min": 0.0,
        "root_z_range": 0.0,
        "body_z_min": 0.0,
        "root_speed_xy_p95": 0.0,
        "root_speed_z_p95": 0.0,
        "root_ang_p95": 0.0,
        "joint_vel_p95": 0.0,
        "reasons": "missing_file",
      })
      continue
    try:
      rows.append(score_motion(path))
    except Exception as exc:  # noqa: BLE001 - scanner should keep going.
      rows.append({
        "score": 999.0,
        "motion": path.stem,
        "path": str(path),
        "duration_s": 0.0,
        "root_z_min": 0.0,
        "root_z_range": 0.0,
        "body_z_min": 0.0,
        "root_speed_xy_p95": 0.0,
        "root_speed_z_p95": 0.0,
        "root_ang_p95": 0.0,
        "joint_vel_p95": 0.0,
        "reasons": f"load_error:{type(exc).__name__}",
      })
    if i % 500 == 0:
      print(f"scanned {i}/{len(entries)}")

  rows.sort(key=lambda row: float(row["score"]), reverse=True)
  output = Path(args.output)
  output.parent.mkdir(parents=True, exist_ok=True)
  fieldnames = [
    "score",
    "motion",
    "path",
    "duration_s",
    "root_z_min",
    "root_z_range",
    "body_z_min",
    "root_speed_xy_p95",
    "root_speed_z_p95",
    "root_ang_p95",
    "joint_vel_p95",
    "reasons",
  ]
  with output.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
    writer.writeheader()
    writer.writerows(rows)

  hard = [row for row in rows if float(row["score"]) >= args.min_score]
  print(f"wrote {output}")
  print(f"motions={len(rows)} hard_score>={args.min_score:g}={len(hard)}")
  print(f"top {min(args.top, len(rows))}:")
  for row in rows[: args.top]:
    print(
      f"{row['score']}\t{row['motion']}\t{row['reasons']}\t{row['path']}"
    )


if __name__ == "__main__":
  main()
