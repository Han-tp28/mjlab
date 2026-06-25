"""Generate encoder_input_mode2.csv for a motion NPZ + SMPL PKL pair.

Usage:
    uv run python scripts/generate_mode2_csv.py \
        --npz  data/vr_m3_1_npz/220705__Idle_Left_001__A017.npz \
        --pkl  /home/hantp/Groot-WholeBodyControl/data/smpl_filtered/Idle_Left_001__A017.pkl \
        --out  export/vr_m3_1_mode2_csv/Idle_Left_001__A017
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import joblib
import numpy as np


# ---------------------------------------------------------------------------
# Quaternion helpers (wxyz convention)
# ---------------------------------------------------------------------------


def _aa_to_quat(aa: np.ndarray) -> np.ndarray:
  """Axis-angle (N,3) → quaternion (N,4) wxyz."""
  aa = np.asarray(aa, dtype=np.float64).reshape(-1, 3)
  angle = np.linalg.norm(aa, axis=-1, keepdims=True)
  safe = np.where(angle > 1e-8, angle, 1.0)
  axis = aa / safe
  s = np.sin(angle / 2)
  c = np.cos(angle / 2)
  return np.concatenate([c, axis * s], axis=-1).astype(np.float32)


def _qmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
  """Quaternion multiply (wxyz), shapes (...,4)."""
  w1, x1, y1, z1 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
  w2, x2, y2, z2 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
  return np.stack(
    [
      w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
      w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
      w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
      w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ],
    axis=-1,
  ).astype(np.float32)


def _qnorm(q: np.ndarray) -> np.ndarray:
  return q / (np.linalg.norm(q, axis=-1, keepdims=True) + 1e-8)


def _qinv(q: np.ndarray) -> np.ndarray:
  inv = q.copy()
  inv[..., 1:] *= -1
  return inv


def _qapply(q: np.ndarray, v: np.ndarray) -> np.ndarray:
  """Rotate vector v by quaternion q (wxyz)."""
  qv = np.zeros((*q.shape[:-1], 4), dtype=np.float32)
  qv[..., 1:] = v
  return _qmul(_qmul(q, qv), _qinv(q))[..., 1:]


def _qmat(q: np.ndarray) -> np.ndarray:
  """Quaternion (wxyz) → 3×3 rotation matrix."""
  w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
  R = np.stack(
    [
      1 - 2 * (y * y + z * z),
      2 * (x * y - w * z),
      2 * (x * z + w * y),
      2 * (x * y + w * z),
      1 - 2 * (x * x + z * z),
      2 * (y * z - w * x),
      2 * (x * z - w * y),
      2 * (y * z + w * x),
      1 - 2 * (x * x + y * y),
    ],
    axis=-1,
  ).reshape(*q.shape[:-1], 3, 3)
  return R.astype(np.float32)


def _rot6d(q: np.ndarray) -> np.ndarray:
  """(N,4) wxyz → (N,6) 6D rotation (row-major first-2-column submatrix)."""
  R = _qmat(q)
  return np.stack(
    [
      R[..., 0, 0],
      R[..., 0, 1],
      R[..., 1, 0],
      R[..., 1, 1],
      R[..., 2, 0],
      R[..., 2, 1],
    ],
    axis=-1,
  ).astype(np.float32)


# ---------------------------------------------------------------------------
# SMPL root quaternion extraction (matching MotionLib in commands.py)
# ---------------------------------------------------------------------------


def smpl_root_quat(pose_aa: np.ndarray, smpl_y_up: bool) -> np.ndarray:
  """pose_aa (N,72) → root quaternion (N,4) wxyz, z-up, base-rot removed."""
  root_aa = pose_aa[:, :3]
  q = _aa_to_quat(root_aa)
  if smpl_y_up:
    # Rotate by Rx(pi/2): converts y-up to z-up
    rx = _aa_to_quat(np.array([[np.pi / 2.0, 0.0, 0.0]]))
    rx = np.broadcast_to(rx, q.shape)
    q = _qnorm(_qmul(rx, q))
  # Remove SMPL T-pose base rotation (conj of [0.5, 0.5, 0.5, 0.5])
  base_conj = np.array([[0.5, -0.5, -0.5, -0.5]], dtype=np.float32)
  base_conj = np.broadcast_to(base_conj, q.shape)
  return _qnorm(_qmul(q, base_conj))


# ---------------------------------------------------------------------------
# Wrist joint indices in the 29-joint NPZ (VR M3.1 ordering)
# SMPL_WRIST_JOINT_NAMES order: left_wrist_roll, right_wrist_roll,
#                                left_wrist_pitch, right_wrist_pitch,
#                                left_wrist_yaw, right_wrist_yaw
# ---------------------------------------------------------------------------
WRIST_IDX = [18, 25, 19, 26, 17, 24]


def generate(
  npz_path: str,
  pkl_path: str,
  out_dir: str,
  num_future: int = 10,
  smpl_y_up: bool = True,
) -> None:
  out = Path(out_dir)
  out.mkdir(parents=True, exist_ok=True)

  npz = np.load(npz_path, allow_pickle=True)
  pkl = joblib.load(pkl_path)

  joint_pos = npz["joint_pos"].astype(np.float32)  # (T, 29)
  body_quat_w = npz["body_quat_w"].astype(np.float32)  # (T, bodies, 4), wxyz
  smpl_joints_w = pkl["smpl_joints"].astype(np.float32)  # (T, 24, 3) world
  pose_aa = pkl["pose_aa"].astype(np.float32)  # (T, 72)

  T = min(len(joint_pos), len(body_quat_w), len(smpl_joints_w), len(pose_aa))
  joint_pos = joint_pos[:T]
  body_quat_w = body_quat_w[:T]
  smpl_joints_w = smpl_joints_w[:T]
  pose_aa = pose_aa[:T]

  root_quat = smpl_root_quat(pose_aa, smpl_y_up)  # (T, 4) wxyz
  # Match mjlab play: smpl_root_ori_b_multi_future = inv(robot_anchor_quat_w) * smpl_root_quat_w.
  # In exported motion data the robot anchor is the root body at index 0.
  anchor_quat_inv = _qinv(_qnorm(body_quat_w[:, 0, :]))

  encoder_mode = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)  # mode 4

  rows: list[list[float]] = []
  for t in range(T):
    steps = np.clip(np.arange(t, t + num_future), 0, T - 1)

    sj = smpl_joints_w[steps]  # (10, 24, 3)
    rq = root_quat[steps]  # (10,  4)

    # Local SMPL joints = qapply(qinv(root_quat), smpl_joints)
    rq_inv = _qinv(rq)  # (10, 4)
    rq_inv_expanded = rq_inv[:, None, :].repeat(24, axis=1)  # (10, 24, 4)
    local_j = _qapply(rq_inv_expanded, sj)  # (10, 24, 3)
    smpl_block = local_j.flatten()  # 720

    # 6D root orientation relative to robot/root anchor, same as mjlab play.
    aq_inv = anchor_quat_inv[steps]
    rq_dif = _qmul(aq_inv, rq)  # (10, 4)
    ori_block = _rot6d(rq_dif).flatten()  # 60

    # Wrist joints from NPZ (absolute, 6 joints × 10 future frames)
    wj = joint_pos[steps][:, WRIST_IDX]  # (10, 6)
    wrist_block = wj.flatten()  # 60

    row = np.concatenate(
      [encoder_mode, smpl_block, ori_block, wrist_block]
    )  # 4+720+60+60 = 844
    rows.append(row.tolist())

  motion_name = Path(npz_path).stem
  csv_path = out / "encoder_input_mode2.csv"
  header = [f"f{i}" for i in range(844)]
  with open(csv_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(header)
    w.writerows(rows)

  # Write metadata
  (out / "metadata.txt").write_text(
    f"motion: {motion_name}\n"
    f"frames: {T}\n"
    f"encoder_input_mode2: ({T}, 844)\n"
    f"encoder layout: encoder_mode_4(4), smpl_joints_10frame_step1(720), "
    f"smpl_anchor_orientation_10frame_step1(60), "
    f"motion_joint_positions_wrists_10frame_step1(60)\n"
    f"encoder_mode_4: [0, 0, 1, 0]\n"
    f"smpl_y_up: {str(smpl_y_up).lower()}\n"
    f"robot: vr_m3_1\n"
    f"robot_action_dof: 27\n"
    f"robot_joint_state_dof: 29\n"
    f"source_npz: {os.path.abspath(npz_path)}\n"
    f"source_pkl: {os.path.abspath(pkl_path)}\n"
  )
  print(f"Generated {T} frames → {csv_path}")


def main() -> None:
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--npz", required=True, help="Motion NPZ file")
  p.add_argument("--pkl", required=True, help="SMPL PKL file (joblib/zlib)")
  p.add_argument("--out", required=True, help="Output directory")
  p.add_argument("--num-future", type=int, default=10)
  p.add_argument("--no-smpl-y-up", action="store_true")
  args = p.parse_args()
  generate(args.npz, args.pkl, args.out, args.num_future, not args.no_smpl_y_up)


if __name__ == "__main__":
  main()
