from pathlib import Path

import pytest

from mjlab.scripts.csv_to_npz import (
  G1_JOINT_NAMES,
  VR_H3_1_27_JOINT_NAMES,
  VR_H3_1_28_SOURCE_JOINT_NAMES,
  VR_H3_1_JOINT_NAMES,
  VR_M3_1_27_JOINT_NAMES,
  VR_M3_1_28_SOURCE_JOINT_NAMES,
  VR_M3_1_30_SOURCE_JOINT_NAMES,
  VR_M3_1_JOINT_NAMES,
  _resolve_motion_joint_mapping,
  _resolve_motion_joint_names,
  _resolve_output,
)


def test_resolve_motion_joint_names_accepts_full_g1_order() -> None:
  assert _resolve_motion_joint_names(G1_JOINT_NAMES, 29) == G1_JOINT_NAMES


def test_resolve_motion_joint_names_accepts_27_dof_without_wrist_yaw() -> None:
  joint_names = _resolve_motion_joint_names(G1_JOINT_NAMES, 27)

  assert len(joint_names) == 27
  assert "left_wrist_yaw_joint" not in joint_names
  assert "right_wrist_yaw_joint" not in joint_names


def test_resolve_motion_joint_names_rejects_unexpected_dof_count() -> None:
  with pytest.raises(ValueError, match="got 28 joint columns"):
    _resolve_motion_joint_names(G1_JOINT_NAMES, 28)


def test_resolve_motion_joint_names_accepts_vr_h3_1_order() -> None:
  assert (
    _resolve_motion_joint_names(VR_H3_1_JOINT_NAMES, 27, robot="vr_h3_1")
    == VR_H3_1_JOINT_NAMES
  )


def test_resolve_motion_joint_names_accepts_27_dof_vr_h3_1_default() -> None:
  joint_names = _resolve_motion_joint_names(VR_H3_1_JOINT_NAMES, 27, robot="vr_h3_1")

  assert joint_names == VR_H3_1_27_JOINT_NAMES
  assert "waist_roll_joint" not in joint_names


def test_resolve_motion_joint_mapping_drops_vr_h3_1_source_waist_roll() -> None:
  joint_names, column_indexes = _resolve_motion_joint_mapping(
    VR_H3_1_JOINT_NAMES,
    len(VR_H3_1_28_SOURCE_JOINT_NAMES),
    robot="vr_h3_1",
  )

  assert joint_names == VR_H3_1_JOINT_NAMES
  assert "waist_roll_joint" not in joint_names
  assert column_indexes.tolist() == [
    i
    for i, name in enumerate(VR_H3_1_28_SOURCE_JOINT_NAMES)
    if name != "waist_roll_joint"
  ]


def test_resolve_motion_joint_names_accepts_custom_vr_h3_1_order() -> None:
  custom_names = VR_H3_1_JOINT_NAMES[:-1]

  assert (
    _resolve_motion_joint_names(
      VR_H3_1_JOINT_NAMES,
      26,
      robot="vr_h3_1",
      motion_joint_names=custom_names,
    )
    == custom_names
  )


def test_resolve_motion_joint_names_accepts_vr_m3_1_full_order() -> None:
  assert (
    _resolve_motion_joint_names(VR_M3_1_JOINT_NAMES, 29, robot="vr_m3_1")
    == VR_M3_1_JOINT_NAMES
  )


def test_resolve_motion_joint_names_accepts_27_dof_vr_m3_1_default() -> None:
  joint_names = _resolve_motion_joint_names(VR_M3_1_JOINT_NAMES, 27, robot="vr_m3_1")

  assert joint_names == VR_M3_1_27_JOINT_NAMES
  assert "head_yaw_joint" not in joint_names
  assert "head_pitch_joint" not in joint_names
  assert "waist_roll_joint" not in joint_names


def test_resolve_motion_joint_mapping_drops_vr_m3_1_source_waist_roll() -> None:
  joint_names, column_indexes = _resolve_motion_joint_mapping(
    VR_M3_1_JOINT_NAMES,
    len(VR_M3_1_28_SOURCE_JOINT_NAMES),
    robot="vr_m3_1",
  )

  assert joint_names == VR_M3_1_27_JOINT_NAMES
  assert "waist_roll_joint" not in joint_names
  assert column_indexes.tolist() == [
    i
    for i, name in enumerate(VR_M3_1_28_SOURCE_JOINT_NAMES)
    if name != "waist_roll_joint"
  ]


def test_resolve_motion_joint_mapping_drops_vr_m3_1_full_source_waist_roll() -> None:
  joint_names, column_indexes = _resolve_motion_joint_mapping(
    VR_M3_1_JOINT_NAMES,
    len(VR_M3_1_30_SOURCE_JOINT_NAMES),
    robot="vr_m3_1",
  )

  assert joint_names == VR_M3_1_JOINT_NAMES
  assert "waist_roll_joint" not in joint_names
  assert column_indexes.tolist() == [
    i
    for i, name in enumerate(VR_M3_1_30_SOURCE_JOINT_NAMES)
    if name != "waist_roll_joint"
  ]


def test_resolve_output_accepts_motion_name_or_npz_path() -> None:
  assert _resolve_output("flip") == (Path("/tmp/motion.npz"), "flip")
  assert _resolve_output("/tmp/flip.npz") == (Path("/tmp/flip.npz"), "flip")
