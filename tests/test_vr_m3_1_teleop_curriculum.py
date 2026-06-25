from mjlab.scripts.build_vr_m3_1_teleop_curriculum import build_playlist_lines


def test_build_playlist_lines_interleaves_teleop_with_prioritized_clean() -> None:
  clean_entries = [
    "vr_m3_1_npz/turn_start_jog_001.npz",
    "vr_m3_1_npz/idle_loop_001.npz",
    "vr_m3_1_npz/walk_forward_001.npz",
    "vr_m3_1_npz/dance_001.npz",
  ]
  teleop_entries = [
    "vr_m3_1_npz_teleop/1.npz",
    "vr_m3_1_npz_teleop/2.npz",
  ]

  playlist = build_playlist_lines(
    clean_entries,
    teleop_entries,
    clean_per_teleop=2,
  )

  assert playlist == [
    "vr_m3_1_npz/idle_loop_001.npz",
    "vr_m3_1_npz/walk_forward_001.npz",
    "vr_m3_1_npz_teleop/1.npz",
    "vr_m3_1_npz/dance_001.npz",
    "vr_m3_1_npz/turn_start_jog_001.npz",
    "vr_m3_1_npz_teleop/2.npz",
  ]


def test_build_playlist_lines_deduplicates_entries() -> None:
  playlist = build_playlist_lines(
    [
      "vr_m3_1_npz/idle_loop_001.npz",
      "vr_m3_1_npz/idle_loop_001.npz",
    ],
    [
      "vr_m3_1_npz_teleop/1.npz",
      "vr_m3_1_npz_teleop/1.npz",
    ],
    clean_per_teleop=1,
  )

  assert playlist == [
    "vr_m3_1_npz/idle_loop_001.npz",
    "vr_m3_1_npz_teleop/1.npz",
  ]
