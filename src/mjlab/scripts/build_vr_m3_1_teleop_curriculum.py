"""Build a staged VR M3.1 playlist from clean and teleop motion data."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import tyro

import mjlab

_NATURAL_SPLIT_RE = re.compile(r"(\d+)")
_BALANCE_MOTION_KEYWORDS = (
  "stand",
  "idle",
  "pause",
  "hold",
  "recover",
  "recovery",
  "stop",
)
_EASY_WALK_KEYWORDS = (
  "walk_forward",
  "walk forward",
)
_HARD_EARLY_MOTION_KEYWORDS = (
  "jog",
  "jump",
  "turn",
  "rotate",
  "sideway",
  "randdir",
  "backward",
  "injured",
  "inj",
  "sick",
  "weakend",
  "grab",
  "fast",
  "one_foot",
  "one foot",
  "dance",
  "dancing",
)


@dataclass(frozen=True)
class PlaylistBuildStats:
  """Summary for a generated VR M3.1 teleop-robust curriculum playlist."""

  clean_count: int
  teleop_count: int
  total_count: int
  output_path: Path


def _natural_key(path: Path) -> list[str]:
  return [
    chunk.zfill(12) if chunk.isdigit() else chunk.lower()
    for chunk in _NATURAL_SPLIT_RE.split(path.as_posix())
  ]


def _strip_playlist_line(raw_line: str) -> str | None:
  line = raw_line.split("#", maxsplit=1)[0].strip()
  return line or None


def _read_clean_playlist(path: Path) -> list[str]:
  entries = []
  for raw_line in path.read_text().splitlines():
    line = _strip_playlist_line(raw_line)
    if line is not None:
      entries.append(line)
  return _dedupe_preserving_order(entries)


def _dedupe_preserving_order(entries: list[str]) -> list[str]:
  seen = set()
  unique_entries = []
  for entry in entries:
    if entry in seen:
      continue
    seen.add(entry)
    unique_entries.append(entry)
  return unique_entries


def _clean_curriculum_key(entry: str) -> tuple[int, str]:
  motion_name = Path(entry).stem.lower()
  is_hard = any(keyword in motion_name for keyword in _HARD_EARLY_MOTION_KEYWORDS)
  if any(keyword in motion_name for keyword in _BALANCE_MOTION_KEYWORDS):
    return (0 if not is_hard else 3, motion_name)
  if any(keyword in motion_name for keyword in _EASY_WALK_KEYWORDS):
    return (1 if not is_hard else 3, motion_name)
  return (3 if is_hard else 2, motion_name)


def _prioritize_clean_entries(entries: list[str]) -> list[str]:
  return sorted(entries, key=_clean_curriculum_key)


def _relative_entry(path: Path, base_dir: Path) -> str:
  resolved_path = path.expanduser().resolve()
  resolved_base = base_dir.expanduser().resolve()
  try:
    return resolved_path.relative_to(resolved_base).as_posix()
  except ValueError:
    return resolved_path.as_posix()


def _resolve_entry(entry: str, base_dir: Path) -> Path:
  path = Path(entry).expanduser()
  if path.is_absolute():
    return path
  return base_dir / path


def _validate_playlist_entries(entries: list[str], base_dir: Path) -> None:
  missing = [
    entry for entry in entries if not _resolve_entry(entry, base_dir).is_file()
  ]
  if missing:
    preview = ", ".join(missing[:5])
    suffix = "" if len(missing) <= 5 else f", ... (+{len(missing) - 5} more)"
    raise FileNotFoundError(f"Playlist contains missing files: {preview}{suffix}")


def build_playlist_lines(
  clean_entries: list[str],
  teleop_entries: list[str],
  clean_per_teleop: int = 16,
) -> list[str]:
  """Interleave clean and teleop entries for ordered curriculum loading.

  Clean motions still dominate the library, but teleop clips appear throughout
  the early ordered stages so the policy sees start/stop/pause/recovery behavior
  while the clean corpus is still being introduced.
  """
  if clean_per_teleop <= 0:
    raise ValueError("`clean_per_teleop` must be positive.")

  clean_entries = _prioritize_clean_entries(_dedupe_preserving_order(clean_entries))
  teleop_entries = _dedupe_preserving_order(teleop_entries)

  playlist: list[str] = []
  clean_index = 0
  teleop_index = 0
  while teleop_index < len(teleop_entries) and clean_index < len(clean_entries):
    next_clean_index = min(clean_index + clean_per_teleop, len(clean_entries))
    playlist.extend(clean_entries[clean_index:next_clean_index])
    clean_index = next_clean_index
    playlist.append(teleop_entries[teleop_index])
    teleop_index += 1

  playlist.extend(teleop_entries[teleop_index:])
  playlist.extend(clean_entries[clean_index:])
  return playlist


def build_playlist(
  clean_playlist: str = "data/vr_m3_1_full_unique_names.txt",
  teleop_dir: str = "data/vr_m3_1_npz_teleop",
  output: str = "data/vr_m3_1_teleop_robust_curriculum.txt",
  clean_per_teleop: int = 16,
  validate: bool = True,
) -> PlaylistBuildStats:
  """Build the VR M3.1 clean+teleop ordered curriculum playlist."""
  clean_playlist_path = Path(clean_playlist).expanduser()
  teleop_dir_path = Path(teleop_dir).expanduser()
  output_path = Path(output).expanduser()

  if not clean_playlist_path.is_file():
    raise FileNotFoundError(f"Clean playlist not found: {clean_playlist_path}")
  if not teleop_dir_path.is_dir():
    raise FileNotFoundError(f"Teleop motion directory not found: {teleop_dir_path}")

  output_base_dir = output_path.parent
  clean_entries = _read_clean_playlist(clean_playlist_path)
  teleop_paths = sorted(teleop_dir_path.rglob("*.npz"), key=_natural_key)
  teleop_entries = [_relative_entry(path, output_base_dir) for path in teleop_paths]

  playlist = build_playlist_lines(
    clean_entries,
    teleop_entries,
    clean_per_teleop=clean_per_teleop,
  )

  if validate:
    _validate_playlist_entries(playlist, output_base_dir)

  output_path.parent.mkdir(parents=True, exist_ok=True)
  output_path.write_text("\n".join(playlist) + ("\n" if playlist else ""))
  return PlaylistBuildStats(
    clean_count=len(clean_entries),
    teleop_count=len(teleop_entries),
    total_count=len(playlist),
    output_path=output_path,
  )


def main(
  clean_playlist: str = "data/vr_m3_1_full_unique_names.txt",
  teleop_dir: str = "data/vr_m3_1_npz_teleop",
  output: str = "data/vr_m3_1_teleop_robust_curriculum.txt",
  clean_per_teleop: int = 16,
  validate: bool = True,
) -> None:
  """Build a clean+teleop playlist for VR M3.1 from-scratch robust training."""
  stats = build_playlist(
    clean_playlist=clean_playlist,
    teleop_dir=teleop_dir,
    output=output,
    clean_per_teleop=clean_per_teleop,
    validate=validate,
  )
  print(
    "[INFO] Wrote "
    f"{stats.total_count} motions ({stats.clean_count} clean, "
    f"{stats.teleop_count} teleop) to {stats.output_path}"
  )


def cli() -> None:
  tyro.cli(main, config=mjlab.TYRO_FLAGS)


if __name__ == "__main__":
  cli()
