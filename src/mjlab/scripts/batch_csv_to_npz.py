"""Batch-convert CSV motion files to NPZ motion libraries."""

from __future__ import annotations

import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path

import tyro

import mjlab
from mjlab.scripts import csv_to_npz


def _safe_output_stem(csv_file: Path, input_dir: Path) -> str:
  relative = csv_file.relative_to(input_dir).with_suffix("")
  return "__".join(relative.parts)


@dataclass(frozen=True)
class _ConvertJob:
  csv_file: str
  output_file: str
  input_fps: float
  output_fps: float
  robot: csv_to_npz.RobotName
  device: str
  render: bool
  upload_wandb: bool


def _convert_one(job: _ConvertJob) -> tuple[str, str | None]:
  try:
    csv_to_npz.main(
      input_file=job.csv_file,
      output_name=job.output_file,
      input_fps=job.input_fps,
      output_fps=job.output_fps,
      device=job.device,
      render=job.render,
      robot=job.robot,
      upload_wandb=job.upload_wandb,
    )
  except Exception:
    return job.csv_file, traceback.format_exc()
  return job.csv_file, None


def main(
  input_dir: str,
  output_dir: str,
  input_fps: float = 120.0,
  output_fps: float = 50.0,
  robot: csv_to_npz.RobotName = "vr_h3_1",
  device: str = "cuda:0",
  render: bool = False,
  upload_wandb: bool = False,
  skip_existing: bool = True,
  quiet_skips: bool = True,
  continue_on_error: bool = True,
  num_workers: int = 1,
  max_tasks_per_child: int = 8,
  glob: str = "*.csv",
  limit: int | None = None,
) -> None:
  """Convert all matching CSV files under a directory into NPZ files.

  Args:
    input_dir: Directory containing raw CSV motions.
    output_dir: Directory to write NPZ motions into.
    input_fps: Frame rate of the source CSV files.
    output_fps: Frame rate of the generated NPZ files.
    robot: Robot model used for replay and body ordering.
    device: Device used by MuJoCo/Warp during conversion.
    render: Whether to render every converted motion.
    upload_wandb: Whether to upload generated motions to W&B.
    skip_existing: Skip output files that already exist.
    quiet_skips: Do not print every skipped file when resuming a large batch.
    continue_on_error: Keep converting when one CSV fails; print failures at the end.
    num_workers: Number of converter worker processes. Use 1 for safest GPU behavior.
    max_tasks_per_child: Restart each worker after this many conversions. Use 1 to
      aggressively cap MuJoCo/Warp native memory growth, or 0 to disable recycling.
    glob: File glob to search recursively under input_dir.
    limit: Optional maximum number of CSVs to convert.
  """
  if num_workers < 1:
    raise ValueError("`num_workers` must be >= 1.")
  if max_tasks_per_child < 0:
    raise ValueError("`max_tasks_per_child` must be >= 0.")
  if render and num_workers > 1:
    raise ValueError("Parallel conversion with rendering is not supported.")

  input_path = Path(input_dir).expanduser().resolve()
  output_path = Path(output_dir).expanduser().resolve()

  if not input_path.is_dir():
    raise FileNotFoundError(f"Input directory not found: {input_path}")

  csv_files = sorted(input_path.rglob(glob))
  if limit is not None:
    csv_files = csv_files[:limit]
  if not csv_files:
    raise FileNotFoundError(f"No files matching {glob!r} found in {input_path}")

  output_path.mkdir(parents=True, exist_ok=True)

  print(
    f"[INFO] Converting {len(csv_files)} CSV files from {input_path} to {output_path}"
  )
  print(
    f"[INFO] robot={robot}, input_fps={input_fps}, output_fps={output_fps}, "
    f"device={device}"
  )

  converted = 0
  skipped = 0
  failures: list[tuple[str, str]] = []
  jobs: list[tuple[int, _ConvertJob]] = []
  for index, csv_file in enumerate(csv_files, start=1):
    output_file = output_path / f"{_safe_output_stem(csv_file, input_path)}.npz"
    if skip_existing and output_file.exists():
      skipped += 1
      if not quiet_skips:
        print(f"[{index}/{len(csv_files)}] Skip existing: {output_file.name}")
      continue

    jobs.append(
      (
        index,
        _ConvertJob(
          csv_file=str(csv_file),
          output_file=str(output_file),
          input_fps=input_fps,
          output_fps=output_fps,
          robot=robot,
          device=device,
          render=render,
          upload_wandb=upload_wandb,
        ),
      )
    )

  if skipped:
    print(f"[INFO] Skipped {skipped} existing output files.")
  print(f"[INFO] Remaining jobs: {len(jobs)}")

  if num_workers == 1 and max_tasks_per_child == 0:
    for index, job in jobs:
      print(f"[{index}/{len(csv_files)}] Convert: {job.csv_file}")
      csv_file, error = _convert_one(job)
      if error is None:
        converted += 1
      else:
        failures.append((csv_file, error))
        print(f"[ERROR] Failed: {csv_file}\n{error}")
        if not continue_on_error:
          break
  else:
    if max_tasks_per_child == 0:
      print(f"[INFO] Running with {num_workers} worker processes.")
      tasks_per_child = None
    else:
      print(
        f"[INFO] Running with {num_workers} worker processes; recycling each "
        f"worker after {max_tasks_per_child} task(s)."
      )
      tasks_per_child = max_tasks_per_child
    ctx = get_context("spawn")
    with ProcessPoolExecutor(
      max_workers=num_workers,
      mp_context=ctx,
      max_tasks_per_child=tasks_per_child,
    ) as executor:
      future_to_job = {
        executor.submit(_convert_one, job): (index, job) for index, job in jobs
      }
      for future in as_completed(future_to_job):
        index, job = future_to_job[future]
        try:
          csv_file, error = future.result()
        except Exception:
          csv_file = job.csv_file
          error = traceback.format_exc()
        if error is None:
          converted += 1
          print(f"[{index}/{len(csv_files)}] Done: {Path(job.output_file).name}")
        else:
          failures.append((csv_file, error))
          print(f"[ERROR] Failed: {csv_file}\n{error}")
          if not continue_on_error:
            executor.shutdown(cancel_futures=True)
            break

  if failures:
    failed_log = output_path / "failed_conversions.txt"
    failed_log.write_text(
      "\n\n".join(f"{csv_file}\n{error}" for csv_file, error in failures)
    )
    print(f"[WARN] Failed conversions: {len(failures)}. See {failed_log}")

  print(
    f"[INFO] Done. converted={converted}, skipped={skipped}, failed={len(failures)}"
  )
  print(f"[INFO] Motion library: {output_path}")


def cli() -> None:
  tyro.cli(main, config=mjlab.TYRO_FLAGS)


if __name__ == "__main__":
  cli()
