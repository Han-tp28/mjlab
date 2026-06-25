import json
import os
import statistics
import time
from pathlib import Path
from typing import cast

import torch
import wandb
from rsl_rl.env.vec_env import VecEnv
from rsl_rl.utils import check_nan
from torch import nn

from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import (
  attach_metadata_to_onnx,
  get_base_metadata,
)
from mjlab.rl.runner import MjlabOnPolicyRunner
from mjlab.tasks.tracking.mdp import MotionCommand


class _OnnxMotionModel(nn.Module):
  """ONNX-exportable model that wraps the policy and bundles motion reference data."""

  def __init__(self, actor, motion):
    super().__init__()
    self.policy = actor.as_onnx(verbose=False)
    self.register_buffer("joint_pos", motion.joint_pos.to("cpu"))
    self.register_buffer("joint_vel", motion.joint_vel.to("cpu"))
    self.register_buffer("body_pos_w", motion.body_pos_w.to("cpu"))
    self.register_buffer("body_quat_w", motion.body_quat_w.to("cpu"))
    self.register_buffer("body_lin_vel_w", motion.body_lin_vel_w.to("cpu"))
    self.register_buffer("body_ang_vel_w", motion.body_ang_vel_w.to("cpu"))
    self.time_step_total: int = self.joint_pos.shape[0]  # type: ignore[index]

  def forward(self, x, time_step):
    time_step_clamped = torch.clamp(
      time_step.long().squeeze(-1), max=self.time_step_total - 1
    )
    return (
      self.policy(x),
      self.joint_pos[time_step_clamped],  # type: ignore[index]
      self.joint_vel[time_step_clamped],  # type: ignore[index]
      self.body_pos_w[time_step_clamped],  # type: ignore[index]
      self.body_quat_w[time_step_clamped],  # type: ignore[index]
      self.body_lin_vel_w[time_step_clamped],  # type: ignore[index]
      self.body_ang_vel_w[time_step_clamped],  # type: ignore[index]
    )


class MotionTrackingOnPolicyRunner(MjlabOnPolicyRunner):
  env: RslRlVecEnvWrapper

  def __init__(
    self,
    env: VecEnv,
    train_cfg: dict,
    log_dir: str | None = None,
    device: str = "cpu",
    registry_name: str | None = None,
  ):
    super().__init__(env, train_cfg, log_dir, device)
    self.registry_name = registry_name
    self._motion_curriculum_gate_pass_start_iteration: int | None = None
    self._motion_curriculum_last_reload_iteration: int | None = None
    self._motion_local_done_counts: torch.Tensor | None = None
    self._motion_local_fail_counts: torch.Tensor | None = None
    self._motion_local_timeout_counts: torch.Tensor | None = None
    self._motion_local_term_counts: dict[str, torch.Tensor] = {}
    self._motion_outcome_counts: dict[str, dict[str, float | str]] = {}

  def _write_motion_manifest(self, iteration: int) -> None:
    if self.logger.log_dir is None:
      return

    motion_term = cast(
      MotionCommand, self.env.unwrapped.command_manager.get_term("motion")
    )
    manifest_dir = Path(self.logger.log_dir) / "motions"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    records = [
      {
        "index": index,
        "name": name,
        "file": file,
        "source_file": source_file,
      }
      for index, (name, file, source_file) in enumerate(
        zip(
          motion_term.motion.motion_names,
          motion_term.motion.motion_files,
          motion_term.motion.motion_source_files,
          strict=True,
        )
      )
    ]
    manifest = {
      "iteration": iteration,
      "source_motion_file": motion_term.motion.source_motion_file,
      "num_motions": motion_term.motion.num_motions,
      "num_seen_motions": motion_term.motion.num_seen_motions,
      "num_available_motions": motion_term.motion.num_available_motions,
      "motions": records,
    }

    latest_json = manifest_dir / "active_motion_pool_latest.json"
    latest_txt = manifest_dir / "active_motion_pool_latest.txt"
    iter_json = manifest_dir / f"active_motion_pool_iter_{iteration:06d}.json"
    latest_json.write_text(json.dumps(manifest, indent=2))
    iter_json.write_text(json.dumps(manifest, indent=2))
    latest_txt.write_text(
      "\n".join(
        f"{record['index']:06d}\t{record['name']}\t{record['source_file']}"
        for record in records
      )
      + "\n"
    )

  def _motion_curriculum_stats(self) -> dict[str, float]:
    motion_term = cast(
      MotionCommand, self.env.unwrapped.command_manager.get_term("motion")
    )
    stats = {
      name: float(value.mean().detach().cpu().item())
      for name, value in motion_term.metrics.items()
    }
    for name in (
      "error_body_pos",
      "error_body_rot",
      "error_anchor_pos",
      "error_anchor_rot",
      "error_joint_pos",
    ):
      logged_value = self._mean_logged_episode_extra(f"Metrics/motion/{name}")
      if logged_value is not None:
        stats[name] = logged_value

    stats["mean_episode_length"] = (
      float(statistics.mean(self.logger.lenbuffer))
      if len(self.logger.lenbuffer) > 0
      else 0.0
    )

    fall_keys = tuple(
      key.strip()
      for key in motion_term.cfg.motion_curriculum_fall_termination_keys.split(",")
      if key.strip()
    )
    stats["fall_rate"] = self._compute_fall_rate(self.logger.ep_extras, fall_keys)
    return stats

  @staticmethod
  def _compute_fall_rate(
    ep_extras: list[dict[str, torch.Tensor | float]],
    fall_termination_keys: tuple[str, ...],
  ) -> float:
    """Fraction of done episodes counted as a "fall" for curriculum gating.

    By default (``fall_termination_keys=()``) every non-timeout termination
    counts, which conflates literal instability with recoverable tracking
    misses (e.g. anchor_xy, ee_body_pos) already covered by the
    error_body_pos/error_anchor_pos gate checks. Restricting to the literal
    fall keys (e.g. fell_over_height/orientation) lets the gate pass on a
    policy that is stable but still imperfectly tracking a hard motion.
    """
    fall_term_names = (
      {f"Episode_Termination/{name}" for name in fall_termination_keys}
      if fall_termination_keys
      else None
    )
    time_out_count = 0.0
    failed_count = 0.0
    fall_count = 0.0
    for ep_info in ep_extras:
      for key, value in ep_info.items():
        if not key.startswith("Episode_Termination/"):
          continue
        count = (
          float(value.detach().sum().cpu().item())
          if isinstance(value, torch.Tensor)
          else float(value)
        )
        if key == "Episode_Termination/time_out":
          time_out_count += count
          continue
        failed_count += count
        if fall_term_names is None or key in fall_term_names:
          fall_count += count
    total_done_count = failed_count + time_out_count
    return fall_count / total_done_count if total_done_count > 0 else 0.0

  def _mean_logged_episode_extra(self, key: str) -> float | None:
    values = []
    for ep_info in self.logger.ep_extras:
      if key not in ep_info:
        continue
      value = ep_info[key]
      if isinstance(value, torch.Tensor):
        values.append(float(value.detach().float().mean().cpu().item()))
      else:
        values.append(float(value))
    if not values:
      return None
    return float(statistics.mean(values))

  @staticmethod
  def _format_motion_curriculum_stats(stats: dict[str, float]) -> str:
    keys = (
      "mean_episode_length",
      "fall_rate",
      "error_body_pos",
      "error_body_rot",
      "error_anchor_pos",
      "error_anchor_rot",
      "error_joint_pos",
    )
    return ", ".join(f"{key}={stats.get(key, 0.0):.4f}" for key in keys)

  def _ensure_motion_outcome_buffers(self, motion_term: MotionCommand) -> None:
    num_motions = motion_term.motion.num_motions
    if (
      self._motion_local_done_counts is not None
      and self._motion_local_done_counts.numel() == num_motions
    ):
      return

    device = motion_term.motion_ids.device
    with torch.inference_mode(False):
      self._motion_local_done_counts = torch.zeros(
        num_motions, device=device, dtype=torch.float
      )
      self._motion_local_fail_counts = torch.zeros_like(self._motion_local_done_counts)
      self._motion_local_timeout_counts = torch.zeros_like(
        self._motion_local_done_counts
      )
    self._motion_local_term_counts = {}

  def _record_motion_step_outcomes(
    self, motion_term: MotionCommand, motion_ids: torch.Tensor
  ) -> None:
    if self.logger.log_dir is None:
      return

    termination_manager = self.env.unwrapped.termination_manager
    terminated = self.env.unwrapped.reset_terminated.detach()
    time_outs = self.env.unwrapped.reset_time_outs.detach()
    dones = terminated | time_outs
    if not torch.any(dones):
      return

    self._ensure_motion_outcome_buffers(motion_term)
    assert self._motion_local_done_counts is not None
    assert self._motion_local_fail_counts is not None
    assert self._motion_local_timeout_counts is not None

    num_motions = motion_term.motion.num_motions
    motion_ids = torch.clamp(motion_ids.detach(), 0, num_motions - 1)
    self._motion_local_done_counts += torch.bincount(
      motion_ids[dones], minlength=num_motions
    )
    self._motion_local_fail_counts += torch.bincount(
      motion_ids[terminated], minlength=num_motions
    )
    self._motion_local_timeout_counts += torch.bincount(
      motion_ids[time_outs], minlength=num_motions
    )

    for term_name in termination_manager.active_terms:
      if termination_manager.get_term_cfg(term_name).time_out:
        continue
      term_dones = termination_manager.get_term(term_name).detach()
      if not torch.any(term_dones):
        continue
      if term_name not in self._motion_local_term_counts:
        with torch.inference_mode(False):
          self._motion_local_term_counts[term_name] = torch.zeros(
            num_motions, device=motion_ids.device, dtype=torch.float
          )
      self._motion_local_term_counts[term_name] += torch.bincount(
        motion_ids[term_dones], minlength=num_motions
      )

  def _flush_motion_outcome_counts(self, motion_term: MotionCommand) -> None:
    if self._motion_local_done_counts is None:
      return

    done_counts = self._motion_local_done_counts.detach().cpu()
    fail_counts = cast(torch.Tensor, self._motion_local_fail_counts).detach().cpu()
    timeout_counts = (
      cast(torch.Tensor, self._motion_local_timeout_counts).detach().cpu()
    )
    term_counts = {
      name: counts.detach().cpu()
      for name, counts in self._motion_local_term_counts.items()
    }

    nonzero_indices = torch.nonzero(done_counts, as_tuple=False).flatten().tolist()
    for index in nonzero_indices:
      source_file = motion_term.motion.motion_source_files[index]
      record = self._motion_outcome_counts.setdefault(
        source_file,
        {
          "name": motion_term.motion.motion_names[index],
          "done_count": 0.0,
          "fail_count": 0.0,
          "timeout_count": 0.0,
        },
      )
      record["done_count"] = float(record["done_count"]) + float(done_counts[index])
      record["fail_count"] = float(record["fail_count"]) + float(fail_counts[index])
      record["timeout_count"] = float(record["timeout_count"]) + float(
        timeout_counts[index]
      )
      for term_name, counts in term_counts.items():
        key = f"term:{term_name}"
        record[key] = float(record.get(key, 0.0)) + float(counts[index])

    self._motion_local_done_counts.zero_()
    cast(torch.Tensor, self._motion_local_fail_counts).zero_()
    cast(torch.Tensor, self._motion_local_timeout_counts).zero_()
    for counts in self._motion_local_term_counts.values():
      counts.zero_()

  def _update_replay_failure_weights(self, motion_term: MotionCommand) -> None:
    """Push lifetime per-motion failure rates into the loader for replay biasing.

    Uses the persistent ``_motion_outcome_counts`` (accumulated across all
    resamples and keyed by source file), so motions that have historically
    failed more are replayed more often even after they leave the active pool.
    """
    weights = {
      source_file: (
        float(record["fail_count"]) / float(record["done_count"])
        if float(record["done_count"]) > 0.0
        else 0.0
      )
      for source_file, record in self._motion_outcome_counts.items()
    }
    motion_term.motion.set_replay_failure_weights(weights)

  def _write_motion_failure_report(
    self, iteration: int, motion_term: MotionCommand | None = None
  ) -> None:
    if self.logger.log_dir is None:
      return
    if motion_term is None:
      motion_term = cast(
        MotionCommand, self.env.unwrapped.command_manager.get_term("motion")
      )

    self._flush_motion_outcome_counts(motion_term)
    motion_dir = Path(self.logger.log_dir) / "motions"
    motion_dir.mkdir(parents=True, exist_ok=True)

    term_names = sorted(
      {
        key.removeprefix("term:")
        for record in self._motion_outcome_counts.values()
        for key in record
        if key.startswith("term:")
      }
    )
    rows = sorted(
      self._motion_outcome_counts.items(),
      key=lambda item: (
        -float(item[1].get("fail_count", 0.0)),
        -float(item[1].get("done_count", 0.0)),
        str(item[1].get("name", "")),
      ),
    )

    latest_path = motion_dir / "motion_failure_report_latest.tsv"
    iter_path = motion_dir / f"motion_failure_report_iter_{iteration:06d}.tsv"
    header = [
      "iteration",
      "fail_count",
      "timeout_count",
      "done_count",
      "fail_rate",
      "top_fail_term",
      "top_fail_term_count",
      "motion_name",
      "source_file",
      *[f"term:{name}" for name in term_names],
    ]
    lines = ["	".join(header)]
    for source_file, record in rows:
      done_count = float(record.get("done_count", 0.0))
      fail_count = float(record.get("fail_count", 0.0))
      timeout_count = float(record.get("timeout_count", 0.0))
      term_values = {
        name: float(record.get(f"term:{name}", 0.0)) for name in term_names
      }
      top_term, top_term_count = (
        max(term_values.items(), key=lambda item: item[1]) if term_values else ("", 0.0)
      )
      fail_rate = fail_count / done_count if done_count > 0.0 else 0.0
      row = [
        str(iteration),
        f"{fail_count:.0f}",
        f"{timeout_count:.0f}",
        f"{done_count:.0f}",
        f"{fail_rate:.6f}",
        top_term if top_term_count > 0.0 else "",
        f"{top_term_count:.0f}",
        str(record.get("name", "")),
        source_file,
        *[f"{term_values[name]:.0f}" for name in term_names],
      ]
      lines.append("	".join(row))

    text = "\n".join(lines) + "\n"
    latest_path.write_text(text)
    iter_path.write_text(text)

  def _next_motion_curriculum_target(self, motion_term: MotionCommand) -> int | None:
    cfg = motion_term.cfg
    max_num_motions = (
      cfg.max_num_load_motions or motion_term.motion.num_available_motions
    )
    max_num_motions = min(max_num_motions, motion_term.motion.num_available_motions)
    if cfg.motion_pool_mode == "streaming":
      if cfg.num_new_motions_per_resample is None:
        return None
      if (
        motion_term.motion.num_seen_motions >= motion_term.motion.num_available_motions
      ):
        return None
      return min(max_num_motions, motion_term.motion.num_available_motions)

    if motion_term.motion.num_motions >= max_num_motions:
      return None

    for stage_size in cfg.motion_curriculum_stage_sizes:
      stage_size = min(stage_size, max_num_motions)
      if stage_size > motion_term.motion.num_motions:
        return stage_size

    if cfg.num_new_motions_per_resample is None:
      return None
    return min(
      motion_term.motion.num_motions + cfg.num_new_motions_per_resample,
      max_num_motions,
    )

  def _motion_curriculum_gate_passed(
    self, motion_term: MotionCommand, stats: dict[str, float]
  ) -> tuple[bool, list[str]]:
    cfg = motion_term.cfg
    if not cfg.motion_curriculum_gate:
      return True, []

    checks = (
      (
        "mean_episode_length",
        cfg.motion_curriculum_min_mean_episode_length,
        ">=",
      ),
      ("fall_rate", cfg.motion_curriculum_max_fall_rate, "<="),
      ("error_body_pos", cfg.motion_curriculum_max_body_pos_error, "<="),
      ("error_body_rot", cfg.motion_curriculum_max_body_rot_error, "<="),
      ("error_anchor_pos", cfg.motion_curriculum_max_anchor_pos_error, "<="),
      ("error_anchor_rot", cfg.motion_curriculum_max_anchor_rot_error, "<="),
      ("error_joint_pos", cfg.motion_curriculum_max_joint_pos_error, "<="),
    )

    failures = []
    for key, threshold, direction in checks:
      if threshold is None:
        continue
      value = stats.get(key, 0.0)
      passed = value >= threshold if direction == ">=" else value <= threshold
      if not passed:
        failures.append(f"{key} {value:.4f} {direction} {threshold:.4f}")
    return len(failures) == 0, failures

  def _maybe_resample_motion_library(
    self, iteration: int, curriculum_stats: dict[str, float]
  ):
    motion_term = cast(
      MotionCommand, self.env.unwrapped.command_manager.get_term("motion")
    )
    target_num_motions = self._next_motion_curriculum_target(motion_term)
    if target_num_motions is None:
      self._motion_curriculum_gate_pass_start_iteration = None
      self._motion_curriculum_last_reload_iteration = None
      return None

    force_after_iterations = motion_term.cfg.motion_curriculum_force_after_iterations
    if self._motion_curriculum_last_reload_iteration is None:
      if force_after_iterations is None or force_after_iterations <= 0:
        self._motion_curriculum_last_reload_iteration = iteration + 1
      else:
        current_iteration = iteration + 1
        self._motion_curriculum_last_reload_iteration = (
          current_iteration // force_after_iterations
        ) * force_after_iterations

    passed, failures = self._motion_curriculum_gate_passed(
      motion_term, curriculum_stats
    )
    if passed:
      if self._motion_curriculum_gate_pass_start_iteration is None:
        self._motion_curriculum_gate_pass_start_iteration = iteration + 1
    else:
      self._motion_curriculum_gate_pass_start_iteration = None

    stable_iterations = 0
    if self._motion_curriculum_gate_pass_start_iteration is not None:
      stable_iterations = (
        iteration + 1 - self._motion_curriculum_gate_pass_start_iteration + 1
      )

    interval = motion_term.cfg.motion_resample_interval
    if interval is None or interval <= 0:
      return None
    if iteration + 1 < motion_term.cfg.motion_resample_start_iteration:
      return None
    if (iteration + 1) % interval != 0:
      return None

    iterations_since_reload = (
      iteration + 1 - self._motion_curriculum_last_reload_iteration
    )
    force_reload = (
      force_after_iterations is not None
      and iterations_since_reload >= force_after_iterations
    )

    min_stable_iterations = motion_term.cfg.motion_curriculum_min_stable_iterations
    gate_ready = passed and stable_iterations >= min_stable_iterations

    if not gate_ready and not force_reload:
      if not passed:
        print(
          "[INFO] Holding motion curriculum at iteration "
          f"{iteration + 1}: active {motion_term.motion.num_motions}/"
          f"{motion_term.motion.num_available_motions}; "
          f"no reload for {iterations_since_reload}/"
          f"{force_after_iterations} iterations; "
          f"stats: {self._format_motion_curriculum_stats(curriculum_stats)}; "
          f"waiting for {', '.join(failures)}"
        )
        return None
      print(
        "[INFO] Holding motion curriculum at iteration "
        f"{iteration + 1}: active {motion_term.motion.num_motions}/"
        f"{motion_term.motion.num_available_motions}; "
        f"gate stable for {stable_iterations}/{min_stable_iterations} iterations; "
        f"no reload for {iterations_since_reload}/"
        f"{force_after_iterations} iterations; "
        f"stats: {self._format_motion_curriculum_stats(curriculum_stats)}"
      )
      return None

    if force_reload and not gate_ready:
      reason = (
        "gate passed but stable window not reached"
        if passed
        else f"waiting for {', '.join(failures)}"
      )
      print(
        "[INFO] Forcing motion curriculum update at iteration "
        f"{iteration + 1}: active {motion_term.motion.num_motions}/"
        f"{motion_term.motion.num_available_motions}; "
        f"no reload for {iterations_since_reload}/"
        f"{force_after_iterations} iterations; {reason}; "
        f"stats: {self._format_motion_curriculum_stats(curriculum_stats)}"
      )

    self._write_motion_failure_report(iteration + 1, motion_term)

    if motion_term.cfg.motion_pool_mode == "grow":
      motion_term.cfg.num_new_motions_per_resample = (
        target_num_motions - motion_term.motion.num_motions
      )

    if motion_term.cfg.motion_replay_failure_weighted:
      self._update_replay_failure_weights(motion_term)

    if not motion_term.reload_motion_library():
      return None
    self._motion_curriculum_gate_pass_start_iteration = None
    self._motion_curriculum_last_reload_iteration = iteration + 1

    print(
      "[INFO] Updated motion replay library at iteration "
      f"{iteration + 1}: loaded {motion_term.motion.num_motions}/"
      f"{motion_term.motion.num_available_motions} active motions, "
      f"seen {motion_term.motion.num_seen_motions}/"
      f"{motion_term.motion.num_available_motions}; "
      f"gate stats: {self._format_motion_curriculum_stats(curriculum_stats)}"
    )
    self._write_motion_manifest(iteration + 1)
    with torch.inference_mode():
      obs, _ = self.env.reset()
    return obs.to(self.device)

  def learn(
    self, num_learning_iterations: int, init_at_random_ep_len: bool = False
  ) -> None:
    """Run learning with optional Groot-style periodic motion library resampling."""
    if init_at_random_ep_len:
      self.env.episode_length_buf = torch.randint_like(
        self.env.episode_length_buf, high=int(self.env.max_episode_length)
      )

    obs = self.env.get_observations().to(self.device)
    self.alg.train_mode()

    if self.is_distributed:
      print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
      self.alg.broadcast_parameters()

    start_it = self.current_learning_iteration
    self.logger.init_logging_writer()
    self._write_motion_manifest(start_it)

    total_it = start_it + num_learning_iterations
    for it in range(start_it, total_it):
      start = time.time()
      with torch.inference_mode():
        motion_term = cast(
          MotionCommand, self.env.unwrapped.command_manager.get_term("motion")
        )
        for _ in range(self.cfg["num_steps_per_env"]):
          motion_ids = motion_term.motion_ids.detach().clone()
          actions = self.alg.act(obs)
          obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
          self._record_motion_step_outcomes(motion_term, motion_ids)
          if self.cfg.get("check_for_nan", True):
            check_nan(obs, rewards, dones)
          obs, rewards, dones = (
            obs.to(self.device),
            rewards.to(self.device),
            dones.to(self.device),
          )
          self.alg.process_env_step(obs, rewards, dones, extras)
          intrinsic_rewards = (
            self.alg.intrinsic_rewards if self.cfg["algorithm"]["rnd_cfg"] else None
          )
          self.logger.process_env_step(rewards, dones, extras, intrinsic_rewards)

        stop = time.time()
        collect_time = stop - start
        start = stop
        self.alg.compute_returns(obs)

      loss_dict = self.alg.update()

      stop = time.time()
      learn_time = stop - start
      self.current_learning_iteration = it
      curriculum_stats = self._motion_curriculum_stats()
      rnd_weight = None
      if self.cfg["algorithm"]["rnd_cfg"]:
        rnd_weight = getattr(self.alg.rnd, "weight", None)

      self.logger.log(
        it=it,
        start_it=start_it,
        total_it=total_it,
        collect_time=collect_time,
        learn_time=learn_time,
        loss_dict=loss_dict,
        learning_rate=self.alg.learning_rate,
        action_std=self.alg.get_policy().output_std,
        rnd_weight=rnd_weight,
      )

      if self.logger.writer is not None and it % 100 == 0:
        self._write_motion_failure_report(it, motion_term)

      if self.logger.writer is not None and it % self.cfg["save_interval"] == 0:
        assert self.logger.log_dir is not None
        self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))

      resampled_obs = self._maybe_resample_motion_library(it, curriculum_stats)
      if resampled_obs is not None:
        obs = resampled_obs

    if self.logger.writer is not None:
      assert self.logger.log_dir is not None
      motion_term = cast(
        MotionCommand, self.env.unwrapped.command_manager.get_term("motion")
      )
      self._write_motion_failure_report(self.current_learning_iteration, motion_term)
      self.save(
        os.path.join(self.logger.log_dir, f"model_{self.current_learning_iteration}.pt")
      )
      self.logger.stop_logging_writer()

  def export_policy_to_onnx(
    self, path: str, filename: str = "policy.onnx", verbose: bool = False
  ) -> None:
    os.makedirs(path, exist_ok=True)
    cmd = cast(MotionCommand, self.env.unwrapped.command_manager.get_term("motion"))
    model = _OnnxMotionModel(self.alg.get_policy(), cmd.motion)
    model.to("cpu")
    model.eval()
    obs = torch.zeros(1, model.policy.input_size)
    time_step = torch.zeros(1, 1)
    torch.onnx.export(
      model,
      (obs, time_step),
      os.path.join(path, filename),
      export_params=True,
      opset_version=18,
      verbose=verbose,
      input_names=["obs", "time_step"],
      output_names=[
        "actions",
        "joint_pos",
        "joint_vel",
        "body_pos_w",
        "body_quat_w",
        "body_lin_vel_w",
        "body_ang_vel_w",
      ],
      dynamic_axes={},
      dynamo=False,
    )

  def save(self, path: str, infos=None):
    super().save(path, infos)
    policy_dir, filename, onnx_path = self._get_export_paths(path)
    try:
      self.export_policy_to_onnx(str(policy_dir), filename)
      run_name: str = (
        wandb.run.name if self.logger.logger_type == "wandb" and wandb.run else "local"
      )  # type: ignore[assignment]
      metadata = get_base_metadata(self.env.unwrapped, run_name)
      motion_term = cast(
        MotionCommand, self.env.unwrapped.command_manager.get_term("motion")
      )
      metadata.update(
        {
          "anchor_body_name": motion_term.cfg.anchor_body_name,
          "body_names": list(motion_term.cfg.body_names),
        }
      )
      attach_metadata_to_onnx(str(onnx_path), metadata)
      if self.logger.logger_type in ["wandb"] and self.cfg["upload_model"]:
        wandb.save(str(onnx_path), base_path=str(policy_dir))
        if self.registry_name is not None:
          wandb.run.use_artifact(self.registry_name)  # type: ignore
          self.registry_name = None
    except Exception as e:
      print(f"[WARN] ONNX export failed (training continues): {e}")
