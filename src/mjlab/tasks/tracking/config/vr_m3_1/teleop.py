"""Teleoperate the VR-M3.1 pose policy from live VR head/hand poses.

Drives the trained multi-encoder pose policy through its ``teleop`` encoder: the
three VR points (headset + two controllers) are retargeted to the
``teleop_3point_*`` observation targets each control step via
``MotionCommand.set_live_teleop``. The lower body and balance are produced by the
policy. See ``mjlab.tasks.tracking.teleop`` for the source/retarget components.

Example (no hardware needed)::

  uv run python -m mjlab.tasks.tracking.config.vr_m3_1.teleop \\
    --source mock \\
    --checkpoint-file logs/rsl_rl/vr_m3_1_pose/<run>/model_55000.pt \\
    --motion-file data/<some_motion>.npz
"""

from __future__ import annotations

import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.config.vr_m3_1.env_cfgs import POSE_BODY_NAMES
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.mdp.commands import MotionCommand
from mjlab.tasks.tracking.teleop import (
  TELEOP_BODY_NAMES,
  MockPoseSource,
  OpenXrPoseSource,
  PoseSource,
  ReplayPoseSource,
  TeleopRetargeter,
)
from mjlab.utils.lab_api.math import quat_apply_inverse, quat_inv, quat_mul
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

TASK_ID = "Mjlab-Tracking-Flat-VR-M3-1-Pose"


@dataclass(frozen=True)
class TeleopConfig:
  checkpoint_file: str | None = None
  """Path to a trained checkpoint (.pt). Required."""
  wandb_run_path: str | None = None
  motion_file: str | None = None
  """Background motion for the loader (the teleop targets override it). The
  pose config's default playlist is used when omitted, if present locally."""
  source: Literal["openxr", "mock", "replay"] = "mock"
  device: str | None = None
  viewer: Literal["auto", "native", "viser"] = "auto"
  turn: bool = True
  """Let the robot turn its base to follow the headset heading."""
  log_root: str = "logs/rsl_rl"


def _build_retargeter(command: MotionCommand) -> TeleopRetargeter:
  """Extract the robot's neutral 3-point pose (in the pelvis frame) + default
  joint targets and build a retargeter."""
  robot = command.robot
  data = robot.data
  anchor_idx = command.robot_anchor_body_index
  anchor_pos = data.body_link_pos_w[0, anchor_idx]
  anchor_quat = data.body_link_quat_w[0, anchor_idx]
  anchor_quat_inv = quat_inv(anchor_quat)

  neutral_pos, neutral_orn = [], []
  for name in TELEOP_BODY_NAMES:
    bi = robot.body_names.index(name)
    pos = data.body_link_pos_w[0, bi]
    quat = data.body_link_quat_w[0, bi]
    neutral_pos.append(quat_apply_inverse(anchor_quat, pos - anchor_pos))
    neutral_orn.append(quat_mul(anchor_quat_inv, quat))

  return TeleopRetargeter(
    neutral_pos_local=torch.stack(neutral_pos).cpu(),
    neutral_orn_local=torch.stack(neutral_orn).cpu(),
    default_joint_pos=data.default_joint_pos[0].cpu(),
  )


def _make_source(cfg: TeleopConfig, dt: float) -> PoseSource:
  if cfg.source == "mock":
    return MockPoseSource(dt=dt)
  if cfg.source == "replay":
    if cfg.motion_file is None:
      raise ValueError("--source replay requires --motion-file")
    return ReplayPoseSource(cfg.motion_file, POSE_BODY_NAMES)
  return OpenXrPoseSource()


def run_teleop(cfg: TeleopConfig) -> None:
  configure_torch_backends()
  if cfg.checkpoint_file is None:
    raise ValueError("--checkpoint-file is required.")
  resume_path = Path(cfg.checkpoint_file)
  if not resume_path.exists():
    raise FileNotFoundError(f"Checkpoint not found: {resume_path}")

  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  env_cfg = load_env_cfg(TASK_ID, play=True)
  agent_cfg = load_rl_cfg(TASK_ID)

  motion_cmd = env_cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  if cfg.motion_file is not None:
    motion_cmd.motion_file = cfg.motion_file
  motion_cmd.forced_encoder_source = "teleop"
  env_cfg.scene.num_envs = 1
  env_cfg.terminations = {}  # never reset/teleport during teleop

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  runner_cls = load_runner_cls(TASK_ID) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=device)
  runner.load(
    str(resume_path), load_cfg={"actor": True}, strict=True, map_location=device
  )
  base_policy = runner.get_inference_policy(device=device)

  base_env = env.unwrapped
  assert isinstance(base_env, ManagerBasedRlEnv)
  env.reset()
  command = base_env.command_manager.get_term("motion")
  assert isinstance(command, MotionCommand)
  retargeter = _build_retargeter(command)
  source = _make_source(cfg, dt=base_env.step_dt)
  num_envs = base_env.num_envs
  print(
    f"[INFO] Teleop ready (source={cfg.source}, device={device}). Calibrating "
    "from the first frame; hold a neutral standing pose."
  )

  def teleop_policy(obs):
    frame = source.poll()
    if frame.reset_calib:
      retargeter.calibrate(frame)
    tgt = retargeter.retarget(frame)
    command.set_live_teleop(
      pos_local=tgt.pos_local.to(device).unsqueeze(0).expand(num_envs, -1, -1),
      orn_local_quat=tgt.orn_local_quat.to(device)
      .unsqueeze(0)
      .expand(num_envs, -1, -1),
      joint_pos=tgt.joint_pos.to(device).unsqueeze(0).expand(num_envs, -1),
      root_yaw=tgt.root_yaw.to(device).reshape(1).expand(num_envs)
      if cfg.turn
      else None,
    )
    return base_policy(obs)

  if cfg.viewer == "auto":
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    resolved_viewer = "native" if has_display else "viser"
  else:
    resolved_viewer = cfg.viewer

  try:
    if resolved_viewer == "native":
      NativeMujocoViewer(env, teleop_policy).run()
    else:
      ViserPlayViewer(env, teleop_policy).run()
  finally:
    source.close()
    env.close()


def main() -> None:
  import mjlab.tasks  # noqa: F401  (populate task registry)

  cfg = tyro.cli(TeleopConfig, prog=sys.argv[0], config=mjlab.TYRO_FLAGS)
  run_teleop(cfg)


if __name__ == "__main__":
  main()
