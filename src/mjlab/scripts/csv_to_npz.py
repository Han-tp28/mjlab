import gc
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import tyro
from tqdm import tqdm

import mjlab
from mjlab.entity import Entity
from mjlab.scene import Scene
from mjlab.sim.sim import Simulation, SimulationCfg
from mjlab.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg
from mjlab.tasks.tracking.config.vr_h3_1.env_cfgs import vr_h3_1_flat_tracking_env_cfg
from mjlab.tasks.tracking.config.vr_m3_1.env_cfgs import vr_m3_1_flat_tracking_env_cfg
from mjlab.utils.lab_api.math import (
  axis_angle_from_quat,
  quat_conjugate,
  quat_mul,
  quat_slerp,
)
from mjlab.viewer.offscreen_renderer import OffscreenRenderer
from mjlab.viewer.viewer_config import ViewerConfig

_G1_WRIST_YAW_JOINTS = frozenset(("left_wrist_yaw_joint", "right_wrist_yaw_joint"))

G1_JOINT_NAMES = (
  "left_hip_pitch_joint",
  "left_hip_roll_joint",
  "left_hip_yaw_joint",
  "left_knee_joint",
  "left_ankle_pitch_joint",
  "left_ankle_roll_joint",
  "right_hip_pitch_joint",
  "right_hip_roll_joint",
  "right_hip_yaw_joint",
  "right_knee_joint",
  "right_ankle_pitch_joint",
  "right_ankle_roll_joint",
  "waist_yaw_joint",
  "waist_roll_joint",
  "waist_pitch_joint",
  "left_shoulder_pitch_joint",
  "left_shoulder_roll_joint",
  "left_shoulder_yaw_joint",
  "left_elbow_joint",
  "left_wrist_roll_joint",
  "left_wrist_pitch_joint",
  "left_wrist_yaw_joint",
  "right_shoulder_pitch_joint",
  "right_shoulder_roll_joint",
  "right_shoulder_yaw_joint",
  "right_elbow_joint",
  "right_wrist_roll_joint",
  "right_wrist_pitch_joint",
  "right_wrist_yaw_joint",
)

VR_H3_1_JOINT_NAMES = (
  "left_hip_pitch_joint",
  "left_hip_roll_joint",
  "left_hip_yaw_joint",
  "left_knee_pitch_joint",
  "left_ankle_pitch_joint",
  "left_ankle_roll_joint",
  "right_hip_pitch_joint",
  "right_hip_roll_joint",
  "right_hip_yaw_joint",
  "right_knee_pitch_joint",
  "right_ankle_pitch_joint",
  "right_ankle_roll_joint",
  "waist_yaw_joint",
  "left_shoulder_pitch_joint",
  "left_shoulder_roll_joint",
  "left_shoulder_yaw_joint",
  "left_elbow_pitch_joint",
  "left_wrist_yaw_joint",
  "left_wrist_roll_joint",
  "left_wrist_pitch_joint",
  "right_shoulder_pitch_joint",
  "right_shoulder_roll_joint",
  "right_shoulder_yaw_joint",
  "right_elbow_pitch_joint",
  "right_wrist_yaw_joint",
  "right_wrist_roll_joint",
  "right_wrist_pitch_joint",
)
VR_H3_1_27_JOINT_NAMES = VR_H3_1_JOINT_NAMES
VR_H3_1_28_SOURCE_JOINT_NAMES = (
  "left_hip_pitch_joint",
  "left_hip_roll_joint",
  "left_hip_yaw_joint",
  "left_knee_pitch_joint",
  "left_ankle_pitch_joint",
  "left_ankle_roll_joint",
  "right_hip_pitch_joint",
  "right_hip_roll_joint",
  "right_hip_yaw_joint",
  "right_knee_pitch_joint",
  "right_ankle_pitch_joint",
  "right_ankle_roll_joint",
  "waist_yaw_joint",
  "waist_roll_joint",
  "left_shoulder_pitch_joint",
  "left_shoulder_roll_joint",
  "left_shoulder_yaw_joint",
  "left_elbow_pitch_joint",
  "left_wrist_yaw_joint",
  "left_wrist_roll_joint",
  "left_wrist_pitch_joint",
  "right_shoulder_pitch_joint",
  "right_shoulder_roll_joint",
  "right_shoulder_yaw_joint",
  "right_elbow_pitch_joint",
  "right_wrist_yaw_joint",
  "right_wrist_roll_joint",
  "right_wrist_pitch_joint",
)
_VR_H3_1_DROPPED_SOURCE_JOINTS = frozenset(("waist_roll_joint",))

VR_M3_1_27_JOINT_NAMES = VR_H3_1_JOINT_NAMES
VR_M3_1_JOINT_NAMES = (
  *VR_M3_1_27_JOINT_NAMES,
  "head_yaw_joint",
  "head_pitch_joint",
)
VR_M3_1_28_SOURCE_JOINT_NAMES = VR_H3_1_28_SOURCE_JOINT_NAMES
VR_M3_1_30_SOURCE_JOINT_NAMES = (
  *VR_M3_1_28_SOURCE_JOINT_NAMES,
  "head_yaw_joint",
  "head_pitch_joint",
)
_VR_M3_1_DROPPED_SOURCE_JOINTS = frozenset(("waist_roll_joint",))

RobotName = Literal["g1", "vr_h3_1", "vr_m3_1"]


class MotionLoader:
  def __init__(
    self,
    motion_file: str,
    input_fps: int,
    output_fps: int,
    device: torch.device | str,
    line_range: tuple[int, int] | None = None,
  ):
    self.motion_file = motion_file
    self.input_fps = input_fps
    self.output_fps = output_fps
    self.input_dt = 1.0 / self.input_fps
    self.output_dt = 1.0 / self.output_fps
    self.current_idx = 0
    self.device = device
    self.line_range = line_range
    self._load_motion()
    self._interpolate_motion()
    self._compute_velocities()

  def _load_motion(self):
    """Loads the motion from the csv file."""
    if self.line_range is None:
      motion = torch.from_numpy(np.loadtxt(self.motion_file, delimiter=","))
    else:
      motion = torch.from_numpy(
        np.loadtxt(
          self.motion_file,
          delimiter=",",
          skiprows=self.line_range[0] - 1,
          max_rows=self.line_range[1] - self.line_range[0] + 1,
        )
      )
    motion = motion.to(torch.float32).to(self.device)
    # motion[:, 2] -= 0.05
    self.motion_base_poss_input = motion[:, :3]
    self.motion_base_rots_input = motion[:, 3:7]
    self.motion_base_rots_input = self.motion_base_rots_input[
      :, [3, 0, 1, 2]
    ]  # convert to wxyz
    self.motion_dof_poss_input = motion[:, 7:]

    self.input_frames = motion.shape[0]
    self.duration = (self.input_frames - 1) * self.input_dt

  def _interpolate_motion(self):
    """Interpolates the motion to the output fps."""
    times = torch.arange(
      0, self.duration, self.output_dt, device=self.device, dtype=torch.float32
    )
    self.output_frames = times.shape[0]
    index_0, index_1, blend = self._compute_frame_blend(times)
    self.motion_base_poss = self._lerp(
      self.motion_base_poss_input[index_0],
      self.motion_base_poss_input[index_1],
      blend.unsqueeze(1),
    )
    self.motion_base_rots = self._slerp(
      self.motion_base_rots_input[index_0],
      self.motion_base_rots_input[index_1],
      blend,
    )
    self.motion_dof_poss = self._lerp(
      self.motion_dof_poss_input[index_0],
      self.motion_dof_poss_input[index_1],
      blend.unsqueeze(1),
    )
    print(
      f"Motion interpolated, input frames: {self.input_frames}, "
      f"input fps: {self.input_fps}, "
      f"output frames: {self.output_frames}, "
      f"output fps: {self.output_fps}"
    )

  def _lerp(
    self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor
  ) -> torch.Tensor:
    """Linear interpolation between two tensors."""
    return a * (1 - blend) + b * blend

  def _slerp(
    self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor
  ) -> torch.Tensor:
    """Spherical linear interpolation between two quaternions."""
    slerped_quats = torch.zeros_like(a)
    for i in range(a.shape[0]):
      slerped_quats[i] = quat_slerp(a[i], b[i], float(blend[i]))
    return slerped_quats

  def _compute_frame_blend(
    self, times: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Computes the frame blend for the motion."""
    phase = times / self.duration
    index_0 = (phase * (self.input_frames - 1)).floor().long()
    index_1 = torch.minimum(index_0 + 1, torch.tensor(self.input_frames - 1))
    blend = phase * (self.input_frames - 1) - index_0
    return index_0, index_1, blend

  def _compute_velocities(self):
    """Computes the velocities of the motion."""
    self.motion_base_lin_vels = torch.gradient(
      self.motion_base_poss, spacing=self.output_dt, dim=0
    )[0]
    self.motion_dof_vels = torch.gradient(
      self.motion_dof_poss, spacing=self.output_dt, dim=0
    )[0]
    self.motion_base_ang_vels = self._so3_derivative(
      self.motion_base_rots, self.output_dt
    )

  def _so3_derivative(self, rotations: torch.Tensor, dt: float) -> torch.Tensor:
    """Computes the derivative of a sequence of SO3 rotations.

    Args:
      rotations: shape (B, 4).
      dt: time step.
    Returns:
      shape (B, 3).
    """
    q_prev, q_next = rotations[:-2], rotations[2:]
    q_rel = quat_mul(q_next, quat_conjugate(q_prev))  # shape (B−2, 4)

    omega = axis_angle_from_quat(q_rel) / (2.0 * dt)  # shape (B−2, 3)
    omega = torch.cat(
      [omega[:1], omega, omega[-1:]], dim=0
    )  # repeat first and last sample
    return omega

  def get_next_state(
    self,
  ) -> tuple[
    tuple[
      torch.Tensor,
      torch.Tensor,
      torch.Tensor,
      torch.Tensor,
      torch.Tensor,
      torch.Tensor,
    ],
    bool,
  ]:
    """Gets the next state of the motion."""
    state = (
      self.motion_base_poss[self.current_idx : self.current_idx + 1],
      self.motion_base_rots[self.current_idx : self.current_idx + 1],
      self.motion_base_lin_vels[self.current_idx : self.current_idx + 1],
      self.motion_base_ang_vels[self.current_idx : self.current_idx + 1],
      self.motion_dof_poss[self.current_idx : self.current_idx + 1],
      self.motion_dof_vels[self.current_idx : self.current_idx + 1],
    )
    self.current_idx += 1
    reset_flag = False
    if self.current_idx >= self.output_frames:
      self.current_idx = 0
      reset_flag = True
    return state, reset_flag


def _resolve_motion_joint_names(
  joint_names: tuple[str, ...],
  motion_dof_count: int,
  robot: RobotName = "g1",
  motion_joint_names: tuple[str, ...] | None = None,
) -> tuple[str, ...]:
  resolved_names, _ = _resolve_motion_joint_mapping(
    joint_names,
    motion_dof_count,
    robot=robot,
    motion_joint_names=motion_joint_names,
  )
  return resolved_names


def _resolve_motion_joint_mapping(
  joint_names: tuple[str, ...],
  motion_dof_count: int,
  robot: RobotName = "g1",
  motion_joint_names: tuple[str, ...] | None = None,
) -> tuple[tuple[str, ...], torch.Tensor]:
  if motion_joint_names is not None:
    if motion_dof_count != len(motion_joint_names):
      raise ValueError(
        "`motion_joint_names` length does not match the CSV joint columns: "
        f"got {len(motion_joint_names)} names but CSV has {motion_dof_count} "
        "joint columns after the 7 root-state columns."
      )
    dropped_names = {
      "g1": frozenset(),
      "vr_h3_1": _VR_H3_1_DROPPED_SOURCE_JOINTS,
      "vr_m3_1": _VR_M3_1_DROPPED_SOURCE_JOINTS,
    }[robot]
    unknown_names = sorted(set(motion_joint_names) - set(joint_names) - dropped_names)
    if unknown_names:
      raise ValueError(
        "`motion_joint_names` contains joints that are not in the selected robot "
        f"'{robot}': {unknown_names}"
      )
    selected = [
      (i, name) for i, name in enumerate(motion_joint_names) if name in joint_names
    ]
    column_indexes = torch.tensor([i for i, _ in selected], dtype=torch.long)
    return tuple(name for _, name in selected), column_indexes

  if motion_dof_count == len(joint_names):
    return tuple(joint_names), torch.arange(motion_dof_count, dtype=torch.long)

  if robot == "g1":
    joint_names_without_wrist_yaw = tuple(
      name for name in joint_names if name not in _G1_WRIST_YAW_JOINTS
    )
  else:
    joint_names_without_wrist_yaw = ()

  if robot == "g1" and motion_dof_count == len(joint_names_without_wrist_yaw):
    print(
      "[INFO]: CSV has 27 joint columns; using the G1 joint order without "
      "left/right wrist yaw joints."
    )
    return joint_names_without_wrist_yaw, torch.arange(
      motion_dof_count, dtype=torch.long
    )
  if robot == "vr_h3_1" and motion_dof_count == len(VR_H3_1_27_JOINT_NAMES):
    print("[INFO]: CSV has 27 joint columns; using the 27-DOF VR H3.1 joint order.")
    return VR_H3_1_27_JOINT_NAMES, torch.arange(motion_dof_count, dtype=torch.long)
  if robot == "vr_h3_1" and motion_dof_count == len(VR_H3_1_28_SOURCE_JOINT_NAMES):
    print(
      "[INFO]: CSV has 28 joint columns; using the VR H3.1 source joint order "
      "and dropping waist_roll_joint for the 27-DOF robot model."
    )
    selected = [
      (i, name)
      for i, name in enumerate(VR_H3_1_28_SOURCE_JOINT_NAMES)
      if name in joint_names
    ]
    column_indexes = torch.tensor([i for i, _ in selected], dtype=torch.long)
    return tuple(name for _, name in selected), column_indexes
  if robot == "vr_m3_1" and motion_dof_count == len(VR_M3_1_27_JOINT_NAMES):
    print(
      "[INFO]: CSV has 27 joint columns; using the 27-DOF VR M3.1 joint order "
      "and leaving head joints at their default pose."
    )
    return VR_M3_1_27_JOINT_NAMES, torch.arange(motion_dof_count, dtype=torch.long)
  if robot == "vr_m3_1" and motion_dof_count == len(VR_M3_1_28_SOURCE_JOINT_NAMES):
    print(
      "[INFO]: CSV has 28 joint columns; using the VR M3.1 source joint order "
      "and dropping waist_roll_joint for the robot model."
    )
    selected = [
      (i, name)
      for i, name in enumerate(VR_M3_1_28_SOURCE_JOINT_NAMES)
      if name in joint_names
    ]
    column_indexes = torch.tensor([i for i, _ in selected], dtype=torch.long)
    return tuple(name for _, name in selected), column_indexes
  if robot == "vr_m3_1" and motion_dof_count == len(VR_M3_1_30_SOURCE_JOINT_NAMES):
    print(
      "[INFO]: CSV has 30 joint columns; using the VR M3.1 source joint order "
      "and dropping waist_roll_joint for the robot model."
    )
    selected = [
      (i, name)
      for i, name in enumerate(VR_M3_1_30_SOURCE_JOINT_NAMES)
      if name in joint_names
    ]
    column_indexes = torch.tensor([i for i, _ in selected], dtype=torch.long)
    return tuple(name for _, name in selected), column_indexes

  raise ValueError(
    "CSV joint column count does not match the configured robot joint order: "
    f"got {motion_dof_count} joint columns after the 7 root-state columns, "
    f"expected {len(joint_names)} for robot '{robot}'."
    + _robot_dof_hint(robot, len(joint_names_without_wrist_yaw))
  )


def _parse_motion_joint_names(motion_joint_names: str | None) -> tuple[str, ...] | None:
  if motion_joint_names is None:
    return None
  names = tuple(name.strip() for name in motion_joint_names.split(",") if name.strip())
  if not names:
    raise ValueError("`motion_joint_names` was provided but no joint names were found.")
  return names


def _robot_dof_hint(robot: RobotName, g1_without_wrist_yaw_count: int) -> str:
  if robot == "g1":
    return (
      f" G1 also supports {g1_without_wrist_yaw_count} columns without "
      "wrist yaw joints."
    )
  if robot == "vr_h3_1":
    return (
      " VR H3.1 supports 27 columns for the robot model, or 28 source "
      "columns with waist_roll_joint automatically dropped."
    )
  return (
    " VR M3.1 supports 27 columns for the main body joints, 28 source "
    "columns with waist_roll_joint automatically dropped, 29 columns for "
    "the full robot model, or 30 source columns with waist_roll_joint and "
    "head joints."
  )


def _resolve_output(output_name: str) -> tuple[Path, str]:
  output_path = Path(output_name)
  if output_path.suffix == ".npz" or output_path.parent != Path("."):
    return output_path, output_path.stem
  return Path("/tmp/motion.npz"), output_name


def run_sim(
  sim: Simulation,
  scene: Scene,
  joint_names: tuple[str, ...],
  robot_name: RobotName,
  input_file,
  input_fps,
  output_fps,
  output_name,
  render,
  line_range,
  motion_joint_names: tuple[str, ...] | None = None,
  upload_wandb: bool = True,
  renderer: OffscreenRenderer | None = None,
):
  motion = MotionLoader(
    motion_file=input_file,
    input_fps=input_fps,
    output_fps=output_fps,
    device=sim.device,
    line_range=line_range,
  )

  robot: Entity = scene["robot"]
  motion_joint_names, motion_column_indexes = _resolve_motion_joint_mapping(
    joint_names,
    motion.motion_dof_poss.shape[1],
    robot=robot_name,
    motion_joint_names=motion_joint_names,
  )
  motion_column_indexes = motion_column_indexes.to(sim.device)
  robot_joint_indexes = robot.find_joints(motion_joint_names, preserve_order=True)[0]
  output_path, collection = _resolve_output(output_name)

  log: dict[str, Any] = {
    "fps": [output_fps],
    "joint_pos": [],
    "joint_vel": [],
    "body_pos_w": [],
    "body_quat_w": [],
    "body_lin_vel_w": [],
    "body_ang_vel_w": [],
  }
  file_saved = False

  frames = []
  scene.reset()

  print(f"\nStarting simulation with {motion.output_frames} frames...")
  if render:
    print("Rendering enabled - generating video frames...")

  # Create progress bar
  pbar = tqdm(
    total=motion.output_frames,
    desc="Processing frames",
    unit="frame",
    ncols=100,
    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
  )

  frame_count = 0
  while not file_saved:
    (
      (
        motion_base_pos,
        motion_base_rot,
        motion_base_lin_vel,
        motion_base_ang_vel,
        motion_dof_pos,
        motion_dof_vel,
      ),
      reset_flag,
    ) = motion.get_next_state()

    root_states = robot.data.default_root_state.clone()
    root_states[:, 0:3] = motion_base_pos
    root_states[:, :2] += scene.env_origins[:, :2]
    root_states[:, 3:7] = motion_base_rot
    root_states[:, 7:10] = motion_base_lin_vel
    root_states[:, 10:] = motion_base_ang_vel
    robot.write_root_state_to_sim(root_states)

    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = robot.data.default_joint_vel.clone()
    joint_pos[:, robot_joint_indexes] = motion_dof_pos[:, motion_column_indexes]
    joint_vel[:, robot_joint_indexes] = motion_dof_vel[:, motion_column_indexes]
    robot.write_joint_state_to_sim(joint_pos, joint_vel)

    sim.forward()
    scene.update(sim.mj_model.opt.timestep)
    if render and renderer is not None:
      renderer.update(sim.data)
      frames.append(renderer.render())

    if not file_saved:
      log["joint_pos"].append(robot.data.joint_pos[0, :].cpu().numpy().copy())
      log["joint_vel"].append(robot.data.joint_vel[0, :].cpu().numpy().copy())
      log["body_pos_w"].append(robot.data.body_link_pos_w[0, :].cpu().numpy().copy())
      log["body_quat_w"].append(robot.data.body_link_quat_w[0, :].cpu().numpy().copy())
      log["body_lin_vel_w"].append(
        robot.data.body_link_lin_vel_w[0, :].cpu().numpy().copy()
      )
      log["body_ang_vel_w"].append(
        robot.data.body_link_ang_vel_w[0, :].cpu().numpy().copy()
      )

      torch.testing.assert_close(
        robot.data.body_link_lin_vel_w[0, 0],
        motion_base_lin_vel[0],
        atol=1e-4,
        rtol=1e-4,
      )
      torch.testing.assert_close(
        robot.data.body_link_ang_vel_w[0, 0],
        motion_base_ang_vel[0],
        atol=1e-4,
        rtol=1e-4,
      )

      frame_count += 1
      pbar.update(1)

      if frame_count % 100 == 0:  # Update every 100 frames to avoid spam
        elapsed_time = frame_count / output_fps
        pbar.set_description(f"Processing frames (t={elapsed_time:.1f}s)")

      if reset_flag and not file_saved:
        file_saved = True
        pbar.close()

        print("\nStacking arrays and saving data...")
        for k in (
          "joint_pos",
          "joint_vel",
          "body_pos_w",
          "body_quat_w",
          "body_lin_vel_w",
          "body_ang_vel_w",
        ):
          log[k] = np.stack(log[k], axis=0)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Saving to {output_path}...")
        np.savez(output_path, **log)

        if upload_wandb:
          print("Uploading to Weights & Biases...")
          import wandb

          run = wandb.init(project="csv_to_npz", name=collection)
          print(f"[INFO]: Logging motion to wandb: {collection}")
          REGISTRY = "motions"
          logged_artifact = run.log_artifact(
            artifact_or_path=str(output_path), name=collection, type=REGISTRY
          )
          run.link_artifact(
            artifact=logged_artifact,
            target_path=f"wandb-registry-{REGISTRY}/{collection}",
          )
          print(f"[INFO]: Motion saved to wandb registry: {REGISTRY}/{collection}")

        if render:
          import mediapy as media

          print("Creating video...")
          media.write_video("./motion.mp4", frames, fps=output_fps)

          if upload_wandb:
            print("Logging video to wandb...")
            import wandb

            wandb.log({"motion_video": wandb.Video("./motion.mp4", format="mp4")})

        if upload_wandb:
          import wandb

          wandb.finish()


def main(
  input_file: str,
  output_name: str,
  input_fps: float = 30.0,
  output_fps: float = 50.0,
  device: str = "cuda:0",
  render: bool = False,
  line_range: tuple[int, int] | None = None,
  robot: RobotName = "g1",
  motion_joint_names: str | None = None,
  upload_wandb: bool = True,
):
  """Replay motion from CSV file and output to npz file.

  Args:
    input_file: Path to the input CSV file.
    output_name: Path to the output npz file.
    input_fps: Frame rate of the CSV file.
    output_fps: Desired output frame rate.
    device: Device to use.
    render: Whether to render the simulation and save a video.
    line_range: Range of lines to process from the CSV file.
    robot: Robot model to use for replay and body ordering.
    motion_joint_names: Comma-separated joint names matching CSV joint columns.
    upload_wandb: Whether to upload the generated NPZ/video to Weights & Biases.
  """
  requested_device = device
  if device.startswith("cuda") and not torch.cuda.is_available():
    print("[WARNING]: CUDA is not available. Falling back to CPU. This may be slow.")
    device = "cpu"

  renderer = None
  sim = None
  scene = None
  model = None
  try:
    sim_cfg = SimulationCfg()
    sim_cfg.mujoco.timestep = 1.0 / output_fps

    if robot == "g1":
      scene_cfg = unitree_g1_flat_tracking_env_cfg().scene
      joint_names = G1_JOINT_NAMES
    elif robot == "vr_h3_1":
      scene_cfg = vr_h3_1_flat_tracking_env_cfg().scene
      joint_names = VR_H3_1_JOINT_NAMES
    elif robot == "vr_m3_1":
      scene_cfg = vr_m3_1_flat_tracking_env_cfg().scene
      joint_names = VR_M3_1_JOINT_NAMES
    else:
      raise ValueError(f"Unsupported robot: {robot}")

    scene_cfg.num_envs = 1
    scene = Scene(scene_cfg, device=device)
    model = scene.compile()

    sim = Simulation(num_envs=1, cfg=sim_cfg, model=model, device=device)

    scene.initialize(sim.mj_model, sim.model, sim.data)

    if render:
      viewer_cfg = ViewerConfig(
        height=480,
        width=640,
        origin_type=ViewerConfig.OriginType.ASSET_ROOT,
        entity_name="robot",
        distance=2.0,
        elevation=-5.0,
        azimuth=20,
      )
      renderer = OffscreenRenderer(
        model=sim.mj_model,
        cfg=viewer_cfg,
        scene=scene,
      )
      renderer.initialize()

    run_sim(
      sim=sim,
      scene=scene,
      joint_names=joint_names,
      robot_name=robot,
      input_fps=input_fps,
      input_file=input_file,
      output_fps=output_fps,
      output_name=output_name,
      render=render,
      line_range=line_range,
      motion_joint_names=_parse_motion_joint_names(motion_joint_names),
      upload_wandb=upload_wandb,
      renderer=renderer,
    )
  finally:
    if renderer is not None:
      renderer.close()

    del renderer
    del sim
    del scene
    del model
    gc.collect()

    if requested_device.startswith("cuda") and torch.cuda.is_available():
      torch.cuda.empty_cache()
      torch.cuda.ipc_collect()


def cli() -> None:
  tyro.cli(main, config=mjlab.TYRO_FLAGS)


if __name__ == "__main__":
  cli()
