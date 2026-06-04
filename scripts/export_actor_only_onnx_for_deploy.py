"""Export an actor-only ONNX policy for the Vinrobotics deploy controller.

The tracking runner normal ONNX export bundles motion reference tensors into the
graph. With large motion libraries that file can be several GB and is not useful
for the deploy controller, which already reads motion.npz separately.

This script exports only the deterministic actor:
  inputs:  obs, h_in, c_in
  outputs: actions, h_out, c_out

The h/c tensors are pass-through dummy tensors so the output stays compatible
with older recurrent deploy policies. By default the ONNX accepts the current
Vinrobotics MotionTracking observation layout and zero-fills trained terms that
the C++ binary does not register yet.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from pathlib import Path

import torch
from torch import nn


class DeployActorOnly(nn.Module):
  """Deterministic MLP actor with deploy-compatible dummy recurrent state IO."""

  def __init__(
    self, actor_state_dict: dict[str, torch.Tensor], input_layout: str
  ) -> None:
    super().__init__()
    self.input_layout = input_layout

    mean = actor_state_dict["obs_normalizer._mean"].detach().clone()
    std = actor_state_dict["obs_normalizer._std"].detach().clone()
    self.register_buffer("obs_mean", mean)
    self.register_buffer("obs_std", std)

    modules: OrderedDict[str, nn.Module] = OrderedDict()
    linear_indices = sorted(
      {
        int(key.split(".")[1])
        for key in actor_state_dict
        if key.startswith("mlp.") and key.endswith(".weight")
      }
    )
    for out_idx, layer_idx in enumerate(linear_indices):
      weight = actor_state_dict[f"mlp.{layer_idx}.weight"]
      bias = actor_state_dict[f"mlp.{layer_idx}.bias"]
      layer = nn.Linear(weight.shape[1], weight.shape[0])
      layer.weight.data.copy_(weight)
      layer.bias.data.copy_(bias)
      modules[f"linear_{layer_idx}"] = layer
      if out_idx != len(linear_indices) - 1:
        modules[f"elu_{layer_idx}"] = nn.ELU()

    self.mlp = nn.Sequential(modules)

  def _expand_legacy_144_obs(self, obs: torch.Tensor) -> torch.Tensor:
    """Expand old C++ deploy obs(144) into the full actor obs layout.

    Old binary layout:
      command(54), anchor_ori(6), base_ang(3), joint_pos(27),
      joint_vel(27), actions(27)

    Trained actor layout:
      command(442), anchor_pos(3), anchor_ori(6), base_lin(3),
      base_ang(3), joint_pos(27), joint_vel(27), actions(27)

    Missing command tail, anchor_pos, and base_lin_vel are zero-filled.
    """
    command = obs[:, 0:54]
    anchor_ori = obs[:, 54:60]
    base_ang = obs[:, 60:63]
    joint_pos = obs[:, 63:90]
    joint_vel = obs[:, 90:117]
    actions = obs[:, 117:144]
    zeros388 = obs.new_zeros((obs.shape[0], 388))
    zeros3 = obs.new_zeros((obs.shape[0], 3))
    return torch.cat(
      [
        command,
        zeros388,
        zeros3,
        anchor_ori,
        zeros3,
        base_ang,
        joint_pos,
        joint_vel,
        actions,
      ],
      dim=-1,
    )

  def _expand_deploy_supported_obs(self, obs: torch.Tensor) -> torch.Tensor:
    """Expand C++ deploy obs into the full actor obs layout.

    The current Vinrobotics binary supports:
      command(442), anchor_ori(6), base_ang(3), joint_pos(27),
      joint_vel(27), actions(27)

    The trained actor expects:
      command(442), anchor_pos(3), anchor_ori(6), base_lin(3),
      base_ang(3), joint_pos(27), joint_vel(27), actions(27)

    Missing anchor_pos and base_lin_vel are zero-filled.
    """
    command = obs[:, 0:442]
    anchor_ori = obs[:, 442:448]
    base_ang = obs[:, 448:451]
    joint_pos = obs[:, 451:478]
    joint_vel = obs[:, 478:505]
    actions = obs[:, 505:532]
    zeros3 = torch.zeros_like(obs[:, 0:3])
    return torch.cat(
      [
        command,
        zeros3,
        anchor_ori,
        zeros3,
        base_ang,
        joint_pos,
        joint_vel,
        actions,
      ],
      dim=-1,
    )

  def forward(
    self, obs: torch.Tensor, h_in: torch.Tensor, c_in: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if self.input_layout == "legacy_144":
      obs = self._expand_legacy_144_obs(obs)
    elif self.input_layout == "deploy_supported":
      obs = self._expand_deploy_supported_obs(obs)
    obs = (obs - self.obs_mean) / (self.obs_std + 1e-2)
    actions = self.mlp(obs)
    return actions, h_in, c_in


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--checkpoint", required=True, type=Path)
  parser.add_argument("--output", required=True, type=Path)
  parser.add_argument(
    "--input-layout",
    choices=("full", "deploy_supported", "legacy_144"),
    default="legacy_144",
    help=(
      "full exports obs_dim from the actor checkpoint. deploy_supported exports "
      "the newer Vinrobotics MotionTracking obs layout and zero-fills "
      "unsupported anchor_pos/base_lin_vel terms. legacy_144 exports the "
      "current binary layout seen in logs: 54+6+3+27+27+27."
    ),
  )
  parser.add_argument("--opset", default=18, type=int)
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  checkpoint = torch.load(args.checkpoint, map_location="cpu")
  actor_state_dict = checkpoint["actor_state_dict"]

  model = DeployActorOnly(actor_state_dict, input_layout=args.input_layout).eval()
  actor_obs_dim = actor_state_dict["obs_normalizer._mean"].shape[-1]
  obs_dim = (
    144
    if args.input_layout == "legacy_144"
    else 532
    if args.input_layout == "deploy_supported"
    else actor_obs_dim
  )
  action_dim = actor_state_dict["mlp.6.bias"].shape[0]

  args.output.parent.mkdir(parents=True, exist_ok=True)
  obs = torch.zeros(1, obs_dim)
  h_in = torch.zeros(1, 1, 256)
  c_in = torch.zeros(1, 1, 256)

  torch.onnx.export(
    model,
    (obs, h_in, c_in),
    args.output,
    export_params=True,
    opset_version=args.opset,
    input_names=["obs", "h_in", "c_in"],
    output_names=["actions", "h_out", "c_out"],
    dynamic_axes={},
    dynamo=False,
  )

  print(f"Exported: {args.output}")
  print(
    f"input_layout={args.input_layout}, onnx_obs_dim={obs_dim}, "
    f"actor_obs_dim={actor_obs_dim}, action_dim={action_dim}"
  )


if __name__ == "__main__":
  main()
