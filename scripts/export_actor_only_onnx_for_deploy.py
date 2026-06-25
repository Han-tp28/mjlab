"""Export deploy ONNX policies for the Vinrobotics controller.

The legacy export path writes one deterministic actor graph for old MLP
checkpoints. The universal-token Mode 2 export path writes two graphs that match
Groot-style SMPL deployment:

  model_encoder.onnx: encoder_input(844) -> token_state(64)
  model_decoder.onnx: decoder_input(994), h_in, c_in -> actions, h_out, c_out

Mode 2 keeps the deploy recurrent state tensors as pass-through placeholders so
existing controller code can share the same call signature.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from pathlib import Path

import torch
from torch import nn


class FiniteScalarQuantizer(nn.Module):
  """Finite scalar quantization with straight-through gradients."""

  def __init__(self, levels: int = 32) -> None:
    super().__init__()
    if levels < 2:
      raise ValueError("FSQ `levels` must be at least 2.")
    self.levels = levels

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    bounded = torch.tanh(x)
    scaled = (bounded + 1.0) * 0.5 * (self.levels - 1)
    quantized = torch.round(scaled)
    dequantized = quantized / (self.levels - 1) * 2.0 - 1.0
    return bounded + (dequantized - bounded).detach()


def _build_mlp(
  state_dict: dict[str, torch.Tensor],
  prefix: str,
  activation: type[nn.Module] = nn.ELU,
) -> nn.Sequential:
  prefix_with_dot = f"{prefix}."
  linear_indices = sorted(
    {
      int(key.removeprefix(prefix_with_dot).split(".")[0])
      for key in state_dict
      if key.startswith(prefix_with_dot) and key.endswith(".weight")
    }
  )
  if not linear_indices:
    raise ValueError(f"No linear layers found for prefix '{prefix}'.")

  modules: OrderedDict[str, nn.Module] = OrderedDict()
  for out_idx, layer_idx in enumerate(linear_indices):
    weight = state_dict[f"{prefix}.{layer_idx}.weight"]
    bias = state_dict[f"{prefix}.{layer_idx}.bias"]
    layer = nn.Linear(weight.shape[1], weight.shape[0])
    layer.weight.data.copy_(weight)
    layer.bias.data.copy_(bias)
    modules[f"linear_{layer_idx}"] = layer
    if out_idx != len(linear_indices) - 1:
      modules[f"elu_{layer_idx}"] = activation()
  return nn.Sequential(modules)


def _last_linear_bias(state_dict: dict[str, torch.Tensor], prefix: str) -> torch.Tensor:
  prefix_with_dot = f"{prefix}."
  linear_indices = sorted(
    int(key.removeprefix(prefix_with_dot).split(".")[0])
    for key in state_dict
    if key.startswith(prefix_with_dot) and key.endswith(".bias")
  )
  if not linear_indices:
    raise ValueError(f"No linear bias found for prefix '{prefix}'.")
  return state_dict[f"{prefix}.{linear_indices[-1]}.bias"]


class DeployActorOnly(nn.Module):
  """Deterministic MLP actor with deploy-compatible dummy recurrent state IO."""

  def __init__(
    self, actor_state_dict: dict[str, torch.Tensor], input_layout: str
  ) -> None:
    super().__init__()
    self.input_layout = input_layout

    mean = actor_state_dict["obs_normalizer._mean"].detach().clone()
    std = actor_state_dict["obs_normalizer._std"].detach().clone()
    self.obs_mean: torch.Tensor
    self.obs_std: torch.Tensor
    self.register_buffer("obs_mean", mean)
    self.register_buffer("obs_std", std)
    self.mlp = _build_mlp(actor_state_dict, "mlp")

  def _expand_legacy_144_obs(self, obs: torch.Tensor) -> torch.Tensor:
    """Expand old C++ deploy obs(144) into the full actor obs layout."""
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
    """Expand current C++ deploy obs into the full actor obs layout."""
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


class UniversalMode2Encoder(nn.Module):
  """SMPL Mode 2 encoder export: 844 deploy obs -> 64 token_state."""

  def __init__(
    self,
    actor_state_dict: dict[str, torch.Tensor],
    expected_smpl_dim: int,
    token_total_dim: int,
    fsq_levels: int,
  ) -> None:
    super().__init__()
    mean = actor_state_dict["normalizers.smpl._mean"].detach().clone()
    std = actor_state_dict["normalizers.smpl._std"].detach().clone()
    if mean.shape[-1] != expected_smpl_dim:
      raise ValueError(
        "Checkpoint SMPL obs dim is "
        f"{mean.shape[-1]}, expected {expected_smpl_dim}. Train/export with "
        "Mode 2 actor_smpl = smpl_joints(720) + root_ori(60) + wrists(60)."
      )
    self.smpl_mean: torch.Tensor
    self.smpl_std: torch.Tensor
    self.register_buffer("smpl_mean", mean)
    self.register_buffer("smpl_std", std)
    self.smpl_encoder = _build_mlp(actor_state_dict, "encoders.smpl")
    self.quantizer = FiniteScalarQuantizer(fsq_levels)
    self.token_total_dim = token_total_dim
    last_bias = _last_linear_bias(actor_state_dict, "encoders.smpl")
    if last_bias.shape[0] != token_total_dim:
      raise ValueError(
        f"SMPL encoder output dim is {last_bias.shape[0]}, expected {token_total_dim}."
      )

  def forward(self, encoder_input: torch.Tensor) -> torch.Tensor:
    smpl_obs = encoder_input[:, 4:]
    smpl_obs = (smpl_obs - self.smpl_mean) / (self.smpl_std + 1e-2)
    latent = self.smpl_encoder(smpl_obs)
    return self.quantizer(latent).reshape(-1, self.token_total_dim)


class UniversalMode2Decoder(nn.Module):
  """Universal-token decoder export: 994 deploy obs -> robot actions."""

  def __init__(
    self,
    actor_state_dict: dict[str, torch.Tensor],
    expected_proprio_dim: int,
    token_total_dim: int,
  ) -> None:
    super().__init__()
    mean = actor_state_dict["normalizers.proprioception._mean"].detach().clone()
    std = actor_state_dict["normalizers.proprioception._std"].detach().clone()
    if mean.shape[-1] != expected_proprio_dim:
      raise ValueError(
        "Checkpoint proprioception dim is "
        f"{mean.shape[-1]}, expected {expected_proprio_dim}. Train/export with "
        "Mode 2 proprio history: ang_vel(30)+joint_pos(290)+joint_vel(290)"
        "+last_actions(290)+gravity(30)."
      )
    self.proprio_mean: torch.Tensor
    self.proprio_std: torch.Tensor
    self.register_buffer("proprio_mean", mean)
    self.register_buffer("proprio_std", std)
    self.decoder = _build_mlp(actor_state_dict, "decoder")
    self.token_total_dim = token_total_dim
    first_weight = actor_state_dict["decoder.0.weight"]
    expected_decoder_dim = token_total_dim + expected_proprio_dim
    if first_weight.shape[1] != expected_decoder_dim:
      raise ValueError(
        f"Decoder input dim is {first_weight.shape[1]}, "
        f"expected {expected_decoder_dim}."
      )
    self.action_dim = _last_linear_bias(actor_state_dict, "decoder").shape[0]

  def forward(
    self,
    decoder_input: torch.Tensor,
    h_in: torch.Tensor,
    c_in: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    token_state = decoder_input[:, : self.token_total_dim]
    proprio = decoder_input[:, self.token_total_dim :]
    proprio = (proprio - self.proprio_mean) / (self.proprio_std + 1e-2)
    actions = self.decoder(torch.cat([token_state, proprio], dim=-1))
    return actions, h_in, c_in


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--checkpoint", required=True, type=Path)
  parser.add_argument("--output", required=True, type=Path)
  parser.add_argument(
    "--export-format",
    choices=("actor_only", "universal_mode2"),
    default="actor_only",
  )
  parser.add_argument(
    "--input-layout",
    choices=("full", "deploy_supported", "legacy_144"),
    default="legacy_144",
    help=(
      "Only used by --export-format actor_only. full exports obs_dim from the "
      "actor checkpoint. deploy_supported exports the newer Vinrobotics "
      "MotionTracking obs layout and zero-fills unsupported terms. legacy_144 "
      "exports the current binary layout: 54+6+3+27+27+27."
    ),
  )
  parser.add_argument("--opset", default=18, type=int)
  parser.add_argument("--mode2-encoder-dim", default=844, type=int)
  parser.add_argument("--mode2-smpl-dim", default=840, type=int)
  parser.add_argument("--mode2-decoder-dim", default=994, type=int)
  parser.add_argument("--mode2-proprio-dim", default=930, type=int)
  parser.add_argument("--mode2-token-dim", default=64, type=int)
  parser.add_argument("--mode2-fsq-levels", default=32, type=int)
  return parser.parse_args()


def _actor_state_dict(checkpoint: dict) -> dict[str, torch.Tensor]:
  actor_state_dict = checkpoint.get("actor_state_dict")
  if actor_state_dict is None:
    raise KeyError("Checkpoint does not contain 'actor_state_dict'.")
  return actor_state_dict


def _mode2_output_paths(output: Path) -> tuple[Path, Path]:
  if output.suffix == ".onnx":
    return (
      output.with_name(f"{output.stem}_encoder.onnx"),
      output.with_name(f"{output.stem}_decoder.onnx"),
    )
  return output / "model_encoder.onnx", output / "model_decoder.onnx"


def export_actor_only(
  actor_state_dict: dict[str, torch.Tensor], args: argparse.Namespace
) -> None:
  model = DeployActorOnly(actor_state_dict, input_layout=args.input_layout).eval()
  actor_obs_dim = actor_state_dict["obs_normalizer._mean"].shape[-1]
  obs_dim = (
    144
    if args.input_layout == "legacy_144"
    else 532
    if args.input_layout == "deploy_supported"
    else actor_obs_dim
  )
  action_dim = _last_linear_bias(actor_state_dict, "mlp").shape[0]

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


def export_universal_mode2(
  actor_state_dict: dict[str, torch.Tensor], args: argparse.Namespace
) -> None:
  encoder_path, decoder_path = _mode2_output_paths(args.output)
  encoder_path.parent.mkdir(parents=True, exist_ok=True)
  decoder_path.parent.mkdir(parents=True, exist_ok=True)

  encoder = UniversalMode2Encoder(
    actor_state_dict,
    expected_smpl_dim=args.mode2_smpl_dim,
    token_total_dim=args.mode2_token_dim,
    fsq_levels=args.mode2_fsq_levels,
  ).eval()
  decoder = UniversalMode2Decoder(
    actor_state_dict,
    expected_proprio_dim=args.mode2_proprio_dim,
    token_total_dim=args.mode2_token_dim,
  ).eval()

  encoder_input = torch.zeros(1, args.mode2_encoder_dim)
  decoder_input = torch.zeros(1, args.mode2_decoder_dim)
  h_in = torch.zeros(1, 1, 256)
  c_in = torch.zeros(1, 1, 256)

  torch.onnx.export(
    encoder,
    (encoder_input,),
    encoder_path,
    export_params=True,
    opset_version=args.opset,
    input_names=["encoder_input"],
    output_names=["token_state"],
    dynamic_axes={},
    dynamo=False,
  )
  torch.onnx.export(
    decoder,
    (decoder_input, h_in, c_in),
    decoder_path,
    export_params=True,
    opset_version=args.opset,
    input_names=["decoder_input", "h_in", "c_in"],
    output_names=["actions", "h_out", "c_out"],
    dynamic_axes={},
    dynamo=False,
  )

  print(f"Exported encoder: {encoder_path}")
  print(f"Exported decoder: {decoder_path}")
  print(
    "mode2_encoder_dim="
    f"{args.mode2_encoder_dim}, mode2_decoder_dim={args.mode2_decoder_dim}, "
    f"token_dim={args.mode2_token_dim}, action_dim={decoder.action_dim}"
  )


def main() -> None:
  args = parse_args()
  checkpoint = torch.load(args.checkpoint, map_location="cpu")
  actor_state_dict = _actor_state_dict(checkpoint)

  if args.export_format == "universal_mode2":
    export_universal_mode2(actor_state_dict, args)
  else:
    export_actor_only(actor_state_dict, args)


if __name__ == "__main__":
  main()
