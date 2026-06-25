from __future__ import annotations

import copy
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from rsl_rl.algorithms import PPO
from rsl_rl.modules import MLP, EmpiricalNormalization, HiddenState
from rsl_rl.modules.distribution import Distribution
from rsl_rl.utils import resolve_callable, unpad_trajectories
from tensordict import TensorDict


class FiniteScalarQuantizer(nn.Module):
  """Small FSQ bottleneck with straight-through gradients."""

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


class UniversalTokenActor(nn.Module):
  """Groot-style multi-encoder actor with an FSQ token bottleneck."""

  is_recurrent: bool = False

  def __init__(
    self,
    obs: TensorDict,
    obs_groups: dict[str, list[str]],
    obs_set: str,
    output_dim: int,
    hidden_dims: tuple[int, ...] | list[int] = (512, 256, 128),
    activation: str = "elu",
    obs_normalization: bool = False,
    distribution_cfg: dict | None = None,
    smpl_obs_set: str = "actor_smpl",
    g1_obs_set: str = "actor_g1",
    teleop_obs_set: str = "actor_teleop",
    tokenizer_obs_set: str = "tokenizer",
    proprioception_obs_set: str = "proprioception",
    encoder_names: tuple[str, ...] | list[str] = ("g1", "teleop", "smpl"),
    num_fsq_levels: int = 32,
    fsq_level_list: int = 32,
    max_num_tokens: int = 2,
    stiff_compliance_threshold: float = 0.01,
  ) -> None:
    super().__init__()
    del obs_set, num_fsq_levels

    self.encoder_names = tuple(encoder_names)
    if self.encoder_names != ("g1", "teleop", "smpl"):
      raise ValueError(
        "UniversalTokenActor expects encoder_names=('g1','teleop','smpl')."
      )
    self.max_num_tokens = max_num_tokens
    self.token_dim = fsq_level_list
    self.token_total_dim = self.max_num_tokens * self.token_dim
    self.stiff_compliance_threshold = stiff_compliance_threshold
    self.obs_normalization = obs_normalization

    self.obs_group_names = {
      "g1": self._get_obs_groups(obs, obs_groups, g1_obs_set),
      "teleop": self._get_obs_groups(obs, obs_groups, teleop_obs_set),
      "smpl": self._get_obs_groups(obs, obs_groups, smpl_obs_set),
      "tokenizer": self._get_obs_groups(obs, obs_groups, tokenizer_obs_set),
      "proprioception": self._get_obs_groups(obs, obs_groups, proprioception_obs_set),
    }
    self.obs_dims = {
      name: self._obs_dim(obs, groups) for name, groups in self.obs_group_names.items()
    }
    self.obs_dim = sum(
      self.obs_dims[name]
      for name in ("tokenizer", "g1", "teleop", "smpl", "proprioception")
    )

    self.normalizers = nn.ModuleDict()
    for name, dim in self.obs_dims.items():
      if name == "tokenizer":
        self.normalizers[name] = nn.Identity()
      else:
        self.normalizers[name] = (
          EmpiricalNormalization(dim) if obs_normalization else nn.Identity()
        )

    hidden_dims = tuple(hidden_dims)
    encoder_hidden_dims = hidden_dims[:-1] or (512, 256)
    decoder_hidden_dims = hidden_dims[:-1] or (512, 256)

    self.encoders = nn.ModuleDict(
      {
        "g1": MLP(
          self.obs_dims["g1"], self.token_total_dim, encoder_hidden_dims, activation
        ),
        "teleop": MLP(
          self.obs_dims["teleop"], self.token_total_dim, encoder_hidden_dims, activation
        ),
        "smpl": MLP(
          self.obs_dims["smpl"], self.token_total_dim, encoder_hidden_dims, activation
        ),
      }
    )
    self.quantizer = FiniteScalarQuantizer(fsq_level_list)

    if distribution_cfg is not None:
      dist_cfg = copy.deepcopy(distribution_cfg)
      dist_class: type[Distribution] = resolve_callable(  # type: ignore[assignment]
        dist_cfg.pop("class_name")
      )
      self.distribution: Distribution | None = dist_class(output_dim, **dist_cfg)
      action_output_dim = self.distribution.input_dim
    else:
      self.distribution = None
      action_output_dim = output_dim

    self.decoder = MLP(
      self.token_total_dim + self.obs_dims["proprioception"],
      action_output_dim,
      decoder_hidden_dims,
      activation,
    )
    self.g1_kin_decoder = MLP(
      self.token_total_dim, self.obs_dims["g1"], decoder_hidden_dims, activation
    )
    if self.distribution is not None:
      self.distribution.init_mlp_weights(self.decoder)

    self._last_aux_losses: dict[str, torch.Tensor] = {}
    self._last_encoder_ratios: dict[str, torch.Tensor] = {}

  def forward(
    self,
    obs: TensorDict,
    masks: torch.Tensor | None = None,
    hidden_state: HiddenState = None,
    stochastic_output: bool = False,
  ) -> torch.Tensor:
    del hidden_state
    obs = cast(TensorDict, unpad_trajectories(obs, masks)) if masks is not None else obs
    tokens, aux_losses, encoder_ratios = self._tokens_and_aux(obs)
    self._last_aux_losses = aux_losses
    self._last_encoder_ratios = encoder_ratios

    proprioception = self._normalized_cat(obs, "proprioception")
    mlp_output = self.decoder(torch.cat([tokens, proprioception], dim=-1))
    if self.distribution is not None:
      if stochastic_output:
        self.distribution.update(mlp_output)
        return self.distribution.sample()
      return self.distribution.deterministic_output(mlp_output)
    return mlp_output

  def _tokens_and_aux(
    self, obs: TensorDict
  ) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    tokenizer = self._normalized_cat(obs, "tokenizer")
    encoder_index = tokenizer[..., : len(self.encoder_names)].float()
    compliance = tokenizer[..., len(self.encoder_names) : len(self.encoder_names) + 3]
    if compliance.shape[-1] != 3:
      compliance = torch.zeros(tokenizer.shape[0], 3, device=tokenizer.device)

    active_encoder_mask = encoder_index > 0.0
    fallback = torch.zeros_like(active_encoder_mask)
    fallback[:, self.encoder_names.index("smpl")] = True
    active_encoder_mask = torch.where(
      active_encoder_mask.any(dim=-1, keepdim=True),
      active_encoder_mask,
      fallback,
    )

    # Groot/SONIC keeps encoder_index multi-hot for paired aux losses, but the
    # action token is still routed from the native source. Later encoders have
    # higher priority because Groot scatters g1, then teleop, then smpl tokens.
    encoder_weights = torch.zeros_like(encoder_index)
    remaining = torch.ones(
      encoder_index.shape[0], dtype=torch.bool, device=encoder_index.device
    )
    for idx in reversed(range(len(self.encoder_names))):
      selected = active_encoder_mask[:, idx] & remaining
      encoder_weights[:, idx] = selected.float()
      remaining = remaining & ~selected

    raw_latents = {
      name: self.encoders[name](self._normalized_cat(obs, name)).view(
        -1, self.max_num_tokens, self.token_dim
      )
      for name in self.encoder_names
    }
    stacked_latents = torch.stack(
      [raw_latents[name] for name in self.encoder_names], dim=1
    )
    selected_latent = (stacked_latents * encoder_weights[:, :, None, None]).sum(dim=1)
    tokens = self.quantizer(selected_latent).reshape(-1, self.token_total_dim)
    quantized_tokens = {
      name: self.quantizer(raw_latents[name]).reshape(-1, self.token_total_dim)
      for name in self.encoder_names
    }

    aux_losses = self._compute_aux_losses(
      obs,
      raw_latents,
      quantized_tokens,
      active_encoder_mask,
      compliance,
      tokens,
    )
    encoder_ratios = {
      f"encoder_ratio/{name}": encoder_weights[:, idx].mean().detach()
      for idx, name in enumerate(self.encoder_names)
    }
    encoder_ratios.update(
      {
        f"encoder_active_ratio/{name}": active_encoder_mask[:, idx]
        .float()
        .mean()
        .detach()
        for idx, name in enumerate(self.encoder_names)
      }
    )
    return tokens, aux_losses, encoder_ratios

  def _compute_aux_losses(
    self,
    obs: TensorDict,
    raw_latents: dict[str, torch.Tensor],
    quantized_tokens: dict[str, torch.Tensor],
    active_encoder_mask: torch.Tensor,
    compliance: torch.Tensor,
    tokens: torch.Tensor,
  ) -> dict[str, torch.Tensor]:
    g1_obs = self._normalized_cat(obs, "g1")
    g1_recon = self.g1_kin_decoder(tokens)
    stiff_mask = (compliance.abs() < self.stiff_compliance_threshold).all(dim=-1)

    def masked_loss(
      lhs: torch.Tensor, rhs: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
      if not torch.any(mask):
        return lhs.sum() * 0.0
      return F.mse_loss(lhs[mask], rhs[mask])

    g1_idx = self.encoder_names.index("g1")
    teleop_idx = self.encoder_names.index("teleop")
    smpl_idx = self.encoder_names.index("smpl")
    g1_active = active_encoder_mask[:, g1_idx]
    teleop_active = active_encoder_mask[:, teleop_idx]
    smpl_active = active_encoder_mask[:, smpl_idx]
    g1_smpl_mask = g1_active & smpl_active & stiff_mask
    g1_teleop_mask = g1_active & teleop_active
    teleop_smpl_mask = teleop_active & smpl_active

    g1_latent = raw_latents["g1"].detach()
    teleop_latent = raw_latents["teleop"]
    smpl_latent = raw_latents["smpl"]
    smpl_recon = self.g1_kin_decoder(quantized_tokens["smpl"])
    reencoded_smpl = self.encoders["g1"](smpl_recon).view(
      -1, self.max_num_tokens, self.token_dim
    )

    return {
      "g1_recon": F.mse_loss(g1_recon, g1_obs),
      "g1_smpl_latent": masked_loss(smpl_latent, g1_latent, g1_smpl_mask),
      "g1_teleop_latent": masked_loss(teleop_latent, g1_latent, g1_teleop_mask),
      "teleop_smpl_latent": masked_loss(
        smpl_latent, teleop_latent.detach(), teleop_smpl_mask
      ),
      "reencoded_smpl_g1_latent": masked_loss(reencoded_smpl, g1_latent, g1_smpl_mask),
    }

  def aux_losses(self) -> dict[str, torch.Tensor]:
    return self._last_aux_losses

  def encoder_ratios(self) -> dict[str, torch.Tensor]:
    return self._last_encoder_ratios

  def reset(
    self, dones: torch.Tensor | None = None, hidden_state: HiddenState = None
  ) -> None:
    del dones, hidden_state

  def get_hidden_state(self) -> HiddenState:
    return None

  def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
    del dones

  @property
  def output_mean(self) -> torch.Tensor:
    assert self.distribution is not None
    return self.distribution.mean

  @property
  def output_std(self) -> torch.Tensor:
    assert self.distribution is not None
    return self.distribution.std

  @property
  def output_entropy(self) -> torch.Tensor:
    assert self.distribution is not None
    return self.distribution.entropy

  @property
  def output_distribution_params(self) -> tuple[torch.Tensor, ...]:
    assert self.distribution is not None
    return self.distribution.params

  def get_output_log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
    assert self.distribution is not None
    return self.distribution.log_prob(outputs)

  def get_kl_divergence(
    self, old_params: tuple[torch.Tensor, ...], new_params: tuple[torch.Tensor, ...]
  ) -> torch.Tensor:
    assert self.distribution is not None
    return self.distribution.kl_divergence(old_params, new_params)

  def as_onnx(self, verbose: bool) -> nn.Module:
    return _OnnxUniversalTokenActor(self, verbose)

  def update_normalization(self, obs: TensorDict) -> None:
    if not self.obs_normalization:
      return
    for name in self.obs_group_names:
      if name == "tokenizer":
        continue
      self.normalizers[name].update(self._cat_obs(obs, name))  # type: ignore[attr-defined]

  def _normalized_cat(self, obs: TensorDict, name: str) -> torch.Tensor:
    return self.normalizers[name](self._cat_obs(obs, name))

  def _cat_obs(self, obs: TensorDict, name: str) -> torch.Tensor:
    return torch.cat(
      [cast(torch.Tensor, obs[group]) for group in self.obs_group_names[name]], dim=-1
    )

  def _get_obs_groups(
    self, obs: TensorDict, obs_groups: dict[str, list[str]], obs_set: str
  ) -> list[str]:
    del obs
    return list(obs_groups[obs_set])

  def _obs_dim(self, obs: TensorDict, groups: list[str]) -> int:
    obs_dim = 0
    for obs_group in groups:
      if len(obs[obs_group].shape) != 2:
        raise ValueError(
          "UniversalTokenActor only supports 1D observations, got shape "
          f"{obs[obs_group].shape} for '{obs_group}'."
        )
      obs_dim += obs[obs_group].shape[-1]
    return obs_dim


class _OnnxUniversalTokenActor(nn.Module):
  is_recurrent: bool = False

  def __init__(self, model: UniversalTokenActor, verbose: bool) -> None:
    super().__init__()
    self.verbose = verbose
    self.model = copy.deepcopy(model)
    self.model.eval()
    self.input_size = model.obs_dim

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    raise RuntimeError(
      "UniversalTokenActor ONNX export needs named observation export; use play/train "
      "checkpoints directly until split encoder export is added."
    )

  def get_dummy_inputs(self) -> tuple[torch.Tensor]:
    return (torch.zeros(1, self.input_size),)

  @property
  def input_names(self) -> list[str]:
    return ["obs"]

  @property
  def output_names(self) -> list[str]:
    return ["actions"]


class UniversalTokenPPO(PPO):
  def __init__(
    self,
    *args,
    aux_loss_coefs: dict[str, float] | None = None,
    aux_loss_cap: float | None = None,
    **kwargs,
  ) -> None:
    super().__init__(*args, **kwargs)
    self.aux_loss_coefs = aux_loss_coefs or {}
    # Upper bound on each aux (latent-distillation) loss term's contribution to
    # the total loss. The cross-encoder latent losses are unbounded MSEs with
    # coef 1.0; when a batch of hard new motions enters the curriculum the
    # teleop/smpl latents diverge from the g1 teacher and the loss spikes ~100x
    # (e.g. ~2 -> ~220), dominating the gradient and pulling the shared trunk to
    # chase a target computed on an already-destabilising policy -> an
    # unrecoverable blow-up (value loss spike, action std explosion, global
    # reward collapse). Capping bounds the spike's gradient magnitude while
    # preserving its direction, turning a local hard-motion difficulty back into
    # a recoverable high-error episode instead of a catastrophe. None = off.
    self.aux_loss_cap = aux_loss_cap

  @staticmethod
  def _capped_aux_term(
    aux_loss: torch.Tensor, coef: float, cap: float | None
  ) -> torch.Tensor:
    """coef * aux_loss, with the term's magnitude bounded to ~coef*cap.

    Scales by ``min(1, cap/|aux_loss|)`` using a detached denominator so the
    gradient direction is unchanged but a spike cannot dominate the objective.
    """
    if cap is None:
      return coef * aux_loss
    scale = torch.clamp(cap / aux_loss.detach().abs().clamp(min=1e-6), max=1.0)
    return coef * scale * aux_loss

  def update(self) -> dict[str, float]:
    mean_value_loss = 0.0
    mean_surrogate_loss = 0.0
    mean_entropy = 0.0
    mean_aux_losses = {name: 0.0 for name in self.aux_loss_coefs}
    mean_encoder_ratios = {
      **{f"encoder_ratio/{name}": 0.0 for name in ("g1", "teleop", "smpl")},
      **{f"encoder_active_ratio/{name}": 0.0 for name in ("g1", "teleop", "smpl")},
    }
    mean_rnd_loss = 0.0 if self.rnd else None
    mean_symmetry_loss = 0.0 if self.symmetry else None

    if self.actor.is_recurrent or self.critic.is_recurrent:
      generator = self.storage.recurrent_mini_batch_generator(
        self.num_mini_batches, self.num_learning_epochs
      )
    else:
      generator = self.storage.mini_batch_generator(
        self.num_mini_batches, self.num_learning_epochs
      )

    for batch in generator:
      observations = cast(TensorDict, batch.observations)
      original_batch_size = observations.batch_size[0]

      if self.normalize_advantage_per_mini_batch:
        with torch.no_grad():
          advantages = cast(torch.Tensor, batch.advantages)
          batch.advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-8
          )

      if self.symmetry:
        self.symmetry.augment_batch(batch, original_batch_size)

      self.actor(
        observations,
        masks=batch.masks,
        hidden_state=batch.hidden_states[0],
        stochastic_output=True,
      )
      actor = cast(UniversalTokenActor, self.actor)
      aux_losses = actor.aux_losses()
      encoder_ratios = actor.encoder_ratios()

      actions_log_prob = self.actor.get_output_log_prob(batch.actions)  # type: ignore[arg-type]
      values = self.critic(
        observations, masks=batch.masks, hidden_state=batch.hidden_states[1]
      )
      distribution_params = tuple(
        p[:original_batch_size] for p in self.actor.output_distribution_params
      )
      entropy = self.actor.output_entropy[:original_batch_size]

      if self.desired_kl is not None and self.schedule == "adaptive":
        with torch.inference_mode():
          kl = self.actor.get_kl_divergence(
            batch.old_distribution_params,  # type: ignore[arg-type]
            distribution_params,
          )
          kl_mean = torch.mean(kl)

          if self.is_multi_gpu:
            torch.distributed.all_reduce(  # type: ignore[attr-defined]
              kl_mean,
              op=torch.distributed.ReduceOp.SUM,  # type: ignore[attr-defined]
            )
            kl_mean /= self.gpu_world_size

          if self.gpu_global_rank == 0:
            if kl_mean > self.desired_kl * 2.0:
              self.learning_rate = max(1e-5, self.learning_rate / 1.5)
            elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
              self.learning_rate = min(1e-2, self.learning_rate * 1.5)

          if self.is_multi_gpu:
            lr_tensor = torch.tensor(self.learning_rate, device=self.device)
            torch.distributed.broadcast(lr_tensor, src=0)  # type: ignore[attr-defined]
            self.learning_rate = lr_tensor.item()

          for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.learning_rate

      ratio = torch.exp(
        actions_log_prob - torch.squeeze(batch.old_actions_log_prob)  # type: ignore[arg-type]
      )
      surrogate = -torch.squeeze(batch.advantages) * ratio  # type: ignore[arg-type]
      surrogate_clipped = -torch.squeeze(batch.advantages) * torch.clamp(  # type: ignore[arg-type]
        ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
      )
      surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

      if self.use_clipped_value_loss:
        value_clipped = batch.values + (values - batch.values).clamp(
          -self.clip_param, self.clip_param
        )
        value_losses = (values - batch.returns).pow(2)
        value_losses_clipped = (value_clipped - batch.returns).pow(2)
        value_loss = torch.max(value_losses, value_losses_clipped).mean()
      else:
        value_loss = (batch.returns - values).pow(2).mean()

      entropy_loss = entropy.mean()
      loss = (
        surrogate_loss
        + self.value_loss_coef * value_loss
        - self.entropy_coef * entropy_loss
      )
      for name, coef in self.aux_loss_coefs.items():
        if name in aux_losses:
          loss = loss + self._capped_aux_term(aux_losses[name], coef, self.aux_loss_cap)

      rnd_loss = (
        self.rnd.compute_loss(observations[:original_batch_size])  # type: ignore[arg-type]
        if self.rnd
        else None
      )
      symmetry_loss = torch.zeros((), device=self.device)

      if self.symmetry:
        symmetry_loss = self.symmetry.compute_loss(
          self.actor, batch, original_batch_size
        )
        if self.symmetry.use_mirror_loss:
          loss = loss + self.symmetry.mirror_loss_coeff * symmetry_loss

      self.optimizer.zero_grad()
      loss.backward()
      if self.rnd:
        self.rnd.optimizer.zero_grad()
        rnd_loss.backward()  # type: ignore[union-attr]

      if self.is_multi_gpu:
        self.reduce_parameters()

      nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
      nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
      self.optimizer.step()
      if self.rnd:
        self.rnd.optimizer.step()

      mean_value_loss += value_loss.item()
      mean_surrogate_loss += surrogate_loss.item()
      mean_entropy += entropy_loss.item()
      for name in mean_aux_losses:
        aux_value = aux_losses[name] if name in aux_losses else value_loss * 0.0
        mean_aux_losses[name] += aux_value.item()
      for name in mean_encoder_ratios:
        ratio_value = (
          encoder_ratios[name] if name in encoder_ratios else value_loss * 0.0
        )
        mean_encoder_ratios[name] += ratio_value.item()
      if mean_rnd_loss is not None:
        mean_rnd_loss += rnd_loss.item()  # type: ignore[union-attr]
      if mean_symmetry_loss is not None:
        mean_symmetry_loss += symmetry_loss.item()

    num_updates = self.num_learning_epochs * self.num_mini_batches
    loss_dict = {
      "value": mean_value_loss / num_updates,
      "surrogate": mean_surrogate_loss / num_updates,
      "entropy": mean_entropy / num_updates,
    }
    for name, value in mean_aux_losses.items():
      loss_dict[name] = value / num_updates
    for name, value in mean_encoder_ratios.items():
      loss_dict[name] = value / num_updates
    if self.rnd:
      loss_dict["rnd"] = mean_rnd_loss / num_updates  # type: ignore[operator]
    if self.symmetry:
      loss_dict["symmetry"] = mean_symmetry_loss / num_updates  # type: ignore[operator]

    self.storage.clear()
    return loss_dict


UniversalPoseActor = UniversalTokenActor
LatentAlignmentPPO = UniversalTokenPPO
