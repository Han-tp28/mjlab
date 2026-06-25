"""Tests for MotionTrackingOnPolicyRunner curriculum helpers."""

import torch

from mjlab.tasks.tracking.rl.runner import MotionTrackingOnPolicyRunner
from mjlab.tasks.tracking.rl.universal import UniversalTokenPPO


def test_capped_aux_term_passes_through_below_cap():
  """A normal-magnitude aux loss is unchanged (scale == 1)."""
  aux = torch.tensor(2.0, requires_grad=True)
  term = UniversalTokenPPO._capped_aux_term(aux, coef=1.0, cap=8.0)
  assert torch.isclose(term, torch.tensor(2.0))


def test_capped_aux_term_bounds_spike_but_keeps_gradient_direction():
  """A spiking aux loss contributes ~coef*cap, not coef*loss, with a smaller
  but same-signed gradient (so distillation still pulls in the right way)."""
  aux = torch.tensor(220.0, requires_grad=True)
  term = UniversalTokenPPO._capped_aux_term(aux, coef=1.0, cap=8.0)
  # Contribution bounded to coef*cap, not the raw 220.
  assert torch.isclose(term, torch.tensor(8.0))
  term.backward()
  # Gradient is positive (same direction) but scaled down by cap/|aux|.
  assert aux.grad is not None
  assert 0.0 < float(aux.grad) < 1.0


def test_capped_aux_term_none_disables_capping():
  aux = torch.tensor(220.0)
  term = UniversalTokenPPO._capped_aux_term(aux, coef=1.0, cap=None)
  assert torch.isclose(term, torch.tensor(220.0))


def test_compute_fall_rate_default_counts_every_non_timeout_termination():
  """With an empty fall set, every non-timeout termination is a fall."""
  ep_extras = [
    {
      "Episode_Termination/time_out": torch.tensor(7.0),
      "Episode_Termination/anchor_xy": torch.tensor(2.0),
      "Episode_Termination/fell_over_height": torch.tensor(1.0),
    }
  ]
  fall_rate = MotionTrackingOnPolicyRunner._compute_fall_rate(ep_extras, ())
  assert fall_rate == 3.0 / 10.0


def test_compute_fall_rate_restricted_to_literal_falls():
  """With fall_termination_keys set, tracking-miss terminations don't count.

  Regression test: anchor_xy (a recoverable tracking miss, already covered by
  the error_anchor_pos gate check) used to inflate fall_rate enough that the
  curriculum gate almost never passed naturally and growth always happened
  via the force_after_iterations override instead.
  """
  ep_extras = [
    {
      "Episode_Termination/time_out": torch.tensor(7.0),
      "Episode_Termination/anchor_xy": torch.tensor(2.0),
      "Episode_Termination/fell_over_height": torch.tensor(1.0),
    }
  ]
  fall_rate = MotionTrackingOnPolicyRunner._compute_fall_rate(
    ep_extras, ("fell_over_height", "fell_over_orientation")
  )
  assert fall_rate == 1.0 / 10.0


def test_compute_fall_rate_no_done_episodes_returns_zero():
  fall_rate = MotionTrackingOnPolicyRunner._compute_fall_rate([], ())
  assert fall_rate == 0.0
