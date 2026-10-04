"""Exact posterior of the binary CTMC used for reverse sampling.

For s < t the reverse step draws X_s from

    q(X_s | X_t, p_theta) = sum_a p_theta(X_0 = a | X_t) q(X_s | X_t, X_0 = a),

where each bridge q(X_s | X_t, X_0 = a) follows from Bayes' rule and the
closed-form transition probabilities of the forward process.
"""

import torch

from diffusion.beta_schedules import BetaSchedule


class ExactCTMCPosterior:
    """Posterior q(X_s | X_t, x0_pred) of the two-state CTMC."""

    def __init__(self, schedule=None, beta_min=0.1, beta_max=1.5):
        """
        Args:
            schedule: BetaSchedule; overrides the linear schedule when given
            beta_min, beta_max: end points of the linear noise-rate schedule
        """
        self.schedule = schedule or BetaSchedule('linear', beta_min, beta_max)

    def transition_probs(self, s, t):
        """(P_same, P_diff) of the forward process over the interval [s, t]."""
        integral = self.schedule.integral(t) - self.schedule.integral(s)
        # expm1 avoids cancellation for short intervals near t = 0.
        p_diff = -0.5 * torch.expm1(-2 * integral)
        return 1 - p_diff, p_diff

    def posterior_prob(self, x_t, x0_pred, t, s):
        """P(X_s = 1 | X_t, x0_pred).

        Args:
            x_t: current binary state
            x0_pred: predicted P(X_0 = 1 | X_t), same shape as x_t
            t: current time, a scalar or one per batch element
            s: target time, s < t
        """
        # float64 inputs are kept (used by the normalization tests); lower
        # precisions are promoted to float32 for numerical stability.
        dtype = torch.float64 if x0_pred.dtype == torch.float64 else torch.float32
        x_t = x_t.to(dtype=dtype)
        x0_pred = x0_pred.to(device=x_t.device, dtype=dtype)
        t = torch.as_tensor(t, device=x_t.device, dtype=dtype)
        s = torch.as_tensor(s, device=x_t.device, dtype=dtype)

        p_same_0s, p_diff_0s = self.transition_probs(torch.zeros_like(s), s)
        p_same_st, p_diff_st = self.transition_probs(s, t)
        while p_same_0s.dim() < x_t.dim():
            p_same_0s, p_diff_0s = p_same_0s.unsqueeze(-1), p_diff_0s.unsqueeze(-1)
            p_same_st, p_diff_st = p_same_st.unsqueeze(-1), p_diff_st.unsqueeze(-1)

        # P(X_t | X_s = k) for k in {0, 1}
        p_xt_given_xs1 = x_t * p_same_st + (1 - x_t) * p_diff_st
        p_xt_given_xs0 = x_t * p_diff_st + (1 - x_t) * p_same_st

        # Each clean-state bridge is normalized before mixing: x0_pred already
        # estimates P(X_0 = 1 | X_t), so mixing unnormalized numerators would
        # condition on X_t twice. Chapman-Kolmogorov gives P(X_t | X_0 = a).
        numerator_a0 = p_xt_given_xs1 * p_diff_0s
        denominator_a0 = numerator_a0 + p_xt_given_xs0 * p_same_0s
        numerator_a1 = p_xt_given_xs1 * p_same_0s
        denominator_a1 = numerator_a1 + p_xt_given_xs0 * p_diff_0s
        tiny = torch.finfo(dtype).tiny
        bridge_a0 = numerator_a0 / denominator_a0.clamp_min(tiny)
        bridge_a1 = numerator_a1 / denominator_a1.clamp_min(tiny)
        return ((1 - x0_pred) * bridge_a0 + x0_pred * bridge_a1).clamp(0, 1)

    def sample(self, x_t, x0_pred, t, s, deterministic=False):
        """Draw X_s ~ q(X_s | X_t, x0_pred); the posterior mode if `deterministic`."""
        posterior = self.posterior_prob(x_t, x0_pred, t, s)
        if deterministic:
            return (posterior > 0.5).float()
        return torch.bernoulli(posterior)
