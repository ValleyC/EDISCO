"""
Exact CTMC Posterior Sampling for EDISCO
Based on Campbell et al., 2022 - Continuous Time Markov Chains
"""

import torch
import torch.nn.functional as F
from typing import Tuple


class ExactCTMCPosterior:
    """
    Exact posterior computation for reverse-time CTMC sampling.
    Computes q(X_s | X_t, X_0) using Bayes' rule.
    """

    def __init__(
        self,
        beta_min: float = 0.1,
        beta_max: float = 1.5,
        epsilon: float = 1e-8
    ):
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.epsilon = epsilon

    def integral_beta(self, t: torch.Tensor) -> torch.Tensor:
        """Integral of beta from 0 to t for linear schedule."""
        return self.beta_min * t + 0.5 * (self.beta_max - self.beta_min) * t ** 2

    def transition_probs(
        self,
        s: torch.Tensor,
        t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Transition probabilities P_same and P_diff for interval [s, t].

        Args:
            s: Start time
            t: End time (t >= s)

        Returns:
            (P_same, P_diff) where P_same = P(X_t = i | X_s = i)
        """
        integral_diff = self.integral_beta(t) - self.integral_beta(s)
        decay = torch.exp(-2 * integral_diff)

        p_same = (1 + decay) / 2
        p_diff = (1 - decay) / 2

        return p_same, p_diff

    def posterior_prob(
        self,
        x_t: torch.Tensor,
        x0_pred: torch.Tensor,
        t: torch.Tensor,
        s: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute q(X_s = 1 | X_t, x0_pred) using exact CTMC posterior.

        Args:
            x_t: Current state (binary)
            x0_pred: Predicted P(X_0 = 1 | X_t)
            t: Current time
            s: Target time (s < t)

        Returns:
            Posterior probability P(X_s = 1 | X_t, x0_pred)
        """
        x_t = x_t.float()

        # Transition probs for [0, s] and [s, t]
        p_same_0s, p_diff_0s = self.transition_probs(torch.zeros_like(s), s)
        p_same_st, p_diff_st = self.transition_probs(s, t)

        # Expand to match x_t shape
        while p_same_0s.dim() < x_t.dim():
            p_same_0s = p_same_0s.unsqueeze(-1)
            p_diff_0s = p_diff_0s.unsqueeze(-1)
            p_same_st = p_same_st.unsqueeze(-1)
            p_diff_st = p_diff_st.unsqueeze(-1)

        # P(X_t | X_s = k) for k in {0, 1}
        p_xt_given_xs1 = x_t * p_same_st + (1 - x_t) * p_diff_st
        p_xt_given_xs0 = x_t * p_diff_st + (1 - x_t) * p_same_st

        # E[P(X_s = k | X_0)] under x0_pred
        p_xs1_given_x0 = x0_pred * p_same_0s + (1 - x0_pred) * p_diff_0s
        p_xs0_given_x0 = x0_pred * p_diff_0s + (1 - x0_pred) * p_same_0s

        # Bayes' rule: q(X_s=1) = P(X_t|X_s=1) * P(X_s=1|X_0) / Z
        num_1 = p_xt_given_xs1 * p_xs1_given_x0
        num_0 = p_xt_given_xs0 * p_xs0_given_x0

        return num_1 / (num_0 + num_1 + self.epsilon)

    def sample(
        self,
        x_t: torch.Tensor,
        x0_pred: torch.Tensor,
        t: torch.Tensor,
        s: torch.Tensor,
        deterministic: bool = False
    ) -> torch.Tensor:
        """
        Sample X_s from exact posterior q(X_s | X_t, x0_pred).

        Args:
            x_t: Current state
            x0_pred: Predicted P(X_0 = 1 | X_t)
            t: Current time
            s: Target time (s < t)
            deterministic: Use argmax if True

        Returns:
            Sampled X_s
        """
        posterior = self.posterior_prob(x_t, x0_pred, t, s)

        if deterministic:
            return (posterior > 0.5).float()
        else:
            return torch.bernoulli(posterior.clamp(0, 1))
