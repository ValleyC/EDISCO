"""Continuous-time categorical diffusion over binary variables.

Forward CTMC and training objective of EDISCO:

    Q(t)       = beta(t) (11^T - K I)
    P_ab(t|s)  = 1/K + (delta_ab - 1/K) exp(-K int_s^t beta(u) du)
    L(theta)   = E[(1 - sqrt(t)) * CE(s_theta(X_t, t), X_0)]

with K = 2 states per edge (or node) variable.
"""

import math

import torch
import torch.nn.functional as F

from diffusion.beta_schedules import BetaSchedule


class ContinuousTimeCategoricalDiffusion:
    """Categorical CTMC over binary edge (or node) variables."""

    num_classes = 2

    def __init__(self, beta_min=0.1, beta_max=1.5, schedule=None):
        """
        Args:
            beta_min, beta_max: end points of the linear noise-rate schedule
            schedule: BetaSchedule; overrides the linear schedule when given
        """
        self.schedule = schedule or BetaSchedule('linear', beta_min, beta_max)

    def beta_integral(self, t, s=0.0):
        """int_s^t beta(u) du."""
        return self.schedule.integral(t) - self.schedule.integral(s)

    def flip_probability(self, t, s=0.0):
        """P(X_t != X_s) = 1/2 - 1/2 exp(-2 int_s^t beta)."""
        integral = self.beta_integral(t, s)
        # expm1 avoids cancellation for short intervals near t = 0.
        if isinstance(integral, torch.Tensor):
            return -0.5 * torch.expm1(-2.0 * integral)
        return -0.5 * math.expm1(-2.0 * integral)

    def stay_probability(self, t, s=0.0):
        """P(X_t = a | X_s = a) = 1/2 + 1/2 exp(-2 int_s^t beta)."""
        return 1.0 - self.flip_probability(t, s)

    def sample_forward(self, x0, t):
        """Draw X_t ~ q(X_t | X_0) exactly from the closed-form kernel.

        Args:
            x0: clean binary state, e.g. (B, n, n), (B, E) or (E,)
            t: diffusion times: a scalar, one per batch element (B,), or one
               per variable (same shape as x0)
        Each variable flips independently with probability
        1/2 - 1/2 exp(-2 int_0^t beta).
        """
        x0 = x0.float()
        t = torch.as_tensor(t, device=x0.device, dtype=torch.float32)
        p_flip = self.flip_probability(t)
        if p_flip.numel() == 1:
            p_flip = p_flip.reshape(())
        elif p_flip.shape != x0.shape:
            if p_flip.dim() != 1 or p_flip.shape[0] != x0.shape[0]:
                raise ValueError('diffusion times must be a scalar, one per batch element, '
                                 'or one per variable')
            p_flip = p_flip.view(-1, *([1] * (x0.dim() - 1)))
        flip = torch.rand_like(x0) < p_flip
        return torch.where(flip, 1.0 - x0, x0)

    def loss(self, x0, t, x0_pred_logits):
        """(1 - sqrt(t))-weighted cross-entropy on the clean-state prediction.

        Args:
            x0: clean state with a leading batch dimension, (B, ...)
            t: one diffusion time per batch element, (B,)
            x0_pred_logits: logits of shape (B, ..., 2)
        The cross-entropy is averaged over the variables of each instance,
        weighted by (1 - sqrt(t)) for that instance, then averaged over the
        batch.
        """
        batch_size = x0.shape[0]
        target = x0.reshape(batch_size, -1).long()
        logits = x0_pred_logits.reshape(batch_size, -1, self.num_classes)
        if logits.shape[1] != target.shape[1]:
            raise ValueError('prediction and target shapes differ')
        ce = F.cross_entropy(logits.reshape(-1, self.num_classes),
                             target.reshape(-1), reduction='none')
        ce = ce.reshape(batch_size, -1).mean(dim=1)
        t = torch.as_tensor(t, device=ce.device, dtype=ce.dtype).reshape(-1)
        if t.shape[0] == 1:
            t = t.expand(batch_size)
        if t.shape[0] != batch_size:
            raise ValueError('one diffusion time per batch element is required')
        weight = 1.0 - torch.sqrt(t.clamp(min=0.0))
        return (weight * ce).mean()
