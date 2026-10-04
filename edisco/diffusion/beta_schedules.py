"""Noise-rate schedules beta(t) of the CTMC forward process.

The forward transition probabilities depend on the schedule only through the
integral int_s^t beta(u) du. Three families are provided:

    linear       beta(t) = beta_min + t (beta_max - beta_min)
    exponential  beta(t) = a b^t log(b)
    cosine       beta(t) = clip(pi/4 * tan(pi t / 2) / sqrt(cos(pi t / 2) + eps), beta_min, beta_max)

The linear schedule with beta_min = 0.1 and beta_max = 1.5 is the default.
"""

import math

import numpy as np
import torch


class BetaSchedule:
    """beta(t) on t in [0, 1] and its integral from 0 to t."""

    def __init__(self, kind='linear', beta_min=0.1, beta_max=1.5, a=0.5, b=4.0,
                 eps=1e-8, grid_size=20001):
        if kind not in ('linear', 'exponential', 'cosine'):
            raise ValueError(f"Unknown beta schedule: {kind}")
        if kind == 'exponential' and (a <= 0 or b <= 1):
            raise ValueError("the exponential schedule needs a > 0 and b > 1")
        self.kind = kind
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.a = a
        self.b = b
        self.eps = eps
        if kind == 'cosine':
            # The clipped schedule has no closed-form integral: tabulate it.
            grid = np.linspace(0.0, 1.0, grid_size)
            beta = self._cosine_beta(grid)
            steps = 0.5 * (beta[1:] + beta[:-1]) * np.diff(grid)
            self._grid = grid
            self._cumulative = np.concatenate([[0.0], np.cumsum(steps)])

    def _cosine_beta(self, t):
        phase = np.pi * np.asarray(t, dtype=np.float64) / 2.0
        raw = np.pi / 4.0 * np.tan(phase) / np.sqrt(np.clip(np.cos(phase), 0.0, None) + self.eps)
        return np.clip(raw, self.beta_min, self.beta_max)

    def beta(self, t):
        """Instantaneous noise rate beta(t)."""
        if self.kind == 'linear':
            return self.beta_min + t * (self.beta_max - self.beta_min)
        if self.kind == 'exponential':
            return self.a * self.b ** t * math.log(self.b)
        if isinstance(t, torch.Tensor):
            return torch.as_tensor(self._cosine_beta(t.detach().cpu().numpy()), dtype=t.dtype, device=t.device)
        return float(self._cosine_beta(t))

    def integral(self, t):
        """int_0^t beta(u) du for a float or a tensor of times."""
        if self.kind == 'linear':
            return self.beta_min * t + 0.5 * (self.beta_max - self.beta_min) * t ** 2
        if self.kind == 'exponential':
            return self.a * (self.b ** t - 1.0)
        if isinstance(t, torch.Tensor):
            cumulative = torch.as_tensor(self._cumulative, dtype=t.dtype, device=t.device)
            position = t.clamp(0.0, 1.0) * (len(self._grid) - 1)
            lower = position.floor().long().clamp(max=len(self._grid) - 2)
            fraction = position - lower.to(t.dtype)
            return cumulative[lower] * (1.0 - fraction) + cumulative[lower + 1] * fraction
        return float(np.interp(t, self._grid, self._cumulative))

    @classmethod
    def from_args(cls, args):
        return cls(kind=getattr(args, 'beta_schedule', 'linear'),
                   beta_min=getattr(args, 'beta_min', 0.1),
                   beta_max=getattr(args, 'beta_max', 1.5),
                   a=getattr(args, 'beta_exp_a', 0.5),
                   b=getattr(args, 'beta_exp_b', 4.0))
