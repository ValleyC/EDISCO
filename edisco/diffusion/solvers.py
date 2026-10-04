"""Reverse-time solvers for the categorical CTMC.

Every solver starts from X_{t_0} at t_0 = 1, walks the schedule
t_0 > t_1 > ... > t_T = 0 and returns the final clean-state probabilities
P(X_0 = 1), which are decoded once into a feasible solution.

Single-evaluation solvers (Euler, DDIM, DEIS-2, PNDM) call the score network
once per step: the clean-state predictions are optionally smoothed with
Adams-Bashforth coefficients, X_{t_{k+1}} is drawn from the exact posterior
q(X_{t_{k+1}} | X_{t_k}, x0_pred), and one last prediction is made at t_T.
They use T + 1 network evaluations. Multi-evaluation solvers (DPM-2, Heun,
RK4) call the network 2, 2 and 4 times per step and return the last combined
prediction.
"""

import math

import torch
import torch.nn.functional as F

from diffusion.exact_ctmc import ExactCTMCPosterior

# Adams-Bashforth coefficients, newest prediction first.
ADAMS_BASHFORTH = {
    1: (1.0,),
    2: (3 / 2, -1 / 2),
    3: (23 / 12, -16 / 12, 5 / 12),
    4: (55 / 24, -59 / 24, 37 / 24, -9 / 24),
}


def get_time_schedule(schedule='linear', num_steps=50):
    """Sampling times t_0 = 1 > t_1 > ... > t_T = 0."""
    if num_steps < 1:
        raise ValueError('num_steps must be positive')
    if schedule == 'linear':
        return torch.linspace(1.0, 0.0, num_steps + 1)
    if schedule == 'cosine':
        phase = torch.linspace(0.0, 1.0, num_steps + 1)
        return 0.5 * (1 + torch.cos(math.pi * phase))
    if schedule == 'quadratic':
        return torch.linspace(1.0, 0.0, num_steps + 1) ** 2
    raise ValueError(f"Unknown time schedule: {schedule}")


class BaseSolver:
    """Common interface: `sample(score_fn, x_T)` returns P(X_0 = 1)."""

    def __init__(self, num_steps=50, schedule=None, beta_min=0.1, beta_max=1.5):
        if num_steps < 1:
            raise ValueError('num_steps must be positive')
        self.num_steps = num_steps
        self.posterior = ExactCTMCPosterior(schedule, beta_min, beta_max)

    @staticmethod
    def predict(score_fn, x_t, t):
        """Clean-state probability P(X_0 = 1 | X_t) from the score network."""
        with torch.no_grad():
            logits = score_fn(x_t, float(t))
        return F.softmax(logits, dim=-1)[..., 1].clamp(0, 1)

    def _start(self, x_T, device, schedule):
        x_t = x_T.float() if device is None else x_T.to(device).float()
        return x_t, get_time_schedule(schedule, self.num_steps).to(x_t.device)

    def sample(self, score_fn, x_T, device=None, schedule='linear'):
        """
        Args:
            score_fn: callable (x_t, t) -> clean-state logits of shape (*x_t.shape, 2)
            x_T: initial binary state, uniform on {0, 1}
            schedule: spacing of the sampling times ('linear', 'cosine', 'quadratic')
        """
        raise NotImplementedError


class MultistepSolver(BaseSolver):
    """One network evaluation per step with Adams-Bashforth prediction smoothing."""

    order = 1
    deterministic = False

    def sample(self, score_fn, x_T, device=None, schedule='linear'):
        x_t, times = self._start(x_T, device, schedule)
        history = []
        for t, t_next in zip(times[:-1], times[1:]):
            history = (history + [self.predict(score_fn, x_t, t)])[-self.order:]
            coefficients = ADAMS_BASHFORTH[len(history)]
            x0_pred = sum(c * p for c, p in zip(coefficients, reversed(history))).clamp(0, 1)
            x_t = self.posterior.sample(x_t, x0_pred, t, t_next, deterministic=self.deterministic)
        # Final clean-state prediction at t_T.
        return self.predict(score_fn, x_t, times[-1])


class EulerSolver(MultistepSolver):
    """Ancestral sampling from the exact posterior without smoothing."""


class DDIMSolver(MultistepSolver):
    """Deterministic variant: every step takes the mode of the exact posterior."""

    deterministic = True


class DEISSolver(MultistepSolver):
    """DEIS-2: two-prediction smoothing, 3/2 x0^(k) - 1/2 x0^(k-1)."""

    order = 2


class PNDMSolver(MultistepSolver):
    """PNDM: four-prediction Adams-Bashforth smoothing."""

    order = 4


class DPMSolver(BaseSolver):
    """DPM-2: averages the predictions at the current time and the interval midpoint."""

    def sample(self, score_fn, x_T, device=None, schedule='linear'):
        x_t, times = self._start(x_T, device, schedule)
        for t, t_next in zip(times[:-1], times[1:]):
            t_mid = (t + t_next) / 2
            p_1 = self.predict(score_fn, x_t, t)
            x_mid = self.posterior.sample(x_t, p_1, t, t_mid)
            p_2 = self.predict(score_fn, x_mid, t_mid)
            x0_pred = (0.5 * (p_1 + p_2)).clamp(0, 1)
            x_t = self.posterior.sample(x_t, x0_pred, t, t_next)
        return x0_pred


class HeunSolver(BaseSolver):
    """Heun: averages a predictor at t_k and a corrector at t_{k+1}."""

    def sample(self, score_fn, x_T, device=None, schedule='linear'):
        x_t, times = self._start(x_T, device, schedule)
        for t, t_next in zip(times[:-1], times[1:]):
            p_1 = self.predict(score_fn, x_t, t)
            x_next = self.posterior.sample(x_t, p_1, t, t_next)
            p_2 = self.predict(score_fn, x_next, t_next)
            x0_pred = (0.5 * (p_1 + p_2)).clamp(0, 1)
            x_t = self.posterior.sample(x_t, x0_pred, t, t_next)
        return x0_pred


class RK4Solver(BaseSolver):
    """RK4: combines four predictions per step with weights (1, 2, 2, 1) / 6."""

    def sample(self, score_fn, x_T, device=None, schedule='linear'):
        x_t, times = self._start(x_T, device, schedule)
        for t, t_next in zip(times[:-1], times[1:]):
            t_mid = (t + t_next) / 2
            p_1 = self.predict(score_fn, x_t, t)
            p_2 = self.predict(score_fn, self.posterior.sample(x_t, p_1, t, t_mid), t_mid)
            p_3 = self.predict(score_fn, self.posterior.sample(x_t, p_2, t, t_mid), t_mid)
            p_4 = self.predict(score_fn, self.posterior.sample(x_t, p_3, t, t_next), t_next)
            x0_pred = ((p_1 + 2 * p_2 + 2 * p_3 + p_4) / 6).clamp(0, 1)
            x_t = self.posterior.sample(x_t, x0_pred, t, t_next)
        return x0_pred


SOLVERS = {
    'euler': EulerSolver,
    'ddim': DDIMSolver,
    'deis': DEISSolver,
    'pndm': PNDMSolver,
    'dpm2': DPMSolver,
    'heun': HeunSolver,
    'rk4': RK4Solver,
}


def get_solver(solver_type, num_steps=50, schedule=None, beta_min=0.1, beta_max=1.5):
    """Build a solver by name.

    Args:
        solver_type: one of 'euler', 'ddim', 'deis', 'pndm', 'dpm2', 'heun', 'rk4'
        num_steps: number of reverse steps T
        schedule: BetaSchedule of the forward process (default: linear from
            beta_min to beta_max); the posterior uses the same schedule as training
    """
    if solver_type not in SOLVERS:
        raise ValueError(f"Unknown solver type: {solver_type}. Choose from: {sorted(SOLVERS)}")
    return SOLVERS[solver_type](num_steps, schedule, beta_min, beta_max)
