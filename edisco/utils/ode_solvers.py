"""ODE solvers for continuous-time diffusion in EDISCO."""

import torch
import torch.nn.functional as F
import math
import numpy as np

from diffusion.exact_ctmc import ExactCTMCPosterior


def get_time_schedule(schedule='linear', num_steps=50):
    """Get time schedule for sampling."""
    if schedule == 'linear':
        return torch.linspace(1.0, 0.0, num_steps + 1)
    elif schedule == 'cosine':
        steps = torch.linspace(0, num_steps, num_steps + 1)
        alpha_bar = 0.5 * (1 + torch.cos((steps / num_steps + 0.008) / 1.008 * math.pi))
        return 1 - alpha_bar
    elif schedule == 'quadratic':
        return torch.linspace(1.0 ** 0.5, 0.0 ** 0.5, num_steps + 1) ** 2
    else:
        raise ValueError(f"Unknown schedule: {schedule}")


class BaseSolver:
    """Base class for ODE/SDE solvers"""
    
    def __init__(self, num_steps=50):
        self.num_steps = num_steps
    
    def get_timesteps(self, schedule='linear'):
        """Get timesteps for the given schedule"""
        return get_time_schedule(schedule, self.num_steps)
    
    def apply_adaptive_mixing(self, x_t, x0_pred, t, adaptive_mixing, deterministic_threshold):
        """
        Apply adaptive mixing strategy
        
        Args:
            x_t: Current state
            x0_pred: Predicted clean data (logits)
            t: Current time
            adaptive_mixing: Whether to use adaptive mixing
            deterministic_threshold: Threshold for deterministic decoding
        """
        if not adaptive_mixing:
            return x0_pred
        
        # For very small t, use deterministic
        if t < deterministic_threshold:
            if len(x0_pred.shape) == 4:  # Dense adjacency
                return x0_pred.argmax(dim=-1).float()
            else:  # Sparse
                return x0_pred.argmax(dim=-1).float()

        # Get probabilities from logits
        if len(x0_pred.shape) == 4:
            x0_probs = F.softmax(x0_pred, dim=-1)
        else:
            x0_probs = F.softmax(x0_pred, dim=-1)

        # Extract edge probabilities (class 1)
        edge_probs = x0_probs[..., 1]
        
        # Sample edges
        return torch.bernoulli(edge_probs.clamp(0, 1))


class EulerSolver(BaseSolver):
    """First-order Euler solver with exact CTMC posterior sampling."""

    def __init__(self, num_steps=50, beta_min=0.1, beta_max=1.5):
        super().__init__(num_steps)
        self.posterior = ExactCTMCPosterior(beta_min, beta_max)

    def sample(self, score_fn, x_T, device='cuda', schedule='linear',
              adaptive_mixing=True, deterministic_threshold=0.1, **kwargs):
        """Euler method with exact CTMC posterior sampling."""
        x_t = x_T.to(device).float()
        timesteps = self.get_timesteps(schedule).to(device)

        for i in range(len(timesteps) - 1):
            t = timesteps[i]
            t_next = timesteps[i + 1]

            # Get predicted x0
            with torch.no_grad():
                x0_logits = score_fn(x_t, t.item())
            x0_probs = F.softmax(x0_logits, dim=-1)
            x0_pred = x0_probs[..., 1].clamp(0, 1)

            # Sample from exact CTMC posterior
            use_deterministic = adaptive_mixing and t_next.item() < deterministic_threshold
            x_t = self.posterior.sample(x_t, x0_pred, t, t_next, deterministic=use_deterministic)

        # Final prediction
        with torch.no_grad():
            x0_logits = score_fn(x_t, 0.0)

        return x0_logits.argmax(dim=-1).float()


class DDIMSolver(BaseSolver):
    """DDIM solver with exact CTMC posterior sampling."""

    def __init__(self, num_steps=50, eta=0.0, beta_min=0.1, beta_max=1.5):
        super().__init__(num_steps)
        self.eta = eta  # eta=0 for deterministic, eta=1 for DDPM
        self.posterior = ExactCTMCPosterior(beta_min, beta_max)

    def sample(self, score_fn, x_T, device='cuda', schedule='linear',
              adaptive_mixing=True, deterministic_threshold=0.1, **kwargs):
        """DDIM sampling with exact CTMC posterior."""
        x_t = x_T.to(device).float()
        timesteps = self.get_timesteps(schedule).to(device)

        for i in range(len(timesteps) - 1):
            t = timesteps[i]
            t_next = timesteps[i + 1]

            # Get predicted x0
            with torch.no_grad():
                x0_logits = score_fn(x_t, t.item())
            x0_probs = F.softmax(x0_logits, dim=-1)
            x0_pred = x0_probs[..., 1].clamp(0, 1)

            # Sample from exact CTMC posterior (eta controls stochasticity)
            use_deterministic = (self.eta == 0) or (adaptive_mixing and t_next.item() < deterministic_threshold)
            x_t = self.posterior.sample(x_t, x0_pred, t, t_next, deterministic=use_deterministic)

        # Final prediction
        with torch.no_grad():
            x0_logits = score_fn(x_t, 0.0)

        return x0_logits.argmax(dim=-1).float()


class PNDMSolver(BaseSolver):
    """
    Pseudo Numerical Methods for Diffusion Models (PNDM)
    Uses multi-step prediction smoothing with exact CTMC posterior sampling.
    """

    def __init__(self, num_steps=50, order=4, beta_min=0.1, beta_max=1.5):
        super().__init__(num_steps)
        self.order = min(order, 4)
        self.posterior = ExactCTMCPosterior(beta_min, beta_max)

    def sample(self, score_fn, x_T, device='cuda', schedule='linear',
              adaptive_mixing=True, deterministic_threshold=0.1, **kwargs):
        """
        PNDM sampling with exact CTMC posterior.

        Multi-step method smooths x0 predictions using Adams-Bashforth coefficients,
        then samples from exact posterior q(X_s | X_t, x0_pred).
        """
        x_t = x_T.to(device).float()
        timesteps = self.get_timesteps(schedule).to(device)

        # History for multi-step prediction smoothing
        x0_history = []

        for i in range(len(timesteps) - 1):
            t = timesteps[i]
            t_next = timesteps[i + 1]

            # Get predicted x0 probabilities
            with torch.no_grad():
                x0_logits = score_fn(x_t, t.item())

            x0_probs = F.softmax(x0_logits, dim=-1)
            x0_history.append(x0_probs)

            # Keep only recent history
            if len(x0_history) > self.order:
                x0_history.pop(0)

            # Multi-step prediction smoothing (Adams-Bashforth coefficients)
            if len(x0_history) == 1:
                combined_probs = x0_probs
            elif len(x0_history) == 2:
                combined_probs = 1.5 * x0_history[-1] - 0.5 * x0_history[-2]
            elif len(x0_history) == 3:
                combined_probs = (23 * x0_history[-1] - 16 * x0_history[-2] +
                                 5 * x0_history[-3]) / 12
            else:
                combined_probs = (55 * x0_history[-1] - 59 * x0_history[-2] +
                                 37 * x0_history[-3] - 9 * x0_history[-4]) / 24

            # Extract smoothed x0 prediction
            x0_pred = combined_probs[..., 1].clamp(0, 1)

            # Sample from exact CTMC posterior q(X_s | X_t, x0_pred)
            use_deterministic = adaptive_mixing and t_next.item() < deterministic_threshold
            x_t = self.posterior.sample(x_t, x0_pred, t, t_next, deterministic=use_deterministic)

        # Final prediction
        with torch.no_grad():
            x0_logits = score_fn(x_t, 0.0)

        return x0_logits.argmax(dim=-1).float()


class DPMSolver(BaseSolver):
    """DPM-Solver with exact CTMC posterior sampling."""

    def __init__(self, num_steps=50, order=2, beta_min=0.1, beta_max=1.5):
        super().__init__(num_steps)
        self.order = order
        self.posterior = ExactCTMCPosterior(beta_min, beta_max)

    def sample(self, score_fn, x_T, device='cuda', schedule='linear',
              adaptive_mixing=True, deterministic_threshold=0.1, **kwargs):
        """DPM-Solver sampling with exact CTMC posterior."""
        x_t = x_T.to(device).float()
        timesteps = self.get_timesteps(schedule).to(device)

        for i in range(len(timesteps) - 1):
            t = timesteps[i]
            t_next = timesteps[i + 1]

            # First-order prediction
            with torch.no_grad():
                x0_logits_1 = score_fn(x_t, t.item())
            x0_probs_1 = F.softmax(x0_logits_1, dim=-1)

            if self.order >= 2 and i < len(timesteps) - 2:
                # Second-order correction at midpoint
                t_mid = (t + t_next) / 2
                x0_pred_mid = x0_probs_1[..., 1].clamp(0, 1)
                x_mid = self.posterior.sample(x_t, x0_pred_mid, t, t_mid, deterministic=False)

                with torch.no_grad():
                    x0_logits_2 = score_fn(x_mid, t_mid.item())
                x0_probs_2 = F.softmax(x0_logits_2, dim=-1)

                # Combine predictions
                combined_probs = 0.5 * (x0_probs_1 + x0_probs_2)
            else:
                combined_probs = x0_probs_1

            # Sample from exact CTMC posterior
            x0_pred = combined_probs[..., 1].clamp(0, 1)
            use_deterministic = adaptive_mixing and t_next.item() < deterministic_threshold
            x_t = self.posterior.sample(x_t, x0_pred, t, t_next, deterministic=use_deterministic)

        # Final prediction
        with torch.no_grad():
            x0_logits = score_fn(x_t, 0.0)

        return x0_logits.argmax(dim=-1).float()


class DEISSolver(BaseSolver):
    """Diffusion Exponential Integrator Sampler (DEIS) with exact CTMC posterior."""

    def __init__(self, num_steps=50, order=2, beta_min=0.1, beta_max=1.5):
        super().__init__(num_steps)
        self.order = order
        self.posterior = ExactCTMCPosterior(beta_min, beta_max)

    def sample(self, score_fn, x_T, device='cuda', schedule='linear',
              adaptive_mixing=True, deterministic_threshold=0.1, **kwargs):
        """
        DEIS sampling with exact CTMC posterior.
        """
        x_t = x_T.to(device).float()
        timesteps = self.get_timesteps(schedule).to(device)

        for i in range(len(timesteps) - 1):
            t = timesteps[i]
            t_next = timesteps[i + 1]

            # Get predicted x0
            with torch.no_grad():
                x0_logits = score_fn(x_t, t.item())

            x0_probs = F.softmax(x0_logits, dim=-1)
            x0_pred = x0_probs[..., 1]

            # Sample from exact CTMC posterior
            use_deterministic = adaptive_mixing and t_next.item() < deterministic_threshold
            x_t = self.posterior.sample(x_t, x0_pred, t, t_next, deterministic=use_deterministic)

        # Final prediction
        with torch.no_grad():
            x0_logits = score_fn(x_t, 0.0)

        return x0_logits.argmax(dim=-1).float()


class RK4Solver(BaseSolver):
    """Fourth-order Runge-Kutta solver with exact CTMC posterior."""

    def __init__(self, num_steps=50, beta_min=0.1, beta_max=1.5):
        super().__init__(num_steps)
        self.posterior = ExactCTMCPosterior(beta_min, beta_max)

    def sample(self, score_fn, x_T, device='cuda', schedule='linear',
              adaptive_mixing=True, deterministic_threshold=0.1, **kwargs):
        """RK4 sampling with exact CTMC posterior."""
        x_t = x_T.to(device).float()
        timesteps = self.get_timesteps(schedule).to(device)

        for i in range(len(timesteps) - 1):
            t = timesteps[i]
            t_next = timesteps[i + 1]
            t_mid = (t + t_next) / 2

            # k1 at current time
            with torch.no_grad():
                k1_logits = score_fn(x_t, t.item())
            k1_probs = F.softmax(k1_logits, dim=-1)

            # k2 at midpoint
            x0_pred_1 = k1_probs[..., 1].clamp(0, 1)
            x_mid1 = self.posterior.sample(x_t, x0_pred_1, t, t_mid, deterministic=False)
            with torch.no_grad():
                k2_logits = score_fn(x_mid1, t_mid.item())
            k2_probs = F.softmax(k2_logits, dim=-1)

            # k3 at midpoint with k2
            x0_pred_2 = k2_probs[..., 1].clamp(0, 1)
            x_mid2 = self.posterior.sample(x_t, x0_pred_2, t, t_mid, deterministic=False)
            with torch.no_grad():
                k3_logits = score_fn(x_mid2, t_mid.item())
            k3_probs = F.softmax(k3_logits, dim=-1)

            # k4 at endpoint
            x0_pred_3 = k3_probs[..., 1].clamp(0, 1)
            x_end = self.posterior.sample(x_t, x0_pred_3, t, t_next, deterministic=False)
            with torch.no_grad():
                k4_logits = score_fn(x_end, t_next.item())
            k4_probs = F.softmax(k4_logits, dim=-1)

            # Combine with RK4 weights
            combined_probs = (k1_probs + 2*k2_probs + 2*k3_probs + k4_probs) / 6
            x0_pred = combined_probs[..., 1].clamp(0, 1)

            # Sample from exact CTMC posterior
            use_deterministic = adaptive_mixing and t_next.item() < deterministic_threshold
            x_t = self.posterior.sample(x_t, x0_pred, t, t_next, deterministic=use_deterministic)

        # Final prediction
        with torch.no_grad():
            x0_logits = score_fn(x_t, 0.0)

        return x0_logits.argmax(dim=-1).float()


class HeunSolver(BaseSolver):
    """Heun's method (Improved Euler / RK2) with exact CTMC posterior."""

    def __init__(self, num_steps=50, beta_min=0.1, beta_max=1.5):
        super().__init__(num_steps)
        self.posterior = ExactCTMCPosterior(beta_min, beta_max)

    def sample(self, score_fn, x_T, device='cuda', schedule='linear',
              adaptive_mixing=True, deterministic_threshold=0.1, **kwargs):
        """Heun's method with exact CTMC posterior."""
        x_t = x_T.to(device).float()
        timesteps = self.get_timesteps(schedule).to(device)

        for i in range(len(timesteps) - 1):
            t = timesteps[i]
            t_next = timesteps[i + 1]

            # Predictor step
            with torch.no_grad():
                x0_logits_pred = score_fn(x_t, t.item())
            x0_probs_pred = F.softmax(x0_logits_pred, dim=-1)

            # Get predicted next state via posterior
            x0_pred_1 = x0_probs_pred[..., 1].clamp(0, 1)
            x_pred = self.posterior.sample(x_t, x0_pred_1, t, t_next, deterministic=False)

            # Corrector step
            with torch.no_grad():
                x0_logits_corr = score_fn(x_pred, t_next.item())
            x0_probs_corr = F.softmax(x0_logits_corr, dim=-1)

            # Average predictor and corrector
            combined_probs = 0.5 * (x0_probs_pred + x0_probs_corr)
            x0_pred = combined_probs[..., 1].clamp(0, 1)

            # Sample from exact CTMC posterior
            use_deterministic = adaptive_mixing and t_next.item() < deterministic_threshold
            x_t = self.posterior.sample(x_t, x0_pred, t, t_next, deterministic=use_deterministic)

        # Final prediction
        with torch.no_grad():
            x0_logits = score_fn(x_t, 0.0)

        return x0_logits.argmax(dim=-1).float()


def get_solver(solver_type, num_steps=50, **kwargs):
    """
    Factory function to get the appropriate solver
    
    Args:
        solver_type: Type of solver ('euler', 'ddim', 'pndm', 'dpm2', 'deis', 'rk4', 'heun')
        num_steps: Number of steps for sampling
        **kwargs: Additional solver-specific arguments
    
    Returns:
        solver: Instance of the requested solver
    """
    solver_map = {
        'euler': lambda n: EulerSolver(n, beta_min=kwargs.get('beta_min', 0.1),
                                        beta_max=kwargs.get('beta_max', 1.5)),
        'ddim': lambda n: DDIMSolver(n, eta=kwargs.get('eta', 0.0),
                                      beta_min=kwargs.get('beta_min', 0.1),
                                      beta_max=kwargs.get('beta_max', 1.5)),
        'pndm': lambda n: PNDMSolver(n, order=kwargs.get('order', 4),
                                      beta_min=kwargs.get('beta_min', 0.1),
                                      beta_max=kwargs.get('beta_max', 1.5)),
        'dpm': lambda n: DPMSolver(n, order=kwargs.get('order', 2),
                                   beta_min=kwargs.get('beta_min', 0.1),
                                   beta_max=kwargs.get('beta_max', 1.5)),
        'dpm2': lambda n: DPMSolver(n, order=2,
                                    beta_min=kwargs.get('beta_min', 0.1),
                                    beta_max=kwargs.get('beta_max', 1.5)),
        'dpm3': lambda n: DPMSolver(n, order=3,
                                    beta_min=kwargs.get('beta_min', 0.1),
                                    beta_max=kwargs.get('beta_max', 1.5)),
        'deis': lambda n: DEISSolver(n, order=kwargs.get('order', 2),
                                     beta_min=kwargs.get('beta_min', 0.1),
                                     beta_max=kwargs.get('beta_max', 1.5)),
        'rk4': lambda n: RK4Solver(n, beta_min=kwargs.get('beta_min', 0.1),
                                   beta_max=kwargs.get('beta_max', 1.5)),
        'heun': lambda n: HeunSolver(n, beta_min=kwargs.get('beta_min', 0.1),
                                     beta_max=kwargs.get('beta_max', 1.5)),
    }
    
    if solver_type not in solver_map:
        raise ValueError(f"Unknown solver type: {solver_type}. "
                        f"Choose from: {list(solver_map.keys())}")
    
    solver_cls = solver_map[solver_type]
    if callable(solver_cls):
        return solver_cls(num_steps)
    else:
        return solver_cls(num_steps)