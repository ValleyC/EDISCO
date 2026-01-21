"""
Continuous-Time Categorical Diffusion for EDISCO
Based on Campbell et al., 2022 - Continuous Time Markov Chains
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


class ContinuousTimeCategoricalDiffusion(nn.Module):
    """
    Continuous-time categorical diffusion process using CTMCs
    For TSP, we have K=2 categories (edge exists or not)
    """
    
    def __init__(
        self,
        num_classes: int = 2,
        beta_min: float = 0.1,
        beta_max: float = 1.5,
        epsilon: float = 1e-8
    ):
        super().__init__()
        self.num_classes = num_classes
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.epsilon = epsilon
        
    def beta_schedule(self, t: torch.Tensor) -> torch.Tensor:
        """
        Linear noise schedule: β(t) = β_min + t(β_max - β_min)
        
        Args:
            t: Time values in [0, 1]
            
        Returns:
            Beta values at time t
        """
        return self.beta_min + t * (self.beta_max - self.beta_min)
    
    def integral_beta(self, t: torch.Tensor) -> torch.Tensor:
        """
        Integral of beta from 0 to t for linear schedule
        ∫β(u)du = β_min*t + 0.5*(β_max - β_min)*t^2
        
        Args:
            t: Time values in [0, 1]
            
        Returns:
            Integral of beta from 0 to t
        """
        return self.beta_min * t + 0.5 * (self.beta_max - self.beta_min) * t ** 2
    
    def transition_probability(self, t: torch.Tensor, s: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute transition probability P(X_t | X_s) for categorical diffusion
        For K=2: P_ij(t|s) = 1/2 + (δ_ij - 1/2) * exp(-2 * ∫_s^t β(u)du)
        
        Args:
            t: Target time
            s: Source time (default 0)
            
        Returns:
            Transition probability matrix
        """
        if s is None:
            s = torch.zeros_like(t)
            
        # Compute integral difference
        integral_diff = self.integral_beta(t) - self.integral_beta(s)
        
        # For binary case (K=2)
        decay = torch.exp(-2 * integral_diff)
        
        # Transition probability matrix (same class vs different class)
        p_same = 0.5 + 0.5 * decay
        p_diff = 0.5 - 0.5 * decay
        
        return p_same, p_diff
    
    def sample_forward(
        self, 
        x0: torch.Tensor, 
        t: torch.Tensor,
        device: str = 'cuda'
    ) -> torch.Tensor:
        """
        Sample X_t given X_0 using closed-form transition probabilities
        
        Args:
            x0: Clean data (binary adjacency matrix) shape: (batch_size, n_edges) or (batch_size, n, n)
            t: Time values (batch_size,)
            device: Device to use
            
        Returns:
            Noisy sample X_t
        """
        x0 = x0.to(device)
        original_shape = x0.shape
        
        # Flatten if needed
        if x0.dim() > 2:
            batch_size = x0.shape[0]
            x0_flat = x0.reshape(batch_size, -1)
        else:
            x0_flat = x0
            
        # Get transition probabilities
        p_same, p_diff = self.transition_probability(t.to(device))
        
        # Expand probabilities to match x0 shape
        p_same = p_same.unsqueeze(-1).expand_as(x0_flat)
        p_diff = p_diff.unsqueeze(-1).expand_as(x0_flat)
        
        # Sample using Gumbel trick for categorical
        # Use float() to avoid tensor type errors
        uniform_noise = torch.rand_like(x0_flat.float(), device=device)
        
        # For binary case: if x0=0, flip with prob p_diff; if x0=1, flip with prob p_diff
        flip_mask = uniform_noise < p_diff
        xt = torch.where(flip_mask, 1 - x0_flat, x0_flat)
        
        # Reshape back to original
        if len(original_shape) > 2:
            xt = xt.reshape(original_shape)
            
        return xt.long()
    
    def posterior_probability(
        self,
        xt: torch.Tensor,
        x0: torch.Tensor,
        t: torch.Tensor,
        s: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute posterior q(X_s | X_t, X_0) using Bayes rule
        
        Args:
            xt: Noisy data at time t
            x0: Clean data prediction
            t: Current time
            s: Target time (s < t for reverse)
            
        Returns:
            Posterior probability
        """
        # Get transition probabilities
        p_t_given_0_same, p_t_given_0_diff = self.transition_probability(t)
        p_s_given_0_same, p_s_given_0_diff = self.transition_probability(s)
        p_t_given_s_same, p_t_given_s_diff = self.transition_probability(t, s)
        
        # Compute posterior using Bayes rule
        # q(x_s | x_t, x_0) ∝ q(x_t | x_s) * q(x_s | x_0)
        
        # For binary case, compute probabilities for each configuration
        # This is simplified for the binary TSP case
        posterior_same = (p_t_given_s_same * p_s_given_0_same) / (p_t_given_0_same + self.epsilon)
        posterior_diff = (p_t_given_s_diff * p_s_given_0_diff) / (p_t_given_0_diff + self.epsilon)
        
        return posterior_same, posterior_diff
    
    def elbo_loss(
        self,
        x0: torch.Tensor,
        xt: torch.Tensor,
        t: torch.Tensor,
        x0_pred_logits: torch.Tensor,
        weight_by_t: bool = True
    ) -> torch.Tensor:
        """
        Compute ELBO loss for continuous-time categorical diffusion
        Loss = E[-log p(x0 | x0_pred)] with time-dependent weighting
        
        Args:
            x0: Clean data (ground truth)
            xt: Noisy data at time t  
            t: Time values
            x0_pred_logits: Predicted logits for x0 from score network
            weight_by_t: Whether to apply time-dependent weighting
            
        Returns:
            ELBO loss
        """
        # Flatten inputs if needed
        original_shape = x0.shape
        if x0.dim() > 2:
            batch_size = x0.shape[0]
            x0 = x0.reshape(batch_size, -1)
            xt = xt.reshape(batch_size, -1)
            
        # Handle logits shape
        if x0_pred_logits.dim() > 3:
            # (batch, n, n, classes) -> (batch, n*n, classes)
            x0_pred_logits = x0_pred_logits.reshape(batch_size, -1, self.num_classes)
        elif x0_pred_logits.dim() == 2 and x0.dim() == 2:
            # Sparse case: (n_edges, classes)
            x0_pred_logits = x0_pred_logits.unsqueeze(0)
            
        # Compute cross-entropy loss
        if x0_pred_logits.shape[-1] == self.num_classes:
            # Categorical cross-entropy
            loss = F.cross_entropy(
                x0_pred_logits.reshape(-1, self.num_classes),
                x0.long().reshape(-1),
                reduction='none'
            )
            loss = loss.reshape(x0.shape[0], -1).mean(dim=1)
        else:
            # Binary cross-entropy (fallback)
            x0_pred_probs = torch.sigmoid(x0_pred_logits.squeeze(-1))
            loss = F.binary_cross_entropy(x0_pred_probs, x0.float(), reduction='none')
            loss = loss.mean(dim=-1)
            
        # Apply time-dependent weighting: (1 - sqrt(t))
        if weight_by_t:
            weight = 1.0 - torch.sqrt(t)
            loss = loss * weight
            
        return loss.mean()
    
    def sample_reverse_step(
        self,
        xt: torch.Tensor,
        x0_pred_probs: torch.Tensor,
        t_current: torch.Tensor,
        t_next: torch.Tensor,
        deterministic: bool = False
    ) -> torch.Tensor:
        """
        Sample one reverse diffusion step: X_{t-dt} ~ q(X_{t-dt} | X_t, X_0_pred)
        
        Args:
            xt: Current noisy state
            x0_pred_probs: Predicted probabilities for clean data
            t_current: Current time
            t_next: Next time (t_next < t_current for reverse)
            deterministic: If True, use argmax instead of sampling
            
        Returns:
            X at next timestep
        """
        # Get posterior probabilities
        p_same, p_diff = self.posterior_probability(xt, x0_pred_probs, t_current, t_next)
        
        if deterministic or t_next < 0.1:  # Deterministic for final steps
            # Use argmax
            x_next = (x0_pred_probs > 0.5).long()
        else:
            # Sample from posterior
            uniform_noise = torch.rand_like(xt.float())
            
            # Adaptive mixing based on time
            mix_weight = t_next  # More stochastic early, more deterministic late
            
            # Mix between stochastic and deterministic
            flip_prob = mix_weight * p_diff + (1 - mix_weight) * (x0_pred_probs < 0.5).float()
            flip_mask = uniform_noise < flip_prob
            x_next = torch.where(flip_mask, 1 - xt, xt)
            
        return x_next


class AdaptiveTimeSchedule:
    """
    Adaptive time scheduling for inference
    Allows flexible step sizes based on solver type
    """
    
    def __init__(
        self,
        num_steps: int = 50,
        schedule_type: str = 'linear',
        t_start: float = 1.0,
        t_end: float = 0.0
    ):
        self.num_steps = num_steps
        self.schedule_type = schedule_type
        self.t_start = t_start
        self.t_end = t_end
        
    def get_schedule(self) -> np.ndarray:
        """Get time schedule for inference"""
        if self.schedule_type == 'linear':
            return np.linspace(self.t_start, self.t_end, self.num_steps + 1)
        elif self.schedule_type == 'cosine':
            # Cosine schedule (more steps near t=0)
            t = np.linspace(0, np.pi, self.num_steps + 1)
            schedule = (np.cos(t) + 1) / 2
            return self.t_start + (self.t_end - self.t_start) * (1 - schedule)
        elif self.schedule_type == 'quadratic':
            # Quadratic schedule (even more steps near t=0)
            t = np.linspace(0, 1, self.num_steps + 1)
            schedule = t ** 2
            return self.t_start + (self.t_end - self.t_start) * (1 - schedule)
        else:
            raise ValueError(f"Unknown schedule type: {self.schedule_type}")