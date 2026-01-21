"""Continuous-time categorical diffusion for EDISCO."""

import numpy as np
import torch
import torch.nn.functional as F
import math


class ContinuousTimeCategoricalDiffusion:
    """Continuous-time categorical diffusion process."""

    def __init__(self, beta_min=0.1, beta_max=1.5, num_classes=2,
                 sparse=False, dense_only=False):
        """
        Args:
            beta_min: Minimum noise rate
            beta_max: Maximum noise rate
            num_classes: Number of categorical states (2 for binary edges)
            sparse: Whether using sparse graphs
            dense_only: Whether using only dense graphs
        """
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.num_classes = num_classes
        self.eps = 1e-8

        # Set execution mode at initialization
        self.sparse = sparse and not dense_only
        self.dense_only = dense_only or not sparse
        
    def beta_t(self, t):
        """Linear noise schedule β(t)"""
        if isinstance(t, torch.Tensor):
            return self.beta_min + t * (self.beta_max - self.beta_min)
        else:
            return self.beta_min + t * (self.beta_max - self.beta_min)
    
    def beta_integral(self, t, s=0.0):
        """∫_s^t β(u) du for the linear schedule"""
        if isinstance(t, torch.Tensor):
            t_val = t
            s_val = s if isinstance(s, torch.Tensor) else torch.tensor(s, device=t.device)
        else:
            t_val = t
            s_val = s
        
        delta_beta = self.beta_max - self.beta_min
        return self.beta_min * (t_val - s_val) + 0.5 * delta_beta * (t_val**2 - s_val**2)
    
    def transition_matrix(self, t, s=0.0, device='cuda'):
        """
        Transition probability matrix P(X_t | X_s) = exp(∫_s^t Q(u) du)
        """
        integral = self.beta_integral(t, s)
        K = self.num_classes
        
        if isinstance(integral, torch.Tensor):
            exp_term = torch.exp(-K * integral)
            P = (1 - exp_term) / K + exp_term
        else:
            exp_term = np.exp(-K * integral)
            P = (1 - exp_term) / K + exp_term
        
        return P
    
    def sample_forward(self, x0, t, device='cuda'):
        """Sample from forward diffusion process."""
        if self.dense_only:
            return self._sample_forward_dense(x0, t, device)
        elif self.sparse:
            return self._sample_forward_sparse(x0, t, device)
        else:
            return self._sample_forward_flexible(x0, t, device)

    def _sample_forward_dense(self, x0, t, device='cuda'):
        """Dense forward sampling (batch_size, n_nodes, n_nodes)."""
        x0 = x0.to(device)
        t = t.to(device)
        
        batch_size, n_nodes, _ = x0.shape
        
        # Vectorized computation for all batches at once
        # Compute transition probabilities for each batch element
        p_flips = torch.stack([
            torch.tensor(self.transition_matrix(t[b].item(), device=device), device=device)
            for b in range(batch_size)
        ])
        
        # Generate uniform noise for all elements
        uniform_noise = torch.rand_like(x0, dtype=torch.float32)
        
        # Vectorized flip operation
        x0_float = x0.float()
        flip_masks = uniform_noise < p_flips.view(batch_size, 1, 1)
        xt = torch.where(flip_masks, 1.0 - x0_float, x0_float)
        
        return xt
    
    def _sample_forward_sparse(self, x0, t, device='cuda'):
        """Sparse forward sampling (batch_size, edges_per_graph)."""
        x0 = x0.to(device)
        t = t.to(device)
        
        if x0.dim() == 1:
            # Single sparse graph - direct vectorized operation
            p_flip = self.transition_matrix(t[0] if t.dim() > 0 else t, device=device)
            uniform_noise = torch.rand_like(x0, dtype=torch.float32)
            x0_float = x0.float()
            flip_mask = uniform_noise < p_flip
            return torch.where(flip_mask, 1.0 - x0_float, x0_float)
        
        # Batched sparse graphs
        batch_size, edges_per_graph = x0.shape
        
        # Vectorized computation
        p_flips = torch.stack([
            torch.tensor(self.transition_matrix(t[b].item(), device=device), device=device)
            for b in range(batch_size)
        ])
        
        uniform_noise = torch.rand_like(x0, dtype=torch.float32)
        x0_float = x0.float()
        flip_masks = uniform_noise < p_flips.view(batch_size, 1)
        xt = torch.where(flip_masks, 1.0 - x0_float, x0_float)
        
        return xt
    
    def _sample_forward_flexible(self, x0, t, device='cuda'):
        """Flexible forward sampling with runtime checks."""
        x0 = x0.to(device)
        t = t.to(device) if isinstance(t, torch.Tensor) else torch.tensor(t, device=device)
        
        if t.dim() == 0:
            t = t.unsqueeze(0)
        
        # Original implementation with all checks
        if x0.dim() == 1:
            if t.shape[0] != 1:
                raise ValueError(f"Single sparse graph expects single time value")
            p_flip = self.transition_matrix(t[0], device=device)
            uniform_noise = torch.rand_like(x0, dtype=torch.float32)
            flip_mask = uniform_noise < p_flip
            xt = torch.where(flip_mask, 1.0 - x0.float(), x0.float())
            return xt
            
        elif x0.dim() == 2:
            if t.shape[0] == x0.shape[0] and t.shape[0] > 1:
                # Batched sparse
                batch_size = x0.shape[0]
                edges_per_graph = x0.shape[1]
                xt = torch.zeros_like(x0, dtype=torch.float32)
                for b in range(batch_size):
                    p_flip = self.transition_matrix(t[b], device=device)
                    uniform_noise = torch.rand(edges_per_graph, device=device)
                    flip_mask = uniform_noise < p_flip
                    x0_b_float = x0[b].float()
                    xt[b] = torch.where(flip_mask, 1.0 - x0_b_float, x0_b_float)
                return xt
            else:
                # Single dense
                x0_expanded = x0.unsqueeze(0)
                p_flip = self.transition_matrix(t[0], device=device)
                uniform_noise = torch.rand_like(x0_expanded[0], dtype=torch.float32)
                flip_mask = uniform_noise < p_flip
                x0_float = x0_expanded[0].float()
                xt = torch.where(flip_mask, 1.0 - x0_float, x0_float)
                return xt.float()
                
        elif x0.dim() == 3:
            # Dense batched
            batch_size = x0.shape[0]
            xt = torch.zeros_like(x0, dtype=torch.float32)
            for b in range(batch_size):
                t_b = t[b] if b < t.shape[0] else t[0]
                p_flip = self.transition_matrix(t_b, device=device)
                uniform_noise = torch.rand_like(x0[b], dtype=torch.float32)
                flip_mask = uniform_noise < p_flip
                x0_b_float = x0[b].float()
                xt[b] = torch.where(flip_mask, 1.0 - x0_b_float, x0_b_float)
            return xt
        else:
            raise ValueError(f"Unexpected x0 dimensions: {x0.shape}")
    
    def sample_forward_dense(self, x0, t, device='cuda'):
        """Public API for dense-only sampling (for explicit use)"""
        return self._sample_forward_dense(x0, t, device)
    
    def sample_forward_sparse(self, x0, t, device='cuda'):
        """Public API for sparse-only sampling (for explicit use)"""
        return self._sample_forward_sparse(x0, t, device)
    
    def elbo_loss(self, x0, xt, t, x0_pred_logits):
        """Compute ELBO loss."""
        if self.dense_only:
            return self._elbo_loss_dense(x0, xt, t, x0_pred_logits)
        else:
            return self._elbo_loss_flexible(x0, xt, t, x0_pred_logits)

    def _elbo_loss_dense(self, x0, xt, t, x0_pred_logits):
        """Dense loss computation."""
        # Direct reshape without checks
        batch_size = x0.shape[0]
        n_nodes = x0.shape[1]
        
        x0_flat = x0.reshape(batch_size, -1)
        x0_pred_logits_flat = x0_pred_logits.reshape(-1, self.num_classes)
        x0_flat_long = x0_flat.long().reshape(-1)
        
        # Direct cross entropy
        loss = F.cross_entropy(x0_pred_logits_flat, x0_flat_long, reduction='mean')
        return loss
    
    def _elbo_loss_flexible(self, x0, xt, t, x0_pred_logits):
        """Flexible loss computation with dimension checks."""
        # Flatten if needed
        if len(x0.shape) == 3:  # Dense adjacency
            x0 = x0.reshape(x0.shape[0], -1)
            xt = xt.reshape(xt.shape[0], -1)
            x0_pred_logits = x0_pred_logits.reshape(x0_pred_logits.shape[0], -1, self.num_classes)
        
        # Cross entropy loss
        loss = F.cross_entropy(
            x0_pred_logits.reshape(-1, self.num_classes),
            x0.long().reshape(-1),
            reduction='mean'
        )
        
        return loss
    
    def sample_reverse(self, xt, x0_pred_logits, t_current, t_next, device='cuda'):
        """Sample from reverse diffusion process."""
        if self.dense_only:
            return self._sample_reverse_dense(xt, x0_pred_logits, t_current, t_next, device)
        else:
            return self._sample_reverse_flexible(xt, x0_pred_logits, t_current, t_next, device)

    def sample_reverse_dense(self, xt, x0_pred_logits, t_current, t_next, device='cuda'):
        """Public API for dense-only reverse sampling."""
        return self._sample_reverse_dense(xt, x0_pred_logits, t_current, t_next, device)

    def _sample_reverse_dense(self, xt, x0_pred_logits, t_current, t_next, device='cuda'):
        """Dense reverse sampling."""
        # Direct softmax and extraction
        x0_pred_probs = F.softmax(x0_pred_logits, dim=-1)
        edge_probs = x0_pred_probs[..., 1]
        
        # Deterministic for small times
        if t_next < 0.01:
            return (edge_probs > 0.5).float()
        
        # Direct Bernoulli sampling
        return torch.bernoulli(edge_probs.clamp(0, 1))
    
    def _sample_reverse_flexible(self, xt, x0_pred_logits, t_current, t_next, device='cuda'):
        """Flexible reverse sampling with dimension checks."""
        x0_pred_probs = F.softmax(x0_pred_logits, dim=-1)
        
        if len(x0_pred_probs.shape) == 4:
            edge_probs = x0_pred_probs[..., 1]
        else:
            edge_probs = x0_pred_probs[..., 1]
        
        if t_next < 0.01:
            return (edge_probs > 0.5).float()
        
        return torch.bernoulli(edge_probs.clamp(0, 1))


class ContinuousTimeGaussianDiffusion:
    """
    Optimized continuous-time Gaussian diffusion.
    """
    
    def __init__(self, beta_min=0.1, beta_max=1.5, sparse=False, dense_only=False):
        """
        Args:
            beta_min: Minimum noise rate
            beta_max: Maximum noise rate
            sparse: Whether using sparse graphs
            dense_only: Whether using only dense graphs
        """
        self.beta_min = beta_min
        self.beta_max = beta_max
        
        # Set execution mode at initialization
        self.sparse = sparse and not dense_only
        self.dense_only = dense_only or not sparse
        
    def beta_t(self, t):
        """Linear noise schedule"""
        return self.beta_min + t * (self.beta_max - self.beta_min)
    
    def alpha_bar(self, t):
        """Cumulative product of alphas"""
        integral = self.beta_min * t + 0.5 * (self.beta_max - self.beta_min) * t**2
        return torch.exp(-integral)
    
    def sample_forward(self, x0, t, device='cuda'):
        """Sample from forward Gaussian diffusion."""
        if self.dense_only:
            return self._sample_forward_dense(x0, t, device)
        else:
            return self._sample_forward_flexible(x0, t, device)

    def _sample_forward_dense(self, x0, t, device='cuda'):
        """Dense Gaussian forward sampling."""
        batch_size, n_nodes, _ = x0.shape
        
        # Direct computation without checks
        alpha_bar = self.alpha_bar(t).view(batch_size, 1, 1)
        epsilon = torch.randn_like(x0, device=device)
        xt = torch.sqrt(alpha_bar) * x0 + torch.sqrt(1 - alpha_bar) * epsilon
        
        return xt
    
    def _sample_forward_flexible(self, x0, t, device='cuda'):
        """Flexible Gaussian sampling with dimension handling."""
        batch_size = x0.shape[0]
        alpha_bar = self.alpha_bar(t).view(batch_size, 1, 1)
        
        epsilon = torch.randn_like(x0, device=device)
        xt = torch.sqrt(alpha_bar) * x0 + torch.sqrt(1 - alpha_bar) * epsilon
        
        return xt
    
    def elbo_loss(self, x0, xt, t, pred):
        """Direct MSE loss - already optimized."""
        return F.mse_loss(pred, x0)
    
    def gaussian_posterior_dense(self, target_t, t, pred, xt):
        """Optimized Gaussian posterior for dense graphs."""
        # Direct computation for dense case
        alpha_bar_t = self.alpha_bar(t)
        alpha_bar_target = self.alpha_bar(target_t)
        
        # Simplified posterior computation
        posterior_mean = (torch.sqrt(alpha_bar_target) * pred + 
                         torch.sqrt(1 - alpha_bar_target) * xt)
        
        return posterior_mean
    
    def gaussian_posterior(self, target_t, t, pred, xt):
        """Standard Gaussian posterior."""
        if self.dense_only:
            return self.gaussian_posterior_dense(target_t, t, pred, xt)
        else:
            alpha_bar_t = self.alpha_bar(t)
            alpha_bar_target = self.alpha_bar(target_t)
            posterior_mean = (torch.sqrt(alpha_bar_target) * pred + 
                             torch.sqrt(1 - alpha_bar_target) * xt)
            return posterior_mean


class ContinuousTimeCategoricalDiffusionDense:
    """
    Pure dense-only implementation for maximum performance.
    No sparse support, no runtime checks.
    """
    
    def __init__(self, beta_min=0.1, beta_max=1.5, num_classes=2):
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.num_classes = num_classes
        self.eps = 1e-8
        
    def beta_t(self, t):
        """Linear noise schedule β(t)"""
        return self.beta_min + t * (self.beta_max - self.beta_min)
    
    def beta_integral(self, t, s=0.0):
        """∫_s^t β(u) du for the linear schedule"""
        t_val = t
        s_val = s if isinstance(s, torch.Tensor) else 0.0
        delta_beta = self.beta_max - self.beta_min
        return self.beta_min * (t_val - s_val) + 0.5 * delta_beta * (t_val**2 - s_val**2)
    
    def transition_matrix_batch(self, t_batch, device='cuda'):
        """Vectorized transition matrix computation for batch."""
        integrals = self.beta_integral(t_batch)
        K = self.num_classes
        exp_terms = torch.exp(-K * integrals)
        P = (1 - exp_terms) / K + exp_terms
        return P
    
    def sample_forward(self, x0, t, device='cuda'):
        """Direct dense sampling - no checks."""
        batch_size, n_nodes, _ = x0.shape
        
        # Vectorized transition probability computation
        p_flips = self.transition_matrix_batch(t, device)
        
        # Generate noise and apply flips
        uniform_noise = torch.rand_like(x0, dtype=torch.float32)
        x0_float = x0.float()
        flip_masks = uniform_noise < p_flips.view(batch_size, 1, 1)
        xt = torch.where(flip_masks, 1.0 - x0_float, x0_float)
        
        return xt
    
    def elbo_loss(self, x0, xt, t, x0_pred_logits):
        """Direct loss computation - no checks."""
        x0_flat = x0.reshape(-1)
        x0_pred_logits_flat = x0_pred_logits.reshape(-1, self.num_classes)
        return F.cross_entropy(x0_pred_logits_flat, x0_flat.long(), reduction='mean')
    
    def sample_reverse(self, xt, x0_pred_logits, t_current, t_next, device='cuda'):
        """Direct reverse sampling - no checks."""
        x0_pred_probs = F.softmax(x0_pred_logits, dim=-1)
        edge_probs = x0_pred_probs[..., 1]
        
        if t_next < 0.01:
            return (edge_probs > 0.5).float()
        
        return torch.bernoulli(edge_probs.clamp(0, 1))