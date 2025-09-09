"""Continuous-time categorical diffusion for EDISCO"""

import numpy as np
import torch
import torch.nn.functional as F
from scipy import linalg


class ContinuousTimeCategoricalDiffusion:
    """
    Continuous-time categorical diffusion based on CTMC (Continuous-Time Markov Chain) theory
    Implements score-based continuous-time discrete diffusion
    """
    
    def __init__(self, beta_min=0.1, beta_max=2.0, num_classes=2):
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.num_classes = num_classes
        self.eps = 1e-8
        
    def beta_t(self, t):
        """Linear noise schedule β(t)"""
        return self.beta_min + t * (self.beta_max - self.beta_min)
    
    def beta_integral(self, t, s=0.0):
        """∫_s^t β(u) du for the linear schedule"""
        delta_beta = self.beta_max - self.beta_min
        return self.beta_min * (t - s) + 0.5 * delta_beta * (t**2 - s**2)
    
    def rate_matrix(self, t):
        """
        Generator matrix Q(t) for the forward CTMC
        Q = β(t) * (1/K * 11^T - I)
        """
        beta = self.beta_t(t)
        K = self.num_classes
        Q = beta * (np.ones((K, K)) / K - np.eye(K))
        return Q
    
    def transition_matrix(self, t, s=0.0):
        """
        Transition probability matrix P(X_t | X_s) = exp(∫_s^t Q(u) du)
        Closed form for linear schedule with uniform target distribution
        """
        if isinstance(t, torch.Tensor):
            t = t.cpu().numpy()
        if isinstance(s, torch.Tensor):
            s = s.cpu().numpy()
            
        integral = self.beta_integral(t, s)
        K = self.num_classes
        
        # Closed form for uniform target: P_ij = 1/K + (δ_ij - 1/K) * exp(-K*integral)
        exp_term = np.exp(-K * integral)
        P = (1 - exp_term) / K * np.ones((K, K)) + exp_term * np.eye(K)
        
        return P
    
    def sample_forward(self, x0, t, device='cuda'):
        """
        Sample X_t | X_0 for dense adjacency matrices
        
        Args:
            x0: initial adjacency matrix (batch_size, n_nodes, n_nodes)
            t: time values (batch_size,) or scalar
            device: computation device
        
        Returns:
            xt: noisy adjacency matrix at time t
        """
        batch_size = x0.shape[0]
        n_nodes = x0.shape[1]
        
        # Flatten adjacency matrix to treat edges independently
        x0_flat = x0.reshape(batch_size, n_nodes * n_nodes)
        n_edges = x0_flat.shape[1]
        
        # Handle scalar or tensor time
        if not isinstance(t, torch.Tensor):
            t = torch.tensor([t] * batch_size, device=device)
        
        # Compute transition matrices for each time in batch
        P_list = []
        for i in range(batch_size):
            P = self.transition_matrix(t[i].item())
            P_list.append(torch.from_numpy(P).float())
        
        P_batch = torch.stack(P_list).to(device)  # (batch_size, K, K)
        
        # Get transition probabilities for each edge
        x0_idx = x0_flat.long()
        x0_expanded = x0_idx.unsqueeze(-1).expand(-1, -1, self.num_classes)
        P_expanded = P_batch.unsqueeze(1).expand(-1, n_edges, -1, -1)
        
        # Extract relevant transition probabilities
        trans_probs = torch.gather(P_expanded, 2, x0_expanded.unsqueeze(2)).squeeze(2)
        
        # Sample new states
        xt_flat = torch.multinomial(trans_probs.view(-1, self.num_classes), 1)
        xt_flat = xt_flat.view(batch_size, n_edges)
        
        # Reshape back to adjacency matrix
        xt = xt_flat.reshape(batch_size, n_nodes, n_nodes).float()
        
        return xt
    
    def sample_reverse_simple(self, xt, x0_logits, t, dt, device='cuda'):
        """
        Simplified reverse sampling - stable for training
        
        Args:
            xt: current state (batch_size, n_nodes, n_nodes)
            x0_logits: predicted X_0 logits (batch_size, n_nodes, n_nodes, num_classes)
            t: current time
            dt: time step (negative for reverse)
            device: computation device
        
        Returns:
            x_next: next state in reverse process
        """
        # Get predicted x0 probabilities
        x0_probs = F.softmax(x0_logits, dim=-1)
        
        # Near t=0, use deterministic transition
        if abs(dt) < 0.02 or t < 0.1:
            x0_pred = x0_probs.argmax(dim=-1)
            return x0_pred.float()
        
        # Get reverse transition matrix
        P_reverse = self.transition_matrix(t, t + dt)
        P_reverse = torch.from_numpy(P_reverse).float().to(device)
        
        # Mix current state with predicted x0 based on time
        mix_weight = t  # More diffusion early, more x0 prediction late
        
        # Compute transition probabilities
        xt_flat = xt.reshape(-1).long()
        xt_onehot = F.one_hot(xt_flat, num_classes=self.num_classes).float()
        
        # Diffusion-based transition
        trans_probs_diffusion = torch.matmul(xt_onehot, P_reverse)
        
        # Mix with predicted x0
        x0_probs_flat = x0_probs.reshape(-1, self.num_classes)
        trans_probs = mix_weight * trans_probs_diffusion + (1 - mix_weight) * x0_probs_flat
        
        # Normalize
        trans_probs = trans_probs / (trans_probs.sum(dim=-1, keepdim=True) + self.eps)
        
        # Sample next state
        x_next = torch.multinomial(trans_probs, 1).reshape(xt.shape)
        
        return x_next.float()
    
    def sample_reverse_pc(self, xt, x0_logits, t, dt, device='cuda', n_corrector_steps=1):
        """
        Predictor-corrector reverse sampling for higher quality
        
        Args:
            xt: current state
            x0_logits: predicted X_0 logits
            t: current time
            dt: time step
            device: computation device
            n_corrector_steps: number of corrector steps
        
        Returns:
            x_next: next state
        """
        # Predictor step (same as simple)
        x_pred = self.sample_reverse_simple(xt, x0_logits, t, dt, device)
        
        # Corrector steps (Langevin dynamics)
        for _ in range(n_corrector_steps):
            # Add small noise
            noise = torch.randn_like(x_pred) * 0.01
            x_pred = x_pred + noise
            x_pred = x_pred.clamp(0, self.num_classes - 1).round()
        
        return x_pred
    
    def elbo_loss(self, x0, xt, t, x0_pred_logits):
        """
        Simplified ELBO loss for continuous-time diffusion
        
        Args:
            x0: ground truth (batch_size, n_nodes, n_nodes)
            xt: noisy state (batch_size, n_nodes, n_nodes)
            t: time values (batch_size,)
            x0_pred_logits: predicted X_0 (batch_size, n_nodes, n_nodes, num_classes)
        
        Returns:
            loss: scalar loss value
        """
        # Flatten for cross-entropy computation
        x0_flat = x0.reshape(-1).long()
        x0_pred_flat = x0_pred_logits.reshape(-1, self.num_classes)
        
        # Cross-entropy loss
        reconstruction_loss = F.cross_entropy(
            x0_pred_flat,
            x0_flat,
            reduction='none'
        ).reshape(x0.shape)
        
        # Time-dependent weighting - emphasize reconstruction near t=0
        if torch.is_tensor(t):
            # Use sqrt for smoother weighting
            weight = 1.0 - torch.sqrt(t)
            # Expand weight to match loss shape
            while len(weight.shape) < len(reconstruction_loss.shape):
                weight = weight.unsqueeze(-1)
        else:
            weight = 1.0 - np.sqrt(t)
        
        # Weighted average
        loss = (reconstruction_loss * weight).mean()
        
        return loss
    
    def score_matching_loss(self, x0, xt, t, x0_pred_logits):
        """
        Score matching loss for continuous-time diffusion
        
        Args:
            x0: ground truth
            xt: noisy state
            t: time values
            x0_pred_logits: predicted X_0
        
        Returns:
            loss: scalar loss value
        """
        # Get predicted probabilities
        x0_pred_probs = F.softmax(x0_pred_logits, dim=-1)
        
        # True probabilities (one-hot)
        x0_onehot = F.one_hot(x0.long(), num_classes=self.num_classes).float()
        
        # Score = (x0 - xt) / t (simplified)
        t_safe = t.view(-1, 1, 1, 1).clamp(min=0.01)
        
        # Predicted score
        pred_score = (x0_pred_probs - F.one_hot(xt.long(), self.num_classes).float()) / t_safe
        
        # True score
        true_score = (x0_onehot - F.one_hot(xt.long(), self.num_classes).float()) / t_safe
        
        # MSE loss on scores
        loss = F.mse_loss(pred_score, true_score)
        
        return loss
    
    def get_prior_logits(self, shape, device):
        """
        Get prior distribution logits (uniform for categorical)
        
        Args:
            shape: shape of the output
            device: computation device
        
        Returns:
            logits: uniform prior logits
        """
        # Uniform prior
        logits = torch.zeros(*shape, self.num_classes, device=device)
        return logits