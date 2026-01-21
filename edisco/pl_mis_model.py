"""PyTorch Lightning module for EDISCO MIS model.

This module implements continuous-time diffusion for the Maximum Independent Set
problem using EDISCO's ODE solvers (PNDM, DEIS, etc.).

Note: MIS is NOT a geometric problem, so we use the standard GNN encoder (not EGNN)
while still benefiting from continuous-time diffusion.
"""

import os

import numpy as np
import scipy.sparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from pytorch_lightning.utilities import rank_zero_info

from co_datasets.mis_dataset import MISDataset
from pl_meta_model import COMetaModel
from utils.mis_utils import mis_decode_np
from utils.ode_solvers import get_solver, get_time_schedule


class MISModel(COMetaModel):
    """EDISCO model for Maximum Independent Set using continuous-time diffusion.

    Uses EDISCO's ODE solvers (PNDM, DEIS, etc.) for inference.
    Note: MIS is NOT a geometric problem, so we use the standard GNN encoder (not EGNN).
    """

    def __init__(self, param_args=None):
        # Force settings for MIS: use GNN (not EGNN), node features only
        param_args.equivariant = False  # MIS is not geometric
        param_args.sparse_factor = -1  # Always sparse for MIS graphs

        super(MISModel, self).__init__(param_args=param_args, node_feature_only=True)

        rank_zero_info("MISModel: Using standard GNN (not EGNN) - MIS is not a geometric problem")
        rank_zero_info(f"MISModel: Using EDISCO continuous-time diffusion with {self.args.solver_type} solver")

        # Load datasets
        data_label_dir = None
        if hasattr(self.args, 'training_split_label_dir') and self.args.training_split_label_dir is not None:
            data_label_dir = os.path.join(self.args.storage_path, self.args.training_split_label_dir)

        self.train_dataset = MISDataset(
            data_file=os.path.join(self.args.storage_path, self.args.training_split),
            data_label_dir=data_label_dir,
        )

        self.test_dataset = MISDataset(
            data_file=os.path.join(self.args.storage_path, self.args.test_split),
        )

        self.validation_dataset = MISDataset(
            data_file=os.path.join(self.args.storage_path, self.args.validation_split),
        )

        # Get solver parameters
        self.solver_type = getattr(self.args, 'solver_type', 'pndm')
        self.solver_steps = getattr(self.args, 'solver_steps', 50)
        self.time_schedule = getattr(self.args, 'time_schedule', 'linear')
        self.beta_min = getattr(self.args, 'beta_min', 0.1)
        self.beta_max = getattr(self.args, 'beta_max', 1.5)
        self.adaptive_mixing = getattr(self.args, 'adaptive_mixing', True)
        self.deterministic_threshold = getattr(self.args, 'deterministic_threshold', 0.1)

    def forward(self, x, t, edge_index):
        """Forward pass through the GNN encoder.

        Args:
            x: Node features (num_nodes,)
            t: Timesteps (num_nodes,)
            edge_index: Edge indices (2, num_edges)

        Returns:
            Node predictions (num_nodes, 2) for categorical or (num_nodes, 1) for gaussian
        """
        return self.model(x, t, edge_index=edge_index)

    def _beta_integral(self, t, s=0.0):
        """Integral of beta from s to t for continuous-time diffusion."""
        delta_beta = self.beta_max - self.beta_min
        return self.beta_min * (t - s) + 0.5 * delta_beta * (t**2 - s**2)

    def _sample_forward_categorical(self, x0, t, device):
        """Sample from forward diffusion process (categorical)."""
        # Compute transition probability
        integral = self._beta_integral(t)
        K = 2  # num_classes
        exp_term = np.exp(-K * integral) if isinstance(integral, (int, float)) else torch.exp(-K * integral)
        p_stay = (1 - exp_term) / K + exp_term

        # Flip with probability 1 - p_stay
        p_flip = 1 - p_stay
        if isinstance(p_flip, torch.Tensor):
            uniform_noise = torch.rand_like(x0.float())
            flip_mask = uniform_noise < p_flip
            xt = torch.where(flip_mask, 1.0 - x0.float(), x0.float())
        else:
            uniform_noise = torch.rand_like(x0.float())
            flip_mask = uniform_noise < p_flip
            xt = torch.where(flip_mask, 1.0 - x0.float(), x0.float())
        return xt

    def categorical_training_step(self, batch, batch_idx):
        """Training step with categorical diffusion."""
        _, graph_data, point_indicator = batch
        node_labels = graph_data.x
        edge_index = graph_data.edge_index
        device = node_labels.device

        # Sample t uniformly from [0, 1]
        batch_size = point_indicator.shape[0]
        t = torch.rand(batch_size, device=device)

        # Expand t to match node count
        t_expanded = t.repeat_interleave(point_indicator.reshape(-1).to(device), dim=0)

        # Sample noisy labels
        xt = self._sample_forward_categorical(node_labels, t_expanded, device)

        # Scale to [-1, 1] and add small noise for robustness
        xt = xt * 2 - 1
        xt = xt * (1.0 + 0.05 * torch.rand_like(xt))

        # Prepare inputs
        xt = xt.reshape(-1)
        t_expanded = t_expanded.reshape(-1)
        edge_index = edge_index.to(device).reshape(2, -1)

        # Scale t to [0, 1000] for compatibility with model timestep embedding
        t_scaled = t_expanded * 1000

        # Predict clean labels
        x0_pred = self.forward(
            xt.float().to(device),
            t_scaled.float().to(device),
            edge_index,
        )

        # Cross-entropy loss
        loss = F.cross_entropy(x0_pred, node_labels.long())
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def gaussian_training_step(self, batch, batch_idx):
        """Training step with Gaussian diffusion."""
        _, graph_data, point_indicator = batch
        node_labels = graph_data.x
        edge_index = graph_data.edge_index
        device = node_labels.device

        batch_size = point_indicator.shape[0]
        t = torch.rand(batch_size, device=device)
        t_expanded = t.repeat_interleave(point_indicator.reshape(-1).to(device), dim=0)

        # Normalize labels to [-1, 1]
        node_labels_norm = node_labels.float() * 2 - 1
        node_labels_norm = node_labels_norm * (1.0 + 0.05 * torch.rand_like(node_labels_norm))

        # Add Gaussian noise based on continuous-time schedule
        alpha_bar = torch.exp(-self._beta_integral(t_expanded))
        epsilon = torch.randn_like(node_labels_norm)
        xt = torch.sqrt(alpha_bar) * node_labels_norm + torch.sqrt(1 - alpha_bar) * epsilon

        t_expanded = t_expanded.reshape(-1)
        xt = xt.reshape(-1)
        edge_index = edge_index.to(device).reshape(2, -1)

        # Scale t to [0, 1000]
        t_scaled = t_expanded * 1000

        # Predict noise
        epsilon_pred = self.forward(
            xt.float().to(device),
            t_scaled.float().to(device),
            edge_index,
        )
        epsilon_pred = epsilon_pred.squeeze(-1)

        # MSE loss
        loss = F.mse_loss(epsilon_pred, epsilon)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        """Route to appropriate training step based on diffusion type."""
        if self.diffusion_type == 'gaussian':
            return self.gaussian_training_step(batch, batch_idx)
        elif self.diffusion_type == 'categorical':
            return self.categorical_training_step(batch, batch_idx)

    def _create_score_fn(self, edge_index, device):
        """Create score function for ODE solver."""
        def score_fn(x_t, t):
            """Score function that returns predicted x0 logits."""
            # Ensure x_t is properly shaped
            x_t = x_t.reshape(-1).float().to(device)

            # Create timestep tensor
            if isinstance(t, (int, float)):
                t_tensor = torch.full((x_t.shape[0],), t * 1000, dtype=torch.float32, device=device)
            else:
                t_tensor = torch.full((x_t.shape[0],), t.item() * 1000, dtype=torch.float32, device=device)

            # Get model prediction
            x0_logits = self.forward(x_t, t_tensor, edge_index)
            return x0_logits

        return score_fn

    def test_step(self, batch, batch_idx, split='test'):
        """Evaluation step using EDISCO ODE solver."""
        device = batch[-1].device

        real_batch_idx, graph_data, point_indicator = batch
        node_labels = graph_data.x
        edge_index = graph_data.edge_index

        stacked_predict_labels = []
        edge_index = edge_index.to(node_labels.device).reshape(2, -1)
        edge_index_np = edge_index.cpu().numpy()

        # Build adjacency matrix for MIS decoding
        adj_mat = scipy.sparse.coo_matrix(
            (np.ones_like(edge_index_np[0]), (edge_index_np[0], edge_index_np[1])),
        )

        num_nodes = node_labels.shape[0]

        for _ in range(self.args.sequential_sampling):
            # Initialize with random noise
            if self.diffusion_type == 'gaussian':
                xt = torch.randn(num_nodes, device=device)
            else:
                # For categorical: random binary
                xt = (torch.rand(num_nodes, device=device) > 0.5).float()
                xt = xt * 2 - 1  # Scale to [-1, 1]

            if self.args.parallel_sampling > 1:
                xt = xt.repeat(self.args.parallel_sampling)
                if self.diffusion_type == 'gaussian':
                    xt = torch.randn_like(xt)
                else:
                    xt = (torch.rand_like(xt) > 0.5).float() * 2 - 1
                edge_index = self.duplicate_edge_index(edge_index, num_nodes, device)

            # Create score function
            score_fn = self._create_score_fn(edge_index, device)

            # Get ODE solver
            solver = get_solver(
                self.solver_type,
                num_steps=self.solver_steps,
                beta_min=self.beta_min,
                beta_max=self.beta_max
            )

            # Run reverse diffusion with ODE solver
            with torch.no_grad():
                # Get timesteps
                timesteps = get_time_schedule(self.time_schedule, self.solver_steps).to(device)

                for i in range(len(timesteps) - 1):
                    t = timesteps[i]
                    t_next = timesteps[i + 1]

                    # Get predicted x0
                    x0_logits = score_fn(xt, t.item())

                    if self.diffusion_type == 'categorical':
                        x0_probs = F.softmax(x0_logits, dim=-1)
                        x0_pred = x0_probs[..., 1]  # Probability of class 1 (in MIS)

                        # Adaptive mixing
                        if self.adaptive_mixing and t_next.item() < self.deterministic_threshold:
                            xt = (x0_pred > 0.5).float() * 2 - 1
                        else:
                            xt = torch.bernoulli(x0_pred.clamp(0, 1)) * 2 - 1
                    else:
                        # Gaussian: predict noise and denoise
                        epsilon_pred = x0_logits.squeeze(-1)
                        alpha_bar_t = np.exp(-self._beta_integral(t.item()))
                        alpha_bar_next = np.exp(-self._beta_integral(t_next.item()))

                        # DDIM-style update
                        x0_pred = (xt - np.sqrt(1 - alpha_bar_t) * epsilon_pred) / np.sqrt(alpha_bar_t)
                        xt = np.sqrt(alpha_bar_next) * x0_pred + np.sqrt(1 - alpha_bar_next) * epsilon_pred

                # Final prediction
                x0_logits = score_fn(xt, 0.0)

                if self.diffusion_type == 'categorical':
                    x0_probs = F.softmax(x0_logits, dim=-1)
                    predict_labels = x0_probs[..., 1].cpu().numpy()
                else:
                    predict_labels = (xt.cpu().numpy() * 0.5 + 0.5)

            stacked_predict_labels.append(predict_labels)

        # Aggregate predictions across sampling runs
        predict_labels = np.concatenate(stacked_predict_labels, axis=0)
        all_sampling = self.args.sequential_sampling * self.args.parallel_sampling

        splitted_predict_labels = np.split(predict_labels, all_sampling)

        # Decode predictions to valid independent sets
        solved_solutions = [mis_decode_np(pl, adj_mat) for pl in splitted_predict_labels]
        solved_costs = [sol.sum() for sol in solved_solutions]
        best_solved_cost = np.max(solved_costs)

        gt_cost = node_labels.cpu().numpy().sum()

        metrics = {
            f"{split}/gt_cost": float(gt_cost),
            f"{split}/solved_cost": float(best_solved_cost),
        }

        # Compute gap (MIS is maximization, so gap = (gt - solved) / gt)
        if gt_cost > 0:
            metrics[f"{split}/gap"] = float((gt_cost - best_solved_cost) / gt_cost * 100)

        # Log metrics (solved_cost with prog_bar for visibility)
        self.log(f"{split}/gt_cost", metrics[f"{split}/gt_cost"], on_epoch=True, sync_dist=True)
        self.log(f"{split}/solved_cost", metrics[f"{split}/solved_cost"], prog_bar=True, on_epoch=True, sync_dist=True)
        if f"{split}/gap" in metrics:
            self.log(f"{split}/gap", metrics[f"{split}/gap"], on_epoch=True, sync_dist=True)

        return metrics

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        return self.test_step(batch, batch_idx, split='val')
