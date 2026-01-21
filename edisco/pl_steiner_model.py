"""Lightning module for training Steiner Tree models with E(2)-equivariant EGNN."""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from pytorch_lightning.utilities import rank_zero_info

from co_datasets.steiner_dataset import SteinerTreeDataset
from pl_meta_model import COMetaModel
from utils.diffusion_schedulers import InferenceSchedule, ContinuousTimeSchedule
from utils.steiner_utils import (
    SteinerTreeEvaluator,
    decode_steiner_tree,
    MSTSolver,
    OneSteinerSolver,
    IteratedOneSteinerSolver
)


class SteinerTreeModel(COMetaModel):
    """Steiner Tree Model using EDISCO's E(2)-equivariant framework.

    This model learns to predict Steiner tree adjacency matrices for the
    Euclidean Steiner Tree Problem using continuous-time categorical diffusion
    and E(2)-equivariant graph neural networks.

    Node features:
        - is_terminal: Binary indicator (1 for terminals, 0 for Steiner candidates)

    Coordinates:
        - Terminal positions (2D)
        - Steiner candidate positions (2D)

    Output:
        - Adjacency matrix representing tree structure
    """

    def __init__(self, param_args=None):
        """Initialize Steiner Tree model.

        Args:
            param_args: Argument namespace containing model hyperparameters
        """
        # Initialize base model with node features (is_terminal)
        # node_feature_only=False means we use both coordinates and features
        super(SteinerTreeModel, self).__init__(param_args=param_args, node_feature_only=False)

        # Load datasets
        self.train_dataset = SteinerTreeDataset(
            data_file=os.path.join(self.args.storage_path, self.args.training_split),
            sparse_factor=self.args.sparse_factor,
        )

        self.test_dataset = SteinerTreeDataset(
            data_file=os.path.join(self.args.storage_path, self.args.test_split),
            sparse_factor=self.args.sparse_factor,
        )

        self.validation_dataset = SteinerTreeDataset(
            data_file=os.path.join(self.args.storage_path, self.args.validation_split),
            sparse_factor=self.args.sparse_factor,
        )

        # Check diffusion type
        self.is_continuous = getattr(self.args, 'continuous_time', False)
        self.is_equivariant = getattr(self.args, 'equivariant', False)

        rank_zero_info(f"Initialized SteinerTreeModel:")
        rank_zero_info(f"  - Continuous-time diffusion: {self.is_continuous}")
        rank_zero_info(f"  - E(2)-equivariant: {self.is_equivariant}")
        rank_zero_info(f"  - Sparse mode: {self.sparse}")

    def forward(self, coords, is_terminal, adj, t, edge_index=None):
        """Forward pass through the model.

        Args:
            coords: (batch_size, n_nodes, 2) or (n_nodes, 2) node coordinates
            is_terminal: (batch_size, n_nodes) or (n_nodes,) binary indicators
            adj: Adjacency matrix or edge features (depending on sparse/dense)
            t: Diffusion timestep (batch_size,) or scalar
            edge_index: Edge index for sparse mode

        Returns:
            Predicted adjacency probabilities
        """
        if self.is_equivariant:
            # E(2)-EGNN expects (coords, features, adj, t, edge_index)
            # Coordinates are handled equivariantly
            # is_terminal is an invariant node feature

            # Ensure is_terminal has correct shape for features
            if len(is_terminal.shape) == 1:
                # (n_nodes,) -> (n_nodes, 1)
                node_features = is_terminal.unsqueeze(-1)
            elif len(is_terminal.shape) == 2 and is_terminal.shape[-1] == 1:
                # Already (n_nodes, 1) or (batch_size, n_nodes, 1)
                node_features = is_terminal
            else:
                # (batch_size, n_nodes) -> (batch_size, n_nodes, 1)
                node_features = is_terminal.unsqueeze(-1)

            # EGNN forward: model expects (x_coords, h_features, adj, t, edge_index)
            # But our EGNN encoder combines them internally
            # We pass coords as x and features will be embedded
            return self.model(coords, adj, t, edge_index)
        else:
            # Standard GNN forward (not typically used for Steiner Tree)
            # Combine coords and is_terminal as features
            if len(is_terminal.shape) == 1:
                features = torch.cat([coords, is_terminal.unsqueeze(-1)], dim=-1)
            else:
                features = torch.cat([coords, is_terminal.unsqueeze(-1)], dim=-1)
            return self.model(features, t, adj, edge_index)

    def categorical_training_step(self, batch, batch_idx):
        """Training step for categorical diffusion on Steiner Tree.

        This implements the continuous-time categorical diffusion training
        procedure for learning to predict Steiner tree adjacency matrices.

        Args:
            batch: Batch from SteinerTreeDataset
            batch_idx: Batch index

        Returns:
            Training loss (ELBO for continuous-time, cross-entropy for discrete)
        """
        edge_index = None

        if not self.sparse:
            # Dense mode: batch contains (idx, coords, adjacency, is_terminal)
            _, coords, adj_matrix, is_terminal = batch
            batch_size = coords.shape[0]

            if self.is_continuous:
                # Sample continuous time uniformly in [0, 1]
                t = torch.rand(batch_size).to(coords.device)
            else:
                # Discrete time sampling
                t = np.random.randint(1, self.diffusion.T + 1, batch_size).astype(int)
        else:
            # Sparse mode: batch contains (idx, graph_data, point_indicator, edge_indicator, is_terminal)
            _, graph_data, point_indicator, edge_indicator, is_terminal = batch
            batch_size = point_indicator.shape[0]

            if self.is_continuous:
                t = torch.rand(batch_size).to(graph_data.x.device)
            else:
                t = np.random.randint(1, self.diffusion.T + 1, batch_size).astype(int)

            # Extract sparse graph components
            tree_edge_flags = graph_data.edge_attr
            coords = graph_data.x
            edge_index = graph_data.edge_index
            num_edges = edge_index.shape[1]

            # Reshape adjacency for sparse mode
            adj_matrix = tree_edge_flags.reshape((batch_size, num_edges // batch_size))

        # Sample from forward diffusion process
        if self.is_continuous:
            # Continuous-time: sample X_t ~ q(X_t | X_0)
            xt = self.diffusion.sample_forward(adj_matrix, t, coords.device)
        else:
            # Discrete-time: sample from categorical distribution
            adj_matrix_onehot = F.one_hot(adj_matrix.long(), num_classes=2).float()
            if self.sparse:
                adj_matrix_onehot = adj_matrix_onehot.reshape((batch_size * num_edges // batch_size, 2))
            else:
                adj_matrix_onehot = adj_matrix_onehot.unsqueeze(1).unsqueeze(1)
            t = torch.from_numpy(t).long()
            if self.sparse:
                t = t.repeat_interleave(edge_indicator.reshape(-1).cpu(), dim=0).numpy()
            xt = self.diffusion.sample(adj_matrix_onehot, t)

        # Forward pass to predict X_0
        if self.is_continuous:
            # Continuous-time: predict X_0 from X_t
            if self.is_equivariant:
                # E(2)-EGNN forward
                pred = self.forward(coords, is_terminal, xt, t, edge_index)
            else:
                # Standard GNN (less common for Steiner)
                pred = self.model(coords, t, xt, edge_index)

            # Compute ELBO loss
            loss = self.diffusion.elbo_loss(adj_matrix, xt, t, pred)
        else:
            # Discrete-time: predict noise or X_0
            xt = xt * 2 - 1  # Scale to [-1, 1]
            xt = xt * (1.0 + 0.05 * torch.rand_like(xt))  # Add noise augmentation

            if not self.sparse:
                pred = self.model(coords, xt, t, edge_index)
            else:
                coords = coords.repeat(batch_size, 1)
                xt = xt.reshape((batch_size * edge_indicator[-1], 2))
                pred = self.model(coords, xt, t, edge_index)

            # Cross-entropy loss
            loss = F.cross_entropy(
                pred.view(-1, 2),
                adj_matrix.reshape(-1).long()
            )

        self.log("train/loss", loss, prog_bar=True)
        return loss

    def gaussian_training_step(self, batch, batch_idx):
        """Gaussian diffusion training (not typically used for Steiner Tree).

        Steiner Tree uses categorical diffusion since adjacency matrices are binary.
        This method is included for completeness but shouldn't be called.
        """
        raise NotImplementedError(
            "Gaussian diffusion not implemented for Steiner Tree. "
            "Use categorical diffusion (--diffusion_type categorical)."
        )

    def training_step(self, batch, batch_idx):
        """Main training step dispatcher.

        Args:
            batch: Batch from dataloader
            batch_idx: Batch index

        Returns:
            Training loss
        """
        if self.diffusion_type == 'gaussian':
            return self.gaussian_training_step(batch, batch_idx)
        elif self.diffusion_type == 'categorical':
            return self.categorical_training_step(batch, batch_idx)

    def test_step(self, batch, batch_idx, split='test'):
        """Test step for evaluating Steiner Tree solutions.

        This method:
        1. Samples adjacency matrices using continuous-time diffusion
        2. Decodes adjacency matrices to valid tree structures
        3. Evaluates tree length against ground truth

        Args:
            batch: Batch from dataloader
            batch_idx: Batch index
            split: 'test' or 'val'

        Returns:
            Dictionary of evaluation metrics
        """
        device = batch[-1].device if isinstance(batch[-1], torch.Tensor) else batch[0].device

        if not self.sparse:
            # Dense mode
            real_batch_idx, coords, gt_adj_matrix, is_terminal = batch
            np_coords = coords.cpu().numpy()[0]  # (n_nodes, 2)
            np_is_terminal = is_terminal.cpu().numpy()[0]  # (n_nodes,)
            np_gt_adj = gt_adj_matrix.cpu().numpy()[0]  # (n_nodes, n_nodes)
            np_edge_index = None
        else:
            # Sparse mode
            real_batch_idx, graph_data, point_indicator, edge_indicator, is_terminal = batch
            coords = graph_data.x
            np_coords = coords.cpu().numpy()
            np_is_terminal = is_terminal.cpu().numpy().reshape(-1)
            edge_index = graph_data.edge_index
            np_edge_index = edge_index.cpu().numpy()

            # Reconstruct ground truth adjacency from edge attributes
            gt_edge_attr = graph_data.edge_attr
            n_nodes = len(np_coords)
            np_gt_adj = np.zeros((n_nodes, n_nodes))
            for i in range(edge_index.shape[1]):
                src, dst = edge_index[0, i].item(), edge_index[1, i].item()
                if gt_edge_attr[i].item() > 0.5:
                    np_gt_adj[src, dst] = 1
                    np_gt_adj[dst, src] = 1

        # Sample tree structures using diffusion
        if self.is_continuous:
            pred_adj_matrices, _ = self._sample_continuous(
                coords, is_terminal, device,
                edge_index if self.sparse else None,
                np_edge_index
            )
        else:
            pred_adj_matrices, _ = self._sample_discrete(
                coords, is_terminal, device,
                edge_index if self.sparse else None,
                np_edge_index,
                point_indicator if self.sparse else None,
                edge_indicator if self.sparse else None
            )

        # Decode adjacency matrices to valid trees
        evaluator = SteinerTreeEvaluator()
        gt_length = evaluator.compute_tree_length(np_coords, np_gt_adj)

        # Evaluate all sampled solutions
        total_sampling = self.args.parallel_sampling * self.args.sequential_sampling
        all_lengths = []
        valid_trees = 0

        for i in range(min(total_sampling, len(pred_adj_matrices))):
            adj_probs = pred_adj_matrices[i]

            # Decode to tree structure
            pred_adj, pred_length = decode_steiner_tree(
                adj_probs, np_coords, np_is_terminal,
                threshold=getattr(self.args, 'decode_threshold', 0.5)
            )

            # Validate tree
            is_valid = evaluator.validate_tree(pred_adj, np_is_terminal)
            if is_valid:
                valid_trees += 1
                all_lengths.append(pred_length)
            else:
                # Use a large penalty for invalid trees
                all_lengths.append(gt_length * 2.0)

        # Get best solution
        if all_lengths:
            best_length = np.min(all_lengths)
            avg_length = np.mean(all_lengths)
        else:
            best_length = gt_length * 2.0
            avg_length = gt_length * 2.0

        # Compute gap to ground truth
        gap = (best_length - gt_length) / gt_length * 100.0

        # Log metrics
        metrics = {
            f"{split}/gt_length": gt_length,
            f"{split}/best_length": best_length,
            f"{split}/avg_length": avg_length,
            f"{split}/gap_percent": gap,
            f"{split}/valid_trees": valid_trees,
            f"{split}/valid_rate": valid_trees / total_sampling if total_sampling > 0 else 0.0,
        }

        for k, v in metrics.items():
            self.log(k, v, on_epoch=True, sync_dist=True)

        self.log(f"{split}/solved_length", best_length, prog_bar=True, on_epoch=True, sync_dist=True)

        return metrics

    def _sample_continuous(self, coords, is_terminal, device, edge_index, np_edge_index):
        """Continuous-time sampling using ODE solver (for EDISCO).

        Args:
            coords: Node coordinates tensor
            is_terminal: Binary terminal indicators
            device: Computation device
            edge_index: Edge index for sparse mode
            np_edge_index: Numpy edge index

        Returns:
            Tuple of (adjacency_matrices, num_sampling_steps)
        """
        from utils.ode_solvers import get_solver
        from models.continuous_score_network import ScoreWrapper

        batch_size = 1 if len(coords.shape) == 2 else coords.shape[0]
        n_nodes = coords.shape[-2] if len(coords.shape) == 3 else coords.shape[0]

        # Initialize at t=1 with random binary adjacency
        if not self.sparse:
            # Dense mode: (batch_size, n_nodes, n_nodes)
            x_T = torch.randint(0, 2, (batch_size, n_nodes, n_nodes),
                              device=device, dtype=torch.float32)
        else:
            # Sparse mode: (n_edges,)
            n_edges = edge_index.shape[1] if edge_index is not None else n_nodes * (n_nodes - 1)
            x_T = torch.randint(0, 2, (n_edges,), device=device, dtype=torch.float32)

        # Get ODE solver with beta parameters for consistent CTMC posterior
        beta_min = getattr(self.args, 'beta_min', 0.1)
        beta_max = getattr(self.args, 'beta_max', 1.5)
        solver = get_solver(
            self.args.solver_type if hasattr(self.args, 'solver_type') else 'pndm',
            self.args.solver_steps if hasattr(self.args, 'solver_steps') else 50,
            beta_min=beta_min, beta_max=beta_max
        )

        # Create score function wrapper
        # Ensure coords and is_terminal have batch dimension
        if len(coords.shape) == 2:
            coords = coords.unsqueeze(0)
        if len(is_terminal.shape) == 1:
            is_terminal = is_terminal.unsqueeze(0)

        # Wrap model to include is_terminal
        def score_fn_with_features(x, t):
            """Score function that includes is_terminal features."""
            return self.forward(coords, is_terminal, x, t, edge_index)

        score_fn = ScoreWrapper(score_fn_with_features, coords, edge_index)

        # Sample from diffusion
        x0_pred = solver.sample(
            score_fn, x_T, device=device,
            schedule=getattr(self.args, 'time_schedule', 'linear'),
            adaptive_mixing=getattr(self.args, 'adaptive_mixing', True),
            deterministic_threshold=getattr(self.args, 'deterministic_threshold', 0.1)
        )

        # Convert to numpy adjacency matrices
        if not self.sparse:
            # Dense mode: (batch_size, n_nodes, n_nodes)
            adj_matrices = x0_pred.cpu().detach().numpy()
        else:
            # Sparse mode: convert edge predictions to dense adjacency
            edge_probs = x0_pred.cpu().detach().numpy()
            adj_matrices = np.zeros((batch_size, n_nodes, n_nodes))

            for i in range(len(edge_probs)):
                src, dst = np_edge_index[0, i], np_edge_index[1, i]
                adj_matrices[0, src, dst] = edge_probs[i]
                adj_matrices[0, dst, src] = edge_probs[i]

        # Return list of adjacency matrices (one per parallel sample)
        num_samples = self.args.parallel_sampling if hasattr(self.args, 'parallel_sampling') else 1
        result = [adj_matrices[i % batch_size] for i in range(num_samples)]

        return result, solver.steps

    def _sample_discrete(self, coords, is_terminal, device, edge_index, np_edge_index,
                        point_indicator, edge_indicator):
        """Discrete-time sampling (original DIFUSCO approach).

        Not recommended for Steiner Tree - use continuous-time instead.

        Args:
            coords: Node coordinates
            is_terminal: Binary indicators
            device: Computation device
            edge_index: Edge index for sparse mode
            np_edge_index: Numpy edge index
            point_indicator: Number of nodes (sparse mode)
            edge_indicator: Number of edges (sparse mode)

        Returns:
            Tuple of (adjacency_matrices, num_steps)
        """
        raise NotImplementedError(
            "Discrete-time sampling not fully implemented for Steiner Tree. "
            "Use continuous-time diffusion (--continuous_time True)."
        )

    def validation_step(self, batch, batch_idx):
        """Validation step - same as test but with 'val' split label."""
        return self.test_step(batch, batch_idx, split='val')

    def test_epoch_end(self, outputs):
        """Aggregate test metrics at epoch end."""
        unmerged_metrics = {}
        for metrics in outputs:
            for k, v in metrics.items():
                if k not in unmerged_metrics:
                    unmerged_metrics[k] = []
                unmerged_metrics[k].append(v)

        merged_metrics = {}
        for k, v in unmerged_metrics.items():
            merged_metrics[k] = float(np.mean(v))

        if hasattr(self, 'logger') and self.logger is not None:
            self.logger.log_metrics(merged_metrics, step=self.global_step)

        # Print summary
        rank_zero_info("\n" + "="*60)
        rank_zero_info("Test Epoch Summary:")
        for k, v in sorted(merged_metrics.items()):
            if 'test/' in k:
                rank_zero_info(f"  {k}: {v:.4f}")
        rank_zero_info("="*60 + "\n")
