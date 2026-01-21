"""PyTorch Lightning module for EDISCO."""

import os
import numpy as np
import scipy.sparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.utilities import rank_zero_info

from co_datasets.tsp_graph_dataset import TSPGraphDataset
from pl_meta_model import COMetaModel
from utils.continuous_diffusion import (
    ContinuousTimeCategoricalDiffusion,
    ContinuousTimeCategoricalDiffusionDense
)
from utils.ode_solvers import get_solver, get_time_schedule
from utils.tsp_utils import merge_tours as _merge_tours, batched_two_opt_torch, TSPEvaluator


def merge_tours_sparse(adj_mat, np_points, edge_index_np, sparse_graph=False, parallel_sampling=1):
    """Wrapper for tour extraction that handles sparse graph representation.

    For sparse graphs with parallel_sampling=1, directly constructs the adjacency
    matrix from edge indices to ensure correct dimension handling.
    """
    if sparse_graph and parallel_sampling == 1:
        # Ensure adj_mat is 1D
        if adj_mat.ndim >= 2:
            adj_mat = adj_mat.reshape(-1)  # Flatten to (n_edges,)

        # Manually create the split result that merge_tours expects
        # Instead of letting np.split create [(1, n_edges)], we create [(n_edges,)]

        # Replicate the logic from merge_tours but with correct dimensions
        # Create adjacency matrix from sparse representation
        n_nodes = np_points.shape[0]
        adj_matrix_full = scipy.sparse.coo_matrix(
            (adj_mat, (edge_index_np[0], edge_index_np[1])),
            shape=(n_nodes, n_nodes)
        ).toarray() + scipy.sparse.coo_matrix(
            (adj_mat, (edge_index_np[1], edge_index_np[0])),
            shape=(n_nodes, n_nodes)
        ).toarray()

        # Use the cython_merge function from tsp_utils
        from utils.tsp_utils import cython_merge
        real_adj_mat, merge_iterations = cython_merge(np_points, adj_matrix_full)

        # Extract tour
        tour = [0]
        while len(tour) < n_nodes + 1:
            n = np.nonzero(real_adj_mat[tour[-1]])[0]
            if len(tour) > 1:
                n = n[n != tour[-2]]
            if len(n) == 0:
                break
            tour.append(n[0] if len(n) == 1 else n.max())

        return [tour], merge_iterations

    return _merge_tours(adj_mat, np_points, edge_index_np, sparse_graph, parallel_sampling)


from models.continuous_score_network import ContinuousScoreNetwork, ScoreWrapper
from utils.equivariance_utils import test_model_equivariance


class EDISCOModel(COMetaModel):
    """EDISCO model for solving TSP using equivariant continuous-time diffusion."""

    def __init__(self, param_args=None):
        # Enable continuous-time diffusion and equivariance for EDISCO
        param_args.continuous_time = True
        param_args.equivariant = True

        super(EDISCOModel, self).__init__(param_args=param_args, node_feature_only=False)

        # Initialize datasets
        self.train_dataset = TSPGraphDataset(
            data_file=os.path.join(self.args.storage_path, self.args.training_split),
            sparse_factor=self.args.sparse_factor,
        )

        self.test_dataset = TSPGraphDataset(
            data_file=os.path.join(self.args.storage_path, self.args.test_split),
            sparse_factor=self.args.sparse_factor,
        )

        self.validation_dataset = TSPGraphDataset(
            data_file=os.path.join(self.args.storage_path, self.args.validation_split),
            sparse_factor=self.args.sparse_factor,
        )

        # Initialize continuous-time categorical diffusion
        if self.dense_only:
            self.continuous_diffusion = ContinuousTimeCategoricalDiffusionDense(
                beta_min=self.args.beta_min if hasattr(self.args, 'beta_min') else 0.1,
                beta_max=self.args.beta_max if hasattr(self.args, 'beta_max') else 1.5,
                num_classes=2
            )
        else:
            self.continuous_diffusion = ContinuousTimeCategoricalDiffusion(
                beta_min=self.args.beta_min if hasattr(self.args, 'beta_min') else 0.1,
                beta_max=self.args.beta_max if hasattr(self.args, 'beta_max') else 1.5,
                num_classes=2,
                sparse=self.sparse,
                dense_only=self.dense_only
            )

        self.score_network = self.model

        # Solver configuration
        self.solver_type = self.args.solver_type if hasattr(self.args, 'solver_type') else 'pndm'
        self.solver_steps = self.args.solver_steps if hasattr(self.args, 'solver_steps') else 50
        self.time_schedule = self.args.time_schedule if hasattr(self.args, 'time_schedule') else 'linear'

        # Adaptive mixing parameters
        self.use_adaptive_mixing = self.args.adaptive_mixing if hasattr(self.args, 'adaptive_mixing') else True
        self.deterministic_threshold = self.args.deterministic_threshold if hasattr(self.args, 'deterministic_threshold') else 0.1
    
    def forward(self, coords, adj_matrix, timesteps, edge_index=None):
        """Forward pass through the score network."""
        return self.model(coords, adj_matrix, timesteps, edge_index)

    def training_step(self, batch, batch_idx):
        """Training step that routes to the appropriate implementation based on mode."""
        if self.dense_only:
            return self._training_step_dense(batch, batch_idx)
        elif self.sparse:
            return self._training_step_sparse(batch, batch_idx)
        else:
            return self._training_step_flexible(batch, batch_idx)

    def _training_step_dense(self, batch, batch_idx):
        """Dense-only training step with optimized computation."""
        _, coords, adj_matrix, _ = batch
        batch_size = coords.shape[0]
        device = coords.device

        # Sample random timesteps and add noise
        t = torch.rand(batch_size, device=device)
        xt = self.continuous_diffusion.sample_forward(adj_matrix, t, device)

        # Predict x0 from noisy input
        x0_pred_logits = self.forward(coords, xt, t, None)

        # Compute ELBO loss
        loss = self.continuous_diffusion.elbo_loss(adj_matrix, xt, t, x0_pred_logits)

        self.log("train/loss", loss, prog_bar=True)
        return loss

    def _training_step_sparse(self, batch, batch_idx):
        """Sparse training step for large graphs."""
        _, graph_data, point_indicator, edge_indicator, _ = batch
        coords = graph_data.x
        edge_index = graph_data.edge_index
        route_edge_flags = graph_data.edge_attr
        batch_size = point_indicator.shape[0]
        device = coords.device

        # Reshape edge features to adjacency matrix
        num_edges = edge_index.shape[1]
        edges_per_graph = num_edges // batch_size
        adj_matrix = route_edge_flags.reshape((batch_size, edges_per_graph))

        # Sample timesteps and add noise
        t = torch.rand(batch_size, device=device)
        xt = self.continuous_diffusion.sample_forward(adj_matrix, t, device)

        # Prepare for sparse forward pass
        xt_flat = xt.reshape(-1)
        t_expanded = t.repeat_interleave(edges_per_graph)

        # Predict x0
        x0_pred_logits = self.forward(coords, xt_flat, t_expanded, edge_index)
        x0_pred_logits = x0_pred_logits.reshape(batch_size, edges_per_graph, 2)

        # Compute loss
        loss = self.continuous_diffusion.elbo_loss(adj_matrix, xt, t, x0_pred_logits)

        self.log("train/loss", loss, prog_bar=True)
        return loss

    def _training_step_flexible(self, batch, batch_idx):
        """Flexible training step with runtime mode detection."""
        if not self.sparse:
            _, coords, adj_matrix, _ = batch
            batch_size = coords.shape[0]
            device = coords.device
            edge_index = None
        else:
            _, graph_data, point_indicator, edge_indicator, _ = batch
            coords = graph_data.x
            edge_index = graph_data.edge_index
            route_edge_flags = graph_data.edge_attr
            batch_size = point_indicator.shape[0]
            device = coords.device
            
            num_edges = edge_index.shape[1]
            edges_per_graph = num_edges // batch_size
            adj_matrix = route_edge_flags.reshape((batch_size, edges_per_graph))
        
        t = torch.rand(batch_size, device=device)
        xt = self.continuous_diffusion.sample_forward(adj_matrix, t, device)
        
        if not self.sparse:
            x0_pred_logits = self.forward(coords, xt, t, None)
        else:
            xt_flat = xt.reshape(-1)
            t_expanded = t.repeat_interleave(edges_per_graph)
            x0_pred_logits = self.forward(coords, xt_flat, t_expanded, edge_index)
            x0_pred_logits = x0_pred_logits.reshape(batch_size, edges_per_graph, 2)
        
        loss = self.continuous_diffusion.elbo_loss(adj_matrix, xt, t, x0_pred_logits)
        self.log("train/loss", loss, prog_bar=True)
        return loss
    
    def sample_with_solver(self, coords, n_steps=None, device='cuda', edge_index=None):
        """Sample tours using ODE solver."""
        if self.dense_only:
            return self._sample_with_solver_dense(coords, n_steps, device)
        else:
            return self._sample_with_solver_flexible(coords, n_steps, device, edge_index)

    def _sample_with_solver_dense(self, coords, n_steps=None, device='cuda'):
        """Dense-only sampling for small to medium problems."""
        if n_steps is None:
            n_steps = self.solver_steps

        # Pass beta parameters to ensure consistency with forward diffusion
        beta_min = self.args.beta_min if hasattr(self.args, 'beta_min') else 0.1
        beta_max = self.args.beta_max if hasattr(self.args, 'beta_max') else 1.5
        solver = get_solver(self.solver_type, n_steps, beta_min=beta_min, beta_max=beta_max)

        if coords.dim() == 2:
            coords = coords.unsqueeze(0)

        batch_size, n_nodes, _ = coords.shape

        # Initialize from noise
        x_T = torch.randint(0, 2, (batch_size, n_nodes, n_nodes),
                           device=device, dtype=torch.float32)

        score_fn = ScoreWrapper(self.score_network, coords, None)

        # Run ODE solver
        x0_pred = solver.sample(
            score_fn, x_T, device=device,
            schedule=self.time_schedule,
            adaptive_mixing=self.use_adaptive_mixing,
            deterministic_threshold=self.deterministic_threshold
        )

        # Convert to numpy for tour extraction
        adj_matrix_np = x0_pred.cpu().numpy()
        coords_np = coords.cpu().numpy()

        if batch_size == 1:
            adj_matrix_np = adj_matrix_np[0:1]
            coords_np = coords_np[0]

        # Extract tours from adjacency matrix
        tours, _ = merge_tours_sparse(
            adj_matrix_np,
            coords_np,
            None,
            sparse_graph=False,
            parallel_sampling=1
        )

        return tours, x0_pred

    def _sample_with_solver_flexible(self, coords, n_steps=None, device='cuda', edge_index=None):
        """Flexible sampling with sparse/dense support."""
        if n_steps is None:
            n_steps = self.solver_steps

        # Pass beta parameters to ensure consistency with forward diffusion
        beta_min = self.args.beta_min if hasattr(self.args, 'beta_min') else 0.1
        beta_max = self.args.beta_max if hasattr(self.args, 'beta_max') else 1.5
        solver = get_solver(self.solver_type, n_steps, beta_min=beta_min, beta_max=beta_max)

        if len(coords.shape) == 2:
            coords = coords.unsqueeze(0)

        batch_size, n_nodes, _ = coords.shape

        if not self.sparse:
            x_T = torch.randint(0, 2, (batch_size, n_nodes, n_nodes),
                               device=device, dtype=torch.float32)
        else:
            if edge_index is None:
                raise ValueError("edge_index required for sparse mode")
            n_edges = edge_index.shape[1]
            x_T = torch.randint(0, 2, (n_edges,), device=device, dtype=torch.float32)

        # For sparse mode, ScoreWrapper needs 2D coords (n_nodes, 2)
        # For dense mode, it needs 3D coords (batch_size, n_nodes, 2)
        coords_for_wrapper = coords[0] if self.sparse and batch_size == 1 else coords
        score_fn = ScoreWrapper(self.score_network, coords_for_wrapper, edge_index)
        
        x0_pred = solver.sample(
            score_fn, x_T, device=device,
            schedule=self.time_schedule,
            adaptive_mixing=self.use_adaptive_mixing,
            deterministic_threshold=self.deterministic_threshold
        )
        
        if not self.sparse:
            adj_matrix = x0_pred
            adj_matrix_np = adj_matrix.cpu().numpy()
            coords_np = coords.cpu().numpy()
            
            if batch_size == 1:
                adj_matrix_np = adj_matrix_np[0:1]
                coords_np = coords_np[0]
                
            edge_index_np = None
        else:
            adj_matrix = x0_pred
            adj_matrix_np = adj_matrix.cpu().numpy()
            coords_np = coords.cpu().numpy()
            edge_index_np = edge_index.cpu().numpy() if edge_index is not None else None

            if coords_np.ndim == 3 and coords_np.shape[0] == 1:
                coords_np = coords_np[0]

            pass  # adj_matrix_np shape is preserved from x0_pred

        # Extract tours from adjacency matrix
        tours, _ = merge_tours_sparse(
            adj_matrix_np.reshape(1, -1) if adj_matrix_np.ndim == 1 else adj_matrix_np,
            coords_np,
            edge_index_np,
            sparse_graph=self.sparse,
            parallel_sampling=1
        )
        
        return tours, adj_matrix
    
    def test_step(self, batch, batch_idx, split='test'):
        """Evaluate model on test data."""
        if self.dense_only:
            return self._test_step_dense(batch, batch_idx, split)
        else:
            return self._test_step_flexible(batch, batch_idx, split)

    def _test_step_dense(self, batch, batch_idx, split='test'):
        """Dense-only test step."""
        device = batch[-1].device
        real_batch_idx, coords, adj_matrix, gt_tour = batch
        np_coords = coords.cpu().numpy()[0]
        np_gt_tour = gt_tour.cpu().numpy()[0]

        # Generate tour using solver
        tours, adj_probs = self._sample_with_solver_dense(coords, device=device)
        pred_tour = tours[0]

        # Apply 2-opt local search if enabled
        if self.args.two_opt_iterations > 0:
            solved_tours, _ = batched_two_opt_torch(
                np_coords.astype("float64"),
                np.array([pred_tour]).astype('int64'),
                max_iterations=self.args.two_opt_iterations,
                device=device
            )
            pred_tour = solved_tours[0]

        # Evaluate tour quality
        tsp_solver = TSPEvaluator(np_coords)
        gt_cost = tsp_solver.evaluate(np_gt_tour)
        pred_cost = tsp_solver.evaluate(pred_tour)

        metrics = {
            f"{split}/gt_cost": gt_cost,
            f"{split}/solved_cost": pred_cost,
            f"{split}/gap": (pred_cost - gt_cost) / gt_cost * 100,
        }

        for k, v in metrics.items():
            self.log(k, v, on_epoch=True, sync_dist=True)

        return metrics

    def _test_step_flexible(self, batch, batch_idx, split='test'):
        """Flexible test step with sparse/dense support."""
        device = batch[-1].device if isinstance(batch[-1], torch.Tensor) else batch[0].device
        
        if not self.sparse:
            real_batch_idx, coords, adj_matrix, gt_tour = batch
            np_coords = coords.cpu().numpy()[0]
            np_gt_tour = gt_tour.cpu().numpy()[0]
            edge_index = None
        else:
            real_batch_idx, graph_data, point_indicator, edge_indicator, gt_tour = batch
            coords = graph_data.x.reshape(-1, 2)
            edge_index = graph_data.edge_index
            np_coords = coords.cpu().numpy()
            np_gt_tour = gt_tour.cpu().numpy().reshape(-1)
        
        if not self.sparse:
            tours, adj_probs = self.sample_with_solver(coords, device=device, edge_index=edge_index)
        else:
            tours, adj_probs = self.sample_with_solver(coords.unsqueeze(0), device=device, edge_index=edge_index)
        pred_tour = tours[0]
        
        if self.args.two_opt_iterations > 0:
            solved_tours, _ = batched_two_opt_torch(
                np_coords.astype("float64"),
                np.array([pred_tour]).astype('int64'),
                max_iterations=self.args.two_opt_iterations,
                device=device
            )
            pred_tour = solved_tours[0]
        
        tsp_solver = TSPEvaluator(np_coords)
        gt_cost = tsp_solver.evaluate(np_gt_tour)
        pred_cost = tsp_solver.evaluate(pred_tour)
        
        metrics = {
            f"{split}/gt_cost": gt_cost,
            f"{split}/solved_cost": pred_cost,
            f"{split}/gap": (pred_cost - gt_cost) / gt_cost * 100,
        }
        
        for k, v in metrics.items():
            self.log(k, v, on_epoch=True, sync_dist=True)
        
        return metrics
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        return self.test_step(batch, batch_idx, split='val')
    
    def on_validation_epoch_end(self):
        """Test E(2) equivariance at the end of validation if enabled."""
        if self.dense_only and self.args.equivariant:
            if hasattr(self.args, 'test_equivariance') and self.args.test_equivariance:
                if self.global_step > 0:
                    sample_batch = next(iter(self.val_dataloader()))
                    _, coords, adj_matrix, _ = sample_batch
                    coords = coords[:1].to(self.device)
                    adj_matrix = adj_matrix[:1].to(self.device)
                    timesteps = torch.tensor([0.5], device=self.device)

                    eq_results = test_model_equivariance(
                        self.score_network,
                        coords, adj_matrix, timesteps,
                        num_tests=5
                    )

                    self.log("val/equivariance_maintained", float(eq_results['all_equivariant']))
                    self.log("val/equivariance_mean_diff", eq_results['mean_difference'])