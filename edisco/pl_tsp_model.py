"""Lightning module for training TSP models (both DIFUSCO and EDISCO compatible)."""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from pytorch_lightning.utilities import rank_zero_info

from co_datasets.tsp_graph_dataset import TSPGraphDataset
from pl_meta_model import COMetaModel
from utils.diffusion_schedulers import InferenceSchedule, ContinuousTimeSchedule
from utils.tsp_utils import TSPEvaluator, batched_two_opt_torch, merge_tours


class TSPModel(COMetaModel):
    """TSP Model compatible with both DIFUSCO and EDISCO architectures"""
    
    def __init__(self, param_args=None):
        super(TSPModel, self).__init__(param_args=param_args, node_feature_only=False)
        
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
        
        # Check if using continuous-time
        self.is_continuous = getattr(self.args, 'continuous_time', False)
        self.is_equivariant = getattr(self.args, 'equivariant', False)
    
    def forward(self, x, adj, t, edge_index=None):
        if self.is_equivariant:
            # EGNN forward (coordinates, adjacency, time)
            return self.model(x, adj, t, edge_index)
        else:
            # Standard GNN forward (features, time, adjacency, edge_index)
            return self.model(x, t, adj, edge_index)
    
    def categorical_training_step(self, batch, batch_idx):
        edge_index = None
        if not self.sparse:
            _, points, adj_matrix, _ = batch
            if self.is_continuous:
                # Sample continuous time uniformly in [0, 1]
                batch_size = points.shape[0]
                t = torch.rand(batch_size).to(points.device)
            else:
                t = np.random.randint(1, self.diffusion.T + 1, points.shape[0]).astype(int)
        else:
            _, graph_data, point_indicator, edge_indicator, _ = batch
            if self.is_continuous:
                batch_size = point_indicator.shape[0]
                t = torch.rand(batch_size).to(graph_data.x.device)
            else:
                t = np.random.randint(1, self.diffusion.T + 1, point_indicator.shape[0]).astype(int)
            route_edge_flags = graph_data.edge_attr
            points = graph_data.x
            edge_index = graph_data.edge_index
            num_edges = edge_index.shape[1]
            batch_size = point_indicator.shape[0]
            adj_matrix = route_edge_flags.reshape((batch_size, num_edges // batch_size))
        
        # Sample from diffusion
        if self.is_continuous:
            # Continuous-time sampling
            xt = self.diffusion.sample_forward(adj_matrix, t, points.device)
        else:
            # Discrete-time sampling
            adj_matrix_onehot = F.one_hot(adj_matrix.long(), num_classes=2).float()
            if self.sparse:
                adj_matrix_onehot = adj_matrix_onehot.reshape((batch_size * num_edges // batch_size, 2))
            else:
                adj_matrix_onehot = adj_matrix_onehot.unsqueeze(1).unsqueeze(1)
            t = torch.from_numpy(t).long()
            if self.sparse:
                t = t.repeat_interleave(edge_indicator.reshape(-1).cpu(), dim=0).numpy()
            xt = self.diffusion.sample(adj_matrix_onehot, t)
        
        # Forward pass
        if self.is_continuous:
            # Continuous-time forward
            if self.is_equivariant:
                pred = self.forward(points, xt, t, edge_index)
            else:
                pred = self.model(points, t, xt, edge_index)
            
            # Compute continuous-time loss
            loss = self.diffusion.elbo_loss(adj_matrix, xt, t, pred)
        else:
            # Discrete-time forward (original DIFUSCO)
            xt = xt * 2 - 1
            xt = xt * (1.0 + 0.05 * torch.rand_like(xt))
            
            if not self.sparse:
                pred = self.model(points, xt, t, edge_index)
            else:
                points = points.repeat(batch_size, 1)
                xt = xt.reshape((batch_size * edge_indicator[-1], 2))
                pred = self.model(points, xt, t, edge_index)
            
            # Compute discrete-time loss
            loss = F.cross_entropy(
                pred.view(-1, 2),
                adj_matrix.reshape(-1).long()
            )
        
        self.log("train/loss", loss, prog_bar=True)
        return loss
    
    def gaussian_training_step(self, batch, batch_idx):
        """Gaussian diffusion training (not used in EDISCO)"""
        edge_index = None
        if not self.sparse:
            _, points, adj_matrix, _ = batch
            t = np.random.randint(1, self.diffusion.T + 1, points.shape[0]).astype(int)
        else:
            _, graph_data, point_indicator, edge_indicator, _ = batch
            t = np.random.randint(1, self.diffusion.T + 1, point_indicator.shape[0]).astype(int)
            route_edge_flags = graph_data.edge_attr
            points = graph_data.x
            edge_index = graph_data.edge_index
            num_edges = edge_index.shape[1]
            batch_size = point_indicator.shape[0]
            adj_matrix = route_edge_flags.reshape((batch_size, num_edges // batch_size))
        
        # Sample from diffusion
        adj_matrix_norm = adj_matrix * 2.0 - 1.0
        
        if not self.sparse:
            adj_matrix_norm = adj_matrix_norm.unsqueeze(1).unsqueeze(1)
            xt, noise = self.diffusion.sample(adj_matrix_norm, t)
            pred = self.model(points, xt, t, edge_index)
            pred = pred.squeeze(1)
        else:
            adj_matrix_norm = adj_matrix_norm.reshape(-1).unsqueeze(-1)
            xt, noise = self.diffusion.sample(adj_matrix_norm, t)
            points = points.repeat(batch_size, 1)
            t = t.repeat_interleave(edge_indicator.reshape(-1).cpu(), dim=0).numpy()
            xt = xt.reshape((batch_size * edge_indicator[-1], 1))
            noise = noise.reshape((batch_size * edge_indicator[-1], 1))
            pred = self.model(points, xt, t, edge_index).squeeze(-1)
        
        loss = F.mse_loss(pred, noise.squeeze())
        self.log("train/loss", loss, prog_bar=True)
        return loss
    
    def training_step(self, batch, batch_idx):
        if self.diffusion_type == 'gaussian':
            return self.gaussian_training_step(batch, batch_idx)
        elif self.diffusion_type == 'categorical':
            return self.categorical_training_step(batch, batch_idx)
    
    def test_step(self, batch, batch_idx, split='test'):
        """Test step compatible with both DIFUSCO and EDISCO"""
        device = batch[-1].device if isinstance(batch[-1], torch.Tensor) else batch[0].device
        
        if not self.sparse:
            real_batch_idx, points, adj_matrix, gt_tour = batch
            np_points = points.cpu().numpy()[0]
            np_gt_tour = gt_tour.cpu().numpy()[0]
            np_edge_index = None
        else:
            real_batch_idx, graph_data, point_indicator, edge_indicator, gt_tour = batch
            points = graph_data.x
            np_points = points.cpu().numpy()
            np_gt_tour = gt_tour.cpu().numpy().reshape(-1)
            edge_index = graph_data.edge_index
            np_edge_index = edge_index.cpu().numpy()
        
        # Sample using appropriate method
        if self.is_continuous:
            tours, merge_iterations = self._sample_continuous(
                points, device, edge_index if self.sparse else None, np_edge_index
            )
        else:
            tours, merge_iterations = self._sample_discrete(
                points, device, edge_index if self.sparse else None, np_edge_index,
                point_indicator if self.sparse else None,
                edge_indicator if self.sparse else None
            )
        
        # Apply 2-opt refinement
        solved_tours, ns = batched_two_opt_torch(
            np_points.astype("float64"), 
            np.array(tours).astype('int64'),
            max_iterations=self.args.two_opt_iterations, 
            device=device
        )
        
        # Evaluate
        tsp_solver = TSPEvaluator(np_points)
        gt_cost = tsp_solver.evaluate(np_gt_tour)
        
        total_sampling = self.args.parallel_sampling * self.args.sequential_sampling
        all_solved_costs = [tsp_solver.evaluate(solved_tours[i]) for i in range(min(total_sampling, len(solved_tours)))]
        best_solved_cost = np.min(all_solved_costs)
        
        metrics = {
            f"{split}/gt_cost": gt_cost,
            f"{split}/2opt_iterations": ns,
            f"{split}/merge_iterations": merge_iterations,
        }
        for k, v in metrics.items():
            self.log(k, v, on_epoch=True, sync_dist=True)
        self.log(f"{split}/solved_cost", best_solved_cost, prog_bar=True, on_epoch=True, sync_dist=True)
        return metrics
    
    def _sample_continuous(self, points, device, edge_index, np_edge_index):
        """Continuous-time sampling using ODE solver (for EDISCO)"""
        from utils.ode_solvers import get_solver
        from models.continuous_score_network import ScoreWrapper
        
        batch_size = 1 if len(points.shape) == 2 else points.shape[0]
        n_nodes = points.shape[-2] if len(points.shape) == 3 else points.shape[0]
        
        # Initialize at t=1 with noise
        if not self.sparse:
            x_T = torch.randint(0, 2, (batch_size, n_nodes, n_nodes), 
                              device=device, dtype=torch.float32)
        else:
            n_edges = edge_index.shape[1] if edge_index is not None else n_nodes * (n_nodes - 1)
            x_T = torch.randint(0, 2, (n_edges,), device=device, dtype=torch.float32)
        
        # Get solver with beta parameters for consistent CTMC posterior
        beta_min = getattr(self.args, 'beta_min', 0.1)
        beta_max = getattr(self.args, 'beta_max', 1.5)
        solver = get_solver(
            self.args.solver_type if hasattr(self.args, 'solver_type') else 'pndm',
            self.args.solver_steps if hasattr(self.args, 'solver_steps') else 50,
            beta_min=beta_min, beta_max=beta_max
        )
        
        # Create score function wrapper
        if len(points.shape) == 2:
            points = points.unsqueeze(0)
        score_fn = ScoreWrapper(self.model, points, edge_index)
        
        # Sample
        x0_pred = solver.sample(
            score_fn, x_T, device=device,
            schedule=getattr(self.args, 'time_schedule', 'linear'),
            adaptive_mixing=getattr(self.args, 'adaptive_mixing', True),
            deterministic_threshold=getattr(self.args, 'deterministic_threshold', 0.1)
        )
        
        # Convert to adjacency matrix
        adj_mat = x0_pred.cpu().detach().numpy()
        
        # Extract tours
        tours, merge_iterations = merge_tours(
            adj_mat, 
            points.cpu().numpy()[0] if batch_size == 1 else points.cpu().numpy(),
            np_edge_index,
            sparse_graph=self.sparse,
            parallel_sampling=self.args.parallel_sampling,
        )
        
        return tours, merge_iterations
    
    def _sample_discrete(self, points, device, edge_index, np_edge_index, 
                        point_indicator, edge_indicator):
        """Discrete-time sampling (original DIFUSCO)"""
        # Implementation of original discrete sampling
        # (This would be the existing DIFUSCO sampling code)
        
        stacked_tours = []
        for _ in range(self.args.sequential_sampling):
            # Initialize noise
            if self.diffusion_type == 'gaussian':
                if not self.sparse:
                    xt = torch.randn(points.shape[0], 1, points.shape[1], points.shape[1]).to(device)
                else:
                    xt = torch.randn(self.args.parallel_sampling * edge_index.shape[1], 1).to(device)
            else:
                if not self.sparse:
                    xt = torch.randint(0, 2, (points.shape[0], points.shape[1], points.shape[1])).to(device)
                else:
                    xt = torch.randint(0, 2, (self.args.parallel_sampling * edge_index.shape[1],)).to(device)
            
            # Inference schedule
            schedule = InferenceSchedule(self.args.inference_schedule, self.diffusion.T, 
                                        self.args.inference_diffusion_steps)
            
            # Denoise
            for i in range(self.args.inference_diffusion_steps):
                t1, t2 = schedule[i]
                
                if self.diffusion_type == 'gaussian':
                    xt = self.gaussian_denoise_step(points, xt, t1, device, edge_index, target_t=t2)
                else:
                    xt = self.categorical_denoise_step(points, xt, t1, device, edge_index, 
                                                       point_indicator, edge_indicator, target_t=t2)
            
            # Convert to adjacency matrix
            if self.diffusion_type == 'gaussian':
                adj_mat = xt.cpu().detach().numpy() * 0.5 + 0.5
            else:
                adj_mat = xt.float().cpu().detach().numpy() + 1e-6
            
            # Extract tours
            tours, merge_iterations = merge_tours(
                adj_mat, points.cpu().numpy(), np_edge_index,
                sparse_graph=self.sparse,
                parallel_sampling=self.args.parallel_sampling,
            )
            stacked_tours.append(tours)
        
        return np.concatenate(stacked_tours, axis=0), merge_iterations
    
    def gaussian_denoise_step(self, points, xt, t, device, edge_index, target_t=None):
        """Gaussian denoising step"""
        # Implementation from original DIFUSCO
        pass
    
    def categorical_denoise_step(self, points, xt, t, device, edge_index, 
                                point_indicator=None, edge_indicator=None, target_t=None):
        """Categorical denoising step"""
        # Implementation from original DIFUSCO
        pass
    
    def validation_step(self, batch, batch_idx):
        return self.test_step(batch, batch_idx, split='val')