"""
PyTorch Lightning module for EDISCO CVRP
Implements training and evaluation for CVRP with E(2) equivariance
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.utilities import rank_zero_info

from co_datasets.cvrp_graph_dataset import CVRPGraphDataset
from pl_meta_model import COMetaModel
from utils.cvrp_utils import (
    CVRPEvaluator, decode_cvrp_greedy, batched_decode_cvrp,
    apply_2opt_cvrp, merge_cvrp_routes
)
from utils.continuous_diffusion import ContinuousTimeCategoricalDiffusion
from utils.ode_solvers import get_solver, get_time_schedule


class CVRPModel(COMetaModel):
    """
    EDISCO model for CVRP
    Maintains E(2) equivariance through proper separation of coordinates and invariant features
    """
    
    def __init__(self, param_args=None):
        # Initialize parent with node_feature_only=True since we use invariant features
        super(CVRPModel, self).__init__(param_args=param_args, node_feature_only=True)
        
        # Force dense graphs for CVRP (sparse not implemented)
        self.sparse = False
        
        # CVRP-specific configuration
        self.invariant_dim = 2  # demands + is_depot indicator
        self.evaluator = CVRPEvaluator()
        
        # Replace the model with CVRP-specific encoder if using equivariant architecture
        if self.equivariant:
            # Import the CVRP-specific EGNN encoder
            from models.egnn_encoder_cvrp import EGNNEncoderCVRP
            
            # Determine output channels based on diffusion type
            if self.continuous_time:
                out_channels = 2  # for categorical diffusion
            else:
                if self.diffusion_type == 'gaussian':
                    out_channels = 1
                elif self.diffusion_type == 'categorical':
                    out_channels = 2
                else:
                    out_channels = 2
            
            # Replace the model with CVRP version
            self.model = EGNNEncoderCVRP(
                n_layers=self.args.n_layers,
                hidden_dim=self.args.hidden_dim,
                node_dim=getattr(self.args, 'node_dim', 64),
                edge_dim=getattr(self.args, 'edge_dim', 64),
                time_dim=getattr(self.args, 'time_dim', 128),  # Add time_dim as independent parameter
                coord_dim=getattr(self.args, 'coord_dim', 2),
                invariant_dim=self.invariant_dim,  # CVRP-specific: demands + is_depot
                out_channels=out_channels,
                num_classes=2,  # binary adjacency matrix
                sparse=False,  # CVRP only uses dense graphs
                use_activation_checkpoint=self.args.use_activation_checkpoint,
                coord_update_alpha=getattr(self.args, 'coord_update_alpha', 0.1),
                weight_temp=getattr(self.args, 'weight_temp', 10.0)
            )
            
            rank_zero_info(f"Initialized EGNNEncoderCVRP with time_dim={getattr(self.args, 'time_dim', 128)}, "
                          f"hidden_dim={self.args.hidden_dim}, node_dim={getattr(self.args, 'node_dim', 64)}, "
                          f"edge_dim={getattr(self.args, 'edge_dim', 64)}")
            rank_zero_info("Replaced EGNNEncoder with EGNNEncoderCVRP for CVRP-specific equivariant processing")
            
            # Optionally reinitialize node embedding for CVRP invariant features
            # This is now handled internally by EGNNEncoderCVRP, but we can still call it
            self._reinit_node_embedding()
        
        # Load datasets (force dense graphs for CVRP)
        self.train_dataset = CVRPGraphDataset(
            data_file=os.path.join(self.args.storage_path, self.args.training_split),
            sparse_factor=0,  # Force dense graphs
        )
        
        self.test_dataset = CVRPGraphDataset(
            data_file=os.path.join(self.args.storage_path, self.args.test_split),
            sparse_factor=0,  # Force dense graphs
        )
        
        self.validation_dataset = CVRPGraphDataset(
            data_file=os.path.join(self.args.storage_path, self.args.validation_split),
            sparse_factor=0,  # Force dense graphs
        )
        
        # Extract problem info
        self.n_customers = self.train_dataset.n_customers
        self.n_nodes = self.train_dataset.n_nodes
        self.capacity = self.train_dataset.capacity
        
        rank_zero_info(f"Initialized CVRP Model - Customers: {self.n_customers}, Capacity: {self.capacity}")
    
    def _reinit_node_embedding(self):
        """Reinitialize node embedding for CVRP invariant features"""
        if hasattr(self.model, 'node_embed'):
            # For EGNN, modify the input dimension to match CVRP features
            import torch.nn as nn
            self.model.node_embed = nn.Sequential(
                nn.Linear(self.invariant_dim, self.model.node_dim),
                nn.LayerNorm(self.model.node_dim),
                nn.SiLU(),
                nn.Linear(self.model.node_dim, self.model.node_dim)
            )
            rank_zero_info(f"Reinitialized node embedding for CVRP invariant features (dim={self.invariant_dim})")
    
    def forward(self, coords, invariant_features, adj_matrix, t, edge_index=None):
        """
        Forward pass for CVRP
        
        Args:
            coords: Node coordinates (batch_size, n_nodes, 2) - equivariant
            invariant_features: Demands + is_depot (batch_size, n_nodes, 2) - invariant
            adj_matrix: Adjacency matrix (batch_size, n_nodes, n_nodes)
            t: Time steps
            edge_index: Not used for CVRP (dense only)
        """
        if self.equivariant:
            # EGNN expects coordinates and invariant features separately
            # Check if the model has forward_cvrp method (it should with EGNNEncoderCVRP)
            if hasattr(self.model, 'forward_cvrp'):
                return self.model.forward_cvrp(coords, invariant_features, adj_matrix, t)
            else:
                # Fallback for standard EGNN
                rank_zero_info("Warning: Using fallback forward method. Check model initialization.")
                return self.model(invariant_features, t, coords, adj_matrix)
        else:
            # Standard GNN uses only invariant features
            return self.model(invariant_features, t, adj_matrix)
    
    def training_step(self, batch, batch_idx):
        """Main training step that routes to appropriate diffusion-specific method"""
        if self.diffusion_type == 'categorical' or self.continuous_time:
            return self.categorical_training_step(batch, batch_idx)
        elif self.diffusion_type == 'gaussian':
            # If you have Gaussian diffusion, implement gaussian_training_step
            # For now, fallback to categorical
            return self.categorical_training_step(batch, batch_idx)
        else:
            raise ValueError(f"Unknown diffusion type: {self.diffusion_type}")
    
    def categorical_training_step(self, batch, batch_idx):
        """Training step for CVRP with categorical diffusion"""
        # CVRP only uses dense graphs
        coords, invariant_features, adj_matrix, capacity, demands = batch
        batch_size = coords.shape[0]
        device = coords.device
        
        if self.continuous_time:
            t = torch.rand(batch_size, device=device)
            xt = self.diffusion.sample_forward(adj_matrix, t, device)
            logits = self.forward(coords, invariant_features, xt, t)
            loss = self.diffusion.elbo_loss(adj_matrix, xt, t, logits)
        else:
            t = np.random.randint(1, self.diffusion.T + 1, batch_size)
            t = torch.from_numpy(t).long().to(device)
            xt = self.diffusion.sample(adj_matrix.unsqueeze(-1), t).squeeze(-1)
            logits = self.forward(coords, invariant_features, xt, t)
            loss = self.diffusion.loss(logits, adj_matrix.long(), xt, t)
        
        self.log('train/loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        return self.test_step(batch, batch_idx, split='val')
    
    def test_step(self, batch, batch_idx, split='test'):
        """Evaluation step for CVRP"""
        # CVRP only uses dense graphs
        coords, invariant_features, adj_matrix, capacity_batch, demands = batch
        batch_size = coords.shape[0]
        
        # Sample solutions using reverse diffusion
        solutions = self.sample_solutions(
            coords, invariant_features, capacity_batch, 
            n_steps=self.args.solver_steps if self.continuous_time else self.args.inference_diffusion_steps
        )
        
        # Evaluate solutions
        gaps = []
        distances = []
        n_routes_list = []
        
        for b in range(batch_size):
            coords_b = coords[b].cpu().numpy()
            demands_b = demands[b].cpu().numpy()
            
            routes = solutions[b]
            
            # Apply 2-opt if enabled
            if self.args.two_opt_iterations > 0:
                routes = apply_2opt_cvrp(routes, coords_b, self.args.two_opt_iterations)
            
            # Try to merge routes
            if hasattr(self.args, 'merge_routes') and self.args.merge_routes:
                routes = merge_cvrp_routes(routes, demands_b, self.capacity)
            
            # Compute metrics
            distance = self.evaluator.compute_total_distance(coords_b, routes)
            distances.append(distance)
            n_routes_list.append(len(routes))
            
            # Compute gap if ground truth is available
            if hasattr(self, f'{split}_dataset'):
                dataset = getattr(self, f'{split}_dataset')
                instance_idx = batch_idx * len(coords) + b
                if instance_idx < len(dataset):
                    info = dataset.get_instance_info(instance_idx)
                    if 'optimal_distance' in info and info['optimal_distance']:
                        gap = (distance - info['optimal_distance']) / info['optimal_distance'] * 100
                        gaps.append(gap)
        
        # Log metrics
        avg_distance = np.mean(distances)
        avg_routes = np.mean(n_routes_list)
        
        self.log(f'{split}/distance', avg_distance, prog_bar=True)
        self.log(f'{split}/n_routes', avg_routes)
        
        if gaps:
            avg_gap = np.mean(gaps)
            self.log(f'{split}/gap', avg_gap, prog_bar=True)
            self.log(f'{split}/solved_cost', avg_gap)  # For compatibility with checkpoint monitoring
        
        return {'distance': avg_distance, 'gap': np.mean(gaps) if gaps else 0.0}
    
    def sample_solutions(self, coords, invariant_features, capacity_batch, n_steps=50):
        """
        Sample CVRP solutions using reverse diffusion
        """
        device = coords.device
        batch_size, n_nodes, _ = coords.shape
        
        # Initialize with noise
        if self.continuous_time:
            xt = torch.randint(0, 2, (batch_size, n_nodes, n_nodes), device=device).float()
        else:
            xt = torch.randint(0, self.diffusion.num_classes, 
                             (batch_size, n_nodes, n_nodes), device=device).float()
        
        # Reverse diffusion
        if self.continuous_time:
            # Use ODE solver for continuous-time with beta parameters
            beta_min = getattr(self.args, 'beta_min', 0.1)
            beta_max = getattr(self.args, 'beta_max', 1.5)
            solver = get_solver(self.args.solver_type, num_steps=n_steps,
                               beta_min=beta_min, beta_max=beta_max)
            
            def score_fn(x, t):
                t_batch = torch.full((batch_size,), t, device=device)
                logits = self.forward(coords, invariant_features, x, t_batch)
                return logits  # Return logits for solvers to process
            
            # Run solver to get edge probabilities
            adj_probs = solver.sample(score_fn, xt, device=device, schedule=self.args.time_schedule)
        else:
            # Discrete-time reverse process
            schedule = self.diffusion.get_inference_schedule(n_steps)
            
            for t in reversed(schedule):
                t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
                logits = self.forward(coords, invariant_features, xt, t_batch)
                
                if t > 1:
                    xt = self.diffusion.sample_reverse(xt, logits, t_batch)
                else:
                    adj_probs = F.softmax(logits, dim=-1)[..., 1]
                    xt = adj_probs
        
        # Decode to routes
        coords_cpu = coords.cpu()
        demands_cpu = invariant_features[..., 0].cpu()  # First column is demands
        capacity_cpu = capacity_batch.cpu()
        
        solutions = batched_decode_cvrp(
            adj_probs.cpu(), coords_cpu, demands_cpu, capacity_cpu,
            decode_type='greedy'
        )
        
        return solutions
    
    def configure_optimizers(self):
        """Use parent's optimizer configuration"""
        return super().configure_optimizers()
    
    def train_dataloader(self):
        """Return training dataloader"""
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=getattr(self.args, 'num_workers', 16),
            pin_memory=True,
            drop_last=True
        )
    
    def val_dataloader(self):
        """Return validation dataloader"""
        return torch.utils.data.DataLoader(
            self.validation_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=getattr(self.args, 'num_workers', 16),
            pin_memory=True,
            drop_last=False
        )
    
    def test_dataloader(self):
        """Return test dataloader"""
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=getattr(self.args, 'num_workers', 16),
            pin_memory=True,
            drop_last=False
        )