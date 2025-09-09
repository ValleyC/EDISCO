"""Lightning module for training the EDISCO TSP model."""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from pytorch_lightning.utilities import rank_zero_info

from co_datasets.tsp_graph_dataset import TSPGraphDataset
from pl_meta_model import COMetaModel
from utils.tsp_utils import (
    TSPEvaluator, 
    batched_two_opt_torch, 
    merge_tours_dense,
    compute_tour_length
)


class TSPModel(COMetaModel):
    def __init__(self, param_args=None):
        super(TSPModel, self).__init__(param_args=param_args)
        
        # Create datasets
        self.train_dataset = TSPGraphDataset(
            data_file=os.path.join(self.args.storage_path, self.args.training_split),
            n_instances=None  # Use all instances
        )
        
        self.test_dataset = TSPGraphDataset(
            data_file=os.path.join(self.args.storage_path, self.args.test_split),
            n_instances=None
        )
        
        self.validation_dataset = TSPGraphDataset(
            data_file=os.path.join(self.args.storage_path, self.args.validation_split),
            n_instances=self.args.validation_examples
        )
        
        # Get number of cities from dataset
        self.n_cities = self.train_dataset.n_cities
        rank_zero_info(f"TSP dataset with {self.n_cities} cities loaded")
    
    def training_step(self, batch, batch_idx):
        """Training step with continuous-time diffusion"""
        coords, adj_matrix, tour_gt = batch
        batch_size = coords.shape[0]
        
        # Sample continuous time uniformly in [0, 1]
        t = self.sample_time(batch_size, coords.device)
        
        # Forward diffusion: sample X_t | X_0
        xt = self.diffusion.sample_forward(adj_matrix, t, coords.device)
        
        # Predict X_0 from (X_t, t)
        x0_pred_logits = self.forward(coords, xt, t)
        
        # Compute loss
        loss = self.compute_loss(adj_matrix, xt, t, x0_pred_logits)
        
        # Log metrics
        self.log('train/loss', loss, prog_bar=True)
        self.log('train/t_mean', t.mean())
        self.log('train/t_std', t.std())
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step with sampling and evaluation"""
        return self._eval_step(batch, batch_idx, split='val')
    
    def test_step(self, batch, batch_idx):
        """Test step with sampling and evaluation"""
        return self._eval_step(batch, batch_idx, split='test')
    
    def _eval_step(self, batch, batch_idx, split='val'):
        """Common evaluation logic for validation and test"""
        coords, adj_matrix, tour_gt = batch
        device = coords.device
        batch_size = coords.shape[0]
        
        # Track all sampled tours
        all_tours = []
        all_gaps = []
        
        # Sequential sampling
        for seq_idx in range(self.args.sequential_sampling):
            # Parallel sampling
            if self.args.parallel_sampling > 1:
                coords_expanded = coords.repeat(self.args.parallel_sampling, 1, 1)
            else:
                coords_expanded = coords
            
            # Sample tours
            adj_probs, _ = self.sample(
                coords_expanded,
                n_steps=self.args.inference_diffusion_steps,
                method=self.args.inference_method,
                device=device
            )
            
            # Decode tours from adjacency probabilities
            for p_idx in range(self.args.parallel_sampling):
                for b_idx in range(batch_size):
                    idx = p_idx * batch_size + b_idx
                    tour = merge_tours_dense(
                        adj_probs[idx].cpu(),
                        coords_expanded[idx].cpu()
                    )
                    
                    # Apply 2-opt if specified
                    if self.args.two_opt_iterations > 0:
                        tour, _ = batched_two_opt_torch(
                            coords_expanded[idx].cpu().numpy(),
                            tour.reshape(1, -1),
                            max_iterations=self.args.two_opt_iterations,
                            device='cpu'
                        )
                        tour = tour[0]
                    
                    all_tours.append(tour)
                    
                    # Compute gap
                    pred_length = compute_tour_length(coords[b_idx].cpu(), tour)
                    gt_length = compute_tour_length(coords[b_idx].cpu(), tour_gt[b_idx].cpu())
                    gap = (pred_length - gt_length) / gt_length * 100
                    all_gaps.append(gap)
        
        # Save heatmaps if requested
        if self.args.save_numpy_heatmap and split == 'test':
            self._save_numpy_heatmap(adj_probs, coords, batch_idx, split)
        
        # Find best tour for each instance
        total_sampling = self.args.sequential_sampling * self.args.parallel_sampling
        best_gaps = []
        
        for b_idx in range(batch_size):
            instance_gaps = all_gaps[b_idx::batch_size][:total_sampling]
            best_gap = min(instance_gaps)
            best_gaps.append(best_gap)
        
        # Log metrics
        metrics = {
            f'{split}/avg_gap': np.mean(best_gaps),
            f'{split}/std_gap': np.std(best_gaps),
            f'{split}/min_gap': np.min(best_gaps),
            f'{split}/max_gap': np.max(best_gaps),
        }
        
        for key, value in metrics.items():
            self.log(key, value, on_epoch=True, sync_dist=True)
        
        return metrics
    
    def _save_numpy_heatmap(self, adj_probs, coords, batch_idx, split):
        """Save adjacency probability heatmaps for MCTS"""
        exp_save_dir = os.path.join(
            self.logger.save_dir, 
            self.logger.name, 
            self.logger.version
        )
        heatmap_path = os.path.join(exp_save_dir, 'numpy_heatmap')
        rank_zero_info(f"Saving heatmap to {heatmap_path}")
        os.makedirs(heatmap_path, exist_ok=True)
        
        batch_size = coords.shape[0]
        for b_idx in range(batch_size):
            global_idx = batch_idx * batch_size + b_idx
            np.save(
                os.path.join(heatmap_path, f"{split}-heatmap-{global_idx}.npy"),
                adj_probs[b_idx].cpu().numpy()
            )
            np.save(
                os.path.join(heatmap_path, f"{split}-points-{global_idx}.npy"),
                coords[b_idx].cpu().numpy()
            )
    
    def __init__(self, param_args=None):
        super(TSPModel, self).__init__(param_args=param_args)
        
        # Initialize instance attributes for storing outputs
        self.validation_outputs = []
        self.test_outputs = []
        
        # Create datasets
        self.train_dataset = TSPGraphDataset(
            data_file=os.path.join(self.args.storage_path, self.args.training_split),
            n_instances=None  # Use all instances
        )
        
        self.test_dataset = TSPGraphDataset(
            data_file=os.path.join(self.args.storage_path, self.args.test_split),
            n_instances=None
        )
        
        self.validation_dataset = TSPGraphDataset(
            data_file=os.path.join(self.args.storage_path, self.args.validation_split),
            n_instances=self.args.validation_examples
        )
        
        # Get number of cities from dataset
        self.n_cities = self.train_dataset.n_cities
        rank_zero_info(f"TSP dataset with {self.n_cities} cities loaded")
    
    def train_dataloader(self):
        batch_size = self.args.batch_size
        train_dataloader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=True,
            persistent_workers=True if self.args.num_workers > 0 else False,
            drop_last=True
        )
        return train_dataloader
    
    def val_dataloader(self):
        val_dataloader = torch.utils.data.DataLoader(
            self.validation_dataset,
            batch_size=1,  # Evaluate one at a time for accurate metrics
            shuffle=False,
            num_workers=self.args.num_workers
        )
        return val_dataloader
    
    def test_dataloader(self):
        test_dataloader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.args.num_workers
        )
        return test_dataloader