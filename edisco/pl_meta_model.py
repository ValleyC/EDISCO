"""A meta PyTorch Lightning model for training and evaluating diffusion models."""

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.utils.data
from torch_geometric.loader import DataLoader as GraphDataLoader
from pytorch_lightning.utilities import rank_zero_info

from models.gnn_encoder import GNNEncoder
from utils.lr_schedulers import get_schedule_fn
from utils.diffusion_schedulers import CategoricalDiffusion, GaussianDiffusion


class COMetaModel(pl.LightningModule):
    """Base PyTorch Lightning model for combinatorial optimization problems."""

    def __init__(self, param_args, node_feature_only=False):
        super(COMetaModel, self).__init__()
        self.args = param_args
        self.diffusion_type = self.args.diffusion_type
        self.diffusion_schedule = self.args.diffusion_schedule
        self.diffusion_steps = self.args.diffusion_steps

        # Determine sparse/dense mode at initialization
        self.sparse = self.args.sparse_factor > 0 or node_feature_only
        self.dense_only = not self.sparse and self.args.sparse_factor == 0

        mode_str = "dense-only" if self.dense_only else ("sparse" if self.sparse else "dense")
        rank_zero_info(f"Initializing model in {mode_str} mode")

        self.continuous_time = getattr(self.args, 'continuous_time', False)
        self.equivariant = getattr(self.args, 'equivariant', False)
        
        # Initialize diffusion process
        if self.continuous_time:
            from utils.continuous_diffusion import (
                ContinuousTimeCategoricalDiffusion,
                ContinuousTimeGaussianDiffusion
            )

            if self.diffusion_type == 'gaussian':
                out_channels = 1
                self.diffusion = ContinuousTimeGaussianDiffusion(
                    beta_min=getattr(self.args, 'beta_min', 0.1),
                    beta_max=getattr(self.args, 'beta_max', 1.5),
                    sparse=self.sparse,
                    dense_only=self.dense_only
                )
            elif self.diffusion_type == 'categorical':
                out_channels = 2
                self.diffusion = ContinuousTimeCategoricalDiffusion(
                    beta_min=getattr(self.args, 'beta_min', 0.1),
                    beta_max=getattr(self.args, 'beta_max', 1.5),
                    num_classes=2,
                    sparse=self.sparse,
                    dense_only=self.dense_only
                )
        else:
            # Discrete-time diffusion
            if self.diffusion_type == 'gaussian':
                out_channels = 1
                self.diffusion = GaussianDiffusion(
                    T=self.diffusion_steps, schedule=self.diffusion_schedule)
            elif self.diffusion_type == 'categorical':
                out_channels = 2
                self.diffusion = CategoricalDiffusion(
                    T=self.diffusion_steps, schedule=self.diffusion_schedule)
        
        # Initialize model architecture
        if self.equivariant:
            from models.egnn_encoder import EGNNEncoder
            self.model = EGNNEncoder(
                n_layers=self.args.n_layers,
                hidden_dim=self.args.hidden_dim,
                node_dim=getattr(self.args, 'node_dim', 64),
                edge_dim=getattr(self.args, 'edge_dim', 64),
                time_dim=getattr(self.args, 'time_dim', 128),
                coord_dim=getattr(self.args, 'coord_dim', 2),
                out_channels=out_channels,
                sparse=self.sparse,
                dense_only=self.dense_only,
                use_activation_checkpoint=self.args.use_activation_checkpoint,
                coord_update_alpha=getattr(self.args, 'coord_update_alpha', 0.1),
                weight_temp=getattr(self.args, 'weight_temp', 10.0)
            )
        else:
            self.model = GNNEncoder(
                n_layers=self.args.n_layers,
                hidden_dim=self.args.hidden_dim,
                out_channels=out_channels,
                aggregation=self.args.aggregation,
                sparse=self.sparse,
                use_activation_checkpoint=self.args.use_activation_checkpoint,
                node_feature_only=node_feature_only,
            )

        self.model_sparse_mode = self.sparse
        self.model_dense_only_mode = self.dense_only
        self.num_training_steps_cached = None

        if self.continuous_time:
            self._init_ode_solver()

    def _init_ode_solver(self):
        """Initialize ODE solver for continuous-time diffusion."""
        from utils.ode_solvers import get_solver
        self.solver_type = getattr(self.args, 'solver_type', 'pndm')
        self.solver_steps = getattr(self.args, 'solver_steps', 50)
        # Pass beta parameters for consistent CTMC posterior
        beta_min = getattr(self.args, 'beta_min', 0.1)
        beta_max = getattr(self.args, 'beta_max', 1.5)
        self.solver = get_solver(
            self.solver_type,
            self.solver_steps,
            beta_min=beta_min,
            beta_max=beta_max
        )
    
    def forward(self, x, adj, t, edge_index=None):
        """Forward pass through the model."""
        if self.dense_only:
            if self.equivariant:
                return self.model(x, adj, t)
            else:
                return self.model(x, t, adj, None)
        else:
            if self.equivariant:
                return self.model(x, adj, t, edge_index)
            else:
                return self.model(x, t, adj, edge_index)
    
    def test_epoch_end(self, outputs):
        unmerged_metrics = {}
        for metrics in outputs:
            for k, v in metrics.items():
                if k not in unmerged_metrics:
                    unmerged_metrics[k] = []
                unmerged_metrics[k].append(v)
        
        merged_metrics = {}
        for k, v in unmerged_metrics.items():
            merged_metrics[k] = float(np.mean(v))
        self.logger.log_metrics(merged_metrics, step=self.global_step)
    
    def get_total_num_training_steps(self) -> int:
        """Total training steps inferred from datamodule and devices."""
        if self.num_training_steps_cached is not None:
            return self.num_training_steps_cached
        dataset = self.train_dataloader()
        if self.trainer.max_steps and self.trainer.max_steps > 0:
            return self.trainer.max_steps
        
        dataset_size = (
            self.trainer.limit_train_batches * len(dataset)
            if self.trainer.limit_train_batches != 0
            else len(dataset)
        )
        
        num_devices = max(1, self.trainer.num_devices)
        effective_batch_size = self.trainer.accumulate_grad_batches * num_devices
        self.num_training_steps_cached = (dataset_size // effective_batch_size) * self.trainer.max_epochs
        return self.num_training_steps_cached
    
    def configure_optimizers(self):
        rank_zero_info('Parameters: %d' % sum([p.numel() for p in self.model.parameters()]))
        rank_zero_info('Training steps: %d' % self.get_total_num_training_steps())
        
        if self.args.lr_scheduler == "constant":
            return torch.optim.AdamW(
                self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        else:
            optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
            scheduler = get_schedule_fn(self.args.lr_scheduler, self.get_total_num_training_steps())(optimizer)
            
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                },
            }
    
    def categorical_posterior(self, target_t, t, x0_pred_prob, xt):
        """Sample from the categorical posterior."""
        if self.continuous_time:
            if self.dense_only:
                return self.diffusion.sample_reverse_dense(xt, x0_pred_prob, t, target_t, self.device)
            else:
                return self.diffusion.sample_reverse(xt, x0_pred_prob, t, target_t, self.device)
        else:
            return self._categorical_posterior_discrete(target_t, t, x0_pred_prob, xt)

    def _categorical_posterior_discrete(self, target_t, t, x0_pred_prob, xt):
        """Discrete-time categorical posterior."""
        diffusion = self.diffusion
        if target_t is None:
            target_t = t - 1
        else:
            target_t = torch.from_numpy(target_t).view(1)
        
        atbar = diffusion.alphabar[t]
        atbar_target = diffusion.alphabar[target_t]
        
        if self.args.inference_trick is None or t <= 1:
            at = diffusion.alpha[t]
            z = torch.randn_like(xt)
            atbar_prev = diffusion.alphabar[t - 1]
            beta_tilde = diffusion.beta[t - 1] * (1 - atbar_prev) / (1 - atbar)
            
            xt_target = (1 / np.sqrt(at)).item() * (xt - ((1 - at) / np.sqrt(1 - atbar)).item() * pred)
            xt_target = xt_target + np.sqrt(beta_tilde).item() * z
        elif self.args.inference_trick == 'ddim':
            xt_target = np.sqrt(atbar_target / atbar).item() * (xt - np.sqrt(1 - atbar).item() * pred)
            xt_target = xt_target + np.sqrt(1 - atbar_target).item() * pred
        else:
            raise ValueError('Unknown inference trick {}'.format(self.args.inference_trick))
        return xt_target
    
    def gaussian_posterior(self, target_t, t, pred, xt):
        """Sample from Gaussian posterior."""
        if self.continuous_time:
            if self.dense_only:
                return self.diffusion.gaussian_posterior_dense(target_t, t, pred, xt)
            else:
                return self.diffusion.gaussian_posterior(target_t, t, pred, xt)
        else:
            return self._gaussian_posterior_discrete(target_t, t, pred, xt)

    def _gaussian_posterior_discrete(self, target_t, t, pred, xt):
        """Discrete-time Gaussian posterior."""
        diffusion = self.diffusion
        if target_t is None:
            target_t = t - 1
        else:
            target_t = torch.from_numpy(target_t).view(1)
        
        atbar = diffusion.alphabar[t]
        atbar_target = diffusion.alphabar[target_t]
        
        if self.args.inference_trick is None or t <= 1:
            at = diffusion.alpha[t]
            z = torch.randn_like(xt)
            atbar_prev = diffusion.alphabar[t - 1]
            beta_tilde = diffusion.beta[t - 1] * (1 - atbar_prev) / (1 - atbar)
            
            xt_target = (1 / np.sqrt(at)).item() * (xt - ((1 - at) / np.sqrt(1 - atbar)).item() * pred)
            xt_target = xt_target + np.sqrt(beta_tilde).item() * z
        elif self.args.inference_trick == 'ddim':
            xt_target = np.sqrt(atbar_target / atbar).item() * (xt - np.sqrt(1 - atbar).item() * pred)
            xt_target = xt_target + np.sqrt(1 - atbar_target).item() * pred
        else:
            raise ValueError('Unknown inference trick {}'.format(self.args.inference_trick))
        return xt_target
    
    def duplicate_edge_index(self, edge_index, num_nodes, device):
        """Duplicate edge index for parallel sampling in sparse mode."""
        if self.dense_only:
            return None

        edge_index = edge_index.reshape((2, 1, -1))
        edge_index_indent = torch.arange(0, self.args.parallel_sampling).view(1, -1, 1).to(device)
        edge_index_indent = edge_index_indent * num_nodes
        edge_index = edge_index + edge_index_indent
        edge_index = edge_index.reshape((2, -1))
        return edge_index
    
    def train_dataloader(self):
        batch_size = self.args.batch_size
        train_dataloader = GraphDataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=self.args.num_workers, pin_memory=True,
            persistent_workers=True, drop_last=True)
        return train_dataloader
    
    def test_dataloader(self):
        batch_size = 1
        print("Test dataset size:", len(self.test_dataset))
        test_dataloader = GraphDataLoader(
            self.test_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=True
        )
        return test_dataloader
    
    def val_dataloader(self):
        batch_size = 1
        val_dataset = torch.utils.data.Subset(self.validation_dataset, range(self.args.validation_examples))
        print("Validation dataset size:", len(val_dataset))
        val_dataloader = GraphDataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=True
        )
        return val_dataloader