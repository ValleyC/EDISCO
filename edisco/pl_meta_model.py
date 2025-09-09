"""A meta PyTorch Lightning model for training and evaluating EDISCO models."""

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.utils.data
from torch_geometric.data import DataLoader as GraphDataLoader
from pytorch_lightning.utilities import rank_zero_info

from models.score_network import ScoreNetwork
from utils.lr_schedulers import get_schedule_fn
from utils.ct_diffusion_schedulers import ContinuousTimeCategoricalDiffusion


class COMetaModel(pl.LightningModule):
    def __init__(self, param_args):
        super(COMetaModel, self).__init__()
        self.args = param_args
        self.save_hyperparameters()
        
        # Continuous-time diffusion parameters
        self.beta_min = self.args.beta_min
        self.beta_max = self.args.beta_max
        self.loss_type = self.args.loss_type
        self.time_sampling = self.args.time_sampling
        
        # Initialize continuous-time categorical diffusion
        self.diffusion = ContinuousTimeCategoricalDiffusion(
            beta_min=self.beta_min,
            beta_max=self.beta_max,
            num_classes=self.args.num_classes
        )
        
        # Initialize score network (EGNN-based)
        self.model = ScoreNetwork(
            n_layers=self.args.n_layers,
            hidden_dim=self.args.hidden_dim,
            node_dim=self.args.node_dim,
            coord_dim=self.args.coord_dim,
            num_classes=self.args.num_classes
        )
        
        self.num_training_steps_cached = None
        
    def forward(self, coords, adj_matrix, timesteps):
        """Forward pass: predict X_0 given X_t and t"""
        return self.model(coords, adj_matrix, timesteps)
    
    def sample_time(self, batch_size, device):
        """Sample continuous time points for training"""
        if self.time_sampling == 'uniform':
            # Uniform sampling in [0, 1]
            t = torch.rand(batch_size, device=device)
        elif self.time_sampling == 'importance':
            # Importance sampling - more samples near t=0
            u = torch.rand(batch_size, device=device)
            t = 1 - torch.sqrt(1 - u)  # Quadratic importance sampling
        elif self.time_sampling == 'cosine':
            # Cosine schedule sampling
            u = torch.rand(batch_size, device=device)
            t = (1 - torch.cos(u * np.pi)) / 2
        else:
            t = torch.rand(batch_size, device=device)
        
        return t
    
    def compute_loss(self, x0, xt, t, x0_pred_logits):
        """Compute training loss"""
        if self.loss_type == 'elbo':
            loss = self.diffusion.elbo_loss(x0, xt, t, x0_pred_logits)
        elif self.loss_type == 'score_matching':
            # Alternative: score matching loss
            loss = self.diffusion.score_matching_loss(x0, xt, t, x0_pred_logits)
        else:
            loss = self.diffusion.elbo_loss(x0, xt, t, x0_pred_logits)
        
        return loss
    
    @torch.no_grad()
    def sample(self, coords, n_steps=None, method=None, adaptive=None, device='cuda'):
        """Sample tours using reverse diffusion"""
        if n_steps is None:
            n_steps = self.args.inference_diffusion_steps
        if method is None:
            method = self.args.inference_method
        if adaptive is None:
            adaptive = (self.args.inference_schedule == 'adaptive')
        
        batch_size = coords.shape[0]
        n_nodes = coords.shape[1]
        
        # Initialize at t=1 with uniform noise
        xt = torch.randint(0, self.args.num_classes, 
                          (batch_size, n_nodes, n_nodes), device=device).float()
        
        # Time schedule for reverse diffusion
        if self.args.inference_schedule == 'cosine':
            s = torch.linspace(0, 1, n_steps + 1, device=device)
            timesteps = 1 - (torch.cos(s * np.pi / 2) ** 2)[:-1]
            timesteps = timesteps.flip(0)
        elif self.args.inference_schedule == 'adaptive':
            # Adaptive time stepping based on score magnitude
            timesteps = self._adaptive_time_schedule(coords, xt, n_steps, device)
        else:  # linear
            timesteps = torch.linspace(1.0, 0.0, n_steps + 1, device=device)[:-1]
        
        # Reverse diffusion process
        for i in range(len(timesteps) - 1):
            t = timesteps[i].item()
            t_next = timesteps[i + 1].item()
            dt = t_next - t  # Negative for reverse
            
            t_tensor = torch.full((batch_size,), t, device=device)
            
            # Predict X_0
            x0_logits = self.forward(coords, xt, t_tensor)
            
            # Sample reverse step
            if method == 'simple':
                if t_next > 0.01:
                    xt = self.diffusion.sample_reverse_simple(xt, x0_logits, t, dt, device)
                else:
                    xt = x0_logits.argmax(dim=-1).float()
            elif method == 'predictor_corrector':
                xt = self.diffusion.sample_reverse_pc(xt, x0_logits, t, dt, device)
            else:
                xt = self.diffusion.sample_reverse_simple(xt, x0_logits, t, dt, device)
        
        # Final prediction at t=0
        final_logits = self.forward(coords, xt, torch.zeros(batch_size, device=device))
        adj_probs = F.softmax(final_logits, dim=-1)[..., 1]  # Probability of edge=1
        
        return adj_probs, xt
    
    def _adaptive_time_schedule(self, coords, xt, n_steps, device):
        """Generate adaptive time schedule based on score magnitude"""
        # Simple adaptive schedule - more steps near t=0
        # Can be made more sophisticated based on score magnitude
        power = 2.0  # Control adaptation strength
        s = torch.linspace(0, 1, n_steps + 1, device=device)
        timesteps = 1 - torch.pow(s, power)[:-1]
        return timesteps.flip(0)
    
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
            optimizer = torch.optim.AdamW(
                self.model.parameters(), 
                lr=self.args.learning_rate, 
                weight_decay=self.args.weight_decay
            )
            return optimizer
        else:
            optimizer = torch.optim.AdamW(
                self.model.parameters(), 
                lr=self.args.learning_rate, 
                weight_decay=self.args.weight_decay
            )
            scheduler = get_schedule_fn(
                self.args.lr_scheduler, 
                self.get_total_num_training_steps()
            )(optimizer)

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                },
            }
    
    def on_train_epoch_start(self):
        """Log training epoch start"""
        self.log('train/epoch', self.current_epoch)
    
    def on_validation_epoch_start(self):
        """Log validation epoch start"""
        self.log('val/epoch', self.current_epoch)