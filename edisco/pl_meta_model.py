"""Base PyTorch Lightning module shared by all EDISCO tasks."""

import time

import pytorch_lightning as pl
import torch
import torch.utils.data
from pytorch_lightning.utilities import rank_zero_info
from torch_geometric.loader import DataLoader as GraphDataLoader

from diffusion.beta_schedules import BetaSchedule
from diffusion.categorical_diffusion import ContinuousTimeCategoricalDiffusion
from diffusion.solvers import get_solver
from utils.lr_schedulers import get_schedule_fn


class COMetaModel(pl.LightningModule):
    """Forward process, solver, optimizer and data loading common to all tasks.

    Subclasses create the score network `self.model` and the datasets
    `self.train_dataset`, `self.validation_dataset` and `self.test_dataset`,
    and implement `training_step` and `test_step`.
    """

    def __init__(self, param_args):
        super().__init__()
        self.args = param_args
        # k-nearest-neighbour graphs for sparse_factor > 0, dense graphs otherwise
        self.sparse = self.args.sparse_factor > 0

        # Categorical CTMC with noise rate beta(t) (linear by default)
        self.beta_schedule = BetaSchedule.from_args(self.args)
        self.diffusion = ContinuousTimeCategoricalDiffusion(schedule=self.beta_schedule)

        self.num_training_steps_cached = None

    def build_solver(self, solver_type=None, num_steps=None):
        """Reverse-time solver whose posterior uses the training noise schedule."""
        return get_solver(solver_type or self.args.solver_type,
                          num_steps or self.args.solver_steps,
                          schedule=self.beta_schedule)

    def wall_clock(self):
        """Wall-clock time after pending GPU work has finished (for per-instance timing)."""
        if torch.cuda.is_available() and self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        return time.perf_counter()

    def validation_step(self, batch, batch_idx):
        return self.test_step(batch, batch_idx, split='val')

    # ------------------------------------------------------------------
    # Optimization
    # ------------------------------------------------------------------

    def get_total_num_training_steps(self):
        """Number of optimizer steps over the whole training run."""
        if self.num_training_steps_cached is not None:
            return self.num_training_steps_cached
        if self.trainer.max_steps and self.trainer.max_steps > 0:
            return self.trainer.max_steps

        num_batches = len(self.train_dataloader())
        if isinstance(self.trainer.limit_train_batches, float):
            num_batches = int(self.trainer.limit_train_batches * num_batches)
        num_devices = max(1, self.trainer.num_devices)
        effective_batches = self.trainer.accumulate_grad_batches * num_devices
        self.num_training_steps_cached = (num_batches // effective_batches) * self.trainer.max_epochs
        return self.num_training_steps_cached

    def configure_optimizers(self):
        rank_zero_info('Training steps: %d' % self.get_total_num_training_steps())
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        if self.args.lr_scheduler == 'constant':
            return optimizer
        scheduler = get_schedule_fn(self.args.lr_scheduler, self.get_total_num_training_steps())(optimizer)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {'scheduler': scheduler, 'interval': 'step'},
        }

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    @property
    def eval_batch_size(self):
        """Instances per validation / test batch."""
        return 1

    def train_dataloader(self):
        return GraphDataLoader(
            self.train_dataset, batch_size=self.args.batch_size, shuffle=True,
            num_workers=self.args.num_workers, pin_memory=True,
            persistent_workers=self.args.num_workers > 0, drop_last=True)

    def val_dataloader(self):
        n_examples = min(self.args.validation_examples, len(self.validation_dataset))
        val_dataset = torch.utils.data.Subset(self.validation_dataset, range(n_examples))
        return GraphDataLoader(
            val_dataset, batch_size=self.eval_batch_size, shuffle=False,
            num_workers=self.args.num_workers, pin_memory=True)

    def test_dataloader(self):
        return GraphDataLoader(
            self.test_dataset, batch_size=self.eval_batch_size, shuffle=False,
            num_workers=self.args.num_workers, pin_memory=True)
