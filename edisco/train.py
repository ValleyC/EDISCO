"""Training and evaluation entry point for EDISCO."""

import os
from argparse import ArgumentParser

import torch
import wandb
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.utilities import rank_zero_info

from pl_tsp_model import TSPModel
from pl_cvrp_model import CVRPModel
from pl_cvrp_partition_model import CVRPPartitionModel
from pl_steiner_model import SteinerTreeModel
from pl_mis_model import MISModel

# task -> (Lightning module, checkpoint selection mode on val/solved_cost)
TASKS = {
    'tsp': (TSPModel, 'min'),
    'cvrp': (CVRPModel, 'min'),
    'cvrp_partition': (CVRPPartitionModel, 'min'),
    'steiner': (SteinerTreeModel, 'min'),
    'mis': (MISModel, 'max'),  # MIS maximizes the size of the independent set
}


def arg_parser():
    parser = ArgumentParser(
        description='EDISCO: equivariant discrete diffusion for Euclidean combinatorial optimization.')

    parser.add_argument('--task', type=str, default='tsp',
                        choices=['tsp', 'cvrp', 'cvrp_partition', 'steiner', 'mis'],
                        help='Problem to solve')

    # Data (paths are relative to --storage_path)
    parser.add_argument('--storage_path', type=str, required=True,
                        help='Root directory of the data splits; checkpoints and logs go to <storage_path>/models')
    parser.add_argument('--training_split', type=str, default='data/tsp/tsp50_train_concorde.txt')
    parser.add_argument('--validation_split', type=str, default='data/tsp/tsp50_test_concorde.txt')
    parser.add_argument('--test_split', type=str, default='data/tsp/tsp50_test_concorde.txt')
    parser.add_argument('--validation_examples', type=int, default=64,
                        help='Number of validation instances evaluated after every epoch')
    parser.add_argument('--training_split_label_dir', type=str, default=None,
                        help='Directory with external label files for MIS training graphs')

    # Optimization
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--accumulate_grad_batches', type=int, default=1,
                        help='Accumulate gradients over this many batches per optimizer step; '
                             'the effective batch size is batch_size * accumulate_grad_batches')
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=2e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--lr_scheduler', type=str, default='cosine-decay',
                        choices=['constant', 'cosine-decay', 'one-cycle'])
    parser.add_argument('--gradient_clip_val', type=float, default=1.0,
                        help='Gradient-norm clipping threshold (0 disables)')
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--fp16', action='store_true', help='Mixed-precision training')
    parser.add_argument('--use_activation_checkpoint', action='store_true',
                        help='Recompute layer activations in the backward pass to save memory')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for training and sampling')

    # Forward process: categorical CTMC with noise rate beta(t)
    parser.add_argument('--beta_schedule', type=str, default='linear',
                        choices=['linear', 'exponential', 'cosine'],
                        help='Noise-rate schedule beta(t)')
    parser.add_argument('--beta_min', type=float, default=0.1)
    parser.add_argument('--beta_max', type=float, default=1.5)
    parser.add_argument('--beta_exp_a', type=float, default=0.5,
                        help='Parameter a of the exponential schedule beta(t) = a b^t log(b)')
    parser.add_argument('--beta_exp_b', type=float, default=4.0,
                        help='Parameter b of the exponential schedule beta(t) = a b^t log(b)')

    # Reverse process
    parser.add_argument('--solver_type', type=str, default='deis',
                        choices=['euler', 'ddim', 'pndm', 'dpm2', 'deis', 'rk4', 'heun'],
                        help='Reverse-time solver')
    parser.add_argument('--solver_steps', type=int, default=5,
                        help='Number of reverse steps')
    parser.add_argument('--time_schedule', type=str, default='linear',
                        choices=['linear', 'cosine', 'quadratic'],
                        help='Spacing of the sampling times between t = 1 and t = 0')
    parser.add_argument('--sequential_sampling', type=int, default=1,
                        help='Number of sampling rounds per instance; the best solution is kept')
    parser.add_argument('--parallel_sampling', type=int, default=1,
                        help='Number of samples per round; the best solution is kept')

    # Decoding
    parser.add_argument('--decoder', choices=['greedy', 'nee'], default='nee',
                        help='Feasible probability-only greedy or distance-aware NEE decoding')
    parser.add_argument('--two_opt_iterations', type=int, default=0,
                        help='2-opt local search iterations (reference comparisons only, 0 disables)')

    # Architecture
    parser.add_argument('--n_layers', type=int, default=12)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--node_dim', type=int, default=64, help='Node embedding size of the EGNN')
    parser.add_argument('--edge_dim', type=int, default=64, help='Edge embedding size of the EGNN')
    parser.add_argument('--time_dim', type=int, default=128, help='Time embedding size of the EGNN')
    parser.add_argument('--coord_update_alpha', type=float, default=0.1,
                        help='Step size of the equivariant coordinate update')
    parser.add_argument('--weight_temp', type=float, default=10.0,
                        help='Temperature of the coordinate-weight tanh')
    parser.add_argument('--sparse_factor', type=int, default=-1,
                        help='k of the k-nearest-neighbour graph (<= 0 uses dense graphs)')

    # CVRP
    parser.add_argument('--disable_capacity_conditioning', action='store_true',
                        help='Train the unconditioned CVRP variant of the capacity-shift study')
    parser.add_argument('--default_capacity', type=float, default=None,
                        help='Reference capacity Q_default (default: capacity of the training set)')
    parser.add_argument('--merge_routes', action='store_true',
                        help='Merge decoded routes when the capacity allows')

    # Partition diffusion for large-scale CVRP (--task cvrp_partition)
    parser.add_argument('--partitioner', type=str, default='diffusion', choices=['diffusion', 'kmeans'],
                        help='Customer partitioner: partition diffusion, or k-means on coordinates (ablation)')
    parser.add_argument('--sub_tsp_ckpt', type=str, default=None,
                        help='EDISCO TSP checkpoint used to solve each cluster plus depot')
    parser.add_argument('--sub_tsp_n_layers', type=int, default=None,
                        help='Layers of the sub-TSP score network (default: --n_layers)')
    parser.add_argument('--sub_tsp_hidden_dim', type=int, default=None,
                        help='Hidden dimension of the sub-TSP score network (default: --hidden_dim)')
    parser.add_argument('--sub_tsp_solver_type', type=str, default='deis')
    parser.add_argument('--sub_tsp_solver_steps', type=int, default=5)

    # Symmetry ablations (TSP)
    parser.add_argument('--disable_equivariance', action='store_true',
                        help='Replace the EGNN with the non-equivariant GNN of matched depth and width')
    parser.add_argument('--data_augmentation', type=str, default='none', choices=['none', 'e2'],
                        help='Random E(2) transformation of the training coordinates')
    parser.add_argument('--symmetry_loss', action='store_true',
                        help='Soft E(2)-consistency regularizer on predicted edge probabilities')
    parser.add_argument('--symmetry_loss_weight', type=float, default=1.0)

    # Evaluation protocol
    parser.add_argument('--test_equivariance', action='store_true',
                        help='Report the edge-probability consistency probe under random E(2) transformations')
    parser.add_argument('--equivariance_probe_samples', type=int, default=16,
                        help='Random E(2) elements per instance for --test_equivariance')
    parser.add_argument('--single_thread', action='store_true',
                        help='Restrict PyTorch to one CPU thread (timing protocol)')

    # Logging and checkpoints
    parser.add_argument('--project_name', type=str, default='edisco')
    parser.add_argument('--wandb_entity', type=str, default=None)
    parser.add_argument('--wandb_logger_name', type=str, default=None)
    parser.add_argument('--resume_id', type=str, default=None, help='W&B run id to resume')
    parser.add_argument('--ckpt_path', type=str, default=None)
    parser.add_argument('--resume_weight_only', action='store_true',
                        help='Initialize the weights from --ckpt_path without restoring the optimizer state')

    # Modes
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_test', action='store_true')
    parser.add_argument('--do_valid_only', action='store_true')

    return parser.parse_args()


def main(args):
    if args.seed is not None:
        seed_everything(args.seed, workers=True)
    if args.single_thread:
        torch.set_num_threads(1)

    model_class, saving_mode = TASKS[args.task]
    model = model_class(param_args=args)

    n_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
    rank_zero_info(f"Task: {args.task}, score network: {model.model.__class__.__name__} "
                   f"({n_params:,} parameters, {args.n_layers} layers, hidden dimension {args.hidden_dim})")
    rank_zero_info(f"Graphs: {'k-NN with k = %d' % args.sparse_factor if args.sparse_factor > 0 else 'dense'}, "
                   f"solver: {args.solver_type} ({args.solver_steps} steps), decoder: {args.decoder}")

    wandb_id = os.getenv("WANDB_RUN_ID") or wandb.util.generate_id()
    wandb_logger = WandbLogger(
        name=args.wandb_logger_name,
        project=args.project_name,
        entity=args.wandb_entity,
        save_dir=os.path.join(args.storage_path, 'models'),
        id=args.resume_id or wandb_id,
        config=vars(args)
    )
    logger_name = args.wandb_logger_name or args.project_name

    checkpoint_callback = ModelCheckpoint(
        monitor='val/solved_cost',
        mode=saving_mode,
        save_top_k=3,
        save_last=True,
        dirpath=os.path.join(wandb_logger.save_dir, logger_name, wandb_logger._id, 'checkpoints'),
        filename='epoch={epoch:03d}-val_cost={val/solved_cost:.5f}',
        auto_insert_metric_name=False,
    )
    rank_zero_info(f"Checkpoints: {checkpoint_callback.dirpath}")

    trainer_kwargs = {
        'accelerator': "auto",
        'devices': torch.cuda.device_count() if torch.cuda.is_available() else None,
        'max_epochs': args.num_epochs,
        'callbacks': [TQDMProgressBar(refresh_rate=20), checkpoint_callback,
                      LearningRateMonitor(logging_interval='step')],
        'logger': wandb_logger,
        'check_val_every_n_epoch': 1,
        'precision': 16 if args.fp16 else 32,
    }
    if args.accumulate_grad_batches > 1:
        trainer_kwargs['accumulate_grad_batches'] = args.accumulate_grad_batches
    if args.gradient_clip_val and args.gradient_clip_val > 0:
        trainer_kwargs['gradient_clip_val'] = args.gradient_clip_val
        trainer_kwargs['gradient_clip_algorithm'] = 'norm'

    # Multi-GPU runs use DDP; a single device needs no distributed strategy.
    # The node and coordinate updates of the last EGNN layer do not reach the
    # output head, so their parameters are unused in the backward pass. The
    # static-graph optimization does not support gradient accumulation.
    if trainer_kwargs['devices'] is not None and trainer_kwargs['devices'] > 1:
        trainer_kwargs['strategy'] = DDPStrategy(
            find_unused_parameters=True,
            static_graph=args.accumulate_grad_batches == 1)

    trainer = Trainer(**trainer_kwargs)

    ckpt_path = args.ckpt_path

    if args.do_train:
        if args.resume_weight_only:
            # Initialize the weights from a checkpoint (curriculum training).
            model = model_class.load_from_checkpoint(ckpt_path, param_args=args)
            trainer.fit(model)
        else:
            trainer.fit(model, ckpt_path=ckpt_path)

        # Record the checkpoint with the best validation performance; it
        # initializes curriculum training at the next problem scale.
        if trainer.is_global_zero and checkpoint_callback.best_model_path:
            with open(os.path.join(checkpoint_callback.dirpath, 'best_checkpoint.txt'), 'w') as f:
                print(checkpoint_callback.best_model_path, file=f)
            rank_zero_info(f"Best checkpoint: {checkpoint_callback.best_model_path} "
                           f"(val/solved_cost={checkpoint_callback.best_model_score})")

        if args.do_test:
            trainer.test(ckpt_path=checkpoint_callback.best_model_path)

    elif args.do_test:
        trainer.validate(model, ckpt_path=ckpt_path)
        if not args.do_valid_only:
            trainer.test(model, ckpt_path=ckpt_path)

    trainer.logger.finalize("success")


if __name__ == '__main__':
    main(arg_parser())
