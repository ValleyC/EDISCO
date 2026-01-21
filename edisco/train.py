"""The handler for training and evaluation with EDISCO."""

import os
from argparse import ArgumentParser

import torch
import wandb
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.utilities import rank_zero_info

from pl_edisco_model import EDISCOModel
from pl_cvrp_model import CVRPModel
from pl_steiner_model import SteinerTreeModel
from pl_mis_model import MISModel


def arg_parser():
    parser = ArgumentParser(description='Train EDISCO: Equivariant Continuous-time Diffusion for Combinatorial Optimization.')
    
    # Task selection
    parser.add_argument('--task', type=str, default='tsp', choices=['tsp', 'cvrp', 'steiner', 'mis'],
                        help='Problem type to solve')
    
    # Data paths
    parser.add_argument('--storage_path', type=str, required=True)
    parser.add_argument('--training_split', type=str, default='data/tsp/tsp50_train_concorde.txt')
    parser.add_argument('--validation_split', type=str, default='data/tsp/tsp50_test_concorde.txt')
    parser.add_argument('--test_split', type=str, default='data/tsp/tsp50_test_concorde.txt')
    parser.add_argument('--validation_examples', type=int, default=64)

    # MIS-specific data paths
    parser.add_argument('--training_split_label_dir', type=str, default=None,
                        help='Directory containing external label files for MIS training data')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--lr_scheduler', type=str, default='constant')
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--use_activation_checkpoint', action='store_true')
    
    # Diffusion parameters
    parser.add_argument('--diffusion_type', type=str, default='categorical', choices=['gaussian', 'categorical'])
    parser.add_argument('--diffusion_schedule', type=str, default='linear')
    parser.add_argument('--diffusion_steps', type=int, default=1000)
    
    # EDISCO architecture choices
    parser.add_argument('--continuous_time', action='store_true', default=True,
                        help='Use continuous-time diffusion (EDISCO default)')
    parser.add_argument('--equivariant', action='store_true', default=True,
                        help='Use E(2)-equivariant architecture (EDISCO default)')
    
    # Continuous-time diffusion parameters
    parser.add_argument('--beta_min', type=float, default=0.1,
                        help='Minimum noise rate for continuous-time diffusion')
    parser.add_argument('--beta_max', type=float, default=1.5,
                        help='Maximum noise rate for continuous-time diffusion')
    
    # Inference parameters
    parser.add_argument('--inference_diffusion_steps', type=int, default=1000)
    parser.add_argument('--inference_schedule', type=str, default='linear')
    parser.add_argument('--inference_trick', type=str, default="ddim")
    parser.add_argument('--sequential_sampling', type=int, default=1)
    parser.add_argument('--parallel_sampling', type=int, default=1)
    
    # EDISCO solver parameters
    parser.add_argument('--solver_type', type=str, default='pndm',
                        choices=['euler', 'ddim', 'pndm', 'dpm2', 'deis', 'rk4', 'heun'],
                        help='ODE solver for continuous-time diffusion')
    parser.add_argument('--solver_steps', type=int, default=50,
                        help='Number of steps for ODE solver')
    parser.add_argument('--time_schedule', type=str, default='linear',
                        choices=['linear', 'cosine', 'quadratic'],
                        help='Time schedule for continuous-time sampling')
    parser.add_argument('--adaptive_mixing', action='store_true', default=True,
                        help='Use adaptive mixing strategy in EDISCO')
    parser.add_argument('--deterministic_threshold', type=float, default=0.1,
                        help='Time threshold for deterministic decoding')
    
    # Architecture parameters
    parser.add_argument('--n_layers', type=int, default=12)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--node_dim', type=int, default=64, help='Node dimension for EGNN')
    parser.add_argument('--edge_dim', type=int, default=64, help='Edge dimension for EGNN')
    parser.add_argument('--time_dim', type=int, default=128, help='Time embedding dimension for EGNN')
    parser.add_argument('--coord_dim', type=int, default=2, help='Coordinate dimension')
    parser.add_argument('--coord_update_alpha', type=float, default=0.1, 
                        help='Step size for coordinate updates in EGNN')
    parser.add_argument('--weight_temp', type=float, default=10.0,
                        help='Temperature for weight scaling in EGNN')
    
    # Graph parameters
    parser.add_argument('--sparse_factor', type=int, default=-1,
                        help='Sparsity factor for k-NN graphs (0=dense, >0=sparse with k edges)')
    parser.add_argument('--aggregation', type=str, default='sum')
    parser.add_argument('--two_opt_iterations', type=int, default=1000)
    parser.add_argument('--save_numpy_heatmap', action='store_true')
    
    # CVRP-specific parameters
    parser.add_argument('--merge_routes', action='store_true', default=False,
                        help='Try to merge routes to reduce vehicle count (CVRP only)')
    
    # Logging and checkpointing
    parser.add_argument('--project_name', type=str, default='edisco_tsp')
    parser.add_argument('--wandb_entity', type=str, default=None)
    parser.add_argument('--wandb_logger_name', type=str, default=None)
    parser.add_argument("--resume_id", type=str, default=None, help="Resume training on wandb.")
    parser.add_argument('--ckpt_path', type=str, default=None)
    parser.add_argument('--resume_weight_only', action='store_true')
    
    # Training/evaluation modes
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_test', action='store_true')
    parser.add_argument('--do_valid_only', action='store_true')
    
    # EDISCO evaluation modes
    parser.add_argument('--compare_solvers', action='store_true',
                        help='Compare different ODE solvers')
    parser.add_argument('--test_equivariance', action='store_true',
                        help='Test E(2) equivariance with rotations')
    
    # Ablation study flags
    parser.add_argument('--disable_continuous_time', action='store_true',
                        help='Disable continuous-time for ablation study')
    parser.add_argument('--disable_equivariance', action='store_true',
                        help='Disable equivariance for ablation study')
    
    # Performance optimization flags
    parser.add_argument('--force_dense_only', action='store_true',
                        help='Force dense-only optimizations (overrides sparse_factor)')
    parser.add_argument('--disable_optimizations', action='store_true',
                        help='Disable all performance optimizations for debugging')

    args = parser.parse_args()

    # Determine execution mode based on sparse_factor
    if args.force_dense_only:
        args.sparse_factor = 0
        args.dense_only = True
        rank_zero_info("Forcing dense-only mode for maximum performance")
    elif args.sparse_factor == 0:
        args.dense_only = True
        rank_zero_info("Using dense-only mode (sparse_factor=0)")
    else:
        args.dense_only = False
        rank_zero_info(f"Using sparse mode (sparse_factor={args.sparse_factor})")

    # Apply ablation study settings if specified
    if args.disable_continuous_time:
        args.continuous_time = False
        rank_zero_info("Ablation study: continuous-time disabled")

    if args.disable_equivariance:
        args.equivariant = False
        rank_zero_info("Ablation study: equivariance disabled")

    # Warn about suboptimal configurations
    if args.diffusion_type == 'gaussian' and args.continuous_time:
        rank_zero_info("Warning: EDISCO works best with categorical diffusion")

    # Log optimization status
    if not args.disable_optimizations:
        mode_name = "dense-only" if args.dense_only else "sparse"
        rank_zero_info(f"Performance optimizations enabled: {mode_name} path")
    else:
        rank_zero_info("Performance optimizations disabled (debug mode)")
    
    return args


def main(args):
    epochs = args.num_epochs
    project_name = args.project_name

    # Log execution mode
    execution_mode = "dense-only" if hasattr(args, 'dense_only') and args.dense_only else "sparse"
    rank_zero_info(f"Execution mode: {execution_mode} (sparse_factor={args.sparse_factor})")

    # Select model based on task
    if args.task == 'tsp':
        model = EDISCOModel(param_args=args)
        saving_mode = 'min'
        rank_zero_info("Using EDISCOModel for TSP")
    elif args.task == 'cvrp':
        model = CVRPModel(param_args=args)
        saving_mode = 'min'
        rank_zero_info("Using CVRPModel for CVRP")
    elif args.task == 'steiner':
        model = SteinerTreeModel(param_args=args)
        saving_mode = 'min'
        rank_zero_info("Using SteinerTreeModel for Euclidean Steiner Tree")
    elif args.task == 'mis':
        model = MISModel(param_args=args)
        saving_mode = 'max'  # MIS is a maximization problem
        rank_zero_info("Using MISModel for Maximum Independent Set")
        rank_zero_info("Note: MIS uses standard GNN (not EGNN) - MIS is not a geometric problem")
    else:
        raise ValueError(f"Unknown task: {args.task}")

    # Log model configuration
    encoder_name = model.model.__class__.__name__ if hasattr(model, 'model') else 'Unknown'
    rank_zero_info(f"Model: {model.__class__.__name__} with {encoder_name} encoder")
    rank_zero_info(f"Continuous-time: {args.continuous_time}, Equivariant: {args.equivariant}")
    
    wandb_id = os.getenv("WANDB_RUN_ID") or wandb.util.generate_id()
    wandb_logger = WandbLogger(
        name=args.wandb_logger_name,
        project=project_name,
        entity=args.wandb_entity,
        save_dir=os.path.join(args.storage_path, 'models'),
        id=args.resume_id or wandb_id,
        config=vars(args)
    )
    rank_zero_info(f"Logging to {wandb_logger.save_dir}/{wandb_logger.name}/{wandb_logger.version}")

    # Use wandb_logger_name if provided, otherwise use project_name
    logger_name = args.wandb_logger_name or project_name

    checkpoint_callback = ModelCheckpoint(
        monitor='val/solved_cost',
        mode=saving_mode,
        save_top_k=3,
        save_last=True,
        dirpath=os.path.join(wandb_logger.save_dir,
                            logger_name,
                            wandb_logger._id,
                            'checkpoints'),
    )
    lr_callback = LearningRateMonitor(logging_interval='step')

    # Configure trainer
    trainer_kwargs = {
        'accelerator': "auto",
        'devices': torch.cuda.device_count() if torch.cuda.is_available() else None,
        'max_epochs': epochs,
        'callbacks': [TQDMProgressBar(refresh_rate=20), checkpoint_callback, lr_callback],
        'logger': wandb_logger,
        'check_val_every_n_epoch': 1,
        'precision': 16 if args.fp16 else 32,
    }

    # Use optimized DDP strategy for dense-only mode
    if getattr(args, 'dense_only', False) and not args.disable_optimizations:
        trainer_kwargs['strategy'] = DDPStrategy(static_graph=True, find_unused_parameters=False)
        rank_zero_info("Using optimized DDP strategy")
    else:
        trainer_kwargs['strategy'] = DDPStrategy(static_graph=True)

    trainer = Trainer(**trainer_kwargs)

    # Log training configuration
    solver_info = f"{args.solver_type} ({args.solver_steps} steps)" if args.continuous_time else "N/A"
    model_type = "EGNN" if args.equivariant else "Standard GNN"
    rank_zero_info(f"Training config: {args.n_layers} layers, {args.hidden_dim}D hidden, {model_type}")
    rank_zero_info(f"Diffusion: {args.diffusion_type}, Solver: {solver_info}")
    if args.continuous_time:
        rank_zero_info(f"Adaptive mixing: {args.adaptive_mixing}")

    # Count model parameters
    if hasattr(model, 'model'):
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        rank_zero_info(f"Parameters: {trainable_params:,} trainable / {total_params:,} total")
    
    ckpt_path = args.ckpt_path
    
    if args.do_train:
        if args.resume_weight_only:
            # Resume from checkpoint (weights only)
            if args.task == 'tsp':
                model = EDISCOModel.load_from_checkpoint(ckpt_path, param_args=args)
            elif args.task == 'cvrp':
                model = CVRPModel.load_from_checkpoint(ckpt_path, param_args=args)
            elif args.task == 'steiner':
                model = SteinerTreeModel.load_from_checkpoint(ckpt_path, param_args=args)
            elif args.task == 'mis':
                model = MISModel.load_from_checkpoint(ckpt_path, param_args=args)
            trainer.fit(model)
        else:
            trainer.fit(model, ckpt_path=ckpt_path)

        if args.do_test:
            trainer.test(ckpt_path=checkpoint_callback.best_model_path)

    elif args.do_test:
        trainer.validate(model, ckpt_path=ckpt_path)
        if not args.do_valid_only:
            trainer.test(model, ckpt_path=ckpt_path)

    trainer.logger.finalize("success")


if __name__ == '__main__':
    args = arg_parser()
    main(args)