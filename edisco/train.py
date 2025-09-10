"""The handler for training and evaluation of EDISCO models."""

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

from pl_tsp_model import TSPModel


def arg_parser():
    parser = ArgumentParser(description='Train EDISCO continuous-time diffusion model on TSP dataset.')
    parser.add_argument('--task', type=str, required=True, choices=['tsp'])
    parser.add_argument('--storage_path', type=str, required=True)
    parser.add_argument('--training_split', type=str, default='data/tsp/tsp50_train_concorde.txt')
    parser.add_argument('--validation_split', type=str, default='data/tsp/tsp50_test_concorde.txt')
    parser.add_argument('--test_split', type=str, default='data/tsp/tsp50_test_concorde.txt')
    parser.add_argument('--validation_examples', type=int, default=64)

    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=2e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--lr_scheduler', type=str, default='cosine-decay',
                        choices=['constant', 'cosine-decay', 'one-cycle'])
    parser.add_argument('--gradient_clip_val', type=float, default=1.0)

    # Model arguments
    parser.add_argument('--n_layers', type=int, default=8)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--node_dim', type=int, default=64)
    parser.add_argument('--coord_dim', type=int, default=2)
    parser.add_argument('--num_classes', type=int, default=2)  # Binary adjacency matrix

    # Continuous-time diffusion arguments
    parser.add_argument('--beta_min', type=float, default=0.1)
    parser.add_argument('--beta_max', type=float, default=2.0)
    parser.add_argument('--loss_type', type=str, default='elbo', choices=['elbo', 'score_matching'])
    parser.add_argument('--time_sampling', type=str, default='uniform', 
                        choices=['uniform', 'importance', 'cosine'])
    
    # Inference arguments
    parser.add_argument('--inference_diffusion_steps', type=int, default=50)
    parser.add_argument('--inference_schedule', type=str, default='linear',
                        choices=['linear', 'cosine', 'adaptive'])
    parser.add_argument('--inference_method', type=str, default='simple',
                        choices=['simple', 'predictor_corrector'])
    parser.add_argument('--sequential_sampling', type=int, default=1)
    parser.add_argument('--parallel_sampling', type=int, default=1)
    parser.add_argument('--two_opt_iterations', type=int, default=1000)
    parser.add_argument('--save_numpy_heatmap', action='store_true')

    # System arguments
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--use_activation_checkpoint', action='store_true')

    # Logging arguments
    parser.add_argument('--project_name', type=str, default='edisco_tsp')
    parser.add_argument('--wandb_entity', type=str, default=None)
    parser.add_argument('--wandb_logger_name', type=str, default=None)
    parser.add_argument("--resume_id", type=str, default=None, help="Resume training on wandb.")
    
    # Checkpoint arguments
    parser.add_argument('--ckpt_path', type=str, default=None)
    parser.add_argument('--resume_weight_only', action='store_true',
                        help='Load weights only for curriculum learning')

    # Action arguments
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_test', action='store_true')
    parser.add_argument('--do_valid_only', action='store_true')

    args = parser.parse_args()
    return args


def main(args):
    epochs = args.num_epochs
    project_name = args.project_name

    if args.task == 'tsp':
        model_class = TSPModel
        saving_mode = 'min'
    else:
        raise NotImplementedError(f"Task {args.task} not implemented")

    model = model_class(param_args=args)

    # Setup WandB logger
    wandb_id = os.getenv("WANDB_RUN_ID") or wandb.util.generate_id()
    wandb_logger = WandbLogger(
        name=args.wandb_logger_name,
        project=project_name,
        entity=args.wandb_entity,
        save_dir=os.path.join(args.storage_path, f'models'),
        id=args.resume_id or wandb_id,
    )
    rank_zero_info(f"Logging to {wandb_logger.save_dir}/{wandb_logger.name}/{wandb_logger.version}")

    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='val/avg_gap',
        mode=saving_mode,
        save_top_k=3,
        save_last=True,
        dirpath=os.path.join(
            wandb_logger.save_dir,
            args.wandb_logger_name or 'default',
            wandb_logger._id,
            'checkpoints'
        ),
        filename='{epoch:02d}-{val_avg_gap:.2f}'
    )
    lr_callback = LearningRateMonitor(logging_interval='step')

    # Setup trainer
    trainer = Trainer(
        accelerator="auto",
        devices=torch.cuda.device_count() if torch.cuda.is_available() else None,
        max_epochs=epochs,
        callbacks=[TQDMProgressBar(refresh_rate=20), checkpoint_callback, lr_callback],
        logger=wandb_logger,
        check_val_every_n_epoch=10,
        strategy=DDPStrategy(static_graph=True) if torch.cuda.device_count() > 1 else 'auto',
        precision=16 if args.fp16 else 32,
        gradient_clip_val=args.gradient_clip_val,
    )

    rank_zero_info(
        f"{'-' * 100}\n"
        f"EDISCO Model Configuration:\n"
        f"  Task: {args.task}\n"
        f"  Model: {args.n_layers} EGNN layers, {args.hidden_dim} hidden dim\n"
        f"  Continuous-time diffusion: β(t) ∈ [{args.beta_min}, {args.beta_max}]\n"
        f"  Loss type: {args.loss_type}\n"
        f"  Batch size: {args.batch_size}\n"
        f"  Learning rate: {args.learning_rate}\n"
        f"{'-' * 100}\n"
    )

    ckpt_path = args.ckpt_path

    if args.do_train:
        if args.resume_weight_only:
            # Curriculum learning - load weights only
            model = model_class.load_from_checkpoint(ckpt_path, param_args=args)
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