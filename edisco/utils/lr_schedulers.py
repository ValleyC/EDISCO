"""Learning rate scheduler implementations for EDISCO"""

from functools import partial
import torch
from torch.optim.lr_scheduler import LambdaLR


def get_schedule_fn(scheduler, num_training_steps):
    """
    Returns a callable scheduler_fn(optimizer).
    
    Args:
        scheduler: scheduler name ('constant', 'cosine-decay', 'one-cycle')
        num_training_steps: total number of training steps
    
    Returns:
        scheduler_fn: function that creates a scheduler given an optimizer
    """
    if scheduler == "constant":
        # No scheduling, constant learning rate
        scheduler_fn = lambda optimizer: None
        
    elif scheduler == "cosine-decay":
        # Cosine annealing to 0
        scheduler_fn = partial(
            torch.optim.lr_scheduler.CosineAnnealingLR,
            T_max=num_training_steps,
            eta_min=0.0,
        )
        
    elif scheduler == "one-cycle":
        # One-cycle learning rate schedule
        scheduler_fn = partial(
            get_one_cycle,
            num_training_steps=num_training_steps,
        )
        
    elif scheduler == "linear":
        # Linear decay to 0
        scheduler_fn = partial(
            get_linear_schedule_with_warmup,
            num_warmup_steps=num_training_steps // 10,
            num_training_steps=num_training_steps,
        )
        
    elif scheduler == "exponential":
        # Exponential decay
        scheduler_fn = partial(
            torch.optim.lr_scheduler.ExponentialLR,
            gamma=0.99,
        )
        
    elif scheduler == "polynomial":
        # Polynomial decay
        scheduler_fn = partial(
            get_polynomial_decay_schedule_with_warmup,
            num_warmup_steps=num_training_steps // 10,
            num_training_steps=num_training_steps,
            power=1.0,
        )
        
    else:
        raise ValueError(f"Invalid schedule {scheduler} given.")
    
    return scheduler_fn


def get_one_cycle(optimizer, num_training_steps):
    """
    Simple single-cycle scheduler (simplified version of Leslie Smith's one-cycle).
    
    Args:
        optimizer: PyTorch optimizer
        num_training_steps: total training steps
    
    Returns:
        scheduler: LambdaLR scheduler
    """
    def lr_lambda(current_step):
        # First half: increase from 0 to 1
        if current_step < num_training_steps / 2:
            return float(current_step / (num_training_steps / 2))
        # Second half: decrease from 1 to 0
        else:
            return float(2 - current_step / (num_training_steps / 2))
    
    return LambdaLR(optimizer, lr_lambda, -1)


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """
    Linear schedule with warmup.
    
    Args:
        optimizer: PyTorch optimizer
        num_warmup_steps: number of warmup steps
        num_training_steps: total training steps
        last_epoch: last epoch number
    
    Returns:
        scheduler: LambdaLR scheduler
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            # Warmup phase
            return float(current_step) / float(max(1, num_warmup_steps))
        # Linear decay
        return max(
            0.0,
            float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )
    
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_polynomial_decay_schedule_with_warmup(
    optimizer, num_warmup_steps, num_training_steps, lr_end=1e-7, power=1.0, last_epoch=-1
):
    """
    Polynomial decay schedule with warmup.
    
    Args:
        optimizer: PyTorch optimizer
        num_warmup_steps: number of warmup steps
        num_training_steps: total training steps
        lr_end: final learning rate
        power: polynomial power
        last_epoch: last epoch number
    
    Returns:
        scheduler: LambdaLR scheduler
    """
    lr_init = optimizer.defaults["lr"]
    if not (lr_init > lr_end):
        raise ValueError(f"lr_end ({lr_end}) must be smaller than initial lr ({lr_init})")
    
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            # Warmup phase
            return float(current_step) / float(max(1, num_warmup_steps))
        elif current_step > num_training_steps:
            # After training, use lr_end
            return lr_end / lr_init
        else:
            # Polynomial decay
            lr_range = lr_init - lr_end
            decay_steps = num_training_steps - num_warmup_steps
            pct_remaining = 1 - (current_step - num_warmup_steps) / decay_steps
            decay = lr_range * pct_remaining ** power + lr_end
            return decay / lr_init
    
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1):
    """
    Cosine schedule with warmup.
    
    Args:
        optimizer: PyTorch optimizer
        num_warmup_steps: number of warmup steps
        num_training_steps: total training steps
        num_cycles: number of cosine cycles
        last_epoch: last epoch number
    
    Returns:
        scheduler: LambdaLR scheduler
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            # Warmup phase
            return float(current_step) / float(max(1, num_warmup_steps))
        # Cosine decay
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
    
    import math
    return LambdaLR(optimizer, lr_lambda, last_epoch)