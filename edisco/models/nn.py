"""Neural-network helpers shared by the encoders.

The timestep embedding and activation checkpointing follow the utilities of
guided-diffusion (Dhariwal & Nichol, 2021) as used in DIFUSCO (Sun & Yang, 2023).
"""

import math

import torch as th
import torch.nn as nn


class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


def linear(*args, **kwargs):
    """Create a linear module."""
    return nn.Linear(*args, **kwargs)


def zero_module(module):
    """Zero out the parameters of a module and return it."""
    for p in module.parameters():
        p.detach().zero_()
    return module


def normalization(channels):
    """Group normalization with 32 groups."""
    return GroupNorm32(32, channels)


def timestep_embedding(timesteps, dim, max_period=10000):
    """Sinusoidal timestep embeddings.

    Args:
        timesteps: 1-D tensor of N (possibly fractional) times
        dim: dimension of the output
        max_period: controls the minimum frequency of the embeddings
    Returns:
        (N, dim) tensor of embeddings
    """
    half = dim // 2
    freqs = th.exp(
        -math.log(max_period) * th.arange(start=0, end=half, dtype=th.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = th.cat([th.cos(args), th.sin(args)], dim=-1)
    if dim % 2:
        embedding = th.cat([embedding, th.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def checkpoint(func, inputs, params, flag):
    """Evaluate a function without caching intermediate activations.

    Memory is traded for a second forward pass during backpropagation; the
    gradients are identical to those of the plain call.

    Args:
        func: the function to evaluate
        inputs: the argument sequence to pass to `func`
        params: parameters `func` depends on but does not take as arguments
        flag: if False, call `func` directly
    """
    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    return func(*inputs)


class CheckpointFunction(th.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])
        # The recomputation in backward must use the same precision mode.
        ctx.autocast = th.is_autocast_enabled()
        with th.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
        with th.enable_grad(), th.cuda.amp.autocast(enabled=ctx.autocast):
            # Shallow copies guard against a first op in run_function that
            # modifies the storage of a detached tensor in place.
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        input_grads = th.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (None, None) + input_grads
