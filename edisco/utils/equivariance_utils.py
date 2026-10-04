"""E(2) transformations of planar coordinates: rotations, reflections, translations."""

import math

import torch


def apply_rotation(coords, angle):
    """Rotate coordinates of shape (..., 2) by `angle` radians about the origin."""
    cos, sin = math.cos(angle), math.sin(angle)
    rotation = torch.tensor([[cos, -sin], [sin, cos]], device=coords.device, dtype=coords.dtype)
    return torch.matmul(coords, rotation.T)


def apply_translation(coords, translation):
    """Translate coordinates of shape (..., 2) by a vector of shape (2,)."""
    return coords + translation


def apply_reflection(coords, axis='x'):
    """Reflect coordinates of shape (..., 2) across the x or y axis."""
    reflected = coords.clone()
    if axis == 'x':
        reflected[..., 1] = -reflected[..., 1]
    elif axis == 'y':
        reflected[..., 0] = -reflected[..., 0]
    else:
        raise ValueError(f"Unknown axis: {axis}")
    return reflected


def apply_random_e2_transform(coords, include_reflection=False):
    """Apply one random rotation, translation and optional reflection to all coordinates.

    Returns the transformed coordinates and the transformation parameters.
    """
    angle = torch.rand(1).item() * 2 * math.pi
    translation = torch.randn(2, device=coords.device, dtype=coords.dtype) * 0.5
    transformed = apply_translation(apply_rotation(coords, angle), translation)
    reflection_axis = None
    if include_reflection and torch.rand(1).item() > 0.5:
        reflection_axis = 'x' if torch.rand(1).item() > 0.5 else 'y'
        transformed = apply_reflection(transformed, reflection_axis)
    params = {'angle': angle, 'translation': translation, 'reflection_axis': reflection_axis}
    return transformed, params


def random_e2_transform(coords, node_batch=None, center=0.5, max_shift=0.5):
    """Apply an independent random element of E(2) to every graph of a batch.

    Each graph is rotated by a uniform angle about `center`, reflected with
    probability one half and translated by a uniform shift in
    [-max_shift, max_shift]^2.

    Args:
        coords: (B, n, 2) dense batch, or (N, 2) nodes of a sparse batch
        node_batch: (N,) graph index of every node for sparse batches
    """
    dense = coords.dim() == 3
    if dense:
        n_graphs = coords.shape[0]
    else:
        n_graphs = 1 if node_batch is None else int(node_batch.max().item()) + 1
    device, dtype = coords.device, coords.dtype
    theta = torch.rand(n_graphs, device=device, dtype=dtype) * (2 * math.pi)
    cos, sin = torch.cos(theta), torch.sin(theta)
    sign = torch.where(torch.rand(n_graphs, device=device) < 0.5, -1.0, 1.0).to(dtype)
    # Rotation composed with an optional reflection of the second axis.
    matrix = torch.stack([torch.stack([cos, -sin * sign], -1),
                          torch.stack([sin, cos * sign], -1)], -2)
    shift = (torch.rand(n_graphs, 2, device=device, dtype=dtype) * 2 - 1) * max_shift
    if dense:
        return torch.matmul(coords - center, matrix.transpose(1, 2)) + center + shift.unsqueeze(1)
    if node_batch is None:
        node_batch = torch.zeros(coords.shape[0], dtype=torch.long, device=device)
    rotated = torch.bmm(matrix[node_batch], (coords - center).unsqueeze(-1)).squeeze(-1)
    return rotated + center + shift[node_batch]
