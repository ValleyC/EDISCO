"""
Utilities for testing and ensuring E(2) equivariance in EDISCO models.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional


def apply_rotation(coords, angle):
    """
    Apply 2D rotation to coordinates.
    
    Args:
        coords: (batch_size, n_points, 2) or (n_points, 2)
        angle: Rotation angle in radians
        
    Returns:
        Rotated coordinates
    """
    cos_angle = torch.cos(torch.tensor(angle))
    sin_angle = torch.sin(torch.tensor(angle))
    
    rotation_matrix = torch.tensor([
        [cos_angle, -sin_angle],
        [sin_angle, cos_angle]
    ], device=coords.device, dtype=coords.dtype)
    
    # Apply rotation
    if coords.dim() == 3:
        # Batched
        return torch.matmul(coords, rotation_matrix.T)
    else:
        # Single sample
        return torch.matmul(coords, rotation_matrix.T)


def apply_translation(coords, translation):
    """
    Apply translation to coordinates.
    
    Args:
        coords: (batch_size, n_points, 2) or (n_points, 2)
        translation: (2,) translation vector
        
    Returns:
        Translated coordinates
    """
    return coords + translation


def apply_reflection(coords, axis='x'):
    """
    Apply reflection to coordinates.
    
    Args:
        coords: (batch_size, n_points, 2) or (n_points, 2)
        axis: 'x' or 'y' axis to reflect across
        
    Returns:
        Reflected coordinates
    """
    reflected = coords.clone()
    if axis == 'x':
        reflected[..., 1] = -reflected[..., 1]
    elif axis == 'y':
        reflected[..., 0] = -reflected[..., 0]
    else:
        raise ValueError(f"Unknown axis: {axis}")
    
    return reflected


def apply_random_e2_transform(coords, include_reflection=False):
    """
    Apply random E(2) transformation (rotation, translation, optional reflection).
    
    Args:
        coords: Coordinates to transform
        include_reflection: Whether to include random reflection
        
    Returns:
        transformed_coords: Transformed coordinates
        params: Dictionary of transformation parameters
    """
    device = coords.device
    
    # Random rotation
    angle = torch.rand(1).item() * 2 * np.pi
    transformed = apply_rotation(coords, angle)
    
    # Random translation
    translation = torch.randn(2, device=device) * 0.5
    transformed = apply_translation(transformed, translation)
    
    # Optional reflection
    reflection_applied = False
    reflection_axis = None
    if include_reflection and torch.rand(1).item() > 0.5:
        reflection_axis = 'x' if torch.rand(1).item() > 0.5 else 'y'
        transformed = apply_reflection(transformed, reflection_axis)
        reflection_applied = True
    
    params = {
        'angle': angle,
        'translation': translation,
        'reflection': reflection_applied,
        'reflection_axis': reflection_axis
    }
    
    return transformed, params


def test_model_equivariance(model, coords, adj_matrix, timesteps, 
                           num_tests=5, tolerance=1e-5):
    """
    Test if a model is E(2) equivariant.
    
    Args:
        model: The EGNN model to test
        coords: Input coordinates
        adj_matrix: Input adjacency matrix
        timesteps: Time values
        num_tests: Number of random transformations to test
        tolerance: Tolerance for equivariance check
        
    Returns:
        Dictionary with equivariance test results
    """
    model.eval()
    results = []
    
    with torch.no_grad():
        # Original output
        orig_output = model(coords, adj_matrix, timesteps)
        
        for _ in range(num_tests):
            # Apply random transformation
            transformed_coords, params = apply_random_e2_transform(coords)
            
            # Get output for transformed input
            trans_output = model(transformed_coords, adj_matrix, timesteps)
            
            # Check if outputs are invariant (for edge predictions)
            # Edge predictions should be the same regardless of transformation
            diff = torch.abs(orig_output - trans_output).mean()
            
            results.append({
                'transformation': params,
                'output_difference': diff.item(),
                'equivariant': diff.item() < tolerance
            })
    
    return {
        'all_equivariant': all(r['equivariant'] for r in results),
        'mean_difference': np.mean([r['output_difference'] for r in results]),
        'max_difference': np.max([r['output_difference'] for r in results]),
        'detailed_results': results
    }


def normalize_coordinates(coords):
    """
    Normalize coordinates to unit square [0, 1]^2.
    
    Args:
        coords: (batch_size, n_points, 2) or (n_points, 2)
        
    Returns:
        Normalized coordinates
    """
    # Get min and max for each dimension
    if coords.dim() == 3:
        mins = coords.min(dim=1, keepdim=True)[0]
        maxs = coords.max(dim=1, keepdim=True)[0]
    else:
        mins = coords.min(dim=0, keepdim=True)[0]
        maxs = coords.max(dim=0, keepdim=True)[0]
    
    # Normalize to [0, 1]
    normalized = (coords - mins) / (maxs - mins + 1e-8)
    
    return normalized


def center_coordinates(coords):
    """
    Center coordinates around origin.
    
    Args:
        coords: (batch_size, n_points, 2) or (n_points, 2)
        
    Returns:
        Centered coordinates
    """
    if coords.dim() == 3:
        center = coords.mean(dim=1, keepdim=True)
    else:
        center = coords.mean(dim=0, keepdim=True)
    
    return coords - center


class E2Augmentation(nn.Module):
    """
    Data augmentation with E(2) transformations.
    Useful for training non-equivariant baselines.
    """
    
    def __init__(self, rotation=True, translation=True, 
                 reflection=False, normalize=True):
        super().__init__()
        self.rotation = rotation
        self.translation = translation
        self.reflection = reflection
        self.normalize = normalize
    
    def forward(self, coords):
        """Apply random E(2) augmentation."""
        augmented = coords.clone()
        
        # Apply transformations
        if self.rotation:
            angle = torch.rand(1).item() * 2 * np.pi
            augmented = apply_rotation(augmented, angle)
        
        if self.translation:
            translation = torch.randn(2, device=coords.device) * 0.1
            augmented = apply_translation(augmented, translation)
        
        if self.reflection and torch.rand(1).item() > 0.5:
            axis = 'x' if torch.rand(1).item() > 0.5 else 'y'
            augmented = apply_reflection(augmented, axis)
        
        # Optionally normalize
        if self.normalize:
            augmented = normalize_coordinates(augmented)
        
        return augmented


def check_invariant_distances(coords1, coords2, tolerance=1e-6):
    """
    Check if pairwise distances are preserved (invariant under E(2)).
    
    Args:
        coords1: First set of coordinates
        coords2: Second set of coordinates
        tolerance: Tolerance for distance comparison
        
    Returns:
        is_invariant: Boolean indicating if distances are preserved
        max_diff: Maximum difference in distances
    """
    # Compute pairwise distances for both sets
    if coords1.dim() == 3:
        # Batched
        dists1 = torch.cdist(coords1, coords1)
        dists2 = torch.cdist(coords2, coords2)
    else:
        # Single sample
        dists1 = torch.cdist(coords1.unsqueeze(0), coords1.unsqueeze(0))[0]
        dists2 = torch.cdist(coords2.unsqueeze(0), coords2.unsqueeze(0))[0]
    
    # Check if distances are preserved
    diff = torch.abs(dists1 - dists2)
    max_diff = diff.max().item()
    is_invariant = max_diff < tolerance
    
    return is_invariant, max_diff


def compute_edge_features_e2_invariant(coords):
    """
    Compute E(2)-invariant edge features from coordinates.
    
    Args:
        coords: (batch_size, n_nodes, 2) or (n_nodes, 2)
        
    Returns:
        edge_features: Dictionary containing invariant features
            - distances: Pairwise distances
            - angles: Angles between node triplets (optional)
    """
    if coords.dim() == 2:
        coords = coords.unsqueeze(0)
    
    batch_size, n_nodes, _ = coords.shape
    
    # Compute pairwise distances (invariant)
    distances = torch.cdist(coords, coords)  # (batch_size, n_nodes, n_nodes)
    
    # Could also compute angles between triplets of nodes (invariant)
    # but this is more complex and may not be needed
    
    return {
        'distances': distances
    }