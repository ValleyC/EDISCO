"""Continuous score network wrapper for EDISCO."""

import torch
import torch.nn as nn
from models.egnn_encoder import EGNNEncoder, EGNNEncoderDense
from models.gnn_encoder import GNNEncoder


class ContinuousScoreNetwork(nn.Module):
    """Score network for continuous-time diffusion."""

    def __init__(self,
                 encoder_type='egnn',
                 n_layers=12,
                 hidden_dim=128,
                 node_dim=64,
                 edge_dim=64,
                 time_dim=128,
                 coord_dim=2,
                 num_classes=2,
                 sparse=False,
                 dense_only=False,
                 **kwargs):
        super().__init__()

        self.encoder_type = encoder_type
        self.num_classes = num_classes
        self.sparse = sparse and not dense_only
        self.dense_only = dense_only or not sparse

        # Create encoder based on type and mode
        if encoder_type == 'egnn':
            if self.dense_only:
                self.encoder = EGNNEncoderDense(
                    n_layers=n_layers,
                    hidden_dim=hidden_dim,
                    node_dim=node_dim,
                    edge_dim=edge_dim,
                    time_dim=time_dim,
                    coord_dim=coord_dim,
                    out_channels=num_classes,
                    **kwargs
                )
            else:
                self.encoder = EGNNEncoder(
                    n_layers=n_layers,
                    hidden_dim=hidden_dim,
                    node_dim=node_dim,
                    edge_dim=edge_dim,
                    time_dim=time_dim,
                    coord_dim=coord_dim,
                    out_channels=num_classes,
                    sparse=self.sparse,
                    dense_only=False,
                    **kwargs
                )
        else:
            self.encoder = GNNEncoder(
                n_layers=n_layers,
                hidden_dim=hidden_dim,
                out_channels=num_classes,
                aggregation=kwargs.get('aggregation', 'sum'),
                sparse=self.sparse,
                use_activation_checkpoint=kwargs.get('use_activation_checkpoint', False),
                node_feature_only=False
            )

    def forward(self, coords, adj_matrix, timesteps, edge_index=None):
        """Forward pass through encoder."""
        return self.encoder(coords, adj_matrix, timesteps, edge_index)

    def get_score(self, x, t, coords, edge_index=None):
        """Compute score function for diffusion."""
        if self.dense_only:
            return self._get_score_dense(x, t, coords)
        else:
            return self._get_score_flexible(x, t, coords, edge_index)

    def _get_score_dense(self, x, t, coords):
        """Dense score computation."""
        if not isinstance(t, torch.Tensor):
            t = torch.tensor([t], device=x.device)

        logits = self.encoder(coords, x, t, None)
        probs = torch.softmax(logits, dim=-1)
        score = probs[..., 1] - x
        return score

    def _get_score_flexible(self, x, t, coords, edge_index=None):
        """Flexible score computation."""
        if not isinstance(t, torch.Tensor):
            t = torch.tensor([t], device=x.device)

        logits = self.forward(coords, x, t, edge_index)
        probs = torch.softmax(logits, dim=-1)
        score = probs[..., 1] - x
        return score


class ScoreWrapperDense(nn.Module):
    """Wrapper for dense score network."""

    def __init__(self, score_network, coords):
        super().__init__()
        self.score_network = score_network
        self.coords = coords

    def forward(self, x, t):
        """Forward pass for dense graphs."""
        if not isinstance(t, torch.Tensor):
            t = torch.tensor([t], device=x.device, dtype=torch.float32)
        elif t.dim() == 0:
            t = t.unsqueeze(0)

        batch_size = x.shape[0]
        if t.shape[0] == 1 and batch_size > 1:
            t = t.expand(batch_size)

        return self.score_network(self.coords, x, t, None)


class ScoreWrapperSparse(nn.Module):
    """
    Optimized wrapper for sparse graphs.
    """
    
    def __init__(self, score_network, coords, edge_index):
        super().__init__()
        self.score_network = score_network
        self.coords = coords
        self.edge_index = edge_index
    
    def forward(self, x, t):
        """Direct forward for sparse graphs."""
        # Ensure tensor format
        if not isinstance(t, torch.Tensor):
            t = torch.tensor([t], device=x.device, dtype=torch.float32)
        elif t.dim() == 0:
            t = t.unsqueeze(0)
        
        # Direct forward
        return self.score_network(self.coords, x, t, self.edge_index)


class ScoreWrapper(nn.Module):
    """
    Flexible wrapper with mode selection at initialization.
    """
    
    def __init__(self, score_network, coords, edge_index=None):
        super().__init__()
        self.score_network = score_network
        self.coords = coords
        self.edge_index = edge_index
        
        # Determine mode at initialization
        self.is_dense = edge_index is None
        self.is_sparse = edge_index is not None

    def forward(self, x, t):
        """Forward pass routing to appropriate implementation."""
        if self.is_dense:
            return self._forward_dense(x, t)
        else:
            return self._forward_sparse(x, t)

    def _forward_dense(self, x, t):
        """Dense forward pass."""
        # Direct tensor handling
        if not isinstance(t, torch.Tensor):
            t = torch.tensor([t], device=x.device, dtype=torch.float32)
        elif t.dim() == 0:
            t = t.unsqueeze(0)
        
        # Batch expansion if needed
        if x.dim() == 3 and t.shape[0] == 1:
            batch_size = x.shape[0]
            if batch_size > 1:
                t = t.expand(batch_size)
        
        return self.score_network(self.coords, x, t, None)
    
    def _forward_sparse(self, x, t):
        """Optimized sparse forward."""
        # Direct tensor handling
        if not isinstance(t, torch.Tensor):
            t = torch.tensor([t], device=x.device, dtype=torch.float32)
        elif t.dim() == 0:
            t = t.unsqueeze(0)

        # Ensure x is in the correct format for sparse mode (1D or 2D with shape (n_edges,) or (n_edges, 1))
        if x.dim() > 2:
            # If x is batched 3D, flatten to match edge_index
            n_edges = self.edge_index.shape[1]
            x = x.reshape(-1) if x.numel() == n_edges else x.reshape(n_edges, -1)

        return self.score_network(self.coords, x, t, self.edge_index)


def create_score_wrapper(score_network, coords, edge_index=None):
    """
    Factory function to create optimized wrapper based on mode.
    """
    if hasattr(score_network, 'dense_only') and score_network.dense_only:
        return ScoreWrapperDense(score_network, coords)
    elif hasattr(score_network, 'sparse') and score_network.sparse:
        return ScoreWrapperSparse(score_network, coords, edge_index)
    else:
        return ScoreWrapper(score_network, coords, edge_index)


class ContinuousScoreNetworkDense(nn.Module):
    """
    Pure dense-only score network for maximum performance.
    """
    
    def __init__(self, 
                 encoder_type='egnn',
                 n_layers=12,
                 hidden_dim=128,
                 node_dim=64,
                 edge_dim=64,
                 time_dim=128,
                 coord_dim=2,
                 num_classes=2,
                 **kwargs):
        super().__init__()
        
        self.num_classes = num_classes
        
        # Create dense-only encoder
        if encoder_type == 'egnn':
            self.encoder = EGNNEncoderDense(
                n_layers=n_layers,
                hidden_dim=hidden_dim,
                node_dim=node_dim,
                edge_dim=edge_dim,
                time_dim=time_dim,
                coord_dim=coord_dim,
                out_channels=num_classes,
                **kwargs
            )
        else:
            # Standard GNN in dense mode
            self.encoder = GNNEncoder(
                n_layers=n_layers,
                hidden_dim=hidden_dim,
                out_channels=num_classes,
                aggregation=kwargs.get('aggregation', 'sum'),
                sparse=False,
                use_activation_checkpoint=kwargs.get('use_activation_checkpoint', False),
                node_feature_only=False
            )
    
    def forward(self, coords, adj_matrix, timesteps, edge_index=None):
        """Direct forward - edge_index ignored."""
        return self.encoder(coords, adj_matrix, timesteps, None)
    
    def get_score(self, x, t, coords):
        """Direct score computation for dense."""
        if not isinstance(t, torch.Tensor):
            t = torch.tensor([t], device=x.device)
        
        logits = self.encoder(coords, x, t, None)
        probs = torch.softmax(logits, dim=-1)
        return probs[..., 1] - x