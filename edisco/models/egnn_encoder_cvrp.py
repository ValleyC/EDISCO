"""
EGNN Encoder extension for CVRP
Adds CVRP-specific forward method to handle coordinates and invariant features separately
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.egnn_encoder import EGNNEncoder
from models.nn import timestep_embedding  # Use the shared timestep_embedding function


class EGNNEncoderCVRP(EGNNEncoder):
    """
    Extended EGNN encoder that properly handles CVRP's separated features
    """
    
    def __init__(self, n_layers=12, hidden_dim=128, node_dim=64, edge_dim=64,
                 time_dim=128, coord_dim=2, out_channels=2, sparse=False,
                 invariant_dim=2, use_activation_checkpoint=False,
                 coord_update_alpha=0.1, weight_temp=10.0, **kwargs):
        """
        Initialize CVRP-specific EGNN encoder
        
        Args:
            invariant_dim: Dimension of invariant features (demands + is_depot)
            All other args inherited from EGNNEncoder
        """
        # Store invariant dimension before calling parent init
        self.invariant_dim = invariant_dim
        
        # Call parent constructor with all parameters including time_dim
        super().__init__(
            n_layers=n_layers,
            hidden_dim=hidden_dim,
            node_dim=node_dim,
            edge_dim=edge_dim,
            time_dim=time_dim,
            coord_dim=coord_dim,
            out_channels=out_channels,
            sparse=sparse,
            use_activation_checkpoint=use_activation_checkpoint,
            coord_update_alpha=coord_update_alpha,
            weight_temp=weight_temp,
            **kwargs
        )
        
        # Embedding for CVRP invariant features (demands, is_depot)
        self.invariant_embed = nn.Sequential(
            nn.Linear(invariant_dim, node_dim),
            nn.LayerNorm(node_dim),
            nn.SiLU(),
            nn.Linear(node_dim, node_dim)
        )
    
    def forward_cvrp(self, coords, invariant_features, adj_matrix, timesteps):
        """
        Forward pass for CVRP with proper feature separation
        
        Args:
            coords: (batch_size, n_nodes, 2) - coordinates (equivariant)
            invariant_features: (batch_size, n_nodes, feat_dim) - demands + is_depot (invariant)
            adj_matrix: (batch_size, n_nodes, n_nodes) - noisy adjacency
            timesteps: (batch_size,) - diffusion time
        
        Returns:
            logits: (batch_size, n_nodes, n_nodes, num_classes)
        """
        batch_size, n_nodes, _ = coords.shape

        # Embed invariant node features
        h = self.invariant_embed(invariant_features)  # (batch_size, n_nodes, node_dim)
        
        # Keep coordinates separate for equivariant processing
        x = coords.clone()
        
        # Prepare adjacency matrix for embedding
        if adj_matrix.dtype != torch.float32:
            adj_matrix = adj_matrix.float()
        
        # Ensure correct shape for edge embedding
        if adj_matrix.dim() == 3:
            adj_input = adj_matrix.unsqueeze(-1)  # (batch, n, n, 1)
        else:
            adj_input = adj_matrix
        
        # Embed adjacency matrix as edge features
        e = self.edge_embed(adj_input)  # (batch_size, n_nodes, n_nodes, edge_dim)
        
        # Time embedding using time_dim
        t_emb = self.time_embed(timestep_embedding(timesteps, self.time_dim))
        
        # Apply EGNN layers
        for i, (layer, time_layer) in enumerate(zip(self.layers, self.time_layers)):
            # Modulate edge features with time
            time_mod = time_layer(t_emb)  # (batch_size, edge_dim)
            time_mod = time_mod.view(batch_size, 1, 1, self.edge_dim)
            
            # Multiplicative time modulation
            e_with_time = e * (1 + time_mod)
            
            # Apply EGNN layer - maintains separation of coords and invariants
            h, x, e = layer(h, x, e_with_time, sparse=False)
        
        # Output predictions using edge features (invariant)
        logits = self.out(e)  # (batch_size, n_nodes, n_nodes, num_classes)
        
        return logits
    
    def forward(self, coords, adj_matrix, timesteps, edge_index=None, invariant_features=None):
        """
        Override parent forward to handle both standard and CVRP calls
        """
        if invariant_features is not None:
            # CVRP mode with separate invariant features
            return self.forward_cvrp(coords, invariant_features, adj_matrix, timesteps)
        else:
            # Standard TSP mode
            return super().forward(coords, adj_matrix, timesteps, edge_index)


def create_egnn_for_cvrp(n_layers, hidden_dim, node_dim, edge_dim, time_dim=128,
                         coord_dim=2, out_channels=2, invariant_dim=2, 
                         sparse=False, **kwargs):
    """
    Factory function to create EGNN encoder for CVRP
    
    Args:
        time_dim: Time embedding dimension (independent from hidden_dim)
        invariant_dim: Dimension of invariant features for CVRP
    """
    model = EGNNEncoderCVRP(
        n_layers=n_layers,
        hidden_dim=hidden_dim,
        node_dim=node_dim,
        edge_dim=edge_dim,
        time_dim=time_dim,
        coord_dim=coord_dim,
        out_channels=out_channels,
        invariant_dim=invariant_dim,
        sparse=sparse,
        **kwargs
    )
    return model