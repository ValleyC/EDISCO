"""E(n) Equivariant GNN Encoder for dense adjacency matrices"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as activation_checkpoint
from .nn import SiLU, timestep_embedding


class EGNNLayerDense(nn.Module):
    """
    E(n) Equivariant GNN Layer for dense adjacency matrices
    Processes all edges implicitly through matrix operations
    """
    
    def __init__(self, node_dim, edge_dim, hidden_dim, coord_dim=2, 
                 dropout=0.0, update_coords=True):
        super().__init__()
        self.coord_dim = coord_dim
        self.hidden_dim = hidden_dim
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.dropout = dropout
        self.update_coords = update_coords
        
        # Message network
        self.message_mlp = nn.Sequential(
            nn.Linear(node_dim * 2 + edge_dim + 1, hidden_dim),
            SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Coordinate network (equivariant)
        if self.update_coords:
            self.coord_mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                SiLU(),
                nn.Linear(hidden_dim, 1, bias=False)  # No bias for equivariance
            )
            # Initialize coordinate network to small weights
            nn.init.xavier_uniform_(self.coord_mlp[-1].weight, gain=0.01)
        
        # Node update network
        self.node_mlp = nn.Sequential(
            nn.Linear(node_dim + hidden_dim, hidden_dim),
            SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, node_dim)
        )
        
        # Edge update network
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_dim + hidden_dim, hidden_dim),
            SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, edge_dim)
        )
        
        # Normalization layers
        self.node_norm = nn.LayerNorm(node_dim)
        self.edge_norm = nn.LayerNorm(edge_dim)
    
    def forward(self, h, x, e, mask=None):
        """
        h: node features (batch_size, n_nodes, node_dim)
        x: coordinates (batch_size, n_nodes, coord_dim)
        e: edge features as adjacency matrix (batch_size, n_nodes, n_nodes, edge_dim)
        mask: optional edge mask (batch_size, n_nodes, n_nodes)
        """
        batch_size, n_nodes, _ = h.shape
        
        # Compute pairwise coordinate differences and distances
        x_i = x.unsqueeze(2).expand(-1, -1, n_nodes, -1)  # (batch, n, n, coord_dim)
        x_j = x.unsqueeze(1).expand(-1, n_nodes, -1, -1)  # (batch, n, n, coord_dim)
        x_diff = x_j - x_i  # (batch, n, n, coord_dim)
        distances = torch.norm(x_diff, dim=-1, keepdim=True)  # (batch, n, n, 1)
        
        # Apply mask if provided
        if mask is not None:
            mask = mask.unsqueeze(-1)  # (batch, n, n, 1)
            distances = distances * mask
        
        # Prepare node features for all pairs
        h_i = h.unsqueeze(2).expand(-1, -1, n_nodes, -1)  # (batch, n, n, node_dim)
        h_j = h.unsqueeze(1).expand(-1, n_nodes, -1, -1)  # (batch, n, n, node_dim)
        
        # Compute messages for all pairs
        msg_input = torch.cat([h_i, h_j, e, distances], dim=-1)  # (batch, n, n, input_dim)
        messages = self.message_mlp(msg_input)  # (batch, n, n, hidden_dim)
        
        # Apply mask to messages
        if mask is not None:
            messages = messages * mask
        
        # Update coordinates (equivariant)
        if self.update_coords:
            coord_weights = self.coord_mlp(messages)  # (batch, n, n, 1)
            coord_weights = torch.tanh(coord_weights / 10.0)  # Stability
            
            # Scale coordinate differences by learned weights
            x_update = coord_weights * x_diff / (distances + 1e-8)  # (batch, n, n, coord_dim)
            
            # Apply mask to coordinate updates
            if mask is not None:
                x_update = x_update * mask
            
            # Aggregate coordinate updates (sum over j dimension)
            x_agg = x_update.sum(dim=2)  # (batch, n, coord_dim)
            x_new = x + 0.1 * x_agg  # Small step for stability
        else:
            x_new = x
        
        # Aggregate messages for node updates (sum over j dimension)
        h_agg = messages.sum(dim=2)  # (batch, n, hidden_dim)
        
        # Update nodes with residual connection
        h_input = torch.cat([h, h_agg], dim=-1)
        h_update = self.node_mlp(h_input)
        h_new = self.node_norm(h + h_update)
        
        # Update edges with residual connection
        e_input = torch.cat([e, messages], dim=-1)
        e_update = self.edge_mlp(e_input)
        e_new = self.edge_norm(e + e_update)
        
        return h_new, x_new, e_new


class EGNNEncoder(nn.Module):
    """
    Stacked EGNN layers for processing dense graphs
    """
    
    def __init__(self, n_layers, node_dim, edge_dim, hidden_dim, coord_dim=2,
                 dropout=0.0, update_coords=True, use_checkpoint=False):
        super().__init__()
        self.n_layers = n_layers
        self.use_checkpoint = use_checkpoint
        
        # Stack of EGNN layers
        self.layers = nn.ModuleList([
            EGNNLayerDense(
                node_dim=node_dim,
                edge_dim=edge_dim if i == 0 else hidden_dim,
                hidden_dim=hidden_dim,
                coord_dim=coord_dim,
                dropout=dropout,
                update_coords=update_coords
            )
            for i in range(n_layers)
        ])
        
        # Layer normalization for stability
        self.h_norm = nn.LayerNorm(node_dim)
        self.e_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, h, x, e, mask=None):
        """
        Forward pass through all EGNN layers
        
        Args:
            h: node features (batch_size, n_nodes, node_dim)
            x: coordinates (batch_size, n_nodes, coord_dim)
            e: edge features (batch_size, n_nodes, n_nodes, edge_dim)
            mask: optional edge mask (batch_size, n_nodes, n_nodes)
        """
        # Normalize inputs
        h = self.h_norm(h)
        
        # Process through layers
        for i, layer in enumerate(self.layers):
            if self.use_checkpoint and self.training:
                # Use gradient checkpointing to save memory
                h, x, e = activation_checkpoint.checkpoint(
                    layer, h, x, e, mask
                )
            else:
                h, x, e = layer(h, x, e, mask)
        
        # Final normalization
        e = self.e_norm(e)
        
        return h, x, e