"""Score network for continuous-time categorical diffusion using EGNN"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .egnn_encoder import EGNNEncoder
from .nn import timestep_embedding, SiLU, zero_module


class ScoreNetwork(nn.Module):
    """
    Network that predicts score (or X_0) for continuous-time diffusion
    Uses E(n)-equivariant GNN architecture for dense adjacency matrices
    """
    
    def __init__(self, n_layers=8, hidden_dim=128, node_dim=64, coord_dim=2, 
                 num_classes=2, dropout=0.0, update_coords=True, 
                 use_checkpoint=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.node_dim = node_dim
        self.n_layers = n_layers
        self.num_classes = num_classes
        
        # Initial embeddings
        self.coord_embed = nn.Sequential(
            nn.Linear(coord_dim, node_dim),
            SiLU(),
            nn.LayerNorm(node_dim)
        )
        
        # Edge embedding (for noisy adjacency matrix)
        self.edge_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            SiLU(),
            nn.LayerNorm(hidden_dim)
        )
        
        # Time embedding network
        self.time_embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            SiLU(),
        )
        
        # EGNN encoder
        self.egnn = EGNNEncoder(
            n_layers=n_layers,
            node_dim=node_dim,
            edge_dim=hidden_dim,
            hidden_dim=hidden_dim,
            coord_dim=coord_dim,
            dropout=dropout,
            update_coords=update_coords,
            use_checkpoint=use_checkpoint
        )
        
        # Time modulation layers for each EGNN layer
        self.time_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                SiLU(),
                nn.Linear(hidden_dim, hidden_dim)
            ) for _ in range(n_layers)
        ])
        
        # Output head for X_0 prediction
        self.out = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            SiLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            SiLU(),
            zero_module(nn.Linear(hidden_dim // 2, num_classes))
        )
        
        # Optional: separate heads for different time ranges
        self.use_time_dependent_head = False
        if self.use_time_dependent_head:
            self.early_time_head = zero_module(nn.Linear(hidden_dim, num_classes))
            self.late_time_head = zero_module(nn.Linear(hidden_dim, num_classes))
    
    def forward(self, coords, adj_matrix, timesteps):
        """
        Predict X_0 given X_t and t for dense adjacency matrices
        
        Args:
            coords: (batch_size, n_nodes, 2) - TSP coordinates
            adj_matrix: (batch_size, n_nodes, n_nodes) - noisy adjacency matrix
            timesteps: (batch_size,) - continuous time values in [0, 1]
        
        Returns:
            logits: (batch_size, n_nodes, n_nodes, num_classes) - predicted X_0 logits
        """
        batch_size, n_nodes, coord_dim = coords.shape
        
        # Initialize node features from coordinates
        h = self.coord_embed(coords)  # (batch_size, n_nodes, node_dim)
        x = coords.clone()  # Keep original coordinates for equivariance
        
        # Embed noisy adjacency matrix as edge features
        e = self.edge_embed(adj_matrix.unsqueeze(-1))  # (batch_size, n_nodes, n_nodes, hidden_dim)
        
        # Time embedding
        t_emb = self.time_embed(timestep_embedding(timesteps, self.hidden_dim))
        
        # Apply time modulation at multiple scales
        time_modulations = []
        for time_layer in self.time_layers:
            time_mod = time_layer(t_emb)  # (batch_size, hidden_dim)
            time_modulations.append(time_mod)
        
        # Multi-scale time modulation of edge features
        for i, time_mod in enumerate(time_modulations):
            # Different modulation strategies at different depths
            time_mod = time_mod.view(batch_size, 1, 1, self.hidden_dim)
            
            if i < len(time_modulations) // 2:
                # Early layers: additive modulation
                e = e + 0.1 * time_mod
            else:
                # Later layers: multiplicative modulation
                e = e * (1 + 0.1 * time_mod)
        
        # Process through EGNN encoder
        h, x, e = self.egnn(h, x, e)
        
        # Output predictions for X_0
        if self.use_time_dependent_head:
            # Use different heads for different time ranges
            t_normalized = timesteps.view(batch_size, 1, 1, 1)
            early_weight = torch.sigmoid(10 * (0.5 - t_normalized))  # Active for t < 0.5
            late_weight = 1 - early_weight
            
            early_logits = self.early_time_head(e)
            late_logits = self.late_time_head(e)
            logits = early_weight * early_logits + late_weight * late_logits
        else:
            # Single output head
            logits = self.out(e)  # (batch_size, n_nodes, n_nodes, num_classes)
        
        return logits
    
    def get_score(self, coords, adj_matrix, timesteps):
        """
        Get the score function ∇_x log p(x|t) for score matching
        
        Args:
            coords: (batch_size, n_nodes, 2)
            adj_matrix: (batch_size, n_nodes, n_nodes)
            timesteps: (batch_size,)
        
        Returns:
            score: (batch_size, n_nodes, n_nodes, num_classes)
        """
        logits = self.forward(coords, adj_matrix, timesteps)
        
        # Convert logits to score
        # Score = ∇_x log p(x|t) ≈ (x_0_pred - x_t) / (1 - t)
        x0_probs = F.softmax(logits, dim=-1)
        xt_onehot = F.one_hot(adj_matrix.long(), num_classes=self.num_classes).float()
        
        # Avoid division by zero near t=0
        t_safe = timesteps.view(-1, 1, 1, 1).clamp(min=0.01)
        score = (x0_probs - xt_onehot) / t_safe
        
        return score