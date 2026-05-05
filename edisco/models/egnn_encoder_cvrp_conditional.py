"""Capacity-conditioned EGNN encoder for CVRP (T1.3-code).

Implements the symmetry-faithful conditional diffusion contribution: capacity-
normalized invariant features at the input plus per-layer FiLM modulation of
the scalar message vector. Conditioning enters only through scalar invariant
channels, so exact E(2)-equivariance is preserved for every fixed value of the
capacity parameters. See revision_plan.md (T1.3) and the corresponding section
in the manuscript.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.egnn_encoder import EGNNLayerDense
from models.nn import timestep_embedding


def build_invariant_capacity_features(demands, capacity, default_capacity):
    """Construct E(2)-invariant capacity-aware features.

    Args:
        demands: (batch_size, n_nodes) per-node demand. The depot has demand 0.
        capacity: (batch_size,) scalar capacity Q for each instance.
        default_capacity: float, the canonical training-time capacity Q0.

    Returns:
        node_feats: (batch_size, n_nodes, 1) per-node feature `d_i / Q`.
        edge_feats: (batch_size, n_nodes, n_nodes, 2) per-edge features
            `(d_i + d_j) / Q` and `|d_i - d_j| / Q`.
        z_input: (batch_size, 2) global capacity input
            `[log(Q / Q_default), sum_i d_i / Q]` for the capacity embedding
            MLP. Both entries are E(2)-invariant scalars.
    """
    batch_size, n_nodes = demands.shape
    Q = capacity.view(batch_size, 1)
    Q_safe = Q.clamp(min=1e-8)

    # Node feature: d_i / Q
    d_over_Q = (demands / Q_safe).unsqueeze(-1)

    # Edge features: (d_i + d_j) / Q and |d_i - d_j| / Q
    d_i = demands.unsqueeze(2)
    d_j = demands.unsqueeze(1)
    pair_sum = ((d_i + d_j) / Q_safe.unsqueeze(-1)).unsqueeze(-1)
    pair_abs = ((d_i - d_j).abs() / Q_safe.unsqueeze(-1)).unsqueeze(-1)
    edge_feats = torch.cat([pair_sum, pair_abs], dim=-1)

    # Global capacity input
    log_lambda = torch.log(Q_safe / float(default_capacity))
    total_over_Q = demands.sum(dim=-1, keepdim=True) / Q_safe
    z_input = torch.cat([log_lambda, total_over_Q], dim=-1)

    return d_over_Q, edge_feats, z_input


class FiLMHead(nn.Module):
    """Identity-initialized FiLM head producing per-layer (gamma, beta).

    Outputs scale and shift vectors with `gamma ~ 1` and `beta ~ 0` at the
    start of training so that the conditional model behaves like the
    unconditional model before any gradient steps.
    """

    def __init__(self, z_dim, hidden_dim, n_layers, channel_dim):
        super().__init__()
        self.n_layers = n_layers
        self.channel_dim = channel_dim
        self.shared = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        # One small head per layer producing 2 * channel_dim values
        # (gamma_residual and beta). gamma is reconstructed as 1 + gamma_residual
        # so that zero output gives identity modulation.
        self.heads = nn.ModuleList(
            [nn.Linear(hidden_dim, 2 * channel_dim) for _ in range(n_layers)]
        )
        for head in self.heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, z_input):
        """Compute (gamma, beta) for each layer.

        Args:
            z_input: (batch_size, z_dim)

        Returns:
            gammas: list of length n_layers, each (batch_size, channel_dim).
            betas: same shape, additive shift.
        """
        h = self.shared(z_input)
        gammas, betas = [], []
        for head in self.heads:
            out = head(h)
            gamma_res, beta = out.chunk(2, dim=-1)
            gammas.append(1.0 + gamma_res)
            betas.append(beta)
        return gammas, betas


class EGNNLayerDenseFiLM(EGNNLayerDense):
    """EGNN dense layer with optional FiLM modulation of the message vector.

    The FiLM modulation is applied to `messages` immediately after the message
    MLP. Coordinate, node, and edge updates downstream then operate on the
    modulated messages. The coordinate update remains a scalar-weighted
    combination of `(x_j - x_i)` directions, so equivariance is preserved as
    long as the FiLM scale and shift do not depend on coordinate orientation.
    """

    def forward(self, h, x, e, gamma=None, beta=None):
        batch_size, n_nodes, _ = h.shape

        x_i = x.unsqueeze(2)
        x_j = x.unsqueeze(1)
        x_diff = x_j - x_i
        distances = torch.norm(x_diff, dim=-1, keepdim=True)

        h_i = h.unsqueeze(2).expand(-1, -1, n_nodes, -1)
        h_j = h.unsqueeze(1).expand(-1, n_nodes, -1, -1)

        msg_input = torch.cat([h_i, h_j, e, distances], dim=-1)
        messages = self.message_mlp(msg_input)

        if gamma is not None and beta is not None:
            # Broadcast (batch_size, hidden_dim) to (batch_size, 1, 1, hidden_dim).
            gamma_b = gamma.view(batch_size, 1, 1, -1)
            beta_b = beta.view(batch_size, 1, 1, -1)
            messages = gamma_b * messages + beta_b

        coord_weights = self.coord_mlp(messages)
        coord_weights = torch.tanh(coord_weights / self.weight_temp)
        x_update = coord_weights * x_diff / (distances + 1e-8)
        x_agg = x_update.sum(dim=2)
        x_new = x + self.coord_update_alpha * x_agg

        h_agg = messages.sum(dim=2)
        h_new = self.node_norm(h + self.node_mlp(torch.cat([h, h_agg], dim=-1)))

        e_new = self.edge_norm(e + self.edge_mlp(torch.cat([e, messages], dim=-1)))

        return h_new, x_new, e_new


class EGNNEncoderCVRPConditional(nn.Module):
    """Capacity-conditioned EGNN encoder that preserves E(2)-equivariance.

    The encoder injects three groups of E(2)-invariant capacity-aware features:
        (1) per-node `d_i / Q` appended to invariant node features,
        (2) per-edge `(d_i + d_j) / Q` and `|d_i - d_j| / Q` appended to the
            edge-feature embedding input,
        (3) per-layer FiLM modulation of the message vector, conditioned on a
            global scalar embedding `z_c = MLP([log(Q/Q0), sum_i d_i / Q])`.

    Coordinates are processed through the standard equivariant coordinate
    update unchanged, so the only directional term remains `(x_j - x_i)`. All
    conditioning enters through scalar invariant channels.
    """

    def __init__(self, n_layers=12, hidden_dim=128, node_dim=64, edge_dim=64,
                 time_dim=128, coord_dim=2, out_channels=2,
                 invariant_dim=2, default_capacity=1.0,
                 z_hidden_dim=64,
                 coord_update_alpha=0.1, weight_temp=10.0, **kwargs):
        super().__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.time_dim = time_dim
        self.coord_dim = coord_dim
        self.out_channels = out_channels
        self.invariant_dim = invariant_dim
        self.default_capacity = float(default_capacity)

        # Node feature embedding: existing CVRP invariants (e.g. demand,
        # is_depot) plus capacity-normalized demand `d_i / Q`.
        node_input_dim = invariant_dim + 1
        self.invariant_embed = nn.Sequential(
            nn.Linear(node_input_dim, node_dim),
            nn.LayerNorm(node_dim),
            nn.SiLU(),
            nn.Linear(node_dim, node_dim),
        )

        # Edge feature embedding: noisy adjacency entry plus two
        # capacity-normalized pair statistics.
        edge_input_dim = 1 + 2
        self.edge_embed = nn.Linear(edge_input_dim, edge_dim)

        # Time embedding mirrors the unconditional encoder.
        self.time_embed = nn.Sequential(
            nn.Linear(time_dim, time_dim * 2),
            nn.SiLU(),
            nn.Linear(time_dim * 2, time_dim),
            nn.SiLU(),
        )
        self.time_layers = nn.ModuleList([
            nn.Sequential(nn.Linear(time_dim, edge_dim), nn.SiLU())
            for _ in range(n_layers)
        ])

        # Capacity-conditioned FiLM head producing per-layer (gamma, beta) for
        # the message vector. The hidden dimension here matches the message
        # network's output dimension, which is `hidden_dim` in the existing
        # layer.
        self.film_head = FiLMHead(
            z_dim=2, hidden_dim=z_hidden_dim,
            n_layers=n_layers, channel_dim=hidden_dim,
        )

        # FiLM-aware EGNN layers.
        self.layers = nn.ModuleList([
            EGNNLayerDenseFiLM(node_dim, edge_dim, hidden_dim, coord_dim,
                               coord_update_alpha, weight_temp)
            for _ in range(n_layers)
        ])

        # Output head matches the unconditional encoder.
        self.out = nn.Sequential(
            nn.LayerNorm(edge_dim),
            nn.Linear(edge_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, out_channels),
        )
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    def forward(self, coords, demands, capacity, invariant_features, adj_matrix,
                timesteps):
        """Conditional forward pass for CVRP.

        Args:
            coords: (batch_size, n_nodes, 2) coordinates (equivariant).
            demands: (batch_size, n_nodes) per-node demands. Depot has demand 0.
            capacity: (batch_size,) per-instance capacity Q.
            invariant_features: (batch_size, n_nodes, invariant_dim) existing
                CVRP invariants such as `[demand, is_depot]`.
            adj_matrix: (batch_size, n_nodes, n_nodes) noisy adjacency matrix.
            timesteps: (batch_size,) diffusion time.

        Returns:
            logits: (batch_size, n_nodes, n_nodes, out_channels).
        """
        batch_size, n_nodes, _ = coords.shape

        # Build capacity-normalized invariant features.
        d_over_Q, edge_capacity_feats, z_input = build_invariant_capacity_features(
            demands, capacity, self.default_capacity,
        )

        # Concatenate `d_i / Q` to the existing invariants and embed.
        node_inputs = torch.cat([invariant_features, d_over_Q], dim=-1)
        h = self.invariant_embed(node_inputs)

        x = coords.clone()

        # Edge-level inputs: noisy adjacency plus pair-demand stats.
        if adj_matrix.dtype != torch.float32:
            adj_matrix = adj_matrix.float()
        adj_input = adj_matrix.unsqueeze(-1) if adj_matrix.dim() == 3 else adj_matrix
        e_input = torch.cat([adj_input, edge_capacity_feats], dim=-1)
        e = self.edge_embed(e_input)

        # Time embedding shared across layers.
        t_emb = self.time_embed(timestep_embedding(timesteps, self.time_dim))

        # Per-layer FiLM parameters for the message vector.
        gammas, betas = self.film_head(z_input)

        for layer, time_layer, gamma, beta in zip(
            self.layers, self.time_layers, gammas, betas,
        ):
            time_mod = time_layer(t_emb).view(batch_size, 1, 1, -1)
            e_with_time = e * (1 + time_mod)
            h, x, e = layer(h, x, e_with_time, gamma=gamma, beta=beta)

        return self.out(e)


def create_conditional_egnn_for_cvrp(n_layers, hidden_dim, node_dim, edge_dim,
                                     time_dim=128, coord_dim=2, out_channels=2,
                                     invariant_dim=2, default_capacity=1.0,
                                     **kwargs):
    """Factory matching the signature of `create_egnn_for_cvrp`."""
    return EGNNEncoderCVRPConditional(
        n_layers=n_layers,
        hidden_dim=hidden_dim,
        node_dim=node_dim,
        edge_dim=edge_dim,
        time_dim=time_dim,
        coord_dim=coord_dim,
        out_channels=out_channels,
        invariant_dim=invariant_dim,
        default_capacity=default_capacity,
        **kwargs,
    )
