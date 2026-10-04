"""Capacity-conditioned E(2)-equivariant encoder for CVRP.

Capacity information enters the score network only through E(2)-invariant
scalar channels: capacity-normalized demand features at the node and edge
inputs, and per-layer FiLM modulation of the scalar message vector by a global
capacity embedding. Coordinates are processed by the unchanged equivariant
coordinate update, so the edge logits are E(2)-invariant for every fixed value
of the capacity parameters.
"""

import torch
import torch.nn as nn

from models.egnn_encoder import EGNNLayerDense, EGNNLayerSparse, build_output_head, build_time_layers
from models.nn import checkpoint, timestep_embedding


def build_invariant_capacity_features(demands, capacity, default_capacity):
    """Construct E(2)-invariant capacity-aware features.

    Args:
        demands: (batch_size, n_nodes) per-node demand. The depot has demand 0.
        capacity: (batch_size,) scalar capacity Q for each instance.
        default_capacity: float, the reference training capacity Q_default.

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


class EGNNEncoderCVRP(nn.Module):
    """EGNN score network for CVRP that preserves E(2)-equivariance.

    With `capacity_conditioning=True` the encoder injects three groups of
    E(2)-invariant capacity-aware features:
        (1) per-node `d_i / Q` appended to the invariant node features,
        (2) per-edge `(d_i + d_j) / Q` and `|d_i - d_j| / Q` appended to the
            edge-feature embedding input,
        (3) per-layer FiLM modulation of the message vector, conditioned on a
            global scalar embedding `z_c = MLP([log(Q/Q_default), sum_i d_i / Q])`.

    With `capacity_conditioning=False` the same network reads only the noisy
    edge state and the invariants `[demand, is_depot]` (the unconditioned
    variants of the capacity-shift study).

    Dense and sparse modes share parameter names and shapes. In sparse mode
    message passing is restricted to an explicit edge list (the k-nearest-
    neighbour graph used above N = 100).
    """

    def __init__(self, n_layers=12, hidden_dim=256, node_dim=64, edge_dim=64,
                 time_dim=128, coord_dim=2, out_channels=2,
                 invariant_dim=2, default_capacity=1.0,
                 z_hidden_dim=64, capacity_conditioning=True,
                 use_activation_checkpoint=False, sparse=False,
                 coord_update_alpha=0.1, weight_temp=10.0):
        super().__init__()
        self.sparse = sparse
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.time_dim = time_dim
        self.coord_dim = coord_dim
        self.out_channels = out_channels
        self.invariant_dim = invariant_dim
        self.default_capacity = float(default_capacity)
        self.capacity_conditioning = capacity_conditioning
        self.use_activation_checkpoint = use_activation_checkpoint

        # Node feature embedding: CVRP invariants (demand, is_depot) plus the
        # capacity-normalized demand `d_i / Q`.
        node_input_dim = invariant_dim + (1 if capacity_conditioning else 0)
        self.invariant_embed = nn.Sequential(
            nn.Linear(node_input_dim, node_dim),
            nn.LayerNorm(node_dim),
            nn.SiLU(),
            nn.Linear(node_dim, node_dim),
        )

        # Edge feature embedding: noisy adjacency entry plus two
        # capacity-normalized pair statistics.
        edge_input_dim = 1 + (2 if capacity_conditioning else 0)
        self.edge_embed = nn.Linear(edge_input_dim, edge_dim)

        # Time embedding mirrors the TSP encoder.
        self.time_embed = nn.Sequential(
            nn.Linear(time_dim, time_dim * 2),
            nn.SiLU(),
            nn.Linear(time_dim * 2, time_dim),
            nn.SiLU(),
        )
        self.time_layers = build_time_layers(n_layers, time_dim, edge_dim)

        # Capacity-conditioned FiLM head producing per-layer (gamma, beta) for
        # the message vector.
        if capacity_conditioning:
            self.film_head = FiLMHead(
                z_dim=2, hidden_dim=z_hidden_dim,
                n_layers=n_layers, channel_dim=hidden_dim,
            )

        layer_cls = EGNNLayerSparse if sparse else EGNNLayerDense
        self.layers = nn.ModuleList([
            layer_cls(node_dim, edge_dim, hidden_dim, coord_dim,
                      coord_update_alpha, weight_temp)
            for _ in range(n_layers)
        ])

        # Output head matches the TSP encoder.
        self.out = build_output_head(edge_dim, hidden_dim, out_channels)

    def forward(self, coords, demands, capacity, invariant_features, adj_matrix,
                timesteps, edge_index=None, node_batch=None):
        """Forward pass for CVRP.

        Dense mode:
            coords: (batch_size, n_nodes, 2) coordinates (equivariant).
            demands: (batch_size, n_nodes) per-node demands. Depot has demand 0.
            capacity: (batch_size,) per-instance capacity Q.
            invariant_features: (batch_size, n_nodes, invariant_dim) CVRP
                invariants `[demand, is_depot]`.
            adj_matrix: (batch_size, n_nodes, n_nodes) noisy edge state.
            timesteps: (batch_size,) diffusion time.
            Returns logits (batch_size, n_nodes, n_nodes, out_channels).

        Sparse mode:
            coords (N, 2), demands (N,), invariant_features (N, invariant_dim)
            for the nodes of all graphs in the batch, adj_matrix (E,),
            edge_index (2, E), node_batch (N,) graph index of every node, and
            capacity / timesteps with one entry per graph.
            Returns logits (E, out_channels).
        """
        if self.sparse:
            return self._forward_sparse(coords, demands, capacity, invariant_features,
                                        adj_matrix, timesteps, edge_index, node_batch)
        return self._forward_dense(coords, demands, capacity, invariant_features,
                                   adj_matrix, timesteps)

    def _forward_dense(self, coords, demands, capacity, invariant_features, adj_matrix,
                       timesteps):
        batch_size, n_nodes, _ = coords.shape
        dtype = self.edge_embed.weight.dtype
        adj_input = adj_matrix.reshape(batch_size, n_nodes, n_nodes, 1).to(dtype)

        if self.capacity_conditioning:
            d_over_Q, edge_capacity_feats, z_input = build_invariant_capacity_features(
                demands.to(dtype), capacity.reshape(-1).to(dtype), self.default_capacity,
            )
            h = self.invariant_embed(torch.cat([invariant_features.to(dtype), d_over_Q], dim=-1))
            e = self.edge_embed(torch.cat([adj_input, edge_capacity_feats], dim=-1))
            gammas, betas = self.film_head(z_input)
        else:
            h = self.invariant_embed(invariant_features.to(dtype))
            e = self.edge_embed(adj_input)
            gammas = betas = [None] * self.n_layers

        x = coords

        timesteps = timesteps.reshape(-1)
        if timesteps.shape[0] == 1 and batch_size > 1:
            timesteps = timesteps.expand(batch_size)
        t_emb = self.time_embed(timestep_embedding(timesteps, self.time_dim).to(dtype))

        use_ckpt = self.use_activation_checkpoint and self.training
        for layer, time_layer, gamma, beta in zip(
            self.layers, self.time_layers, gammas, betas,
        ):
            time_scale = 1 + time_layer(t_emb).view(batch_size, 1, 1, -1)
            if gamma is None:
                func = lambda h, x, e, ts, layer=layer: layer(h, x, e, ts)
                inputs = (h, x, e, time_scale)
            else:
                func = lambda h, x, e, ts, g, b, layer=layer: layer(h, x, e, ts, g, b)
                inputs = (h, x, e, time_scale,
                          gamma.view(batch_size, 1, 1, -1), beta.view(batch_size, 1, 1, -1))
            h, x, e = checkpoint(func, inputs, tuple(layer.parameters()), use_ckpt)

        return checkpoint(self.out, (e,), tuple(self.out.parameters()), use_ckpt)

    def _forward_sparse(self, coords, demands, capacity, invariant_features, adj_matrix,
                        timesteps, edge_index, node_batch):
        if edge_index is None:
            raise ValueError('edge_index is required in sparse mode')
        dtype = self.edge_embed.weight.dtype
        coords = coords.reshape(-1, coords.shape[-1])
        n_nodes = coords.shape[0]
        if node_batch is None:
            node_batch = torch.zeros(n_nodes, dtype=torch.long, device=coords.device)
        capacity = capacity.reshape(-1).to(dtype).clamp(min=1e-8)
        n_graphs = capacity.shape[0]
        row, col = edge_index[0], edge_index[1]
        edge_batch = node_batch[row]
        demands = demands.reshape(-1).to(dtype)
        invariant_features = invariant_features.reshape(n_nodes, -1).to(dtype)
        adj_input = adj_matrix.reshape(-1, 1).to(dtype)
        if adj_input.shape[0] != edge_index.shape[1]:
            raise ValueError('one noisy edge state per edge is required')

        if self.capacity_conditioning:
            # Same invariant capacity features as the dense model, per edge.
            q_edge = capacity[edge_batch]
            d_over_Q = (demands / capacity[node_batch]).unsqueeze(-1)
            pair_sum = ((demands[row] + demands[col]) / q_edge).unsqueeze(-1)
            pair_abs = ((demands[row] - demands[col]).abs() / q_edge).unsqueeze(-1)
            total = torch.zeros(n_graphs, dtype=dtype, device=coords.device)
            total.index_add_(0, node_batch, demands)
            z_input = torch.stack([torch.log(capacity / self.default_capacity), total / capacity], dim=-1)
            h = self.invariant_embed(torch.cat([invariant_features, d_over_Q], dim=-1))
            e = self.edge_embed(torch.cat([adj_input, pair_sum, pair_abs], dim=-1))
            gammas, betas = self.film_head(z_input)
        else:
            h = self.invariant_embed(invariant_features)
            e = self.edge_embed(adj_input)
            gammas = betas = [None] * self.n_layers

        x = coords

        timesteps = timesteps.reshape(-1)
        if timesteps.shape[0] == 1 and n_graphs > 1:
            timesteps = timesteps.expand(n_graphs)
        t_emb = self.time_embed(timestep_embedding(timesteps, self.time_dim).to(dtype))

        # Per-graph time scales and FiLM parameters are expanded to edges
        # inside the (optionally checkpointed) layer call.
        use_ckpt = self.use_activation_checkpoint and self.training
        for layer, time_layer, gamma, beta in zip(
            self.layers, self.time_layers, gammas, betas,
        ):
            time_scale = 1 + time_layer(t_emb)
            if gamma is None:
                func = lambda h, x, e, ts, layer=layer: layer(h, x, e, edge_index, ts[edge_batch])
                inputs = (h, x, e, time_scale)
            else:
                func = lambda h, x, e, ts, g, b, layer=layer: layer(
                    h, x, e, edge_index, ts[edge_batch], g[edge_batch], b[edge_batch])
                inputs = (h, x, e, time_scale, gamma, beta)
            h, x, e = checkpoint(func, inputs, tuple(layer.parameters()), use_ckpt)

        return checkpoint(self.out, (e,), tuple(self.out.parameters()), use_ckpt)
