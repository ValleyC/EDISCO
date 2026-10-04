"""E(2)-equivariant graph neural network encoder of EDISCO.

Each layer applies the following updates:

    m_ij      = MLP_m([h_i, h_j, e_ij, ||x_i - x_j||])                      (message)
    x_i      <- x_i + alpha * sum_{j != i} w_ij (x_j - x_i) / (||x_j - x_i|| + eps)
    w_ij      = tanh(MLP_c(m_ij) / tau)                                     (coordinates)
    e_ij     <- LN((e_ij + MLP_e([e_ij, m_ij])) * (1 + MLP_t(t_emb)))       (edges)
    h_i      <- LN(h_i + MLP_h([h_i, sum_{j != i} m_ij]))                   (nodes)

For CVRP the scalar messages are optionally FiLM-modulated,
m_ij <- gamma(z_c) * m_ij + beta(z_c), before every downstream use.
"""

import torch
import torch.nn as nn
from models.nn import checkpoint, timestep_embedding


class _EGNNLayer(nn.Module):
    """Parameters shared by the dense and sparse layer implementations."""

    def __init__(self, node_dim, edge_dim, hidden_dim, coord_dim=2,
                 coord_update_alpha=0.1, weight_temp=10.0):
        super().__init__()
        self.coord_dim = coord_dim
        self.hidden_dim = hidden_dim
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.coord_update_alpha = coord_update_alpha
        self.weight_temp = weight_temp

        # Message network
        self.message_mlp = nn.Sequential(
            nn.Linear(node_dim * 2 + edge_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Coordinate network (final layer has no bias)
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1, bias=False)
        )

        # Node update network
        self.node_mlp = nn.Sequential(
            nn.Linear(node_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, node_dim)
        )

        # Edge update network
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, edge_dim)
        )

        self.node_norm = nn.LayerNorm(node_dim)
        self.edge_norm = nn.LayerNorm(edge_dim)

    def _update_edges(self, e, messages, time_scale):
        e_new = e + self.edge_mlp(torch.cat([e, messages], dim=-1))
        if time_scale is not None:
            e_new = e_new * time_scale
        return self.edge_norm(e_new)


class EGNNLayerDense(_EGNNLayer):
    """Dense E(2)-equivariant layer over all node pairs."""

    def forward(self, h, x, e, time_scale=None, gamma=None, beta=None):
        """
        Args:
            h: (B, n, node_dim) invariant node features
            x: (B, n, coord_dim) equivariant coordinates
            e: (B, n, n, edge_dim) invariant edge features
            time_scale: (B, 1, 1, edge_dim) multiplicative factor 1 + MLP_t(t_emb)
            gamma, beta: optional FiLM parameters broadcastable to the messages
        """
        n_nodes = h.shape[1]

        # x_diff[b, i, j] = x_j - x_i
        x_diff = x.unsqueeze(1) - x.unsqueeze(2)
        distances = torch.norm(x_diff, dim=-1, keepdim=True)

        h_i = h.unsqueeze(2).expand(-1, -1, n_nodes, -1)
        h_j = h.unsqueeze(1).expand(-1, n_nodes, -1, -1)

        messages = self.message_mlp(torch.cat([h_i, h_j, e, distances], dim=-1))
        if gamma is not None:
            messages = gamma * messages + beta

        # Coordinates: the j = i term vanishes because x_diff[i, i] = 0.
        coord_weights = torch.tanh(self.coord_mlp(messages) / self.weight_temp)
        x_agg = (coord_weights * x_diff / (distances + 1e-8)).sum(dim=2)
        x_new = x + self.coord_update_alpha * x_agg

        # Nodes: aggregate messages over j != i.
        h_agg = messages.sum(dim=2) - torch.diagonal(messages, dim1=1, dim2=2).transpose(1, 2)
        h_new = self.node_norm(h + self.node_mlp(torch.cat([h, h_agg], dim=-1)))

        e_new = self._update_edges(e, messages, time_scale)
        return h_new, x_new, e_new


class EGNNLayerSparse(_EGNNLayer):
    """Sparse E(2)-equivariant layer over an explicit edge list."""

    def forward(self, h, x, e, edge_index, time_scale=None, gamma=None, beta=None):
        """
        Args:
            h: (N, node_dim), x: (N, coord_dim), e: (E, edge_dim)
            edge_index: (2, E) with row = receiver i and col = sender j
            time_scale: (E, edge_dim) multiplicative factor 1 + MLP_t(t_emb)
            gamma, beta: optional per-edge FiLM parameters, (E, hidden_dim)
        """
        n_nodes = h.shape[0]
        row, col = edge_index[0], edge_index[1]

        x_diff = x[col] - x[row]
        distances = torch.norm(x_diff, dim=-1, keepdim=True)

        messages = self.message_mlp(torch.cat([h[row], h[col], e, distances], dim=-1))
        if gamma is not None:
            messages = gamma * messages + beta

        # Coordinates: self-loops contribute zero because x_diff = 0.
        coord_weights = torch.tanh(self.coord_mlp(messages) / self.weight_temp)
        x_update = coord_weights * x_diff / (distances + 1e-8)
        x_agg = torch.zeros(n_nodes, self.coord_dim, device=x.device, dtype=x_update.dtype)
        x_agg.index_add_(0, row, x_update)
        x_new = x + self.coord_update_alpha * x_agg

        # Nodes: aggregate messages over j != i.
        not_self = (row != col).unsqueeze(-1).to(messages.dtype)
        h_agg = torch.zeros(n_nodes, self.hidden_dim, device=h.device, dtype=messages.dtype)
        h_agg.index_add_(0, row, messages * not_self)
        h_new = self.node_norm(h + self.node_mlp(torch.cat([h, h_agg], dim=-1)))

        e_new = self._update_edges(e, messages, time_scale)
        return h_new, x_new, e_new


def build_time_layers(n_layers, time_dim, edge_dim):
    return nn.ModuleList([
        nn.Sequential(nn.Linear(time_dim, edge_dim), nn.SiLU())
        for _ in range(n_layers)
    ])


def build_output_head(edge_dim, hidden_dim, out_channels):
    head = nn.Sequential(
        nn.LayerNorm(edge_dim),
        nn.Linear(edge_dim, hidden_dim),
        nn.SiLU(),
        nn.LayerNorm(hidden_dim),
        nn.Linear(hidden_dim, hidden_dim // 2),
        nn.SiLU(),
        nn.Linear(hidden_dim // 2, out_channels)
    )
    nn.init.zeros_(head[-1].weight)
    nn.init.zeros_(head[-1].bias)
    return head


class EGNNEncoder(nn.Module):
    """E(2)-equivariant score network for EDISCO.

    Dense and sparse modes share parameter names and shapes, so a checkpoint
    trained on dense graphs initializes the sparse model used for curriculum
    training at larger scales.
    """

    def __init__(self, n_layers=12, hidden_dim=256, node_dim=64, edge_dim=64,
                 time_dim=128, coord_dim=2, out_channels=2, sparse=False,
                 use_activation_checkpoint=False, coord_update_alpha=0.1,
                 weight_temp=10.0, node_feature_dim=0):
        """
        Args:
            n_layers: number of EGNN layers
            hidden_dim: width of the message, coordinate, node and edge MLPs
            node_dim, edge_dim, time_dim: node, edge and time embedding sizes
            sparse: message passing over an explicit edge list instead of all pairs
            use_activation_checkpoint: recompute activations in the backward pass
            coord_update_alpha: step size alpha of the coordinate update
            weight_temp: temperature tau of the coordinate-weight tanh
            node_feature_dim: number of E(2)-invariant node inputs (0 for TSP)
        """
        super().__init__()
        self.n_layers = n_layers
        self.node_feature_dim = node_feature_dim
        self.hidden_dim = hidden_dim
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.time_dim = time_dim
        self.coord_dim = coord_dim
        self.out_channels = out_channels

        self.sparse = sparse
        self.use_activation_checkpoint = use_activation_checkpoint

        # Initial embeddings, all E(2)-invariant: one shared learnable node
        # vector (TSP) or an embedding of invariant node inputs such as the
        # terminal indicator (ESTP), and a linear embedding of the noisy edge state.
        if node_feature_dim > 0:
            self.node_feature_embed = nn.Sequential(
                nn.Linear(node_feature_dim, node_dim),
                nn.LayerNorm(node_dim),
                nn.SiLU(),
                nn.Linear(node_dim, node_dim),
            )
        else:
            self.node_embed = nn.Parameter(torch.randn(1, 1, node_dim) * 0.02)
        self.edge_embed = nn.Linear(1, edge_dim)

        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(time_dim, time_dim * 2),
            nn.SiLU(),
            nn.Linear(time_dim * 2, time_dim),
            nn.SiLU(),
        )

        layer_cls = EGNNLayerSparse if self.sparse else EGNNLayerDense
        self.layers = nn.ModuleList([
            layer_cls(node_dim, edge_dim, hidden_dim, coord_dim,
                      coord_update_alpha, weight_temp)
            for _ in range(n_layers)
        ])

        # Per-layer time injection
        self.time_layers = build_time_layers(n_layers, time_dim, edge_dim)

        # Output head reads only invariant edge states
        self.out = build_output_head(edge_dim, hidden_dim, out_channels)

    @classmethod
    def from_args(cls, args, **overrides):
        """Build the encoder from the command-line arguments of train.py."""
        kwargs = dict(
            n_layers=args.n_layers,
            hidden_dim=args.hidden_dim,
            node_dim=getattr(args, 'node_dim', 64),
            edge_dim=getattr(args, 'edge_dim', 64),
            time_dim=getattr(args, 'time_dim', 128),
            sparse=getattr(args, 'sparse_factor', -1) > 0,
            use_activation_checkpoint=getattr(args, 'use_activation_checkpoint', False),
            coord_update_alpha=getattr(args, 'coord_update_alpha', 0.1),
            weight_temp=getattr(args, 'weight_temp', 10.0),
        )
        kwargs.update(overrides)
        return cls(**kwargs)

    def forward(self, coords, adj_matrix, timesteps, edge_index=None, node_features=None):
        """Predict clean-edge logits from noisy edge states.

        Dense mode:
            coords (B, n, 2), adj_matrix (B, n, n), timesteps (B,) or (1,),
            node_features (B, n, node_feature_dim); returns (B, n, n, out_channels).
        Sparse mode:
            coords (N, 2), adj_matrix (E,), edge_index (2, E), timesteps (1,) or
            one entry per edge, node_features (N, node_feature_dim); returns
            (E, out_channels).
        """
        if self.sparse:
            return self._forward_sparse(coords, adj_matrix, timesteps, edge_index, node_features)
        return self._forward_dense(coords, adj_matrix, timesteps, node_features)

    def _initial_nodes(self, shape, node_features, dtype):
        """Initial invariant node states with leading dimensions `shape`."""
        if self.node_feature_dim > 0:
            if node_features is None:
                raise ValueError('this encoder expects invariant node features')
            return self.node_feature_embed(
                node_features.reshape(*shape, self.node_feature_dim).to(dtype))
        return self.node_embed.reshape(*([1] * len(shape)), -1).expand(*shape, -1)

    def _run_layer(self, layer, tensors, edge_index=None, time_index=None):
        """Apply one layer, optionally with gradient checkpointing.

        `time_index` maps every edge to its row of the per-graph time scale,
        so the per-edge expansion happens inside the checkpointed function.
        """
        if edge_index is None:
            func = lambda h, x, e, ts: layer(h, x, e, ts)
        elif time_index is None:
            func = lambda h, x, e, ts: layer(h, x, e, edge_index, ts)
        else:
            func = lambda h, x, e, ts: layer(h, x, e, edge_index, ts[time_index])
        use_ckpt = self.use_activation_checkpoint and self.training
        return checkpoint(func, tensors, tuple(layer.parameters()), use_ckpt)

    def _run_head(self, e):
        """Output head, optionally with gradient checkpointing."""
        use_ckpt = self.use_activation_checkpoint and self.training
        return checkpoint(self.out, (e,), tuple(self.out.parameters()), use_ckpt)

    def _forward_dense(self, coords, adj_matrix, timesteps, node_features=None):
        """Dense forward pass on (B, n, n) edge states."""
        if coords.dim() == 2:
            coords = coords.unsqueeze(0)
        batch_size, n_nodes, _ = coords.shape

        dtype = self.edge_embed.weight.dtype
        h = self._initial_nodes((batch_size, n_nodes), node_features, dtype)
        x = coords
        adj_input = adj_matrix.reshape(batch_size, n_nodes, n_nodes, 1).to(dtype)
        e = self.edge_embed(adj_input)

        timesteps = timesteps.reshape(-1)
        if timesteps.shape[0] == 1 and batch_size > 1:
            timesteps = timesteps.expand(batch_size)
        t_emb = self.time_embed(timestep_embedding(timesteps, self.time_dim).to(dtype))

        for layer, time_layer in zip(self.layers, self.time_layers):
            time_scale = 1 + time_layer(t_emb).view(batch_size, 1, 1, -1)
            h, x, e = self._run_layer(layer, (h, x, e, time_scale))

        return self._run_head(e)

    def _forward_sparse(self, coords, adj_matrix, timesteps, edge_index, node_features=None):
        """Sparse forward pass on (E,) edge states of one or several graphs."""
        if edge_index is None:
            raise ValueError('edge_index is required in sparse mode')
        if coords.dim() == 3:
            coords = coords.reshape(-1, coords.shape[-1])
        n_nodes = coords.shape[0]
        n_edges = edge_index.shape[1]

        if edge_index.max() >= n_nodes or edge_index.min() < 0:
            raise ValueError('edge_index lies outside the node range')

        dtype = self.edge_embed.weight.dtype
        h = self._initial_nodes((n_nodes,), node_features, dtype)
        x = coords

        adj_input = adj_matrix.reshape(-1, 1).to(dtype)
        if adj_input.shape[0] != n_edges:
            raise ValueError(f'expected {n_edges} edge states, got {adj_input.shape[0]}')
        e = self.edge_embed(adj_input)

        # One diffusion time per graph. Per-edge times are reduced to their
        # distinct values so the time MLPs run once per graph, not per edge.
        timesteps = timesteps.reshape(-1)
        if timesteps.shape[0] == 1:
            inverse = None
            t_unique = timesteps
        elif timesteps.shape[0] == n_edges:
            t_unique, inverse = torch.unique(timesteps, return_inverse=True)
        else:
            raise ValueError('timesteps must be a scalar or have one entry per edge')
        t_emb = self.time_embed(timestep_embedding(t_unique, self.time_dim).to(dtype))

        for layer, time_layer in zip(self.layers, self.time_layers):
            time_scale = 1 + time_layer(t_emb)
            h, x, e = self._run_layer(layer, (h, x, e, time_scale), edge_index, inverse)

        return self._run_head(e)
