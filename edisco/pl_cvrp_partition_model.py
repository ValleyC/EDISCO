"""PyTorch Lightning module for partition diffusion on large-scale CVRP.

Stage 1 diffuses the symmetric same-route indicator over customer pairs with
the same categorical CTMC and capacity-conditioned E(2)-equivariant score
network as end-to-end CVRP. A combinatorial projection (spectral clustering
followed by capacity-feasible re-balancing) turns the denoised affinities into
customer clusters. Stage 2 solves every cluster together with the depot as a
small Euclidean TSP with the EDISCO TSP score network and NEE.
"""

import numpy as np
import torch
from pytorch_lightning.utilities import rank_zero_info

from models.egnn_encoder import EGNNEncoder
from pl_cvrp_model import CVRPModel
from utils.partition_utils import (
    customer_affinity, distance_affinity, kmeans_partition, num_clusters,
    partition_to_routes, rebalance_capacity, spectral_partition,
)


class CVRPPartitionModel(CVRPModel):
    """Partition diffusion: categorical CTMC over the pairwise same-route indicator."""

    target = 'partition'

    def __init__(self, param_args=None):
        super().__init__(param_args=param_args)

        # 'diffusion' is the proposed partitioner. 'kmeans' replaces it with
        # Lloyd's algorithm on raw coordinates and keeps the projection and
        # the sub-TSP solver unchanged (ablation).
        self.partitioner = getattr(self.args, 'partitioner', 'diffusion')

        # Stage 2: the unmodified EDISCO TSP score network. Without a
        # checkpoint, clusters are ordered by NEE on distances alone.
        self._sub_tsp = [None]  # kept out of the module tree: it is not trained or checkpointed
        sub_tsp_ckpt = getattr(self.args, 'sub_tsp_ckpt', None)
        if sub_tsp_ckpt:
            sub_tsp_model = EGNNEncoder.from_args(
                self.args, sparse=False, use_activation_checkpoint=False,
                n_layers=getattr(self.args, 'sub_tsp_n_layers', None) or self.args.n_layers,
                hidden_dim=getattr(self.args, 'sub_tsp_hidden_dim', None) or self.args.hidden_dim)
            state = torch.load(sub_tsp_ckpt, map_location='cpu')['state_dict']
            state = {k[len('model.'):]: v for k, v in state.items() if k.startswith('model.')}
            sub_tsp_model.load_state_dict(state, strict=True)
            self._sub_tsp[0] = sub_tsp_model.requires_grad_(False).eval()
            rank_zero_info(f"Loaded sub-TSP score network from {sub_tsp_ckpt}")

    def decode_instance(self, probs, coords, demands, capacity, edge_index=None):
        """Project one affinity heatmap onto capacity-feasible routes."""
        points = coords.numpy() if torch.is_tensor(coords) else np.asarray(coords)
        customer_demands = (demands.numpy() if torch.is_tensor(demands) else np.asarray(demands))[1:]
        n_nodes = len(points)
        k = num_clusters(customer_demands, capacity)
        seed = getattr(self.args, 'seed', None) or 0

        if self.partitioner == 'kmeans':
            affinity = distance_affinity(points)
            labels = kmeans_partition(points, k, seed=seed)
        else:
            affinity = customer_affinity(probs, n_nodes, edge_index)
            labels = spectral_partition(affinity, k, seed=seed)

        clusters = rebalance_capacity(labels, affinity, customer_demands, capacity)
        sub_tsp_model = self._sub_tsp[0]
        if sub_tsp_model is not None:
            sub_tsp_model.to(self.device)
        return partition_to_routes(
            clusters, points, sub_tsp_model,
            solver_type=getattr(self.args, 'sub_tsp_solver_type', 'deis'),
            solver_steps=getattr(self.args, 'sub_tsp_solver_steps', 5),
            beta_min=getattr(self.args, 'beta_min', 0.1),
            beta_max=getattr(self.args, 'beta_max', 1.5))
