"""PyTorch Lightning module for EDISCO on the Capacitated Vehicle Routing Problem."""

import os

import numpy as np
import torch
from pytorch_lightning.utilities import rank_zero_info

from co_datasets.cvrp_graph_dataset import CVRPGraphDataset
from models.egnn_encoder_cvrp import EGNNEncoderCVRP
from pl_meta_model import COMetaModel
from utils.cvrp_utils import (
    CVRPEvaluator, apply_2opt_cvrp, decode_cvrp_greedy, decode_cvrp_nee, merge_cvrp_routes
)


class CVRPModel(COMetaModel):
    """Capacity-conditioned E(2)-equivariant diffusion over the edges of a CVRP instance.

    Coordinates are the only equivariant input; demands, the depot indicator
    and the vehicle capacity enter through invariant channels. Dense graphs
    are used by default; --sparse_factor k restricts the edge variables to the
    k-nearest-neighbour graph (used above N = 100).
    """

    # Binary target over node pairs, see CVRPGraphDataset
    target = 'routes'

    def __init__(self, param_args=None):
        super().__init__(param_args=param_args)

        self.invariant_dim = 2  # demand and depot indicator
        self.evaluator = CVRPEvaluator()

        # The training set is only needed for training.
        def load(split):
            return CVRPGraphDataset(
                data_file=os.path.join(self.args.storage_path, split),
                sparse_factor=self.args.sparse_factor,
                target=self.target,
            )
        self.train_dataset = load(self.args.training_split) if getattr(self.args, 'do_train', True) else None
        self.test_dataset = load(self.args.test_split)
        self.validation_dataset = load(self.args.validation_split)

        reference = self.train_dataset or self.validation_dataset
        self.n_customers = reference.n_customers
        self.n_nodes = reference.n_nodes
        self.capacity = reference.capacity

        # Capacity conditioning (invariant capacity features and FiLM on the
        # scalar messages) is disabled only for the unconditioned variants of
        # the capacity-shift study.
        self.capacity_conditioning = not getattr(self.args, 'disable_capacity_conditioning', False)
        reference_capacity = getattr(self.args, 'default_capacity', None)
        self.model = EGNNEncoderCVRP(
            n_layers=self.args.n_layers, hidden_dim=self.args.hidden_dim,
            node_dim=getattr(self.args, 'node_dim', 64),
            edge_dim=getattr(self.args, 'edge_dim', 64),
            time_dim=getattr(self.args, 'time_dim', 128),
            invariant_dim=self.invariant_dim, out_channels=2,
            default_capacity=self.capacity if reference_capacity is None else reference_capacity,
            capacity_conditioning=self.capacity_conditioning,
            use_activation_checkpoint=self.args.use_activation_checkpoint,
            sparse=self.sparse,
            coord_update_alpha=getattr(self.args, 'coord_update_alpha', 0.1),
            weight_temp=getattr(self.args, 'weight_temp', 10.0))

        rank_zero_info(f"{self.__class__.__name__}: {self.n_customers} customers, capacity {self.capacity}, "
                       f"capacity conditioning: {self.capacity_conditioning}")

    @property
    def eval_batch_size(self):
        return self.args.batch_size

    def forward(self, coords, invariant_features, adj_matrix, t, edge_index=None,
                capacity=None, demands=None, node_batch=None):
        """Clean-edge logits.

        Args:
            coords: node coordinates (batch_size, n_nodes, 2), or (N, 2) in sparse mode
            invariant_features: demand and depot indicator, same leading shape as coords
            adj_matrix: noisy edge state (batch_size, n_nodes, n_nodes), or (E,) in sparse mode
            t: one diffusion time per instance
            edge_index: (2, E) candidate edges in sparse mode
            capacity: vehicle capacity per instance, (batch_size,) or (batch_size, 1)
            node_batch: (N,) instance index of every node in sparse mode
        """
        if capacity is None:
            raise ValueError('capacity must be supplied to the CVRP encoder')
        if demands is None:
            demands = invariant_features[..., 0]
        return self.model(coords, demands, capacity.reshape(-1),
                          invariant_features, adj_matrix, t, edge_index, node_batch)

    def _unpack_batch(self, batch):
        """Bring dense and sparse batches to one set of named tensors."""
        if self.sparse:
            graph, capacity, reference = batch
            batch_size = capacity.shape[0]
            return dict(
                coords=graph.x, invariant=graph.invariant, demands=graph.demands,
                capacity=capacity.reshape(-1), reference=reference.reshape(-1),
                target=graph.edge_attr.reshape(batch_size, -1).float(),
                edge_index=graph.edge_index, node_batch=graph.batch, batch_size=batch_size)
        coords, invariant_features, target, capacity, demands, reference = batch
        return dict(
            coords=coords, invariant=invariant_features, demands=demands,
            capacity=capacity.reshape(-1), reference=reference.reshape(-1),
            target=target, edge_index=None, node_batch=None, batch_size=coords.shape[0])

    def _predict_x0_logits(self, b, xt, t):
        """Clean-state logits for the noisy state `xt`, in the shape of `xt`."""
        return self.forward(b['coords'], b['invariant'], xt, t, b['edge_index'],
                            capacity=b['capacity'], demands=b['demands'], node_batch=b['node_batch'])

    def training_step(self, batch, batch_idx):
        b = self._unpack_batch(batch)
        batch_size, target = b['batch_size'], b['target']

        # t ~ U(0, 1), X_t ~ q(X_t | X_0)
        t = torch.rand(batch_size, device=target.device)
        xt = self.diffusion.sample_forward(target, t)

        # x0 prediction and (1 - sqrt(t))-weighted cross-entropy
        logits = self._predict_x0_logits(b, xt.reshape(-1) if self.sparse else xt, t)
        loss = self.diffusion.loss(target, t, logits.reshape(batch_size, -1, 2))

        self.log('train/loss', loss, prog_bar=True, batch_size=batch_size)
        return loss

    @torch.no_grad()
    def sample_heatmap(self, b):
        """Reverse CTMC sampling; returns final probabilities shaped like the target."""
        device = b['target'].device
        batch_size = b['batch_size']
        shape = (b['target'].numel(),) if self.sparse else tuple(b['target'].shape)

        def score_fn(x, t):
            return self._predict_x0_logits(b, x, torch.full((batch_size,), float(t), device=device))

        # X_{t_0} ~ Uniform({0, 1})
        x_T = torch.randint(0, 2, shape, device=device).float()
        probs = self.build_solver().sample(score_fn, x_T, schedule=self.args.time_schedule)
        return probs.reshape(batch_size, -1) if self.sparse else probs

    def decode_instance(self, probs, coords, demands, capacity, edge_index=None):
        """Feasibility projection of one heatmap onto capacity-feasible routes."""
        decode = decode_cvrp_greedy if getattr(self.args, 'decoder', 'nee') == 'greedy' else decode_cvrp_nee
        return decode(probs, coords, demands, capacity, edge_index=edge_index)

    def test_step(self, batch, batch_idx, split='test'):
        b = self._unpack_batch(batch)
        batch_size = b['batch_size']
        start = self.wall_clock()
        probs = self.sample_heatmap(b).cpu()

        coords, demands = b['coords'].cpu(), b['demands'].cpu()
        if self.sparse:
            # nodes and edges of each instance are contiguous in the batch
            n_nodes = coords.shape[0] // batch_size
            n_edges = b['edge_index'].shape[1] // batch_size
            coords = coords.reshape(batch_size, n_nodes, 2)
            demands = demands.reshape(batch_size, n_nodes)
            edge_index = b['edge_index'].cpu().reshape(2, batch_size, n_edges)

        distances, n_routes, gaps = [], [], []
        for i in range(batch_size):
            capacity_i = float(b['capacity'][i])
            local_edges = None if not self.sparse else edge_index[:, i] - i * n_nodes
            routes = self.decode_instance(probs[i], coords[i], demands[i], capacity_i, local_edges)
            coords_i, demands_i = coords[i].numpy(), demands[i].numpy()

            # Optional 2-opt within routes (reference comparisons only)
            if self.args.two_opt_iterations > 0:
                routes = apply_2opt_cvrp(routes, coords_i, self.args.two_opt_iterations)

            # Optional route merging
            if getattr(self.args, 'merge_routes', False):
                routes = merge_cvrp_routes(routes, demands_i, capacity_i)

            distance = self.evaluator.compute_total_distance(coords_i, routes)
            distances.append(distance)
            n_routes.append(len(routes))

            # Gap to the reference solution when one is stored with the instance
            reference_i = float(b['reference'][i])
            if np.isfinite(reference_i) and reference_i > 0:
                gaps.append((distance - reference_i) / reference_i * 100)

        metrics = {
            f'{split}/solved_cost': float(np.mean(distances)),
            f'{split}/n_routes': float(np.mean(n_routes)),
            # sampling, decoding and optional post-processing, per instance
            f'{split}/time': (self.wall_clock() - start) / batch_size,
        }
        if gaps:
            metrics[f'{split}/gap'] = float(np.mean(gaps))
        for key, value in metrics.items():
            self.log(key, value, on_epoch=True, sync_dist=True, batch_size=batch_size,
                     prog_bar=key.endswith('gap'))
        return metrics
