"""PyTorch Lightning module for EDISCO on the Travelling Salesman Problem."""

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from co_datasets.tsp_graph_dataset import TSPGraphDataset
from models.egnn_encoder import EGNNEncoder
from models.gnn_encoder import GNNEncoder
from pl_meta_model import COMetaModel
from utils.equivariance_utils import random_e2_transform
from utils.native_decoder import decode_tsp_batch
from utils.tsp_utils import TSPEvaluator, batched_two_opt_torch


class NonEquivariantScoreNetwork(nn.Module):
    """Adapter giving the non-equivariant GNN the EGNN call signature.

    Used for the encoder-substitution ablation: the anisotropic GNN reads
    raw coordinates, so its edge logits depend on the coordinate frame.
    """

    def __init__(self, gnn, sparse):
        super().__init__()
        self.gnn = gnn
        self.sparse = sparse

    def forward(self, coords, adj_matrix, timesteps, edge_index=None):
        timesteps = timesteps.reshape(-1)
        if self.sparse:
            coords = coords.reshape(-1, coords.shape[-1])
            adj = adj_matrix.reshape(-1).float()
            if timesteps.shape[0] == 1:
                timesteps = timesteps.expand(adj.shape[0])
            return self.gnn(coords, timesteps, adj, edge_index)
        if coords.dim() == 2:
            coords = coords.unsqueeze(0)
        batch_size, n_nodes, _ = coords.shape
        if timesteps.shape[0] == 1 and batch_size > 1:
            timesteps = timesteps.expand(batch_size)
        adj = adj_matrix.reshape(batch_size, n_nodes, n_nodes).float()
        logits = self.gnn(coords, timesteps, adj, None)
        return logits.permute(0, 2, 3, 1)


class TSPModel(COMetaModel):
    """E(2)-equivariant categorical diffusion over the edges of a TSP instance."""

    def __init__(self, param_args=None):
        super().__init__(param_args=param_args)

        if getattr(self.args, 'disable_equivariance', False):
            # Encoder ablation: the non-equivariant GNN with matched depth and width.
            gnn = GNNEncoder(
                n_layers=self.args.n_layers, hidden_dim=self.args.hidden_dim,
                out_channels=2, sparse=self.sparse,
                use_activation_checkpoint=self.args.use_activation_checkpoint)
            self.model = NonEquivariantScoreNetwork(gnn, self.sparse)
        else:
            self.model = EGNNEncoder.from_args(self.args)

        # The training set is only needed for training.
        def load(split):
            return TSPGraphDataset(
                data_file=os.path.join(self.args.storage_path, split),
                sparse_factor=self.args.sparse_factor)
        self.train_dataset = load(self.args.training_split) if getattr(self.args, 'do_train', True) else None
        self.test_dataset = load(self.args.test_split)
        self.validation_dataset = load(self.args.validation_split)

        # Symmetry ablations for the non-equivariant encoder
        self.data_augmentation = getattr(self.args, 'data_augmentation', 'none')
        self.symmetry_loss = getattr(self.args, 'symmetry_loss', False)
        self.symmetry_loss_weight = getattr(self.args, 'symmetry_loss_weight', 1.0)

    def forward(self, coords, adj_matrix, timesteps, edge_index=None):
        """Clean-edge logits from coordinates and a noisy edge state."""
        return self.model(coords, adj_matrix, timesteps, edge_index)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _unpack_batch(self, batch):
        """Return coords, clean edge state (B, ...), edge_index and node->graph index."""
        if self.sparse:
            _, graph_data, point_indicator, _, _ = batch
            batch_size = point_indicator.shape[0]
            adj_matrix = graph_data.edge_attr.reshape(batch_size, -1).float()
            return graph_data.x, adj_matrix, graph_data.edge_index, graph_data.batch
        _, coords, adj_matrix, _ = batch
        return coords, adj_matrix, None, None

    def _predict_x0_logits(self, coords, xt, t, edge_index):
        """Clean-edge logits with a leading batch dimension."""
        if edge_index is None:
            return self.forward(coords, xt, t, None)
        batch_size, edges_per_graph = xt.shape
        logits = self.forward(coords, xt.reshape(-1),
                              t.repeat_interleave(edges_per_graph), edge_index)
        return logits.reshape(batch_size, edges_per_graph, 2)

    def training_step(self, batch, batch_idx):
        coords, adj_matrix, edge_index, node_batch = self._unpack_batch(batch)
        batch_size = adj_matrix.shape[0]

        if self.data_augmentation == 'e2':
            coords = random_e2_transform(coords, node_batch)

        # t ~ U(0, 1), X_t ~ q(X_t | X_0)
        t = torch.rand(batch_size, device=coords.device)
        xt = self.diffusion.sample_forward(adj_matrix, t)

        # x0 prediction and (1 - sqrt(t))-weighted cross-entropy
        x0_pred_logits = self._predict_x0_logits(coords, xt, t, edge_index)
        loss = self.diffusion.loss(adj_matrix, t, x0_pred_logits)

        if self.symmetry_loss:
            # Soft regularizer: predictions on a transformed copy of the
            # instance should match those on the original coordinates.
            coords_g = random_e2_transform(coords, node_batch)
            logits_g = self._predict_x0_logits(coords_g, xt, t, edge_index)
            sym_loss = (F.softmax(x0_pred_logits, dim=-1)[..., 1]
                        - F.softmax(logits_g, dim=-1)[..., 1]).pow(2).mean()
            self.log("train/symmetry_loss", sym_loss, batch_size=batch_size)
            loss = loss + self.symmetry_loss_weight * sym_loss

        self.log("train/loss", loss, prog_bar=True, batch_size=batch_size)
        return loss

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample_heatmap(self, coords, edge_index=None, n_steps=None):
        """Run the reverse CTMC and return the final clean-edge probabilities.

        Args:
            coords: (B, n, 2) for dense graphs, (n, 2) for one sparse graph
            edge_index: (2, E) for sparse graphs
        Returns:
            (B, n, n) probabilities for dense graphs, (E,) for a sparse graph
        """
        device = coords.device
        if edge_index is None:
            if coords.dim() == 2:
                coords = coords.unsqueeze(0)
            batch_size, n_nodes, _ = coords.shape
            shape = (batch_size, n_nodes, n_nodes)
        else:
            coords = coords.reshape(-1, coords.shape[-1])
            shape = (edge_index.shape[1],)

        def score_fn(x, t):
            return self.model(coords, x, torch.full((1,), float(t), device=device), edge_index)

        # X_{t_0} ~ Uniform({0, 1})
        x_T = torch.randint(0, 2, shape, device=device, dtype=torch.float32)
        return self.build_solver(num_steps=n_steps).sample(
            score_fn, x_T, schedule=self.args.time_schedule)

    def sample_with_solver(self, coords, n_steps=None, edge_index=None):
        """Sample tours: reverse diffusion followed by one decoder call."""
        heatmap = self.sample_heatmap(coords, edge_index if self.sparse else None, n_steps)
        decoder = getattr(self.args, 'decoder', 'nee')
        if self.sparse:
            points = coords.reshape(-1, coords.shape[-1]).cpu().numpy()
            tours, _ = decode_tsp_batch(heatmap.cpu().numpy().reshape(1, -1), points,
                                        edge_index.cpu().numpy(), True, 1, decoder)
        else:
            points = coords if coords.dim() == 3 else coords.unsqueeze(0)
            tours, _ = decode_tsp_batch(heatmap.cpu().numpy(), points.cpu().numpy(),
                                        None, False, 1, decoder)
        return tours, heatmap

    def _refine(self, points, tour, device):
        """Optional 2-opt local search (reference comparisons only)."""
        if self.args.two_opt_iterations <= 0:
            return tour
        refined, _ = batched_two_opt_torch(
            points.astype("float64"), np.array([tour], dtype='int64'),
            max_iterations=self.args.two_opt_iterations, device=device)
        return refined[0]

    def test_step(self, batch, batch_idx, split='test'):
        if self.sparse:
            return self._test_step_sparse(batch, batch_idx, split)
        return self._test_step_dense(batch, batch_idx, split)

    def _test_step_dense(self, batch, batch_idx, split='test'):
        """Evaluate every dense instance, retaining the best sampled candidate."""
        _, coords, adj_matrix, gt_tour = batch
        device, batch_size = coords.device, coords.shape[0]
        parallel = max(1, getattr(self.args, 'parallel_sampling', 1))
        sequential = max(1, getattr(self.args, 'sequential_sampling', 1))
        points, gt = coords.cpu().numpy(), gt_tour.cpu().numpy()
        evaluators = [TSPEvaluator(p) for p in points]
        gt_costs = np.array([e.evaluate(t) for e, t in zip(evaluators, gt)])
        best = np.full(batch_size, np.inf)
        start = self.wall_clock()
        for _ in range(sequential):
            tours, _ = self.sample_with_solver(coords.repeat_interleave(parallel, dim=0))
            for b, evaluator in enumerate(evaluators):
                for k in range(parallel):
                    tour = self._refine(points[b], tours[b * parallel + k], device)
                    best[b] = min(best[b], evaluator.evaluate(tour))
        elapsed = self.wall_clock() - start
        metrics = {
            f"{split}/gt_cost": float(gt_costs.mean()),
            f"{split}/solved_cost": float(best.mean()),
            f"{split}/gap": float(((best - gt_costs) / gt_costs * 100).mean()),
            # sampling, decoding and optional local search, per instance
            f"{split}/time": elapsed / batch_size,
        }
        if split == 'test' and getattr(self.args, 'test_equivariance', False):
            metrics[f"{split}/consistency_max_abs_dp"] = self.consistency_probe(coords, adj_matrix)
        for key, value in metrics.items():
            self.log(key, value, on_epoch=True, sync_dist=True, batch_size=batch_size)
        return metrics

    def _test_step_sparse(self, batch, batch_idx, split='test'):
        """Evaluate one sparse instance, retaining the best sampled candidate."""
        _, graph_data, point_indicator, _, gt_tour = batch
        if point_indicator.numel() != 1:
            raise ValueError('Sparse evaluation requires batch_size=1 to keep instances separate')
        coords = graph_data.x.reshape(-1, 2)
        edge_index = graph_data.edge_index
        device = coords.device
        points = coords.cpu().numpy()
        samples = (max(1, getattr(self.args, 'parallel_sampling', 1))
                   * max(1, getattr(self.args, 'sequential_sampling', 1)))

        evaluator = TSPEvaluator(points)
        gt_cost = evaluator.evaluate(gt_tour.cpu().numpy().reshape(-1))
        best = np.inf
        start = self.wall_clock()
        for _ in range(samples):
            tours, _ = self.sample_with_solver(coords, edge_index=edge_index)
            best = min(best, evaluator.evaluate(self._refine(points, tours[0], device)))
        elapsed = self.wall_clock() - start

        metrics = {
            f"{split}/gt_cost": gt_cost,
            f"{split}/solved_cost": best,
            f"{split}/gap": (best - gt_cost) / gt_cost * 100,
            f"{split}/time": elapsed,
        }
        for k, v in metrics.items():
            self.log(k, v, on_epoch=True, sync_dist=True, batch_size=1)
        return metrics

    # ------------------------------------------------------------------
    # Edge-probability consistency probe
    # ------------------------------------------------------------------

    @torch.no_grad()
    def consistency_probe(self, coords, adj_matrix):
        """Mean over instances and random g of max_ij |P_ij(x) - P_ij(g x)|.

        The noisy edge state and the diffusion time are held fixed while the
        coordinates are transformed by random elements of E(2).
        """
        num_g = getattr(self.args, 'equivariance_probe_samples', 16)
        batch_size = coords.shape[0]
        t = torch.rand(batch_size, device=coords.device)
        xt = self.diffusion.sample_forward(adj_matrix, t)
        reference = F.softmax(self.forward(coords, xt, t, None), dim=-1)[..., 1]
        deltas = []
        for _ in range(num_g):
            transformed = F.softmax(self.forward(random_e2_transform(coords), xt, t, None), dim=-1)[..., 1]
            deltas.append((reference - transformed).abs().reshape(batch_size, -1).max(dim=1).values)
        return float(torch.stack(deltas).mean())
