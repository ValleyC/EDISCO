"""PyTorch Lightning module for EDISCO on the Euclidean Steiner Tree Problem."""

import os

import numpy as np
import torch

from co_datasets.steiner_dataset import SteinerTreeDataset
from models.egnn_encoder import EGNNEncoder
from pl_meta_model import COMetaModel
from utils.steiner_utils import SteinerTreeEvaluator, decode_steiner_tree


class SteinerTreeModel(COMetaModel):
    """E(2)-equivariant categorical diffusion over the edges of a Steiner candidate graph.

    The graph contains the terminals and pre-sampled Steiner candidates. The
    score network is the TSP encoder with one E(2)-invariant node input, the
    terminal indicator; coordinates are the only equivariant input. Edge
    probabilities are decoded into a feasible tree by a Kruskal-style greedy
    decoder.
    """

    def __init__(self, param_args=None):
        super().__init__(param_args=param_args)

        self.model = EGNNEncoder.from_args(self.args, node_feature_dim=1)

        # The training set is only needed for training.
        def load(split):
            return SteinerTreeDataset(
                data_file=os.path.join(self.args.storage_path, split),
                sparse_factor=self.args.sparse_factor,
            )
        self.train_dataset = load(self.args.training_split) if getattr(self.args, 'do_train', True) else None
        self.test_dataset = load(self.args.test_split)
        self.validation_dataset = load(self.args.validation_split)

        self.evaluator = SteinerTreeEvaluator()

    def forward(self, coords, is_terminal, adj, t, edge_index=None):
        """Clean-edge logits.

        Args:
            coords: (batch_size, n_nodes, 2) for dense graphs, (N, 2) for sparse
            is_terminal: binary indicators with the same leading shape as coords
            adj: noisy edge state, (batch_size, n_nodes, n_nodes) or (E,)
            t: diffusion time, (batch_size,), (E,) or a single value
            edge_index: (2, E) for sparse graphs
        """
        node_features = is_terminal.reshape(*coords.shape[:-1], 1)
        return self.model(coords, adj, t, edge_index, node_features=node_features)

    def _unpack_batch(self, batch):
        """Return coords, terminal indicator, clean edge state (B, ...), edge_index."""
        if self.sparse:
            _, graph_data, point_indicator, _, is_terminal = batch
            batch_size = point_indicator.shape[0]
            adj_matrix = graph_data.edge_attr.reshape(batch_size, -1).float()
            return graph_data.x, is_terminal.reshape(-1), adj_matrix, graph_data.edge_index
        _, coords, adj_matrix, is_terminal = batch
        return coords, is_terminal, adj_matrix, None

    def training_step(self, batch, batch_idx):
        coords, is_terminal, adj_matrix, edge_index = self._unpack_batch(batch)
        batch_size = adj_matrix.shape[0]

        # t ~ U(0, 1), X_t ~ q(X_t | X_0)
        t = torch.rand(batch_size, device=coords.device)
        xt = self.diffusion.sample_forward(adj_matrix, t)

        # x0 prediction and (1 - sqrt(t))-weighted cross-entropy
        if edge_index is None:
            logits = self.forward(coords, is_terminal, xt, t)
        else:
            edges_per_graph = xt.shape[1]
            logits = self.forward(coords, is_terminal, xt.reshape(-1),
                                  t.repeat_interleave(edges_per_graph), edge_index)
            logits = logits.reshape(batch_size, edges_per_graph, 2)
        loss = self.diffusion.loss(adj_matrix, t, logits)

        self.log("train/loss", loss, prog_bar=True, batch_size=batch_size)
        return loss

    @torch.no_grad()
    def sample_heatmap(self, coords, is_terminal, edge_index=None):
        """Run the reverse CTMC and return edge probabilities.

        Returns an (n, n) matrix for one instance; pairs outside a sparse
        candidate graph have probability zero.
        """
        device = coords.device
        n_nodes = coords.shape[-2]
        if edge_index is None:
            coords = coords.reshape(1, n_nodes, 2)
            is_terminal = is_terminal.reshape(1, n_nodes)
            shape = (1, n_nodes, n_nodes)
        else:
            shape = (edge_index.shape[1],)

        def score_fn(x, t):
            return self.forward(coords, is_terminal, x, torch.full((1,), float(t), device=device), edge_index)

        # X_{t_0} ~ Uniform({0, 1})
        x_T = torch.randint(0, 2, shape, device=device, dtype=torch.float32)
        probs = self.build_solver().sample(score_fn, x_T, schedule=self.args.time_schedule)
        if edge_index is None:
            return probs[0].cpu().numpy()
        heatmap = np.zeros((n_nodes, n_nodes))
        index = edge_index.cpu().numpy()
        heatmap[index[0], index[1]] = probs.cpu().numpy()
        return heatmap

    def test_step(self, batch, batch_idx, split='test'):
        """Sample, decode to a feasible tree and compare with the reference tree."""
        if self.sparse:
            instance_idx, graph_data, point_indicator, _, is_terminal = batch
            if point_indicator.numel() != 1:
                raise ValueError('Steiner evaluation processes one instance at a time')
            coords, edge_index = graph_data.x, graph_data.edge_index
            is_terminal = is_terminal.reshape(-1)
            # The reference tree may use edges outside the sparse candidate
            # graph, so its length is taken from the stored full adjacency.
            dataset = self.validation_dataset if split == 'val' else self.test_dataset
            _, _, np_gt_adj = dataset.get_example(int(instance_idx.reshape(-1)[0]))
        else:
            _, coords, gt_adj_matrix, is_terminal = batch
            if coords.shape[0] != 1:
                raise ValueError('Steiner evaluation processes one instance at a time')
            coords, is_terminal, edge_index = coords[0], is_terminal[0], None
            np_gt_adj = gt_adj_matrix[0].cpu().numpy()

        np_coords = coords.cpu().numpy()
        np_is_terminal = is_terminal.cpu().numpy()
        gt_length = self.evaluator.compute_tree_length(np_coords, np_gt_adj)

        samples = (max(1, getattr(self.args, 'parallel_sampling', 1))
                   * max(1, getattr(self.args, 'sequential_sampling', 1)))
        best_length = np.inf
        start = self.wall_clock()
        for _ in range(samples):
            heatmap = self.sample_heatmap(coords, is_terminal, edge_index)
            _, length = decode_steiner_tree(heatmap, np_coords, np_is_terminal)
            best_length = min(best_length, length)
        elapsed = self.wall_clock() - start

        metrics = {
            f"{split}/gt_cost": gt_length,
            f"{split}/solved_cost": best_length,
            f"{split}/gap": (best_length - gt_length) / gt_length * 100.0,
            f"{split}/time": elapsed,
        }
        for k, v in metrics.items():
            self.log(k, v, on_epoch=True, sync_dist=True, batch_size=1,
                     prog_bar=k.endswith('gap'))
        return metrics
