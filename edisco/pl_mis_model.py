"""PyTorch Lightning module for EDISCO on the Maximum Independent Set problem.

MIS has no geometric structure, so the score network is the anisotropic GNN
over node variables instead of the EGNN. The forward process, the training
objective and the reverse solvers are the same as for the routing problems.
"""

import os

import numpy as np
import scipy.sparse
import torch
import torch.nn.functional as F

from co_datasets.mis_dataset import MISDataset
from models.gnn_encoder import GNNEncoder
from pl_meta_model import COMetaModel
from utils.mis_utils import mis_decode_np

# The GNN embeds diffusion times on the scale [0, 1000].
TIME_SCALE = 1000.0


class MISModel(COMetaModel):
    """Categorical diffusion over the node variables of a graph."""

    def __init__(self, param_args=None):
        super().__init__(param_args=param_args)

        self.model = GNNEncoder(
            n_layers=self.args.n_layers, hidden_dim=self.args.hidden_dim,
            out_channels=2, sparse=True, node_feature_only=True,
            use_activation_checkpoint=self.args.use_activation_checkpoint)

        label_dir = getattr(self.args, 'training_split_label_dir', None)
        if label_dir is not None:
            label_dir = os.path.join(self.args.storage_path, label_dir)

        # The training set is only needed for training.
        self.train_dataset = MISDataset(
            data_file=os.path.join(self.args.storage_path, self.args.training_split),
            data_label_dir=label_dir,
        ) if getattr(self.args, 'do_train', True) else None
        self.test_dataset = MISDataset(
            data_file=os.path.join(self.args.storage_path, self.args.test_split))
        self.validation_dataset = MISDataset(
            data_file=os.path.join(self.args.storage_path, self.args.validation_split))

    def forward(self, x, t, edge_index):
        """Clean-node logits (num_nodes, 2).

        Args:
            x: noisy node state encoded as -1 / +1, (num_nodes,)
            t: diffusion time of every node in [0, 1], (num_nodes,)
            edge_index: (2, num_edges)
        """
        return self.model(x.float(), t.float() * TIME_SCALE, edge_index=edge_index)

    def training_step(self, batch, batch_idx):
        _, graph_data, point_indicator = batch
        node_labels = graph_data.x
        edge_index = graph_data.edge_index.reshape(2, -1)
        device = node_labels.device
        node_counts = point_indicator.reshape(-1).to(device)
        batch_size = node_counts.shape[0]

        # t ~ U(0, 1) per graph, X_t ~ q(X_t | X_0)
        t = torch.rand(batch_size, device=device)
        t_nodes = t.repeat_interleave(node_counts)
        xt = self.diffusion.sample_forward(node_labels, t_nodes)

        x0_pred = self.forward(xt * 2 - 1, t_nodes, edge_index)

        # (1 - sqrt(t))-weighted cross-entropy: averaged over the nodes of
        # each graph, weighted per graph, then averaged over the batch
        graph_index = torch.arange(batch_size, device=device).repeat_interleave(node_counts)
        ce = F.cross_entropy(x0_pred, node_labels.long(), reduction='none')
        ce_per_graph = torch.zeros(batch_size, device=device, dtype=ce.dtype).index_add_(0, graph_index, ce)
        ce_per_graph = ce_per_graph / node_counts.to(ce.dtype)
        loss = ((1.0 - torch.sqrt(t)) * ce_per_graph).mean()
        self.log("train/loss", loss, prog_bar=True, batch_size=batch_size)
        return loss

    def test_step(self, batch, batch_idx, split='test'):
        """Reverse CTMC sampling followed by greedy decoding."""
        _, graph_data, _ = batch
        node_labels = graph_data.x
        device = node_labels.device
        num_nodes = node_labels.shape[0]
        edge_index = graph_data.edge_index.reshape(2, -1)
        edge_index_np = edge_index.cpu().numpy()
        adj_mat = scipy.sparse.coo_matrix(
            (np.ones_like(edge_index_np[0]), (edge_index_np[0], edge_index_np[1])),
        )

        # Parallel samples are drawn on disjoint copies of the graph.
        parallel = max(1, self.args.parallel_sampling)
        sequential = max(1, self.args.sequential_sampling)
        if parallel > 1:
            offsets = torch.arange(parallel, device=device).view(1, -1, 1) * num_nodes
            edge_index = (edge_index.reshape(2, 1, -1) + offsets).reshape(2, -1)

        def score_fn(x, t):
            t_nodes = torch.full((x.shape[0],), float(t), device=device)
            return self.forward(x * 2 - 1, t_nodes, edge_index)

        solver = self.build_solver()
        samples = []
        start = self.wall_clock()
        for _ in range(sequential):
            # X_{t_0} ~ Uniform({0, 1})
            x_T = torch.randint(0, 2, (num_nodes * parallel,), device=device).float()
            probs = solver.sample(score_fn, x_T, schedule=self.args.time_schedule)
            samples.extend(np.split(probs.cpu().numpy(), parallel))

        # Greedy decoding: vertices in decreasing probability, added subject to independence
        best_size = max(mis_decode_np(sample, adj_mat).sum() for sample in samples)
        elapsed = self.wall_clock() - start

        gt_size = node_labels.cpu().numpy().sum()
        metrics = {
            f"{split}/gt_cost": float(gt_size),
            f"{split}/solved_cost": float(best_size),
            f"{split}/time": elapsed,
        }
        # MIS is a maximization problem: gap = (reference - solved) / reference
        if gt_size > 0:
            metrics[f"{split}/gap"] = float((gt_size - best_size) / gt_size * 100)
        for key, value in metrics.items():
            self.log(key, value, on_epoch=True, sync_dist=True, batch_size=1,
                     prog_bar=key.endswith('solved_cost'))
        return metrics
