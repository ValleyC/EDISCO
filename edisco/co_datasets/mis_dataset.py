"""MIS (Maximum Independent Set) dataset for EDISCO."""

import glob
import os

import numpy as np
import torch
from torch_geometric.data import Data as GraphData

# Try to import pickle5 for Python 3.7 compatibility, fallback to pickle
try:
    import pickle5 as pickle
except ImportError:
    import pickle


class MISDataset(torch.utils.data.Dataset):
    """Dataset for Maximum Independent Set problem.

    Loads graphs from .gpickle files with optional external labels.
    Each graph instance contains:
        - Node labels (binary: 0=not in MIS, 1=in MIS)
        - Edge indices (undirected, with self-loops)
    """

    def __init__(self, data_file, data_label_dir=None):
        """
        Args:
            data_file: Glob pattern for graph files (e.g., "data/mis/*.gpickle")
            data_label_dir: Optional directory containing external label files
        """
        self.data_file = data_file
        self.file_lines = glob.glob(data_file)
        self.data_label_dir = data_label_dir
        print(f'Loaded "{data_file}" with {len(self.file_lines)} examples')

    def __len__(self):
        return len(self.file_lines)

    def get_example(self, idx):
        """Load a single graph instance.

        Returns:
            num_nodes: Number of nodes in the graph
            node_labels: Binary labels for each node (1 if in MIS)
            edges: Edge index array (2, num_edges) with bidirectional edges and self-loops
        """
        with open(self.file_lines[idx], "rb") as f:
            graph = pickle.load(f)

        num_nodes = graph.number_of_nodes()

        # Load node labels from graph metadata or external file
        if self.data_label_dir is None:
            node_labels = [_[1] for _ in graph.nodes(data='label')]
            if node_labels is not None and node_labels[0] is not None:
                node_labels = np.array(node_labels, dtype=np.int64)
            else:
                node_labels = np.zeros(num_nodes, dtype=np.int64)
        else:
            base_label_file = os.path.basename(self.file_lines[idx]).replace('.gpickle', '_unweighted.result')
            node_label_file = os.path.join(self.data_label_dir, base_label_file)
            with open(node_label_file, 'r') as f:
                node_labels = [int(_) for _ in f.read().splitlines()]
            node_labels = np.array(node_labels, dtype=np.int64)
            assert node_labels.shape[0] == num_nodes

        # Build edge index (bidirectional edges + self-loops)
        edges = np.array(graph.edges, dtype=np.int64)
        edges = np.concatenate([edges, edges[:, ::-1]], axis=0)  # Add reverse edges

        # Add self-loops
        self_loop = np.arange(num_nodes).reshape(-1, 1).repeat(2, axis=1)
        edges = np.concatenate([edges, self_loop], axis=0)
        edges = edges.T  # Shape: (2, num_edges)

        return num_nodes, node_labels, edges

    def __getitem__(self, idx):
        """Get a single data point.

        Returns:
            Tuple of (index, graph_data, point_indicator):
                - index: Original index in dataset
                - graph_data: PyTorch Geometric Data object with x (labels) and edge_index
                - point_indicator: Number of nodes (for batching)
        """
        num_nodes, node_labels, edge_index = self.get_example(idx)
        graph_data = GraphData(
            x=torch.from_numpy(node_labels),
            edge_index=torch.from_numpy(edge_index)
        )

        point_indicator = np.array([num_nodes], dtype=np.int64)
        return (
            torch.LongTensor(np.array([idx], dtype=np.int64)),
            graph_data,
            torch.from_numpy(point_indicator).long(),
        )
