"""Euclidean Steiner Tree Graph Dataset

Dataset for Euclidean Steiner Tree Problem, following TSPGraphDataset pattern.
Compatible with both dense and sparse modes.
"""

import numpy as np
import torch
from sklearn.neighbors import KDTree
from torch_geometric.data import Data as GraphData


class SteinerTreeDataset(torch.utils.data.Dataset):
    """
    Dataset for Euclidean Steiner Tree Problem

    Format:
        Text file where each line contains:
        <terminals> SEP <candidates> output <adjacency>

        terminals: x1 y1 x2 y2 ... (2*n_terminals numbers)
        candidates: x1 y1 x2 y2 ... (2*n_candidates numbers)
        adjacency: 0 1 0 1 ... (flattened adjacency matrix)

    Args:
        data_file: Path to dataset file
        sparse_factor: If >0, use sparse graph with k-NN connections
    """

    def __init__(self, data_file, sparse_factor=-1):
        self.data_file = data_file
        self.sparse_factor = sparse_factor
        self.file_lines = open(data_file).read().splitlines()
        print(f'Loaded "{data_file}" with {len(self.file_lines)} lines')

    def __len__(self):
        return len(self.file_lines)

    def get_example(self, idx):
        """
        Parse a single example from the dataset

        Returns:
            coords: (n_total, 2) array of all node coordinates (terminals + candidates)
            is_terminal: (n_total,) binary array (1 for terminals, 0 for candidates)
            adjacency: (n_total, n_total) adjacency matrix of the Steiner tree
        """
        line = self.file_lines[idx].strip()

        # Split line into components
        parts = line.split(' output ')
        if len(parts) != 2:
            raise ValueError(f"Invalid line format at index {idx}")

        coords_part = parts[0]
        adj_part = parts[1]

        # Parse coordinates (terminals SEP candidates)
        if ' SEP ' in coords_part:
            terminals_str, candidates_str = coords_part.split(' SEP ')
        else:
            # Fallback: assume all are terminals (for compatibility)
            terminals_str = coords_part
            candidates_str = ""

        # Parse terminal coordinates
        terminals_vals = terminals_str.split(' ')
        terminals = np.array([
            [float(terminals_vals[i]), float(terminals_vals[i + 1])]
            for i in range(0, len(terminals_vals), 2)
        ])

        # Parse candidate coordinates
        if candidates_str:
            candidates_vals = candidates_str.split(' ')
            candidates = np.array([
                [float(candidates_vals[i]), float(candidates_vals[i + 1])]
                for i in range(0, len(candidates_vals), 2)
            ])
        else:
            candidates = np.zeros((0, 2))

        # Combine all coordinates
        coords = np.vstack([terminals, candidates]) if candidates.size > 0 else terminals
        n_terminals = len(terminals)
        n_total = len(coords)

        # Create is_terminal indicator
        is_terminal = np.zeros(n_total)
        is_terminal[:n_terminals] = 1.0

        # Parse adjacency matrix
        adj_vals = adj_part.split(' ')
        adj_flat = np.array([int(v) for v in adj_vals])

        # Reshape to square matrix
        expected_size = n_total * n_total
        if len(adj_flat) != expected_size:
            raise ValueError(
                f"Adjacency matrix size mismatch at index {idx}: "
                f"expected {expected_size}, got {len(adj_flat)}"
            )

        adjacency = adj_flat.reshape(n_total, n_total)

        return coords, is_terminal, adjacency

    def __getitem__(self, idx):
        """
        Get a single example as tensors

        Returns for dense mode (sparse_factor <= 0):
            - idx: Instance index
            - coords: (n_total, 2) coordinates tensor
            - adjacency: (n_total, n_total) adjacency tensor
            - is_terminal: (n_total,) indicator tensor

        Returns for sparse mode (sparse_factor > 0):
            - idx: Instance index
            - graph_data: PyG Data object with x, edge_index, edge_attr
            - point_indicator: Number of nodes
            - edge_indicator: Number of edges
            - is_terminal: (n_total,) indicator tensor
        """
        coords, is_terminal, adjacency = self.get_example(idx)
        n_total = len(coords)

        if self.sparse_factor <= 0:
            # Dense mode
            return (
                torch.LongTensor(np.array([idx], dtype=np.int64)),
                torch.from_numpy(coords).float(),
                torch.from_numpy(adjacency).float(),
                torch.from_numpy(is_terminal).float(),
            )
        else:
            # Sparse mode with k-NN graph
            sparse_factor = self.sparse_factor
            kdt = KDTree(coords, leaf_size=30, metric='euclidean')
            dis_knn, idx_knn = kdt.query(coords, k=sparse_factor, return_distance=True)

            # Build edge index
            edge_index_0 = torch.arange(n_total).reshape((-1, 1)).repeat(1, sparse_factor).reshape(-1)
            edge_index_1 = torch.from_numpy(idx_knn.reshape(-1))
            edge_index = torch.stack([edge_index_0, edge_index_1], dim=0)

            # Mark which edges are in the Steiner tree
            tree_edges = torch.zeros(edge_index.shape[1], dtype=torch.long)
            for i in range(n_total):
                for j in range(n_total):
                    if adjacency[i, j] > 0:
                        # Find if this edge exists in sparse graph
                        mask = (edge_index[0] == i) & (edge_index[1] == j)
                        tree_edges[mask] = 1

            # Create PyG graph data
            graph_data = GraphData(
                x=torch.from_numpy(coords).float(),
                edge_index=edge_index,
                edge_attr=tree_edges.reshape(-1, 1).float()
            )

            point_indicator = np.array([n_total], dtype=np.int64)
            edge_indicator = np.array([edge_index.shape[1]], dtype=np.int64)

            return (
                torch.LongTensor(np.array([idx], dtype=np.int64)),
                graph_data,
                torch.from_numpy(point_indicator).long(),
                torch.from_numpy(edge_indicator).long(),
                torch.from_numpy(is_terminal).float(),
            )
