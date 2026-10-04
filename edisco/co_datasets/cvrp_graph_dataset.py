"""CVRP dataset (dense graphs or kNN-sparsified graphs)."""

import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data as GraphData
from pytorch_lightning.utilities import rank_zero_info

from co_datasets.tsp_graph_dataset import knn_indices


class CVRPGraphDataset(Dataset):
    """
    CVRP dataset that keeps equivariant coordinates and invariant features separate.

    Each instance provides coordinates (node 0 is the depot), the invariant
    node features `[demand, is_depot]`, the vehicle capacity and a binary
    target over node pairs:

        target='routes'     directed route adjacency of the reference solution
                            (depot -> first customer -> ... -> last customer -> depot)
        target='partition'  symmetric same-route indicator over customer pairs
                            (partition diffusion for large-scale CVRP)

    With sparse_factor > 0 the node pairs are restricted to the k-nearest-
    neighbour graph built from pairwise distances with ties broken by node
    index, the same rule as for TSP.
    """

    def __init__(self, data_file, sparse_factor=-1, max_instances=None, target='routes'):
        """
        Args:
            data_file: Path to pickle file with CVRP instances
            sparse_factor: k for k-nearest neighbor sparsification (<= 0 for dense)
            max_instances: Limit number of instances to load
            target: 'routes' or 'partition'
        """
        if target not in ('routes', 'partition'):
            raise ValueError("target must be 'routes' or 'partition'")
        self.data_file = data_file
        self.sparse_factor = sparse_factor
        self.sparse = sparse_factor > 0
        self.target = target

        # Load data
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Data file not found: {data_file}")

        with open(data_file, 'rb') as f:
            self.data = pickle.load(f)

        if max_instances:
            self.data = self.data[:max_instances]

        # Extract problem info from first instance
        first = self.data[0]
        self.n_customers = first.get('n_customers', len(first['coords']) - 1)
        self.n_nodes = first.get('n_nodes', len(first['coords']))
        self.capacity = first.get('capacity', 50)

        rank_zero_info(f"Loaded {len(self.data)} CVRP instances from {data_file}")
        rank_zero_info(f"  Customers: {self.n_customers}, Capacity: {self.capacity}")
        rank_zero_info(f"  Graph mode: {'SPARSE (k={})'.format(sparse_factor) if self.sparse else 'DENSE'}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        instance = self.data[idx]

        # Coordinates (equivariant) and demands / depot indicator (invariant)
        coords = torch.FloatTensor(instance['coords'])  # (n_nodes, 2)
        demands = torch.FloatTensor(instance['demands'])  # (n_nodes,)
        n_nodes = coords.shape[0]
        is_depot = torch.zeros(n_nodes)
        is_depot[0] = 1.0  # First node is depot by convention
        invariant_features = torch.stack([demands, is_depot], dim=-1)  # (n_nodes, 2)

        # Target from the reference solution (zeros when unavailable)
        reference = float('nan')
        target = torch.zeros(n_nodes, n_nodes)
        if 'solution' in instance and instance['solution']:
            routes = instance['solution']['routes']
            if self.target == 'routes':
                target = self._create_adjacency_from_routes(routes, n_nodes)
            else:
                target = self._create_partition_from_routes(routes, n_nodes)
            distance = instance['solution'].get('total_distance')
            reference = float('nan') if distance is None else float(distance)
        reference = torch.FloatTensor([reference])

        capacity = torch.FloatTensor([instance.get('capacity', self.capacity)])

        if not self.sparse:
            return coords, invariant_features, target, capacity, demands, reference

        # Sparse candidate graph: k nearest neighbours of every node
        k = min(self.sparse_factor, n_nodes)
        neighbours = torch.from_numpy(knn_indices(instance['coords'].astype(np.float64), k))
        edge_index = torch.stack([
            torch.arange(n_nodes).reshape(-1, 1).repeat(1, k).reshape(-1),
            neighbours.reshape(-1),
        ], dim=0)
        graph = GraphData(
            x=coords,
            edge_index=edge_index,
            edge_attr=target[edge_index[0], edge_index[1]].reshape(-1, 1),
            invariant=invariant_features,
            demands=demands,
        )
        return graph, capacity, reference

    @staticmethod
    def _create_adjacency_from_routes(routes, n_nodes):
        """Directed route adjacency of a CVRP solution"""
        adj_matrix = torch.zeros(n_nodes, n_nodes)

        for route in routes:
            if len(route) == 0:
                continue

            # Depot -> first customer
            adj_matrix[0, route[0]] = 1.0

            # Customer -> customer edges within route
            for i in range(len(route) - 1):
                adj_matrix[route[i], route[i+1]] = 1.0

            # Last customer -> depot
            adj_matrix[route[-1], 0] = 1.0

        return adj_matrix

    @staticmethod
    def _create_partition_from_routes(routes, n_nodes):
        """Symmetric same-route indicator: A_ij = 1 iff customers i != j share a route"""
        partition = torch.zeros(n_nodes, n_nodes)
        for route in routes:
            members = torch.as_tensor(list(route), dtype=torch.long)
            if len(members) > 1:
                partition[members.unsqueeze(1), members.unsqueeze(0)] = 1.0
        partition.fill_diagonal_(0.0)
        return partition

    def get_instance_info(self, idx):
        """Get additional information about an instance for evaluation"""
        instance = self.data[idx]
        info = {
            'n_customers': self.n_customers,
            'capacity': instance.get('capacity', self.capacity),
        }

        if 'solution' in instance and instance['solution']:
            info['optimal_routes'] = instance['solution']['routes']
            info['optimal_distance'] = instance['solution'].get('total_distance', None)

        return info
