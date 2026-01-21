"""
CVRP Graph Dataset for EDISCO
Handles both dense and sparse graph representations with proper feature separation
"""

import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse
from pytorch_lightning.utilities import rank_zero_info


class CVRPGraphDataset(Dataset):
    """
    CVRP Dataset compatible with EDISCO framework
    Maintains separation of equivariant coordinates and invariant features
    """
    
    def __init__(self, data_file, sparse_factor=-1, max_instances=None):
        """
        Args:
            data_file: Path to pickle file with CVRP instances
            sparse_factor: k for k-nearest neighbor sparsification (-1 for dense)
            max_instances: Limit number of instances to load
        """
        self.data_file = data_file
        self.sparse_factor = sparse_factor
        self.sparse = sparse_factor > 0
        
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
        
        # Extract coordinates (equivariant features)
        coords = torch.FloatTensor(instance['coords'])  # (n_nodes, 2)
        
        # Extract demands (invariant features)
        demands = torch.FloatTensor(instance['demands'])  # (n_nodes,)
        
        # Create depot indicator (invariant feature)
        is_depot = torch.zeros(self.n_nodes)
        is_depot[0] = 1.0  # First node is depot by convention
        
        # Combine invariant features
        invariant_features = torch.stack([demands, is_depot], dim=-1)  # (n_nodes, 2)
        
        # Create adjacency matrix from solution if available
        if 'solution' in instance and instance['solution']:
            adj_matrix = self._create_adjacency_from_routes(instance['solution']['routes'])
        else:
            # Initialize with zeros for unsupervised learning
            adj_matrix = torch.zeros(self.n_nodes, self.n_nodes)
        
        # Get capacity
        capacity = torch.FloatTensor([instance.get('capacity', self.capacity)])
        
        if self.sparse:
            # Convert to sparse representation for large instances
            return self._to_sparse_graph(coords, invariant_features, adj_matrix, capacity, demands)
        else:
            # Dense representation for small instances
            return self._to_dense_batch(coords, invariant_features, adj_matrix, capacity, demands)
    
    def _create_adjacency_from_routes(self, routes):
        """Create adjacency matrix from CVRP solution routes"""
        adj_matrix = torch.zeros(self.n_nodes, self.n_nodes)
        
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
    
    def _to_dense_batch(self, coords, invariant_features, adj_matrix, capacity, demands):
        """Return dense format for batch processing"""
        # For dense graphs, return tensors directly
        # They will be batched by DataLoader
        return coords, invariant_features, adj_matrix, capacity, demands
    
    def _to_sparse_graph(self, coords, invariant_features, adj_matrix, capacity, demands):
        """Convert to sparse graph format using PyTorch Geometric Data"""
        # Create edge index based on k-nearest neighbors if sparse_factor > 0
        if self.sparse_factor > 0:
            edge_index = self._create_knn_edges(coords)
        else:
            # Convert dense adjacency to sparse format
            edge_index, edge_attr = dense_to_sparse(adj_matrix)
        
        # Create edge features from adjacency matrix
        edge_features = torch.zeros(edge_index.size(1))
        for i in range(edge_index.size(1)):
            src, dst = edge_index[:, i]
            edge_features[i] = adj_matrix[src, dst]
        
        # Create PyTorch Geometric Data object
        graph = Data(
            x=invariant_features,  # Node features (demands, is_depot)
            pos=coords,  # Coordinates (kept separate for equivariance)
            edge_index=edge_index,
            edge_attr=edge_features.unsqueeze(-1),
            capacity=capacity,
            demands=demands,
            adj_matrix=adj_matrix  # Keep full adjacency for evaluation
        )
        
        return graph
    
    def _create_knn_edges(self, coords):
        """Create k-nearest neighbor edge connections"""
        n_nodes = coords.size(0)
        k = min(self.sparse_factor, n_nodes - 1)
        
        # Compute pairwise distances
        dist_matrix = torch.cdist(coords, coords)
        
        # For each node, connect to k nearest neighbors
        edge_list = []
        for i in range(n_nodes):
            # Get k nearest neighbors (excluding self)
            dist_row = dist_matrix[i].clone()
            dist_row[i] = float('inf')
            _, nearest = torch.topk(dist_row, k, largest=False)
            
            for j in nearest:
                edge_list.append([i, j.item()])
                # Add reverse edge for undirected graph
                edge_list.append([j.item(), i])
        
        # Remove duplicates and create edge index
        edge_set = set(tuple(e) for e in edge_list)
        edge_index = torch.tensor(list(edge_set), dtype=torch.long).t()
        
        return edge_index
    
    def get_instance_info(self, idx):
        """Get additional information about an instance for evaluation"""
        instance = self.data[idx]
        info = {
            'n_customers': self.n_customers,
            'capacity': self.capacity,
        }
        
        if 'solution' in instance and instance['solution']:
            info['optimal_routes'] = instance['solution']['routes']
            info['optimal_distance'] = instance['solution'].get('total_distance', None)
        
        return info


class CVRPDataModule:
    """
    Data module for CVRP compatible with PyTorch Lightning
    """
    
    def __init__(self, args):
        self.args = args
        self.sparse = args.sparse_factor > 0
        
    def setup(self):
        """Setup train, validation, and test datasets"""
        storage_path = self.args.storage_path
        
        self.train_dataset = CVRPGraphDataset(
            data_file=os.path.join(storage_path, self.args.training_split),
            sparse_factor=self.args.sparse_factor
        )
        
        self.val_dataset = CVRPGraphDataset(
            data_file=os.path.join(storage_path, self.args.validation_split),
            sparse_factor=self.args.sparse_factor,
            max_instances=self.args.validation_examples
        )
        
        self.test_dataset = CVRPGraphDataset(
            data_file=os.path.join(storage_path, self.args.test_split),
            sparse_factor=self.args.sparse_factor
        )
    
    def train_dataloader(self):
        if self.sparse:
            from torch_geometric.loader import DataLoader
            return DataLoader(
                self.train_dataset,
                batch_size=self.args.batch_size,
                shuffle=True,
                num_workers=self.args.num_workers,
                pin_memory=True
            )
        else:
            from torch.utils.data import DataLoader
            return DataLoader(
                self.train_dataset,
                batch_size=self.args.batch_size,
                shuffle=True,
                num_workers=self.args.num_workers,
                pin_memory=True
            )
    
    def val_dataloader(self):
        if self.sparse:
            from torch_geometric.loader import DataLoader
            return DataLoader(
                self.val_dataset,
                batch_size=self.args.batch_size,
                shuffle=False,
                num_workers=self.args.num_workers,
                pin_memory=True
            )
        else:
            from torch.utils.data import DataLoader
            return DataLoader(
                self.val_dataset,
                batch_size=self.args.batch_size,
                shuffle=False,
                num_workers=self.args.num_workers,
                pin_memory=True
            )
    
    def test_dataloader(self):
        if self.sparse:
            from torch_geometric.loader import DataLoader
            return DataLoader(
                self.test_dataset,
                batch_size=self.args.batch_size,
                shuffle=False,
                num_workers=self.args.num_workers,
                pin_memory=True
            )
        else:
            from torch.utils.data import DataLoader
            return DataLoader(
                self.test_dataset,
                batch_size=self.args.batch_size,
                shuffle=False,
                num_workers=self.args.num_workers,
                pin_memory=True
            )