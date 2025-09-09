"""TSP (Traveling Salesman Problem) Graph Dataset for dense adjacency matrices"""

import numpy as np
import torch
from torch.utils.data import Dataset


class TSPGraphDataset(Dataset):
    """
    TSP Dataset that returns dense adjacency matrices
    Compatible with continuous-time diffusion models
    """
    
    def __init__(self, data_file, n_instances=None):
        self.data_file = data_file
        self.data = []
        self.n_cities = None
        
        # Load data from file
        with open(data_file, 'r') as f:
            lines = f.readlines()
        
        if n_instances:
            lines = lines[:n_instances]
        
        print(f'Loading TSP data from "{data_file}"')
        
        for line_idx, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            try:
                # Parse line format: coordinates... output tour...
                parts = line.split(' output ')
                if len(parts) != 2:
                    continue
                
                # Extract coordinates
                coords_str = parts[0].split()
                coords_flat = [float(x) for x in coords_str]
                n_cities = len(coords_flat) // 2
                
                if self.n_cities is None:
                    self.n_cities = n_cities
                elif self.n_cities != n_cities:
                    # Skip instances with different number of cities
                    continue
                
                coords = np.array(coords_flat).reshape(n_cities, 2)
                
                # Extract tour (1-indexed in file, convert to 0-indexed)
                tour_str = parts[1].split()
                tour = [int(x) - 1 for x in tour_str]
                
                # Handle tour format (might have repeated last city)
                if len(tour) > n_cities:
                    tour = tour[:n_cities]
                
                # Create adjacency matrix from tour
                adj_matrix = self._tour_to_adjacency_matrix(tour, n_cities)
                
                self.data.append({
                    'coordinates': coords.astype(np.float32),
                    'adjacency_matrix': adj_matrix.astype(np.float32),
                    'tour': np.array(tour, dtype=np.int64)
                })
                
            except (ValueError, IndexError) as e:
                print(f"Error parsing line {line_idx}: {e}")
                continue
        
        print(f'Loaded {len(self.data)} TSP instances with {self.n_cities} cities')
        print(f'Graph representation: Dense adjacency matrices')
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Convert to tensors
        coords = torch.tensor(item['coordinates'])
        adj_matrix = torch.tensor(item['adjacency_matrix'])
        tour = torch.tensor(item['tour'])
        
        return coords, adj_matrix, tour
    
    def _tour_to_adjacency_matrix(self, tour, n_cities):
        """Convert tour to dense adjacency matrix"""
        adj_matrix = np.zeros((n_cities, n_cities), dtype=np.float32)
        
        for i in range(len(tour)):
            u = tour[i]
            v = tour[(i + 1) % len(tour)]
            adj_matrix[u, v] = 1.0
            # Note: keeping directed edges for TSP
        
        return adj_matrix
    
    def _adjacency_matrix_to_tour(self, adj_matrix):
        """Convert adjacency matrix back to tour (for validation)"""
        n_cities = adj_matrix.shape[0]
        tour = []
        visited = set()
        
        # Start from city 0
        current = 0
        tour.append(current)
        visited.add(current)
        
        # Follow edges to build tour
        while len(tour) < n_cities:
            # Find next city
            next_cities = np.where(adj_matrix[current] > 0)[0]
            
            # Find unvisited next city
            next_city = None
            for nc in next_cities:
                if nc not in visited:
                    next_city = nc
                    break
            
            if next_city is None:
                # No valid next city found, tour is broken
                break
            
            tour.append(next_city)
            visited.add(next_city)
            current = next_city
        
        return tour
    
    def get_statistics(self):
        """Get dataset statistics"""
        if not self.data:
            return {}
        
        all_coords = np.stack([item['coordinates'] for item in self.data])
        
        stats = {
            'n_instances': len(self.data),
            'n_cities': self.n_cities,
            'coord_mean': all_coords.mean(axis=(0, 1)),
            'coord_std': all_coords.std(axis=(0, 1)),
            'coord_min': all_coords.min(axis=(0, 1)),
            'coord_max': all_coords.max(axis=(0, 1))
        }
        
        return stats