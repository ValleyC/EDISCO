"""TSP utility functions for EDISCO"""

import numpy as np
import torch
import scipy.spatial


def batched_two_opt_torch(points, tours, max_iterations=1000, device="cpu"):
    """
    Batched 2-opt improvement for multiple tours
    
    Args:
        points: coordinates (batch_size, n_cities, 2) or (n_cities, 2)
        tours: initial tours (batch_size, n_cities) or (n_cities,)
        max_iterations: maximum 2-opt iterations
        device: computation device
    
    Returns:
        improved_tours: improved tours
        iterations: number of iterations performed
    """
    # Handle input shapes properly
    single_tour = False
    
    # Check if tours is 1D (single tour without batch dimension)
    if len(tours.shape) == 1:
        tours = tours[np.newaxis, :]
        single_tour = True
        if len(points.shape) == 2:
            points = points[np.newaxis, :]
    # Check if points needs batch dimension
    elif len(points.shape) == 2 and len(tours.shape) == 2:
        # tours already has batch dimension but points doesn't
        points = points[np.newaxis, :].repeat(tours.shape[0], axis=0)
    
    batch_size = tours.shape[0]
    n_cities = tours.shape[1]
    
    tours = tours.copy()
    iterator = 0
    
    with torch.no_grad():
        # Convert to torch tensors - FIX: explicitly convert to long (int64)
        cuda_points = torch.from_numpy(points).to(device)
        cuda_tours = torch.from_numpy(tours).long().to(device)  # Added .long() for int64
        
        min_change = -1.0
        while min_change < 0.0 and iterator < max_iterations:
            # Get tour points
            tour_points = cuda_points.gather(
                1, 
                cuda_tours.unsqueeze(-1).expand(-1, -1, 2).long()  # Ensure long dtype
            )
            
            # Compute all pairwise improvements
            points_i = tour_points.unsqueeze(2)  # (batch, n, 1, 2)
            points_j = tour_points.unsqueeze(1)  # (batch, 1, n, 2)
            
            # Next points in tour (with wraparound)
            next_indices = torch.arange(n_cities, device=device)
            next_indices = (next_indices + 1) % n_cities
            points_i_next = tour_points[:, next_indices, :].unsqueeze(2)
            points_j_next = tour_points[:, next_indices, :].unsqueeze(1)
            
            # Calculate distances
            dist_ij = torch.norm(points_i - points_j, dim=-1)
            dist_i_next_j_next = torch.norm(points_i_next - points_j_next, dim=-1)
            dist_i_i_next = torch.norm(points_i - points_i_next, dim=-1)
            dist_j_j_next = torch.norm(points_j - points_j_next, dim=-1)
            
            # Calculate improvement
            improvement = dist_ij + dist_i_next_j_next - dist_i_i_next - dist_j_j_next
            
            # Only consider valid swaps (i < j-1)
            mask = torch.triu(torch.ones(n_cities, n_cities, device=device), diagonal=2)
            improvement = improvement * mask - (1 - mask) * 1e10
            
            # Find best improvement for each tour
            improvement_flat = improvement.view(batch_size, -1)
            best_improvement, best_idx = improvement_flat.min(dim=1)
            
            # Check if any improvement exists
            min_change = best_improvement.min().item()
            
            if min_change < -1e-6:
                # Apply best 2-opt move for each tour
                for b in range(batch_size):
                    if best_improvement[b] < -1e-6:
                        idx = best_idx[b].item()
                        i = idx // n_cities
                        j = idx % n_cities
                        
                        # Reverse tour segment [i+1, j]
                        if i + 1 < j:
                            cuda_tours[b, i+1:j+1] = cuda_tours[b, i+1:j+1].flip(0)
                
                iterator += 1
            else:
                break
        
        tours = cuda_tours.cpu().numpy()
    
    if single_tour:
        return tours[0], iterator
    else:
        return tours, iterator


def merge_tours_dense(adj_probs, coords):
    """
    Merge tours from dense adjacency probability matrix
    
    Args:
        adj_probs: adjacency probabilities (n_nodes, n_nodes)
        coords: node coordinates (n_nodes, 2)
    
    Returns:
        tour: merged tour as node sequence
    """
    n_nodes = coords.shape[0]
    
    # Compute edge weights (probability / distance)
    weights = adj_probs.clone()
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j and adj_probs[i, j] > 0:
                dist = torch.norm(coords[i] - coords[j])
                weights[i, j] = adj_probs[i, j] / (dist + 1e-6)
    
    # Build tour using greedy edge insertion
    tour_edges = []
    degree = torch.zeros(n_nodes, dtype=torch.long)
    visited_edges = set()
    
    # Sort edges by weight (descending)
    edges = []
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j and weights[i, j] > 0:
                edges.append((i, j, weights[i, j].item()))
    
    sorted_edges = sorted(edges, key=lambda x: x[2], reverse=True)
    
    # Greedy tour construction
    for u, v, _ in sorted_edges:
        if (u, v) in visited_edges:
            continue
        
        # Check degree constraints
        if degree[u] >= 2 or degree[v] >= 2:
            continue
        
        # Check if adding edge creates subtour
        if len(tour_edges) == n_nodes - 1:
            # Last edge must complete the tour
            if degree[u] == 1 and degree[v] == 1:
                tour_edges.append((u, v))
                break
        else:
            # Add edge if valid
            tour_edges.append((u, v))
            degree[u] += 1
            degree[v] += 1
            visited_edges.add((u, v))
            visited_edges.add((v, u))
    
    # Convert edges to tour sequence
    if len(tour_edges) == n_nodes:
        tour = edges_to_tour(tour_edges, n_nodes)
        if tour is not None:
            return tour
    
    # Fallback to nearest neighbor
    return nearest_neighbor_tour(coords)


def edges_to_tour(tour_edges, n_nodes):
    """
    Convert edge list to tour sequence
    
    Args:
        tour_edges: list of (u, v) edges
        n_nodes: number of nodes
    
    Returns:
        tour: node sequence or None if invalid
    """
    # Build adjacency lists
    adj_list = {i: [] for i in range(n_nodes)}
    for u, v in tour_edges:
        adj_list[u].append(v)
        adj_list[v].append(u)
    
    # Check if valid tour (all nodes have degree 2)
    for node, neighbors in adj_list.items():
        if len(neighbors) != 2:
            return None
    
    # Build tour starting from node 0
    tour = [0]
    current = 0
    prev = -1
    
    while len(tour) < n_nodes:
        # Find next node
        next_node = None
        for neighbor in adj_list[current]:
            if neighbor != prev:
                next_node = neighbor
                break
        
        if next_node is None:
            return None
        
        tour.append(next_node)
        prev = current
        current = next_node
    
    return np.array(tour, dtype=np.int64)  # Ensure int64 dtype


def nearest_neighbor_tour(coords):
    """
    Construct tour using nearest neighbor heuristic
    
    Args:
        coords: node coordinates (n_nodes, 2)
    
    Returns:
        tour: nearest neighbor tour
    """
    if torch.is_tensor(coords):
        coords = coords.numpy()
    
    n_nodes = coords.shape[0]
    tour = [0]
    unvisited = set(range(1, n_nodes))
    current = 0
    
    while unvisited:
        # Find nearest unvisited node
        min_dist = float('inf')
        nearest = None
        
        for node in unvisited:
            dist = np.linalg.norm(coords[current] - coords[node])
            if dist < min_dist:
                min_dist = dist
                nearest = node
        
        tour.append(nearest)
        unvisited.remove(nearest)
        current = nearest
    
    return np.array(tour, dtype=np.int64)  # Ensure int64 dtype


def compute_tour_length(coords, tour):
    """
    Compute total tour length
    
    Args:
        coords: node coordinates (n_nodes, 2)
        tour: tour sequence (n_nodes,)
    
    Returns:
        length: total tour length
    """
    if torch.is_tensor(coords):
        coords = coords.numpy()
    if torch.is_tensor(tour):
        tour = tour.numpy()
    
    # Ensure tour is int64
    tour = tour.astype(np.int64)
    
    length = 0.0
    n = len(tour)
    
    for i in range(n):
        current = tour[i]
        next_city = tour[(i + 1) % n]
        length += np.linalg.norm(coords[current] - coords[next_city])
    
    return length


class TSPEvaluator:
    """
    TSP solution evaluator
    """
    
    def __init__(self, coords):
        """
        Initialize evaluator with problem coordinates
        
        Args:
            coords: node coordinates (n_nodes, 2)
        """
        if torch.is_tensor(coords):
            coords = coords.numpy()
        
        self.coords = coords
        self.n_nodes = coords.shape[0]
        
        # Precompute distance matrix
        self.dist_matrix = scipy.spatial.distance_matrix(coords, coords)
    
    def evaluate(self, tour):
        """
        Evaluate tour quality
        
        Args:
            tour: tour sequence (n_nodes,)
        
        Returns:
            cost: total tour cost
        """
        if torch.is_tensor(tour):
            tour = tour.numpy()
        
        # Ensure tour is int64
        tour = tour.astype(np.int64)
        
        cost = 0.0
        for i in range(len(tour)):
            u = tour[i]
            v = tour[(i + 1) % len(tour)]
            cost += self.dist_matrix[u, v]
        
        return cost
    
    def evaluate_batch(self, tours):
        """
        Evaluate multiple tours
        
        Args:
            tours: batch of tours (batch_size, n_nodes)
        
        Returns:
            costs: tour costs (batch_size,)
        """
        if torch.is_tensor(tours):
            tours = tours.numpy()
        
        # Ensure tours are int64
        tours = tours.astype(np.int64)
        
        batch_size = tours.shape[0]
        costs = np.zeros(batch_size)
        
        for b in range(batch_size):
            costs[b] = self.evaluate(tours[b])
        
        return costs
    
    def gap(self, tour, optimal_cost):
        """
        Compute optimality gap
        
        Args:
            tour: tour sequence
            optimal_cost: optimal tour cost
        
        Returns:
            gap: percentage gap from optimal
        """
        tour_cost = self.evaluate(tour)
        gap = (tour_cost - optimal_cost) / optimal_cost * 100
        return gap