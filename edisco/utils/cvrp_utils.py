"""
CVRP-specific utilities for EDISCO
Includes capacity-aware decoding, evaluation metrics, and route construction
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional


class CVRPEvaluator:
    """Evaluator for CVRP solutions"""
    
    @staticmethod
    def compute_route_distance(coords: np.ndarray, route: List[int]) -> float:
        """Compute distance for a single route including depot connections"""
        if len(route) == 0:
            return 0.0
        
        distance = 0.0
        
        # Depot (0) to first customer
        distance += np.linalg.norm(coords[0] - coords[route[0]])
        
        # Customer to customer
        for i in range(len(route) - 1):
            distance += np.linalg.norm(coords[route[i]] - coords[route[i+1]])
        
        # Last customer to depot
        distance += np.linalg.norm(coords[route[-1]] - coords[0])
        
        return distance
    
    @staticmethod
    def compute_total_distance(coords: np.ndarray, routes: List[List[int]]) -> float:
        """Compute total distance for all routes"""
        total_distance = 0.0
        for route in routes:
            total_distance += CVRPEvaluator.compute_route_distance(coords, route)
        return total_distance
    
    @staticmethod
    def verify_solution(routes: List[List[int]], demands: np.ndarray, 
                       capacity: float, n_customers: int) -> Dict[str, any]:
        """Verify CVRP solution feasibility"""
        visited = set()
        feasible = True
        violations = []
        
        for route_idx, route in enumerate(routes):
            # Check capacity constraint
            route_demand = sum(demands[i] for i in route)
            if route_demand > capacity:
                feasible = False
                violations.append(f"Route {route_idx}: demand {route_demand} > capacity {capacity}")
            
            # Check for duplicate visits
            for customer in route:
                if customer in visited:
                    feasible = False
                    violations.append(f"Customer {customer} visited multiple times")
                visited.add(customer)
        
        # Check all customers are visited (excluding depot at index 0)
        missing = set(range(1, n_customers + 1)) - visited
        if missing:
            feasible = False
            violations.append(f"Unvisited customers: {missing}")
        
        return {
            'feasible': feasible,
            'violations': violations,
            'n_routes': len(routes),
            'visited_customers': len(visited),
            'total_customers': n_customers
        }


def decode_cvrp_greedy(adj_probs: torch.Tensor, 
                       coords: torch.Tensor,
                       demands: torch.Tensor,
                       capacity: float,
                       symmetrize: bool = True) -> List[List[int]]:
    """
    Capacity-aware greedy decoder for CVRP as described in the EDISCO paper
    
    Args:
        adj_probs: Edge probabilities from diffusion model (n_nodes, n_nodes)
        coords: Node coordinates (n_nodes, 2)
        demands: Customer demands (n_nodes,)
        capacity: Vehicle capacity
        symmetrize: Whether to symmetrize adjacency probabilities (as in paper)
    
    Returns:
        routes: List of routes, each route is a list of customer indices
    """
    n_nodes = coords.shape[0]
    
    # Symmetrize adjacency as described in paper: s_ij = (P_ij + P_ji) / d_ij
    if symmetrize:
        adj_probs = (adj_probs + adj_probs.T) / 2
    
    # Compute edge scores
    edge_scores = torch.zeros_like(adj_probs)
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:
                dist = torch.norm(coords[i] - coords[j])
                edge_scores[i, j] = adj_probs[i, j] / (dist + 1e-6)
    
    routes = []
    unvisited = set(range(1, n_nodes))  # Skip depot (0)
    
    while unvisited:
        route = []
        current_capacity = 0
        
        # Find best starting customer from depot
        feasible_starts = []
        for j in unvisited:
            if demands[j] <= capacity:
                score = edge_scores[0, j].item()
                feasible_starts.append((score, j))
        
        if not feasible_starts:
            # No feasible customers, create individual routes for remaining
            for customer in list(unvisited):
                routes.append([customer])
            break
        
        # Start route with highest scoring feasible edge from depot
        feasible_starts.sort(reverse=True)
        _, current_customer = feasible_starts[0]
        route.append(current_customer)
        current_capacity = demands[current_customer].item()
        unvisited.remove(current_customer)
        
        # Extend route greedily
        while unvisited:
            # Find feasible next customers
            candidates = []
            for next_customer in unvisited:
                if current_capacity + demands[next_customer] <= capacity:
                    score = edge_scores[current_customer, next_customer].item()
                    candidates.append((score, next_customer))
            
            if not candidates:
                # No feasible extension, return to depot
                break
            
            # Select highest scoring feasible customer
            candidates.sort(reverse=True)
            _, next_customer = candidates[0]
            
            route.append(next_customer)
            current_capacity += demands[next_customer].item()
            unvisited.remove(next_customer)
            current_customer = next_customer
        
        if route:
            routes.append(route)
    
    return routes


def decode_cvrp_sampling(adj_probs: torch.Tensor,
                        coords: torch.Tensor, 
                        demands: torch.Tensor,
                        capacity: float,
                        temperature: float = 1.0) -> List[List[int]]:
    """
    Sampling-based decoder for CVRP with temperature control
    """
    n_nodes = coords.shape[0]
    
    # Symmetrize and compute edge scores
    adj_probs = (adj_probs + adj_probs.T) / 2
    
    edge_scores = torch.zeros_like(adj_probs)
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:
                dist = torch.norm(coords[i] - coords[j])
                edge_scores[i, j] = adj_probs[i, j] / (dist + 1e-6)
    
    # Apply temperature
    edge_probs = F.softmax(edge_scores / temperature, dim=-1)
    
    routes = []
    unvisited = set(range(1, n_nodes))
    
    while unvisited:
        route = []
        current_capacity = 0
        current_node = 0  # Start at depot
        
        # Sample first customer
        feasible_mask = torch.zeros(n_nodes)
        for j in unvisited:
            if demands[j] <= capacity:
                feasible_mask[j] = 1.0
        
        if feasible_mask.sum() == 0:
            # Handle remaining customers
            for customer in list(unvisited):
                routes.append([customer])
            break
        
        probs = edge_probs[current_node] * feasible_mask
        probs = probs / probs.sum()
        
        next_customer = torch.multinomial(probs, 1).item()
        if next_customer == 0:  # Avoid sampling depot
            continue
            
        route.append(next_customer)
        current_capacity = demands[next_customer].item()
        unvisited.remove(next_customer)
        current_node = next_customer
        
        # Extend route by sampling
        while unvisited:
            feasible_mask = torch.zeros(n_nodes)
            for j in unvisited:
                if current_capacity + demands[j] <= capacity:
                    feasible_mask[j] = 1.0
            
            if feasible_mask.sum() == 0:
                break
            
            # Include depot as option to end route
            feasible_mask[0] = 0.5  # Moderate probability to return
            
            probs = edge_probs[current_node] * feasible_mask
            probs = probs / probs.sum()
            
            next_node = torch.multinomial(probs, 1).item()
            
            if next_node == 0:
                # Chose to return to depot
                break
            
            route.append(next_node)
            current_capacity += demands[next_node].item()
            unvisited.remove(next_node)
            current_node = next_node
        
        if route:
            routes.append(route)
    
    return routes


def batched_decode_cvrp(adj_probs_batch: torch.Tensor,
                        coords_batch: torch.Tensor,
                        demands_batch: torch.Tensor,
                        capacity_batch: torch.Tensor,
                        decode_type: str = 'greedy') -> List[List[List[int]]]:
    """
    Decode CVRP solutions for a batch of instances
    
    Args:
        adj_probs_batch: (batch_size, n_nodes, n_nodes)
        coords_batch: (batch_size, n_nodes, 2)
        demands_batch: (batch_size, n_nodes)
        capacity_batch: (batch_size, 1) or (batch_size,)
        decode_type: 'greedy' or 'sampling'
    
    Returns:
        List of solutions, each solution is a list of routes
    """
    batch_size = adj_probs_batch.shape[0]
    solutions = []
    
    for b in range(batch_size):
        adj_probs = adj_probs_batch[b]
        coords = coords_batch[b]
        demands = demands_batch[b]
        capacity = capacity_batch[b].item() if capacity_batch.dim() > 1 else capacity_batch.item()
        
        if decode_type == 'greedy':
            routes = decode_cvrp_greedy(adj_probs, coords, demands, capacity)
        elif decode_type == 'sampling':
            routes = decode_cvrp_sampling(adj_probs, coords, demands, capacity)
        else:
            raise ValueError(f"Unknown decode type: {decode_type}")
        
        solutions.append(routes)
    
    return solutions


def cvrp_tours_to_adjacency(routes: List[List[int]], n_nodes: int) -> torch.Tensor:
    """Convert CVRP routes to adjacency matrix representation"""
    adj_matrix = torch.zeros(n_nodes, n_nodes)
    
    for route in routes:
        if len(route) == 0:
            continue
        
        # Depot to first customer
        adj_matrix[0, route[0]] = 1.0
        adj_matrix[route[0], 0] = 1.0  # Symmetric
        
        # Customer to customer
        for i in range(len(route) - 1):
            adj_matrix[route[i], route[i+1]] = 1.0
            adj_matrix[route[i+1], route[i]] = 1.0  # Symmetric
        
        # Last customer to depot
        adj_matrix[route[-1], 0] = 1.0
        adj_matrix[0, route[-1]] = 1.0  # Symmetric
    
    return adj_matrix


def apply_2opt_cvrp(routes: List[List[int]], 
                    coords: np.ndarray,
                    max_iterations: int = 100) -> List[List[int]]:
    """
    Apply 2-opt local search to improve CVRP routes
    2-opt is applied within each route independently
    """
    improved_routes = []
    
    for route in routes:
        if len(route) < 2:
            improved_routes.append(route.copy())
            continue
        
        # Add depot to create full route for 2-opt
        full_route = [0] + route + [0]
        improved = True
        iteration = 0
        
        while improved and iteration < max_iterations:
            improved = False
            best_delta = 0
            best_i, best_j = -1, -1
            
            # Try all 2-opt swaps
            for i in range(1, len(full_route) - 2):
                for j in range(i + 1, len(full_route) - 1):
                    # Current distance
                    current = (np.linalg.norm(coords[full_route[i-1]] - coords[full_route[i]]) +
                             np.linalg.norm(coords[full_route[j]] - coords[full_route[j+1]]))
                    
                    # New distance after swap
                    new = (np.linalg.norm(coords[full_route[i-1]] - coords[full_route[j]]) +
                          np.linalg.norm(coords[full_route[i]] - coords[full_route[j+1]]))
                    
                    delta = new - current
                    
                    if delta < best_delta:
                        best_delta = delta
                        best_i, best_j = i, j
            
            if best_delta < -1e-6:
                # Apply best 2-opt swap
                full_route[best_i:best_j+1] = full_route[best_i:best_j+1][::-1]
                improved = True
            
            iteration += 1
        
        # Remove depot from route
        improved_route = [node for node in full_route if node != 0]
        improved_routes.append(improved_route)
    
    return improved_routes


def merge_cvrp_routes(routes: List[List[int]], 
                     demands: np.ndarray,
                     capacity: float) -> List[List[int]]:
    """
    Try to merge routes to reduce the number of vehicles
    This is a post-processing step to improve solution quality
    """
    if len(routes) <= 1:
        return routes
    
    merged_routes = []
    used = set()
    
    for i, route_i in enumerate(routes):
        if i in used:
            continue
        
        current_route = route_i.copy()
        current_demand = sum(demands[c] for c in current_route)
        
        # Try to merge with other routes
        for j, route_j in enumerate(routes):
            if j <= i or j in used:
                continue
            
            route_j_demand = sum(demands[c] for c in route_j)
            
            if current_demand + route_j_demand <= capacity:
                # Merge routes
                current_route.extend(route_j)
                current_demand += route_j_demand
                used.add(j)
        
        merged_routes.append(current_route)
    
    return merged_routes