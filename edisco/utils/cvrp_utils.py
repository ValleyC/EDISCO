"""CVRP utilities: capacity-feasible decoding, evaluation and optional post-processing."""

from typing import Dict, List

import numpy as np
import torch


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


def _expand_cvrp_edges(n_nodes, candidates, demand, capacity):
    """Feasibility projection of ranked candidate edges onto CVRP routes.

    Node 0 is the depot. `candidates` yields undirected pairs (i, j), i < j,
    in rank order. Customers keep degree two and the depot accepts one edge
    pair per route. A union-find over customers tracks the partial routes and
    their accumulated demand:

    - a customer-customer edge is accepted when both endpoints have degree
      below two, lie in different partial routes, and the merged demand does
      not exceed the capacity;
    - a candidate rejected only because of capacity closes every involved
      partial route that already has its first depot edge, by adding the
      matching depot-return edge at its free end, before the next candidate;
    - a depot edge is accepted when the customer has degree below two and its
      partial route has fewer than two depot edges.

    Partial routes left open after the last candidate are closed with depot
    edges. Every check reads node indices, demands and the capacity only.
    """
    parent = list(range(n_nodes))
    load = [float(d) for d in demand]
    degree = [0] * n_nodes
    depot_edges = [0] * n_nodes           # per partial route (indexed by root)
    ends = [(v, v) for v in range(n_nodes)]  # path endpoints per root
    neighbours = [[] for _ in range(n_nodes)]

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def close(root):
        for end in ends[root]:
            while degree[end] < 2 and depot_edges[root] < 2:
                degree[end] += 1
                depot_edges[root] += 1

    for i, j in candidates:
        i, j = int(i), int(j)
        if i == 0:
            root = find(j)
            if degree[j] < 2 and depot_edges[root] < 2:
                degree[j] += 1
                depot_edges[root] += 1
            continue
        if degree[i] >= 2 or degree[j] >= 2:
            continue
        ri, rj = find(i), find(j)
        if ri == rj or depot_edges[ri] + depot_edges[rj] > 2:
            continue
        if load[ri] + load[rj] > capacity:
            for root in (ri, rj):
                if depot_edges[root] == 1:
                    close(root)
            continue
        # merge the two partial routes through their endpoints i and j
        other_i = ends[ri][1] if ends[ri][0] == i else ends[ri][0]
        other_j = ends[rj][1] if ends[rj][0] == j else ends[rj][0]
        degree[i] += 1
        degree[j] += 1
        neighbours[i].append(j)
        neighbours[j].append(i)
        parent[rj] = ri
        load[ri] += load[rj]
        depot_edges[ri] += depot_edges[rj]
        ends[ri] = (other_i, other_j)

    routes = []
    for customer in range(1, n_nodes):
        if find(customer) != customer:
            continue
        close(customer)
        # walk the path from one endpoint to the other
        start, previous = ends[customer][0], None
        route = [start]
        while True:
            following = [v for v in neighbours[route[-1]] if v != previous]
            if not following:
                break
            previous = route[-1]
            route.append(following[0])
        routes.append(route)
    routes.sort(key=lambda route: min(route))
    return routes


def _decode_cvrp_ranked(adj_probs, coords, demands, capacity,
                        symmetrize=True, distance_aware=False, edge_index=None):
    """Rank candidate edges and project them onto capacity-feasible routes.

    Dense input: `adj_probs` is (n, n). Sparse input: `adj_probs` is (E,) with
    `edge_index` (2, E); pairs outside the candidate graph are not ranked.
    Ties are broken by node indices.
    """
    capacity = float(capacity)
    if not np.isfinite(capacity) or capacity <= 0:
        raise ValueError("capacity must be finite and positive")
    if not torch.isfinite(demands).all() or (demands < 0).any():
        raise ValueError("demands must be finite and nonnegative")
    if (demands[1:] > capacity).any():
        raise ValueError("every customer demand must be at most vehicle capacity")
    if not torch.isfinite(adj_probs).all() or ((adj_probs < 0) | (adj_probs > 1)).any():
        raise ValueError("edge probabilities must be finite and in [0, 1]")
    n = len(coords)
    points = coords.detach().cpu().double().numpy()
    probs = adj_probs.detach().cpu().double().numpy()
    if edge_index is None:
        i, j = np.triu_indices(n, k=1)
        scores = (probs[i, j] + probs[j, i]) / 2 if symmetrize else probs[i, j]
    else:
        index = edge_index.detach().cpu().numpy() if torch.is_tensor(edge_index) else np.asarray(edge_index)
        src, dst = index
        mask = src != dst
        lo, hi = np.minimum(src[mask], dst[mask]), np.maximum(src[mask], dst[mask])
        keys, inverse = np.unique(lo * n + hi, return_inverse=True)
        scores = np.bincount(inverse, weights=probs[mask], minlength=len(keys)) / 2
        i, j = keys // n, keys % n
    if distance_aware:
        scores = scores / (np.linalg.norm(points[i] - points[j], axis=-1) + 1e-8)
    order = np.lexsort((j, i, -scores))
    return _expand_cvrp_edges(n, zip(i[order], j[order]), demands.detach().cpu().numpy(), capacity)


def decode_cvrp_greedy(adj_probs, coords, demands, capacity, symmetrize=True, edge_index=None):
    """Capacity-feasible edge expansion ranked by edge probability only."""
    return _decode_cvrp_ranked(adj_probs, coords, demands, capacity, symmetrize, False, edge_index)


def decode_cvrp_nee(adj_probs, coords, demands, capacity, symmetrize=True, edge_index=None):
    """Capacity-feasible NEE ranked by probability divided by distance plus epsilon."""
    return _decode_cvrp_ranked(adj_probs, coords, demands, capacity, symmetrize, True, edge_index)


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
        decode_type: 'greedy' or 'nee'

    Returns:
        List of solutions, each solution is a list of routes
    """
    batch_size = adj_probs_batch.shape[0]
    solutions = []

    for b in range(batch_size):
        adj_probs = adj_probs_batch[b]
        coords = coords_batch[b]
        demands = demands_batch[b]
        capacity = capacity_batch.reshape(-1)[b].item()

        if decode_type == 'greedy':
            routes = decode_cvrp_greedy(adj_probs, coords, demands, capacity)
        elif decode_type == 'nee':
            routes = decode_cvrp_nee(adj_probs, coords, demands, capacity)
        else:
            raise ValueError(f"Unknown decode type: {decode_type}")

        solutions.append(routes)

    return solutions


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
