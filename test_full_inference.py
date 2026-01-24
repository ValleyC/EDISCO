"""
Full EDISCO inference test with pure Python merge_tours (no Cython).
Verifies the inference pipeline produces valid tours.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'edisco'))

import torch
import numpy as np
import time
from argparse import Namespace


# ============ Pure Python merge_tours ============

def numpy_merge(points, adj_mat):
    """
    Pure Python/NumPy implementation of tour merging.
    Greedy edge insertion based on (probability / distance) ratio.
    """
    n = adj_mat.shape[0]
    dists = np.linalg.norm(points[:, None] - points, axis=-1)
    dists[dists == 0] = 1e-10  # Avoid division by zero

    # Score = probability / distance (higher is better)
    scores = adj_mat / dists

    # Track components (each node starts as its own component)
    # component[i] = (endpoint1, endpoint2) of the path containing node i
    parent = list(range(n))  # Union-find parent
    rank = [0] * n
    endpoints = {i: [i, i] for i in range(n)}  # Each component's endpoints
    degree = [0] * n  # Degree of each node in current solution

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        px, py = find(x), find(y)
        if px == py:
            return False
        if rank[px] < rank[py]:
            px, py = py, px
        parent[py] = px
        if rank[px] == rank[py]:
            rank[px] += 1
        # Merge endpoints
        ex, ey = endpoints[px], endpoints[py]
        # New endpoints are the ones not connected
        new_endpoints = []
        for e in ex + ey:
            if e != x and e != y:
                new_endpoints.append(e)
        if len(new_endpoints) == 0:
            new_endpoints = [x, y]
        elif len(new_endpoints) == 1:
            new_endpoints = new_endpoints + new_endpoints
        endpoints[px] = new_endpoints[:2]
        return True

    real_adj = np.zeros((n, n))

    # Sort edges by score (descending)
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((scores[i, j], i, j))
    edges.sort(reverse=True)

    edge_count = 0
    for score, i, j in edges:
        if edge_count >= n:
            break

        # Check if adding edge (i, j) is valid:
        # 1. Both nodes must have degree < 2
        # 2. Must not create a cycle (unless it completes the tour)

        if degree[i] >= 2 or degree[j] >= 2:
            continue

        pi, pj = find(i), find(j)

        if pi == pj:
            # Same component - only add if this completes the tour
            if edge_count == n - 1:
                real_adj[i, j] = 1
                real_adj[j, i] = 1
                edge_count += 1
            continue

        # Check if i and j are endpoints of their components
        ei, ej = endpoints[pi], endpoints[pj]
        if i not in ei or j not in ej:
            continue

        # Add edge
        real_adj[i, j] = 1
        real_adj[j, i] = 1
        degree[i] += 1
        degree[j] += 1
        union(i, j)
        edge_count += 1

    return real_adj, edge_count


def extract_tour_from_adj(adj_mat):
    """Extract tour from adjacency matrix."""
    n = adj_mat.shape[0]
    tour = [0]
    visited = {0}

    while len(tour) < n:
        current = tour[-1]
        neighbors = np.where(adj_mat[current] > 0)[0]

        next_node = None
        for nb in neighbors:
            if nb not in visited:
                next_node = nb
                break

        if next_node is None:
            # No unvisited neighbor, try to find any unvisited node
            for i in range(n):
                if i not in visited:
                    next_node = i
                    break

        if next_node is None:
            break

        tour.append(next_node)
        visited.add(next_node)

    # Close the tour
    tour.append(tour[0])
    return tour


def merge_tours_python(adj_probs, coords):
    """
    Pure Python merge_tours implementation.

    Args:
        adj_probs: (n, n) edge probability matrix
        coords: (n, 2) node coordinates

    Returns:
        tour: list of node indices
    """
    # Symmetrize
    adj_probs = (adj_probs + adj_probs.T) / 2

    # Merge
    real_adj, _ = numpy_merge(coords, adj_probs)

    # Extract tour
    tour = extract_tour_from_adj(real_adj)

    return tour


def compute_tour_cost(coords, tour):
    """Compute tour cost."""
    total = 0.0
    for i in range(len(tour) - 1):
        total += np.linalg.norm(coords[tour[i]] - coords[tour[i + 1]])
    return total


# ============ Model Loading ============

def create_dummy_tsp_file(filepath, n_instances=10, n_nodes=50):
    """Create a minimal dummy TSP file for model initialization."""
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    with open(filepath, 'w') as f:
        for _ in range(n_instances):
            coords = ' '.join([f'{np.random.rand():.6f}' for _ in range(n_nodes * 2)])
            tour = ' '.join([str(i) for i in range(n_nodes)])
            f.write(f"{coords} output {tour}\n")
    return filepath


def load_edisco_model(ckpt_path, device='cuda', n_nodes=50):
    """
    Load EDISCO model directly from checkpoint.
    Uses EGNNEncoder (official class) with dense_only=True.
    """
    from models.egnn_encoder import EGNNEncoder
    from utils.continuous_diffusion import ContinuousTimeCategoricalDiffusionDense
    from utils.ode_solvers import get_solver
    from models.continuous_score_network import ScoreWrapper

    # Load checkpoint
    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint['state_dict']

    # Extract config from checkpoint keys
    # Check layer count
    layer_keys = [k for k in state_dict.keys() if 'model.layers.' in k]
    layer_nums = set(int(k.split('.')[2]) for k in layer_keys if k.split('.')[2].isdigit())
    n_layers = max(layer_nums) + 1 if layer_nums else 12

    # Check hidden_dim from a known layer
    hidden_dim = state_dict.get('model.time_embed.0.weight', torch.zeros(256, 128)).shape[0]
    node_dim = state_dict.get('model.node_embed', torch.zeros(1, 1, 64)).shape[-1]
    edge_dim = state_dict.get('model.edge_embed.weight', torch.zeros(64, 1)).shape[0]
    time_dim = state_dict.get('model.time_embed.2.weight', torch.zeros(128, 256)).shape[0]

    print(f"  Detected config: n_layers={n_layers}, hidden_dim={hidden_dim}, "
          f"node_dim={node_dim}, edge_dim={edge_dim}, time_dim={time_dim}")

    # Create encoder using official EGNNEncoder class with dense_only=True
    # This matches what pl_meta_model.py does when sparse_factor=0
    encoder = EGNNEncoder(
        n_layers=n_layers,
        hidden_dim=hidden_dim,
        node_dim=node_dim,
        edge_dim=edge_dim,
        time_dim=time_dim,
        coord_dim=2,
        out_channels=2,
        sparse=False,
        dense_only=True,  # Critical: matches official training config
        use_activation_checkpoint=False,
        coord_update_alpha=0.1,
        weight_temp=10.0,
    )

    # Load weights - use 'model.*' keys directly (encoder is the model)
    encoder_state = {}
    for k, v in state_dict.items():
        if k.startswith('model.'):
            new_key = k[6:]  # Remove 'model.' prefix
            encoder_state[new_key] = v

    missing, unexpected = encoder.load_state_dict(encoder_state, strict=False)
    print(f"  Loaded weights: {len(encoder_state)} keys")
    if missing:
        print(f"  Missing keys: {len(missing)} - {missing[:5]}...")
    if unexpected:
        print(f"  Unexpected keys: {len(unexpected)} - {unexpected[:5]}...")

    encoder.eval()
    encoder.to(device)

    # Freeze
    for p in encoder.parameters():
        p.requires_grad = False

    # Create diffusion
    diffusion = ContinuousTimeCategoricalDiffusionDense(
        beta_min=0.1,
        beta_max=1.5,
        num_classes=2
    )

    config = {
        'n_layers': n_layers,
        'hidden_dim': hidden_dim,
        'beta_min': 0.1,
        'beta_max': 1.5,
        'solver_type': 'pndm',
        'solver_steps': 50,
        'time_schedule': 'linear',
        'adaptive_mixing': True,
        'deterministic_threshold': 0.1,
    }

    return encoder, diffusion, config


def run_diffusion(encoder, coords, config, device='cuda', n_steps=50, return_probs=True):
    """
    Run diffusion to get edge probabilities using official ODE solver.

    Args:
        encoder: The EGNN encoder model
        coords: Node coordinates (batch, n_nodes, 2) or (n_nodes, 2)
        config: Configuration dict with solver settings
        device: Device to run on
        n_steps: Number of diffusion steps
        return_probs: If True, return softmax probabilities; if False, return binary

    Returns:
        edge_probs: Edge probability matrix (batch, n_nodes, n_nodes)
    """
    from utils.ode_solvers import get_solver
    from models.continuous_score_network import ScoreWrapper

    if coords.dim() == 2:
        coords = coords.unsqueeze(0)

    coords = coords.to(device)
    batch_size, n_nodes, _ = coords.shape

    # Initialize from noise (uniform random binary)
    x_T = torch.randint(0, 2, (batch_size, n_nodes, n_nodes),
                       device=device, dtype=torch.float32)

    # Use official ScoreWrapper (same as pl_edisco_model.py)
    score_fn = ScoreWrapper(encoder, coords, edge_index=None)

    # Get beta parameters
    beta_min = config.get('beta_min', 0.1)
    beta_max = config.get('beta_max', 1.5)

    # Use official solver with correct parameters
    solver = get_solver(
        config['solver_type'],
        n_steps,
        beta_min=beta_min,
        beta_max=beta_max
    )

    # Run official ODE solver (includes multi-step smoothing for PNDM)
    edge_probs = solver.sample(
        score_fn, x_T, device=device,
        schedule=config['time_schedule'],
        adaptive_mixing=config['adaptive_mixing'],
        deterministic_threshold=config['deterministic_threshold']
    )

    # Official solver returns binary (argmax) - this is what merge_tours expects
    return edge_probs


# ============ Tests ============

def test_single_instance(encoder, config, n_nodes=20, device='cuda'):
    """Test single instance with full pipeline."""
    print(f"\n{'='*60}")
    print(f"Testing single instance: TSP-{n_nodes}")
    print(f"{'='*60}")

    # Generate random instance
    torch.manual_seed(42)
    coords = torch.rand(1, n_nodes, 2, device=device)
    coords_np = coords[0].cpu().numpy()

    # Run diffusion (returns binary adjacency matrix from official solver)
    start_time = time.time()
    with torch.no_grad():
        adj_matrix = run_diffusion(encoder, coords, config, device, n_steps=50)
    diffusion_time = time.time() - start_time

    # Get adjacency matrix (binary: 0 or 1)
    adj_np = adj_matrix[0].cpu().numpy()
    adj_np = (adj_np + adj_np.T) / 2  # Symmetrize

    # Count edges (binary)
    n_edges = (adj_np > 0.5).sum() / 2  # Divide by 2 for undirected
    mean_degree = (adj_np > 0.5).sum(axis=1).mean()

    print(f"  Adjacency range: [{adj_np.min():.3f}, {adj_np.max():.3f}]")
    print(f"  Number of edges: {n_edges:.0f} (expected: {n_nodes})")
    print(f"  Mean degree: {mean_degree:.2f} (expected: 2.0)")

    # Merge tours using greedy algorithm
    start_time = time.time()
    tour = merge_tours_python(adj_np, coords_np)
    merge_time = time.time() - start_time

    # Remove closing node if present
    if len(tour) == n_nodes + 1 and tour[0] == tour[-1]:
        tour = tour[:-1]

    # Compute cost
    tour_cost = compute_tour_cost(coords_np, tour + [tour[0]])

    print(f"  Tour: {tour[:10]}... (first 10)")
    print(f"  Tour length: {len(tour)} nodes")
    print(f"  Tour cost: {tour_cost:.4f}")
    print(f"  Diffusion time: {diffusion_time:.3f}s")
    print(f"  Merge time: {merge_time:.3f}s")

    # Validate tour
    if len(tour) == n_nodes and len(set(tour)) == n_nodes:
        print(f"  Tour valid: Yes")
    else:
        print(f"  Tour valid: No (length={len(tour)}, unique={len(set(tour))})")

    return tour, tour_cost


def test_batch(encoder, config, batch_size=8, n_nodes=20, device='cuda'):
    """Test batch of instances."""
    print(f"\n{'='*60}")
    print(f"Testing batch: {batch_size} x TSP-{n_nodes}")
    print(f"{'='*60}")

    # Generate random instances
    coords = torch.rand(batch_size, n_nodes, 2, device=device)

    # Run diffusion (batched) - returns binary adjacency matrix
    start_time = time.time()
    with torch.no_grad():
        adj_matrix = run_diffusion(encoder, coords, config, device, n_steps=50)
    diffusion_time = time.time() - start_time

    coords_np = coords.cpu().numpy()
    adj_np = adj_matrix.cpu().numpy()

    # Extract tours (sequential - merge_tours is not batched)
    tours = []
    costs = []
    start_time = time.time()
    for i in range(batch_size):
        adj_i = (adj_np[i] + adj_np[i].T) / 2
        tour = merge_tours_python(adj_i, coords_np[i])
        if len(tour) == n_nodes + 1 and tour[0] == tour[-1]:
            tour = tour[:-1]
        tours.append(tour)
        costs.append(compute_tour_cost(coords_np[i], tour + [tour[0]]))
    merge_time = time.time() - start_time

    print(f"  Batch size: {batch_size}")
    print(f"  Diffusion time: {diffusion_time:.3f}s ({diffusion_time/batch_size*1000:.1f}ms/inst)")
    print(f"  Merge time: {merge_time:.3f}s ({merge_time/batch_size*1000:.1f}ms/inst)")
    print(f"  Tour costs: mean={np.mean(costs):.4f}, std={np.std(costs):.4f}")
    print(f"  Cost range: [{min(costs):.4f}, {max(costs):.4f}]")

    # Validate
    valid = all(len(t) == n_nodes and len(set(t)) == n_nodes for t in tours)
    print(f"  All tours valid: {valid}")

    return tours, costs


def test_compare_soft_vs_real(encoder, config, n_nodes=20, device='cuda', n_instances=100):
    """Compare soft cost vs real tour cost."""
    print(f"\n{'='*60}")
    print(f"Testing {n_instances} x TSP-{n_nodes} instances")
    print(f"{'='*60}")

    # Coordinates in [0, 1] - standard TSP benchmark format
    torch.manual_seed(0)  # For reproducibility
    coords = torch.rand(n_instances, n_nodes, 2, device=device)

    print(f"  Coordinates range: [{coords.min().item():.4f}, {coords.max().item():.4f}]")

    # Process in batches to avoid OOM
    batch_size = 16
    n_batches = (n_instances + batch_size - 1) // batch_size

    real_costs = []
    degrees = []
    n_edges_list = []

    start_time = time.time()
    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, n_instances)
        batch_coords = coords[start_idx:end_idx]

        with torch.no_grad():
            adj_matrix = run_diffusion(encoder, batch_coords, config, device, n_steps=50)

        coords_np = batch_coords.cpu().numpy()
        adj_np = adj_matrix.cpu().numpy()

        for i in range(len(coords_np)):
            adj_i = (adj_np[i] + adj_np[i].T) / 2

            # Degree stats (binary)
            mean_deg = (adj_i > 0.5).sum(axis=1).mean()
            degrees.append(mean_deg)
            n_edges_list.append((adj_i > 0.5).sum() / 2)

            # Real cost from merge_tours
            tour = merge_tours_python(adj_i, coords_np[i])
            if len(tour) == n_nodes + 1:
                tour = tour[:-1]
            real_cost = compute_tour_cost(coords_np[i], tour + [tour[0]])
            real_costs.append(real_cost)

        print(f"  Batch {batch_idx+1}/{n_batches}: processed {end_idx}/{n_instances} instances")

    total_time = time.time() - start_time

    print(f"\n  === Results on {n_instances} instances ===")
    print(f"  Coordinates: [0, 1] x [0, 1]")
    print(f"  Mean edges: {np.mean(n_edges_list):.1f} (expected: {n_nodes})")
    print(f"  Mean degree: {np.mean(degrees):.2f} (expected: 2.0)")
    print(f"  Tour costs: mean={np.mean(real_costs):.4f}, std={np.std(real_costs):.4f}")
    print(f"  Cost range: [{min(real_costs):.4f}, {max(real_costs):.4f}]")
    print(f"  Total time: {total_time:.2f}s ({total_time/n_instances*1000:.1f}ms/inst)")

    return real_costs


def main():
    print("="*60)
    print("EDISCO Full Inference Test (Official Solver)")
    print("="*60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Find checkpoint
    ckpt_path = 'pretrained/edisco_tsp20.ckpt'
    if not os.path.exists(ckpt_path):
        print(f"ERROR: Checkpoint not found at {ckpt_path}")
        return

    print(f"Checkpoint: {ckpt_path}")
    n_nodes = 20

    # Load model
    print("\nLoading model...")
    try:
        encoder, diffusion, config = load_edisco_model(ckpt_path, device, n_nodes)
        print("Model loaded successfully!")
        print(f"  Using solver: {config['solver_type']} with {config['solver_steps']} steps")
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return

    # Run tests
    try:
        test_single_instance(encoder, config, n_nodes=n_nodes, device=device)
        test_batch(encoder, config, batch_size=8, n_nodes=n_nodes, device=device)
        test_compare_soft_vs_real(encoder, config, n_nodes=n_nodes, device=device, n_instances=100)

        print("\n" + "="*60)
        print("All tests completed!")
        print("Expected TSP-20 cost: ~3.84 (matching official inference)")
        print("="*60)

    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
