"""
Test EDISCO heatmap inference without tour extraction.
Produces edge probability heatmaps and computes soft expected cost.
"""

import sys
import os

# Add edisco to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'edisco'))

import torch
import numpy as np
import time
from argparse import Namespace


def create_dummy_tsp_file(filepath, n_instances=10, n_nodes=50):
    """Create a minimal dummy TSP file for model initialization."""
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    with open(filepath, 'w') as f:
        for _ in range(n_instances):
            coords = ' '.join([f'{np.random.rand():.6f}' for _ in range(n_nodes * 2)])
            tour = ' '.join([str(i) for i in range(n_nodes)])
            f.write(f"{coords} output {tour}\n")
    return filepath


def create_minimal_args(n_nodes=50):
    """Create minimal args for inference only."""
    dummy_file = f'/tmp/dummy_tsp{n_nodes}.txt'
    create_dummy_tsp_file(dummy_file, n_instances=10, n_nodes=n_nodes)

    return Namespace(
        storage_path='',
        training_split=dummy_file,
        validation_split=dummy_file,
        test_split=dummy_file,
        sparse_factor=0,
        n_layers=12,
        hidden_dim=256,
        node_dim=64,
        edge_dim=64,
        time_dim=128,
        coord_dim=2,
        aggregation='sum',
        diffusion_type='categorical',
        diffusion_schedule='linear',
        diffusion_steps=1000,
        continuous_time=True,
        equivariant=True,
        beta_min=0.1,
        beta_max=1.5,
        solver_type='pndm',
        solver_steps=50,
        time_schedule='linear',
        adaptive_mixing=True,
        deterministic_threshold=0.1,
        two_opt_iterations=0,
        validation_examples=1,
        batch_size=1,
        num_workers=0,
        use_activation_checkpoint=False,
    )


# ============ Soft Cost Functions ============

def compute_distance_matrix(coords):
    """
    Compute pairwise distance matrix.

    Args:
        coords: (batch, n_nodes, 2) or (n_nodes, 2)

    Returns:
        dist_matrix: (batch, n_nodes, n_nodes) or (n_nodes, n_nodes)
    """
    return torch.cdist(coords, coords)


def soft_tour_cost(edge_probs, dist_matrix):
    """
    Compute soft expected tour cost from edge probabilities.

    Args:
        edge_probs: (batch, n, n) - edge probabilities
        dist_matrix: (batch, n, n) - pairwise distances

    Returns:
        expected_cost: (batch,) - soft tour cost
    """
    # Symmetrize probabilities
    edge_probs = (edge_probs + edge_probs.transpose(-1, -2)) / 2

    # Expected cost = sum(prob * distance) / 2 (undirected)
    expected_cost = (edge_probs * dist_matrix).sum(dim=(-1, -2)) / 2

    return expected_cost


def soft_tour_cost_with_penalty(edge_probs, dist_matrix, degree_weight=0.1):
    """
    Soft tour cost with degree constraint penalty.

    For a valid TSP tour, each node should have exactly degree 2.
    """
    batch = edge_probs.shape[0] if edge_probs.dim() == 3 else 1

    # Symmetrize
    edge_probs = (edge_probs + edge_probs.transpose(-1, -2)) / 2

    # Expected cost
    expected_cost = (edge_probs * dist_matrix).sum(dim=(-1, -2)) / 2

    # Degree penalty: each node should have degree ≈ 2
    degrees = edge_probs.sum(dim=-1)  # (batch, n)
    degree_penalty = ((degrees - 2.0) ** 2).mean(dim=-1)

    total_cost = expected_cost + degree_weight * degree_penalty

    return total_cost, expected_cost, degree_penalty


def compute_degree_stats(edge_probs):
    """
    Compute node degree statistics from edge probabilities.
    """
    # Symmetrize
    edge_probs = (edge_probs + edge_probs.transpose(-1, -2)) / 2

    degrees = edge_probs.sum(dim=-1)  # (batch, n) or (n,)

    return {
        'mean_degree': degrees.mean().item(),
        'std_degree': degrees.std().item(),
        'min_degree': degrees.min().item(),
        'max_degree': degrees.max().item(),
    }


# ============ EDISCO Heatmap Inference ============

class EDISCOHeatmapSolver:
    """
    EDISCO solver that returns edge probability heatmaps.
    No tour extraction - just soft costs.
    """

    def __init__(self, checkpoint_path, device='cuda', n_nodes=50):
        self.device = device
        self.n_nodes = n_nodes

        # Load model
        self.model = self._load_model(checkpoint_path, n_nodes)
        self.model.eval()
        self.model.to(device)

        # Freeze
        for p in self.model.parameters():
            p.requires_grad = False

    def _load_model(self, ckpt_path, n_nodes):
        from pl_edisco_model import EDISCOModel
        args = create_minimal_args(n_nodes=n_nodes)
        model = EDISCOModel.load_from_checkpoint(
            ckpt_path, param_args=args, map_location=self.device
        )
        return model

    @torch.no_grad()
    def get_heatmap(self, coords, n_steps=50):
        """
        Get edge probability heatmap from EDISCO.

        Args:
            coords: (batch, n_nodes, 2) - node coordinates
            n_steps: number of diffusion steps

        Returns:
            edge_probs: (batch, n_nodes, n_nodes) - edge probabilities
        """
        from utils.ode_solvers import get_solver
        from models.continuous_score_network import ScoreWrapper

        if coords.dim() == 2:
            coords = coords.unsqueeze(0)

        coords = coords.to(self.device)
        batch_size, n_nodes, _ = coords.shape

        # Initialize from noise
        x_T = torch.randint(0, 2, (batch_size, n_nodes, n_nodes),
                           device=self.device, dtype=torch.float32)

        # Create score function wrapper
        score_fn = ScoreWrapper(self.model.score_network, coords, None)

        # Get solver
        solver = get_solver(self.model.solver_type, n_steps)

        # Run diffusion
        edge_probs = solver.sample(
            score_fn, x_T, device=self.device,
            schedule=self.model.time_schedule,
            adaptive_mixing=self.model.use_adaptive_mixing,
            deterministic_threshold=self.model.deterministic_threshold
        )

        # Symmetrize
        edge_probs = (edge_probs + edge_probs.transpose(-1, -2)) / 2

        return edge_probs

    @torch.no_grad()
    def get_soft_cost(self, coords, n_steps=50):
        """
        Get soft expected tour cost.

        Args:
            coords: (batch, n_nodes, 2)
            n_steps: number of diffusion steps

        Returns:
            soft_cost: (batch,) - expected tour cost
            edge_probs: (batch, n, n) - edge probability heatmap
        """
        coords = coords.to(self.device)

        # Get heatmap
        edge_probs = self.get_heatmap(coords, n_steps)

        # Compute distance matrix
        dist_matrix = compute_distance_matrix(coords)

        # Compute soft cost
        soft_cost = soft_tour_cost(edge_probs, dist_matrix)

        return soft_cost, edge_probs

    @torch.no_grad()
    def solve_batch(self, coords, n_steps=50):
        """
        Batch solve TSP instances and return soft costs + heatmaps.

        Args:
            coords: (batch, n_nodes, 2)

        Returns:
            dict with soft_costs, edge_probs, degree_stats
        """
        coords = coords.to(self.device)

        if coords.dim() == 2:
            coords = coords.unsqueeze(0)

        batch_size = coords.shape[0]

        # Get heatmaps (BATCHED!)
        edge_probs = self.get_heatmap(coords, n_steps)

        # Compute distances
        dist_matrix = compute_distance_matrix(coords)

        # Soft costs
        soft_costs = soft_tour_cost(edge_probs, dist_matrix)

        # With penalties
        total_costs, expected_costs, degree_penalties = soft_tour_cost_with_penalty(
            edge_probs, dist_matrix
        )

        # Degree stats
        degree_stats = compute_degree_stats(edge_probs)

        return {
            'soft_costs': soft_costs,
            'total_costs': total_costs,
            'expected_costs': expected_costs,
            'degree_penalties': degree_penalties,
            'edge_probs': edge_probs,
            'degree_stats': degree_stats,
        }


# ============ Visualization ============

def visualize_heatmap(coords, edge_probs, save_path=None):
    """Visualize edge probability heatmap."""
    import matplotlib.pyplot as plt

    coords_np = coords.cpu().numpy() if torch.is_tensor(coords) else coords
    probs_np = edge_probs.cpu().numpy() if torch.is_tensor(edge_probs) else edge_probs

    if coords_np.ndim == 3:
        coords_np = coords_np[0]
    if probs_np.ndim == 3:
        probs_np = probs_np[0]

    n_nodes = len(coords_np)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Edge probability heatmap
    im = axes[0].imshow(probs_np, cmap='hot', vmin=0, vmax=1)
    axes[0].set_title('Edge Probability Heatmap')
    axes[0].set_xlabel('Node j')
    axes[0].set_ylabel('Node i')
    plt.colorbar(im, ax=axes[0])

    # Right: Graph visualization with edge weights
    ax = axes[1]
    ax.scatter(coords_np[:, 0], coords_np[:, 1], c='blue', s=100, zorder=5)

    # Draw edges with opacity = probability
    for i in range(n_nodes):
        for j in range(i+1, n_nodes):
            prob = probs_np[i, j]
            if prob > 0.1:  # Only draw visible edges
                ax.plot([coords_np[i, 0], coords_np[j, 0]],
                       [coords_np[i, 1], coords_np[j, 1]],
                       'r-', alpha=prob, linewidth=2*prob)

    # Label nodes
    for i in range(n_nodes):
        ax.annotate(str(i), (coords_np[i, 0], coords_np[i, 1]),
                   xytext=(3, 3), textcoords='offset points', fontsize=8)

    ax.set_title('Graph with Edge Probabilities')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")

    plt.show()
    return fig


# ============ Tests ============

def test_single_heatmap(solver, n_nodes=20, device='cuda'):
    """Test single instance heatmap inference."""
    print(f"\n{'='*60}")
    print(f"Testing single heatmap inference: TSP-{n_nodes}")
    print(f"{'='*60}")

    # Generate random instance
    coords = torch.rand(1, n_nodes, 2, device=device)

    # Get heatmap and soft cost
    start_time = time.time()
    soft_cost, edge_probs = solver.get_soft_cost(coords, n_steps=50)
    elapsed = time.time() - start_time

    # Compute degree stats
    degree_stats = compute_degree_stats(edge_probs)

    print(f"  Soft cost: {soft_cost.item():.4f}")
    print(f"  Inference time: {elapsed:.3f}s")
    print(f"  Edge probs shape: {edge_probs.shape}")
    print(f"  Edge probs range: [{edge_probs.min().item():.3f}, {edge_probs.max().item():.3f}]")
    print(f"  Degree stats: mean={degree_stats['mean_degree']:.2f}, "
          f"std={degree_stats['std_degree']:.2f}, "
          f"range=[{degree_stats['min_degree']:.2f}, {degree_stats['max_degree']:.2f}]")

    return coords, edge_probs, soft_cost


def test_batched_heatmap(solver, batch_size=16, n_nodes=20, device='cuda'):
    """Test batched heatmap inference."""
    print(f"\n{'='*60}")
    print(f"Testing BATCHED heatmap inference: {batch_size} x TSP-{n_nodes}")
    print(f"{'='*60}")

    # Generate random instances
    coords = torch.rand(batch_size, n_nodes, 2, device=device)

    # Get batched results
    start_time = time.time()
    results = solver.solve_batch(coords, n_steps=50)
    elapsed = time.time() - start_time

    soft_costs = results['soft_costs']
    edge_probs = results['edge_probs']
    degree_stats = results['degree_stats']

    print(f"  Batch size: {batch_size}")
    print(f"  Total time: {elapsed:.3f}s")
    print(f"  Per instance: {elapsed/batch_size*1000:.1f}ms")
    print(f"  Soft costs: mean={soft_costs.mean().item():.4f}, "
          f"std={soft_costs.std().item():.4f}")
    print(f"  Edge probs shape: {edge_probs.shape}")
    print(f"  Degree stats: mean={degree_stats['mean_degree']:.2f}")

    return coords, edge_probs, soft_costs


def test_different_steps(solver, n_nodes=20, device='cuda'):
    """Test different numbers of diffusion steps."""
    print(f"\n{'='*60}")
    print(f"Testing different solver steps: TSP-{n_nodes}")
    print(f"{'='*60}")

    # Use same instance
    torch.manual_seed(42)
    coords = torch.rand(1, n_nodes, 2, device=device)

    step_counts = [10, 20, 50, 100]

    for n_steps in step_counts:
        torch.manual_seed(123)  # Reset for fair comparison

        start_time = time.time()
        soft_cost, edge_probs = solver.get_soft_cost(coords, n_steps=n_steps)
        elapsed = time.time() - start_time

        degree_stats = compute_degree_stats(edge_probs)

        print(f"  Steps={n_steps:3d}: cost={soft_cost.item():.4f}, "
              f"time={elapsed:.3f}s, degree={degree_stats['mean_degree']:.2f}")


def test_scaling(solver, device='cuda'):
    """Test scaling with batch size."""
    print(f"\n{'='*60}")
    print(f"Testing batch scaling")
    print(f"{'='*60}")

    n_nodes = 20
    batch_sizes = [1, 4, 16, 64]

    for bs in batch_sizes:
        coords = torch.rand(bs, n_nodes, 2, device=device)

        # Warmup
        if bs == batch_sizes[0]:
            _ = solver.get_heatmap(coords[:1], n_steps=10)

        torch.cuda.synchronize() if device == 'cuda' else None
        start_time = time.time()

        results = solver.solve_batch(coords, n_steps=50)

        torch.cuda.synchronize() if device == 'cuda' else None
        elapsed = time.time() - start_time

        print(f"  Batch={bs:3d}: total={elapsed:.3f}s, "
              f"per_instance={elapsed/bs*1000:.1f}ms, "
              f"throughput={bs/elapsed:.1f} inst/s")


def main():
    """Main test function."""
    print("="*60)
    print("EDISCO Heatmap Inference Test (No Tour Extraction)")
    print("="*60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Find checkpoint
    ckpt_paths = [
        'pretrained/edisco_tsp20.ckpt',
        'pretrained/edisco_tsp50.ckpt',
    ]

    ckpt_path = None
    for path in ckpt_paths:
        if os.path.exists(path):
            ckpt_path = path
            break

    if ckpt_path is None:
        print("ERROR: No checkpoint found!")
        return

    print(f"Checkpoint: {ckpt_path}")

    # Determine n_nodes
    n_nodes = 20 if 'tsp20' in ckpt_path else 50
    print(f"Problem size: TSP-{n_nodes}")

    # Load solver
    print("\nLoading EDISCO solver...")
    try:
        solver = EDISCOHeatmapSolver(ckpt_path, device=device, n_nodes=n_nodes)
        print("Solver loaded successfully!")
    except Exception as e:
        print(f"Error loading solver: {e}")
        import traceback
        traceback.print_exc()
        return

    # Run tests
    try:
        # Test 1: Single heatmap
        coords, edge_probs, soft_cost = test_single_heatmap(
            solver, n_nodes=n_nodes, device=device
        )

        # Test 2: Batched heatmap
        test_batched_heatmap(solver, batch_size=16, n_nodes=n_nodes, device=device)

        # Test 3: Different steps
        test_different_steps(solver, n_nodes=n_nodes, device=device)

        # Test 4: Scaling
        test_scaling(solver, device=device)

        # Visualize (optional - comment out if no display)
        try:
            print(f"\n{'='*60}")
            print("Visualization")
            print("="*60)
            visualize_heatmap(coords, edge_probs, save_path='/tmp/edisco_heatmap.png')
        except Exception as e:
            print(f"  Skipping visualization: {e}")

        print("\n" + "="*60)
        print("All tests passed!")
        print("="*60)

    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
