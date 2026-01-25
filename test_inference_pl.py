"""
Custom EDISCO inference using PyTorch Lightning's load_from_checkpoint.
This ensures identical configuration to official inference.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'edisco'))

import torch
import numpy as np
import time
from argparse import Namespace


# ============ Pure Python merge_tours ============

def merge_tours_python(adj_mat, coords):
    """
    Pure Python merge_tours using official numpy_merge from tsp_utils.
    No Cython dependency - uses the same algorithm as cython_merge.

    Args:
        adj_mat: Symmetrized adjacency matrix (n, n) - should be adj + adj.T
        coords: Node coordinates (n, 2)

    Returns:
        tour: List of node indices forming a valid TSP tour
    """
    from utils.tsp_utils import numpy_merge

    # Use the official numpy_merge algorithm (same logic as cython_merge)
    real_adj_mat, merge_iterations = numpy_merge(coords, adj_mat)

    # Extract tour from adjacency matrix (same as official merge_tours)
    n = adj_mat.shape[0]
    tour = [0]
    while len(tour) < n + 1:
        neighbors = np.nonzero(real_adj_mat[tour[-1]])[0]
        if len(tour) > 1:
            neighbors = neighbors[neighbors != tour[-2]]
        if len(neighbors) == 0:
            break
        tour.append(neighbors.max())

    return tour


def compute_tour_cost(coords, tour):
    """Compute tour cost."""
    total = 0.0
    for i in range(len(tour) - 1):
        total += np.linalg.norm(coords[tour[i]] - coords[tour[i + 1]])
    return total


# ============ Model Loading ============

def create_dummy_args(n_nodes=20):
    """Create minimal args for model loading."""
    return Namespace(
        # Task
        task='tsp',

        # Data paths (dummy)
        storage_path='.',
        training_split='dummy_train.txt',
        validation_split='dummy_val.txt',
        test_split='dummy_test.txt',
        validation_examples=1,

        # Training params
        batch_size=1,
        num_epochs=1,
        learning_rate=1e-4,
        weight_decay=0.0,
        lr_scheduler='constant',
        num_workers=0,
        fp16=False,
        use_activation_checkpoint=False,

        # Diffusion
        diffusion_type='categorical',
        diffusion_schedule='linear',
        diffusion_steps=1000,
        continuous_time=True,
        equivariant=True,
        beta_min=0.1,
        beta_max=1.5,

        # Inference
        inference_diffusion_steps=1000,
        inference_schedule='linear',
        inference_trick='ddim',
        sequential_sampling=1,
        parallel_sampling=1,

        # Solver
        solver_type='pndm',
        solver_steps=50,
        time_schedule='linear',
        adaptive_mixing=True,
        deterministic_threshold=0.1,

        # Architecture
        n_layers=12,
        hidden_dim=256,
        node_dim=64,
        edge_dim=64,
        time_dim=128,
        coord_dim=2,
        coord_update_alpha=0.1,
        weight_temp=10.0,

        # Graph
        sparse_factor=0,  # Dense mode
        aggregation='sum',
        two_opt_iterations=0,
        save_numpy_heatmap=False,

        # CVRP (not used but required)
        merge_routes=False,

        # Logging (not used)
        project_name='edisco_test',
        wandb_entity=None,
        wandb_logger_name=None,
        resume_id=None,
        ckpt_path=None,
        resume_weight_only=False,

        # Modes
        do_train=False,
        do_test=True,
        do_valid_only=False,
        compare_solvers=False,
        test_equivariance=False,
        disable_continuous_time=False,
        disable_equivariance=False,
        force_dense_only=True,
        disable_optimizations=False,

        # Derived
        dense_only=True,
    )


def create_dummy_tsp_file(filepath, n_instances=10, n_nodes=20):
    """Create a minimal dummy TSP file."""
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    with open(filepath, 'w') as f:
        for _ in range(n_instances):
            coords = ' '.join([f'{np.random.rand():.6f}' for _ in range(n_nodes * 2)])
            tour = ' '.join([str(i) for i in range(n_nodes)])
            f.write(f"{coords} output {tour}\n")
    return filepath


def load_model_from_checkpoint(ckpt_path, device='cuda', n_nodes=20):
    """
    Load EDISCO model using PyTorch Lightning's load_from_checkpoint.
    This ensures identical configuration to official inference.
    """
    from pl_edisco_model import EDISCOModel

    # Create dummy data files for model initialization
    dummy_dir = os.path.join(os.path.dirname(ckpt_path), 'dummy_data')
    os.makedirs(dummy_dir, exist_ok=True)

    dummy_train = os.path.join(dummy_dir, 'train.txt')
    dummy_val = os.path.join(dummy_dir, 'val.txt')
    dummy_test = os.path.join(dummy_dir, 'test.txt')

    for f in [dummy_train, dummy_val, dummy_test]:
        if not os.path.exists(f):
            create_dummy_tsp_file(f, n_instances=10, n_nodes=n_nodes)

    # Create args
    args = create_dummy_args(n_nodes)
    args.storage_path = dummy_dir
    args.training_split = 'train.txt'
    args.validation_split = 'val.txt'
    args.test_split = 'test.txt'

    print(f"Loading model from checkpoint...")
    print(f"  Checkpoint: {ckpt_path}")
    print(f"  Args: sparse_factor={args.sparse_factor}, dense_only={args.dense_only}")

    # Load model using PL's load_from_checkpoint
    model = EDISCOModel.load_from_checkpoint(ckpt_path, param_args=args)
    model.eval()
    model.to(device)

    # Freeze all parameters
    for p in model.parameters():
        p.requires_grad = False

    print(f"  Model loaded: {model.__class__.__name__}")
    print(f"  Encoder: {model.model.__class__.__name__}")
    print(f"  Solver: {model.solver_type} ({model.solver_steps} steps)")
    print(f"  Dense mode: {model.dense_only}")

    return model


# ============ Inference ============

def run_inference_with_model(model, coords, device='cuda'):
    """
    Run inference using the loaded model's sample_with_solver method.
    This is the exact same method used in official inference.
    """
    if coords.dim() == 2:
        coords = coords.unsqueeze(0)
    coords = coords.to(device)

    with torch.no_grad():
        tours, adj_probs = model.sample_with_solver(coords, device=device)

    return tours, adj_probs


def run_custom_diffusion(model, coords, device='cuda', n_steps=50, debug=False):
    """
    Run diffusion manually using model components.
    For debugging - should match official inference.
    """
    from utils.ode_solvers import get_solver
    from models.continuous_score_network import ScoreWrapper

    if coords.dim() == 2:
        coords = coords.unsqueeze(0)
    coords = coords.to(device)

    batch_size, n_nodes, _ = coords.shape

    # Initialize from noise
    x_T = torch.randint(0, 2, (batch_size, n_nodes, n_nodes),
                       device=device, dtype=torch.float32)

    if debug:
        print(f"    DEBUG: x_T shape={x_T.shape}, sum={x_T.sum().item():.0f}")
        print(f"    DEBUG: coords shape={coords.shape}")
        print(f"    DEBUG: score_network type={type(model.score_network).__name__}")
        print(f"    DEBUG: solver_type={model.solver_type}, n_steps={n_steps}")
        print(f"    DEBUG: beta_min={model.args.beta_min}, beta_max={model.args.beta_max}")
        print(f"    DEBUG: time_schedule={model.time_schedule}")
        print(f"    DEBUG: adaptive_mixing={model.use_adaptive_mixing}")
        print(f"    DEBUG: deterministic_threshold={model.deterministic_threshold}")

    # Use model's score_network and solver settings
    score_fn = ScoreWrapper(model.score_network, coords, None)

    # Get solver with model's configuration
    solver = get_solver(
        model.solver_type,
        n_steps,
        beta_min=model.args.beta_min,
        beta_max=model.args.beta_max
    )

    # Run solver
    x0_pred = solver.sample(
        score_fn, x_T, device=device,
        schedule=model.time_schedule,
        adaptive_mixing=model.use_adaptive_mixing,
        deterministic_threshold=model.deterministic_threshold
    )

    if debug:
        print(f"    DEBUG: x0_pred shape={x0_pred.shape}, sum={x0_pred.sum().item():.0f}")

    return x0_pred


# ============ Tests ============

def test_single_instance(model, n_nodes=20, device='cuda'):
    """Test single instance with official inference method."""
    print(f"\n{'='*60}")
    print(f"Test 1: Single instance TSP-{n_nodes} (Official Method)")
    print(f"{'='*60}")

    torch.manual_seed(42)
    coords = torch.rand(1, n_nodes, 2, device=device)
    coords_np = coords[0].cpu().numpy()

    # Use official inference method
    start_time = time.time()
    tours, adj_probs = run_inference_with_model(model, coords, device)
    inference_time = time.time() - start_time

    pred_tour = tours[0]

    # Remove closing node if present
    if len(pred_tour) == n_nodes + 1 and pred_tour[0] == pred_tour[-1]:
        pred_tour = pred_tour[:-1]

    tour_cost = compute_tour_cost(coords_np, pred_tour + [pred_tour[0]])

    print(f"  Tour: {pred_tour[:10]}... (first 10)")
    print(f"  Tour length: {len(pred_tour)} nodes")
    print(f"  Tour cost: {tour_cost:.4f}")
    print(f"  Inference time: {inference_time:.3f}s")

    # Validate tour
    if len(pred_tour) == n_nodes and len(set(pred_tour)) == n_nodes:
        print(f"  Tour valid: Yes")
    else:
        print(f"  Tour valid: No (length={len(pred_tour)}, unique={len(set(pred_tour))})")

    return pred_tour, tour_cost


def test_custom_vs_official(model, n_nodes=20, device='cuda'):
    """Compare custom diffusion to official inference."""
    print(f"\n{'='*60}")
    print(f"Test 2: Custom diffusion vs Official method")
    print(f"{'='*60}")

    torch.manual_seed(42)
    coords = torch.rand(1, n_nodes, 2, device=device)
    coords_np = coords[0].cpu().numpy()

    # Run custom diffusion
    print("  Running custom diffusion...")
    adj_custom = run_custom_diffusion(model, coords, device, n_steps=50, debug=True)
    adj_custom_np = adj_custom[0].cpu().numpy()

    # Use SUM symmetrization (like official), not max or average
    adj_custom_sym = adj_custom_np + adj_custom_np.T

    n_edges_custom = (adj_custom_sym > 0.5).sum() / 2
    mean_degree_custom = (adj_custom_sym > 0.5).sum(axis=1).mean()

    print(f"  Custom: edges={n_edges_custom:.0f}, mean_degree={mean_degree_custom:.2f}")

    # Use OFFICIAL merge_tours (same as pl_edisco_model uses)
    from utils.tsp_utils import merge_tours as official_merge_tours
    tours_custom, _ = official_merge_tours(
        adj_custom_sym[np.newaxis, ...],  # Add batch dim
        coords_np,
        None,  # edge_index_np
        sparse_graph=False,
        parallel_sampling=1
    )
    tour_custom = tours_custom[0]
    if len(tour_custom) == n_nodes + 1:
        tour_custom = tour_custom[:-1]
    cost_custom = compute_tour_cost(coords_np, tour_custom + [tour_custom[0]])
    print(f"  Custom tour cost (official merge): {cost_custom:.4f}")

    # Also test with Python merge for comparison
    tour_python = merge_tours_python(adj_custom_sym, coords_np)
    if len(tour_python) == n_nodes + 1:
        tour_python = tour_python[:-1]
    cost_python = compute_tour_cost(coords_np, tour_python + [tour_python[0]])
    print(f"  Custom tour cost (Python merge): {cost_python:.4f}")

    # Run official inference
    print("  Running official inference...")
    torch.manual_seed(42)  # Reset seed for same x_T
    tours, adj_probs = run_inference_with_model(model, coords, device)
    pred_tour = tours[0]
    if len(pred_tour) == n_nodes + 1:
        pred_tour = pred_tour[:-1]
    cost_official = compute_tour_cost(coords_np, pred_tour + [pred_tour[0]])
    print(f"  Official tour cost: {cost_official:.4f}")

    # Debug: trace through official method manually
    print("\n  Tracing official method internals...")
    torch.manual_seed(42)
    from utils.ode_solvers import get_solver
    from models.continuous_score_network import ScoreWrapper

    coords_test = coords.clone()
    if coords_test.dim() == 2:
        coords_test = coords_test.unsqueeze(0)
    batch_size, n_nodes_t, _ = coords_test.shape

    x_T = torch.randint(0, 2, (batch_size, n_nodes_t, n_nodes_t),
                       device=device, dtype=torch.float32)
    print(f"    x_T sum: {x_T.sum().item():.0f}")

    # Check if model.score_network is same as model.model
    print(f"    model.score_network is model.model: {model.score_network is model.model}")

    # Get solver exactly as official does
    beta_min = model.args.beta_min if hasattr(model.args, 'beta_min') else 0.1
    beta_max = model.args.beta_max if hasattr(model.args, 'beta_max') else 1.5
    solver = get_solver(model.solver_type, model.solver_steps, beta_min=beta_min, beta_max=beta_max)

    score_fn = ScoreWrapper(model.score_network, coords_test, None)

    # Test score_fn directly
    with torch.no_grad():
        test_out = score_fn(x_T, 0.5)
    print(f"    score_fn output shape: {test_out.shape}")
    probs = torch.softmax(test_out, dim=-1)
    print(f"    At t=0.5: edge_prob mean={probs[...,1].mean().item():.4f}")

    # Run solver
    x0_pred_trace = solver.sample(
        score_fn, x_T, device=device,
        schedule=model.time_schedule,
        adaptive_mixing=model.use_adaptive_mixing,
        deterministic_threshold=model.deterministic_threshold
    )
    print(f"    x0_pred_trace sum: {x0_pred_trace.sum().item():.0f}")

    return cost_custom, cost_official


def test_many_instances(model, n_nodes=20, n_instances=100, device='cuda'):
    """Test many instances using official inference."""
    print(f"\n{'='*60}")
    print(f"Test 3: {n_instances} instances TSP-{n_nodes} (Official Method)")
    print(f"{'='*60}")

    torch.manual_seed(0)

    batch_size = 16
    n_batches = (n_instances + batch_size - 1) // batch_size

    all_costs = []

    start_time = time.time()
    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, n_instances)
        actual_batch_size = end_idx - start_idx

        coords = torch.rand(actual_batch_size, n_nodes, 2, device=device)

        # Process one at a time (official method expects batch=1)
        for i in range(actual_batch_size):
            single_coords = coords[i:i+1]
            coords_np = single_coords[0].cpu().numpy()

            with torch.no_grad():
                tours, _ = run_inference_with_model(model, single_coords, device)

            pred_tour = tours[0]
            if len(pred_tour) == n_nodes + 1:
                pred_tour = pred_tour[:-1]

            cost = compute_tour_cost(coords_np, pred_tour + [pred_tour[0]])
            all_costs.append(cost)

        print(f"  Batch {batch_idx+1}/{n_batches}: processed {end_idx}/{n_instances}")

    total_time = time.time() - start_time

    print(f"\n  Results on {n_instances} instances:")
    print(f"  Mean cost: {np.mean(all_costs):.4f}")
    print(f"  Std cost: {np.std(all_costs):.4f}")
    print(f"  Cost range: [{min(all_costs):.4f}, {max(all_costs):.4f}]")
    print(f"  Total time: {total_time:.2f}s ({total_time/n_instances*1000:.1f}ms/inst)")

    return all_costs


def main():
    print("="*60)
    print("EDISCO Inference Test (PyTorch Lightning)")
    print("="*60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Find checkpoint
    ckpt_path = 'pretrained/edisco_tsp20.ckpt'
    if not os.path.exists(ckpt_path):
        print(f"ERROR: Checkpoint not found at {ckpt_path}")
        return

    n_nodes = 20

    # Load model using PL
    try:
        model = load_model_from_checkpoint(ckpt_path, device, n_nodes)
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return

    # Run tests
    try:
        test_single_instance(model, n_nodes=n_nodes, device=device)
        test_custom_vs_official(model, n_nodes=n_nodes, device=device)
        test_many_instances(model, n_nodes=n_nodes, n_instances=100, device=device)

        print("\n" + "="*60)
        print("All tests completed!")
        print("Expected TSP-20 cost: ~3.84 (without 2-opt)")
        print("="*60)

    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
