"""
Generate Euclidean Steiner Tree datasets.

This script generates Steiner Tree instances with Iterated 1-Steiner solutions
in the same format as TSP data for EDISCO training.

Data format (per line):
    terminals_x1 terminals_y1 ... SEP candidates_x1 candidates_y1 ... output adj_00 adj_01 ...

Example usage:
    python generate_steiner_data.py --problem_size 10 --num_samples 10000 \\
        --filename steiner10_train.txt --solver iterated_1steiner
"""

import argparse
import numpy as np
from tqdm import tqdm
import sys
import os

# Add EDISCO to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../edisco')))

try:
    from utils.steiner_utils import IteratedOneSteinerSolver, SteinerTreeEvaluator, GeoSteinerSolver
except ImportError:
    print("Warning: Could not import steiner_utils, using MST fallback")
    from scipy.sparse.csgraph import minimum_spanning_tree
    from scipy.spatial import distance_matrix

    class IteratedOneSteinerSolver:
        @staticmethod
        def solve(coords, is_terminal, max_iterations=5):
            n = len(coords)
            dist_mat = distance_matrix(coords, coords)
            mst = minimum_spanning_tree(dist_mat)
            mst_array = mst.toarray()
            adjacency = np.zeros((n, n), dtype=np.float32)
            total_length = 0.0
            for i in range(n):
                for j in range(i+1, n):
                    if mst_array[i, j] > 0 or mst_array[j, i] > 0:
                        adjacency[i, j] = 1.0
                        adjacency[j, i] = 1.0
                        total_length += dist_mat[i, j]
            return adjacency, total_length

    class GeoSteinerSolver:
        @staticmethod
        def solve(coords, is_terminal, **kwargs):
            raise ImportError("GeoSteiner solver requires steiner_utils module")


def generate_steiner_instance(problem_size, seed=None, solver='iterated_1steiner'):
    """Generate a single Steiner Tree instance.

    Args:
        problem_size: Number of terminal points
        seed: Random seed
        solver: Solver to use ('iterated_1steiner', 'geosteiner', etc.)

    Returns:
        terminals: (n_terminals, 2) coordinates
        candidates: (n_candidates, 2) coordinates
        adjacency: (n_total, n_total) adjacency matrix
        is_terminal: (n_total,) binary indicators
        total_length: Total tree length
    """
    if seed is not None:
        np.random.seed(seed)

    n_terminals = problem_size
    n_candidates = problem_size  # Equal number of candidates

    # Generate terminals uniformly in [0, 1]²
    terminals = np.random.uniform(0, 1, size=(n_terminals, 2)).astype(np.float32)

    # Generate candidate Steiner points
    candidates = np.random.uniform(0, 1, size=(n_candidates, 2)).astype(np.float32)

    # Combine all coordinates
    all_coords = np.vstack([terminals, candidates])
    n_total = len(all_coords)

    # Create is_terminal indicator
    is_terminal = np.zeros(n_total, dtype=np.float32)
    is_terminal[:n_terminals] = 1.0

    # Solve using specified solver
    if solver == 'geosteiner':
        adjacency, total_length = GeoSteinerSolver.solve(
            all_coords, is_terminal
        )
    elif solver == 'iterated_1steiner':
        adjacency, total_length = IteratedOneSteinerSolver.solve(
            terminals, candidates, max_outer_iters=3
        )
    else:
        raise ValueError(f"Unknown solver: {solver}")

    return terminals, candidates, adjacency, is_terminal, total_length


def instance_to_line(terminals, candidates, adjacency):
    """Convert instance to text line following EDISCO data format.

    Format: <terminals> SEP <candidates> output <adjacency>

    Args:
        terminals: (n_terminals, 2) coordinates
        candidates: (n_candidates, 2) coordinates
        adjacency: (n_total, n_total) binary matrix

    Returns:
        String line for dataset file
    """
    # Flatten coordinates
    terminals_flat = ' '.join([f'{x:.6f}' for x in terminals.flatten()])
    candidates_flat = ' '.join([f'{x:.6f}' for x in candidates.flatten()])

    # Flatten adjacency matrix (row-major)
    adjacency_flat = ' '.join([str(int(x)) for x in adjacency.flatten()])

    # Combine in EDISCO format
    line = f"{terminals_flat} SEP {candidates_flat} output {adjacency_flat}"

    return line


def generate_dataset(args):
    """Generate full dataset and save to file."""

    print(f"Generating {args.num_samples} Steiner Tree instances")
    print(f"  Problem size: {args.problem_size} terminals + {args.problem_size} candidates")
    print(f"  Solver: {args.solver}")
    print(f"  Output: {args.filename}")
    print(f"  Seed: {args.seed}")

    # Create output directory if needed
    os.makedirs(os.path.dirname(args.filename) if os.path.dirname(args.filename) else '.', exist_ok=True)

    # Generate instances
    lines = []
    lengths = []

    for i in tqdm(range(args.num_samples), desc="Generating"):
        terminals, candidates, adjacency, is_terminal, length = generate_steiner_instance(
            problem_size=args.problem_size,
            seed=args.seed + i if args.seed is not None else None,
            solver=args.solver
        )

        # Convert to text format
        line = instance_to_line(terminals, candidates, adjacency)
        lines.append(line)
        lengths.append(length)

    # Write to file
    with open(args.filename, 'w') as f:
        for line in lines:
            f.write(line + '\n')

    # Print statistics
    print(f"\nDataset generation complete!")
    print(f"  Instances: {args.num_samples}")
    print(f"  Avg tree length: {np.mean(lengths):.6f} ± {np.std(lengths):.6f}")
    print(f"  Min/Max length: {np.min(lengths):.6f} / {np.max(lengths):.6f}")
    print(f"  File size: {os.path.getsize(args.filename) / 1024 / 1024:.2f} MB")
    print(f"  Saved to: {args.filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate Steiner Tree dataset for EDISCO')

    parser.add_argument('--problem_size', type=int, required=True,
                       help='Number of terminal points (10, 20, or 50)')
    parser.add_argument('--num_samples', type=int, required=True,
                       help='Number of instances to generate')
    parser.add_argument('--filename', type=str, required=True,
                       help='Output filename (e.g., steiner10_train.txt)')
    parser.add_argument('--solver', type=str, default='iterated_1steiner',
                       choices=['mst', '1steiner', 'iterated_1steiner', 'geosteiner'],
                       help='Solver for ground truth (default: iterated_1steiner). '
                            'geosteiner requires GeoSteiner installed (http://www.geosteiner.com/)')
    parser.add_argument('--seed', type=int, default=1234,
                       help='Random seed (default: 1234)')

    args = parser.parse_args()

    generate_dataset(args)
