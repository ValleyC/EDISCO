"""
Generate TSP datasets with different distributions.

Distributions:
- uniform: Points uniformly sampled from [0, 1]²
- cluster: Points grouped into clusters
- explosion: Gap/hole created by pushing points away from center
- implosion: Points pulled toward center
- gaussian: 2D normal distribution

Based on Bi et al. (NeurIPS 2022) for OOD evaluation.
"""

import argparse
import pprint as pp
import time
import warnings
from multiprocessing import Pool

import lkh
import numpy as np
import tqdm
import tsplib95

# Optional: only import if using Concorde solver
try:
  from concorde.tsp import TSPSolver
  HAS_CONCORDE = True
except (ImportError, ModuleNotFoundError):
  HAS_CONCORDE = False

warnings.filterwarnings("ignore")


def generate_distribution(num_nodes, distribution, seed=None):
  """Generate TSP instance according to specified distribution."""
  if seed is not None:
    np.random.seed(seed)

  if distribution == "uniform":
    return np.random.uniform(0, 1, size=(num_nodes, 2))

  elif distribution == "cluster":
    # Points grouped into clusters
    n_clusters = max(5, int(np.sqrt(num_nodes)))
    centers = np.random.uniform(0, 1, size=(n_clusters, 2))
    nodes_per_cluster = num_nodes // n_clusters
    remainder = num_nodes % n_clusters

    points = []
    for i in range(n_clusters):
      n_points = nodes_per_cluster + (1 if i < remainder else 0)
      cluster_points = np.random.normal(centers[i], 0.07, size=(n_points, 2))
      cluster_points = np.clip(cluster_points, 0, 1)
      points.append(cluster_points)
    return np.vstack(points)

  elif distribution == "explosion":
    # Create gap by pushing points away from center
    points = np.random.uniform(0, 1, size=(num_nodes, 2))
    center = np.random.uniform(0.3, 0.7, size=2)
    explosion_radius = 0.25
    push_distance = 0.3

    for i in range(len(points)):
      dist = np.linalg.norm(points[i] - center)
      if dist < explosion_radius and dist > 1e-6:
        direction = (points[i] - center) / dist
        new_dist = explosion_radius + push_distance
        points[i] = center + direction * new_dist
        points[i] = np.clip(points[i], 0, 1)
    return points

  elif distribution == "implosion":
    # Pull points toward center
    points = np.random.uniform(0, 1, size=(num_nodes, 2))
    center = np.random.uniform(0.3, 0.7, size=2)
    attraction_strength = 0.6
    attraction_radius = 0.4

    for i in range(len(points)):
      dist = np.linalg.norm(points[i] - center)
      if dist < attraction_radius:
        direction = center - points[i]
        points[i] = points[i] + direction * attraction_strength
        points[i] = np.clip(points[i], 0, 1)
    return points

  elif distribution == "gaussian":
    # 2D normal distribution
    points = np.random.normal(0.5, 0.17, size=(num_nodes, 2))
    points = np.clip(points, 0, 1)
    return points

  else:
    raise ValueError(f"Unknown distribution: {distribution}")


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--min_nodes", type=int, default=20)
  parser.add_argument("--max_nodes", type=int, default=50)
  parser.add_argument("--num_samples", type=int, default=128000)
  parser.add_argument("--batch_size", type=int, default=128)
  parser.add_argument("--filename", type=str, default=None)
  parser.add_argument("--solver", type=str, default="lkh")
  parser.add_argument("--lkh_path", type=str, default="LKH-3.0.6/LKH",
                      help="Path to LKH executable")
  parser.add_argument("--lkh_trails", type=int, default=1000)
  parser.add_argument("--seed", type=int, default=1234)
  parser.add_argument("--distribution", type=str, default="uniform",
                      choices=["uniform", "cluster", "explosion", "implosion", "gaussian"],
                      help="Distribution type for OOD evaluation")
  opts = parser.parse_args()

  assert opts.num_samples % opts.batch_size == 0, "Number of samples must be divisible by batch size"

  np.random.seed(opts.seed)

  if opts.filename is None:
    if opts.min_nodes == opts.max_nodes:
      opts.filename = f"tsp{opts.min_nodes}_{opts.distribution}_{opts.solver}.txt"
    else:
      opts.filename = f"tsp{opts.min_nodes}-{opts.max_nodes}_{opts.distribution}_{opts.solver}.txt"

  # Pretty print the run args
  pp.pprint(vars(opts))

  with open(opts.filename, "w") as f:
    start_time = time.time()
    for b_idx in tqdm.tqdm(range(opts.num_samples // opts.batch_size)):
      num_nodes = np.random.randint(low=opts.min_nodes, high=opts.max_nodes + 1)
      assert opts.min_nodes <= num_nodes <= opts.max_nodes

      # Generate batch according to distribution
      batch_nodes_coord = np.array([
        generate_distribution(num_nodes, opts.distribution, seed=opts.seed + b_idx * opts.batch_size + i)
        for i in range(opts.batch_size)
      ])

      def solve_tsp(nodes_coord):
        if opts.solver == "concorde":
          if not HAS_CONCORDE:
            raise ImportError("Concorde solver requested but not installed. Install with: pip install pyconcorde")
          scale = 1e6
          solver = TSPSolver.from_data(nodes_coord[:, 0] * scale, nodes_coord[:, 1] * scale, norm="EUC_2D")
          solution = solver.solve(verbose=False)
          tour = solution.tour
        elif opts.solver == "lkh":
          scale = 1e6
          lkh_path = opts.lkh_path
          problem = tsplib95.models.StandardProblem()
          problem.name = 'TSP'
          problem.type = 'TSP'
          problem.dimension = num_nodes
          problem.edge_weight_type = 'EUC_2D'
          problem.node_coords = {n + 1: nodes_coord[n] * scale for n in range(num_nodes)}

          solution = lkh.solve(lkh_path, problem=problem, max_trials=opts.lkh_trails, runs=10)
          tour = [n - 1 for n in solution[0]]
        else:
          raise ValueError(f"Unknown solver: {opts.solver}")

        return tour

      with Pool(opts.batch_size) as p:
        tours = p.map(solve_tsp, [batch_nodes_coord[idx] for idx in range(opts.batch_size)])

      for idx, tour in enumerate(tours):
        if (np.sort(tour) == np.arange(num_nodes)).all():
          f.write(" ".join(str(x) + str(" ") + str(y) for x, y in batch_nodes_coord[idx]))
          f.write(str(" ") + str('output') + str(" "))
          f.write(str(" ").join(str(node_idx + 1) for node_idx in tour))
          f.write(str(" ") + str(tour[0] + 1) + str(" "))
          f.write("\n")

    end_time = time.time() - start_time

    assert b_idx == opts.num_samples // opts.batch_size - 1

  print(f"Completed generation of {opts.num_samples} samples of TSP{opts.min_nodes}-{opts.max_nodes}.")
  print(f"Total time: {end_time / 60:.1f}m")
  print(f"Average time: {end_time / opts.num_samples:.1f}s")
