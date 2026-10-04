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
import os
import pprint as pp
import shutil
import subprocess
import tempfile
import time
import warnings
from multiprocessing import Pool

import numpy as np
import tqdm

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


SCALE = 1e6


def solve_lkh(lkh_path, nodes_coord, max_trials, runs):
  """Label one instance with LKH-3 in its default configuration.

  Only MAX_TRIALS and RUNS are set; every other LKH-3 parameter keeps its
  default value (5-opt sequential moves, alpha-nearness candidates).
  """
  num_nodes = len(nodes_coord)
  workdir = tempfile.mkdtemp(prefix="lkh_")
  try:
    problem_file = os.path.join(workdir, "problem.tsp")
    tour_file = os.path.join(workdir, "problem.tour")
    param_file = os.path.join(workdir, "problem.par")
    with open(problem_file, "w") as f:
      f.write(f"NAME : TSP\nTYPE : TSP\nDIMENSION : {num_nodes}\n"
              "EDGE_WEIGHT_TYPE : EUC_2D\nNODE_COORD_SECTION\n")
      for n, (x, y) in enumerate(nodes_coord * SCALE):
        f.write(f"{n + 1} {x} {y}\n")
      f.write("EOF\n")
    with open(param_file, "w") as f:
      f.write(f"PROBLEM_FILE = {problem_file}\nTOUR_FILE = {tour_file}\n"
              f"MAX_TRIALS = {max_trials}\nRUNS = {runs}\nTRACE_LEVEL = 0\n")
    subprocess.run([lkh_path, param_file], check=True, stdin=subprocess.DEVNULL,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    with open(tour_file) as f:
      section = f.read().split("TOUR_SECTION")[1].split()
    return [int(node) - 1 for node in section[:num_nodes]]
  finally:
    shutil.rmtree(workdir, ignore_errors=True)


def solve_farthest_insertion(nodes_coord):
  """Farthest Insertion heuristic tour (suboptimal labels for the data-quality study)."""
  num_nodes = len(nodes_coord)
  dist = np.linalg.norm(nodes_coord[:, None] - nodes_coord[None, :], axis=-1)
  first, second = np.unravel_index(np.argmax(dist), dist.shape)
  tour = [int(first), int(second)]
  in_tour = np.zeros(num_nodes, dtype=bool)
  in_tour[tour] = True
  nearest = np.minimum(dist[first], dist[second])  # distance of every node to the tour
  while len(tour) < num_nodes:
    node = int(np.argmax(np.where(in_tour, -np.inf, nearest)))
    current = np.array(tour)
    following = np.roll(current, -1)
    cost = dist[current, node] + dist[node, following] - dist[current, following]
    tour.insert(int(np.argmin(cost)) + 1, node)
    in_tour[node] = True
    nearest = np.minimum(nearest, dist[node])
  return tour


def solve_tsp(task):
  """Generate and label one instance. Instance `index` uses seed + index."""
  index, opts = task
  rng = np.random.RandomState(opts.seed + index)
  num_nodes = rng.randint(low=opts.min_nodes, high=opts.max_nodes + 1)
  nodes_coord = generate_distribution(num_nodes, opts.distribution, seed=opts.seed + index)

  if opts.solver == "concorde":
    if not HAS_CONCORDE:
      raise ImportError("Concorde solver requested but not installed. Install with: pip install pyconcorde")
    solver = TSPSolver.from_data(nodes_coord[:, 0] * SCALE, nodes_coord[:, 1] * SCALE, norm="EUC_2D")
    tour = list(solver.solve(verbose=False).tour)
  elif opts.solver == "lkh":
    tour = solve_lkh(opts.lkh_path, nodes_coord, opts.lkh_trails, opts.lkh_runs)
  elif opts.solver == "farthest_insertion":
    tour = solve_farthest_insertion(nodes_coord)
  else:
    raise ValueError(f"Unknown solver: {opts.solver}")

  if not (np.sort(tour) == np.arange(num_nodes)).all():
    raise RuntimeError(f"solver returned an invalid tour for instance {index}")
  line = " ".join(str(x) + " " + str(y) for x, y in nodes_coord)
  line += " output " + " ".join(str(node_idx + 1) for node_idx in tour)
  line += " " + str(tour[0] + 1) + " \n"
  return index, line


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--min_nodes", type=int, default=20)
  parser.add_argument("--max_nodes", type=int, default=50)
  parser.add_argument("--num_samples", type=int, default=128000)
  parser.add_argument("--batch_size", type=int, default=128,
                      help="Instances written per flush")
  parser.add_argument("--num_workers", type=int, default=None,
                      help="Parallel solver processes (default: batch_size)")
  parser.add_argument("--filename", type=str, default=None)
  parser.add_argument("--solver", type=str, default="lkh",
                      choices=["lkh", "concorde", "farthest_insertion"])
  parser.add_argument("--lkh_path", type=str, default="LKH-3.0.6/LKH",
                      help="Path to the LKH-3 executable")
  parser.add_argument("--lkh_trails", type=int, default=1000,
                      help="LKH-3 MAX_TRIALS")
  parser.add_argument("--lkh_runs", type=int, default=10,
                      help="LKH-3 RUNS (LKH-3 default)")
  parser.add_argument("--seed", type=int, default=1234)
  parser.add_argument("--distribution", type=str, default="uniform",
                      choices=["uniform", "cluster", "explosion", "implosion", "gaussian"],
                      help="Distribution type for OOD evaluation")
  opts = parser.parse_args()

  if opts.filename is None:
    if opts.min_nodes == opts.max_nodes:
      opts.filename = f"tsp{opts.min_nodes}_{opts.distribution}_{opts.solver}.txt"
    else:
      opts.filename = f"tsp{opts.min_nodes}-{opts.max_nodes}_{opts.distribution}_{opts.solver}.txt"
  num_workers = opts.num_workers or opts.batch_size
  if opts.solver == "lkh":
    opts.lkh_path = os.path.expanduser(opts.lkh_path)
    if shutil.which(opts.lkh_path) is None:
      raise FileNotFoundError(f"LKH-3 executable not found: {opts.lkh_path}")

  # Pretty print the run args
  pp.pprint(vars(opts))

  # Resume: instances are written in index order, so the number of existing
  # lines is the index of the next instance to generate.
  done = 0
  if os.path.exists(opts.filename):
    with open(opts.filename) as f:
      done = sum(1 for _ in f)
    print(f"Resuming from instance {done}")

  start_time = time.time()
  with open(opts.filename, "a") as f, Pool(num_workers) as pool:
    tasks = ((index, opts) for index in range(done, opts.num_samples))
    progress = tqdm.tqdm(pool.imap(solve_tsp, tasks), total=opts.num_samples - done)
    for count, (index, line) in enumerate(progress, 1):
      f.write(line)
      if count % opts.batch_size == 0:
        f.flush()

  end_time = time.time() - start_time
  generated = opts.num_samples - done
  print(f"Completed generation of {generated} samples of TSP{opts.min_nodes}-{opts.max_nodes}.")
  print(f"Total time: {end_time / 60:.1f}m")
  print(f"Average time: {end_time / max(generated, 1):.2f}s")
