"""
Generate CVRP datasets labelled with HGS-CVRP.

Instances follow the standard uniform protocol: depot and customers uniform on
the unit square, integer demands in {1, ..., 9}, and vehicle capacity
Q = 40 / 50 / 80 / 100 for N = 50 / 100 / 200 / 500 customers. Reference
solutions come from HGS-CVRP (Vidal, 2022) through the `hygese` package.

For the capacity-shift experiments, pass --capacity_min/--capacity_max to draw
one capacity per instance uniformly from an integer range.

Output: a pickle file with a list of dicts
    coords        (N + 1, 2) float32, node 0 is the depot
    demands       (N + 1,)   float32, depot demand is 0
    capacity      float
    n_customers   int
    n_nodes       int
    solution      {'routes': [[customer, ...], ...], 'total_distance': float, 'solver': 'hgs'}
"""

import argparse
import pickle
import pprint as pp
import time
from multiprocessing import Pool

import numpy as np
import tqdm

DEFAULT_CAPACITY = {20: 30, 50: 40, 100: 50, 200: 80, 500: 100}
DEFAULT_TIME_LIMIT = {20: 1.0, 50: 1.0, 100: 20.0, 200: 60.0, 500: 240.0}
SCALE = 1e4


def route_length(coords, routes):
  total = 0.0
  for route in routes:
    path = [0] + list(route) + [0]
    total += np.linalg.norm(coords[path[1:]] - coords[path[:-1]], axis=1).sum()
  return float(total)


def solve_instance(task):
  """Generate and label one instance. Instance `index` uses seed + index."""
  import hygese as hgs

  index, opts = task
  rng = np.random.RandomState(opts.seed + index)
  n = opts.num_customers
  coords = rng.uniform(0, 1, size=(n + 1, 2))
  demands = np.concatenate([[0], rng.randint(opts.demand_low, opts.demand_high + 1, size=n)])
  if opts.capacity_min is not None:
    capacity = int(rng.randint(opts.capacity_min, opts.capacity_max + 1))
  else:
    capacity = int(opts.capacity)

  data = dict(
      x_coordinates=coords[:, 0] * SCALE,
      y_coordinates=coords[:, 1] * SCALE,
      demands=demands,
      vehicle_capacity=capacity,
      num_vehicles=n,
      depot=0,
      service_times=np.zeros(n + 1),
  )
  params = hgs.AlgorithmParameters(timeLimit=opts.time_limit, seed=opts.seed + index)
  result = hgs.Solver(parameters=params, verbose=False).solve_cvrp(data, rounding=False)
  routes = [[int(c) for c in route] for route in result.routes if len(route) > 0]

  visited = sorted(c for route in routes for c in route)
  if visited != list(range(1, n + 1)):
    raise RuntimeError(f"HGS returned an incomplete solution for instance {index}")
  if max(demands[route].sum() for route in routes) > capacity:
    raise RuntimeError(f"HGS returned a capacity-infeasible solution for instance {index}")

  return {
      'coords': coords.astype(np.float32),
      'demands': demands.astype(np.float32),
      'capacity': float(capacity),
      'n_customers': n,
      'n_nodes': n + 1,
      'solution': {
          'routes': routes,
          'total_distance': route_length(coords, routes),
          'solver': 'hgs',
      },
  }


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--num_customers", type=int, default=100)
  parser.add_argument("--num_samples", type=int, default=1000)
  parser.add_argument("--capacity", type=int, default=None,
                      help="Vehicle capacity (default: 40/50/80/100 for N=50/100/200/500)")
  parser.add_argument("--capacity_min", type=int, default=None,
                      help="Draw one capacity per instance uniformly from [capacity_min, capacity_max]")
  parser.add_argument("--capacity_max", type=int, default=None)
  parser.add_argument("--demand_low", type=int, default=1)
  parser.add_argument("--demand_high", type=int, default=9)
  parser.add_argument("--time_limit", type=float, default=None,
                      help="HGS-CVRP time limit per instance in seconds "
                           "(default: 1/20/60/240 for N=50/100/200/500)")
  parser.add_argument("--num_workers", type=int, default=32)
  parser.add_argument("--filename", type=str, default=None)
  parser.add_argument("--seed", type=int, default=1234)
  opts = parser.parse_args()

  if (opts.capacity_min is None) != (opts.capacity_max is None):
    parser.error("--capacity_min and --capacity_max must be given together")
  if opts.capacity is None and opts.capacity_min is None:
    if opts.num_customers not in DEFAULT_CAPACITY:
      parser.error("--capacity is required for this problem size")
    opts.capacity = DEFAULT_CAPACITY[opts.num_customers]
  if opts.time_limit is None:
    opts.time_limit = DEFAULT_TIME_LIMIT.get(opts.num_customers, 60.0)
  if opts.filename is None:
    opts.filename = f"cvrp{opts.num_customers}_hgs.pkl"

  pp.pprint(vars(opts))

  start_time = time.time()
  with Pool(opts.num_workers) as pool:
    tasks = ((index, opts) for index in range(opts.num_samples))
    instances = list(tqdm.tqdm(pool.imap(solve_instance, tasks), total=opts.num_samples))

  with open(opts.filename, "wb") as f:
    pickle.dump(instances, f)

  elapsed = time.time() - start_time
  mean_cost = np.mean([inst['solution']['total_distance'] for inst in instances])
  print(f"Saved {len(instances)} CVRP-{opts.num_customers} instances to {opts.filename}")
  print(f"Mean HGS-CVRP cost: {mean_cost:.4f}")
  print(f"Total time: {elapsed / 60:.1f}m")
