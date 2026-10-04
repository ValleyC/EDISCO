"""Partition projection and sub-TSP solving for large-scale CVRP.

Stage 1 of the partition-diffusion pipeline produces an affinity matrix over
customer pairs (the denoised same-route probabilities). The projection turns
it into capacity-feasible customer clusters:

1. spectral clustering on the affinity matrix,
2. capacity-feasible re-balancing: an over-capacity cluster sheds its
   lowest-affinity customer to the feasible cluster it is most affine to,
   until every cluster fits.

The projection reads only cluster affinities and demand sums. Stage 2 solves
each cluster together with the depot as a small Euclidean TSP.
"""

import numpy as np
import torch

from utils.native_decoder import decode_tsp_batch
from diffusion.solvers import get_solver


def customer_affinity(probs, n_nodes, edge_index=None):
    """Symmetric customer-customer affinity from same-route probabilities.

    Args:
        probs: (n_nodes, n_nodes) dense probabilities, or (E,) with edge_index
        n_nodes: number of nodes including the depot (node 0)
    Returns:
        (n_nodes - 1, n_nodes - 1) array; entry (a, b) refers to customers a + 1 and b + 1
    """
    probs = probs.detach().cpu().numpy() if torch.is_tensor(probs) else np.asarray(probs)
    if edge_index is None:
        dense = probs.astype(np.float64)
    else:
        index = edge_index.detach().cpu().numpy() if torch.is_tensor(edge_index) else np.asarray(edge_index)
        dense = np.zeros((n_nodes, n_nodes))
        dense[index[0], index[1]] = probs
    affinity = 0.5 * (dense + dense.T)[1:, 1:]
    np.fill_diagonal(affinity, 0.0)
    return affinity


def distance_affinity(coords):
    """Affinity for the k-means ablation: a Gaussian kernel of customer distances."""
    points = np.asarray(coords, dtype=np.float64)[1:]
    distances = np.linalg.norm(points[:, None] - points[None, :], axis=-1)
    nearest = np.sort(distances + np.diag(np.full(len(points), np.inf)), axis=1)[:, 0]
    scale = max(float(np.mean(nearest)), 1e-12)
    affinity = np.exp(-(distances / scale) ** 2 / 2.0)
    np.fill_diagonal(affinity, 0.0)
    return affinity


def num_clusters(demands, capacity):
    """Smallest number of vehicles that can carry the total demand."""
    return max(1, int(np.ceil(float(np.sum(demands)) / float(capacity) - 1e-9)))


def spectral_partition(affinity, n_clusters, seed=0):
    """Spectral clustering of customers on a precomputed affinity matrix."""
    n = affinity.shape[0]
    if n_clusters <= 1:
        return np.zeros(n, dtype=np.int64)
    if n <= n_clusters:
        return np.arange(n)
    from sklearn.cluster import SpectralClustering
    # a small uniform term keeps the affinity graph connected
    model = SpectralClustering(n_clusters=n_clusters, affinity='precomputed',
                               assign_labels='kmeans', random_state=seed)
    return model.fit_predict(affinity + 1e-6)


def kmeans_partition(coords, n_clusters, seed=0):
    """Ablation baseline: Lloyd's algorithm on raw customer coordinates."""
    points = np.asarray(coords, dtype=np.float64)[1:]
    if n_clusters <= 1 or len(points) <= n_clusters:
        return np.arange(len(points)) % max(n_clusters, 1)
    from sklearn.cluster import KMeans
    return KMeans(n_clusters=n_clusters, n_init=10, random_state=seed).fit_predict(points)


def rebalance_capacity(labels, affinity, demands, capacity):
    """Capacity-feasible re-balancing of customer clusters.

    Args:
        labels: (n_customers,) cluster label of every customer
        affinity: (n_customers, n_customers) symmetric affinity
        demands: (n_customers,) customer demands
        capacity: vehicle capacity
    Returns:
        list of clusters, each a sorted list of customer indices (0-based),
        with total demand at most `capacity`
    """
    demands = np.asarray(demands, dtype=np.float64)
    if np.any(demands > capacity):
        raise ValueError("every customer demand must be at most vehicle capacity")
    clusters = [list(np.nonzero(labels == label)[0]) for label in np.unique(labels)]
    clusters = [cluster for cluster in clusters if cluster]
    load = [float(demands[cluster].sum()) for cluster in clusters]

    def mean_affinity(customer, members):
        others = [m for m in members if m != customer]
        return float(affinity[customer, others].mean()) if others else 0.0

    while True:
        over = [k for k in range(len(clusters)) if load[k] > capacity + 1e-9]
        if not over:
            break
        k = over[0]
        # the customer least affine to its own cluster leaves first (ties by index)
        customer = min(clusters[k], key=lambda c: (mean_affinity(c, clusters[k]), c))
        feasible = [j for j in range(len(clusters))
                    if j != k and load[j] + demands[customer] <= capacity + 1e-9]
        clusters[k].remove(customer)
        load[k] -= demands[customer]
        if feasible:
            j = max(feasible, key=lambda j: (mean_affinity(customer, clusters[j]), -j))
            clusters[j].append(customer)
            load[j] += demands[customer]
        else:
            clusters.append([customer])
            load.append(float(demands[customer]))
    return [sorted(int(c) for c in cluster) for cluster in clusters if cluster]


@torch.no_grad()
def tsp_heatmap(tsp_model, coords, solver_type='deis', solver_steps=5, beta_min=0.1, beta_max=1.5):
    """Edge probabilities of one small TSP instance from an EDISCO TSP score network."""
    device = next(tsp_model.parameters()).device
    points = torch.as_tensor(coords, dtype=torch.float32, device=device).unsqueeze(0)
    n = points.shape[1]
    solver = get_solver(solver_type, solver_steps, beta_min=beta_min, beta_max=beta_max)

    def score_fn(x, t):
        return tsp_model(points, x, torch.full((1,), float(t), device=device))

    x_T = torch.randint(0, 2, (1, n, n), device=device, dtype=torch.float32)
    return solver.sample(score_fn, x_T)[0].cpu().numpy()


def solve_sub_tsp(coords, tsp_model=None, **sampler_kwargs):
    """Order the nodes of one cluster plus depot (node 0 of `coords`) into a tour.

    With a TSP score network the tour is decoded by NEE from its heatmap.
    Without one, NEE runs on a uniform heatmap, i.e. it ranks edges by
    distance alone.

    Returns the visiting order of nodes 1..n-1 starting after the depot.
    """
    coords = np.asarray(coords, dtype=np.float64)
    n = len(coords)
    if n <= 3:
        return list(range(1, n))
    heatmap = np.full((n, n), 0.5) if tsp_model is None else tsp_heatmap(tsp_model, coords, **sampler_kwargs)
    tours, _ = decode_tsp_batch(heatmap, coords, decoder='nee')
    return [node for node in tours[0][:-1] if node != 0]


def partition_to_routes(clusters, coords, tsp_model=None, **sampler_kwargs):
    """Solve every cluster (0-based customer indices) with the depot as a TSP.

    Returns routes as lists of node indices (customers are 1-based, depot is 0).
    """
    coords = np.asarray(coords, dtype=np.float64)
    routes = []
    for cluster in clusters:
        nodes = [0] + [c + 1 for c in cluster]
        order = solve_sub_tsp(coords[nodes], tsp_model, **sampler_kwargs)
        routes.append([nodes[k] for k in order])
    return routes
