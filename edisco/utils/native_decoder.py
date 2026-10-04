"""Feasible probability-only greedy and distance-aware NEE decoders.

Both operate once on the final clean-edge heatmap. NEE uses
(P_ij + P_ji) / (2 * (d_ij + epsilon)). Greedy uses only the symmetrized
probability. Degree and union-find checks are identical, with node-index
tie breaking. Sparse decoding scores represented edges first and can use
zero-probability missing edges to complete a feasible cycle.
"""

from itertools import combinations

import numpy as np
import torch


class _UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1
        return True


def _numpy(value):
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _ranked_cycle(n, candidates, sparse_completion=False):
    if n < 3:
        raise ValueError("Need at least 3 nodes for a Hamiltonian cycle")
    degree, uf, selected, inspected = [0] * n, _UnionFind(n), set(), 0
    phases = [candidates]
    if sparse_completion:
        # Unrepresented edges have probability zero. Node-index ordering
        # preserves the dense decoder's tie rule without an n-by-n allocation.
        phases.append(combinations(range(n), 2))
    for phase in phases:
        for i, j in phase:
            if len(selected) == n:
                break
            i, j = int(i), int(j)
            inspected += 1
            if degree[i] >= 2 or degree[j] >= 2 or (i, j) in selected:
                continue
            if uf.find(i) == uf.find(j) and len(selected) != n - 1:
                continue
            selected.add((i, j))
            degree[i] += 1
            degree[j] += 1
            uf.union(i, j)
        if len(selected) == n:
            break
    if len(selected) != n or any(d != 2 for d in degree):
        raise RuntimeError("Could not construct a Hamiltonian cycle")
    return frozenset(selected), inspected


def _as_tour(n, edges):
    adjacency = [[] for _ in range(n)]
    for i, j in edges:
        adjacency[i].append(j)
        adjacency[j].append(i)
    tour, visited = [0], {0}
    while len(tour) < n:
        available = sorted(v for v in adjacency[tour[-1]] if v not in visited)
        if not available:
            raise RuntimeError("Decoded edge set is not a single cycle")
        tour.append(available[0])
        visited.add(tour[-1])
    return tour + [0]


def _dense_decode(edge_probs, distances=None, epsilon=1e-8,
                  return_edges=False, return_stats=False):
    p = _numpy(edge_probs).astype(np.float64, copy=False)
    if p.ndim != 2 or p.shape[0] != p.shape[1]:
        raise ValueError("edge_probs must be a square matrix")
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("edge probabilities must be finite and in [0, 1]")
    n = p.shape[0]
    iu = np.triu_indices(n, k=1)
    scores = 0.5 * (p[iu] + p.T[iu])
    if distances is not None:
        d = _numpy(distances).astype(np.float64, copy=False)
        if d.shape != p.shape or not np.isfinite(d).all() or np.any(d < 0):
            raise ValueError("distances must be finite, nonnegative and match edge_probs")
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        scores = scores / (d[iu] + epsilon)
    order = np.lexsort((iu[1], iu[0], -scores))
    edges, inspected = _ranked_cycle(n, zip(iu[0][order], iu[1][order]))
    result = edges if return_edges else _as_tour(n, edges)
    return (result, inspected) if return_stats else result


def native_edge_expansion_decode(edge_probs, distances, return_edges=False,
                                  epsilon=1e-8, return_stats=False):
    """NEE with symmetrized probability divided by distance plus epsilon."""
    return _dense_decode(edge_probs, distances, epsilon, return_edges, return_stats)


def greedy_edge_decode(edge_probs, return_edges=False, return_stats=False):
    """Feasible greedy decoding ranked only by symmetrized probability."""
    return _dense_decode(edge_probs, return_edges=return_edges, return_stats=return_stats)


def _sparse_decode(values, coords, edge_index, decoder, epsilon=1e-8):
    n = len(coords)
    src, dst = edge_index
    if np.any(edge_index < 0) or np.any(edge_index >= n):
        raise ValueError("edge indices lie outside the coordinate array")
    if len(values) != len(src) or not np.isfinite(values).all():
        raise ValueError("invalid sparse edge probabilities")
    if np.any((values < 0) | (values > 1)):
        raise ValueError("edge probabilities must be in [0, 1]")
    # One prediction per directed edge, with absent reverse entries set to 0.
    directed_keys = src * n + dst
    if len(np.unique(directed_keys)) != len(directed_keys):
        raise ValueError("duplicate directed edges are not supported")
    mask = src != dst
    lo, hi = np.minimum(src[mask], dst[mask]), np.maximum(src[mask], dst[mask])
    keys, inverse = np.unique(lo * n + hi, return_inverse=True)
    scores = np.bincount(inverse, weights=values[mask], minlength=len(keys)) / 2
    i, j = keys // n, keys % n
    if decoder == "nee":
        scores /= np.linalg.norm(coords[i] - coords[j], axis=-1) + epsilon
    # Zero entries share the same node-index order as all absent edges.
    keep = scores > 0
    i, j, scores = i[keep], j[keep], scores[keep]
    order = np.lexsort((j, i, -scores))
    edges, inspected = _ranked_cycle(n, zip(i[order], j[order]), sparse_completion=True)
    return _as_tour(n, edges), inspected


def decode_tsp_batch(edge_probs, coords, edge_index=None, sparse_graph=False,
                     parallel_sampling=1, decoder="nee"):
    """Decode each heatmap once, for dense batches or one sparse graph.

    Returns tours and the mean number of candidate inspections. Sparse input
    may contain multiple samples of the same graph, with a shared edge index.
    """
    if decoder not in ("greedy", "nee"):
        raise ValueError("decoder must be greedy or nee")
    p, c = _numpy(edge_probs), _numpy(coords).astype(np.float64, copy=False)
    if c.ndim == 2:
        c = c[None]
    if c.ndim != 3 or c.shape[-1] != 2 or not np.isfinite(c).all():
        raise ValueError("coords must be finite and have shape (n,2) or (batch,n,2)")
    if sparse_graph:
        ei = _numpy(edge_index).astype(np.int64, copy=False)
        if ei.ndim != 2 or ei.shape[0] != 2 or ei.shape[1] == 0:
            raise ValueError("sparse decoding requires edge_index with shape (2,E)")
        p = p.reshape(-1, ei.shape[1])
    else:
        n = c.shape[1]
        p = p.reshape(-1, n, n)
    if len(c) not in (1, len(p)):
        raise ValueError("coordinate and heatmap batch sizes differ")
    tours, counts = [], []
    for b, heatmap in enumerate(p):
        points = c[0 if len(c) == 1 else b]
        if sparse_graph:
            tour, inspected = _sparse_decode(heatmap, points, ei, decoder)
        elif decoder == "greedy":
            tour, inspected = greedy_edge_decode(heatmap, return_stats=True)
        else:
            d = np.linalg.norm(points[:, None] - points[None, :], axis=-1)
            tour, inspected = native_edge_expansion_decode(heatmap, d, return_stats=True)
        tours.append(tour)
        counts.append(inspected)
    return tours, float(np.mean(counts))


def tour_edge_set(tour):
    """Undirected edge set of a closed or open tour representation."""
    seq = tour[:-1] if tour[0] == tour[-1] else tour
    return frozenset(
        (min(seq[k], seq[(k + 1) % len(seq)]), max(seq[k], seq[(k + 1) % len(seq)]))
        for k in range(len(seq))
    )
