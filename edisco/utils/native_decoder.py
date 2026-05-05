"""Equivariance-preserving native edge-expansion decoder for TSP (T1.4-code).

Maintains a partial edge set `S_t` and iteratively scores candidate edges using
only E(2)-invariant quantities (edge probability `P_ij` from the score net,
pairwise distance `d_ij`, partial edge state `S_ij`, current node degrees, and
subtour status from a union-find structure). Edges are added one at a time
subject to feasibility projection (degree at most two, no premature subtour,
final edge closes a Hamiltonian cycle). Because every input the decoder reads
is invariant under E(2) and the feasibility projection is purely combinatorial,
the selected node-index-space tour is unchanged when coordinates are
translated, rotated, or reflected.

See revision_plan.md (T1.4) and the corresponding section in the manuscript.
"""

import numpy as np
import torch


class _UnionFind:
    """Simple union-find with path compression for subtour detection."""

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


def native_edge_expansion_decode(edge_probs, distances, return_edges=False):
    """Decode a Hamiltonian cycle via equivariance-preserving edge expansion.

    The decoder ranks candidate edges by the invariant score `P_ij / d_ij`
    (matching the existing greedy decoder's symmetrized form before symmetry
    breaking), filters candidates by feasibility, and adds edges one at a time
    until a Hamiltonian cycle is complete. All inputs are E(2)-invariant
    scalars, and the score, ranking, and feasibility checks operate only on
    these scalars and the combinatorial state (degree, union-find). The
    selected edge set therefore depends only on `edge_probs` and `distances`,
    not on coordinate orientation.

    Args:
        edge_probs: (n, n) symmetric numpy array or torch tensor of edge
            probabilities. Diagonal entries are ignored.
        distances: (n, n) symmetric numpy array or torch tensor of pairwise
            Euclidean distances. Diagonal entries are ignored.
        return_edges: if True, return the set of selected edges as a frozen
            list of sorted (i, j) tuples; otherwise return the cycle as a
            sequence of node indices starting at 0.

    Returns:
        Either a list of node indices forming a Hamiltonian cycle (length
        n + 1, with the last index equal to 0) or a frozenset of selected
        (i, j) tuples with i < j.
    """
    if torch.is_tensor(edge_probs):
        edge_probs = edge_probs.detach().cpu().numpy()
    if torch.is_tensor(distances):
        distances = distances.detach().cpu().numpy()

    edge_probs = np.asarray(edge_probs, dtype=np.float64)
    distances = np.asarray(distances, dtype=np.float64)
    n = edge_probs.shape[0]
    if edge_probs.shape != (n, n) or distances.shape != (n, n):
        raise ValueError("edge_probs and distances must both be (n, n)")
    if n < 3:
        raise ValueError("Need at least 3 nodes for a Hamiltonian cycle")

    # Symmetrize the edge probability to remove any directional asymmetry the
    # score net may produce. Combined with the symmetric distance matrix, the
    # score is then a function of an undirected pair only.
    P = 0.5 * (edge_probs + edge_probs.T)

    # Invariant edge score: probability per unit distance. Higher is better.
    iu = np.triu_indices(n, k=1)
    safe_d = np.where(distances[iu] > 1e-12, distances[iu], 1.0)
    scores = P[iu] / safe_d

    # Sort candidate edges by score, descending. Ties are broken by
    # (distance ascending, index ascending) for deterministic output that is
    # itself invariant under E(2) since both keys depend only on invariant
    # quantities.
    order = np.lexsort((iu[1], iu[0], distances[iu], -scores))

    candidates = list(zip(iu[0][order].tolist(), iu[1][order].tolist()))

    degree = [0] * n
    uf = _UnionFind(n)
    selected = set()

    for i, j in candidates:
        if len(selected) == n:
            break
        if degree[i] >= 2 or degree[j] >= 2:
            continue
        # Closing the cycle is allowed only on the final edge.
        if uf.find(i) == uf.find(j):
            if len(selected) == n - 1:
                selected.add((i, j))
                degree[i] += 1
                degree[j] += 1
                break
            continue
        selected.add((i, j))
        degree[i] += 1
        degree[j] += 1
        uf.union(i, j)

    if len(selected) != n:
        # Fallback: connect dangling endpoints if the candidate sweep finished
        # before the cycle closed. This preserves invariance because the
        # fallback selection uses the same invariant scoring restricted to
        # feasible closure edges.
        endpoints = [v for v in range(n) if degree[v] < 2]
        if len(endpoints) != 2:
            raise RuntimeError(
                f"Native decoder could not construct a Hamiltonian cycle: "
                f"{len(selected)} edges selected, {len(endpoints)} dangling endpoints"
            )
        a, b = sorted(endpoints)
        selected.add((a, b))

    if return_edges:
        return frozenset(selected)

    # Reconstruct the cycle as a node sequence starting at 0.
    adjacency = {v: [] for v in range(n)}
    for i, j in selected:
        adjacency[i].append(j)
        adjacency[j].append(i)

    tour = [0]
    visited = {0}
    while len(tour) < n:
        current = tour[-1]
        next_node = None
        for nb in adjacency[current]:
            if nb not in visited:
                next_node = nb
                break
        if next_node is None:
            raise RuntimeError("Tour reconstruction failed: graph is not a single cycle")
        tour.append(next_node)
        visited.add(next_node)
    tour.append(0)
    return tour


def tour_edge_set(tour):
    """Convert a cycle (list of node indices, possibly closed) to a frozenset
    of sorted (i, j) tuples for set-equality comparison."""
    if tour[0] == tour[-1]:
        seq = tour[:-1]
    else:
        seq = tour
    edges = set()
    n = len(seq)
    for k in range(n):
        a, b = seq[k], seq[(k + 1) % n]
        edges.add((min(a, b), max(a, b)))
    return frozenset(edges)
