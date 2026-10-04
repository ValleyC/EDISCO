"""Tests of the CVRP edge-expansion decoder, sparse CVRP and partition diffusion."""

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "edisco"))
from models.egnn_encoder_cvrp import EGNNEncoderCVRP
from utils.cvrp_utils import CVRPEvaluator, decode_cvrp_greedy, decode_cvrp_nee
from utils.partition_utils import (
    customer_affinity, distance_affinity, kmeans_partition, num_clusters,
    partition_to_routes, rebalance_capacity, solve_sub_tsp, spectral_partition,
)


def _instance(rng, n_customers, capacity):
    coords = torch.tensor(rng.random((n_customers + 1, 2)))
    demands = torch.tensor(np.r_[0, rng.integers(1, 10, n_customers)], dtype=torch.float64)
    return coords, demands, float(capacity)


def _routes_adjacency(routes, n_nodes):
    adjacency = torch.zeros(n_nodes, n_nodes)
    for route in routes:
        path = [0] + route + [0]
        for a, b in zip(path[:-1], path[1:]):
            adjacency[a, b] = 1.0
    return adjacency


def _assert_feasible(routes, demands, capacity):
    visited = sorted(c for route in routes for c in route)
    assert visited == list(range(1, len(demands)))
    assert all(sum(float(demands[c]) for c in route) <= capacity + 1e-9 for route in routes)


@pytest.mark.parametrize("decode", [decode_cvrp_greedy, decode_cvrp_nee])
@pytest.mark.parametrize("capacity", [9, 20, 60, 1000])
def test_cvrp_decoder_is_feasible_for_arbitrary_heatmaps(decode, capacity):
    rng = np.random.default_rng(capacity)
    for _ in range(25):
        n = int(rng.integers(1, 30))
        coords, demands, q = _instance(rng, n, capacity)
        probs = torch.tensor(rng.random((n + 1, n + 1)))
        if rng.random() < 0.3:
            probs = torch.zeros_like(probs)  # no information: single-customer routes
        _assert_feasible(decode(probs, coords, demands, q), demands, q)


@pytest.mark.parametrize("decode", [decode_cvrp_greedy, decode_cvrp_nee])
def test_cvrp_decoder_recovers_a_reference_solution(decode):
    rng = np.random.default_rng(1)
    coords, demands, capacity = _instance(rng, 12, 30)
    # any feasible solution, written as a confident heatmap, is reproduced
    reference = decode(torch.tensor(rng.random((13, 13))), coords, demands, capacity)
    heatmap = _routes_adjacency(reference, 13) * 0.98 + 0.001
    decoded = decode(heatmap, coords, demands, capacity)
    as_sets = lambda routes: sorted(tuple(min(r, r[::-1])) for r in routes)
    assert as_sets(decoded) == as_sets(reference)


def test_cvrp_decoder_is_e2_invariant_and_matches_sparse_input():
    rng = np.random.default_rng(2)
    coords, demands, capacity = _instance(rng, 20, 25)
    n = 21
    mask = torch.tensor(rng.random((n, n)) < 0.3)
    mask.fill_diagonal_(False)
    probs = torch.tensor(rng.random((n, n))) * mask
    dense = decode_cvrp_nee(probs, coords, demands, capacity)
    # a rigid motion of the coordinates leaves the routes unchanged
    theta = 0.7
    rotation = torch.tensor([[math.cos(theta), math.sin(theta)], [math.sin(theta), -math.cos(theta)]],
                            dtype=torch.float64)
    assert decode_cvrp_nee(probs, coords @ rotation.T + 3.0, demands, capacity) == dense
    # the sparse edge list holding the same nonzero entries gives the same routes
    edge_index = mask.nonzero().T
    sparse = decode_cvrp_nee(probs[mask], coords, demands, capacity, edge_index=edge_index)
    assert sparse == dense
    _assert_feasible(sparse, demands, capacity)


def test_cvrp_sparse_and_dense_encoders_share_parameters_and_outputs():
    """A dense CVRP checkpoint initializes the sparse model used above N = 100."""
    torch.manual_seed(3)
    kwargs = dict(n_layers=3, hidden_dim=32, node_dim=16, edge_dim=16, time_dim=32, default_capacity=40)
    dense = EGNNEncoderCVRP(**kwargs).double().eval()
    sparse = EGNNEncoderCVRP(sparse=True, **kwargs).double().eval()
    torch.nn.init.normal_(dense.out[-1].weight, std=0.5)
    for head in dense.film_head.heads:
        torch.nn.init.uniform_(head.weight, -0.1, 0.1)
    sparse.load_state_dict(dense.state_dict(), strict=True)

    batch_size, n = 2, 7
    coords = torch.rand(batch_size, n, 2, dtype=torch.float64)
    demands = torch.randint(1, 10, (batch_size, n)).double()
    demands[:, 0] = 0
    is_depot = torch.zeros(batch_size, n, dtype=torch.float64)
    is_depot[:, 0] = 1
    features = torch.stack((demands, is_depot), -1)
    capacity = torch.tensor([40.0, 17.0], dtype=torch.float64)
    noisy = torch.randint(0, 2, (batch_size, n, n)).double()
    time = torch.tensor([0.2, 0.9], dtype=torch.float64)

    row = torch.arange(n).repeat_interleave(n)
    col = torch.arange(n).repeat(n)
    edge_index = torch.cat([torch.stack([row, col]) + b * n for b in range(batch_size)], dim=1)
    node_batch = torch.arange(batch_size).repeat_interleave(n)
    with torch.no_grad():
        dense_out = dense(coords, demands, capacity, features, noisy, time)
        sparse_out = sparse(coords.reshape(-1, 2), demands.reshape(-1), capacity, features.reshape(-1, 2),
                            noisy.reshape(-1), time, edge_index, node_batch)
    torch.testing.assert_close(sparse_out.reshape(batch_size, n, n, 2), dense_out, atol=1e-10, rtol=0)


def test_partition_target_and_affinity():
    pytest.importorskip("torch_geometric")
    from co_datasets.cvrp_graph_dataset import CVRPGraphDataset
    partition = CVRPGraphDataset._create_partition_from_routes([[1, 3], [2], [4, 5, 6]], 7)
    assert torch.equal(partition, partition.T) and partition.diagonal().sum() == 0
    assert partition[0].sum() == 0  # the depot belongs to no cluster
    assert partition[1, 3] == 1 and partition[4, 6] == 1 and partition[1, 2] == 0 and partition[2].sum() == 0
    affinity = customer_affinity(partition, 7)
    assert affinity.shape == (6, 6) and affinity[0, 2] == 1 and affinity[0, 1] == 0
    edge_index = partition.nonzero().T
    assert np.array_equal(customer_affinity(partition[partition > 0], 7, edge_index), affinity)


def test_projection_recovers_clean_partition_and_respects_capacity():
    pytest.importorskip("sklearn")
    rng = np.random.default_rng(4)
    n_customers, capacity = 40, 30.0
    demands = rng.integers(1, 10, n_customers).astype(float)
    # a feasible reference partition with the minimum number of clusters
    order, clusters, load = rng.permutation(n_customers), [[]], 0.0
    for c in order:
        if load + demands[c] > capacity:
            clusters.append([])
            load = 0.0
        clusters[-1].append(int(c))
        load += demands[c]
    affinity = np.zeros((n_customers, n_customers))
    for cluster in clusters:
        affinity[np.ix_(cluster, cluster)] = 1.0
    np.fill_diagonal(affinity, 0.0)

    labels = spectral_partition(affinity, len(clusters), seed=0)
    projected = rebalance_capacity(labels, affinity, demands, capacity)
    assert sorted(map(sorted, projected)) == sorted(map(sorted, clusters))

    # noisy affinities and an optimistic cluster count still give feasible clusters
    noisy = np.clip(affinity + rng.normal(0, 0.6, affinity.shape), 0, 1)
    noisy = (noisy + noisy.T) / 2
    k = num_clusters(demands, capacity)
    projected = rebalance_capacity(spectral_partition(noisy, k, seed=0), noisy, demands, capacity)
    assert sorted(c for cluster in projected for c in cluster) == list(range(n_customers))
    assert all(demands[cluster].sum() <= capacity for cluster in projected)


def test_projection_is_coordinate_free_and_kmeans_ablation_runs():
    pytest.importorskip("sklearn")
    rng = np.random.default_rng(5)
    coords = rng.random((31, 2))
    demands = rng.integers(1, 10, 30).astype(float)
    capacity = 25.0
    k = num_clusters(demands, capacity)
    labels = kmeans_partition(coords, k, seed=0)
    clusters = rebalance_capacity(labels, distance_affinity(coords), demands, capacity)
    assert all(demands[cluster].sum() <= capacity for cluster in clusters)
    routes = partition_to_routes(clusters, coords)
    assert sorted(c for route in routes for c in route) == list(range(1, 31))
    # a rigid motion changes neither the distance affinity nor the sub-TSP order
    theta = 1.3
    rotation = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
    moved = coords @ rotation.T + np.array([2.0, -1.0])
    np.testing.assert_allclose(distance_affinity(moved), distance_affinity(coords), atol=1e-9)
    assert partition_to_routes(clusters, moved) == routes
    # total length of the decoded routes is finite and positive
    assert 0 < CVRPEvaluator.compute_total_distance(coords, routes) < np.inf


def test_sub_tsp_uses_the_tsp_score_network():
    from models.egnn_encoder import EGNNEncoder
    torch.manual_seed(6)
    model = EGNNEncoder(n_layers=2, hidden_dim=32, sparse=False).eval()
    coords = np.random.default_rng(6).random((9, 2))
    order = solve_sub_tsp(coords, model, solver_type="deis", solver_steps=5)
    assert sorted(order) == list(range(1, 9))
    assert solve_sub_tsp(coords[:3]) == [1, 2] and solve_sub_tsp(coords[:2]) == [1]
