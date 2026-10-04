"""Tests of the CVRP and Steiner training paths."""

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "edisco"))
from models.egnn_encoder import EGNNEncoder
from models.egnn_encoder_cvrp import EGNNEncoderCVRP
from diffusion.categorical_diffusion import ContinuousTimeCategoricalDiffusion
from utils.steiner_utils import SteinerTreeEvaluator, decode_steiner_tree


def _rotation(theta, reflect=False):
    matrix = torch.tensor([[math.cos(theta), -math.sin(theta)],
                           [math.sin(theta), math.cos(theta)]], dtype=torch.float64)
    return matrix @ torch.diag(torch.tensor([1.0, -1.0], dtype=torch.float64)) if reflect else matrix


def _cvrp_inputs(batch_size=2, n_nodes=9):
    torch.manual_seed(0)
    coords = torch.rand(batch_size, n_nodes, 2, dtype=torch.float64)
    demands = torch.randint(1, 10, (batch_size, n_nodes)).double()
    demands[:, 0] = 0
    is_depot = torch.zeros(batch_size, n_nodes, dtype=torch.float64)
    is_depot[:, 0] = 1
    features = torch.stack((demands, is_depot), -1)
    capacity = torch.tensor([40.0, 15.0], dtype=torch.float64)[:batch_size]
    noisy = torch.randint(0, 2, (batch_size, n_nodes, n_nodes)).double()
    time = torch.tensor([0.3, 0.8], dtype=torch.float64)[:batch_size]
    return coords, demands, capacity, features, noisy, time


@pytest.mark.parametrize("conditioning", [True, False])
def test_cvrp_encoder_is_e2_invariant_with_and_without_conditioning(conditioning):
    torch.manual_seed(1)
    model = EGNNEncoderCVRP(n_layers=3, hidden_dim=32, node_dim=16, edge_dim=16, time_dim=32,
                                       default_capacity=40, capacity_conditioning=conditioning).double().eval()
    torch.nn.init.normal_(model.out[-1].weight, std=0.5)
    if conditioning:
        for head in model.film_head.heads:
            torch.nn.init.uniform_(head.weight, -0.1, 0.1)
    coords, demands, capacity, features, noisy, time = _cvrp_inputs()
    with torch.no_grad():
        reference = model(coords, demands, capacity, features, noisy, time)
        moved = model(coords @ _rotation(0.9, reflect=True).T + 2.5, demands, capacity, features, noisy, time)
        other_capacity = model(coords, demands, capacity * 3, features, noisy, time)
    assert reference.shape == (2, 9, 9, 2)
    assert (moved - reference).abs().max() < 1e-9
    # Only the conditioned network reads the capacity.
    assert ((other_capacity - reference).abs().max() > 1e-6) == conditioning
    assert hasattr(model, "film_head") == conditioning


def test_cvrp_activation_checkpointing_matches_plain_gradients():
    torch.manual_seed(2)
    kwargs = dict(n_layers=3, hidden_dim=32, node_dim=16, edge_dim=16, time_dim=32, default_capacity=40)
    plain = EGNNEncoderCVRP(**kwargs).double()
    ckpt = EGNNEncoderCVRP(use_activation_checkpoint=True, **kwargs).double()
    torch.nn.init.normal_(plain.out[-1].weight, std=0.5)
    for head in plain.film_head.heads:
        torch.nn.init.uniform_(head.weight, -0.1, 0.1)
    ckpt.load_state_dict(plain.state_dict())
    coords, demands, capacity, features, noisy, time = _cvrp_inputs()
    target = torch.randint(0, 2, noisy.shape).double()
    diffusion = ContinuousTimeCategoricalDiffusion()
    grads = []
    for model in (plain, ckpt):
        model.train()
        diffusion.loss(target, time, model(coords, demands, capacity, features, noisy, time)).backward()
        grads.append(torch.cat([(torch.zeros_like(p) if p.grad is None else p.grad).reshape(-1)
                                for p in model.parameters()]))
    assert grads[0].abs().max() > 0
    torch.testing.assert_close(grads[0], grads[1], atol=1e-10, rtol=1e-8)


@pytest.mark.parametrize("sparse", [False, True])
def test_terminal_indicator_is_an_invariant_node_input(sparse):
    torch.manual_seed(3)
    n = 10
    model = EGNNEncoder(n_layers=3, hidden_dim=32, sparse=sparse, node_feature_dim=1).double().eval()
    torch.nn.init.normal_(model.out[-1].weight, std=0.5)
    assert not hasattr(model, "node_embed")
    coords = torch.rand(n, 2, dtype=torch.float64)
    is_terminal = torch.zeros(n, 1, dtype=torch.float64)
    is_terminal[:6] = 1
    time = torch.tensor([0.4], dtype=torch.float64)
    moved = coords @ _rotation(2.1).T - 1.0
    if sparse:
        neighbours = torch.cdist(coords, coords).argsort(dim=1)[:, :5]
        edge_index = torch.stack([torch.arange(n).repeat_interleave(5), neighbours.reshape(-1)])
        noisy = torch.randint(0, 2, (n * 5,)).double()
        call = lambda c, f: model(c, noisy, time, edge_index, node_features=f)
    else:
        noisy = torch.randint(0, 2, (1, n, n)).double()
        call = lambda c, f: model(c[None], noisy, time, node_features=f[None])
    with torch.no_grad():
        reference = call(coords, is_terminal)
        assert (call(moved, is_terminal) - reference).abs().max() < 1e-9
        assert (call(coords, 1 - is_terminal) - reference).abs().max() > 1e-6
    with pytest.raises(ValueError, match="invariant node features"):
        model(coords[None] if not sparse else coords, noisy, time, None if not sparse else edge_index)


@pytest.mark.parametrize("mode", ["dense", "sparse", "empty"])
def test_steiner_decoder_is_feasible_hub_only_and_invariant(mode):
    rng = np.random.default_rng({"dense": 0, "sparse": 1, "empty": 2}[mode])
    evaluator = SteinerTreeEvaluator()
    for trial in range(40):
        n_terminals, n_candidates = int(rng.integers(3, 12)), int(rng.integers(0, 12))
        n = n_terminals + n_candidates
        coords = rng.random((n, 2))
        is_terminal = np.r_[np.ones(n_terminals), np.zeros(n_candidates)]
        probs = rng.random((n, n))
        if mode == "sparse":
            probs *= rng.random((n, n)) < 0.15
        elif mode == "empty":
            probs[:] = 0
        adjacency, length = decode_steiner_tree(probs, coords, is_terminal)
        valid, message = evaluator.validate_tree(adjacency, n_terminals)
        assert valid, message
        assert np.array_equal(adjacency, adjacency.T)
        assert length == pytest.approx(evaluator.compute_tree_length(coords, adjacency))
        # Steiner points are kept only as hubs of degree three or more.
        steiner_degree = adjacency.sum(0)[n_terminals:]
        assert not np.any((steiner_degree == 1) | (steiner_degree == 2))
        theta = rng.random() * 2 * math.pi
        rotation = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
        if trial % 2:
            rotation = rotation @ np.diag([1.0, -1.0])
        moved, _ = decode_steiner_tree(probs, coords @ rotation.T + rng.normal(size=2), is_terminal)
        assert np.array_equal(adjacency, moved)


def test_steiner_decoder_follows_the_heatmap():
    # Three terminals around one Steiner point: a confident star heatmap is reproduced,
    # an uninformative heatmap falls back to joining terminals directly.
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.9], [0.5, 0.3]])
    is_terminal = np.array([1, 1, 1, 0])
    star = np.zeros((4, 4))
    star[[0, 1, 2], 3] = 0.99
    adjacency, _ = decode_steiner_tree(star, coords, is_terminal)
    assert adjacency[3].sum() == 3 and adjacency[:3, :3].sum() == 0
    adjacency, _ = decode_steiner_tree(np.zeros((4, 4)), coords, is_terminal)
    assert adjacency[3].sum() == 0 and adjacency.sum() == 4
