"""Tests of the training path.

Covers the forward CTMC kernel, the weighted training objective, the EGNN
layer equations, E(2)-invariance of the edge logits, the dense/sparse
parameter compatibility used by curriculum training, the kNN tie-breaking
rule, the noise-rate schedules and the model size.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "edisco"))
from models.egnn_encoder import EGNNEncoder, EGNNLayerDense, EGNNLayerSparse
from diffusion.categorical_diffusion import ContinuousTimeCategoricalDiffusion
from diffusion.solvers import get_solver


def _integral(t, beta_min=0.1, beta_max=1.5):
    return beta_min * t + 0.5 * (beta_max - beta_min) * t * t


@pytest.mark.parametrize("t", [0.01, 0.2, 0.5, 1.0])
@pytest.mark.parametrize("shape", [(6, 120, 120), (6, 14400)])
def test_forward_kernel_matches_closed_form(t, shape):
    torch.manual_seed(0)
    diffusion = ContinuousTimeCategoricalDiffusion(0.1, 1.5)
    x0 = (torch.rand(shape) < 0.03).float()
    xt = diffusion.sample_forward(x0, torch.full((shape[0],), t))
    expected = 0.5 - 0.5 * math.exp(-2 * _integral(t))
    flipped = (xt != x0).float()
    assert set(xt.unique().tolist()) <= {0.0, 1.0}
    # Both states flip with the same probability under the uniform-rate CTMC.
    assert abs(flipped.mean().item() - expected) < 0.01
    assert abs(flipped[x0 == 1].mean().item() - expected) < 0.03


def test_forward_kernel_limits_and_per_instance_times():
    torch.manual_seed(1)
    diffusion = ContinuousTimeCategoricalDiffusion()
    x0 = (torch.rand(3, 80, 80) < 0.05).float()
    xt = diffusion.sample_forward(x0, torch.tensor([0.0, 0.0, 1.0]))
    assert torch.equal(xt[:2], x0[:2])
    assert 0.3 < (xt[2] != x0[2]).float().mean() < 0.5
    # The stationary distribution is uniform, and t = 1 is close to it.
    assert diffusion.stay_probability(1.0) == pytest.approx(0.5 + 0.5 * math.exp(-1.6))
    # One time per variable (node-level diffusion with graphs of different sizes).
    labels = (torch.rand(500) < 0.3).float()
    times = torch.cat([torch.zeros(250), torch.ones(250)])
    noisy = diffusion.sample_forward(labels, times)
    assert torch.equal(noisy[:250], labels[:250])
    assert 0.3 < (noisy[250:] != labels[250:]).float().mean() < 0.5
    with pytest.raises(ValueError, match="diffusion times"):
        diffusion.sample_forward(x0, torch.rand(5))


def test_loss_is_one_minus_sqrt_t_weighted_cross_entropy():
    torch.manual_seed(2)
    diffusion = ContinuousTimeCategoricalDiffusion()
    x0 = (torch.rand(4, 30, 30) < 0.1).float()
    logits = torch.randn(4, 30, 30, 2)
    t = torch.tensor([0.04, 0.25, 0.49, 0.81])
    ce = F.cross_entropy(logits.reshape(-1, 2), x0.reshape(-1).long(),
                         reduction="none").reshape(4, -1).mean(1)
    expected = ((1 - t.sqrt()) * ce).mean()
    torch.testing.assert_close(diffusion.loss(x0, t, logits), expected)
    # Sparse batches (B, E) give the same value.
    torch.testing.assert_close(
        diffusion.loss(x0.reshape(4, -1), t, logits.reshape(4, -1, 2)), expected)
    assert diffusion.loss(x0, torch.ones(4), logits) == 0


def _random_e2(coords, generator):
    theta = torch.rand((), generator=generator, dtype=coords.dtype) * 2 * math.pi
    rotation = torch.stack([torch.stack([theta.cos(), -theta.sin()]),
                            torch.stack([theta.sin(), theta.cos()])])
    if torch.rand((), generator=generator) < 0.5:
        rotation = rotation @ torch.tensor([[1.0, 0.0], [0.0, -1.0]], dtype=coords.dtype)
    shift = torch.randn(2, generator=generator, dtype=coords.dtype)
    return coords @ rotation.T + shift


def _complete_edge_index(n):
    row = torch.arange(n).repeat_interleave(n)
    col = torch.arange(n).repeat(n)
    return torch.stack([row, col])


def test_dense_edge_logits_are_e2_invariant():
    torch.manual_seed(3)
    generator = torch.Generator().manual_seed(3)
    model = EGNNEncoder(n_layers=4, hidden_dim=64).double().eval()
    torch.nn.init.normal_(model.out[-1].weight, std=0.5)
    coords = torch.rand(2, 12, 2, dtype=torch.float64)
    xt = torch.randint(0, 2, (2, 12, 12)).double()
    t = torch.tensor([0.3, 0.8], dtype=torch.float64)
    with torch.no_grad():
        reference = model(coords, xt, t)
        for _ in range(8):
            moved = torch.stack([_random_e2(c, generator) for c in coords])
            assert (model(moved, xt, t) - reference).abs().max() < 1e-9


def test_sparse_edge_logits_are_e2_invariant():
    torch.manual_seed(4)
    generator = torch.Generator().manual_seed(4)
    n, k = 30, 8
    model = EGNNEncoder(n_layers=4, hidden_dim=64, sparse=True).double().eval()
    torch.nn.init.normal_(model.out[-1].weight, std=0.5)
    coords = torch.rand(n, 2, dtype=torch.float64)
    neighbours = torch.cdist(coords, coords).argsort(dim=1)[:, :k]
    edge_index = torch.stack([torch.arange(n).repeat_interleave(k), neighbours.reshape(-1)])
    xt = torch.randint(0, 2, (n * k,)).double()
    t = torch.tensor([0.4], dtype=torch.float64)
    with torch.no_grad():
        reference = model(coords, xt, t, edge_index)
        for _ in range(8):
            assert (model(_random_e2(coords, generator), xt, t, edge_index) - reference).abs().max() < 1e-9


def test_dense_and_sparse_models_share_parameters_and_outputs():
    """A dense checkpoint initializes the sparse model (curriculum training)."""
    torch.manual_seed(5)
    n = 9
    dense = EGNNEncoder(n_layers=3, hidden_dim=48, sparse=False).double().eval()
    sparse = EGNNEncoder(n_layers=3, hidden_dim=48, sparse=True).double().eval()
    torch.nn.init.normal_(dense.out[-1].weight, std=0.5)
    sparse.load_state_dict(dense.state_dict(), strict=True)
    coords = torch.rand(n, 2, dtype=torch.float64)
    xt = torch.randint(0, 2, (n, n)).double()
    t = torch.tensor([0.6], dtype=torch.float64)
    with torch.no_grad():
        dense_out = dense(coords[None], xt[None], t)[0]
        sparse_out = sparse(coords, xt.reshape(-1), t, _complete_edge_index(n))
    torch.testing.assert_close(sparse_out.reshape(n, n, 2), dense_out, atol=1e-10, rtol=0)


def test_node_update_excludes_self_messages():
    torch.manual_seed(6)
    n = 6
    layer = EGNNLayerDense(8, 8, 16).double()
    h, x = torch.randn(1, n, 8, dtype=torch.float64), torch.rand(1, n, 2, dtype=torch.float64)
    e = torch.randn(1, n, n, 8, dtype=torch.float64)
    h_ref, x_ref, _ = layer(h, x, e)
    # Changing the self-edge features e_ii must not change node or coordinate updates.
    e_changed = e.clone()
    e_changed[0, torch.arange(n), torch.arange(n)] += 5.0
    h_new, x_new, _ = layer(h, x, e_changed)
    torch.testing.assert_close(h_new, h_ref)
    torch.testing.assert_close(x_new, x_ref)


def test_edge_update_applies_time_modulation_inside_layer_norm():
    torch.manual_seed(7)
    n = 5
    layer = EGNNLayerSparse(8, 8, 16).double()
    edge_index = _complete_edge_index(n)
    h, x = torch.randn(n, 8, dtype=torch.float64), torch.rand(n, 2, dtype=torch.float64)
    e = torch.randn(n * n, 8, dtype=torch.float64)
    scale = 1 + torch.rand(n * n, 8, dtype=torch.float64)
    _, _, e_new = layer(h, x, e, edge_index, scale)
    x_diff = x[edge_index[1]] - x[edge_index[0]]
    messages = layer.message_mlp(torch.cat(
        [h[edge_index[0]], h[edge_index[1]], e, x_diff.norm(dim=-1, keepdim=True)], -1))
    expected = layer.edge_norm((e + layer.edge_mlp(torch.cat([e, messages], -1))) * scale)
    torch.testing.assert_close(e_new, expected)


def test_model_sizes_match_the_paper():
    count = lambda m: sum(p.numel() for p in m.parameters())
    # EDISCO-Full, -Medium and -Small
    assert count(EGNNEncoder(n_layers=12, hidden_dim=256)) == 5571650
    assert count(EGNNEncoder(n_layers=12, hidden_dim=128)) == 1879554
    assert count(EGNNEncoder(n_layers=8, hidden_dim=128)) == 1280770
    # Layer dimensions of the architecture appendix.
    layer = EGNNLayerDense(64, 64, 256)
    assert layer.message_mlp[0].in_features == 193
    assert layer.node_mlp[0].in_features == layer.edge_mlp[0].in_features == 320
    assert layer.coord_mlp[-1].bias is None


def test_activation_checkpointing_matches_plain_gradients():
    torch.manual_seed(8)
    n, k = 20, 6
    plain = EGNNEncoder(n_layers=3, hidden_dim=32, sparse=True).double()
    ckpt = EGNNEncoder(n_layers=3, hidden_dim=32, sparse=True, use_activation_checkpoint=True).double()
    torch.nn.init.normal_(plain.out[-1].weight, std=0.5)
    ckpt.load_state_dict(plain.state_dict())
    coords = torch.rand(n, 2, dtype=torch.float64)
    neighbours = torch.cdist(coords, coords).argsort(dim=1)[:, :k]
    edge_index = torch.stack([torch.arange(n).repeat_interleave(k), neighbours.reshape(-1)])
    xt = torch.randint(0, 2, (n * k,)).double()
    t = torch.full((n * k,), 0.5, dtype=torch.float64)
    grads = []
    for model in (plain, ckpt):
        model.train()
        model(coords, xt, t, edge_index).square().sum().backward()
        # The last layer's node and coordinate updates do not reach the output
        # head, so their parameters receive no gradient in either model.
        grads.append(torch.cat([(torch.zeros_like(p) if p.grad is None else p.grad).reshape(-1)
                                for p in model.parameters()]))
    assert grads[0].abs().max() > 0
    torch.testing.assert_close(grads[0], grads[1], atol=1e-10, rtol=1e-8)


def test_reverse_sampler_recovers_a_known_clean_state():
    """With an oracle x0 prediction every solver returns the oracle."""
    torch.manual_seed(9)
    target = (torch.rand(2, 10, 10) < 0.2).float()

    def oracle(x, t):
        p = target.clamp(1e-6, 1 - 1e-6)
        return torch.stack((torch.log1p(-p), p.log()), -1)

    for name in ("euler", "ddim", "deis", "pndm", "dpm2", "heun", "rk4"):
        heatmap = get_solver(name, 5).sample(oracle, torch.randint(0, 2, (2, 10, 10)).float(), device="cpu")
        torch.testing.assert_close(heatmap, target, atol=1e-5, rtol=0)


def test_knn_tie_breaking_by_node_index():
    pytest.importorskip("torch_geometric")
    from co_datasets.tsp_graph_dataset import knn_indices
    rng = np.random.default_rng(0)
    points = rng.random((300, 2))
    distances = np.linalg.norm(points[:, None] - points[None], axis=-1)
    brute = np.stack([np.lexsort((np.arange(300), distances[i]))[:40] for i in range(300)])
    assert np.array_equal(knn_indices(points, 40), brute)
    # A lattice has many exact ties; a 90-degree rotation keeps distances exact.
    grid = np.stack(np.meshgrid(np.arange(12.0), np.arange(12.0)), -1).reshape(-1, 2)
    rotated = grid @ np.array([[0.0, -1.0], [1.0, 0.0]]).T + np.array([7.0, -3.0])
    assert np.array_equal(knn_indices(grid, 20), knn_indices(rotated, 20))


@pytest.mark.parametrize("kind,params", [
    ("linear", dict(beta_min=0.1, beta_max=1.5)),
    ("linear", dict(beta_min=0.1, beta_max=2.0)),
    ("exponential", dict(a=0.5, b=4.0)),
    ("exponential", dict(a=0.8, b=5.0)),
    ("cosine", dict(beta_min=0.01, beta_max=5.0)),
    ("cosine", dict(beta_min=0.1, beta_max=3.0)),
])
def test_beta_schedules_integrate_their_rate_and_drive_the_kernel(kind, params):
    from diffusion.beta_schedules import BetaSchedule
    from diffusion.exact_ctmc import ExactCTMCPosterior
    schedule = BetaSchedule(kind, **params)
    grid = np.linspace(0.0, 1.0, 200001)
    rate = np.array([schedule.beta(float(u)) for u in grid[::2000]])
    assert np.all(rate > 0)
    dense_rate = schedule.beta(torch.tensor(grid, dtype=torch.float64)).numpy()
    for t in (0.1, 0.5, 1.0):
        k = int(round(t * 200000))
        numeric = float((0.5 * (dense_rate[1:k + 1] + dense_rate[:k]) * np.diff(grid[:k + 1])).sum())
        assert schedule.integral(t) == pytest.approx(numeric, rel=2e-3, abs=1e-4)
        tensor_integral = float(schedule.integral(torch.tensor(t, dtype=torch.float64)))
        assert tensor_integral == pytest.approx(schedule.integral(t), rel=1e-6)
    # forward kernel and posterior both read the same schedule
    diffusion = ContinuousTimeCategoricalDiffusion(schedule=schedule)
    torch.manual_seed(0)
    x0 = (torch.rand(4, 150, 150) < 0.05).float()
    xt = diffusion.sample_forward(x0, torch.full((4,), 0.7))
    expected = 0.5 - 0.5 * math.exp(-2 * schedule.integral(0.7))
    assert abs((xt != x0).float().mean().item() - expected) < 0.01
    posterior = ExactCTMCPosterior(schedule=schedule)
    stay, flip = posterior.transition_probs(torch.tensor(0.0, dtype=torch.float64),
                                            torch.tensor(0.7, dtype=torch.float64))
    assert float(flip) == pytest.approx(expected, abs=1e-6)
    assert get_solver("deis", 5, schedule=schedule).posterior.schedule is schedule
    assert get_solver("deis", 5).posterior.schedule.kind == "linear"


def test_farthest_insertion_labels_are_valid_tours():
    import importlib.util
    spec = importlib.util.spec_from_file_location("generate_tsp_data", ROOT / "data" / "generate_tsp_data.py")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except ImportError:
        pytest.skip("generator dependencies not installed")
    rng = np.random.default_rng(0)
    ratios = []
    for _ in range(20):
        points = rng.random((50, 2))
        tour = module.solve_farthest_insertion(points)
        assert sorted(tour) == list(range(50))
        length = np.linalg.norm(points[tour] - points[np.roll(tour, -1)], axis=1).sum()
        ratios.append(length)
    # Farthest Insertion tours on uniform TSP-50 are a few percent above optimal (about 5.7)
    assert 5.5 < np.mean(ratios) < 6.6
