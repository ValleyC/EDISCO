"""Tests of the inference path: solvers, decoders and evaluation."""

import ast
from argparse import ArgumentParser
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "edisco"))
from diffusion.solvers import get_solver, get_time_schedule
from utils.native_decoder import decode_tsp_batch, tour_edge_set
from utils.cvrp_utils import decode_cvrp_greedy, decode_cvrp_nee, batched_decode_cvrp
from models.egnn_encoder import EGNNEncoder
from models.egnn_encoder_cvrp import EGNNEncoderCVRP
from utils.tsp_utils import TSPEvaluator


@pytest.mark.parametrize("name,nfe", [("euler", 6), ("ddim", 6), ("pndm", 6),
                                       ("deis", 6), ("dpm2", 10), ("heun", 10), ("rk4", 20)])
def test_network_evaluations_and_probability_output(name, nfe):
    # Single-evaluation solvers: T reverse steps plus one final clean-state prediction at t_T.
    calls = []
    def score(x, t):
        calls.append(t)
        p = torch.full_like(x, 0.3)
        return torch.stack((torch.log1p(-p), p.log()), -1)
    output = get_solver(name, 5).sample(score, torch.zeros(2, 4, 4), device="cpu")
    assert len(calls) == nfe
    torch.testing.assert_close(output, torch.full((2, 4, 4), 0.3))


@pytest.mark.parametrize("name,order", [("deis", 2), ("pndm", 4)])
def test_smoothing_bridge_calls_and_final_prediction(name, order):
    predictions, observed = [0.2, 0.3, 0.25, 0.28, 0.27, 0.26], []
    solver = get_solver(name, 5)
    calls = []
    def score(x, t):
        p = torch.full_like(x, predictions[len(calls)])
        calls.append(t)
        return torch.stack((torch.log1p(-p), p.log()), -1)
    def bridge(x, p, t, s, deterministic=False):
        assert not deterministic
        assert 0 <= s < t <= 1
        observed.append(p.clone())
        return torch.zeros_like(x)
    solver.posterior.sample = bridge
    output = solver.sample(score, torch.zeros(3), device="cpu")
    coefficients = [[1], [1.5, -0.5], [23/12, -16/12, 5/12], [55/24, -59/24, 37/24, -9/24]]
    assert len(observed) == 5
    for k, p in enumerate(observed):
        count = min(k + 1, order)
        expected = np.clip(np.dot(coefficients[count-1], predictions[k-count+1:k+1][::-1]), 0, 1)
        torch.testing.assert_close(p, torch.full_like(p, expected))
    # The decoder input is the raw clean-state prediction made at t_T = 0.
    torch.testing.assert_close(output, torch.full_like(output, predictions[-1]))
    assert len(calls) == 6 and calls[-1] == 0 and min(calls[:-1]) > 0


@pytest.mark.parametrize("schedule", ["linear", "cosine", "quadratic"])
def test_reverse_schedule_runs_from_one_to_zero(schedule):
    times = get_time_schedule(schedule, 7)
    assert times[0] == 1 and times[-1] == 0
    assert torch.all(times[:-1] > times[1:])


@pytest.mark.parametrize("decoder", ["greedy", "nee"])
@pytest.mark.parametrize("n", [3, 8, 25])
def test_dense_and_sparse_decoders_match_and_enforce_feasibility(decoder, n):
    rng = np.random.default_rng(10 + n)
    points = rng.random((n, 2))
    mask = rng.random((n, n)) < 0.25
    np.fill_diagonal(mask, False)
    p = rng.random((n, n)) * mask
    ei = np.array(np.nonzero(mask))
    dense, _ = decode_tsp_batch(p, points, decoder=decoder)
    sparse, _ = decode_tsp_batch(p[mask], points, ei, sparse_graph=True, decoder=decoder)
    assert tour_edge_set(dense[0]) == tour_edge_set(sparse[0])
    assert dense[0][0] == dense[0][-1] == 0
    assert sorted(dense[0][:-1]) == list(range(n))


def test_greedy_is_probability_only_and_nee_can_choose_different_edges():
    rng = np.random.default_rng(1)
    points, p = rng.random((10, 2)), rng.random((10, 10))
    greedy, _ = decode_tsp_batch(p, points, decoder="greedy")
    distorted, _ = decode_tsp_batch(p, points * [1000, 0.01], decoder="greedy")
    nee, _ = decode_tsp_batch(p, points, decoder="nee")
    assert greedy == distorted
    assert tour_edge_set(nee[0]) != tour_edge_set(greedy[0])


@pytest.mark.parametrize("decoder", ["greedy", "nee"])
def test_dense_batch_and_coincident_coordinates(decoder):
    rng = np.random.default_rng(2)
    p = rng.random((2, 7, 7))
    points = np.zeros((2, 7, 2))
    batch, _ = decode_tsp_batch(p, points, decoder=decoder)
    assert len(batch) == 2
    for b in range(2):
        single, _ = decode_tsp_batch(p[b], points[b], decoder=decoder)
        assert batch[b] == single[0]
        assert sorted(batch[b][:-1]) == list(range(7))


@pytest.mark.parametrize("decode", [decode_cvrp_greedy, decode_cvrp_nee])
def test_cvrp_capacity_rejection_closes_the_active_route(decode):
    p = torch.tensor([[0., .9, .2, .1], [.9, 0., .8, .7], [.2, .8, 0., .1], [.1, .7, .1, 0.]])
    demand = torch.tensor([0., 6., 6., 4.])
    routes = decode(p, torch.zeros(4, 2), demand, 10)
    # Ranked edges: (0,1) starts a route at customer 1. (1,2) would exceed the
    # capacity, so it is rejected and the depot-return edge closes route [1].
    # (1,3) then fails the degree check. (0,2), (0,3) and (2,3) form route [2, 3].
    assert routes == [[1], [2, 3]]
    assert sorted(j for route in routes for j in route) == [1, 2, 3]
    assert all(sum(demand[j] for j in route) <= 10 for route in routes)
    with pytest.raises(ValueError, match="every customer demand"):
        decode(p, torch.zeros(4, 2), demand, 5)


@pytest.mark.parametrize("decoder", ["greedy", "nee"])
def test_cvrp_vector_capacity_batch(decoder):
    p = torch.full((2, 4, 4), 0.5)
    demand = torch.tensor([[0., 6., 6., 4.], [0., 6., 6., 4.]])
    capacities = torch.tensor([10., 6.])
    results = batched_decode_cvrp(p, torch.zeros(2, 4, 2), demand, capacities, decoder)
    assert len(results) == 2
    for b, routes in enumerate(results):
        assert all(sum(demand[b, j] for j in route) <= capacities[b] for route in routes)


def test_cli_defaults_match_main_table_configuration():
    # Exercise the actual parser without importing optional trainer packages.
    source = ast.parse((ROOT / "edisco" / "train.py").read_text(encoding="utf-8"))
    function = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == "arg_parser")
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    namespace = {"ArgumentParser": ArgumentParser, "rank_zero_info": lambda *args: None}
    exec(compile(module, "train.py", "exec"), namespace)
    original = sys.argv
    try:
        sys.argv = ["train.py", "--storage_path", "."]
        args = namespace["arg_parser"]()
    finally:
        sys.argv = original
    assert args.solver_type == "deis" and args.solver_steps == 5
    assert args.decoder == "nee" and args.two_opt_iterations == 0
    assert (args.beta_schedule, args.beta_min, args.beta_max) == ("linear", 0.1, 1.5)
    assert (args.n_layers, args.hidden_dim, args.node_dim, args.edge_dim, args.time_dim) == (12, 256, 64, 64, 128)
    assert (args.coord_update_alpha, args.weight_temp) == (0.1, 10.0)
    assert (args.learning_rate, args.weight_decay, args.gradient_clip_val) == (2e-4, 1e-5, 1.0)
    assert args.lr_scheduler == "cosine-decay"


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable"))])
def test_real_encoder_sampler_decoder_smoke(device):
    torch.manual_seed(43)
    model = EGNNEncoder(n_layers=2, hidden_dim=32, node_dim=16,
                             edge_dim=16, time_dim=32, out_channels=2).to(device).eval()
    coords = torch.rand(2, 9, 2, device=device)
    calls = []
    def score(x, t):
        calls.append(t)
        return model(coords, x, torch.full((2,), t, device=device), None)
    heatmap = get_solver("deis", 5).sample(score, torch.zeros(2, 9, 9), device=device)
    assert len(calls) == 6 and heatmap.shape == (2, 9, 9)
    assert torch.isfinite(heatmap).all() and ((heatmap >= 0) & (heatmap <= 1)).all()
    for decoder in ("greedy", "nee"):
        tours, _ = decode_tsp_batch(heatmap, coords, decoder=decoder)
        assert len(tours) == 2
        assert all(sorted(tour[:-1]) == list(range(9)) for tour in tours)


def _production_method(filename, class_name, method_name, namespace):
    source = ast.parse((ROOT / 'edisco' / filename).read_text(encoding='utf-8'))
    cls = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method_name)
    module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    exec(compile(module, filename, 'exec'), namespace)
    return namespace[method_name]


def test_conditioned_cvrp_production_dispatch_passes_capacity():
    forward = _production_method('pl_cvrp_model.py', 'CVRPModel', 'forward', {})
    model = EGNNEncoderCVRP(n_layers=1, hidden_dim=16, node_dim=8,
                                       edge_dim=8, time_dim=16, default_capacity=10).eval()
    self = SimpleNamespace(capacity_conditioning=True, model=model)
    coords, demands = torch.rand(2, 4, 2), torch.tensor([[0., 2., 3., 1.]]).repeat(2, 1)
    features = torch.stack((demands, (demands == 0).float()), -1)
    capacity = torch.tensor([[10.], [20.]])
    noisy, time = torch.zeros(2, 4, 4), torch.full((2,), .5)
    actual = forward(self, coords, features, noisy, time, capacity=capacity)
    expected = model(coords, demands, capacity.flatten(), features, noisy, time)
    torch.testing.assert_close(actual, expected)
    with pytest.raises(ValueError, match='capacity must'):
        forward(self, coords, features, noisy, time)


def test_dense_evaluation_includes_every_instance_in_batch():
    evaluate = _production_method('pl_tsp_model.py', 'TSPModel', '_test_step_dense',
                                  {'np': np, 'TSPEvaluator': TSPEvaluator})
    calls = []
    def sample(coords):
        calls.append(len(coords))
        return [[0, 1, 2, 3, 0] for _ in coords], None
    self = SimpleNamespace(args=SimpleNamespace(parallel_sampling=2, sequential_sampling=1,
                                                two_opt_iterations=0),
                           sample_with_solver=sample, _refine=lambda points, tour, device: tour,
                           wall_clock=lambda: 0.0, log=lambda *args, **kwargs: None)
    square = torch.tensor([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
    coords = torch.stack((square, 2 * square))
    tours = torch.tensor([[0, 1, 2, 3, 0], [0, 1, 2, 3, 0]])
    metrics = evaluate(self, (None, coords, None, tours), 0)
    assert metrics['test/gt_cost'] == metrics['test/solved_cost'] == 6
    assert metrics['test/gap'] == 0 and calls == [4]
    assert metrics['test/time'] == 0


def test_tsplib_conversion_preserves_geometry_and_checks_the_reference(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("convert_tsplib", ROOT / "data" / "convert_tsplib.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # A 4 x 2 rectangle of unit squares: the boundary tour of length 12 is optimal.
    boundary = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (4, 1), (3, 1), (2, 1), (1, 1), (0, 1)]
    order = [0, 9, 1, 8, 2, 7, 3, 6, 4, 5]  # file order differs from tour order
    points = [boundary[k] for k in order]
    (tmp_path / "grid.tsp").write_text(
        "NAME : grid\nTYPE : TSP\nDIMENSION : 10\nEDGE_WEIGHT_TYPE : EUC_2D\nNODE_COORD_SECTION\n"
        + "".join(f"{k + 1} {100 * x} {100 * y}\n" for k, (x, y) in enumerate(points)) + "EOF\n")
    tour = [order.index(k) for k in range(10)]
    (tmp_path / "grid.opt.tour").write_text(
        "NAME : grid.opt.tour\nTYPE : TOUR\nDIMENSION : 10\nTOUR_SECTION\n"
        + "".join(f"{node + 1}\n" for node in tour) + "-1\nEOF\n")
    coords = module.read_tsp(tmp_path / "grid.tsp")
    assert module.read_tour(tmp_path / "grid.opt.tour", 10) == tour
    assert module.tsplib_length(coords, tour) == 1000
    # One common scale factor: distances keep their ratios and the long side spans [0, 1].
    scaled = module.normalize(coords)
    assert scaled.min() == 0 and scaled.max() == 1 and scaled[:, 1].max() == pytest.approx(0.25)
    opts = SimpleNamespace(input_dir=str(tmp_path), download=False, lkh_path=None, lkh_runs=10)
    line, info = module.convert_instance("grid", opts)
    assert info["reference_length"] == 1000 and info["reference_source"] == "opt.tour"
    values, closed = line.split(" output ")
    assert np.allclose(np.array(values.split(), dtype=float).reshape(-1, 2), scaled)
    assert [int(node) - 1 for node in closed.split()] == tour + tour[:1]
    with pytest.raises(FileNotFoundError):
        module.convert_instance("missing", opts)
