"""Independent checks of the exact per-variable CTMC posterior."""

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "edisco"))
from diffusion.exact_ctmc import ExactCTMCPosterior


def transition(a, b, s, t):
    integral = 0.1 * (t - s) + 0.7 * (t * t - s * s)
    return 0.5 + (float(a == b) - 0.5) * math.exp(-2 * integral)


@pytest.mark.parametrize("s,t", [(0.0, 1.0), (0.2, 0.6), (0.8, 1.0), (0.0, 1e-6)])
@pytest.mark.parametrize("xt", [0, 1])
@pytest.mark.parametrize("prob", [0.0, 0.3, 0.5, 1.0])
def test_matches_enumerated_conditional_bridges(s, t, xt, prob):
    probabilities = []
    for xs in (0, 1):
        probabilities.append(sum(
            transition(xs, xt, s, t)
            * transition(a, xs, 0, s)
            / transition(a, xt, 0, t)
            * (prob if a == 1 else 1 - prob)
            for a in (0, 1)
        ))
    assert sum(probabilities) == pytest.approx(1, abs=1e-9)
    result = ExactCTMCPosterior().posterior_prob(
        torch.tensor([xt], dtype=torch.float64),
        torch.tensor([prob], dtype=torch.float64),
        torch.tensor(t, dtype=torch.float64),
        torch.tensor(s, dtype=torch.float64),
    )
    assert result.item() == pytest.approx(probabilities[1], abs=1e-9)


def test_final_step_recovers_clean_prediction_for_batched_edges():
    pred = torch.tensor([[0.1, 0.3, 0.9], [0.2, 0.6, 0.8]])
    xt = torch.tensor([[0., 1., 0.], [1., 0., 1.]])
    result = ExactCTMCPosterior().posterior_prob(
        xt, pred, torch.tensor([0.5, 1.0]), torch.zeros(2)
    )
    torch.testing.assert_close(result, pred)


def test_tiny_float32_interval_preserves_final_prediction():
    pred = torch.tensor([0.3, 0.7])
    result = ExactCTMCPosterior().posterior_prob(
        torch.tensor([0., 1.]), pred, torch.tensor(1e-12), torch.tensor(0.)
    )
    torch.testing.assert_close(result, pred)
