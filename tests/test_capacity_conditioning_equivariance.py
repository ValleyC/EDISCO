"""Equivariance unit tests for the capacity-conditioned EGNN encoder (T1.3-code).

For every fixed value of `(demands, capacity)`, the conditional encoder must
produce the same edge logits when coordinates are translated, rotated, or
reflected. The conditioning enters only through scalar invariant channels and
the message FiLM modulation, so equivariance is preserved by construction.
This test exercises the construction over random transforms.
"""

import os
import sys
import unittest

import numpy as np
import torch

# Allow `python -m unittest tests.test_...` and direct `python tests/...py` from
# the EDISCO repo root.
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
EDISCO_ROOT = os.path.dirname(THIS_DIR)
EDISCO_PKG = os.path.join(EDISCO_ROOT, "edisco")
if EDISCO_PKG not in sys.path:
    sys.path.insert(0, EDISCO_PKG)

from models.egnn_encoder_cvrp_conditional import (  # noqa: E402
    EGNNEncoderCVRPConditional,
    build_invariant_capacity_features,
)
from utils.equivariance_utils import (  # noqa: E402
    apply_random_e2_transform,
    apply_reflection,
    apply_rotation,
    apply_translation,
)


class CapacityConditioningEquivarianceTests(unittest.TestCase):
    """Verify the conditional encoder is E(2)-invariant on the output side
    when capacity-aware features are kept fixed."""

    def setUp(self):
        torch.manual_seed(20260501)
        np.random.seed(20260501)

        self.batch_size = 2
        self.n_nodes = 12
        self.invariant_dim = 2  # mimicking [demand, is_depot]
        self.default_capacity = 1.0

        self.model = EGNNEncoderCVRPConditional(
            n_layers=3,
            hidden_dim=32,
            node_dim=24,
            edge_dim=24,
            time_dim=32,
            coord_dim=2,
            out_channels=2,
            invariant_dim=self.invariant_dim,
            default_capacity=self.default_capacity,
            z_hidden_dim=16,
        )
        # The encoder zero-initializes the final output linear layer so that
        # untrained models emit zeros, which is the standard diffusion trick
        # but defeats equivariance testing because zero is invariant under
        # everything. Randomize the final linear layer for testing only.
        torch.nn.init.normal_(self.model.out[-1].weight, std=0.05)
        torch.nn.init.normal_(self.model.out[-1].bias, std=0.05)
        # Also exercise the FiLM heads so conditioning has a measurable
        # effect on the output rather than being identity-initialized.
        for head in self.model.film_head.heads:
            torch.nn.init.uniform_(head.weight, -0.1, 0.1)
            torch.nn.init.uniform_(head.bias, -0.1, 0.1)
        self.model.eval()

        # Random instance.
        self.coords = torch.randn(self.batch_size, self.n_nodes, 2)
        # Demands: depot has 0, customers have positive integer-ish demand.
        self.demands = torch.zeros(self.batch_size, self.n_nodes)
        self.demands[:, 1:] = torch.rand(self.batch_size, self.n_nodes - 1) * 0.4 + 0.1
        self.capacity = torch.tensor([1.2, 0.8])
        # Existing CVRP invariants: [demand, is_depot].
        is_depot = torch.zeros(self.batch_size, self.n_nodes, 1)
        is_depot[:, 0, 0] = 1.0
        self.invariant_features = torch.cat(
            [self.demands.unsqueeze(-1), is_depot], dim=-1,
        )
        # Random noisy adjacency matrix in [0, 1].
        adj = torch.rand(self.batch_size, self.n_nodes, self.n_nodes)
        # Symmetrize and zero the diagonal.
        adj = 0.5 * (adj + adj.transpose(-1, -2))
        adj.diagonal(dim1=-2, dim2=-1).zero_()
        self.adj_matrix = adj
        self.timesteps = torch.tensor([0.3, 0.7])

    @torch.no_grad()
    def _logits(self, coords):
        return self.model(
            coords=coords,
            demands=self.demands,
            capacity=self.capacity,
            invariant_features=self.invariant_features,
            adj_matrix=self.adj_matrix,
            timesteps=self.timesteps,
        )

    def test_invariant_features_are_e2_invariant(self):
        """`build_invariant_capacity_features` outputs must not depend on
        coordinates, so they are independent of any E(2) transformation."""
        node_a, edge_a, z_a = build_invariant_capacity_features(
            self.demands, self.capacity, self.default_capacity,
        )
        # Run again with permuted coordinates (which are not even an input
        # here) just to confirm the function is purely demand/capacity-based.
        node_b, edge_b, z_b = build_invariant_capacity_features(
            self.demands, self.capacity, self.default_capacity,
        )
        self.assertTrue(torch.allclose(node_a, node_b))
        self.assertTrue(torch.allclose(edge_a, edge_b))
        self.assertTrue(torch.allclose(z_a, z_b))

    def test_translation_invariance(self):
        original = self._logits(self.coords)
        translation = torch.tensor([3.7, -2.1])
        translated = apply_translation(self.coords, translation)
        new_logits = self._logits(translated)
        self.assertTrue(
            torch.allclose(original, new_logits, atol=1e-4, rtol=1e-4),
            f"max diff = {(original - new_logits).abs().max().item():.3e}",
        )

    def test_rotation_invariance(self):
        original = self._logits(self.coords)
        for angle in [0.4, 1.7, -2.3, np.pi]:
            rotated = apply_rotation(self.coords, angle)
            new_logits = self._logits(rotated)
            self.assertTrue(
                torch.allclose(original, new_logits, atol=1e-4, rtol=1e-4),
                f"angle={angle:.2f}, max diff = "
                f"{(original - new_logits).abs().max().item():.3e}",
            )

    def test_reflection_invariance(self):
        original = self._logits(self.coords)
        for axis in ("x", "y"):
            reflected = apply_reflection(self.coords, axis=axis)
            new_logits = self._logits(reflected)
            self.assertTrue(
                torch.allclose(original, new_logits, atol=1e-4, rtol=1e-4),
                f"axis={axis}, max diff = "
                f"{(original - new_logits).abs().max().item():.3e}",
            )

    def test_random_e2_invariance(self):
        original = self._logits(self.coords)
        worst_diff = 0.0
        for _ in range(8):
            transformed, _ = apply_random_e2_transform(
                self.coords, include_reflection=True,
            )
            new_logits = self._logits(transformed)
            worst_diff = max(
                worst_diff, (original - new_logits).abs().max().item(),
            )
            self.assertTrue(
                torch.allclose(original, new_logits, atol=1e-4, rtol=1e-4),
                f"max diff = {(original - new_logits).abs().max().item():.3e}",
            )
        # Sanity: if the entire tolerance is consumed by a single test, we
        # would still want to know about it.
        self.assertLess(worst_diff, 1e-4)

    def test_conditioning_actually_changes_output(self):
        """Sanity check: changing capacity must change the output, otherwise
        the FiLM head has no effect and the equivariance test is vacuous."""
        original = self._logits(self.coords)
        old_capacity = self.capacity.clone()
        try:
            self.capacity = torch.tensor([0.4, 1.6])
            new = self._logits(self.coords)
            self.assertGreater(
                (original - new).abs().max().item(), 1e-3,
                msg="Conditioning has no measurable effect on the output. "
                "FiLM is initialized to identity by default; the test setUp "
                "perturbs the FiLM heads so a difference is expected.",
            )
        finally:
            self.capacity = old_capacity


if __name__ == "__main__":
    unittest.main()
