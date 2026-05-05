"""Equivariance unit tests for the native edge-expansion decoder (T1.4-code).

For every E(2) transformation `g` of the coordinates, the decoder must produce
the same selected edge set in node-index space. Distances are E(2)-invariant
and the edge-probability input does not depend on coordinates inside this
test, so the decoder reads only invariant inputs and the feasibility
projection is purely combinatorial. The selected edge set therefore must be
identical under any rigid motion of the coordinates.
"""

import os
import sys
import unittest

import numpy as np
import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
EDISCO_ROOT = os.path.dirname(THIS_DIR)
EDISCO_PKG = os.path.join(EDISCO_ROOT, "edisco")
if EDISCO_PKG not in sys.path:
    sys.path.insert(0, EDISCO_PKG)

from utils.equivariance_utils import (  # noqa: E402
    apply_random_e2_transform,
    apply_reflection,
    apply_rotation,
    apply_translation,
)
from utils.native_decoder import (  # noqa: E402
    native_edge_expansion_decode,
    tour_edge_set,
)


def _pairwise_distances(coords):
    diff = coords.unsqueeze(0) - coords.unsqueeze(1)
    return torch.norm(diff, dim=-1)


class NativeDecoderEquivarianceTests(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(20260501)
        np.random.seed(20260501)
        self.n = 25
        self.coords = torch.rand(self.n, 2) * 2.0 - 1.0
        # Edge probabilities are produced by an equivariant score net at
        # inference time, so they are themselves E(2)-invariant under
        # coordinate transforms. Here we model that by sampling a fixed
        # symmetric probability matrix that the test holds constant.
        raw = torch.rand(self.n, self.n)
        sym = 0.5 * (raw + raw.T)
        sym.diagonal().zero_()
        self.edge_probs = sym

    def _decode_for(self, coords):
        distances = _pairwise_distances(coords)
        return native_edge_expansion_decode(
            self.edge_probs, distances, return_edges=True,
        )

    def test_decoder_returns_a_hamiltonian_cycle(self):
        distances = _pairwise_distances(self.coords)
        tour = native_edge_expansion_decode(self.edge_probs, distances)
        self.assertEqual(len(tour), self.n + 1)
        self.assertEqual(tour[0], tour[-1])
        # Every node visited exactly once before the closing repetition.
        self.assertEqual(sorted(tour[:-1]), list(range(self.n)))

    def test_translation_invariance(self):
        edges = self._decode_for(self.coords)
        translated = apply_translation(
            self.coords, torch.tensor([3.7, -2.1]),
        )
        self.assertEqual(edges, self._decode_for(translated))

    def test_rotation_invariance(self):
        edges = self._decode_for(self.coords)
        for angle in [0.4, 1.7, -2.3, np.pi]:
            rotated = apply_rotation(self.coords, angle)
            self.assertEqual(
                edges, self._decode_for(rotated),
                msg=f"angle={angle:.2f} changed edge set",
            )

    def test_reflection_invariance(self):
        edges = self._decode_for(self.coords)
        for axis in ("x", "y"):
            reflected = apply_reflection(self.coords, axis=axis)
            self.assertEqual(
                edges, self._decode_for(reflected),
                msg=f"axis={axis} changed edge set",
            )

    def test_random_e2_invariance(self):
        edges = self._decode_for(self.coords)
        for _ in range(10):
            transformed, _ = apply_random_e2_transform(
                self.coords, include_reflection=True,
            )
            self.assertEqual(edges, self._decode_for(transformed))

    def test_tour_and_edge_set_agree(self):
        distances = _pairwise_distances(self.coords)
        tour = native_edge_expansion_decode(self.edge_probs, distances)
        edges_via_tour = tour_edge_set(tour)
        edges_direct = native_edge_expansion_decode(
            self.edge_probs, distances, return_edges=True,
        )
        self.assertEqual(edges_via_tour, edges_direct)


if __name__ == "__main__":
    unittest.main()
