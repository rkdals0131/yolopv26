from __future__ import annotations

import unittest

import numpy as np
import torch

from tools.probe_pv26_stopline_dense_map_set_decoder import (
    _best_assignment,
    _canonical_segment,
    _set_decoder_loss,
)


class StoplineDenseMapSetDecoderTests(unittest.TestCase):
    def test_canonical_segment_orders_endpoints(self) -> None:
        segment = np.asarray([[0.8, 0.4], [0.2, 0.3]], dtype=np.float32)

        canonical = _canonical_segment(segment)

        np.testing.assert_allclose(canonical[0], np.asarray([0.2, 0.3], dtype=np.float32))
        np.testing.assert_allclose(canonical[1], np.asarray([0.8, 0.4], dtype=np.float32))

    def test_best_assignment_minimizes_total_cost(self) -> None:
        cost = torch.tensor(
            [
                [5.0, 1.0],
                [1.0, 5.0],
                [2.0, 2.0],
            ],
            dtype=torch.float32,
        )

        assignment = _best_assignment(cost, 2)

        self.assertEqual(assignment, [(1, 0), (0, 1)])

    def test_set_decoder_loss_is_finite_with_empty_and_positive_targets(self) -> None:
        logits = torch.zeros((2, 2), dtype=torch.float32, requires_grad=True)
        points = torch.full((2, 2, 2, 2), 0.5, dtype=torch.float32, requires_grad=True)
        targets = [
            torch.empty((0, 2, 2), dtype=torch.float32),
            torch.tensor([[[0.2, 0.3], [0.8, 0.3]]], dtype=torch.float32),
        ]

        loss = _set_decoder_loss(logits, points, targets, pos_weight=2.0)

        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()
        self.assertIsNotNone(logits.grad)
        self.assertIsNotNone(points.grad)


if __name__ == "__main__":
    unittest.main()
