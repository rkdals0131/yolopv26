from __future__ import annotations

import math
import unittest

import torch

from model.engine._det_geometry import _decode_anchor_relative_boxes, _make_anchor_grid


def _inverse_softplus(value: float) -> float:
    return math.log(math.exp(value) - 1.0)


class PV26DetGeometryTests(unittest.TestCase):
    def test_make_anchor_grid_stacks_feature_levels_in_query_order(self) -> None:
        anchor_points, stride_tensor = _make_anchor_grid(
            [(1, 2), (2, 1)],
            [8, 16],
            dtype=torch.float32,
            device=torch.device("cpu"),
        )

        expected_points = torch.tensor(
            [
                [4.0, 4.0],
                [12.0, 4.0],
                [8.0, 8.0],
                [8.0, 24.0],
            ],
            dtype=torch.float32,
        )
        expected_strides = torch.tensor([[8.0], [8.0], [16.0], [16.0]], dtype=torch.float32)

        self.assertTrue(torch.equal(anchor_points, expected_points))
        self.assertTrue(torch.equal(stride_tensor, expected_strides))

    def test_decode_anchor_relative_boxes_scales_softplus_distances_by_stride(self) -> None:
        anchor_points = torch.tensor([[10.0, 10.0], [20.0, 20.0]], dtype=torch.float32)
        stride_tensor = torch.tensor([[2.0], [4.0]], dtype=torch.float32)
        logits = torch.tensor(
            [
                [
                    [_inverse_softplus(0.5), _inverse_softplus(1.0), _inverse_softplus(1.5), _inverse_softplus(2.0)],
                    [_inverse_softplus(1.0), _inverse_softplus(2.0), _inverse_softplus(3.0), _inverse_softplus(4.0)],
                ]
            ],
            dtype=torch.float32,
        )

        boxes = _decode_anchor_relative_boxes(logits, anchor_points, stride_tensor)
        expected = torch.tensor(
            [
                [
                    [9.0, 8.0, 13.0, 14.0],
                    [16.0, 12.0, 32.0, 36.0],
                ]
            ],
            dtype=torch.float32,
        )

        self.assertTrue(torch.allclose(boxes, expected, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
