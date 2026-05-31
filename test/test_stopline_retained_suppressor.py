from __future__ import annotations

import argparse
import unittest

import numpy as np
import torch

from tools.probe_pv26_stopline_retained_suppressor import (
    FEATURE_MAP_KEYS,
    _apply_suppressor,
    _assign_stopline_labels,
    _line_feature_vector,
)


def _meta() -> dict[str, object]:
    return {
        "raw_hw": (100, 200),
        "network_hw": (100, 200),
        "transform": {
            "scale": 1.0,
            "pad_left": 0,
            "pad_top": 0,
            "pad_right": 0,
            "pad_bottom": 0,
            "resized_hw": (100, 200),
        },
    }


class _FixedSuppressor(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("feature_mean", torch.zeros(1, dtype=torch.float32))
        self.register_buffer("feature_std", torch.ones(1, dtype=torch.float32))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features[:, 0]


class StoplineRetainedSuppressorTests(unittest.TestCase):
    def test_assign_stopline_labels_keeps_one_match_per_gt(self) -> None:
        gt = [{"points_xy": [[10.0, 20.0], [90.0, 20.0]]}]
        lines = [
            {"points_xy": [[10.0, 20.0], [90.0, 20.0]]},
            {"points_xy": [[12.0, 20.0], [92.0, 20.0]]},
            {"points_xy": [[10.0, 80.0], [90.0, 80.0]]},
        ]

        labels = _assign_stopline_labels(lines, gt, match_threshold=40.0)

        self.assertEqual(labels, [1, 0, 0])

    def test_apply_suppressor_removes_candidates_without_adding_lines(self) -> None:
        baseline = [
            {
                "stop_lines": [
                    {"points_xy": [[0.0, 0.0], [10.0, 0.0]], "score": 0.8},
                    {"points_xy": [[0.0, 10.0], [10.0, 10.0]], "score": 0.7},
                ],
                "lanes": [],
                "crosswalks": [],
            }
        ]
        examples = [
            {"features": np.asarray([2.0], dtype=np.float32), "sample_index": 0, "line_index": 0, "label": 1},
            {"features": np.asarray([-2.0], dtype=np.float32), "sample_index": 0, "line_index": 1, "label": 0},
        ]

        suppressed, rows = _apply_suppressor(
            examples=examples,
            baseline_predictions=baseline,
            model=_FixedSuppressor(),
            args=argparse.Namespace(keep_threshold=0.5),
            device="cpu",
        )

        self.assertEqual(len(suppressed[0]["stop_lines"]), 1)
        self.assertEqual(suppressed[0]["stop_lines"][0]["points_xy"], [[0.0, 0.0], [10.0, 0.0]])
        self.assertEqual([row["keep"] for row in rows], [1, 0])

    def test_line_feature_vector_is_finite_and_fixed_width(self) -> None:
        predictions = {
            key: torch.full((1, 1, 10, 10), 0.25, dtype=torch.float32)
            for key in FEATURE_MAP_KEYS
        }
        line = {"points_xy": [[20.0, 50.0], [180.0, 50.0]], "score": 0.9}

        vector = _line_feature_vector(
            line,
            predictions=predictions,
            sample_index=0,
            meta=_meta(),
            rank=0,
            candidate_count=2,
        )

        self.assertEqual(vector.shape, (26 + len(FEATURE_MAP_KEYS) * 7,))
        self.assertTrue(np.isfinite(vector).all())


if __name__ == "__main__":
    unittest.main()
