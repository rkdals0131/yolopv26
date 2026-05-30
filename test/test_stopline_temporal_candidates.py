from __future__ import annotations

import unittest

import numpy as np

from model.data.transform import compute_letterbox_transform
from tools.probe_pv26_stopline_temporal_candidates import (
    TEMPORAL_FEATURES,
    _build_temporal_candidates,
    _select_temporal_stop_lines,
    _stopline_temporal_features,
)


def _meta() -> dict[str, object]:
    transform = compute_letterbox_transform((608, 800), (608, 800))
    return {
        "raw_hw": (608, 800),
        "network_hw": (608, 800),
        "transform": transform.as_meta(),
        "sample_id": "scene_000001",
        "dataset_key": "aihub_lane_seoul",
        "image_path": "/tmp/example.jpg",
    }


class StoplineTemporalCandidateTests(unittest.TestCase):
    def test_temporal_features_are_fixed_and_finite(self) -> None:
        candidate = {"points_xy": [[100.0, 300.0], [700.0, 300.0]], "score": 0.8}
        mask = np.ones((76, 100), dtype=np.float32) * 0.5
        center = np.ones((76, 100), dtype=np.float32) * 0.4
        selector = np.ones((76, 100), dtype=np.float32) * 0.3

        features = _stopline_temporal_features(
            candidate,
            meta=_meta(),
            mask_probs=mask,
            center_probs=center,
            selector_probs=selector,
            neighbor_offset=-1,
            neighbor_rank=1,
            current_stop_lines=[],
        )

        self.assertEqual(tuple(features.keys()), TEMPORAL_FEATURES)
        self.assertTrue(np.isfinite(np.asarray(list(features.values()), dtype=np.float32)).all())
        self.assertGreater(features["temporal_neighbor_length"], 0.0)
        self.assertGreater(features["temporal_proposal_mean"], 0.0)

    def test_build_candidates_marks_oracle_positive_neighbor_line(self) -> None:
        mask = np.ones((76, 100), dtype=np.float32) * 0.5
        center = np.ones((76, 100), dtype=np.float32) * 0.4
        selector = np.ones((76, 100), dtype=np.float32) * 0.3
        neighbor_prediction = {
            "stop_lines": [{"points_xy": [[100.0, 300.0], [700.0, 300.0]], "score": 0.8}]
        }
        gt = [{"points_xy": [[100.0, 300.0], [700.0, 300.0]]}]

        candidates = _build_temporal_candidates(
            meta=_meta(),
            mask_probs=mask,
            center_probs=center,
            selector_probs=selector,
            current_stop_lines=[],
            gt_stop_lines=gt,
            neighbor_predictions=[(-1, 12, neighbor_prediction)],
            max_candidates=4,
        )

        self.assertEqual(len(candidates), 1)
        self.assertTrue(candidates[0]["is_oracle_positive"])
        self.assertEqual(candidates[0]["neighbor_dataset_index"], 12)

    def test_select_temporal_stop_lines_applies_score_and_cap(self) -> None:
        candidates = [
            {"points_xy": [[100.0, 300.0], [700.0, 300.0]], "temporal_mlp_score": 0.9, "length": 600.0},
            {"points_xy": [[100.0, 330.0], [700.0, 330.0]], "temporal_mlp_score": 0.2, "length": 600.0},
        ]

        selected = _select_temporal_stop_lines(
            candidates,
            score_key="temporal_mlp_score",
            threshold=0.5,
            top_k=2,
            max_components=1,
        )

        self.assertEqual(len(selected), 1)
        self.assertAlmostEqual(float(selected[0]["score"]), 0.9)


if __name__ == "__main__":
    unittest.main()
