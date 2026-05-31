from __future__ import annotations

import unittest

import numpy as np

from model.data.transform import compute_letterbox_transform
from tools.probe_pv26_stopline_lane_pair_candidates import (
    LANE_PAIR_FEATURES,
    _build_lane_pair_candidates,
    _interpolate_lane_at_y,
    _lane_pair_candidates_from_lanes,
    _lane_pair_features,
    _parse_y_fractions,
    _select_lane_pair_stop_lines,
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


class StoplineLanePairCandidateTests(unittest.TestCase):
    def test_parse_y_fractions_rejects_empty_or_out_of_range_values(self) -> None:
        self.assertEqual(_parse_y_fractions("0.65,0.75,0.85"), (0.65, 0.75, 0.85))

        with self.assertRaises(ValueError):
            _parse_y_fractions(" , ")

        with self.assertRaises(ValueError):
            _parse_y_fractions("1.2")

    def test_interpolate_lane_at_y_returns_x_and_unit_tangent(self) -> None:
        points = np.asarray([[100.0, 100.0], [140.0, 300.0], [180.0, 500.0]], dtype=np.float32)

        hit = _interpolate_lane_at_y(points, 300.0)

        self.assertIsNotNone(hit)
        assert hit is not None
        x_value, tangent = hit
        self.assertAlmostEqual(x_value, 140.0, places=4)
        self.assertAlmostEqual(float(np.linalg.norm(tangent)), 1.0, places=5)

    def test_lane_pair_candidates_connect_predicted_lane_pairs(self) -> None:
        lanes = [
            {"points_xy": [[100.0, 100.0], [140.0, 500.0]], "score": 0.9},
            {"points_xy": [[700.0, 100.0], [660.0, 500.0]], "score": 0.8},
        ]

        candidates = _lane_pair_candidates_from_lanes(
            lanes,
            meta=_meta(),
            y_fractions=(0.75,),
            max_lanes=4,
            min_overlap_px=32.0,
            min_gap_px=24.0,
            max_gap_frac=0.95,
        )

        self.assertEqual(len(candidates), 1)
        points = np.asarray(candidates[0]["points_xy"], dtype=np.float32)
        self.assertTrue(np.isfinite(points).all())
        self.assertLess(points[0, 0], points[1, 0])
        self.assertAlmostEqual(float(points[0, 1]), float(points[1, 1]), places=4)
        self.assertGreater(float(candidates[0]["lane_pair_gap_px"]), 24.0)

    def test_lane_pair_features_are_fixed_and_finite(self) -> None:
        candidate = {
            "points_xy": [[120.0, 420.0], [680.0, 420.0]],
            "score": 0.8,
            "lane_pair_left_rank": 1,
            "lane_pair_right_rank": 2,
            "lane_pair_y_fraction": 0.75,
            "lane_pair_gap_px": 560.0,
            "lane_pair_overlap_px": 400.0,
            "lane_pair_lane_score_mean": 0.85,
            "lane_pair_lane_score_min": 0.8,
            "lane_pair_tangent_cos_abs": 0.95,
            "lane_pair_axis_dot_tangent_abs_mean": 0.1,
        }
        stop_mask = np.ones((76, 100), dtype=np.float32) * 0.5
        stop_center = np.ones((76, 100), dtype=np.float32) * 0.4
        stop_selector = np.ones((76, 100), dtype=np.float32) * 0.3
        lane_center = np.ones((76, 100), dtype=np.float32) * 0.6
        lane_support = np.ones((76, 100), dtype=np.float32) * 0.7

        features = _lane_pair_features(
            candidate,
            meta=_meta(),
            stop_mask_probs=stop_mask,
            stop_center_probs=stop_center,
            stop_selector_probs=stop_selector,
            lane_centerline_probs=lane_center,
            lane_support_probs=lane_support,
            current_stop_lines=[],
            candidate_rank=1,
        )

        self.assertEqual(tuple(features.keys()), LANE_PAIR_FEATURES)
        self.assertTrue(np.isfinite(np.asarray(list(features.values()), dtype=np.float32)).all())
        self.assertGreater(features["lane_pair_stop_proposal_mean"], 0.0)
        self.assertAlmostEqual(features["lane_pair_lane_endpoint_support_mean"], 0.7, places=5)

    def test_build_candidates_runtime_sort_does_not_use_oracle_label(self) -> None:
        stop_mask = np.ones((76, 100), dtype=np.float32) * 0.1
        stop_center = np.ones((76, 100), dtype=np.float32) * 0.2
        stop_selector = np.ones((76, 100), dtype=np.float32) * 0.3
        baseline_prediction = {
            "stop_lines": [],
            "lanes": [
                {"points_xy": [[120.0, 100.0], [140.0, 500.0]], "score": 0.9},
                {"points_xy": [[680.0, 100.0], [660.0, 500.0]], "score": 0.9},
            ],
        }
        gt_stop_lines = [{"points_xy": [[120.0, 200.0], [680.0, 200.0]]}]

        candidates = _build_lane_pair_candidates(
            meta=_meta(),
            baseline_prediction=baseline_prediction,
            gt_stop_lines=gt_stop_lines,
            stop_mask_probs=stop_mask,
            stop_center_probs=stop_center,
            stop_selector_probs=stop_selector,
            lane_centerline_probs=None,
            lane_support_probs=None,
            y_fractions=(0.25, 0.75),
            max_lanes=4,
            min_overlap_px=32.0,
            min_gap_px=24.0,
            max_gap_frac=0.95,
            max_candidates=4,
        )

        self.assertEqual(len(candidates), 2)
        self.assertTrue(any(bool(candidate["is_oracle_positive"]) for candidate in candidates))
        self.assertFalse(bool(candidates[0]["is_oracle_positive"]))

    def test_select_lane_pair_stop_lines_applies_score_and_cap(self) -> None:
        candidates = [
            {
                "points_xy": [[100.0, 300.0], [700.0, 300.0]],
                "lane_pair_mlp_score": 0.9,
                "lane_pair_length": 600.0,
            },
            {
                "points_xy": [[100.0, 330.0], [700.0, 330.0]],
                "lane_pair_mlp_score": 0.2,
                "lane_pair_length": 600.0,
            },
        ]

        selected = _select_lane_pair_stop_lines(
            candidates,
            score_key="lane_pair_mlp_score",
            threshold=0.5,
            top_k=2,
            max_components=1,
        )

        self.assertEqual(len(selected), 1)
        self.assertAlmostEqual(float(selected[0]["score"]), 0.9)


if __name__ == "__main__":
    unittest.main()
