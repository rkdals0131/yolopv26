from __future__ import annotations

import unittest

import numpy as np

from model.data.transform import compute_letterbox_transform
from tools.probe_pv26_stopline_crosswalk_edge_candidates import (
    CROSSWALK_EDGE_FEATURES,
    _build_crosswalk_edge_candidates,
    _crosswalk_edge_candidates_from_polygon,
    _edge_features,
    _parse_offsets,
    _select_crosswalk_edge_stop_lines,
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


class StoplineCrosswalkEdgeCandidateTests(unittest.TestCase):
    def test_parse_offsets_rejects_empty_values(self) -> None:
        self.assertEqual(_parse_offsets("0,16,32"), (0.0, 16.0, 32.0))

        with self.assertRaises(ValueError):
            _parse_offsets(" , ")

    def test_crosswalk_polygon_edges_emit_clipped_offset_candidates(self) -> None:
        crosswalk = {
            "points_xy": [[100.0, 240.0], [700.0, 240.0], [700.0, 300.0], [100.0, 300.0]],
            "score": 0.8,
        }

        candidates = _crosswalk_edge_candidates_from_polygon(crosswalk, meta=_meta(), offsets=(0.0, 32.0))

        self.assertEqual(len(candidates), 4)
        signed_sides = {float(candidate["crosswalk_edge_signed_side"]) for candidate in candidates}
        offsets = {float(candidate["crosswalk_edge_offset_px"]) for candidate in candidates}
        self.assertEqual(signed_sides, {-1.0, 1.0})
        self.assertEqual(offsets, {0.0, 32.0})
        for candidate in candidates:
            points = np.asarray(candidate["points_xy"], dtype=np.float32)
            self.assertTrue(np.isfinite(points).all())
            self.assertGreater(float(np.linalg.norm(points[-1] - points[0])), 8.0)
            self.assertGreaterEqual(float(points[:, 0].min()), 0.0)
            self.assertLessEqual(float(points[:, 0].max()), 799.0)
            self.assertGreaterEqual(float(points[:, 1].min()), 0.0)
            self.assertLessEqual(float(points[:, 1].max()), 607.0)

    def test_edge_features_are_fixed_and_finite(self) -> None:
        candidate = {"points_xy": [[100.0, 300.0], [700.0, 300.0]], "score": 0.8}
        stop_mask = np.ones((76, 100), dtype=np.float32) * 0.5
        stop_center = np.ones((76, 100), dtype=np.float32) * 0.4
        stop_selector = np.ones((76, 100), dtype=np.float32) * 0.3
        crosswalk_mask = np.ones((76, 100), dtype=np.float32) * 0.6
        crosswalk_center = np.ones((76, 100), dtype=np.float32) * 0.7

        features = _edge_features(
            candidate,
            meta=_meta(),
            stop_mask_probs=stop_mask,
            stop_center_probs=stop_center,
            stop_selector_probs=stop_selector,
            crosswalk_mask_probs=crosswalk_mask,
            crosswalk_center_probs=crosswalk_center,
            current_stop_lines=[],
            candidate_rank=1,
        )

        self.assertEqual(tuple(features.keys()), CROSSWALK_EDGE_FEATURES)
        self.assertTrue(np.isfinite(np.asarray(list(features.values()), dtype=np.float32)).all())
        self.assertGreater(features["crosswalk_edge_stop_proposal_mean"], 0.0)
        self.assertGreater(features["crosswalk_edge_endpoint_proposal_mean"], 0.0)
        self.assertAlmostEqual(features["crosswalk_edge_crosswalk_center_max"], 0.7, places=5)

    def test_build_candidates_runtime_sort_does_not_use_oracle_label(self) -> None:
        stop_mask = np.ones((76, 100), dtype=np.float32) * 0.1
        stop_center = np.ones((76, 100), dtype=np.float32) * 0.2
        stop_selector = np.ones((76, 100), dtype=np.float32) * 0.3
        baseline_prediction = {
            "stop_lines": [],
            "crosswalks": [
                {
                    "points_xy": [[100.0, 360.0], [700.0, 360.0], [700.0, 420.0], [100.0, 420.0]],
                    "score": 0.95,
                },
                {
                    "points_xy": [[100.0, 240.0], [700.0, 240.0], [700.0, 300.0], [100.0, 300.0]],
                    "score": 0.25,
                }
            ],
        }
        gt_stop_lines = [{"points_xy": [[100.0, 208.0], [700.0, 208.0]]}]

        candidates = _build_crosswalk_edge_candidates(
            meta=_meta(),
            baseline_prediction=baseline_prediction,
            gt_stop_lines=gt_stop_lines,
            stop_mask_probs=stop_mask,
            stop_center_probs=stop_center,
            stop_selector_probs=stop_selector,
            crosswalk_mask_probs=None,
            crosswalk_center_probs=None,
            offsets=(0.0, 32.0),
            max_candidates=8,
        )

        self.assertGreater(len(candidates), 1)
        self.assertTrue(any(bool(candidate["is_oracle_positive"]) for candidate in candidates))
        self.assertFalse(bool(candidates[0]["is_oracle_positive"]))

    def test_select_crosswalk_edge_stop_lines_applies_score_and_cap(self) -> None:
        candidates = [
            {
                "points_xy": [[100.0, 300.0], [700.0, 300.0]],
                "crosswalk_edge_mlp_score": 0.9,
                "crosswalk_edge_length": 600.0,
            },
            {
                "points_xy": [[100.0, 330.0], [700.0, 330.0]],
                "crosswalk_edge_mlp_score": 0.2,
                "crosswalk_edge_length": 600.0,
            },
        ]

        selected = _select_crosswalk_edge_stop_lines(
            candidates,
            score_key="crosswalk_edge_mlp_score",
            threshold=0.5,
            top_k=2,
            max_components=1,
        )

        self.assertEqual(len(selected), 1)
        self.assertAlmostEqual(float(selected[0]["score"]), 0.9)


if __name__ == "__main__":
    unittest.main()
