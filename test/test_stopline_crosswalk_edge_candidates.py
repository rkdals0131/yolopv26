from __future__ import annotations

import unittest

import numpy as np

from tools.probe_pv26_stopline_crosswalk_edge_candidates import (
    CROSSWALK_EDGE_FEATURES,
    _build_crosswalk_edge_candidates,
    _crosswalk_axes,
    _crosswalk_edge_candidates_from_crosswalks,
    _crosswalk_edge_features,
)


def _identity_meta() -> dict[str, object]:
    return {
        "raw_hw": (120, 160),
        "network_hw": (120, 160),
        "transform": {
            "scale": 1.0,
            "pad_left": 0,
            "pad_top": 0,
            "pad_right": 0,
            "pad_bottom": 0,
            "resized_hw": (120, 160),
        },
    }


class StoplineCrosswalkEdgeCandidateTests(unittest.TestCase):
    def test_crosswalk_axes_finds_long_axis_and_width(self) -> None:
        points = np.asarray(
            [
                [20.0, 40.0],
                [140.0, 40.0],
                [140.0, 60.0],
                [20.0, 60.0],
            ],
            dtype=np.float32,
        )

        axes = _crosswalk_axes(points)

        self.assertIsNotNone(axes)
        assert axes is not None
        self.assertGreater(float(axes["half_length"]), 50.0)
        self.assertGreater(float(axes["half_width"]), 8.0)
        self.assertGreater(abs(float(axes["axis"][0])), 0.9)

    def test_crosswalk_edge_candidates_emit_both_sides(self) -> None:
        crosswalks = [
            {
                "score": 0.9,
                "points_xy": [
                    [20.0, 40.0],
                    [140.0, 40.0],
                    [140.0, 60.0],
                    [20.0, 60.0],
                ],
            }
        ]

        candidates = _crosswalk_edge_candidates_from_crosswalks(
            crosswalks,
            meta=_identity_meta(),
            offsets=(8.0,),
            length_scales=(1.0,),
            max_crosswalks=1,
            max_candidates=8,
        )

        self.assertEqual(len(candidates), 2)
        ys = sorted(float(np.asarray(candidate["points_xy"], dtype=np.float32)[:, 1].mean()) for candidate in candidates)
        self.assertLess(ys[0], 40.0)
        self.assertGreater(ys[1], 60.0)
        self.assertEqual({float(candidate["crosswalk_edge_side"]) for candidate in candidates}, {-1.0, 1.0})

    def test_crosswalk_edge_features_are_finite(self) -> None:
        candidate = {
            "points_xy": [[20.0, 32.0], [140.0, 32.0]],
            "score": 0.9,
            "crosswalk_edge_side": -1.0,
            "crosswalk_edge_offset_px": 8.0,
            "crosswalk_edge_length_scale": 1.0,
            "crosswalk_edge_crosswalk_rank": 1,
            "crosswalk_edge_crosswalk_length_px": 120.0,
            "crosswalk_edge_crosswalk_width_px": 20.0,
        }
        stop_map = np.ones((12, 16), dtype=np.float32) * 0.5
        cross_map = np.ones((12, 16), dtype=np.float32) * 0.7

        features = _crosswalk_edge_features(
            candidate,
            meta=_identity_meta(),
            stop_mask_probs=stop_map,
            stop_center_probs=stop_map,
            stop_selector_probs=stop_map,
            crosswalk_mask_probs=cross_map,
            crosswalk_center_probs=cross_map,
            current_stop_lines=[],
            candidate_rank=1,
        )

        self.assertEqual(set(features), set(CROSSWALK_EDGE_FEATURES))
        self.assertTrue(np.isfinite(np.asarray(list(features.values()), dtype=np.float32)).all())
        self.assertGreater(features["crosswalk_edge_crosswalk_aspect"], 1.0)

    def test_build_crosswalk_edge_candidates_labels_near_gt(self) -> None:
        baseline = {
            "crosswalks": [
                {
                    "score": 0.9,
                    "points_xy": [
                        [20.0, 40.0],
                        [140.0, 40.0],
                        [140.0, 60.0],
                        [20.0, 60.0],
                    ],
                }
            ],
            "stop_lines": [],
        }
        gt_stop_lines = [{"points_xy": [[20.0, 32.0], [140.0, 32.0]]}]

        candidates = _build_crosswalk_edge_candidates(
            meta=_identity_meta(),
            baseline_prediction=baseline,
            gt_stop_lines=gt_stop_lines,
            stop_mask_probs=np.ones((12, 16), dtype=np.float32),
            stop_center_probs=np.ones((12, 16), dtype=np.float32),
            stop_selector_probs=np.ones((12, 16), dtype=np.float32),
            crosswalk_mask_probs=np.ones((12, 16), dtype=np.float32),
            crosswalk_center_probs=np.ones((12, 16), dtype=np.float32),
            offsets=(8.0,),
            length_scales=(1.0,),
            max_crosswalks=1,
            max_candidates=8,
        )

        self.assertTrue(any(bool(candidate["is_oracle_positive"]) for candidate in candidates))


if __name__ == "__main__":
    unittest.main()
