from __future__ import annotations

import unittest

import numpy as np

from model.data.transform import compute_letterbox_transform
from tools.probe_pv26_stopline_temporal_candidates import (
    TEMPORAL_FEATURES,
    _build_temporal_candidates,
    _phase_correlation_shift,
    _select_temporal_stop_lines,
    _sparse_affine_alignment_from_arrays,
    _stopline_temporal_features,
    _translate_stop_line_points,
    _warp_stop_line_points,
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
            alignment={"dx": -16.0, "dy": 8.0, "response": 0.5},
            current_stop_lines=[],
        )

        self.assertEqual(tuple(features.keys()), TEMPORAL_FEATURES)
        self.assertTrue(np.isfinite(np.asarray(list(features.values()), dtype=np.float32)).all())
        self.assertGreater(features["temporal_neighbor_length"], 0.0)
        self.assertGreater(features["temporal_proposal_mean"], 0.0)
        self.assertLess(features["temporal_alignment_dx_norm"], 0.0)
        self.assertGreater(features["temporal_alignment_dy_norm"], 0.0)
        self.assertAlmostEqual(features["temporal_alignment_response"], 0.5)

    def test_phase_correlation_returns_shift_to_apply_to_moving(self) -> None:
        reference = np.zeros((32, 32), dtype=np.float32)
        reference[10, 10] = 1.0
        moving = np.roll(np.roll(reference, 3, axis=0), 5, axis=1)

        dx, dy, response = _phase_correlation_shift(reference, moving)

        self.assertEqual((dx, dy), (-5.0, -3.0))
        self.assertGreater(response, 0.0)

    def test_translate_stop_line_points_clips_to_raw_frame(self) -> None:
        translated = _translate_stop_line_points(
            {"points_xy": [[10.0, 20.0], [799.0, 607.0]]},
            dx=-15.0,
            dy=10.0,
            meta=_meta(),
        )

        self.assertEqual(translated[0], [0.0, 30.0])
        self.assertEqual(translated[1], [784.0, 607.0])

    def test_sparse_affine_alignment_maps_moving_to_reference(self) -> None:
        cv2 = __import__("cv2")
        reference = np.zeros((96, 128), dtype=np.float32)
        for y in range(16, 80, 16):
            for x in range(16, 112, 16):
                reference[y - 2 : y + 3, x - 2 : x + 3] = 1.0
        matrix = np.asarray([[1.0, 0.0, 7.0], [0.0, 1.0, -5.0]], dtype=np.float32)
        moving = cv2.warpAffine(reference, matrix, (128, 96))

        alignment = _sparse_affine_alignment_from_arrays(
            reference,
            moving,
            raw_hw=(96, 128),
            size=(128, 96),
            max_shift_frac=0.25,
        )

        self.assertEqual(float(alignment["applied"]), 1.0)
        self.assertAlmostEqual(float(alignment["dx"]), -7.0, delta=1.5)
        self.assertAlmostEqual(float(alignment["dy"]), 5.0, delta=1.5)
        self.assertGreater(float(alignment["response"]), 0.1)

    def test_warp_stop_line_points_applies_affine_matrix(self) -> None:
        warped = _warp_stop_line_points(
            {"points_xy": [[10.0, 20.0], [30.0, 20.0]]},
            alignment={
                "applied": 1.0,
                "m00": 1.0,
                "m01": 0.0,
                "m02": 5.0,
                "m10": 0.0,
                "m11": 1.0,
                "m12": -3.0,
            },
            meta=_meta(),
        )

        self.assertEqual(warped[0], [15.0, 17.0])
        self.assertEqual(warped[1], [35.0, 17.0])

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
