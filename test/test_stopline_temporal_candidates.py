from __future__ import annotations

import unittest

import numpy as np

from model.data.transform import compute_letterbox_transform
from tools.probe_pv26_stopline_temporal_candidates import (
    TEMPORAL_FEATURES,
    UNION_SCORE_KEY,
    _build_baseline_union_candidates,
    _build_dense_component_stop_lines,
    _build_temporal_endpoint_envelopes,
    _build_temporal_candidates,
    _build_union_candidates,
    _lane_affine_alignment_from_predictions,
    _merge_stop_lines_with_extra,
    _orb_homography_alignment_from_arrays,
    _phase_correlation_shift,
    _select_union_stop_lines,
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
        self.assertEqual(features["temporal_source_neighbor"], 0.0)
        self.assertEqual(features["temporal_source_envelope"], 0.0)
        self.assertEqual(features["temporal_source_dense_component"], 0.0)
        self.assertLess(features["temporal_alignment_dx_norm"], 0.0)
        self.assertGreater(features["temporal_alignment_dy_norm"], 0.0)
        self.assertAlmostEqual(features["temporal_alignment_response"], 0.5)

    def test_dense_component_stop_lines_extracts_segment_from_maps(self) -> None:
        mask = np.zeros((76, 100), dtype=np.float32)
        center = np.zeros((76, 100), dtype=np.float32)
        selector = np.zeros((76, 100), dtype=np.float32)
        mask[38:43, 20:81] = 0.9
        center[40, 20:81] = 0.8
        selector[39:42, 24:77] = 0.75

        candidates = _build_dense_component_stop_lines(
            meta=_meta(),
            mask_probs=mask,
            center_probs=center,
            selector_probs=selector,
            max_candidates=2,
        )

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["source"], "temporal_dense_component")
        self.assertGreater(candidates[0]["length"], 300.0)
        self.assertGreater(candidates[0]["temporal_dense_component_support_sum"], 1.0)

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

    def test_orb_homography_alignment_maps_moving_to_reference(self) -> None:
        cv2 = __import__("cv2")
        reference = np.zeros((96, 128), dtype=np.float32)
        for y in range(12, 88, 11):
            for x in range(10, 120, 13):
                radius = 2 + int((x + y) % 3)
                intensity = 0.45 + float((x * y) % 5) * 0.1
                cv2.circle(reference, (x, y), radius, intensity, -1)
        cv2.putText(reference, "PV26", (17, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1.0, 2)
        matrix = np.asarray([[1.0, 0.0, 7.0], [0.0, 1.0, -5.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        moving = cv2.warpPerspective(reference, matrix, (128, 96))

        alignment = _orb_homography_alignment_from_arrays(
            reference,
            moving,
            raw_hw=(96, 128),
            size=(128, 96),
            max_shift_frac=0.25,
        )

        self.assertEqual(float(alignment["applied"]), 1.0)
        self.assertAlmostEqual(float(alignment["dx"]), -7.0, delta=3.0)
        self.assertAlmostEqual(float(alignment["dy"]), 5.0, delta=3.0)
        self.assertGreater(float(alignment["response"]), 0.05)

    def test_lane_affine_alignment_uses_predicted_lane_tracks(self) -> None:
        neighbor_prediction = {
            "lanes": [
                {"points_xy": [[120.0, 120.0], [150.0, 280.0], [190.0, 520.0]]},
                {"points_xy": [[610.0, 120.0], [570.0, 280.0], [520.0, 520.0]]},
            ]
        }
        current_prediction = {
            "lanes": [
                {"points_xy": [[132.0, 120.0], [162.0, 280.0], [202.0, 520.0]]},
                {"points_xy": [[622.0, 120.0], [582.0, 280.0], [532.0, 520.0]]},
            ]
        }

        alignment = _lane_affine_alignment_from_predictions(
            current_prediction,
            neighbor_prediction,
            raw_hw=(608, 800),
            max_shift_frac=0.25,
        )

        self.assertEqual(float(alignment["applied"]), 1.0)
        self.assertAlmostEqual(float(alignment["dx"]), 12.0, delta=1.0)
        self.assertAlmostEqual(float(alignment["dy"]), 0.0, delta=1.0)
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

    def test_warp_stop_line_points_applies_homography_matrix(self) -> None:
        warped = _warp_stop_line_points(
            {"points_xy": [[10.0, 20.0], [30.0, 20.0]]},
            alignment={
                "applied": 1.0,
                "h00": 1.0,
                "h01": 0.0,
                "h02": 5.0,
                "h10": 0.0,
                "h11": 1.0,
                "h12": -3.0,
                "h20": 0.0,
                "h21": 0.0,
                "h22": 1.0,
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

    def test_build_candidates_adds_dense_component_temporal_source(self) -> None:
        mask = np.ones((76, 100), dtype=np.float32) * 0.5
        center = np.ones((76, 100), dtype=np.float32) * 0.4
        selector = np.ones((76, 100), dtype=np.float32) * 0.3
        neighbor_mask = np.zeros((76, 100), dtype=np.float32)
        neighbor_center = np.zeros((76, 100), dtype=np.float32)
        neighbor_selector = np.zeros((76, 100), dtype=np.float32)
        neighbor_mask[38:43, 20:81] = 0.9
        neighbor_center[40, 20:81] = 0.8
        neighbor_selector[39:42, 24:77] = 0.75
        gt = [{"points_xy": [[160.0, 320.0], [640.0, 320.0]]}]

        candidates = _build_temporal_candidates(
            meta=_meta(),
            mask_probs=mask,
            center_probs=center,
            selector_probs=selector,
            current_stop_lines=[],
            gt_stop_lines=gt,
            neighbor_predictions=[
                (
                    -1,
                    12,
                    {"stop_lines": []},
                    {},
                    {
                        "meta": _meta(),
                        "mask_probs": neighbor_mask,
                        "center_probs": neighbor_center,
                        "selector_probs": neighbor_selector,
                    },
                )
            ],
            max_candidates=4,
            temporal_dense_component_enabled=True,
            max_temporal_dense_components=2,
        )

        dense_candidates = [candidate for candidate in candidates if candidate["source"] == "temporal_dense_component"]
        self.assertEqual(len(dense_candidates), 1)
        self.assertTrue(dense_candidates[0]["is_oracle_positive"])
        self.assertEqual(dense_candidates[0]["temporal_source_dense_component"], 1.0)

    def test_build_temporal_endpoint_envelope_extends_supported_axis(self) -> None:
        mask = np.ones((76, 100), dtype=np.float32) * 0.5
        center = np.ones((76, 100), dtype=np.float32) * 0.4
        selector = np.ones((76, 100), dtype=np.float32) * 0.3
        current = [{"points_xy": [[180.0, 300.0], [520.0, 300.0]], "score": 0.8}]
        temporal = [
            {
                "points_xy": [[120.0, 302.0], [680.0, 302.0]],
                "score": 0.7,
                "source": "temporal_neighbor",
                "neighbor_offset": -1,
                "neighbor_dataset_index": 12,
                "neighbor_rank": 1,
            }
        ]
        gt = [{"points_xy": [[120.0, 301.0], [680.0, 301.0]]}]

        envelopes = _build_temporal_endpoint_envelopes(
            meta=_meta(),
            temporal_candidates=temporal,
            current_stop_lines=current,
            gt_stop_lines=gt,
            mask_probs=mask,
            center_probs=center,
            selector_probs=selector,
        )

        self.assertEqual(len(envelopes), 1)
        self.assertEqual(envelopes[0]["source"], "temporal_endpoint_envelope")
        self.assertGreater(envelopes[0]["length"], 500.0)
        self.assertGreater(envelopes[0]["temporal_envelope_length_gain"], 0.0)
        self.assertTrue(envelopes[0]["is_oracle_positive"])
        self.assertEqual(envelopes[0]["temporal_source_envelope"], 1.0)

    def test_build_union_candidates_assigns_single_match_label(self) -> None:
        mask = np.ones((76, 100), dtype=np.float32) * 0.5
        center = np.ones((76, 100), dtype=np.float32) * 0.4
        selector = np.ones((76, 100), dtype=np.float32) * 0.3
        baseline_prediction = {
            "stop_lines": [{"points_xy": [[100.0, 300.0], [700.0, 300.0]], "score": 0.8}]
        }
        temporal_candidates = [
            {"points_xy": [[100.0, 300.0], [700.0, 300.0]], "score": 0.7, "temporal_rank_score": 1.0}
        ]
        gt = [{"points_xy": [[100.0, 300.0], [700.0, 300.0]]}]

        baseline_candidates = _build_baseline_union_candidates(
            meta=_meta(),
            mask_probs=mask,
            center_probs=center,
            selector_probs=selector,
            baseline_prediction=baseline_prediction,
        )
        union_candidates = _build_union_candidates(
            baseline_candidates=baseline_candidates,
            temporal_candidates=temporal_candidates,
            gt_stop_lines=gt,
        )

        self.assertEqual(len(union_candidates), 2)
        self.assertEqual(sum(1 for candidate in union_candidates if candidate["is_union_positive"]), 1)
        self.assertEqual({candidate["source"] for candidate in union_candidates}, {"retained_projection_comp", "temporal_neighbor"})

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

    def test_select_union_stop_lines_applies_score_and_cap(self) -> None:
        candidates = [
            {
                "points_xy": [[100.0, 300.0], [700.0, 300.0]],
                UNION_SCORE_KEY: 0.9,
                "temporal_rank_score": 0.5,
                "length": 600.0,
                "source": "retained_projection_comp",
            },
            {
                "points_xy": [[100.0, 330.0], [700.0, 330.0]],
                UNION_SCORE_KEY: 0.2,
                "temporal_rank_score": 0.7,
                "length": 600.0,
                "source": "temporal_neighbor",
            },
        ]

        selected = _select_union_stop_lines(candidates, threshold=0.5, top_k=2, max_components=1)

        self.assertEqual(len(selected), 1)
        self.assertAlmostEqual(float(selected[0]["score"]), 0.9)

    def test_merge_stop_lines_with_extra_dedupes_and_caps(self) -> None:
        base = [
            {"points_xy": [[100.0, 300.0], [700.0, 300.0]], "score": 0.9},
            {"points_xy": [[100.0, 360.0], [700.0, 360.0]], "score": 0.8},
        ]
        extra = [
            {"points_xy": [[101.0, 301.0], [701.0, 301.0]], "score": 0.95},
            {"points_xy": [[100.0, 430.0], [700.0, 430.0]], "score": 0.7},
        ]

        merged = _merge_stop_lines_with_extra(base, extra, max_components=2)

        self.assertEqual(len(merged), 2)
        self.assertAlmostEqual(float(merged[0]["score"]), 0.95)
        self.assertNotIn([[100.0, 430.0], [700.0, 430.0]], [line["points_xy"] for line in merged])


if __name__ == "__main__":
    unittest.main()
