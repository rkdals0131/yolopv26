from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np
import torch

from tools.probe_pv26_lane_area_roi_verifier import (
    _accumulate_task_counts,
    _alignment_context_features,
    _apply_verifier,
    _baseline_matched_gt_indices,
    _candidate_label,
    _empty_task_count_payload,
    _finalize_task_counts,
    _lane_cross_task_conflict_features,
    _lane_raw_image_line_features,
    _lane_set_geometry_support,
    _lane_side_contrast_features,
    _matched_lane_prediction_indices,
    _near_any_lane,
    _nearest_lane_index,
)


def _lane(x: float) -> dict[str, object]:
    return {"points_xy": [[x, 0.0], [x, 100.0]], "class_name": "white", "lane_type": "solid"}


def _horizontal_lane(y: float) -> dict[str, object]:
    return {"points_xy": [[0.0, y], [100.0, y]], "class_name": "white", "lane_type": "solid"}


class LaneAreaRoiVerifierTests(unittest.TestCase):
    def test_baseline_matched_gt_indices_uses_lane_metric_threshold(self) -> None:
        matched = _baseline_matched_gt_indices([_lane(10.0), _lane(150.0)], [_lane(12.0), _lane(260.0)])
        self.assertEqual(matched, {0})

    def test_matched_lane_prediction_indices_return_prediction_matches(self) -> None:
        matched = _matched_lane_prediction_indices([_lane(10.0), _lane(150.0)], [_lane(12.0), _lane(260.0)])

        self.assertEqual(set(matched), {0})
        self.assertEqual(matched[0][0], 0)
        self.assertLess(matched[0][1], 5.0)

    def test_candidate_label_requires_unmatched_gt(self) -> None:
        positive, negative, gt_index, distance = _candidate_label(
            candidate=_lane(100.0),
            gt_lanes=[_lane(100.0)],
            baseline_matched_gt=set(),
            positive_distance_px=40.0,
            negative_distance_px=60.0,
        )
        self.assertTrue(positive)
        self.assertFalse(negative)
        self.assertEqual(gt_index, 0)
        self.assertLessEqual(distance, 40.0)

        positive, negative, _, _ = _candidate_label(
            candidate=_lane(100.0),
            gt_lanes=[_lane(100.0)],
            baseline_matched_gt={0},
            positive_distance_px=40.0,
            negative_distance_px=60.0,
        )
        self.assertFalse(positive)
        self.assertTrue(negative)

    def test_near_any_lane_is_schema_agnostic(self) -> None:
        candidate = {"points_xy": [[12.0, 0.0], [12.0, 100.0]], "class_name": "yellow", "lane_type": "dashed"}
        self.assertTrue(_near_any_lane(candidate, [_lane(10.0)], threshold_px=5.0))
        self.assertFalse(_near_any_lane(candidate, [_lane(30.0)], threshold_px=5.0))

    def test_nearest_lane_index_returns_distance(self) -> None:
        index, distance = _nearest_lane_index(_lane(42.0), [_lane(5.0), _lane(50.0), _lane(120.0)])

        self.assertEqual(index, 1)
        self.assertLess(distance, 10.0)

    def test_replace_nearest_integration_preserves_lane_count(self) -> None:
        class ConstantVerifier(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.register_buffer("feature_mean", torch.zeros(1), persistent=True)
                self.register_buffer("feature_std", torch.ones(1), persistent=True)

            def forward(self, features: torch.Tensor) -> torch.Tensor:
                return torch.full((int(features.shape[0]),), 10.0, dtype=features.dtype, device=features.device)

        predictions = [{"lanes": [_lane(0.0), _lane(220.0)]}]
        examples = [
            {
                "features": np.asarray([1.0], dtype=np.float32),
                "positive": 1.0,
                "negative": 0.0,
                "nearest_gt_index": 0,
                "nearest_gt_distance": 3.0,
                "sample_index": 0,
                "candidate_index": 0,
                "candidate": _lane(75.0),
            }
        ]
        repaired, rows = _apply_verifier(
            examples=examples,
            predictions_all=predictions,
            model=ConstantVerifier(),
            args=SimpleNamespace(
                quality_threshold=0.8,
                candidate_duplicate_distance_px=5.0,
                max_appends_per_sample=1,
                candidate_integration_mode="replace_nearest",
                replace_nearest_max_distance_px=120.0,
            ),
            device="cpu",
        )

        self.assertEqual(len(repaired[0]["lanes"]), 2)
        self.assertEqual(repaired[0]["lanes"][0]["points_xy"][0][0], 75.0)
        self.assertEqual(repaired[0]["lanes"][1]["points_xy"][0][0], 220.0)
        self.assertEqual(rows[0]["selected"], 1)
        self.assertEqual(rows[0]["integration_action"], "replace_nearest")
        self.assertEqual(rows[0]["replaced_lane_index"], 0)

    def test_suppress_low_quality_removes_retained_lane_by_candidate_index(self) -> None:
        class IndexedVerifier(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.register_buffer("feature_mean", torch.zeros(1), persistent=True)
                self.register_buffer("feature_std", torch.ones(1), persistent=True)

            def forward(self, features: torch.Tensor) -> torch.Tensor:
                return torch.tensor([-10.0, 10.0], dtype=features.dtype, device=features.device)

        predictions = [{"lanes": [_lane(0.0), _lane(100.0)]}]
        examples = [
            {
                "features": np.asarray([0.0], dtype=np.float32),
                "positive": 0.0,
                "negative": 1.0,
                "nearest_gt_index": -1,
                "nearest_gt_distance": 80.0,
                "sample_index": 0,
                "candidate_index": 0,
                "candidate": _lane(0.0),
            },
            {
                "features": np.asarray([1.0], dtype=np.float32),
                "positive": 1.0,
                "negative": 0.0,
                "nearest_gt_index": 0,
                "nearest_gt_distance": 1.0,
                "sample_index": 0,
                "candidate_index": 1,
                "candidate": _lane(100.0),
            },
        ]
        repaired, rows = _apply_verifier(
            examples=examples,
            predictions_all=predictions,
            model=IndexedVerifier(),
            args=SimpleNamespace(
                quality_threshold=0.5,
                candidate_duplicate_distance_px=5.0,
                max_appends_per_sample=1,
                max_suppressions_per_sample=1,
                candidate_integration_mode="suppress_low_quality",
                replace_nearest_max_distance_px=120.0,
            ),
            device="cpu",
        )

        self.assertEqual(len(repaired[0]["lanes"]), 1)
        self.assertEqual(repaired[0]["lanes"][0]["points_xy"][0][0], 100.0)
        self.assertEqual(rows[0]["selected"], 1)
        self.assertEqual(rows[0]["integration_action"], "suppress_low_quality")
        self.assertEqual(rows[1]["selected"], 0)

    def test_select_topk_union_reselects_retained_and_dropped_candidates(self) -> None:
        class FeatureLogitVerifier(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.register_buffer("feature_mean", torch.zeros(1), persistent=True)
                self.register_buffer("feature_std", torch.ones(1), persistent=True)

            def forward(self, features: torch.Tensor) -> torch.Tensor:
                return features[:, 0]

        predictions = [{"lanes": [_lane(0.0), _lane(220.0)]}]
        examples = [
            {
                "features": np.asarray([-10.0], dtype=np.float32),
                "positive": 0.0,
                "negative": 1.0,
                "nearest_gt_index": -1,
                "nearest_gt_distance": 90.0,
                "sample_index": 0,
                "candidate_index": 0,
                "candidate_source_kind": "retained",
                "baseline_lane_count": 2,
                "candidate": _lane(0.0),
            },
            {
                "features": np.asarray([8.0], dtype=np.float32),
                "positive": 1.0,
                "negative": 0.0,
                "nearest_gt_index": 1,
                "nearest_gt_distance": 2.0,
                "sample_index": 0,
                "candidate_index": 1,
                "candidate_source_kind": "retained",
                "baseline_lane_count": 2,
                "candidate": _lane(220.0),
            },
            {
                "features": np.asarray([10.0], dtype=np.float32),
                "positive": 1.0,
                "negative": 0.0,
                "nearest_gt_index": 0,
                "nearest_gt_distance": 2.0,
                "sample_index": 0,
                "candidate_index": 2,
                "candidate_source_kind": "dropped_area",
                "baseline_lane_count": 2,
                "candidate": _lane(75.0),
            },
        ]

        repaired, rows = _apply_verifier(
            examples=examples,
            predictions_all=predictions,
            model=FeatureLogitVerifier(),
            args=SimpleNamespace(
                quality_threshold=0.5,
                candidate_duplicate_distance_px=5.0,
                max_appends_per_sample=0,
                max_suppressions_per_sample=0,
                candidate_integration_mode="select_topk_union",
                replace_nearest_max_distance_px=120.0,
            ),
            device="cpu",
        )

        self.assertEqual(len(repaired[0]["lanes"]), 2)
        self.assertEqual([lane["points_xy"][0][0] for lane in repaired[0]["lanes"]], [75.0, 220.0])
        selected_sources = [row["candidate_source_kind"] for row in rows if row["selected"]]
        self.assertEqual(selected_sources, ["retained", "dropped_area"])

    def test_lane_set_geometry_support_prefers_parallel_set_context(self) -> None:
        retained = [_lane(10.0), _lane(100.0)]

        supported = _lane_set_geometry_support(_lane(70.0), retained)
        unsupported = _lane_set_geometry_support(_horizontal_lane(50.0), retained)

        self.assertGreater(supported, 0.8)
        self.assertLess(unsupported, 0.45)

    def test_select_topk_union_geometry_keeps_retained_over_unsupported_drop(self) -> None:
        class FeatureLogitVerifier(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.register_buffer("feature_mean", torch.zeros(1), persistent=True)
                self.register_buffer("feature_std", torch.ones(1), persistent=True)

            def forward(self, features: torch.Tensor) -> torch.Tensor:
                return features[:, 0]

        predictions = [{"lanes": [_lane(0.0), _lane(220.0)]}]
        examples = [
            {
                "features": np.asarray([7.0], dtype=np.float32),
                "positive": 1.0,
                "negative": 0.0,
                "nearest_gt_index": 0,
                "nearest_gt_distance": 2.0,
                "sample_index": 0,
                "candidate_index": 0,
                "candidate_source_kind": "retained",
                "baseline_lane_count": 2,
                "candidate": _lane(0.0),
            },
            {
                "features": np.asarray([6.0], dtype=np.float32),
                "positive": 1.0,
                "negative": 0.0,
                "nearest_gt_index": 1,
                "nearest_gt_distance": 2.0,
                "sample_index": 0,
                "candidate_index": 1,
                "candidate_source_kind": "retained",
                "baseline_lane_count": 2,
                "candidate": _lane(220.0),
            },
            {
                "features": np.asarray([10.0], dtype=np.float32),
                "positive": 0.0,
                "negative": 1.0,
                "nearest_gt_index": -1,
                "nearest_gt_distance": 120.0,
                "sample_index": 0,
                "candidate_index": 2,
                "candidate_source_kind": "dropped_area",
                "baseline_lane_count": 2,
                "candidate": _horizontal_lane(50.0),
            },
        ]

        repaired, rows = _apply_verifier(
            examples=examples,
            predictions_all=predictions,
            model=FeatureLogitVerifier(),
            args=SimpleNamespace(
                quality_threshold=0.5,
                candidate_duplicate_distance_px=5.0,
                max_appends_per_sample=0,
                max_suppressions_per_sample=0,
                candidate_integration_mode="select_topk_union_geometry",
                replace_nearest_max_distance_px=120.0,
            ),
            device="cpu",
        )

        self.assertEqual(len(repaired[0]["lanes"]), 2)
        self.assertEqual([lane["points_xy"][0][0] for lane in repaired[0]["lanes"]], [0.0, 220.0])
        selected_sources = [row["candidate_source_kind"] for row in rows if row["selected"]]
        self.assertEqual(selected_sources, ["retained", "retained"])
        unsupported_rows = [row for row in rows if row["candidate_index"] == 2]
        self.assertEqual(unsupported_rows[0]["selected"], 0)

    def test_ensemble_probability_mode_can_require_member_agreement(self) -> None:
        class ConstantVerifier(torch.nn.Module):
            def __init__(self, logit: float) -> None:
                super().__init__()
                self.logit = float(logit)
                self.register_buffer("feature_mean", torch.zeros(1), persistent=True)
                self.register_buffer("feature_std", torch.ones(1), persistent=True)

            def forward(self, features: torch.Tensor) -> torch.Tensor:
                return torch.full(
                    (int(features.shape[0]),),
                    self.logit,
                    dtype=features.dtype,
                    device=features.device,
                )

        examples = [
            {
                "features": np.asarray([1.0], dtype=np.float32),
                "positive": 1.0,
                "negative": 0.0,
                "nearest_gt_index": 0,
                "nearest_gt_distance": 3.0,
                "sample_index": 0,
                "candidate_index": 0,
                "candidate": _lane(75.0),
            }
        ]
        predictions = [{"lanes": [_lane(0.0)]}]

        _, rows = _apply_verifier(
            examples=examples,
            predictions_all=predictions,
            model=[ConstantVerifier(10.0), ConstantVerifier(-10.0)],
            args=SimpleNamespace(
                quality_threshold=0.8,
                candidate_duplicate_distance_px=5.0,
                max_appends_per_sample=1,
                candidate_integration_mode="append",
                replace_nearest_max_distance_px=120.0,
                ensemble_probability_mode="min",
            ),
            device="cpu",
        )

        self.assertEqual(rows[0]["selected"], 0)
        self.assertLess(rows[0]["verifier_probability"], 0.1)

    def test_alignment_context_features_describe_nearest_retained_lane(self) -> None:
        features = _alignment_context_features(_lane(70.0), [_lane(10.0), _lane(100.0)])

        self.assertEqual(features.shape, (10,))
        self.assertEqual(float(features[0]), 1.0)
        self.assertLess(float(features[1]), 1.0)
        self.assertGreater(float(features[5]), 0.9)
        self.assertLess(float(features[6]), 0.01)

    def test_alignment_context_features_handles_empty_retained_lanes(self) -> None:
        features = _alignment_context_features(_lane(70.0), [])

        self.assertEqual(features.shape, (10,))
        self.assertEqual(float(features[0]), 0.0)
        self.assertEqual(float(features[5]), 0.0)

    def test_lane_side_contrast_features_describe_thin_dense_ridge(self) -> None:
        centerline = torch.zeros((1, 32, 32), dtype=torch.float32)
        support = torch.zeros((1, 32, 32), dtype=torch.float32)
        centerline[:, :, 16] = 1.0
        support[:, :, 16] = 0.8
        tangent_axis = torch.zeros((2, 32, 32), dtype=torch.float32)
        tangent_axis[1, :, :] = 1.0
        sampled = np.stack(
            [np.full(20, 16.0, dtype=np.float32), np.linspace(4.0, 28.0, 20, dtype=np.float32)],
            axis=1,
        )

        features = _lane_side_contrast_features(
            sampled,
            maps={
                "centerline_core": centerline,
                "support": support,
                "tangent_axis": tangent_axis,
            },
        )

        self.assertEqual(features.shape, (52,))
        self.assertTrue(np.isfinite(features).all())
        self.assertGreater(float(features[0]), 0.9)
        self.assertLess(float(features[6]), 0.1)
        self.assertGreater(float(features[-4]), 0.9)

    def test_lane_raw_image_line_features_describe_bright_image_ridge(self) -> None:
        image = torch.zeros((3, 64, 64), dtype=torch.float32)
        image[:, :, 31:34] = 1.0
        sampled = np.stack(
            [np.full(20, 16.0, dtype=np.float32), np.linspace(4.0, 28.0, 20, dtype=np.float32)],
            axis=1,
        )

        features = _lane_raw_image_line_features(sampled, map_hw=(32, 32), image=image)

        self.assertEqual(features.shape, (68,))
        self.assertTrue(np.isfinite(features).all())
        self.assertGreater(float(features[0]), 0.9)
        self.assertLess(float(features[20]), 0.2)
        self.assertGreater(float(features[23]), 0.7)

    def test_lane_cross_task_conflict_features_sample_dense_task_maps(self) -> None:
        sampled = np.stack(
            [np.full(20, 16.0, dtype=np.float32), np.linspace(4.0, 28.0, 20, dtype=np.float32)],
            axis=1,
        )
        high_logit = torch.full((1, 32, 32), -10.0, dtype=torch.float32)
        high_logit[:, :, 16] = 10.0
        low_logit = torch.full((1, 16, 16), -10.0, dtype=torch.float32)

        features = _lane_cross_task_conflict_features(
            sampled,
            predictions={
                "stop_line_mask_logits": high_logit,
                "crosswalk_mask_logits": low_logit,
            },
            map_hw=(32, 32),
        )

        self.assertEqual(features.shape, (42,))
        self.assertTrue(np.isfinite(features).all())
        self.assertGreater(float(features[0]), 0.99)
        self.assertLess(float(features[24]), 0.01)

    def test_task_count_accumulator_sums_chunked_metrics(self) -> None:
        counts = _empty_task_count_payload()
        _accumulate_task_counts(
            counts,
            {
                "lane": {"tp": 2, "fp": 1, "fn": 3},
                "stop_line": {"tp": 1, "fp": 0, "fn": 1},
                "crosswalk": {"tp": 0, "fp": 2, "fn": 4},
            },
        )
        _accumulate_task_counts(
            counts,
            {
                "lane": {"tp": 3, "fp": 2, "fn": 1},
                "stop_line": {"tp": 2, "fp": 1, "fn": 0},
                "crosswalk": {"tp": 4, "fp": 0, "fn": 0},
            },
        )

        summary = _finalize_task_counts(counts)
        self.assertEqual(summary["lane"]["tp"], 5.0)
        self.assertEqual(summary["lane"]["fp"], 3.0)
        self.assertEqual(summary["lane"]["fn"], 4.0)
        self.assertAlmostEqual(summary["lane"]["f1"], 10.0 / 17.0)
        self.assertEqual(summary["stop_line"]["tp"], 3.0)
        self.assertAlmostEqual(summary["crosswalk"]["recall"], 0.5)


if __name__ == "__main__":
    unittest.main()
