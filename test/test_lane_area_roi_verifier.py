from __future__ import annotations

import unittest

from tools.probe_pv26_lane_area_roi_verifier import (
    _accumulate_task_counts,
    _alignment_context_features,
    _baseline_matched_gt_indices,
    _candidate_label,
    _empty_task_count_payload,
    _finalize_task_counts,
    _near_any_lane,
)


def _lane(x: float) -> dict[str, object]:
    return {"points_xy": [[x, 0.0], [x, 100.0]], "class_name": "white", "lane_type": "solid"}


class LaneAreaRoiVerifierTests(unittest.TestCase):
    def test_baseline_matched_gt_indices_uses_lane_metric_threshold(self) -> None:
        matched = _baseline_matched_gt_indices([_lane(10.0), _lane(150.0)], [_lane(12.0), _lane(260.0)])
        self.assertEqual(matched, {0})

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
