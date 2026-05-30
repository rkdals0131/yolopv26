from __future__ import annotations

import unittest

from tools.probe_pv26_lane_area_roi_verifier import (
    _baseline_matched_gt_indices,
    _candidate_label,
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


if __name__ == "__main__":
    unittest.main()
