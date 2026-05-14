import math
import json
import unittest

from tools.probe_pv26_lane_fn_recovery_audit import (
    _distance_bin,
    _f1,
    _geometry_summaries,
    _nearest_gt_lane,
    _nearest_prediction,
    _pair_geometry_features,
    _points_json,
    _prediction_shape_features,
    _upper_bound_rows,
)


class LaneFnRecoveryAuditTest(unittest.TestCase):
    def test_distance_bins_keep_metric_cutoff_explicit(self) -> None:
        self.assertEqual(_distance_bin(math.inf), "none")
        self.assertEqual(_distance_bin(40.0), "le40")
        self.assertEqual(_distance_bin(80.0), "40_80")
        self.assertEqual(_distance_bin(120.0), "80_120")
        self.assertEqual(_distance_bin(200.0), "120_200")
        self.assertEqual(_distance_bin(201.0), "gt200")

    def test_nearest_prediction_can_restrict_unmatched_set(self) -> None:
        gt = {"points_xy": [[0.0, 0.0], [0.0, 10.0]]}
        predictions = [
            {"points_xy": [[0.0, 0.0], [0.0, 10.0]]},
            {"points_xy": [[30.0, 0.0], [30.0, 10.0]]},
        ]

        self.assertEqual(_nearest_prediction(gt, predictions)[0], 0)
        self.assertEqual(_nearest_prediction(gt, predictions, allowed_indices={1})[0], 1)

    def test_upper_bounds_preserve_fp_and_remove_only_counted_fn(self) -> None:
        rows = [
            {"gt_center_point_mean": 0.8, "gt_center_point_q10": 0.4, "nearest_unmatched_pred_distance": 200.0},
            {"gt_center_point_mean": 0.1, "gt_center_point_q10": 0.0, "nearest_unmatched_pred_distance": 60.0},
        ]

        bounds = _upper_bound_rows(rows, baseline_tp=10, baseline_fp=5, baseline_fn=4)
        best = bounds[0]

        self.assertEqual(best["upper_bound_lane_fp"], 5)
        self.assertLessEqual(best["recovered_fn_count"], 4)
        self.assertGreaterEqual(best["upper_bound_lane_f1_no_new_fp"], _f1(10, 5, 4))

    def test_pair_geometry_features_align_reversed_predictions(self) -> None:
        gt = {"points_xy": [[0.0, 0.0], [0.0, 10.0]]}
        pred = {"points_xy": [[3.0, 10.0], [3.0, 0.0]]}

        features = _pair_geometry_features(gt, pred, prefix="nearest_unmatched_pred")

        self.assertAlmostEqual(features["nearest_unmatched_pred_length_ratio"], 1.0)
        self.assertAlmostEqual(features["nearest_unmatched_pred_sample_mean_distance"], 3.0)
        self.assertAlmostEqual(features["nearest_unmatched_pred_center_dx"], 3.0)
        self.assertAlmostEqual(features["nearest_unmatched_pred_center_dy"], 0.0)
        self.assertAlmostEqual(features["nearest_unmatched_pred_angle_error_degrees"], 0.0)
        self.assertAlmostEqual(features["nearest_unmatched_pred_y_overlap_fraction"], 1.0)

    def test_nearest_gt_lane_can_restrict_unmatched_set(self) -> None:
        pred = {"points_xy": [[0.0, 0.0], [0.0, 10.0]]}
        gt_rows = [
            {"points_xy": [[0.0, 0.0], [0.0, 10.0]]},
            {"points_xy": [[25.0, 0.0], [25.0, 10.0]]},
        ]

        self.assertEqual(_nearest_gt_lane(pred, gt_rows)[0], 0)
        self.assertEqual(_nearest_gt_lane(pred, gt_rows, allowed_indices={1})[0], 1)

    def test_prediction_shape_features_are_no_gt_track_features(self) -> None:
        features = _prediction_shape_features({"points_xy": [[2.0, 1.0], [2.0, 11.0], [4.0, 21.0]]})

        self.assertEqual(features["pred_point_count"], 3.0)
        self.assertGreater(features["pred_polyline_length"], 20.0)
        self.assertAlmostEqual(features["pred_center_x"], 8.0 / 3.0, places=6)
        self.assertAlmostEqual(features["pred_top_y"], 1.0)
        self.assertAlmostEqual(features["pred_bottom_y"], 21.0)
        self.assertGreater(features["pred_bbox_aspect"], 1.0)

    def test_points_json_exports_compact_points_for_geometry_replay(self) -> None:
        payload = _points_json({"points_xy": [[1.5, 2.0], [3.0, 4.25]]})

        self.assertEqual(json.loads(payload), [[1.5, 2.0], [3.0, 4.25]])
        self.assertEqual(_points_json(None), "[]")

    def test_geometry_summaries_keep_joint_groups_explicit(self) -> None:
        rows = [
            {
                "nearest_unmatched_pred_index": 0,
                "nearest_unmatched_pred_distance": 70.0,
                "gt_center_point_mean": 0.6,
                "nearest_unmatched_pred_length_ratio": 1.0,
                "nearest_unmatched_pred_angle_error_degrees": 3.0,
                "nearest_unmatched_pred_center_distance": 20.0,
                "nearest_unmatched_pred_sample_mean_abs_dx": 5.0,
                "nearest_unmatched_pred_sample_mean_distance": 20.0,
                "nearest_unmatched_pred_endpoint_mean_distance": 30.0,
                "nearest_unmatched_pred_y_overlap_fraction": 0.8,
            },
            {
                "nearest_unmatched_pred_index": 1,
                "nearest_unmatched_pred_distance": 100.0,
                "gt_center_point_mean": 0.2,
                "nearest_unmatched_pred_length_ratio": 0.5,
                "nearest_unmatched_pred_angle_error_degrees": 10.0,
                "nearest_unmatched_pred_center_distance": 60.0,
                "nearest_unmatched_pred_sample_mean_abs_dx": 40.0,
                "nearest_unmatched_pred_sample_mean_distance": 60.0,
                "nearest_unmatched_pred_endpoint_mean_distance": 70.0,
                "nearest_unmatched_pred_y_overlap_fraction": 0.4,
            },
        ]

        summaries = {row["name"]: row for row in _geometry_summaries(rows)}

        self.assertEqual(summaries["center050_and_unmatched_le80"]["count"], 1)
        self.assertEqual(summaries["unmatched_le120_without_center050"]["count"], 1)
        self.assertAlmostEqual(
            summaries["center050_and_unmatched_le80"]["nearest_unmatched_length_ratio_q50"],
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
