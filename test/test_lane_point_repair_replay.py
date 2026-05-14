import unittest

from tools.replay_pv26_lane_point_repair import (
    _candidate_is_selected,
    _replace_lane_points,
    _summarize_selected_targets,
)


class LanePointRepairReplayTest(unittest.TestCase):
    def test_candidate_selection_keeps_oracle_modes_explicit(self) -> None:
        broad = {
            "nearest_fn_gt_index": 3,
            "nearest_fn_gt_distance": 100.0,
            "nearest_fn_gt_center_point_mean": 0.1,
        }
        tight = {
            "nearest_fn_gt_index": 4,
            "nearest_fn_gt_distance": 70.0,
            "nearest_fn_gt_center_point_mean": 0.6,
        }
        missing = {
            "nearest_fn_gt_index": -1,
            "nearest_fn_gt_distance": 10.0,
            "nearest_fn_gt_center_point_mean": 1.0,
        }

        self.assertTrue(_candidate_is_selected(broad, "oracle_le120_any_center"))
        self.assertFalse(_candidate_is_selected(broad, "oracle_le80_center050"))
        self.assertTrue(_candidate_is_selected(tight, "oracle_le80_center050"))
        self.assertFalse(_candidate_is_selected(missing, "all_unmatched_with_fn"))

    def test_replace_lane_points_does_not_mutate_original_lane(self) -> None:
        lane = {"points_xy": [[0.0, 0.0], [1.0, 1.0]], "score": 0.5}

        repaired = _replace_lane_points(lane, [[2, 3], [4, 5]])

        self.assertEqual(lane["points_xy"], [[0.0, 0.0], [1.0, 1.0]])
        self.assertEqual(repaired["points_xy"], [[2.0, 3.0], [4.0, 5.0]])
        self.assertEqual(repaired["score"], 0.5)

    def test_selected_target_summary_reports_duplicate_targets(self) -> None:
        rows = [
            {"sample_index": 0, "nearest_fn_gt_index": 1, "selected": True},
            {"sample_index": 0, "nearest_fn_gt_index": 1, "selected": True},
            {"sample_index": 1, "nearest_fn_gt_index": 1, "selected": True},
            {"sample_index": 2, "nearest_fn_gt_index": 2, "selected": False},
        ]

        summary = _summarize_selected_targets(rows)

        self.assertEqual(summary["candidate_count"], 4)
        self.assertEqual(summary["selected_count"], 3)
        self.assertEqual(summary["selected_unique_target_count"], 2)
        self.assertEqual(summary["selected_duplicate_target_count"], 1)
        self.assertEqual(summary["samples_with_selected"], 2)


if __name__ == "__main__":
    unittest.main()
