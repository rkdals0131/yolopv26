from __future__ import annotations

import unittest

import numpy as np

from tools.probe_pv26_lane_repair_hypothesis_selector import (
    _feature_vector,
    _parse_repair_modes,
    select_hypothesis_indices,
)


class LaneRepairHypothesisSelectorTests(unittest.TestCase):
    def test_parse_repair_modes_rejects_unknown(self) -> None:
        self.assertEqual(_parse_repair_modes("translate_x,local_2d_snap"), ("translate_x", "local_2d_snap"))
        with self.assertRaises(ValueError):
            _parse_repair_modes("translate_x,unknown_mode")

    def test_feature_vector_has_original_repaired_delta_stats_and_mode(self) -> None:
        original = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)
        repaired = np.asarray([2.0, 4.0, 6.0], dtype=np.float32)
        vector = _feature_vector(
            mode="local_2d_snap",
            mode_index=1,
            modes=("translate_x", "local_2d_snap"),
            original_features=original,
            repaired_features=repaired,
            stats={"mean_move": 5.0, "max_move": 7.0},
            mean_move=5.0,
        )

        self.assertTrue(np.isfinite(vector).all())
        self.assertEqual(vector[:3].tolist(), [1.0, 2.0, 3.0])
        self.assertEqual(vector[3:6].tolist(), [2.0, 4.0, 6.0])
        self.assertEqual(vector[6:9].tolist(), [1.0, 2.0, 3.0])
        self.assertEqual(vector[-2:].tolist(), [0.0, 1.0])

    def test_selection_keeps_one_hypothesis_per_lane_and_sample_budget(self) -> None:
        examples = [
            {"sample_index": 0, "pred_index": 0, "mean_move": 2.0},
            {"sample_index": 0, "pred_index": 0, "mean_move": 3.0},
            {"sample_index": 0, "pred_index": 1, "mean_move": 2.0},
            {"sample_index": 1, "pred_index": 0, "mean_move": 0.0},
        ]
        selected = select_hypothesis_indices(
            examples,
            np.asarray([0.9, 0.95, 0.92, 0.99], dtype=np.float32),
            quality_threshold=0.8,
            max_repairs_per_sample=2,
            min_mean_move_px=0.25,
            max_mean_move_px=90.0,
        )

        self.assertEqual(selected, {1, 2})


if __name__ == "__main__":
    unittest.main()
