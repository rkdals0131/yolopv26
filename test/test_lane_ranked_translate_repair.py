from __future__ import annotations

import unittest

import numpy as np

from tools.probe_pv26_lane_ranked_translate_repair import (
    _auto_repair_topk,
    _fit_affine_map_points_to_targets,
    _project_points_to_row_profile,
    _project_points_to_component_rows,
    _snap_map_points_to_local_centerline,
    score_repairability,
)


class LaneRankedTranslateRepairTests(unittest.TestCase):
    def test_auto_repair_topk_scales_reference_budget(self) -> None:
        self.assertEqual(_auto_repair_topk(16), 4)
        self.assertEqual(_auto_repair_topk(2048), 500)
        self.assertEqual(_auto_repair_topk(1), 1)

    def test_score_repairability_uses_standardized_features(self) -> None:
        model = {
            "features": ["center", "length"],
            "means": [0.5, 100.0],
            "scales": [0.25, 50.0],
            "weights": [1.0, 0.5],
            "bias": 0.0,
        }

        low = score_repairability({"center": 0.25, "length": 50.0}, model)
        high = score_repairability({"center": 0.75, "length": 150.0}, model)

        self.assertGreater(high, low)
        self.assertTrue(np.isfinite(high))
        self.assertTrue(np.isfinite(low))

    def test_local_2d_snap_moves_points_to_nearby_peak(self) -> None:
        centerline = np.zeros((8, 8), dtype=np.float32)
        centerline[3, 4] = 0.9
        centerline[0, 0] = 0.6
        points = np.asarray([[0.0, 0.0], [3.0, 3.0]], dtype=np.float32)

        snapped, stats = _snap_map_points_to_local_centerline(points, centerline, radius=2)

        self.assertEqual(snapped.tolist(), [[0.0, 0.0], [4.0, 3.0]])
        self.assertEqual(stats["moved_points"], 1.0)
        self.assertAlmostEqual(stats["max_move"], 1.0)

    def test_affine_fit_moves_track_coherently_to_targets(self) -> None:
        points = np.asarray([[1.0, 1.0], [3.0, 1.0], [2.0, 4.0], [5.0, 5.0]], dtype=np.float32)
        targets = points + np.asarray([2.0, 1.0], dtype=np.float32)

        repaired, stats = _fit_affine_map_points_to_targets(points, targets, map_hw=(16, 16))

        self.assertTrue(np.allclose(repaired, targets, atol=1.0e-4))
        self.assertEqual(stats["moved_points"], 4.0)
        self.assertAlmostEqual(stats["affine_residual"], 0.0, places=5)

    def test_component_row_project_moves_points_to_component_rows(self) -> None:
        points = np.asarray([[1.0, 1.0], [1.0, 3.0], [1.0, 5.0]], dtype=np.float32)
        component = np.asarray([[6.0, 1.0], [7.0, 3.0], [8.0, 5.0]], dtype=np.float32)

        repaired, stats = _project_points_to_component_rows(points, component)

        self.assertTrue(np.allclose(repaired, component))
        self.assertEqual(stats["moved_points"], 3.0)
        self.assertGreater(stats["mean_move"], 0.0)

    def test_row_profile_projection_uses_soft_centerline_mass(self) -> None:
        centerline = np.zeros((5, 9), dtype=np.float32)
        centerline[2, 4] = 0.2
        centerline[2, 5] = 0.8
        centerline[2, 6] = 0.4
        points = np.asarray([[4.0, 2.0]], dtype=np.float32)

        repaired, stats = _project_points_to_row_profile(points, centerline, radius=2)

        self.assertGreater(repaired[0, 0], 4.0)
        self.assertAlmostEqual(repaired[0, 1], 2.0)
        self.assertEqual(stats["moved_points"], 1.0)
        self.assertGreater(stats["mean_profile_mass"], 0.0)


if __name__ == "__main__":
    unittest.main()
