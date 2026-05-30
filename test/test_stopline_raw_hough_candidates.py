from __future__ import annotations

import unittest

import numpy as np

from tools.probe_pv26_stopline_raw_hough_candidates import (
    SCORE_KEY,
    _attach_scores,
    _line_points,
    _select_hough_stop_lines,
)


class StopLineRawHoughCandidateTests(unittest.TestCase):
    def test_line_points_interpolates_requested_count(self) -> None:
        points = _line_points(np.asarray([0.0, 2.0]), np.asarray([10.0, 12.0]), count=5)
        self.assertEqual(points.shape, (5, 2))
        np.testing.assert_allclose(points[0], np.asarray([0.0, 2.0], dtype=np.float32))
        np.testing.assert_allclose(points[-1], np.asarray([10.0, 12.0], dtype=np.float32))

    def test_attach_scores_skips_candidates_outside_top_k(self) -> None:
        row_a: dict[str, float] = {}
        row_b: dict[str, float] = {}
        records = [
            {
                "candidates": [{"proposal_rank": 1}, {"proposal_rank": 9}],
                "candidate_feature_rows": [row_a, row_b],
            }
        ]
        _attach_scores(records, np.asarray([0.6], dtype=np.float32), top_k=8)
        self.assertAlmostEqual(records[0]["candidates"][0][SCORE_KEY], 0.6, places=6)
        self.assertAlmostEqual(row_a[SCORE_KEY], 0.6, places=6)
        self.assertNotIn(SCORE_KEY, records[0]["candidates"][1])
        self.assertNotIn(SCORE_KEY, row_b)

    def test_select_hough_stop_lines_applies_threshold_and_cap(self) -> None:
        candidates = [
            {
                "proposal_rank": 1,
                SCORE_KEY: 0.8,
                "raw_hough_score": 0.7,
                "length": 50.0,
                "points_xy": [[0.0, 0.0], [50.0, 0.0]],
            },
            {
                "proposal_rank": 2,
                SCORE_KEY: 0.4,
                "raw_hough_score": 0.9,
                "length": 80.0,
                "points_xy": [[0.0, 1.0], [80.0, 1.0]],
            },
        ]
        selected = _select_hough_stop_lines(
            candidates,
            score_key=SCORE_KEY,
            threshold=0.5,
            top_k=8,
            max_components=2,
        )
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["points_xy"], [[0.0, 0.0], [50.0, 0.0]])


if __name__ == "__main__":
    unittest.main()
