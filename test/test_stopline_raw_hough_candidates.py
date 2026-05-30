from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from tools.pv26_train.config import (
    DatasetConfig,
    MetaTrainScenario,
    PhaseConfig,
    PreviewConfig,
    RunConfig,
    SelectionConfig,
    TrainDefaultsConfig,
)
from tools.probe_pv26_stopline_raw_hough_candidates import (
    SCORE_KEY,
    _attach_scores,
    _detect_raw_line_segments,
    _generate_raw_hough_candidates,
    _line_points,
    _scenario_with_dataset_root,
    _select_hough_stop_lines,
)


class StopLineRawHoughCandidateTests(unittest.TestCase):
    def test_line_points_interpolates_requested_count(self) -> None:
        points = _line_points(np.asarray([0.0, 2.0]), np.asarray([10.0, 12.0]), count=5)
        self.assertEqual(points.shape, (5, 2))
        np.testing.assert_allclose(points[0], np.asarray([0.0, 2.0], dtype=np.float32))
        np.testing.assert_allclose(points[-1], np.asarray([10.0, 12.0], dtype=np.float32))

    def test_lsd_generator_detects_synthetic_line(self) -> None:
        image = np.zeros((64, 96), dtype=np.uint8)
        image[30:33, 12:84] = 255
        lines, edge = _detect_raw_line_segments(image, candidate_generator="lsd")

        self.assertEqual(lines.ndim, 2)
        self.assertEqual(lines.shape[1], 4)
        self.assertEqual(edge.shape, image.shape)
        self.assertGreater(lines.shape[0], 0)

    def test_support_pca_generator_uses_dense_and_raw_support(self) -> None:
        image = np.zeros((64, 96), dtype=np.float32)
        image[30:33, 12:84] = 1.0
        mask = np.zeros((64, 96), dtype=np.float32)
        center = np.zeros((64, 96), dtype=np.float32)
        selector = np.zeros((64, 96), dtype=np.float32)
        mask[29:34, 10:86] = 0.8
        center[30:33, 12:84] = 0.7
        selector[30:33, 12:84] = 0.65
        meta = {
            "raw_hw": (64, 96),
            "network_hw": (64, 96),
            "transform": {
                "scale": 1.0,
                "pad_left": 0,
                "pad_top": 0,
                "pad_right": 0,
                "pad_bottom": 0,
                "resized_hw": (64, 96),
            },
        }

        candidates = _generate_raw_hough_candidates(
            image=image,
            meta=meta,
            mask_probs=mask,
            center_probs=center,
            selector_probs=selector,
            gt_stop_lines=[],
            max_candidates=4,
            candidate_generator="support_pca",
        )

        self.assertGreaterEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["proposal_source"], "raw_support_pca")
        self.assertGreater(float(candidates[0]["length"]), 12.0)

    def test_scenario_with_dataset_root_updates_dataset_config(self) -> None:
        scenario = MetaTrainScenario(
            dataset=DatasetConfig(root=Path("/old/root"), additional_roots=(Path("/extra/root"),)),
            run=RunConfig(),
            train_defaults=TrainDefaultsConfig(),
            selection=SelectionConfig(),
            preview=PreviewConfig(enabled=False),
            phases=(
                PhaseConfig(
                    name="phase",
                    stage="stage_4_lane_family_finetune",
                    min_epochs=1,
                    max_epochs=1,
                    patience=1,
                ),
            ),
        )

        updated = _scenario_with_dataset_root(scenario, "/new/root")

        self.assertEqual(updated.dataset.root, Path("/new/root").resolve())
        self.assertEqual(updated.dataset.additional_roots, ())
        self.assertEqual(updated.run, scenario.run)
        self.assertEqual(updated.phases, scenario.phases)

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
