import unittest

import numpy as np

from tools.probe_pv26_lane_instance_evidence import (
    FLIP_FEATURE_NAMES,
    _feature_names,
    _flip_consistency_features,
    _oracle_tp_scores_by_row,
    _needs_flip_forward,
    _resolve_device,
)


class LaneInstanceEvidenceProbeTest(unittest.TestCase):
    def test_feature_names_keep_default_surface_without_flip_features(self) -> None:
        self.assertFalse(set(FLIP_FEATURE_NAMES) & set(_feature_names(use_flip_consistency=False)))
        self.assertTrue(set(FLIP_FEATURE_NAMES) <= set(_feature_names(use_flip_consistency=True)))

    def test_resolve_device_does_not_pass_auto_to_torch(self) -> None:
        self.assertEqual(_resolve_device("cpu", "auto"), "cpu")
        self.assertIn(_resolve_device("auto", "auto"), {"cpu", "cuda:0"})

    def test_needs_flip_forward_for_features_or_merge_variant(self) -> None:
        self.assertFalse(_needs_flip_forward(use_flip_consistency=False, lane_flip_variant="baseline"))
        self.assertTrue(_needs_flip_forward(use_flip_consistency=True, lane_flip_variant="baseline"))
        self.assertTrue(_needs_flip_forward(use_flip_consistency=False, lane_flip_variant="flip_centerline_avg"))

    def test_oracle_tp_scores_keep_only_matched_rows(self) -> None:
        scores = _oracle_tp_scores_by_row(
            [
                {"row_id": 2, "is_tp": 1},
                {"row_id": 3, "is_tp": 0},
                {"row_id": 4},
            ]
        )

        self.assertEqual(scores, {2: 1.0, 3: 0.0, 4: 0.0})

    def test_flip_consistency_features_score_consensus_and_agreement(self) -> None:
        centerline = np.asarray([[0.9, 0.2], [0.6, 0.1]], dtype=np.float32)
        flip_centerline = np.asarray([[0.8, 0.4], [0.1, 0.1]], dtype=np.float32)
        mask = np.asarray([[True, False], [True, True]])
        sampled_points = np.asarray(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
            dtype=np.float32,
        )

        features = _flip_consistency_features(
            centerline,
            flip_centerline,
            mask=mask,
            sampled_points=sampled_points,
        )

        self.assertAlmostEqual(features["flip_center_mask_mean"], (0.8 + 0.1 + 0.1) / 3.0)
        self.assertAlmostEqual(features["center_consensus_point_mean"], (0.8 + 0.2 + 0.1 + 0.1) / 4.0)
        self.assertAlmostEqual(features["center_agreement_mask_mean"], (0.9 + 0.5 + 1.0) / 3.0)
        self.assertAlmostEqual(features["center_agreement_point_q10"], 0.59, places=6)


if __name__ == "__main__":
    unittest.main()
