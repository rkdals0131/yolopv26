from __future__ import annotations

import unittest

import torch

from model.engine.batch import (
    augment_lane_family_metrics,
    move_batch_to_device,
    raw_batch_for_metrics,
)


class EngineBatchHelpersTests(unittest.TestCase):
    def test_move_batch_to_device_recurses_nested_containers(self) -> None:
        cpu = torch.device("cpu")
        sample = {
            "tensor": torch.tensor([1.0]),
            "nested": [torch.tensor([2.0]), (torch.tensor([3.0]), "keep-me")],
            "scalar": 4,
        }

        moved = move_batch_to_device(sample, cpu)

        self.assertEqual(moved["tensor"].device.type, "cpu")
        self.assertEqual(moved["nested"][0].device.type, "cpu")
        self.assertEqual(moved["nested"][1][0].device.type, "cpu")
        self.assertEqual(moved["nested"][1][1], "keep-me")
        self.assertEqual(moved["scalar"], 4)

    def test_raw_batch_for_metrics_prefers_embedded_raw_batch(self) -> None:
        embedded = {"det_targets": [{"sample_id": "raw"}]}
        batch = {"_raw_batch": embedded, "det_targets": [{"sample_id": "encoded"}]}

        self.assertIs(raw_batch_for_metrics(batch), embedded)

    def test_raw_batch_for_metrics_falls_back_to_raw_batch_shape(self) -> None:
        batch = {"det_targets": [{"sample_id": "raw"}]}

        self.assertIs(raw_batch_for_metrics(batch), batch)

    def test_raw_batch_for_metrics_returns_none_without_metric_payload(self) -> None:
        self.assertIsNone(raw_batch_for_metrics({"image": torch.zeros(1, 3, 16, 16)}))

    def test_augment_lane_family_metrics_adds_summary_without_mutating_input(self) -> None:
        metrics = {
            "lane": {"f1": 0.9},
            "stop_line": {"f1": 0.6},
            "crosswalk": {"f1": 0.3},
        }

        augmented = augment_lane_family_metrics(metrics)

        self.assertNotIn("lane_family", metrics)
        self.assertAlmostEqual(augmented["lane_family"]["mean_f1"], 0.6)
        self.assertAlmostEqual(augmented["lane_family"]["min_f1"], 0.3)

    def test_augment_lane_family_metrics_ignores_missing_scores(self) -> None:
        metrics = {"lane": {}, "stop_line": {"f1": 0.5}, "crosswalk": "n/a"}

        augmented = augment_lane_family_metrics(metrics)

        self.assertAlmostEqual(augmented["lane_family"]["mean_f1"], 0.5)
        self.assertAlmostEqual(augmented["lane_family"]["min_f1"], 0.5)


if __name__ == "__main__":
    unittest.main()
