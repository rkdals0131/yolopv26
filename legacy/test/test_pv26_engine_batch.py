from __future__ import annotations

import unittest

import torch

from model.engine.batch import (
    augment_lane_family_metrics,
    merge_raw_batches,
    move_batch_to_device,
    raw_batch_for_metrics,
    validate_prediction_batch_matches_image,
    validate_raw_batch_matches_image,
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

    def test_merge_raw_batches_concatenates_metric_payload_and_images(self) -> None:
        first = {
            "image": torch.zeros(1, 3, 4, 4),
            "det_targets": [{"sample_id": "a"}],
            "tl_attr_targets": [{"sample_id": "a"}],
            "lane_targets": [{"sample_id": "a"}],
            "source_mask": [{"det": True}],
            "valid_mask": [{"lane": True}],
            "meta": [{"sample_id": "a"}],
        }
        second = {
            "image": torch.ones(1, 3, 4, 4),
            "det_targets": [{"sample_id": "b"}],
            "tl_attr_targets": [{"sample_id": "b"}],
            "lane_targets": [{"sample_id": "b"}],
            "source_mask": [{"det": False}],
            "valid_mask": [{"lane": False}],
            "meta": [{"sample_id": "b"}],
        }

        merged = merge_raw_batches([first, second])

        self.assertEqual([item["sample_id"] for item in merged["meta"]], ["a", "b"])
        self.assertEqual(tuple(merged["image"].shape), (2, 3, 4, 4))
        self.assertTrue(torch.equal(merged["image"][0], first["image"][0]))
        self.assertTrue(torch.equal(merged["image"][1], second["image"][0]))

    def test_merge_raw_batches_rejects_empty_input(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot merge zero raw batches"):
            merge_raw_batches([])

    def test_merge_raw_batches_rejects_per_batch_field_length_mismatch(self) -> None:
        malformed = {
            "det_targets": [{"sample_id": "a"}],
            "tl_attr_targets": [{"sample_id": "a"}],
            "lane_targets": [{"sample_id": "a"}],
            "source_mask": [{"det": True}],
            "valid_mask": [{"lane": True}],
            "meta": [{"sample_id": "a"}, {"sample_id": "extra"}],
        }

        with self.assertRaisesRegex(ValueError, "raw batch field lengths must match"):
            merge_raw_batches([malformed])

    def test_merge_raw_batches_rejects_image_batch_size_mismatch(self) -> None:
        malformed = {
            "image": torch.zeros(2, 3, 4, 4),
            "det_targets": [{"sample_id": "a"}],
            "tl_attr_targets": [{"sample_id": "a"}],
            "lane_targets": [{"sample_id": "a"}],
            "source_mask": [{"det": True}],
            "valid_mask": [{"lane": True}],
            "meta": [{"sample_id": "a"}],
        }

        with self.assertRaisesRegex(ValueError, "raw batch image batch size must match meta length"):
            merge_raw_batches([malformed])

    def test_validate_raw_batch_rejects_missing_metric_fields(self) -> None:
        raw_batch = {
            "det_targets": [{"sample_id": "a"}],
            "tl_attr_targets": [{"sample_id": "a"}],
            "lane_targets": [{"sample_id": "a"}],
            "source_mask": [{"det": True}],
            "meta": [{"sample_id": "a"}],
        }

        with self.assertRaisesRegex(ValueError, "encoded _raw_batch missing required fields"):
            validate_raw_batch_matches_image(
                raw_batch,
                torch.zeros((1, 3, 4, 4), dtype=torch.float32),
                context="encoded",
            )

    def test_validate_raw_batch_rejects_non_batched_image_tensor(self) -> None:
        raw_batch = {
            "det_targets": [{"sample_id": str(index)} for index in range(3)],
            "tl_attr_targets": [{"sample_id": str(index)} for index in range(3)],
            "lane_targets": [{"sample_id": str(index)} for index in range(3)],
            "source_mask": [{"det": True} for _ in range(3)],
            "valid_mask": [{"lane": True} for _ in range(3)],
            "meta": [{"sample_id": str(index)} for index in range(3)],
        }

        with self.assertRaisesRegex(ValueError, "encoded image must be a 4D tensor batch"):
            validate_raw_batch_matches_image(
                raw_batch,
                torch.zeros((3, 4, 4), dtype=torch.float32),
                context="encoded",
            )

    def test_validate_prediction_batch_rejects_non_batched_image_tensor(self) -> None:
        predictions = {"det": torch.zeros((3, 10, 12), dtype=torch.float32)}

        with self.assertRaisesRegex(ValueError, "prediction image must be a 4D tensor batch"):
            validate_prediction_batch_matches_image(
                predictions,
                torch.zeros((3, 4, 4), dtype=torch.float32),
            )

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
