from __future__ import annotations

import unittest
from pathlib import Path

import torch

from model.loading import PV26BalancedBatchSampler, build_pv26_train_dataloader, dataset_group_for_key
from model.loading.pv26_loader import SampleRecord


class _ToyCanonicalDataset:
    def __init__(self) -> None:
        self.records: list[SampleRecord] = []
        self.samples: list[dict] = []
        dataset_keys = (
            ("bdd100k_det_100k", 4),
            ("aihub_traffic_seoul", 4),
            ("aihub_lane_seoul", 4),
        )
        for split in ("train", "val"):
            for dataset_key, count in dataset_keys:
                for index in range(count):
                    sample_id = f"{dataset_key}_{split}_{index}"
                    record = SampleRecord(
                        dataset_root=Path("/tmp"),
                        dataset_key=dataset_key,
                        split=split,
                        sample_id=sample_id,
                        scene_path=Path("/tmp") / f"{sample_id}.json",
                        image_path=Path("/tmp") / f"{sample_id}.jpg",
                        det_path=None,
                    )
                    self.records.append(record)
                    self.samples.append(
                        {
                            "image": torch.zeros((3, 608, 800), dtype=torch.float32),
                            "det_targets": {
                                "boxes_xyxy": torch.zeros((0, 4), dtype=torch.float32),
                                "classes": torch.zeros((0,), dtype=torch.long),
                            },
                            "tl_attr_targets": {
                                "bits": torch.zeros((0, 4), dtype=torch.float32),
                                "is_traffic_light": torch.zeros((0,), dtype=torch.bool),
                                "collapse_reason": [],
                            },
                            "lane_targets": {
                                "lanes": [],
                                "stop_lines": [],
                                "crosswalks": [],
                            },
                            "source_mask": {
                                "det": dataset_key != "aihub_lane_seoul",
                                "tl_attr": dataset_key == "aihub_traffic_seoul",
                                "lane": dataset_key == "aihub_lane_seoul",
                                "stop_line": dataset_key == "aihub_lane_seoul",
                                "crosswalk": dataset_key == "aihub_lane_seoul",
                            },
                            "valid_mask": {
                                "det": torch.zeros((0,), dtype=torch.bool),
                                "tl_attr": torch.zeros((0,), dtype=torch.bool),
                                "lane": torch.zeros((0,), dtype=torch.bool),
                                "stop_line": torch.zeros((0,), dtype=torch.bool),
                                "crosswalk": torch.zeros((0,), dtype=torch.bool),
                            },
                            "meta": {
                                "sample_id": sample_id,
                                "dataset_key": dataset_key,
                                "split": split,
                                "image_path": str(record.image_path),
                                "raw_hw": (720, 1280),
                                "network_hw": (608, 800),
                                "transform": {
                                    "scale": 0.625,
                                    "pad_left": 0,
                                    "pad_top": 79,
                                    "pad_right": 0,
                                    "pad_bottom": 79,
                                    "resized_hw": (450, 800),
                                },
                            },
                        }
                    )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        return self.samples[index]


class PV26BalancedSamplerTests(unittest.TestCase):
    def test_dataset_group_mapping_is_stable(self) -> None:
        self.assertEqual(dataset_group_for_key("bdd100k_det_100k"), "bdd100k")
        self.assertEqual(dataset_group_for_key("aihub_traffic_seoul"), "aihub_traffic")
        self.assertEqual(dataset_group_for_key("aihub_lane_seoul"), "aihub_lane")

    def test_balanced_batch_sampler_uses_expected_per_batch_counts(self) -> None:
        dataset = _ToyCanonicalDataset()
        sampler = PV26BalancedBatchSampler(dataset, batch_size=20, num_batches=2, split="train", seed=7)

        for batch_indices in sampler:
            counts = {"bdd100k": 0, "aihub_traffic": 0, "aihub_lane": 0}
            for index in batch_indices:
                group = dataset_group_for_key(dataset.records[index].dataset_key)
                counts[group] += 1
                self.assertEqual(dataset.records[index].split, "train")
            self.assertEqual(counts, {"bdd100k": 7, "aihub_traffic": 7, "aihub_lane": 6})

    def test_balanced_dataloader_respects_split_filter(self) -> None:
        dataset = _ToyCanonicalDataset()
        loader = build_pv26_train_dataloader(dataset, batch_size=20, num_batches=1, split="val", seed=11)

        batch = next(iter(loader))

        self.assertEqual(tuple(batch["image"].shape), (20, 3, 608, 800))
        self.assertTrue(all(item["split"] == "val" for item in batch["meta"]))
        groups = {dataset_group_for_key(item["dataset_key"]) for item in batch["meta"]}
        self.assertEqual(groups, {"bdd100k", "aihub_traffic", "aihub_lane"})


if __name__ == "__main__":
    unittest.main()
