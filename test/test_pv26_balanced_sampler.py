from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import torch

from common.pv26_schema import (
    DET_SUPERVISION_BY_DATASET,
    EXHAUSTIVE_DATASET_KEY_BY_SOURCE,
    OD_CLASSES,
    SOURCE_MASK_BY_DATASET,
)
from model.data import (
    PV26BalancedBatchSampler,
    PV26SampleIdPositiveBatchSampler,
    PV26TaskCooccurrenceBatchSampler,
    PV26TaskPositiveMultiBatchSampler,
    build_pv26_eval_dataloader,
    build_pv26_train_dataloader,
    dataset_group_for_key,
)
from model.data.sampler import DATASET_GROUP_BY_KEY
from model.data.dataset import SampleRecord


class _ToyCanonicalDataset:
    def __init__(
        self,
        *,
        dataset_keys: tuple[tuple[str, int], ...] = (
            ("bdd100k_det_100k", 4),
            ("aihub_traffic_seoul", 4),
            ("aihub_obstacle_seoul", 4),
            ("aihub_lane_seoul", 4),
        ),
    ) -> None:
        self.records: list[SampleRecord] = []
        self.samples: list[dict] = []
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
                                "det_supervised_classes": self._det_supervised_classes(dataset_key),
                                "det_supervised_class_ids": self._det_supervised_class_ids(dataset_key),
                                "det_allow_objectness_negatives": False,
                                "det_allow_unmatched_class_negatives": dataset_key != "aihub_lane_seoul",
                            },
                        }
                    )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        return self.samples[index]

    @staticmethod
    def _det_supervised_classes(dataset_key: str) -> list[str]:
        mapping = {
            "pv26_exhaustive_bdd100k_det_100k": ["vehicle", "bike", "pedestrian", "traffic_cone", "obstacle", "traffic_light", "sign"],
            "pv26_exhaustive_aihub_traffic_seoul": ["vehicle", "bike", "pedestrian", "traffic_cone", "obstacle", "traffic_light", "sign"],
            "pv26_exhaustive_aihub_obstacle_seoul": ["vehicle", "bike", "pedestrian", "traffic_cone", "obstacle", "traffic_light", "sign"],
            "bdd100k_det_100k": ["vehicle", "bike", "pedestrian"],
            "aihub_traffic_seoul": ["traffic_light", "sign"],
            "aihub_obstacle_seoul": ["traffic_cone", "obstacle"],
            "aihub_lane_seoul": [],
        }
        return list(mapping[dataset_key])

    @classmethod
    def _det_supervised_class_ids(cls, dataset_key: str) -> list[int]:
        return [OD_CLASSES.index(class_name) for class_name in cls._det_supervised_classes(dataset_key)]


def _scene_flags(scene_path: Path) -> dict[str, bool]:
    payload = json.loads(scene_path.read_text(encoding="utf-8"))
    return {
        "lane": bool(payload.get("lanes")),
        "stop_line": bool((payload.get("tasks") or {}).get("has_stop_line")),
        "crosswalk": bool((payload.get("tasks") or {}).get("has_crosswalk")),
    }


class PV26BalancedSamplerTests(unittest.TestCase):
    def test_dataset_key_contracts_cover_loader_and_sampler_surfaces(self) -> None:
        expected_keys = {
            "bdd100k_det_100k",
            "aihub_traffic_seoul",
            "aihub_obstacle_seoul",
            "aihub_lane_seoul",
            *EXHAUSTIVE_DATASET_KEY_BY_SOURCE.values(),
        }

        self.assertEqual(set(SOURCE_MASK_BY_DATASET), expected_keys)
        self.assertEqual(set(DET_SUPERVISION_BY_DATASET), expected_keys)
        self.assertEqual(set(DATASET_GROUP_BY_KEY), expected_keys)

        for dataset_key in expected_keys:
            source_mask = SOURCE_MASK_BY_DATASET[dataset_key]
            det_policy = DET_SUPERVISION_BY_DATASET[dataset_key]
            self.assertEqual(
                set(source_mask),
                {"det", "tl_attr", "lane", "stop_line", "crosswalk"},
            )
            if source_mask["det"]:
                self.assertGreater(len(det_policy["class_names"]), 0, msg=dataset_key)
            else:
                self.assertEqual(tuple(det_policy["class_names"]), (), msg=dataset_key)

    def test_dataset_group_mapping_is_stable(self) -> None:
        self.assertEqual(dataset_group_for_key("bdd100k_det_100k"), "bdd100k")
        self.assertEqual(dataset_group_for_key("aihub_traffic_seoul"), "aihub_traffic")
        self.assertEqual(dataset_group_for_key("aihub_obstacle_seoul"), "aihub_obstacle")
        self.assertEqual(dataset_group_for_key("aihub_lane_seoul"), "aihub_lane")

    def test_dataset_group_mapping_accepts_exhaustive_dataset_keys(self) -> None:
        self.assertEqual(dataset_group_for_key("pv26_exhaustive_bdd100k_det_100k"), "bdd100k")
        self.assertEqual(dataset_group_for_key("pv26_exhaustive_aihub_traffic_seoul"), "aihub_traffic")
        self.assertEqual(dataset_group_for_key("pv26_exhaustive_aihub_obstacle_seoul"), "aihub_obstacle")

    def test_balanced_batch_sampler_supports_exhaustive_dataset_keys(self) -> None:
        dataset = _ToyCanonicalDataset(
            dataset_keys=(
                ("pv26_exhaustive_bdd100k_det_100k", 4),
                ("pv26_exhaustive_aihub_traffic_seoul", 4),
                ("pv26_exhaustive_aihub_obstacle_seoul", 4),
                ("aihub_lane_seoul", 4),
            ),
        )
        sampler = PV26BalancedBatchSampler(dataset, batch_size=20, num_batches=2, split="train", seed=7)

        for batch_indices in sampler:
            counts = {"bdd100k": 0, "aihub_traffic": 0, "aihub_obstacle": 0, "aihub_lane": 0}
            for index in batch_indices:
                group = dataset_group_for_key(dataset.records[index].dataset_key)
                counts[group] += 1
                self.assertEqual(dataset.records[index].split, "train")
            self.assertEqual(counts, {"bdd100k": 6, "aihub_traffic": 6, "aihub_obstacle": 3, "aihub_lane": 5})

    def test_balanced_batch_sampler_uses_expected_per_batch_counts(self) -> None:
        dataset = _ToyCanonicalDataset()
        sampler = PV26BalancedBatchSampler(dataset, batch_size=20, num_batches=2, split="train", seed=7)

        for batch_indices in sampler:
            counts = {"bdd100k": 0, "aihub_traffic": 0, "aihub_obstacle": 0, "aihub_lane": 0}
            for index in batch_indices:
                group = dataset_group_for_key(dataset.records[index].dataset_key)
                counts[group] += 1
                self.assertEqual(dataset.records[index].split, "train")
            self.assertEqual(counts, {"bdd100k": 6, "aihub_traffic": 6, "aihub_obstacle": 3, "aihub_lane": 5})

    def test_balanced_batch_sampler_epoch_length_ignores_zero_ratio_groups(self) -> None:
        dataset = _ToyCanonicalDataset(
            dataset_keys=(
                ("bdd100k_det_100k", 8),
                ("aihub_lane_seoul", 2),
            ),
        )
        sampler = PV26BalancedBatchSampler(
            dataset,
            batch_size=4,
            split="train",
            seed=7,
            ratios={
                "bdd100k": 0.0,
                "aihub_traffic": 0.0,
                "aihub_obstacle": 0.0,
                "aihub_lane": 1.0,
            },
        )

        self.assertEqual(len(sampler), 1)
        only_batch = next(iter(sampler))
        self.assertTrue(all(dataset.records[index].dataset_key == "aihub_lane_seoul" for index in only_batch))

    def test_task_positive_multi_sampler_uses_three_positive_slots_and_one_background_slot(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset = _ToyCanonicalDataset(
                dataset_keys=(
                    ("bdd100k_det_100k", 4),
                    ("aihub_lane_seoul", 12),
                ),
            )
            scene_payloads = (
                {"lanes": [{"points_xy": [[100, 80], [120, 240], [140, 420]], "visibility": [1, 1, 1]}]},
                {"tasks": {"has_stop_line": True}},
                {"tasks": {"has_crosswalk": True}},
            )
            for record_index, record in enumerate(dataset.records):
                scene_path = root / f"{record.sample_id}.json"
                if record.dataset_key == "aihub_lane_seoul":
                    payload = scene_payloads[int(record.sample_id.rsplit("_", 1)[-1]) % len(scene_payloads)]
                else:
                    payload = {"tasks": {}}
                scene_path.write_text(json.dumps(payload), encoding="utf-8")
                dataset.records[record_index] = replace(record, scene_path=scene_path)

            sampler = PV26TaskPositiveMultiBatchSampler(
                dataset,
                batch_size=4,
                task_names=["lane", "stopline", "crosswalk"],
                positive_fraction=0.75,
                num_batches=3,
                split="train",
                seed=7,
            )

            self.assertEqual(sampler.positive_count, 3)
            self.assertEqual(sampler.negative_count, 1)
            self.assertEqual(sampler.negative_pool_policy, "det_source_not_task_positive")
            for batch_indices in sampler:
                batch_records = [dataset.records[index] for index in batch_indices]
                self.assertEqual(len(batch_records), 4)
                self.assertEqual(
                    sum(record.dataset_key == "bdd100k_det_100k" for record in batch_records),
                    1,
                )
                self.assertEqual(
                    sum(record.dataset_key == "aihub_lane_seoul" for record in batch_records),
                    3,
                )

    def test_sample_id_positive_sampler_uses_manifest_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest_path = root / "hard_samples.csv"
            train_lane_id = "aihub_lane_seoul_train_1"
            train_bdd_id = "bdd100k_det_100k_train_2"
            val_lane_id = "aihub_lane_seoul_val_1"
            manifest_path.write_text(
                "\n".join(
                    [
                        "dataset_key,split,sample_id",
                        f"aihub_lane_seoul,train,{train_lane_id}",
                        f"bdd100k_det_100k,train,{train_bdd_id}",
                        f"aihub_lane_seoul,val,{val_lane_id}",
                    ]
                ),
                encoding="utf-8",
            )
            dataset = _ToyCanonicalDataset()

            sampler = PV26SampleIdPositiveBatchSampler(
                dataset,
                batch_size=4,
                manifest_path=str(manifest_path),
                positive_fraction=0.5,
                num_batches=2,
                split="train",
                seed=7,
            )

            self.assertEqual(sampler.positive_count, 2)
            self.assertEqual(sampler.positive_pool_size, 2)
            self.assertEqual(sampler.manifest_entry_count, 3)
            positive_ids = {train_lane_id, train_bdd_id}
            for batch_indices in sampler:
                batch_ids = {dataset.records[index].sample_id for index in batch_indices}
                self.assertEqual(len(batch_ids & positive_ids), 2)
                self.assertNotIn(val_lane_id, batch_ids)

    def test_sample_id_positive_sampler_accepts_text_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest_path = root / "hard_samples.txt"
            manifest_path.write_text(
                "\n".join(
                    [
                        "# comments are ignored",
                        "aihub_lane_seoul_train_0",
                        "aihub_lane_seoul_train_1",
                    ]
                ),
                encoding="utf-8",
            )
            dataset = _ToyCanonicalDataset()

            sampler = PV26SampleIdPositiveBatchSampler(
                dataset,
                batch_size=2,
                manifest_path=str(manifest_path),
                positive_fraction=1.0,
                num_batches=1,
                split="train",
                seed=7,
            )

            batch_ids = {dataset.records[index].sample_id for index in next(iter(sampler))}
            self.assertEqual(batch_ids, {"aihub_lane_seoul_train_0", "aihub_lane_seoul_train_1"})

    def test_sample_id_positive_dataloader_records_sampler_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest_path = root / "hard_samples.csv"
            manifest_path.write_text(
                "\n".join(
                    [
                        "dataset_key,sample_id",
                        "aihub_lane_seoul,aihub_lane_seoul_train_0",
                        "aihub_lane_seoul,aihub_lane_seoul_train_1",
                    ]
                ),
                encoding="utf-8",
            )
            dataset = _ToyCanonicalDataset()

            loader = build_pv26_train_dataloader(
                dataset,
                batch_size=4,
                num_batches=1,
                split="train",
                seed=7,
                task_positive_task=f"sample_ids:{manifest_path}",
                task_positive_fraction=1.0,
            )

            metadata = loader._sampling_metadata  # type: ignore[attr-defined]
            self.assertEqual(metadata["requested_task_positive_mode"], "sample_ids")
            self.assertEqual(metadata["sampler_type"], "PV26SampleIdPositiveBatchSampler")
            self.assertEqual(metadata["positive_pool_size"], 2)
            self.assertEqual(metadata["positive_manifest_entry_count"], 2)
            batch = next(iter(loader))
            self.assertEqual({item["sample_id"] for item in batch["meta"]}, {"aihub_lane_seoul_train_0", "aihub_lane_seoul_train_1"})

    def test_task_positive_multi_sampler_prefers_det_source_for_negative_slot(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset = _ToyCanonicalDataset(
                dataset_keys=(
                    ("bdd100k_det_100k", 4),
                    ("aihub_lane_seoul", 16),
                ),
            )
            scene_payloads = (
                {"lanes": [{"points_xy": [[100, 80], [120, 240], [140, 420]], "visibility": [1, 1, 1]}]},
                {"tasks": {"has_stop_line": True}},
                {"tasks": {"has_crosswalk": True}},
                {"tasks": {}},
            )
            for record_index, record in enumerate(dataset.records):
                scene_path = root / f"{record.sample_id}.json"
                if record.dataset_key == "aihub_lane_seoul":
                    payload = scene_payloads[int(record.sample_id.rsplit("_", 1)[-1]) % len(scene_payloads)]
                else:
                    payload = {"tasks": {}}
                scene_path.write_text(json.dumps(payload), encoding="utf-8")
                dataset.records[record_index] = replace(record, scene_path=scene_path)

            sampler = PV26TaskPositiveMultiBatchSampler(
                dataset,
                batch_size=4,
                task_names=["lane", "stopline", "crosswalk"],
                positive_fraction=0.75,
                num_batches=4,
                split="train",
                seed=11,
            )

            self.assertEqual(sampler.negative_pool_policy, "det_source_not_task_positive")
            for batch_indices in sampler:
                batch_records = [dataset.records[index] for index in batch_indices]
                self.assertEqual(
                    sum(record.dataset_key == "bdd100k_det_100k" for record in batch_records),
                    1,
                )

    def test_task_positive_multi_sampler_gives_extra_slot_to_first_task(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset = _ToyCanonicalDataset(
                dataset_keys=(
                    ("bdd100k_det_100k", 4),
                    ("aihub_lane_seoul", 18),
                ),
            )
            scene_payloads = (
                {"lanes": [{"points_xy": [[100, 80], [120, 240], [140, 420]], "visibility": [1, 1, 1]}]},
                {"tasks": {"has_stop_line": True}},
                {"tasks": {"has_crosswalk": True}},
            )
            for record_index, record in enumerate(dataset.records):
                scene_path = root / f"{record.sample_id}.json"
                if record.dataset_key == "aihub_lane_seoul":
                    payload = scene_payloads[int(record.sample_id.rsplit("_", 1)[-1]) % len(scene_payloads)]
                else:
                    payload = {"tasks": {}}
                scene_path.write_text(json.dumps(payload), encoding="utf-8")
                dataset.records[record_index] = replace(record, scene_path=scene_path)

            sampler = PV26TaskPositiveMultiBatchSampler(
                dataset,
                batch_size=4,
                task_names=["stopline", "lane", "crosswalk"],
                positive_fraction=1.0,
                num_batches=2,
                split="train",
                seed=13,
            )

            self.assertEqual(sampler.positive_count, 4)
            self.assertEqual(sampler.negative_count, 0)
            self.assertEqual(sampler.task_names, ["stop_line", "lane", "crosswalk"])
            for batch_indices in sampler:
                flags = [_scene_flags(dataset.records[index].scene_path) for index in batch_indices]
                self.assertEqual(sum(item["stop_line"] for item in flags), 2)
                self.assertEqual(sum(item["lane"] for item in flags), 1)
                self.assertEqual(sum(item["crosswalk"] for item in flags), 1)

    def test_task_cooccurrence_sampler_draws_records_with_all_requested_tasks(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset = _ToyCanonicalDataset(
                dataset_keys=(
                    ("bdd100k_det_100k", 4),
                    ("aihub_lane_seoul", 16),
                ),
            )
            scene_payloads = (
                {
                    "tasks": {"has_stop_line": True, "has_crosswalk": True},
                    "lanes": [{"points_xy": [[100, 80], [120, 240], [140, 420]], "visibility": [1, 1, 1]}],
                },
                {"tasks": {"has_stop_line": True}},
                {"tasks": {"has_crosswalk": True}},
                {"lanes": [{"points_xy": [[100, 80], [120, 240], [140, 420]], "visibility": [1, 1, 1]}]},
            )
            for record_index, record in enumerate(dataset.records):
                scene_path = root / f"{record.sample_id}.json"
                if record.dataset_key == "aihub_lane_seoul":
                    payload = scene_payloads[int(record.sample_id.rsplit("_", 1)[-1]) % len(scene_payloads)]
                else:
                    payload = {"tasks": {}}
                scene_path.write_text(json.dumps(payload), encoding="utf-8")
                dataset.records[record_index] = replace(record, scene_path=scene_path)

            sampler = PV26TaskCooccurrenceBatchSampler(
                dataset,
                batch_size=4,
                task_names=["lane", "stopline", "crosswalk"],
                positive_fraction=0.75,
                num_batches=3,
                split="train",
                seed=17,
            )

            self.assertEqual(sampler.positive_count, 3)
            self.assertEqual(sampler.negative_count, 1)
            self.assertEqual(sampler.negative_pool_policy, "det_source_not_task_cooccurrence")
            for batch_indices in sampler:
                flags = [_scene_flags(dataset.records[index].scene_path) for index in batch_indices]
                self.assertEqual(sum(all(item.values()) for item in flags), 3)

    def test_cooccurrence_task_positive_dataloader_records_sampler_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset = _ToyCanonicalDataset(
                dataset_keys=(
                    ("bdd100k_det_100k", 4),
                    ("aihub_lane_seoul", 8),
                ),
            )
            for record_index, record in enumerate(dataset.records):
                scene_path = root / f"{record.sample_id}.json"
                if record.dataset_key == "aihub_lane_seoul":
                    payload = {
                        "tasks": {"has_stop_line": True, "has_crosswalk": True},
                        "lanes": [{"points_xy": [[100, 80], [120, 240], [140, 420]], "visibility": [1, 1, 1]}],
                    }
                else:
                    payload = {"tasks": {}}
                scene_path.write_text(json.dumps(payload), encoding="utf-8")
                dataset.records[record_index] = replace(record, scene_path=scene_path)

            loader = build_pv26_train_dataloader(
                dataset,
                batch_size=4,
                num_batches=1,
                split="train",
                seed=19,
                task_positive_task="cooccur:lane,stopline,crosswalk",
                task_positive_fraction=1.0,
            )

            metadata = loader._sampling_metadata  # type: ignore[attr-defined]
            self.assertEqual(metadata["requested_task_positive_mode"], "cooccur")
            self.assertEqual(metadata["sampler_type"], "PV26TaskCooccurrenceBatchSampler")
            self.assertEqual(metadata["effective_task_positive_tasks"], ["lane", "stop_line", "crosswalk"])

    def test_multi_task_positive_dataloader_fails_fast_when_a_task_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset = _ToyCanonicalDataset(
                dataset_keys=(
                    ("bdd100k_det_100k", 4),
                    ("aihub_lane_seoul", 8),
                ),
            )
            scene_payloads = (
                {"lanes": [{"points_xy": [[100, 80], [120, 240], [140, 420]], "visibility": [1, 1, 1]}]},
                {"tasks": {"has_stop_line": True}},
            )
            for record_index, record in enumerate(dataset.records):
                scene_path = root / f"{record.sample_id}.json"
                if record.dataset_key == "aihub_lane_seoul":
                    payload = scene_payloads[int(record.sample_id.rsplit("_", 1)[-1]) % len(scene_payloads)]
                else:
                    payload = {"tasks": {}}
                scene_path.write_text(json.dumps(payload), encoding="utf-8")
                dataset.records[record_index] = replace(record, scene_path=scene_path)

            with self.assertRaisesRegex(ValueError, "task-positive multi sampler found no positive samples for task='crosswalk'"):
                build_pv26_train_dataloader(
                    dataset,
                    batch_size=4,
                    num_batches=1,
                    split="train",
                    seed=7,
                    task_positive_task="multi:lane,stopline,crosswalk",
                    task_positive_fraction=0.75,
                )

    def test_balanced_dataloader_respects_split_filter(self) -> None:
        dataset = _ToyCanonicalDataset()
        loader = build_pv26_train_dataloader(dataset, batch_size=20, num_batches=1, split="val", seed=11)

        batch = next(iter(loader))

        self.assertEqual(tuple(batch["image"].shape), (20, 3, 608, 800))
        self.assertTrue(all(item["split"] == "val" for item in batch["meta"]))
        groups = {dataset_group_for_key(item["dataset_key"]) for item in batch["meta"]}
        self.assertEqual(groups, {"bdd100k", "aihub_traffic", "aihub_obstacle", "aihub_lane"})

    def test_eval_dataloader_uses_random_val_subset_instead_of_prefix_slice(self) -> None:
        dataset = _ToyCanonicalDataset()
        loader = build_pv26_eval_dataloader(dataset, batch_size=5, num_batches=2, split="val")

        batches = list(loader)

        self.assertEqual(len(batches), 2)
        flat_sample_ids = [item["sample_id"] for batch in batches for item in batch["meta"]]
        sequential_prefix = [
            record.sample_id
            for record in dataset.records
            if record.split == "val"
        ][:10]
        self.assertEqual(len(flat_sample_ids), 10)
        self.assertEqual(len(set(flat_sample_ids)), 10)
        self.assertNotEqual(flat_sample_ids, sequential_prefix)

    def test_eval_dataloader_can_draw_random_val_subsets_without_prefix_bias(self) -> None:
        dataset = _ToyCanonicalDataset()
        loader = build_pv26_eval_dataloader(
            dataset,
            batch_size=5,
            num_batches=2,
            split="val",
            seed=11,
        )

        first_epoch = [item["sample_id"] for batch in loader for item in batch["meta"]]
        second_epoch = [item["sample_id"] for batch in loader for item in batch["meta"]]
        sequential_prefix = [
            record.sample_id
            for record in dataset.records
            if record.split == "val"
        ][:10]

        self.assertEqual(len(first_epoch), 10)
        self.assertEqual(len(set(first_epoch)), 10)
        self.assertNotEqual(first_epoch, sequential_prefix)
        self.assertEqual(len(second_epoch), 10)
        self.assertEqual(len(set(second_epoch)), 10)
        self.assertGreater(len(set(first_epoch) | set(second_epoch)), 10)

    def test_eval_dataloader_can_encode_batches_and_preserve_raw_bundle_for_metrics(self) -> None:
        dataset = _ToyCanonicalDataset()
        loader = build_pv26_eval_dataloader(
            dataset,
            batch_size=5,
            num_batches=1,
            split="val",
            encode_batches=True,
        )

        batch = next(iter(loader))

        self.assertIn("det_gt", batch)
        self.assertIn("_raw_batch", batch)
        self.assertEqual(len(batch["_raw_batch"]["meta"]), 5)
        self.assertTrue(all(item["split"] == "val" for item in batch["_raw_batch"]["meta"]))

    def test_balanced_dataloader_can_encode_batches_in_worker_collate(self) -> None:
        dataset = _ToyCanonicalDataset()
        loader = build_pv26_train_dataloader(
            dataset,
            batch_size=20,
            num_batches=1,
            split="val",
            seed=11,
            encode_batches=True,
        )

        batch = next(iter(loader))

        self.assertEqual(tuple(batch["image"].shape), (20, 3, 608, 800))
        self.assertIn("det_gt", batch)
        self.assertIn("mask", batch)
        self.assertNotIn("_raw_batch", batch)
        self.assertTrue(all(item["split"] == "val" for item in batch["meta"]))


if __name__ == "__main__":
    unittest.main()
