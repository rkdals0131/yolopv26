from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn

from model.data import PV26CanonicalDataset, collate_pv26_samples
from model.engine.trainer import PV26Trainer
from pv26_prepared_dataset_fixture import (
    EXHAUSTIVE_BDD_DATASET_KEY,
    EXHAUSTIVE_OBSTACLE_DATASET_KEY,
    EXHAUSTIVE_TRAFFIC_DATASET_KEY,
    AIHUB_LANE_DATASET_KEY,
    create_prepared_pv26_dataset,
    prepared_sample_id,
    select_prepared_samples,
)
from tools.run_pv26_train import _build_phase_train_loaders, PhaseConfig, TrainDefaultsConfig


class _DummyAdapter:
    def __init__(self) -> None:
        self.raw_model = nn.Identity()
        self.trunk = nn.Identity()

    def freeze_trunk(self) -> None:
        return None

    def unfreeze_trunk(self) -> None:
        return None


class _NaNCriterion(nn.Module):
    def forward(self, predictions, encoded):  # type: ignore[override]
        del predictions, encoded
        nan_value = torch.tensor(float("nan"), requires_grad=True)
        zero_value = torch.tensor(0.0)
        return {
            "total": nan_value,
            "det": zero_value,
            "tl_attr": zero_value,
            "lane": zero_value,
            "stop_line": zero_value,
            "crosswalk": zero_value,
        }


class _ConflictCriterion(nn.Module):
    stage = "stage_3_end_to_end_finetune"
    loss_weights = {"det": 1.0, "tl_attr": 1.0, "lane": 1.0, "stop_line": 1.0, "crosswalk": 1.0}
    last_det_assignment_mode = "unit"
    last_lane_assignment_modes: dict[str, str] = {}
    last_det_loss_breakdown: dict[str, float] = {}

    def forward(self, predictions, encoded):  # type: ignore[override]
        del encoded
        scalar = predictions["scalar"]
        det = scalar
        lane = -scalar
        zero = scalar * 0.0
        return {
            "total": det + lane + zero,
            "det": det,
            "tl_attr": zero,
            "lane": lane,
            "stop_line": zero,
            "crosswalk": zero,
        }


class _TinyAdapter:
    def __init__(self) -> None:
        self.raw_model = nn.Linear(1, 1, bias=False)
        self.trunk = self.raw_model

    def freeze_trunk(self) -> None:
        for parameter in self.trunk.parameters():
            parameter.requires_grad = False

    def unfreeze_trunk(self) -> None:
        for parameter in self.trunk.parameters():
            parameter.requires_grad = True


def _dummy_optimizer() -> torch.optim.Optimizer:
    return torch.optim.SGD([nn.Parameter(torch.tensor(0.0, requires_grad=True))], lr=1e-3)


def _default_phase(stage: str = "stage_1_frozen_trunk_warmup") -> PhaseConfig:
    return PhaseConfig(
        name=f"{stage}_sanity",
        stage=stage,
        min_epochs=1,
        max_epochs=1,
        patience=1,
    )


def _default_train_config() -> TrainDefaultsConfig:
    return TrainDefaultsConfig(
        device="cpu",
        batch_size=4,
        train_batches=1,
        val_batches=1,
        amp=False,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        prefetch_factor=None,
        backbone_variant="n",
    )


class PV26PreparedDatasetRuntimeSanityTests(unittest.TestCase):
    def test_trainer_applies_pcgrad_style_multitask_conflict_snapshot(self) -> None:
        adapter = _TinyAdapter()
        heads = nn.Linear(1, 1, bias=False)
        trainer = PV26Trainer(
            adapter,
            heads,
            stage="stage_3_end_to_end_finetune",
            device="cpu",
            criterion=_ConflictCriterion(),
            amp=False,
            accumulate_steps=1,
            grad_clip_norm=5.0,
            multitask_conflict={
                "enabled": True,
                "mode": "pcgrad_style",
                "tasks": ["det", "lane"],
            },
        )

        def _forward_encoded_batch(encoded):
            del encoded
            return {"scalar": adapter.raw_model.weight.sum() + heads.weight.sum()}

        trainer.forward_encoded_batch = _forward_encoded_batch  # type: ignore[method-assign]
        batch = {
            "image": torch.zeros((2, 3, 4, 4), dtype=torch.float32),
            "det_gt": {
                "valid_mask": torch.zeros((2, 0), dtype=torch.bool),
                "classes": torch.zeros((2, 0), dtype=torch.long),
            },
            "mask": {
                "det_source": torch.zeros((2,), dtype=torch.bool),
                "tl_attr_source": torch.zeros((2,), dtype=torch.bool),
                "lane_source": torch.zeros((2,), dtype=torch.bool),
                "stop_line_source": torch.zeros((2,), dtype=torch.bool),
                "crosswalk_source": torch.zeros((2,), dtype=torch.bool),
                "det_allow_objectness_negatives": torch.zeros((2,), dtype=torch.bool),
                "det_allow_unmatched_class_negatives": torch.zeros((2,), dtype=torch.bool),
                "det_supervised_class_mask": torch.zeros((2, 7), dtype=torch.bool),
            },
        }

        summary = trainer.train_step(batch)

        self.assertTrue(summary["successful"])
        self.assertTrue(summary["optimizer_step"])
        self.assertEqual(summary["multitask_conflict"]["mode"], "pcgrad_style")
        self.assertEqual(summary["multitask_conflict"]["tasks"], ["det", "lane"])
        self.assertIn(["det", "lane"], summary["multitask_conflict"]["conflict_pairs"])

    def test_loader_rejects_malformed_detection_label_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = create_prepared_pv26_dataset(Path(temp_dir) / "prepared")
            sample_id = prepared_sample_id(EXHAUSTIVE_TRAFFIC_DATASET_KEY, "train")
            det_path = root / "labels_det" / "train" / f"{sample_id}.txt"
            det_path.write_text("5 0.25 0.25 0.10\n", encoding="utf-8")

            dataset = PV26CanonicalDataset([root])
            index = next(i for i, record in enumerate(dataset.records) if record.sample_id == sample_id)

            with self.assertRaisesRegex(ValueError, "expected 5 columns"):
                dataset[index]

    def test_loader_rejects_out_of_range_detection_class_id(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = create_prepared_pv26_dataset(Path(temp_dir) / "prepared")
            sample_id = prepared_sample_id(EXHAUSTIVE_TRAFFIC_DATASET_KEY, "train")
            det_path = root / "labels_det" / "train" / f"{sample_id}.txt"
            det_path.write_text("999 0.25 0.25 0.10 0.10\n", encoding="utf-8")

            dataset = PV26CanonicalDataset([root])
            index = next(i for i, record in enumerate(dataset.records) if record.sample_id == sample_id)

            with self.assertRaisesRegex(ValueError, "invalid detection class id"):
                dataset[index]

    def test_loader_rejects_non_finite_detection_geometry(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = create_prepared_pv26_dataset(Path(temp_dir) / "prepared")
            sample_id = prepared_sample_id(EXHAUSTIVE_TRAFFIC_DATASET_KEY, "train")
            det_path = root / "labels_det" / "train" / f"{sample_id}.txt"
            det_path.write_text("1 nan 0.25 0.10 0.10\n", encoding="utf-8")

            dataset = PV26CanonicalDataset([root])
            index = next(i for i, record in enumerate(dataset.records) if record.sample_id == sample_id)

            with self.assertRaisesRegex(ValueError, "non-finite detection center_x"):
                dataset[index]

    def test_loader_rejects_invalid_normalized_detection_center(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = create_prepared_pv26_dataset(Path(temp_dir) / "prepared")
            sample_id = prepared_sample_id(EXHAUSTIVE_TRAFFIC_DATASET_KEY, "train")
            det_path = root / "labels_det" / "train" / f"{sample_id}.txt"
            det_path.write_text("1 1.25 0.25 0.10 0.10\n", encoding="utf-8")

            dataset = PV26CanonicalDataset([root])
            index = next(i for i, record in enumerate(dataset.records) if record.sample_id == sample_id)

            with self.assertRaisesRegex(ValueError, "invalid normalized detection center"):
                dataset[index]

    def test_loader_rejects_invalid_normalized_detection_size(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = create_prepared_pv26_dataset(Path(temp_dir) / "prepared")
            sample_id = prepared_sample_id(EXHAUSTIVE_TRAFFIC_DATASET_KEY, "train")
            det_path = root / "labels_det" / "train" / f"{sample_id}.txt"
            det_path.write_text("1 0.25 0.25 0.00 0.10\n", encoding="utf-8")

            dataset = PV26CanonicalDataset([root])
            index = next(i for i, record in enumerate(dataset.records) if record.sample_id == sample_id)

            with self.assertRaisesRegex(ValueError, "invalid normalized detection size"):
                dataset[index]

    def test_loader_requires_detection_labels_for_detector_supervised_sample(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = create_prepared_pv26_dataset(Path(temp_dir) / "prepared")
            sample_id = prepared_sample_id(EXHAUSTIVE_BDD_DATASET_KEY, "train")
            det_path = root / "labels_det" / "train" / f"{sample_id}.txt"
            det_path.unlink()

            dataset = PV26CanonicalDataset([root])
            index = next(i for i, record in enumerate(dataset.records) if record.sample_id == sample_id)

            with self.assertRaisesRegex(FileNotFoundError, "det label file not found"):
                dataset[index]

    def test_loader_raises_when_prepared_image_path_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = create_prepared_pv26_dataset(Path(temp_dir) / "prepared")
            sample_id = prepared_sample_id(AIHUB_LANE_DATASET_KEY, "train")
            image_path = root / "images" / "train" / f"{sample_id}.png"
            image_path.unlink()

            dataset = PV26CanonicalDataset([root])
            index = next(i for i, record in enumerate(dataset.records) if record.sample_id == sample_id)

            with self.assertRaises(FileNotFoundError):
                dataset[index]

    def test_loader_raises_when_scene_image_metadata_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = create_prepared_pv26_dataset(Path(temp_dir) / "prepared")
            sample_id = prepared_sample_id(EXHAUSTIVE_OBSTACLE_DATASET_KEY, "train")
            scene_path = root / "labels_scene" / "train" / f"{sample_id}.json"
            scene = json.loads(scene_path.read_text(encoding="utf-8"))
            scene["image"].pop("height", None)
            scene_path.write_text(json.dumps(scene, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

            dataset = PV26CanonicalDataset([root])
            index = next(i for i, record in enumerate(dataset.records) if record.sample_id == sample_id)

            with self.assertRaises(KeyError):
                dataset[index]

    def test_build_phase_train_loaders_requires_train_split(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = create_prepared_pv26_dataset(Path(temp_dir) / "prepared", splits=("val",))
            dataset = PV26CanonicalDataset([root])

            with self.assertRaisesRegex(ValueError, "balanced sampler found no eligible samples"):
                _build_phase_train_loaders(
                    dataset,
                    train_config=_default_train_config(),
                    phase=_default_phase(),
                )

    def test_build_phase_train_loaders_requires_val_split_when_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = create_prepared_pv26_dataset(Path(temp_dir) / "prepared", splits=("train",))
            dataset = PV26CanonicalDataset([root])

            with self.assertRaisesRegex(ValueError, "eval sampler found no eligible samples"):
                _build_phase_train_loaders(
                    dataset,
                    train_config=_default_train_config(),
                    phase=_default_phase(),
                )

    def test_stage4_requires_lane_family_dataset_presence(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = create_prepared_pv26_dataset(
                Path(temp_dir) / "prepared",
                dataset_keys=(
                    EXHAUSTIVE_BDD_DATASET_KEY,
                    EXHAUSTIVE_TRAFFIC_DATASET_KEY,
                    EXHAUSTIVE_OBSTACLE_DATASET_KEY,
                ),
            )
            dataset = PV26CanonicalDataset([root])

            with self.assertRaisesRegex(ValueError, "requires lane-family samples"):
                _build_phase_train_loaders(
                    dataset,
                    train_config=_default_train_config(),
                    phase=_default_phase("stage_4_lane_family_finetune"),
                )

    def test_prepared_batch_non_finite_loss_fails_fast(self) -> None:
        from model.engine.trainer import PV26Trainer

        with tempfile.TemporaryDirectory() as temp_dir:
            root = create_prepared_pv26_dataset(Path(temp_dir) / "prepared")
            dataset = PV26CanonicalDataset([root])
            train_samples = select_prepared_samples(
                dataset,
                split="train",
                dataset_keys=(
                    EXHAUSTIVE_BDD_DATASET_KEY,
                    EXHAUSTIVE_TRAFFIC_DATASET_KEY,
                    EXHAUSTIVE_OBSTACLE_DATASET_KEY,
                    AIHUB_LANE_DATASET_KEY,
                ),
            )
            batch = collate_pv26_samples(train_samples)
            trainer = PV26Trainer(
                _DummyAdapter(),
                nn.Identity(),
                criterion=_NaNCriterion(),
                optimizer=_dummy_optimizer(),
            )
            trainer.forward_encoded_batch = lambda encoded: {}  # type: ignore[method-assign]

            with self.assertRaisesRegex(FloatingPointError, "non-finite PV26 total loss encountered"):
                trainer.train_step(batch)


if __name__ == "__main__":
    unittest.main()
