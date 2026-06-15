from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image
import torch

from common.io import read_json, write_json
from tools.od_bootstrap.signal_attr import (
    BASE_COLOR_TO_INDEX,
    SIGNAL_ATTR_TEACHER_REASON_AMBIGUOUS_BITS,
    SIGNAL_ATTR_TEACHER_REASON_LOW_CONFIDENCE,
    SIGNAL_ATTR_TEACHER_REASON_NONFINITE_LOGITS,
    SIGNAL_ATTR_TEACHER_REASON_VALID,
    SignalAttrClassifierConfig,
    SignalAttrCropClassifier,
    SignalAttrCropTorchDataset,
    SignalAttrThresholdPolicy,
    SignalAttrTrainConfig,
    signal_attr_collate,
    signal_attr_prediction_from_logits,
    train_signal_attr_classifier,
)


class SignalAttrClassifierTests(unittest.TestCase):
    def test_signal_attr_classifier_outputs_base_color_and_arrow_logits(self) -> None:
        model = SignalAttrCropClassifier(SignalAttrClassifierConfig(width=4, dropout=0.0))

        outputs = model(torch.zeros((2, 3, 128, 128), dtype=torch.float32))

        self.assertEqual(tuple(outputs["base_color_logits"].shape), (2, 4))
        self.assertEqual(tuple(outputs["arrow_logit"].shape), (2,))

    def test_signal_attr_prediction_adapter_outputs_hard_bits_or_invalid_reason(self) -> None:
        valid = signal_attr_prediction_from_logits(
            [0.0, 8.0, -4.0, -4.0],
            4.0,
            policy=SignalAttrThresholdPolicy(base_color_min_confidence=0.70, arrow_ambiguity_band=0.0),
        )

        self.assertEqual(valid.collapse_reason, SIGNAL_ATTR_TEACHER_REASON_VALID)
        self.assertEqual(valid.tl_attr_valid, 1)
        self.assertEqual(valid.base_color, "red")
        self.assertEqual(valid.arrow, 1)
        self.assertEqual(valid.tl_bits, {"red": 1, "yellow": 0, "green": 0, "arrow": 1})

        low_confidence = signal_attr_prediction_from_logits(
            [0.0, 0.0, 0.0, 0.0],
            -5.0,
            policy=SignalAttrThresholdPolicy(base_color_min_confidence=0.70),
        )
        self.assertEqual(low_confidence.tl_attr_valid, 0)
        self.assertEqual(low_confidence.collapse_reason, SIGNAL_ATTR_TEACHER_REASON_LOW_CONFIDENCE)
        self.assertEqual(low_confidence.tl_bits, {"red": 0, "yellow": 0, "green": 0, "arrow": 0})

        ambiguous = signal_attr_prediction_from_logits(
            [-4.0, -4.0, 8.0, -4.0],
            0.0,
            policy=SignalAttrThresholdPolicy(base_color_min_confidence=0.70, arrow_ambiguity_band=0.20),
        )
        self.assertEqual(ambiguous.tl_attr_valid, 0)
        self.assertEqual(ambiguous.collapse_reason, SIGNAL_ATTR_TEACHER_REASON_AMBIGUOUS_BITS)

        nonfinite = signal_attr_prediction_from_logits([float("nan"), 0.0, 0.0, 0.0], 1.0)
        self.assertEqual(nonfinite.tl_attr_valid, 0)
        self.assertEqual(nonfinite.collapse_reason, SIGNAL_ATTR_TEACHER_REASON_NONFINITE_LOGITS)

    def test_signal_attr_crop_torch_dataset_reads_hard_labels(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write_signal_attr_dataset(root)

            dataset = SignalAttrCropTorchDataset(root, split="train", input_size=32, normalization="none")
            item = dataset[0]
            batch = signal_attr_collate([dataset[0], dataset[1]])

            self.assertEqual(len(dataset), 4)
            self.assertEqual(tuple(item["image"].shape), (3, 32, 32))
            self.assertEqual(item["base_color_target"], BASE_COLOR_TO_INDEX["red"])
            self.assertEqual(item["arrow_target"], 1.0)
            self.assertEqual(tuple(batch["image"].shape), (2, 3, 32, 32))
            self.assertEqual(batch["base_color_target"].tolist(), [BASE_COLOR_TO_INDEX["red"], BASE_COLOR_TO_INDEX["green"]])
            self.assertEqual(batch["arrow_target"].tolist(), [1.0, 0.0])

    def test_train_signal_attr_classifier_writes_best_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_root = root / "dataset"
            output_root = root / "run"
            _write_signal_attr_dataset(dataset_root)

            summary = train_signal_attr_classifier(
                dataset_root,
                output_root,
                train_config=SignalAttrTrainConfig(
                    epochs=1,
                    batch_size=2,
                    learning_rate=1.0e-3,
                    device="cpu",
                    num_workers=0,
                    seed=7,
                ),
                model_config=SignalAttrClassifierConfig(input_size=32, width=4, dropout=0.0),
                threshold_policy=SignalAttrThresholdPolicy(base_color_min_confidence=0.60),
            )

            best_checkpoint = output_root / "best_signal_attr.pt"
            last_checkpoint = output_root / "last_signal_attr.pt"
            summary_path = output_root / "train_summary.json"
            self.assertTrue(best_checkpoint.is_file())
            self.assertTrue(last_checkpoint.is_file())
            self.assertTrue(summary_path.is_file())
            self.assertEqual(summary["best_checkpoint"], str(best_checkpoint))
            self.assertEqual(read_json(summary_path)["history"][0]["val"]["sample_count"], 4)
            checkpoint = torch.load(best_checkpoint, map_location="cpu", weights_only=False)
            self.assertEqual(checkpoint["model_type"], "SignalAttrCropClassifier")
            self.assertEqual(checkpoint["base_colors"], ["off", "red", "yellow", "green"])
            self.assertEqual(checkpoint["tl_bits"], ["red", "yellow", "green", "arrow"])


def _write_signal_attr_dataset(root: Path) -> None:
    write_json(root / "meta" / "crop_config.json", {"input_size": 32, "normalization": "none"})
    rows = [
        ("red_arrow", "red", 1, {"red": 1, "yellow": 0, "green": 0, "arrow": 1}, "#ff0000"),
        ("green", "green", 0, {"red": 0, "yellow": 0, "green": 1, "arrow": 0}, "#00ff00"),
        ("yellow_arrow", "yellow", 1, {"red": 0, "yellow": 1, "green": 0, "arrow": 1}, "#ffff00"),
        ("off", "off", 0, {"red": 0, "yellow": 0, "green": 0, "arrow": 0}, "#202020"),
    ]
    for split in ("train", "val"):
        label_path = root / "labels" / f"{split}.jsonl"
        label_path.parent.mkdir(parents=True, exist_ok=True)
        serialized_rows: list[str] = []
        for sample_id, base_color, arrow, tl_bits, color in rows:
            crop_path = Path("images") / split / f"{sample_id}.jpg"
            image_path = root / crop_path
            image_path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (32, 32), color).save(image_path)
            serialized_rows.append(
                json.dumps(
                    {
                        "sample_id": sample_id,
                        "crop_path": crop_path.as_posix(),
                        "base_color": base_color,
                        "arrow": arrow,
                        "tl_bits": tl_bits,
                        "collapse_reason": "valid",
                    },
                    ensure_ascii=True,
                    sort_keys=True,
                )
            )
        label_path.write_text("\n".join(serialized_rows) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
