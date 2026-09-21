from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image
import torch

from common.io import read_json, read_jsonl, write_json, write_jsonl_sorted
from tools.od_bootstrap.signal_attr import (
    BASE_COLOR_TO_INDEX,
    SIGNAL_ATTR_TEACHER_REASON_AMBIGUOUS_BITS,
    SIGNAL_ATTR_TEACHER_REASON_LOW_CONFIDENCE,
    SIGNAL_ATTR_TEACHER_REASON_NONFINITE_LOGITS,
    SIGNAL_ATTR_TEACHER_REASON_VALID,
    SignalAttrClassifierConfig,
    SignalAttrCropClassifier,
    SignalAttrCropTorchDataset,
    SignalAttrCropConfig,
    SignalAttrRuntime,
    build_signal_attr_focused_run,
    SignalAttrThresholdPolicy,
    SignalAttrTrainConfig,
    evaluate_signal_attr_checkpoint,
    evaluate_signal_attr_classifier,
    load_signal_attr_classifier_checkpoint,
    product_signal_attr_prediction_from_logits,
    signal_attr_collate,
    signal_attr_loss,
    signal_attr_prediction_from_logits,
    train_signal_attr_classifier,
)


class SignalAttrClassifierTests(unittest.TestCase):
    def test_macro_state_f1_penalizes_false_left_when_no_left_gt(self) -> None:
        class GreenModel(torch.nn.Module):
            def __init__(self, arrow_logit: float) -> None:
                super().__init__()
                self.arrow_logit = arrow_logit

            def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
                count = images.shape[0]
                return {
                    "base_color_logits": images.new_tensor([[-4.0, -4.0, -4.0, 8.0]]).expand(count, -1),
                    "arrow_logit": images.new_full((count,), self.arrow_logit),
                }

        batch = signal_attr_collate([{
            "image": torch.zeros((3, 2, 2)),
            "base_color_target": BASE_COLOR_TO_INDEX["green"],
            "arrow_target": 0.0,
            "arrow_target_valid": 1.0,
            "row": {"light_type": "car", "base_color": "green", "arrow": 0},
        }])
        correct = evaluate_signal_attr_classifier(
            GreenModel(-8.0), [batch], device=torch.device("cpu"), product_semantics=True
        )
        false_left = evaluate_signal_attr_classifier(
            GreenModel(8.0), [batch], device=torch.device("cpu"), product_semantics=True
        )
        self.assertEqual(correct["macro_state_f1"], 1.0)
        self.assertEqual(false_left["by_light_type"]["car"]["states"]["left_arrow"]["support"], 0)
        self.assertEqual(false_left["by_light_type"]["car"]["states"]["left_arrow"]["fp"], 1)
        self.assertEqual(false_left["macro_state_f1"], 0.5)

    def test_product_evaluation_reports_state_errors_and_valid_coverage(self) -> None:
        class IndexedModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.calls = 0
                self.register_buffer("colors", torch.tensor([
                    [-4.0, 8.0, -4.0, -4.0],
                    [-4.0, -4.0, -4.0, 8.0],
                    [-4.0, -4.0, -4.0, 8.0],
                    [8.0, -4.0, -4.0, -4.0],
                    [8.0, -4.0, -4.0, -4.0],
                ]))
                self.register_buffer("arrows", torch.tensor([8.0, -8.0, 0.0, 0.0, -8.0]))

            def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
                self.calls += 1
                indices = images[:, 0, 0, 0].long()
                return {"base_color_logits": self.colors[indices], "arrow_logit": self.arrows[indices]}

        labels = [
            ("car", "red", 1),
            ("car", "yellow", 0),
            ("pedestrian", "green", 0),
            ("pedestrian", "red", 0),
            ("car", "off", 0),
        ]
        batch = signal_attr_collate([
            {
                "image": torch.full((3, 2, 2), float(index)),
                "base_color_target": BASE_COLOR_TO_INDEX[color],
                "arrow_target": float(arrow),
                "arrow_target_valid": float(light_type == "car"),
                "row": {"light_type": light_type, "base_color": color, "arrow": arrow},
            }
            for index, (light_type, color, arrow) in enumerate(labels)
        ])
        model = IndexedModel()
        report = evaluate_signal_attr_classifier(
            model, [batch], device=torch.device("cpu"), product_semantics=True,
            all_off_is_valid=False,
        )
        self.assertEqual(model.calls, 1)
        self.assertAlmostEqual(report["valid_coverage"], 3 / 5)
        self.assertAlmostEqual(report["macro_state_f1"], 3 / 6)
        self.assertEqual(report["by_light_type"]["car"]["states"]["yellow"]["fn"], 1)
        self.assertEqual(report["by_light_type"]["car"]["states"]["green"]["fp"], 1)
        self.assertEqual(report["by_light_type"]["car"]["states"]["green"]["support"], 0)
        self.assertEqual(report["by_light_type"]["pedestrian"]["states"]["red"]["fn"], 1)

    def test_balanced_crop_sampler_exposes_rare_signal_states(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_root = root / "dataset"
            _write_signal_attr_dataset(dataset_root)
            write_json(dataset_root / "meta" / "signal_attr_dataset_manifest.json", {
                "state_semantics": "left_arrow", "all_off_is_valid": False,
            })
            labels = dataset_root / "labels" / "train.jsonl"
            by_id = {row["sample_id"]: dict(row) for row in read_jsonl(labels)}
            common = [
                {**by_id["green"], "sample_id": f"common_{index:02d}", "light_type": "car"}
                for index in range(30)
            ]
            rare = [
                {**by_id["red_arrow"], "light_type": "car"},
                {**by_id["yellow_arrow"], "light_type": "car"},
                {**by_id["green"], "sample_id": "pedestrian_green", "light_type": "pedestrian"},
            ]
            write_jsonl_sorted(labels, common + rare)
            options = dict(logical_batch_size=24, microbatch_size=4, device="cpu",
                           precision="fp32", initial_checkpoint=None, seed=26)
            natural = build_signal_attr_focused_run(dataset_root, root / "natural", sampling="natural", **options)
            balanced = build_signal_attr_focused_run(dataset_root, root / "balanced", sampling="balanced", **options)

            def selected_ids(run) -> list[str]:
                keys = next(iter(run.trainer.sampler))
                return [run.train_loader.dataset.rows[index]["sample_id"] for index, _ in keys]

            natural_ids = selected_ids(natural)
            balanced_ids = selected_ids(balanced)
            rare_ids = {"red_arrow", "yellow_arrow", "pedestrian_green"}
            self.assertEqual(balanced.group_counts["car:green:left=0"], 30)
            self.assertEqual(sum(balanced.group_counts.values()), 33)
            self.assertTrue(rare_ids.issubset(balanced_ids))
            self.assertGreater(sum(sample_id in rare_ids for sample_id in balanced_ids),
                               sum(sample_id in rare_ids for sample_id in natural_ids))
            self.assertEqual(balanced.trainer.run_metadata["sampling"], "balanced")
            balanced.trainer.fit(balanced.train_loader, max_steps=1, planned_steps=2)
            expected_next_ids = selected_ids(balanced)
            resumed = build_signal_attr_focused_run(
                dataset_root, root / "balanced", sampling="balanced", resume=True, **options
            )
            self.assertEqual(selected_ids(resumed), expected_next_ids)

    def test_resume_before_first_checkpoint_reuses_or_creates_frozen_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_root = root / "dataset"
            _write_signal_attr_dataset(dataset_root)
            write_json(dataset_root / "meta" / "signal_attr_dataset_manifest.json", {
                "state_semantics": "left_arrow", "all_off_is_valid": False,
            })
            options = dict(logical_batch_size=2, microbatch_size=1, device="cpu",
                           precision="fp32", initial_checkpoint=None)
            first = build_signal_attr_focused_run(dataset_root, root / "run_a", **options)
            initial_weight = next(first.trainer.model.parameters()).detach().clone()
            saved_colors = [row["base_color"] for row in first.train_loader.dataset.rows]
            live_labels = dataset_root / "labels" / "train.jsonl"
            changed_rows = [dict(row) for row in read_jsonl(live_labels)]
            changed_rows[0]["base_color"] = "yellow"
            write_jsonl_sorted(live_labels, changed_rows)
            recovered = build_signal_attr_focused_run(dataset_root, root / "run_a", resume=True, **options)
            self.assertEqual(recovered.trainer.global_step, 0)
            self.assertTrue(torch.equal(initial_weight, next(recovered.trainer.model.parameters())))
            self.assertEqual([row["base_color"] for row in recovered.train_loader.dataset.rows], saved_colors)

            before_snapshot = build_signal_attr_focused_run(
                dataset_root, root / "run_b", resume=True, **options
            )
            self.assertEqual(before_snapshot.trainer.global_step, 0)
            self.assertTrue(before_snapshot.snapshot_root.is_dir())
            self.assertEqual(before_snapshot.train_loader.dataset.rows[0]["base_color"], "yellow")

    def test_product_step_training_resume_and_runtime_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_root = root / "dataset"
            _write_signal_attr_dataset(dataset_root)
            write_json(dataset_root / "meta" / "signal_attr_dataset_manifest.json", {
                "state_semantics": "left_arrow", "all_off_is_valid": False,
            })
            for split in ("train", "val"):
                path = dataset_root / "labels" / f"{split}.jsonl"
                rows = [dict(row) for row in read_jsonl(path) if row["sample_id"] in ("red_arrow", "green")]
                for row in rows:
                    row["light_type"] = "pedestrian" if row["sample_id"] == "green" else "car"
                write_jsonl_sorted(path, rows)
            options = dict(
                logical_batch_size=2, microbatch_size=1, device="cpu", precision="fp32",
                num_workers=0, initial_checkpoint=None,
            )
            run = build_signal_attr_focused_run(dataset_root, root / "run", **options)
            first = run.trainer.fit(run.train_loader, max_steps=1, planned_steps=2)
            self.assertEqual(first["global_step"], 1)
            report = run.evaluate()
            run.trainer.update_best(float(report["combo_accuracy"]))
            deployment = run.publish_checkpoint(root / "best_signal_attr.pt")
            self.assertEqual(SignalAttrRuntime.from_checkpoint(deployment).state_semantics, "left_arrow")
            self.assertEqual(read_json(run.snapshot_root / "manifest.json")["state_semantics"], "left_arrow")
            published = torch.load(deployment, map_location="cpu", weights_only=False)
            self.assertEqual(published["model_config"]["width"], 24)

            live_labels = dataset_root / "labels" / "train.jsonl"
            changed_rows = [dict(row) for row in read_jsonl(live_labels)]
            changed_rows[0]["base_color"] = "yellow"
            write_jsonl_sorted(live_labels, changed_rows)
            resumed = build_signal_attr_focused_run(dataset_root, root / "run", resume=True, **options)
            self.assertEqual([row["base_color"] for row in resumed.train_loader.dataset.rows],
                             ["red", "green"])
            second = resumed.trainer.fit(resumed.train_loader, max_steps=2)
            self.assertEqual(second["global_step"], 2)
            self.assertEqual(resumed.trainer.sampler.position, 4)

    def test_product_training_preserves_state_meaning_and_reports_types(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_root = root / "dataset"
            _write_signal_attr_dataset(dataset_root)
            write_json(dataset_root / "meta" / "signal_attr_dataset_manifest.json", {
                "state_semantics": "left_arrow", "all_off_is_valid": False,
            })
            for split in ("train", "val"):
                label_path = dataset_root / "labels" / f"{split}.jsonl"
                rows = [dict(row) for row in read_jsonl(label_path) if row["sample_id"] in ("red_arrow", "green")]
                for row in rows:
                    row["light_type"] = "pedestrian" if row["sample_id"] == "green" else "car"
                write_jsonl_sorted(label_path, rows)
            output_root = root / "run"
            train_signal_attr_classifier(
                dataset_root,
                output_root,
                train_config=SignalAttrTrainConfig(epochs=1, batch_size=2, device="cpu", num_workers=0),
                model_config=SignalAttrClassifierConfig(input_size=32, width=4, dropout=0.0),
            )
            checkpoint = output_root / "best_signal_attr.pt"
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            self.assertEqual(payload["state_semantics"], "left_arrow")
            self.assertFalse(payload["all_off_is_valid"])
            report = evaluate_signal_attr_checkpoint(
                dataset_root, checkpoint, root / "eval", device="cpu", num_workers=0
            )
            self.assertEqual(report["state_semantics"], "left_arrow")
            self.assertEqual(report["by_light_type"]["car"]["sample_count"], 1)
            self.assertEqual(report["by_light_type"]["pedestrian"]["sample_count"], 1)
            self.assertIsNone(report["by_light_type"]["pedestrian"]["arrow_accuracy"])

    def test_pedestrian_arrow_is_not_supervised(self) -> None:
        outputs = {
            "base_color_logits": torch.tensor([[0.0, 8.0, 0.0, 0.0], [0.0, 8.0, 0.0, 0.0]]),
            "arrow_logit": torch.tensor([4.0, 4.0]),
        }
        batch = {
            "base_color_target": torch.tensor([1, 1]),
            "arrow_target": torch.tensor([1.0, 0.0]),
            "arrow_target_valid": torch.tensor([1.0, 0.0]),
        }
        losses = signal_attr_loss(outputs, batch)
        self.assertLess(float(losses["arrow"]), 0.03)

    def test_product_decoder_does_not_publish_unverified_all_off(self) -> None:
        all_off = product_signal_attr_prediction_from_logits(
            [8.0, 0.0, 0.0, 0.0], -8.0,
            light_type="car", all_off_is_valid=False,
        )
        left_only = product_signal_attr_prediction_from_logits(
            [8.0, 0.0, 0.0, 0.0], 8.0,
            light_type="car", all_off_is_valid=False,
        )
        pedestrian = product_signal_attr_prediction_from_logits(
            [0.0, 8.0, 0.0, 0.0], 0.0,
            light_type="pedestrian", all_off_is_valid=False,
        )
        pedestrian_off_rejected = product_signal_attr_prediction_from_logits(
            [8.0, 0.0, 0.0, 0.0], 8.0,
            light_type="pedestrian", all_off_is_valid=False,
        )
        pedestrian_off_valid = product_signal_attr_prediction_from_logits(
            [8.0, 0.0, 0.0, 0.0], 8.0,
            light_type="pedestrian", all_off_is_valid=True,
        )
        self.assertEqual((all_off.tl_attr_valid, all_off.collapse_reason), (0, "all_off_unverified"))
        self.assertEqual((left_only.tl_attr_valid, left_only.tl_bits["arrow"]), (1, 1))
        self.assertEqual((pedestrian.tl_attr_valid, pedestrian.tl_bits["arrow"]), (1, 0))
        self.assertEqual((pedestrian_off_rejected.tl_attr_valid, pedestrian_off_rejected.collapse_reason),
                         (0, "all_off_unverified"))
        self.assertEqual((pedestrian_off_valid.tl_attr_valid, pedestrian_off_valid.tl_bits),
                         (1, {"red": 0, "yellow": 0, "green": 0, "arrow": 0}))

    def test_runtime_batches_rois_and_does_not_relabel_legacy_arrow(self) -> None:
        class FixedModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.calls = 0

            def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
                self.calls += 1
                count = images.shape[0]
                return {
                    "base_color_logits": images.new_tensor([[-4.0, 8.0, -4.0, -4.0]]).expand(count, -1),
                    "arrow_logit": images.new_full((count,), 5.0),
                }

        detections = [
            {"id": 12, "class_name": "vehicle_signal", "bbox_xyxy": [5, 5, 25, 35]},
            {"id": 3, "class_name": "pedestrian_signal", "bbox_xyxy": [30, 5, 50, 35]},
            {"id": 99, "class_name": "lane", "bbox_xyxy": [1, 1, 20, 20]},
        ]
        model = FixedModel()
        runtime = SignalAttrRuntime(
            model,
            crop_config=SignalAttrCropConfig(input_size=16),
            threshold_policy=SignalAttrThresholdPolicy(),
            state_semantics="left_arrow",
            device=torch.device("cpu"),
        )
        rows = runtime.predict(Image.new("RGB", (64, 48)), detections)
        self.assertEqual(model.calls, 1)
        self.assertEqual([row["detection_id"] for row in rows], [12, 3])
        self.assertEqual((rows[0]["state_valid"], rows[0]["left_arrow"]), (True, 1))
        self.assertEqual((rows[1]["state_valid"], rows[1]["left_arrow"]), (True, None))

        runtime.state_semantics = "legacy_arrow"
        legacy_rows = runtime.predict(Image.new("RGB", (64, 48)), detections)
        self.assertEqual([row["state_valid"] for row in legacy_rows], [False, False])
        self.assertEqual([row["reason"] for row in legacy_rows], ["left_arrow_untrained"] * 2)

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
            logs: list[str] = []

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
                log_fn=logs.append,
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
            joined_logs = "\n".join(logs)
            self.assertIn("[teacher:signal_attr] train start", joined_logs)
            self.assertIn("[teacher:signal_attr] epoch 1/1 train start", joined_logs)
            self.assertIn("wait=", joined_logs)
            self.assertIn("compute=", joined_logs)
            self.assertIn("\n", joined_logs)
            self.assertIn("fwd=", joined_logs)
            self.assertIn("bwd=", joined_logs)
            self.assertIn("opt=", joined_logs)
            self.assertIn("[teacher:signal_attr] train done", joined_logs)

            loaded = load_signal_attr_classifier_checkpoint(best_checkpoint, device="cpu")
            self.assertEqual(loaded["model_config"].input_size, 32)

            eval_logs: list[str] = []
            eval_summary = evaluate_signal_attr_checkpoint(
                dataset_root,
                best_checkpoint,
                output_root / "eval",
                split="val",
                batch_size=2,
                device="cpu",
                num_workers=0,
                log_fn=eval_logs.append,
            )
            self.assertEqual(eval_summary["sample_count"], 4)
            self.assertEqual(set(eval_summary["bit_metrics"]), {"red", "yellow", "green", "arrow"})
            self.assertTrue((output_root / "eval" / "signal_attr_eval_report.json").is_file())
            self.assertTrue((output_root / "eval" / "signal_attr_predictions.jsonl").is_file())
            joined_eval_logs = "\n".join(eval_logs)
            self.assertIn("[teacher:signal_attr] eval start", joined_eval_logs)
            self.assertIn("[teacher:signal_attr] eval done", joined_eval_logs)


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
                        "light_type": "car",
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
