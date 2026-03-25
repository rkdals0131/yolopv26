from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn

from model.loss.spec import build_loss_spec
from runtime_support import has_yolo26_runtime


OD_CLASSES = tuple(build_loss_spec()["model_contract"]["od_classes"])
TL_CLASS_ID = OD_CLASSES.index("traffic_light")


def _default_components_for_test(matched_count: int, unmatched_count: int) -> dict[str, float | int]:
    return {
        "det_obj_loss": 0.2,
        "det_cls_matched_loss": 0.3,
        "det_cls_unmatched_neg_loss": 0.1,
        "det_iou_loss": 0.15,
        "det_l1_loss": 0.25,
        "det_cls_matched_count": int(matched_count),
        "det_cls_unmatched_neg_count": int(unmatched_count),
    }


def _make_encoded_batch(batch_size: int, q_det: int) -> dict:
    del q_det
    det_boxes = torch.zeros((batch_size, 3, 4), dtype=torch.float32)
    det_classes = torch.full((batch_size, 3), -1, dtype=torch.long)
    det_valid = torch.zeros((batch_size, 3), dtype=torch.bool)
    tl_bits = torch.zeros((batch_size, 3, 4), dtype=torch.float32)
    tl_mask = torch.zeros((batch_size, 3), dtype=torch.bool)

    lane = torch.zeros((batch_size, 12, 54), dtype=torch.float32)
    stop_line = torch.zeros((batch_size, 6, 9), dtype=torch.float32)
    crosswalk = torch.zeros((batch_size, 4, 17), dtype=torch.float32)
    lane_valid = torch.zeros((batch_size, 12), dtype=torch.bool)
    stop_line_valid = torch.zeros((batch_size, 6), dtype=torch.bool)
    crosswalk_valid = torch.zeros((batch_size, 4), dtype=torch.bool)

    for batch_index in range(batch_size):
        det_boxes[batch_index, 0] = torch.tensor([40.0, 50.0, 120.0, 180.0])
        det_boxes[batch_index, 1] = torch.tensor([220.0, 80.0, 280.0, 160.0])
        det_classes[batch_index, 0] = TL_CLASS_ID
        det_classes[batch_index, 1] = 0
        det_valid[batch_index, :2] = True
        tl_bits[batch_index, 0] = torch.tensor([1.0, 0.0, 0.0, 1.0])
        tl_mask[batch_index, 0] = True

        lane[batch_index, 0, 0] = 1.0
        lane[batch_index, 0, 1] = 1.0
        lane[batch_index, 0, 4] = 1.0
        lane[batch_index, 0, 6:38] = torch.linspace(0.0, 31.0, 32)
        lane[batch_index, 0, 38:54] = 1.0
        lane_valid[batch_index, 0] = True

        stop_line[batch_index, 0, 0] = 1.0
        stop_line[batch_index, 0, 1:9] = torch.linspace(0.0, 7.0, 8)
        stop_line_valid[batch_index, 0] = True

        crosswalk[batch_index, 0, 0] = 1.0
        crosswalk[batch_index, 0, 1:17] = torch.linspace(0.0, 15.0, 16)
        crosswalk_valid[batch_index, 0] = True

    return {
        "image": torch.randn(batch_size, 3, 608, 800),
        "det_gt": {
            "boxes_xyxy": det_boxes,
            "classes": det_classes,
            "valid_mask": det_valid,
        },
        "tl_attr_gt_bits": tl_bits,
        "tl_attr_gt_mask": tl_mask,
        "lane": lane,
        "stop_line": stop_line,
        "crosswalk": crosswalk,
        "mask": {
            "det_source": torch.ones(batch_size, dtype=torch.bool),
            "det_supervised_class_mask": torch.ones((batch_size, len(OD_CLASSES)), dtype=torch.bool),
            "det_allow_objectness_negatives": torch.ones(batch_size, dtype=torch.bool),
            "det_allow_unmatched_class_negatives": torch.ones(batch_size, dtype=torch.bool),
            "tl_attr_source": torch.ones(batch_size, dtype=torch.bool),
            "lane_source": torch.ones(batch_size, dtype=torch.bool),
            "stop_line_source": torch.ones(batch_size, dtype=torch.bool),
            "crosswalk_source": torch.ones(batch_size, dtype=torch.bool),
            "lane_valid": lane_valid,
            "stop_line_valid": stop_line_valid,
            "crosswalk_valid": crosswalk_valid,
        },
        "meta": [{"sample_id": f"sample_{index}"} for index in range(batch_size)],
    }


class _DummyAdapter:
    def __init__(self) -> None:
        self.raw_model = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.Conv2d(8, 8, kernel_size=3, padding=1),
            nn.Conv2d(8, 8, kernel_size=3, padding=1),
        )
        self.trunk = self.raw_model

    def freeze_trunk(self) -> None:
        for parameter in self.trunk.parameters():
            parameter.requires_grad = False

    def unfreeze_trunk(self) -> None:
        for parameter in self.trunk.parameters():
            parameter.requires_grad = True


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


class _OOMCriterion(nn.Module):
    def forward(self, predictions, encoded):  # type: ignore[override]
        del predictions, encoded
        raise RuntimeError("CUDA out of memory. simulated for test")


class _AssignmentFailureCriterion(nn.Module):
    def forward(self, predictions, encoded):  # type: ignore[override]
        del predictions, encoded
        from model.loss import PV26DetAssignmentUnavailable

        raise PV26DetAssignmentUnavailable("det_feature_metadata_invalid")


class _FiniteCriterion(nn.Module):
    def __init__(self, total: float = 1.0) -> None:
        super().__init__()
        self.total = float(total)
        self.last_det_assignment_mode = "task_aligned"
        self.last_lane_assignment_modes = {
            "lane": "hungarian",
            "stop_line": "hungarian",
            "crosswalk": "hungarian",
        }
        self.last_det_loss_breakdown = {
            "det_obj_loss": 0.2,
            "det_cls_matched_loss": 0.3,
            "det_cls_unmatched_neg_loss": 0.1,
            "det_iou_loss": 0.15,
            "det_l1_loss": 0.25,
            "det_cls_matched_count": 2,
            "det_cls_unmatched_neg_count": 4,
        }

    def forward(self, predictions, encoded):  # type: ignore[override]
        del encoded
        total = predictions["det"].sum() * 0.0 + self.total
        zero = predictions["det"].sum() * 0.0
        return {
            "total": total,
            "det": zero + 0.5,
            "tl_attr": zero + 0.1,
            "lane": zero + 0.2,
            "stop_line": zero + 0.1,
            "crosswalk": zero + 0.1,
        }


class PV26TrainerTests(unittest.TestCase):
    def test_stage_configuration_freezes_and_unfreezes_expected_modules(self) -> None:
        from model.training import configure_pv26_train_stage

        adapter = _DummyAdapter()
        heads = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 4))

        stage1 = configure_pv26_train_stage(adapter, heads, "stage_1_frozen_trunk_warmup")
        self.assertEqual(stage1["stage"], "stage_1_frozen_trunk_warmup")
        self.assertEqual(stage1["trainable_trunk_params"], 0)
        self.assertGreater(stage1["trainable_head_params"], 0)

        stage2 = configure_pv26_train_stage(adapter, heads, "stage_2_partial_unfreeze")
        self.assertGreater(stage2["trainable_trunk_params"], 0)
        self.assertLess(stage2["trainable_trunk_params"], sum(p.numel() for p in adapter.trunk.parameters()))

        stage3 = configure_pv26_train_stage(adapter, heads, "stage_3_end_to_end_finetune")
        self.assertEqual(
            stage3["trainable_trunk_params"],
            sum(parameter.numel() for parameter in adapter.trunk.parameters()),
        )

    @unittest.skipUnless(has_yolo26_runtime(), "requires ultralytics yolo26 runtime")
    def test_train_step_with_real_runtime_returns_finite_losses(self) -> None:
        from model.heads import PV26Heads
        from model.training import PV26Trainer
        from model.trunk import build_yolo26n_trunk

        adapter = build_yolo26n_trunk()
        heads = PV26Heads(in_channels=(64, 128, 256))
        trainer = PV26Trainer(adapter, heads, stage="stage_0_smoke")

        summary = trainer.train_step(_make_encoded_batch(batch_size=1, q_det=9975))

        self.assertEqual(summary["global_step"], 1)
        self.assertEqual(summary["batch_size"], 1)
        self.assertTrue(summary["successful"])
        self.assertIn("trunk", summary["optimizer_lrs"])
        self.assertIn("heads", summary["optimizer_lrs"])
        self.assertGreater(summary["losses"]["total"], 0.0)
        self.assertTrue(torch.isfinite(torch.tensor(summary["losses"]["total"])))
        self.assertIn("assignment", summary)
        self.assertIn("det", summary["assignment"])
        self.assertIn("det_components", summary)
        self.assertIn("det_cls_unmatched_neg_loss", summary["det_components"])
        self.assertIn("timing", summary)
        self.assertIn("iteration_sec", summary["timing"])
        self.assertIn("source_counts", summary)
        self.assertIn("det_supervision", summary)

    @unittest.skipUnless(has_yolo26_runtime(), "requires ultralytics yolo26 runtime")
    def test_trainer_checkpoint_and_history_roundtrip(self) -> None:
        from model.heads import PV26Heads
        from model.training import PV26Trainer
        from model.trunk import build_yolo26n_trunk

        adapter = build_yolo26n_trunk()
        heads = PV26Heads(in_channels=(64, 128, 256))
        trainer = PV26Trainer(adapter, heads, stage="stage_1_frozen_trunk_warmup")
        batch = _make_encoded_batch(batch_size=1, q_det=9975)

        trainer.train_step(batch)
        trainer.train_step(batch)
        history_summary = trainer.summarize_history()

        self.assertEqual(history_summary["steps"], 2)
        self.assertEqual(history_summary["stage"], "stage_1_frozen_trunk_warmup")
        self.assertIn("total", history_summary["losses"])
        self.assertIn("lane", history_summary["assignment"]["lane"])

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            history_path = trainer.save_history_jsonl(root / "history.jsonl")
            checkpoint_path = trainer.save_checkpoint(root / "checkpoints" / "trainer.pt", extra_state={"tag": "smoke"})

            history_lines = history_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(history_lines), 2)
            self.assertEqual(json.loads(history_lines[-1])["global_step"], 2)

            reloaded_trainer = PV26Trainer(
                build_yolo26n_trunk(),
                PV26Heads(in_channels=(64, 128, 256)),
                stage="stage_0_smoke",
            )
            checkpoint = reloaded_trainer.load_checkpoint(checkpoint_path, map_location="cpu")

            self.assertEqual(reloaded_trainer.stage, "stage_1_frozen_trunk_warmup")
            self.assertEqual(reloaded_trainer.global_step, 2)
            self.assertEqual(len(reloaded_trainer.history), 2)
            self.assertEqual(checkpoint["extra_state"]["tag"], "smoke")
            self.assertEqual(
                reloaded_trainer.stage_summary["trainable_head_params"],
                trainer.stage_summary["trainable_head_params"],
            )

    @unittest.skipUnless(has_yolo26_runtime(), "requires ultralytics yolo26 runtime")
    def test_fit_writes_epoch_history_and_checkpoints(self) -> None:
        from model.heads import PV26Heads
        from model.training import PV26Trainer
        from model.trunk import build_yolo26n_trunk

        adapter = build_yolo26n_trunk()
        heads = PV26Heads(in_channels=(64, 128, 256))
        trainer = PV26Trainer(adapter, heads, stage="stage_0_smoke")
        batch = _make_encoded_batch(batch_size=1, q_det=9975)

        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "fit_smoke"
            summary = trainer.fit(
                [batch, batch],
                epochs=1,
                val_loader=[batch],
                run_dir=run_dir,
                run_manifest_extra={"tag": "fit_smoke"},
            )

            self.assertEqual(summary["completed_epochs"], 1)
            self.assertEqual(summary["best_epoch"], 1)
            self.assertEqual(trainer.global_step, 2)
            self.assertEqual(len(trainer.epoch_history), 1)
            self.assertTrue((run_dir / "history" / "train_steps.jsonl").is_file())
            self.assertTrue((run_dir / "history" / "epochs.jsonl").is_file())
            self.assertTrue((run_dir / "checkpoints" / "last.pt").is_file())
            self.assertTrue((run_dir / "checkpoints" / "best.pt").is_file())
            self.assertTrue((run_dir / "checkpoints" / "epoch_001.pt").is_file())
            self.assertTrue((run_dir / "summary.json").is_file())
            self.assertTrue((run_dir / "run_manifest.json").is_file())

            step_lines = (run_dir / "history" / "train_steps.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(step_lines), 2)
            step_payload = json.loads(step_lines[-1])
            self.assertIn("timing", step_payload)
            self.assertIn("profile", step_payload)
            self.assertIn("progress", step_payload)
            self.assertIn("iteration_sec", step_payload["timing"])
            self.assertIn("iteration_sec", step_payload["profile"])
            self.assertEqual(step_payload["progress"]["iteration"], 2)

            summary_payload = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary_payload["completed_epochs"], 1)
            self.assertEqual(summary_payload["last_epoch"]["train"]["batches"], 2)
            self.assertEqual(summary_payload["last_epoch"]["train"]["attempted_batches"], 2)
            self.assertEqual(summary_payload["last_epoch"]["train"]["successful_batches"], 2)
            self.assertEqual(summary_payload["last_epoch"]["train"]["skipped_batches"], 0)
            self.assertEqual(summary_payload["last_epoch"]["val"]["batches"], 1)
            self.assertEqual(summary_payload["manifest_path"], str(run_dir / "run_manifest.json"))
            self.assertIn("timing_profile", summary_payload["last_epoch"]["train"])
            self.assertIn("det_components", summary_payload["last_epoch"]["train"])
            manifest_payload = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest_payload["extra"]["tag"], "fit_smoke")
            self.assertEqual(manifest_payload["artifacts"]["summary"], str(run_dir / "summary.json"))
            self.assertEqual(manifest_payload["trainer"]["log_every_n_steps"], 1)
            self.assertEqual(manifest_payload["trainer"]["profile_window"], 20)
            if manifest_payload["artifacts"]["tensorboard"]["enabled"]:
                self.assertTrue(any((run_dir / "tensorboard").glob("events.out.tfevents.*")))

    @unittest.skipUnless(has_yolo26_runtime(), "requires ultralytics yolo26 runtime")
    def test_train_step_supports_grad_accumulation(self) -> None:
        from model.heads import PV26Heads
        from model.training import PV26Trainer
        from model.trunk import build_yolo26n_trunk

        trainer = PV26Trainer(
            build_yolo26n_trunk(),
            PV26Heads(in_channels=(64, 128, 256)),
            stage="stage_0_smoke",
            accumulate_steps=2,
            grad_clip_norm=1.0,
        )
        batch = _make_encoded_batch(batch_size=1, q_det=9975)

        first = trainer.train_step(batch)
        second = trainer.train_step(batch)

        self.assertTrue(first["successful"])
        self.assertFalse(first["optimizer_step"])
        self.assertEqual(first["global_step"], 0)
        self.assertEqual(first["micro_step"], 1)
        self.assertTrue(second["successful"])
        self.assertTrue(second["optimizer_step"])
        self.assertEqual(second["global_step"], 1)
        self.assertEqual(second["micro_step"], 0)

    @unittest.skipUnless(has_yolo26_runtime(), "requires ultralytics yolo26 runtime")
    def test_train_step_skips_non_finite_loss_when_enabled(self) -> None:
        from model.heads import PV26Heads
        from model.training import PV26Trainer
        from model.trunk import build_yolo26n_trunk

        trainer = PV26Trainer(
            build_yolo26n_trunk(),
            PV26Heads(in_channels=(64, 128, 256)),
            criterion=_NaNCriterion(),
            skip_non_finite_loss=True,
        )

        summary = trainer.train_step(_make_encoded_batch(batch_size=1, q_det=8))

        self.assertEqual(summary["skipped_reason"], "non_finite_loss")
        self.assertFalse(summary["successful"])
        self.assertEqual(summary["global_step"], 0)
        self.assertEqual(trainer.skipped_steps, 1)

    @unittest.skipUnless(has_yolo26_runtime(), "requires ultralytics yolo26 runtime")
    def test_train_step_recovers_from_oom_guard(self) -> None:
        from model.heads import PV26Heads
        from model.training import PV26Trainer
        from model.trunk import build_yolo26n_trunk

        trainer = PV26Trainer(
            build_yolo26n_trunk(),
            PV26Heads(in_channels=(64, 128, 256)),
            criterion=_OOMCriterion(),
            oom_guard=True,
        )

        summary = trainer.train_step(_make_encoded_batch(batch_size=1, q_det=8))

        self.assertEqual(summary["skipped_reason"], "oom_recovered")
        self.assertFalse(summary["successful"])
        self.assertEqual(summary["global_step"], 0)
        self.assertEqual(trainer.skipped_steps, 1)

    def test_train_step_skips_assignment_failure_without_advancing_counters(self) -> None:
        from model.training import PV26Trainer

        trainer = PV26Trainer(
            _DummyAdapter(),
            nn.Identity(),
            criterion=_AssignmentFailureCriterion(),
            accumulate_steps=2,
        )
        trainer.forward_encoded_batch = lambda encoded: {  # type: ignore[method-assign]
            "det": torch.zeros((1, 2, 12), requires_grad=True),
            "tl_attr": torch.zeros((1, 2, 4), requires_grad=True),
            "lane": torch.zeros((1, 12, 54), requires_grad=True),
            "stop_line": torch.zeros((1, 6, 9), requires_grad=True),
            "crosswalk": torch.zeros((1, 4, 17), requires_grad=True),
        }

        summary = trainer.train_step(_make_encoded_batch(batch_size=1, q_det=2))

        self.assertFalse(summary["successful"])
        self.assertEqual(summary["skipped_reason"], "det_assignment_unavailable")
        self.assertEqual(summary["assignment"]["det"], "det_assignment_unavailable")
        self.assertEqual(summary["global_step"], 0)
        self.assertEqual(summary["micro_step"], 0)
        self.assertEqual(trainer.skipped_steps, 1)
        self.assertIn("det_feature_metadata_invalid", summary["skipped_reason_detail"])

    def test_train_epoch_aggregates_only_successful_batches(self) -> None:
        from model.training import PV26Trainer

        trainer = PV26Trainer(_DummyAdapter(), nn.Identity(), criterion=_FiniteCriterion(total=1.0))

        outcomes = iter(
            [
                {
                    "history_index": 1,
                    "global_step": 1,
                    "stage": trainer.stage,
                    "batch_size": 1,
                    "successful": True,
                    "losses": {"total": 2.0, "det": 1.0, "tl_attr": 0.0, "lane": 0.5, "stop_line": 0.25, "crosswalk": 0.25},
                    "det_components": _default_components_for_test(2, 8),
                    "optimizer_step": True,
                    "micro_step": 0,
                    "accumulate_steps": 1,
                    "skipped_reason": None,
                    "skipped_reason_detail": None,
                    "skipped_steps": 0,
                    "amp_enabled": False,
                    "gradient_scale": 1.0,
                    "optimizer_lrs": {"trunk": 1e-4},
                    "trainable": dict(trainer.stage_summary),
                    "assignment": {"det": "task_aligned", "lane": {"lane": "hungarian", "stop_line": "hungarian", "crosswalk": "hungarian"}},
                    "timing": {key: 0.01 for key in ("wait_sec", "load_sec", "forward_sec", "loss_sec", "backward_sec", "iteration_sec")},
                    "source_counts": {"det_source_samples": 1, "tl_attr_source_samples": 0, "lane_source_samples": 0, "stop_line_source_samples": 0, "crosswalk_source_samples": 0},
                    "det_supervision": {
                        "det_source_samples": 1,
                        "partial_det_samples": 1,
                        "objectness_negative_enabled_samples": 0,
                        "class_negative_enabled_samples": 1,
                        "partial_det_ratio": 1.0,
                        "supervised_class_sample_counts": {class_name: 0 for class_name in OD_CLASSES},
                        "gt_class_counts": {class_name: 0 for class_name in OD_CLASSES},
                    },
                },
                {
                    "history_index": 2,
                    "global_step": 1,
                    "stage": trainer.stage,
                    "batch_size": 1,
                    "successful": False,
                    "losses": {"total": float("nan"), "det": float("nan"), "tl_attr": float("nan"), "lane": float("nan"), "stop_line": float("nan"), "crosswalk": float("nan")},
                    "det_components": _default_components_for_test(0, 0),
                    "optimizer_step": False,
                    "micro_step": 0,
                    "accumulate_steps": 1,
                    "skipped_reason": "det_assignment_unavailable",
                    "skipped_reason_detail": "det_feature_metadata_invalid",
                    "skipped_steps": 1,
                    "amp_enabled": False,
                    "gradient_scale": 1.0,
                    "optimizer_lrs": {"trunk": 1e-4},
                    "trainable": dict(trainer.stage_summary),
                    "assignment": {"det": "det_assignment_unavailable", "lane": {}},
                    "timing": {key: 0.01 for key in ("wait_sec", "load_sec", "forward_sec", "loss_sec", "backward_sec", "iteration_sec")},
                    "source_counts": {"det_source_samples": 1, "tl_attr_source_samples": 0, "lane_source_samples": 0, "stop_line_source_samples": 0, "crosswalk_source_samples": 0},
                    "det_supervision": {
                        "det_source_samples": 1,
                        "partial_det_samples": 1,
                        "objectness_negative_enabled_samples": 0,
                        "class_negative_enabled_samples": 1,
                        "partial_det_ratio": 1.0,
                        "supervised_class_sample_counts": {class_name: 0 for class_name in OD_CLASSES},
                        "gt_class_counts": {class_name: 0 for class_name in OD_CLASSES},
                    },
                },
            ]
        )
        trainer.train_step = lambda batch, **kwargs: next(outcomes)  # type: ignore[method-assign]

        summary = trainer.train_epoch([_make_encoded_batch(batch_size=1, q_det=2), _make_encoded_batch(batch_size=1, q_det=2)], epoch=1)

        self.assertEqual(summary["attempted_batches"], 2)
        self.assertEqual(summary["successful_batches"], 1)
        self.assertEqual(summary["skipped_batches"], 1)
        self.assertEqual(summary["skipped_reasons"]["det_assignment_unavailable"], 1)
        self.assertEqual(summary["losses"]["total"]["mean"], 2.0)
        self.assertEqual(summary["det_components"]["det_cls_unmatched_neg_count"], 8)

    def test_train_epoch_fails_when_every_batch_is_skipped(self) -> None:
        from model.training import PV26Trainer

        trainer = PV26Trainer(_DummyAdapter(), nn.Identity(), criterion=_FiniteCriterion(total=1.0))
        trainer.train_step = lambda batch, **kwargs: {  # type: ignore[method-assign]
            "history_index": 1,
            "global_step": 0,
            "stage": trainer.stage,
            "batch_size": 1,
            "successful": False,
            "losses": {"total": float("nan"), "det": float("nan"), "tl_attr": float("nan"), "lane": float("nan"), "stop_line": float("nan"), "crosswalk": float("nan")},
            "det_components": _default_components_for_test(0, 0),
            "optimizer_step": False,
            "micro_step": 0,
            "accumulate_steps": 1,
            "skipped_reason": "det_assignment_unavailable",
            "skipped_reason_detail": "det_feature_metadata_invalid",
            "skipped_steps": 1,
            "amp_enabled": False,
            "gradient_scale": 1.0,
            "optimizer_lrs": {"trunk": 1e-4},
            "trainable": dict(trainer.stage_summary),
            "assignment": {"det": "det_assignment_unavailable", "lane": {}},
            "timing": {key: 0.01 for key in ("wait_sec", "load_sec", "forward_sec", "loss_sec", "backward_sec", "iteration_sec")},
            "source_counts": {"det_source_samples": 1, "tl_attr_source_samples": 0, "lane_source_samples": 0, "stop_line_source_samples": 0, "crosswalk_source_samples": 0},
            "det_supervision": {
                "det_source_samples": 1,
                "partial_det_samples": 1,
                "objectness_negative_enabled_samples": 0,
                "class_negative_enabled_samples": 1,
                "partial_det_ratio": 1.0,
                "supervised_class_sample_counts": {class_name: 0 for class_name in OD_CLASSES},
                "gt_class_counts": {class_name: 0 for class_name in OD_CLASSES},
            },
        }

        with self.assertRaisesRegex(ValueError, "zero successful batches"):
            trainer.train_epoch([_make_encoded_batch(batch_size=1, q_det=2)], epoch=1)

    @unittest.skipUnless(has_yolo26_runtime(), "requires ultralytics yolo26 runtime")
    def test_fit_auto_resume_continues_from_last_checkpoint(self) -> None:
        from model.heads import PV26Heads
        from model.training import PV26Trainer
        from model.trunk import build_yolo26n_trunk

        batch = _make_encoded_batch(batch_size=1, q_det=9975)
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "resume_fit"

            trainer = PV26Trainer(build_yolo26n_trunk(), PV26Heads(in_channels=(64, 128, 256)), stage="stage_0_smoke")
            first = trainer.fit([batch], epochs=1, val_loader=None, run_dir=run_dir)
            self.assertEqual(first["completed_epochs"], 1)

            resumed = PV26Trainer(build_yolo26n_trunk(), PV26Heads(in_channels=(64, 128, 256)), stage="stage_0_smoke")
            second = resumed.fit([batch], epochs=2, val_loader=None, run_dir=run_dir, auto_resume=True)

            self.assertTrue(second["auto_resumed"])
            self.assertEqual(second["resume_start_epoch"], 2)
            self.assertEqual(second["completed_epochs"], 2)
            self.assertEqual(resumed.global_step, 2)


if __name__ == "__main__":
    unittest.main()
