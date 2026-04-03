from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
import torch.nn as nn

from model.engine.loss import build_loss_spec
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
        from model.engine.loss import PV26DetAssignmentUnavailable

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


def _dummy_optimizer() -> torch.optim.Optimizer:
    return torch.optim.SGD([nn.Parameter(torch.tensor(0.0, requires_grad=True))], lr=1e-3)


class _FakeSummaryWriter:
    def __init__(self, log_dir: str) -> None:
        self.log_dir = log_dir
        self.scalars: list[tuple[str, float, int]] = []
        self.flushed = False
        self.closed = False
        self.layouts: list[dict] = []

    def add_scalar(self, name: str, value: float, global_step: int) -> None:
        self.scalars.append((name, float(value), int(global_step)))

    def add_custom_scalars(self, layout: dict) -> None:
        self.layouts.append(layout)

    def flush(self) -> None:
        self.flushed = True

    def close(self) -> None:
        self.closed = True


class PV26TrainerTests(unittest.TestCase):
    def test_format_train_live_detail_compacts_to_two_lines(self) -> None:
        from model.engine.trainer import _format_train_live_detail

        message = _format_train_live_detail(
            losses={
                "total": 1.25,
                "det": 0.5,
                "tl_attr": 0.1,
                "lane": 0.3,
                "stop_line": 0.2,
                "crosswalk": 0.15,
            },
            profile_summary={
                "iteration_sec": {"mean": 0.25, "p50": 0.2, "p99": 0.4},
                "wait_sec": {"mean": 0.01, "p50": 0.01, "p99": 0.01},
                "load_sec": {"mean": 0.02, "p50": 0.02, "p99": 0.02},
                "forward_sec": {"mean": 0.03, "p50": 0.03, "p99": 0.03},
                "loss_sec": {"mean": 0.04, "p50": 0.04, "p99": 0.04},
                "backward_sec": {"mean": 0.05, "p50": 0.05, "p99": 0.05},
            },
        )

        self.assertEqual(message.count("\n"), 0)
        self.assertIn("timing_ms  |  load=20.000", message)
        self.assertIn("fwd=30.000", message)
        self.assertIn("bwd=50.000", message)
        self.assertNotIn("total=1.2500", message)

    def test_format_train_progress_log_includes_phase_epoch_and_multiline_groups(self) -> None:
        from model.engine.trainer import _format_train_progress_log

        message = _format_train_progress_log(
            stage="stage_2_partial_unfreeze",
            phase_index=2,
            phase_count=3,
            phase_name="partial_unfreeze",
            epoch=4,
            epoch_total=8,
            batch_index=120,
            total_batches=1650,
            global_step=3420,
            epoch_started_at_iso="2026-03-26T12:34:56",
            elapsed_sec=125.0,
            eta_sec=333.0,
            losses={
                "total": 1.25,
                "det": 0.5,
                "tl_attr": 0.1,
                "lane": 0.3,
                "stop_line": 0.2,
                "crosswalk": 0.15,
            },
            profile_summary={
                "iteration_sec": {"mean": 0.25, "p50": 0.2, "p99": 0.4},
                "wait_sec": {"mean": 0.01, "p50": 0.01, "p99": 0.01},
                "load_sec": {"mean": 0.02, "p50": 0.02, "p99": 0.02},
                "forward_sec": {"mean": 0.03, "p50": 0.03, "p99": 0.03},
                "loss_sec": {"mean": 0.04, "p50": 0.04, "p99": 0.04},
                "backward_sec": {"mean": 0.05, "p50": 0.05, "p99": 0.05},
            },
        )

        self.assertIn("phase=2/3", message)
        self.assertIn("epoch=4/8", message)
        self.assertIn("iter=120/1650", message)
        self.assertIn("partial_unfreeze", message)
        self.assertIn("[#.......]   7%", message)
        self.assertGreaterEqual(message.count("\n"), 3)
        self.assertIn("  |  ", message)
        self.assertIn("  loss  |  total=1.2500", message)
        self.assertIn("  timing_ms  |  load=20.000", message)

    def test_format_validate_progress_log_includes_phase_epoch_and_eval_timing(self) -> None:
        from model.engine.trainer_reporting import _format_validate_progress_log

        message = _format_validate_progress_log(
            stage="stage_2_partial_unfreeze",
            phase_index=2,
            phase_count=3,
            phase_name="partial_unfreeze",
            epoch=4,
            epoch_total=8,
            batch_index=12,
            total_batches=165,
            epoch_started_at_iso="2026-03-26T12:34:56",
            elapsed_sec=125.0,
            eta_sec=333.0,
            batch_summary={
                "losses": {
                    "total": 1.25,
                    "det": 0.5,
                    "tl_attr": 0.1,
                    "lane": 0.3,
                    "stop_line": 0.2,
                    "crosswalk": 0.15,
                }
            },
            profile_summary={
                "iteration_sec": {"mean": 0.25, "p50": 0.2, "p99": 0.4},
                "wait_sec": {"mean": 0.01, "p50": 0.01, "p99": 0.01},
                "evaluate_sec": {"mean": 0.03, "p50": 0.03, "p99": 0.03},
            },
        )

        self.assertIn("phase=2/3", message)
        self.assertIn("epoch=4/8", message)
        self.assertIn("iter=12/165", message)
        self.assertIn("partial_unfreeze", message)
        self.assertGreaterEqual(message.count("\n"), 2)
        self.assertIn("  loss  |  det=0.5000", message)
        self.assertIn("  timing_ms  |  eval=30.000", message)

    def test_format_epoch_completion_log_is_concise_and_includes_checkpoint_state(self) -> None:
        from model.engine.trainer import _format_epoch_completion_log

        message = _format_epoch_completion_log(
            phase_index=1,
            phase_count=4,
            phase_name="head_warmup",
            epoch=2,
            epoch_total=12,
            train_summary={"losses": {"total": {"mean": 1.5}}},
            val_summary={"losses": {"total": {"mean": 1.25}}},
            best_metric_value=1.25,
            best_epoch=2,
            is_best=True,
        )

        self.assertIn("[epoch]", message)
        self.assertIn("phase=1/4", message)
        self.assertIn("head_warmup", message)
        self.assertIn("epoch=2/12", message)
        self.assertIn("train=1.5000", message)
        self.assertIn("val=1.2500", message)
        self.assertIn("best=1.2500@2", message)
        self.assertIn("checkpoint=last,best", message)

    def test_tensorboard_train_step_payload_keeps_only_core_scalars(self) -> None:
        import model.engine.trainer as pv26_trainer

        summary = {
            "successful": True,
            "losses": {
                "total": 10.0,
                "det": 1.0,
                "tl_attr": 0.5,
                "lane": 2.0,
                "stop_line": 3.0,
                "crosswalk": 4.0,
            },
            "optimizer_lrs": {"trunk": 1e-4, "heads": 5e-3, "aux": 1e-3},
            "timing": {key: 0.1 for key in pv26_trainer.TIMING_KEYS},
            "source_counts": {
                "det_source_samples": 8,
                "tl_attr_source_samples": 4,
                "lane_source_samples": 2,
                "stop_line_source_samples": 2,
                "crosswalk_source_samples": 2,
            },
            "det_supervision": {
                "partial_det_ratio": 0.75,
                "det_source_samples": 8,
            },
            "det_components": {
                "det_obj_loss": 0.2,
                "det_cls_matched_loss": 0.3,
            },
            "optimizer_step": True,
            "micro_step": 0,
            "skipped_steps": 0,
            "amp_enabled": True,
            "gradient_scale": 1024.0,
            "skipped_reason": None,
        }

        payload = pv26_trainer._tensorboard_train_step_payload(summary)
        scalar_names = {name for name, _ in pv26_trainer._flatten_scalar_tree("train_step", payload)}

        self.assertIn("train_step/loss/total", scalar_names)
        self.assertIn("train_step/profile_sec/iteration_sec", scalar_names)
        self.assertNotIn("train_step/lr/trunk", scalar_names)
        self.assertNotIn("train_step/health/gradient_scale", scalar_names)
        self.assertNotIn("train_step/source/det_source_samples", scalar_names)
        self.assertNotIn("train_step/det_supervision/partial_det_ratio", scalar_names)
        self.assertNotIn("train_step/det_components/det_obj_loss", scalar_names)

    def test_tensorboard_epoch_payload_keeps_named_lr_loss_and_val_metrics_only(self) -> None:
        import model.engine.trainer as pv26_trainer

        epoch_summary = {
            "train": {
                "losses": {
                    "total": {"mean": 1.0},
                    "det": {"mean": 0.2},
                },
                "duration_sec": 12.5,
                "timing_profile": {
                    "iteration_sec": {"mean": 0.4},
                },
                "optimizer_lrs": {"trunk": 1e-4, "heads": 5e-4},
            },
            "val": {
                "losses": {
                    "total": {"mean": 0.8},
                    "det": {"mean": 0.1},
                },
                "duration_sec": 3.0,
                "metrics": {
                    "detector": {"precision": 0.5, "recall": 0.4, "f1": 0.44, "map50": 0.6},
                    "traffic_light": {"combo_accuracy": 0.75, "mean_f1": 0.7},
                    "lane": {"precision": 0.8, "recall": 0.7, "f1": 0.74},
                },
            },
        }

        payload = pv26_trainer._tensorboard_epoch_payload(epoch_summary)
        scalar_names = {name for name, _ in pv26_trainer._flatten_scalar_tree("epoch", payload)}

        self.assertIn("epoch/lr/trunk", scalar_names)
        self.assertIn("epoch/lr/heads", scalar_names)
        self.assertIn("epoch/train/loss_mean/total", scalar_names)
        self.assertIn("epoch/val/loss_mean/total", scalar_names)
        self.assertIn("epoch/val/metrics/detector/map50", scalar_names)
        self.assertIn("epoch/val/metrics/lane/f1", scalar_names)
        self.assertNotIn("epoch/train/duration_sec", scalar_names)
        self.assertNotIn("epoch/train/profile_sec/iteration_sec", scalar_names)
        self.assertNotIn("epoch/val/duration_sec", scalar_names)

    def test_stage_configuration_freezes_and_unfreezes_expected_modules(self) -> None:
        from model.engine.trainer import configure_pv26_train_stage

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

    def test_stage4_freezes_trunk_and_detector_tl_heads_but_keeps_lane_family_trainable(self) -> None:
        from model.engine.trainer import configure_pv26_train_stage
        from model.net import PV26Heads

        adapter = _DummyAdapter()
        heads = PV26Heads(in_channels=(64, 128, 256))

        stage4 = configure_pv26_train_stage(adapter, heads, "stage_4_lane_family_finetune")

        self.assertEqual(stage4["stage"], "stage_4_lane_family_finetune")
        self.assertEqual(stage4["trainable_trunk_params"], 0)
        self.assertEqual(stage4["trainable_det_head_params"], 0)
        self.assertEqual(stage4["trainable_tl_attr_head_params"], 0)
        self.assertGreater(stage4["trainable_lane_family_head_params"], 0)
        self.assertEqual(stage4["head_training_policy"], "lane_family_only")
        self.assertFalse(any(parameter.requires_grad for parameter in adapter.trunk.parameters()))
        self.assertFalse(any(parameter.requires_grad for parameter in heads.det_heads.parameters()))
        self.assertFalse(any(parameter.requires_grad for parameter in heads.tl_attr_heads.parameters()))
        self.assertTrue(any(parameter.requires_grad for parameter in heads.lane_head.parameters()))
        self.assertTrue(any(parameter.requires_grad for parameter in heads.stop_line_head.parameters()))
        self.assertTrue(any(parameter.requires_grad for parameter in heads.crosswalk_head.parameters()))

    @unittest.skipUnless(has_yolo26_runtime(), "requires ultralytics yolo26 runtime")
    def test_train_step_with_real_runtime_returns_finite_losses(self) -> None:
        from model.net import PV26Heads
        from model.engine.trainer import PV26Trainer
        from model.net import build_yolo26n_trunk

        adapter = build_yolo26n_trunk()
        heads = PV26Heads(in_channels=(64, 128, 256))
        trainer = PV26Trainer(adapter, heads, stage="stage_1_frozen_trunk_warmup")

        summary = trainer.train_step(_make_encoded_batch(batch_size=1, q_det=9975))

        self.assertEqual(summary["global_step"], 1)
        self.assertEqual(summary["batch_size"], 1)
        self.assertTrue(summary["successful"])
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
        from model.net import PV26Heads
        from model.engine.loss import PV26MultiTaskLoss
        from model.engine.trainer import PV26Trainer
        from model.net import build_yolo26n_trunk

        adapter = build_yolo26n_trunk()
        heads = PV26Heads(in_channels=(64, 128, 256))
        criterion = PV26MultiTaskLoss(
            stage="stage_1_frozen_trunk_warmup",
            det_cls_negative_weight=0.2,
        )
        trainer = PV26Trainer(adapter, heads, stage="stage_1_frozen_trunk_warmup", criterion=criterion)
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
            checkpoint_path = trainer.save_checkpoint(root / "checkpoints" / "trainer.pt", extra_state={"tag": "regression"})

            history_lines = history_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(history_lines), 2)
            self.assertEqual(json.loads(history_lines[-1])["global_step"], 2)

            reloaded_trainer = PV26Trainer(
                build_yolo26n_trunk(),
                PV26Heads(in_channels=(64, 128, 256)),
                stage="stage_1_frozen_trunk_warmup",
            )
            checkpoint = reloaded_trainer.load_checkpoint(checkpoint_path, map_location="cpu")

            self.assertEqual(reloaded_trainer.stage, "stage_1_frozen_trunk_warmup")
            self.assertEqual(reloaded_trainer.global_step, 2)
            self.assertEqual(len(reloaded_trainer.history), 2)
            self.assertEqual(checkpoint["extra_state"]["tag"], "regression")
            self.assertEqual(
                reloaded_trainer.stage_summary["trainable_head_params"],
                trainer.stage_summary["trainable_head_params"],
            )
            self.assertEqual(checkpoint["criterion_config"]["stage"], "stage_1_frozen_trunk_warmup")
            self.assertAlmostEqual(float(reloaded_trainer.criterion.det_cls_negative_weight), 0.2)

    @unittest.skipUnless(has_yolo26_runtime(), "requires ultralytics yolo26 runtime")
    def test_fit_writes_epoch_history_and_checkpoints(self) -> None:
        from model.net import PV26Heads
        from model.engine.trainer import PV26Trainer
        from model.net import build_yolo26n_trunk

        adapter = build_yolo26n_trunk()
        heads = PV26Heads(in_channels=(64, 128, 256))
        trainer = PV26Trainer(adapter, heads, stage="stage_1_frozen_trunk_warmup")
        batch = _make_encoded_batch(batch_size=1, q_det=9975)

        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "fit_regression"
            summary = trainer.fit(
                [batch, batch],
                epochs=1,
                val_loader=[batch],
                run_dir=run_dir,
                run_manifest_extra={"tag": "fit_regression"},
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
            self.assertEqual(manifest_payload["extra"]["tag"], "fit_regression")
            self.assertEqual(manifest_payload["artifacts"]["summary"], str(run_dir / "summary.json"))
            self.assertEqual(manifest_payload["trainer"]["log_every_n_steps"], 1)
            self.assertEqual(manifest_payload["trainer"]["profile_window"], 20)
            if manifest_payload["artifacts"]["tensorboard"]["enabled"]:
                self.assertTrue(any((run_dir / "tensorboard").glob("events.out.tfevents.*")))

    def test_validate_epoch_aggregates_metrics_from_encoded_eval_batches(self) -> None:
        from model.data import encode_pv26_batch
        from model.engine.trainer import PV26Trainer
        from test_pv26_eval_metrics import make_prediction_bundle, make_raw_sample_batch

        trainer = PV26Trainer(
            _DummyAdapter(),
            nn.Identity(),
            criterion=_FiniteCriterion(),
            stage="stage_1_frozen_trunk_warmup",
            optimizer=_dummy_optimizer(),
        )
        raw_batch = make_raw_sample_batch()
        encoded_batch = encode_pv26_batch(raw_batch)
        encoded_batch["_raw_batch"] = {
            "det_targets": list(raw_batch["det_targets"]),
            "tl_attr_targets": list(raw_batch["tl_attr_targets"]),
            "lane_targets": list(raw_batch["lane_targets"]),
            "source_mask": list(raw_batch["source_mask"]),
            "valid_mask": list(raw_batch["valid_mask"]),
            "meta": list(raw_batch["meta"]),
        }
        evaluator = mock.Mock()
        evaluator.evaluate_batch.return_value = {
            "losses": {"total": 1.0, "det": 0.2, "tl_attr": 0.1, "lane": 0.3, "stop_line": 0.2, "crosswalk": 0.2},
            "counts": {"det_gt": 2, "tl_attr_gt": 1, "lane_rows": 1, "stop_line_rows": 1, "crosswalk_rows": 1},
            "metrics": {},
            "predictions": make_prediction_bundle(),
        }

        summary = trainer.validate_epoch([encoded_batch], epoch=1, evaluator=evaluator)

        self.assertEqual(summary["metrics"]["lane"]["tp"], 1)
        self.assertIn("detector", summary["metrics"])
        evaluator.evaluate_batch.assert_called_once_with(encoded_batch, include_predictions=True)

    @unittest.skipUnless(has_yolo26_runtime(), "requires ultralytics yolo26 runtime")
    def test_train_step_supports_grad_accumulation(self) -> None:
        from model.net import PV26Heads
        from model.engine.trainer import PV26Trainer
        from model.net import build_yolo26n_trunk

        trainer = PV26Trainer(
            build_yolo26n_trunk(),
            PV26Heads(in_channels=(64, 128, 256)),
            stage="stage_1_frozen_trunk_warmup",
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
        from model.net import PV26Heads
        from model.engine.trainer import PV26Trainer
        from model.net import build_yolo26n_trunk

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
        from model.net import PV26Heads
        from model.engine.trainer import PV26Trainer
        from model.net import build_yolo26n_trunk

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
        from model.engine.trainer import PV26Trainer

        trainer = PV26Trainer(
            _DummyAdapter(),
            nn.Identity(),
            criterion=_AssignmentFailureCriterion(),
            accumulate_steps=2,
            optimizer=_dummy_optimizer(),
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
        from model.engine.trainer import PV26Trainer

        trainer = PV26Trainer(_DummyAdapter(), nn.Identity(), criterion=_FiniteCriterion(total=1.0), optimizer=_dummy_optimizer())

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
        self.assertEqual(summary["attempted_source_counts"]["det_source_samples"], 2)
        self.assertEqual(summary["skipped_source_counts"]["det_source_samples"], 1)
        self.assertEqual(summary["source_counts"]["det_source_samples"], 1)

    def test_build_evaluator_clones_pv26_criterion_config(self) -> None:
        from model.engine.evaluator import PV26Evaluator
        from model.engine.loss import PV26MultiTaskLoss
        from model.engine.trainer import PV26Trainer

        criterion = PV26MultiTaskLoss(stage="stage_1_frozen_trunk_warmup", det_cls_negative_weight=0.2)
        trainer = PV26Trainer(_DummyAdapter(), nn.Identity(), criterion=criterion, optimizer=_dummy_optimizer())

        evaluator = trainer.build_evaluator()

        self.assertIsInstance(evaluator, PV26Evaluator)
        self.assertIsNot(evaluator.criterion, trainer.criterion)
        self.assertEqual(evaluator.criterion.export_config(), trainer.criterion.export_config())

    def test_train_epoch_fails_when_every_batch_is_skipped(self) -> None:
        from model.engine.trainer import PV26Trainer

        trainer = PV26Trainer(_DummyAdapter(), nn.Identity(), criterion=_FiniteCriterion(total=1.0), optimizer=_dummy_optimizer())
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

        with self.assertRaisesRegex(
            ValueError,
            "zero successful batches.*det_assignment_unavailable.*det_feature_metadata_invalid",
        ):
            trainer.train_epoch([_make_encoded_batch(batch_size=1, q_det=2)], epoch=1)

    @unittest.skipUnless(has_yolo26_runtime(), "requires ultralytics yolo26 runtime")
    def test_fit_auto_resume_continues_from_last_checkpoint(self) -> None:
        from model.net import PV26Heads
        from model.engine.trainer import PV26Trainer
        from model.net import build_yolo26n_trunk

        batch = _make_encoded_batch(batch_size=1, q_det=9975)
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "resume_fit"

            trainer = PV26Trainer(build_yolo26n_trunk(), PV26Heads(in_channels=(64, 128, 256)), stage="stage_1_frozen_trunk_warmup")
            first = trainer.fit([batch], epochs=1, val_loader=None, run_dir=run_dir)
            self.assertEqual(first["completed_epochs"], 1)

            resumed = PV26Trainer(build_yolo26n_trunk(), PV26Heads(in_channels=(64, 128, 256)), stage="stage_1_frozen_trunk_warmup")
            second = resumed.fit([batch], epochs=2, val_loader=None, run_dir=run_dir, auto_resume=True)

            self.assertTrue(second["auto_resumed"])
            self.assertEqual(second["resume_start_epoch"], 2)
            self.assertEqual(second["completed_epochs"], 2)
            self.assertEqual(resumed.global_step, 2)

    @unittest.skipUnless(has_yolo26_runtime(), "requires ultralytics yolo26 runtime")
    def test_fit_auto_resume_purges_stale_tensorboard_steps_and_continues_step_index(self) -> None:
        from model.net import PV26Heads
        from model.engine.trainer import PV26Trainer
        from model.net import build_yolo26n_trunk
        import model.engine.trainer as pv26_trainer

        batch = _make_encoded_batch(batch_size=1, q_det=9975)
        writer_calls: list[tuple[Path, int | None, _FakeSummaryWriter]] = []

        def _fake_build_summary_writer(log_dir: Path, *, purge_step: int | None = None):
            writer = _FakeSummaryWriter(str(log_dir))
            writer_calls.append((Path(log_dir), purge_step, writer))
            return writer, {
                "enabled": True,
                "status": "active",
                "error": None,
                "log_dir": str(log_dir),
                "purge_step": purge_step,
            }

        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "resume_fit_tb"
            with mock.patch.object(pv26_trainer, "_maybe_build_summary_writer", side_effect=_fake_build_summary_writer):
                trainer = PV26Trainer(build_yolo26n_trunk(), PV26Heads(in_channels=(64, 128, 256)), stage="stage_1_frozen_trunk_warmup")
                first = trainer.fit([batch], epochs=1, val_loader=None, run_dir=run_dir, enable_tensorboard=True)

                resumed = PV26Trainer(build_yolo26n_trunk(), PV26Heads(in_channels=(64, 128, 256)), stage="stage_1_frozen_trunk_warmup")
                second = resumed.fit([batch], epochs=2, val_loader=None, run_dir=run_dir, auto_resume=True, enable_tensorboard=True)

        self.assertEqual(first["completed_epochs"], 1)
        self.assertEqual(second["completed_epochs"], 2)
        self.assertEqual(len(writer_calls), 2)
        self.assertIsNone(writer_calls[0][1])
        self.assertEqual(writer_calls[1][1], 2)
        first_train_steps = [step for name, _, step in writer_calls[0][2].scalars if name == "train_step/loss/total"]
        second_train_steps = [step for name, _, step in writer_calls[1][2].scalars if name == "train_step/loss/total"]
        self.assertEqual(first_train_steps, [1])
        self.assertEqual(second_train_steps, [2])
        self.assertEqual(second["tensorboard"]["purge_step"], 2)


if __name__ == "__main__":
    unittest.main()
