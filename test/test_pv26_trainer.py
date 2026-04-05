from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
import torch.nn as nn

from model.engine import _trainer_checkpoint
from common.io import (
    append_jsonl_sorted as common_append_jsonl_sorted,
    now_iso as common_now_iso,
    write_json as common_write_json,
    write_jsonl_sorted as common_write_jsonl_sorted,
)
from common.scalars import flatten_scalar_tree
from common.train_runtime import maybe_build_summary_writer as common_maybe_build_summary_writer
from model.engine import _trainer_io
from model.engine.loss import build_loss_spec
from runtime_support import has_yolo26_runtime


OD_CLASSES = tuple(build_loss_spec()["model_contract"]["od_classes"])
TL_CLASS_ID = OD_CLASSES.index("traffic_light")
LANE_QUERY_COUNT = int(build_loss_spec()["heads"]["lane"]["query_count"])
LANE_ANCHOR_COUNT = int(build_loss_spec()["heads"]["lane"]["target_encoding"]["anchor_rows"])
LANE_VECTOR_DIM = int(build_loss_spec()["heads"]["lane"]["shape"].split(" x ")[-1])
STOP_LINE_QUERY_COUNT = int(build_loss_spec()["heads"]["stop_line"]["query_count"])
STOP_LINE_VECTOR_DIM = int(build_loss_spec()["heads"]["stop_line"]["shape"].split(" x ")[-1])
CROSSWALK_QUERY_COUNT = int(build_loss_spec()["heads"]["crosswalk"]["query_count"])
CROSSWALK_VECTOR_DIM = int(build_loss_spec()["heads"]["crosswalk"]["shape"].split(" x ")[-1])
LANE_X_SLICE = slice(6, 6 + LANE_ANCHOR_COUNT)
LANE_VIS_SLICE = slice(LANE_X_SLICE.stop, LANE_X_SLICE.stop + LANE_ANCHOR_COUNT)


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

    lane = torch.zeros((batch_size, LANE_QUERY_COUNT, LANE_VECTOR_DIM), dtype=torch.float32)
    stop_line = torch.zeros((batch_size, STOP_LINE_QUERY_COUNT, STOP_LINE_VECTOR_DIM), dtype=torch.float32)
    crosswalk = torch.zeros((batch_size, CROSSWALK_QUERY_COUNT, CROSSWALK_VECTOR_DIM), dtype=torch.float32)
    lane_valid = torch.zeros((batch_size, LANE_QUERY_COUNT), dtype=torch.bool)
    stop_line_valid = torch.zeros((batch_size, STOP_LINE_QUERY_COUNT), dtype=torch.bool)
    stop_line_width_valid = torch.zeros((batch_size, STOP_LINE_QUERY_COUNT), dtype=torch.bool)
    crosswalk_valid = torch.zeros((batch_size, CROSSWALK_QUERY_COUNT), dtype=torch.bool)

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
        lane[batch_index, 0, LANE_X_SLICE] = torch.linspace(120.0, 270.0, LANE_ANCHOR_COUNT)
        lane[batch_index, 0, LANE_VIS_SLICE] = 1.0
        lane_valid[batch_index, 0] = True

        stop_line[batch_index, 0, 0] = 1.0
        stop_line[batch_index, 0, 1:5] = torch.tensor([100.0, 500.0, 340.0, 500.0])
        stop_line[batch_index, 0, 5] = 12.0
        stop_line_valid[batch_index, 0] = True
        stop_line_width_valid[batch_index, 0] = True

        crosswalk[batch_index, 0, 0] = 1.0
        crosswalk[batch_index, 0, 1:9] = torch.tensor([200.0, 400.0, 380.0, 400.0, 380.0, 480.0, 200.0, 480.0])
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
            "stop_line_width_valid": stop_line_width_valid,
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
        self.histograms: list[tuple[str, object, int]] = []
        self.graphs: list[tuple[tuple[object, ...], dict[str, object]]] = []
        self.flushed = False
        self.closed = False
        self.layouts: list[dict] = []

    def add_scalar(self, name: str, value: float, global_step: int) -> None:
        self.scalars.append((name, float(value), int(global_step)))

    def add_histogram(self, name: str, value: object, global_step: int) -> None:
        self.histograms.append((name, value, int(global_step)))

    def add_graph(self, *args: object, **kwargs: object) -> None:
        self.graphs.append((args, kwargs))

    def add_custom_scalars(self, layout: dict) -> None:
        self.layouts.append(layout)

    def flush(self) -> None:
        self.flushed = True

    def close(self) -> None:
        self.closed = True


class PV26TrainerTests(unittest.TestCase):
    def test_trainer_io_reuses_common_helper_aliases(self) -> None:
        self.assertIs(_trainer_io._now_iso, common_now_iso)
        self.assertIs(_trainer_io._write_json, common_write_json)
        self.assertIs(_trainer_io._append_jsonl, common_append_jsonl_sorted)
        self.assertIs(_trainer_io._write_jsonl_rows, common_write_jsonl_sorted)
        self.assertIs(_trainer_io._maybe_build_summary_writer, common_maybe_build_summary_writer)

        with mock.patch.object(_trainer_io, "_common_timestamp_token", return_value="20260403_010203"):
            self.assertEqual(_trainer_io._default_run_dir(), Path("runs/pv26_train/pv26_fit_20260403_010203"))

    def test_format_train_live_detail_compacts_to_two_lines(self) -> None:
        from model.engine.trainer_reporting import _format_train_live_detail

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
        from model.engine.trainer_reporting import _format_train_progress_log

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

    def test_trainer_reporting_public_module_exports_progress_helpers(self) -> None:
        from model.engine import trainer_reporting as public_reporting

        exported_names = (
            "_phase_label",
            "_train_progress_desc",
            "_train_progress_postfix",
            "_validation_timing_profile",
            "_validate_progress_desc",
            "_validate_progress_postfix",
            "_format_validate_progress_log",
        )

        for name in exported_names:
            with self.subTest(name=name):
                self.assertIn(name, public_reporting.__all__)
                self.assertTrue(callable(getattr(public_reporting, name)))

    def test_trainer_progress_helpers_trim_windows_and_force_final_logs(self) -> None:
        from model.engine.trainer_progress import should_log_progress, summarize_progress, update_timing_window

        timing_window: list[dict[str, float]] = []
        timing_window = update_timing_window(timing_window, {"iteration_sec": 0.5}, profile_window=2)
        timing_window = update_timing_window(timing_window, {"iteration_sec": 1.5}, profile_window=2)
        timing_window = update_timing_window(timing_window, {"iteration_sec": 2.0}, profile_window=2)

        self.assertEqual(timing_window, [{"iteration_sec": 1.5}, {"iteration_sec": 2.0}])

        with mock.patch("model.engine.trainer_progress.time.perf_counter", return_value=25.0):
            profile_summary, elapsed_sec, eta_sec = summarize_progress(
                started_at=10.0,
                batch_index=3,
                total_batches=5,
                timing_window=timing_window,
                profile_builder=lambda items: {
                    "iteration_sec": {"mean": sum(item["iteration_sec"] for item in items) / len(items)}
                },
            )

        self.assertEqual(profile_summary["iteration_sec"]["mean"], 1.75)
        self.assertEqual(elapsed_sec, 15.0)
        self.assertEqual(eta_sec, 3.5)
        self.assertFalse(should_log_progress(batch_index=1, total_batches=5, log_every_n_steps=2))
        self.assertTrue(should_log_progress(batch_index=2, total_batches=5, log_every_n_steps=2))
        self.assertTrue(should_log_progress(batch_index=5, total_batches=5, log_every_n_steps=10))

    def test_format_epoch_completion_log_is_concise_and_includes_checkpoint_state(self) -> None:
        from model.engine.trainer_reporting import _format_epoch_completion_log

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
        from model.engine import trainer_reporting as trainer_reporting

        summary = {
            "stage": "stage_1_frozen_trunk_warmup",
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
            "timing": {key: 0.1 for key in trainer_reporting.TIMING_KEYS},
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

        payload = trainer_reporting._tensorboard_train_step_payload(summary)
        scalar_names = {name for name, _ in flatten_scalar_tree("train_step", payload)}

        self.assertIn("train_step/loss/total", scalar_names)
        self.assertIn("train_step/loss_weighted/det", scalar_names)
        self.assertIn("train_step/loss_weighted/tl_attr", scalar_names)
        self.assertIn("train_step/loss_weighted/lane", scalar_names)
        self.assertIn("train_step/loss_weighted/stop_line", scalar_names)
        self.assertIn("train_step/loss_weighted/crosswalk", scalar_names)
        self.assertIn("train_step/profile_sec/iteration_sec", scalar_names)
        self.assertNotIn("train_step/lr/trunk", scalar_names)
        self.assertNotIn("train_step/health/gradient_scale", scalar_names)
        self.assertNotIn("train_step/source/det_source_samples", scalar_names)
        self.assertNotIn("train_step/det_supervision/partial_det_ratio", scalar_names)
        self.assertNotIn("train_step/det_components/det_obj_loss", scalar_names)

    def test_tensorboard_epoch_payload_keeps_named_lr_loss_and_val_metrics_only(self) -> None:
        from model.engine import trainer_reporting as trainer_reporting

        epoch_summary = {
            "train": {
                "optimizer_lrs": {"trunk": 1e-4, "heads": 5e-4},
            },
            "val": {
                "losses": {
                    "total": {"mean": 0.8},
                    "det": {"mean": 0.1},
                    "tl_attr": {"mean": 0.05},
                    "lane": {"mean": 0.2},
                    "stop_line": {"mean": 0.15},
                    "crosswalk": {"mean": 0.12},
                    "weighted": {
                        "det": {"mean": 0.1},
                        "tl_attr": {"mean": 0.025},
                        "lane": {"mean": 0.3},
                        "stop_line": {"mean": 0.225},
                        "crosswalk": {"mean": 0.12},
                    },
                },
                "metrics": {
                    "detector": {"map50": 0.6, "map50_95": 0.4},
                    "traffic_light": {"combo_accuracy": 0.75},
                    "lane": {"mean_point_distance": 2.5},
                    "stop_line": {"mean_angle_error": 3.0},
                    "crosswalk": {"mean_polygon_iou": 0.82},
                    "lane_family": {"mean_f1": 0.71},
                },
            },
        }

        payload = trainer_reporting._tensorboard_epoch_payload(epoch_summary)
        scalar_names = {name for name, _ in flatten_scalar_tree("epoch", payload)}

        self.assertIn("epoch/lr/trunk", scalar_names)
        self.assertIn("epoch/lr/heads", scalar_names)
        self.assertIn("epoch/val/loss/total", scalar_names)
        self.assertIn("epoch/val/loss_weighted/det", scalar_names)
        self.assertIn("epoch/val/metrics/detector/map50", scalar_names)
        self.assertIn("epoch/val/metrics/detector/map50_95", scalar_names)
        self.assertIn("epoch/val/metrics/traffic_light/combo_accuracy", scalar_names)
        self.assertIn("epoch/val/metrics/lane/mean_point_distance", scalar_names)
        self.assertIn("epoch/val/metrics/stop_line/mean_angle_error", scalar_names)
        self.assertIn("epoch/val/metrics/crosswalk/mean_polygon_iou", scalar_names)
        self.assertIn("epoch/val/metrics/lane_family/mean_f1", scalar_names)
        self.assertNotIn("epoch/train/loss", scalar_names)
        self.assertNotIn("epoch/train/duration_sec", scalar_names)
        self.assertNotIn("epoch/train/profile_sec/iteration_sec", scalar_names)
        self.assertNotIn("epoch/val/metrics/detector/precision", scalar_names)
        self.assertNotIn("epoch/val/metrics/lane/f1", scalar_names)

    def test_trainer_module_exports_runtime_public_surface_only(self) -> None:
        import model.engine.trainer as pv26_trainer

        for name in (
            "_format_epoch_completion_log",
            "_format_train_live_detail",
            "_format_train_progress_log",
            "_flatten_scalar_tree",
            "_maybe_build_summary_writer",
            "_resolve_summary_path",
            "_tensorboard_epoch_payload",
            "_tensorboard_train_step_payload",
        ):
            with self.subTest(name=name):
                self.assertNotIn(name, pv26_trainer.__all__)
                self.assertFalse(hasattr(pv26_trainer, name))

        self.assertIn("build_pv26_scheduler", pv26_trainer.__all__)

    def test_resolve_summary_path_public_helper_reads_nested_float_values(self) -> None:
        from model.engine.train_summary import resolve_summary_path

        summary = {"train": {"losses": {"total": {"mean": 1.25}}}}

        self.assertEqual(resolve_summary_path(summary, "train.losses.total.mean"), 1.25)

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
        self.assertTrue(any(parameter.requires_grad for parameter in heads.spatial_fusion_stem.parameters()))
        self.assertTrue(any(parameter.requires_grad for parameter in heads.geometry_memory.parameters()))
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
            self.assertEqual(
                checkpoint["checkpoint_metadata"]["architecture_generation"],
                _trainer_checkpoint.ARCHITECTURE_GENERATION,
            )

    def test_load_checkpoint_rejects_incompatible_architecture_generation_for_exact_resume(self) -> None:
        from model.engine.trainer import PV26Trainer
        from model.net import PV26Heads

        trainer = PV26Trainer(_DummyAdapter(), PV26Heads(in_channels=(64, 128, 256)), stage="stage_1_frozen_trunk_warmup")

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = trainer.save_checkpoint(Path(temp_dir) / "resume_break.pt")
            payload = torch.load(checkpoint_path, map_location="cpu")
            payload["checkpoint_metadata"]["architecture_generation"] = "pv26-pre-road-marking"
            torch.save(payload, checkpoint_path)

            reloaded = PV26Trainer(_DummyAdapter(), PV26Heads(in_channels=(64, 128, 256)), stage="stage_1_frozen_trunk_warmup")
            with self.assertRaisesRegex(RuntimeError, "exact resume unsupported"):
                reloaded.load_checkpoint(checkpoint_path, map_location="cpu")

    def test_load_model_weights_uses_shape_aware_partial_load_for_migration(self) -> None:
        from model.engine.trainer import PV26Trainer
        from model.net import PV26Heads

        source = PV26Trainer(_DummyAdapter(), PV26Heads(in_channels=(64, 128, 256)), stage="stage_1_frozen_trunk_warmup")
        target = PV26Trainer(_DummyAdapter(), PV26Heads(in_channels=(64, 128, 256)), stage="stage_1_frozen_trunk_warmup")

        checkpoint_payload = {
            "adapter_state_dict": source.adapter.raw_model.state_dict(),
            "heads_state_dict": source.heads.state_dict(),
        }
        checkpoint_payload["heads_state_dict"]["det_heads.0.block.0.weight"] = torch.full_like(
            checkpoint_payload["heads_state_dict"]["det_heads.0.block.0.weight"],
            7.0,
        )
        checkpoint_payload["heads_state_dict"]["lane_head.predictor.weight"] = torch.ones((1, 1), dtype=torch.float32)

        original_lane_predictor = target.heads.lane_head.predictor.weight.detach().clone()
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "migration.pt"
            torch.save(checkpoint_payload, checkpoint_path)
            loaded = target.load_model_weights(checkpoint_path, map_location="cpu")

        self.assertEqual(loaded["load_policy"], "shape_aware_partial")
        self.assertIn("lane_head.predictor.weight", loaded["heads_load_report"]["skipped_shape_keys"])
        self.assertAlmostEqual(float(target.heads.det_heads[0].block[0].weight.detach().flatten()[0]), 7.0)
        self.assertTrue(torch.equal(target.heads.lane_head.predictor.weight.detach(), original_lane_predictor))

    def test_run_fit_selection_metric_callback_populates_custom_best_metric_path(self) -> None:
        from model.engine import _trainer_fit
        from model.engine.train_summary import resolve_summary_path

        class _StubTrainer:
            def __init__(self) -> None:
                self.stage = "stage_1_frozen_trunk_warmup"
                self.device = torch.device("cpu")
                self.epoch_history: list[dict[str, object]] = []
                self.history: list[dict[str, object]] = []
                self.global_step = 0
                self.skipped_steps = 0
                self.scheduler = None
                self.optimizer = object()
                self.amp_enabled = False
                self.accumulate_steps = 1
                self.grad_clip_norm = 0.0
                self.skip_non_finite_loss = False
                self.oom_guard = False
                self.tensorboard_writer = None
                self.tensorboard_status = {"enabled": False}
                self._tensorboard_train_step = 0
                self._tensorboard_graph_written = False

            def build_evaluator(self) -> object:
                return object()

            def train_epoch(self, *args, **kwargs) -> dict[str, object]:
                return {"batches": 1}

            def validate_epoch(self, *args, **kwargs) -> dict[str, object]:
                return {"losses": {"total": {"mean": 1.0}}}

            def save_checkpoint(self, path: Path, extra_state: dict[str, object] | None = None) -> Path:
                del extra_state
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("checkpoint\n", encoding="utf-8")
                return path

            def save_history_jsonl(self, path: Path) -> Path:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("", encoding="utf-8")
                return path

            def save_epoch_history_jsonl(self, path: Path) -> Path:
                path.parent.mkdir(parents=True, exist_ok=True)
                rows = [json.dumps(item, ensure_ascii=True) for item in self.epoch_history]
                path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
                return path

        def _write_json(path: str | Path, payload: dict[str, object]) -> Path:
            target = Path(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(payload, indent=2, ensure_ascii=True, default=str) + "\n", encoding="utf-8")
            return target

        def _is_better(candidate: float, best: float | None, mode: str) -> bool:
            if best is None:
                return True
            if mode == "min":
                return candidate < best
            return candidate > best

        trainer = _StubTrainer()
        callback_calls: list[int] = []

        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "selection_metric_callback"
            summary = _trainer_fit.run_fit(
                trainer,
                [object()],
                epochs=1,
                val_loader=[object()],
                run_dir=run_dir,
                best_metric="selection_metrics.phase_objective",
                best_mode="max",
                enable_tensorboard=False,
                selection_metric_callback=lambda epoch_summary: (
                    callback_calls.append(int(epoch_summary["epoch"])),
                    epoch_summary.setdefault("selection_metrics", {"phase_objective": 0.75}),
                ),
                default_run_dir_fn=lambda: run_dir,
                now_iso_fn=lambda: "2026-04-06T00:00:00",
                write_json_fn=_write_json,
                json_ready_fn=lambda value: value,
                maybe_build_summary_writer_fn=lambda *args, **kwargs: (None, {"enabled": False}),
                optimizer_group_hparams_fn=lambda optimizer: {},
                resolve_summary_path_fn=resolve_summary_path,
                is_better_fn=_is_better,
                write_tensorboard_scalars_fn=lambda *args, **kwargs: None,
                write_tensorboard_histograms_fn=lambda *args, **kwargs: None,
                tensorboard_epoch_payload_fn=lambda payload: payload,
            )

        self.assertEqual(callback_calls, [1])
        self.assertEqual(summary["best_metric_path"], "selection_metrics.phase_objective")
        self.assertAlmostEqual(summary["best_metric_value"], 0.75)
        self.assertAlmostEqual(trainer.epoch_history[0]["selection_metrics"]["phase_objective"], 0.75)

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
            "lane": torch.zeros((1, LANE_QUERY_COUNT, LANE_VECTOR_DIM), requires_grad=True),
            "stop_line": torch.zeros((1, STOP_LINE_QUERY_COUNT, STOP_LINE_VECTOR_DIM), requires_grad=True),
            "crosswalk": torch.zeros((1, CROSSWALK_QUERY_COUNT, CROSSWALK_VECTOR_DIM), requires_grad=True),
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
        import model.engine._trainer_io as trainer_io
        from model.net import build_yolo26n_trunk

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
            with mock.patch.object(trainer_io, "_maybe_build_summary_writer", side_effect=_fake_build_summary_writer):
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
        self.assertEqual(len(writer_calls[0][2].graphs), 1)
        self.assertEqual(len(writer_calls[1][2].graphs), 1)
        self.assertEqual(second["tensorboard"]["purge_step"], 2)


if __name__ == "__main__":
    unittest.main()
