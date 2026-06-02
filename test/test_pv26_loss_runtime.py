from __future__ import annotations

import unittest
import torch
import torch.nn as nn

from model.engine.loss import build_loss_spec
from runtime_support import has_yolo26_runtime


OD_CLASSES = tuple(build_loss_spec()["model_contract"]["od_classes"])
TL_CLASS_ID = OD_CLASSES.index("traffic_light")
SIGN_CLASS_ID = OD_CLASSES.index("sign")
LANE_QUERY_COUNT = int(build_loss_spec()["heads"]["lane"]["query_count"])
LANE_ANCHOR_COUNT = int(build_loss_spec()["heads"]["lane"]["target_encoding"]["anchor_rows"])
LANE_COLOR_DIM = int(build_loss_spec()["heads"]["lane"]["target_encoding"]["color_logits"])
LANE_TYPE_DIM = int(build_loss_spec()["heads"]["lane"]["target_encoding"]["type_logits"])
LANE_VECTOR_DIM = int(build_loss_spec()["heads"]["lane"]["shape"].split(" x ")[-1])
STOP_LINE_QUERY_COUNT = int(build_loss_spec()["heads"]["stop_line"]["query_count"])
STOP_LINE_VECTOR_DIM = int(build_loss_spec()["heads"]["stop_line"]["shape"].split(" x ")[-1])
CROSSWALK_QUERY_COUNT = int(build_loss_spec()["heads"]["crosswalk"]["query_count"])
CROSSWALK_VECTOR_DIM = int(build_loss_spec()["heads"]["crosswalk"]["shape"].split(" x ")[-1])
LANE_X_SLICE = slice(6, 6 + LANE_ANCHOR_COUNT)
LANE_VIS_SLICE = slice(LANE_X_SLICE.stop, LANE_X_SLICE.stop + LANE_ANCHOR_COUNT)


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
        stop_line[batch_index, 0, 1:] = torch.tensor(
            [100.0, 500.0, 180.0, 500.0, 260.0, 500.0, 340.0, 500.0]
        )
        stop_line_valid[batch_index, 0] = True

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
            "crosswalk_valid": crosswalk_valid,
        },
        "meta": [{"sample_id": f"sample_{index}"} for index in range(batch_size)],
    }


def _with_zero_segfirst_targets(encoded: dict) -> dict:
    from model.data.roadmark_v2_targets import ROADMARK_DENSE_OUTPUT_HW

    batch_size = int(encoded["image"].shape[0])
    h, w = ROADMARK_DENSE_OUTPUT_HW
    encoded = dict(encoded)
    roadmark_v2 = dict(encoded.get("roadmark_v2") or {})
    roadmark_v2.update(
        {
            "stop_line_center_heatmap": torch.zeros((batch_size, 1, h, w), dtype=torch.float32),
            "stop_line_center_offset": torch.zeros((batch_size, 2, h, w), dtype=torch.float32),
            "stop_line_angle": torch.zeros((batch_size, 2, h, w), dtype=torch.float32),
            "stop_line_half_length": torch.zeros((batch_size, 1, h, w), dtype=torch.float32),
            "stop_line_haf_endpoint": torch.zeros((batch_size, 4, h, w), dtype=torch.float32),
            "stop_line_haf_valid": torch.zeros((batch_size, 1, h, w), dtype=torch.float32),
            "stop_line_haf_ignore": torch.zeros((batch_size, 1, h, w), dtype=torch.float32),
            "stop_line_axis_distance": torch.zeros((batch_size, 3, h, w), dtype=torch.float32),
            "stop_line_axis_direction": torch.zeros((batch_size, 2, h, w), dtype=torch.float32),
            "stop_line_axis_valid": torch.zeros((batch_size, 1, h, w), dtype=torch.float32),
            "stop_line_axis_ignore": torch.zeros((batch_size, 1, h, w), dtype=torch.float32),
            "stop_line_mask": torch.zeros((batch_size, 1, h, w), dtype=torch.float32),
            "stop_line_centerline": torch.zeros((batch_size, 1, h, w), dtype=torch.float32),
            "crosswalk_mask": torch.zeros((batch_size, 1, h, w), dtype=torch.float32),
            "crosswalk_boundary": torch.zeros((batch_size, 1, h, w), dtype=torch.float32),
            "crosswalk_center": torch.zeros((batch_size, 1, h, w), dtype=torch.float32),
            "lane_seg_centerline_core": torch.zeros((batch_size, 1, h, w), dtype=torch.float32),
            "lane_seg_centerline_soft": torch.zeros((batch_size, 1, h, w), dtype=torch.float32),
            "lane_seg_support": torch.zeros((batch_size, 1, h, w), dtype=torch.float32),
            "lane_seg_center_offset": torch.zeros((batch_size, 2, h, w), dtype=torch.float32),
            "lane_seg_center_offset_valid": torch.zeros((batch_size, 1, h, w), dtype=torch.float32),
            "lane_seg_anchor_offset": torch.zeros((batch_size, 1, h, w), dtype=torch.float32),
            "lane_seg_anchor_offset_valid": torch.zeros((batch_size, 1, h, w), dtype=torch.float32),
            "lane_seg_row_link_delta": torch.zeros((batch_size, 1, h, w), dtype=torch.float32),
            "lane_seg_row_link_valid": torch.zeros((batch_size, 1, h, w), dtype=torch.float32),
            "lane_seg_residual_risk_core": torch.zeros((batch_size, 1, h, w), dtype=torch.float32),
            "lane_seg_residual_risk_ring_negative": torch.zeros((batch_size, 1, h, w), dtype=torch.float32),
            "lane_seg_tangent_axis": torch.zeros((batch_size, 2, h, w), dtype=torch.float32),
            "lane_seg_instance_id": torch.zeros((batch_size, h, w), dtype=torch.long),
            "lane_seg_instance_ignore": torch.zeros((batch_size, 1, h, w), dtype=torch.float32),
            "lane_seg_color": torch.zeros((batch_size, LANE_COLOR_DIM, h, w), dtype=torch.float32),
            "lane_seg_type": torch.zeros((batch_size, LANE_TYPE_DIM, h, w), dtype=torch.float32),
            "lane_seg_ignore": torch.zeros((batch_size, 1, h, w), dtype=torch.float32),
            "lane_seg_negative": torch.zeros((batch_size, 1, h, w), dtype=torch.float32),
            "lane_seg_stop_line_ignore": torch.zeros((batch_size, 1, h, w), dtype=torch.float32),
            "lane_seg_crosswalk_ignore": torch.zeros((batch_size, 1, h, w), dtype=torch.float32),
            "lane_seg_tangent_count": torch.zeros((batch_size, 1, h, w), dtype=torch.float32),
        }
    )
    encoded["roadmark_v2"] = roadmark_v2
    return encoded


def _zero_predictions(batch_size: int, q_det: int) -> dict[str, torch.Tensor]:
    return {
        "det": torch.zeros((batch_size, q_det, 12), dtype=torch.float32, requires_grad=True),
        "tl_attr": torch.zeros((batch_size, q_det, 4), dtype=torch.float32, requires_grad=True),
        "lane": torch.zeros((batch_size, LANE_QUERY_COUNT, LANE_VECTOR_DIM), dtype=torch.float32, requires_grad=True),
        "stop_line": torch.zeros((batch_size, STOP_LINE_QUERY_COUNT, STOP_LINE_VECTOR_DIM), dtype=torch.float32, requires_grad=True),
        "crosswalk": torch.zeros((batch_size, CROSSWALK_QUERY_COUNT, CROSSWALK_VECTOR_DIM), dtype=torch.float32, requires_grad=True),
        "det_feature_shapes": [(1, q_det)],
        "det_feature_strides": [8],
    }


class _FakeSingleMatchTaskAlignedAssigner(nn.Module):
    def forward(
        self,
        pred_scores: torch.Tensor,
        pred_bboxes: torch.Tensor,
        anchor_points: torch.Tensor,
        gt_labels: torch.Tensor,
        gt_bboxes: torch.Tensor,
        mask_gt: torch.Tensor,
    ):
        del pred_bboxes, anchor_points, mask_gt
        batch_size, query_count, num_classes = pred_scores.shape
        assigned_labels = torch.zeros((batch_size, query_count), device=pred_scores.device, dtype=torch.long)
        assigned_bboxes = torch.zeros((batch_size, query_count, 4), device=pred_scores.device, dtype=torch.float32)
        assigned_scores = torch.zeros(
            (batch_size, query_count, num_classes),
            device=pred_scores.device,
            dtype=torch.float32,
        )
        assigned_fg = torch.zeros((batch_size, query_count), device=pred_scores.device, dtype=torch.bool)
        assigned_gt_idx = torch.full((batch_size, query_count), -1, device=pred_scores.device, dtype=torch.long)

        assigned_fg[:, 0] = True
        assigned_gt_idx[:, 0] = 0
        assigned_labels[:, 0] = gt_labels[:, 0, 0]
        assigned_bboxes[:, 0] = gt_bboxes[:, 0]
        for batch_index in range(batch_size):
            assigned_scores[batch_index, 0, int(gt_labels[batch_index, 0, 0].item())] = 1.0
        return assigned_labels, assigned_bboxes, assigned_scores, assigned_fg, assigned_gt_idx


class PV26LossRuntimeTests(unittest.TestCase):
    def test_zero_segfirst_targets_include_all_aux_lane_keys(self) -> None:
        encoded = _with_zero_segfirst_targets(_make_encoded_batch(batch_size=1, q_det=2))

        self.assertTrue(
            {
                "lane_seg_row_link_delta",
                "lane_seg_row_link_valid",
                "lane_seg_residual_risk_core",
                "lane_seg_residual_risk_ring_negative",
            }.issubset(encoded["roadmark_v2"])
        )

    def test_current_family_denoise_vectors_contribute_to_task_losses(self) -> None:
        from model.engine.loss import PV26MultiTaskLoss

        encoded = _make_encoded_batch(batch_size=1, q_det=4)
        predictions = _zero_predictions(batch_size=1, q_det=4)
        predictions["lane_denoise"] = encoded["lane"].clone().requires_grad_(True)
        predictions["lane_denoise_target"] = encoded["lane"].clone()
        predictions["lane_denoise_valid"] = encoded["mask"]["lane_valid"].clone()
        predictions["stop_line_denoise"] = encoded["stop_line"].clone().requires_grad_(True)
        predictions["stop_line_denoise_target"] = encoded["stop_line"].clone()
        predictions["stop_line_denoise_valid"] = encoded["mask"]["stop_line_valid"].clone()
        predictions["crosswalk_denoise"] = encoded["crosswalk"].clone().requires_grad_(True)
        predictions["crosswalk_denoise_target"] = encoded["crosswalk"].clone()
        predictions["crosswalk_denoise_valid"] = encoded["mask"]["crosswalk_valid"].clone()
        criterion = PV26MultiTaskLoss(
            stage="stage_4_lane_family_finetune",
            loss_weights={"lane": 1.0, "stop_line": 1.0, "crosswalk": 1.0},
        )

        losses = criterion(predictions, encoded)

        self.assertTrue(torch.isfinite(losses["total"]))
        losses["total"].backward()
        self.assertIsNotNone(predictions["lane_denoise"].grad)
        self.assertIsNotNone(predictions["stop_line_denoise"].grad)
        self.assertIsNotNone(predictions["crosswalk_denoise"].grad)

    def test_current_family_dense_seed_logits_contribute_to_task_losses(self) -> None:
        from model.engine.loss import PV26MultiTaskLoss

        encoded = _make_encoded_batch(batch_size=1, q_det=4)
        predictions = _zero_predictions(batch_size=1, q_det=4)
        predictions["lane_dense_seed_logits"] = torch.zeros((1, 1, 32, 40), dtype=torch.float32, requires_grad=True)
        predictions["stop_line_dense_seed_logits"] = torch.zeros((1, 1, 32, 40), dtype=torch.float32, requires_grad=True)
        predictions["crosswalk_dense_seed_logits"] = torch.zeros((1, 1, 32, 40), dtype=torch.float32, requires_grad=True)
        criterion = PV26MultiTaskLoss(
            stage="stage_4_lane_family_finetune",
            loss_weights={"lane": 1.0, "stop_line": 1.0, "crosswalk": 1.0},
        )

        losses = criterion(predictions, encoded)

        self.assertTrue(torch.isfinite(losses["total"]))
        losses["total"].backward()
        self.assertIsNotNone(predictions["lane_dense_seed_logits"].grad)
        self.assertIsNotNone(predictions["stop_line_dense_seed_logits"].grad)
        self.assertIsNotNone(predictions["crosswalk_dense_seed_logits"].grad)

    def test_lane_family_query_metric_objectness_removes_quality_floor(self) -> None:
        from model.engine import loss as loss_module
        from model.engine.loss import PV26MultiTaskLoss

        encoded = _make_encoded_batch(batch_size=1, q_det=2)
        predictions = _zero_predictions(batch_size=1, q_det=2)
        pred_rows = predictions["lane"].detach()
        target_rows = encoded["lane"]
        valid_mask = encoded["mask"]["lane_valid"]
        source_mask = encoded["mask"]["lane_source"]
        floor_criterion = PV26MultiTaskLoss(stage="stage_4_lane_family_finetune")
        metric_criterion = PV26MultiTaskLoss(
            stage="stage_4_lane_family_finetune",
            lane_family_query_objectness_target_mode="metric_quality",
        )

        floor_assignment = floor_criterion._build_query_assignment(
            pred_rows,
            target_rows,
            valid_mask,
            source_mask,
            task_name="lane",
            cost_builder=loss_module._lane_cost_matrix,
            quality_builder=loss_module._lane_match_quality,
        )
        metric_assignment = metric_criterion._build_query_assignment(
            pred_rows,
            target_rows,
            valid_mask,
            source_mask,
            task_name="lane",
            cost_builder=loss_module._lane_cost_matrix,
            quality_builder=loss_module._lane_match_quality,
        )

        floor_value = float(floor_assignment["obj_target"][0].max().item())
        metric_value = float(metric_assignment["obj_target"][0].max().item())
        self.assertGreaterEqual(floor_value, 0.35)
        self.assertLess(metric_value, floor_value)
        self.assertLess(metric_value, 0.35)

    def test_lane_family_query_loss_can_use_teacher_runtime_targets(self) -> None:
        from model.engine.loss import PV26MultiTaskLoss

        encoded = _make_encoded_batch(batch_size=1, q_det=2)
        predictions = _zero_predictions(batch_size=1, q_det=2)
        teacher_lane = torch.zeros_like(encoded["lane"])
        teacher_valid = torch.zeros_like(encoded["mask"]["lane_valid"])
        teacher_lane[0, 0] = encoded["lane"][0, 0]
        teacher_lane[0, 0, 0] = 1.0
        teacher_valid[0, 0] = True
        encoded["teacher_cache"] = {
            "teacher_runtime_lane": teacher_lane,
            "teacher_runtime_lane_valid": teacher_valid,
        }
        criterion = PV26MultiTaskLoss(
            stage="stage_4_lane_family_finetune",
            loss_weights={"lane": 1.0, "stop_line": 0.0, "crosswalk": 0.0},
            lane_family_query_target_source="teacher_runtime",
        )

        losses = criterion(predictions, encoded)

        self.assertTrue(torch.isfinite(losses["total"]))
        self.assertEqual(criterion.export_config()["lane_family_query_target_source"], "teacher_runtime")

    def test_task_loss_ema_normalizer_scales_ready_task_losses(self) -> None:
        from model.engine.loss import PV26MultiTaskLoss

        criterion = PV26MultiTaskLoss(
            stage="stage_4_lane_family_finetune",
            loss_weights={"lane": 1.0, "stop_line": 1.0, "crosswalk": 1.0},
            task_loss_normalize_mode="ema",
            task_loss_ema_decay=0.0,
            task_loss_ema_warmup_steps=0,
            task_loss_scale_min=0.5,
            task_loss_scale_max=2.0,
        )
        task_losses = {
            "lane": torch.tensor(4.0),
            "stop_line": torch.tensor(1.0),
            "crosswalk": torch.tensor(1.0),
        }

        normalized = criterion._normalize_task_losses(task_losses, {"_distill_phase": "train"})

        self.assertAlmostEqual(float(normalized["lane"]), 2.0)
        self.assertAlmostEqual(float(normalized["stop_line"]), 2.0)
        self.assertAlmostEqual(float(normalized["crosswalk"]), 2.0)
        self.assertAlmostEqual(criterion.last_task_loss_normalization["lane"]["scale"], 0.5)
        self.assertAlmostEqual(criterion.last_task_loss_normalization["stop_line"]["scale"], 2.0)
        self.assertAlmostEqual(criterion.last_task_loss_normalization["crosswalk"]["scale"], 2.0)

    def test_task_loss_ema_normalizer_validates_scale_bounds(self) -> None:
        from model.engine.loss import PV26MultiTaskLoss

        with self.assertRaisesRegex(ValueError, "task_loss_scale_max"):
            PV26MultiTaskLoss(
                stage="stage_4_lane_family_finetune",
                task_loss_normalize_mode="ema",
                task_loss_scale_min=2.0,
                task_loss_scale_max=1.0,
            )

    def test_task_uncertainty_weighting_adds_trainable_task_log_vars(self) -> None:
        from model.engine.loss import PV26MultiTaskLoss

        criterion = PV26MultiTaskLoss(
            stage="stage_4_lane_family_finetune",
            loss_weights={"lane": 1.0, "stop_line": 1.0, "crosswalk": 1.0},
            task_uncertainty_weighting_enabled=True,
            task_uncertainty_tasks=("lane", "stop_line"),
            task_uncertainty_init_log_vars={"lane": 0.25, "stop_line": -0.25},
        )
        task_losses = {
            "lane": torch.tensor(2.0, requires_grad=True),
            "stop_line": torch.tensor(1.0, requires_grad=True),
            "crosswalk": torch.tensor(3.0, requires_grad=True),
        }

        weighted = criterion._apply_task_uncertainty_losses(task_losses)
        total = weighted["lane"] + weighted["stop_line"] + weighted["crosswalk"]
        total.backward()

        self.assertIn("lane", criterion.task_uncertainty_log_vars)
        self.assertIn("stop_line", criterion.task_uncertainty_log_vars)
        self.assertNotIn("crosswalk", criterion.task_uncertainty_log_vars)
        self.assertIsNotNone(criterion.task_uncertainty_log_vars["lane"].grad)
        self.assertIsNotNone(criterion.task_uncertainty_log_vars["stop_line"].grad)
        self.assertAlmostEqual(
            criterion.export_config()["task_uncertainty_init_log_vars"]["lane"],
            0.25,
        )

    def test_stage4_disables_detector_and_tl_attr_loss_paths(self) -> None:
        from model.engine.loss import PV26MultiTaskLoss

        encoded = _make_encoded_batch(batch_size=1, q_det=2)
        predictions = _zero_predictions(batch_size=1, q_det=2)
        predictions["det"] = torch.full((1, 2, 12), float("nan"), dtype=torch.float32, requires_grad=True)
        predictions["tl_attr"] = torch.full((1, 2, 4), float("nan"), dtype=torch.float32, requires_grad=True)
        predictions.pop("det_feature_shapes")
        predictions.pop("det_feature_strides")

        criterion = PV26MultiTaskLoss(stage="stage_4_lane_family_finetune")
        losses = criterion(predictions, encoded)

        self.assertEqual(criterion.last_det_assignment_mode, "disabled")
        self.assertEqual(criterion.last_det_positive_count, 0)
        self.assertEqual(float(losses["det"].detach().cpu()), 0.0)
        self.assertEqual(float(losses["tl_attr"].detach().cpu()), 0.0)
        self.assertTrue(torch.isfinite(losses["total"]))
        losses["total"].backward()
        self.assertIsNotNone(predictions["lane"].grad)

    def test_zero_weight_lane_family_aux_losses_skip_non_finite_heads(self) -> None:
        from model.engine.loss import PV26MultiTaskLoss

        encoded = _make_encoded_batch(batch_size=1, q_det=2)
        predictions = _zero_predictions(batch_size=1, q_det=2)
        predictions["stop_line"] = torch.full((1, STOP_LINE_QUERY_COUNT, STOP_LINE_VECTOR_DIM), float("nan"), dtype=torch.float32, requires_grad=True)
        predictions["crosswalk"] = torch.full((1, CROSSWALK_QUERY_COUNT, CROSSWALK_VECTOR_DIM), float("nan"), dtype=torch.float32, requires_grad=True)

        criterion = PV26MultiTaskLoss(
            stage="stage_4_lane_family_finetune",
            loss_weights={"stop_line": 0.0, "crosswalk": 0.0},
        )
        losses = criterion(predictions, encoded)

        self.assertEqual(criterion.last_lane_assignment_modes["stop_line"], "disabled")
        self.assertEqual(criterion.last_lane_assignment_modes["crosswalk"], "disabled")
        self.assertTrue(torch.isfinite(losses["stop_line"]))
        self.assertTrue(torch.isfinite(losses["crosswalk"]))
        self.assertTrue(torch.isfinite(losses["total"]))
        losses["total"].backward()
        self.assertIsNotNone(predictions["lane"].grad)

    def test_stopline_distill_tolerates_missing_segfirst_lane_row_logits(self) -> None:
        from model.engine.loss import PV26MultiTaskLoss
        from model.data.roadmark_v2_targets import ROADMARK_DENSE_OUTPUT_HW

        batch_size = 1
        h, w = ROADMARK_DENSE_OUTPUT_HW
        encoded = _with_zero_segfirst_targets(_make_encoded_batch(batch_size=batch_size, q_det=2))
        predictions = _zero_predictions(batch_size=batch_size, q_det=2)
        predictions.update(
            {
                "lane_seg_centerline_logits": torch.zeros((batch_size, 1, h, w), requires_grad=True),
                "lane_seg_support_logits": torch.zeros((batch_size, 1, h, w), requires_grad=True),
                "lane_seg_tangent_axis": torch.zeros((batch_size, 2, h, w), requires_grad=True),
                "lane_seg_color_logits": torch.zeros((batch_size, LANE_COLOR_DIM, h, w), requires_grad=True),
                "lane_seg_type_logits": torch.zeros((batch_size, LANE_TYPE_DIM, h, w), requires_grad=True),
                "stop_line_mask_logits": torch.zeros((batch_size, 1, h, w), requires_grad=True),
                "stop_line_center_logits": torch.zeros((batch_size, 1, h, w), requires_grad=True),
                "stop_line_center_offset": torch.zeros((batch_size, 2, h, w), requires_grad=True),
                "stop_line_angle": torch.zeros((batch_size, 2, h, w), requires_grad=True),
                "stop_line_half_length": torch.zeros((batch_size, 1, h, w), requires_grad=True),
                "stop_line_feature": torch.zeros((batch_size, 4, h, w), requires_grad=True),
                "crosswalk_mask_logits": torch.zeros((batch_size, 1, h, w), requires_grad=True),
                "crosswalk_boundary_logits": torch.zeros((batch_size, 1, h, w), requires_grad=True),
                "crosswalk_center_logits": torch.zeros((batch_size, 1, h, w), requires_grad=True),
                "crosswalk_feature": torch.zeros((batch_size, 4, h, w), requires_grad=True),
            }
        )
        encoded["teacher_cache"] = {
            key: value.detach().clone()
            for key, value in predictions.items()
            if key.startswith("stop_line_")
        }

        criterion = PV26MultiTaskLoss(
            stage="stage_4_lane_family_finetune",
            task_mode="roadmark_joint",
            distill_enabled=True,
            distill_loss_weights={"lane": 0.0, "stop_line": 0.1, "crosswalk": 0.0},
        )
        losses = criterion(predictions, encoded)

        self.assertTrue(torch.isfinite(losses["total"]))
        self.assertEqual(float(criterion.last_distill_breakdown["lane"]["loss"]), 0.0)
        self.assertIsNotNone(criterion.last_distill_breakdown["stop_line"]["loss"])
        losses["total"].backward()
        self.assertIsNotNone(predictions["stop_line_mask_logits"].grad)

    def test_stopline_haf_aux_loss_backprops_when_enabled(self) -> None:
        from model.engine.loss import PV26MultiTaskLoss
        from model.data.roadmark_v2_targets import ROADMARK_DENSE_OUTPUT_HW

        batch_size = 1
        h, w = ROADMARK_DENSE_OUTPUT_HW
        encoded = _with_zero_segfirst_targets(_make_encoded_batch(batch_size=batch_size, q_det=2))
        encoded["roadmark_v2"]["stop_line_haf_valid"][0, 0, 20, 30] = 1.0
        encoded["roadmark_v2"]["stop_line_haf_endpoint"][0, :, 20, 30] = torch.tensor(
            [-5.0, 0.0, 5.0, 0.0],
            dtype=torch.float32,
        )
        predictions = _zero_predictions(batch_size=batch_size, q_det=2)
        predictions.update(
            {
                "stop_line_mask_logits": torch.zeros((batch_size, 1, h, w), requires_grad=True),
                "stop_line_center_logits": torch.zeros((batch_size, 1, h, w), requires_grad=True),
                "stop_line_center_offset": torch.zeros((batch_size, 2, h, w), requires_grad=True),
                "stop_line_angle": torch.zeros((batch_size, 2, h, w), requires_grad=True),
                "stop_line_half_length": torch.zeros((batch_size, 1, h, w), requires_grad=True),
                "stop_line_haf_endpoint": torch.zeros((batch_size, 4, h, w), requires_grad=True),
                "stop_line_haf_valid_logits": torch.zeros((batch_size, 1, h, w), requires_grad=True),
            }
        )

        criterion = PV26MultiTaskLoss(
            stage="stage_4_lane_family_finetune",
            task_mode="roadmark_joint",
            loss_weights={"lane": 0.0, "crosswalk": 0.0},
            stopline_haf_aux_weight=1.0,
        )
        losses = criterion(predictions, encoded)

        self.assertTrue(torch.isfinite(losses["total"]))
        losses["total"].backward()
        self.assertIsNotNone(predictions["stop_line_haf_endpoint"].grad)
        self.assertIsNotNone(predictions["stop_line_haf_valid_logits"].grad)

    def test_stopline_axis_distance_aux_loss_backprops_when_enabled(self) -> None:
        from model.engine.loss import PV26MultiTaskLoss
        from model.data.roadmark_v2_targets import ROADMARK_DENSE_OUTPUT_HW

        batch_size = 1
        h, w = ROADMARK_DENSE_OUTPUT_HW
        encoded = _with_zero_segfirst_targets(_make_encoded_batch(batch_size=batch_size, q_det=2))
        encoded["roadmark_v2"]["stop_line_axis_valid"][0, 0, 20, 30] = 1.0
        encoded["roadmark_v2"]["stop_line_axis_distance"][0, :, 20, 30] = torch.tensor(
            [-0.05, 0.05, 0.0],
            dtype=torch.float32,
        )
        encoded["roadmark_v2"]["stop_line_axis_direction"][0, :, 20, 30] = torch.tensor(
            [1.0, 0.0],
            dtype=torch.float32,
        )
        predictions = _zero_predictions(batch_size=batch_size, q_det=2)
        predictions.update(
            {
                "stop_line_mask_logits": torch.zeros((batch_size, 1, h, w), requires_grad=True),
                "stop_line_center_logits": torch.zeros((batch_size, 1, h, w), requires_grad=True),
                "stop_line_center_offset": torch.zeros((batch_size, 2, h, w), requires_grad=True),
                "stop_line_angle": torch.zeros((batch_size, 2, h, w), requires_grad=True),
                "stop_line_half_length": torch.zeros((batch_size, 1, h, w), requires_grad=True),
                "stop_line_axis_distance": torch.zeros((batch_size, 3, h, w), requires_grad=True),
                "stop_line_axis_direction": torch.zeros((batch_size, 2, h, w), requires_grad=True),
                "stop_line_axis_valid_logits": torch.zeros((batch_size, 1, h, w), requires_grad=True),
            }
        )

        criterion = PV26MultiTaskLoss(
            stage="stage_4_lane_family_finetune",
            task_mode="roadmark_joint",
            loss_weights={"lane": 0.0, "crosswalk": 0.0},
            stopline_axis_distance_aux_weight=1.0,
        )
        losses = criterion(predictions, encoded)

        self.assertTrue(torch.isfinite(losses["total"]))
        losses["total"].backward()
        self.assertIsNotNone(predictions["stop_line_axis_distance"].grad)
        self.assertIsNotNone(predictions["stop_line_axis_direction"].grad)
        self.assertIsNotNone(predictions["stop_line_axis_valid_logits"].grad)

    def test_stopline_positive_only_mode_ignores_empty_source_samples(self) -> None:
        from model.engine.loss import PV26MultiTaskLoss
        from model.data.roadmark_v2_targets import ROADMARK_DENSE_OUTPUT_HW

        batch_size = 2
        h, w = ROADMARK_DENSE_OUTPUT_HW
        encoded = _with_zero_segfirst_targets(_make_encoded_batch(batch_size=batch_size, q_det=2))
        encoded["mask"]["stop_line_valid"][1].zero_()
        mask_logits = torch.zeros((batch_size, 1, h, w), dtype=torch.float32)
        mask_logits[1].fill_(6.0)

        def _loss(mode: str) -> torch.Tensor:
            predictions = _zero_predictions(batch_size=batch_size, q_det=2)
            predictions["stop_line_mask_logits"] = mask_logits.clone().requires_grad_(True)
            criterion = PV26MultiTaskLoss(
                stage="stage_4_lane_family_finetune",
                task_mode="roadmark_joint",
                loss_weights={"lane": 0.0, "crosswalk": 0.0},
                stopline_empty_sample_mode=mode,
            )
            return criterion(predictions, encoded)["total"]

        full_loss = _loss("full")
        positive_only_loss = _loss("positive_only")

        self.assertGreater(float(full_loss.detach().cpu()), float(positive_only_loss.detach().cpu()) + 1.0)

    def test_roadmark_joint_accepts_vector_only_lane_family_outputs(self) -> None:
        from model.engine.loss import PV26MultiTaskLoss

        batch_size = 2
        predictions = {
            "lane": torch.zeros((batch_size, LANE_QUERY_COUNT, LANE_VECTOR_DIM), dtype=torch.float32, requires_grad=True),
            "stop_line": torch.zeros(
                (batch_size, STOP_LINE_QUERY_COUNT, STOP_LINE_VECTOR_DIM), dtype=torch.float32, requires_grad=True
            ),
            "crosswalk": torch.zeros(
                (batch_size, CROSSWALK_QUERY_COUNT, CROSSWALK_VECTOR_DIM), dtype=torch.float32, requires_grad=True
            ),
        }
        encoded = _make_encoded_batch(batch_size=batch_size, q_det=2)
        criterion = PV26MultiTaskLoss(
            stage="stage_4_lane_family_finetune",
            task_mode="roadmark_joint",
            loss_weights={"det": 0.0, "tl_attr": 0.0},
        )

        loss = criterion(predictions, encoded)["total"]
        loss.backward()

        self.assertTrue(torch.isfinite(loss.detach()).item())
        self.assertIsNotNone(predictions["lane"].grad)
        self.assertIsNotNone(predictions["stop_line"].grad)
        self.assertIsNotNone(predictions["crosswalk"].grad)

    def test_stopline_segment_set_aux_loss_backprops_when_enabled(self) -> None:
        from model.engine.loss import PV26MultiTaskLoss
        from model.data.roadmark_v2_targets import ROADMARK_DENSE_OUTPUT_HW

        batch_size = 1
        h, w = ROADMARK_DENSE_OUTPUT_HW
        encoded = _with_zero_segfirst_targets(_make_encoded_batch(batch_size=batch_size, q_det=2))
        predictions = _zero_predictions(batch_size=batch_size, q_det=2)
        predictions.update(
            {
                "stop_line_mask_logits": torch.zeros((batch_size, 1, h, w), requires_grad=True),
                "stop_line_center_logits": torch.zeros((batch_size, 1, h, w), requires_grad=True),
                "stop_line_center_offset": torch.zeros((batch_size, 2, h, w), requires_grad=True),
                "stop_line_angle": torch.zeros((batch_size, 2, h, w), requires_grad=True),
                "stop_line_half_length": torch.zeros((batch_size, 1, h, w), requires_grad=True),
                "stop_line_segment_seed_logits": torch.zeros((batch_size, 1, h, w), requires_grad=True),
                "stop_line_segment_logits": torch.zeros((batch_size, STOP_LINE_QUERY_COUNT), requires_grad=True),
                "stop_line_segment_verifier_logits": torch.zeros(
                    (batch_size, STOP_LINE_QUERY_COUNT),
                    requires_grad=True,
                ),
                "stop_line_segment_points": torch.full(
                    (batch_size, STOP_LINE_QUERY_COUNT, 2, 2),
                    0.5,
                    requires_grad=True,
                ),
                "stop_line_context_segment_seed_logits": torch.zeros((batch_size, 1, h, w), requires_grad=True),
                "stop_line_context_segment_logits": torch.zeros(
                    (batch_size, STOP_LINE_QUERY_COUNT),
                    requires_grad=True,
                ),
                "stop_line_context_segment_verifier_logits": torch.zeros(
                    (batch_size, STOP_LINE_QUERY_COUNT),
                    requires_grad=True,
                ),
                "stop_line_context_segment_points": torch.full(
                    (batch_size, STOP_LINE_QUERY_COUNT, 2, 2),
                    0.5,
                    requires_grad=True,
                ),
                "stop_line_patch_segment_seed_logits": torch.zeros((batch_size, 1, h, w), requires_grad=True),
                "stop_line_patch_segment_logits": torch.zeros((batch_size, STOP_LINE_QUERY_COUNT), requires_grad=True),
                "stop_line_patch_segment_verifier_logits": torch.zeros(
                    (batch_size, STOP_LINE_QUERY_COUNT),
                    requires_grad=True,
                ),
                "stop_line_patch_segment_points": torch.full(
                    (batch_size, STOP_LINE_QUERY_COUNT, 2, 2),
                    0.5,
                    requires_grad=True,
                ),
                "stop_line_segment_denoise_logits": torch.zeros(
                    (batch_size, STOP_LINE_QUERY_COUNT),
                    requires_grad=True,
                ),
                "stop_line_segment_denoise_points": torch.full(
                    (batch_size, STOP_LINE_QUERY_COUNT, 2, 2),
                    0.5,
                    requires_grad=True,
                ),
                "stop_line_segment_denoise_targets": torch.zeros(
                    (batch_size, STOP_LINE_QUERY_COUNT, 2, 2),
                ),
                "stop_line_segment_denoise_valid": torch.zeros(
                    (batch_size, STOP_LINE_QUERY_COUNT),
                    dtype=torch.bool,
                ),
            }
        )
        predictions["stop_line_segment_denoise_targets"][0, 0] = torch.tensor(
            [[100.0 / 800.0, 500.0 / 608.0], [340.0 / 800.0, 500.0 / 608.0]]
        )
        predictions["stop_line_segment_denoise_valid"][0, 0] = True

        criterion = PV26MultiTaskLoss(
            stage="stage_4_lane_family_finetune",
            task_mode="roadmark_joint",
            loss_weights={"lane": 0.0, "crosswalk": 0.0},
            stopline_segment_set_aux_weight=1.0,
            stopline_segment_verifier_aux_weight=1.0,
            stopline_context_segment_set_aux_weight=1.0,
            stopline_context_segment_verifier_aux_weight=1.0,
            stopline_patch_segment_set_aux_weight=1.0,
            stopline_patch_segment_verifier_aux_weight=1.0,
            stopline_segment_denoise_aux_weight=1.0,
            stopline_segment_verifier_target_mode="metric_quality",
            stopline_segment_objectness_target_mode="metric_quality",
            stopline_segment_verifier_quality_tau_px=18.0,
        )
        losses = criterion(predictions, encoded)

        self.assertTrue(torch.isfinite(losses["total"]))
        self.assertEqual(criterion.export_config()["stopline_segment_verifier_target_mode"], "metric_quality")
        self.assertEqual(criterion.export_config()["stopline_segment_objectness_target_mode"], "metric_quality")
        self.assertAlmostEqual(criterion.export_config()["stopline_segment_verifier_quality_tau_px"], 18.0)
        self.assertAlmostEqual(criterion.export_config()["stopline_context_segment_set_aux_weight"], 1.0)
        self.assertAlmostEqual(criterion.export_config()["stopline_patch_segment_set_aux_weight"], 1.0)
        losses["total"].backward()
        self.assertIsNotNone(predictions["stop_line_segment_logits"].grad)
        self.assertIsNotNone(predictions["stop_line_segment_verifier_logits"].grad)
        self.assertIsNotNone(predictions["stop_line_segment_points"].grad)
        self.assertIsNotNone(predictions["stop_line_segment_seed_logits"].grad)
        self.assertIsNotNone(predictions["stop_line_context_segment_logits"].grad)
        self.assertIsNotNone(predictions["stop_line_context_segment_verifier_logits"].grad)
        self.assertIsNotNone(predictions["stop_line_context_segment_points"].grad)
        self.assertIsNotNone(predictions["stop_line_context_segment_seed_logits"].grad)
        self.assertIsNotNone(predictions["stop_line_patch_segment_logits"].grad)
        self.assertIsNotNone(predictions["stop_line_patch_segment_verifier_logits"].grad)
        self.assertIsNotNone(predictions["stop_line_patch_segment_points"].grad)
        self.assertIsNotNone(predictions["stop_line_patch_segment_seed_logits"].grad)
        self.assertIsNotNone(predictions["stop_line_segment_denoise_logits"].grad)
        self.assertIsNotNone(predictions["stop_line_segment_denoise_points"].grad)

    def test_lane_segfirst_task_conflict_negative_penalizes_ignored_crosswalk_pixels(self) -> None:
        from model.engine.loss import PV26MultiTaskLoss
        from model.data.roadmark_v2_targets import ROADMARK_DENSE_OUTPUT_HW

        batch_size = 1
        h, w = ROADMARK_DENSE_OUTPUT_HW
        encoded = _with_zero_segfirst_targets(_make_encoded_batch(batch_size=batch_size, q_det=2))
        encoded["roadmark_v2"]["lane_seg_ignore"][:, :, 0, 0] = 1.0
        encoded["roadmark_v2"]["lane_seg_crosswalk_ignore"][:, :, 0, 0] = 1.0
        predictions = _zero_predictions(batch_size=batch_size, q_det=2)
        predictions.update(
            {
                "lane_seg_centerline_logits": torch.zeros((batch_size, 1, h, w), requires_grad=True),
                "lane_seg_support_logits": torch.zeros((batch_size, 1, h, w), requires_grad=True),
                "lane_seg_tangent_axis": torch.zeros((batch_size, 2, h, w), requires_grad=True),
                "lane_seg_color_logits": torch.zeros((batch_size, LANE_COLOR_DIM, h, w), requires_grad=True),
                "lane_seg_type_logits": torch.zeros((batch_size, LANE_TYPE_DIM, h, w), requires_grad=True),
            }
        )

        disabled = PV26MultiTaskLoss(
            stage="stage_4_lane_family_finetune",
            loss_weights={"stop_line": 0.0, "crosswalk": 0.0},
        )
        disabled_loss = disabled(predictions, encoded)["lane"].detach()
        enabled = PV26MultiTaskLoss(
            stage="stage_4_lane_family_finetune",
            loss_weights={"stop_line": 0.0, "crosswalk": 0.0},
            lane_segfirst_task_conflict_negative_mode="crosswalk",
            lane_segfirst_task_conflict_negative_weight=1.0,
            lane_segfirst_task_conflict_negative_margin=0.25,
        )
        enabled_losses = enabled(predictions, encoded)

        self.assertGreater(
            float(enabled.last_lane_loss_breakdown["seg_task_conflict_negative"]),
            0.0,
        )
        self.assertGreater(float(enabled_losses["lane"].detach()), float(disabled_loss))
        enabled_losses["total"].backward()
        self.assertIsNotNone(predictions["lane_seg_centerline_logits"].grad)

    def test_stopline_task_conflict_negative_penalizes_lane_crosswalk_pixels(self) -> None:
        from model.engine.loss import PV26MultiTaskLoss
        from model.data.roadmark_v2_targets import ROADMARK_DENSE_OUTPUT_HW

        batch_size = 1
        h, w = ROADMARK_DENSE_OUTPUT_HW
        encoded = _with_zero_segfirst_targets(_make_encoded_batch(batch_size=batch_size, q_det=2))
        encoded["roadmark_v2"]["crosswalk_mask"][:, :, 2, 3] = 1.0
        encoded["roadmark_v2"]["lane_seg_support"][:, :, 4, 5] = 1.0
        predictions = _zero_predictions(batch_size=batch_size, q_det=2)
        predictions.update(
            {
                "stop_line_mask_logits": torch.zeros((batch_size, 1, h, w), requires_grad=True),
                "stop_line_center_logits": torch.zeros((batch_size, 1, h, w), requires_grad=True),
                "stop_line_row_logits": torch.zeros((batch_size, 1, h, w), requires_grad=True),
                "stop_line_x_logits": torch.zeros((batch_size, 1, h, w), requires_grad=True),
                "stop_line_selector_map_logits": torch.zeros((batch_size, 1, h, w), requires_grad=True),
                "stop_line_center_offset": torch.zeros((batch_size, 2, h, w), requires_grad=True),
                "stop_line_angle": torch.zeros((batch_size, 2, h, w), requires_grad=True),
                "stop_line_half_length": torch.zeros((batch_size, 1, h, w), requires_grad=True),
            }
        )

        disabled = PV26MultiTaskLoss(
            stage="stage_4_lane_family_finetune",
            task_mode="roadmark_joint",
            loss_weights={"lane": 0.0, "crosswalk": 0.0},
        )
        disabled_loss = disabled(predictions, encoded)["total"].detach()
        enabled = PV26MultiTaskLoss(
            stage="stage_4_lane_family_finetune",
            task_mode="roadmark_joint",
            loss_weights={"lane": 0.0, "crosswalk": 0.0},
            stopline_task_conflict_negative_mode="lane_crosswalk",
            stopline_task_conflict_negative_weight=1.0,
            stopline_task_conflict_negative_margin=0.25,
        )
        enabled_losses = enabled(predictions, encoded)

        self.assertGreater(float(enabled_losses["total"].detach()), float(disabled_loss))
        self.assertEqual(enabled.export_config()["stopline_task_conflict_negative_mode"], "lane_crosswalk")
        enabled_losses["total"].backward()
        self.assertIsNotNone(predictions["stop_line_mask_logits"].grad)
        self.assertIsNotNone(predictions["stop_line_center_logits"].grad)

    def test_lane_conditional_row_aux_loss_backprops_when_enabled(self) -> None:
        from model.engine.loss import PV26MultiTaskLoss
        from model.data.roadmark_v2_targets import ROADMARK_DENSE_OUTPUT_HW

        batch_size = 1
        h, w = ROADMARK_DENSE_OUTPUT_HW
        encoded = _with_zero_segfirst_targets(_make_encoded_batch(batch_size=batch_size, q_det=2))
        encoded["roadmark_v2"]["lane_seg_centerline_core"][:, :, 40:44, 50:54] = 1.0
        predictions = _zero_predictions(batch_size=batch_size, q_det=2)
        predictions.update(
            {
                "lane_seg_centerline_logits": torch.zeros((batch_size, 1, h, w), requires_grad=True),
                "lane_seg_support_logits": torch.zeros((batch_size, 1, h, w), requires_grad=True),
                "lane_seg_tangent_axis": torch.zeros((batch_size, 2, h, w), requires_grad=True),
                "lane_seg_color_logits": torch.zeros((batch_size, LANE_COLOR_DIM, h, w), requires_grad=True),
                "lane_seg_type_logits": torch.zeros((batch_size, LANE_TYPE_DIM, h, w), requires_grad=True),
                "lane_conditional_seed_logits": torch.zeros((batch_size, 1, h, w), requires_grad=True),
                "lane_conditional_rows": torch.zeros(
                    (batch_size, LANE_QUERY_COUNT, LANE_VECTOR_DIM),
                    requires_grad=True,
                ),
            }
        )

        criterion = PV26MultiTaskLoss(
            stage="stage_4_lane_family_finetune",
            loss_weights={"stop_line": 0.0, "crosswalk": 0.0},
            lane_conditional_row_aux_weight=1.0,
            lane_conditional_seed_target_mode="bottom_anchor",
            lane_conditional_objectness_target_mode="metric_quality",
            lane_conditional_row_x_weight=0.5,
        )
        losses = criterion(predictions, encoded)

        self.assertTrue(torch.isfinite(losses["total"]))
        self.assertEqual(criterion.export_config()["lane_conditional_seed_target_mode"], "bottom_anchor")
        self.assertEqual(criterion.export_config()["lane_conditional_objectness_target_mode"], "metric_quality")
        self.assertAlmostEqual(criterion.export_config()["lane_conditional_row_x_weight"], 0.5)
        losses["total"].backward()
        self.assertIsNotNone(predictions["lane_conditional_rows"].grad)
        self.assertIsNotNone(predictions["lane_conditional_seed_logits"].grad)

    def test_lane_conditional_row_aux_loss_can_use_teacher_runtime_targets(self) -> None:
        from model.engine.loss import PV26MultiTaskLoss
        from model.data.roadmark_v2_targets import ROADMARK_DENSE_OUTPUT_HW

        batch_size = 1
        h, w = ROADMARK_DENSE_OUTPUT_HW
        encoded = _with_zero_segfirst_targets(_make_encoded_batch(batch_size=batch_size, q_det=2))
        teacher_lane = encoded["lane"].clone()
        teacher_lane[:, :, LANE_X_SLICE] = teacher_lane[:, :, LANE_X_SLICE] + 96.0
        encoded["teacher_cache"] = {
            "teacher_runtime_lane": teacher_lane,
            "teacher_runtime_lane_valid": encoded["mask"]["lane_valid"].clone(),
        }
        conditional_rows = torch.zeros((batch_size, LANE_QUERY_COUNT, LANE_VECTOR_DIM), dtype=torch.float32)
        conditional_rows[0, 0] = teacher_lane[0, 0]
        conditional_rows[0, 0, 0] = 8.0
        predictions = _zero_predictions(batch_size=batch_size, q_det=2)
        predictions.update(
            {
                "lane_seg_centerline_logits": torch.zeros((batch_size, 1, h, w), requires_grad=True),
                "lane_seg_support_logits": torch.zeros((batch_size, 1, h, w), requires_grad=True),
                "lane_seg_tangent_axis": torch.zeros((batch_size, 2, h, w), requires_grad=True),
                "lane_seg_color_logits": torch.zeros((batch_size, LANE_COLOR_DIM, h, w), requires_grad=True),
                "lane_seg_type_logits": torch.zeros((batch_size, LANE_TYPE_DIM, h, w), requires_grad=True),
                "lane_conditional_seed_logits": torch.zeros((batch_size, 1, h, w), requires_grad=True),
                "lane_conditional_rows": conditional_rows.requires_grad_(),
            }
        )

        common_kwargs = {
            "stage": "stage_4_lane_family_finetune",
            "loss_weights": {"stop_line": 0.0, "crosswalk": 0.0},
            "lane_conditional_row_aux_weight": 1.0,
            "lane_conditional_seed_aux_weight": 0.0,
            "lane_conditional_seed_target_mode": "bottom_anchor",
            "lane_conditional_objectness_target_mode": "metric_quality",
            "lane_conditional_row_x_weight": 1.0,
            "lane_segfirst_loss_weights": {
                "centerline_bce": 0.0,
                "centerline_dice": 0.0,
                "support_bce": 0.0,
                "tangent": 0.0,
                "color": 0.0,
                "type": 0.0,
            },
        }
        encoded_criterion = PV26MultiTaskLoss(**common_kwargs)
        teacher_criterion = PV26MultiTaskLoss(
            **common_kwargs,
            distill_enabled=True,
            lane_family_query_target_source="teacher_runtime",
        )

        encoded_loss = encoded_criterion(predictions, encoded)["total"].detach()
        teacher_losses = teacher_criterion(predictions, encoded)

        self.assertLess(float(teacher_losses["total"].detach()), float(encoded_loss))
        self.assertEqual(teacher_criterion.export_config()["lane_family_query_target_source"], "teacher_runtime")
        teacher_losses["total"].backward()
        self.assertIsNotNone(predictions["lane_conditional_rows"].grad)

    def test_lane_conditional_denoise_aux_loss_backprops_when_enabled(self) -> None:
        from model.engine.loss import PV26MultiTaskLoss
        from model.data.roadmark_v2_targets import ROADMARK_DENSE_OUTPUT_HW

        batch_size = 1
        h, w = ROADMARK_DENSE_OUTPUT_HW
        encoded = _with_zero_segfirst_targets(_make_encoded_batch(batch_size=batch_size, q_det=2))
        predictions = _zero_predictions(batch_size=batch_size, q_det=2)
        denoise_targets = encoded["lane"].clone()
        denoise_valid = encoded["mask"]["lane_valid"].clone()
        predictions.update(
            {
                "lane_seg_centerline_logits": torch.zeros((batch_size, 1, h, w), requires_grad=True),
                "lane_seg_support_logits": torch.zeros((batch_size, 1, h, w), requires_grad=True),
                "lane_seg_tangent_axis": torch.zeros((batch_size, 2, h, w), requires_grad=True),
                "lane_seg_color_logits": torch.zeros((batch_size, LANE_COLOR_DIM, h, w), requires_grad=True),
                "lane_seg_type_logits": torch.zeros((batch_size, LANE_TYPE_DIM, h, w), requires_grad=True),
                "lane_conditional_seed_logits": torch.zeros((batch_size, 1, h, w), requires_grad=True),
                "lane_conditional_rows": torch.zeros(
                    (batch_size, LANE_QUERY_COUNT, LANE_VECTOR_DIM),
                    requires_grad=True,
                ),
                "lane_conditional_denoise_rows": torch.zeros(
                    (batch_size, LANE_QUERY_COUNT, LANE_VECTOR_DIM),
                    requires_grad=True,
                ),
                "lane_conditional_denoise_targets": denoise_targets,
                "lane_conditional_denoise_valid": denoise_valid,
            }
        )

        criterion = PV26MultiTaskLoss(
            stage="stage_4_lane_family_finetune",
            loss_weights={"stop_line": 0.0, "crosswalk": 0.0},
            lane_conditional_denoise_aux_weight=1.0,
            lane_conditional_row_x_weight=0.5,
        )
        losses = criterion(predictions, encoded)

        self.assertTrue(torch.isfinite(losses["total"]))
        self.assertAlmostEqual(criterion.export_config()["lane_conditional_denoise_aux_weight"], 1.0)
        losses["total"].backward()
        self.assertIsNotNone(predictions["lane_conditional_denoise_rows"].grad)
        self.assertIsNone(predictions["lane_conditional_rows"].grad)

    def test_lane_center_offset_aux_loss_backprops_when_enabled(self) -> None:
        from model.engine.loss import PV26MultiTaskLoss
        from model.data.roadmark_v2_targets import ROADMARK_DENSE_OUTPUT_HW

        batch_size = 1
        h, w = ROADMARK_DENSE_OUTPUT_HW
        encoded = _with_zero_segfirst_targets(_make_encoded_batch(batch_size=batch_size, q_det=2))
        encoded["roadmark_v2"]["lane_seg_center_offset_valid"][:, :, 20, 20] = 1.0
        encoded["roadmark_v2"]["lane_seg_center_offset"][:, 0, 20, 20] = 2.0
        predictions = _zero_predictions(batch_size=batch_size, q_det=2)
        predictions.update(
            {
                "lane_seg_centerline_logits": torch.zeros((batch_size, 1, h, w), requires_grad=True),
                "lane_seg_support_logits": torch.zeros((batch_size, 1, h, w), requires_grad=True),
                "lane_seg_center_offset": torch.zeros((batch_size, 2, h, w), requires_grad=True),
                "lane_seg_tangent_axis": torch.zeros((batch_size, 2, h, w), requires_grad=True),
                "lane_seg_color_logits": torch.zeros((batch_size, LANE_COLOR_DIM, h, w), requires_grad=True),
                "lane_seg_type_logits": torch.zeros((batch_size, LANE_TYPE_DIM, h, w), requires_grad=True),
                "stop_line_mask_logits": torch.zeros((batch_size, 1, h, w), requires_grad=True),
                "stop_line_center_logits": torch.zeros((batch_size, 1, h, w), requires_grad=True),
                "stop_line_center_offset": torch.zeros((batch_size, 2, h, w), requires_grad=True),
                "stop_line_angle": torch.zeros((batch_size, 2, h, w), requires_grad=True),
                "stop_line_half_length": torch.zeros((batch_size, 1, h, w), requires_grad=True),
                "stop_line_feature": torch.zeros((batch_size, 4, h, w), requires_grad=True),
                "crosswalk_mask_logits": torch.zeros((batch_size, 1, h, w), requires_grad=True),
                "crosswalk_boundary_logits": torch.zeros((batch_size, 1, h, w), requires_grad=True),
                "crosswalk_center_logits": torch.zeros((batch_size, 1, h, w), requires_grad=True),
                "crosswalk_feature": torch.zeros((batch_size, 4, h, w), requires_grad=True),
            }
        )
        criterion = PV26MultiTaskLoss(
            stage="stage_4_lane_family_finetune",
            loss_weights={"det": 0.0, "tl_attr": 0.0, "lane": 1.0, "stop_line": 0.0, "crosswalk": 0.0},
            task_mode="roadmark_joint",
            lane_segfirst_center_offset_aux_weight=1.0,
        )
        losses = criterion(predictions, encoded)
        losses["total"].backward()

        self.assertTrue(torch.isfinite(losses["lane"]))
        self.assertAlmostEqual(criterion.export_config()["lane_segfirst_center_offset_aux_weight"], 1.0)
        self.assertIsNotNone(predictions["lane_seg_center_offset"].grad)

    def test_lane_anchor_offset_aux_loss_backprops_when_enabled(self) -> None:
        from model.engine.loss import PV26MultiTaskLoss
        from model.data.roadmark_v2_targets import ROADMARK_DENSE_OUTPUT_HW

        batch_size = 1
        h, w = ROADMARK_DENSE_OUTPUT_HW
        encoded = _with_zero_segfirst_targets(_make_encoded_batch(batch_size=batch_size, q_det=2))
        encoded["roadmark_v2"]["lane_seg_anchor_offset_valid"][:, :, 20, 20] = 1.0
        encoded["roadmark_v2"]["lane_seg_anchor_offset"][:, :, 20, 20] = 12.0
        predictions = _zero_predictions(batch_size=batch_size, q_det=2)
        predictions.update(
            {
                "lane_seg_centerline_logits": torch.zeros((batch_size, 1, h, w), requires_grad=True),
                "lane_seg_support_logits": torch.zeros((batch_size, 1, h, w), requires_grad=True),
                "lane_seg_anchor_offset": torch.zeros((batch_size, 1, h, w), requires_grad=True),
                "lane_seg_tangent_axis": torch.zeros((batch_size, 2, h, w), requires_grad=True),
                "lane_seg_color_logits": torch.zeros((batch_size, LANE_COLOR_DIM, h, w), requires_grad=True),
                "lane_seg_type_logits": torch.zeros((batch_size, LANE_TYPE_DIM, h, w), requires_grad=True),
                "stop_line_mask_logits": torch.zeros((batch_size, 1, h, w), requires_grad=True),
                "stop_line_center_logits": torch.zeros((batch_size, 1, h, w), requires_grad=True),
                "stop_line_center_offset": torch.zeros((batch_size, 2, h, w), requires_grad=True),
                "stop_line_angle": torch.zeros((batch_size, 2, h, w), requires_grad=True),
                "stop_line_half_length": torch.zeros((batch_size, 1, h, w), requires_grad=True),
                "stop_line_feature": torch.zeros((batch_size, 4, h, w), requires_grad=True),
                "crosswalk_mask_logits": torch.zeros((batch_size, 1, h, w), requires_grad=True),
                "crosswalk_boundary_logits": torch.zeros((batch_size, 1, h, w), requires_grad=True),
                "crosswalk_center_logits": torch.zeros((batch_size, 1, h, w), requires_grad=True),
                "crosswalk_feature": torch.zeros((batch_size, 4, h, w), requires_grad=True),
            }
        )
        criterion = PV26MultiTaskLoss(
            stage="stage_4_lane_family_finetune",
            loss_weights={"det": 0.0, "tl_attr": 0.0, "lane": 1.0, "stop_line": 0.0, "crosswalk": 0.0},
            task_mode="roadmark_joint",
            lane_segfirst_anchor_offset_aux_weight=1.0,
        )
        losses = criterion(predictions, encoded)
        losses["total"].backward()

        self.assertTrue(torch.isfinite(losses["lane"]))
        self.assertAlmostEqual(criterion.export_config()["lane_segfirst_anchor_offset_aux_weight"], 1.0)
        self.assertIsNotNone(predictions["lane_seg_anchor_offset"].grad)

    def test_lane_row_link_aux_loss_backprops_when_enabled(self) -> None:
        from model.engine.loss import PV26MultiTaskLoss
        from model.data.roadmark_v2_targets import ROADMARK_DENSE_OUTPUT_HW

        batch_size = 1
        h, w = ROADMARK_DENSE_OUTPUT_HW
        encoded = _with_zero_segfirst_targets(_make_encoded_batch(batch_size=batch_size, q_det=2))
        encoded["roadmark_v2"]["lane_seg_row_link_valid"][:, :, 20, 20] = 1.0
        encoded["roadmark_v2"]["lane_seg_row_link_delta"][:, :, 20, 20] = 3.0
        predictions = _zero_predictions(batch_size=batch_size, q_det=2)
        predictions.update(
            {
                "lane_seg_centerline_logits": torch.zeros((batch_size, 1, h, w), requires_grad=True),
                "lane_seg_support_logits": torch.zeros((batch_size, 1, h, w), requires_grad=True),
                "lane_seg_row_link_delta": torch.zeros((batch_size, 1, h, w), requires_grad=True),
                "lane_seg_tangent_axis": torch.zeros((batch_size, 2, h, w), requires_grad=True),
                "lane_seg_color_logits": torch.zeros((batch_size, LANE_COLOR_DIM, h, w), requires_grad=True),
                "lane_seg_type_logits": torch.zeros((batch_size, LANE_TYPE_DIM, h, w), requires_grad=True),
                "stop_line_mask_logits": torch.zeros((batch_size, 1, h, w), requires_grad=True),
                "stop_line_center_logits": torch.zeros((batch_size, 1, h, w), requires_grad=True),
                "stop_line_center_offset": torch.zeros((batch_size, 2, h, w), requires_grad=True),
                "stop_line_angle": torch.zeros((batch_size, 2, h, w), requires_grad=True),
                "stop_line_half_length": torch.zeros((batch_size, 1, h, w), requires_grad=True),
                "stop_line_feature": torch.zeros((batch_size, 4, h, w), requires_grad=True),
                "crosswalk_mask_logits": torch.zeros((batch_size, 1, h, w), requires_grad=True),
                "crosswalk_boundary_logits": torch.zeros((batch_size, 1, h, w), requires_grad=True),
                "crosswalk_center_logits": torch.zeros((batch_size, 1, h, w), requires_grad=True),
                "crosswalk_feature": torch.zeros((batch_size, 4, h, w), requires_grad=True),
            }
        )
        criterion = PV26MultiTaskLoss(
            stage="stage_4_lane_family_finetune",
            loss_weights={"det": 0.0, "tl_attr": 0.0, "lane": 1.0, "stop_line": 0.0, "crosswalk": 0.0},
            task_mode="roadmark_joint",
            lane_segfirst_row_link_aux_weight=1.0,
        )
        losses = criterion(predictions, encoded)
        losses["total"].backward()

        self.assertTrue(torch.isfinite(losses["lane"]))
        self.assertAlmostEqual(criterion.export_config()["lane_segfirst_row_link_aux_weight"], 1.0)
        self.assertIsNotNone(predictions["lane_seg_row_link_delta"].grad)

    def test_lane_instance_embedding_aux_loss_backprops_when_enabled(self) -> None:
        from model.engine.loss import PV26MultiTaskLoss
        from model.data.roadmark_v2_targets import ROADMARK_DENSE_OUTPUT_HW

        batch_size = 1
        h, w = ROADMARK_DENSE_OUTPUT_HW
        encoded = _with_zero_segfirst_targets(_make_encoded_batch(batch_size=batch_size, q_det=2))
        encoded["roadmark_v2"]["lane_seg_centerline_core"][:, :, 20:24, 20:24] = 1.0
        encoded["roadmark_v2"]["lane_seg_centerline_core"][:, :, 50:54, 60:64] = 1.0
        encoded["roadmark_v2"]["lane_seg_instance_id"][:, 20:24, 20:24] = 1
        encoded["roadmark_v2"]["lane_seg_instance_id"][:, 50:54, 60:64] = 2
        predictions = _zero_predictions(batch_size=batch_size, q_det=2)
        predictions.update(
            {
                "lane_seg_centerline_logits": torch.zeros((batch_size, 1, h, w), requires_grad=True),
                "lane_seg_support_logits": torch.zeros((batch_size, 1, h, w), requires_grad=True),
                "lane_seg_tangent_axis": torch.zeros((batch_size, 2, h, w), requires_grad=True),
                "lane_seg_instance_embedding": torch.zeros((batch_size, 2, h, w), requires_grad=True),
                "lane_seg_color_logits": torch.zeros((batch_size, LANE_COLOR_DIM, h, w), requires_grad=True),
                "lane_seg_type_logits": torch.zeros((batch_size, LANE_TYPE_DIM, h, w), requires_grad=True),
            }
        )

        criterion = PV26MultiTaskLoss(
            stage="stage_4_lane_family_finetune",
            loss_weights={"stop_line": 0.0, "crosswalk": 0.0},
            lane_segfirst_instance_embedding_aux_weight=1.0,
        )
        losses = criterion(predictions, encoded)

        self.assertTrue(torch.isfinite(losses["total"]))
        self.assertAlmostEqual(criterion.export_config()["lane_segfirst_instance_embedding_aux_weight"], 1.0)
        self.assertIn("seg_instance_embedding", criterion.last_lane_loss_breakdown)
        losses["total"].backward()
        self.assertIsNotNone(predictions["lane_seg_instance_embedding"].grad)

    def test_lane_conditional_seed_aux_loss_backprops_without_row_aux(self) -> None:
        from model.engine.loss import PV26MultiTaskLoss
        from model.data.roadmark_v2_targets import ROADMARK_DENSE_OUTPUT_HW

        batch_size = 1
        h, w = ROADMARK_DENSE_OUTPUT_HW
        encoded = _with_zero_segfirst_targets(_make_encoded_batch(batch_size=batch_size, q_det=2))
        predictions = _zero_predictions(batch_size=batch_size, q_det=2)
        predictions.update(
            {
                "lane_seg_centerline_logits": torch.zeros((batch_size, 1, h, w), requires_grad=True),
                "lane_seg_support_logits": torch.zeros((batch_size, 1, h, w), requires_grad=True),
                "lane_seg_tangent_axis": torch.zeros((batch_size, 2, h, w), requires_grad=True),
                "lane_seg_color_logits": torch.zeros((batch_size, LANE_COLOR_DIM, h, w), requires_grad=True),
                "lane_seg_type_logits": torch.zeros((batch_size, LANE_TYPE_DIM, h, w), requires_grad=True),
                "lane_conditional_seed_logits": torch.zeros((batch_size, 1, h, w), requires_grad=True),
                "lane_conditional_rows": torch.zeros(
                    (batch_size, LANE_QUERY_COUNT, LANE_VECTOR_DIM),
                    requires_grad=True,
                ),
            }
        )

        criterion = PV26MultiTaskLoss(
            stage="stage_4_lane_family_finetune",
            loss_weights={"stop_line": 0.0, "crosswalk": 0.0},
            lane_conditional_seed_aux_weight=1.0,
            lane_conditional_seed_target_mode="bottom_anchor",
        )
        losses = criterion(predictions, encoded)

        self.assertTrue(torch.isfinite(losses["total"]))
        self.assertAlmostEqual(criterion.export_config()["lane_conditional_seed_aux_weight"], 1.0)
        losses["total"].backward()
        self.assertIsNone(predictions["lane_conditional_rows"].grad)
        self.assertIsNotNone(predictions["lane_conditional_seed_logits"].grad)

    def test_stage4_promotes_half_precision_lane_predictions_to_float32_for_loss(self) -> None:
        from model.engine.loss import PV26MultiTaskLoss

        encoded = _make_encoded_batch(batch_size=1, q_det=2)
        predictions = _zero_predictions(batch_size=1, q_det=2)
        predictions["lane"] = predictions["lane"].detach().to(dtype=torch.float16).requires_grad_(True)

        observed: dict[str, torch.dtype] = {}
        criterion = PV26MultiTaskLoss(
            stage="stage_4_lane_family_finetune",
            loss_weights={"stop_line": 0.0, "crosswalk": 0.0},
        )
        original_build_query_assignment = criterion._build_query_assignment

        def _record_dtype(
            pred_rows: torch.Tensor,
            target_rows: torch.Tensor,
            valid_mask: torch.Tensor,
            source_mask: torch.Tensor,
            *,
            task_name: str,
            cost_builder,
            quality_builder,
        ):
            observed[task_name] = pred_rows.dtype
            return original_build_query_assignment(
                pred_rows,
                target_rows,
                valid_mask,
                source_mask,
                task_name=task_name,
                cost_builder=cost_builder,
                quality_builder=quality_builder,
            )

        criterion._build_query_assignment = _record_dtype  # type: ignore[method-assign]
        losses = criterion(predictions, encoded)

        self.assertEqual(observed["lane"], torch.float32)
        self.assertTrue(torch.isfinite(losses["lane"]))
        self.assertTrue(torch.isfinite(losses["total"]))
        losses["total"].backward()
        self.assertIsNotNone(predictions["lane"].grad)

    def test_stop_line_assignment_sanitizes_partial_invalid_cost_entries(self) -> None:
        from model.engine.loss import PV26MultiTaskLoss

        encoded = _make_encoded_batch(batch_size=1, q_det=2)
        predictions = _zero_predictions(batch_size=1, q_det=2)
        stop_line = predictions["stop_line"].detach().clone()
        stop_line[0, 0, 1:] = float("nan")
        predictions["stop_line"] = stop_line.requires_grad_(True)

        criterion = PV26MultiTaskLoss(stage="stage_4_lane_family_finetune")
        losses = criterion(predictions, encoded)

        self.assertEqual(criterion.last_lane_assignment_modes["stop_line"], "hungarian")
        self.assertTrue(torch.isfinite(losses["stop_line"]))
        self.assertTrue(torch.isfinite(losses["total"]))
        losses["total"].backward()
        self.assertIsNotNone(predictions["stop_line"].grad)

    def test_stop_line_assignment_skips_samples_with_all_invalid_cost_entries(self) -> None:
        from model.engine.loss import PV26MultiTaskLoss

        encoded = _make_encoded_batch(batch_size=1, q_det=2)
        predictions = _zero_predictions(batch_size=1, q_det=2)
        stop_line = predictions["stop_line"].detach().clone()
        stop_line[0, :, 1:] = float("nan")
        predictions["stop_line"] = stop_line.requires_grad_(True)

        criterion = PV26MultiTaskLoss(stage="stage_4_lane_family_finetune")
        losses = criterion(predictions, encoded)

        self.assertEqual(criterion.last_lane_assignment_modes["stop_line"], "hungarian")
        self.assertTrue(torch.isfinite(losses["stop_line"]))
        self.assertTrue(torch.isfinite(losses["total"]))
        losses["total"].backward()
        self.assertIsNotNone(predictions["stop_line"].grad)

    def test_task_aligned_assignment_promotes_amp_inputs_to_float32(self) -> None:
        from model.engine.loss import PV26MultiTaskLoss

        encoded = _make_encoded_batch(batch_size=1, q_det=2)
        predictions = _zero_predictions(batch_size=1, q_det=2)
        predictions["det"] = predictions["det"].detach().to(dtype=torch.float16).requires_grad_(True)
        predictions["det_feature_shapes"] = [(1, 2)]
        predictions["det_feature_strides"] = [8]

        observed: dict[str, torch.dtype] = {}

        class FakeTaskAlignedAssigner(nn.Module):
            def forward(
                self,
                pred_scores: torch.Tensor,
                pred_bboxes: torch.Tensor,
                anchor_points: torch.Tensor,
                gt_labels: torch.Tensor,
                gt_bboxes: torch.Tensor,
                mask_gt: torch.Tensor,
            ):
                observed["pred_scores"] = pred_scores.dtype
                observed["pred_bboxes"] = pred_bboxes.dtype
                observed["anchor_points"] = anchor_points.dtype
                batch_size, query_count, num_classes = pred_scores.shape
                assigned_labels = torch.zeros((batch_size, query_count), device=pred_scores.device, dtype=torch.long)
                assigned_bboxes = torch.zeros((batch_size, query_count, 4), device=pred_scores.device, dtype=torch.float32)
                assigned_scores = torch.zeros(
                    (batch_size, query_count, num_classes),
                    device=pred_scores.device,
                    dtype=torch.float32,
                )
                assigned_fg = torch.zeros((batch_size, query_count), device=pred_scores.device, dtype=torch.bool)
                assigned_gt_idx = torch.full((batch_size, query_count), -1, device=pred_scores.device, dtype=torch.long)
                assigned_fg[0, 0] = True
                assigned_gt_idx[0, 0] = 0
                assigned_bboxes[0, 0] = gt_bboxes[0, 0]
                assigned_scores[0, 0, int(gt_labels[0, 0, 0].item())] = 1.0
                return assigned_labels, assigned_bboxes, assigned_scores, assigned_fg, assigned_gt_idx

        criterion = PV26MultiTaskLoss(stage="stage_1_frozen_trunk_warmup")
        criterion.assigner = FakeTaskAlignedAssigner()

        assignment = criterion._build_det_assignment(predictions, encoded)

        self.assertEqual(str(assignment["mode"]), "task_aligned")
        self.assertEqual(observed["pred_scores"], torch.float32)
        self.assertEqual(observed["pred_bboxes"], torch.float32)
        self.assertEqual(observed["anchor_points"], torch.float32)
        self.assertEqual(tuple(assignment["target_bboxes"].shape), (1, 2, 4))
        self.assertTrue(bool(assignment["fg_mask"][0, 0]))

    def test_loss_returns_finite_components_and_supports_backward(self) -> None:
        from model.engine.loss import PV26MultiTaskLoss

        encoded = _make_encoded_batch(batch_size=2, q_det=8)
        predictions = _zero_predictions(batch_size=2, q_det=8)
        predictions["det"] = torch.randn(2, 8, 12, requires_grad=True)
        predictions["tl_attr"] = torch.randn(2, 8, 4, requires_grad=True)
        predictions["lane"] = torch.randn(2, LANE_QUERY_COUNT, LANE_VECTOR_DIM, requires_grad=True)
        predictions["stop_line"] = torch.randn(2, STOP_LINE_QUERY_COUNT, STOP_LINE_VECTOR_DIM, requires_grad=True)
        predictions["crosswalk"] = torch.randn(2, CROSSWALK_QUERY_COUNT, CROSSWALK_VECTOR_DIM, requires_grad=True)

        criterion = PV26MultiTaskLoss(stage="stage_1_frozen_trunk_warmup")
        losses = criterion(predictions, encoded)
        self.assertEqual(criterion.last_det_assignment_mode, "task_aligned")
        self.assertEqual(criterion.last_lane_assignment_modes["lane"], "hungarian")
        self.assertEqual(criterion.last_lane_assignment_modes["stop_line"], "hungarian")
        self.assertEqual(criterion.last_lane_assignment_modes["crosswalk"], "disabled")

        self.assertTrue(torch.isfinite(losses["total"]))
        self.assertTrue(torch.isfinite(losses["det"]))
        self.assertTrue(torch.isfinite(losses["tl_attr"]))
        self.assertTrue(torch.isfinite(losses["lane"]))
        self.assertTrue(torch.isfinite(losses["stop_line"]))
        self.assertTrue(torch.isfinite(losses["crosswalk"]))

        losses["total"].backward()
        self.assertIsNotNone(predictions["det"].grad)
        self.assertIsNotNone(predictions["tl_attr"].grad)
        self.assertIsNotNone(predictions["lane"].grad)

    def test_loss_handles_no_source_batch_without_nan(self) -> None:
        from model.engine.loss import PV26MultiTaskLoss

        encoded = _make_encoded_batch(batch_size=1, q_det=8)
        for key in ("det_source", "tl_attr_source", "lane_source", "stop_line_source", "crosswalk_source"):
            encoded["mask"][key] = torch.zeros(1, dtype=torch.bool)
        encoded["det_gt"]["valid_mask"].zero_()
        encoded["tl_attr_gt_mask"].zero_()
        encoded["mask"]["lane_valid"].zero_()
        encoded["mask"]["stop_line_valid"].zero_()
        encoded["mask"]["crosswalk_valid"].zero_()
        encoded["lane"].zero_()
        encoded["stop_line"].zero_()
        encoded["crosswalk"].zero_()

        predictions = {
            "det": torch.randn(1, 8, 12, requires_grad=True),
            "tl_attr": torch.randn(1, 8, 4, requires_grad=True),
            "lane": torch.randn(1, LANE_QUERY_COUNT, LANE_VECTOR_DIM, requires_grad=True),
            "stop_line": torch.randn(1, STOP_LINE_QUERY_COUNT, STOP_LINE_VECTOR_DIM, requires_grad=True),
            "crosswalk": torch.randn(1, CROSSWALK_QUERY_COUNT, CROSSWALK_VECTOR_DIM, requires_grad=True),
        }

        criterion = PV26MultiTaskLoss(stage="stage_1_frozen_trunk_warmup")
        losses = criterion(predictions, encoded)
        self.assertEqual(criterion.last_det_assignment_mode, "zero_positive")
        self.assertEqual(criterion.last_lane_assignment_modes["lane"], "hungarian")
        self.assertEqual(criterion.last_lane_assignment_modes["stop_line"], "hungarian")
        self.assertEqual(criterion.last_lane_assignment_modes["crosswalk"], "disabled")

        self.assertTrue(torch.isfinite(losses["total"]))
        self.assertEqual(float(losses["total"].detach().cpu()), 0.0)
        losses["total"].backward()
        self.assertIsNotNone(predictions["det"].grad)

    def test_tl_attr_loss_only_binds_to_traffic_light_matches(self) -> None:
        from model.engine.loss import PV26MultiTaskLoss

        for class_id, expected_non_zero in ((TL_CLASS_ID, True), (SIGN_CLASS_ID, False)):
            encoded = {
                "image": torch.randn(1, 3, 608, 800),
                "det_gt": {
                    "boxes_xyxy": torch.tensor([[[40.0, 50.0, 120.0, 180.0]]], dtype=torch.float32),
                    "classes": torch.tensor([[class_id]], dtype=torch.long),
                    "valid_mask": torch.tensor([[True]], dtype=torch.bool),
                },
                "tl_attr_gt_bits": torch.tensor([[[1.0, 0.0, 0.0, 1.0]]], dtype=torch.float32),
                "tl_attr_gt_mask": torch.tensor([[True]], dtype=torch.bool),
                "lane": torch.zeros((1, LANE_QUERY_COUNT, LANE_VECTOR_DIM), dtype=torch.float32),
                "stop_line": torch.zeros((1, STOP_LINE_QUERY_COUNT, STOP_LINE_VECTOR_DIM), dtype=torch.float32),
                "crosswalk": torch.zeros((1, CROSSWALK_QUERY_COUNT, CROSSWALK_VECTOR_DIM), dtype=torch.float32),
                "mask": {
                    "det_source": torch.tensor([True], dtype=torch.bool),
                    "det_supervised_class_mask": torch.ones((1, len(OD_CLASSES)), dtype=torch.bool),
                    "det_allow_objectness_negatives": torch.tensor([False], dtype=torch.bool),
                    "det_allow_unmatched_class_negatives": torch.tensor([True], dtype=torch.bool),
                    "tl_attr_source": torch.tensor([True], dtype=torch.bool),
                    "lane_source": torch.tensor([False], dtype=torch.bool),
                    "stop_line_source": torch.tensor([False], dtype=torch.bool),
                    "crosswalk_source": torch.tensor([False], dtype=torch.bool),
                    "lane_valid": torch.zeros((1, LANE_QUERY_COUNT), dtype=torch.bool),
                    "stop_line_valid": torch.zeros((1, STOP_LINE_QUERY_COUNT), dtype=torch.bool),
                    "crosswalk_valid": torch.zeros((1, CROSSWALK_QUERY_COUNT), dtype=torch.bool),
                },
                "meta": [{"sample_id": f"class_{class_id}"}],
            }
            predictions = _zero_predictions(batch_size=1, q_det=2)
            criterion = PV26MultiTaskLoss(stage="stage_1_frozen_trunk_warmup")
            criterion.assigner = _FakeSingleMatchTaskAlignedAssigner()

            losses = criterion(predictions, encoded)
            losses["tl_attr"].backward()
            tl_grad_sum = float(predictions["tl_attr"].grad.abs().sum().item())

            with self.subTest(class_id=class_id):
                if expected_non_zero:
                    self.assertGreater(float(losses["tl_attr"].detach().cpu()), 0.0)
                    self.assertGreater(tl_grad_sum, 0.0)
                else:
                    self.assertEqual(float(losses["tl_attr"].detach().cpu()), 0.0)
                    self.assertEqual(tl_grad_sum, 0.0)

    def test_partial_det_samples_enable_class_negatives_without_objectness_negatives(self) -> None:
        from model.engine.loss import PV26MultiTaskLoss

        encoded = {
            "image": torch.randn(1, 3, 608, 800),
            "det_gt": {
                "boxes_xyxy": torch.tensor([[[40.0, 50.0, 120.0, 180.0]]], dtype=torch.float32),
                "classes": torch.tensor([[TL_CLASS_ID]], dtype=torch.long),
                "valid_mask": torch.tensor([[True]], dtype=torch.bool),
            },
            "tl_attr_gt_bits": torch.zeros((1, 1, 4), dtype=torch.float32),
            "tl_attr_gt_mask": torch.zeros((1, 1), dtype=torch.bool),
            "lane": torch.zeros((1, LANE_QUERY_COUNT, LANE_VECTOR_DIM), dtype=torch.float32),
            "stop_line": torch.zeros((1, STOP_LINE_QUERY_COUNT, STOP_LINE_VECTOR_DIM), dtype=torch.float32),
            "crosswalk": torch.zeros((1, CROSSWALK_QUERY_COUNT, CROSSWALK_VECTOR_DIM), dtype=torch.float32),
            "mask": {
                "det_source": torch.tensor([True], dtype=torch.bool),
                "det_supervised_class_mask": torch.tensor([[False, False, False, False, False, True, True]], dtype=torch.bool),
                "det_allow_objectness_negatives": torch.tensor([False], dtype=torch.bool),
                "det_allow_unmatched_class_negatives": torch.tensor([True], dtype=torch.bool),
                "tl_attr_source": torch.tensor([False], dtype=torch.bool),
                "lane_source": torch.tensor([False], dtype=torch.bool),
                "stop_line_source": torch.tensor([False], dtype=torch.bool),
                "crosswalk_source": torch.tensor([False], dtype=torch.bool),
                "lane_valid": torch.zeros((1, LANE_QUERY_COUNT), dtype=torch.bool),
                "stop_line_valid": torch.zeros((1, STOP_LINE_QUERY_COUNT), dtype=torch.bool),
                "crosswalk_valid": torch.zeros((1, CROSSWALK_QUERY_COUNT), dtype=torch.bool),
            },
            "meta": [{"sample_id": "partial_det"}],
        }
        predictions = _zero_predictions(batch_size=1, q_det=2)

        criterion = PV26MultiTaskLoss(stage="stage_1_frozen_trunk_warmup")
        criterion.assigner = _FakeSingleMatchTaskAlignedAssigner()
        losses = criterion(predictions, encoded)
        losses["det"].backward()

        det_grad = predictions["det"].grad[0]
        self.assertNotEqual(float(det_grad[0, 4]), 0.0)
        self.assertEqual(float(det_grad[1, 4]), 0.0)
        self.assertEqual(float(det_grad[1, 5:10].abs().sum()), 0.0)
        self.assertGreater(float(det_grad[1, 10].abs()), 0.0)
        self.assertGreater(float(det_grad[1, 11].abs()), 0.0)
        self.assertGreater(float(det_grad[0, 10].abs()), 0.0)
        self.assertGreater(float(det_grad[0, 11].abs()), 0.0)

    def test_large_query_class_negative_scaling_stays_finite(self) -> None:
        from model.engine.loss import PV26MultiTaskLoss

        encoded = {
            "image": torch.randn(1, 3, 608, 800),
            "det_gt": {
                "boxes_xyxy": torch.tensor([[[40.0, 50.0, 120.0, 180.0]]], dtype=torch.float32),
                "classes": torch.tensor([[TL_CLASS_ID]], dtype=torch.long),
                "valid_mask": torch.tensor([[True]], dtype=torch.bool),
            },
            "tl_attr_gt_bits": torch.zeros((1, 1, 4), dtype=torch.float32),
            "tl_attr_gt_mask": torch.zeros((1, 1), dtype=torch.bool),
            "lane": torch.zeros((1, LANE_QUERY_COUNT, LANE_VECTOR_DIM), dtype=torch.float32),
            "stop_line": torch.zeros((1, STOP_LINE_QUERY_COUNT, STOP_LINE_VECTOR_DIM), dtype=torch.float32),
            "crosswalk": torch.zeros((1, CROSSWALK_QUERY_COUNT, CROSSWALK_VECTOR_DIM), dtype=torch.float32),
            "mask": {
                "det_source": torch.tensor([True], dtype=torch.bool),
                "det_supervised_class_mask": torch.tensor([[False, False, False, False, False, True, True]], dtype=torch.bool),
                "det_allow_objectness_negatives": torch.tensor([False], dtype=torch.bool),
                "det_allow_unmatched_class_negatives": torch.tensor([True], dtype=torch.bool),
                "tl_attr_source": torch.tensor([False], dtype=torch.bool),
                "lane_source": torch.tensor([False], dtype=torch.bool),
                "stop_line_source": torch.tensor([False], dtype=torch.bool),
                "crosswalk_source": torch.tensor([False], dtype=torch.bool),
                "lane_valid": torch.zeros((1, LANE_QUERY_COUNT), dtype=torch.bool),
                "stop_line_valid": torch.zeros((1, STOP_LINE_QUERY_COUNT), dtype=torch.bool),
                "crosswalk_valid": torch.zeros((1, CROSSWALK_QUERY_COUNT), dtype=torch.bool),
            },
            "meta": [{"sample_id": "stress_det"}],
        }
        predictions = _zero_predictions(batch_size=1, q_det=9975)

        criterion = PV26MultiTaskLoss(stage="stage_1_frozen_trunk_warmup")
        losses = criterion(predictions, encoded)
        losses["det"].backward()

        det_grad = predictions["det"].grad[0]
        self.assertEqual(float(det_grad[100, 4]), 0.0)
        self.assertEqual(float(det_grad[100, 5:10].abs().sum()), 0.0)
        self.assertGreater(float(det_grad[100, 10].abs()), 0.0)
        self.assertGreater(float(det_grad[100, 11].abs()), 0.0)
        self.assertTrue(torch.isfinite(losses["det"]))
        self.assertTrue(torch.isfinite(torch.tensor(float(criterion.last_det_loss_breakdown["det_cls_matched_loss"]))))
        self.assertTrue(torch.isfinite(torch.tensor(float(criterion.last_det_loss_breakdown["det_cls_unmatched_neg_loss"]))))
        self.assertGreater(int(criterion.last_det_loss_breakdown["det_cls_unmatched_neg_count"]), 1000)
        self.assertGreater(int(criterion.last_det_loss_breakdown["det_cls_unmatched_neg_count"]), int(criterion.last_det_loss_breakdown["det_cls_matched_count"]))
        self.assertGreaterEqual(float(criterion.last_det_loss_breakdown["det_cls_unmatched_neg_loss"]), 0.0)

    def test_encoded_det_supervision_contract_requires_explicit_masks(self) -> None:
        from model.engine.loss import PV26MultiTaskLoss

        encoded = _make_encoded_batch(batch_size=1, q_det=2)
        encoded["mask"].pop("det_allow_objectness_negatives")
        predictions = _zero_predictions(batch_size=1, q_det=2)
        criterion = PV26MultiTaskLoss(stage="stage_1_frozen_trunk_warmup")

        with self.assertRaisesRegex(ValueError, "missing mask fields det_allow_objectness_negatives"):
            criterion(predictions, encoded)

    def test_encoded_det_supervision_contract_rejects_shape_mismatch(self) -> None:
        from model.engine.loss import PV26MultiTaskLoss

        encoded = _make_encoded_batch(batch_size=1, q_det=2)
        encoded["mask"]["det_allow_objectness_negatives"] = torch.ones((1, 1), dtype=torch.bool)
        predictions = _zero_predictions(batch_size=1, q_det=2)
        criterion = PV26MultiTaskLoss(stage="stage_1_frozen_trunk_warmup")

        with self.assertRaisesRegex(ValueError, "det_allow_objectness_negatives shape"):
            criterion(predictions, encoded)

    def test_encoded_det_supervision_contract_rejects_empty_supervised_rows(self) -> None:
        from model.engine.loss import PV26MultiTaskLoss

        encoded = _make_encoded_batch(batch_size=1, q_det=2)
        encoded["mask"]["det_supervised_class_mask"].zero_()
        predictions = _zero_predictions(batch_size=1, q_det=2)
        criterion = PV26MultiTaskLoss(stage="stage_1_frozen_trunk_warmup")

        with self.assertRaisesRegex(ValueError, "require at least one supervised detector class"):
            criterion(predictions, encoded)

    def test_missing_detector_feature_metadata_raises(self) -> None:
        from model.engine.loss import PV26MultiTaskLoss

        encoded = _make_encoded_batch(batch_size=1, q_det=2)
        predictions = _zero_predictions(batch_size=1, q_det=2)
        predictions["det"] = torch.full((1, 2, 12), float("nan"), dtype=torch.float32, requires_grad=True)
        predictions["tl_attr"] = torch.full((1, 2, 4), float("nan"), dtype=torch.float32, requires_grad=True)
        predictions.pop("det_feature_shapes")
        predictions.pop("det_feature_strides")
        criterion = PV26MultiTaskLoss(stage="stage_1_frozen_trunk_warmup")

        with self.assertRaisesRegex(Exception, "det_feature_metadata_invalid"):
            criterion(predictions, encoded)

    @unittest.skipUnless(has_yolo26_runtime(), "requires ultralytics yolo26 runtime")
    def test_real_trunk_heads_and_loss_support_backward_regression(self) -> None:
        from model.net import PV26Heads
        from model.engine.loss import PV26MultiTaskLoss
        from model.net import build_yolo26_roadmark_trunk
        from model.net import forward_pyramid_features

        encoded = _with_zero_segfirst_targets(_make_encoded_batch(batch_size=1, q_det=9975))
        adapter = build_yolo26_roadmark_trunk(variant="n")
        image = encoded["image"].clone().requires_grad_(False)

        features = forward_pyramid_features(adapter, image)
        heads = PV26Heads(in_channels=tuple(int(feature.shape[1]) for feature in features))
        predictions = heads(features)
        criterion = PV26MultiTaskLoss(stage="stage_1_frozen_trunk_warmup")
        losses = criterion(predictions, encoded)
        self.assertEqual(criterion.last_det_assignment_mode, "task_aligned")
        self.assertEqual(criterion.last_lane_assignment_modes["lane"], "seg_first_dense")
        self.assertEqual(criterion.last_lane_assignment_modes["stop_line"], "hungarian")
        self.assertEqual(criterion.last_lane_assignment_modes["crosswalk"], "disabled")

        self.assertTrue(torch.isfinite(losses["total"]))
        losses["total"].backward()

        head_parameter = next(heads.parameters())
        trunk_parameter = next(adapter.trunk.parameters())
        self.assertIsNotNone(head_parameter.grad)
        self.assertIsNotNone(trunk_parameter.grad)


if __name__ == "__main__":
    unittest.main()
