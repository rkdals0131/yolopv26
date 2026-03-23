from __future__ import annotations

from typing import Any

from scipy.optimize import linear_sum_assignment
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils.tal import TaskAlignedAssigner

from .spec import build_loss_spec


SPEC = build_loss_spec()
TL_BITS = tuple(SPEC["model_contract"]["tl_bits"])
OD_CLASSES = tuple(SPEC["model_contract"]["od_classes"])
STAGE_LOSS_WEIGHTS = {
    stage["name"]: dict(stage["loss_weights"]) for stage in SPEC["training_schedule"]
}
STAGE_ALIASES = {
    "stage_1_head_warmup": "stage_1_frozen_trunk_warmup",
}
DEFAULT_DET_FEATURE_STRIDES = (8, 16, 32)


def _canonical_stage(stage: str) -> str:
    return STAGE_ALIASES.get(stage, stage)


def _zero_graph(*tensors: torch.Tensor) -> torch.Tensor:
    zero = None
    for tensor in tensors:
        if not isinstance(tensor, torch.Tensor):
            continue
        term = tensor.sum() * 0.0
        zero = term if zero is None else zero + term
    if zero is None:
        zero = torch.tensor(0.0)
    return zero


def _mean_or_zero(values: list[torch.Tensor], *graph_tensors: torch.Tensor) -> torch.Tensor:
    if values:
        return torch.stack(values).mean()
    return _zero_graph(*graph_tensors)


def _smoothness_regularizer(points: torch.Tensor) -> torch.Tensor:
    if points.shape[1] < 3:
        return points.sum() * 0.0
    second = points[:, 2:] - 2.0 * points[:, 1:-1] + points[:, :-2]
    return second.abs().mean()


def _polygon_shape_regularizer(points: torch.Tensor) -> torch.Tensor:
    rolled = torch.roll(points, shifts=-1, dims=1)
    edge_lengths = torch.linalg.norm(rolled - points, dim=-1)
    return (edge_lengths - edge_lengths.mean(dim=1, keepdim=True)).abs().mean()


def _make_anchor_grid(
    feature_shapes: list[tuple[int, int]],
    feature_strides: list[int],
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    anchor_points: list[torch.Tensor] = []
    stride_tensors: list[torch.Tensor] = []
    for (height, width), stride in zip(feature_shapes, feature_strides):
        sy = (torch.arange(height, device=device, dtype=dtype) + 0.5) * float(stride)
        sx = (torch.arange(width, device=device, dtype=dtype) + 0.5) * float(stride)
        grid_y, grid_x = torch.meshgrid(sy, sx, indexing="ij")
        anchor_points.append(torch.stack((grid_x, grid_y), dim=-1).reshape(-1, 2))
        stride_tensors.append(torch.full((height * width, 1), float(stride), dtype=dtype, device=device))
    return torch.cat(anchor_points, dim=0), torch.cat(stride_tensors, dim=0)


def _decode_anchor_relative_boxes(
    pred_ltrb_logits: torch.Tensor,
    anchor_points: torch.Tensor,
    stride_tensor: torch.Tensor,
) -> torch.Tensor:
    distances = F.softplus(pred_ltrb_logits) * stride_tensor.unsqueeze(0)
    x1 = anchor_points[:, 0].unsqueeze(0) - distances[..., 0]
    y1 = anchor_points[:, 1].unsqueeze(0) - distances[..., 1]
    x2 = anchor_points[:, 0].unsqueeze(0) + distances[..., 2]
    y2 = anchor_points[:, 1].unsqueeze(0) + distances[..., 3]
    return torch.stack((x1, y1, x2, y2), dim=-1)


def _batched_box_from_points(points: torch.Tensor) -> torch.Tensor:
    x = points[..., 0]
    y = points[..., 1]
    return torch.stack((x.min(dim=-1).values, y.min(dim=-1).values, x.max(dim=-1).values, y.max(dim=-1).values), dim=-1)


def _batched_angle_length_cost(pred_points: torch.Tensor, target_points: torch.Tensor) -> torch.Tensor:
    pred_vec = pred_points[..., -1, :] - pred_points[..., 0, :]
    target_vec = target_points[..., -1, :] - target_points[..., 0, :]
    pred_len = torch.linalg.norm(pred_vec, dim=-1)
    target_len = torch.linalg.norm(target_vec, dim=-1)
    length_cost = (pred_len - target_len).abs() / target_len.clamp(min=1.0)
    cosine = (pred_vec * target_vec).sum(dim=-1) / (pred_len * target_len).clamp(min=1e-6)
    angle_cost = 1.0 - cosine.clamp(min=-1.0, max=1.0)
    return length_cost + angle_cost


def _hungarian_match(cost_matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if cost_matrix.numel() == 0 or cost_matrix.shape[0] == 0 or cost_matrix.shape[1] == 0:
        empty = torch.empty(0, dtype=torch.long, device=cost_matrix.device)
        return empty, empty
    row_ind, col_ind = linear_sum_assignment(cost_matrix.detach().cpu().numpy())
    return (
        torch.as_tensor(row_ind, dtype=torch.long, device=cost_matrix.device),
        torch.as_tensor(col_ind, dtype=torch.long, device=cost_matrix.device),
    )


def _lane_cost_matrix(pred_rows: torch.Tensor, gt_rows: torch.Tensor) -> torch.Tensor:
    pred_points = pred_rows[:, 6:38].view(pred_rows.shape[0], 16, 2)
    gt_points = gt_rows[:, 6:38].view(gt_rows.shape[0], 16, 2)
    points_cost = (pred_points[:, None] - gt_points[None]).abs().mean(dim=(2, 3))

    color_targets = gt_rows[:, 1:4].argmax(dim=-1)
    type_targets = gt_rows[:, 4:6].argmax(dim=-1)
    color_cost = -F.log_softmax(pred_rows[:, 1:4], dim=-1)[:, color_targets]
    type_cost = -F.log_softmax(pred_rows[:, 4:6], dim=-1)[:, type_targets]
    vis_cost = F.binary_cross_entropy_with_logits(
        pred_rows[:, None, 38:54].expand(-1, gt_rows.shape[0], -1),
        gt_rows[None, :, 38:54].expand(pred_rows.shape[0], -1, -1),
        reduction="none",
    ).mean(dim=-1)
    return 3.0 * points_cost + 1.0 * color_cost + 0.5 * type_cost + 0.5 * vis_cost


def _stop_line_cost_matrix(pred_rows: torch.Tensor, gt_rows: torch.Tensor) -> torch.Tensor:
    pred_points = pred_rows[:, 1:9].view(pred_rows.shape[0], 4, 2)
    gt_points = gt_rows[:, 1:9].view(gt_rows.shape[0], 4, 2)
    points_cost = (pred_points[:, None] - gt_points[None]).abs().mean(dim=(2, 3))
    angle_length_cost = _batched_angle_length_cost(pred_points[:, None], gt_points[None])
    return 4.0 * points_cost + 0.5 * angle_length_cost


def _crosswalk_cost_matrix(pred_rows: torch.Tensor, gt_rows: torch.Tensor) -> torch.Tensor:
    pred_points = pred_rows[:, 1:17].view(pred_rows.shape[0], 8, 2)
    gt_points = gt_rows[:, 1:17].view(gt_rows.shape[0], 8, 2)
    points_cost = (pred_points[:, None] - gt_points[None]).abs().mean(dim=(2, 3))
    pred_boxes = _batched_box_from_points(pred_points)
    gt_boxes = _batched_box_from_points(gt_points)
    overlap = bbox_iou(pred_boxes[:, None, :], gt_boxes[None, :, :], xywh=False).squeeze(-1)
    return 3.0 * points_cost + (1.0 - overlap)


class PV26MultiTaskLoss(nn.Module):
    def __init__(self, stage: str = "stage_0_smoke") -> None:
        super().__init__()
        stage = _canonical_stage(stage)
        try:
            self.loss_weights = STAGE_LOSS_WEIGHTS[stage]
        except KeyError as exc:
            raise KeyError(f"unsupported PV26 loss stage: {stage}") from exc
        self.stage = stage
        self.register_buffer(
            "tl_bit_weights",
            torch.tensor(
                [1.0, 2.5, 1.0, 1.8],
                dtype=torch.float32,
            ),
            persistent=False,
        )
        self.assigner = TaskAlignedAssigner(
            topk=10,
            num_classes=len(OD_CLASSES),
            alpha=0.5,
            beta=6.0,
            stride=list(DEFAULT_DET_FEATURE_STRIDES),
        )
        self.last_det_assignment_mode = "uninitialized"
        self.last_det_positive_count = 0
        self.last_lane_assignment_modes = {
            "lane": "uninitialized",
            "stop_line": "uninitialized",
            "crosswalk": "uninitialized",
        }

    def forward(self, predictions: dict[str, torch.Tensor], encoded: dict[str, Any]) -> dict[str, torch.Tensor]:
        det_assignment = self._build_det_assignment(predictions, encoded)
        self.last_det_assignment_mode = str(det_assignment["mode"])
        self.last_det_positive_count = int(det_assignment["fg_mask"].sum().item())

        det = self._det_loss(predictions["det"], encoded, det_assignment)
        tl_attr = self._tl_attr_loss(predictions["tl_attr"], encoded, det_assignment)
        lane = self._lane_loss(predictions["lane"], encoded)
        stop_line = self._stop_line_loss(predictions["stop_line"], encoded)
        crosswalk = self._crosswalk_loss(predictions["crosswalk"], encoded)

        total = (
            self.loss_weights["det"] * det
            + self.loss_weights["tl_attr"] * tl_attr
            + self.loss_weights["lane"] * lane
            + self.loss_weights["stop_line"] * stop_line
            + self.loss_weights["crosswalk"] * crosswalk
        )
        return {
            "total": total,
            "det": det,
            "tl_attr": tl_attr,
            "lane": lane,
            "stop_line": stop_line,
            "crosswalk": crosswalk,
        }

    def _build_det_assignment(
        self,
        predictions: dict[str, torch.Tensor],
        encoded: dict[str, Any],
    ) -> dict[str, torch.Tensor | str]:
        det_gt = encoded["det_gt"]
        det_pred = predictions["det"]
        det_source = encoded["mask"]["det_source"].to(device=det_pred.device, dtype=torch.bool)
        obj_logits = det_pred[..., 4]
        cls_logits = det_pred[..., 5:]
        batch_size, query_count, _ = det_pred.shape
        num_classes = cls_logits.shape[-1]

        target_bboxes = torch.zeros((batch_size, query_count, 4), device=det_pred.device, dtype=torch.float32)
        target_scores = torch.zeros((batch_size, query_count, num_classes), device=det_pred.device, dtype=torch.float32)
        target_gt_idx = torch.full((batch_size, query_count), -1, device=det_pred.device, dtype=torch.long)
        fg_mask = torch.zeros((batch_size, query_count), device=det_pred.device, dtype=torch.bool)
        obj_target = torch.zeros((batch_size, query_count), device=det_pred.device, dtype=torch.float32)

        feature_shapes = predictions.get("det_feature_shapes")
        feature_strides = predictions.get("det_feature_strides")
        if (
            isinstance(feature_shapes, list)
            and isinstance(feature_strides, list)
            and len(feature_shapes) == len(feature_strides)
            and sum(int(height) * int(width) for height, width in feature_shapes) == query_count
        ):
            anchor_points, stride_tensor = _make_anchor_grid(
                [(int(height), int(width)) for height, width in feature_shapes],
                [int(value) for value in feature_strides],
                dtype=det_pred.dtype,
                device=det_pred.device,
            )
            pred_boxes = _decode_anchor_relative_boxes(det_pred[..., :4], anchor_points, stride_tensor)
            gt_labels = det_gt["classes"].to(device=det_pred.device, dtype=torch.long).clamp(min=0).unsqueeze(-1)
            gt_bboxes = det_gt["boxes_xyxy"].to(device=det_pred.device, dtype=torch.float32)
            mask_gt = det_gt["valid_mask"].to(device=det_pred.device, dtype=torch.bool).unsqueeze(-1)
            mask_gt = mask_gt & det_source[:, None, None]

            _, assigned_bboxes, assigned_scores, assigned_fg, assigned_gt_idx = self.assigner(
                cls_logits.detach().sigmoid(),
                pred_boxes.detach(),
                anchor_points,
                gt_labels,
                gt_bboxes,
                mask_gt,
            )
            assigned_fg = assigned_fg.to(dtype=torch.bool)
            assigned_scores = assigned_scores * det_source[:, None, None].to(dtype=det_pred.dtype)
            assigned_fg = assigned_fg & det_source[:, None]
            assigned_gt_idx = assigned_gt_idx.to(dtype=torch.long)

            target_bboxes = assigned_bboxes.to(dtype=torch.float32)
            target_scores = assigned_scores.to(dtype=torch.float32)
            fg_mask = assigned_fg
            target_gt_idx = torch.where(
                fg_mask,
                assigned_gt_idx,
                torch.full_like(assigned_gt_idx, -1),
            )
            obj_target = target_scores.max(dim=-1).values
            mode = "task_aligned"
        else:
            pred_boxes = det_pred[..., :4]
            anchor_points = torch.zeros((query_count, 2), device=det_pred.device, dtype=det_pred.dtype)
            stride_tensor = torch.ones((query_count, 1), device=det_pred.device, dtype=det_pred.dtype)
            for batch_index in range(batch_size):
                if not bool(det_source[batch_index]):
                    continue
                valid = det_gt["valid_mask"][batch_index].to(device=det_pred.device, dtype=torch.bool)
                valid_indices = torch.nonzero(valid, as_tuple=False).flatten()
                if valid_indices.numel() == 0:
                    continue
                positive_count = min(int(valid_indices.numel()), int(query_count))
                positive_indices = valid_indices[:positive_count]
                fg_mask[batch_index, :positive_count] = True
                obj_target[batch_index, :positive_count] = 1.0
                target_gt_idx[batch_index, :positive_count] = positive_indices
                target_bboxes[batch_index, :positive_count] = det_gt["boxes_xyxy"][batch_index, positive_indices].to(
                    device=det_pred.device, dtype=torch.float32
                )
                classes = det_gt["classes"][batch_index, positive_indices].to(device=det_pred.device, dtype=torch.long)
                target_scores[batch_index, torch.arange(positive_count, device=det_pred.device), classes] = 1.0
            mode = "prefix_smoke"

        target_score_sum = target_scores.sum().clamp(min=1.0)
        return {
            "mode": mode,
            "pred_boxes": pred_boxes,
            "obj_logits": obj_logits,
            "cls_logits": cls_logits,
            "target_bboxes": target_bboxes,
            "target_scores": target_scores,
            "target_gt_idx": target_gt_idx,
            "fg_mask": fg_mask,
            "obj_target": obj_target,
            "target_score_sum": target_score_sum,
            "anchor_points": anchor_points,
            "stride_tensor": stride_tensor,
            "det_source": det_source,
        }

    def _det_loss(
        self,
        det_pred: torch.Tensor,
        encoded: dict[str, Any],
        assignment: dict[str, torch.Tensor | str],
    ) -> torch.Tensor:
        obj_logits = assignment["obj_logits"]
        cls_logits = assignment["cls_logits"]
        pred_boxes = assignment["pred_boxes"]
        target_bboxes = assignment["target_bboxes"]
        target_scores = assignment["target_scores"]
        fg_mask = assignment["fg_mask"]
        obj_target = assignment["obj_target"]
        target_score_sum = assignment["target_score_sum"]
        det_source = assignment["det_source"]

        obj_mask = det_source[:, None].expand_as(obj_logits)
        obj_loss = F.binary_cross_entropy_with_logits(obj_logits, obj_target, reduction="none")
        obj_loss = (obj_loss * obj_mask.to(dtype=det_pred.dtype)).sum() / obj_mask.sum().clamp(min=1)

        cls_mask = det_source[:, None, None].expand_as(cls_logits)
        cls_loss = F.binary_cross_entropy_with_logits(cls_logits, target_scores, reduction="none")
        cls_loss = (cls_loss * cls_mask.to(dtype=det_pred.dtype)).sum() / target_score_sum

        if not bool(fg_mask.any()):
            return obj_loss + cls_loss

        weights = target_scores.sum(dim=-1)[fg_mask]
        iou = bbox_iou(pred_boxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True).squeeze(-1)
        iou_loss = ((1.0 - iou) * weights).sum() / target_score_sum
        l1_loss = (
            F.l1_loss(pred_boxes[fg_mask], target_bboxes[fg_mask], reduction="none").mean(dim=-1) * weights
        ).sum() / target_score_sum
        return obj_loss + cls_loss + iou_loss + l1_loss

    def _tl_attr_loss(
        self,
        tl_pred: torch.Tensor,
        encoded: dict[str, Any],
        assignment: dict[str, torch.Tensor | str],
    ) -> torch.Tensor:
        tl_source = encoded["mask"]["tl_attr_source"].to(device=tl_pred.device, dtype=torch.bool)
        tl_bits = encoded["tl_attr_gt_bits"].to(device=tl_pred.device, dtype=torch.float32)
        tl_mask = encoded["tl_attr_gt_mask"].to(device=tl_pred.device, dtype=torch.bool)
        det_classes = encoded["det_gt"]["classes"].to(device=tl_pred.device, dtype=torch.long)
        fg_mask = assignment["fg_mask"]
        target_gt_idx = assignment["target_gt_idx"]
        per_sample: list[torch.Tensor] = []

        for batch_index in range(tl_pred.shape[0]):
            if not bool(tl_source[batch_index]):
                continue
            assigned = fg_mask[batch_index] & (target_gt_idx[batch_index] >= 0)
            if not bool(assigned.any()):
                continue
            matched_gt = target_gt_idx[batch_index, assigned]
            valid = tl_mask[batch_index, matched_gt] & det_classes[batch_index, matched_gt].eq(len(OD_CLASSES) - 1)
            if not bool(valid.any()):
                continue
            logits = tl_pred[batch_index, assigned][valid]
            targets = tl_bits[batch_index, matched_gt][valid]
            loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
            weighted = loss * self.tl_bit_weights.to(device=tl_pred.device)
            per_sample.append(weighted.mean())

        return _mean_or_zero(per_sample, tl_pred)

    def _lane_loss(self, lane_pred: torch.Tensor, encoded: dict[str, Any]) -> torch.Tensor:
        lane_target = encoded["lane"].to(device=lane_pred.device, dtype=torch.float32)
        lane_source = encoded["mask"]["lane_source"].to(device=lane_pred.device, dtype=torch.bool)
        lane_valid = encoded["mask"]["lane_valid"].to(device=lane_pred.device, dtype=torch.bool)
        assignment = self._build_query_assignment(
            lane_pred,
            lane_target,
            lane_valid,
            lane_source,
            task_name="lane",
            cost_builder=_lane_cost_matrix,
        )

        source_mask = lane_source[:, None].expand_as(lane_pred[..., 0])
        obj_loss = F.binary_cross_entropy_with_logits(
            lane_pred[..., 0],
            assignment["obj_target"],
            reduction="none",
        )
        obj_loss = (obj_loss * source_mask.to(dtype=lane_pred.dtype)).sum() / source_mask.sum().clamp(min=1)

        valid = assignment["fg_mask"]
        if not bool(valid.any()):
            return obj_loss

        assigned_target = assignment["assigned_target"]
        color_target = assigned_target[..., 1:4].argmax(dim=-1)
        type_target = assigned_target[..., 4:6].argmax(dim=-1)
        color_loss = F.cross_entropy(lane_pred[..., 1:4][valid], color_target[valid], reduction="mean")
        type_loss = F.cross_entropy(lane_pred[..., 4:6][valid], type_target[valid], reduction="mean")
        points_loss = F.l1_loss(lane_pred[..., 6:38][valid], assigned_target[..., 6:38][valid], reduction="mean")
        vis_loss = F.binary_cross_entropy_with_logits(
            lane_pred[..., 38:54][valid],
            assigned_target[..., 38:54][valid],
            reduction="mean",
        )
        smoothness = _smoothness_regularizer(lane_pred[..., 6:38][valid].view(-1, 16, 2))
        return obj_loss + color_loss + 0.5 * type_loss + 5.0 * points_loss + vis_loss + 0.25 * smoothness

    def _stop_line_loss(self, stop_pred: torch.Tensor, encoded: dict[str, Any]) -> torch.Tensor:
        stop_target = encoded["stop_line"].to(device=stop_pred.device, dtype=torch.float32)
        stop_source = encoded["mask"]["stop_line_source"].to(device=stop_pred.device, dtype=torch.bool)
        stop_valid = encoded["mask"]["stop_line_valid"].to(device=stop_pred.device, dtype=torch.bool)
        assignment = self._build_query_assignment(
            stop_pred,
            stop_target,
            stop_valid,
            stop_source,
            task_name="stop_line",
            cost_builder=_stop_line_cost_matrix,
        )

        source_mask = stop_source[:, None].expand_as(stop_pred[..., 0])
        obj_loss = F.binary_cross_entropy_with_logits(
            stop_pred[..., 0],
            assignment["obj_target"],
            reduction="none",
        )
        obj_loss = (obj_loss * source_mask.to(dtype=stop_pred.dtype)).sum() / source_mask.sum().clamp(min=1)

        valid = assignment["fg_mask"]
        if not bool(valid.any()):
            return obj_loss

        assigned_target = assignment["assigned_target"]
        points_pred = stop_pred[..., 1:9][valid].view(-1, 4, 2)
        points_target = assigned_target[..., 1:9][valid].view(-1, 4, 2)
        points_loss = F.l1_loss(points_pred, points_target, reduction="mean")
        straightness = _smoothness_regularizer(points_pred)
        return obj_loss + 6.0 * points_loss + 0.5 * straightness

    def _crosswalk_loss(self, cross_pred: torch.Tensor, encoded: dict[str, Any]) -> torch.Tensor:
        cross_target = encoded["crosswalk"].to(device=cross_pred.device, dtype=torch.float32)
        cross_source = encoded["mask"]["crosswalk_source"].to(device=cross_pred.device, dtype=torch.bool)
        cross_valid = encoded["mask"]["crosswalk_valid"].to(device=cross_pred.device, dtype=torch.bool)
        assignment = self._build_query_assignment(
            cross_pred,
            cross_target,
            cross_valid,
            cross_source,
            task_name="crosswalk",
            cost_builder=_crosswalk_cost_matrix,
        )

        source_mask = cross_source[:, None].expand_as(cross_pred[..., 0])
        obj_loss = F.binary_cross_entropy_with_logits(
            cross_pred[..., 0],
            assignment["obj_target"],
            reduction="none",
        )
        obj_loss = (obj_loss * source_mask.to(dtype=cross_pred.dtype)).sum() / source_mask.sum().clamp(min=1)

        valid = assignment["fg_mask"]
        if not bool(valid.any()):
            return obj_loss

        assigned_target = assignment["assigned_target"]
        points_pred = cross_pred[..., 1:17][valid].view(-1, 8, 2)
        points_target = assigned_target[..., 1:17][valid].view(-1, 8, 2)
        points_loss = F.l1_loss(points_pred, points_target, reduction="mean")
        shape = _polygon_shape_regularizer(points_pred)
        return obj_loss + 4.0 * points_loss + 0.5 * shape

    def _build_query_assignment(
        self,
        pred_rows: torch.Tensor,
        target_rows: torch.Tensor,
        valid_mask: torch.Tensor,
        source_mask: torch.Tensor,
        *,
        task_name: str,
        cost_builder,
    ) -> dict[str, torch.Tensor]:
        batch_size, query_count, vector_dim = pred_rows.shape
        assigned_target = torch.zeros((batch_size, query_count, vector_dim), device=pred_rows.device, dtype=pred_rows.dtype)
        obj_target = torch.zeros((batch_size, query_count), device=pred_rows.device, dtype=pred_rows.dtype)
        fg_mask = torch.zeros((batch_size, query_count), device=pred_rows.device, dtype=torch.bool)
        matched_target_idx = torch.full((batch_size, query_count), -1, device=pred_rows.device, dtype=torch.long)
        mode = "hungarian"

        for batch_index in range(batch_size):
            if not bool(source_mask[batch_index]):
                continue
            valid_indices = torch.nonzero(valid_mask[batch_index], as_tuple=False).flatten()
            if valid_indices.numel() == 0:
                continue
            sample_pred = pred_rows[batch_index].detach()
            sample_gt = target_rows[batch_index, valid_indices].detach()
            cost = cost_builder(sample_pred, sample_gt)
            query_idx, gt_local_idx = _hungarian_match(cost)
            if query_idx.numel() == 0:
                continue
            gt_idx = valid_indices[gt_local_idx]
            assigned_target[batch_index, query_idx] = target_rows[batch_index, gt_idx].to(dtype=pred_rows.dtype)
            obj_target[batch_index, query_idx] = 1.0
            fg_mask[batch_index, query_idx] = True
            matched_target_idx[batch_index, query_idx] = gt_idx

        self.last_lane_assignment_modes[task_name] = mode
        return {
            "assigned_target": assigned_target,
            "obj_target": obj_target,
            "fg_mask": fg_mask,
            "matched_target_idx": matched_target_idx,
        }


__all__ = ["PV26MultiTaskLoss"]
