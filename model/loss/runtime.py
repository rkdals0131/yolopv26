from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .spec import build_loss_spec


SPEC = build_loss_spec()
TL_BITS = tuple(SPEC["model_contract"]["tl_bits"])
STAGE_LOSS_WEIGHTS = {
    stage["name"]: dict(stage["loss_weights"]) for stage in SPEC["training_schedule"]
}


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


class PV26MultiTaskLoss(nn.Module):
    def __init__(self, stage: str = "stage_0_smoke") -> None:
        super().__init__()
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

    def forward(self, predictions: dict[str, torch.Tensor], encoded: dict[str, Any]) -> dict[str, torch.Tensor]:
        det = self._det_loss(predictions["det"], encoded)
        tl_attr = self._tl_attr_loss(predictions["tl_attr"], encoded)
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

    def _det_loss(self, det_pred: torch.Tensor, encoded: dict[str, Any]) -> torch.Tensor:
        det_gt = encoded["det_gt"]
        det_source = encoded["mask"]["det_source"].to(device=det_pred.device, dtype=torch.bool)
        obj_logits = det_pred[..., 4]
        box_pred = det_pred[..., :4]
        cls_logits = det_pred[..., 5:]
        per_sample: list[torch.Tensor] = []

        for batch_index in range(det_pred.shape[0]):
            if not bool(det_source[batch_index]):
                continue
            valid = det_gt["valid_mask"][batch_index].to(device=det_pred.device, dtype=torch.bool)
            gt_boxes = det_gt["boxes_xyxy"][batch_index].to(device=det_pred.device, dtype=torch.float32)[valid]
            gt_classes = det_gt["classes"][batch_index].to(device=det_pred.device, dtype=torch.long)[valid]
            positive_count = min(int(gt_boxes.shape[0]), int(det_pred.shape[1]))
            obj_target = torch.zeros(det_pred.shape[1], device=det_pred.device, dtype=torch.float32)
            if positive_count > 0:
                obj_target[:positive_count] = 1.0
            sample_loss = F.binary_cross_entropy_with_logits(obj_logits[batch_index], obj_target, reduction="mean")
            if positive_count > 0:
                sample_loss = sample_loss + F.l1_loss(
                    box_pred[batch_index, :positive_count],
                    gt_boxes[:positive_count],
                    reduction="mean",
                )
                sample_loss = sample_loss + F.cross_entropy(
                    cls_logits[batch_index, :positive_count],
                    gt_classes[:positive_count],
                    reduction="mean",
                )
            per_sample.append(sample_loss)

        return _mean_or_zero(per_sample, det_pred)

    def _tl_attr_loss(self, tl_pred: torch.Tensor, encoded: dict[str, Any]) -> torch.Tensor:
        tl_source = encoded["mask"]["tl_attr_source"].to(device=tl_pred.device, dtype=torch.bool)
        tl_bits = encoded["tl_attr_gt_bits"].to(device=tl_pred.device, dtype=torch.float32)
        tl_mask = encoded["tl_attr_gt_mask"].to(device=tl_pred.device, dtype=torch.bool)
        per_sample: list[torch.Tensor] = []

        for batch_index in range(tl_pred.shape[0]):
            if not bool(tl_source[batch_index]):
                continue
            target_count = min(int(tl_bits.shape[1]), int(tl_pred.shape[1]))
            valid = tl_mask[batch_index, :target_count]
            if not bool(valid.any()):
                continue
            logits = tl_pred[batch_index, :target_count][valid]
            targets = tl_bits[batch_index, :target_count][valid]
            loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
            weighted = loss * self.tl_bit_weights.to(device=tl_pred.device)
            per_sample.append(weighted.mean())

        return _mean_or_zero(per_sample, tl_pred)

    def _lane_loss(self, lane_pred: torch.Tensor, encoded: dict[str, Any]) -> torch.Tensor:
        lane_target = encoded["lane"].to(device=lane_pred.device, dtype=torch.float32)
        lane_source = encoded["mask"]["lane_source"].to(device=lane_pred.device, dtype=torch.bool)
        lane_valid = encoded["mask"]["lane_valid"].to(device=lane_pred.device, dtype=torch.bool)

        source_mask = lane_source[:, None].expand_as(lane_target[..., 0])
        obj_loss = F.binary_cross_entropy_with_logits(
            lane_pred[..., 0],
            lane_target[..., 0],
            reduction="none",
        )
        obj_loss = (obj_loss * source_mask.to(dtype=lane_pred.dtype)).sum() / source_mask.sum().clamp(min=1)

        valid = lane_valid & lane_source[:, None]
        if not bool(valid.any()):
            return obj_loss

        color_target = lane_target[..., 1:4].argmax(dim=-1)
        type_target = lane_target[..., 4:6].argmax(dim=-1)
        color_loss = F.cross_entropy(lane_pred[..., 1:4][valid], color_target[valid], reduction="mean")
        type_loss = F.cross_entropy(lane_pred[..., 4:6][valid], type_target[valid], reduction="mean")
        points_loss = F.l1_loss(lane_pred[..., 6:38][valid], lane_target[..., 6:38][valid], reduction="mean")
        vis_loss = F.binary_cross_entropy_with_logits(
            lane_pred[..., 38:54][valid],
            lane_target[..., 38:54][valid],
            reduction="mean",
        )
        smoothness = _smoothness_regularizer(lane_pred[..., 6:38][valid].view(-1, 16, 2))
        return obj_loss + color_loss + 0.5 * type_loss + 5.0 * points_loss + vis_loss + 0.25 * smoothness

    def _stop_line_loss(self, stop_pred: torch.Tensor, encoded: dict[str, Any]) -> torch.Tensor:
        stop_target = encoded["stop_line"].to(device=stop_pred.device, dtype=torch.float32)
        stop_source = encoded["mask"]["stop_line_source"].to(device=stop_pred.device, dtype=torch.bool)
        stop_valid = encoded["mask"]["stop_line_valid"].to(device=stop_pred.device, dtype=torch.bool)

        source_mask = stop_source[:, None].expand_as(stop_target[..., 0])
        obj_loss = F.binary_cross_entropy_with_logits(
            stop_pred[..., 0],
            stop_target[..., 0],
            reduction="none",
        )
        obj_loss = (obj_loss * source_mask.to(dtype=stop_pred.dtype)).sum() / source_mask.sum().clamp(min=1)

        valid = stop_valid & stop_source[:, None]
        if not bool(valid.any()):
            return obj_loss

        points_pred = stop_pred[..., 1:9][valid].view(-1, 4, 2)
        points_target = stop_target[..., 1:9][valid].view(-1, 4, 2)
        points_loss = F.l1_loss(points_pred, points_target, reduction="mean")
        straightness = _smoothness_regularizer(points_pred)
        return obj_loss + 6.0 * points_loss + 0.5 * straightness

    def _crosswalk_loss(self, cross_pred: torch.Tensor, encoded: dict[str, Any]) -> torch.Tensor:
        cross_target = encoded["crosswalk"].to(device=cross_pred.device, dtype=torch.float32)
        cross_source = encoded["mask"]["crosswalk_source"].to(device=cross_pred.device, dtype=torch.bool)
        cross_valid = encoded["mask"]["crosswalk_valid"].to(device=cross_pred.device, dtype=torch.bool)

        source_mask = cross_source[:, None].expand_as(cross_target[..., 0])
        obj_loss = F.binary_cross_entropy_with_logits(
            cross_pred[..., 0],
            cross_target[..., 0],
            reduction="none",
        )
        obj_loss = (obj_loss * source_mask.to(dtype=cross_pred.dtype)).sum() / source_mask.sum().clamp(min=1)

        valid = cross_valid & cross_source[:, None]
        if not bool(valid.any()):
            return obj_loss

        points_pred = cross_pred[..., 1:17][valid].view(-1, 8, 2)
        points_target = cross_target[..., 1:17][valid].view(-1, 8, 2)
        points_loss = F.l1_loss(points_pred, points_target, reduction="mean")
        shape = _polygon_shape_regularizer(points_pred)
        return obj_loss + 4.0 * points_loss + 0.5 * shape


__all__ = ["PV26MultiTaskLoss"]
