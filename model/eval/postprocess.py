from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from ..loading.transform import (
    clip_box_xyxy,
    clip_points,
    inverse_transform_box_xyxy,
    inverse_transform_points,
    transform_from_meta,
    unique_point_count,
)
from ..loss.spec import build_loss_spec


SPEC = build_loss_spec()
OD_CLASSES = tuple(SPEC["model_contract"]["od_classes"])
TL_BITS = tuple(SPEC["model_contract"]["tl_bits"])
LANE_CLASSES = ("white_lane", "yellow_lane", "blue_lane")
LANE_TYPES = ("solid", "dotted")


@dataclass(frozen=True)
class PV26PostprocessConfig:
    det_conf_threshold: float = 0.25
    det_iou_threshold: float = 0.70
    max_detections: int = 300
    lane_obj_threshold: float = 0.50
    stop_line_obj_threshold: float = 0.50
    crosswalk_obj_threshold: float = 0.50
    lane_visibility_threshold: float = 0.50


def _nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    if boxes.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=boxes.device)
    order = scores.argsort(descending=True)
    keep: list[int] = []
    while order.numel() > 0:
        current = int(order[0].item())
        keep.append(current)
        if order.numel() == 1:
            break
        current_box = boxes[current].unsqueeze(0)
        other_indices = order[1:]
        other_boxes = boxes[other_indices]
        x1 = torch.maximum(current_box[:, 0], other_boxes[:, 0])
        y1 = torch.maximum(current_box[:, 1], other_boxes[:, 1])
        x2 = torch.minimum(current_box[:, 2], other_boxes[:, 2])
        y2 = torch.minimum(current_box[:, 3], other_boxes[:, 3])
        inter_w = (x2 - x1).clamp(min=0.0)
        inter_h = (y2 - y1).clamp(min=0.0)
        inter = inter_w * inter_h
        area_current = (current_box[:, 2] - current_box[:, 0]).clamp(min=0.0) * (
            current_box[:, 3] - current_box[:, 1]
        ).clamp(min=0.0)
        area_other = (other_boxes[:, 2] - other_boxes[:, 0]).clamp(min=0.0) * (
            other_boxes[:, 3] - other_boxes[:, 1]
        ).clamp(min=0.0)
        union = area_current + area_other - inter
        iou = inter / union.clamp(min=1e-6)
        order = other_indices[iou <= float(iou_threshold)]
    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


def _run_batched_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    class_ids: torch.Tensor,
    iou_threshold: float,
) -> torch.Tensor:
    try:
        from torchvision.ops import batched_nms as torchvision_batched_nms

        return torchvision_batched_nms(boxes, scores, class_ids, float(iou_threshold))
    except Exception:
        kept_indices: list[torch.Tensor] = []
        for class_id in class_ids.unique(sorted=True):
            class_mask = class_ids == class_id
            class_indices = torch.nonzero(class_mask, as_tuple=False).flatten()
            class_keep = _nms(boxes[class_mask], scores[class_mask], iou_threshold)
            if class_keep.numel() > 0:
                kept_indices.append(class_indices[class_keep])
        if not kept_indices:
            return torch.empty(0, dtype=torch.long, device=boxes.device)
        keep = torch.cat(kept_indices, dim=0)
        keep_scores = scores[keep]
        return keep[keep_scores.argsort(descending=True)]


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


def _decode_detection_rows(
    det_rows: torch.Tensor,
    tl_rows: torch.Tensor,
    *,
    meta: dict[str, Any],
    feature_shapes: list[tuple[int, int]],
    feature_strides: list[int],
    config: PV26PostprocessConfig,
) -> list[dict[str, Any]]:
    if len(feature_shapes) != len(feature_strides):
        raise ValueError("det feature shape/stride metadata mismatch")
    if sum(int(height) * int(width) for height, width in feature_shapes) != int(det_rows.shape[0]):
        raise ValueError("det feature metadata does not match detector query count")

    transform = transform_from_meta(meta)
    anchor_points, stride_tensor = _make_anchor_grid(
        [(int(height), int(width)) for height, width in feature_shapes],
        [int(value) for value in feature_strides],
        dtype=det_rows.dtype,
        device=det_rows.device,
    )
    boxes = _decode_anchor_relative_boxes(det_rows[:, :4].unsqueeze(0), anchor_points, stride_tensor).squeeze(0)
    obj_scores = det_rows[:, 4].sigmoid()
    cls_scores = det_rows[:, 5:].sigmoid()
    best_cls_scores, class_ids = cls_scores.max(dim=-1)
    scores = obj_scores * best_cls_scores

    net_h, net_w = transform.network_hw
    boxes[:, 0::2] = boxes[:, 0::2].clamp(0.0, net_w - 1.0)
    boxes[:, 1::2] = boxes[:, 1::2].clamp(0.0, net_h - 1.0)
    valid = (boxes[:, 2] - boxes[:, 0] > 1.0) & (boxes[:, 3] - boxes[:, 1] > 1.0)
    valid &= scores > float(config.det_conf_threshold)
    if not bool(valid.any()):
        return []

    boxes = boxes[valid]
    scores = scores[valid]
    class_ids = class_ids[valid]
    tl_scores = tl_rows.sigmoid()[valid]

    keep = _run_batched_nms(boxes, scores, class_ids, float(config.det_iou_threshold))
    keep = keep[: int(config.max_detections)]

    detections: list[dict[str, Any]] = []
    for index in keep.tolist():
        raw_box = inverse_transform_box_xyxy(boxes[index].tolist(), transform)
        if raw_box is None:
            continue
        detections.append(
            {
                "box_xyxy": [float(value) for value in raw_box],
                "score": float(scores[index].item()),
                "class_id": int(class_ids[index].item()),
                "class_name": OD_CLASSES[int(class_ids[index].item())],
                "tl_attr_scores": {
                    bit: float(tl_scores[index, bit_index].item())
                    for bit_index, bit in enumerate(TL_BITS)
                },
            }
        )
    detections.sort(key=lambda item: item["score"], reverse=True)
    return detections


def _decode_lane_rows(
    lane_rows: torch.Tensor,
    *,
    meta: dict[str, Any],
    config: PV26PostprocessConfig,
) -> list[dict[str, Any]]:
    transform = transform_from_meta(meta)
    predictions: list[dict[str, Any]] = []
    for row in lane_rows:
        score = float(row[0].sigmoid().item())
        if score <= config.lane_obj_threshold:
            continue
        visible = row[38:54].sigmoid() >= config.lane_visibility_threshold
        points = row[6:38].view(16, 2)
        active_points = points[visible] if int(visible.sum().item()) >= 2 else points
        network_points = clip_points(active_points.tolist(), transform.network_hw)
        if unique_point_count(network_points) < 2:
            continue
        raw_points = inverse_transform_points(network_points, transform)
        if unique_point_count(raw_points) < 2:
            continue
        predictions.append(
            {
                "score": score,
                "class_name": LANE_CLASSES[int(row[1:4].argmax().item())],
                "lane_type": LANE_TYPES[int(row[4:6].argmax().item())],
                "points_xy": [[float(x), float(y)] for x, y in raw_points],
            }
        )
    predictions.sort(key=lambda item: item["score"], reverse=True)
    return predictions


def _decode_polyline_rows(
    rows: torch.Tensor,
    *,
    meta: dict[str, Any],
    obj_threshold: float,
    min_unique_points: int,
    start_index: int,
    point_count: int,
) -> list[dict[str, Any]]:
    transform = transform_from_meta(meta)
    predictions: list[dict[str, Any]] = []
    end_index = start_index + point_count * 2
    for row in rows:
        score = float(row[0].sigmoid().item())
        if score <= obj_threshold:
            continue
        points = row[start_index:end_index].view(point_count, 2)
        network_points = clip_points(points.tolist(), transform.network_hw)
        if unique_point_count(network_points) < min_unique_points:
            continue
        raw_points = inverse_transform_points(network_points, transform)
        if unique_point_count(raw_points) < min_unique_points:
            continue
        predictions.append(
            {
                "score": score,
                "points_xy": [[float(x), float(y)] for x, y in raw_points],
            }
        )
    predictions.sort(key=lambda item: item["score"], reverse=True)
    return predictions


def postprocess_pv26_batch(
    predictions: dict[str, torch.Tensor | list[Any]],
    meta: list[dict[str, Any]],
    *,
    config: PV26PostprocessConfig | None = None,
) -> list[dict[str, Any]]:
    config = config or PV26PostprocessConfig()
    det_pred = predictions["det"]
    tl_attr_pred = predictions["tl_attr"]
    lane_pred = predictions["lane"]
    stop_line_pred = predictions["stop_line"]
    crosswalk_pred = predictions["crosswalk"]
    feature_shapes = predictions.get("det_feature_shapes")
    feature_strides = predictions.get("det_feature_strides")

    if not isinstance(feature_shapes, list) or not isinstance(feature_strides, list):
        raise ValueError("postprocess requires det_feature_shapes and det_feature_strides metadata")

    batch_predictions: list[dict[str, Any]] = []
    for batch_index, sample_meta in enumerate(meta):
        batch_predictions.append(
            {
                "meta": dict(sample_meta),
                "detections": _decode_detection_rows(
                    det_pred[batch_index],
                    tl_attr_pred[batch_index],
                    meta=sample_meta,
                    feature_shapes=feature_shapes,
                    feature_strides=feature_strides,
                    config=config,
                ),
                "lanes": _decode_lane_rows(
                    lane_pred[batch_index],
                    meta=sample_meta,
                    config=config,
                ),
                "stop_lines": _decode_polyline_rows(
                    stop_line_pred[batch_index],
                    meta=sample_meta,
                    obj_threshold=config.stop_line_obj_threshold,
                    min_unique_points=2,
                    start_index=1,
                    point_count=4,
                ),
                "crosswalks": _decode_polyline_rows(
                    crosswalk_pred[batch_index],
                    meta=sample_meta,
                    obj_threshold=config.crosswalk_obj_threshold,
                    min_unique_points=3,
                    start_index=1,
                    point_count=8,
                ),
            }
        )
    return batch_predictions


__all__ = [
    "PV26PostprocessConfig",
    "postprocess_pv26_batch",
]
