from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
import torch

from ..data.transform import (
    clip_box_xyxy,
    clip_points,
    inverse_transform_box_xyxy,
    inverse_transform_points,
    transform_from_meta,
    unique_point_count,
)
from .det_geometry import decode_anchor_relative_boxes, make_anchor_grid
from .spec import build_loss_spec


SPEC = build_loss_spec()
OD_CLASSES = tuple(SPEC["model_contract"]["od_classes"])
TL_BITS = tuple(SPEC["model_contract"]["tl_bits"])
LANE_CLASSES = ("white_lane", "yellow_lane", "blue_lane")
LANE_TYPES = ("solid", "dotted")
LANE_COLOR_DIM = int(SPEC["heads"]["lane"]["target_encoding"]["color_logits"])
LANE_TYPE_DIM = int(SPEC["heads"]["lane"]["target_encoding"]["type_logits"])
LANE_ANCHOR_COUNT = int(SPEC["heads"]["lane"]["target_encoding"]["anchor_rows"])
LANE_COLOR_SLICE = slice(1, 1 + LANE_COLOR_DIM)
LANE_TYPE_SLICE = slice(LANE_COLOR_SLICE.stop, LANE_COLOR_SLICE.stop + LANE_TYPE_DIM)
LANE_X_SLICE = slice(LANE_TYPE_SLICE.stop, LANE_TYPE_SLICE.stop + LANE_ANCHOR_COUNT)
LANE_VIS_SLICE = slice(LANE_X_SLICE.stop, LANE_X_SLICE.stop + LANE_ANCHOR_COUNT)
CROSSWALK_CORNER_COUNT = int(SPEC["heads"]["crosswalk"]["target_encoding"]["quad_corners"])


@dataclass(frozen=True)
class PV26PostprocessConfig:
    det_conf_threshold: float = 0.25
    det_iou_threshold: float = 0.70
    max_detections: int = 300
    lane_obj_threshold: float = 0.50
    stop_line_obj_threshold: float = 0.50
    crosswalk_obj_threshold: float = 0.50
    lane_visibility_threshold: float = 0.50
    allow_python_nms_fallback: bool = False


def _resample_polyline(points_xy: list[list[float]], target_count: int) -> torch.Tensor:
    points = torch.tensor(points_xy, dtype=torch.float32).reshape(-1, 2)
    if points.shape[0] == 0:
        return torch.zeros((target_count, 2), dtype=torch.float32)
    if points.shape[0] == 1:
        return points.repeat(target_count, 1)
    deltas = points[1:] - points[:-1]
    segment_lengths = torch.linalg.norm(deltas, dim=1)
    cumulative = torch.cat(
        [
            torch.zeros(1, dtype=torch.float32),
            torch.cumsum(segment_lengths, dim=0),
        ]
    )
    total_length = float(cumulative[-1].item())
    if total_length <= 1.0e-6:
        return points[:1].repeat(target_count, 1)
    targets = torch.linspace(0.0, total_length, target_count, dtype=torch.float32)
    resampled: list[torch.Tensor] = []
    for target in targets:
        upper = int(torch.searchsorted(cumulative, target, right=False).item())
        upper = min(max(upper, 1), points.shape[0] - 1)
        lower = upper - 1
        left_distance = cumulative[lower]
        right_distance = cumulative[upper]
        interval = float((right_distance - left_distance).item())
        if interval <= 1.0e-6:
            resampled.append(points[lower])
            continue
        ratio = float(((target - left_distance) / interval).item())
        resampled.append(points[lower] + ratio * (points[upper] - points[lower]))
    return torch.stack(resampled, dim=0)


def _mean_point_distance(points_a: list[list[float]], points_b: list[list[float]], *, target_count: int) -> float:
    resampled_a = _resample_polyline(points_a, target_count)
    resampled_b = _resample_polyline(points_b, target_count)
    return float(torch.linalg.norm(resampled_a - resampled_b, dim=1).mean().item())


def _segment_angle_error(points_a: list[list[float]], points_b: list[list[float]], *, target_count: int) -> float:
    resampled_a = _resample_polyline(points_a, target_count)
    resampled_b = _resample_polyline(points_b, target_count)
    vector_a = resampled_a[-1] - resampled_a[0]
    vector_b = resampled_b[-1] - resampled_b[0]
    norm_a = float(torch.linalg.norm(vector_a).item())
    norm_b = float(torch.linalg.norm(vector_b).item())
    if norm_a <= 1.0e-6 or norm_b <= 1.0e-6:
        return 0.0
    cosine = float(torch.clamp(torch.dot(vector_a, vector_b) / (norm_a * norm_b), min=-1.0, max=1.0).item())
    return float(torch.rad2deg(torch.arccos(torch.tensor(cosine))).item())


def _polygon_iou(points_a: list[list[float]], points_b: list[list[float]]) -> float:
    polygon_a = np.asarray(points_a, dtype=np.float32).reshape(-1, 2)
    polygon_b = np.asarray(points_b, dtype=np.float32).reshape(-1, 2)
    min_x = int(np.floor(min(float(polygon_a[:, 0].min()), float(polygon_b[:, 0].min()))))
    min_y = int(np.floor(min(float(polygon_a[:, 1].min()), float(polygon_b[:, 1].min()))))
    max_x = int(np.ceil(max(float(polygon_a[:, 0].max()), float(polygon_b[:, 0].max()))))
    max_y = int(np.ceil(max(float(polygon_a[:, 1].max()), float(polygon_b[:, 1].max()))))
    width = max(1, max_x - min_x + 3)
    height = max(1, max_y - min_y + 3)

    def rasterize(points: np.ndarray) -> np.ndarray:
        canvas = Image.new("1", (width, height), 0)
        shifted = [(float(x - min_x + 1.0), float(y - min_y + 1.0)) for x, y in points]
        ImageDraw.Draw(canvas).polygon(shifted, outline=1, fill=1)
        return np.asarray(canvas, dtype=bool)

    mask_a = rasterize(polygon_a)
    mask_b = rasterize(polygon_b)
    intersection = float(np.logical_and(mask_a, mask_b).sum())
    union = float(np.logical_or(mask_a, mask_b).sum())
    if union <= 0.0:
        return 0.0
    return intersection / union


def _lane_anchor_rows(transform: Any, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    network_h = int(transform.network_hw[0])
    return torch.linspace(float(network_h - 1), 0.0, LANE_ANCHOR_COUNT, device=device, dtype=dtype)


def _longest_true_run(mask: torch.BoolTensor) -> torch.BoolTensor:
    best_start = -1
    best_length = 0
    current_start = -1
    current_length = 0
    mask_list = mask.tolist()
    for index, value in enumerate(mask_list + [False]):
        if value:
            if current_start < 0:
                current_start = index
                current_length = 0
            current_length += 1
            continue
        if current_start >= 0 and current_length > best_length:
            best_start = current_start
            best_length = current_length
        current_start = -1
        current_length = 0
    output = torch.zeros_like(mask)
    if best_start >= 0 and best_length > 0:
        output[best_start : best_start + best_length] = True
    return output


def _dedupe_lane_predictions(predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for candidate in predictions:
        candidate_mask = candidate["_anchor_mask"]
        candidate_x = candidate["_anchor_x"]
        is_duplicate = False
        for existing in kept:
            if candidate["class_name"] != existing["class_name"] or candidate["lane_type"] != existing["lane_type"]:
                continue
            existing_mask = existing["_anchor_mask"]
            overlap_mask = candidate_mask & existing_mask
            overlap_count = int(overlap_mask.sum().item())
            if overlap_count < 2:
                continue
            min_visible = max(1, min(int(candidate_mask.sum().item()), int(existing_mask.sum().item())))
            overlap_ratio = overlap_count / float(min_visible)
            mean_x_distance = float((candidate_x[overlap_mask] - existing["_anchor_x"][overlap_mask]).abs().mean().item())
            if overlap_ratio >= 0.75 and mean_x_distance <= 16.0:
                is_duplicate = True
                break
        if not is_duplicate:
            kept.append(candidate)
    for prediction in kept:
        prediction.pop("_anchor_mask", None)
        prediction.pop("_anchor_x", None)
    return kept


def _dedupe_stop_line_predictions(predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for candidate in predictions:
        candidate_points = candidate["points_xy"]
        is_duplicate = False
        for existing in kept:
            mean_distance = _mean_point_distance(candidate_points, existing["points_xy"], target_count=4)
            angle_error = _segment_angle_error(candidate_points, existing["points_xy"], target_count=4)
            if mean_distance <= 12.0 and angle_error <= 5.0:
                is_duplicate = True
                break
        if not is_duplicate:
            kept.append(candidate)
    return kept


def _dedupe_crosswalk_predictions(predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for candidate in predictions:
        is_duplicate = False
        for existing in kept:
            if _polygon_iou(candidate["points_xy"], existing["points_xy"]) >= 0.7:
                is_duplicate = True
                break
        if not is_duplicate:
            kept.append(candidate)
    return kept


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
    *,
    allow_python_nms_fallback: bool,
) -> torch.Tensor:
    try:
        from torchvision.ops import batched_nms as torchvision_batched_nms

        return torchvision_batched_nms(boxes, scores, class_ids, float(iou_threshold))
    except Exception:
        if not allow_python_nms_fallback:
            raise
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
    anchor_points, stride_tensor = make_anchor_grid(
        [(int(height), int(width)) for height, width in feature_shapes],
        [int(value) for value in feature_strides],
        dtype=det_rows.dtype,
        device=det_rows.device,
    )
    boxes = decode_anchor_relative_boxes(det_rows[:, :4].unsqueeze(0), anchor_points, stride_tensor).squeeze(0)
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

    keep = _run_batched_nms(
        boxes,
        scores,
        class_ids,
        float(config.det_iou_threshold),
        allow_python_nms_fallback=bool(config.allow_python_nms_fallback),
    )
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
    anchor_rows = _lane_anchor_rows(transform, device=lane_rows.device, dtype=lane_rows.dtype)
    for row in lane_rows:
        score = float(row[0].sigmoid().item())
        if score <= config.lane_obj_threshold:
            continue
        visible = row[LANE_VIS_SLICE].sigmoid() >= config.lane_visibility_threshold
        visible = _longest_true_run(visible)
        if int(visible.sum().item()) < 2:
            continue
        points = torch.stack((row[LANE_X_SLICE], anchor_rows), dim=-1)
        active_points = points[visible]
        network_points = clip_points(active_points.tolist(), transform.network_hw)
        if unique_point_count(network_points) < 2:
            continue
        raw_points = inverse_transform_points(network_points, transform)
        if unique_point_count(raw_points) < 2:
            continue
        predictions.append(
            {
                "score": score,
                "class_name": LANE_CLASSES[int(row[LANE_COLOR_SLICE].argmax().item())],
                "lane_type": LANE_TYPES[int(row[LANE_TYPE_SLICE].argmax().item())],
                "points_xy": [[float(x), float(y)] for x, y in raw_points],
                "_anchor_mask": visible.detach().cpu(),
                "_anchor_x": row[LANE_X_SLICE].detach().cpu(),
            }
        )
    predictions.sort(key=lambda item: item["score"], reverse=True)
    return _dedupe_lane_predictions(predictions)


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


def _decode_stop_line_rows(
    rows: torch.Tensor,
    *,
    meta: dict[str, Any],
    obj_threshold: float,
) -> list[dict[str, Any]]:
    transform = transform_from_meta(meta)
    predictions: list[dict[str, Any]] = []
    for row in rows:
        score = float(row[0].sigmoid().item())
        if score <= obj_threshold:
            continue
        endpoints = row[1:5].view(2, 2)
        start = endpoints[0]
        end = endpoints[1]
        interpolation = torch.linspace(0.0, 1.0, 4, device=row.device, dtype=row.dtype).unsqueeze(-1)
        points = start.unsqueeze(0) + interpolation * (end - start).unsqueeze(0)
        network_points = clip_points(points.tolist(), transform.network_hw)
        if unique_point_count(network_points) < 2:
            continue
        raw_points = inverse_transform_points(network_points, transform)
        if unique_point_count(raw_points) < 2:
            continue
        predictions.append(
            {
                "score": score,
                "points_xy": [[float(x), float(y)] for x, y in raw_points],
            }
        )
    predictions.sort(key=lambda item: item["score"], reverse=True)
    return _dedupe_stop_line_predictions(predictions)


def _decode_crosswalk_rows(
    rows: torch.Tensor,
    *,
    meta: dict[str, Any],
    obj_threshold: float,
) -> list[dict[str, Any]]:
    transform = transform_from_meta(meta)
    predictions: list[dict[str, Any]] = []
    for row in rows:
        score = float(row[0].sigmoid().item())
        if score <= obj_threshold:
            continue
        points = row[1:9].view(CROSSWALK_CORNER_COUNT, 2)
        network_points = clip_points(points.tolist(), transform.network_hw)
        if unique_point_count(network_points) < 3:
            continue
        raw_points = inverse_transform_points(network_points, transform)
        if unique_point_count(raw_points) < 3:
            continue
        predictions.append(
            {
                "score": score,
                "points_xy": [[float(x), float(y)] for x, y in raw_points],
            }
        )
    predictions.sort(key=lambda item: item["score"], reverse=True)
    return _dedupe_crosswalk_predictions(predictions)


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
                "stop_lines": _decode_stop_line_rows(
                    stop_line_pred[batch_index],
                    meta=sample_meta,
                    obj_threshold=config.stop_line_obj_threshold,
                ),
                "crosswalks": _decode_crosswalk_rows(
                    crosswalk_pred[batch_index],
                    meta=sample_meta,
                    obj_threshold=config.crosswalk_obj_threshold,
                ),
            }
        )
    return batch_predictions


__all__ = [
    "PV26PostprocessConfig",
    "postprocess_pv26_batch",
]
