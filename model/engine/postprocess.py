from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
import torch

from common.geometry import (
    canonicalize_crosswalk_points,
    sample_crosswalk_contour,
    sample_stop_line_centerline,
)
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
STOP_LINE_POINT_COUNT = int(SPEC["heads"]["stop_line"]["target_encoding"]["polyline_points"])
CROSSWALK_POINT_COUNT = int(SPEC["heads"]["crosswalk"]["target_encoding"]["sequence_points"])
STOPLINE_MIN_COMPONENT_PIXELS = 4
STOPLINE_MIN_COMPONENT_LENGTH = 3.0
STOPLINE_BINARY_DILATION_ITERATIONS = 1
STOPLINE_MIN_ASPECT_RATIO = 1.25
STOPLINE_HORIZONTAL_BRIDGE_WIDTH = 9
STOPLINE_HORIZONTAL_BRIDGE_HEIGHT = 3
STOPLINE_CENTER_ROW_TOLERANCE = 4
STOPLINE_CENTER_ANCHOR_BAND = 1
STOPLINE_BACKUP_LENGTH_FLOOR_RATIO = 0.28
STOPLINE_BACKUP_MAX_SHORT_LENGTH = 12.0
STOPLINE_BACKUP_MAX_THICKNESS = 4.5
STOPLINE_BACKUP_MAX_CENTER_SCORE = 0.05


@dataclass(frozen=True)
class PV26PostprocessConfig:
    det_conf_threshold: float = 0.25
    det_iou_threshold: float = 0.70
    max_detections: int = 300
    lane_obj_threshold: float = 0.50
    lane_segfirst_min_polyline_length_px: float = 0.0
    lane_segfirst_min_polyline_bottom_y_fraction: float = 0.0
    lane_segfirst_semantic_vote_mode: str = "component"
    stop_line_obj_threshold: float = 0.50
    stop_line_mask_binary_threshold: float = 0.50
    crosswalk_obj_threshold: float = 0.50
    crosswalk_mask_binary_threshold: float = 0.50
    lane_visibility_threshold: float = 0.50
    allow_python_nms_fallback: bool = False


def _tensor_all_finite(value: torch.Tensor) -> bool:
    return bool(torch.isfinite(value).all())


def _points_all_finite(points_xy: list[list[float]] | np.ndarray) -> bool:
    points = np.asarray(points_xy, dtype=np.float32)
    return bool(np.isfinite(points).all())


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
    if not _points_all_finite(points_a) or not _points_all_finite(points_b):
        return float("inf")
    resampled_a = _resample_polyline(points_a, target_count)
    resampled_b = _resample_polyline(points_b, target_count)
    return float(torch.linalg.norm(resampled_a - resampled_b, dim=1).mean().item())


def _segment_angle_error(points_a: list[list[float]], points_b: list[list[float]], *, target_count: int) -> float:
    if not _points_all_finite(points_a) or not _points_all_finite(points_b):
        return 180.0
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
    if polygon_a.shape[0] < 3 or polygon_b.shape[0] < 3:
        return 0.0
    if not bool(np.isfinite(polygon_a).all()) or not bool(np.isfinite(polygon_b).all()):
        return 0.0
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


def _visibility_envelope(mask: torch.BoolTensor) -> torch.BoolTensor:
    output = torch.zeros_like(mask)
    indices = torch.nonzero(mask, as_tuple=False).flatten()
    if indices.numel() == 0:
        return output
    start = int(indices[0].item())
    end = int(indices[-1].item())
    output[start : end + 1] = True
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
            if overlap_ratio >= 0.5 and mean_x_distance <= 24.0:
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
            mean_distance = _mean_point_distance(candidate_points, existing["points_xy"], target_count=STOP_LINE_POINT_COUNT)
            angle_error = _segment_angle_error(candidate_points, existing["points_xy"], target_count=STOP_LINE_POINT_COUNT)
            if mean_distance <= 12.0 and angle_error <= 5.0:
                is_duplicate = True
                break
        if not is_duplicate:
            kept.append(candidate)
    return kept


def _suppress_stop_line_fragments(
    predictions: list[dict[str, Any]],
    *,
    min_length_ratio: float = 0.35,
) -> list[dict[str, Any]]:
    if len(predictions) <= 1:
        return predictions
    best_length = max(float(item.get("length", 0.0)) for item in predictions)
    if best_length <= 0.0:
        return predictions
    min_length = max(4.0, float(min_length_ratio) * best_length)
    kept: list[dict[str, Any]] = []
    for index, item in enumerate(predictions):
        if index == 0:
            kept.append(item)
            continue
        if float(item.get("length", 0.0)) < min_length:
            continue
        kept.append(item)
    return kept


def _promote_stop_line_structured_fallback(predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(predictions) <= 1:
        return predictions
    top = predictions[0]
    top_orientation = float(top.get("orientation_score", 0.0))
    top_length = float(top.get("length", 0.0))
    top_center_score = float(top.get("center_score", 0.0))
    if not bool(top.get("allowed", False)):
        return predictions
    if top_orientation >= 0.90 or top_length >= 8.0 or top_center_score >= 0.25:
        return predictions
    for index, candidate in enumerate(predictions[1:], start=1):
        candidate_orientation = float(candidate.get("orientation_score", 0.0))
        candidate_length = float(candidate.get("length", 0.0))
        candidate_score = float(candidate.get("score", 0.0))
        if candidate_orientation < 0.95:
            continue
        if candidate_length < max(6.0, 2.0 * max(top_length, 1.0)):
            continue
        if candidate_score < 0.5 * float(top.get("score", 0.0)):
            continue
        return [candidate, *predictions[:index], *predictions[index + 1 :]]
    return predictions


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


def _crosswalk_mask_to_polygon(
    mask_logits: torch.Tensor,
    center_logits: torch.Tensor | None,
    *,
    meta: dict[str, Any],
    obj_threshold: float,
    mask_binary_threshold: float,
) -> list[dict[str, Any]]:
    if not _tensor_all_finite(mask_logits):
        return []
    mask_probs = mask_logits.sigmoid().squeeze(0).detach().cpu().numpy()
    if mask_probs.ndim == 3:
        mask_probs = mask_probs.squeeze(0)
    if mask_probs.ndim == 3:
        mask_probs = mask_probs.squeeze(0)
    if mask_probs.ndim == 3:
        mask_probs = mask_probs.squeeze(0)
    score = float(mask_probs.max())
    if score <= obj_threshold:
        return []
    binary = mask_probs >= float(mask_binary_threshold)
    if not bool(binary.any()):
        return []

    labels, component_count = ndimage.label(binary)
    if component_count <= 0:
        return []
    component_scores = ndimage.maximum(mask_probs, labels, index=np.arange(1, component_count + 1))
    center_probs = None
    if isinstance(center_logits, torch.Tensor) and _tensor_all_finite(center_logits):
        center_probs = center_logits.sigmoid().squeeze(0).detach().cpu().numpy()

    predictions: list[dict[str, Any]] = []
    output_h, output_w = mask_probs.shape
    transform = transform_from_meta(meta)
    for label_index in range(1, component_count + 1):
        rows, cols = np.nonzero(labels == label_index)
        if len(rows) < 4:
            continue

        points = np.stack([cols.astype(np.float32), rows.astype(np.float32)], axis=1)
        center = points.mean(axis=0, keepdims=True)
        centered = points - center
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        axis1, axis2 = vh[0], vh[1]
        proj1 = centered @ axis1
        proj2 = centered @ axis2
        min1, max1 = float(proj1.min()), float(proj1.max())
        min2, max2 = float(proj2.min()), float(proj2.max())
        rect = np.array(
            [
                center[0] + axis1 * min1 + axis2 * min2,
                center[0] + axis1 * max1 + axis2 * min2,
                center[0] + axis1 * max1 + axis2 * max2,
                center[0] + axis1 * min1 + axis2 * max2,
            ],
            dtype=np.float32,
        )

        rect[:, 0] = (rect[:, 0] + 0.5) * (float(meta["network_hw"][1]) / float(output_w))
        rect[:, 1] = (rect[:, 1] + 0.5) * (float(meta["network_hw"][0]) / float(output_h))
        network_points = canonicalize_crosswalk_points(
            sample_crosswalk_contour(rect, target_count=CROSSWALK_POINT_COUNT)
        ).tolist()
        if unique_point_count(network_points) < 3:
            continue
        raw_points = canonicalize_crosswalk_points(inverse_transform_points(network_points, transform)).tolist()
        if unique_point_count(raw_points) < 3:
            continue

        instance_score = float(component_scores[label_index - 1])
        if center_probs is not None:
            center_score = float(center_probs[rows, cols].max())
            instance_score = 0.5 * instance_score + 0.5 * center_score
        if instance_score <= obj_threshold:
            continue
        predictions.append(
            {
                "score": instance_score,
                "points_xy": [[float(x), float(y)] for x, y in raw_points],
            }
        )
    predictions.sort(key=lambda item: item["score"], reverse=True)
    return _dedupe_crosswalk_predictions(predictions)


def _prepare_stopline_binary_mask(mask_probs: np.ndarray, *, threshold: float) -> np.ndarray:
    binary = mask_probs >= float(threshold)
    if not bool(binary.any()):
        return binary
    if STOPLINE_BINARY_DILATION_ITERATIONS > 0:
        binary = ndimage.binary_dilation(
            binary,
            structure=np.ones((3, 3), dtype=bool),
            iterations=STOPLINE_BINARY_DILATION_ITERATIONS,
        )
    binary = ndimage.binary_closing(
        binary,
        structure=np.ones((STOPLINE_HORIZONTAL_BRIDGE_HEIGHT, STOPLINE_HORIZONTAL_BRIDGE_WIDTH), dtype=bool),
        iterations=1,
    )
    return binary


def _fit_stopline_segment(
    component_points: np.ndarray,
    *,
    mask_values: np.ndarray | None = None,
    center_anchor: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, float, float] | None:
    fit_points = component_points
    anchor_point = center_anchor.astype(np.float32) if center_anchor is not None else None

    if anchor_point is not None and fit_points.shape[0] >= STOPLINE_MIN_COMPONENT_PIXELS:
        initial_center = fit_points.mean(axis=0, keepdims=True)
        initial_centered = fit_points - initial_center
        try:
            _, _, initial_vh = np.linalg.svd(initial_centered, full_matrices=False)
        except np.linalg.LinAlgError:
            initial_vh = np.eye(2, dtype=np.float32)
        initial_axis = initial_vh[0]
        if abs(float(initial_axis[0])) < abs(float(initial_axis[1])):
            initial_axis = initial_vh[1]
        initial_axis = initial_axis / max(float(np.linalg.norm(initial_axis)), 1.0e-6)
        initial_normal = np.array([-initial_axis[1], initial_axis[0]], dtype=np.float32)
        anchor_offsets = fit_points - anchor_point[None, :]
        normal_distances = np.abs(anchor_offsets @ initial_normal)
        if mask_values is not None and mask_values.shape[0] == fit_points.shape[0]:
            support_floor = float(np.quantile(mask_values, 0.6))
            support_mask = mask_values >= max(0.35, support_floor)
        else:
            support_mask = np.ones((fit_points.shape[0],), dtype=bool)
        local_mask = normal_distances <= 2.5
        refined_mask = support_mask & local_mask
        if int(refined_mask.sum()) >= STOPLINE_MIN_COMPONENT_PIXELS:
            refined_points = fit_points[refined_mask]
            refined_center = refined_points.mean(axis=0, keepdims=True)
            refined_projection = (refined_points - refined_center) @ initial_axis
            refined_length = float(refined_projection.max() - refined_projection.min()) if refined_projection.size > 0 else 0.0
            full_projection = (fit_points - initial_center) @ initial_axis
            full_length = float(full_projection.max() - full_projection.min()) if full_projection.size > 0 else 0.0
            min_adopt_length = max(STOPLINE_MIN_COMPONENT_LENGTH, 0.6 * full_length)
            if refined_length >= min_adopt_length:
                fit_points = refined_points

    if fit_points.shape[0] < STOPLINE_MIN_COMPONENT_PIXELS:
        return None

    center = fit_points.mean(axis=0, keepdims=True)
    centered = fit_points - center
    try:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    axis = vh[0]
    if abs(float(axis[0])) < abs(float(axis[1])):
        axis = vh[1]
    axis = axis / max(float(np.linalg.norm(axis)), 1.0e-6)
    normal = np.array([-axis[1], axis[0]], dtype=np.float32)
    endpoint_anchor = anchor_point[None, :] if anchor_point is not None else center
    anchored = fit_points - endpoint_anchor
    projection = anchored @ axis
    thickness_projection = anchored @ normal
    if projection.size == 0 or thickness_projection.size == 0:
        return None
    lower_q, upper_q = (0.05, 0.95) if projection.shape[0] >= 8 else (0.0, 1.0)
    start_proj = float(np.quantile(projection, lower_q))
    end_proj = float(np.quantile(projection, upper_q))
    full_start_proj = float(projection.min())
    full_end_proj = float(projection.max())
    full_length = float(full_end_proj - full_start_proj)
    if anchor_point is not None and full_length >= STOPLINE_MIN_COMPONENT_LENGTH:
        quantile_length = float(end_proj - start_proj)
        if quantile_length < 0.6 * full_length:
            start_proj = full_start_proj
            end_proj = full_end_proj
    if anchor_point is not None:
        start_proj = min(start_proj, 0.0)
        end_proj = max(end_proj, 0.0)
    length = float(end_proj - start_proj)
    thickness = float(np.quantile(thickness_projection, 0.95) - np.quantile(thickness_projection, 0.05))
    start = endpoint_anchor[0] + axis * start_proj
    end = endpoint_anchor[0] + axis * end_proj
    return start.astype(np.float32), end.astype(np.float32), length, thickness


def _decode_stopline_direct_selector_segment(
    *,
    mask_logits: torch.Tensor | None,
    selector_map_logits: torch.Tensor | None,
    row_logits: torch.Tensor | None,
    x_logits: torch.Tensor | None,
    center_logits: torch.Tensor | None,
    center_offset: torch.Tensor | None,
    angle: torch.Tensor | None,
    half_length: torch.Tensor | None,
    meta: dict[str, Any],
    obj_threshold: float,
) -> list[dict[str, Any]]:
    if not all(
        isinstance(value, torch.Tensor)
        for value in (row_logits, x_logits, center_offset, angle, half_length)
    ):
        return []
    if not all(_tensor_all_finite(value) for value in (row_logits, x_logits, center_offset, angle, half_length)):
        return []

    row_probs = row_logits.sigmoid().detach().cpu()
    x_probs = x_logits.sigmoid().detach().cpu()
    if row_probs.ndim == 4:
        row_probs = row_probs.squeeze(0)
    if row_probs.ndim == 3:
        row_probs = row_probs.squeeze(0)
    if row_probs.ndim == 2:
        row_probs = row_probs.max(dim=-1).values
    if x_probs.ndim == 4:
        x_probs = x_probs.squeeze(0)
    if x_probs.ndim == 3:
        x_probs = x_probs.squeeze(0)
    if x_probs.ndim == 2:
        x_probs = x_probs.max(dim=0).values
    if row_probs.ndim != 1 or x_probs.ndim != 1 or int(row_probs.numel()) == 0 or int(x_probs.numel()) == 0:
        return []

    selector_score_component = 0.0
    selector_map = None
    if isinstance(selector_map_logits, torch.Tensor) and _tensor_all_finite(selector_map_logits):
        selector_map = selector_map_logits.sigmoid().detach().cpu()
        if selector_map.ndim == 4:
            selector_map = selector_map.squeeze(0)
        if selector_map.ndim == 3:
            selector_map = selector_map.squeeze(0)
    if isinstance(selector_map, torch.Tensor) and selector_map.ndim == 2:
        flat_index = int(torch.argmax(selector_map).item())
        row_index, col_index = divmod(flat_index, int(selector_map.shape[1]))
        row_score = float(row_probs[min(row_index, int(row_probs.numel()) - 1)].item())
        col_score = float(x_probs[min(col_index, int(x_probs.numel()) - 1)].item())
        selector_score_component = float(selector_map[row_index, col_index].item())
    else:
        row_index = int(torch.argmax(row_probs).item())
        col_index = int(torch.argmax(x_probs).item())
        row_score = float(row_probs[row_index].item())
        col_score = float(x_probs[col_index].item())
    mask_score = 0.0
    if isinstance(mask_logits, torch.Tensor) and _tensor_all_finite(mask_logits):
        mask_score = float(mask_logits.sigmoid().max().item())
    center_score = 0.0
    if isinstance(center_logits, torch.Tensor) and _tensor_all_finite(center_logits):
        center_map = center_logits.sigmoid().detach().cpu()
        if center_map.ndim == 4:
            center_map = center_map.squeeze(0)
        if center_map.ndim == 3:
            center_map = center_map.squeeze(0)
        center_score = float(center_map[row_index, col_index].item()) if center_map.ndim == 2 else float(center_map.max().item())

    selector_score = max(mask_score, 0.0) * 0.3 + row_score * 0.15 + col_score * 0.15 + center_score * 0.1 + selector_score_component * 0.3
    if selector_score <= obj_threshold:
        return []

    offset_map = center_offset.detach().cpu()
    angle_map = angle.detach().cpu()
    length_map = half_length.detach().cpu()
    if offset_map.ndim == 4:
        offset_map = offset_map.squeeze(0)
    if angle_map.ndim == 4:
        angle_map = angle_map.squeeze(0)
    if length_map.ndim == 4:
        length_map = length_map.squeeze(0)
    if offset_map.ndim != 3 or angle_map.ndim != 3 or length_map.ndim != 3:
        return []

    center_xy = np.array(
        [
            float(col_index) + float(offset_map[0, row_index, col_index].item()),
            float(row_index) + float(offset_map[1, row_index, col_index].item()),
        ],
        dtype=np.float32,
    )
    angle_vec = angle_map[:, row_index, col_index].numpy().astype(np.float32)
    angle_norm = float(np.linalg.norm(angle_vec))
    if angle_norm <= 1.0e-6:
        return []
    angle_vec = angle_vec / angle_norm
    half_len = float(length_map[0, row_index, col_index].item())
    if not np.isfinite(half_len) or half_len <= 0.5:
        return []

    start = center_xy - angle_vec * half_len
    end = center_xy + angle_vec * half_len
    output_h = int(angle_map.shape[1])
    output_w = int(angle_map.shape[2])
    network_segment = np.stack([start, end], axis=0).astype(np.float32)
    network_segment[:, 0] = (network_segment[:, 0] + 0.5) * (float(meta["network_hw"][1]) / float(output_w))
    network_segment[:, 1] = (network_segment[:, 1] + 0.5) * (float(meta["network_hw"][0]) / float(output_h))
    transform = transform_from_meta(meta)
    network_points = sample_stop_line_centerline(network_segment.tolist(), target_count=STOP_LINE_POINT_COUNT).tolist()
    if unique_point_count(network_points) < 2:
        return []
    raw_points = sample_stop_line_centerline(inverse_transform_points(network_points, transform), target_count=STOP_LINE_POINT_COUNT).tolist()
    if unique_point_count(raw_points) < 2:
        return []
    return [
        {
            "allowed": True,
            "score": float(selector_score),
            "center_score": float(center_score),
            "orientation_score": float(_stopline_orientation_score(raw_points)),
            "length": float(half_len * 2.0),
            "thickness": 1.0,
            "points_xy": [[float(x), float(y)] for x, y in raw_points],
        }
    ]


def _decode_stopline_direct_center_segment(
    *,
    mask_logits: torch.Tensor | None,
    center_logits: torch.Tensor | None,
    center_offset: torch.Tensor | None,
    angle: torch.Tensor | None,
    half_length: torch.Tensor | None,
    meta: dict[str, Any],
    obj_threshold: float,
) -> list[dict[str, Any]]:
    if not all(
        isinstance(value, torch.Tensor)
        for value in (center_logits, center_offset, angle, half_length)
    ):
        return []
    if not all(_tensor_all_finite(value) for value in (center_logits, center_offset, angle, half_length)):
        return []

    center_map = center_logits.sigmoid().detach().cpu()
    if center_map.ndim == 4:
        center_map = center_map.squeeze(0)
    if center_map.ndim == 3:
        center_map = center_map.squeeze(0)
    if center_map.ndim != 2:
        return []

    flat_index = int(torch.argmax(center_map).item())
    row_index, col_index = divmod(flat_index, int(center_map.shape[1]))
    center_score = float(center_map[row_index, col_index].item())
    mask_score = 0.0
    if isinstance(mask_logits, torch.Tensor) and _tensor_all_finite(mask_logits):
        mask_score = float(mask_logits.sigmoid().max().item())
    direct_score = 0.5 * center_score + 0.5 * max(mask_score, 0.0)
    if direct_score <= obj_threshold:
        return []

    offset_map = center_offset.detach().cpu()
    angle_map = angle.detach().cpu()
    length_map = half_length.detach().cpu()
    if offset_map.ndim == 4:
        offset_map = offset_map.squeeze(0)
    if angle_map.ndim == 4:
        angle_map = angle_map.squeeze(0)
    if length_map.ndim == 4:
        length_map = length_map.squeeze(0)
    if offset_map.ndim != 3 or angle_map.ndim != 3 or length_map.ndim != 3:
        return []

    center_xy = np.array(
        [
            float(col_index) + float(offset_map[0, row_index, col_index].item()),
            float(row_index) + float(offset_map[1, row_index, col_index].item()),
        ],
        dtype=np.float32,
    )
    angle_vec = angle_map[:, row_index, col_index].numpy().astype(np.float32)
    angle_norm = float(np.linalg.norm(angle_vec))
    if angle_norm <= 1.0e-6:
        return []
    angle_vec = angle_vec / angle_norm
    half_len = float(length_map[0, row_index, col_index].item())
    if not np.isfinite(half_len) or half_len <= 0.5:
        return []

    start = center_xy - angle_vec * half_len
    end = center_xy + angle_vec * half_len
    output_h = int(angle_map.shape[1])
    output_w = int(angle_map.shape[2])
    network_segment = np.stack([start, end], axis=0).astype(np.float32)
    network_segment[:, 0] = (network_segment[:, 0] + 0.5) * (float(meta["network_hw"][1]) / float(output_w))
    network_segment[:, 1] = (network_segment[:, 1] + 0.5) * (float(meta["network_hw"][0]) / float(output_h))
    transform = transform_from_meta(meta)
    network_points = sample_stop_line_centerline(network_segment.tolist(), target_count=STOP_LINE_POINT_COUNT).tolist()
    if unique_point_count(network_points) < 2:
        return []
    raw_points = sample_stop_line_centerline(inverse_transform_points(network_points, transform), target_count=STOP_LINE_POINT_COUNT).tolist()
    if unique_point_count(raw_points) < 2:
        return []
    return [
        {
            "allowed": True,
            "score": float(direct_score),
            "center_score": float(center_score),
            "orientation_score": float(_stopline_orientation_score(raw_points)),
            "length": float(half_len * 2.0),
            "thickness": 1.0,
            "points_xy": [[float(x), float(y)] for x, y in raw_points],
        }
    ]


def _stopline_orientation_score(points_xy: list[list[float]] | np.ndarray) -> float:
    points = np.asarray(points_xy, dtype=np.float32)
    if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] != 2:
        return 0.0
    delta = points[-1] - points[0]
    norm = float(np.linalg.norm(delta))
    if not np.isfinite(norm) or norm <= 1.0e-6:
        return 0.0
    return float(min(1.0, abs(float(delta[0])) / norm))


def _promote_stop_line_endpoint_floor_backup(
    predictions: list[dict[str, Any]],
    *,
    meta: dict[str, Any],
) -> list[dict[str, Any]]:
    if not predictions:
        return predictions
    raw_hw = meta.get("raw_hw") or ()
    if len(raw_hw) != 2:
        return predictions
    raw_h = float(raw_hw[0])
    raw_w = float(raw_hw[1])
    target_half_len = STOPLINE_BACKUP_LENGTH_FLOOR_RATIO * raw_w
    if target_half_len <= 0.0:
        return predictions

    injected: list[dict[str, Any]] = []
    for item in predictions:
        points = np.asarray(item.get("points_xy", []), dtype=np.float32)
        if points.ndim != 2 or points.shape[0] < 2:
            continue
        orientation = float(item.get("orientation_score", 0.0))
        length = float(item.get("length", 0.0))
        thickness = float(item.get("thickness", 0.0))
        center_score = float(item.get("center_score", 0.0))
        if orientation < 0.995:
            continue
        if length <= 0.0 or length > STOPLINE_BACKUP_MAX_SHORT_LENGTH:
            continue
        if thickness > STOPLINE_BACKUP_MAX_THICKNESS:
            continue
        if bool(item.get("allowed", False)) and center_score > STOPLINE_BACKUP_MAX_CENTER_SCORE:
            continue

        midpoint = points.mean(axis=0)
        direction = points[-1] - points[0]
        direction_norm = float(np.linalg.norm(direction))
        if direction_norm <= 1.0e-6:
            continue
        direction = direction / direction_norm
        current_half_len = 0.5 * direction_norm
        if target_half_len <= current_half_len + 1.0:
            continue
        start = midpoint - direction * target_half_len
        end = midpoint + direction * target_half_len
        start[0] = np.clip(start[0], 0.0, raw_w - 1.0)
        start[1] = np.clip(start[1], 0.0, raw_h - 1.0)
        end[0] = np.clip(end[0], 0.0, raw_w - 1.0)
        end[1] = np.clip(end[1], 0.0, raw_h - 1.0)
        raw_points = sample_stop_line_centerline(
            [[float(start[0]), float(start[1])], [float(end[0]), float(end[1])]],
            target_count=STOP_LINE_POINT_COUNT,
        ).tolist()
        injected.append(
            {
                **item,
                "score": float(item.get("score", 0.0)),
                "length": float(np.linalg.norm(end - start)),
                "points_xy": [[float(x), float(y)] for x, y in raw_points],
            }
        )
        break
    if not injected:
        return predictions
    return [*predictions, *injected]


def _stopline_prediction_sort_key(item: dict[str, Any]) -> tuple[float, ...]:
    return (
        1.0 if item.get("allowed") else 0.0,
        float(item.get("orientation_score", 0.0)),
        float(item.get("center_score", 0.0)),
        float(item.get("score", 0.0)),
        float(item.get("length", 0.0)),
        -float(item.get("thickness", 0.0)),
    )


def _stopline_component_anchor(
    rows: np.ndarray,
    cols: np.ndarray,
    *,
    mask_values: np.ndarray | None,
    center_probs: np.ndarray | None,
    center_offset: torch.Tensor | None,
    row_probs: np.ndarray | None,
    x_probs: np.ndarray | None,
) -> np.ndarray:
    component_points = np.stack([cols.astype(np.float32), rows.astype(np.float32)], axis=1)
    if rows.size == 0:
        return component_points.mean(axis=0, dtype=np.float32)

    row_scores: dict[int, float] = {}
    for row_value in np.unique(rows):
        row_index = int(row_value)
        if row_probs is not None and 0 <= row_index < int(row_probs.shape[0]):
            row_scores[row_index] = float(row_probs[row_index])
            continue
        if center_probs is not None:
            row_mask = rows == row_value
            row_scores[row_index] = float(center_probs[rows[row_mask], cols[row_mask]].max())
    if not row_scores:
        return component_points.mean(axis=0, dtype=np.float32)
    best_row = max(row_scores.items(), key=lambda item: item[1])[0]

    band_mask = np.abs(rows - best_row) <= STOPLINE_CENTER_ANCHOR_BAND
    if not bool(band_mask.any()):
        band_mask = rows == best_row
    if not bool(band_mask.any()):
        band_mask = np.ones_like(rows, dtype=bool)

    band_cols = cols[band_mask].astype(np.float32)
    if x_probs is not None and band_cols.size > 0:
        x_indices = cols[band_mask].astype(np.int64)
        band_weights = x_probs[x_indices].astype(np.float32)
        if mask_values is not None and mask_values.shape[0] == rows.shape[0]:
            band_weights = band_weights * mask_values[band_mask].astype(np.float32)
        if float(band_weights.sum()) > 1.0e-6:
            anchor_x = float(np.average(band_cols, weights=band_weights))
        else:
            anchor_x = float(np.median(band_cols))
    elif mask_values is not None and mask_values.shape[0] == rows.shape[0]:
        band_weights = mask_values[band_mask].astype(np.float32)
        if float(band_weights.sum()) > 1.0e-6:
            anchor_x = float(np.average(band_cols, weights=band_weights))
        else:
            anchor_x = float(np.median(band_cols))
    else:
        anchor_x = float(np.median(band_cols))
    anchor_y = float(best_row)

    if isinstance(center_offset, torch.Tensor) and _tensor_all_finite(center_offset):
        offset_map = center_offset.detach().cpu().numpy()
        if offset_map.ndim == 4:
            offset_map = offset_map.squeeze(0)
        if offset_map.ndim == 3 and offset_map.shape[0] >= 2:
            band_rows = rows[band_mask]
            band_cols_int = cols[band_mask]
            local_index = int(np.argmax(center_probs[band_rows, band_cols_int]))
            offset_x = float(offset_map[0, band_rows[local_index], band_cols_int[local_index]])
            offset_y = float(offset_map[1, band_rows[local_index], band_cols_int[local_index]])
            anchor_x += offset_x - 0.5
            anchor_y += offset_y - 0.5

    return np.array([anchor_x, anchor_y], dtype=np.float32)


def _stopline_mask_to_polyline(
    mask_logits: torch.Tensor,
    selector_map_logits: torch.Tensor | None,
    center_logits: torch.Tensor | None,
    center_offset: torch.Tensor | None,
    row_logits: torch.Tensor | None,
    x_logits: torch.Tensor | None,
    angle: torch.Tensor | None,
    half_length: torch.Tensor | None,
    *,
    meta: dict[str, Any],
    obj_threshold: float,
    mask_binary_threshold: float,
) -> list[dict[str, Any]]:
    if not _tensor_all_finite(mask_logits):
        return []
    mask_probs = mask_logits.sigmoid().squeeze(0).detach().cpu().numpy()
    if mask_probs.ndim == 3:
        mask_probs = mask_probs.squeeze(0)
    score = float(mask_probs.max())
    if score <= obj_threshold:
        return []
    binary = _prepare_stopline_binary_mask(mask_probs, threshold=mask_binary_threshold)
    if not bool(binary.any()):
        return []

    labels, component_count = ndimage.label(binary)
    if component_count <= 0:
        return []

    component_scores = ndimage.maximum(mask_probs, labels, index=np.arange(1, component_count + 1))
    center_probs = None
    row_probs = None
    x_probs = None
    if isinstance(center_logits, torch.Tensor) and _tensor_all_finite(center_logits):
        center_probs = center_logits.sigmoid().squeeze(0).detach().cpu().numpy()
        if center_probs.ndim == 3:
            center_probs = center_probs.squeeze(0)
    if isinstance(row_logits, torch.Tensor) and _tensor_all_finite(row_logits):
        row_probs = row_logits.sigmoid().detach().cpu().numpy()
        if row_probs.ndim == 4:
            row_probs = row_probs.squeeze(0)
        if row_probs.ndim == 3:
            row_probs = row_probs.squeeze(0)
        if row_probs.ndim == 2:
            row_probs = row_probs.max(axis=-1)
    if isinstance(x_logits, torch.Tensor) and _tensor_all_finite(x_logits):
        x_probs = x_logits.sigmoid().detach().cpu().numpy()
        if x_probs.ndim == 4:
            x_probs = x_probs.squeeze(0)
        if x_probs.ndim == 3:
            x_probs = x_probs.squeeze(0)
        if x_probs.ndim == 2:
            x_probs = x_probs.max(axis=0)

    allowed_labels = _stopline_allowed_labels(labels, center_probs, row_probs=row_probs)

    predictions: list[dict[str, Any]] = _decode_stopline_direct_center_segment(
        mask_logits=mask_logits,
        center_logits=center_logits,
        center_offset=center_offset,
        angle=angle,
        half_length=half_length,
        meta=meta,
        obj_threshold=obj_threshold,
    )
    for label_index in range(1, component_count + 1):
        rows, cols = np.nonzero(labels == label_index)
        if len(rows) < STOPLINE_MIN_COMPONENT_PIXELS:
            continue
        component_points = np.stack([cols.astype(np.float32), rows.astype(np.float32)], axis=1)
        mask_values = mask_probs[rows, cols]
        center_anchor = None
        if center_probs is not None:
            center_anchor = _stopline_component_anchor(
                rows,
                cols,
                mask_values=mask_values,
                center_probs=center_probs,
                center_offset=center_offset,
                row_probs=row_probs,
                x_probs=x_probs,
            )
        fitted = _fit_stopline_segment(component_points, mask_values=mask_values, center_anchor=center_anchor)
        if fitted is None:
            continue
        start, end, length, thickness = fitted
        if length < STOPLINE_MIN_COMPONENT_LENGTH:
            continue
        if thickness > 12.0:
            continue
        if length / max(thickness, 1.0) < STOPLINE_MIN_ASPECT_RATIO:
            continue

        output_h, output_w = mask_probs.shape
        network_segment = np.stack([start, end], axis=0).astype(np.float32)
        network_segment[:, 0] = (network_segment[:, 0] + 0.5) * (float(meta["network_hw"][1]) / float(output_w))
        network_segment[:, 1] = (network_segment[:, 1] + 0.5) * (float(meta["network_hw"][0]) / float(output_h))
        transform = transform_from_meta(meta)
        network_points = sample_stop_line_centerline(network_segment.tolist(), target_count=STOP_LINE_POINT_COUNT).tolist()
        if unique_point_count(network_points) < 2:
            continue
        raw_points = sample_stop_line_centerline(inverse_transform_points(network_points, transform), target_count=STOP_LINE_POINT_COUNT).tolist()
        if unique_point_count(raw_points) < 2:
            continue
        instance_score = float(component_scores[label_index - 1])
        center_score = 0.0
        if center_probs is not None:
            center_score = float(center_probs[rows, cols].max())
            instance_score = 0.25 * center_score + 0.75 * instance_score
            if allowed_labels is not None and label_index in allowed_labels:
                instance_score += 0.02
        if instance_score <= obj_threshold:
            continue
        predictions.append(
            {
                "allowed": bool(allowed_labels is None or label_index in allowed_labels),
                "score": instance_score,
                "center_score": center_score,
                "orientation_score": float(_stopline_orientation_score(raw_points)),
                "length": length,
                "thickness": thickness,
                "points_xy": [[float(x), float(y)] for x, y in raw_points],
            }
        )
    predictions.sort(
        key=_stopline_prediction_sort_key,
        reverse=True,
    )
    predictions = _dedupe_stop_line_predictions(predictions)
    predictions = _suppress_stop_line_fragments(predictions)
    predictions = _promote_stop_line_structured_fallback(predictions)
    predictions = _promote_stop_line_endpoint_floor_backup(predictions, meta=meta)
    predictions.sort(
        key=_stopline_prediction_sort_key,
        reverse=True,
    )
    predictions = _dedupe_stop_line_predictions(predictions)
    return predictions[:3]


def _probe_stopline_mask_decode(
    mask_logits: torch.Tensor,
    selector_map_logits: torch.Tensor | None,
    center_logits: torch.Tensor | None,
    row_logits: torch.Tensor | None,
    x_logits: torch.Tensor | None,
    *,
    obj_threshold: float,
    mask_binary_threshold: float,
) -> dict[str, float | int]:
    if not _tensor_all_finite(mask_logits):
        return {
            "mask_score": 0.0,
            "binary_pixels": 0,
            "component_count": 0,
            "center_peak": 0.0,
            "selected_label_count": 0,
            "geometry_pass_count": 0,
            "best_length": 0.0,
            "best_thickness": 0.0,
        }
    mask_probs = mask_logits.sigmoid().squeeze(0).detach().cpu().numpy()
    if mask_probs.ndim == 3:
        mask_probs = mask_probs.squeeze(0)
    score = float(mask_probs.max())
    binary = _prepare_stopline_binary_mask(mask_probs, threshold=mask_binary_threshold)
    binary_pixels = int(binary.sum())
    if score <= obj_threshold or not bool(binary.any()):
        center_peak = 0.0
        if isinstance(selector_map_logits, torch.Tensor) and _tensor_all_finite(selector_map_logits):
            center_peak = float(selector_map_logits.sigmoid().max().item())
        elif isinstance(center_logits, torch.Tensor) and _tensor_all_finite(center_logits):
            center_peak = float(center_logits.sigmoid().max().item())
        return {
            "mask_score": score,
            "binary_pixels": binary_pixels,
            "component_count": 0,
            "center_peak": center_peak,
            "selected_label_count": 0,
            "geometry_pass_count": 0,
            "best_length": 0.0,
            "best_thickness": 0.0,
        }
    labels, component_count = ndimage.label(binary)
    center_peak = 0.0
    selector_probs = None
    allowed_labels: set[int] | None = None
    row_probs = None
    if isinstance(selector_map_logits, torch.Tensor) and _tensor_all_finite(selector_map_logits):
        selector_probs = selector_map_logits.sigmoid().detach().cpu().numpy()
        if selector_probs.ndim == 4:
            selector_probs = selector_probs.squeeze(0)
        if selector_probs.ndim == 3:
            selector_probs = selector_probs.squeeze(0)
        center_peak = float(np.max(selector_probs))
    if isinstance(center_logits, torch.Tensor) and _tensor_all_finite(center_logits):
        center_probs = center_logits.sigmoid().squeeze(0).detach().cpu().numpy()
        if center_probs.ndim == 3:
            center_probs = center_probs.squeeze(0)
        if selector_probs is None:
            center_peak = float(center_probs.max())
    else:
        center_probs = None
    if isinstance(row_logits, torch.Tensor) and _tensor_all_finite(row_logits):
        row_probs = row_logits.sigmoid().detach().cpu().numpy()
        if row_probs.ndim == 4:
            row_probs = row_probs.squeeze(0)
        if row_probs.ndim == 3:
            row_probs = row_probs.squeeze(0)
        if row_probs.ndim == 2:
            row_probs = row_probs.max(axis=-1)
    if isinstance(x_logits, torch.Tensor) and _tensor_all_finite(x_logits):
        x_probs = x_logits.sigmoid().detach().cpu().numpy()
        if x_probs.ndim == 4:
            x_probs = x_probs.squeeze(0)
        if x_probs.ndim == 3:
            x_probs = x_probs.squeeze(0)
        if x_probs.ndim == 2:
            x_probs = x_probs.max(axis=0)
    else:
        x_probs = None
    allowed_labels = _stopline_allowed_labels(labels, selector_probs if selector_probs is not None else center_probs, row_probs=row_probs)

    geometry_pass_count = 0
    best_length = 0.0
    best_thickness = 0.0
    for label_index in range(1, component_count + 1):
        if allowed_labels is not None and label_index not in allowed_labels:
            continue
        rows, cols = np.nonzero(labels == label_index)
        if len(rows) < STOPLINE_MIN_COMPONENT_PIXELS:
            continue
        component_points = np.stack([cols.astype(np.float32), rows.astype(np.float32)], axis=1)
        center = component_points.mean(axis=0, keepdims=True)
        centered = component_points - center
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        axis = vh[0]
        normal = vh[1]
        projection = centered @ axis
        thickness_projection = centered @ normal
        length = float(projection.max() - projection.min())
        thickness = float(thickness_projection.max() - thickness_projection.min())
        best_length = max(best_length, length)
        best_thickness = max(best_thickness, thickness)
        if length < STOPLINE_MIN_COMPONENT_LENGTH:
            continue
        if thickness > 12.0:
            continue
        if length / max(thickness, 1.0) < STOPLINE_MIN_ASPECT_RATIO:
            continue
        geometry_pass_count += 1

    return {
        "mask_score": score,
        "binary_pixels": binary_pixels,
        "component_count": int(component_count),
        "center_peak": center_peak,
        "selected_label_count": 0 if allowed_labels is None else int(len(allowed_labels)),
        "geometry_pass_count": int(geometry_pass_count),
        "best_length": float(best_length),
        "best_thickness": float(best_thickness),
    }


def _stopline_allowed_labels(
    labels: np.ndarray,
    center_probs: np.ndarray | None,
    *,
    row_probs: np.ndarray | None = None,
) -> set[int] | None:
    if center_probs is None and row_probs is None:
        return None
    if row_probs is not None:
        row_scores = np.asarray(row_probs, dtype=np.float32).reshape(-1)
    else:
        if center_probs is None or center_probs.ndim != 2:
            return None
        row_scores = center_probs.max(axis=1)
    topk = min(1, int(row_scores.shape[0]))
    if topk <= 0:
        return None
    peak_rows = np.argpartition(-row_scores, topk - 1)[:topk]
    positive_rows, positive_cols = np.nonzero(labels > 0)
    selected: set[int] = set()
    for peak_row in peak_rows.tolist():
        peak_score = float(row_scores[peak_row])
        if peak_score < 0.2:
            continue
        candidate_labels: set[int] = set()
        for label_value in range(1, int(labels.max()) + 1):
            rows, _ = np.nonzero(labels == label_value)
            if len(rows) == 0:
                continue
            if int(rows.min()) - STOPLINE_CENTER_ROW_TOLERANCE <= peak_row <= int(rows.max()) + STOPLINE_CENTER_ROW_TOLERANCE:
                candidate_labels.add(int(label_value))
        if candidate_labels:
            selected.update(candidate_labels)
            continue
        if len(positive_rows) > 0:
            nearest = np.argmin((positive_rows - peak_row) ** 2)
            label_value = int(labels[positive_rows[nearest], positive_cols[nearest]])
            if label_value > 0:
                selected.add(label_value)
    return selected or None


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
    valid = torch.isfinite(det_rows).all(dim=-1) & torch.isfinite(tl_rows).all(dim=-1)
    valid &= torch.isfinite(boxes).all(dim=-1) & torch.isfinite(scores)
    valid &= (boxes[:, 2] - boxes[:, 0] > 1.0) & (boxes[:, 3] - boxes[:, 1] > 1.0)
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
        if not _tensor_all_finite(row):
            continue
        score = float(row[0].sigmoid().item())
        if score <= config.lane_obj_threshold:
            continue
        visible = row[LANE_VIS_SLICE].sigmoid() >= config.lane_visibility_threshold
        visible = _visibility_envelope(visible)
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


def _decode_segfirst_lane_rows(
    predictions: dict[str, torch.Tensor | list[Any]],
    *,
    batch_index: int,
    meta: dict[str, Any],
    config: PV26PostprocessConfig,
) -> list[dict[str, Any]] | None:
    if "lane_seg_centerline_logits" not in predictions:
        return None
    required = (
        "lane_seg_centerline_logits",
        "lane_seg_support_logits",
        "lane_seg_tangent_axis",
        "lane_seg_color_logits",
        "lane_seg_type_logits",
    )
    if not all(isinstance(predictions.get(key), torch.Tensor) for key in required):
        raise KeyError("seg-first lane postprocess requires all dense lane prediction maps")
    from .lane_segfirst_vectorizer import (
        LaneSegFirstVectorizerConfig,
        lane_segfirst_prediction_maps,
        vectorize_lane_segfirst_maps,
    )

    maps = lane_segfirst_prediction_maps(predictions, batch_index=batch_index)
    return vectorize_lane_segfirst_maps(
        maps,
        meta=meta,
        config=LaneSegFirstVectorizerConfig(
            centerline_threshold=float(config.lane_obj_threshold),
            min_polyline_length_px=float(config.lane_segfirst_min_polyline_length_px),
            min_polyline_bottom_y_fraction=float(config.lane_segfirst_min_polyline_bottom_y_fraction),
            semantic_vote_mode=str(config.lane_segfirst_semantic_vote_mode),
        ),
    )


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
    mask_binary_threshold: float = 0.5,
    mask_logits: torch.Tensor | None = None,
    selector_map_logits: torch.Tensor | None = None,
    center_logits: torch.Tensor | None = None,
    center_offset: torch.Tensor | None = None,
    row_logits: torch.Tensor | None = None,
    x_logits: torch.Tensor | None = None,
    angle: torch.Tensor | None = None,
    half_length: torch.Tensor | None = None,
) -> list[dict[str, Any]]:
    if isinstance(mask_logits, torch.Tensor):
        decoded = _stopline_mask_to_polyline(
            mask_logits,
            selector_map_logits,
            center_logits,
            center_offset,
            row_logits,
            x_logits,
            angle,
            half_length,
            meta=meta,
            obj_threshold=obj_threshold,
            mask_binary_threshold=mask_binary_threshold,
        )
        if decoded:
            return decoded
    transform = transform_from_meta(meta)
    predictions: list[dict[str, Any]] = []
    for row in rows:
        if not _tensor_all_finite(row):
            continue
        score = float(row[0].sigmoid().item())
        if score <= obj_threshold:
            continue
        encoded_points = row[1 : 1 + STOP_LINE_POINT_COUNT * 2].view(STOP_LINE_POINT_COUNT, 2).tolist()
        network_points = sample_stop_line_centerline(
            clip_points(encoded_points, transform.network_hw),
            target_count=STOP_LINE_POINT_COUNT,
        ).tolist()
        if unique_point_count(network_points) < 2:
            continue
        raw_points = sample_stop_line_centerline(
            inverse_transform_points(network_points, transform),
            target_count=STOP_LINE_POINT_COUNT,
        ).tolist()
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
    mask_binary_threshold: float = 0.5,
    mask_logits: torch.Tensor | None = None,
    center_logits: torch.Tensor | None = None,
) -> list[dict[str, Any]]:
    if isinstance(mask_logits, torch.Tensor):
        decoded = _crosswalk_mask_to_polygon(
            mask_logits,
            center_logits,
            meta=meta,
            obj_threshold=obj_threshold,
            mask_binary_threshold=mask_binary_threshold,
        )
        if decoded:
            return decoded
    transform = transform_from_meta(meta)
    predictions: list[dict[str, Any]] = []
    for row in rows:
        if not _tensor_all_finite(row):
            continue
        score = float(row[0].sigmoid().item())
        if score <= obj_threshold:
            continue
        points = row[1 : 1 + CROSSWALK_POINT_COUNT * 2].view(CROSSWALK_POINT_COUNT, 2).tolist()
        network_points = canonicalize_crosswalk_points(
            sample_crosswalk_contour(clip_points(points, transform.network_hw), target_count=CROSSWALK_POINT_COUNT)
        ).tolist()
        if unique_point_count(network_points) < 3:
            continue
        raw_points = canonicalize_crosswalk_points(
            inverse_transform_points(network_points, transform)
        ).tolist()
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
    stop_line_mask_logits = predictions.get("stop_line_mask_logits")
    stop_line_row_logits = predictions.get("stop_line_row_logits")
    stop_line_x_logits = predictions.get("stop_line_x_logits")
    stop_line_selector_map_logits = predictions.get("stop_line_selector_map_logits")
    stop_line_center_logits = predictions.get("stop_line_center_logits")
    stop_line_center_offset = predictions.get("stop_line_center_offset")
    stop_line_angle = predictions.get("stop_line_angle")
    stop_line_half_length = predictions.get("stop_line_half_length")
    crosswalk_mask_logits = predictions.get("crosswalk_mask_logits")
    crosswalk_center_logits = predictions.get("crosswalk_center_logits")
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
                "lanes": _decode_segfirst_lane_rows(
                    predictions,
                    batch_index=batch_index,
                    meta=sample_meta,
                    config=config,
                )
                or _decode_lane_rows(
                    lane_pred[batch_index],
                    meta=sample_meta,
                    config=config,
                ),
                "stop_lines": _decode_stop_line_rows(
                    stop_line_pred[batch_index],
                    meta=sample_meta,
                    obj_threshold=config.stop_line_obj_threshold,
                    mask_binary_threshold=config.stop_line_mask_binary_threshold,
                    mask_logits=(
                        stop_line_mask_logits[batch_index]
                        if isinstance(stop_line_mask_logits, torch.Tensor)
                        else None
                    ),
                    selector_map_logits=(
                        stop_line_selector_map_logits[batch_index]
                        if isinstance(stop_line_selector_map_logits, torch.Tensor)
                        else None
                    ),
                    center_logits=(
                        stop_line_center_logits[batch_index]
                        if isinstance(stop_line_center_logits, torch.Tensor)
                        else None
                    ),
                    center_offset=(
                        stop_line_center_offset[batch_index]
                        if isinstance(stop_line_center_offset, torch.Tensor)
                        else None
                    ),
                    row_logits=(
                        stop_line_row_logits[batch_index]
                        if isinstance(stop_line_row_logits, torch.Tensor)
                        else None
                    ),
                    x_logits=(
                        stop_line_x_logits[batch_index]
                        if isinstance(stop_line_x_logits, torch.Tensor)
                        else None
                    ),
                    angle=(
                        stop_line_angle[batch_index]
                        if isinstance(stop_line_angle, torch.Tensor)
                        else None
                    ),
                    half_length=(
                        stop_line_half_length[batch_index]
                        if isinstance(stop_line_half_length, torch.Tensor)
                        else None
                    ),
                ),
                "crosswalks": _decode_crosswalk_rows(
                    crosswalk_pred[batch_index],
                    meta=sample_meta,
                    obj_threshold=config.crosswalk_obj_threshold,
                    mask_binary_threshold=config.crosswalk_mask_binary_threshold,
                    mask_logits=(
                        crosswalk_mask_logits[batch_index]
                        if isinstance(crosswalk_mask_logits, torch.Tensor)
                        else None
                    ),
                    center_logits=(
                        crosswalk_center_logits[batch_index]
                        if isinstance(crosswalk_center_logits, torch.Tensor)
                        else None
                    ),
                ),
            }
        )
    return batch_predictions


__all__ = [
    "PV26PostprocessConfig",
    "postprocess_pv26_batch",
]
