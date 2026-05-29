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
    lane_obj_threshold: float = 0.45
    lane_segfirst_min_polyline_length_px: float = 0.0
    lane_segfirst_min_polyline_bottom_y_fraction: float = 0.0
    lane_segfirst_min_bbox_area_px: float = 4096.0
    lane_segfirst_max_bbox_aspect: float = 6.0
    lane_segfirst_semantic_vote_mode: str = "component"
    lane_segfirst_track_mode: str = "component"
    lane_segfirst_max_row_gap: int = 12
    lane_segfirst_max_link_dx: float = 8.0
    lane_segfirst_max_turn_degrees: float = 0.0
    lane_segfirst_seed_threshold: float = 0.50
    lane_segfirst_seed_trace_max_seeds: int = 24
    lane_segfirst_center_offset_enabled: bool = False
    lane_segfirst_center_offset_max_shift_px: float = 4.0
    lane_segfirst_center_offset_min_support_score: float = 0.50
    lane_conditional_row_enabled: bool = False
    lane_conditional_row_merge_mode: str = "replace"
    stop_line_obj_threshold: float = 0.50
    stop_line_mask_binary_threshold: float = 0.50
    stop_line_min_component_pixels: int = 24
    stop_line_max_components: int = 1
    stop_line_min_bbox_area_px: float = 0.0
    stop_line_min_bbox_aspect: float = 6.0
    stop_line_min_instance_score: float = 0.94
    stop_line_presence_threshold: float = 0.0
    stop_line_component_gate_source: str = "center"
    stop_line_haf_enabled: bool = False
    stop_line_haf_valid_threshold: float = 0.50
    stop_line_haf_min_votes: int = 4
    stop_line_haf_cluster_endpoint_tolerance: float = 3.0
    stop_line_haf_max_endpoint_covariance: float = 9.0
    stop_line_haf_max_segments: int = 3
    stop_line_axis_distance_enabled: bool = False
    stop_line_axis_distance_valid_threshold: float = 0.75
    stop_line_axis_distance_min_votes: int = 3
    stop_line_axis_distance_cluster_endpoint_tolerance: float = 4.0
    stop_line_axis_distance_max_endpoint_covariance: float = 16.0
    stop_line_axis_distance_min_support_score: float = 0.35
    stop_line_axis_distance_max_segments: int = 3
    stop_line_endpoint_pair_enabled: bool = False
    stop_line_endpoint_pair_score_threshold: float = 0.55
    stop_line_endpoint_pair_topk: int = 8
    stop_line_endpoint_pair_max_segments: int = 3
    stop_line_endpoint_haf_consensus_enabled: bool = False
    stop_line_endpoint_haf_consensus_score_threshold: float = 0.55
    stop_line_endpoint_haf_consensus_topk: int = 8
    stop_line_endpoint_haf_consensus_haf_valid_threshold: float = 0.65
    stop_line_endpoint_haf_consensus_min_votes: int = 4
    stop_line_endpoint_haf_consensus_max_endpoint_error: float = 8.0
    stop_line_endpoint_haf_consensus_max_endpoint_covariance: float = 32.0
    stop_line_endpoint_haf_consensus_max_segments: int = 3
    stop_line_endpoint_pair_segment_enabled: bool = False
    stop_line_endpoint_pair_segment_score_threshold: float = 0.50
    stop_line_endpoint_pair_segment_max_segments: int = 3
    stop_line_endpoint_pair_verifier_score_weight: float = 0.0
    stop_line_segment_set_enabled: bool = False
    stop_line_segment_set_score_threshold: float = 0.50
    stop_line_segment_set_max_segments: int = 3
    stop_line_segment_verifier_score_weight: float = 0.0
    stop_line_context_segment_set_enabled: bool = False
    stop_line_context_segment_set_score_threshold: float = 0.50
    stop_line_context_segment_set_max_segments: int = 3
    stop_line_context_segment_verifier_score_weight: float = 0.0
    stop_line_axis_segment_set_enabled: bool = False
    stop_line_axis_segment_set_score_threshold: float = 0.50
    stop_line_axis_segment_set_max_segments: int = 3
    stop_line_axis_segment_verifier_score_weight: float = 0.0
    stop_line_patch_segment_set_enabled: bool = False
    stop_line_patch_segment_set_score_threshold: float = 0.50
    stop_line_patch_segment_set_max_segments: int = 3
    stop_line_patch_segment_verifier_score_weight: float = 0.0
    stop_line_projection_comp_enabled: bool = False
    stop_line_projection_comp_proposal_source: str = "max"
    stop_line_projection_comp_min_gap: float = 4.0
    stop_line_projection_comp_topk: int = 50
    stop_line_projection_comp_union_min_score: float = 0.80
    stop_line_projection_comp_single_min_score: float = 0.90
    stop_line_projection_comp_angle_threshold_deg: float = 16.0
    stop_line_projection_comp_offset_threshold_px: float = 48.0
    stop_line_projection_comp_min_cluster_count: int = 2
    stop_line_projection_comp_projection_gap_px: float = 320.0
    stop_line_projection_comp_max_predictions: int = 2
    stop_line_projection_comp_second_min_score: float = 0.0
    stop_line_projection_comp_second_min_fragment_count: int = 5
    stop_line_projection_comp_second_min_length_ratio: float = 0.0
    crosswalk_obj_threshold: float = 0.50
    crosswalk_mask_binary_threshold: float = 0.20
    crosswalk_min_component_pixels: int = 24
    crosswalk_max_components: int = 0
    crosswalk_min_polygon_area_px: float = 640.0
    crosswalk_min_bbox_aspect: float = 3.0
    crosswalk_polygon_mode: str = "rect"
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


def _polygon_area(points_xy: list[list[float]] | np.ndarray) -> float:
    points = np.asarray(points_xy, dtype=np.float32).reshape(-1, 2)
    if points.shape[0] < 3 or not bool(np.isfinite(points).all()):
        return 0.0
    x = points[:, 0]
    y = points[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) * 0.5)


def _bbox_aspect(points_xy: list[list[float]] | np.ndarray) -> float:
    points = np.asarray(points_xy, dtype=np.float32).reshape(-1, 2)
    if points.shape[0] == 0 or not bool(np.isfinite(points).all()):
        return 0.0
    min_xy = points.min(axis=0)
    max_xy = points.max(axis=0)
    width = max(float(max_xy[0] - min_xy[0]), 0.0)
    height = max(float(max_xy[1] - min_xy[1]), 0.0)
    return max(width, height) / max(min(width, height), 1.0)


def _bbox_area(points_xy: list[list[float]] | np.ndarray) -> float:
    points = np.asarray(points_xy, dtype=np.float32).reshape(-1, 2)
    if points.shape[0] == 0 or not bool(np.isfinite(points).all()):
        return 0.0
    min_xy = points.min(axis=0)
    max_xy = points.max(axis=0)
    width = max(float(max_xy[0] - min_xy[0]), 0.0)
    height = max(float(max_xy[1] - min_xy[1]), 0.0)
    return width * height


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


def _dedupe_lane_predictions_by_distance(
    predictions: list[dict[str, Any]],
    *,
    distance_threshold: float = 24.0,
) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    ordered = sorted(predictions, key=lambda item: float(item.get("score", 0.0)), reverse=True)
    for candidate in ordered:
        candidate_points = candidate.get("points_xy", [])
        is_duplicate = False
        for existing in kept:
            if candidate.get("class_name") != existing.get("class_name") or candidate.get("lane_type") != existing.get("lane_type"):
                continue
            try:
                distance = _mean_point_distance(candidate_points, existing.get("points_xy", []), target_count=20)
            except Exception:
                distance = float("inf")
            if distance <= float(distance_threshold):
                is_duplicate = True
                break
        if not is_duplicate:
            kept.append(candidate)
    return kept


def _filter_lane_predictions(
    predictions: list[dict[str, Any]] | None,
    *,
    min_bbox_area_px: float = 0.0,
    max_bbox_aspect: float = 0.0,
) -> list[dict[str, Any]]:
    if predictions is None:
        return []
    if float(min_bbox_area_px) <= 0.0 and float(max_bbox_aspect) <= 0.0:
        return predictions
    kept: list[dict[str, Any]] = []
    for prediction in predictions:
        points_xy = prediction.get("points_xy", [])
        if float(min_bbox_area_px) > 0.0 and _bbox_area(points_xy) < float(min_bbox_area_px):
            continue
        if float(max_bbox_aspect) > 0.0 and _bbox_aspect(points_xy) > float(max_bbox_aspect):
            continue
        kept.append(prediction)
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


def _filter_stop_line_predictions(
    predictions: list[dict[str, Any]],
    *,
    min_bbox_area_px: float = 0.0,
    min_bbox_aspect: float = 0.0,
    min_instance_score: float = 0.0,
) -> list[dict[str, Any]]:
    if (
        float(min_bbox_area_px) <= 0.0
        and float(min_bbox_aspect) <= 0.0
        and float(min_instance_score) <= 0.0
    ):
        return predictions
    kept: list[dict[str, Any]] = []
    for prediction in predictions:
        points_xy = prediction.get("points_xy", [])
        if float(min_instance_score) > 0.0 and float(prediction.get("score", 0.0)) < float(min_instance_score):
            continue
        if float(min_bbox_area_px) > 0.0 and _bbox_area(points_xy) < float(min_bbox_area_px):
            continue
        if float(min_bbox_aspect) > 0.0 and _bbox_aspect(points_xy) < float(min_bbox_aspect):
            continue
        kept.append(prediction)
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


def _convex_hull_points(points: np.ndarray) -> np.ndarray:
    unique = np.unique(np.asarray(points, dtype=np.float32), axis=0)
    if unique.shape[0] <= 2:
        return unique
    order = np.lexsort((unique[:, 1], unique[:, 0]))
    sorted_points = unique[order]

    def cross(origin: np.ndarray, left: np.ndarray, right: np.ndarray) -> float:
        return float((left[0] - origin[0]) * (right[1] - origin[1]) - (left[1] - origin[1]) * (right[0] - origin[0]))

    lower: list[np.ndarray] = []
    for point in sorted_points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0.0:
            lower.pop()
        lower.append(point)
    upper: list[np.ndarray] = []
    for point in reversed(sorted_points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0.0:
            upper.pop()
        upper.append(point)
    hull = np.asarray(lower[:-1] + upper[:-1], dtype=np.float32)
    return hull if hull.shape[0] >= 3 else unique


def _minimum_area_rect(points: np.ndarray) -> np.ndarray | None:
    hull = _convex_hull_points(points)
    if hull.shape[0] < 3:
        return None
    best_area: float | None = None
    best_rect: np.ndarray | None = None
    for index in range(hull.shape[0]):
        edge = hull[(index + 1) % hull.shape[0]] - hull[index]
        edge_norm = float(np.linalg.norm(edge))
        if edge_norm <= 1.0e-6:
            continue
        axis1 = edge / edge_norm
        axis2 = np.array([-axis1[1], axis1[0]], dtype=np.float32)
        proj1 = hull @ axis1
        proj2 = hull @ axis2
        min1, max1 = float(proj1.min()), float(proj1.max())
        min2, max2 = float(proj2.min()), float(proj2.max())
        area = max(max1 - min1, 0.0) * max(max2 - min2, 0.0)
        if best_area is not None and area >= best_area:
            continue
        best_area = area
        best_rect = np.array(
            [
                axis1 * min1 + axis2 * min2,
                axis1 * max1 + axis2 * min2,
                axis1 * max1 + axis2 * max2,
                axis1 * min1 + axis2 * max2,
            ],
            dtype=np.float32,
        )
    return best_rect


def _crosswalk_mask_to_polygon(
    mask_logits: torch.Tensor,
    center_logits: torch.Tensor | None,
    *,
    meta: dict[str, Any],
    obj_threshold: float,
    mask_binary_threshold: float,
    min_component_pixels: int = 4,
    max_components: int = 0,
    min_polygon_area_px: float = 0.0,
    min_bbox_aspect: float = 0.0,
    polygon_mode: str = "rect",
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
        if len(rows) < max(4, int(min_component_pixels)):
            continue

        points = np.stack([cols.astype(np.float32), rows.astype(np.float32)], axis=1)
        normalized_polygon_mode = str(polygon_mode).strip().lower()
        if normalized_polygon_mode == "hull":
            polygon = _convex_hull_points(points)
        elif normalized_polygon_mode == "rect":
            polygon = _minimum_area_rect(points)
        else:
            raise ValueError("crosswalk_polygon_mode must be one of: rect, hull")
        if polygon is None or int(polygon.shape[0]) < 3:
            continue

        polygon[:, 0] = (polygon[:, 0] + 0.5) * (float(meta["network_hw"][1]) / float(output_w))
        polygon[:, 1] = (polygon[:, 1] + 0.5) * (float(meta["network_hw"][0]) / float(output_h))
        network_points = canonicalize_crosswalk_points(
            sample_crosswalk_contour(polygon, target_count=CROSSWALK_POINT_COUNT)
        ).tolist()
        if unique_point_count(network_points) < 3:
            continue
        raw_points = canonicalize_crosswalk_points(inverse_transform_points(network_points, transform)).tolist()
        if unique_point_count(raw_points) < 3:
            continue
        if float(min_polygon_area_px) > 0.0 and _polygon_area(raw_points) < float(min_polygon_area_px):
            continue
        if float(min_bbox_aspect) > 0.0 and _bbox_aspect(raw_points) < float(min_bbox_aspect):
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
    deduped = _dedupe_crosswalk_predictions(predictions)
    if int(max_components) > 0:
        return deduped[: int(max_components)]
    return deduped


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


def _canonical_segment_endpoints(start: np.ndarray, end: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if (float(start[0]), float(start[1])) <= (float(end[0]), float(end[1])):
        return start.astype(np.float32), end.astype(np.float32)
    return end.astype(np.float32), start.astype(np.float32)


def _stopline_2d_sigmoid_array(tensor: torch.Tensor | None) -> np.ndarray | None:
    if not isinstance(tensor, torch.Tensor) or not _tensor_all_finite(tensor):
        return None
    array = tensor.sigmoid().detach().cpu().numpy()
    while array.ndim > 2 and int(array.shape[0]) == 1:
        array = array.squeeze(0)
    if array.ndim == 3 and int(array.shape[0]) == 1:
        array = array.squeeze(0)
    if array.ndim != 2:
        return None
    return np.asarray(array, dtype=np.float32)


def _stopline_top_spaced_cells(
    map_scores: np.ndarray,
    *,
    top_k: int,
    threshold: float,
    min_gap: float,
) -> list[tuple[int, int, float]]:
    if map_scores.ndim != 2:
        return []
    flat_scores = map_scores.reshape(-1)
    flat_order = np.argsort(-flat_scores)
    output_h, output_w = map_scores.shape
    selected: list[tuple[int, int, float]] = []
    for flat_index in flat_order.tolist():
        score = float(flat_scores[flat_index])
        if score < float(threshold):
            break
        row, col = divmod(int(flat_index), int(output_w))
        too_close = False
        for prev_row, prev_col, _ in selected:
            if float(np.hypot(float(col - prev_col), float(row - prev_row))) < float(min_gap):
                too_close = True
                break
        if too_close:
            continue
        selected.append((int(row), int(col), score))
        if len(selected) >= int(top_k):
            break
    return selected


def _stopline_union_source_cells(
    source_maps: list[np.ndarray | None],
    *,
    top_k: int,
    threshold: float,
    min_gap: float,
) -> list[tuple[int, int, float]]:
    """Keep source-local proposal peaks instead of collapsing sources before top-k."""

    merged: dict[tuple[int, int], float] = {}
    for source_map in source_maps:
        if source_map is None:
            continue
        for row, col, score in _stopline_top_spaced_cells(
            source_map,
            top_k=int(top_k),
            threshold=float(threshold),
            min_gap=float(min_gap),
        ):
            key = (int(row), int(col))
            merged[key] = max(float(score), float(merged.get(key, -float("inf"))))
    cells = [(row, col, score) for (row, col), score in merged.items()]
    cells.sort(key=lambda item: float(item[2]), reverse=True)
    return cells


def _stopline_nearest_component_label(
    labels: np.ndarray,
    *,
    center_xy: np.ndarray,
    search_radius: float,
) -> int:
    output_h, output_w = labels.shape
    row = max(0, min(int(output_h) - 1, int(round(float(center_xy[1])))))
    col = max(0, min(int(output_w) - 1, int(round(float(center_xy[0])))))
    center_label = int(labels[row, col])
    if center_label > 0:
        return center_label
    rows, cols = np.nonzero(labels > 0)
    if rows.size == 0:
        return 0
    distances = np.sqrt(
        (cols.astype(np.float32) - float(center_xy[0])) ** 2
        + (rows.astype(np.float32) - float(center_xy[1])) ** 2
    )
    best_index = int(np.argmin(distances))
    if float(distances[best_index]) > float(search_radius):
        return 0
    return int(labels[int(rows[best_index]), int(cols[best_index])])


def _stopline_raw_points_from_map_segment(
    *,
    center_xy: np.ndarray,
    axis: np.ndarray,
    start_proj: float,
    end_proj: float,
    meta: dict[str, Any],
    output_hw: tuple[int, int],
) -> list[list[float]] | None:
    center_xy = np.asarray(center_xy, dtype=np.float32).reshape(2)
    axis = np.asarray(axis, dtype=np.float32).reshape(2)
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm <= 1.0e-6 or not np.isfinite(axis_norm):
        return None
    axis = axis / axis_norm
    if not np.isfinite(float(start_proj)) or not np.isfinite(float(end_proj)):
        return None
    if float(end_proj) - float(start_proj) <= 1.0:
        return None
    start = center_xy + axis * float(start_proj)
    end = center_xy + axis * float(end_proj)
    output_h, output_w = int(output_hw[0]), int(output_hw[1])
    network_segment = np.stack([start, end], axis=0).astype(np.float32)
    network_segment[:, 0] = (network_segment[:, 0] + 0.5) * (float(meta["network_hw"][1]) / float(output_w))
    network_segment[:, 1] = (network_segment[:, 1] + 0.5) * (float(meta["network_hw"][0]) / float(output_h))
    transform = transform_from_meta(meta)
    network_points = sample_stop_line_centerline(network_segment.tolist(), target_count=STOP_LINE_POINT_COUNT)
    raw_points = sample_stop_line_centerline(
        inverse_transform_points(network_points.tolist(), transform),
        target_count=STOP_LINE_POINT_COUNT,
    )
    if raw_points.shape[0] < 2 or not bool(np.isfinite(raw_points).all()):
        return None
    if unique_point_count(raw_points.tolist()) < 2:
        return None
    return [[float(x), float(y)] for x, y in raw_points.tolist()]


def _stopline_normalize_axis(vector: np.ndarray) -> np.ndarray | None:
    norm = float(np.linalg.norm(vector))
    if norm <= 1.0e-6 or not np.isfinite(norm):
        return None
    axis = np.asarray(vector, dtype=np.float32) / norm
    if float(axis[0]) < 0.0 or (abs(float(axis[0])) <= 1.0e-6 and float(axis[1]) < 0.0):
        axis = -axis
    return axis.astype(np.float32)


def _stopline_projection_candidate_from_points(
    *,
    points_xy: list[list[float]],
    score: float,
    rank_length: float,
    proposal_rank: int,
) -> dict[str, Any] | None:
    points = np.asarray(points_xy, dtype=np.float32).reshape(-1, 2)
    if points.shape[0] < 2 or not bool(np.isfinite(points).all()):
        return None
    axis = _stopline_normalize_axis(points[-1] - points[0])
    if axis is None:
        return None
    normal = np.asarray([-float(axis[1]), float(axis[0])], dtype=np.float32)
    center_xy = points.mean(axis=0).astype(np.float32)
    raw_length = float(np.linalg.norm(points[-1] - points[0]))
    return {
        "score": float(score),
        "center_score": float(score),
        "length": raw_length,
        "rank_length": float(rank_length),
        "component_svd_length": float(rank_length),
        "points_xy": [[float(x), float(y)] for x, y in points.tolist()],
        "center_xy": center_xy,
        "axis": axis,
        "normal": normal,
        "offset": float(np.dot(center_xy, normal)),
        "proposal_rank": int(proposal_rank),
    }


def _stopline_map_extent_candidate(
    *,
    mask_probs: np.ndarray,
    center_xy: np.ndarray,
    angle_vec: np.ndarray,
    meta: dict[str, Any],
    output_hw: tuple[int, int],
    score: float,
    proposal_rank: int,
    mask_threshold: float = 0.50,
    normal_band: float = 4.0,
    search_radius: float = 8.0,
) -> dict[str, Any] | None:
    binary = _prepare_stopline_binary_mask(mask_probs, threshold=float(mask_threshold))
    labels, _ = ndimage.label(binary)
    label = _stopline_nearest_component_label(labels, center_xy=center_xy, search_radius=float(search_radius))
    if label <= 0:
        return None
    rows, cols = np.nonzero(labels == label)
    if rows.size < 2:
        return None
    points = np.stack([cols.astype(np.float32), rows.astype(np.float32)], axis=1)
    axis = np.asarray(angle_vec, dtype=np.float32).reshape(2)
    axis = _stopline_normalize_axis(axis)
    if axis is None:
        return None
    normal = np.asarray([-float(axis[1]), float(axis[0])], dtype=np.float32)
    offsets = points - np.asarray(center_xy, dtype=np.float32).reshape(1, 2)
    if float(normal_band) > 0.0:
        local_mask = np.abs(offsets @ normal) <= float(normal_band)
        if int(local_mask.sum()) >= 2:
            offsets = offsets[local_mask]
    if offsets.shape[0] < 2:
        return None
    projections = offsets @ axis
    if projections.size < 2:
        return None
    if projections.size >= 8:
        start_proj = float(np.quantile(projections, 0.05))
        end_proj = float(np.quantile(projections, 0.95))
    else:
        start_proj = float(projections.min())
        end_proj = float(projections.max())
    raw_points = _stopline_raw_points_from_map_segment(
        center_xy=center_xy,
        axis=axis,
        start_proj=start_proj,
        end_proj=end_proj,
        meta=meta,
        output_hw=output_hw,
    )
    if raw_points is None:
        return None
    return _stopline_projection_candidate_from_points(
        points_xy=raw_points,
        score=float(score),
        rank_length=float(end_proj - start_proj),
        proposal_rank=int(proposal_rank),
    )


def _stopline_projection_angle_error(left: dict[str, Any], right: dict[str, Any]) -> float:
    left_axis = np.asarray(left["axis"], dtype=np.float32)
    right_axis = np.asarray(right["axis"], dtype=np.float32)
    cosine = float(np.clip(abs(float(np.dot(left_axis, right_axis))), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _stopline_projection_offset_error(left: dict[str, Any], right: dict[str, Any]) -> float:
    left_center = np.asarray(left["center_xy"], dtype=np.float32)
    right_center = np.asarray(right["center_xy"], dtype=np.float32)
    left_normal = np.asarray(left["normal"], dtype=np.float32)
    right_normal = np.asarray(right["normal"], dtype=np.float32)
    left_error = abs(float(np.dot(right_center, left_normal) - float(left["offset"])))
    right_error = abs(float(np.dot(left_center, right_normal) - float(right["offset"])))
    return float(max(left_error, right_error))


def _stopline_projection_union_groups(
    candidates: list[dict[str, Any]],
    *,
    angle_threshold_deg: float,
    offset_threshold_px: float,
) -> list[list[dict[str, Any]]]:
    if not candidates:
        return []
    parent = list(range(len(candidates)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for left_index, left in enumerate(candidates):
        for right_index in range(left_index + 1, len(candidates)):
            right = candidates[right_index]
            if _stopline_projection_angle_error(left, right) > float(angle_threshold_deg):
                continue
            if _stopline_projection_offset_error(left, right) > float(offset_threshold_px):
                continue
            union(left_index, right_index)

    groups_by_root: dict[int, list[dict[str, Any]]] = {}
    for index, candidate in enumerate(candidates):
        groups_by_root.setdefault(find(index), []).append(candidate)
    return list(groups_by_root.values())


def _stopline_projection_group_axis(group: list[dict[str, Any]]) -> np.ndarray | None:
    if not group:
        return None
    reference_axis = np.asarray(group[0]["axis"], dtype=np.float32)
    weighted_axis = np.zeros(2, dtype=np.float32)
    for candidate in group:
        axis = np.asarray(candidate["axis"], dtype=np.float32)
        if float(np.dot(axis, reference_axis)) < 0.0:
            axis = -axis
        weighted_axis += axis * max(1.0e-3, float(candidate.get("score", 0.0)))
    return _stopline_normalize_axis(weighted_axis)


def _stopline_projection_interval(candidate: dict[str, Any], axis: np.ndarray) -> tuple[float, float]:
    points = np.asarray(candidate.get("points_xy", []), dtype=np.float32).reshape(-1, 2)
    projections = [float(np.dot(point, axis)) for point in points]
    return min(projections), max(projections)


def _stopline_split_group_by_projection_gap(
    group: list[dict[str, Any]],
    projection_gap_px: float,
) -> list[list[dict[str, Any]]]:
    if len(group) <= 1:
        return [group]
    axis = _stopline_projection_group_axis(group)
    if axis is None:
        return [group]
    intervals = [(*_stopline_projection_interval(candidate, axis), candidate) for candidate in group]
    intervals.sort(key=lambda item: item[0])
    parts: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_hi: float | None = None
    for lo, hi, candidate in intervals:
        if current and current_hi is not None and float(lo) - float(current_hi) > float(projection_gap_px):
            parts.append(current)
            current = []
        current.append(candidate)
        current_hi = max(float(current_hi) if current_hi is not None else float(hi), float(hi))
    if current:
        parts.append(current)
    return parts


def _stopline_merge_projection_group(group: list[dict[str, Any]], *, min_cluster_count: int) -> dict[str, Any] | None:
    if len(group) < max(1, int(min_cluster_count)):
        return None
    axis = _stopline_projection_group_axis(group)
    if axis is None:
        return None
    normal = np.asarray([-float(axis[1]), float(axis[0])], dtype=np.float32)
    weight_sum = 0.0
    weighted_offset = 0.0
    projections: list[float] = []
    for candidate in group:
        weight = max(1.0e-3, float(candidate.get("score", 0.0)))
        center_xy = np.asarray(candidate["center_xy"], dtype=np.float32)
        weighted_offset += float(np.dot(center_xy, normal)) * weight
        weight_sum += weight
        points = np.asarray(candidate.get("points_xy", []), dtype=np.float32).reshape(-1, 2)
        projections.extend(float(np.dot(point, axis)) for point in points)
    if not projections:
        return None
    offset = weighted_offset / max(1.0e-6, weight_sum)
    start_t = min(projections)
    end_t = max(projections)
    if end_t <= start_t:
        return None
    start_xy = axis * start_t + normal * offset
    end_xy = axis * end_t + normal * offset
    raw_points = sample_stop_line_centerline(
        [[float(start_xy[0]), float(start_xy[1])], [float(end_xy[0]), float(end_xy[1])]],
        target_count=STOP_LINE_POINT_COUNT,
    ).tolist()
    length = float(np.linalg.norm(end_xy - start_xy))
    max_score = max(float(candidate.get("score", 0.0)) for candidate in group)
    mean_score = float(sum(float(candidate.get("score", 0.0)) for candidate in group) / max(1, len(group)))
    rank_feature = max(float(candidate.get("rank_length", 0.0)) for candidate in group)
    return {
        "allowed": True,
        "score": float(max_score + 0.02 * float(max(0, len(group) - 1))),
        "center_score": float(mean_score),
        "length": length,
        "points_xy": [[float(x), float(y)] for x, y in raw_points],
        "fragment_count": int(len(group)),
        "_proposal_kind": "union",
        "_rank_feature": float(rank_feature),
    }


def _stopline_projection_keep_additional(
    prediction: dict[str, Any],
    primary_prediction: dict[str, Any],
    *,
    second_min_score: float,
    second_min_fragment_count: int,
    second_min_length_ratio: float,
) -> bool:
    if float(prediction.get("score", 0.0)) < float(second_min_score):
        return False
    if int(prediction.get("fragment_count", 0)) < int(second_min_fragment_count):
        return False
    primary_length = max(1.0e-6, float(primary_prediction.get("length", 0.0)))
    if float(prediction.get("length", 0.0)) / primary_length < float(second_min_length_ratio):
        return False
    return True


def _stopline_projection_apply_cap(
    predictions: list[dict[str, Any]],
    *,
    max_predictions: int,
    second_min_score: float,
    second_min_fragment_count: int,
    second_min_length_ratio: float,
) -> list[dict[str, Any]]:
    max_predictions = max(1, int(max_predictions))
    if len(predictions) <= 1 or max_predictions <= 1:
        return predictions[:1]
    capped = [predictions[0]]
    for prediction in predictions[1:]:
        if len(capped) >= max_predictions:
            break
        if _stopline_projection_keep_additional(
            prediction,
            predictions[0],
            second_min_score=float(second_min_score),
            second_min_fragment_count=int(second_min_fragment_count),
            second_min_length_ratio=float(second_min_length_ratio),
        ):
            capped.append(prediction)
    return capped


def _decode_stopline_projection_competition(
    *,
    mask_logits: torch.Tensor | None,
    center_logits: torch.Tensor | None,
    midpoint_logits: torch.Tensor | None,
    selector_map_logits: torch.Tensor | None,
    center_offset: torch.Tensor | None,
    angle: torch.Tensor | None,
    meta: dict[str, Any],
    proposal_source: str,
    min_gap: float,
    top_k: int,
    union_min_score: float,
    single_min_score: float,
    angle_threshold_deg: float,
    offset_threshold_px: float,
    min_cluster_count: int,
    projection_gap_px: float,
    max_predictions: int,
    second_min_score: float,
    second_min_fragment_count: int,
    second_min_length_ratio: float,
) -> list[dict[str, Any]]:
    mask_probs = _stopline_2d_sigmoid_array(mask_logits)
    center_probs = _stopline_2d_sigmoid_array(center_logits)
    midpoint_probs = _stopline_2d_sigmoid_array(midpoint_logits)
    selector_probs = _stopline_2d_sigmoid_array(selector_map_logits)
    if mask_probs is None or (center_probs is None and selector_probs is None and midpoint_probs is None):
        return []
    if not isinstance(center_offset, torch.Tensor) or not isinstance(angle, torch.Tensor):
        return []
    if not _tensor_all_finite(center_offset) or not _tensor_all_finite(angle):
        return []
    offset_map = center_offset.detach().cpu()
    angle_map = angle.detach().cpu()
    if offset_map.ndim == 4:
        offset_map = offset_map.squeeze(0)
    if angle_map.ndim == 4:
        angle_map = angle_map.squeeze(0)
    if offset_map.ndim != 3 or angle_map.ndim != 3 or int(offset_map.shape[0]) < 2 or int(angle_map.shape[0]) < 2:
        return []
    proposal_source = str(proposal_source or "max").strip().lower()
    proposal_inputs = {
        "center": center_probs,
        "selector": selector_probs,
        "midpoint": midpoint_probs,
    }
    top_cells: list[tuple[int, int, float]]
    if proposal_source == "max":
        available = [value for value in proposal_inputs.values() if value is not None]
        if not available:
            return []
        proposal_map = available[0]
        for value in available[1:]:
            proposal_map = np.maximum(proposal_map, value)
        top_cells = _stopline_top_spaced_cells(
            proposal_map,
            top_k=int(top_k),
            threshold=0.0,
            min_gap=float(min_gap),
        )
    elif proposal_source in proposal_inputs:
        proposal_map = proposal_inputs[proposal_source]
        if proposal_map is None:
            return []
        top_cells = _stopline_top_spaced_cells(
            proposal_map,
            top_k=int(top_k),
            threshold=0.0,
            min_gap=float(min_gap),
        )
    elif proposal_source == "midpoint_max":
        if midpoint_probs is None:
            return []
        proposal_map = midpoint_probs
        for value in (center_probs, selector_probs):
            if value is not None:
                proposal_map = np.maximum(proposal_map, value)
        top_cells = _stopline_top_spaced_cells(
            proposal_map,
            top_k=int(top_k),
            threshold=0.0,
            min_gap=float(min_gap),
        )
    elif proposal_source == "center_selector_union":
        top_cells = _stopline_union_source_cells(
            [center_probs, selector_probs],
            top_k=int(top_k),
            threshold=0.0,
            min_gap=float(min_gap),
        )
    elif proposal_source == "source_union":
        top_cells = _stopline_union_source_cells(
            [center_probs, selector_probs, midpoint_probs],
            top_k=int(top_k),
            threshold=0.0,
            min_gap=float(min_gap),
        )
    else:
        raise ValueError(f"unsupported stop_line_projection_comp_proposal_source: {proposal_source}")
    output_hw = (int(mask_probs.shape[0]), int(mask_probs.shape[1]))

    candidates: list[dict[str, Any]] = []
    offset_np = offset_map.numpy().astype(np.float32)
    angle_np = angle_map.numpy().astype(np.float32)
    for proposal_rank, (row, col, score) in enumerate(top_cells, start=1):
        center_xy = np.asarray(
            [float(col) + float(offset_np[0, row, col]), float(row) + float(offset_np[1, row, col])],
            dtype=np.float32,
        )
        candidate = _stopline_map_extent_candidate(
            mask_probs=mask_probs,
            center_xy=center_xy,
            angle_vec=angle_np[:, row, col],
            meta=meta,
            output_hw=output_hw,
            score=float(score),
            proposal_rank=int(proposal_rank),
            mask_threshold=0.50,
            normal_band=4.0,
        )
        if candidate is not None:
            candidates.append(candidate)
    if not candidates:
        return []

    union_candidates = [candidate for candidate in candidates if float(candidate["score"]) >= float(union_min_score)]
    union_groups = _stopline_projection_union_groups(
        union_candidates,
        angle_threshold_deg=float(angle_threshold_deg),
        offset_threshold_px=float(offset_threshold_px),
    )
    proposals: list[dict[str, Any]] = []
    for group in union_groups:
        for split_group in _stopline_split_group_by_projection_gap(group, float(projection_gap_px)):
            merged = _stopline_merge_projection_group(split_group, min_cluster_count=int(min_cluster_count))
            if merged is not None:
                proposals.append(merged)

    for candidate in candidates:
        if float(candidate["score"]) < float(single_min_score):
            continue
        proposals.append(
            {
                "allowed": True,
                "score": float(candidate["score"]),
                "center_score": float(candidate["score"]),
                "length": float(candidate["length"]),
                "points_xy": candidate["points_xy"],
                "fragment_count": 1,
                "_proposal_kind": "single",
                "_rank_feature": float(candidate.get("rank_length", 0.0)),
            }
        )

    if not proposals:
        return []
    proposals.sort(
        key=lambda item: (
            float(item.get("_rank_feature", 0.0)),
            1.0 if str(item.get("_proposal_kind", "")) == "union" else 0.0,
            float(item.get("score", 0.0)),
            float(item.get("length", 0.0)),
        ),
        reverse=True,
    )
    proposals = _dedupe_stop_line_predictions(proposals)
    capped = _stopline_projection_apply_cap(
        proposals,
        max_predictions=int(max_predictions),
        second_min_score=float(second_min_score),
        second_min_fragment_count=int(second_min_fragment_count),
        second_min_length_ratio=float(second_min_length_ratio),
    )
    cleaned: list[dict[str, Any]] = []
    for prediction in capped:
        output = dict(prediction)
        output.pop("_proposal_kind", None)
        output.pop("_rank_feature", None)
        output["orientation_score"] = float(_stopline_orientation_score(output.get("points_xy", [])))
        cleaned.append(output)
    return cleaned


def _decode_stopline_haf_consensus_segments(
    *,
    haf_endpoint: torch.Tensor | None,
    haf_valid_logits: torch.Tensor | None,
    meta: dict[str, Any],
    valid_threshold: float,
    min_votes: int,
    cluster_endpoint_tolerance: float,
    max_endpoint_covariance: float,
    max_segments: int,
) -> list[dict[str, Any]]:
    if not isinstance(haf_endpoint, torch.Tensor) or not isinstance(haf_valid_logits, torch.Tensor):
        return []
    if not _tensor_all_finite(haf_endpoint) or not _tensor_all_finite(haf_valid_logits):
        return []

    endpoint_map = haf_endpoint.detach().cpu()
    valid_map = haf_valid_logits.sigmoid().detach().cpu()
    if endpoint_map.ndim == 4:
        endpoint_map = endpoint_map.squeeze(0)
    if valid_map.ndim == 4:
        valid_map = valid_map.squeeze(0)
    if valid_map.ndim == 3:
        valid_map = valid_map.squeeze(0)
    if endpoint_map.ndim != 3 or int(endpoint_map.shape[0]) != 4 or valid_map.ndim != 2:
        return []
    output_h, output_w = int(valid_map.shape[0]), int(valid_map.shape[1])
    if endpoint_map.shape[1:] != valid_map.shape:
        return []

    support_rows, support_cols = np.nonzero(valid_map.numpy() >= float(valid_threshold))
    if support_rows.size == 0:
        return []

    votes: list[dict[str, Any]] = []
    endpoint_np = endpoint_map.numpy().astype(np.float32)
    valid_np = valid_map.numpy().astype(np.float32)
    for row_index, col_index in zip(support_rows.tolist(), support_cols.tolist()):
        point = np.array([float(col_index), float(row_index)], dtype=np.float32)
        start = point + endpoint_np[0:2, row_index, col_index]
        end = point + endpoint_np[2:4, row_index, col_index]
        if not np.isfinite(start).all() or not np.isfinite(end).all():
            continue
        start, end = _canonical_segment_endpoints(start, end)
        length = float(np.linalg.norm(end - start))
        if length < STOPLINE_MIN_COMPONENT_LENGTH:
            continue
        votes.append(
            {
                "start": start,
                "end": end,
                "weight": max(float(valid_np[row_index, col_index]), 1.0e-6),
            }
        )
    if not votes:
        return []

    votes.sort(key=lambda item: float(item["weight"]), reverse=True)
    clusters: list[dict[str, Any]] = []
    tolerance = max(float(cluster_endpoint_tolerance), 1.0e-6)
    for vote in votes:
        assigned = False
        for cluster in clusters:
            start_mean = np.asarray(cluster["start_sum"], dtype=np.float32) / max(float(cluster["weight_sum"]), 1.0e-6)
            end_mean = np.asarray(cluster["end_sum"], dtype=np.float32) / max(float(cluster["weight_sum"]), 1.0e-6)
            endpoint_error = max(
                float(np.linalg.norm(vote["start"] - start_mean)),
                float(np.linalg.norm(vote["end"] - end_mean)),
            )
            if endpoint_error > tolerance:
                continue
            weight = float(vote["weight"])
            cluster["start_sum"] = np.asarray(cluster["start_sum"], dtype=np.float32) + vote["start"] * weight
            cluster["end_sum"] = np.asarray(cluster["end_sum"], dtype=np.float32) + vote["end"] * weight
            cluster["weight_sum"] = float(cluster["weight_sum"]) + weight
            cluster["votes"].append(vote)
            assigned = True
            break
        if not assigned:
            weight = float(vote["weight"])
            clusters.append(
                {
                    "start_sum": vote["start"] * weight,
                    "end_sum": vote["end"] * weight,
                    "weight_sum": weight,
                    "votes": [vote],
                }
            )

    transform = transform_from_meta(meta)
    decoded: list[dict[str, Any]] = []
    for cluster in clusters:
        cluster_votes = list(cluster["votes"])
        vote_count = len(cluster_votes)
        if vote_count < int(min_votes):
            continue
        weight_sum = max(float(cluster["weight_sum"]), 1.0e-6)
        start = np.asarray(cluster["start_sum"], dtype=np.float32) / weight_sum
        end = np.asarray(cluster["end_sum"], dtype=np.float32) / weight_sum
        start, end = _canonical_segment_endpoints(start, end)
        endpoint_errors = [
            max(float(np.linalg.norm(vote["start"] - start)), float(np.linalg.norm(vote["end"] - end)))
            for vote in cluster_votes
        ]
        endpoint_covariance = float(np.mean(np.square(endpoint_errors))) if endpoint_errors else float("inf")
        if endpoint_covariance > float(max_endpoint_covariance):
            continue
        grid_segment = np.stack([start, end], axis=0).astype(np.float32)
        network_segment = grid_segment.copy()
        network_segment[:, 0] = network_segment[:, 0] * (float(meta["network_hw"][1]) / float(output_w))
        network_segment[:, 1] = network_segment[:, 1] * (float(meta["network_hw"][0]) / float(output_h))
        network_points = sample_stop_line_centerline(
            clip_points(network_segment.tolist(), transform.network_hw),
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
        mean_vote_score = weight_sum / float(vote_count)
        vote_score = min(1.0, float(vote_count) / 16.0)
        covariance_score = 1.0 / (1.0 + max(endpoint_covariance, 0.0))
        score = 0.55 * mean_vote_score + 0.25 * vote_score + 0.20 * covariance_score
        decoded.append(
            {
                "allowed": True,
                "score": float(score),
                "center_score": float(mean_vote_score),
                "orientation_score": float(_stopline_orientation_score(raw_points)),
                "length": float(np.linalg.norm(end - start)),
                "thickness": 1.0,
                "haf_vote_count": int(vote_count),
                "haf_endpoint_covariance": float(endpoint_covariance),
                "points_xy": [[float(x), float(y)] for x, y in raw_points],
            }
        )

    decoded.sort(key=_stopline_prediction_sort_key, reverse=True)
    if int(max_segments) > 0:
        decoded = decoded[: int(max_segments)]
    return _dedupe_stop_line_predictions(decoded)


def _decode_stopline_axis_distance_segments(
    *,
    axis_distance: torch.Tensor | None,
    axis_direction: torch.Tensor | None,
    axis_valid_logits: torch.Tensor | None,
    mask_logits: torch.Tensor | None,
    selector_map_logits: torch.Tensor | None,
    center_logits: torch.Tensor | None,
    meta: dict[str, Any],
    valid_threshold: float,
    min_votes: int,
    cluster_endpoint_tolerance: float,
    max_endpoint_covariance: float,
    min_support_score: float,
    max_segments: int,
) -> list[dict[str, Any]]:
    if (
        not isinstance(axis_distance, torch.Tensor)
        or not isinstance(axis_direction, torch.Tensor)
        or not isinstance(axis_valid_logits, torch.Tensor)
    ):
        return []
    if (
        not _tensor_all_finite(axis_distance)
        or not _tensor_all_finite(axis_direction)
        or not _tensor_all_finite(axis_valid_logits)
    ):
        return []

    distance_map = axis_distance.detach().cpu()
    direction_map = axis_direction.detach().cpu()
    valid_map = axis_valid_logits.sigmoid().detach().cpu()
    if distance_map.ndim == 4:
        distance_map = distance_map.squeeze(0)
    if direction_map.ndim == 4:
        direction_map = direction_map.squeeze(0)
    if valid_map.ndim == 4:
        valid_map = valid_map.squeeze(0)
    if valid_map.ndim == 3:
        valid_map = valid_map.squeeze(0)
    if (
        distance_map.ndim != 3
        or direction_map.ndim != 3
        or int(distance_map.shape[0]) != 3
        or int(direction_map.shape[0]) != 2
        or valid_map.ndim != 2
    ):
        return []
    output_h, output_w = int(valid_map.shape[0]), int(valid_map.shape[1])
    if distance_map.shape[1:] != valid_map.shape or direction_map.shape[1:] != valid_map.shape:
        return []

    valid_np = valid_map.numpy().astype(np.float32)
    distance_np = distance_map.numpy().astype(np.float32)
    direction_np = direction_map.numpy().astype(np.float32)
    support_map = valid_np.copy()
    for optional_map in (mask_logits, selector_map_logits, center_logits):
        if isinstance(optional_map, torch.Tensor) and _tensor_all_finite(optional_map):
            item = optional_map.detach().cpu()
            if item.ndim == 3:
                item = item.squeeze(0)
            if item.ndim == 2 and tuple(item.shape) == tuple(valid_map.shape):
                support_map = np.maximum(support_map, item.sigmoid().numpy().astype(np.float32))

    rows, cols = np.nonzero(valid_np >= float(valid_threshold))
    if rows.size == 0:
        return []
    order = np.argsort(-valid_np[rows, cols])
    # Keep decode bounded when an undertrained valid map fires everywhere.
    order = order[:512]
    distance_norm = max(float(output_w), float(output_h), 1.0)
    votes: list[dict[str, Any]] = []
    for index in order.tolist():
        row_index = int(rows[index])
        col_index = int(cols[index])
        point = np.array([float(col_index), float(row_index)], dtype=np.float32)
        axis = direction_np[:, row_index, col_index].astype(np.float32)
        axis_norm = float(np.linalg.norm(axis))
        if not np.isfinite(axis_norm) or axis_norm <= 1.0e-6:
            continue
        axis = axis / axis_norm
        if axis[0] < 0.0 or (abs(float(axis[0])) <= 1.0e-6 and axis[1] < 0.0):
            axis = -axis
        normal = np.array([-float(axis[1]), float(axis[0])], dtype=np.float32)
        start_distance = float(np.clip(distance_np[0, row_index, col_index], -2.0, 2.0) * distance_norm)
        end_distance = float(np.clip(distance_np[1, row_index, col_index], -2.0, 2.0) * distance_norm)
        normal_distance = float(np.clip(distance_np[2, row_index, col_index], -0.5, 0.5) * distance_norm)
        line_point = point + normal * normal_distance
        start = line_point + axis * start_distance
        end = line_point + axis * end_distance
        start, end = _canonical_segment_endpoints(start, end)
        length = float(np.linalg.norm(end - start))
        if length < STOPLINE_MIN_COMPONENT_LENGTH:
            continue
        votes.append(
            {
                "start": start,
                "end": end,
                "weight": max(float(valid_np[row_index, col_index]), 1.0e-6),
            }
        )
    if not votes:
        return []

    clusters: list[dict[str, Any]] = []
    tolerance = max(float(cluster_endpoint_tolerance), 1.0e-6)
    for vote in votes:
        assigned = False
        for cluster in clusters:
            start_mean = np.asarray(cluster["start_sum"], dtype=np.float32) / max(float(cluster["weight_sum"]), 1.0e-6)
            end_mean = np.asarray(cluster["end_sum"], dtype=np.float32) / max(float(cluster["weight_sum"]), 1.0e-6)
            endpoint_error = max(
                float(np.linalg.norm(vote["start"] - start_mean)),
                float(np.linalg.norm(vote["end"] - end_mean)),
            )
            if endpoint_error > tolerance:
                continue
            weight = float(vote["weight"])
            cluster["start_sum"] = np.asarray(cluster["start_sum"], dtype=np.float32) + vote["start"] * weight
            cluster["end_sum"] = np.asarray(cluster["end_sum"], dtype=np.float32) + vote["end"] * weight
            cluster["weight_sum"] = float(cluster["weight_sum"]) + weight
            cluster["votes"].append(vote)
            assigned = True
            break
        if not assigned:
            weight = float(vote["weight"])
            clusters.append(
                {
                    "start_sum": vote["start"] * weight,
                    "end_sum": vote["end"] * weight,
                    "weight_sum": weight,
                    "votes": [vote],
                }
            )

    transform = transform_from_meta(meta)
    decoded: list[dict[str, Any]] = []
    for cluster in clusters:
        cluster_votes = list(cluster["votes"])
        vote_count = len(cluster_votes)
        if vote_count < int(min_votes):
            continue
        weight_sum = max(float(cluster["weight_sum"]), 1.0e-6)
        start = np.asarray(cluster["start_sum"], dtype=np.float32) / weight_sum
        end = np.asarray(cluster["end_sum"], dtype=np.float32) / weight_sum
        start, end = _canonical_segment_endpoints(start, end)
        endpoint_errors = [
            max(float(np.linalg.norm(vote["start"] - start)), float(np.linalg.norm(vote["end"] - end)))
            for vote in cluster_votes
        ]
        endpoint_covariance = float(np.mean(np.square(endpoint_errors))) if endpoint_errors else float("inf")
        if endpoint_covariance > float(max_endpoint_covariance):
            continue
        sample_count = 33
        ts = np.linspace(0.0, 1.0, sample_count, dtype=np.float32)
        segment_points = start[None, :] * (1.0 - ts[:, None]) + end[None, :] * ts[:, None]
        sample_cols = np.clip(np.rint(segment_points[:, 0]).astype(np.int64), 0, output_w - 1)
        sample_rows = np.clip(np.rint(segment_points[:, 1]).astype(np.int64), 0, output_h - 1)
        support_score = float(np.mean(support_map[sample_rows, sample_cols]))
        if support_score < float(min_support_score):
            continue
        grid_segment = np.stack([start, end], axis=0).astype(np.float32)
        network_segment = grid_segment.copy()
        network_segment[:, 0] = network_segment[:, 0] * (float(meta["network_hw"][1]) / float(output_w))
        network_segment[:, 1] = network_segment[:, 1] * (float(meta["network_hw"][0]) / float(output_h))
        network_points = sample_stop_line_centerline(
            clip_points(network_segment.tolist(), transform.network_hw),
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
        mean_vote_score = weight_sum / float(vote_count)
        vote_score = min(1.0, float(vote_count) / 16.0)
        covariance_score = 1.0 / (1.0 + max(endpoint_covariance, 0.0))
        score = 0.40 * mean_vote_score + 0.35 * support_score + 0.15 * vote_score + 0.10 * covariance_score
        decoded.append(
            {
                "allowed": True,
                "score": float(score),
                "center_score": float(mean_vote_score),
                "orientation_score": float(_stopline_orientation_score(raw_points)),
                "length": float(np.linalg.norm(end - start)),
                "thickness": 1.0,
                "axis_vote_count": int(vote_count),
                "axis_endpoint_covariance": float(endpoint_covariance),
                "axis_support_score": float(support_score),
                "points_xy": [[float(x), float(y)] for x, y in raw_points],
            }
        )

    decoded.sort(key=_stopline_prediction_sort_key, reverse=True)
    if int(max_segments) > 0:
        decoded = decoded[: int(max_segments)]
    return _dedupe_stop_line_predictions(decoded)


def _decode_stopline_segment_set(
    *,
    segment_logits: torch.Tensor | None,
    segment_points: torch.Tensor | None,
    segment_verifier_logits: torch.Tensor | None,
    meta: dict[str, Any],
    score_threshold: float,
    max_segments: int,
    verifier_score_weight: float = 0.0,
) -> list[dict[str, Any]]:
    if not isinstance(segment_logits, torch.Tensor) or not isinstance(segment_points, torch.Tensor):
        return []
    if not _tensor_all_finite(segment_logits) or not _tensor_all_finite(segment_points):
        return []
    logits = segment_logits.detach().cpu()
    verifier_logits = segment_verifier_logits.detach().cpu() if isinstance(segment_verifier_logits, torch.Tensor) else None
    points = segment_points.detach().cpu()
    if logits.ndim == 2:
        logits = logits.squeeze(0)
    if isinstance(verifier_logits, torch.Tensor) and verifier_logits.ndim == 2:
        verifier_logits = verifier_logits.squeeze(0)
    if points.ndim == 4:
        points = points.squeeze(0)
    if logits.ndim != 1 or points.ndim != 3 or points.shape[1:] != (2, 2):
        return []
    score_logits = logits
    weight = min(max(float(verifier_score_weight), 0.0), 1.0)
    if weight > 0.0 and isinstance(verifier_logits, torch.Tensor) and verifier_logits.shape == logits.shape:
        score_logits = logits * (1.0 - weight) + verifier_logits * weight
    scores = score_logits.sigmoid().numpy().astype(np.float32)
    verifier_scores = (
        verifier_logits.sigmoid().numpy().astype(np.float32)
        if isinstance(verifier_logits, torch.Tensor) and verifier_logits.shape == logits.shape
        else None
    )
    points_np = points.numpy().astype(np.float32)
    transform = transform_from_meta(meta)
    network_w = float(meta["network_hw"][1])
    network_h = float(meta["network_hw"][0])
    decoded: list[dict[str, Any]] = []
    for query_index, score in enumerate(scores.tolist()):
        if float(score) < float(score_threshold):
            continue
        normalized_segment = np.clip(points_np[query_index], 0.0, 1.0)
        network_segment = normalized_segment.copy()
        network_segment[:, 0] *= network_w
        network_segment[:, 1] *= network_h
        start, end = _canonical_segment_endpoints(network_segment[0], network_segment[1])
        network_points = sample_stop_line_centerline(
            clip_points(np.stack([start, end], axis=0).tolist(), transform.network_hw),
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
        length = float(np.linalg.norm(end - start))
        if length < STOPLINE_MIN_COMPONENT_LENGTH:
            continue
        decoded.append(
            {
                "allowed": True,
                "score": float(score),
                "center_score": float(score),
                "orientation_score": float(_stopline_orientation_score(raw_points)),
                "length": length,
                "thickness": 1.0,
                "segment_set_query_index": int(query_index),
                "segment_set_base_score": float(logits[query_index].sigmoid().item()),
                "segment_set_verifier_score": (
                    float(verifier_scores[query_index]) if verifier_scores is not None else float(score)
                ),
                "points_xy": [[float(x), float(y)] for x, y in raw_points],
            }
        )
    decoded.sort(key=_stopline_prediction_sort_key, reverse=True)
    decoded = _dedupe_stop_line_predictions(decoded)
    if int(max_segments) > 0:
        decoded = decoded[: int(max_segments)]
    return decoded


def _decode_stopline_endpoint_pair_segments(
    *,
    endpoint_logits: torch.Tensor | None,
    endpoint_offset: torch.Tensor | None,
    mask_logits: torch.Tensor | None,
    selector_map_logits: torch.Tensor | None,
    meta: dict[str, Any],
    score_threshold: float,
    topk: int,
    max_segments: int,
) -> list[dict[str, Any]]:
    if not isinstance(endpoint_logits, torch.Tensor) or not isinstance(endpoint_offset, torch.Tensor):
        return []
    if not _tensor_all_finite(endpoint_logits) or not _tensor_all_finite(endpoint_offset):
        return []
    logits = endpoint_logits.detach().cpu()
    offsets = endpoint_offset.detach().cpu()
    if logits.ndim == 4:
        logits = logits.squeeze(0)
    if offsets.ndim == 4:
        offsets = offsets.squeeze(0)
    if logits.ndim != 3 or offsets.ndim != 3 or int(logits.shape[0]) != 2 or int(offsets.shape[0]) != 4:
        return []
    if offsets.shape[1:] != logits.shape[1:]:
        return []

    scores_np = logits.sigmoid().numpy().astype(np.float32)
    offsets_np = offsets.numpy().astype(np.float32)
    output_h, output_w = int(scores_np.shape[1]), int(scores_np.shape[2])
    if output_h <= 0 or output_w <= 0:
        return []

    support_maps: list[np.ndarray] = []
    for candidate_map in (selector_map_logits, mask_logits):
        if not isinstance(candidate_map, torch.Tensor) or not _tensor_all_finite(candidate_map):
            continue
        map_tensor = candidate_map.detach().cpu()
        if map_tensor.ndim == 4:
            map_tensor = map_tensor.squeeze(0)
        if map_tensor.ndim == 3:
            map_tensor = map_tensor.squeeze(0)
        if map_tensor.ndim == 2 and tuple(map_tensor.shape) == (output_h, output_w):
            support_maps.append(map_tensor.sigmoid().numpy().astype(np.float32))

    def _top_endpoints(side_index: int) -> list[dict[str, Any]]:
        flat_scores = scores_np[side_index].reshape(-1)
        count = min(max(int(topk), 1), int(flat_scores.shape[0]))
        if count <= 0:
            return []
        top_indices = np.argpartition(-flat_scores, kth=count - 1)[:count]
        top_indices = top_indices[np.argsort(-flat_scores[top_indices])]
        endpoints: list[dict[str, Any]] = []
        for flat_index in top_indices.tolist():
            score = float(flat_scores[flat_index])
            if score < float(score_threshold):
                continue
            row_index = int(flat_index // output_w)
            col_index = int(flat_index % output_w)
            offset = offsets_np[side_index * 2 : side_index * 2 + 2, row_index, col_index]
            point = np.array([float(col_index), float(row_index)], dtype=np.float32) + offset.astype(np.float32)
            if not np.isfinite(point).all():
                continue
            point[0] = float(np.clip(point[0], 0.0, float(output_w - 1)))
            point[1] = float(np.clip(point[1], 0.0, float(output_h - 1)))
            endpoints.append({"point": point, "score": score, "row": row_index, "col": col_index})
        return endpoints

    left_endpoints = _top_endpoints(0)
    right_endpoints = _top_endpoints(1)
    if not left_endpoints or not right_endpoints:
        return []

    transform = transform_from_meta(meta)
    decoded: list[dict[str, Any]] = []
    for left in left_endpoints:
        for right in right_endpoints:
            start, end = _canonical_segment_endpoints(left["point"], right["point"])
            length = float(np.linalg.norm(end - start))
            if length < STOPLINE_MIN_COMPONENT_LENGTH:
                continue
            support_score = 0.0
            if support_maps:
                sample_count = 16
                xs = np.linspace(float(start[0]), float(end[0]), sample_count)
                ys = np.linspace(float(start[1]), float(end[1]), sample_count)
                cols = np.clip(np.rint(xs).astype(np.int64), 0, output_w - 1)
                rows = np.clip(np.rint(ys).astype(np.int64), 0, output_h - 1)
                support_values = [float(support_map[rows, cols].mean()) for support_map in support_maps]
                support_score = float(max(support_values))
            endpoint_score = float(np.sqrt(max(float(left["score"]) * float(right["score"]), 0.0)))
            score = 0.65 * endpoint_score + 0.35 * support_score
            if score < float(score_threshold):
                continue
            grid_segment = np.stack([start, end], axis=0).astype(np.float32)
            network_segment = grid_segment.copy()
            network_segment[:, 0] = network_segment[:, 0] * (float(meta["network_hw"][1]) / float(output_w))
            network_segment[:, 1] = network_segment[:, 1] * (float(meta["network_hw"][0]) / float(output_h))
            network_points = sample_stop_line_centerline(
                clip_points(network_segment.tolist(), transform.network_hw),
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
            decoded.append(
                {
                    "allowed": True,
                    "score": float(score),
                    "center_score": float(support_score),
                    "orientation_score": float(_stopline_orientation_score(raw_points)),
                    "length": length,
                    "thickness": 1.0,
                    "endpoint_pair_score": float(endpoint_score),
                    "endpoint_pair_support_score": float(support_score),
                    "points_xy": [[float(x), float(y)] for x, y in raw_points],
                }
            )
    decoded.sort(key=_stopline_prediction_sort_key, reverse=True)
    decoded = _dedupe_stop_line_predictions(decoded)
    if int(max_segments) > 0:
        decoded = decoded[: int(max_segments)]
    return decoded


def _decode_stopline_endpoint_haf_consensus_segments(
    *,
    endpoint_logits: torch.Tensor | None,
    endpoint_offset: torch.Tensor | None,
    haf_endpoint: torch.Tensor | None,
    haf_valid_logits: torch.Tensor | None,
    mask_logits: torch.Tensor | None,
    selector_map_logits: torch.Tensor | None,
    meta: dict[str, Any],
    score_threshold: float,
    topk: int,
    haf_valid_threshold: float,
    min_votes: int,
    max_endpoint_error: float,
    max_endpoint_covariance: float,
    max_segments: int,
) -> list[dict[str, Any]]:
    if (
        not isinstance(endpoint_logits, torch.Tensor)
        or not isinstance(endpoint_offset, torch.Tensor)
        or not isinstance(haf_endpoint, torch.Tensor)
        or not isinstance(haf_valid_logits, torch.Tensor)
    ):
        return []
    if (
        not _tensor_all_finite(endpoint_logits)
        or not _tensor_all_finite(endpoint_offset)
        or not _tensor_all_finite(haf_endpoint)
        or not _tensor_all_finite(haf_valid_logits)
    ):
        return []
    logits = endpoint_logits.detach().cpu()
    offsets = endpoint_offset.detach().cpu()
    endpoint_votes = haf_endpoint.detach().cpu()
    valid_map = haf_valid_logits.sigmoid().detach().cpu()
    if logits.ndim == 4:
        logits = logits.squeeze(0)
    if offsets.ndim == 4:
        offsets = offsets.squeeze(0)
    if endpoint_votes.ndim == 4:
        endpoint_votes = endpoint_votes.squeeze(0)
    if valid_map.ndim == 4:
        valid_map = valid_map.squeeze(0)
    if valid_map.ndim == 3:
        valid_map = valid_map.squeeze(0)
    if (
        logits.ndim != 3
        or offsets.ndim != 3
        or endpoint_votes.ndim != 3
        or valid_map.ndim != 2
        or int(logits.shape[0]) != 2
        or int(offsets.shape[0]) != 4
        or int(endpoint_votes.shape[0]) != 4
    ):
        return []
    if offsets.shape[1:] != logits.shape[1:] or endpoint_votes.shape[1:] != logits.shape[1:]:
        return []
    output_h, output_w = int(valid_map.shape[0]), int(valid_map.shape[1])
    if tuple(logits.shape[1:]) != (output_h, output_w):
        return []

    scores_np = logits.sigmoid().numpy().astype(np.float32)
    offsets_np = offsets.numpy().astype(np.float32)
    endpoint_vote_np = endpoint_votes.numpy().astype(np.float32)
    valid_np = valid_map.numpy().astype(np.float32)
    support_maps: list[np.ndarray] = []
    for candidate_map in (selector_map_logits, mask_logits):
        if not isinstance(candidate_map, torch.Tensor) or not _tensor_all_finite(candidate_map):
            continue
        map_tensor = candidate_map.detach().cpu()
        if map_tensor.ndim == 4:
            map_tensor = map_tensor.squeeze(0)
        if map_tensor.ndim == 3:
            map_tensor = map_tensor.squeeze(0)
        if map_tensor.ndim == 2 and tuple(map_tensor.shape) == (output_h, output_w):
            support_maps.append(map_tensor.sigmoid().numpy().astype(np.float32))

    def _top_endpoints(side_index: int) -> list[dict[str, Any]]:
        flat_scores = scores_np[side_index].reshape(-1)
        count = min(max(int(topk), 1), int(flat_scores.shape[0]))
        if count <= 0:
            return []
        top_indices = np.argpartition(-flat_scores, kth=count - 1)[:count]
        top_indices = top_indices[np.argsort(-flat_scores[top_indices])]
        endpoints: list[dict[str, Any]] = []
        for flat_index in top_indices.tolist():
            score = float(flat_scores[flat_index])
            row_index = int(flat_index // output_w)
            col_index = int(flat_index % output_w)
            offset = offsets_np[side_index * 2 : side_index * 2 + 2, row_index, col_index]
            point = np.array([float(col_index), float(row_index)], dtype=np.float32) + offset.astype(np.float32)
            if not np.isfinite(point).all():
                continue
            point[0] = float(np.clip(point[0], 0.0, float(output_w - 1)))
            point[1] = float(np.clip(point[1], 0.0, float(output_h - 1)))
            endpoints.append({"point": point, "score": score})
        return endpoints

    left_endpoints = _top_endpoints(0)
    right_endpoints = _top_endpoints(1)
    if not left_endpoints or not right_endpoints:
        return []

    transform = transform_from_meta(meta)
    sample_count = 24
    decoded: list[dict[str, Any]] = []
    for left in left_endpoints:
        for right in right_endpoints:
            start, end = _canonical_segment_endpoints(left["point"], right["point"])
            length = float(np.linalg.norm(end - start))
            if length < STOPLINE_MIN_COMPONENT_LENGTH:
                continue
            ts = np.linspace(0.0, 1.0, sample_count, dtype=np.float32)
            sample_points = start[None, :] * (1.0 - ts[:, None]) + end[None, :] * ts[:, None]
            sample_cols = np.clip(np.rint(sample_points[:, 0]).astype(np.int64), 0, output_w - 1)
            sample_rows = np.clip(np.rint(sample_points[:, 1]).astype(np.int64), 0, output_h - 1)
            candidate_votes: list[dict[str, Any]] = []
            for sample_row, sample_col in zip(sample_rows.tolist(), sample_cols.tolist()):
                valid_score = float(valid_np[sample_row, sample_col])
                if valid_score < float(haf_valid_threshold):
                    continue
                point = np.array([float(sample_col), float(sample_row)], dtype=np.float32)
                vote_start = point + endpoint_vote_np[0:2, sample_row, sample_col]
                vote_end = point + endpoint_vote_np[2:4, sample_row, sample_col]
                if not np.isfinite(vote_start).all() or not np.isfinite(vote_end).all():
                    continue
                vote_start, vote_end = _canonical_segment_endpoints(vote_start, vote_end)
                endpoint_error = max(
                    float(np.linalg.norm(vote_start - start)),
                    float(np.linalg.norm(vote_end - end)),
                )
                if endpoint_error > float(max_endpoint_error):
                    continue
                candidate_votes.append(
                    {
                        "start": vote_start,
                        "end": vote_end,
                        "weight": max(valid_score, 1.0e-6),
                    }
                )
            vote_count = len(candidate_votes)
            if vote_count < int(min_votes):
                continue
            weight_sum = sum(float(vote["weight"]) for vote in candidate_votes)
            if weight_sum <= 0.0:
                continue
            refined_start = sum(
                np.asarray(vote["start"], dtype=np.float32) * float(vote["weight"])
                for vote in candidate_votes
            ) / weight_sum
            refined_end = sum(
                np.asarray(vote["end"], dtype=np.float32) * float(vote["weight"])
                for vote in candidate_votes
            ) / weight_sum
            refined_start, refined_end = _canonical_segment_endpoints(refined_start, refined_end)
            endpoint_errors = [
                max(
                    float(np.linalg.norm(np.asarray(vote["start"], dtype=np.float32) - refined_start)),
                    float(np.linalg.norm(np.asarray(vote["end"], dtype=np.float32) - refined_end)),
                )
                for vote in candidate_votes
            ]
            endpoint_covariance = float(np.mean(np.square(endpoint_errors))) if endpoint_errors else float("inf")
            if endpoint_covariance > float(max_endpoint_covariance):
                continue
            refined_length = float(np.linalg.norm(refined_end - refined_start))
            if refined_length < STOPLINE_MIN_COMPONENT_LENGTH:
                continue
            support_score = 0.0
            if support_maps:
                support_values = [float(support_map[sample_rows, sample_cols].mean()) for support_map in support_maps]
                support_score = float(max(support_values))
            endpoint_score = float(np.sqrt(max(float(left["score"]) * float(right["score"]), 0.0)))
            mean_vote_score = float(weight_sum / max(float(vote_count), 1.0))
            consensus_score = min(1.0, float(vote_count) / float(sample_count))
            covariance_score = 1.0 / (1.0 + max(endpoint_covariance, 0.0))
            score = (
                0.35 * endpoint_score
                + 0.25 * mean_vote_score
                + 0.20 * support_score
                + 0.15 * consensus_score
                + 0.05 * covariance_score
            )
            if score < float(score_threshold):
                continue
            grid_segment = np.stack([refined_start, refined_end], axis=0).astype(np.float32)
            network_segment = grid_segment.copy()
            network_segment[:, 0] = network_segment[:, 0] * (float(meta["network_hw"][1]) / float(output_w))
            network_segment[:, 1] = network_segment[:, 1] * (float(meta["network_hw"][0]) / float(output_h))
            network_points = sample_stop_line_centerline(
                clip_points(network_segment.tolist(), transform.network_hw),
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
            decoded.append(
                {
                    "allowed": True,
                    "score": float(score),
                    "center_score": float(mean_vote_score),
                    "orientation_score": float(_stopline_orientation_score(raw_points)),
                    "length": refined_length,
                    "thickness": 1.0,
                    "endpoint_haf_endpoint_score": float(endpoint_score),
                    "endpoint_haf_support_score": float(support_score),
                    "endpoint_haf_vote_count": int(vote_count),
                    "endpoint_haf_endpoint_covariance": float(endpoint_covariance),
                    "points_xy": [[float(x), float(y)] for x, y in raw_points],
                }
            )
    decoded.sort(key=_stopline_prediction_sort_key, reverse=True)
    decoded = _dedupe_stop_line_predictions(decoded)
    if int(max_segments) > 0:
        decoded = decoded[: int(max_segments)]
    return decoded


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
    min_component_pixels: int = STOPLINE_MIN_COMPONENT_PIXELS,
    max_components: int = 3,
    min_bbox_area_px: float = 0.0,
    min_bbox_aspect: float = 0.0,
    min_instance_score: float = 0.0,
    presence_logits: torch.Tensor | None = None,
    presence_threshold: float = 0.0,
    component_gate_source: str = "center",
) -> list[dict[str, Any]]:
    if float(presence_threshold) > 0.0 and isinstance(presence_logits, torch.Tensor):
        if not _tensor_all_finite(presence_logits):
            return []
        presence_score = float(presence_logits.reshape(-1)[0].sigmoid().detach().cpu().item())
        if presence_score < float(presence_threshold):
            return []
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
    selector_probs = None
    if isinstance(selector_map_logits, torch.Tensor) and _tensor_all_finite(selector_map_logits):
        selector_probs = selector_map_logits.sigmoid().squeeze(0).detach().cpu().numpy()
        if selector_probs.ndim == 3:
            selector_probs = selector_probs.squeeze(0)

    gate_source = str(component_gate_source or "center").strip().lower()
    if gate_source not in {"center", "selector", "max"}:
        raise ValueError(f"unsupported stop_line_component_gate_source: {component_gate_source}")
    gate_probs = center_probs
    if gate_source == "selector" and selector_probs is not None:
        gate_probs = selector_probs
    elif gate_source == "max" and selector_probs is not None:
        gate_probs = selector_probs if center_probs is None else np.maximum(center_probs, selector_probs)

    allowed_labels = _stopline_allowed_labels(labels, gate_probs, row_probs=row_probs)

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
        if len(rows) < max(STOPLINE_MIN_COMPONENT_PIXELS, int(min_component_pixels)):
            continue
        component_points = np.stack([cols.astype(np.float32), rows.astype(np.float32)], axis=1)
        mask_values = mask_probs[rows, cols]
        center_anchor = None
        if gate_probs is not None:
            center_anchor = _stopline_component_anchor(
                rows,
                cols,
                mask_values=mask_values,
                center_probs=gate_probs,
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
        if gate_probs is not None:
            center_score = float(gate_probs[rows, cols].max())
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
    predictions = _filter_stop_line_predictions(
        predictions,
        min_bbox_area_px=min_bbox_area_px,
        min_bbox_aspect=min_bbox_aspect,
        min_instance_score=min_instance_score,
    )
    return predictions[: max(1, int(max_components))]


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
    segfirst_lanes = vectorize_lane_segfirst_maps(
        maps,
        meta=meta,
        config=LaneSegFirstVectorizerConfig(
            track_mode=str(config.lane_segfirst_track_mode),
            centerline_threshold=float(config.lane_obj_threshold),
            min_polyline_length_px=float(config.lane_segfirst_min_polyline_length_px),
            min_polyline_bottom_y_fraction=float(config.lane_segfirst_min_polyline_bottom_y_fraction),
            semantic_vote_mode=str(config.lane_segfirst_semantic_vote_mode),
            max_row_gap=int(config.lane_segfirst_max_row_gap),
            max_link_dx=float(config.lane_segfirst_max_link_dx),
            max_turn_degrees=float(config.lane_segfirst_max_turn_degrees),
            seed_threshold=float(config.lane_segfirst_seed_threshold),
            seed_trace_max_seeds=int(config.lane_segfirst_seed_trace_max_seeds),
            center_offset_enabled=bool(config.lane_segfirst_center_offset_enabled),
            center_offset_max_shift_px=float(config.lane_segfirst_center_offset_max_shift_px),
            center_offset_min_support_score=float(config.lane_segfirst_center_offset_min_support_score),
        ),
    )
    conditional_rows = predictions.get("lane_conditional_rows")
    if not bool(config.lane_conditional_row_enabled) or not isinstance(conditional_rows, torch.Tensor):
        return segfirst_lanes
    conditional_lanes = _decode_lane_rows(
        conditional_rows[batch_index],
        meta=meta,
        config=config,
    )
    merge_mode = str(config.lane_conditional_row_merge_mode).strip().lower()
    if merge_mode == "replace":
        return conditional_lanes
    if merge_mode == "append":
        return _dedupe_lane_predictions_by_distance([*segfirst_lanes, *conditional_lanes])
    raise ValueError(f"unsupported lane_conditional_row_merge_mode: {config.lane_conditional_row_merge_mode}")


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
    min_component_pixels: int = STOPLINE_MIN_COMPONENT_PIXELS,
    max_components: int = 3,
    min_bbox_area_px: float = 0.0,
    min_bbox_aspect: float = 0.0,
    min_instance_score: float = 0.0,
    mask_logits: torch.Tensor | None = None,
    selector_map_logits: torch.Tensor | None = None,
    center_logits: torch.Tensor | None = None,
    midpoint_logits: torch.Tensor | None = None,
    center_offset: torch.Tensor | None = None,
    row_logits: torch.Tensor | None = None,
    x_logits: torch.Tensor | None = None,
    angle: torch.Tensor | None = None,
    half_length: torch.Tensor | None = None,
    haf_endpoint: torch.Tensor | None = None,
    haf_valid_logits: torch.Tensor | None = None,
    axis_distance: torch.Tensor | None = None,
    axis_direction: torch.Tensor | None = None,
    axis_valid_logits: torch.Tensor | None = None,
    endpoint_logits: torch.Tensor | None = None,
    endpoint_offset: torch.Tensor | None = None,
    endpoint_pair_logits: torch.Tensor | None = None,
    endpoint_pair_points: torch.Tensor | None = None,
    endpoint_pair_verifier_logits: torch.Tensor | None = None,
    segment_logits: torch.Tensor | None = None,
    segment_points: torch.Tensor | None = None,
    segment_verifier_logits: torch.Tensor | None = None,
    context_segment_logits: torch.Tensor | None = None,
    context_segment_points: torch.Tensor | None = None,
    context_segment_verifier_logits: torch.Tensor | None = None,
    axis_segment_logits: torch.Tensor | None = None,
    axis_segment_points: torch.Tensor | None = None,
    axis_segment_verifier_logits: torch.Tensor | None = None,
    patch_segment_logits: torch.Tensor | None = None,
    patch_segment_points: torch.Tensor | None = None,
    patch_segment_verifier_logits: torch.Tensor | None = None,
    segment_set_enabled: bool = False,
    segment_set_score_threshold: float = 0.50,
    segment_set_max_segments: int = 3,
    segment_verifier_score_weight: float = 0.0,
    context_segment_set_enabled: bool = False,
    context_segment_set_score_threshold: float = 0.50,
    context_segment_set_max_segments: int = 3,
    context_segment_verifier_score_weight: float = 0.0,
    axis_segment_set_enabled: bool = False,
    axis_segment_set_score_threshold: float = 0.50,
    axis_segment_set_max_segments: int = 3,
    axis_segment_verifier_score_weight: float = 0.0,
    patch_segment_set_enabled: bool = False,
    patch_segment_set_score_threshold: float = 0.50,
    patch_segment_set_max_segments: int = 3,
    patch_segment_verifier_score_weight: float = 0.0,
    haf_enabled: bool = False,
    haf_valid_threshold: float = 0.50,
    haf_min_votes: int = 4,
    haf_cluster_endpoint_tolerance: float = 3.0,
    haf_max_endpoint_covariance: float = 9.0,
    haf_max_segments: int = 3,
    axis_distance_enabled: bool = False,
    axis_distance_valid_threshold: float = 0.75,
    axis_distance_min_votes: int = 3,
    axis_distance_cluster_endpoint_tolerance: float = 4.0,
    axis_distance_max_endpoint_covariance: float = 16.0,
    axis_distance_min_support_score: float = 0.35,
    axis_distance_max_segments: int = 3,
    endpoint_pair_enabled: bool = False,
    endpoint_pair_score_threshold: float = 0.55,
    endpoint_pair_topk: int = 8,
    endpoint_pair_max_segments: int = 3,
    endpoint_haf_consensus_enabled: bool = False,
    endpoint_haf_consensus_score_threshold: float = 0.55,
    endpoint_haf_consensus_topk: int = 8,
    endpoint_haf_consensus_haf_valid_threshold: float = 0.65,
    endpoint_haf_consensus_min_votes: int = 4,
    endpoint_haf_consensus_max_endpoint_error: float = 8.0,
    endpoint_haf_consensus_max_endpoint_covariance: float = 32.0,
    endpoint_haf_consensus_max_segments: int = 3,
    endpoint_pair_segment_enabled: bool = False,
    endpoint_pair_segment_score_threshold: float = 0.50,
    endpoint_pair_segment_max_segments: int = 3,
    endpoint_pair_verifier_score_weight: float = 0.0,
    projection_comp_enabled: bool = False,
    projection_comp_proposal_source: str = "max",
    projection_comp_min_gap: float = 4.0,
    projection_comp_topk: int = 50,
    projection_comp_union_min_score: float = 0.80,
    projection_comp_single_min_score: float = 0.90,
    projection_comp_angle_threshold_deg: float = 16.0,
    projection_comp_offset_threshold_px: float = 48.0,
    projection_comp_min_cluster_count: int = 2,
    projection_comp_projection_gap_px: float = 320.0,
    projection_comp_max_predictions: int = 2,
    projection_comp_second_min_score: float = 0.0,
    projection_comp_second_min_fragment_count: int = 5,
    projection_comp_second_min_length_ratio: float = 0.0,
    presence_logits: torch.Tensor | None = None,
    presence_threshold: float = 0.0,
    component_gate_source: str = "center",
) -> list[dict[str, Any]]:
    if float(presence_threshold) > 0.0 and isinstance(presence_logits, torch.Tensor):
        if not _tensor_all_finite(presence_logits):
            return []
        presence_score = float(presence_logits.reshape(-1)[0].sigmoid().detach().cpu().item())
        if presence_score < float(presence_threshold):
            return []
    if bool(projection_comp_enabled):
        return _decode_stopline_projection_competition(
            mask_logits=mask_logits,
            center_logits=center_logits,
            midpoint_logits=midpoint_logits,
            selector_map_logits=selector_map_logits,
            center_offset=center_offset,
            angle=angle,
            meta=meta,
            proposal_source=str(projection_comp_proposal_source),
            min_gap=float(projection_comp_min_gap),
            top_k=int(projection_comp_topk),
            union_min_score=float(projection_comp_union_min_score),
            single_min_score=float(projection_comp_single_min_score),
            angle_threshold_deg=float(projection_comp_angle_threshold_deg),
            offset_threshold_px=float(projection_comp_offset_threshold_px),
            min_cluster_count=int(projection_comp_min_cluster_count),
            projection_gap_px=float(projection_comp_projection_gap_px),
            max_predictions=int(projection_comp_max_predictions),
            second_min_score=float(projection_comp_second_min_score),
            second_min_fragment_count=int(projection_comp_second_min_fragment_count),
            second_min_length_ratio=float(projection_comp_second_min_length_ratio),
        )
    segment_set_decoded: list[dict[str, Any]] = []
    if bool(segment_set_enabled):
        segment_set_decoded = _decode_stopline_segment_set(
            segment_logits=segment_logits,
            segment_points=segment_points,
            segment_verifier_logits=segment_verifier_logits,
            meta=meta,
            score_threshold=float(segment_set_score_threshold),
            max_segments=int(segment_set_max_segments),
            verifier_score_weight=float(segment_verifier_score_weight),
        )
    if bool(context_segment_set_enabled):
        segment_set_decoded.extend(
            _decode_stopline_segment_set(
                segment_logits=context_segment_logits,
                segment_points=context_segment_points,
                segment_verifier_logits=context_segment_verifier_logits,
                meta=meta,
                score_threshold=float(context_segment_set_score_threshold),
                max_segments=int(context_segment_set_max_segments),
                verifier_score_weight=float(context_segment_verifier_score_weight),
            )
        )
    if bool(axis_segment_set_enabled):
        segment_set_decoded.extend(
            _decode_stopline_segment_set(
                segment_logits=axis_segment_logits,
                segment_points=axis_segment_points,
                segment_verifier_logits=axis_segment_verifier_logits,
                meta=meta,
                score_threshold=float(axis_segment_set_score_threshold),
                max_segments=int(axis_segment_set_max_segments),
                verifier_score_weight=float(axis_segment_verifier_score_weight),
            )
        )
    if bool(patch_segment_set_enabled):
        segment_set_decoded.extend(
            _decode_stopline_segment_set(
                segment_logits=patch_segment_logits,
                segment_points=patch_segment_points,
                segment_verifier_logits=patch_segment_verifier_logits,
                meta=meta,
                score_threshold=float(patch_segment_set_score_threshold),
                max_segments=int(patch_segment_set_max_segments),
                verifier_score_weight=float(patch_segment_verifier_score_weight),
            )
        )
    if bool(endpoint_pair_enabled):
        segment_set_decoded.extend(
            _decode_stopline_endpoint_pair_segments(
                endpoint_logits=endpoint_logits,
                endpoint_offset=endpoint_offset,
                mask_logits=mask_logits,
                selector_map_logits=selector_map_logits,
                meta=meta,
                score_threshold=float(endpoint_pair_score_threshold),
                topk=int(endpoint_pair_topk),
                max_segments=int(endpoint_pair_max_segments),
            )
        )
    if bool(endpoint_haf_consensus_enabled):
        decoded = _decode_stopline_endpoint_haf_consensus_segments(
            endpoint_logits=endpoint_logits,
            endpoint_offset=endpoint_offset,
            haf_endpoint=haf_endpoint,
            haf_valid_logits=haf_valid_logits,
            mask_logits=mask_logits,
            selector_map_logits=selector_map_logits,
            meta=meta,
            score_threshold=float(endpoint_haf_consensus_score_threshold),
            topk=int(endpoint_haf_consensus_topk),
            haf_valid_threshold=float(endpoint_haf_consensus_haf_valid_threshold),
            min_votes=int(endpoint_haf_consensus_min_votes),
            max_endpoint_error=float(endpoint_haf_consensus_max_endpoint_error),
            max_endpoint_covariance=float(endpoint_haf_consensus_max_endpoint_covariance),
            max_segments=int(endpoint_haf_consensus_max_segments),
        )
        if decoded:
            return decoded
    if bool(endpoint_pair_segment_enabled):
        segment_set_decoded.extend(
            _decode_stopline_segment_set(
                segment_logits=endpoint_pair_logits,
                segment_points=endpoint_pair_points,
                segment_verifier_logits=endpoint_pair_verifier_logits,
                meta=meta,
                score_threshold=float(endpoint_pair_segment_score_threshold),
                max_segments=int(endpoint_pair_segment_max_segments),
                verifier_score_weight=float(endpoint_pair_verifier_score_weight),
            )
        )
    if bool(axis_distance_enabled):
        decoded = _decode_stopline_axis_distance_segments(
            axis_distance=axis_distance,
            axis_direction=axis_direction,
            axis_valid_logits=axis_valid_logits,
            mask_logits=mask_logits,
            selector_map_logits=selector_map_logits,
            center_logits=center_logits,
            meta=meta,
            valid_threshold=float(axis_distance_valid_threshold),
            min_votes=int(axis_distance_min_votes),
            cluster_endpoint_tolerance=float(axis_distance_cluster_endpoint_tolerance),
            max_endpoint_covariance=float(axis_distance_max_endpoint_covariance),
            min_support_score=float(axis_distance_min_support_score),
            max_segments=int(axis_distance_max_segments),
        )
        if decoded:
            return decoded
    if bool(haf_enabled):
        decoded = _decode_stopline_haf_consensus_segments(
            haf_endpoint=haf_endpoint,
            haf_valid_logits=haf_valid_logits,
            meta=meta,
            valid_threshold=float(haf_valid_threshold),
            min_votes=int(haf_min_votes),
            cluster_endpoint_tolerance=float(haf_cluster_endpoint_tolerance),
            max_endpoint_covariance=float(haf_max_endpoint_covariance),
            max_segments=int(haf_max_segments),
        )
        if decoded:
            return decoded
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
            min_component_pixels=min_component_pixels,
            max_components=max_components,
            min_bbox_area_px=min_bbox_area_px,
            min_bbox_aspect=min_bbox_aspect,
            min_instance_score=min_instance_score,
            presence_logits=presence_logits,
            presence_threshold=presence_threshold,
            component_gate_source=component_gate_source,
        )
        if decoded:
            if segment_set_decoded:
                merged = _dedupe_stop_line_predictions(decoded + segment_set_decoded)
                merged.sort(key=_stopline_prediction_sort_key, reverse=True)
                return merged[
                    : max(
                        1,
                        int(max_components),
                        int(segment_set_max_segments),
                        int(context_segment_set_max_segments),
                        int(axis_segment_set_max_segments),
                        int(patch_segment_set_max_segments),
                        int(endpoint_pair_max_segments),
                        int(endpoint_haf_consensus_max_segments),
                        int(endpoint_pair_segment_max_segments),
                    )
                ]
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
    if segment_set_decoded:
        predictions = predictions + segment_set_decoded
    predictions = _dedupe_stop_line_predictions(predictions)
    if not segment_set_decoded:
        return predictions
    predictions.sort(key=_stopline_prediction_sort_key, reverse=True)
    return predictions[
        : max(
            1,
            int(max_components),
            int(segment_set_max_segments),
            int(axis_segment_set_max_segments),
            int(patch_segment_set_max_segments),
            int(endpoint_pair_max_segments),
            int(endpoint_haf_consensus_max_segments),
            int(endpoint_pair_segment_max_segments),
        )
    ]


def _decode_crosswalk_rows(
    rows: torch.Tensor,
    *,
    meta: dict[str, Any],
    obj_threshold: float,
    mask_binary_threshold: float = 0.5,
    min_component_pixels: int = 4,
    max_components: int = 0,
    min_polygon_area_px: float = 0.0,
    min_bbox_aspect: float = 0.0,
    polygon_mode: str = "rect",
    mask_logits: torch.Tensor | None = None,
    center_logits: torch.Tensor | None = None,
) -> list[dict[str, Any]]:
    filter_fallback = isinstance(mask_logits, torch.Tensor)
    if isinstance(mask_logits, torch.Tensor):
        decoded = _crosswalk_mask_to_polygon(
            mask_logits,
            center_logits,
            meta=meta,
            obj_threshold=obj_threshold,
            mask_binary_threshold=mask_binary_threshold,
            min_component_pixels=min_component_pixels,
            max_components=max_components,
            min_polygon_area_px=min_polygon_area_px,
            min_bbox_aspect=min_bbox_aspect,
            polygon_mode=polygon_mode,
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
        if filter_fallback:
            if float(min_polygon_area_px) > 0.0 and _polygon_area(raw_points) < float(min_polygon_area_px):
                continue
            if float(min_bbox_aspect) > 0.0 and _bbox_aspect(raw_points) < float(min_bbox_aspect):
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
    stop_line_presence_logits = predictions.get("stop_line_presence_logits")
    stop_line_center_logits = predictions.get("stop_line_center_logits")
    stop_line_midpoint_logits = predictions.get("stop_line_midpoint_logits")
    stop_line_center_offset = predictions.get("stop_line_center_offset")
    stop_line_angle = predictions.get("stop_line_angle")
    stop_line_half_length = predictions.get("stop_line_half_length")
    stop_line_haf_endpoint = predictions.get("stop_line_haf_endpoint")
    stop_line_haf_valid_logits = predictions.get("stop_line_haf_valid_logits")
    stop_line_axis_distance = predictions.get("stop_line_axis_distance")
    stop_line_axis_direction = predictions.get("stop_line_axis_direction")
    stop_line_axis_valid_logits = predictions.get("stop_line_axis_valid_logits")
    stop_line_endpoint_logits = predictions.get("stop_line_endpoint_logits")
    stop_line_endpoint_offset = predictions.get("stop_line_endpoint_offset")
    stop_line_endpoint_pair_logits = predictions.get("stop_line_endpoint_pair_logits")
    stop_line_endpoint_pair_points = predictions.get("stop_line_endpoint_pair_points")
    stop_line_endpoint_pair_verifier_logits = predictions.get("stop_line_endpoint_pair_verifier_logits")
    stop_line_segment_logits = predictions.get("stop_line_segment_logits")
    stop_line_segment_points = predictions.get("stop_line_segment_points")
    stop_line_segment_verifier_logits = predictions.get("stop_line_segment_verifier_logits")
    stop_line_context_segment_logits = predictions.get("stop_line_context_segment_logits")
    stop_line_context_segment_points = predictions.get("stop_line_context_segment_points")
    stop_line_context_segment_verifier_logits = predictions.get("stop_line_context_segment_verifier_logits")
    stop_line_axis_segment_logits = predictions.get("stop_line_axis_segment_logits")
    stop_line_axis_segment_points = predictions.get("stop_line_axis_segment_points")
    stop_line_axis_segment_verifier_logits = predictions.get("stop_line_axis_segment_verifier_logits")
    stop_line_patch_segment_logits = predictions.get("stop_line_patch_segment_logits")
    stop_line_patch_segment_points = predictions.get("stop_line_patch_segment_points")
    stop_line_patch_segment_verifier_logits = predictions.get("stop_line_patch_segment_verifier_logits")
    crosswalk_mask_logits = predictions.get("crosswalk_mask_logits")
    crosswalk_center_logits = predictions.get("crosswalk_center_logits")
    feature_shapes = predictions.get("det_feature_shapes")
    feature_strides = predictions.get("det_feature_strides")

    if not isinstance(feature_shapes, list) or not isinstance(feature_strides, list):
        raise ValueError("postprocess requires det_feature_shapes and det_feature_strides metadata")

    batch_predictions: list[dict[str, Any]] = []
    for batch_index, sample_meta in enumerate(meta):
        lane_predictions = _filter_lane_predictions(
            _decode_segfirst_lane_rows(
                predictions,
                batch_index=batch_index,
                meta=sample_meta,
                config=config,
            ),
            min_bbox_area_px=config.lane_segfirst_min_bbox_area_px,
            max_bbox_aspect=config.lane_segfirst_max_bbox_aspect,
        )
        if not lane_predictions:
            lane_predictions = _filter_lane_predictions(
                _decode_lane_rows(
                    lane_pred[batch_index],
                    meta=sample_meta,
                    config=config,
                ),
                min_bbox_area_px=config.lane_segfirst_min_bbox_area_px,
                max_bbox_aspect=config.lane_segfirst_max_bbox_aspect,
            )
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
                "lanes": lane_predictions,
                "stop_lines": _decode_stop_line_rows(
                    stop_line_pred[batch_index],
                    meta=sample_meta,
                    obj_threshold=config.stop_line_obj_threshold,
                    mask_binary_threshold=config.stop_line_mask_binary_threshold,
                    min_component_pixels=config.stop_line_min_component_pixels,
                    max_components=config.stop_line_max_components,
                    min_bbox_area_px=config.stop_line_min_bbox_area_px,
                    min_bbox_aspect=config.stop_line_min_bbox_aspect,
                    min_instance_score=config.stop_line_min_instance_score,
                    presence_logits=(
                        stop_line_presence_logits[batch_index]
                        if isinstance(stop_line_presence_logits, torch.Tensor)
                        else None
                    ),
                    presence_threshold=config.stop_line_presence_threshold,
                    component_gate_source=config.stop_line_component_gate_source,
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
                    midpoint_logits=(
                        stop_line_midpoint_logits[batch_index]
                        if isinstance(stop_line_midpoint_logits, torch.Tensor)
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
                    haf_endpoint=(
                        stop_line_haf_endpoint[batch_index]
                        if isinstance(stop_line_haf_endpoint, torch.Tensor)
                        else None
                    ),
                    haf_valid_logits=(
                        stop_line_haf_valid_logits[batch_index]
                        if isinstance(stop_line_haf_valid_logits, torch.Tensor)
                        else None
                    ),
                    axis_distance=(
                        stop_line_axis_distance[batch_index]
                        if isinstance(stop_line_axis_distance, torch.Tensor)
                        else None
                    ),
                    axis_direction=(
                        stop_line_axis_direction[batch_index]
                        if isinstance(stop_line_axis_direction, torch.Tensor)
                        else None
                    ),
                    axis_valid_logits=(
                        stop_line_axis_valid_logits[batch_index]
                        if isinstance(stop_line_axis_valid_logits, torch.Tensor)
                        else None
                    ),
                    endpoint_logits=(
                        stop_line_endpoint_logits[batch_index]
                        if isinstance(stop_line_endpoint_logits, torch.Tensor)
                        else None
                    ),
                    endpoint_offset=(
                        stop_line_endpoint_offset[batch_index]
                        if isinstance(stop_line_endpoint_offset, torch.Tensor)
                        else None
                    ),
                    endpoint_pair_logits=(
                        stop_line_endpoint_pair_logits[batch_index]
                        if isinstance(stop_line_endpoint_pair_logits, torch.Tensor)
                        else None
                    ),
                    endpoint_pair_points=(
                        stop_line_endpoint_pair_points[batch_index]
                        if isinstance(stop_line_endpoint_pair_points, torch.Tensor)
                        else None
                    ),
                    endpoint_pair_verifier_logits=(
                        stop_line_endpoint_pair_verifier_logits[batch_index]
                        if isinstance(stop_line_endpoint_pair_verifier_logits, torch.Tensor)
                        else None
                    ),
                    segment_logits=(
                        stop_line_segment_logits[batch_index]
                        if isinstance(stop_line_segment_logits, torch.Tensor)
                        else None
                    ),
                    segment_points=(
                        stop_line_segment_points[batch_index]
                        if isinstance(stop_line_segment_points, torch.Tensor)
                        else None
                    ),
                    segment_verifier_logits=(
                        stop_line_segment_verifier_logits[batch_index]
                        if isinstance(stop_line_segment_verifier_logits, torch.Tensor)
                        else None
                    ),
                    context_segment_logits=(
                        stop_line_context_segment_logits[batch_index]
                        if isinstance(stop_line_context_segment_logits, torch.Tensor)
                        else None
                    ),
                    context_segment_points=(
                        stop_line_context_segment_points[batch_index]
                        if isinstance(stop_line_context_segment_points, torch.Tensor)
                        else None
                    ),
                    context_segment_verifier_logits=(
                        stop_line_context_segment_verifier_logits[batch_index]
                        if isinstance(stop_line_context_segment_verifier_logits, torch.Tensor)
                        else None
                    ),
                    axis_segment_logits=(
                        stop_line_axis_segment_logits[batch_index]
                        if isinstance(stop_line_axis_segment_logits, torch.Tensor)
                        else None
                    ),
                    axis_segment_points=(
                        stop_line_axis_segment_points[batch_index]
                        if isinstance(stop_line_axis_segment_points, torch.Tensor)
                        else None
                    ),
                    axis_segment_verifier_logits=(
                        stop_line_axis_segment_verifier_logits[batch_index]
                        if isinstance(stop_line_axis_segment_verifier_logits, torch.Tensor)
                        else None
                    ),
                    patch_segment_logits=(
                        stop_line_patch_segment_logits[batch_index]
                        if isinstance(stop_line_patch_segment_logits, torch.Tensor)
                        else None
                    ),
                    patch_segment_points=(
                        stop_line_patch_segment_points[batch_index]
                        if isinstance(stop_line_patch_segment_points, torch.Tensor)
                        else None
                    ),
                    patch_segment_verifier_logits=(
                        stop_line_patch_segment_verifier_logits[batch_index]
                        if isinstance(stop_line_patch_segment_verifier_logits, torch.Tensor)
                        else None
                    ),
                    segment_set_enabled=config.stop_line_segment_set_enabled,
                    segment_set_score_threshold=config.stop_line_segment_set_score_threshold,
                    segment_set_max_segments=config.stop_line_segment_set_max_segments,
                    segment_verifier_score_weight=config.stop_line_segment_verifier_score_weight,
                    context_segment_set_enabled=config.stop_line_context_segment_set_enabled,
                    context_segment_set_score_threshold=config.stop_line_context_segment_set_score_threshold,
                    context_segment_set_max_segments=config.stop_line_context_segment_set_max_segments,
                    context_segment_verifier_score_weight=(
                        config.stop_line_context_segment_verifier_score_weight
                    ),
                    axis_segment_set_enabled=config.stop_line_axis_segment_set_enabled,
                    axis_segment_set_score_threshold=config.stop_line_axis_segment_set_score_threshold,
                    axis_segment_set_max_segments=config.stop_line_axis_segment_set_max_segments,
                    axis_segment_verifier_score_weight=config.stop_line_axis_segment_verifier_score_weight,
                    patch_segment_set_enabled=config.stop_line_patch_segment_set_enabled,
                    patch_segment_set_score_threshold=config.stop_line_patch_segment_set_score_threshold,
                    patch_segment_set_max_segments=config.stop_line_patch_segment_set_max_segments,
                    patch_segment_verifier_score_weight=config.stop_line_patch_segment_verifier_score_weight,
                    haf_enabled=config.stop_line_haf_enabled,
                    haf_valid_threshold=config.stop_line_haf_valid_threshold,
                    haf_min_votes=config.stop_line_haf_min_votes,
                    haf_cluster_endpoint_tolerance=config.stop_line_haf_cluster_endpoint_tolerance,
                    haf_max_endpoint_covariance=config.stop_line_haf_max_endpoint_covariance,
                    haf_max_segments=config.stop_line_haf_max_segments,
                    axis_distance_enabled=config.stop_line_axis_distance_enabled,
                    axis_distance_valid_threshold=config.stop_line_axis_distance_valid_threshold,
                    axis_distance_min_votes=config.stop_line_axis_distance_min_votes,
                    axis_distance_cluster_endpoint_tolerance=(
                        config.stop_line_axis_distance_cluster_endpoint_tolerance
                    ),
                    axis_distance_max_endpoint_covariance=config.stop_line_axis_distance_max_endpoint_covariance,
                    axis_distance_min_support_score=config.stop_line_axis_distance_min_support_score,
                    axis_distance_max_segments=config.stop_line_axis_distance_max_segments,
                    endpoint_pair_enabled=config.stop_line_endpoint_pair_enabled,
                    endpoint_pair_score_threshold=config.stop_line_endpoint_pair_score_threshold,
                    endpoint_pair_topk=config.stop_line_endpoint_pair_topk,
                    endpoint_pair_max_segments=config.stop_line_endpoint_pair_max_segments,
                    endpoint_haf_consensus_enabled=config.stop_line_endpoint_haf_consensus_enabled,
                    endpoint_haf_consensus_score_threshold=(
                        config.stop_line_endpoint_haf_consensus_score_threshold
                    ),
                    endpoint_haf_consensus_topk=config.stop_line_endpoint_haf_consensus_topk,
                    endpoint_haf_consensus_haf_valid_threshold=(
                        config.stop_line_endpoint_haf_consensus_haf_valid_threshold
                    ),
                    endpoint_haf_consensus_min_votes=config.stop_line_endpoint_haf_consensus_min_votes,
                    endpoint_haf_consensus_max_endpoint_error=(
                        config.stop_line_endpoint_haf_consensus_max_endpoint_error
                    ),
                    endpoint_haf_consensus_max_endpoint_covariance=(
                        config.stop_line_endpoint_haf_consensus_max_endpoint_covariance
                    ),
                    endpoint_haf_consensus_max_segments=config.stop_line_endpoint_haf_consensus_max_segments,
                    endpoint_pair_segment_enabled=config.stop_line_endpoint_pair_segment_enabled,
                    endpoint_pair_segment_score_threshold=config.stop_line_endpoint_pair_segment_score_threshold,
                    endpoint_pair_segment_max_segments=config.stop_line_endpoint_pair_segment_max_segments,
                    endpoint_pair_verifier_score_weight=config.stop_line_endpoint_pair_verifier_score_weight,
                    projection_comp_enabled=config.stop_line_projection_comp_enabled,
                    projection_comp_proposal_source=config.stop_line_projection_comp_proposal_source,
                    projection_comp_min_gap=config.stop_line_projection_comp_min_gap,
                    projection_comp_topk=config.stop_line_projection_comp_topk,
                    projection_comp_union_min_score=config.stop_line_projection_comp_union_min_score,
                    projection_comp_single_min_score=config.stop_line_projection_comp_single_min_score,
                    projection_comp_angle_threshold_deg=config.stop_line_projection_comp_angle_threshold_deg,
                    projection_comp_offset_threshold_px=config.stop_line_projection_comp_offset_threshold_px,
                    projection_comp_min_cluster_count=config.stop_line_projection_comp_min_cluster_count,
                    projection_comp_projection_gap_px=config.stop_line_projection_comp_projection_gap_px,
                    projection_comp_max_predictions=config.stop_line_projection_comp_max_predictions,
                    projection_comp_second_min_score=config.stop_line_projection_comp_second_min_score,
                    projection_comp_second_min_fragment_count=(
                        config.stop_line_projection_comp_second_min_fragment_count
                    ),
                    projection_comp_second_min_length_ratio=config.stop_line_projection_comp_second_min_length_ratio,
                ),
                "crosswalks": _decode_crosswalk_rows(
                    crosswalk_pred[batch_index],
                    meta=sample_meta,
                    obj_threshold=config.crosswalk_obj_threshold,
                    mask_binary_threshold=config.crosswalk_mask_binary_threshold,
                    min_component_pixels=config.crosswalk_min_component_pixels,
                    max_components=config.crosswalk_max_components,
                    min_polygon_area_px=config.crosswalk_min_polygon_area_px,
                    min_bbox_aspect=config.crosswalk_min_bbox_aspect,
                    polygon_mode=config.crosswalk_polygon_mode,
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
