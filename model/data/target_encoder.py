from __future__ import annotations

from typing import Any

import torch

from common.geometry import (
    canonicalize_crosswalk_points,
    canonicalize_stop_line_points,
    sample_crosswalk_contour,
    sample_stop_line_centerline,
)
from common.task_mode import LANE_FAMILY_TASK_MODE, active_tasks_for_mode, filter_source_mask_for_task_mode
from common.pv26_schema import OD_CLASSES
from .transform import NETWORK_HW
from .roadmark_v2_targets import (
    LANE_CENTERLINE_OUTPUT_HW,
    LANE_ROW_CLASS_OUTPUT_HW,
    LANE_ROW_SLOT_COUNT,
    LANE_SEED_OUTPUT_HW,
    ROADMARK_DENSE_OUTPUT_HW,
    build_crosswalk_mask_targets,
    build_lane_centerline_targets,
    build_lane_dense_row_targets,
    build_lane_row_class_targets,
    build_lane_segfirst_targets,
    build_stopline_mask_targets,
    build_stopline_dense_targets,
)
from ..engine.spec import build_loss_spec


SPEC = build_loss_spec()
LANE_QUERY_COUNT = int(SPEC["heads"]["lane"]["query_count"])
STOP_LINE_QUERY_COUNT = int(SPEC["heads"]["stop_line"]["query_count"])
CROSSWALK_QUERY_COUNT = int(SPEC["heads"]["crosswalk"]["query_count"])
LANE_ANCHOR_COUNT = int(SPEC["heads"]["lane"]["target_encoding"]["anchor_rows"])
STOP_LINE_POINT_COUNT = int(SPEC["heads"]["stop_line"]["target_encoding"]["polyline_points"])
CROSSWALK_POINT_COUNT = int(SPEC["heads"]["crosswalk"]["target_encoding"]["sequence_points"])
LANE_COLOR_DIM = int(SPEC["heads"]["lane"]["target_encoding"]["color_logits"])
LANE_TYPE_DIM = int(SPEC["heads"]["lane"]["target_encoding"]["type_logits"])
LANE_X_DIM = int(SPEC["heads"]["lane"]["target_encoding"]["x_coordinates"])
LANE_VIS_DIM = int(SPEC["heads"]["lane"]["target_encoding"]["visibility_logits"])
LANE_COLOR_SLICE = slice(1, 1 + LANE_COLOR_DIM)
LANE_TYPE_SLICE = slice(LANE_COLOR_SLICE.stop, LANE_COLOR_SLICE.stop + LANE_TYPE_DIM)
LANE_X_SLICE = slice(LANE_TYPE_SLICE.stop, LANE_TYPE_SLICE.stop + LANE_X_DIM)
LANE_VIS_SLICE = slice(LANE_X_SLICE.stop, LANE_X_SLICE.stop + LANE_VIS_DIM)
LANE_VECTOR_SIZE = LANE_VIS_SLICE.stop
STOP_LINE_VECTOR_SIZE = 1 + STOP_LINE_POINT_COUNT * 2
CROSSWALK_VECTOR_SIZE = 1 + CROSSWALK_POINT_COUNT * 2
LANE_ANCHOR_ROWS = torch.linspace(float(NETWORK_HW[0] - 1), 0.0, LANE_ANCHOR_COUNT, dtype=torch.float32)


def _as_float_tensor(points: Any) -> torch.FloatTensor:
    if isinstance(points, torch.Tensor):
        return points.to(dtype=torch.float32)
    return torch.tensor(points, dtype=torch.float32)


def _resample_points(points_xy: Any, target_count: int) -> torch.FloatTensor:
    points = _as_float_tensor(points_xy).reshape(-1, 2)
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
    total_length = float(cumulative[-1])
    if total_length <= 1e-6:
        return points[:1].repeat(target_count, 1)

    targets = torch.linspace(0.0, total_length, target_count, dtype=torch.float32)
    resampled: list[torch.FloatTensor] = []
    for target in targets:
        upper = int(torch.searchsorted(cumulative, target, right=False).item())
        upper = min(max(upper, 1), points.shape[0] - 1)
        lower = upper - 1
        left_distance = cumulative[lower]
        right_distance = cumulative[upper]
        interval = float(right_distance - left_distance)
        if interval <= 1e-6:
            resampled.append(points[lower])
            continue
        ratio = float((target - left_distance) / interval)
        resampled.append(points[lower] + ratio * (points[upper] - points[lower]))
    return torch.stack(resampled, dim=0)


def _coerce_lane_visibility(visibility: Any, point_count: int) -> torch.FloatTensor:
    if isinstance(visibility, torch.Tensor):
        values = visibility.to(dtype=torch.float32).reshape(-1)
    elif isinstance(visibility, (list, tuple)):
        values = torch.tensor(visibility, dtype=torch.float32).reshape(-1)
    else:
        values = torch.ones(point_count, dtype=torch.float32)
    if values.numel() != point_count:
        values = torch.ones(point_count, dtype=torch.float32)
    return values.clamp(0.0, 1.0)


def _sort_lane_points(points_xy: Any, visibility: Any = None) -> tuple[torch.FloatTensor, torch.FloatTensor]:
    points = _as_float_tensor(points_xy).reshape(-1, 2)
    lane_visibility = _coerce_lane_visibility(visibility, int(points.shape[0]))
    if points.shape[0] <= 1:
        return points, lane_visibility
    order = torch.argsort(points[:, 1], descending=True)
    return points[order], lane_visibility[order]


def _sort_stop_line_points(points_xy: Any) -> torch.FloatTensor:
    return torch.as_tensor(canonicalize_stop_line_points(points_xy), dtype=torch.float32).reshape(-1, 2)


def _sample_stop_line_points(points_xy: Any) -> torch.FloatTensor:
    return torch.as_tensor(sample_stop_line_centerline(points_xy, STOP_LINE_POINT_COUNT), dtype=torch.float32).reshape(-1, 2)


def _sort_crosswalk_points(points_xy: Any) -> torch.FloatTensor:
    return torch.as_tensor(canonicalize_crosswalk_points(points_xy), dtype=torch.float32).reshape(-1, 2)


def _sample_crosswalk_points(points_xy: Any) -> torch.FloatTensor:
    return torch.as_tensor(sample_crosswalk_contour(points_xy, CROSSWALK_POINT_COUNT), dtype=torch.float32).reshape(-1, 2)


def _lane_sort_key(row: dict[str, Any]) -> tuple[float, float]:
    points = _as_float_tensor(row.get("points_xy", [])).reshape(-1, 2)
    if points.shape[0] == 0:
        return (float("inf"), float("inf"))
    return (-float(points[:, 1].max()), float(points[:, 0].mean()))


def _stop_line_sort_key(row: dict[str, Any]) -> float:
    points = _as_float_tensor(row.get("points_xy", [])).reshape(-1, 2)
    if points.shape[0] == 0:
        return float("inf")
    return float(points[:, 0].mean())


def _crosswalk_sort_key(row: dict[str, Any]) -> tuple[float, float]:
    points = _as_float_tensor(row.get("points_xy", [])).reshape(-1, 2)
    if points.shape[0] == 0:
        return (float("inf"), float("inf"))
    center = points.mean(dim=0)
    return (-float(center[1]), float(center[0]))


def _interpolate_lane_anchor_targets(
    points: torch.FloatTensor,
    visibility: torch.FloatTensor,
    anchor_rows: torch.FloatTensor,
) -> tuple[torch.FloatTensor, torch.FloatTensor, torch.BoolTensor]:
    x_targets = torch.zeros(anchor_rows.shape[0], dtype=torch.float32)
    vis_targets = torch.zeros(anchor_rows.shape[0], dtype=torch.float32)
    anchor_hits = torch.zeros(anchor_rows.shape[0], dtype=torch.bool)
    if points.shape[0] == 0:
        return x_targets, vis_targets, anchor_hits
    if points.shape[0] == 1:
        closest = int(torch.argmin((anchor_rows - points[0, 1]).abs()).item())
        x_targets[closest] = float(points[0, 0])
        vis_targets[closest] = float(visibility[0])
        anchor_hits[closest] = True
        return x_targets, vis_targets.clamp(0.0, 1.0), anchor_hits

    for point_index in range(points.shape[0] - 1):
        p0 = points[point_index]
        p1 = points[point_index + 1]
        v0 = visibility[point_index]
        v1 = visibility[point_index + 1]
        y0 = float(p0[1])
        y1 = float(p1[1])
        lower = min(y0, y1) - 1.0e-6
        upper = max(y0, y1) + 1.0e-6
        anchor_mask = (~anchor_hits) & (anchor_rows >= lower) & (anchor_rows <= upper)
        anchor_indices = torch.nonzero(anchor_mask, as_tuple=False).flatten()
        if anchor_indices.numel() == 0:
            continue
        if abs(y1 - y0) <= 1.0e-6:
            x_targets[anchor_indices] = (p0[0] + p1[0]) * 0.5
            vis_targets[anchor_indices] = torch.maximum(v0, v1)
            anchor_hits[anchor_indices] = True
            continue
        rows = anchor_rows[anchor_indices]
        ratios = (rows - p0[1]) / (p1[1] - p0[1])
        x_targets[anchor_indices] = p0[0] + ratios * (p1[0] - p0[0])
        vis_targets[anchor_indices] = v0 + ratios * (v1 - v0)
        anchor_hits[anchor_indices] = True
    return x_targets, vis_targets.clamp(0.0, 1.0), anchor_hits


def _sample_label(sample_meta: Any, batch_index: int) -> str:
    if isinstance(sample_meta, dict):
        dataset_key = str(sample_meta.get("dataset_key") or "unknown_dataset")
        sample_id = str(sample_meta.get("sample_id") or f"batch_{batch_index}")
        return f"dataset={dataset_key} sample_id={sample_id}"
    return f"batch_index={batch_index}"


def _raise_det_contract_error(sample_meta: Any, batch_index: int, detail: str) -> None:
    raise ValueError(f"det supervision contract violation for {_sample_label(sample_meta, batch_index)}: {detail}")


def _encode_lane_rows(
    rows: list[dict[str, Any]],
    valid_mask: torch.BoolTensor,
    source_enabled: bool,
) -> tuple[torch.FloatTensor, torch.BoolTensor]:
    encoded = torch.zeros((LANE_QUERY_COUNT, LANE_VECTOR_SIZE), dtype=torch.float32)
    row_valid = torch.zeros(LANE_QUERY_COUNT, dtype=torch.bool)
    if not source_enabled:
        return encoded, row_valid

    supervised_mask = lane_supervised_valid_mask(rows, valid_mask)
    sorted_rows = [
        (source_index, row)
        for source_index, row in sorted(enumerate(rows), key=lambda item: _lane_sort_key(item[1]))
        if bool(supervised_mask[source_index])
    ]
    for query_index, (_, row) in enumerate(sorted_rows[:LANE_QUERY_COUNT]):
        points, visibility = _sort_lane_points(row.get("points_xy", []), row.get("visibility"))
        color_index = int(row.get("color", -1))
        lane_type_index = int(row.get("lane_type", -1))
        anchor_x, anchor_visibility, anchor_hits = _interpolate_lane_anchor_targets(points, visibility, LANE_ANCHOR_ROWS)
        visible_anchor_mask = anchor_hits & (anchor_visibility >= 0.5)
        encoded[query_index, 0] = 1.0
        encoded[query_index, LANE_COLOR_SLICE.start + color_index] = 1.0
        encoded[query_index, LANE_TYPE_SLICE.start + lane_type_index] = 1.0
        encoded[query_index, LANE_X_SLICE] = anchor_x
        encoded[query_index, LANE_VIS_SLICE] = visible_anchor_mask.to(dtype=torch.float32)
        row_valid[query_index] = True
    return encoded, row_valid


def _encode_stop_line_rows(
    rows: list[dict[str, Any]],
    valid_mask: torch.BoolTensor,
    source_enabled: bool,
) -> tuple[torch.FloatTensor, torch.BoolTensor]:
    encoded = torch.zeros((STOP_LINE_QUERY_COUNT, STOP_LINE_VECTOR_SIZE), dtype=torch.float32)
    row_valid = torch.zeros(STOP_LINE_QUERY_COUNT, dtype=torch.bool)
    if not source_enabled:
        return encoded, row_valid

    sorted_rows = [
        row
        for source_index, row in sorted(enumerate(rows), key=lambda item: _stop_line_sort_key(item[1]))
        if bool(valid_mask[source_index])
    ]
    for query_index, row in enumerate(sorted_rows[:STOP_LINE_QUERY_COUNT]):
        points = _sample_stop_line_points(row.get("points_xy", []))
        if points.shape[0] < 2:
            continue
        encoded[query_index, 0] = 1.0
        encoded[query_index, 1:] = points.flatten()
        row_valid[query_index] = True
    return encoded, row_valid


def _encode_crosswalk_rows(
    rows: list[dict[str, Any]],
    valid_mask: torch.BoolTensor,
    source_enabled: bool,
) -> tuple[torch.FloatTensor, torch.BoolTensor]:
    encoded = torch.zeros((CROSSWALK_QUERY_COUNT, CROSSWALK_VECTOR_SIZE), dtype=torch.float32)
    row_valid = torch.zeros(CROSSWALK_QUERY_COUNT, dtype=torch.bool)
    if not source_enabled:
        return encoded, row_valid

    sorted_rows = [
        row
        for source_index, row in sorted(enumerate(rows), key=lambda item: _crosswalk_sort_key(item[1]))
        if bool(valid_mask[source_index])
    ]
    for query_index, row in enumerate(sorted_rows[:CROSSWALK_QUERY_COUNT]):
        encoded_points = _sample_crosswalk_points(row.get("points_xy", []))
        encoded[query_index, 0] = 1.0
        encoded[query_index, 1:] = encoded_points.flatten()
        row_valid[query_index] = True
    return encoded, row_valid


def lane_supervised_valid_mask(
    rows: list[dict[str, Any]],
    valid_mask: torch.BoolTensor | list[bool] | tuple[bool, ...],
    *,
    require_semantics: bool = True,
) -> torch.BoolTensor:
    source_valid = torch.as_tensor(valid_mask, dtype=torch.bool).reshape(-1)
    supervised = torch.zeros(len(rows), dtype=torch.bool)
    for source_index, row in enumerate(rows):
        if source_index >= int(source_valid.numel()) or not bool(source_valid[source_index]):
            continue
        points, visibility = _sort_lane_points(row.get("points_xy", []), row.get("visibility"))
        color_index = int(row.get("color", -1))
        lane_type_index = int(row.get("lane_type", -1))
        _, anchor_visibility, anchor_hits = _interpolate_lane_anchor_targets(points, visibility, LANE_ANCHOR_ROWS)
        visible_anchor_mask = anchor_hits & (anchor_visibility >= 0.5)
        semantic_ok = color_index >= 0 and lane_type_index >= 0
        if bool(require_semantics):
            supervised[source_index] = semantic_ok and int(visible_anchor_mask.sum().item()) >= 2
        else:
            supervised[source_index] = int(visible_anchor_mask.sum().item()) >= 2
    return supervised


def encode_pv26_batch(
    batch: dict[str, Any],
    *,
    task_mode: str = LANE_FAMILY_TASK_MODE,
    include_lane_segfirst_targets: bool = False,
) -> dict[str, Any]:
    images = batch["image"]
    det_targets = batch["det_targets"]
    tl_attr_targets = batch["tl_attr_targets"]
    lane_targets = batch["lane_targets"]
    source_masks = [filter_source_mask_for_task_mode(item, task_mode) for item in batch["source_mask"]]
    valid_masks = batch["valid_mask"]
    meta = batch["meta"]
    active_tasks = set(active_tasks_for_mode(task_mode))

    batch_size = int(images.shape[0])
    max_det_gt = max((int(item["boxes_xyxy"].shape[0]) for item in det_targets), default=0)

    det_boxes = torch.zeros((batch_size, max_det_gt, 4), dtype=torch.float32)
    det_classes = torch.full((batch_size, max_det_gt), -1, dtype=torch.long)
    det_valid = torch.zeros((batch_size, max_det_gt), dtype=torch.bool)
    tl_bits = torch.zeros((batch_size, max_det_gt, 4), dtype=torch.float32)
    tl_mask = torch.zeros((batch_size, max_det_gt), dtype=torch.bool)

    lane_encoded = torch.zeros((batch_size, LANE_QUERY_COUNT, LANE_VECTOR_SIZE), dtype=torch.float32)
    lane_valid = torch.zeros((batch_size, LANE_QUERY_COUNT), dtype=torch.bool)
    lane_raw_count = torch.zeros(batch_size, dtype=torch.int64)
    lane_input_valid_count = torch.zeros(batch_size, dtype=torch.int64)
    lane_supervised_count = torch.zeros(batch_size, dtype=torch.int64)
    stop_line_encoded = torch.zeros((batch_size, STOP_LINE_QUERY_COUNT, STOP_LINE_VECTOR_SIZE), dtype=torch.float32)
    stop_line_valid = torch.zeros((batch_size, STOP_LINE_QUERY_COUNT), dtype=torch.bool)
    crosswalk_encoded = torch.zeros((batch_size, CROSSWALK_QUERY_COUNT, CROSSWALK_VECTOR_SIZE), dtype=torch.float32)
    crosswalk_valid = torch.zeros((batch_size, CROSSWALK_QUERY_COUNT), dtype=torch.bool)
    lane_seed_heatmap = torch.zeros((batch_size, *LANE_SEED_OUTPUT_HW), dtype=torch.float32)
    lane_seed_offset = torch.zeros((batch_size, *LANE_SEED_OUTPUT_HW), dtype=torch.float32)
    lane_seed_positive_rows = torch.full((batch_size, LANE_QUERY_COUNT), -1, dtype=torch.long)
    lane_seed_positive_cols = torch.full((batch_size, LANE_QUERY_COUNT), -1, dtype=torch.long)
    lane_slot_valid = torch.zeros((batch_size, LANE_ROW_SLOT_COUNT), dtype=torch.bool)
    lane_row_exists = torch.zeros((batch_size, LANE_ROW_SLOT_COUNT, LANE_ROW_CLASS_OUTPUT_HW[0]), dtype=torch.float32)
    lane_row_col_index = torch.full((batch_size, LANE_ROW_SLOT_COUNT, LANE_ROW_CLASS_OUTPUT_HW[0]), -1, dtype=torch.long)
    lane_row_col_target = torch.full((batch_size, LANE_ROW_SLOT_COUNT, LANE_ROW_CLASS_OUTPUT_HW[0]), -1.0, dtype=torch.float32)
    lane_row_soft_target = torch.zeros((batch_size, LANE_ROW_SLOT_COUNT, LANE_ROW_CLASS_OUTPUT_HW[0], LANE_ROW_CLASS_OUTPUT_HW[1]), dtype=torch.float32)
    lane_slot_color = torch.zeros((batch_size, LANE_ROW_SLOT_COUNT), dtype=torch.long)
    lane_slot_type = torch.zeros((batch_size, LANE_ROW_SLOT_COUNT), dtype=torch.long)
    lane_centerline = torch.zeros((batch_size, 1, *LANE_CENTERLINE_OUTPUT_HW), dtype=torch.float32)
    stop_line_center_heatmap = torch.zeros((batch_size, 1, *ROADMARK_DENSE_OUTPUT_HW), dtype=torch.float32)
    stop_line_center_offset = torch.zeros((batch_size, 2, *ROADMARK_DENSE_OUTPUT_HW), dtype=torch.float32)
    stop_line_angle = torch.zeros((batch_size, 2, *ROADMARK_DENSE_OUTPUT_HW), dtype=torch.float32)
    stop_line_half_length = torch.zeros((batch_size, 1, *ROADMARK_DENSE_OUTPUT_HW), dtype=torch.float32)
    stop_line_mask = torch.zeros((batch_size, 1, *ROADMARK_DENSE_OUTPUT_HW), dtype=torch.float32)
    stop_line_centerline = torch.zeros((batch_size, 1, *ROADMARK_DENSE_OUTPUT_HW), dtype=torch.float32)
    crosswalk_mask = torch.zeros((batch_size, 1, *ROADMARK_DENSE_OUTPUT_HW), dtype=torch.float32)
    crosswalk_boundary = torch.zeros((batch_size, 1, *ROADMARK_DENSE_OUTPUT_HW), dtype=torch.float32)
    crosswalk_center = torch.zeros((batch_size, 1, *ROADMARK_DENSE_OUTPUT_HW), dtype=torch.float32)
    lane_segfirst: dict[str, torch.Tensor] = {}
    if bool(include_lane_segfirst_targets):
        lane_segfirst = {
            "lane_seg_centerline_core": torch.zeros((batch_size, 1, *ROADMARK_DENSE_OUTPUT_HW), dtype=torch.float32),
            "lane_seg_centerline_soft": torch.zeros((batch_size, 1, *ROADMARK_DENSE_OUTPUT_HW), dtype=torch.float32),
            "lane_seg_support": torch.zeros((batch_size, 1, *ROADMARK_DENSE_OUTPUT_HW), dtype=torch.float32),
            "lane_seg_tangent_axis": torch.zeros((batch_size, 2, *ROADMARK_DENSE_OUTPUT_HW), dtype=torch.float32),
            "lane_seg_color": torch.zeros((batch_size, LANE_COLOR_DIM, *ROADMARK_DENSE_OUTPUT_HW), dtype=torch.float32),
            "lane_seg_type": torch.zeros((batch_size, LANE_TYPE_DIM, *ROADMARK_DENSE_OUTPUT_HW), dtype=torch.float32),
            "lane_seg_ignore": torch.zeros((batch_size, 1, *ROADMARK_DENSE_OUTPUT_HW), dtype=torch.float32),
            "lane_seg_negative": torch.zeros((batch_size, 1, *ROADMARK_DENSE_OUTPUT_HW), dtype=torch.float32),
            "lane_seg_stop_line_ignore": torch.zeros((batch_size, 1, *ROADMARK_DENSE_OUTPUT_HW), dtype=torch.float32),
            "lane_seg_crosswalk_ignore": torch.zeros((batch_size, 1, *ROADMARK_DENSE_OUTPUT_HW), dtype=torch.float32),
            "lane_seg_tangent_count": torch.zeros((batch_size, 1, *ROADMARK_DENSE_OUTPUT_HW), dtype=torch.float32),
        }

    det_source = torch.tensor([bool(mask["det"]) for mask in source_masks], dtype=torch.bool)
    tl_attr_source = torch.tensor([bool(mask["tl_attr"]) for mask in source_masks], dtype=torch.bool)
    lane_source = torch.tensor([bool(mask["lane"]) for mask in source_masks], dtype=torch.bool)
    stop_line_source = torch.tensor([bool(mask["stop_line"]) for mask in source_masks], dtype=torch.bool)
    crosswalk_source = torch.tensor([bool(mask["crosswalk"]) for mask in source_masks], dtype=torch.bool)
    det_supervised_class_mask = torch.zeros((batch_size, len(OD_CLASSES)), dtype=torch.bool)
    det_allow_objectness_negatives = torch.zeros(batch_size, dtype=torch.bool)
    det_allow_unmatched_class_negatives = torch.zeros(batch_size, dtype=torch.bool)

    for batch_index in range(batch_size):
        sample_det = det_targets[batch_index]
        sample_tl = tl_attr_targets[batch_index]
        sample_lane = lane_targets[batch_index]
        sample_valid = valid_masks[batch_index]
        sample_meta = meta[batch_index] if batch_index < len(meta) else {}
        if not isinstance(sample_meta, dict):
            sample_meta = {}
        if bool(det_source[batch_index]):
            class_ids = sample_meta.get("det_supervised_class_ids")
            if not isinstance(class_ids, (list, tuple)):
                _raise_det_contract_error(sample_meta, batch_index, "missing meta.det_supervised_class_ids")
            if not class_ids:
                _raise_det_contract_error(sample_meta, batch_index, "meta.det_supervised_class_ids must not be empty")
            for class_id in class_ids:
                try:
                    index = int(class_id)
                except (TypeError, ValueError):
                    _raise_det_contract_error(sample_meta, batch_index, f"invalid class id: {class_id!r}")
                if not 0 <= index < len(OD_CLASSES):
                    _raise_det_contract_error(sample_meta, batch_index, f"class id out of range: {index}")
                det_supervised_class_mask[batch_index, index] = True
            if not bool(det_supervised_class_mask[batch_index].any()):
                _raise_det_contract_error(sample_meta, batch_index, "at least one detector class must be supervised")
            if "det_allow_objectness_negatives" not in sample_meta:
                _raise_det_contract_error(sample_meta, batch_index, "missing meta.det_allow_objectness_negatives")
            if "det_allow_unmatched_class_negatives" not in sample_meta:
                _raise_det_contract_error(sample_meta, batch_index, "missing meta.det_allow_unmatched_class_negatives")
            det_allow_objectness_negatives[batch_index] = bool(sample_meta["det_allow_objectness_negatives"])
            det_allow_unmatched_class_negatives[batch_index] = bool(sample_meta["det_allow_unmatched_class_negatives"])

        det_count = int(sample_det["boxes_xyxy"].shape[0])
        if det_count > 0:
            det_boxes[batch_index, :det_count] = sample_det["boxes_xyxy"].to(dtype=torch.float32)
            det_classes[batch_index, :det_count] = sample_det["classes"].to(dtype=torch.long)
            det_valid[batch_index, :det_count] = sample_valid["det"].to(dtype=torch.bool)
            tl_bits[batch_index, :det_count] = sample_tl["bits"].to(dtype=torch.float32)
            tl_mask[batch_index, :det_count] = sample_valid["tl_attr"].to(dtype=torch.bool)

        lane_raw_count[batch_index] = int(len(sample_lane["lanes"]))
        lane_input_valid_count[batch_index] = int(torch.as_tensor(sample_valid["lane"], dtype=torch.int64).sum().item())
        lane_supervised_mask = lane_supervised_valid_mask(
            sample_lane["lanes"],
            sample_valid["lane"],
            require_semantics=task_mode != "lane_only",
        )
        lane_supervised_count[batch_index] = int(lane_supervised_mask.to(dtype=torch.int64).sum().item())

        lane_rows, lane_rows_valid = _encode_lane_rows(
            sample_lane["lanes"],
            sample_valid["lane"],
            bool(lane_source[batch_index]),
        )
        lane_encoded[batch_index] = lane_rows
        lane_valid[batch_index] = lane_rows_valid

        if "stop_line" in active_tasks:
            stop_rows, stop_rows_valid = _encode_stop_line_rows(
                sample_lane["stop_lines"],
                sample_valid["stop_line"],
                bool(stop_line_source[batch_index]),
            )
            stop_line_encoded[batch_index] = stop_rows
            stop_line_valid[batch_index] = stop_rows_valid

        if "crosswalk" in active_tasks:
            cross_rows, cross_rows_valid = _encode_crosswalk_rows(
                sample_lane["crosswalks"],
                sample_valid["crosswalk"],
                bool(crosswalk_source[batch_index]),
            )
            crosswalk_encoded[batch_index] = cross_rows
            crosswalk_valid[batch_index] = cross_rows_valid

        if "lane" in active_tasks:
            lane_v2 = build_lane_dense_row_targets(
                sample_lane["lanes"],
                sample_valid["lane"],
                source_enabled=bool(lane_source[batch_index]),
            )
            lane_seed_heatmap[batch_index] = lane_v2["lane_seed_heatmap"]
            lane_seed_offset[batch_index] = lane_v2["lane_seed_offset"]
            lane_seed_positive_rows[batch_index] = lane_v2["lane_seed_positive_rows"]
            lane_seed_positive_cols[batch_index] = lane_v2["lane_seed_positive_cols"]
            lane_row_targets = build_lane_row_class_targets(
                sample_lane["lanes"],
                sample_valid["lane"],
                source_enabled=bool(lane_source[batch_index]),
            )
            lane_slot_valid[batch_index] = lane_row_targets["lane_slot_valid"]
            lane_row_exists[batch_index] = lane_row_targets["lane_row_exists"]
            lane_row_col_index[batch_index] = lane_row_targets["lane_row_col_index"]
            lane_row_col_target[batch_index] = lane_row_targets["lane_row_col_target"]
            lane_row_soft_target[batch_index] = lane_row_targets["lane_row_soft_target"]
            lane_slot_color[batch_index] = lane_row_targets["lane_slot_color"]
            lane_slot_type[batch_index] = lane_row_targets["lane_slot_type"]
            lane_centerline_targets = build_lane_centerline_targets(
                sample_lane["lanes"],
                sample_valid["lane"],
                source_enabled=bool(lane_source[batch_index]),
            )
            lane_centerline[batch_index] = lane_centerline_targets["lane_centerline"]
            if bool(include_lane_segfirst_targets):
                lane_segfirst_targets = build_lane_segfirst_targets(
                    sample_lane["lanes"],
                    sample_valid["lane"],
                    stop_line_rows=sample_lane["stop_lines"],
                    stop_line_valid_mask=sample_valid["stop_line"],
                    crosswalk_rows=sample_lane["crosswalks"],
                    crosswalk_valid_mask=sample_valid["crosswalk"],
                    source_enabled=bool(lane_source[batch_index]),
                )
                for key, value in lane_segfirst_targets.items():
                    lane_segfirst[key][batch_index] = value

        if "stop_line" in active_tasks:
            stop_line_v2 = build_stopline_dense_targets(
                sample_lane["stop_lines"],
                sample_valid["stop_line"],
                source_enabled=bool(stop_line_source[batch_index]),
            )
            stop_line_center_heatmap[batch_index] = stop_line_v2["stop_line_center_heatmap"]
            stop_line_center_offset[batch_index] = stop_line_v2["stop_line_center_offset"]
            stop_line_angle[batch_index] = stop_line_v2["stop_line_angle"]
            stop_line_half_length[batch_index] = stop_line_v2["stop_line_half_length"]
            stop_line_mask_targets = build_stopline_mask_targets(
                sample_lane["stop_lines"],
                sample_valid["stop_line"],
                source_enabled=bool(stop_line_source[batch_index]),
            )
            stop_line_mask[batch_index] = stop_line_mask_targets["stop_line_mask"]
            stop_line_centerline[batch_index] = stop_line_mask_targets["stop_line_centerline"]

        if "crosswalk" in active_tasks:
            crosswalk_v2 = build_crosswalk_mask_targets(
                sample_lane["crosswalks"],
                sample_valid["crosswalk"],
                source_enabled=bool(crosswalk_source[batch_index]),
            )
            crosswalk_mask[batch_index] = crosswalk_v2["crosswalk_mask"]
            crosswalk_boundary[batch_index] = crosswalk_v2["crosswalk_boundary"]
            crosswalk_center[batch_index] = crosswalk_v2["crosswalk_center"]

    roadmark_v2 = {
        "lane_seed_heatmap": lane_seed_heatmap,
        "lane_seed_offset": lane_seed_offset,
        "lane_seed_positive_rows": lane_seed_positive_rows,
        "lane_seed_positive_cols": lane_seed_positive_cols,
        "lane_slot_valid": lane_slot_valid,
        "lane_centerline": lane_centerline,
        "lane_row_exists": lane_row_exists,
        "lane_row_col_index": lane_row_col_index,
        "lane_row_col_target": lane_row_col_target,
        "lane_row_soft_target": lane_row_soft_target,
        "lane_slot_color": lane_slot_color,
        "lane_slot_type": lane_slot_type,
        "stop_line_center_heatmap": stop_line_center_heatmap,
        "stop_line_center_offset": stop_line_center_offset,
        "stop_line_angle": stop_line_angle,
        "stop_line_half_length": stop_line_half_length,
        "stop_line_mask": stop_line_mask,
        "stop_line_centerline": stop_line_centerline,
        "crosswalk_mask": crosswalk_mask,
        "crosswalk_boundary": crosswalk_boundary,
        "crosswalk_center": crosswalk_center,
    }
    roadmark_v2.update(lane_segfirst)

    return {
        "image": images,
        "det_gt": {
            "boxes_xyxy": det_boxes,
            "classes": det_classes,
            "valid_mask": det_valid,
        },
        "tl_attr_gt_bits": tl_bits,
        "tl_attr_gt_mask": tl_mask,
        "lane": lane_encoded,
        "stop_line": stop_line_encoded,
        "crosswalk": crosswalk_encoded,
        "roadmark_v2": roadmark_v2,
        "mask": {
            "det_source": det_source,
            "det_supervised_class_mask": det_supervised_class_mask,
            "det_allow_objectness_negatives": det_allow_objectness_negatives,
            "det_allow_unmatched_class_negatives": det_allow_unmatched_class_negatives,
            "tl_attr_source": tl_attr_source,
            "lane_source": lane_source,
            "stop_line_source": stop_line_source,
            "crosswalk_source": crosswalk_source,
            "lane_valid": lane_valid,
            "lane_raw_count": lane_raw_count,
            "lane_input_valid_count": lane_input_valid_count,
            "lane_supervised_count": lane_supervised_count,
            "stop_line_valid": stop_line_valid,
            "crosswalk_valid": crosswalk_valid,
        },
        "meta": meta,
        "task_mode": str(task_mode),
    }


__all__ = ["encode_pv26_batch", "lane_supervised_valid_mask"]
