from __future__ import annotations

from typing import Any

import torch

from ..preprocess.aihub_standardize import OD_CLASSES
from ..loss.spec import build_loss_spec


SPEC = build_loss_spec()
LANE_QUERY_COUNT = int(SPEC["heads"]["lane"]["query_count"])
STOP_LINE_QUERY_COUNT = int(SPEC["heads"]["stop_line"]["query_count"])
CROSSWALK_QUERY_COUNT = int(SPEC["heads"]["crosswalk"]["query_count"])
LANE_POINT_COUNT = int(SPEC["heads"]["lane"]["target_encoding"]["polyline_points"])
STOP_LINE_POINT_COUNT = int(SPEC["heads"]["stop_line"]["target_encoding"]["polyline_points"])
CROSSWALK_POINT_COUNT = int(SPEC["heads"]["crosswalk"]["target_encoding"]["polygon_points"])
LANE_VECTOR_SIZE = 54
STOP_LINE_VECTOR_SIZE = 9
CROSSWALK_VECTOR_SIZE = 17


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


def _sort_lane_points(points_xy: Any) -> torch.FloatTensor:
    points = _as_float_tensor(points_xy).reshape(-1, 2)
    if points.shape[0] <= 1:
        return points
    order = torch.argsort(points[:, 1], descending=True)
    return points[order]


def _sort_stop_line_points(points_xy: Any) -> torch.FloatTensor:
    points = _as_float_tensor(points_xy).reshape(-1, 2)
    if points.shape[0] <= 1:
        return points
    order = torch.argsort(points[:, 0], descending=False)
    return points[order]


def _sort_crosswalk_points(points_xy: Any) -> torch.FloatTensor:
    points = _as_float_tensor(points_xy).reshape(-1, 2)
    if points.shape[0] <= 2:
        return points
    center = points.mean(dim=0)
    angles = torch.atan2(points[:, 1] - center[1], points[:, 0] - center[0])
    order = torch.argsort(angles, descending=True)
    ordered = points[order]
    top_left_score = ordered[:, 1] * 10000.0 + ordered[:, 0]
    start = int(torch.argmin(top_left_score).item())
    return torch.roll(ordered, shifts=-start, dims=0)


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

    sorted_rows = sorted(enumerate(rows), key=lambda item: _lane_sort_key(item[1]))
    for query_index, (source_index, row) in enumerate(sorted_rows[:LANE_QUERY_COUNT]):
        points = _sort_lane_points(row.get("points_xy", []))
        color_index = int(row.get("color", -1))
        lane_type_index = int(row.get("lane_type", -1))
        is_valid = bool(valid_mask[source_index]) and color_index >= 0 and lane_type_index >= 0
        if not is_valid:
            continue

        resampled = _resample_points(points, LANE_POINT_COUNT)
        encoded[query_index, 0] = 1.0
        encoded[query_index, 1 + color_index] = 1.0
        encoded[query_index, 4 + lane_type_index] = 1.0
        encoded[query_index, 6:38] = resampled.flatten()
        encoded[query_index, 38:54] = 1.0
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

    sorted_rows = sorted(enumerate(rows), key=lambda item: _stop_line_sort_key(item[1]))
    for query_index, (source_index, row) in enumerate(sorted_rows[:STOP_LINE_QUERY_COUNT]):
        if not bool(valid_mask[source_index]):
            continue
        points = _sort_stop_line_points(row.get("points_xy", []))
        resampled = _resample_points(points, STOP_LINE_POINT_COUNT)
        encoded[query_index, 0] = 1.0
        encoded[query_index, 1:9] = resampled.flatten()
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

    sorted_rows = sorted(enumerate(rows), key=lambda item: _crosswalk_sort_key(item[1]))
    for query_index, (source_index, row) in enumerate(sorted_rows[:CROSSWALK_QUERY_COUNT]):
        if not bool(valid_mask[source_index]):
            continue
        points = _sort_crosswalk_points(row.get("points_xy", []))
        resampled = _resample_points(points, CROSSWALK_POINT_COUNT)
        encoded[query_index, 0] = 1.0
        encoded[query_index, 1:17] = resampled.flatten()
        row_valid[query_index] = True
    return encoded, row_valid


def encode_pv26_batch(batch: dict[str, Any]) -> dict[str, Any]:
    images = batch["image"]
    det_targets = batch["det_targets"]
    tl_attr_targets = batch["tl_attr_targets"]
    lane_targets = batch["lane_targets"]
    source_masks = batch["source_mask"]
    valid_masks = batch["valid_mask"]
    meta = batch["meta"]

    batch_size = int(images.shape[0])
    max_det_gt = max((int(item["boxes_xyxy"].shape[0]) for item in det_targets), default=0)

    det_boxes = torch.zeros((batch_size, max_det_gt, 4), dtype=torch.float32)
    det_classes = torch.full((batch_size, max_det_gt), -1, dtype=torch.long)
    det_valid = torch.zeros((batch_size, max_det_gt), dtype=torch.bool)
    tl_bits = torch.zeros((batch_size, max_det_gt, 4), dtype=torch.float32)
    tl_mask = torch.zeros((batch_size, max_det_gt), dtype=torch.bool)

    lane_encoded = torch.zeros((batch_size, LANE_QUERY_COUNT, LANE_VECTOR_SIZE), dtype=torch.float32)
    lane_valid = torch.zeros((batch_size, LANE_QUERY_COUNT), dtype=torch.bool)
    stop_line_encoded = torch.zeros((batch_size, STOP_LINE_QUERY_COUNT, STOP_LINE_VECTOR_SIZE), dtype=torch.float32)
    stop_line_valid = torch.zeros((batch_size, STOP_LINE_QUERY_COUNT), dtype=torch.bool)
    crosswalk_encoded = torch.zeros((batch_size, CROSSWALK_QUERY_COUNT, CROSSWALK_VECTOR_SIZE), dtype=torch.float32)
    crosswalk_valid = torch.zeros((batch_size, CROSSWALK_QUERY_COUNT), dtype=torch.bool)

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

        lane_rows, lane_rows_valid = _encode_lane_rows(
            sample_lane["lanes"],
            sample_valid["lane"],
            bool(lane_source[batch_index]),
        )
        lane_encoded[batch_index] = lane_rows
        lane_valid[batch_index] = lane_rows_valid

        stop_rows, stop_rows_valid = _encode_stop_line_rows(
            sample_lane["stop_lines"],
            sample_valid["stop_line"],
            bool(stop_line_source[batch_index]),
        )
        stop_line_encoded[batch_index] = stop_rows
        stop_line_valid[batch_index] = stop_rows_valid

        cross_rows, cross_rows_valid = _encode_crosswalk_rows(
            sample_lane["crosswalks"],
            sample_valid["crosswalk"],
            bool(crosswalk_source[batch_index]),
        )
        crosswalk_encoded[batch_index] = cross_rows
        crosswalk_valid[batch_index] = cross_rows_valid

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
            "stop_line_valid": stop_line_valid,
            "crosswalk_valid": crosswalk_valid,
        },
        "meta": meta,
    }


__all__ = ["encode_pv26_batch"]
