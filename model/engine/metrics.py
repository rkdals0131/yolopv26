from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from scipy.optimize import linear_sum_assignment
import torch

from ..data.transform import inverse_transform_box_xyxy, inverse_transform_points, transform_from_meta
from .spec import build_loss_spec


SPEC = build_loss_spec()
OD_CLASSES = tuple(SPEC["model_contract"]["od_classes"])
TL_BITS = tuple(SPEC["model_contract"]["tl_bits"])
LANE_CLASSES = ("white_lane", "yellow_lane", "blue_lane")
LANE_TYPES = ("solid", "dotted")


@dataclass(frozen=True)
class PV26MetricConfig:
    det_iou_threshold: float = 0.50
    tl_iou_threshold: float = 0.50
    tl_bit_threshold: float = 0.50
    det_tiny_max_side: float = 16.0
    det_small_max_side: float = 32.0
    lane_match_threshold: float = 40.0
    stop_line_match_threshold: float = 40.0
    crosswalk_iou_threshold: float = 0.30


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator <= 0.0:
        return 0.0
    return numerator / denominator


def _prf(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2.0 * precision * recall, precision + recall)
    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _box_iou(box_a: list[float], box_b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(value) for value in box_a]
    bx1, by1, bx2, by2 = [float(value) for value in box_b]
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union_area = area_a + area_b - inter_area
    return _safe_div(inter_area, union_area)


def _det_box_bucket(box: Iterable[float], *, config: PV26MetricConfig) -> str:
    x1, y1, x2, y2 = [float(value) for value in box]
    min_side = min(max(0.0, x2 - x1), max(0.0, y2 - y1))
    if min_side < float(config.det_tiny_max_side):
        return "tiny"
    if min_side < float(config.det_small_max_side):
        return "small"
    return "medium_plus"


def _compute_ap(records: list[tuple[float, int]], gt_count: int) -> float:
    if gt_count <= 0 or not records:
        return 0.0
    sorted_records = sorted(records, key=lambda item: item[0], reverse=True)
    tp = np.array([int(item[1]) for item in sorted_records], dtype=np.float32)
    fp = 1.0 - tp
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recall = tp_cum / max(float(gt_count), 1e-12)
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)

    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for index in range(mpre.size - 1, 0, -1):
        mpre[index - 1] = max(mpre[index - 1], mpre[index])
    indices = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[indices + 1] - mrec[indices]) * mpre[indices + 1]))


def _resample_points(points_xy: list[list[float]], target_count: int) -> np.ndarray:
    points = np.asarray(points_xy, dtype=np.float32).reshape(-1, 2)
    if points.shape[0] == 0:
        return np.zeros((target_count, 2), dtype=np.float32)
    if points.shape[0] == 1:
        return np.repeat(points, target_count, axis=0)
    deltas = points[1:] - points[:-1]
    segment_lengths = np.linalg.norm(deltas, axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    total_length = float(cumulative[-1])
    if total_length <= 1e-6:
        return np.repeat(points[:1], target_count, axis=0)

    targets = np.linspace(0.0, total_length, target_count, dtype=np.float32)
    resampled: list[np.ndarray] = []
    for target in targets:
        upper = int(np.searchsorted(cumulative, target, side="left"))
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
    return np.stack(resampled, axis=0)


def _mean_point_distance(points_a: list[list[float]], points_b: list[list[float]], target_count: int) -> float:
    resampled_a = _resample_points(points_a, target_count)
    resampled_b = _resample_points(points_b, target_count)
    return float(np.linalg.norm(resampled_a - resampled_b, axis=1).mean())


def _segment_angle_error(points_a: list[list[float]], points_b: list[list[float]], target_count: int) -> float:
    resampled_a = _resample_points(points_a, target_count)
    resampled_b = _resample_points(points_b, target_count)
    vector_a = resampled_a[-1] - resampled_a[0]
    vector_b = resampled_b[-1] - resampled_b[0]
    norm_a = float(np.linalg.norm(vector_a))
    norm_b = float(np.linalg.norm(vector_b))
    if norm_a <= 1e-6 or norm_b <= 1e-6:
        return 0.0
    cosine = float(np.clip(np.dot(vector_a, vector_b) / (norm_a * norm_b), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _polygon_iou(points_a: list[list[float]], points_b: list[list[float]]) -> float:
    polygon_a = np.asarray(points_a, dtype=np.float32).reshape(-1, 2)
    polygon_b = np.asarray(points_b, dtype=np.float32).reshape(-1, 2)
    min_x = math.floor(min(float(polygon_a[:, 0].min()), float(polygon_b[:, 0].min())))
    min_y = math.floor(min(float(polygon_a[:, 1].min()), float(polygon_b[:, 1].min())))
    max_x = math.ceil(max(float(polygon_a[:, 0].max()), float(polygon_b[:, 0].max())))
    max_y = math.ceil(max(float(polygon_a[:, 1].max()), float(polygon_b[:, 1].max())))
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
    return _safe_div(intersection, union)


def _hungarian_from_cost(cost_matrix: np.ndarray, *, max_cost: float) -> list[tuple[int, int]]:
    if cost_matrix.size == 0 or cost_matrix.shape[0] == 0 or cost_matrix.shape[1] == 0:
        return []
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matches: list[tuple[int, int]] = []
    for pred_index, gt_index in zip(row_ind.tolist(), col_ind.tolist()):
        if float(cost_matrix[pred_index, gt_index]) <= max_cost:
            matches.append((pred_index, gt_index))
    return matches


def _hungarian_from_similarity(similarity_matrix: np.ndarray, *, min_similarity: float) -> list[tuple[int, int]]:
    if similarity_matrix.size == 0 or similarity_matrix.shape[0] == 0 or similarity_matrix.shape[1] == 0:
        return []
    row_ind, col_ind = linear_sum_assignment(1.0 - similarity_matrix)
    matches: list[tuple[int, int]] = []
    for pred_index, gt_index in zip(row_ind.tolist(), col_ind.tolist()):
        if float(similarity_matrix[pred_index, gt_index]) >= min_similarity:
            matches.append((pred_index, gt_index))
    return matches


def _extract_gt_samples(batch: dict[str, Any]) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for sample_index, meta in enumerate(batch["meta"]):
        transform = transform_from_meta(meta)
        sample_det = batch["det_targets"][sample_index]
        sample_tl = batch["tl_attr_targets"][sample_index]
        sample_lane = batch["lane_targets"][sample_index]
        sample_valid = batch["valid_mask"][sample_index]

        detections: list[dict[str, Any]] = []
        for row_index in range(int(sample_det["boxes_xyxy"].shape[0])):
            if not bool(sample_valid["det"][row_index]):
                continue
            raw_box = inverse_transform_box_xyxy(sample_det["boxes_xyxy"][row_index].tolist(), transform)
            if raw_box is None:
                continue
            class_id = int(sample_det["classes"][row_index].item())
            detections.append(
                {
                    "box_xyxy": [float(value) for value in raw_box],
                    "class_id": class_id,
                    "class_name": OD_CLASSES[class_id],
                    "tl_valid": bool(sample_valid["tl_attr"][row_index]) if row_index < len(sample_valid["tl_attr"]) else False,
                    "tl_bits": [
                        int(value)
                        for value in sample_tl["bits"][row_index].tolist()
                    ]
                    if row_index < int(sample_tl["bits"].shape[0])
                    else [0, 0, 0, 0],
                }
            )

        lanes: list[dict[str, Any]] = []
        for row_index, row in enumerate(sample_lane["lanes"]):
            if not bool(sample_valid["lane"][row_index]):
                continue
            raw_points = inverse_transform_points(row["points_xy"].tolist(), transform)
            lanes.append(
                {
                    "class_name": LANE_CLASSES[int(row["color"])],
                    "lane_type": LANE_TYPES[int(row["lane_type"])],
                    "points_xy": [[float(x), float(y)] for x, y in raw_points],
                }
            )

        stop_lines: list[dict[str, Any]] = []
        for row_index, row in enumerate(sample_lane["stop_lines"]):
            if not bool(sample_valid["stop_line"][row_index]):
                continue
            raw_points = inverse_transform_points(row["points_xy"].tolist(), transform)
            stop_lines.append({"points_xy": [[float(x), float(y)] for x, y in raw_points]})

        crosswalks: list[dict[str, Any]] = []
        for row_index, row in enumerate(sample_lane["crosswalks"]):
            if not bool(sample_valid["crosswalk"][row_index]):
                continue
            raw_points = inverse_transform_points(row["points_xy"].tolist(), transform)
            crosswalks.append({"points_xy": [[float(x), float(y)] for x, y in raw_points]})

        samples.append(
            {
                "meta": dict(meta),
                "detections": detections,
                "lanes": lanes,
                "stop_lines": stop_lines,
                "crosswalks": crosswalks,
            }
        )
    return samples


def _detector_metrics(
    predictions: list[dict[str, Any]],
    gt_samples: list[dict[str, Any]],
    *,
    config: PV26MetricConfig,
    iou_threshold: float,
) -> dict[str, Any]:
    per_class: dict[str, Any] = {}
    map_candidates: list[float] = []
    total_tp = 0
    total_fp = 0
    total_fn = 0
    bucket_counts = {
        name: {"tp": 0, "fp": 0, "fn": 0, "gt_count": 0, "records": []}
        for name in ("tiny", "small", "medium_plus")
    }

    for class_id, class_name in enumerate(OD_CLASSES):
        records: list[tuple[float, int]] = []
        gt_count = 0
        class_tp = 0
        class_fp = 0

        for pred_sample, gt_sample in zip(predictions, gt_samples):
            gt_rows = [row for row in gt_sample["detections"] if int(row["class_id"]) == class_id]
            pred_rows = [row for row in pred_sample["detections"] if int(row["class_id"]) == class_id]
            gt_count += len(gt_rows)
            for gt in gt_rows:
                bucket_counts[_det_box_bucket(gt["box_xyxy"], config=config)]["gt_count"] += 1
            matched_gt: set[int] = set()
            for pred in sorted(pred_rows, key=lambda item: float(item["score"]), reverse=True):
                best_iou = 0.0
                best_gt_index = -1
                for gt_index, gt in enumerate(gt_rows):
                    if gt_index in matched_gt:
                        continue
                    iou = _box_iou(pred["box_xyxy"], gt["box_xyxy"])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_index = gt_index
                if best_gt_index >= 0 and best_iou >= iou_threshold:
                    matched_gt.add(best_gt_index)
                    class_tp += 1
                    records.append((float(pred["score"]), 1))
                    bucket = _det_box_bucket(gt_rows[best_gt_index]["box_xyxy"], config=config)
                    bucket_counts[bucket]["tp"] += 1
                    bucket_counts[bucket]["records"].append((float(pred["score"]), 1))
                else:
                    class_fp += 1
                    records.append((float(pred["score"]), 0))
                    bucket = _det_box_bucket(pred["box_xyxy"], config=config)
                    bucket_counts[bucket]["fp"] += 1
                    bucket_counts[bucket]["records"].append((float(pred["score"]), 0))
            for gt_index, gt in enumerate(gt_rows):
                if gt_index not in matched_gt:
                    bucket_counts[_det_box_bucket(gt["box_xyxy"], config=config)]["fn"] += 1

        class_fn = gt_count - class_tp
        ap50 = _compute_ap(records, gt_count)
        if gt_count > 0:
            map_candidates.append(ap50)
        class_summary = _prf(class_tp, class_fp, class_fn)
        class_summary["gt_count"] = int(gt_count)
        class_summary["ap50"] = ap50
        per_class[class_name] = class_summary
        total_tp += class_tp
        total_fp += class_fp
        total_fn += class_fn

    summary = _prf(total_tp, total_fp, total_fn)
    summary["map50"] = sum(map_candidates) / len(map_candidates) if map_candidates else 0.0
    summary["per_class"] = per_class
    summary["size_buckets"] = {
        bucket_name: {
            **_prf(payload["tp"], payload["fp"], payload["fn"]),
            "gt_count": int(payload["gt_count"]),
            "ap50": _compute_ap(payload["records"], int(payload["gt_count"])),
        }
        for bucket_name, payload in bucket_counts.items()
    }
    return summary


def summarize_pv26_tensorboard_histograms(
    predictions: list[dict[str, Any]],
    batch: dict[str, Any],
    *,
    config: PV26MetricConfig | None = None,
) -> dict[str, Any]:
    config = config or PV26MetricConfig()
    gt_samples = _extract_gt_samples(batch)
    detector_prediction_confidence: list[float] = []
    detector_per_class_confidence: dict[str, list[float]] = {class_name: [] for class_name in OD_CLASSES}
    detector_matched_positive_iou: list[float] = []
    traffic_light_attr_confidence: list[float] = []
    lane_point_distances: list[float] = []
    stop_line_angle_errors: list[float] = []
    crosswalk_polygon_ious: list[float] = []

    for pred_sample, gt_sample in zip(predictions, gt_samples):
        for class_id, class_name in enumerate(OD_CLASSES):
            gt_rows = [row for row in gt_sample["detections"] if int(row["class_id"]) == class_id]
            pred_rows = [row for row in pred_sample["detections"] if int(row["class_id"]) == class_id]
            matched_gt: set[int] = set()
            for pred in sorted(pred_rows, key=lambda item: float(item["score"]), reverse=True):
                score = float(pred["score"])
                detector_prediction_confidence.append(score)
                detector_per_class_confidence[class_name].append(score)
                best_iou = 0.0
                best_gt_index = -1
                for gt_index, gt in enumerate(gt_rows):
                    if gt_index in matched_gt:
                        continue
                    iou = _box_iou(pred["box_xyxy"], gt["box_xyxy"])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_index = gt_index
                if best_gt_index >= 0 and best_iou >= config.det_iou_threshold:
                    matched_gt.add(best_gt_index)
                    detector_matched_positive_iou.append(best_iou)

        gt_rows = [
            row
            for row in gt_sample["detections"]
            if row["class_name"] == "traffic_light" and bool(row.get("tl_valid"))
        ]
        pred_rows = [row for row in pred_sample["detections"] if row["class_name"] == "traffic_light"]
        matched_gt: set[int] = set()
        for pred in sorted(pred_rows, key=lambda item: float(item["score"]), reverse=True):
            best_iou = 0.0
            best_gt_index = -1
            for gt_index, gt in enumerate(gt_rows):
                if gt_index in matched_gt:
                    continue
                iou = _box_iou(pred["box_xyxy"], gt["box_xyxy"])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_index = gt_index
            if best_gt_index >= 0 and best_iou >= config.tl_iou_threshold:
                matched_gt.add(best_gt_index)
                attr_scores = pred["tl_attr_scores"]
                values = [float(attr_scores[bit]) for bit in TL_BITS if bit in attr_scores]
                if values:
                    traffic_light_attr_confidence.append(sum(values) / len(values))

        lane_point_distances.extend(
            _collect_lane_family_histogram_values(
                pred_sample,
                gt_sample,
                field_name="lanes",
                target_count=16,
                match_threshold=config.lane_match_threshold,
            )
        )
        stop_line_angle_errors.extend(
            _collect_lane_family_histogram_values(
                pred_sample,
                gt_sample,
                field_name="stop_lines",
                target_count=4,
                match_threshold=config.stop_line_match_threshold,
            )
        )
        crosswalk_polygon_ious.extend(
            _collect_lane_family_histogram_values(
                pred_sample,
                gt_sample,
                field_name="crosswalks",
                target_count=8,
                iou_threshold=config.crosswalk_iou_threshold,
            )
        )

    return {
        "detector": {
            "prediction_confidence": detector_prediction_confidence,
            "per_class_confidence": detector_per_class_confidence,
            "matched_positive_iou": detector_matched_positive_iou,
        },
        "traffic_light": {
            "attr_confidence": traffic_light_attr_confidence,
        },
        "lane": {
            "mean_point_distance": lane_point_distances,
        },
        "stop_line": {
            "mean_angle_error": stop_line_angle_errors,
        },
        "crosswalk": {
            "mean_polygon_iou": crosswalk_polygon_ious,
        },
    }


def _tl_metrics(
    predictions: list[dict[str, Any]],
    gt_samples: list[dict[str, Any]],
    *,
    iou_threshold: float,
    bit_threshold: float,
) -> dict[str, Any]:
    bit_counts = {bit: {"tp": 0, "fp": 0, "fn": 0} for bit in TL_BITS}
    matched_pairs = 0
    combo_correct = 0

    for pred_sample, gt_sample in zip(predictions, gt_samples):
        gt_rows = [
            row
            for row in gt_sample["detections"]
            if row["class_name"] == "traffic_light" and bool(row.get("tl_valid"))
        ]
        pred_rows = [row for row in pred_sample["detections"] if row["class_name"] == "traffic_light"]
        matched_gt: set[int] = set()
        for pred in sorted(pred_rows, key=lambda item: float(item["score"]), reverse=True):
            best_iou = 0.0
            best_gt_index = -1
            for gt_index, gt in enumerate(gt_rows):
                if gt_index in matched_gt:
                    continue
                iou = _box_iou(pred["box_xyxy"], gt["box_xyxy"])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_index = gt_index

            pred_bits = {
                bit: float(pred["tl_attr_scores"][bit]) > bit_threshold
                for bit in TL_BITS
            }
            if best_gt_index >= 0 and best_iou >= iou_threshold:
                matched_gt.add(best_gt_index)
                gt_bits = {bit: bool(gt_rows[best_gt_index]["tl_bits"][bit_index]) for bit_index, bit in enumerate(TL_BITS)}
                matched_pairs += 1
                if all(pred_bits[bit] == gt_bits[bit] for bit in TL_BITS):
                    combo_correct += 1
                for bit in TL_BITS:
                    if pred_bits[bit] and gt_bits[bit]:
                        bit_counts[bit]["tp"] += 1
                    elif pred_bits[bit] and not gt_bits[bit]:
                        bit_counts[bit]["fp"] += 1
                    elif (not pred_bits[bit]) and gt_bits[bit]:
                        bit_counts[bit]["fn"] += 1
            else:
                for bit in TL_BITS:
                    if pred_bits[bit]:
                        bit_counts[bit]["fp"] += 1

        for gt_index, gt in enumerate(gt_rows):
            if gt_index in matched_gt:
                continue
            for bit_index, bit in enumerate(TL_BITS):
                if bool(gt["tl_bits"][bit_index]):
                    bit_counts[bit]["fn"] += 1

    per_bit = {bit: _prf(counts["tp"], counts["fp"], counts["fn"]) for bit, counts in bit_counts.items()}
    mean_f1 = sum(float(per_bit[bit]["f1"]) for bit in TL_BITS) / len(TL_BITS)
    return {
        "matched_pairs": int(matched_pairs),
        "combo_accuracy": _safe_div(combo_correct, matched_pairs),
        "mean_f1": mean_f1,
        "per_bit": per_bit,
    }


def _lane_family_metrics(
    predictions: list[dict[str, Any]],
    gt_samples: list[dict[str, Any]],
    *,
    field_name: str,
    target_count: int,
    match_threshold: float | None = None,
    iou_threshold: float | None = None,
) -> dict[str, Any]:
    total_tp = 0
    total_fp = 0
    total_fn = 0
    distances: list[float] = []
    angle_errors: list[float] = []
    polygon_ious: list[float] = []
    color_hits = 0
    type_hits = 0
    attr_total = 0

    for pred_sample, gt_sample in zip(predictions, gt_samples):
        pred_rows = list(pred_sample[field_name])
        gt_rows = list(gt_sample[field_name])

        if field_name == "crosswalks":
            similarity_matrix = np.zeros((len(pred_rows), len(gt_rows)), dtype=np.float32)
            for pred_index, pred in enumerate(pred_rows):
                for gt_index, gt in enumerate(gt_rows):
                    similarity_matrix[pred_index, gt_index] = _polygon_iou(pred["points_xy"], gt["points_xy"])
            matches = _hungarian_from_similarity(similarity_matrix, min_similarity=float(iou_threshold or 0.0))
        else:
            cost_matrix = np.zeros((len(pred_rows), len(gt_rows)), dtype=np.float32)
            for pred_index, pred in enumerate(pred_rows):
                for gt_index, gt in enumerate(gt_rows):
                    cost_matrix[pred_index, gt_index] = _mean_point_distance(
                        pred["points_xy"],
                        gt["points_xy"],
                        target_count,
                    )
            matches = _hungarian_from_cost(cost_matrix, max_cost=float(match_threshold or 0.0))

        matched_pred = {pred_index for pred_index, _ in matches}
        matched_gt = {gt_index for _, gt_index in matches}
        total_tp += len(matches)
        total_fp += len(pred_rows) - len(matches)
        total_fn += len(gt_rows) - len(matches)

        for pred_index, gt_index in matches:
            pred = pred_rows[pred_index]
            gt = gt_rows[gt_index]
            distances.append(_mean_point_distance(pred["points_xy"], gt["points_xy"], target_count))
            if field_name == "lanes":
                attr_total += 1
                if pred["class_name"] == gt["class_name"]:
                    color_hits += 1
                if pred["lane_type"] == gt["lane_type"]:
                    type_hits += 1
            elif field_name == "stop_lines":
                angle_errors.append(_segment_angle_error(pred["points_xy"], gt["points_xy"], target_count))
            elif field_name == "crosswalks":
                polygon_ious.append(_polygon_iou(pred["points_xy"], gt["points_xy"]))

    summary = _prf(total_tp, total_fp, total_fn)
    if distances:
        summary["mean_point_distance"] = float(sum(distances) / len(distances))
    if field_name == "lanes":
        summary["color_accuracy"] = _safe_div(color_hits, attr_total)
        summary["type_accuracy"] = _safe_div(type_hits, attr_total)
    if field_name == "stop_lines":
        summary["mean_angle_error"] = float(sum(angle_errors) / len(angle_errors)) if angle_errors else 0.0
    if field_name == "crosswalks":
        summary["mean_polygon_iou"] = float(sum(polygon_ious) / len(polygon_ious)) if polygon_ious else 0.0
        summary["mean_vertex_distance"] = float(sum(distances) / len(distances)) if distances else 0.0
    return summary


def _collect_lane_family_histogram_values(
    pred_sample: dict[str, Any],
    gt_sample: dict[str, Any],
    *,
    field_name: str,
    target_count: int,
    match_threshold: float | None = None,
    iou_threshold: float | None = None,
) -> list[float]:
    values: list[float] = []
    pred_rows = list(pred_sample[field_name])
    gt_rows = list(gt_sample[field_name])

    if field_name == "crosswalks":
        similarity_matrix = np.zeros((len(pred_rows), len(gt_rows)), dtype=np.float32)
        for pred_index, pred in enumerate(pred_rows):
            for gt_index, gt in enumerate(gt_rows):
                similarity_matrix[pred_index, gt_index] = _polygon_iou(pred["points_xy"], gt["points_xy"])
        matches = _hungarian_from_similarity(similarity_matrix, min_similarity=float(iou_threshold or 0.0))
    else:
        cost_matrix = np.zeros((len(pred_rows), len(gt_rows)), dtype=np.float32)
        for pred_index, pred in enumerate(pred_rows):
            for gt_index, gt in enumerate(gt_rows):
                cost_matrix[pred_index, gt_index] = _mean_point_distance(
                    pred["points_xy"],
                    gt["points_xy"],
                    target_count,
                )
        matches = _hungarian_from_cost(cost_matrix, max_cost=float(match_threshold or 0.0))

    for pred_index, gt_index in matches:
        pred = pred_rows[pred_index]
        gt = gt_rows[gt_index]
        if field_name == "lanes":
            values.append(_mean_point_distance(pred["points_xy"], gt["points_xy"], target_count))
        elif field_name == "stop_lines":
            values.append(_segment_angle_error(pred["points_xy"], gt["points_xy"], target_count))
        elif field_name == "crosswalks":
            values.append(_polygon_iou(pred["points_xy"], gt["points_xy"]))
    return values


def summarize_pv26_metrics(
    predictions: list[dict[str, Any]],
    batch: dict[str, Any],
    *,
    config: PV26MetricConfig | None = None,
) -> dict[str, Any]:
    config = config or PV26MetricConfig()
    gt_samples = _extract_gt_samples(batch)
    detector_summary = _detector_metrics(
        predictions,
        gt_samples,
        config=config,
        iou_threshold=config.det_iou_threshold,
    )
    detector_thresholds = [0.50 + 0.05 * index for index in range(10)]
    detector_map50_95 = sum(
        _detector_metrics(
            predictions,
            gt_samples,
            config=config,
            iou_threshold=threshold,
        )["map50"]
        for threshold in detector_thresholds
    ) / len(detector_thresholds)
    detector_summary["map50_95"] = detector_map50_95
    return {
        "detector": detector_summary,
        "traffic_light": _tl_metrics(
            predictions,
            gt_samples,
            iou_threshold=config.tl_iou_threshold,
            bit_threshold=config.tl_bit_threshold,
        ),
        "lane": _lane_family_metrics(
            predictions,
            gt_samples,
            field_name="lanes",
            target_count=16,
            match_threshold=config.lane_match_threshold,
        ),
        "stop_line": _lane_family_metrics(
            predictions,
            gt_samples,
            field_name="stop_lines",
            target_count=4,
            match_threshold=config.stop_line_match_threshold,
        ),
        "crosswalk": _lane_family_metrics(
            predictions,
            gt_samples,
            field_name="crosswalks",
            target_count=8,
            iou_threshold=config.crosswalk_iou_threshold,
        ),
    }


__all__ = [
    "PV26MetricConfig",
    "summarize_pv26_metrics",
    "summarize_pv26_tensorboard_histograms",
]
