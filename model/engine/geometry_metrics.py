"""Match decoded road-marking lines to raw-image annotation polylines."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from common.schema import ROADMARK_CLASSES


def _sample_line(points_xy: Sequence[Sequence[float]], spacing_px: float) -> np.ndarray:
    points = np.asarray(points_xy, dtype=np.float64)
    if len(points) < 2:
        return points.reshape(-1, 2)
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    keep = np.r_[True, segment_lengths > 0]
    points = points[keep]
    if len(points) == 1:
        return points
    distances = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    positions = np.linspace(0.0, distances[-1], max(2, math.ceil(distances[-1] / spacing_px) + 1))
    return np.column_stack((np.interp(positions, distances, points[:, 0]),
                            np.interp(positions, distances, points[:, 1])))


def _line_distance_and_coverage(
    predicted: np.ndarray, ground_truth: np.ndarray, tolerance_px: float
) -> tuple[float, float, float]:
    if not len(predicted) or not len(ground_truth):
        return math.inf, 0.0, 0.0
    pred_to_gt = cKDTree(ground_truth).query(predicted)[0]
    gt_to_pred = cKDTree(predicted).query(ground_truth)[0]
    distance = (float(pred_to_gt.mean()) + float(gt_to_pred.mean())) / 2.0
    return distance, float(np.mean(gt_to_pred <= tolerance_px)), float(np.mean(pred_to_gt <= tolerance_px))


def _stop_angle_error_deg(predicted: np.ndarray, ground_truth: np.ndarray) -> float | None:
    if len(predicted) < 2 or len(ground_truth) < 2:
        return None
    pred_vector = predicted[-1] - predicted[0]
    gt_vector = ground_truth[-1] - ground_truth[0]
    if np.linalg.norm(pred_vector) == 0 or np.linalg.norm(gt_vector) == 0:
        return None
    pred_angle = math.atan2(float(pred_vector[1]), float(pred_vector[0]))
    gt_angle = math.atan2(float(gt_vector[1]), float(gt_vector[0]))
    difference = pred_angle - gt_angle
    # A line has no forward direction: reversing its endpoints changes no angle.
    return math.degrees(abs(math.atan2(math.sin(2 * difference), math.cos(2 * difference))) / 2)


def match_roadmark_lines(
    predicted: Sequence[Mapping[str, Any]],
    ground_truth: Sequence[Mapping[str, Any]],
    *,
    tolerance_px: float = 8.0,
) -> dict[str, dict[str, int | float]]:
    """Return per-class one-to-one line counts and raw-pixel geometry sums.

    Each line is sampled uniformly along arc length at no more than half the
    matching tolerance. A pair matches when its symmetric mean nearest-point
    distance is within ``tolerance_px``. The tolerance is an evaluation setting,
    not a model-acceptance threshold. Coverage reports the sampled fraction of
    each matched line lying within that same distance.
    """
    if not math.isfinite(tolerance_px) or tolerance_px <= 0:
        raise ValueError("tolerance_px must be finite and positive")
    result: dict[str, dict[str, int | float]] = {}
    for class_name in ROADMARK_CLASSES:
        preds = [line for line in predicted if line["class_name"] == class_name]
        gts = [line for line in ground_truth if line["class_name"] == class_name]
        metrics: dict[str, int | float] = {
            "tp": 0, "fp": len(preds), "fn": len(gts),
            "matched_count": 0, "matched_distance_sum": 0.0,
            "matched_gt_coverage_sum": 0.0, "matched_pred_coverage_sum": 0.0,
        }
        if class_name == "stop_line":
            metrics.update(matched_angle_error_sum_deg=0.0, matched_angle_count=0)
        if preds and gts:
            spacing_px = tolerance_px / 2.0
            pred_points = [_sample_line(line["points_xy"], spacing_px) for line in preds]
            gt_points = [_sample_line(line["points_xy"], spacing_px) for line in gts]
            distances = np.empty((len(preds), len(gts)), dtype=np.float64)
            gt_coverage = np.empty_like(distances)
            pred_coverage = np.empty_like(distances)
            for pred_index, pred_line in enumerate(pred_points):
                for gt_index, gt_line in enumerate(gt_points):
                    distance, gt_fraction, pred_fraction = _line_distance_and_coverage(
                        pred_line, gt_line, tolerance_px
                    )
                    distances[pred_index, gt_index] = distance
                    gt_coverage[pred_index, gt_index] = gt_fraction
                    pred_coverage[pred_index, gt_index] = pred_fraction
            # A disallowed pair must cost more than every allowed pair combined,
            # so assignment maximizes valid matches before minimizing distance.
            disallowed_cost = tolerance_px * (min(len(preds), len(gts)) + 1) + 1.0
            assignment_cost = np.where(distances <= tolerance_px, distances, disallowed_cost)
            pred_indices, gt_indices = linear_sum_assignment(assignment_cost)
            for pred_index, gt_index in zip(pred_indices, gt_indices):
                if distances[pred_index, gt_index] > tolerance_px:
                    continue
                metrics["tp"] += 1
                metrics["fp"] -= 1
                metrics["fn"] -= 1
                metrics["matched_count"] += 1
                metrics["matched_distance_sum"] += float(distances[pred_index, gt_index])
                metrics["matched_gt_coverage_sum"] += float(gt_coverage[pred_index, gt_index])
                metrics["matched_pred_coverage_sum"] += float(pred_coverage[pred_index, gt_index])
                if class_name == "stop_line":
                    angle = _stop_angle_error_deg(pred_points[pred_index], gt_points[gt_index])
                    if angle is not None:
                        metrics["matched_angle_error_sum_deg"] += angle
                        metrics["matched_angle_count"] += 1
        result[class_name] = metrics
    return result
