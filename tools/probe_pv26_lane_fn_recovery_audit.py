from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import site
import sys
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
repo_root = str(REPO_ROOT)
if repo_root not in sys.path:
    site.addsitedir(repo_root)

from model.engine import raw_batch_for_metrics
from model.engine._trainer_epochs import _merge_raw_batches
from model.engine.batch import augment_lane_family_metrics
from model.engine.lane_segfirst_vectorizer import lane_segfirst_prediction_maps
from model.engine.metrics import _extract_gt_samples, _hungarian_from_cost, _mean_point_distance, summarize_pv26_metrics
from model.engine.postprocess import postprocess_pv26_batch
from tools.evaluate_pv26_lane60_checkpoint import _advance_validation_sampler, _postprocess_override_config
from tools.probe_pv26_lane_flip_tta import _merge_lane_dense_predictions, _unflip_lane_dense_outputs
from tools.probe_pv26_lane_instance_evidence import (
    _as_channel,
    _max_low_run_fraction,
    _polyline_length,
    _raw_points_to_map,
    _resolve_dataset_root,
    _resolve_device,
    _safe_stat,
    _sample_polyline,
    _track_mask,
    _values_at_points,
    _write_csv,
)
from tools.probe_pv26_lane60_postprocess_thresholds import _detach_to_cpu
from tools.pv26_train import cli as train_cli
from tools.pv26_train import config as train_config_api
from tools.run_pv26_lane60_probe import _lane60_scenario


SOURCE_RUN = (
    REPO_ROOT
    / "runs"
    / "pv26_exhaustive_od_lane_train"
    / "lane60_core_centerline_refine_cross_retain_from_exhaustive_od_lane_default_20260505_032217_default_20260510_003412"
)
DEFAULT_CHECKPOINT = (
    REPO_ROOT
    / "runs"
    / "pv26_exhaustive_od_lane_train"
    / "lane60_lane_head_transplant_original_stop_pca_20260512"
    / "merged_lane_head.pt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only lane FN recovery audit. It checks whether missed GT lanes "
            "already have predicted centerline evidence or nearby unmatched row-scan tracks."
        )
    )
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--source-run", default=str(SOURCE_RUN))
    parser.add_argument("--lane60-experiment", default="core_centerline_refine_row_scan_tangent_link")
    parser.add_argument("--preset", default="default")
    parser.add_argument("--phase-index", type=int, default=4)
    parser.add_argument("--max-val-batches", type=int, default=128)
    parser.add_argument("--validation-epoch", type=int, default=2)
    parser.add_argument("--train-batches", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--dataset-root",
        default="",
        help="Optional canonical dataset root. If omitted, infer from source-run/scenario paths.",
    )
    parser.add_argument("--lane-flip-variant", choices=("baseline", "flip_centerline_avg"), default="flip_centerline_avg")
    parser.add_argument("--lane-obj-threshold", type=float, default=None)
    parser.add_argument("--lane-segfirst-track-mode", default=None)
    parser.add_argument("--lane-segfirst-max-row-gap", type=int, default=None)
    parser.add_argument("--lane-segfirst-max-link-dx", type=float, default=None)
    parser.add_argument("--lane-segfirst-max-turn-degrees", type=float, default=None)
    parser.add_argument("--stop-line-mask-binary-threshold", type=float, default=None)
    parser.add_argument("--stop-line-min-instance-score", type=float, default=None)
    parser.add_argument("--stop-line-presence-threshold", type=float, default=None)
    parser.add_argument(
        "--stop-line-component-gate-source",
        choices=("center", "selector", "max", "validator", "max_validator", "product_validator"),
        default=None,
    )
    parser.add_argument("--crosswalk-obj-threshold", type=float, default=None)
    parser.add_argument("--crosswalk-mask-binary-threshold", type=float, default=None)
    parser.add_argument("--crosswalk-min-component-pixels", type=int, default=None)
    parser.add_argument("--crosswalk-max-components", type=int, default=None)
    parser.add_argument("--crosswalk-min-polygon-area-px", type=float, default=None)
    parser.add_argument("--crosswalk-min-bbox-aspect", type=float, default=None)
    parser.add_argument("--crosswalk-polygon-mode", choices=("rect", "hull"), default="hull")
    parser.add_argument("--output-dir", default="")
    return parser.parse_args()


def _f1(tp: int, fp: int, fn: int) -> float:
    denom = 2 * int(tp) + int(fp) + int(fn)
    if denom <= 0:
        return 0.0
    return float(2 * int(tp) / denom)


def _distance_bin(distance: float) -> str:
    if not math.isfinite(distance):
        return "none"
    if distance <= 40.0:
        return "le40"
    if distance <= 80.0:
        return "40_80"
    if distance <= 120.0:
        return "80_120"
    if distance <= 200.0:
        return "120_200"
    return "gt200"


def _lane_distance(pred: dict[str, Any], gt: dict[str, Any]) -> float:
    return float(_mean_point_distance(pred.get("points_xy", []), gt.get("points_xy", []), 16))


def _polyline_samples(points_xy: list[list[float]], *, count: int = 16) -> np.ndarray:
    points = np.asarray(points_xy, dtype=np.float32).reshape(-1, 2)
    if points.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32)
    return _sample_polyline(points, count=count)


def _polyline_axis(points_xy: list[list[float]]) -> np.ndarray | None:
    samples = _polyline_samples(points_xy, count=16)
    if samples.shape[0] < 2:
        return None
    delta = np.asarray(samples[-1] - samples[0], dtype=np.float32)
    norm = float(np.linalg.norm(delta))
    if norm <= 1.0e-6 or not bool(np.isfinite(delta).all()):
        return None
    return delta / norm


def _axis_angle_error_degrees(first: list[list[float]], second: list[list[float]]) -> float:
    axis_a = _polyline_axis(first)
    axis_b = _polyline_axis(second)
    if axis_a is None or axis_b is None:
        return math.inf
    dot = float(np.clip(abs(float(np.dot(axis_a, axis_b))), -1.0, 1.0))
    return float(math.degrees(math.acos(dot)))


def _range_overlap_fraction(
    first_min: float,
    first_max: float,
    second_min: float,
    second_max: float,
) -> float:
    denom = max(float(first_max) - float(first_min), 1.0e-6)
    overlap = max(0.0, min(float(first_max), float(second_max)) - max(float(first_min), float(second_min)))
    return float(overlap / denom)


def _aligned_polyline_samples(
    gt_lane: dict[str, Any],
    pred_lane: dict[str, Any],
    *,
    count: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    gt_samples = _polyline_samples(list(gt_lane.get("points_xy", [])), count=count)
    pred_samples = _polyline_samples(list(pred_lane.get("points_xy", [])), count=count)
    if gt_samples.shape[0] == 0 or pred_samples.shape[0] == 0:
        return gt_samples, pred_samples
    if gt_samples.shape[0] != pred_samples.shape[0]:
        sample_count = min(int(gt_samples.shape[0]), int(pred_samples.shape[0]))
        gt_samples = gt_samples[:sample_count]
        pred_samples = pred_samples[:sample_count]
    direct = float(np.linalg.norm(pred_samples - gt_samples, axis=1).mean())
    flipped = float(np.linalg.norm(pred_samples[::-1] - gt_samples, axis=1).mean())
    if flipped < direct:
        pred_samples = pred_samples[::-1]
    return gt_samples, pred_samples


def _pair_geometry_features(
    gt_lane: dict[str, Any],
    pred_lane: dict[str, Any] | None,
    *,
    prefix: str,
) -> dict[str, float]:
    keys = {
        f"{prefix}_polyline_length": math.inf,
        f"{prefix}_length_ratio": math.inf,
        f"{prefix}_angle_error_degrees": math.inf,
        f"{prefix}_center_dx": math.inf,
        f"{prefix}_center_dy": math.inf,
        f"{prefix}_center_distance": math.inf,
        f"{prefix}_sample_mean_dx": math.inf,
        f"{prefix}_sample_mean_dy": math.inf,
        f"{prefix}_sample_mean_abs_dx": math.inf,
        f"{prefix}_sample_mean_distance": math.inf,
        f"{prefix}_sample_q90_distance": math.inf,
        f"{prefix}_endpoint_mean_distance": math.inf,
        f"{prefix}_endpoint_max_distance": math.inf,
        f"{prefix}_x_overlap_fraction": 0.0,
        f"{prefix}_y_overlap_fraction": 0.0,
    }
    if pred_lane is None:
        return keys
    gt_points = np.asarray(gt_lane.get("points_xy", []), dtype=np.float32).reshape(-1, 2)
    pred_points = np.asarray(pred_lane.get("points_xy", []), dtype=np.float32).reshape(-1, 2)
    if gt_points.shape[0] < 2 or pred_points.shape[0] < 2:
        return keys
    gt_samples, pred_samples = _aligned_polyline_samples(gt_lane, pred_lane, count=16)
    if gt_samples.shape[0] < 2 or pred_samples.shape[0] < 2:
        return keys
    deltas = pred_samples - gt_samples
    distances = np.linalg.norm(deltas, axis=1)
    endpoint_distances = np.asarray([distances[0], distances[-1]], dtype=np.float32)
    gt_center = gt_samples.mean(axis=0)
    pred_center = pred_samples.mean(axis=0)
    center_delta = pred_center - gt_center
    gt_length = float(_polyline_length(list(gt_lane.get("points_xy", []))))
    pred_length = float(_polyline_length(list(pred_lane.get("points_xy", []))))
    keys.update(
        {
            f"{prefix}_polyline_length": pred_length,
            f"{prefix}_length_ratio": float(pred_length / max(gt_length, 1.0e-6)),
            f"{prefix}_angle_error_degrees": _axis_angle_error_degrees(
                list(gt_lane.get("points_xy", [])),
                list(pred_lane.get("points_xy", [])),
            ),
            f"{prefix}_center_dx": float(center_delta[0]),
            f"{prefix}_center_dy": float(center_delta[1]),
            f"{prefix}_center_distance": float(np.linalg.norm(center_delta)),
            f"{prefix}_sample_mean_dx": float(deltas[:, 0].mean()),
            f"{prefix}_sample_mean_dy": float(deltas[:, 1].mean()),
            f"{prefix}_sample_mean_abs_dx": float(np.abs(deltas[:, 0]).mean()),
            f"{prefix}_sample_mean_distance": float(distances.mean()),
            f"{prefix}_sample_q90_distance": float(np.quantile(distances, 0.90)),
            f"{prefix}_endpoint_mean_distance": float(endpoint_distances.mean()),
            f"{prefix}_endpoint_max_distance": float(endpoint_distances.max()),
            f"{prefix}_x_overlap_fraction": _range_overlap_fraction(
                float(gt_points[:, 0].min()),
                float(gt_points[:, 0].max()),
                float(pred_points[:, 0].min()),
                float(pred_points[:, 0].max()),
            ),
            f"{prefix}_y_overlap_fraction": _range_overlap_fraction(
                float(gt_points[:, 1].min()),
                float(gt_points[:, 1].max()),
                float(pred_points[:, 1].min()),
                float(pred_points[:, 1].max()),
            ),
        }
    )
    return keys


def _match_predictions(
    pred_rows: list[dict[str, Any]],
    gt_rows: list[dict[str, Any]],
) -> tuple[dict[int, tuple[int, float]], dict[int, tuple[int, float]]]:
    if not pred_rows or not gt_rows:
        return {}, {}
    cost = np.zeros((len(pred_rows), len(gt_rows)), dtype=np.float32)
    for pred_index, pred in enumerate(pred_rows):
        for gt_index, gt in enumerate(gt_rows):
            cost[pred_index, gt_index] = _lane_distance(pred, gt)
    matches = _hungarian_from_cost(cost, max_cost=40.0)
    pred_to_gt = {
        int(pred_index): (int(gt_index), float(cost[pred_index, gt_index]))
        for pred_index, gt_index in matches
    }
    gt_to_pred = {
        int(gt_index): (int(pred_index), float(cost[pred_index, gt_index]))
        for pred_index, gt_index in matches
    }
    return pred_to_gt, gt_to_pred


def _nearest_prediction(
    gt: dict[str, Any],
    predictions: list[dict[str, Any]],
    *,
    allowed_indices: set[int] | None = None,
) -> tuple[int, float]:
    best_index = -1
    best_distance = math.inf
    for index, pred in enumerate(predictions):
        if allowed_indices is not None and index not in allowed_indices:
            continue
        distance = _lane_distance(pred, gt)
        if distance < best_distance:
            best_index = int(index)
            best_distance = float(distance)
    return best_index, best_distance


def _nearest_gt_lane(
    pred: dict[str, Any],
    gt_rows: list[dict[str, Any]],
    *,
    allowed_indices: set[int] | None = None,
) -> tuple[int, float]:
    best_index = -1
    best_distance = math.inf
    for index, gt in enumerate(gt_rows):
        if allowed_indices is not None and index not in allowed_indices:
            continue
        distance = _lane_distance(pred, gt)
        if distance < best_distance:
            best_index = int(index)
            best_distance = float(distance)
    return best_index, best_distance


def _gt_centerline_evidence(
    gt_lane: dict[str, Any],
    *,
    maps: dict[str, torch.Tensor],
    meta: dict[str, Any],
) -> dict[str, float]:
    centerline = _as_channel(maps["centerline_core"])
    support = _as_channel(maps["support"])
    map_hw = (int(centerline.shape[0]), int(centerline.shape[1]))
    map_points = _raw_points_to_map(list(gt_lane.get("points_xy", [])), meta, map_hw)
    mask = _track_mask(map_points, map_hw, width=3)
    sampled_points = _sample_polyline(map_points, count=64)
    center_values = centerline[mask]
    support_values = support[mask]
    center_samples = _values_at_points(centerline, sampled_points)
    support_samples = _values_at_points(support, sampled_points)
    return {
        "gt_track_pixels": float(mask.sum()),
        "gt_center_mask_mean": _safe_stat(center_values, "mean"),
        "gt_center_mask_q10": _safe_stat(center_values, "q10"),
        "gt_center_mask_active05": float((center_values >= 0.5).mean()) if center_values.size else 0.0,
        "gt_center_point_mean": _safe_stat(center_samples, "mean"),
        "gt_center_point_q10": _safe_stat(center_samples, "q10"),
        "gt_center_point_active05": float((center_samples >= 0.5).mean()) if center_samples.size else 0.0,
        "gt_center_point_low_run05": _max_low_run_fraction(center_samples, threshold=0.5),
        "gt_support_point_mean": _safe_stat(support_samples, "mean"),
        "gt_support_point_q10": _safe_stat(support_samples, "q10"),
        "gt_support_mask_mean": _safe_stat(support_values, "mean"),
    }


def _track_map_evidence(
    points_xy: list[list[float]],
    *,
    maps: dict[str, torch.Tensor],
    meta: dict[str, Any],
    prefix: str,
) -> dict[str, float]:
    centerline = _as_channel(maps["centerline_core"])
    support = _as_channel(maps["support"])
    map_hw = (int(centerline.shape[0]), int(centerline.shape[1]))
    map_points = _raw_points_to_map(points_xy, meta, map_hw)
    mask = _track_mask(map_points, map_hw, width=3)
    sampled_points = _sample_polyline(map_points, count=64)
    center_values = centerline[mask]
    support_values = support[mask]
    center_samples = _values_at_points(centerline, sampled_points)
    support_samples = _values_at_points(support, sampled_points)
    return {
        f"{prefix}_track_pixels": float(mask.sum()),
        f"{prefix}_center_mask_mean": _safe_stat(center_values, "mean"),
        f"{prefix}_center_mask_q10": _safe_stat(center_values, "q10"),
        f"{prefix}_center_mask_active05": float((center_values >= 0.5).mean()) if center_values.size else 0.0,
        f"{prefix}_center_point_mean": _safe_stat(center_samples, "mean"),
        f"{prefix}_center_point_q10": _safe_stat(center_samples, "q10"),
        f"{prefix}_center_point_active05": float((center_samples >= 0.5).mean()) if center_samples.size else 0.0,
        f"{prefix}_center_point_low_run05": _max_low_run_fraction(center_samples, threshold=0.5),
        f"{prefix}_support_point_mean": _safe_stat(support_samples, "mean"),
        f"{prefix}_support_point_q10": _safe_stat(support_samples, "q10"),
        f"{prefix}_support_mask_mean": _safe_stat(support_values, "mean"),
    }


def _prediction_shape_features(pred_lane: dict[str, Any]) -> dict[str, float]:
    points = np.asarray(pred_lane.get("points_xy", []), dtype=np.float32).reshape(-1, 2)
    if points.shape[0] == 0:
        return {
            "pred_point_count": 0.0,
            "pred_polyline_length": 0.0,
            "pred_center_x": math.inf,
            "pred_center_y": math.inf,
            "pred_top_y": math.inf,
            "pred_bottom_y": math.inf,
            "pred_bbox_width": 0.0,
            "pred_bbox_height": 0.0,
            "pred_bbox_aspect": math.inf,
            "pred_axis_angle_degrees": math.inf,
        }
    width = float(points[:, 0].max() - points[:, 0].min())
    height = float(points[:, 1].max() - points[:, 1].min())
    return {
        "pred_point_count": float(points.shape[0]),
        "pred_polyline_length": float(_polyline_length(list(pred_lane.get("points_xy", [])))),
        "pred_center_x": float(points[:, 0].mean()),
        "pred_center_y": float(points[:, 1].mean()),
        "pred_top_y": float(points[:, 1].min()),
        "pred_bottom_y": float(points[:, 1].max()),
        "pred_bbox_width": width,
        "pred_bbox_height": height,
        "pred_bbox_aspect": float(max(width, height) / max(min(width, height), 1.0e-6)),
        "pred_axis_angle_degrees": float(_axis_angle_error_degrees(
            [[0.0, 0.0], [1.0, 0.0]],
            list(pred_lane.get("points_xy", [])),
        )),
    }


def _points_json(lane: dict[str, Any] | None) -> str:
    if lane is None:
        return "[]"
    points = np.asarray(lane.get("points_xy", []), dtype=np.float32).reshape(-1, 2)
    if points.shape[0] == 0:
        return "[]"
    payload = [[float(x), float(y)] for x, y in points.tolist()]
    return json.dumps(payload, separators=(",", ":"))


def _nearest_other_prediction_distance(pred_index: int, predictions: list[dict[str, Any]]) -> float:
    if pred_index < 0 or pred_index >= len(predictions):
        return math.inf
    best = math.inf
    pred = predictions[pred_index]
    for other_index, other in enumerate(predictions):
        if int(other_index) == int(pred_index):
            continue
        best = min(best, _lane_distance(pred, other))
    return float(best)


def _bucket_counts(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key, ""))
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _count_at_least(rows: list[dict[str, Any]], key: str, threshold: float) -> int:
    return sum(1 for row in rows if float(row.get(key, 0.0)) >= float(threshold))


def _count_at_most(rows: list[dict[str, Any]], key: str, threshold: float) -> int:
    return sum(1 for row in rows if float(row.get(key, math.inf)) <= float(threshold))


def _finite_float(row: dict[str, Any], key: str, default: float = math.inf) -> float:
    try:
        value = float(row.get(key, default))
    except (TypeError, ValueError):
        return float(default)
    return value if math.isfinite(value) else float(default)


def _quantile(rows: list[dict[str, Any]], key: str, q: float) -> float:
    values = np.asarray([_finite_float(row, key) for row in rows], dtype=np.float32)
    values = values[np.isfinite(values)]
    if int(values.size) == 0:
        return math.inf
    return float(np.quantile(values, float(q)))


def _geometry_group_summary(
    rows: list[dict[str, Any]],
    *,
    name: str,
    predicate: Any,
) -> dict[str, Any]:
    group = [row for row in rows if bool(predicate(row))]
    return {
        "name": name,
        "count": int(len(group)),
        "nearest_unmatched_distance_q50": _quantile(group, "nearest_unmatched_pred_distance", 0.50),
        "nearest_unmatched_length_ratio_q50": _quantile(group, "nearest_unmatched_pred_length_ratio", 0.50),
        "nearest_unmatched_angle_error_degrees_q50": _quantile(group, "nearest_unmatched_pred_angle_error_degrees", 0.50),
        "nearest_unmatched_center_distance_q50": _quantile(group, "nearest_unmatched_pred_center_distance", 0.50),
        "nearest_unmatched_sample_mean_abs_dx_q50": _quantile(group, "nearest_unmatched_pred_sample_mean_abs_dx", 0.50),
        "nearest_unmatched_sample_mean_distance_q50": _quantile(group, "nearest_unmatched_pred_sample_mean_distance", 0.50),
        "nearest_unmatched_endpoint_mean_distance_q50": _quantile(group, "nearest_unmatched_pred_endpoint_mean_distance", 0.50),
        "nearest_unmatched_y_overlap_fraction_q50": _quantile(group, "nearest_unmatched_pred_y_overlap_fraction", 0.50),
    }


def _geometry_summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        _geometry_group_summary(
            rows,
            name="all_with_unmatched",
            predicate=lambda row: int(row.get("nearest_unmatched_pred_index", -1)) >= 0,
        ),
        _geometry_group_summary(
            rows,
            name="unmatched_le80",
            predicate=lambda row: _finite_float(row, "nearest_unmatched_pred_distance") <= 80.0,
        ),
        _geometry_group_summary(
            rows,
            name="center050_and_unmatched_le80",
            predicate=lambda row: _finite_float(row, "gt_center_point_mean", -math.inf) >= 0.50
            and _finite_float(row, "nearest_unmatched_pred_distance") <= 80.0,
        ),
        _geometry_group_summary(
            rows,
            name="center050_and_unmatched_le120",
            predicate=lambda row: _finite_float(row, "gt_center_point_mean", -math.inf) >= 0.50
            and _finite_float(row, "nearest_unmatched_pred_distance") <= 120.0,
        ),
        _geometry_group_summary(
            rows,
            name="center050_without_unmatched_le120",
            predicate=lambda row: _finite_float(row, "gt_center_point_mean", -math.inf) >= 0.50
            and _finite_float(row, "nearest_unmatched_pred_distance") > 120.0,
        ),
        _geometry_group_summary(
            rows,
            name="unmatched_le120_without_center050",
            predicate=lambda row: _finite_float(row, "nearest_unmatched_pred_distance") <= 120.0
            and _finite_float(row, "gt_center_point_mean", -math.inf) < 0.50,
        ),
    ]


def _unmatched_prediction_group_summary(
    rows: list[dict[str, Any]],
    *,
    name: str,
    predicate: Any,
) -> dict[str, Any]:
    group = [row for row in rows if bool(predicate(row))]
    return {
        "name": name,
        "count": int(len(group)),
        "pred_polyline_length_q50": _quantile(group, "pred_polyline_length", 0.50),
        "pred_bbox_aspect_q50": _quantile(group, "pred_bbox_aspect", 0.50),
        "pred_center_point_mean_q50": _quantile(group, "pred_center_point_mean", 0.50),
        "pred_center_point_q10_q50": _quantile(group, "pred_center_point_q10", 0.50),
        "pred_support_point_mean_q50": _quantile(group, "pred_support_point_mean", 0.50),
        "nearest_other_pred_distance_q50": _quantile(group, "nearest_other_pred_distance", 0.50),
        "nearest_fn_gt_distance_q50": _quantile(group, "nearest_fn_gt_distance", 0.50),
    }


def _unmatched_prediction_summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        _unmatched_prediction_group_summary(
            rows,
            name="all_unmatched_predictions",
            predicate=lambda row: True,
        ),
        _unmatched_prediction_group_summary(
            rows,
            name="repairable_le80_center050",
            predicate=lambda row: bool(row.get("repairable_le80_center050")),
        ),
        _unmatched_prediction_group_summary(
            rows,
            name="repairable_le120_any_center",
            predicate=lambda row: bool(row.get("repairable_le120_any_center")),
        ),
        _unmatched_prediction_group_summary(
            rows,
            name="not_repairable_le120_any_center",
            predicate=lambda row: not bool(row.get("repairable_le120_any_center")),
        ),
    ]


def _upper_bound_rows(
    fn_rows: list[dict[str, Any]],
    *,
    baseline_tp: int,
    baseline_fp: int,
    baseline_fn: int,
) -> list[dict[str, Any]]:
    rules: list[tuple[str, int]] = []
    for threshold in (0.30, 0.50, 0.70):
        rules.append((f"recover_gt_center_point_mean_ge_{threshold:.2f}", _count_at_least(fn_rows, "gt_center_point_mean", threshold)))
    for threshold in (0.10, 0.30, 0.50):
        rules.append((f"recover_gt_center_point_q10_ge_{threshold:.2f}", _count_at_least(fn_rows, "gt_center_point_q10", threshold)))
    for threshold in (40.0, 80.0, 120.0, 200.0):
        rules.append((f"recover_nearest_unmatched_pred_le_{int(threshold)}", _count_at_most(fn_rows, "nearest_unmatched_pred_distance", threshold)))
    for center_threshold, distance_threshold in ((0.50, 80.0), (0.50, 120.0), (0.30, 120.0)):
        count = sum(
            1
            for row in fn_rows
            if float(row.get("gt_center_point_mean", 0.0)) >= center_threshold
            or float(row.get("nearest_unmatched_pred_distance", math.inf)) <= distance_threshold
        )
        rules.append((f"recover_center_ge_{center_threshold:.2f}_or_unmatched_le_{int(distance_threshold)}", count))

    rows = []
    for name, recovered in rules:
        recovered = min(int(recovered), int(baseline_fn))
        tp = int(baseline_tp) + recovered
        fn = max(0, int(baseline_fn) - recovered)
        rows.append(
            {
                "rule": name,
                "recovered_fn_count": recovered,
                "upper_bound_lane_tp": tp,
                "upper_bound_lane_fp": int(baseline_fp),
                "upper_bound_lane_fn": fn,
                "upper_bound_lane_f1_no_new_fp": _f1(tp, int(baseline_fp), fn),
            }
        )
    rows.sort(key=lambda row: float(row["upper_bound_lane_f1_no_new_fp"]), reverse=True)
    return rows


def _metric(metrics: dict[str, Any], task: str, key: str) -> float:
    payload = metrics.get(task, {}) if isinstance(metrics.get(task), dict) else {}
    value = payload.get(key, 0.0)
    return float(value) if isinstance(value, (int, float)) else 0.0


def main() -> int:
    args = parse_args()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    source_run = Path(args.source_run).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not source_run.is_dir():
        raise FileNotFoundError(source_run)

    scenario_args = argparse.Namespace(
        preset=str(args.preset),
        source_run=str(source_run),
        seed_checkpoint=str(checkpoint),
        experiment=str(args.lane60_experiment),
        epochs=1,
        train_batches=int(args.train_batches),
        val_batches=int(args.max_val_batches),
        batch_size=int(args.batch_size),
        device=str(args.device),
        run_root="",
        preview=False,
    )
    scenario, scenario_path, options = _lane60_scenario(
        scenario_args,
        source_run=source_run,
        seed_checkpoint=checkpoint,
    )
    dataset_root = _resolve_dataset_root(args, source_run, scenario.dataset.root)
    scenario = replace(
        scenario,
        dataset=train_config_api.DatasetConfig(
            root=dataset_root,
            additional_roots=tuple(scenario.dataset.additional_roots),
        ),
    )
    phase_index = int(tuple(options["selected_phase_indices"])[0])
    phase = scenario.phases[phase_index - 1]
    train_config = train_config_api.scenario_phase_defaults(scenario.train_defaults, phase.overrides)
    train_config = replace(
        train_config,
        device=_resolve_device(args.device, train_config.device),
        val_batches=int(args.max_val_batches),
    )
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if str(args.output_dir).strip()
        else source_run / "analysis_exports" / f"lane_fn_recovery_audit_val{int(args.max_val_batches)}_epoch{int(args.validation_epoch)}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    train_cli._configure_torch_multiprocessing()
    dataset = train_cli.PV26CanonicalDataset(
        train_cli._existing_dataset_roots(scenario),
        train_augmentation=False,
        progress_callback=lambda message: print(f"[lane_fn_recovery] {message}", flush=True),
    )
    _, val_loader = train_cli._build_phase_train_loaders(dataset, train_config=train_config, phase=phase)
    if val_loader is None:
        raise ValueError("lane FN recovery audit requires validation batches")
    _advance_validation_sampler(val_loader, validation_epoch=int(args.validation_epoch))

    trainer = train_cli._build_phase_trainer(phase, train_config)
    trainer.load_model_weights(checkpoint, map_location=train_config.device)
    postprocess_config = _postprocess_override_config(args, trainer) or train_cli._build_postprocess_config(train_config)
    evaluator = trainer.build_evaluator()
    evaluator.adapter.raw_model.eval()
    evaluator.heads.eval()

    predictions_all: list[dict[str, Any]] = []
    raw_batches: list[dict[str, Any]] = []
    fn_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    unmatched_prediction_rows: list[dict[str, Any]] = []
    global_sample_index = 0
    with torch.no_grad():
        for batch_index, batch in enumerate(val_loader, start=1):
            if batch_index > int(args.max_val_batches):
                break
            if batch_index == 1 or batch_index % 20 == 0:
                print(f"[lane_fn_recovery] eval batch {batch_index}/{args.max_val_batches}", flush=True)
            raw_batch = raw_batch_for_metrics(batch)
            if raw_batch is None:
                raise ValueError("validation loader must provide raw batches for metrics")
            encoded = evaluator.prepare_batch(batch)
            predictions = _detach_to_cpu(evaluator.forward_encoded_batch(encoded))
            flip_predictions = None
            if str(args.lane_flip_variant) != "baseline":
                flipped_encoded = dict(encoded)
                flipped_encoded["image"] = torch.flip(encoded["image"], dims=(-1,))
                flip_predictions = _unflip_lane_dense_outputs(
                    _detach_to_cpu(evaluator.forward_encoded_batch(flipped_encoded))
                )
            postprocess_predictions = (
                _merge_lane_dense_predictions(
                    predictions,
                    flip_predictions if flip_predictions is not None else {},
                    variant=str(args.lane_flip_variant),
                )
                if str(args.lane_flip_variant) != "baseline"
                else predictions
            )
            meta = _detach_to_cpu(encoded["meta"])
            batch_predictions = postprocess_pv26_batch(postprocess_predictions, meta, config=postprocess_config)
            batch_gt = _extract_gt_samples(raw_batch)
            predictions_all.extend(batch_predictions)
            raw_batches.append(raw_batch)
            for sample_batch_index, (sample_pred, sample_gt, sample_meta) in enumerate(
                zip(batch_predictions, batch_gt, meta)
            ):
                maps = lane_segfirst_prediction_maps(postprocess_predictions, batch_index=sample_batch_index)
                pred_lanes = list(sample_pred.get("lanes", []))
                gt_lanes = list(sample_gt.get("lanes", []))
                pred_to_gt, gt_to_pred = _match_predictions(pred_lanes, gt_lanes)
                matched_pred_indices = set(pred_to_gt)
                unmatched_pred_indices = set(range(len(pred_lanes))) - matched_pred_indices
                matched_gt_indices = set(gt_to_pred)
                unmatched_gt_indices = set(range(len(gt_lanes))) - matched_gt_indices
                fn_evidence_by_gt_index: dict[int, dict[str, float]] = {}
                sample_fn_count = max(0, len(gt_lanes) - len(matched_gt_indices))
                sample_rows.append(
                    {
                        "batch_index": int(batch_index),
                        "sample_index": int(global_sample_index),
                        "sample_batch_index": int(sample_batch_index),
                        "gt_lane_count": int(len(gt_lanes)),
                        "pred_lane_count": int(len(pred_lanes)),
                        "matched_lane_count": int(len(matched_gt_indices)),
                        "fn_lane_count": int(sample_fn_count),
                        "fp_lane_count": int(len(unmatched_pred_indices)),
                    }
                )
                for gt_index, gt_lane in enumerate(gt_lanes):
                    if gt_index in matched_gt_indices:
                        continue
                    nearest_any_index, nearest_any_distance = _nearest_prediction(gt_lane, pred_lanes)
                    nearest_unmatched_index, nearest_unmatched_distance = _nearest_prediction(
                        gt_lane,
                        pred_lanes,
                        allowed_indices=unmatched_pred_indices,
                    )
                    evidence = _gt_centerline_evidence(gt_lane, maps=maps, meta=sample_meta)
                    fn_evidence_by_gt_index[int(gt_index)] = evidence
                    row = {
                        "batch_index": int(batch_index),
                        "sample_index": int(global_sample_index),
                        "sample_batch_index": int(sample_batch_index),
                        "gt_index": int(gt_index),
                        "sample_gt_lane_count": int(len(gt_lanes)),
                        "sample_pred_lane_count": int(len(pred_lanes)),
                        "sample_matched_lane_count": int(len(matched_gt_indices)),
                        "sample_unmatched_pred_count": int(len(unmatched_pred_indices)),
                        "gt_polyline_length": float(_polyline_length(list(gt_lane.get("points_xy", [])))),
                        "nearest_any_pred_index": int(nearest_any_index),
                        "nearest_any_pred_distance": float(nearest_any_distance),
                        "nearest_any_pred_distance_bin": _distance_bin(nearest_any_distance),
                        "nearest_unmatched_pred_index": int(nearest_unmatched_index),
                        "nearest_unmatched_pred_distance": float(nearest_unmatched_distance),
                        "nearest_unmatched_pred_distance_bin": _distance_bin(nearest_unmatched_distance),
                        "gt_points_json": _points_json(gt_lane),
                        "nearest_any_pred_points_json": _points_json(
                            pred_lanes[nearest_any_index] if nearest_any_index >= 0 else None
                        ),
                        "nearest_unmatched_pred_points_json": _points_json(
                            pred_lanes[nearest_unmatched_index] if nearest_unmatched_index >= 0 else None
                        ),
                    }
                    row.update(
                        _pair_geometry_features(
                            gt_lane,
                            pred_lanes[nearest_any_index] if nearest_any_index >= 0 else None,
                            prefix="nearest_any_pred",
                        )
                    )
                    row.update(
                        _pair_geometry_features(
                            gt_lane,
                            pred_lanes[nearest_unmatched_index] if nearest_unmatched_index >= 0 else None,
                            prefix="nearest_unmatched_pred",
                        )
                    )
                    row.update(evidence)
                    fn_rows.append(row)
                for pred_index in sorted(unmatched_pred_indices):
                    pred_lane = pred_lanes[pred_index]
                    nearest_fn_gt_index, nearest_fn_gt_distance = _nearest_gt_lane(
                        pred_lane,
                        gt_lanes,
                        allowed_indices=unmatched_gt_indices,
                    )
                    nearest_fn_evidence = (
                        fn_evidence_by_gt_index.get(int(nearest_fn_gt_index), {})
                        if int(nearest_fn_gt_index) >= 0
                        else {}
                    )
                    nearest_fn_center_mean = float(nearest_fn_evidence.get("gt_center_point_mean", 0.0))
                    pred_row = {
                        "batch_index": int(batch_index),
                        "sample_index": int(global_sample_index),
                        "sample_batch_index": int(sample_batch_index),
                        "pred_index": int(pred_index),
                        "sample_gt_lane_count": int(len(gt_lanes)),
                        "sample_pred_lane_count": int(len(pred_lanes)),
                        "sample_matched_lane_count": int(len(matched_gt_indices)),
                        "sample_unmatched_pred_count": int(len(unmatched_pred_indices)),
                        "pred_class_name": str(pred_lane.get("class_name", "")),
                        "pred_lane_type": str(pred_lane.get("lane_type", "")),
                        "pred_score": float(pred_lane.get("score", 0.0)),
                        "pred_points_json": _points_json(pred_lane),
                        "nearest_fn_gt_index": int(nearest_fn_gt_index),
                        "nearest_fn_gt_distance": float(nearest_fn_gt_distance),
                        "nearest_fn_gt_distance_bin": _distance_bin(nearest_fn_gt_distance),
                        "nearest_fn_gt_points_json": _points_json(
                            gt_lanes[nearest_fn_gt_index] if nearest_fn_gt_index >= 0 else None
                        ),
                        "nearest_fn_gt_center_point_mean": nearest_fn_center_mean,
                        "repairable_le80_center050": bool(
                            nearest_fn_gt_distance <= 80.0 and nearest_fn_center_mean >= 0.50
                        ),
                        "repairable_le120_center050": bool(
                            nearest_fn_gt_distance <= 120.0 and nearest_fn_center_mean >= 0.50
                        ),
                        "repairable_le80_any_center": bool(nearest_fn_gt_distance <= 80.0),
                        "repairable_le120_any_center": bool(nearest_fn_gt_distance <= 120.0),
                        "nearest_other_pred_distance": _nearest_other_prediction_distance(pred_index, pred_lanes),
                    }
                    pred_row.update(_prediction_shape_features(pred_lane))
                    pred_row.update(
                        _track_map_evidence(
                            list(pred_lane.get("points_xy", [])),
                            maps=maps,
                            meta=sample_meta,
                            prefix="pred",
                        )
                    )
                    unmatched_prediction_rows.append(pred_row)
                global_sample_index += 1

    if not raw_batches:
        raise ValueError("no validation batches were processed")

    metrics = augment_lane_family_metrics(summarize_pv26_metrics(predictions_all, _merge_raw_batches(raw_batches)))
    baseline_tp = int(round(_metric(metrics, "lane", "tp")))
    baseline_fp = int(round(_metric(metrics, "lane", "fp")))
    baseline_fn = int(round(_metric(metrics, "lane", "fn")))
    upper_bounds = _upper_bound_rows(
        fn_rows,
        baseline_tp=baseline_tp,
        baseline_fp=baseline_fp,
        baseline_fn=baseline_fn,
    )

    _write_csv(output_dir / "lane_fn_recovery_rows.csv", fn_rows)
    _write_csv(output_dir / "lane_fn_recovery_samples.csv", sample_rows)
    _write_csv(output_dir / "lane_fn_recovery_upper_bounds.csv", upper_bounds)
    _write_csv(output_dir / "lane_unmatched_prediction_repair_rows.csv", unmatched_prediction_rows)

    summary = {
        "checkpoint": str(checkpoint),
        "source_run": str(source_run),
        "scenario_path": str(scenario_path),
        "lane60_experiment": str(args.lane60_experiment),
        "phase_index": int(phase_index),
        "max_val_batches": int(args.max_val_batches),
        "validation_epoch": int(args.validation_epoch),
        "lane_flip_variant": str(args.lane_flip_variant),
        "baseline_lane": {
            "precision": _metric(metrics, "lane", "precision"),
            "recall": _metric(metrics, "lane", "recall"),
            "f1": _metric(metrics, "lane", "f1"),
            "tp": baseline_tp,
            "fp": baseline_fp,
            "fn": baseline_fn,
        },
        "baseline_stop_line_f1": _metric(metrics, "stop_line", "f1"),
        "baseline_crosswalk_f1": _metric(metrics, "crosswalk", "f1"),
        "fn_lane_count": int(len(fn_rows)),
        "sample_count": int(len(sample_rows)),
        "samples_with_fn": int(sum(1 for row in sample_rows if int(row.get("fn_lane_count", 0)) > 0)),
        "nearest_any_pred_distance_bins": _bucket_counts(fn_rows, "nearest_any_pred_distance_bin"),
        "nearest_unmatched_pred_distance_bins": _bucket_counts(fn_rows, "nearest_unmatched_pred_distance_bin"),
        "gt_center_point_mean_ge_030": _count_at_least(fn_rows, "gt_center_point_mean", 0.30),
        "gt_center_point_mean_ge_050": _count_at_least(fn_rows, "gt_center_point_mean", 0.50),
        "gt_center_point_mean_ge_070": _count_at_least(fn_rows, "gt_center_point_mean", 0.70),
        "gt_center_point_q10_ge_030": _count_at_least(fn_rows, "gt_center_point_q10", 0.30),
        "best_upper_bounds": upper_bounds[:8],
        "nearest_unmatched_geometry": _geometry_summaries(fn_rows),
        "unmatched_prediction_count": int(len(unmatched_prediction_rows)),
        "repairable_unmatched_predictions_le80_center050": int(
            sum(1 for row in unmatched_prediction_rows if bool(row.get("repairable_le80_center050")))
        ),
        "repairable_unmatched_predictions_le120_any_center": int(
            sum(1 for row in unmatched_prediction_rows if bool(row.get("repairable_le120_any_center")))
        ),
        "unmatched_prediction_feature_summaries": _unmatched_prediction_summaries(unmatched_prediction_rows),
        "interpretation": (
            "This is not a production decoder. It estimates whether missed GT lanes have enough "
            "predicted centerline evidence or nearby unmatched tracks to justify a recall-preserving "
            "decoder/model-side instance recovery contract."
        ),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
