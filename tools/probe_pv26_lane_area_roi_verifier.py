from __future__ import annotations

import argparse
from itertools import islice
import json
from pathlib import Path
import site
import sys
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

REPO_ROOT = Path(__file__).resolve().parents[1]
repo_root = str(REPO_ROOT)
if repo_root not in sys.path:
    site.addsitedir(repo_root)

from model.engine import raw_batch_for_metrics
from model.engine._trainer_epochs import _merge_raw_batches
from model.engine.lane_segfirst_vectorizer import LaneSegFirstVectorizerConfig, vectorize_lane_segfirst_maps
from model.engine.metrics import _extract_gt_samples, _mean_point_distance, summarize_pv26_metrics
from model.engine.postprocess import _filter_lane_predictions, postprocess_pv26_batch
from tools.evaluate_pv26_lane60_checkpoint import _postprocess_override_config
from tools.probe_pv26_lane_feature_roi_repair import (
    DEFAULT_CHECKPOINT,
    LANE_MATCH_THRESHOLD,
    SOURCE_RUN,
    _as_channel,
    _build_scenario,
    _forward_predictions,
    _json_ready,
    _lane_features,
    _nearest_gt,
    _task_delta,
    _values_at_points,
)
from tools.probe_pv26_lane_flip_tta import _detach_to_cpu
from tools.probe_pv26_lane_instance_evidence import _resolve_device, _write_csv
from tools.pv26_train import cli as train_cli


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a no-GT line-ROI verifier over raw seg-first lane candidates "
            "that the default bbox/area filter drops, then append selected "
            "candidates and report actual lane-family TP/FP/FN."
        )
    )
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--source-run", default=str(SOURCE_RUN))
    parser.add_argument("--lane60-experiment", default="stopline_projection_comp_runtime")
    parser.add_argument("--preset", default="default")
    parser.add_argument("--verifier-train-batches", type=int, default=64)
    parser.add_argument("--max-val-batches", type=int, default=4)
    parser.add_argument("--validation-epoch", type=int, default=2)
    parser.add_argument("--train-batches", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dataset-root", default="")
    parser.add_argument(
        "--lane-flip-variant",
        choices=("baseline", "flip_centerline_avg", "flip_centerline_avg_lane_cross_comp050"),
        default="flip_centerline_avg_lane_cross_comp050",
    )
    parser.add_argument("--positive-distance-px", type=float, default=LANE_MATCH_THRESHOLD)
    parser.add_argument("--negative-distance-px", type=float, default=60.0)
    parser.add_argument("--baseline-duplicate-distance-px", type=float, default=LANE_MATCH_THRESHOLD)
    parser.add_argument("--candidate-duplicate-distance-px", type=float, default=LANE_MATCH_THRESHOLD)
    parser.add_argument("--quality-threshold", type=float, default=0.80)
    parser.add_argument("--max-appends-per-sample", type=int, default=2)
    parser.add_argument("--max-suppressions-per-sample", type=int, default=2)
    parser.add_argument(
        "--candidate-source",
        choices=("dropped_area", "retained"),
        default="dropped_area",
        help=(
            "dropped_area trains the historical rescue verifier over lanes removed "
            "by bbox/area filters. retained trains a suppress-only instance-quality "
            "verifier over lanes emitted by the retained runtime decoder."
        ),
    )
    parser.add_argument(
        "--candidate-integration-mode",
        choices=("append", "replace_nearest", "suppress_low_quality"),
        default="append",
        help=(
            "How verifier-scored candidates are integrated. "
            "append preserves the historical replay. replace_nearest keeps "
            "the lane count fixed by replacing the nearest retained lane. "
            "suppress_low_quality removes retained lanes whose keep probability "
            "falls below --quality-threshold."
        ),
    )
    parser.add_argument("--replace-nearest-max-distance-px", type=float, default=120.0)
    parser.add_argument(
        "--alignment-context-features",
        action="store_true",
        help=(
            "Append no-GT geometry context between each dropped candidate and "
            "the retained baseline lane predictions. This tests an "
            "instance-alignment FP-control signal, not a verifier threshold sweep."
        ),
    )
    parser.add_argument(
        "--side-contrast-features",
        action="store_true",
        help=(
            "Append no-GT center-vs-side-band dense evidence features for each "
            "dropped candidate. This tests whether raw candidates lie on a thin "
            "lane ridge rather than broad support/noise."
        ),
    )
    parser.add_argument(
        "--raw-image-line-features",
        action="store_true",
        help=(
            "Append no-GT raw-image line evidence sampled along each dropped "
            "candidate and its side bands. This tests image-space lane markings "
            "as a distinct signal from dense lane logits."
        ),
    )
    parser.add_argument(
        "--cross-task-conflict-features",
        action="store_true",
        help=(
            "Append no-GT stop-line/crosswalk dense-map evidence sampled along "
            "each lane candidate. This tests whether rejected lane candidates "
            "are actually explained by another lane-family task."
        ),
    )
    parser.add_argument("--verifier-epochs", type=int, default=40)
    parser.add_argument("--verifier-ensemble-size", type=int, default=1)
    parser.add_argument(
        "--ensemble-probability-mode",
        choices=("mean", "min", "mean_minus_std"),
        default="mean",
        help=(
            "How multiple independently seeded verifier probabilities are "
            "collapsed. This is an uncertainty/stability signal, not another "
            "single-model score threshold."
        ),
    )
    parser.add_argument("--verifier-batch-size", type=int, default=256)
    parser.add_argument("--verifier-lr", type=float, default=1.0e-3)
    parser.add_argument("--save-verifier-model", default="")
    parser.add_argument("--load-verifier-model", default="")
    parser.add_argument("--val-start-batch", type=int, default=0)
    parser.add_argument(
        "--eval-chunk-batches",
        type=int,
        default=128,
        help="Validation batches to keep in memory while replaying verifier metrics.",
    )
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--seed", type=int, default=26)
    parser.add_argument("--backbone-weights", default="")
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
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


class LaneAreaRoiVerifierNet(nn.Module):
    def __init__(self, input_dim: int, *, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(input_dim), int(hidden_dim)),
            nn.SiLU(inplace=True),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.SiLU(inplace=True),
            nn.Linear(int(hidden_dim), 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


def _lane_distance(a: dict[str, Any], b: dict[str, Any]) -> float:
    try:
        return float(_mean_point_distance(a.get("points_xy", []), b.get("points_xy", []), target_count=20))
    except Exception:
        return float("inf")


def _near_any_lane(candidate: dict[str, Any], lanes: list[dict[str, Any]], *, threshold_px: float) -> bool:
    return any(_lane_distance(candidate, lane) <= float(threshold_px) for lane in lanes)


def _nearest_lane_index(candidate: dict[str, Any], lanes: list[dict[str, Any]]) -> tuple[int, float]:
    if not lanes:
        return -1, float("inf")
    distances = [float(_lane_distance(candidate, lane)) for lane in lanes]
    if not distances:
        return -1, float("inf")
    index = int(np.argmin(np.asarray(distances, dtype=np.float32)))
    return index, float(distances[index])


def _polyline_array(lane: dict[str, Any]) -> np.ndarray:
    return np.asarray(lane.get("points_xy", []), dtype=np.float32).reshape(-1, 2)


def _polyline_length(lane: dict[str, Any]) -> float:
    points = _polyline_array(lane)
    if points.shape[0] < 2:
        return 0.0
    return float(np.linalg.norm(points[1:] - points[:-1], axis=1).sum())


def _lane_center(points: np.ndarray) -> np.ndarray:
    if points.shape[0] == 0:
        return np.zeros(2, dtype=np.float32)
    return points.mean(axis=0).astype(np.float32)


def _lane_axis(points: np.ndarray) -> np.ndarray | None:
    if points.shape[0] < 2:
        return None
    delta = points[-1] - points[0]
    norm = float(np.linalg.norm(delta))
    if norm <= 1.0e-6:
        return None
    axis = delta / norm
    if float(axis[1]) > 0.0:
        axis = -axis
    return axis.astype(np.float32)


def _angle_error_degrees(a: np.ndarray, b: np.ndarray) -> float:
    axis_a = _lane_axis(a)
    axis_b = _lane_axis(b)
    if axis_a is None or axis_b is None:
        return 180.0
    dot = float(np.clip(abs(float(np.dot(axis_a, axis_b))), 0.0, 1.0))
    return float(np.degrees(np.arccos(dot)))


def _y_overlap_fraction(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape[0] == 0 or b.shape[0] == 0:
        return 0.0
    a_min, a_max = float(a[:, 1].min()), float(a[:, 1].max())
    b_min, b_max = float(b[:, 1].min()), float(b[:, 1].max())
    overlap = max(0.0, min(a_max, b_max) - max(a_min, b_min))
    span = max(max(a_max, b_max) - min(a_min, b_min), 1.0e-6)
    return float(overlap / span)


def _alignment_context_features(candidate: dict[str, Any], baseline_lanes: list[dict[str, Any]]) -> np.ndarray:
    """No-GT context between a rescue candidate and retained lane instances."""

    candidate_points = _polyline_array(candidate)
    candidate_center = _lane_center(candidate_points)
    candidate_length = _polyline_length(candidate)
    if not baseline_lanes:
        return np.asarray(
            [
                0.0,  # has retained lane
                1.0,  # normalized nearest distance sentinel
                1.0,  # normalized center distance sentinel
                1.0,  # normalized abs dx sentinel
                1.0,  # normalized abs dy sentinel
                0.0,  # y overlap
                1.0,  # normalized angle error sentinel
                0.0,  # candidate / nearest length ratio
                0.0,  # nearest / candidate length ratio
                0.0,  # sample retained lane count norm
            ],
            dtype=np.float32,
        )

    best_lane = min(baseline_lanes, key=lambda lane: _lane_distance(candidate, lane))
    best_points = _polyline_array(best_lane)
    best_distance = _lane_distance(candidate, best_lane)
    best_center = _lane_center(best_points)
    center_delta = candidate_center - best_center
    best_length = _polyline_length(best_lane)
    length_den = max(candidate_length, best_length, 1.0e-6)
    return np.asarray(
        [
            1.0,
            min(float(best_distance), 240.0) / 240.0,
            min(float(np.linalg.norm(center_delta)), 240.0) / 240.0,
            min(abs(float(center_delta[0])), 240.0) / 240.0,
            min(abs(float(center_delta[1])), 240.0) / 240.0,
            _y_overlap_fraction(candidate_points, best_points),
            min(_angle_error_degrees(candidate_points, best_points), 90.0) / 90.0,
            min(candidate_length / length_den, 2.0) / 2.0,
            min(best_length / length_den, 2.0) / 2.0,
            min(float(len(baseline_lanes)), 16.0) / 16.0,
        ],
        dtype=np.float32,
    )


def _finite_feature_array(values: list[float]) -> np.ndarray:
    return np.asarray([0.0 if not np.isfinite(float(value)) else float(value) for value in values], dtype=np.float32)


def _summary_stats(values: np.ndarray) -> list[float]:
    finite = np.asarray(values, dtype=np.float32).reshape(-1)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    return [
        float(finite.mean()),
        float(finite.max(initial=0.0)),
        float(finite.std()),
        float((finite >= 0.25).mean()),
        float((finite >= 0.50).mean()),
        float((finite >= 0.75).mean()),
    ]


def _sample_tangent_and_normal(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sampled = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if sampled.shape[0] == 0:
        tangent = np.zeros((0, 2), dtype=np.float32)
        normal = np.zeros((0, 2), dtype=np.float32)
        return tangent, normal
    if sampled.shape[0] == 1:
        tangent = np.tile(np.asarray([[0.0, -1.0]], dtype=np.float32), (1, 1))
    else:
        previous_points = np.vstack([sampled[:1], sampled[:-1]])
        next_points = np.vstack([sampled[1:], sampled[-1:]])
        tangent = next_points - previous_points
        norms = np.linalg.norm(tangent, axis=1, keepdims=True)
        tangent = tangent / np.maximum(norms, 1.0e-6)
    normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1).astype(np.float32)
    return tangent.astype(np.float32), normal


def _lane_side_contrast_features(
    sampled_map_points: np.ndarray,
    *,
    maps: dict[str, torch.Tensor],
    offsets_px: tuple[float, ...] = (4.0, 8.0, 16.0),
) -> np.ndarray:
    """No-GT ridge-vs-side evidence around a raw lane candidate."""

    sampled = np.asarray(sampled_map_points, dtype=np.float32).reshape(-1, 2)
    if sampled.shape[0] == 0:
        return np.zeros(52, dtype=np.float32)
    centerline = _as_channel(maps["centerline_core"])
    support = _as_channel(maps["support"])
    tangent_axis = maps["tangent_axis"].detach().cpu().numpy().astype(np.float32)
    candidate_tangent, candidate_normal = _sample_tangent_and_normal(sampled)
    features: list[float] = []
    for dense_map in (centerline, support):
        center_values = _values_at_points(dense_map, sampled).astype(np.float32).reshape(-1)
        center_stats = _summary_stats(center_values)
        features.extend(center_stats)
        for offset_px in offsets_px:
            offset = candidate_normal * float(offset_px)
            left_values = _values_at_points(dense_map, sampled + offset).astype(np.float32).reshape(-1)
            right_values = _values_at_points(dense_map, sampled - offset).astype(np.float32).reshape(-1)
            side_values = np.concatenate([left_values, right_values], axis=0)
            side_stats = _summary_stats(side_values)
            features.extend(
                [
                    side_stats[0],
                    side_stats[1],
                    side_stats[2],
                    float(center_stats[0] - side_stats[0]),
                    float(center_stats[1] - side_stats[1]),
                    float(abs(float(left_values.mean()) - float(right_values.mean())))
                    if left_values.size and right_values.size
                    else 0.0,
                ]
            )
    tangent_values = _values_at_points(tangent_axis, sampled).astype(np.float32).reshape(-1, 2)
    if tangent_values.shape[0] == candidate_tangent.shape[0] and tangent_values.size:
        tangent_norm = np.linalg.norm(tangent_values, axis=1, keepdims=True)
        normalized_tangent = tangent_values / np.maximum(tangent_norm, 1.0e-6)
        alignment = np.abs(np.sum(normalized_tangent * candidate_tangent, axis=1))
        features.extend(_summary_stats(alignment)[:4])
    else:
        features.extend([0.0, 0.0, 0.0, 0.0])
    return _finite_feature_array(features)


def _lane_raw_image_line_features(
    sampled_map_points: np.ndarray,
    *,
    map_hw: tuple[int, int],
    image: torch.Tensor | np.ndarray | None,
    offsets_px: tuple[float, ...] = (3.0, 6.0, 12.0, 24.0),
) -> np.ndarray:
    """No-GT image-space line evidence around a raw lane candidate."""

    sampled = np.asarray(sampled_map_points, dtype=np.float32).reshape(-1, 2)
    feature_count = 20 + 12 * len(offsets_px)
    if sampled.shape[0] == 0 or image is None:
        return np.zeros(feature_count, dtype=np.float32)
    if isinstance(image, torch.Tensor):
        image_array = image.detach().cpu().numpy().astype(np.float32)
    else:
        image_array = np.asarray(image, dtype=np.float32)
    if image_array.ndim == 3 and int(image_array.shape[0]) in (1, 3):
        chw = image_array
    elif image_array.ndim == 3 and int(image_array.shape[-1]) in (1, 3):
        chw = np.moveaxis(image_array, -1, 0).astype(np.float32)
    else:
        return np.zeros(feature_count, dtype=np.float32)
    if int(chw.shape[0]) == 1:
        gray = chw[0]
    else:
        gray = 0.299 * chw[0] + 0.587 * chw[1] + 0.114 * chw[2]
    finite_gray = gray[np.isfinite(gray)]
    if finite_gray.size and float(finite_gray.max(initial=0.0)) > 2.0:
        gray = gray / 255.0
    gray = np.nan_to_num(gray.astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    image_h, image_w = int(gray.shape[0]), int(gray.shape[1])
    if image_h < 2 or image_w < 2:
        return np.zeros(feature_count, dtype=np.float32)
    map_h, map_w = int(map_hw[0]), int(map_hw[1])
    image_points = sampled.copy()
    image_points[:, 0] = image_points[:, 0] * max(float(image_w - 1), 1.0) / max(float(map_w - 1), 1.0)
    image_points[:, 1] = image_points[:, 1] * max(float(image_h - 1), 1.0) / max(float(map_h - 1), 1.0)
    candidate_tangent, candidate_normal = _sample_tangent_and_normal(image_points)

    grad_y, grad_x = np.gradient(gray)
    grad_x = grad_x.astype(np.float32)
    grad_y = grad_y.astype(np.float32)
    grad_mag = np.sqrt((grad_x * grad_x) + (grad_y * grad_y)).astype(np.float32)
    grad_xy = np.stack([grad_x, grad_y], axis=0)
    center_gray = _values_at_points(gray, image_points).astype(np.float32).reshape(-1)
    center_grad = _values_at_points(grad_mag, image_points).astype(np.float32).reshape(-1)
    center_grad_xy = _values_at_points(grad_xy, image_points).astype(np.float32).reshape(-1, 2)
    if center_grad_xy.shape[0] == candidate_normal.shape[0] and center_grad_xy.size:
        center_normal_edge = np.abs(np.sum(center_grad_xy * candidate_normal, axis=1))
        center_tangent_edge = np.abs(np.sum(center_grad_xy * candidate_tangent, axis=1))
    else:
        center_normal_edge = np.zeros_like(center_grad)
        center_tangent_edge = np.zeros_like(center_grad)

    center_gray_stats = _summary_stats(center_gray)
    center_grad_stats = _summary_stats(center_grad)
    center_normal_stats = _summary_stats(center_normal_edge)
    center_tangent_stats = _summary_stats(center_tangent_edge)
    features: list[float] = []
    features.extend(center_gray_stats)
    features.extend(center_grad_stats)
    features.extend(center_normal_stats[:4])
    features.extend(center_tangent_stats[:4])
    for offset_px in offsets_px:
        offset = candidate_normal * float(offset_px)
        left_points = image_points + offset
        right_points = image_points - offset
        left_gray = _values_at_points(gray, left_points).astype(np.float32).reshape(-1)
        right_gray = _values_at_points(gray, right_points).astype(np.float32).reshape(-1)
        side_gray = np.concatenate([left_gray, right_gray], axis=0)
        side_gray_stats = _summary_stats(side_gray)
        left_grad = _values_at_points(grad_mag, left_points).astype(np.float32).reshape(-1)
        right_grad = _values_at_points(grad_mag, right_points).astype(np.float32).reshape(-1)
        side_grad = np.concatenate([left_grad, right_grad], axis=0)
        side_grad_stats = _summary_stats(side_grad)
        features.extend(
            [
                side_gray_stats[0],
                side_gray_stats[1],
                side_gray_stats[2],
                float(center_gray_stats[0] - side_gray_stats[0]),
                float(center_gray_stats[1] - side_gray_stats[1]),
                float(abs(float(left_gray.mean()) - float(right_gray.mean())))
                if left_gray.size and right_gray.size
                else 0.0,
                side_grad_stats[0],
                side_grad_stats[1],
                side_grad_stats[2],
                float(center_grad_stats[0] - side_grad_stats[0]),
                float(center_grad_stats[1] - side_grad_stats[1]),
                float(abs(float(left_grad.mean()) - float(right_grad.mean())))
                if left_grad.size and right_grad.size
                else 0.0,
            ]
        )
    return _finite_feature_array(features)


CROSS_TASK_CONFLICT_MAP_KEYS = (
    "stop_line_mask_logits",
    "stop_line_center_logits",
    "stop_line_selector_map_logits",
    "stop_line_segment_seed_logits",
    "crosswalk_mask_logits",
    "crosswalk_boundary_logits",
    "crosswalk_center_logits",
)


def _probability_channel_from_prediction(value: Any) -> np.ndarray | None:
    if not isinstance(value, torch.Tensor):
        return None
    tensor = value.detach().float().cpu()
    if tensor.ndim == 3:
        tensor = tensor[0]
    elif tensor.ndim != 2:
        return None
    return torch.sigmoid(tensor).numpy().astype(np.float32)


def _lane_cross_task_conflict_features(
    sampled_map_points: np.ndarray,
    *,
    predictions: dict[str, Any],
    map_hw: tuple[int, int],
) -> np.ndarray:
    """No-GT evidence that a lane candidate overlaps another task's dense maps."""

    sampled = np.asarray(sampled_map_points, dtype=np.float32).reshape(-1, 2)
    features: list[float] = []
    source_h, source_w = int(map_hw[0]), int(map_hw[1])
    for key in CROSS_TASK_CONFLICT_MAP_KEYS:
        dense_map = _probability_channel_from_prediction(predictions.get(key))
        if dense_map is None or sampled.shape[0] == 0:
            features.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            continue
        target_h, target_w = int(dense_map.shape[0]), int(dense_map.shape[1])
        points = sampled.copy()
        points[:, 0] = points[:, 0] * max(float(target_w - 1), 1.0) / max(float(source_w - 1), 1.0)
        points[:, 1] = points[:, 1] * max(float(target_h - 1), 1.0) / max(float(source_h - 1), 1.0)
        values = _values_at_points(dense_map, points).astype(np.float32).reshape(-1)
        features.extend(_summary_stats(values))
    return _finite_feature_array(features)


def _baseline_matched_gt_indices(
    baseline_lanes: list[dict[str, Any]],
    gt_lanes: list[dict[str, Any]],
    *,
    threshold_px: float = LANE_MATCH_THRESHOLD,
) -> set[int]:
    if not baseline_lanes or not gt_lanes:
        return set()
    cost = np.zeros((len(baseline_lanes), len(gt_lanes)), dtype=np.float32)
    for pred_index, prediction in enumerate(baseline_lanes):
        for gt_index, gt_lane in enumerate(gt_lanes):
            cost[pred_index, gt_index] = float(_lane_distance(prediction, gt_lane))
    pred_indices, gt_indices = linear_sum_assignment(cost)
    matched: set[int] = set()
    for pred_index, gt_index in zip(pred_indices.tolist(), gt_indices.tolist()):
        if float(cost[pred_index, gt_index]) <= float(threshold_px):
            matched.add(int(gt_index))
    return matched


def _matched_lane_prediction_indices(
    baseline_lanes: list[dict[str, Any]],
    gt_lanes: list[dict[str, Any]],
    *,
    threshold_px: float = LANE_MATCH_THRESHOLD,
) -> dict[int, tuple[int, float]]:
    if not baseline_lanes or not gt_lanes:
        return {}
    cost = np.zeros((len(baseline_lanes), len(gt_lanes)), dtype=np.float32)
    for pred_index, prediction in enumerate(baseline_lanes):
        for gt_index, gt_lane in enumerate(gt_lanes):
            cost[pred_index, gt_index] = float(_lane_distance(prediction, gt_lane))
    pred_indices, gt_indices = linear_sum_assignment(cost)
    matched: dict[int, tuple[int, float]] = {}
    for pred_index, gt_index in zip(pred_indices.tolist(), gt_indices.tolist()):
        distance = float(cost[pred_index, gt_index])
        if distance <= float(threshold_px):
            matched[int(pred_index)] = (int(gt_index), distance)
    return matched


def _is_dropped_by_lane_filter(candidate: dict[str, Any], *, postprocess_config: Any) -> bool:
    kept = _filter_lane_predictions(
        [candidate],
        min_bbox_area_px=float(postprocess_config.lane_segfirst_min_bbox_area_px),
        max_bbox_aspect=float(postprocess_config.lane_segfirst_max_bbox_aspect),
    )
    return len(kept) == 0


def _raw_lane_candidates(
    *,
    maps: dict[str, torch.Tensor],
    meta: dict[str, Any],
    postprocess_config: Any,
) -> list[dict[str, Any]]:
    return vectorize_lane_segfirst_maps(
        maps,
        meta=meta,
        config=LaneSegFirstVectorizerConfig(
            track_mode=str(postprocess_config.lane_segfirst_track_mode),
            centerline_threshold=float(postprocess_config.lane_obj_threshold),
            min_polyline_length_px=float(postprocess_config.lane_segfirst_min_polyline_length_px),
            min_polyline_bottom_y_fraction=float(postprocess_config.lane_segfirst_min_polyline_bottom_y_fraction),
            semantic_vote_mode=str(postprocess_config.lane_segfirst_semantic_vote_mode),
            max_row_gap=int(postprocess_config.lane_segfirst_max_row_gap),
            max_link_dx=float(postprocess_config.lane_segfirst_max_link_dx),
            max_turn_degrees=float(postprocess_config.lane_segfirst_max_turn_degrees),
            seed_threshold=float(postprocess_config.lane_segfirst_seed_threshold),
            seed_trace_max_seeds=int(postprocess_config.lane_segfirst_seed_trace_max_seeds),
            center_offset_enabled=bool(postprocess_config.lane_segfirst_center_offset_enabled),
            center_offset_max_shift_px=float(postprocess_config.lane_segfirst_center_offset_max_shift_px),
            center_offset_min_support_score=float(postprocess_config.lane_segfirst_center_offset_min_support_score),
        ),
    )


def _candidate_label(
    *,
    candidate: dict[str, Any],
    gt_lanes: list[dict[str, Any]],
    baseline_matched_gt: set[int],
    positive_distance_px: float,
    negative_distance_px: float,
) -> tuple[bool, bool, int, float]:
    gt_index, distance = _nearest_gt(candidate, gt_lanes)
    positive = bool(
        gt_index >= 0
        and int(gt_index) not in baseline_matched_gt
        and float(distance) <= float(positive_distance_px)
    )
    negative = bool(
        gt_index < 0
        or int(gt_index) in baseline_matched_gt
        or float(distance) >= float(negative_distance_px)
    )
    return positive, negative, int(gt_index), float(distance)


def _collect_examples(
    *,
    loader: Any,
    evaluator: Any,
    postprocess_config: Any,
    args: argparse.Namespace,
    max_batches: int,
    training: bool,
    batch_index_offset: int = 0,
    progress_total: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    from model.engine.lane_segfirst_vectorizer import lane_segfirst_prediction_maps

    examples: list[dict[str, Any]] = []
    predictions_all: list[dict[str, Any]] = []
    raw_batches: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    global_sample_index = 0
    split = "train" if training else "val"
    candidate_source = str(getattr(args, "candidate_source", "dropped_area"))
    with torch.no_grad():
        for batch_index, batch in enumerate(islice(loader, max(0, int(max_batches))), start=1):
            display_batch_index = int(batch_index_offset) + int(batch_index)
            display_total = int(progress_total) if progress_total is not None else int(max_batches)
            if batch_index == 1 or display_batch_index % 20 == 0:
                print(
                    f"[lane_area_roi_verifier] collect {split} batch {display_batch_index}/{display_total}",
                    flush=True,
                )
            raw_batch = raw_batch_for_metrics(batch)
            if raw_batch is None:
                raise ValueError("lane area ROI verifier requires raw batches for metrics")
            batch_images = batch.get("image") if isinstance(batch, dict) else None
            encoded = evaluator.prepare_batch(batch)
            predictions = _forward_predictions(evaluator, encoded, lane_flip_variant=str(args.lane_flip_variant))
            meta = _detach_to_cpu(encoded["meta"])
            batch_predictions = postprocess_pv26_batch(predictions, meta, config=postprocess_config)
            batch_gt = _extract_gt_samples(raw_batch)
            predictions_all.extend(batch_predictions)
            raw_batches.append(raw_batch)
            for sample_batch_index, (sample_pred, sample_gt, sample_meta) in enumerate(
                zip(batch_predictions, batch_gt, meta)
            ):
                maps = lane_segfirst_prediction_maps(predictions, batch_index=sample_batch_index)
                sample_prediction_tensors = {
                    key: value[sample_batch_index]
                    for key, value in predictions.items()
                    if isinstance(value, torch.Tensor) and int(value.shape[0]) > sample_batch_index
                }
                gt_lanes = list(sample_gt.get("lanes", []))
                baseline_lanes = list(sample_pred.get("lanes", []))
                baseline_matched_gt = _baseline_matched_gt_indices(baseline_lanes, gt_lanes)
                if candidate_source == "retained":
                    matched_predictions = _matched_lane_prediction_indices(baseline_lanes, gt_lanes)
                    for pred_index, candidate in enumerate(baseline_lanes):
                        match = matched_predictions.get(int(pred_index))
                        if match is not None:
                            gt_index, distance = int(match[0]), float(match[1])
                            positive = True
                            negative = False
                        else:
                            gt_index, distance = _nearest_gt(candidate, gt_lanes)
                            positive = False
                            negative = True
                        features, sampled_map_points, map_hw = _lane_features(
                            candidate,
                            predictions=sample_prediction_tensors,
                            maps=maps,
                            meta=sample_meta,
                        )
                        if bool(args.alignment_context_features):
                            other_lanes = [
                                lane for lane_index, lane in enumerate(baseline_lanes) if lane_index != int(pred_index)
                            ]
                            features = np.concatenate(
                                [features, _alignment_context_features(candidate, other_lanes)]
                            ).astype(np.float32)
                        if bool(args.side_contrast_features):
                            features = np.concatenate(
                                [features, _lane_side_contrast_features(sampled_map_points, maps=maps)]
                            ).astype(np.float32)
                        if bool(args.raw_image_line_features):
                            sample_image = None
                            if isinstance(batch_images, torch.Tensor) and int(batch_images.shape[0]) > sample_batch_index:
                                sample_image = batch_images[sample_batch_index]
                            features = np.concatenate(
                                [
                                    features,
                                    _lane_raw_image_line_features(
                                        sampled_map_points,
                                        map_hw=map_hw,
                                        image=sample_image,
                                    ),
                                ]
                            ).astype(np.float32)
                        if bool(args.cross_task_conflict_features):
                            features = np.concatenate(
                                [
                                    features,
                                    _lane_cross_task_conflict_features(
                                        sampled_map_points,
                                        predictions=sample_prediction_tensors,
                                        map_hw=map_hw,
                                    ),
                                ]
                            ).astype(np.float32)
                        examples.append(
                            {
                                "features": features.astype(np.float32),
                                "positive": float(1.0 if positive else 0.0),
                                "negative": float(1.0 if negative else 0.0),
                                "nearest_gt_index": int(gt_index),
                                "nearest_gt_distance": float(distance),
                                "sample_index": int(global_sample_index),
                                "sample_batch_index": int(sample_batch_index),
                                "candidate_index": int(pred_index),
                                "candidate": dict(candidate),
                            }
                        )
                        rows.append(
                            {
                                "split": split,
                                "batch_index": int(display_batch_index),
                                "sample_index": int(global_sample_index),
                                "sample_batch_index": int(sample_batch_index),
                                "candidate_index": int(pred_index),
                                "nearest_gt_index": int(gt_index),
                                "nearest_gt_distance": float(distance),
                                "positive": int(positive),
                                "negative": int(negative),
                                "baseline_matched_gt_count": int(len(baseline_matched_gt)),
                            }
                        )
                    global_sample_index += 1
                    continue
                raw_candidates = _raw_lane_candidates(
                    maps=maps,
                    meta=sample_meta,
                    postprocess_config=postprocess_config,
                )
                sample_candidate_index = 0
                for candidate in raw_candidates:
                    if not _is_dropped_by_lane_filter(candidate, postprocess_config=postprocess_config):
                        continue
                    if str(getattr(args, "candidate_integration_mode", "append")) == "append" and _near_any_lane(
                        candidate,
                        baseline_lanes,
                        threshold_px=float(args.baseline_duplicate_distance_px),
                    ):
                        continue
                    positive, negative, gt_index, distance = _candidate_label(
                        candidate=candidate,
                        gt_lanes=gt_lanes,
                        baseline_matched_gt=baseline_matched_gt,
                        positive_distance_px=float(args.positive_distance_px),
                        negative_distance_px=float(args.negative_distance_px),
                    )
                    if training and not positive and not negative:
                        continue
                    features, sampled_map_points, map_hw = _lane_features(
                        candidate,
                        predictions=sample_prediction_tensors,
                        maps=maps,
                        meta=sample_meta,
                    )
                    if bool(args.alignment_context_features):
                        features = np.concatenate(
                            [features, _alignment_context_features(candidate, baseline_lanes)]
                        ).astype(np.float32)
                    if bool(args.side_contrast_features):
                        features = np.concatenate(
                            [features, _lane_side_contrast_features(sampled_map_points, maps=maps)]
                        ).astype(np.float32)
                    if bool(args.raw_image_line_features):
                        sample_image = None
                        if isinstance(batch_images, torch.Tensor) and int(batch_images.shape[0]) > sample_batch_index:
                            sample_image = batch_images[sample_batch_index]
                        features = np.concatenate(
                            [
                                features,
                                _lane_raw_image_line_features(
                                    sampled_map_points,
                                    map_hw=map_hw,
                                    image=sample_image,
                                ),
                            ]
                        ).astype(np.float32)
                    if bool(args.cross_task_conflict_features):
                        features = np.concatenate(
                            [
                                features,
                                _lane_cross_task_conflict_features(
                                    sampled_map_points,
                                    predictions=sample_prediction_tensors,
                                    map_hw=map_hw,
                                ),
                            ]
                        ).astype(np.float32)
                    example = {
                        "features": features.astype(np.float32),
                        "positive": float(1.0 if positive else 0.0),
                        "negative": float(1.0 if negative else 0.0),
                        "nearest_gt_index": int(gt_index),
                        "nearest_gt_distance": float(distance),
                        "sample_index": int(global_sample_index),
                        "sample_batch_index": int(sample_batch_index),
                        "candidate_index": int(sample_candidate_index),
                        "candidate": dict(candidate),
                    }
                    examples.append(example)
                    rows.append(
                        {
                            "split": split,
                            "batch_index": int(display_batch_index),
                            "sample_index": int(global_sample_index),
                            "sample_batch_index": int(sample_batch_index),
                            "candidate_index": int(sample_candidate_index),
                            "nearest_gt_index": int(gt_index),
                            "nearest_gt_distance": float(distance),
                            "positive": int(positive),
                            "negative": int(negative),
                            "baseline_matched_gt_count": int(len(baseline_matched_gt)),
                        }
                    )
                    sample_candidate_index += 1
                global_sample_index += 1
    return examples, predictions_all, raw_batches, rows


def _offset_sample_rows(rows: list[dict[str, Any]], *, sample_index_offset: int) -> list[dict[str, Any]]:
    if int(sample_index_offset) == 0:
        return rows
    output: list[dict[str, Any]] = []
    for row in rows:
        updated = dict(row)
        if "sample_index" in updated:
            updated["sample_index"] = int(updated["sample_index"]) + int(sample_index_offset)
        output.append(updated)
    return output


def _empty_task_count_payload() -> dict[str, dict[str, float]]:
    return {task: {"tp": 0.0, "fp": 0.0, "fn": 0.0} for task in ("lane", "stop_line", "crosswalk")}


def _accumulate_task_counts(target: dict[str, dict[str, float]], metrics: dict[str, Any]) -> None:
    for task in ("lane", "stop_line", "crosswalk"):
        payload = metrics.get(task, {}) if isinstance(metrics.get(task), dict) else {}
        target[task]["tp"] += float(payload.get("tp", 0.0))
        target[task]["fp"] += float(payload.get("fp", 0.0))
        target[task]["fn"] += float(payload.get("fn", 0.0))


def _counts_to_metric_payload(counts: dict[str, float]) -> dict[str, float]:
    tp = float(counts.get("tp", 0.0))
    fp = float(counts.get("fp", 0.0))
    fn = float(counts.get("fn", 0.0))
    precision = tp / (tp + fp) if (tp + fp) > 0.0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0.0 else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0.0 else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def _finalize_task_counts(counts: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    return {task: _counts_to_metric_payload(counts[task]) for task in ("lane", "stop_line", "crosswalk")}


def _train_verifier(
    examples: list[dict[str, Any]],
    *,
    args: argparse.Namespace,
    device: str,
    seed: int | None = None,
) -> tuple[LaneAreaRoiVerifierNet, dict[str, Any]]:
    if not examples:
        raise ValueError("no area-ROI verifier training examples collected")
    effective_seed = int(args.seed) if seed is None else int(seed)
    torch.manual_seed(effective_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(effective_seed)
    features = torch.tensor(np.stack([row["features"] for row in examples]), dtype=torch.float32)
    labels = torch.tensor([float(row["positive"]) for row in examples], dtype=torch.float32)
    mean = features.mean(dim=0)
    std = features.std(dim=0).clamp(min=1.0e-6)
    features = (features - mean) / std
    model = LaneAreaRoiVerifierNet(int(features.shape[1]), hidden_dim=int(args.hidden_dim)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.verifier_lr), weight_decay=1.0e-4)
    features = features.to(device)
    labels = labels.to(device)
    positive_count = int(labels.sum().item())
    negative_count = int(labels.numel() - positive_count)
    pos_weight = torch.tensor(
        [max(float(negative_count) / max(float(positive_count), 1.0), 1.0)],
        dtype=torch.float32,
        device=device,
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(effective_seed)
    batch_size = max(1, int(args.verifier_batch_size))
    history: list[dict[str, float]] = []
    for epoch in range(1, max(1, int(args.verifier_epochs)) + 1):
        order = torch.randperm(int(labels.numel()), generator=generator)
        losses: list[float] = []
        for start in range(0, int(order.numel()), batch_size):
            index = order[start : start + batch_size].to(device)
            logits = model(features[index])
            loss = F.binary_cross_entropy_with_logits(logits, labels[index], pos_weight=pos_weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        if epoch == 1 or epoch == int(args.verifier_epochs) or epoch % 10 == 0:
            history.append({"epoch": float(epoch), "loss": float(np.mean(losses) if losses else 0.0)})
    model.register_buffer("feature_mean", mean.to(device), persistent=True)
    model.register_buffer("feature_std", std.to(device), persistent=True)
    summary = {
        "example_count": int(labels.numel()),
        "positive_count": positive_count,
        "negative_count": negative_count,
        "input_dim": int(features.shape[1]),
        "history": history,
        "seed": effective_seed,
    }
    return model, summary


def _train_verifier_or_ensemble(
    examples: list[dict[str, Any]],
    *,
    args: argparse.Namespace,
    device: str,
) -> tuple[LaneAreaRoiVerifierNet | list[LaneAreaRoiVerifierNet], dict[str, Any]]:
    ensemble_size = max(1, int(getattr(args, "verifier_ensemble_size", 1)))
    if ensemble_size == 1:
        return _train_verifier(examples, args=args, device=device)
    models: list[LaneAreaRoiVerifierNet] = []
    members: list[dict[str, Any]] = []
    for member_index in range(ensemble_size):
        model, member_summary = _train_verifier(
            examples,
            args=args,
            device=device,
            seed=int(args.seed) + (9973 * int(member_index)),
        )
        model.eval()
        models.append(model)
        member_summary["member_index"] = int(member_index)
        members.append(member_summary)
    first = members[0]
    summary = {
        "ensemble_size": int(ensemble_size),
        "ensemble_probability_mode": str(getattr(args, "ensemble_probability_mode", "mean")),
        "example_count": int(first.get("example_count", 0)),
        "positive_count": int(first.get("positive_count", 0)),
        "negative_count": int(first.get("negative_count", 0)),
        "input_dim": int(first.get("input_dim", 0)),
        "members": members,
    }
    return models, summary


def _save_verifier_model(
    path: str,
    *,
    model: LaneAreaRoiVerifierNet | list[LaneAreaRoiVerifierNet],
    train_summary: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    if not path:
        return
    if isinstance(model, list):
        raise ValueError("--save-verifier-model currently supports a single verifier model, not an ensemble")
    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": int(train_summary["input_dim"]),
            "hidden_dim": int(args.hidden_dim),
            "train_summary": train_summary,
        },
        output_path,
    )


def _load_verifier_model(path: str, *, args: argparse.Namespace, device: str) -> tuple[LaneAreaRoiVerifierNet, dict[str, Any]]:
    input_path = Path(path).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    payload = torch.load(input_path, map_location=device)
    train_summary = dict(payload.get("train_summary", {}))
    input_dim = int(payload.get("input_dim", train_summary.get("input_dim", 0)))
    if input_dim <= 0:
        raise ValueError(f"verifier checkpoint is missing input_dim: {input_path}")
    hidden_dim = int(payload.get("hidden_dim", int(args.hidden_dim)))
    model = LaneAreaRoiVerifierNet(input_dim, hidden_dim=hidden_dim).to(device)
    state_dict = dict(payload["state_dict"])
    feature_mean = state_dict.pop("feature_mean", None)
    feature_std = state_dict.pop("feature_std", None)
    model.load_state_dict(state_dict)
    if feature_mean is None or feature_std is None:
        raise ValueError(f"verifier checkpoint is missing normalization buffers: {input_path}")
    model.register_buffer("feature_mean", feature_mean.to(device), persistent=True)
    model.register_buffer("feature_std", feature_std.to(device), persistent=True)
    model.eval()
    if not train_summary:
        train_summary = {"input_dim": input_dim, "loaded_from": str(input_path)}
    else:
        train_summary["loaded_from"] = str(input_path)
    return model, train_summary


def _apply_verifier(
    *,
    examples: list[dict[str, Any]],
    predictions_all: list[dict[str, Any]],
    model: LaneAreaRoiVerifierNet | list[LaneAreaRoiVerifierNet],
    args: argparse.Namespace,
    device: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    repaired = [dict(sample, lanes=[dict(lane) for lane in sample.get("lanes", [])]) for sample in predictions_all]
    if not examples:
        return repaired, []
    features = torch.tensor(np.stack([row["features"] for row in examples]), dtype=torch.float32, device=device)
    models = model if isinstance(model, list) else [model]
    member_probabilities: list[np.ndarray] = []
    with torch.no_grad():
        for member in models:
            normalized = (features - member.feature_mean) / member.feature_std
            member_probabilities.append(torch.sigmoid(member(normalized)).detach().cpu().numpy().astype(np.float32))
    stacked_probabilities = np.stack(member_probabilities, axis=0)
    if len(models) == 1:
        probabilities = stacked_probabilities[0]
    else:
        mode = str(getattr(args, "ensemble_probability_mode", "mean"))
        if mode == "mean":
            probabilities = stacked_probabilities.mean(axis=0)
        elif mode == "min":
            probabilities = stacked_probabilities.min(axis=0)
        elif mode == "mean_minus_std":
            probabilities = stacked_probabilities.mean(axis=0) - stacked_probabilities.std(axis=0)
        else:
            raise ValueError(f"unsupported ensemble probability mode: {mode}")
        probabilities = np.clip(probabilities, 0.0, 1.0).astype(np.float32)
    by_sample: dict[int, list[tuple[float, int]]] = {}
    for index, (example, probability) in enumerate(zip(examples, probabilities.tolist())):
        by_sample.setdefault(int(example["sample_index"]), []).append((float(probability), int(index)))
    selected_records: dict[int, dict[str, Any]] = {}
    integration_mode = str(getattr(args, "candidate_integration_mode", "append"))
    if integration_mode == "suppress_low_quality":
        max_suppressions = max(0, int(getattr(args, "max_suppressions_per_sample", 0)))
        for sample_index, candidates in by_sample.items():
            candidates.sort(key=lambda item: item[0])
            if sample_index < 0 or sample_index >= len(repaired):
                continue
            lanes = repaired[sample_index].setdefault("lanes", [])
            suppressed_lane_indices: set[int] = set()
            for probability, index in candidates:
                if len(suppressed_lane_indices) >= max_suppressions:
                    break
                if probability >= float(args.quality_threshold):
                    continue
                lane_index = int(examples[index]["candidate_index"])
                if lane_index < 0 or lane_index >= len(lanes):
                    continue
                if lane_index in suppressed_lane_indices:
                    continue
                suppressed_lane_indices.add(lane_index)
                selected_records[int(index)] = {
                    "action": "suppress_low_quality",
                    "replaced_lane_index": int(lane_index),
                    "replaced_lane_distance": 0.0,
                }
            if suppressed_lane_indices:
                repaired[sample_index]["lanes"] = [
                    lane for lane_index, lane in enumerate(lanes) if lane_index not in suppressed_lane_indices
                ]
    else:
        for sample_index, candidates in by_sample.items():
            candidates.sort(reverse=True)
            if sample_index < 0 or sample_index >= len(repaired):
                continue
            lanes = repaired[sample_index].setdefault("lanes", [])
            selected_for_sample = 0
            replaced_lane_indices: set[int] = set()
            for probability, index in candidates:
                if selected_for_sample >= int(args.max_appends_per_sample):
                    break
                if probability < float(args.quality_threshold):
                    continue
                candidate = dict(examples[index]["candidate"])
                candidate["area_roi_verifier_score"] = float(probability)
                candidate["area_roi_verifier_integration"] = integration_mode
                if integration_mode == "append":
                    if _near_any_lane(
                        candidate,
                        lanes,
                        threshold_px=float(args.candidate_duplicate_distance_px),
                    ):
                        continue
                    lanes.append(candidate)
                    selected_records[int(index)] = {
                        "action": "append",
                        "replaced_lane_index": -1,
                        "replaced_lane_distance": float("nan"),
                    }
                    selected_for_sample += 1
                    continue

                if integration_mode == "replace_nearest":
                    replace_index, replace_distance = _nearest_lane_index(candidate, lanes)
                    if replace_index < 0:
                        continue
                    if replace_index in replaced_lane_indices:
                        continue
                    if float(replace_distance) > float(args.replace_nearest_max_distance_px):
                        continue
                    other_lanes = [lane for lane_index, lane in enumerate(lanes) if lane_index != int(replace_index)]
                    if _near_any_lane(
                        candidate,
                        other_lanes,
                        threshold_px=float(args.candidate_duplicate_distance_px),
                    ):
                        continue
                    candidate["area_roi_replaced_lane_index"] = int(replace_index)
                    candidate["area_roi_replaced_lane_distance_px"] = float(replace_distance)
                    lanes[int(replace_index)] = candidate
                    replaced_lane_indices.add(int(replace_index))
                    selected_records[int(index)] = {
                        "action": "replace_nearest",
                        "replaced_lane_index": int(replace_index),
                        "replaced_lane_distance": float(replace_distance),
                    }
                    selected_for_sample += 1
                    continue

                raise ValueError(f"unsupported candidate integration mode: {integration_mode}")
    rows: list[dict[str, Any]] = []
    for index, example in enumerate(examples):
        sample_index = int(example["sample_index"])
        selected_record = selected_records.get(int(index), {})
        selected = bool(selected_record)
        probability = float(probabilities[index])
        rows.append(
            {
                "sample_index": sample_index,
                "candidate_index": int(example["candidate_index"]),
                "nearest_gt_index": int(example["nearest_gt_index"]),
                "nearest_gt_distance": float(example["nearest_gt_distance"]),
                "positive": int(float(example["positive"]) > 0.5),
                "negative": int(float(example["negative"]) > 0.5),
                "verifier_probability": probability,
                "selected": int(bool(selected)),
                "integration_action": str(selected_record.get("action", "")),
                "replaced_lane_index": int(selected_record.get("replaced_lane_index", -1)),
                "replaced_lane_distance": float(selected_record.get("replaced_lane_distance", float("nan"))),
            }
        )
    return repaired, rows


def _evaluate_verifier_streaming(
    *,
    val_loader: Any,
    evaluator: Any,
    postprocess_config: Any,
    model: LaneAreaRoiVerifierNet,
    args: argparse.Namespace,
    device: str,
) -> dict[str, Any]:
    max_val_batches = max(0, int(args.max_val_batches))
    val_start_batch = max(0, int(args.val_start_batch))
    chunk_batches = max(1, int(args.eval_chunk_batches))
    baseline_counts = _empty_task_count_payload()
    repaired_counts = _empty_task_count_payload()
    val_rows: list[dict[str, Any]] = []
    verifier_rows: list[dict[str, Any]] = []
    val_candidate_count = 0
    selected_candidate_count = 0
    selected_oracle_positive_count = 0
    evaluated_batches = 0
    sample_offset = 0
    val_iter = iter(val_loader)
    for _ in range(val_start_batch):
        try:
            next(val_iter)
        except StopIteration:
            break

    while evaluated_batches < max_val_batches:
        chunk_size = min(chunk_batches, max_val_batches - evaluated_batches)
        chunk_examples, baseline_predictions, raw_batches, chunk_val_rows = _collect_examples(
            loader=val_iter,
            evaluator=evaluator,
            postprocess_config=postprocess_config,
            args=args,
            max_batches=chunk_size,
            training=False,
            batch_index_offset=val_start_batch + evaluated_batches,
            progress_total=val_start_batch + max_val_batches,
        )
        if not raw_batches:
            break
        repaired_predictions, chunk_verifier_rows = _apply_verifier(
            examples=chunk_examples,
            predictions_all=baseline_predictions,
            model=model,
            args=args,
            device=device,
        )
        merged_raw = _merge_raw_batches(raw_batches)
        _accumulate_task_counts(baseline_counts, summarize_pv26_metrics(baseline_predictions, merged_raw))
        _accumulate_task_counts(repaired_counts, summarize_pv26_metrics(repaired_predictions, merged_raw))

        selected_rows = [row for row in chunk_verifier_rows if int(row.get("selected", 0))]
        val_candidate_count += int(len(chunk_examples))
        selected_candidate_count += int(len(selected_rows))
        selected_oracle_positive_count += int(sum(int(row.get("positive", 0)) for row in selected_rows))
        val_rows.extend(_offset_sample_rows(chunk_val_rows, sample_index_offset=sample_offset))
        verifier_rows.extend(_offset_sample_rows(chunk_verifier_rows, sample_index_offset=sample_offset))
        batch_count = int(len(raw_batches))
        evaluated_batches += batch_count
        sample_offset += int(len(baseline_predictions))
        if batch_count < chunk_size:
            break

    return {
        "baseline_tasks": _finalize_task_counts(baseline_counts),
        "repaired_tasks": _finalize_task_counts(repaired_counts),
        "val_rows": val_rows,
        "verifier_rows": verifier_rows,
        "val_candidate_count": int(val_candidate_count),
        "selected_candidate_count": int(selected_candidate_count),
        "selected_oracle_positive_count": int(selected_oracle_positive_count),
        "evaluated_val_batches": int(evaluated_batches),
    }


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    scenario_args = argparse.Namespace(**vars(args))
    scenario_args.max_val_batches = max(1, int(args.val_start_batch) + int(args.max_val_batches))
    scenario, scenario_path, options, phase, train_config = _build_scenario(scenario_args)
    phase_index = int(tuple(options["selected_phase_indices"])[0])
    train_cli._configure_torch_multiprocessing()
    dataset = train_cli.PV26CanonicalDataset(
        train_cli._existing_dataset_roots(scenario),
        train_augmentation=False,
        progress_callback=lambda message: print(f"[lane_area_roi_verifier] {message}", flush=True),
    )
    train_loader, val_loader = train_cli._build_phase_train_loaders(dataset, train_config=train_config, phase=phase)
    if train_loader is None or val_loader is None:
        raise ValueError("lane area ROI verifier requires train and validation loaders")
    from tools.evaluate_pv26_lane60_checkpoint import _advance_validation_sampler

    _advance_validation_sampler(val_loader, validation_epoch=int(args.validation_epoch))
    trainer = train_cli._build_phase_trainer(phase, train_config)
    trainer.load_model_weights(Path(args.checkpoint).expanduser().resolve(), map_location=train_config.device)
    postprocess_config = _postprocess_override_config(args, trainer) or train_cli._build_postprocess_config(train_config)
    evaluator = trainer.build_evaluator()
    evaluator.adapter.raw_model.eval()
    evaluator.heads.eval()
    device = _resolve_device(str(args.device), str(train_config.device))

    if str(args.load_verifier_model):
        model, train_summary = _load_verifier_model(str(args.load_verifier_model), args=args, device=device)
        train_rows: list[dict[str, Any]] = []
    else:
        train_examples, _, _, train_rows = _collect_examples(
            loader=train_loader,
            evaluator=evaluator,
            postprocess_config=postprocess_config,
            args=args,
            max_batches=int(args.verifier_train_batches),
            training=True,
        )
        model, train_summary = _train_verifier_or_ensemble(train_examples, args=args, device=device)
        _save_verifier_model(str(args.save_verifier_model), model=model, train_summary=train_summary, args=args)
    eval_payload = _evaluate_verifier_streaming(
        val_loader=val_loader,
        evaluator=evaluator,
        postprocess_config=postprocess_config,
        model=model,
        args=args,
        device=device,
    )
    baseline_tasks = eval_payload["baseline_tasks"]
    repaired_tasks = eval_payload["repaired_tasks"]
    deltas = {task: _task_delta(repaired_tasks[task], baseline_tasks[task]) for task in baseline_tasks}
    summary = {
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "source_run": str(Path(args.source_run).expanduser().resolve()),
        "scenario_path": str(scenario_path),
        "lane60_experiment": str(args.lane60_experiment),
        "phase_index": int(phase_index),
        "lane_flip_variant": str(args.lane_flip_variant),
        "verifier_train_batches": int(args.verifier_train_batches),
        "verifier_ensemble_size": int(args.verifier_ensemble_size),
        "ensemble_probability_mode": str(args.ensemble_probability_mode),
        "candidate_source": str(args.candidate_source),
        "max_suppressions_per_sample": int(args.max_suppressions_per_sample),
        "max_val_batches": int(args.max_val_batches),
        "val_start_batch": int(args.val_start_batch),
        "eval_chunk_batches": int(args.eval_chunk_batches),
        "evaluated_val_batches": int(eval_payload["evaluated_val_batches"]),
        "validation_epoch": int(args.validation_epoch),
        "positive_distance_px": float(args.positive_distance_px),
        "negative_distance_px": float(args.negative_distance_px),
        "quality_threshold": float(args.quality_threshold),
        "candidate_integration_mode": str(args.candidate_integration_mode),
        "replace_nearest_max_distance_px": float(args.replace_nearest_max_distance_px),
        "max_appends_per_sample": int(args.max_appends_per_sample),
        "alignment_context_features": bool(args.alignment_context_features),
        "side_contrast_features": bool(args.side_contrast_features),
        "raw_image_line_features": bool(args.raw_image_line_features),
        "cross_task_conflict_features": bool(args.cross_task_conflict_features),
        "train_summary": train_summary,
        "val_candidate_count": int(eval_payload["val_candidate_count"]),
        "selected_candidate_count": int(eval_payload["selected_candidate_count"]),
        "selected_oracle_positive_count": int(eval_payload["selected_oracle_positive_count"]),
        "baseline": baseline_tasks,
        "repaired": repaired_tasks,
        "delta": deltas,
        "interpretation": (
            "No-GT runtime replay of a learned line-ROI verifier. "
            "candidate_source=dropped_area scores raw seg-first lane candidates dropped by "
            "the default lane bbox/area filter; candidate_source=retained scores emitted "
            "runtime lanes for suppress-only instance-quality tests. GT is used only for "
            "train labels and final audit metrics, not candidate selection."
        ),
    }
    return {
        "summary": summary,
        "train_rows": train_rows,
        "val_rows": eval_payload["val_rows"],
        "verifier_rows": eval_payload["verifier_rows"],
    }


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = run_probe(args)
    (output_dir / "summary.json").write_text(json.dumps(_json_ready(payload["summary"]), indent=2), encoding="utf-8")
    _write_csv(output_dir / "train_candidates.csv", payload["train_rows"])
    _write_csv(output_dir / "val_candidates.csv", payload["val_rows"])
    _write_csv(output_dir / "verifier_replay_rows.csv", payload["verifier_rows"])
    print(json.dumps(_json_ready({"summary": payload["summary"]}), indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
