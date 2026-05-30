from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, is_dataclass, replace as dataclasses_replace
import json
import math
from pathlib import Path
import random
import site
import sys
from typing import Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
repo_root = str(REPO_ROOT)
if repo_root not in sys.path:
    site.addsitedir(repo_root)

from model.engine._trainer_epochs import _merge_raw_batches
from model.engine.batch import augment_lane_family_metrics, raw_batch_for_metrics
from model.engine.metrics import (
    STOP_LINE_POINT_COUNT,
    _extract_gt_samples,
    _lane_family_metrics,
    _mean_point_distance,
    summarize_pv26_metrics,
)
from model.engine.postprocess import postprocess_pv26_batch
from model.data.transform import transform_from_meta, transform_points
from tools.evaluate_pv26_lane60_checkpoint import _advance_validation_sampler, _postprocess_override_config
from tools.probe_pv26_lane_flip_tta import (
    DEFAULT_CHECKPOINT,
    SOURCE_RUN,
    _apply_lane_task_mask_competition,
    _dedupe_stop_lines_by_distance,
    _detach_to_cpu,
    _json_ready,
    _merge_stop_line_outputs,
    _resolve_device,
    _stop_line_distance,
    _unflip_lane_dense_outputs,
    _build_eval_contract,
)
from tools.pv26_train import cli as train_cli


DEFAULT_STOP_LINE_CHECKPOINT = (
    REPO_ROOT
    / "runs"
    / "pv26_exhaustive_od_lane_train"
    / "lane60_stopline_priority_positive_sampler_from_lane60_lane_head_transplant_original_stop_pca_20260512_default_20260529_101745"
    / "phase_4"
    / "checkpoints"
    / "best.pt"
)
ROUTER_MODES = ("primary", "specialist", "endpoint_fusion", "union_dedupe", "agreement", "empty")
DEFAULT_ROUTER_RASTER_SIZE = (64, 96)
LINE_PROFILE_MAP_KEYS = (
    "stop_line_mask_logits",
    "stop_line_center_logits",
    "stop_line_selector_map_logits",
    "stop_line_midpoint_logits",
)


class SourceRouterMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class SourceRouterRasterCNN(nn.Module):
    def __init__(self, input_channels: int, output_dim: int, *, base_channels: int = 16) -> None:
        super().__init__()
        channels = max(4, int(base_channels))
        self.net = nn.Sequential(
            nn.Conv2d(int(input_channels), channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(channels, channels * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(channels * 2, channels * 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels * 4, int(output_dim)),
        )

    def forward(self, rasters: torch.Tensor) -> torch.Tensor:
        return self.net(rasters)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a small no-GT runtime router that chooses per-sample stop-line "
            "source outputs from the retained primary checkpoint, the stop-line "
            "specialist checkpoint, their deduped union/agreement, or an empty "
            "FP-suppression source."
        )
    )
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--stop-line-checkpoint", default=str(DEFAULT_STOP_LINE_CHECKPOINT))
    parser.add_argument("--source-run", default=str(SOURCE_RUN))
    parser.add_argument("--lane60-experiment", default="stopline_projection_comp_runtime")
    parser.add_argument("--stop-line-lane60-experiment", default="stopline_projection_comp_runtime")
    parser.add_argument("--preset", default="default")
    parser.add_argument("--router-train-batches", type=int, default=64)
    parser.add_argument("--max-val-batches", type=int, default=4)
    parser.add_argument("--validation-epoch", type=int, default=2)
    parser.add_argument("--train-batches", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--backbone-weights", default="")
    parser.add_argument("--dataset-root", default="")
    parser.add_argument("--router-hidden-dim", type=int, default=32)
    parser.add_argument("--router-epochs", type=int, default=80)
    parser.add_argument("--router-lr", type=float, default=3.0e-3)
    parser.add_argument("--router-weight-decay", type=float, default=1.0e-4)
    parser.add_argument(
        "--feature-mode",
        choices=("output_stats", "dense_aligned", "line_profile", "lane_topology", "raster_cnn"),
        default="output_stats",
        help=(
            "output_stats reproduces the closed scalar source-router premise. "
            "dense_aligned appends no-GT line-aligned dense-map/raw-image quality features. "
            "line_profile appends fixed along-axis source line profiles without spatial CNN pooling. "
            "lane_topology appends no-GT source-line geometry relative to retained lane predictions. "
            "raster_cnn trains a tiny router over raw-image, dense-map, and source prediction rasters."
        ),
    )
    parser.add_argument("--dense-feature-samples", type=int, default=25)
    parser.add_argument("--dense-feature-side-offset", type=float, default=3.0)
    parser.add_argument("--raster-height", type=int, default=DEFAULT_ROUTER_RASTER_SIZE[0])
    parser.add_argument("--raster-width", type=int, default=DEFAULT_ROUTER_RASTER_SIZE[1])
    parser.add_argument("--raster-base-channels", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260530)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--stop-line-mask-binary-threshold", type=float, default=None)
    parser.add_argument("--stop-line-min-instance-score", type=float, default=None)
    parser.add_argument("--stop-line-presence-threshold", type=float, default=None)
    parser.add_argument(
        "--stop-line-component-gate-source",
        choices=("center", "selector", "max"),
        default=None,
    )
    parser.add_argument("--crosswalk-polygon-mode", choices=("rect", "hull"), default=None)
    parser.add_argument("--save-router-model", default="")
    return parser.parse_args()


def _line_score(line: dict[str, Any]) -> float:
    candidates = (
        line.get("score"),
        line.get("center_score"),
        line.get("instance_score"),
        line.get("orientation_score"),
    )
    values = [float(value) for value in candidates if isinstance(value, (int, float))]
    return max(values) if values else 0.0


def _line_length(line: dict[str, Any]) -> float:
    value = line.get("length")
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    points = np.asarray(line.get("points_xy", []), dtype=np.float32).reshape(-1, 2)
    if points.shape[0] < 2 or not bool(np.isfinite(points).all()):
        return 0.0
    deltas = points[1:] - points[:-1]
    return float(np.linalg.norm(deltas, axis=1).sum())


def _line_angle(line: dict[str, Any]) -> float:
    points = np.asarray(line.get("points_xy", []), dtype=np.float32).reshape(-1, 2)
    if points.shape[0] < 2 or not bool(np.isfinite(points).all()):
        return 0.0
    vector = points[-1] - points[0]
    if float(np.linalg.norm(vector)) <= 1.0e-6:
        return 0.0
    return float(math.atan2(float(vector[1]), float(vector[0])))


def _line_stats(lines: list[dict[str, Any]]) -> list[float]:
    if not lines:
        return [0.0] * 13
    scores = np.asarray([_line_score(line) for line in lines], dtype=np.float32)
    lengths = np.asarray([_line_length(line) for line in lines], dtype=np.float32)
    fragments = np.asarray(
        [
            float(line.get("fragment_count", 0.0))
            if isinstance(line.get("fragment_count"), (int, float))
            else 0.0
            for line in lines
        ],
        dtype=np.float32,
    )
    angles = np.asarray([_line_angle(line) for line in lines], dtype=np.float32)
    points = [
        np.asarray(line.get("points_xy", []), dtype=np.float32).reshape(-1, 2)
        for line in lines
        if len(line.get("points_xy", [])) > 0
    ]
    centers = np.asarray([point.mean(axis=0) for point in points if point.size], dtype=np.float32)
    if centers.size == 0:
        centers = np.zeros((1, 2), dtype=np.float32)
    return [
        float(len(lines)),
        float(scores.max(initial=0.0)),
        float(scores.mean()),
        float(scores.sum()),
        float(lengths.max(initial=0.0)),
        float(lengths.mean()),
        float(lengths.sum()),
        float(fragments.max(initial=0.0)),
        float(fragments.mean()),
        float(np.cos(angles).mean()),
        float(np.sin(angles).mean()),
        float(centers[:, 0].mean()),
        float(centers[:, 1].mean()),
    ]


def _pair_stats(primary_lines: list[dict[str, Any]], specialist_lines: list[dict[str, Any]]) -> list[float]:
    if not primary_lines or not specialist_lines:
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    distances = np.asarray(
        [
            _stop_line_distance(primary, specialist)
            for primary in primary_lines
            for specialist in specialist_lines
        ],
        dtype=np.float32,
    )
    finite = distances[np.isfinite(distances)]
    if finite.size == 0:
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    primary_best = max((_line_score(line) for line in primary_lines), default=0.0)
    specialist_best = max((_line_score(line) for line in specialist_lines), default=0.0)
    primary_len = max((_line_length(line) for line in primary_lines), default=0.0)
    specialist_len = max((_line_length(line) for line in specialist_lines), default=0.0)
    return [
        float(finite.min()),
        float(finite.mean()),
        float((finite <= 40.0).sum()),
        float((finite <= 64.0).sum()),
        float(specialist_best - primary_best),
        float(specialist_len - primary_len),
    ]


def _endpoint_array(line: dict[str, Any]) -> np.ndarray | None:
    points = np.asarray(line.get("points_xy", []), dtype=np.float32).reshape(-1, 2)
    if points.shape[0] < 2 or not bool(np.isfinite(points).all()):
        return None
    return np.stack([points[0], points[-1]], axis=0).astype(np.float32)


def _aligned_endpoint_arrays(
    primary_line: dict[str, Any],
    specialist_line: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray] | None:
    primary_points = _endpoint_array(primary_line)
    specialist_points = _endpoint_array(specialist_line)
    if primary_points is None or specialist_points is None:
        return None
    same_cost = float(np.linalg.norm(primary_points[0] - specialist_points[0])) + float(
        np.linalg.norm(primary_points[1] - specialist_points[1])
    )
    flipped_cost = float(np.linalg.norm(primary_points[0] - specialist_points[1])) + float(
        np.linalg.norm(primary_points[1] - specialist_points[0])
    )
    if flipped_cost < same_cost:
        specialist_points = specialist_points[::-1].copy()
    return primary_points, specialist_points


def _fuse_endpoint_pair(primary_line: dict[str, Any], specialist_line: dict[str, Any]) -> dict[str, Any] | None:
    aligned = _aligned_endpoint_arrays(primary_line, specialist_line)
    if aligned is None:
        return None
    primary_points, specialist_points = aligned
    fused_points = 0.5 * (primary_points + specialist_points)
    if not bool(np.isfinite(fused_points).all()):
        return None
    fused = dict(primary_line)
    fused["points_xy"] = fused_points.astype(float).tolist()
    fused["score"] = max(_line_score(primary_line), _line_score(specialist_line))
    fused["source"] = "endpoint_fusion"
    fused["primary_source_score"] = _line_score(primary_line)
    fused["specialist_source_score"] = _line_score(specialist_line)
    fused["source_pair_distance"] = _endpoint_pair_distance(primary_line, specialist_line)
    return fused


def _endpoint_pair_distance(primary_line: dict[str, Any], specialist_line: dict[str, Any]) -> float:
    aligned = _aligned_endpoint_arrays(primary_line, specialist_line)
    if aligned is None:
        return float("inf")
    primary_points, specialist_points = aligned
    return float(np.linalg.norm(primary_points - specialist_points, axis=1).mean())


def _endpoint_fusion_lines(
    primary_lines: list[dict[str, Any]],
    specialist_lines: list[dict[str, Any]],
    *,
    max_pair_distance: float = 96.0,
) -> list[dict[str, Any]]:
    """Fuse agreeing primary/specialist endpoints into a new geometry candidate.

    This is a fixed no-GT geometry source, not a threshold sweep: each primary
    candidate greedily pairs with the nearest unused specialist candidate within
    a broad same-line distance and averages aligned endpoints.
    """

    unused_specialists = set(range(len(specialist_lines)))
    fused: list[dict[str, Any]] = []
    for primary_line in sorted(primary_lines, key=_line_score, reverse=True):
        best_index = -1
        best_distance = float("inf")
        for specialist_index in list(unused_specialists):
            distance = _endpoint_pair_distance(primary_line, specialist_lines[specialist_index])
            if distance < best_distance:
                best_distance = float(distance)
                best_index = int(specialist_index)
        if best_index < 0 or best_distance > float(max_pair_distance):
            continue
        unused_specialists.remove(best_index)
        fused_line = _fuse_endpoint_pair(primary_line, specialist_lines[best_index])
        if fused_line is not None:
            fused.append(fused_line)
    return _dedupe_stop_lines_by_distance(fused)


def _points_to_segment_distance(points: np.ndarray, start: np.ndarray, end: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if points.ndim != 2 or points.shape[0] == 0 or points.shape[1] != 2:
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    vector = end.astype(np.float32) - start.astype(np.float32)
    denom = float(np.dot(vector, vector))
    if denom <= 1.0e-6 or not math.isfinite(denom):
        distances = np.linalg.norm(points.astype(np.float32) - start.astype(np.float32)[None, :], axis=1)
        return distances.astype(np.float32), np.zeros_like(distances, dtype=np.float32)
    rel = points.astype(np.float32) - start.astype(np.float32)[None, :]
    t = np.clip((rel @ vector) / denom, 0.0, 1.0).astype(np.float32)
    closest = start.astype(np.float32)[None, :] + t[:, None] * vector[None, :]
    distances = np.linalg.norm(points.astype(np.float32) - closest, axis=1)
    return distances.astype(np.float32), t.astype(np.float32)


def _lane_points(lane: dict[str, Any]) -> np.ndarray | None:
    points = np.asarray(lane.get("points_xy", []), dtype=np.float32).reshape(-1, 2)
    if points.shape[0] < 2 or not bool(np.isfinite(points).all()):
        return None
    dense_segments: list[np.ndarray] = []
    for start, end in zip(points[:-1], points[1:]):
        delta = end - start
        steps = max(2, int(np.ceil(float(np.linalg.norm(delta)) / 16.0)) + 1)
        weights = np.linspace(0.0, 1.0, num=steps, dtype=np.float32)
        segment = start[None, :] * (1.0 - weights[:, None]) + end[None, :] * weights[:, None]
        if dense_segments:
            segment = segment[1:]
        dense_segments.append(segment.astype(np.float32))
    dense = np.concatenate(dense_segments, axis=0) if dense_segments else points
    if dense.shape[0] < 2 or not bool(np.isfinite(dense).all()):
        return points
    return dense


def _safe_distance(value: float, cap: float = 512.0) -> float:
    if not math.isfinite(float(value)):
        return float(cap)
    return float(np.clip(float(value), 0.0, float(cap)))


def _single_line_lane_topology_features(line: dict[str, Any], lanes: list[dict[str, Any]]) -> list[float]:
    raw_points = np.asarray(line.get("points_xy", []), dtype=np.float32).reshape(-1, 2)
    lane_points = [_lane_points(lane) for lane in lanes]
    valid_lanes = [points for points in lane_points if points is not None]
    # Keep a fixed finite contract even for absent source lines; source line count is
    # still supplied by the enclosing source feature.
    missing = [
        0.0,
        float(len(valid_lanes)),
        512.0,
        512.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        512.0,
        512.0,
        0.0,
        512.0,
    ]
    if raw_points.shape[0] < 2 or not bool(np.isfinite(raw_points).all()) or not valid_lanes:
        return missing

    start = raw_points[0].astype(np.float32)
    end = raw_points[-1].astype(np.float32)
    line_vector = end - start
    line_norm = float(np.linalg.norm(line_vector))
    if line_norm <= 1.0e-6 or not math.isfinite(line_norm):
        return missing
    axis = line_vector / line_norm
    midpoint = 0.5 * (start + end)

    closest_distances: list[float] = []
    closest_t: list[float] = []
    abs_dots: list[float] = []
    start_distances: list[float] = []
    end_distances: list[float] = []
    mid_distances: list[float] = []

    for points in valid_lanes:
        distances, t_values = _points_to_segment_distance(points, start, end)
        if distances.size == 0:
            continue
        closest_index = int(np.argmin(distances))
        closest_distances.append(float(distances[closest_index]))
        closest_t.append(float(t_values[closest_index]))
        prev_index = max(0, closest_index - 1)
        next_index = min(points.shape[0] - 1, closest_index + 1)
        tangent = points[next_index] - points[prev_index]
        tangent_norm = float(np.linalg.norm(tangent))
        if tangent_norm <= 1.0e-6 or not math.isfinite(tangent_norm):
            abs_dots.append(1.0)
        else:
            abs_dots.append(float(abs(np.dot(axis, tangent / tangent_norm))))
        start_distances.append(float(np.linalg.norm(points - start[None, :], axis=1).min()))
        end_distances.append(float(np.linalg.norm(points - end[None, :], axis=1).min()))
        mid_distances.append(float(np.linalg.norm(points - midpoint[None, :], axis=1).min()))

    if not closest_distances:
        return missing

    distances_array = np.asarray(closest_distances, dtype=np.float32)
    t_array = np.asarray(closest_t, dtype=np.float32)
    dot_array = np.asarray(abs_dots, dtype=np.float32)
    start_array = np.asarray(start_distances, dtype=np.float32)
    end_array = np.asarray(end_distances, dtype=np.float32)
    mid_array = np.asarray(mid_distances, dtype=np.float32)

    close_mask = distances_array <= 64.0
    close_t = t_array[close_mask]
    close_dot = dot_array[close_mask]
    sorted_distances = np.sort(distances_array)
    top3 = sorted_distances[: min(3, sorted_distances.size)]
    t_min = float(close_t.min()) if close_t.size else 0.0
    t_max = float(close_t.max()) if close_t.size else 0.0
    t_span = max(0.0, t_max - t_min) if close_t.size else 0.0
    t_center_error = float(abs(float(close_t.mean()) - 0.5)) if close_t.size else 0.0
    crossness = 1.0 - close_dot if close_dot.size else np.zeros((0,), dtype=np.float32)
    start_min = float(start_array.min(initial=512.0))
    end_min = float(end_array.min(initial=512.0))
    return [
        1.0,
        float(len(valid_lanes)),
        _safe_distance(float(distances_array.min(initial=512.0))),
        _safe_distance(float(top3.mean()) if top3.size else 512.0),
        float((distances_array <= 16.0).sum()),
        float((distances_array <= 32.0).sum()),
        float((distances_array <= 48.0).sum()),
        float((distances_array <= 64.0).sum()),
        float((distances_array <= 96.0).sum()),
        float(t_min),
        float(t_max),
        float(t_span),
        float(t_center_error),
        float(crossness.mean()) if crossness.size else 0.0,
        float(crossness.max(initial=0.0)) if crossness.size else 0.0,
        float(dot_array.min(initial=1.0)),
        _safe_distance(start_min),
        _safe_distance(end_min),
        _safe_distance(abs(start_min - end_min)),
        _safe_distance(float(mid_array.min(initial=512.0))),
    ]


def _source_lane_topology_features(lines: list[dict[str, Any]], lanes: list[dict[str, Any]]) -> list[float]:
    single_dim = 20
    if not lines:
        empty_line = _single_line_lane_topology_features({}, lanes)
        return [0.0, float(len(lanes)), *empty_line, *empty_line, *empty_line]
    per_line = np.asarray(
        [_single_line_lane_topology_features(line, lanes) for line in lines],
        dtype=np.float32,
    )
    if per_line.ndim != 2 or per_line.shape[1] != single_dim or not bool(np.isfinite(per_line).all()):
        empty_line = _single_line_lane_topology_features({}, lanes)
        return [float(len(lines)), float(len(lanes)), *empty_line, *empty_line, *empty_line]
    best_index = int(np.argmin(per_line[:, 2]))
    mean_values = per_line.mean(axis=0)
    max_values = per_line.max(axis=0)
    best_values = per_line[best_index]
    output = [
        float(len(lines)),
        float(len(lanes)),
        *[float(value) for value in mean_values.tolist()],
        *[float(value) for value in max_values.tolist()],
        *[float(value) for value in best_values.tolist()],
    ]
    return [0.0 if not math.isfinite(float(value)) else float(value) for value in output]


def _lane_topology_features_for_sample(
    primary_sample: dict[str, Any],
    specialist_sample: dict[str, Any],
) -> list[float]:
    primary_lines = [dict(line) for line in primary_sample.get("stop_lines", [])]
    specialist_lines = [dict(line) for line in specialist_sample.get("stop_lines", [])]
    union_lines = _dedupe_stop_lines_by_distance([*primary_lines, *specialist_lines])
    agreement_lines = _source_prediction(primary_sample, specialist_sample, "agreement").get("stop_lines", [])
    lanes = [dict(lane) for lane in primary_sample.get("lanes", [])]
    source_groups = (primary_lines, specialist_lines, union_lines, [dict(line) for line in agreement_lines])
    features: list[float] = []
    for lines in source_groups:
        features.extend(_source_lane_topology_features(lines, lanes))
    return [0.0 if not math.isfinite(float(value)) else float(value) for value in features]


def _source_prediction(
    primary_sample: dict[str, Any],
    specialist_sample: dict[str, Any],
    mode: str,
) -> dict[str, Any]:
    primary_lines = [dict(line) for line in primary_sample.get("stop_lines", [])]
    specialist_lines = [dict(line) for line in specialist_sample.get("stop_lines", [])]
    if mode == "primary":
        chosen = primary_lines
    elif mode == "specialist":
        chosen = specialist_lines
    elif mode == "endpoint_fusion":
        chosen = _endpoint_fusion_lines(primary_lines, specialist_lines)
    elif mode == "union_dedupe":
        chosen = _dedupe_stop_lines_by_distance([*primary_lines, *specialist_lines])
    elif mode == "agreement":
        chosen = []
        for line in primary_lines:
            if any(_stop_line_distance(line, other) <= 64.0 for other in specialist_lines):
                chosen.append(dict(line))
        for line in specialist_lines:
            if any(_stop_line_distance(line, other) <= 64.0 for other in primary_lines):
                chosen.append(dict(line))
        chosen = _dedupe_stop_lines_by_distance(chosen)
    elif mode == "empty":
        chosen = []
    else:
        raise ValueError(f"unknown router mode: {mode}")
    sample = dict(primary_sample)
    sample["stop_lines"] = chosen
    return sample


def _sample_stopline_metrics(prediction: dict[str, Any], gt_sample: dict[str, Any]) -> dict[str, Any]:
    return _lane_family_metrics(
        [prediction],
        [gt_sample],
        field_name="stop_lines",
        target_count=STOP_LINE_POINT_COUNT,
        match_threshold=40.0,
    )


def _label_for_sample(
    primary_sample: dict[str, Any],
    specialist_sample: dict[str, Any],
    gt_sample: dict[str, Any],
) -> int:
    gt_count = len(gt_sample.get("stop_lines", []))
    ranked: list[tuple[float, float, float, int]] = []
    for index, mode in enumerate(ROUTER_MODES):
        prediction = _source_prediction(primary_sample, specialist_sample, mode)
        metrics = _sample_stopline_metrics(prediction, gt_sample)
        tp = float(metrics.get("tp", 0.0))
        fp = float(metrics.get("fp", 0.0))
        f1 = float(metrics.get("f1", 0.0))
        if gt_count <= 0:
            ranked.append((-fp, 0.0, 0.0, index))
        else:
            # Prefer actual matches, but allow empty only when no source can match.
            ranked.append((f1, tp, -fp, index))
    ranked.sort(reverse=True)
    return int(ranked[0][3])


def _features_for_sample(primary_sample: dict[str, Any], specialist_sample: dict[str, Any]) -> list[float]:
    primary_lines = [dict(line) for line in primary_sample.get("stop_lines", [])]
    specialist_lines = [dict(line) for line in specialist_sample.get("stop_lines", [])]
    union_lines = _dedupe_stop_lines_by_distance([*primary_lines, *specialist_lines])
    features = [
        *_line_stats(primary_lines),
        *_line_stats(specialist_lines),
        *_line_stats(union_lines),
        *_pair_stats(primary_lines, specialist_lines),
    ]
    return [0.0 if not math.isfinite(float(value)) else float(value) for value in features]


def _as_2d_prob(outputs: dict[str, torch.Tensor], key: str, sample_index: int) -> np.ndarray | None:
    tensor = outputs.get(key)
    if not isinstance(tensor, torch.Tensor):
        return None
    if tensor.ndim < 3 or int(sample_index) >= int(tensor.shape[0]):
        return None
    sample = tensor[int(sample_index)].detach().cpu()
    sample = sample.sigmoid()
    while sample.ndim > 2 and int(sample.shape[0]) == 1:
        sample = sample.squeeze(0)
    if sample.ndim != 2:
        return None
    array = sample.numpy().astype(np.float32)
    if not bool(np.isfinite(array).all()):
        return None
    return array


def _as_gray_image(image: torch.Tensor | None, sample_index: int) -> np.ndarray | None:
    if not isinstance(image, torch.Tensor):
        return None
    if image.ndim != 4 or int(sample_index) >= int(image.shape[0]):
        return None
    sample = image[int(sample_index)].detach().cpu().float()
    if sample.ndim != 3 or int(sample.shape[0]) < 1:
        return None
    if int(sample.shape[0]) >= 3:
        gray = 0.299 * sample[0] + 0.587 * sample[1] + 0.114 * sample[2]
    else:
        gray = sample[0]
    array = gray.numpy().astype(np.float32)
    if not bool(np.isfinite(array).all()):
        return None
    return array


def _resize_array(array: np.ndarray | None, *, size: tuple[int, int]) -> np.ndarray:
    height, width = int(size[0]), int(size[1])
    if array is None or array.size == 0:
        return np.zeros((height, width), dtype=np.float32)
    tensor = torch.as_tensor(array, dtype=torch.float32).reshape(1, 1, int(array.shape[0]), int(array.shape[1]))
    resized = F.interpolate(tensor, size=(height, width), mode="bilinear", align_corners=False)
    output = resized[0, 0].detach().cpu().numpy().astype(np.float32)
    if not bool(np.isfinite(output).all()):
        return np.zeros((height, width), dtype=np.float32)
    return output


def _draw_stopline_raster(
    lines: list[dict[str, Any]],
    meta: dict[str, Any],
    *,
    size: tuple[int, int],
    thickness: int = 1,
) -> np.ndarray:
    height, width = int(size[0]), int(size[1])
    raster = np.zeros((height, width), dtype=np.float32)
    transform = transform_from_meta(meta)
    network_h, network_w = int(transform.network_hw[0]), int(transform.network_hw[1])
    radius = max(0, int(thickness))
    for line in lines:
        raw_points = np.asarray(line.get("points_xy", []), dtype=np.float32).reshape(-1, 2)
        if raw_points.shape[0] < 2 or not bool(np.isfinite(raw_points).all()):
            continue
        network_points = np.asarray(transform_points(raw_points.tolist(), transform), dtype=np.float32)
        for start, end in zip(network_points[:-1], network_points[1:]):
            delta = end - start
            steps = max(2, int(np.ceil(float(np.linalg.norm(delta)) / 2.0)))
            weights = np.linspace(0.0, 1.0, num=steps, dtype=np.float32)
            points = start[None, :] * (1.0 - weights[:, None]) + end[None, :] * weights[:, None]
            x = np.rint(points[:, 0] * float(width - 1) / max(float(network_w - 1), 1.0)).astype(np.int64)
            y = np.rint(points[:, 1] * float(height - 1) / max(float(network_h - 1), 1.0)).astype(np.int64)
            x = np.clip(x, 0, width - 1)
            y = np.clip(y, 0, height - 1)
            for dy in range(-radius, radius + 1):
                yy = np.clip(y + dy, 0, height - 1)
                for dx in range(-radius, radius + 1):
                    xx = np.clip(x + dx, 0, width - 1)
                    raster[yy, xx] = 1.0
    return raster


def _source_raster_for_sample(
    primary_sample: dict[str, Any],
    specialist_sample: dict[str, Any],
    *,
    primary_outputs: dict[str, torch.Tensor],
    specialist_outputs: dict[str, torch.Tensor],
    image: torch.Tensor | None,
    sample_index: int,
    meta: dict[str, Any],
    size: tuple[int, int],
) -> np.ndarray:
    primary_lines = [dict(line) for line in primary_sample.get("stop_lines", [])]
    specialist_lines = [dict(line) for line in specialist_sample.get("stop_lines", [])]
    union_lines = _dedupe_stop_lines_by_distance([*primary_lines, *specialist_lines])
    channels = [
        _resize_array(_as_gray_image(image, sample_index), size=size),
        _draw_stopline_raster(primary_lines, meta, size=size, thickness=1),
        _draw_stopline_raster(specialist_lines, meta, size=size, thickness=1),
        _draw_stopline_raster(union_lines, meta, size=size, thickness=1),
        _resize_array(_as_2d_prob(primary_outputs, "stop_line_mask_logits", sample_index), size=size),
        _resize_array(_as_2d_prob(specialist_outputs, "stop_line_mask_logits", sample_index), size=size),
        _resize_array(_as_2d_prob(primary_outputs, "stop_line_center_logits", sample_index), size=size),
        _resize_array(_as_2d_prob(specialist_outputs, "stop_line_center_logits", sample_index), size=size),
        _resize_array(_as_2d_prob(primary_outputs, "stop_line_selector_map_logits", sample_index), size=size),
        _resize_array(_as_2d_prob(specialist_outputs, "stop_line_selector_map_logits", sample_index), size=size),
    ]
    raster = np.stack(channels, axis=0).astype(np.float32)
    if not bool(np.isfinite(raster).all()):
        return np.zeros((len(channels), int(size[0]), int(size[1])), dtype=np.float32)
    return raster


def _bilinear_values(map_array: np.ndarray | None, xy: np.ndarray) -> np.ndarray:
    if map_array is None or xy.size == 0:
        return np.zeros((0,), dtype=np.float32)
    height, width = int(map_array.shape[0]), int(map_array.shape[1])
    x = np.clip(xy[:, 0].astype(np.float32), 0.0, float(width - 1))
    y = np.clip(xy[:, 1].astype(np.float32), 0.0, float(height - 1))
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = np.clip(x0 + 1, 0, width - 1)
    y1 = np.clip(y0 + 1, 0, height - 1)
    wx = x - x0.astype(np.float32)
    wy = y - y0.astype(np.float32)
    top = (1.0 - wx) * map_array[y0, x0] + wx * map_array[y0, x1]
    bottom = (1.0 - wx) * map_array[y1, x0] + wx * map_array[y1, x1]
    return ((1.0 - wy) * top + wy * bottom).astype(np.float32)


def _value_stats(values: np.ndarray) -> list[float]:
    finite = values[np.isfinite(values)] if values.size else values
    if finite.size == 0:
        return [0.0, 0.0, 0.0]
    return [float(finite.mean()), float(finite.max(initial=0.0)), float(finite.std())]


def _line_dense_points(
    line: dict[str, Any],
    meta: dict[str, Any],
    *,
    output_hw: tuple[int, int],
    sample_count: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    raw_points = np.asarray(line.get("points_xy", []), dtype=np.float32).reshape(-1, 2)
    if raw_points.shape[0] < 2 or not bool(np.isfinite(raw_points).all()):
        return None
    transform = transform_from_meta(meta)
    network_points = np.asarray(transform_points(raw_points.tolist(), transform), dtype=np.float32)
    start = network_points[0]
    end = network_points[-1]
    vector = end - start
    norm = float(np.linalg.norm(vector))
    if norm <= 1.0e-6 or not math.isfinite(norm):
        return None
    count = max(3, int(sample_count))
    weights = np.linspace(0.0, 1.0, num=count, dtype=np.float32)
    points = start[None, :] * (1.0 - weights[:, None]) + end[None, :] * weights[:, None]
    output_h, output_w = int(output_hw[0]), int(output_hw[1])
    network_h, network_w = int(transform.network_hw[0]), int(transform.network_hw[1])
    points[:, 0] *= float(output_w - 1) / max(float(network_w - 1), 1.0)
    points[:, 1] *= float(output_h - 1) / max(float(network_h - 1), 1.0)
    dense_vector = points[-1] - points[0]
    dense_norm = float(np.linalg.norm(dense_vector))
    if dense_norm <= 1.0e-6 or not math.isfinite(dense_norm):
        return None
    normal = np.asarray([-dense_vector[1], dense_vector[0]], dtype=np.float32) / dense_norm
    return points.astype(np.float32), normal.astype(np.float32)


def _single_line_dense_features(
    line: dict[str, Any],
    *,
    maps: list[np.ndarray | None],
    image_gray: np.ndarray | None,
    meta: dict[str, Any],
    sample_count: int,
    side_offset: float,
) -> list[float]:
    reference = next((array for array in maps if isinstance(array, np.ndarray)), None)
    if reference is None:
        reference = image_gray
    if reference is None:
        return [0.0] * 35
    line_points = _line_dense_points(
        line,
        meta,
        output_hw=(int(reference.shape[0]), int(reference.shape[1])),
        sample_count=int(sample_count),
    )
    if line_points is None:
        return [0.0] * 35
    points, normal = line_points
    side_delta = normal[None, :] * float(side_offset)
    side_points = np.concatenate([points + side_delta, points - side_delta], axis=0)
    center_index = points.shape[0] // 2
    edge_indices = np.unique(np.asarray([0, 1, points.shape[0] - 2, points.shape[0] - 1], dtype=np.int64))
    features: list[float] = []
    for array in maps:
        line_values = _bilinear_values(array, points)
        side_values = _bilinear_values(array, side_points)
        edge_values = line_values[edge_indices] if line_values.size else np.zeros((0,), dtype=np.float32)
        center_value = float(line_values[center_index]) if line_values.size else 0.0
        line_stats = _value_stats(line_values)
        side_mean = float(side_values.mean()) if side_values.size else 0.0
        features.extend(
            [
                *line_stats,
                side_mean,
                float(line_stats[0] - side_mean),
                float(edge_values.mean()) if edge_values.size else 0.0,
                center_value,
            ]
        )
    image_values = np.zeros((0,), dtype=np.float32)
    side_image_values = np.zeros((0,), dtype=np.float32)
    image_center_index = 0
    image_edge_indices = np.zeros((0,), dtype=np.int64)
    if image_gray is not None:
        image_line_points = _line_dense_points(
            line,
            meta,
            output_hw=(int(image_gray.shape[0]), int(image_gray.shape[1])),
            sample_count=int(sample_count),
        )
        if image_line_points is not None:
            image_points, image_normal = image_line_points
            image_delta = image_normal[None, :] * float(side_offset)
            image_side_points = np.concatenate([image_points + image_delta, image_points - image_delta], axis=0)
            image_center_index = image_points.shape[0] // 2
            image_edge_indices = np.unique(
                np.asarray([0, 1, image_points.shape[0] - 2, image_points.shape[0] - 1], dtype=np.int64)
            )
            image_values = _bilinear_values(image_gray, image_points)
            side_image_values = _bilinear_values(image_gray, image_side_points)
    image_stats = _value_stats(image_values)
    side_image_mean = float(side_image_values.mean()) if side_image_values.size else 0.0
    features.extend(
        [
            *image_stats,
            side_image_mean,
            float(image_stats[0] - side_image_mean),
            float(image_values[image_edge_indices].mean()) if image_values.size else 0.0,
            float(image_values[image_center_index]) if image_values.size else 0.0,
        ]
    )
    return [0.0 if not math.isfinite(float(value)) else float(value) for value in features]


def _source_dense_features(
    lines: list[dict[str, Any]],
    *,
    outputs: dict[str, torch.Tensor],
    image: torch.Tensor | None,
    sample_index: int,
    meta: dict[str, Any],
    sample_count: int,
    side_offset: float,
) -> list[float]:
    maps = [
        _as_2d_prob(outputs, "stop_line_mask_logits", sample_index),
        _as_2d_prob(outputs, "stop_line_center_logits", sample_index),
        _as_2d_prob(outputs, "stop_line_selector_map_logits", sample_index),
        _as_2d_prob(outputs, "stop_line_midpoint_logits", sample_index),
    ]
    image_gray = _as_gray_image(image, sample_index)
    if not lines:
        return [0.0] * 71
    per_line = np.asarray(
        [
            _single_line_dense_features(
                line,
                maps=maps,
                image_gray=image_gray,
                meta=meta,
                sample_count=int(sample_count),
                side_offset=float(side_offset),
            )
            for line in lines
        ],
        dtype=np.float32,
    )
    if per_line.ndim != 2 or per_line.shape[0] == 0:
        return [0.0] * 71
    mean_values = per_line.mean(axis=0)
    max_values = per_line.max(axis=0)
    return [
        float(len(lines)),
        *[float(value) for value in mean_values.tolist()],
        *[float(value) for value in max_values.tolist()],
    ]


def _dense_aligned_features_for_sample(
    primary_sample: dict[str, Any],
    specialist_sample: dict[str, Any],
    *,
    primary_outputs: dict[str, torch.Tensor],
    specialist_outputs: dict[str, torch.Tensor],
    image: torch.Tensor | None,
    sample_index: int,
    meta: dict[str, Any],
    sample_count: int,
    side_offset: float,
) -> list[float]:
    primary_lines = [dict(line) for line in primary_sample.get("stop_lines", [])]
    specialist_lines = [dict(line) for line in specialist_sample.get("stop_lines", [])]
    primary_on_primary = _source_dense_features(
        primary_lines,
        outputs=primary_outputs,
        image=image,
        sample_index=int(sample_index),
        meta=meta,
        sample_count=int(sample_count),
        side_offset=float(side_offset),
    )
    primary_on_specialist = _source_dense_features(
        primary_lines,
        outputs=specialist_outputs,
        image=image,
        sample_index=int(sample_index),
        meta=meta,
        sample_count=int(sample_count),
        side_offset=float(side_offset),
    )
    specialist_on_primary = _source_dense_features(
        specialist_lines,
        outputs=primary_outputs,
        image=image,
        sample_index=int(sample_index),
        meta=meta,
        sample_count=int(sample_count),
        side_offset=float(side_offset),
    )
    specialist_on_specialist = _source_dense_features(
        specialist_lines,
        outputs=specialist_outputs,
        image=image,
        sample_index=int(sample_index),
        meta=meta,
        sample_count=int(sample_count),
        side_offset=float(side_offset),
    )
    dense_values = [
        *primary_on_primary,
        *primary_on_specialist,
        *specialist_on_primary,
        *specialist_on_specialist,
    ]
    return [0.0 if not math.isfinite(float(value)) else float(value) for value in dense_values]


def _fixed_profile_values(array: np.ndarray | None, points: np.ndarray, count: int) -> np.ndarray:
    if array is None or points.size == 0:
        return np.zeros((int(count),), dtype=np.float32)
    values = _bilinear_values(array, points)
    if values.size != int(count):
        return np.zeros((int(count),), dtype=np.float32)
    return values.astype(np.float32)


def _single_line_profile_features(
    line: dict[str, Any],
    *,
    outputs: dict[str, torch.Tensor],
    image: torch.Tensor | None,
    sample_index: int,
    meta: dict[str, Any],
    sample_count: int,
    side_offset: float,
) -> list[float]:
    count = max(3, int(sample_count))
    angle = _line_angle(line)
    features: list[float] = [
        _line_score(line),
        _line_length(line),
        math.cos(angle),
        math.sin(angle),
    ]
    maps = [_as_2d_prob(outputs, key, sample_index) for key in LINE_PROFILE_MAP_KEYS]
    for array in maps:
        if array is None:
            features.extend([0.0] * (count * 2))
            continue
        line_points = _line_dense_points(
            line,
            meta,
            output_hw=(int(array.shape[0]), int(array.shape[1])),
            sample_count=count,
        )
        if line_points is None:
            features.extend([0.0] * (count * 2))
            continue
        points, normal = line_points
        side_delta = normal[None, :] * float(side_offset)
        center_values = _fixed_profile_values(array, points, count)
        left_values = _fixed_profile_values(array, points + side_delta, count)
        right_values = _fixed_profile_values(array, points - side_delta, count)
        side_values = 0.5 * (left_values + right_values)
        features.extend(float(value) for value in center_values.tolist())
        features.extend(float(value) for value in (center_values - side_values).tolist())

    gray = _as_gray_image(image, sample_index)
    if gray is None:
        features.extend([0.0] * (count * 2))
    else:
        image_points = _line_dense_points(
            line,
            meta,
            output_hw=(int(gray.shape[0]), int(gray.shape[1])),
            sample_count=count,
        )
        if image_points is None:
            features.extend([0.0] * (count * 2))
        else:
            points, normal = image_points
            side_delta = normal[None, :] * float(side_offset)
            center_values = _fixed_profile_values(gray, points, count)
            left_values = _fixed_profile_values(gray, points + side_delta, count)
            right_values = _fixed_profile_values(gray, points - side_delta, count)
            side_values = 0.5 * (left_values + right_values)
            features.extend(float(value) for value in center_values.tolist())
            features.extend(float(value) for value in (center_values - side_values).tolist())
    return [0.0 if not math.isfinite(float(value)) else float(value) for value in features]


def _line_profile_features_for_lines(
    lines: list[dict[str, Any]],
    *,
    outputs: dict[str, torch.Tensor],
    image: torch.Tensor | None,
    sample_index: int,
    meta: dict[str, Any],
    sample_count: int,
    side_offset: float,
    max_lines: int = 1,
) -> list[float]:
    count = max(3, int(sample_count))
    single_dim = 4 + (len(LINE_PROFILE_MAP_KEYS) + 1) * 2 * count
    selected = sorted(lines, key=_line_score, reverse=True)[: max(0, int(max_lines))]
    features: list[float] = []
    for line in selected:
        values = _single_line_profile_features(
            line,
            outputs=outputs,
            image=image,
            sample_index=int(sample_index),
            meta=meta,
            sample_count=count,
            side_offset=float(side_offset),
        )
        if len(values) != single_dim:
            values = [0.0] * single_dim
        features.extend(values)
    missing = max(0, int(max_lines) - len(selected))
    if missing:
        features.extend([0.0] * (missing * single_dim))
    return features


def _line_profile_features_for_sample(
    primary_sample: dict[str, Any],
    specialist_sample: dict[str, Any],
    *,
    primary_outputs: dict[str, torch.Tensor],
    specialist_outputs: dict[str, torch.Tensor],
    image: torch.Tensor | None,
    sample_index: int,
    meta: dict[str, Any],
    sample_count: int,
    side_offset: float,
) -> list[float]:
    primary_lines = [dict(line) for line in primary_sample.get("stop_lines", [])]
    specialist_lines = [dict(line) for line in specialist_sample.get("stop_lines", [])]
    union_lines = _dedupe_stop_lines_by_distance([*primary_lines, *specialist_lines])
    agreement_lines = _source_prediction(primary_sample, specialist_sample, "agreement").get("stop_lines", [])
    source_groups = (primary_lines, specialist_lines, union_lines, [dict(line) for line in agreement_lines])
    output_groups = (primary_outputs, specialist_outputs)
    features: list[float] = []
    for lines in source_groups:
        features.extend(_line_stats(lines))
        for outputs in output_groups:
            features.extend(
                _line_profile_features_for_lines(
                    lines,
                    outputs=outputs,
                    image=image,
                    sample_index=int(sample_index),
                    meta=meta,
                    sample_count=int(sample_count),
                    side_offset=float(side_offset),
                    max_lines=1,
                )
            )
    return [0.0 if not math.isfinite(float(value)) else float(value) for value in features]


def _build_predictions_for_loader(
    *,
    loader: Any,
    max_batches: int,
    evaluator: Any,
    stop_line_evaluator: Any,
    postprocess_config: Any,
    stop_line_postprocess_config: Any,
    split_name: str,
    feature_mode: str,
    dense_feature_samples: int,
    dense_feature_side_offset: float,
    raster_size: tuple[int, int],
) -> tuple[list[dict[str, Any]], list[list[float]], list[np.ndarray], list[int], dict[str, list[dict[str, Any]]]]:
    raw_batches: list[dict[str, Any]] = []
    features: list[list[float]] = []
    rasters: list[np.ndarray] = []
    labels: list[int] = []
    predictions_by_mode: dict[str, list[dict[str, Any]]] = {mode: [] for mode in ROUTER_MODES}

    with torch.no_grad():
        for batch_index, batch in enumerate(loader, start=1):
            if batch_index > int(max_batches):
                break
            if batch_index == 1 or batch_index % 20 == 0:
                print(f"[stopline_source_router] {split_name} batch {batch_index}/{max_batches}", flush=True)
            raw_batch = raw_batch_for_metrics(batch)
            if raw_batch is None:
                raw_batch = batch
            raw_batches.append(raw_batch)
            gt_samples = _extract_gt_samples(raw_batch)
            encoded = evaluator.prepare_batch(batch)
            encoded_meta = _detach_to_cpu(encoded["meta"])
            base_outputs = _detach_to_cpu(evaluator.forward_encoded_batch(encoded))
            flipped_encoded = dict(encoded)
            flipped_encoded["image"] = torch.flip(encoded["image"], dims=(-1,))
            flipped_outputs = _detach_to_cpu(evaluator.forward_encoded_batch(flipped_encoded))
            flipped_unflipped = _unflip_lane_dense_outputs(flipped_outputs)
            stop_line_outputs = _detach_to_cpu(stop_line_evaluator.forward_encoded_batch(encoded))
            lane_outputs = _apply_lane_task_mask_competition(
                {
                    **base_outputs,
                    "lane_seg_centerline_logits": 0.5 * base_outputs["lane_seg_centerline_logits"]
                    + 0.5 * flipped_unflipped["lane_seg_centerline_logits"],
                },
                mask_keys=("crosswalk_mask_logits",),
                strength=0.50,
            )
            primary_predictions = postprocess_pv26_batch(
                lane_outputs,
                _detach_to_cpu(encoded["meta"]),
                config=postprocess_config,
            )
            specialist_predictions = postprocess_pv26_batch(
                _merge_stop_line_outputs(lane_outputs, stop_line_outputs),
                _detach_to_cpu(encoded["meta"]),
                config=stop_line_postprocess_config,
            )
            for sample_index, (primary_sample, specialist_sample, gt_sample) in enumerate(
                zip(
                    primary_predictions,
                    specialist_predictions,
                    gt_samples,
                )
            ):
                sample_features = _features_for_sample(primary_sample, specialist_sample)
                normalized_feature_mode = str(feature_mode).strip().lower()
                if normalized_feature_mode == "dense_aligned":
                    sample_meta = (
                        encoded_meta[sample_index]
                        if isinstance(encoded_meta, list)
                        and sample_index < len(encoded_meta)
                        and isinstance(encoded_meta[sample_index], dict)
                        else primary_sample.get("meta", {})
                    )
                    if not isinstance(sample_meta, dict):
                        sample_meta = {}
                    sample_features.extend(
                        _dense_aligned_features_for_sample(
                            primary_sample,
                            specialist_sample,
                            primary_outputs=lane_outputs,
                            specialist_outputs=stop_line_outputs,
                            image=encoded.get("image") if isinstance(encoded, dict) else None,
                            sample_index=int(sample_index),
                            meta=sample_meta,
                            sample_count=int(dense_feature_samples),
                            side_offset=float(dense_feature_side_offset),
                        )
                    )
                elif normalized_feature_mode == "line_profile":
                    sample_meta = (
                        encoded_meta[sample_index]
                        if isinstance(encoded_meta, list)
                        and sample_index < len(encoded_meta)
                        and isinstance(encoded_meta[sample_index], dict)
                        else primary_sample.get("meta", {})
                    )
                    if not isinstance(sample_meta, dict):
                        sample_meta = {}
                    sample_features.extend(
                        _line_profile_features_for_sample(
                            primary_sample,
                            specialist_sample,
                            primary_outputs=lane_outputs,
                            specialist_outputs=stop_line_outputs,
                            image=encoded.get("image") if isinstance(encoded, dict) else None,
                            sample_index=int(sample_index),
                            meta=sample_meta,
                            sample_count=int(dense_feature_samples),
                            side_offset=float(dense_feature_side_offset),
                        )
                    )
                elif normalized_feature_mode == "lane_topology":
                    sample_features.extend(_lane_topology_features_for_sample(primary_sample, specialist_sample))
                elif normalized_feature_mode == "raster_cnn":
                    sample_meta = (
                        encoded_meta[sample_index]
                        if isinstance(encoded_meta, list)
                        and sample_index < len(encoded_meta)
                        and isinstance(encoded_meta[sample_index], dict)
                        else primary_sample.get("meta", {})
                    )
                    if not isinstance(sample_meta, dict):
                        sample_meta = {}
                    rasters.append(
                        _source_raster_for_sample(
                            primary_sample,
                            specialist_sample,
                            primary_outputs=lane_outputs,
                            specialist_outputs=stop_line_outputs,
                            image=encoded.get("image") if isinstance(encoded, dict) else None,
                            sample_index=int(sample_index),
                            meta=sample_meta,
                            size=raster_size,
                        )
                    )
                features.append(sample_features)
                labels.append(_label_for_sample(primary_sample, specialist_sample, gt_sample))
                for mode in ROUTER_MODES:
                    predictions_by_mode[mode].append(_source_prediction(primary_sample, specialist_sample, mode))

    return raw_batches, features, rasters, labels, predictions_by_mode


def _train_router(
    features: list[list[float]],
    labels: list[int],
    *,
    hidden_dim: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    seed: int,
) -> tuple[SourceRouterMLP, torch.Tensor, torch.Tensor, dict[str, Any]]:
    if not features:
        raise ValueError("no router training features were collected")
    torch.manual_seed(int(seed))
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))
    x = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)
    mean = x.mean(dim=0)
    std = x.std(dim=0).clamp_min(1.0e-6)
    x_norm = (x - mean) / std
    model = SourceRouterMLP(x_norm.shape[1], int(hidden_dim), len(ROUTER_MODES))
    counts = torch.bincount(y, minlength=len(ROUTER_MODES)).to(dtype=torch.float32)
    weights = torch.where(counts > 0.0, float(y.numel()) / (counts * len(ROUTER_MODES)), torch.zeros_like(counts))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    losses: list[float] = []
    for _ in range(int(epochs)):
        optimizer.zero_grad(set_to_none=True)
        logits = model(x_norm)
        loss = F.cross_entropy(logits, y, weight=weights)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    with torch.no_grad():
        train_logits = model(x_norm)
        train_pred = train_logits.argmax(dim=1)
        train_accuracy = float((train_pred == y).float().mean().item())
    diagnostics = {
        "feature_count": int(x.shape[0]),
        "feature_dim": int(x.shape[1]),
        "label_counts": {mode: int(counts[index].item()) for index, mode in enumerate(ROUTER_MODES)},
        "class_weights": {mode: float(weights[index].item()) for index, mode in enumerate(ROUTER_MODES)},
        "train_accuracy": train_accuracy,
        "loss_first": losses[0] if losses else None,
        "loss_last": losses[-1] if losses else None,
    }
    return model, mean, std, diagnostics


def _train_raster_router(
    rasters: list[np.ndarray],
    labels: list[int],
    *,
    base_channels: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    seed: int,
    device: str,
) -> tuple[SourceRouterRasterCNN, torch.Tensor, torch.Tensor, dict[str, Any]]:
    if not rasters:
        raise ValueError("no raster router training inputs were collected")
    if len(rasters) != len(labels):
        raise ValueError(f"raster/label length mismatch: {len(rasters)} != {len(labels)}")
    torch.manual_seed(int(seed))
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))
    x_cpu = torch.tensor(np.stack(rasters, axis=0), dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)
    mean = x_cpu.mean(dim=(0, 2, 3), keepdim=True)
    std = x_cpu.std(dim=(0, 2, 3), keepdim=True).clamp_min(1.0e-6)
    target_device = torch.device(device if str(device).startswith("cuda") and torch.cuda.is_available() else "cpu")
    x = ((x_cpu - mean) / std).to(target_device)
    y_device = y.to(target_device)
    model = SourceRouterRasterCNN(
        input_channels=int(x.shape[1]),
        output_dim=len(ROUTER_MODES),
        base_channels=int(base_channels),
    ).to(target_device)
    counts = torch.bincount(y, minlength=len(ROUTER_MODES)).to(dtype=torch.float32)
    weights = torch.where(counts > 0.0, float(y.numel()) / (counts * len(ROUTER_MODES)), torch.zeros_like(counts))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    losses: list[float] = []
    for _ in range(int(epochs)):
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.cross_entropy(logits, y_device, weight=weights.to(target_device))
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    with torch.no_grad():
        train_logits = model(x)
        train_pred = train_logits.argmax(dim=1).detach().cpu()
        train_accuracy = float((train_pred == y).float().mean().item())
    diagnostics = {
        "feature_count": int(x_cpu.shape[0]),
        "input_channels": int(x_cpu.shape[1]),
        "raster_height": int(x_cpu.shape[2]),
        "raster_width": int(x_cpu.shape[3]),
        "label_counts": {mode: int(counts[index].item()) for index, mode in enumerate(ROUTER_MODES)},
        "class_weights": {mode: float(weights[index].item()) for index, mode in enumerate(ROUTER_MODES)},
        "train_accuracy": train_accuracy,
        "loss_first": losses[0] if losses else None,
        "loss_last": losses[-1] if losses else None,
    }
    return model, mean.cpu(), std.cpu(), diagnostics


def _predict_router_modes(
    model: SourceRouterMLP,
    mean: torch.Tensor,
    std: torch.Tensor,
    features: list[list[float]],
) -> list[int]:
    if not features:
        return []
    with torch.no_grad():
        x = torch.tensor(features, dtype=torch.float32)
        logits = model((x - mean) / std)
        return [int(value) for value in logits.argmax(dim=1).tolist()]


def _predict_raster_router_modes(
    model: SourceRouterRasterCNN,
    mean: torch.Tensor,
    std: torch.Tensor,
    rasters: list[np.ndarray],
) -> list[int]:
    if not rasters:
        return []
    target_device = next(model.parameters()).device
    with torch.no_grad():
        x = torch.tensor(np.stack(rasters, axis=0), dtype=torch.float32)
        x = ((x - mean.cpu()) / std.cpu()).to(target_device)
        logits = model(x)
        return [int(value) for value in logits.argmax(dim=1).detach().cpu().tolist()]


def _prediction_rows_from_choices(
    predictions_by_mode: dict[str, list[dict[str, Any]]],
    choices: list[int],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for sample_index, choice in enumerate(choices):
        mode = ROUTER_MODES[int(choice)]
        output.append(dict(predictions_by_mode[mode][sample_index]))
    return output


def _row_from_metrics(name: str, metrics: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {"variant": name}
    for task in ("lane", "stop_line", "crosswalk"):
        values = metrics.get(task, {}) if isinstance(metrics.get(task), dict) else {}
        for key in ("precision", "recall", "f1", "tp", "fp", "fn", "mean_point_distance", "mean_polygon_iou"):
            value = values.get(key)
            if isinstance(value, (int, float)):
                row[f"{task}_{key}"] = value
    lane_family = metrics.get("lane_family", {}) if isinstance(metrics.get("lane_family"), dict) else {}
    for key in ("mean_f1", "min_f1"):
        value = lane_family.get(key)
        if isinstance(value, (int, float)):
            row[f"lane_family_{key}"] = value
    row["phase4_objective_proxy"] = (
        0.50 * float(metrics.get("lane", {}).get("f1", 0.0))
        + 0.30 * float(metrics.get("stop_line", {}).get("f1", 0.0))
        + 0.20 * float(metrics.get("crosswalk", {}).get("f1", 0.0))
    )
    return row


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _json_safe(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _json_safe(asdict(value))
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return _json_ready(value)


def _choice_counts(choices: list[int]) -> dict[str, int]:
    return {mode: int(sum(1 for choice in choices if int(choice) == index)) for index, mode in enumerate(ROUTER_MODES)}


def _config_brief(config: Any) -> dict[str, Any]:
    return {
        "device": str(getattr(config, "device", "")),
        "batch_size": int(getattr(config, "batch_size", 0)),
        "train_batches": int(getattr(config, "train_batches", 0)),
        "val_batches": int(getattr(config, "val_batches", 0)),
        "task_mode": str(getattr(config, "task_mode", "")),
        "roadmark_architecture": str(getattr(config, "roadmark_architecture", "")),
        "lane_head_mode": str(getattr(config, "lane_head_mode", "")),
        "lane_segfirst_track_mode": str(getattr(config, "lane_segfirst_track_mode", "")),
        "crosswalk_polygon_mode": str(getattr(config, "crosswalk_polygon_mode", "")),
    }


def _postprocess_brief(config: Any) -> dict[str, Any]:
    fields = (
        "lane_segfirst_track_mode",
        "lane_segfirst_semantic_vote_mode",
        "stop_line_projection_comp_enabled",
        "stop_line_projection_comp_min_gap",
        "stop_line_projection_comp_topk",
        "stop_line_projection_comp_max_predictions",
        "crosswalk_polygon_mode",
    )
    return {field: _json_safe(getattr(config, field, None)) for field in fields}


def main() -> int:
    args = parse_args()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    stop_line_checkpoint = Path(args.stop_line_checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    if not stop_line_checkpoint.is_file():
        raise FileNotFoundError(f"stop-line checkpoint not found: {stop_line_checkpoint}")

    source_run = Path(args.source_run).expanduser().resolve()
    scenario, scenario_path, _options, phase, _phase_selection, train_config = _build_eval_contract(
        args,
        source_run=source_run,
        checkpoint=checkpoint,
        experiment=str(args.lane60_experiment),
    )
    (
        stop_line_scenario,
        stop_line_scenario_path,
        _stop_line_options,
        stop_line_phase,
        _stop_line_phase_selection,
        stop_line_train_config,
    ) = _build_eval_contract(
        args,
        source_run=source_run,
        checkpoint=stop_line_checkpoint,
        experiment=str(args.stop_line_lane60_experiment),
    )
    train_config = dataclasses_replace(
        train_config,
        device=_resolve_device(str(args.device), train_config.device),
        encode_train_batches_in_loader=False,
    )
    stop_line_train_config = dataclasses_replace(
        stop_line_train_config,
        device=_resolve_device(str(args.device), stop_line_train_config.device),
    )

    train_cli._configure_torch_multiprocessing()
    dataset = train_cli.PV26CanonicalDataset(
        train_cli._existing_dataset_roots(scenario),
        train_augmentation=False,
        progress_callback=lambda message: print(f"[stopline_source_router] {message}", flush=True),
    )
    train_loader, val_loader = train_cli._build_phase_train_loaders(dataset, train_config=train_config, phase=phase)
    if train_loader is None or val_loader is None:
        raise ValueError("source-router probe requires both train and validation loaders")
    _advance_validation_sampler(val_loader, validation_epoch=int(args.validation_epoch))

    trainer = train_cli._build_phase_trainer(phase, train_config)
    load_report = trainer.load_model_weights(checkpoint, map_location=train_config.device)
    evaluator = trainer.build_evaluator()
    evaluator.adapter.raw_model.eval()
    evaluator.heads.eval()
    postprocess_config = _postprocess_override_config(args, trainer) or train_cli._build_postprocess_config(train_config)

    stop_line_trainer = train_cli._build_phase_trainer(stop_line_phase, stop_line_train_config)
    stop_line_load_report = stop_line_trainer.load_model_weights(
        stop_line_checkpoint,
        map_location=stop_line_train_config.device,
    )
    stop_line_evaluator = stop_line_trainer.build_evaluator()
    stop_line_evaluator.adapter.raw_model.eval()
    stop_line_evaluator.heads.eval()
    stop_line_postprocess_config = (
        _postprocess_override_config(args, stop_line_trainer)
        or train_cli._build_postprocess_config(stop_line_train_config)
    )

    raster_size = (int(args.raster_height), int(args.raster_width))
    train_raw_batches, train_features, train_rasters, train_labels, _train_predictions = _build_predictions_for_loader(
        loader=train_loader,
        max_batches=int(args.router_train_batches),
        evaluator=evaluator,
        stop_line_evaluator=stop_line_evaluator,
        postprocess_config=postprocess_config,
        stop_line_postprocess_config=stop_line_postprocess_config,
        split_name="train",
        feature_mode=str(args.feature_mode),
        dense_feature_samples=int(args.dense_feature_samples),
        dense_feature_side_offset=float(args.dense_feature_side_offset),
        raster_size=raster_size,
    )
    if str(args.feature_mode).strip().lower() == "raster_cnn":
        router, feature_mean, feature_std, router_diagnostics = _train_raster_router(
            train_rasters,
            train_labels,
            base_channels=int(args.raster_base_channels),
            epochs=int(args.router_epochs),
            lr=float(args.router_lr),
            weight_decay=float(args.router_weight_decay),
            seed=int(args.seed),
            device=train_config.device,
        )
    else:
        router, feature_mean, feature_std, router_diagnostics = _train_router(
            train_features,
            train_labels,
            hidden_dim=int(args.router_hidden_dim),
            epochs=int(args.router_epochs),
            lr=float(args.router_lr),
            weight_decay=float(args.router_weight_decay),
            seed=int(args.seed),
        )
    val_raw_batches, val_features, val_rasters, val_labels, val_predictions_by_mode = _build_predictions_for_loader(
        loader=val_loader,
        max_batches=int(args.max_val_batches),
        evaluator=evaluator,
        stop_line_evaluator=stop_line_evaluator,
        postprocess_config=postprocess_config,
        stop_line_postprocess_config=stop_line_postprocess_config,
        split_name="val",
        feature_mode=str(args.feature_mode),
        dense_feature_samples=int(args.dense_feature_samples),
        dense_feature_side_offset=float(args.dense_feature_side_offset),
        raster_size=raster_size,
    )
    if str(args.feature_mode).strip().lower() == "raster_cnn":
        val_choices = _predict_raster_router_modes(router, feature_mean, feature_std, val_rasters)
    else:
        val_choices = _predict_router_modes(router, feature_mean, feature_std, val_features)
    val_oracle_choices = [int(value) for value in val_labels]

    merged_raw = _merge_raw_batches(val_raw_batches)
    predictions_by_variant: dict[str, list[dict[str, Any]]] = {
        mode: val_predictions_by_mode[mode] for mode in ROUTER_MODES
    }
    predictions_by_variant["learned_router"] = _prediction_rows_from_choices(val_predictions_by_mode, val_choices)
    predictions_by_variant["oracle_router"] = _prediction_rows_from_choices(val_predictions_by_mode, val_oracle_choices)
    metrics_by_variant = {
        name: augment_lane_family_metrics(summarize_pv26_metrics(predictions, merged_raw))
        for name, predictions in predictions_by_variant.items()
    }
    rows = [_row_from_metrics(name, metrics) for name, metrics in metrics_by_variant.items()]
    rows.sort(key=lambda row: float(row.get("phase4_objective_proxy", 0.0)), reverse=True)

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if str(args.output_dir).strip()
        else checkpoint.parent / "analysis_exports" / "stopline_source_router"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "metrics.csv", rows)
    if str(args.save_router_model).strip():
        router_state_kind = "raster_cnn" if isinstance(router, SourceRouterRasterCNN) else "mlp"
        torch.save(
            {
                "state_dict": router.state_dict(),
                "feature_mean": feature_mean,
                "feature_std": feature_std,
                "router_modes": ROUTER_MODES,
                "router_kind": router_state_kind,
                "diagnostics": router_diagnostics,
            },
            Path(args.save_router_model).expanduser().resolve(),
        )
    payload = {
        "checkpoint": str(checkpoint),
        "stop_line_checkpoint": str(stop_line_checkpoint),
        "scenario_path": str(scenario_path),
        "stop_line_scenario_path": str(stop_line_scenario_path),
        "lane60_experiment": str(args.lane60_experiment),
        "stop_line_lane60_experiment": str(args.stop_line_lane60_experiment),
        "validation_epoch": int(args.validation_epoch),
        "router_train_batches": int(args.router_train_batches),
        "max_val_batches": int(args.max_val_batches),
        "feature_mode": str(args.feature_mode),
        "router_kind": "raster_cnn" if isinstance(router, SourceRouterRasterCNN) else "mlp",
        "dense_feature_samples": int(args.dense_feature_samples),
        "dense_feature_side_offset": float(args.dense_feature_side_offset),
        "raster_height": int(args.raster_height),
        "raster_width": int(args.raster_width),
        "raster_base_channels": int(args.raster_base_channels),
        "processed_train_batches": int(len(train_raw_batches)),
        "processed_val_batches": int(len(val_raw_batches)),
        "router_modes": list(ROUTER_MODES),
        "router_diagnostics": router_diagnostics,
        "val_label_counts": _choice_counts(val_oracle_choices),
        "val_router_choice_counts": _choice_counts(val_choices),
        "train_config": _config_brief(train_config),
        "stop_line_train_config": _config_brief(stop_line_train_config),
        "postprocess_config": _postprocess_brief(postprocess_config),
        "stop_line_postprocess_config": _postprocess_brief(stop_line_postprocess_config),
        "load_report": {
            "missing_keys": len(load_report.get("missing_keys", [])) if isinstance(load_report, dict) else None,
            "unexpected_keys": len(load_report.get("unexpected_keys", [])) if isinstance(load_report, dict) else None,
        },
        "stop_line_load_report": {
            "missing_keys": len(stop_line_load_report.get("missing_keys", []))
            if isinstance(stop_line_load_report, dict)
            else None,
            "unexpected_keys": len(stop_line_load_report.get("unexpected_keys", []))
            if isinstance(stop_line_load_report, dict)
            else None,
        },
        "rows": rows,
        "metrics_by_variant": _json_safe(metrics_by_variant),
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(rows, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    print(f"[stopline_source_router] wrote {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
