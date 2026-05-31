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

from model.engine._trainer_epochs import _merge_raw_batches
from model.engine.batch import augment_lane_family_metrics, raw_batch_for_metrics
from model.engine.metrics import _extract_gt_samples, summarize_pv26_metrics
from model.engine.postprocess import _dedupe_stop_line_predictions, _stopline_prediction_sort_key, postprocess_pv26_batch
from tools.evaluate_pv26_lane60_checkpoint import _advance_validation_sampler, _postprocess_override_config
from tools.probe_pv26_lane60_postprocess_thresholds import _detach_to_cpu
from tools.probe_pv26_lane_temporal_neighbor_union import _build_scenario
from tools.probe_pv26_stopline_angle_mask_extent import _as_2d_array, _row_from_metrics, _sample_tensor, _write_csv
from tools.probe_pv26_stopline_candidate_pool import (
    _fit_raw_patch_mlp,
    _nearest_gt,
    _predict_raw_patch_mlp,
    _standardize_from_train,
    _write_candidate_features_csv,
)
from tools.probe_pv26_stopline_raw_hough_candidates import (
    _endpoint_proposal_mean,
    _line_stats,
    _raw_points_to_output,
    _slice_raw_batch_sample,
)
from tools.probe_pv26_stopline_temporal_candidates import _copy_stop_line, _line_length, _nearest_current_distance
from tools.pv26_train import cli as train_cli


SCORE_KEY = "lane_pair_mlp_score"
LANE_PAIR_FEATURES = (
    "lane_pair_score",
    "lane_pair_rank_norm",
    "lane_pair_dense_anchor_score",
    "lane_pair_left_lane_rank_norm",
    "lane_pair_right_lane_rank_norm",
    "lane_pair_y_fraction",
    "lane_pair_length_norm",
    "lane_pair_gap_px_norm",
    "lane_pair_overlap_norm",
    "lane_pair_lane_score_mean",
    "lane_pair_lane_score_min",
    "lane_pair_tangent_cos_abs",
    "lane_pair_axis_dot_tangent_abs_mean",
    "lane_pair_current_stopline_count",
    "lane_pair_nearest_current_distance_norm",
    "lane_pair_stop_mask_mean",
    "lane_pair_stop_mask_max",
    "lane_pair_stop_center_mean",
    "lane_pair_stop_center_max",
    "lane_pair_stop_selector_mean",
    "lane_pair_stop_selector_max",
    "lane_pair_stop_proposal_mean",
    "lane_pair_stop_proposal_max",
    "lane_pair_endpoint_proposal_mean",
    "lane_pair_lane_centerline_mean",
    "lane_pair_lane_centerline_max",
    "lane_pair_lane_support_mean",
    "lane_pair_lane_support_max",
    "lane_pair_lane_endpoint_centerline_mean",
    "lane_pair_lane_endpoint_support_mean",
    "lane_pair_center_x_norm",
    "lane_pair_center_y_norm",
    "lane_pair_abs_cos",
    "lane_pair_abs_sin",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate stop-line candidates from predicted lane-pair topology, train a small "
            "no-GT verifier on canonical train batches, and replay it on validation."
        )
    )
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--source-run", default="")
    parser.add_argument("--lane60-experiment", default="stopline_projection_comp_runtime")
    parser.add_argument("--preset", default="default")
    parser.add_argument("--phase-index", type=int, default=4)
    parser.add_argument("--train-record-batches", type=int, default=64)
    parser.add_argument("--max-val-batches", type=int, default=4)
    parser.add_argument("--validation-epoch", type=int, default=2)
    parser.add_argument("--train-batches", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dataset-root", default="")
    parser.add_argument("--lane-pair-y-fractions", default="0.65,0.75,0.85")
    parser.add_argument("--max-lane-pair-lanes", type=int, default=12)
    parser.add_argument("--lane-pair-min-overlap-px", type=float, default=32.0)
    parser.add_argument("--lane-pair-min-gap-px", type=float, default=24.0)
    parser.add_argument("--lane-pair-max-gap-frac", type=float, default=0.95)
    parser.add_argument("--lane-pair-dense-anchor-count", type=int, default=3)
    parser.add_argument("--lane-pair-dense-anchor-stride-px", type=float, default=12.0)
    parser.add_argument("--lane-pair-top-k", type=int, default=8)
    parser.add_argument("--max-lane-pair-candidates", type=int, default=16)
    parser.add_argument("--max-components", type=int, default=2)
    parser.add_argument("--threshold-grid", type=int, default=101)
    parser.add_argument("--verifier-epochs", type=int, default=60)
    parser.add_argument("--verifier-lr", type=float, default=1.0e-3)
    parser.add_argument("--stop-line-mask-binary-threshold", type=float, default=None)
    parser.add_argument("--stop-line-min-instance-score", type=float, default=None)
    parser.add_argument("--stop-line-presence-threshold", type=float, default=None)
    parser.add_argument(
        "--stop-line-component-gate-source",
        choices=("center", "selector", "max", "validator", "max_validator", "product_validator"),
        default=None,
    )
    parser.add_argument("--crosswalk-polygon-mode", choices=("rect", "hull"), default="hull")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def _parse_y_fractions(value: str) -> tuple[float, ...]:
    fractions = tuple(float(part.strip()) for part in str(value).split(",") if part.strip())
    if not fractions:
        raise ValueError("lane-pair y fractions must contain at least one value")
    for fraction in fractions:
        if not math.isfinite(float(fraction)) or float(fraction) < 0.0 or float(fraction) > 1.0:
            raise ValueError(f"lane-pair y fraction must be in [0, 1], got {fraction}")
    return fractions


def _lane_points_array(lane: dict[str, Any]) -> np.ndarray:
    points = np.asarray(lane.get("points_xy", []), dtype=np.float32).reshape(-1, 2)
    if points.shape[0] < 2 or not bool(np.isfinite(points).all()):
        return np.zeros((0, 2), dtype=np.float32)
    keep = np.ones((points.shape[0],), dtype=bool)
    if points.shape[0] > 1:
        delta = np.linalg.norm(points[1:] - points[:-1], axis=1)
        keep[1:] = delta > 1.0e-4
    points = points[keep]
    if points.shape[0] < 2:
        return np.zeros((0, 2), dtype=np.float32)
    order = np.argsort(points[:, 1], kind="mergesort")
    return points[order].astype(np.float32)


def _polyline_length(points: np.ndarray) -> float:
    points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if points.shape[0] < 2:
        return 0.0
    return float(np.linalg.norm(points[1:] - points[:-1], axis=1).sum())


def _interpolate_lane_at_y(points: np.ndarray, y_value: float) -> tuple[float, np.ndarray] | None:
    points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if points.shape[0] < 2:
        return None
    y = float(y_value)
    hits: list[tuple[float, np.ndarray]] = []
    for start, end in zip(points[:-1], points[1:]):
        y0, y1 = float(start[1]), float(end[1])
        x0, x1 = float(start[0]), float(end[0])
        if abs(y1 - y0) <= 1.0e-6:
            continue
        lo = min(y0, y1) - 1.0e-5
        hi = max(y0, y1) + 1.0e-5
        if y < lo or y > hi:
            continue
        t = (y - y0) / (y1 - y0)
        x_value = x0 + t * (x1 - x0)
        tangent = np.asarray([x1 - x0, y1 - y0], dtype=np.float32)
        norm = float(np.linalg.norm(tangent))
        if not math.isfinite(float(x_value)) or norm <= 1.0e-6:
            continue
        tangent = tangent / norm
        hits.append((float(x_value), tangent.astype(np.float32)))
    if not hits:
        return None
    hits.sort(key=lambda item: item[0])
    median_index = len(hits) // 2
    return float(hits[median_index][0]), hits[median_index][1].astype(np.float32)


def _clip_points(points: np.ndarray, meta: dict[str, Any]) -> np.ndarray:
    raw_h, raw_w = int(meta.get("raw_hw", (1, 1))[0]), int(meta.get("raw_hw", (1, 1))[1])
    clipped = np.asarray(points, dtype=np.float32).reshape(-1, 2).copy()
    clipped[:, 0] = np.clip(clipped[:, 0], 0.0, max(float(raw_w - 1), 0.0))
    clipped[:, 1] = np.clip(clipped[:, 1], 0.0, max(float(raw_h - 1), 0.0))
    return clipped


def _proposal_line_score(
    points_raw: np.ndarray,
    *,
    meta: dict[str, Any],
    proposal_map: np.ndarray | None,
) -> float:
    if proposal_map is None:
        return 0.0
    points = np.asarray(points_raw, dtype=np.float32).reshape(-1, 2)
    if points.shape[0] < 2:
        return 0.0
    output_hw = (int(proposal_map.shape[0]), int(proposal_map.shape[1]))
    output_points = _raw_points_to_output(points, meta, output_hw)
    mean_value, max_value = _line_stats(proposal_map, output_points)
    return float(0.5 * (float(mean_value) + float(max_value)))


def _candidate_from_lane_pair_anchor(
    left_lane: dict[str, Any],
    right_lane: dict[str, Any],
    *,
    y_anchor: float,
    y_fraction: float,
    fraction_index: int,
    dense_anchor_score: float,
    meta: dict[str, Any],
    min_gap_px: float,
    max_gap_px: float,
) -> dict[str, Any] | None:
    left_hit = _interpolate_lane_at_y(left_lane["points"], y_anchor)
    right_hit = _interpolate_lane_at_y(right_lane["points"], y_anchor)
    if left_hit is None or right_hit is None:
        return None
    left_x, left_tangent = left_hit
    right_x, right_tangent = right_hit
    gap = abs(float(right_x) - float(left_x))
    if gap < float(min_gap_px) or gap > float(max_gap_px):
        return None
    start = np.asarray([left_x, y_anchor], dtype=np.float32)
    end = np.asarray([right_x, y_anchor], dtype=np.float32)
    if float(start[0]) > float(end[0]):
        start, end = end, start
        left_lane, right_lane = right_lane, left_lane
        left_tangent, right_tangent = right_tangent, left_tangent
    clipped = _clip_points(np.stack([start, end], axis=0), meta)
    length = float(np.linalg.norm(clipped[-1] - clipped[0]))
    if length < float(min_gap_px):
        return None
    tangent_cos_abs = abs(float(np.dot(left_tangent, right_tangent)))
    axis = clipped[-1] - clipped[0]
    axis_norm = float(np.linalg.norm(axis))
    axis_unit = axis / max(axis_norm, 1.0e-6)
    axis_dot_tangent_abs_mean = 0.5 * (
        abs(float(np.dot(axis_unit, left_tangent))) + abs(float(np.dot(axis_unit, right_tangent)))
    )
    lane_score_mean = 0.5 * (float(left_lane["score"]) + float(right_lane["score"]))
    lane_score_min = min(float(left_lane["score"]), float(right_lane["score"]))
    overlap = float(min(float(left_lane["y_max"]), float(right_lane["y_max"])) - max(float(left_lane["y_min"]), float(right_lane["y_min"])))
    return {
        "points_xy": [[float(x), float(y)] for x, y in clipped.tolist()],
        "score": float(lane_score_mean),
        "center_score": float(lane_score_mean),
        "source": "lane_pair",
        "proposal_source": "lane_pair",
        "lane_pair_left_rank": int(left_lane["rank"]),
        "lane_pair_right_rank": int(right_lane["rank"]),
        "lane_pair_fraction_index": int(fraction_index),
        "lane_pair_y_fraction": float(y_fraction),
        "lane_pair_dense_anchor_score": float(dense_anchor_score),
        "lane_pair_gap_px": float(gap),
        "lane_pair_overlap_px": float(overlap),
        "lane_pair_lane_score_mean": float(lane_score_mean),
        "lane_pair_lane_score_min": float(lane_score_min),
        "lane_pair_tangent_cos_abs": float(tangent_cos_abs),
        "lane_pair_axis_dot_tangent_abs_mean": float(axis_dot_tangent_abs_mean),
        "lane_pair_length": float(length),
    }


def _lane_pair_candidates_from_lanes(
    lanes: list[dict[str, Any]],
    *,
    meta: dict[str, Any],
    y_fractions: tuple[float, ...],
    max_lanes: int,
    min_overlap_px: float,
    min_gap_px: float,
    max_gap_frac: float,
    proposal_map: np.ndarray | None = None,
    dense_anchor_count: int = 0,
    dense_anchor_stride_px: float = 12.0,
) -> list[dict[str, Any]]:
    raw_h, raw_w = int(meta.get("raw_hw", (1, 1))[0]), int(meta.get("raw_hw", (1, 1))[1])
    prepared: list[dict[str, Any]] = []
    for lane_rank, lane in enumerate(lanes, start=1):
        points = _lane_points_array(lane)
        if points.shape[0] < 2:
            continue
        score = float(lane.get("score", lane.get("instance_score", 1.0)) or 0.0)
        prepared.append(
            {
                "lane": lane,
                "points": points,
                "rank": int(lane_rank),
                "score": score,
                "length": _polyline_length(points),
                "y_min": float(points[:, 1].min()),
                "y_max": float(points[:, 1].max()),
            }
        )
    prepared.sort(key=lambda item: (float(item["score"]), float(item["length"])), reverse=True)
    prepared = prepared[: max(2, int(max_lanes))]
    candidates: list[dict[str, Any]] = []
    max_gap_px = max(float(min_gap_px), float(max_gap_frac) * max(float(raw_w), 1.0))
    for left_index, first in enumerate(prepared):
        for second in prepared[left_index + 1 :]:
            y_low = max(float(first["y_min"]), float(second["y_min"]), 0.0)
            y_high = min(float(first["y_max"]), float(second["y_max"]), max(float(raw_h - 1), 0.0))
            overlap = y_high - y_low
            if overlap < float(min_overlap_px):
                continue
            anchors: list[tuple[int, float, float, float]] = [
                (index, float(fraction), y_low + float(fraction) * overlap, 0.0)
                for index, fraction in enumerate(y_fractions)
            ]
            if proposal_map is not None and int(dense_anchor_count) > 0:
                stride = max(2.0, float(dense_anchor_stride_px))
                dense_rows = np.arange(y_low, y_high + 1.0e-3, stride, dtype=np.float32)
                dense_scores: list[tuple[float, float, float]] = []
                for y_anchor_value in dense_rows.tolist():
                    left_hit = _interpolate_lane_at_y(first["points"], float(y_anchor_value))
                    right_hit = _interpolate_lane_at_y(second["points"], float(y_anchor_value))
                    if left_hit is None or right_hit is None:
                        continue
                    left_x, _left_tangent = left_hit
                    right_x, _right_tangent = right_hit
                    gap = abs(float(right_x) - float(left_x))
                    if gap < float(min_gap_px) or gap > max_gap_px:
                        continue
                    points_raw = _clip_points(
                        np.asarray([[left_x, y_anchor_value], [right_x, y_anchor_value]], dtype=np.float32),
                        meta,
                    )
                    score = _proposal_line_score(points_raw, meta=meta, proposal_map=proposal_map)
                    dense_scores.append((float(score), float((float(y_anchor_value) - y_low) / max(overlap, 1.0)), float(y_anchor_value)))
                dense_scores.sort(key=lambda item: item[0], reverse=True)
                for dense_index, (score, fraction, y_anchor_value) in enumerate(
                    dense_scores[: max(0, int(dense_anchor_count))],
                    start=1,
                ):
                    anchors.append((-dense_index, float(fraction), float(y_anchor_value), float(score)))
            for fraction_index, fraction, y_anchor, dense_score in anchors:
                candidate = _candidate_from_lane_pair_anchor(
                    first,
                    second,
                    y_anchor=float(y_anchor),
                    y_fraction=float(fraction),
                    fraction_index=int(fraction_index),
                    dense_anchor_score=float(dense_score),
                    meta=meta,
                    min_gap_px=float(min_gap_px),
                    max_gap_px=float(max_gap_px),
                )
                if candidate is not None:
                    candidates.append(candidate)
    candidates.sort(
        key=lambda item: (
            float(item.get("lane_pair_lane_score_mean", 0.0)),
            float(item.get("lane_pair_dense_anchor_score", 0.0)),
            float(item.get("lane_pair_overlap_px", 0.0)),
            -abs(float(item.get("lane_pair_y_fraction", 0.0)) - 0.75),
            float(item.get("lane_pair_gap_px", 0.0)),
        ),
        reverse=True,
    )
    return candidates


def _endpoint_mean(map_array: np.ndarray | None, output_points: np.ndarray) -> float:
    if map_array is None or output_points.shape[0] < 2:
        return 0.0
    endpoints = np.stack([output_points[0], output_points[-1]], axis=0)
    values = _line_stats(map_array, endpoints)
    return float(values[0])


def _lane_pair_features(
    candidate: dict[str, Any],
    *,
    meta: dict[str, Any],
    stop_mask_probs: np.ndarray | None,
    stop_center_probs: np.ndarray | None,
    stop_selector_probs: np.ndarray | None,
    lane_centerline_probs: np.ndarray | None,
    lane_support_probs: np.ndarray | None,
    current_stop_lines: list[dict[str, Any]],
    candidate_rank: int,
) -> dict[str, float]:
    points_raw = np.asarray(candidate.get("points_xy", []), dtype=np.float32).reshape(-1, 2)
    reference_map = next(
        (
            array
            for array in (
                stop_mask_probs,
                stop_center_probs,
                stop_selector_probs,
                lane_centerline_probs,
                lane_support_probs,
            )
            if isinstance(array, np.ndarray)
        ),
        None,
    )
    if points_raw.shape[0] < 2 or reference_map is None:
        return {name: 0.0 for name in LANE_PAIR_FEATURES}
    output_hw = (int(reference_map.shape[0]), int(reference_map.shape[1]))
    output_points = _raw_points_to_output(points_raw, meta, output_hw)
    proposal_map = None
    if stop_center_probs is not None and stop_selector_probs is not None:
        proposal_map = np.maximum(stop_center_probs, stop_selector_probs)
    elif stop_center_probs is not None:
        proposal_map = stop_center_probs
    elif stop_selector_probs is not None:
        proposal_map = stop_selector_probs
    stop_mask_mean, stop_mask_max = _line_stats(stop_mask_probs, output_points)
    stop_center_mean, stop_center_max = _line_stats(stop_center_probs, output_points)
    stop_selector_mean, stop_selector_max = _line_stats(stop_selector_probs, output_points)
    proposal_mean, proposal_max = _line_stats(proposal_map, output_points)
    endpoint_mean = _endpoint_proposal_mean(proposal_map, output_points)
    lane_center_mean, lane_center_max = _line_stats(lane_centerline_probs, output_points)
    lane_support_mean, lane_support_max = _line_stats(lane_support_probs, output_points)
    lane_endpoint_center = _endpoint_mean(lane_centerline_probs, output_points)
    lane_endpoint_support = _endpoint_mean(lane_support_probs, output_points)
    raw_h, raw_w = int(meta.get("raw_hw", (1, 1))[0]), int(meta.get("raw_hw", (1, 1))[1])
    length = _line_length(candidate)
    center = points_raw.mean(axis=0)
    delta = points_raw[-1] - points_raw[0]
    norm = float(np.linalg.norm(delta))
    axis = delta / max(norm, 1.0e-6)
    nearest_distance = _nearest_current_distance(candidate, current_stop_lines)
    features = {
        "lane_pair_score": float(candidate.get("score", 0.0)),
        "lane_pair_rank_norm": float(1.0 / max(int(candidate_rank), 1)),
        "lane_pair_dense_anchor_score": float(candidate.get("lane_pair_dense_anchor_score", 0.0)),
        "lane_pair_left_lane_rank_norm": float(1.0 / max(int(candidate.get("lane_pair_left_rank", 999)), 1)),
        "lane_pair_right_lane_rank_norm": float(1.0 / max(int(candidate.get("lane_pair_right_rank", 999)), 1)),
        "lane_pair_y_fraction": float(candidate.get("lane_pair_y_fraction", 0.0)),
        "lane_pair_length_norm": float(min(length / max(float(raw_w), 1.0), 1.0)),
        "lane_pair_gap_px_norm": float(min(float(candidate.get("lane_pair_gap_px", 0.0)) / max(float(raw_w), 1.0), 2.0)),
        "lane_pair_overlap_norm": float(
            min(float(candidate.get("lane_pair_overlap_px", 0.0)) / max(float(raw_h), 1.0), 1.0)
        ),
        "lane_pair_lane_score_mean": float(candidate.get("lane_pair_lane_score_mean", 0.0)),
        "lane_pair_lane_score_min": float(candidate.get("lane_pair_lane_score_min", 0.0)),
        "lane_pair_tangent_cos_abs": float(candidate.get("lane_pair_tangent_cos_abs", 0.0)),
        "lane_pair_axis_dot_tangent_abs_mean": float(candidate.get("lane_pair_axis_dot_tangent_abs_mean", 0.0)),
        "lane_pair_current_stopline_count": float(len(current_stop_lines)),
        "lane_pair_nearest_current_distance_norm": float(min(nearest_distance / max(float(raw_w), 1.0), 4.0)),
        "lane_pair_stop_mask_mean": float(stop_mask_mean),
        "lane_pair_stop_mask_max": float(stop_mask_max),
        "lane_pair_stop_center_mean": float(stop_center_mean),
        "lane_pair_stop_center_max": float(stop_center_max),
        "lane_pair_stop_selector_mean": float(stop_selector_mean),
        "lane_pair_stop_selector_max": float(stop_selector_max),
        "lane_pair_stop_proposal_mean": float(proposal_mean),
        "lane_pair_stop_proposal_max": float(proposal_max),
        "lane_pair_endpoint_proposal_mean": float(endpoint_mean),
        "lane_pair_lane_centerline_mean": float(lane_center_mean),
        "lane_pair_lane_centerline_max": float(lane_center_max),
        "lane_pair_lane_support_mean": float(lane_support_mean),
        "lane_pair_lane_support_max": float(lane_support_max),
        "lane_pair_lane_endpoint_centerline_mean": float(lane_endpoint_center),
        "lane_pair_lane_endpoint_support_mean": float(lane_endpoint_support),
        "lane_pair_center_x_norm": float(np.clip(float(center[0]) / max(float(raw_w), 1.0), 0.0, 1.0)),
        "lane_pair_center_y_norm": float(np.clip(float(center[1]) / max(float(raw_h), 1.0), 0.0, 1.0)),
        "lane_pair_abs_cos": float(abs(float(axis[0]))),
        "lane_pair_abs_sin": float(abs(float(axis[1]))),
    }
    return {key: 0.0 if not math.isfinite(float(value)) else float(value) for key, value in features.items()}


def _build_lane_pair_candidates(
    *,
    meta: dict[str, Any],
    baseline_prediction: dict[str, Any],
    gt_stop_lines: list[dict[str, Any]],
    stop_mask_probs: np.ndarray | None,
    stop_center_probs: np.ndarray | None,
    stop_selector_probs: np.ndarray | None,
    lane_centerline_probs: np.ndarray | None,
    lane_support_probs: np.ndarray | None,
    y_fractions: tuple[float, ...],
    max_lanes: int,
    min_overlap_px: float,
    min_gap_px: float,
    max_gap_frac: float,
    dense_anchor_count: int = 0,
    dense_anchor_stride_px: float = 12.0,
    max_candidates: int = 16,
) -> list[dict[str, Any]]:
    current_stop_lines = list(baseline_prediction.get("stop_lines", []))
    candidates = _lane_pair_candidates_from_lanes(
        list(baseline_prediction.get("lanes", [])),
        meta=meta,
        y_fractions=y_fractions,
        max_lanes=int(max_lanes),
        min_overlap_px=float(min_overlap_px),
        min_gap_px=float(min_gap_px),
        max_gap_frac=float(max_gap_frac),
        proposal_map=(
            np.maximum(stop_center_probs, stop_selector_probs)
            if stop_center_probs is not None and stop_selector_probs is not None
            else stop_center_probs
            if stop_center_probs is not None
            else stop_selector_probs
        ),
        dense_anchor_count=int(dense_anchor_count),
        dense_anchor_stride_px=float(dense_anchor_stride_px),
    )
    output: list[dict[str, Any]] = []
    for rank, candidate in enumerate(candidates[: max(1, int(max_candidates))], start=1):
        distance, angle_error, gt_index = _nearest_gt(candidate, gt_stop_lines)
        candidate["nearest_gt_distance"] = float(distance)
        candidate["nearest_gt_angle_error"] = float(angle_error)
        candidate["nearest_gt_index"] = int(gt_index)
        candidate["is_oracle_positive"] = bool(float(distance) <= 40.0)
        candidate["lane_pair_rank_score"] = float(1.0 / max(rank, 1))
        candidate.update(
            _lane_pair_features(
                candidate,
                meta=meta,
                stop_mask_probs=stop_mask_probs,
                stop_center_probs=stop_center_probs,
                stop_selector_probs=stop_selector_probs,
                lane_centerline_probs=lane_centerline_probs,
                lane_support_probs=lane_support_probs,
                current_stop_lines=current_stop_lines,
                candidate_rank=rank,
            )
        )
        output.append(candidate)
    output.sort(
        key=lambda item: (
            float(item.get("lane_pair_stop_proposal_max", 0.0)),
            float(item.get("lane_pair_rank_score", 0.0)),
            float(item.get("lane_pair_lane_endpoint_centerline_mean", 0.0)),
            float(item.get("lane_pair_length", _line_length(item))),
        ),
        reverse=True,
    )
    return output


def _candidate_row(candidate: dict[str, Any], *, batch_index: int, sample_index: int, meta: dict[str, Any]) -> dict[str, Any]:
    points = np.asarray(candidate.get("points_xy", []), dtype=np.float32).reshape(-1, 2)
    row = {
        "batch_index": int(batch_index),
        "sample_index": int(sample_index),
        "sample_id": str(meta.get("sample_id", "")),
        "dataset_key": str(meta.get("dataset_key", "")),
        "image_path": str(meta.get("image_path", "")),
        "candidate_points_json": json.dumps([[float(x), float(y)] for x, y in points.tolist()], separators=(",", ":")),
        "score": float(candidate.get("score", 0.0)),
        "length": float(_line_length(candidate)),
        "lane_pair_left_rank": int(candidate.get("lane_pair_left_rank", 0)),
        "lane_pair_right_rank": int(candidate.get("lane_pair_right_rank", 0)),
        "lane_pair_fraction_index": int(candidate.get("lane_pair_fraction_index", 0)),
        "lane_pair_y_fraction": float(candidate.get("lane_pair_y_fraction", 0.0)),
        "nearest_gt_distance": float(candidate.get("nearest_gt_distance", float("inf"))),
        "nearest_gt_angle_error": float(candidate.get("nearest_gt_angle_error", 180.0)),
        "nearest_gt_index": int(candidate.get("nearest_gt_index", -1)),
        "is_oracle_positive": int(bool(candidate.get("is_oracle_positive", False))),
        "nearest_current_distance": float(candidate.get("lane_pair_nearest_current_distance_norm", 0.0)),
    }
    for name in LANE_PAIR_FEATURES:
        row[name] = float(candidate.get(name, 0.0))
    return row


def _collect_records(
    *,
    loader: Any,
    evaluator: Any,
    postprocess_config: Any,
    max_batches: int,
    y_fractions: tuple[float, ...],
    max_lanes: int,
    min_overlap_px: float,
    min_gap_px: float,
    max_gap_frac: float,
    dense_anchor_count: int,
    dense_anchor_stride_px: float,
    max_candidates: int,
    split_name: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch_index, batch in enumerate(loader, start=1):
            if batch_index > int(max_batches):
                break
            if batch_index == 1 or batch_index % 20 == 0:
                print(f"[stopline_lane_pair] collect {split_name} batch {batch_index}/{max_batches}", flush=True)
            raw_batch = raw_batch_for_metrics(batch)
            if raw_batch is None:
                raise ValueError("lane-pair stop-line probe requires raw batches for metrics")
            gt_samples = _extract_gt_samples(raw_batch)
            encoded = evaluator.prepare_batch(batch)
            outputs = _detach_to_cpu(evaluator.forward_encoded_batch(encoded))
            meta_rows = _detach_to_cpu(encoded["meta"])
            baseline_predictions = postprocess_pv26_batch(outputs, meta_rows, config=postprocess_config)
            for sample_index, (meta, baseline_prediction, gt_sample) in enumerate(
                zip(meta_rows, baseline_predictions, gt_samples)
            ):
                stop_mask_probs = _as_2d_array(_sample_tensor(outputs, "stop_line_mask_logits", sample_index), sigmoid=True)
                stop_center_probs = _as_2d_array(
                    _sample_tensor(outputs, "stop_line_center_logits", sample_index),
                    sigmoid=True,
                )
                stop_selector_probs = _as_2d_array(
                    _sample_tensor(outputs, "stop_line_selector_map_logits", sample_index),
                    sigmoid=True,
                )
                lane_centerline_probs = _as_2d_array(
                    _sample_tensor(outputs, "lane_seg_centerline_logits", sample_index),
                    sigmoid=True,
                )
                lane_support_probs = _as_2d_array(
                    _sample_tensor(outputs, "lane_seg_support_logits", sample_index),
                    sigmoid=True,
                )
                candidates = _build_lane_pair_candidates(
                    meta=meta,
                    baseline_prediction=baseline_prediction,
                    gt_stop_lines=list(gt_sample.get("stop_lines", [])),
                    stop_mask_probs=stop_mask_probs,
                    stop_center_probs=stop_center_probs,
                    stop_selector_probs=stop_selector_probs,
                    lane_centerline_probs=lane_centerline_probs,
                    lane_support_probs=lane_support_probs,
                    y_fractions=y_fractions,
                    max_lanes=int(max_lanes),
                    min_overlap_px=float(min_overlap_px),
                    min_gap_px=float(min_gap_px),
                    max_gap_frac=float(max_gap_frac),
                    dense_anchor_count=int(dense_anchor_count),
                    dense_anchor_stride_px=float(dense_anchor_stride_px),
                    max_candidates=int(max_candidates),
                )
                candidate_rows = [
                    _candidate_row(candidate, batch_index=batch_index, sample_index=sample_index, meta=meta)
                    for candidate in candidates
                ]
                for row in candidate_rows:
                    row["split"] = str(split_name)
                rows.extend(candidate_rows)
                records.append(
                    {
                        "batch_index": int(batch_index),
                        "sample_index": int(sample_index),
                        "sample_id": str(meta.get("sample_id", "")),
                        "raw_batch": _slice_raw_batch_sample(raw_batch, sample_index),
                        "baseline_prediction": dict(baseline_prediction),
                        "gt_stop_lines": list(gt_sample.get("stop_lines", [])),
                        "candidates": candidates,
                        "candidate_feature_rows": candidate_rows,
                    }
                )
                sample_rows.append(
                    {
                        "split": str(split_name),
                        "batch_index": int(batch_index),
                        "sample_index": int(sample_index),
                        "sample_id": str(meta.get("sample_id", "")),
                        "dataset_key": str(meta.get("dataset_key", "")),
                        "baseline_stopline_count": int(len(list(baseline_prediction.get("stop_lines", [])))),
                        "lane_count": int(len(list(baseline_prediction.get("lanes", [])))),
                        "gt_stopline_count": int(len(list(gt_sample.get("stop_lines", [])))),
                        "lane_pair_candidate_count": int(len(candidates)),
                        "lane_pair_oracle_positive_count": int(
                            sum(1 for candidate in candidates if bool(candidate.get("is_oracle_positive", False)))
                        ),
                    }
                )
    return records, rows, sample_rows


def _feature_matrix(records: list[dict[str, Any]], *, top_k: int) -> tuple[np.ndarray, np.ndarray]:
    features: list[list[float]] = []
    labels: list[float] = []
    for record in records:
        for rank, candidate in enumerate(record.get("candidates", []), start=1):
            if rank > int(top_k):
                continue
            features.append([float(candidate.get(name, 0.0)) for name in LANE_PAIR_FEATURES])
            labels.append(float(bool(candidate.get("is_oracle_positive", False))))
    if not features:
        return np.zeros((0, len(LANE_PAIR_FEATURES)), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    return np.asarray(features, dtype=np.float32), np.asarray(labels, dtype=np.float32)


def _attach_scores(records: list[dict[str, Any]], scores: np.ndarray, *, top_k: int) -> None:
    index = 0
    for record in records:
        for rank, (candidate, row) in enumerate(
            zip(record.get("candidates", []), record.get("candidate_feature_rows", [])),
            start=1,
        ):
            if rank > int(top_k):
                continue
            score = float(scores[index])
            candidate[SCORE_KEY] = score
            row[SCORE_KEY] = score
            index += 1
    if index != int(scores.shape[0]):
        raise ValueError(f"score length mismatch: attached {index}, got {int(scores.shape[0])}")


def _score_records(
    *,
    train_records: list[dict[str, Any]],
    val_records: list[dict[str, Any]],
    top_k: int,
    epochs: int,
    lr: float,
) -> dict[str, Any]:
    train_x, train_y = _feature_matrix(train_records, top_k=int(top_k))
    val_x, val_y = _feature_matrix(val_records, top_k=int(top_k))
    if train_x.shape[0] == 0 or val_x.shape[0] == 0:
        raise ValueError("lane-pair verifier requires non-empty train and validation candidates")
    combined_x = np.concatenate([train_x, val_x], axis=0)
    combined_std, mean, std = _standardize_from_train(train_x.astype(np.float64), combined_x.astype(np.float64))
    train_std = combined_std[: train_x.shape[0]].astype(np.float32)
    val_std = combined_std[train_x.shape[0] :].astype(np.float32)
    model = _fit_raw_patch_mlp(train_std, train_y.astype(np.float32), epochs=int(epochs), lr=float(lr))
    train_scores = _predict_raw_patch_mlp(model, train_std)
    val_scores = _predict_raw_patch_mlp(model, val_std)
    _attach_scores(train_records, train_scores, top_k=int(top_k))
    _attach_scores(val_records, val_scores, top_k=int(top_k))
    return {
        "train_candidate_count": int(train_x.shape[0]),
        "train_positive_count": int(train_y.sum()),
        "val_candidate_count": int(val_x.shape[0]),
        "val_positive_count": int(val_y.sum()),
        "feature_dim": int(train_x.shape[1]),
        "mean_dim": int(mean.shape[0]),
        "std_min": float(std.min()) if std.size else 0.0,
    }


def _select_lane_pair_stop_lines(
    candidates: list[dict[str, Any]],
    *,
    score_key: str,
    threshold: float,
    top_k: int,
    max_components: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for rank, candidate in enumerate(candidates, start=1):
        if rank > int(top_k):
            continue
        if float(candidate.get(score_key, 0.0)) < float(threshold):
            continue
        selected.append(candidate)
    selected.sort(
        key=lambda item: (
            float(item.get(score_key, 0.0)),
            float(item.get("lane_pair_stop_proposal_max", 0.0)),
            float(item.get("lane_pair_rank_score", 0.0)),
            float(item.get("lane_pair_length", 0.0)),
        ),
        reverse=True,
    )
    predictions = [
        _copy_stop_line(candidate, score=float(candidate.get(score_key, 0.0)), source="lane_pair")
        for candidate in selected
    ]
    predictions.sort(key=_stopline_prediction_sort_key, reverse=True)
    predictions = _dedupe_stop_line_predictions(predictions)
    return predictions[: max(1, int(max_components))]


def _metrics_row(
    records: list[dict[str, Any]],
    *,
    name: str,
    split: str,
    score_key: str = "",
    threshold: float = 0.0,
    top_k: int = 0,
    max_components: int = 2,
    union_baseline: bool = False,
) -> dict[str, Any]:
    predictions: list[dict[str, Any]] = []
    raw_batches: list[dict[str, Any]] = []
    for record in records:
        raw_batches.append(record["raw_batch"])
        if score_key:
            stop_lines = _select_lane_pair_stop_lines(
                list(record.get("candidates", [])),
                score_key=score_key,
                threshold=float(threshold),
                top_k=int(top_k),
                max_components=int(max_components),
            )
            if bool(union_baseline):
                stop_lines = list(record["baseline_prediction"].get("stop_lines", [])) + stop_lines
                stop_lines.sort(key=_stopline_prediction_sort_key, reverse=True)
                stop_lines = _dedupe_stop_line_predictions(stop_lines)[: max(1, int(max_components))]
            predictions.append({**record["baseline_prediction"], "stop_lines": stop_lines})
        else:
            predictions.append(dict(record["baseline_prediction"]))
    merged_raw = _merge_raw_batches(raw_batches)
    metrics = augment_lane_family_metrics(summarize_pv26_metrics(predictions, merged_raw))
    prediction_count = sum(len(sample.get("stop_lines", [])) for sample in predictions)
    row = _row_from_metrics(name, metrics, prediction_count=prediction_count, stats={})
    row.update(
        {
            "split": str(split),
            "score_key": str(score_key),
            "threshold": "" if not score_key else float(threshold),
            "top_k": "" if not score_key else int(top_k),
            "union_baseline": int(bool(union_baseline)),
            "sample_count": int(len(records)),
        }
    )
    return row


def _best_threshold(
    records: list[dict[str, Any]],
    *,
    score_key: str,
    top_k: int,
    max_components: int,
    grid_size: int,
) -> float:
    best_threshold = 0.5
    best_key: tuple[float, float, float] = (-1.0, -1.0, 0.0)
    for threshold in np.linspace(0.0, 1.0, max(2, int(grid_size))).tolist():
        row = _metrics_row(
            records,
            name="threshold_search",
            split="train",
            score_key=score_key,
            threshold=float(threshold),
            top_k=int(top_k),
            max_components=int(max_components),
        )
        key = (
            float(row.get("stop_line_f1", 0.0)),
            float(row.get("stop_line_tp", 0.0)),
            -float(row.get("stop_line_fp", 0.0)),
        )
        if key > best_key:
            best_key = key
            best_threshold = float(threshold)
    return float(best_threshold)


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    if not str(args.checkpoint).strip():
        from tools.probe_pv26_lane_flip_tta import DEFAULT_CHECKPOINT

        args.checkpoint = str(DEFAULT_CHECKPOINT)
    if not str(args.source_run).strip():
        from tools.probe_pv26_lane_flip_tta import SOURCE_RUN

        args.source_run = str(SOURCE_RUN)
    y_fractions = _parse_y_fractions(str(args.lane_pair_y_fractions))
    scenario, scenario_path, options, phase, train_config = _build_scenario(args)
    train_config = replace(
        train_config,
        encode_train_batches_in_loader=False,
        encode_val_batches_in_loader=False,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        prefetch_factor=None,
    )
    train_cli._configure_torch_multiprocessing()
    dataset = train_cli.PV26CanonicalDataset(
        train_cli._existing_dataset_roots(scenario),
        train_augmentation=False,
        progress_callback=lambda message: print(f"[stopline_lane_pair] {message}", flush=True),
    )
    train_loader, val_loader = train_cli._build_phase_train_loaders(dataset, train_config=train_config, phase=phase)
    if train_loader is None or val_loader is None:
        raise ValueError("lane-pair stop-line probe requires train and validation loaders")
    _advance_validation_sampler(val_loader, validation_epoch=int(args.validation_epoch))
    trainer = train_cli._build_phase_trainer(phase, train_config)
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    trainer.load_model_weights(checkpoint, map_location=train_config.device)
    evaluator = trainer.build_evaluator()
    evaluator.adapter.raw_model.eval()
    evaluator.heads.eval()
    postprocess_config = _postprocess_override_config(args, trainer) or train_cli._build_postprocess_config(train_config)
    train_records, train_candidate_rows, train_sample_rows = _collect_records(
        loader=train_loader,
        evaluator=evaluator,
        postprocess_config=postprocess_config,
        max_batches=int(args.train_record_batches),
        y_fractions=y_fractions,
        max_lanes=int(args.max_lane_pair_lanes),
        min_overlap_px=float(args.lane_pair_min_overlap_px),
        min_gap_px=float(args.lane_pair_min_gap_px),
        max_gap_frac=float(args.lane_pair_max_gap_frac),
        dense_anchor_count=int(args.lane_pair_dense_anchor_count),
        dense_anchor_stride_px=float(args.lane_pair_dense_anchor_stride_px),
        max_candidates=int(args.max_lane_pair_candidates),
        split_name="train",
    )
    val_records, val_candidate_rows, val_sample_rows = _collect_records(
        loader=val_loader,
        evaluator=evaluator,
        postprocess_config=postprocess_config,
        max_batches=int(args.max_val_batches),
        y_fractions=y_fractions,
        max_lanes=int(args.max_lane_pair_lanes),
        min_overlap_px=float(args.lane_pair_min_overlap_px),
        min_gap_px=float(args.lane_pair_min_gap_px),
        max_gap_frac=float(args.lane_pair_max_gap_frac),
        dense_anchor_count=int(args.lane_pair_dense_anchor_count),
        dense_anchor_stride_px=float(args.lane_pair_dense_anchor_stride_px),
        max_candidates=int(args.max_lane_pair_candidates),
        split_name="val",
    )
    score_summary = _score_records(
        train_records=train_records,
        val_records=val_records,
        top_k=int(args.lane_pair_top_k),
        epochs=int(args.verifier_epochs),
        lr=float(args.verifier_lr),
    )
    threshold = _best_threshold(
        train_records,
        score_key=SCORE_KEY,
        top_k=int(args.lane_pair_top_k),
        max_components=int(args.max_components),
        grid_size=int(args.threshold_grid),
    )
    rows = [
        _metrics_row(train_records, name="baseline", split="train"),
        _metrics_row(val_records, name="baseline", split="val"),
        _metrics_row(
            train_records,
            name="lane_pair_mlp",
            split="train",
            score_key=SCORE_KEY,
            threshold=threshold,
            top_k=int(args.lane_pair_top_k),
            max_components=int(args.max_components),
        ),
        _metrics_row(
            val_records,
            name="lane_pair_mlp",
            split="val",
            score_key=SCORE_KEY,
            threshold=threshold,
            top_k=int(args.lane_pair_top_k),
            max_components=int(args.max_components),
        ),
        _metrics_row(
            train_records,
            name="baseline_plus_lane_pair_mlp",
            split="train",
            score_key=SCORE_KEY,
            threshold=threshold,
            top_k=int(args.lane_pair_top_k),
            max_components=int(args.max_components),
            union_baseline=True,
        ),
        _metrics_row(
            val_records,
            name="baseline_plus_lane_pair_mlp",
            split="val",
            score_key=SCORE_KEY,
            threshold=threshold,
            top_k=int(args.lane_pair_top_k),
            max_components=int(args.max_components),
            union_baseline=True,
        ),
        _metrics_row(
            val_records,
            name="oracle_lane_pair",
            split="val",
            score_key="is_oracle_positive",
            threshold=0.5,
            top_k=int(args.lane_pair_top_k),
            max_components=int(args.max_components),
        ),
        _metrics_row(
            val_records,
            name="baseline_plus_oracle_lane_pair",
            split="val",
            score_key="is_oracle_positive",
            threshold=0.5,
            top_k=int(args.lane_pair_top_k),
            max_components=int(args.max_components),
            union_baseline=True,
        ),
    ]
    summary = {
        "checkpoint": str(checkpoint),
        "source_run": str(Path(args.source_run).expanduser().resolve()),
        "scenario_path": str(scenario_path),
        "lane60_experiment": str(args.lane60_experiment),
        "phase_index": int(tuple(options["selected_phase_indices"])[0]),
        "train_record_batches": int(args.train_record_batches),
        "max_val_batches": int(args.max_val_batches),
        "validation_epoch": int(args.validation_epoch),
        "lane_pair_y_fractions": [float(value) for value in y_fractions],
        "max_lane_pair_lanes": int(args.max_lane_pair_lanes),
        "lane_pair_min_overlap_px": float(args.lane_pair_min_overlap_px),
        "lane_pair_min_gap_px": float(args.lane_pair_min_gap_px),
        "lane_pair_max_gap_frac": float(args.lane_pair_max_gap_frac),
        "lane_pair_dense_anchor_count": int(args.lane_pair_dense_anchor_count),
        "lane_pair_dense_anchor_stride_px": float(args.lane_pair_dense_anchor_stride_px),
        "lane_pair_top_k": int(args.lane_pair_top_k),
        "max_lane_pair_candidates": int(args.max_lane_pair_candidates),
        "max_components": int(args.max_components),
        "threshold": float(threshold),
        "score_summary": score_summary,
        "train_sample_count": int(len(train_records)),
        "val_sample_count": int(len(val_records)),
        "train_lane_pair_candidate_count": int(len(train_candidate_rows)),
        "val_lane_pair_candidate_count": int(len(val_candidate_rows)),
        "train_lane_pair_oracle_positive_count": int(
            sum(int(row.get("is_oracle_positive", 0)) for row in train_candidate_rows)
        ),
        "val_lane_pair_oracle_positive_count": int(
            sum(int(row.get("is_oracle_positive", 0)) for row in val_candidate_rows)
        ),
        "rows": rows,
        "interpretation": (
            "Lane-pair topology stop-line candidate probe. Runtime candidates are generated from "
            "predicted lane polyline pairs and current-frame dense maps; GT is used for train labels, "
            "oracle diagnostics, and final metrics only. This tests candidate generation, not only "
            "source-router feature re-ranking."
        ),
    }
    return {
        "summary": summary,
        "rows": rows,
        "train_candidate_rows": train_candidate_rows,
        "val_candidate_rows": val_candidate_rows,
        "sample_rows": [*train_sample_rows, *val_sample_rows],
    }


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = run_probe(args)
    _write_csv(output_dir / "lane_pair_variants.csv", payload["rows"])
    _write_candidate_features_csv(output_dir / "train_candidate_features.csv", payload["train_candidate_rows"])
    _write_candidate_features_csv(output_dir / "val_candidate_features.csv", payload["val_candidate_rows"])
    _write_csv(output_dir / "lane_pair_samples.csv", payload["sample_rows"])
    (output_dir / "summary.json").write_text(
        json.dumps(payload["summary"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
