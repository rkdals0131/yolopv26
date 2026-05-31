from __future__ import annotations

import argparse
from copy import deepcopy
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

from model.data.transform import inverse_transform_points, transform_from_meta, transform_points
from model.engine import raw_batch_for_metrics
from model.engine._trainer_epochs import _merge_raw_batches
from model.engine.batch import augment_lane_family_metrics
from model.engine.lane_segfirst_vectorizer import lane_segfirst_prediction_maps
from model.engine.metrics import summarize_pv26_metrics
from model.engine.postprocess import postprocess_pv26_batch
from tools.evaluate_pv26_lane60_checkpoint import _advance_validation_sampler, _postprocess_override_config
from tools.probe_pv26_lane_flip_tta import _merge_lane_dense_predictions, _unflip_lane_dense_outputs
from tools.probe_pv26_lane_instance_evidence import (
    _as_channel,
    _max_low_run_fraction,
    _polyline_length,
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
            "Read-only smoke for a fixed no-GT lane repairability ranker. "
            "It scores decoded lane predictions, translates only the top-ranked "
            "budget to the local centerline ridge, and compares actual metrics."
        )
    )
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--source-run", default=str(SOURCE_RUN))
    parser.add_argument("--ranker-parameters", required=True)
    parser.add_argument("--ranker-label", default="repairable_le120_any_center")
    parser.add_argument("--lane60-experiment", default="core_centerline_refine_row_scan_tangent_link")
    parser.add_argument("--preset", default="default")
    parser.add_argument("--max-val-batches", type=int, default=4)
    parser.add_argument("--validation-epoch", type=int, default=2)
    parser.add_argument("--train-batches", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--dataset-root",
        default="",
        help="Optional canonical dataset root. If omitted, infer from source-run/scenario paths.",
    )
    parser.add_argument(
        "--lane-flip-variant",
        choices=("baseline", "flip_centerline_avg", "flip_centerline_avg_lane_cross_comp050"),
        default="flip_centerline_avg",
    )
    parser.add_argument("--repair-topk", type=int, default=0, help="Global repair budget. 0 uses val-size-scaled top500/2048.")
    parser.add_argument(
        "--repair-mode",
        choices=(
            "translate_x",
            "local_2d_snap",
            "affine_2d_snap",
            "component_row_project",
            "row_profile_softargmax",
            "ridge_path_dp",
        ),
        default="translate_x",
    )
    parser.add_argument("--translation-radius", type=int, default=4)
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
    parser.add_argument("--crosswalk-polygon-mode", choices=("rect", "hull"), default="hull")
    parser.add_argument("--output-dir", default="")
    return parser.parse_args()


def _f1(tp: int, fp: int, fn: int) -> float:
    denom = 2 * int(tp) + int(fp) + int(fn)
    return 0.0 if denom <= 0 else float(2 * int(tp) / denom)


def _metric(metrics: dict[str, Any], task: str, key: str) -> float:
    payload = metrics.get(task, {}) if isinstance(metrics.get(task), dict) else {}
    value = payload.get(key, 0.0)
    return float(value) if isinstance(value, (int, float)) else 0.0


def _raw_points_to_map(points_xy: list[list[float]], meta: dict[str, Any], map_hw: tuple[int, int]) -> np.ndarray:
    transform = transform_from_meta(meta)
    network_points = np.asarray(transform_points(points_xy, transform), dtype=np.float32).reshape(-1, 2)
    if network_points.size == 0:
        return network_points.reshape(0, 2)
    map_h, map_w = int(map_hw[0]), int(map_hw[1])
    net_h, net_w = int(transform.network_hw[0]), int(transform.network_hw[1])
    network_points[:, 0] = np.clip(network_points[:, 0] * float(map_w) / max(float(net_w), 1.0), 0.0, float(map_w - 1))
    network_points[:, 1] = np.clip(network_points[:, 1] * float(map_h) / max(float(net_h), 1.0), 0.0, float(map_h - 1))
    return network_points


def _map_points_to_raw(map_points: np.ndarray, meta: dict[str, Any], map_hw: tuple[int, int]) -> list[list[float]]:
    transform = transform_from_meta(meta)
    points = np.asarray(map_points, dtype=np.float32).reshape(-1, 2).copy()
    if points.size == 0:
        return []
    map_h, map_w = int(map_hw[0]), int(map_hw[1])
    net_h, net_w = int(transform.network_hw[0]), int(transform.network_hw[1])
    points[:, 0] = np.clip(points[:, 0] * float(net_w) / max(float(map_w), 1.0), 0.0, float(net_w - 1))
    points[:, 1] = np.clip(points[:, 1] * float(net_h) / max(float(map_h), 1.0), 0.0, float(net_h - 1))
    return [[float(x), float(y)] for x, y in inverse_transform_points(points.tolist(), transform)]


def _translated_centerline_score(map_points: np.ndarray, centerline: np.ndarray, *, dx: int) -> float:
    values = np.nan_to_num(np.asarray(centerline, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if values.ndim != 2 or map_points.shape[0] == 0:
        return 0.0
    h, w = int(values.shape[0]), int(values.shape[1])
    scores: list[float] = []
    for x_value, y_value in map_points:
        x = min(max(int(round(float(x_value) + int(dx))), 0), w - 1)
        y = min(max(int(round(float(y_value))), 0), h - 1)
        scores.append(float(values[y, x]))
    return float(np.mean(scores)) if scores else 0.0


def translate_points_to_centerline(
    points_xy: list[list[float]],
    *,
    centerline: np.ndarray,
    meta: dict[str, Any],
    radius: int,
) -> tuple[list[list[float]], int]:
    map_hw = (int(centerline.shape[0]), int(centerline.shape[1]))
    map_points = _raw_points_to_map(points_xy, meta, map_hw)
    if map_points.shape[0] == 0:
        return [], 0
    search_radius = max(0, int(radius))
    candidate_offsets = list(range(-search_radius, search_radius + 1))
    best_dx = max(
        candidate_offsets,
        key=lambda dx: (
            _translated_centerline_score(map_points, centerline, dx=dx),
            -abs(int(dx)),
        ),
    )
    translated = map_points.copy()
    translated[:, 0] = np.clip(translated[:, 0] + float(best_dx), 0.0, float(map_hw[1] - 1))
    return _map_points_to_raw(translated, meta, map_hw), int(best_dx)


def _snap_map_points_to_local_centerline(
    map_points: np.ndarray,
    centerline: np.ndarray,
    *,
    radius: int,
) -> tuple[np.ndarray, dict[str, float]]:
    values = np.nan_to_num(np.asarray(centerline, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    points = np.asarray(map_points, dtype=np.float32).reshape(-1, 2)
    if values.ndim != 2 or points.shape[0] == 0:
        return points.copy(), {"moved_points": 0.0, "mean_move": 0.0, "max_move": 0.0}
    h, w = int(values.shape[0]), int(values.shape[1])
    search_radius = max(0, int(radius))
    snapped = points.copy()
    moves: list[float] = []
    for point_index, (x_value, y_value) in enumerate(points):
        x0 = min(max(int(round(float(x_value))), 0), w - 1)
        y0 = min(max(int(round(float(y_value))), 0), h - 1)
        best_x, best_y = x0, y0
        best_key = (float(values[y0, x0]), 0.0)
        for dy in range(-search_radius, search_radius + 1):
            y = y0 + int(dy)
            if y < 0 or y >= h:
                continue
            for dx in range(-search_radius, search_radius + 1):
                x = x0 + int(dx)
                if x < 0 or x >= w:
                    continue
                distance_sq = float(dx * dx + dy * dy)
                key = (float(values[y, x]), -distance_sq)
                if key > best_key:
                    best_key = key
                    best_x, best_y = x, y
        snapped[point_index, 0] = float(best_x)
        snapped[point_index, 1] = float(best_y)
        moves.append(float(np.linalg.norm(snapped[point_index] - points[point_index])))
    move_values = np.asarray(moves, dtype=np.float32)
    return snapped, {
        "moved_points": float((move_values > 1.0e-3).sum()),
        "mean_move": float(move_values.mean()) if move_values.size else 0.0,
        "max_move": float(move_values.max()) if move_values.size else 0.0,
    }


def snap_points_to_local_centerline(
    points_xy: list[list[float]],
    *,
    centerline: np.ndarray,
    meta: dict[str, Any],
    radius: int,
) -> tuple[list[list[float]], dict[str, float]]:
    map_hw = (int(centerline.shape[0]), int(centerline.shape[1]))
    map_points = _raw_points_to_map(points_xy, meta, map_hw)
    snapped, stats = _snap_map_points_to_local_centerline(map_points, centerline, radius=radius)
    return _map_points_to_raw(snapped, meta, map_hw), stats


def _fit_affine_map_points_to_targets(
    map_points: np.ndarray,
    target_points: np.ndarray,
    *,
    map_hw: tuple[int, int],
) -> tuple[np.ndarray, dict[str, float]]:
    points = np.asarray(map_points, dtype=np.float32).reshape(-1, 2)
    targets = np.asarray(target_points, dtype=np.float32).reshape(-1, 2)
    if points.shape[0] < 3 or targets.shape != points.shape:
        return points.copy(), {"moved_points": 0.0, "mean_move": 0.0, "max_move": 0.0, "affine_residual": 0.0}
    design = np.concatenate([points, np.ones((points.shape[0], 1), dtype=np.float32)], axis=1)
    coeff_x, _, _, _ = np.linalg.lstsq(design, targets[:, 0], rcond=None)
    coeff_y, _, _, _ = np.linalg.lstsq(design, targets[:, 1], rcond=None)
    repaired = np.stack([design @ coeff_x, design @ coeff_y], axis=1).astype(np.float32)
    repaired[:, 0] = np.clip(repaired[:, 0], 0.0, float(int(map_hw[1]) - 1))
    repaired[:, 1] = np.clip(repaired[:, 1], 0.0, float(int(map_hw[0]) - 1))
    moves = np.linalg.norm(repaired - points, axis=1).astype(np.float32)
    residual = np.linalg.norm(repaired - targets, axis=1).astype(np.float32)
    return repaired, {
        "moved_points": float((moves > 1.0e-3).sum()),
        "mean_move": float(moves.mean()) if moves.size else 0.0,
        "max_move": float(moves.max()) if moves.size else 0.0,
        "affine_residual": float(residual.mean()) if residual.size else 0.0,
    }


def affine_snap_points_to_local_centerline(
    points_xy: list[list[float]],
    *,
    centerline: np.ndarray,
    meta: dict[str, Any],
    radius: int,
) -> tuple[list[list[float]], dict[str, float]]:
    map_hw = (int(centerline.shape[0]), int(centerline.shape[1]))
    map_points = _raw_points_to_map(points_xy, meta, map_hw)
    snapped, snap_stats = _snap_map_points_to_local_centerline(map_points, centerline, radius=radius)
    repaired, affine_stats = _fit_affine_map_points_to_targets(map_points, snapped, map_hw=map_hw)
    affine_stats["snap_moved_points"] = float(snap_stats.get("moved_points", 0.0))
    affine_stats["snap_mean_move"] = float(snap_stats.get("mean_move", 0.0))
    affine_stats["snap_max_move"] = float(snap_stats.get("max_move", 0.0))
    return _map_points_to_raw(repaired, meta, map_hw), affine_stats


def _centerline_components(centerline: np.ndarray, *, threshold: float = 0.5) -> list[np.ndarray]:
    values = np.nan_to_num(np.asarray(centerline, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if values.ndim != 2:
        return []
    active = values >= float(threshold)
    visited = np.zeros(active.shape, dtype=bool)
    components: list[np.ndarray] = []
    height, width = int(active.shape[0]), int(active.shape[1])
    for row in range(height):
        for col in range(width):
            if not bool(active[row, col]) or bool(visited[row, col]):
                continue
            stack = [(row, col)]
            visited[row, col] = True
            pixels: list[tuple[float, float]] = []
            while stack:
                y, x = stack.pop()
                pixels.append((float(x), float(y)))
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dx == 0 and dy == 0:
                            continue
                        ny, nx = y + dy, x + dx
                        if ny < 0 or ny >= height or nx < 0 or nx >= width:
                            continue
                        if bool(active[ny, nx]) and not bool(visited[ny, nx]):
                            visited[ny, nx] = True
                            stack.append((ny, nx))
            components.append(np.asarray(pixels, dtype=np.float32))
    return components


def _nearest_component_distances(points: np.ndarray, component: np.ndarray) -> np.ndarray:
    if points.shape[0] == 0 or component.shape[0] == 0:
        return np.asarray([], dtype=np.float32)
    distances: list[float] = []
    for point in points:
        delta = component - point.reshape(1, 2)
        distances.append(float(np.sqrt(np.square(delta).sum(axis=1)).min()))
    return np.asarray(distances, dtype=np.float32)


def _select_centerline_component(map_points: np.ndarray, centerline: np.ndarray) -> tuple[np.ndarray | None, dict[str, float]]:
    components = _centerline_components(centerline, threshold=0.5)
    points = np.asarray(map_points, dtype=np.float32).reshape(-1, 2)
    if not components or points.shape[0] == 0:
        return None, {"component_count": float(len(components)), "component_size": 0.0, "component_median_distance": 0.0}
    best_component: np.ndarray | None = None
    best_stats = {"component_count": float(len(components)), "component_size": 0.0, "component_median_distance": 0.0}
    best_key = (float("inf"), 0.0)
    for component in components:
        distances = _nearest_component_distances(points, component)
        if distances.size == 0:
            continue
        median_distance = float(np.median(distances))
        key = (median_distance, -float(component.shape[0]))
        if key < best_key:
            best_key = key
            best_component = component
            best_stats = {
                "component_count": float(len(components)),
                "component_size": float(component.shape[0]),
                "component_median_distance": median_distance,
            }
    return best_component, best_stats


def _project_points_to_component_rows(
    map_points: np.ndarray,
    component: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    points = np.asarray(map_points, dtype=np.float32).reshape(-1, 2)
    pixels = np.asarray(component, dtype=np.float32).reshape(-1, 2)
    if points.shape[0] == 0 or pixels.shape[0] == 0:
        return points.copy(), {"moved_points": 0.0, "mean_move": 0.0, "max_move": 0.0}
    repaired = points.copy()
    moves: list[float] = []
    for point_index, point in enumerate(points):
        row_delta = np.abs(pixels[:, 1] - float(point[1]))
        min_row_delta = float(row_delta.min())
        same_row = pixels[row_delta <= min_row_delta + 1.0e-6]
        if same_row.shape[0] == 0:
            same_row = pixels
        col_delta = np.abs(same_row[:, 0] - float(point[0]))
        target = same_row[int(col_delta.argmin())]
        repaired[point_index] = target
        moves.append(float(np.linalg.norm(repaired[point_index] - points[point_index])))
    move_values = np.asarray(moves, dtype=np.float32)
    return repaired, {
        "moved_points": float((move_values > 1.0e-3).sum()),
        "mean_move": float(move_values.mean()) if move_values.size else 0.0,
        "max_move": float(move_values.max()) if move_values.size else 0.0,
    }


def _project_points_to_row_profile(
    map_points: np.ndarray,
    centerline: np.ndarray,
    *,
    radius: int,
) -> tuple[np.ndarray, dict[str, float]]:
    values = np.nan_to_num(np.asarray(centerline, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    points = np.asarray(map_points, dtype=np.float32).reshape(-1, 2)
    if values.ndim != 2 or points.shape[0] == 0:
        return points.copy(), {"moved_points": 0.0, "mean_move": 0.0, "max_move": 0.0, "mean_profile_mass": 0.0}
    height, width = int(values.shape[0]), int(values.shape[1])
    search_radius = max(1, int(radius))
    repaired = points.copy()
    moves: list[float] = []
    masses: list[float] = []
    for point_index, (x_value, y_value) in enumerate(points):
        row = min(max(int(round(float(y_value))), 0), height - 1)
        center_x = min(max(int(round(float(x_value))), 0), width - 1)
        start = max(0, center_x - search_radius)
        end = min(width, center_x + search_radius + 1)
        xs = np.arange(start, end, dtype=np.float32)
        profile = values[row, start:end].astype(np.float32)
        weights = np.clip(profile - float(profile.min()), 0.0, None)
        if float(weights.sum()) <= 1.0e-6:
            weights = np.clip(profile, 0.0, None)
        mass = float(weights.sum())
        if mass > 1.0e-6:
            repaired[point_index, 0] = float(np.sum(xs * weights) / mass)
        repaired[point_index, 1] = float(row)
        moves.append(float(np.linalg.norm(repaired[point_index] - points[point_index])))
        masses.append(mass)
    move_values = np.asarray(moves, dtype=np.float32)
    return repaired, {
        "moved_points": float((move_values > 1.0e-3).sum()),
        "mean_move": float(move_values.mean()) if move_values.size else 0.0,
        "max_move": float(move_values.max()) if move_values.size else 0.0,
        "mean_profile_mass": float(np.asarray(masses, dtype=np.float32).mean()) if masses else 0.0,
    }


def row_profile_project_points_to_centerline(
    points_xy: list[list[float]],
    *,
    centerline: np.ndarray,
    meta: dict[str, Any],
    radius: int,
) -> tuple[list[list[float]], dict[str, float]]:
    map_hw = (int(centerline.shape[0]), int(centerline.shape[1]))
    map_points = _raw_points_to_map(points_xy, meta, map_hw)
    repaired, repair_stats = _project_points_to_row_profile(map_points, centerline, radius=radius)
    return _map_points_to_raw(repaired, meta, map_hw), repair_stats


def _project_points_to_ridge_path(
    map_points: np.ndarray,
    centerline: np.ndarray,
    support: np.ndarray,
    *,
    radius: int,
    smoothness_weight: float = 0.35,
) -> tuple[np.ndarray, dict[str, float]]:
    center_values = np.nan_to_num(np.asarray(centerline, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    support_values = np.nan_to_num(np.asarray(support, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    points = np.asarray(map_points, dtype=np.float32).reshape(-1, 2)
    if center_values.ndim != 2 or support_values.shape != center_values.shape or points.shape[0] == 0:
        return points.copy(), {
            "moved_points": 0.0,
            "mean_move": 0.0,
            "max_move": 0.0,
            "mean_path_score": 0.0,
            "mean_center_gain": 0.0,
        }

    height, width = int(center_values.shape[0]), int(center_values.shape[1])
    search_radius = max(1, int(radius))
    candidate_xs: list[np.ndarray] = []
    unary_scores: list[np.ndarray] = []
    original_scores: list[float] = []
    for x_value, y_value in points:
        row = min(max(int(round(float(y_value))), 0), height - 1)
        center_x = min(max(int(round(float(x_value))), 0), width - 1)
        start = max(0, center_x - search_radius)
        end = min(width, center_x + search_radius + 1)
        xs = np.arange(start, end, dtype=np.float32)
        center_profile = center_values[row, start:end].astype(np.float32)
        support_profile = support_values[row, start:end].astype(np.float32)
        move_penalty = 0.015 * np.abs(xs - float(center_x)) / max(float(search_radius), 1.0)
        scores = center_profile + 0.25 * support_profile - move_penalty.astype(np.float32)
        candidate_xs.append(xs)
        unary_scores.append(scores.astype(np.float32))
        original_scores.append(float(center_values[row, center_x] + 0.25 * support_values[row, center_x]))

    dp_scores: list[np.ndarray] = [unary_scores[0].copy()]
    backpointers: list[np.ndarray] = [np.full(unary_scores[0].shape, -1, dtype=np.int32)]
    for point_index in range(1, len(points)):
        previous_xs = candidate_xs[point_index - 1]
        current_xs = candidate_xs[point_index]
        original_step = float(points[point_index, 0] - points[point_index - 1, 0])
        current_scores = np.full(current_xs.shape, -1.0e9, dtype=np.float32)
        current_back = np.zeros(current_xs.shape, dtype=np.int32)
        for current_index, current_x in enumerate(current_xs):
            continuity_error = np.abs((float(current_x) - previous_xs.astype(np.float32)) - original_step)
            transition = dp_scores[-1] - float(smoothness_weight) * continuity_error / max(float(search_radius), 1.0)
            best_previous = int(np.argmax(transition))
            current_scores[current_index] = float(unary_scores[point_index][current_index] + transition[best_previous])
            current_back[current_index] = best_previous
        dp_scores.append(current_scores)
        backpointers.append(current_back)

    selected_indices = [0] * len(points)
    selected_indices[-1] = int(np.argmax(dp_scores[-1]))
    for point_index in range(len(points) - 1, 0, -1):
        selected_indices[point_index - 1] = int(backpointers[point_index][selected_indices[point_index]])

    repaired = points.copy()
    path_scores: list[float] = []
    for point_index, selected_index in enumerate(selected_indices):
        x = float(candidate_xs[point_index][selected_index])
        repaired[point_index, 0] = x
        row = min(max(int(round(float(points[point_index, 1]))), 0), height - 1)
        col = min(max(int(round(x)), 0), width - 1)
        path_scores.append(float(center_values[row, col] + 0.25 * support_values[row, col]))

    moves = np.linalg.norm(repaired - points, axis=1).astype(np.float32)
    path_values = np.asarray(path_scores, dtype=np.float32)
    original_values = np.asarray(original_scores, dtype=np.float32)
    return repaired, {
        "moved_points": float((moves > 1.0e-3).sum()),
        "mean_move": float(moves.mean()) if moves.size else 0.0,
        "max_move": float(moves.max()) if moves.size else 0.0,
        "mean_path_score": float(path_values.mean()) if path_values.size else 0.0,
        "mean_center_gain": float((path_values - original_values).mean()) if path_values.size else 0.0,
    }


def ridge_path_project_points_to_dense_lane(
    points_xy: list[list[float]],
    *,
    centerline: np.ndarray,
    support: np.ndarray,
    meta: dict[str, Any],
    radius: int,
) -> tuple[list[list[float]], dict[str, float]]:
    map_hw = (int(centerline.shape[0]), int(centerline.shape[1]))
    map_points = _raw_points_to_map(points_xy, meta, map_hw)
    repaired, repair_stats = _project_points_to_ridge_path(
        map_points,
        centerline,
        support,
        radius=radius,
    )
    return _map_points_to_raw(repaired, meta, map_hw), repair_stats


def component_project_points_to_centerline(
    points_xy: list[list[float]],
    *,
    centerline: np.ndarray,
    meta: dict[str, Any],
) -> tuple[list[list[float]], dict[str, float]]:
    map_hw = (int(centerline.shape[0]), int(centerline.shape[1]))
    map_points = _raw_points_to_map(points_xy, meta, map_hw)
    component, component_stats = _select_centerline_component(map_points, centerline)
    if component is None:
        return _map_points_to_raw(map_points, meta, map_hw), {
            **component_stats,
            "moved_points": 0.0,
            "mean_move": 0.0,
            "max_move": 0.0,
        }
    repaired, repair_stats = _project_points_to_component_rows(map_points, component)
    return _map_points_to_raw(repaired, meta, map_hw), {**component_stats, **repair_stats}


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
            "pred_bbox_width": 0.0,
            "pred_bbox_height": 0.0,
            "pred_bbox_aspect": 0.0,
        }
    width = float(points[:, 0].max() - points[:, 0].min())
    height = float(points[:, 1].max() - points[:, 1].min())
    return {
        "pred_point_count": float(points.shape[0]),
        "pred_polyline_length": float(_polyline_length(list(pred_lane.get("points_xy", [])))),
        "pred_bbox_width": width,
        "pred_bbox_height": height,
        "pred_bbox_aspect": float(max(width, height) / max(min(width, height), 1.0e-6)),
    }


def _lane_distance(pred: dict[str, Any], other: dict[str, Any]) -> float:
    from model.engine.metrics import _mean_point_distance

    return float(_mean_point_distance(pred.get("points_xy", []), other.get("points_xy", []), 16))


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


def _safe_sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-min(value, 60.0))
        return 1.0 / (1.0 + z)
    z = math.exp(max(value, -60.0))
    return z / (1.0 + z)


def score_repairability(features: dict[str, float], model: dict[str, Any]) -> float:
    score = float(model.get("bias", 0.0))
    for feature, mean, scale, weight in zip(
        model.get("features", []),
        model.get("means", []),
        model.get("scales", []),
        model.get("weights", []),
    ):
        value = float(features.get(str(feature), 0.0))
        if not math.isfinite(value):
            value = float(mean)
        score += float(weight) * ((value - float(mean)) / max(float(scale), 1.0e-6))
    return _safe_sigmoid(score)


def _prediction_features(
    pred_index: int,
    pred_lanes: list[dict[str, Any]],
    *,
    maps: dict[str, torch.Tensor],
    meta: dict[str, Any],
) -> dict[str, float]:
    pred_lane = pred_lanes[pred_index]
    features = _prediction_shape_features(pred_lane)
    features.update(_track_map_evidence(list(pred_lane.get("points_xy", [])), maps=maps, meta=meta, prefix="pred"))
    features["nearest_other_pred_distance"] = _nearest_other_prediction_distance(pred_index, pred_lanes)
    features["sample_pred_lane_count"] = float(len(pred_lanes))
    return features


def _auto_repair_topk(sample_count: int, *, reference_topk: int = 500, reference_samples: int = 2048) -> int:
    return max(1, int(round(float(sample_count) * float(reference_topk) / float(reference_samples))))


def _metrics_row(metrics: dict[str, Any], *, name: str) -> dict[str, Any]:
    lane_tp = int(round(_metric(metrics, "lane", "tp")))
    lane_fp = int(round(_metric(metrics, "lane", "fp")))
    lane_fn = int(round(_metric(metrics, "lane", "fn")))
    return {
        "variant": name,
        "lane_f1": _metric(metrics, "lane", "f1"),
        "lane_tp": lane_tp,
        "lane_fp": lane_fp,
        "lane_fn": lane_fn,
        "lane_f1_from_counts": _f1(lane_tp, lane_fp, lane_fn),
        "stop_line_f1": _metric(metrics, "stop_line", "f1"),
        "crosswalk_f1": _metric(metrics, "crosswalk", "f1"),
    }


def main() -> int:
    args = parse_args()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    source_run = Path(args.source_run).expanduser().resolve()
    ranker_path = Path(args.ranker_parameters).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not source_run.is_dir():
        raise FileNotFoundError(source_run)
    if not ranker_path.is_file():
        raise FileNotFoundError(ranker_path)

    rankers = json.loads(ranker_path.read_text(encoding="utf-8"))
    if str(args.ranker_label) not in rankers:
        raise KeyError(f"ranker label not found: {args.ranker_label}")
    ranker = rankers[str(args.ranker_label)]

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
        else source_run / "analysis_exports" / f"lane_ranked_translate_repair_smoke_val{int(args.max_val_batches)}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    train_cli._configure_torch_multiprocessing()
    dataset = train_cli.PV26CanonicalDataset(
        train_cli._existing_dataset_roots(scenario),
        train_augmentation=False,
        progress_callback=lambda message: print(f"[lane_ranked_repair] {message}", flush=True),
    )
    _, val_loader = train_cli._build_phase_train_loaders(dataset, train_config=train_config, phase=phase)
    if val_loader is None:
        raise ValueError("ranked repair smoke requires validation batches")
    _advance_validation_sampler(val_loader, validation_epoch=int(args.validation_epoch))

    trainer = train_cli._build_phase_trainer(phase, train_config)
    trainer.load_model_weights(checkpoint, map_location=train_config.device)
    postprocess_config = _postprocess_override_config(args, trainer) or train_cli._build_postprocess_config(train_config)
    evaluator = trainer.build_evaluator()
    evaluator.adapter.raw_model.eval()
    evaluator.heads.eval()

    base_predictions_all: list[dict[str, Any]] = []
    repaired_predictions_all: list[dict[str, Any]] = []
    raw_batches: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    sample_count = 0

    with torch.no_grad():
        for batch_index, batch in enumerate(val_loader, start=1):
            if batch_index > int(args.max_val_batches):
                break
            print(f"[lane_ranked_repair] eval batch {batch_index}/{args.max_val_batches}", flush=True)
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
            batch_repaired = deepcopy(batch_predictions)
            raw_batches.append(raw_batch)
            base_offset = len(base_predictions_all)
            base_predictions_all.extend(batch_predictions)
            repaired_predictions_all.extend(batch_repaired)
            for sample_batch_index, (sample_pred, sample_meta) in enumerate(zip(batch_predictions, meta)):
                maps = lane_segfirst_prediction_maps(postprocess_predictions, batch_index=sample_batch_index)
                centerline = _as_channel(maps["centerline_core"])
                support = _as_channel(maps["support"])
                pred_lanes = list(sample_pred.get("lanes", []))
                for pred_index, pred_lane in enumerate(pred_lanes):
                    features = _prediction_features(pred_index, pred_lanes, maps=maps, meta=sample_meta)
                    score = score_repairability(features, ranker)
                    if str(args.repair_mode) == "local_2d_snap":
                        translated_points, repair_stats = snap_points_to_local_centerline(
                            list(pred_lane.get("points_xy", [])),
                            centerline=centerline,
                            meta=sample_meta,
                            radius=int(args.translation_radius),
                        )
                    elif str(args.repair_mode) == "affine_2d_snap":
                        translated_points, repair_stats = affine_snap_points_to_local_centerline(
                            list(pred_lane.get("points_xy", [])),
                            centerline=centerline,
                            meta=sample_meta,
                            radius=int(args.translation_radius),
                        )
                    elif str(args.repair_mode) == "component_row_project":
                        translated_points, repair_stats = component_project_points_to_centerline(
                            list(pred_lane.get("points_xy", [])),
                            centerline=centerline,
                            meta=sample_meta,
                        )
                    elif str(args.repair_mode) == "row_profile_softargmax":
                        translated_points, repair_stats = row_profile_project_points_to_centerline(
                            list(pred_lane.get("points_xy", [])),
                            centerline=centerline,
                            meta=sample_meta,
                            radius=int(args.translation_radius),
                        )
                    elif str(args.repair_mode) == "ridge_path_dp":
                        translated_points, repair_stats = ridge_path_project_points_to_dense_lane(
                            list(pred_lane.get("points_xy", [])),
                            centerline=centerline,
                            support=support,
                            meta=sample_meta,
                            radius=int(args.translation_radius),
                        )
                    else:
                        translated_points, dx = translate_points_to_centerline(
                            list(pred_lane.get("points_xy", [])),
                            centerline=centerline,
                            meta=sample_meta,
                            radius=int(args.translation_radius),
                        )
                        repair_stats = {
                            "dx": float(dx),
                            "moved_points": float(
                                sum(
                                    1
                                    for (x, y), (tx, ty) in zip(
                                        pred_lane.get("points_xy", []),
                                        translated_points,
                                    )
                                    if abs(float(x) - float(tx)) > 1.0e-3
                                    or abs(float(y) - float(ty)) > 1.0e-3
                                )
                            ),
                            "mean_move": float(abs(dx)),
                            "max_move": float(abs(dx)),
                        }
                    moved = any(
                        abs(float(x) - float(tx)) > 1.0e-3 or abs(float(y) - float(ty)) > 1.0e-3
                        for (x, y), (tx, ty) in zip(pred_lane.get("points_xy", []), translated_points)
                    )
                    candidates.append(
                        {
                            "score": float(score),
                            "global_sample_index": int(base_offset + sample_batch_index),
                            "batch_index": int(batch_index),
                            "sample_batch_index": int(sample_batch_index),
                            "pred_index": int(pred_index),
                            "moved": bool(moved),
                            "translated_points": translated_points,
                            **{key: float(value) for key, value in repair_stats.items()},
                            **{key: float(features.get(key, 0.0)) for key in ranker.get("features", [])},
                        }
                    )
                sample_count += 1

    if not raw_batches:
        raise ValueError("no validation batches were processed")
    repair_topk = int(args.repair_topk) if int(args.repair_topk) > 0 else _auto_repair_topk(sample_count)
    selected = sorted(candidates, key=lambda row: float(row["score"]), reverse=True)[:repair_topk]
    selected_keys: set[tuple[int, int]] = set()
    selected_rows: list[dict[str, Any]] = []
    for row in selected:
        global_sample_index = int(row["global_sample_index"])
        pred_index = int(row["pred_index"])
        if (global_sample_index, pred_index) in selected_keys:
            continue
        selected_keys.add((global_sample_index, pred_index))
        lane = repaired_predictions_all[global_sample_index]["lanes"][pred_index]
        lane["points_xy"] = [[float(x), float(y)] for x, y in row["translated_points"]]
        selected_rows.append({key: value for key, value in row.items() if key != "translated_points"})

    raw_all = _merge_raw_batches(raw_batches)
    base_metrics = augment_lane_family_metrics(summarize_pv26_metrics(base_predictions_all, raw_all))
    repaired_metrics = augment_lane_family_metrics(summarize_pv26_metrics(repaired_predictions_all, raw_all))
    metric_rows = [
        _metrics_row(base_metrics, name="baseline"),
        _metrics_row(repaired_metrics, name=f"ranked_{args.repair_mode}_repair"),
    ]
    lane_delta = {
        "f1_delta": float(metric_rows[1]["lane_f1"] - metric_rows[0]["lane_f1"]),
        "tp_delta": int(metric_rows[1]["lane_tp"] - metric_rows[0]["lane_tp"]),
        "fp_delta": int(metric_rows[1]["lane_fp"] - metric_rows[0]["lane_fp"]),
        "fn_delta": int(metric_rows[1]["lane_fn"] - metric_rows[0]["lane_fn"]),
    }
    summary = {
        "checkpoint": str(checkpoint),
        "source_run": str(source_run),
        "scenario_path": str(scenario_path),
        "ranker_parameters": str(ranker_path),
        "ranker_label": str(args.ranker_label),
        "lane60_experiment": str(args.lane60_experiment),
        "phase_index": int(phase_index),
        "max_val_batches": int(args.max_val_batches),
        "validation_epoch": int(args.validation_epoch),
        "lane_flip_variant": str(args.lane_flip_variant),
        "repair_mode": str(args.repair_mode),
        "translation_radius": int(args.translation_radius),
        "repair_topk": int(repair_topk),
        "sample_count": int(sample_count),
        "candidate_count": int(len(candidates)),
        "selected_count": int(len(selected_rows)),
        "selected_moved_count": int(sum(1 for row in selected_rows if bool(row.get("moved")))),
        "selected_moved_points": int(sum(int(row.get("moved_points", 0)) for row in selected_rows)),
        "metrics": metric_rows,
        "lane_delta": lane_delta,
        "interpretation": (
            "This is a read-only postprocess smoke. It uses a fixed no-GT ranker parameter artifact, "
            "but the ranker was trained from GT-derived repair labels. Success requires actual metric "
            "movement here, not the prior oracle replay."
        ),
    }
    _write_csv(output_dir / "metrics.csv", metric_rows)
    _write_csv(output_dir / "selected_repairs.csv", selected_rows)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
