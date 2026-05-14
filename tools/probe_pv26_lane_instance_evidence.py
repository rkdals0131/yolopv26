from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
import math
from pathlib import Path
import site
import sys
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
repo_root = str(REPO_ROOT)
if repo_root not in sys.path:
    site.addsitedir(repo_root)

from model.data.transform import transform_from_meta, transform_points
from model.engine import raw_batch_for_metrics
from model.engine._trainer_epochs import _merge_raw_batches
from model.engine.batch import augment_lane_family_metrics
from model.engine.lane_segfirst_vectorizer import lane_segfirst_prediction_maps
from model.engine.metrics import (
    _extract_gt_samples,
    _hungarian_from_cost,
    _mean_point_distance,
    summarize_pv26_metrics,
)
from model.engine.postprocess import postprocess_pv26_batch
from tools.evaluate_pv26_lane60_checkpoint import _advance_validation_sampler, _postprocess_override_config
from tools.probe_pv26_lane_flip_tta import _merge_lane_dense_predictions, _unflip_lane_dense_outputs
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
DEFAULT_CHECKPOINT = SOURCE_RUN / "phase_4" / "checkpoints" / "best.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit whether row-scan lane instances can be validated by map-local "
            "centerline/support/tangent evidence without retraining."
        )
    )
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--source-run", default=str(SOURCE_RUN))
    parser.add_argument("--lane60-experiment", default="core_centerline_refine_row_scan_vectorizer")
    parser.add_argument("--preset", default="default")
    parser.add_argument("--phase-index", type=int, default=4)
    parser.add_argument("--max-val-batches", type=int, default=128)
    parser.add_argument("--validation-epoch", type=int, default=2)
    parser.add_argument("--train-batches", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dataset-root",
        default="",
        help="Optional canonical dataset root. If omitted, infer from the source-run repo parent when the worktree has no dataset.",
    )
    parser.add_argument("--train-fraction", type=float, default=0.5)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--task-threshold-quantiles", type=int, default=31)
    parser.add_argument(
        "--skip-single-feature-replays",
        action="store_true",
        help="Skip per-feature task-threshold replay when only baseline/logistic/oracle summaries are needed.",
    )
    parser.add_argument(
        "--use-flip-consistency",
        action="store_true",
        help=(
            "Run a horizontal-flip forward pass and add normal/flip centerline "
            "agreement features to lane instance validation."
        ),
    )
    parser.add_argument(
        "--lane-flip-variant",
        choices=("baseline", "flip_centerline_avg"),
        default="baseline",
        help="Optional lane dense merge variant to apply before postprocess and instance evidence.",
    )
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
    parser.add_argument("--crosswalk-polygon-mode", choices=("rect", "hull"), default=None)
    parser.add_argument("--output-dir", default="")
    return parser.parse_args()


def _resolve_device(requested: str, scenario_device: str) -> str:
    value = str(requested or "auto").strip().lower()
    scenario_value = str(scenario_device or "auto").strip()
    if value == "auto" and scenario_value.lower() == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    else:
        device = scenario_value if value == "auto" else str(requested)
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[lane_instance] CUDA requested but unavailable; falling back to CPU", flush=True)
        return "cpu"
    return device


def _resolve_dataset_root(args: argparse.Namespace, source_run: Path, scenario_root: Path) -> Path:
    explicit = Path(str(args.dataset_root)).expanduser().resolve() if str(args.dataset_root).strip() else None
    if explicit is not None:
        return explicit
    if Path(scenario_root).is_dir():
        return Path(scenario_root).resolve()
    for parent in (source_run, *source_run.parents):
        candidate = parent / "seg_dataset" / "pv26_exhaustive_od_lane_dataset"
        if candidate.is_dir():
            return candidate.resolve()
    return Path(scenario_root).resolve()


def _points(points_xy: list[list[float]]) -> np.ndarray:
    return np.asarray(points_xy, dtype=np.float32).reshape(-1, 2)


def _polyline_length(points_xy: list[list[float]]) -> float:
    points = _points(points_xy)
    if points.shape[0] < 2 or not bool(np.isfinite(points).all()):
        return 0.0
    return float(np.linalg.norm(points[1:] - points[:-1], axis=1).sum())


def _shape_features(points_xy: list[list[float]]) -> dict[str, float]:
    points = _points(points_xy)
    if points.size == 0 or not bool(np.isfinite(points).all()):
        return {
            "bbox_w": 0.0,
            "bbox_h": 0.0,
            "bbox_area": 0.0,
            "bbox_aspect": 0.0,
            "center_x": 0.0,
            "center_y": 0.0,
            "polyline_length": 0.0,
            "max_turn_degrees": 0.0,
        }
    min_xy = points.min(axis=0)
    max_xy = points.max(axis=0)
    width = float(max_xy[0] - min_xy[0])
    height = float(max_xy[1] - min_xy[1])
    return {
        "bbox_w": width,
        "bbox_h": height,
        "bbox_area": max(width, 0.0) * max(height, 0.0),
        "bbox_aspect": max(width, height) / max(min(width, height), 1.0),
        "center_x": float((min_xy[0] + max_xy[0]) * 0.5),
        "center_y": float((min_xy[1] + max_xy[1]) * 0.5),
        "polyline_length": _polyline_length(points_xy),
        "max_turn_degrees": _max_turn_degrees(points),
    }


def _max_turn_degrees(points: np.ndarray) -> float:
    if points.shape[0] < 3 or not bool(np.isfinite(points).all()):
        return 0.0
    deltas = points[1:] - points[:-1]
    norms = np.linalg.norm(deltas, axis=1)
    valid = norms > 1.0e-6
    deltas = deltas[valid]
    norms = norms[valid]
    if deltas.shape[0] < 2:
        return 0.0
    unit = deltas / norms[:, None]
    dots = np.clip((unit[1:] * unit[:-1]).sum(axis=1), -1.0, 1.0)
    return float(np.degrees(np.arccos(dots)).max(initial=0.0))


def _as_channel(array: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(array, torch.Tensor):
        array = array.detach().cpu().numpy()
    out = np.asarray(array, dtype=np.float32)
    if out.ndim == 3 and out.shape[0] == 1:
        out = out[0]
    return out


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


def _track_mask(points: np.ndarray, map_hw: tuple[int, int], *, width: int = 3) -> np.ndarray:
    h, w = int(map_hw[0]), int(map_hw[1])
    image = Image.new("L", (w, h), 0)
    if points.shape[0] >= 2:
        draw = ImageDraw.Draw(image)
        coords = [(float(point[0]), float(point[1])) for point in points]
        draw.line(coords, fill=1, width=max(1, int(width)))
    return np.asarray(image, dtype=bool)


def _sample_polyline(points: np.ndarray, *, count: int = 64) -> np.ndarray:
    if points.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32)
    if points.shape[0] == 1:
        return np.repeat(points.astype(np.float32), repeats=max(1, int(count)), axis=0)
    deltas = points[1:] - points[:-1]
    lengths = np.linalg.norm(deltas, axis=1)
    total = float(lengths.sum())
    if total <= 1.0e-6:
        return np.repeat(points[:1].astype(np.float32), repeats=max(1, int(count)), axis=0)
    targets = np.linspace(0.0, total, max(2, int(count)), dtype=np.float32)
    cumulative = np.concatenate([[0.0], np.cumsum(lengths)])
    sampled: list[np.ndarray] = []
    for target in targets:
        index = int(np.searchsorted(cumulative, float(target), side="right") - 1)
        index = max(0, min(index, len(lengths) - 1))
        denom = max(float(lengths[index]), 1.0e-6)
        ratio = (float(target) - float(cumulative[index])) / denom
        sampled.append(points[index] + ratio * (points[index + 1] - points[index]))
    return np.asarray(sampled, dtype=np.float32)


def _values_at_points(array: np.ndarray, points: np.ndarray) -> np.ndarray:
    if points.shape[0] == 0:
        return np.zeros(0, dtype=np.float32)
    h, w = int(array.shape[0]), int(array.shape[1])
    xs = np.clip(np.rint(points[:, 0]).astype(np.int64), 0, w - 1)
    ys = np.clip(np.rint(points[:, 1]).astype(np.int64), 0, h - 1)
    return np.asarray(array[ys, xs], dtype=np.float32)


def _safe_stat(values: np.ndarray, op: str) -> float:
    finite = np.asarray(values[np.isfinite(values)], dtype=np.float32)
    if finite.size == 0:
        return 0.0
    if op == "mean":
        return float(finite.mean())
    if op == "min":
        return float(finite.min())
    if op == "max":
        return float(finite.max())
    if op.startswith("q"):
        return float(np.quantile(finite, float(op[1:]) / 100.0))
    raise ValueError(op)


def _max_low_run_fraction(values: np.ndarray, *, threshold: float) -> float:
    if values.size == 0:
        return 0.0
    best = 0
    current = 0
    for value in values:
        if float(value) < float(threshold):
            current += 1
            best = max(best, current)
        else:
            current = 0
    return float(best / max(1, int(values.size)))


def _tangent_alignment(tangent_axis: np.ndarray, points: np.ndarray) -> np.ndarray:
    if points.shape[0] < 2:
        return np.zeros(0, dtype=np.float32)
    h, w = int(tangent_axis.shape[1]), int(tangent_axis.shape[2])
    out: list[float] = []
    for start, end in zip(points[:-1], points[1:]):
        delta = end - start
        norm = float(np.linalg.norm(delta))
        if norm <= 1.0e-6:
            continue
        unit = delta / norm
        mid = (start + end) * 0.5
        x = int(np.clip(round(float(mid[0])), 0, w - 1))
        y = int(np.clip(round(float(mid[1])), 0, h - 1))
        axis = tangent_axis[:, y, x]
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm <= 1.0e-6:
            continue
        axis = axis / axis_norm
        out.append(abs(float(axis[0] * unit[0] + axis[1] * unit[1])))
    return np.asarray(out, dtype=np.float32)


def _flip_consistency_features(
    centerline: np.ndarray,
    flip_centerline: np.ndarray,
    *,
    mask: np.ndarray,
    sampled_points: np.ndarray,
) -> dict[str, float]:
    if tuple(centerline.shape) != tuple(flip_centerline.shape):
        raise ValueError(f"centerline shape mismatch: {centerline.shape} vs {flip_centerline.shape}")
    agreement = 1.0 - np.abs(centerline - flip_centerline)
    consensus = np.minimum(centerline, flip_centerline)
    flip_values = flip_centerline[mask]
    consensus_values = consensus[mask]
    agreement_values = agreement[mask]
    flip_samples = _values_at_points(flip_centerline, sampled_points)
    consensus_samples = _values_at_points(consensus, sampled_points)
    agreement_samples = _values_at_points(agreement, sampled_points)
    return {
        "flip_center_mask_mean": _safe_stat(flip_values, "mean"),
        "flip_center_point_mean": _safe_stat(flip_samples, "mean"),
        "center_consensus_mask_mean": _safe_stat(consensus_values, "mean"),
        "center_consensus_point_mean": _safe_stat(consensus_samples, "mean"),
        "center_consensus_point_q10": _safe_stat(consensus_samples, "q10"),
        "center_agreement_mask_mean": _safe_stat(agreement_values, "mean"),
        "center_agreement_point_mean": _safe_stat(agreement_samples, "mean"),
        "center_agreement_point_q10": _safe_stat(agreement_samples, "q10"),
    }


def _evidence_features(
    pred: dict[str, Any],
    *,
    maps: dict[str, torch.Tensor],
    flip_maps: dict[str, torch.Tensor] | None = None,
    meta: dict[str, Any],
) -> dict[str, float]:
    centerline = _as_channel(maps["centerline_core"])
    support = _as_channel(maps["support"])
    color_map = np.asarray(maps["color_map"].detach().cpu().numpy(), dtype=np.float32)
    lane_type_map = np.asarray(maps["lane_type_map"].detach().cpu().numpy(), dtype=np.float32)
    tangent_axis = np.asarray(maps["tangent_axis"].detach().cpu().numpy(), dtype=np.float32)
    map_hw = (int(centerline.shape[0]), int(centerline.shape[1]))
    map_points = _raw_points_to_map(list(pred.get("points_xy", [])), meta, map_hw)
    mask = _track_mask(map_points, map_hw, width=3)
    center_values = centerline[mask]
    support_values = support[mask]
    sampled_points = _sample_polyline(map_points, count=64)
    center_samples = _values_at_points(centerline, sampled_points)
    support_samples = _values_at_points(support, sampled_points)
    align = _tangent_alignment(tangent_axis, sampled_points)
    color_conf = color_map.max(axis=0)[mask] if mask.any() else np.zeros(0, dtype=np.float32)
    type_conf = lane_type_map.max(axis=0)[mask] if mask.any() else np.zeros(0, dtype=np.float32)
    track_pixels = int(mask.sum())
    features = {
        "track_pixels": float(track_pixels),
        "center_mask_mean": _safe_stat(center_values, "mean"),
        "center_mask_q10": _safe_stat(center_values, "q10"),
        "center_mask_q25": _safe_stat(center_values, "q25"),
        "center_mask_q50": _safe_stat(center_values, "q50"),
        "center_mask_q90": _safe_stat(center_values, "q90"),
        "center_mask_max": _safe_stat(center_values, "max"),
        "center_mask_active05": float((center_values >= 0.5).mean()) if center_values.size else 0.0,
        "center_mask_active07": float((center_values >= 0.7).mean()) if center_values.size else 0.0,
        "support_mask_mean": _safe_stat(support_values, "mean"),
        "support_mask_q25": _safe_stat(support_values, "q25"),
        "support_mask_active05": float((support_values >= 0.5).mean()) if support_values.size else 0.0,
        "center_point_mean": _safe_stat(center_samples, "mean"),
        "center_point_min": _safe_stat(center_samples, "min"),
        "center_point_q10": _safe_stat(center_samples, "q10"),
        "center_point_active05": float((center_samples >= 0.5).mean()) if center_samples.size else 0.0,
        "center_point_low_run05": _max_low_run_fraction(center_samples, threshold=0.5),
        "support_point_mean": _safe_stat(support_samples, "mean"),
        "support_point_q10": _safe_stat(support_samples, "q10"),
        "support_point_active05": float((support_samples >= 0.5).mean()) if support_samples.size else 0.0,
        "tangent_align_mean": _safe_stat(align, "mean"),
        "tangent_align_q25": _safe_stat(align, "q25"),
        "color_conf_mean": _safe_stat(color_conf, "mean"),
        "type_conf_mean": _safe_stat(type_conf, "mean"),
    }
    if flip_maps is not None:
        flip_centerline = _as_channel(flip_maps["centerline_core"])
        features.update(
            _flip_consistency_features(
                centerline,
                flip_centerline,
                mask=mask,
                sampled_points=sampled_points,
            )
        )
    return features


def _match_lane_rows(
    pred_rows: list[dict[str, Any]],
    gt_rows: list[dict[str, Any]],
) -> tuple[dict[int, tuple[int, float]], int]:
    cost = np.zeros((len(pred_rows), len(gt_rows)), dtype=np.float32)
    for pred_index, pred in enumerate(pred_rows):
        for gt_index, gt in enumerate(gt_rows):
            cost[pred_index, gt_index] = _mean_point_distance(pred["points_xy"], gt["points_xy"], 16)
    matches = _hungarian_from_cost(cost, max_cost=40.0)
    matched = {
        int(pred_index): (int(gt_index), float(cost[pred_index, gt_index]))
        for pred_index, gt_index in matches
    }
    return matched, max(0, len(gt_rows) - len(matches))


BASE_FEATURE_NAMES = (
    "bbox_w",
    "bbox_h",
    "bbox_area",
    "bbox_aspect",
    "center_x",
    "center_y",
    "polyline_length",
    "max_turn_degrees",
    "track_pixels",
    "center_mask_mean",
    "center_mask_q10",
    "center_mask_q25",
    "center_mask_q50",
    "center_mask_q90",
    "center_mask_max",
    "center_mask_active05",
    "center_mask_active07",
    "support_mask_mean",
    "support_mask_q25",
    "support_mask_active05",
    "center_point_mean",
    "center_point_min",
    "center_point_q10",
    "center_point_active05",
    "center_point_low_run05",
    "support_point_mean",
    "support_point_q10",
    "support_point_active05",
    "tangent_align_mean",
    "tangent_align_q25",
    "color_conf_mean",
    "type_conf_mean",
)

FLIP_FEATURE_NAMES = (
    "flip_center_mask_mean",
    "flip_center_point_mean",
    "center_consensus_mask_mean",
    "center_consensus_point_mean",
    "center_consensus_point_q10",
    "center_agreement_mask_mean",
    "center_agreement_point_mean",
    "center_agreement_point_q10",
)

BASE_SINGLE_FEATURES = (
    "center_mask_mean",
    "center_mask_q10",
    "center_mask_q25",
    "center_mask_active05",
    "center_point_mean",
    "center_point_q10",
    "center_point_active05",
    "support_mask_mean",
    "support_point_mean",
    "tangent_align_mean",
    "tangent_align_q25",
    "color_conf_mean",
    "type_conf_mean",
)

FLIP_SINGLE_FEATURES = (
    "flip_center_point_mean",
    "center_consensus_point_mean",
    "center_consensus_point_q10",
    "center_agreement_point_mean",
    "center_agreement_point_q10",
)


def _feature_names(*, use_flip_consistency: bool) -> tuple[str, ...]:
    if use_flip_consistency:
        return BASE_FEATURE_NAMES + FLIP_FEATURE_NAMES
    return BASE_FEATURE_NAMES


def _single_features(*, use_flip_consistency: bool) -> tuple[str, ...]:
    if use_flip_consistency:
        return BASE_SINGLE_FEATURES + FLIP_SINGLE_FEATURES
    return BASE_SINGLE_FEATURES


def _needs_flip_forward(*, use_flip_consistency: bool, lane_flip_variant: str) -> bool:
    return bool(use_flip_consistency) or str(lane_flip_variant).strip() != "baseline"


def _feature_matrix(rows: list[dict[str, Any]], feature_names: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    matrix = [[float(row.get(name, 0.0)) for name in feature_names] for row in rows]
    labels = [1.0 if int(row.get("is_tp", 0)) else 0.0 for row in rows]
    return np.asarray(matrix, dtype=np.float64), np.asarray(labels, dtype=np.float64)


def _standardize(train_x: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    std = np.where(std < 1.0e-6, 1.0, std)
    return (x - mean) / std, mean, std


def _sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -40.0, 40.0)))


def _fit_logistic(train_x: np.ndarray, train_y: np.ndarray, *, steps: int, lr: float) -> tuple[np.ndarray, float]:
    x = np.concatenate([np.ones((train_x.shape[0], 1), dtype=np.float64), train_x], axis=1)
    weights = np.zeros(x.shape[1], dtype=np.float64)
    pos = max(float(train_y.sum()), 1.0)
    neg = max(float(train_y.shape[0] - train_y.sum()), 1.0)
    sample_weights = np.where(train_y > 0.5, 0.5 / pos, 0.5 / neg)
    for _ in range(max(1, int(steps))):
        probs = _sigmoid(x @ weights)
        grad = x.T @ ((probs - train_y) * sample_weights)
        weights -= float(lr) * grad
    return weights[1:], float(weights[0])


def _predict(x: np.ndarray, weights: np.ndarray, bias: float) -> np.ndarray:
    return _sigmoid(x @ weights + float(bias))


def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    positives = labels > 0.5
    n_pos = int(positives.sum())
    n_neg = int(labels.shape[0] - n_pos)
    if n_pos == 0 or n_neg == 0:
        return 0.0
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, labels.shape[0] + 1, dtype=np.float64)
    pos_rank_sum = float(ranks[positives].sum())
    return (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / float(n_pos * n_neg)


def _average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    positives = labels > 0.5
    n_pos = int(positives.sum())
    if n_pos == 0:
        return 0.0
    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = positives[order].astype(np.float64)
    precision = np.cumsum(sorted_labels) / np.arange(1, sorted_labels.shape[0] + 1, dtype=np.float64)
    return float((precision * sorted_labels).sum() / float(n_pos))


def _split_masks(rows: list[dict[str, Any]], train_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    batch_indices = np.asarray([int(row["batch_index"]) for row in rows], dtype=np.int64)
    min_batch = int(batch_indices.min(initial=1))
    max_batch = int(batch_indices.max(initial=1))
    cutoff = min_batch + int(round((max_batch - min_batch + 1) * float(train_fraction))) - 1
    train_mask = batch_indices <= cutoff
    return train_mask, ~train_mask


def _metric(metrics: dict[str, Any], task: str, key: str) -> float:
    payload = metrics.get(task, {}) if isinstance(metrics.get(task), dict) else {}
    value = payload.get(key, 0.0)
    return float(value) if isinstance(value, (int, float)) else 0.0


def _row_from_metrics(name: str, split: str, threshold: float | None, metrics: dict[str, Any]) -> dict[str, Any]:
    row = {
        "variant": name,
        "split": split,
        "threshold": "" if threshold is None else float(threshold),
    }
    for task in ("lane", "stop_line", "crosswalk"):
        for key in ("precision", "recall", "f1", "tp", "fp", "fn"):
            row[f"{task}_{key}"] = _metric(metrics, task, key)
    lane_family = metrics.get("lane_family", {}) if isinstance(metrics.get("lane_family"), dict) else {}
    row["lane_family_mean_f1"] = float(lane_family.get("mean_f1", 0.0))
    row["lane_family_min_f1"] = float(lane_family.get("min_f1", 0.0))
    row["phase4_objective_proxy"] = (
        0.50 * float(row["lane_f1"])
        + 0.30 * float(row["stop_line_f1"])
        + 0.20 * float(row["crosswalk_f1"])
    )
    return row


def _filter_prediction_samples(
    samples: list[dict[str, Any]],
    rows_by_sample: dict[int, list[dict[str, Any]]],
    scores_by_row: dict[int, float],
    *,
    threshold: float,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for sample_position, sample in enumerate(samples):
        lane_rows = rows_by_sample.get(sample_position, [])
        kept_lanes = []
        for lane_index, lane in enumerate(sample.get("lanes", [])):
            row = lane_rows[lane_index] if lane_index < len(lane_rows) else None
            row_id = int(row["row_id"]) if row is not None else -1
            if row_id >= 0 and float(scores_by_row.get(row_id, 0.0)) >= float(threshold):
                kept_lanes.append(lane)
        cloned = dict(sample)
        cloned["lanes"] = kept_lanes
        output.append(cloned)
    return output


def _oracle_tp_scores_by_row(rows: list[dict[str, Any]]) -> dict[int, float]:
    return {
        int(row["row_id"]): 1.0 if int(row.get("is_tp", 0)) else 0.0
        for row in rows
    }


def _metrics_for_split(
    batch_records: list[dict[str, Any]],
    rows_by_sample: dict[int, list[dict[str, Any]]],
    *,
    split_batches: set[int],
    scores_by_row: dict[int, float] | None = None,
    threshold: float | None = None,
) -> dict[str, Any]:
    selected_predictions: list[dict[str, Any]] = []
    selected_raw_batches: list[dict[str, Any]] = []
    sample_offset = 0
    for record in batch_records:
        batch_index = int(record["batch_index"])
        samples = list(record["predictions"])
        if batch_index in split_batches:
            if scores_by_row is not None and threshold is not None:
                local_rows = {
                    local_index: rows_by_sample.get(sample_offset + local_index, [])
                    for local_index in range(len(samples))
                }
                filtered = _filter_prediction_samples(
                    samples,
                    local_rows,
                    scores_by_row,
                    threshold=float(threshold),
                )
                selected_predictions.extend(filtered)
            else:
                selected_predictions.extend(samples)
            selected_raw_batches.append(record["raw_batch"])
        sample_offset += len(samples)
    if not selected_raw_batches:
        return {}
    return augment_lane_family_metrics(summarize_pv26_metrics(selected_predictions, _merge_raw_batches(selected_raw_batches)))


def _best_task_threshold(
    batch_records: list[dict[str, Any]],
    rows_by_sample: dict[int, list[dict[str, Any]]],
    scores_by_row: dict[int, float],
    *,
    train_batches: set[int],
    quantile_count: int,
) -> tuple[float, dict[str, Any]]:
    scores = np.asarray(list(scores_by_row.values()), dtype=np.float64)
    if scores.size == 0:
        return 1.0, {}
    candidates = np.unique(np.quantile(scores, np.linspace(0.0, 1.0, max(3, int(quantile_count)))))
    best_threshold = float(candidates[0])
    best_metrics = _metrics_for_split(
        batch_records,
        rows_by_sample,
        split_batches=train_batches,
        scores_by_row=scores_by_row,
        threshold=best_threshold,
    )
    for threshold in candidates:
        metrics = _metrics_for_split(
            batch_records,
            rows_by_sample,
            split_batches=train_batches,
            scores_by_row=scores_by_row,
            threshold=float(threshold),
        )
        if (
            _metric(metrics, "lane", "f1"),
            _metric(metrics, "lane", "precision"),
            _metric(metrics, "lane", "recall"),
        ) > (
            _metric(best_metrics, "lane", "f1"),
            _metric(best_metrics, "lane", "precision"),
            _metric(best_metrics, "lane", "recall"),
        ):
            best_threshold = float(threshold)
            best_metrics = metrics
    return best_threshold, best_metrics


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
        else source_run / "analysis_exports" / f"lane_instance_evidence_val{int(args.max_val_batches)}_epoch{int(args.validation_epoch)}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    train_cli._configure_torch_multiprocessing()
    dataset = train_cli.PV26CanonicalDataset(
        train_cli._existing_dataset_roots(scenario),
        train_augmentation=False,
        progress_callback=lambda message: print(f"[lane_instance] {message}", flush=True),
    )
    _, val_loader = train_cli._build_phase_train_loaders(dataset, train_config=train_config, phase=phase)
    if val_loader is None:
        raise ValueError("lane instance evidence audit requires validation batches")
    _advance_validation_sampler(val_loader, validation_epoch=int(args.validation_epoch))

    trainer = train_cli._build_phase_trainer(phase, train_config)
    trainer.load_model_weights(checkpoint, map_location=train_config.device)
    postprocess_config = _postprocess_override_config(args, trainer) or train_cli._build_postprocess_config(train_config)
    evaluator = trainer.build_evaluator()
    evaluator.adapter.raw_model.eval()
    evaluator.heads.eval()

    batch_records: list[dict[str, Any]] = []
    gt_samples: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    rows_by_sample: dict[int, list[dict[str, Any]]] = {}
    global_sample_index = 0
    row_id = 0
    with torch.no_grad():
        for batch_index, batch in enumerate(val_loader, start=1):
            if batch_index > int(args.max_val_batches):
                break
            if batch_index == 1 or batch_index % 20 == 0:
                print(f"[lane_instance] eval batch {batch_index}/{args.max_val_batches}", flush=True)
            raw_batch = raw_batch_for_metrics(batch)
            if raw_batch is None:
                raise ValueError("validation loader must provide raw batches for metrics")
            encoded = evaluator.prepare_batch(batch)
            predictions = _detach_to_cpu(evaluator.forward_encoded_batch(encoded))
            flip_predictions = None
            if _needs_flip_forward(
                use_flip_consistency=bool(args.use_flip_consistency),
                lane_flip_variant=str(args.lane_flip_variant),
            ):
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
            batch_records.append({"batch_index": int(batch_index), "predictions": batch_predictions, "raw_batch": raw_batch})
            gt_samples.extend(batch_gt)
            for sample_batch_index, (sample_pred, sample_gt, sample_meta) in enumerate(
                zip(batch_predictions, batch_gt, meta)
            ):
                maps = lane_segfirst_prediction_maps(postprocess_predictions, batch_index=sample_batch_index)
                flip_maps = (
                    lane_segfirst_prediction_maps(flip_predictions, batch_index=sample_batch_index)
                    if flip_predictions is not None
                    else None
                )
                lane_rows = list(sample_pred.get("lanes", []))
                matches, fn_count = _match_lane_rows(lane_rows, list(sample_gt.get("lanes", [])))
                sample_rows: list[dict[str, Any]] = []
                for pred_index, lane in enumerate(lane_rows):
                    gt_match = matches.get(pred_index)
                    row: dict[str, Any] = {
                        "row_id": int(row_id),
                        "batch_index": int(batch_index),
                        "sample_index": int(global_sample_index),
                        "sample_batch_index": int(sample_batch_index),
                        "pred_index": int(pred_index),
                        "is_tp": int(gt_match is not None),
                        "match_quality": float(gt_match[1]) if gt_match is not None else 0.0,
                        "matched_gt_index": int(gt_match[0]) if gt_match is not None else -1,
                        "sample_fn_count": int(fn_count),
                    }
                    row.update(_shape_features(list(lane.get("points_xy", []))))
                    row.update(_evidence_features(lane, maps=maps, flip_maps=flip_maps, meta=sample_meta))
                    candidate_rows.append(row)
                    sample_rows.append(row)
                    row_id += 1
                rows_by_sample[global_sample_index] = sample_rows
                global_sample_index += 1

    if not candidate_rows:
        raise ValueError("no lane candidate rows were produced")

    train_mask, test_mask = _split_masks(candidate_rows, float(args.train_fraction))
    feature_names = _feature_names(use_flip_consistency=bool(args.use_flip_consistency))
    single_features = _single_features(use_flip_consistency=bool(args.use_flip_consistency))
    features, labels = _feature_matrix(candidate_rows, feature_names)
    train_x = features[train_mask]
    train_y = labels[train_mask]
    test_x = features[test_mask]
    test_y = labels[test_mask]
    if train_x.shape[0] == 0 or test_x.shape[0] == 0:
        raise ValueError("lane instance audit requires non-empty train and test splits")
    train_x_std, mean, std = _standardize(train_x, train_x)
    test_x_std = (test_x - mean) / std
    weights, bias = _fit_logistic(train_x_std, train_y, steps=int(args.steps), lr=float(args.lr))
    all_scores = _predict((features - mean) / std, weights, bias)

    batch_indices = {int(record["batch_index"]) for record in batch_records}
    cutoff = min(batch_indices) + int(round((max(batch_indices) - min(batch_indices) + 1) * float(args.train_fraction))) - 1
    train_batches = {index for index in batch_indices if index <= cutoff}
    test_batches = batch_indices - train_batches
    all_batches = set(batch_indices)

    rows_by_id = {int(row["row_id"]): row for row in candidate_rows}
    logistic_scores_by_row = {int(row["row_id"]): float(score) for row, score in zip(candidate_rows, all_scores)}
    oracle_tp_scores_by_row = _oracle_tp_scores_by_row(candidate_rows)
    variant_rows: list[dict[str, Any]] = []
    baseline_train = _metrics_for_split(batch_records, rows_by_sample, split_batches=train_batches)
    baseline_test = _metrics_for_split(batch_records, rows_by_sample, split_batches=test_batches)
    baseline_full = _metrics_for_split(batch_records, rows_by_sample, split_batches=all_batches)
    variant_rows.append(_row_from_metrics("row_scan_baseline", "train", None, baseline_train))
    variant_rows.append(_row_from_metrics("row_scan_baseline", "heldout", None, baseline_test))
    variant_rows.append(_row_from_metrics("row_scan_baseline", "full", None, baseline_full))
    oracle_train = _metrics_for_split(
        batch_records,
        rows_by_sample,
        split_batches=train_batches,
        scores_by_row=oracle_tp_scores_by_row,
        threshold=0.5,
    )
    oracle_test = _metrics_for_split(
        batch_records,
        rows_by_sample,
        split_batches=test_batches,
        scores_by_row=oracle_tp_scores_by_row,
        threshold=0.5,
    )
    oracle_full = _metrics_for_split(
        batch_records,
        rows_by_sample,
        split_batches=all_batches,
        scores_by_row=oracle_tp_scores_by_row,
        threshold=0.5,
    )
    variant_rows.append(_row_from_metrics("oracle_keep_tp_only", "train", 0.5, oracle_train))
    variant_rows.append(_row_from_metrics("oracle_keep_tp_only", "heldout", 0.5, oracle_test))
    variant_rows.append(_row_from_metrics("oracle_keep_tp_only", "full", 0.5, oracle_full))
    print("[lane_instance] threshold replay feature=logistic", flush=True)
    logistic_threshold, logistic_train_metrics = _best_task_threshold(
        batch_records,
        rows_by_sample,
        logistic_scores_by_row,
        train_batches=train_batches,
        quantile_count=int(args.task_threshold_quantiles),
    )
    logistic_test_metrics = _metrics_for_split(
        batch_records,
        rows_by_sample,
        split_batches=test_batches,
        scores_by_row=logistic_scores_by_row,
        threshold=logistic_threshold,
    )
    variant_rows.append(_row_from_metrics("logistic_task_threshold", "train", logistic_threshold, logistic_train_metrics))
    variant_rows.append(_row_from_metrics("logistic_task_threshold", "heldout", logistic_threshold, logistic_test_metrics))

    feature_reports: list[dict[str, Any]] = []
    if not bool(args.skip_single_feature_replays):
        for feature_name in single_features:
            print(f"[lane_instance] threshold replay feature={feature_name}", flush=True)
            scores_by_row = {
                int(row_id_value): float(rows_by_id[row_id_value].get(feature_name, 0.0))
                for row_id_value in rows_by_id
            }
            threshold, train_metrics = _best_task_threshold(
                batch_records,
                rows_by_sample,
                scores_by_row,
                train_batches=train_batches,
                quantile_count=int(args.task_threshold_quantiles),
            )
            test_metrics = _metrics_for_split(
                batch_records,
                rows_by_sample,
                split_batches=test_batches,
                scores_by_row=scores_by_row,
                threshold=threshold,
            )
            variant_name = f"{feature_name}_task_threshold"
            variant_rows.append(_row_from_metrics(variant_name, "train", threshold, train_metrics))
            variant_rows.append(_row_from_metrics(variant_name, "heldout", threshold, test_metrics))
            raw_train = np.asarray([float(row.get(feature_name, 0.0)) for row, keep in zip(candidate_rows, train_mask) if keep], dtype=np.float64)
            raw_test = np.asarray([float(row.get(feature_name, 0.0)) for row, keep in zip(candidate_rows, test_mask) if keep], dtype=np.float64)
            feature_reports.append(
                {
                    "feature": feature_name,
                    "train_auc": _auc(raw_train, train_y),
                    "train_ap": _average_precision(raw_train, train_y),
                    "heldout_auc": _auc(raw_test, test_y),
                    "heldout_ap": _average_precision(raw_test, test_y),
                    "task_threshold": float(threshold),
                    "train_lane_f1": _metric(train_metrics, "lane", "f1"),
                    "heldout_lane_f1": _metric(test_metrics, "lane", "f1"),
                    "heldout_lane_tp": _metric(test_metrics, "lane", "tp"),
                    "heldout_lane_fp": _metric(test_metrics, "lane", "fp"),
                    "heldout_lane_fn": _metric(test_metrics, "lane", "fn"),
                }
            )

    variant_rows.sort(
        key=lambda row: (
            str(row["split"]) != "heldout",
            -float(row.get("lane_f1", 0.0)),
            -float(row.get("phase4_objective_proxy", 0.0)),
        )
    )

    _write_csv(output_dir / "lane_instance_features.csv", candidate_rows)
    _write_csv(output_dir / "feature_reports.csv", feature_reports)
    _write_csv(output_dir / "validator_variants.csv", variant_rows)
    summary = {
        "checkpoint": str(checkpoint),
        "source_run": str(source_run),
        "scenario_path": str(scenario_path),
        "lane60_experiment": str(args.lane60_experiment),
        "phase_index": int(phase_index),
        "max_val_batches": int(args.max_val_batches),
        "validation_epoch": int(args.validation_epoch),
        "use_flip_consistency": bool(args.use_flip_consistency),
        "lane_flip_variant": str(args.lane_flip_variant),
        "candidate_count": int(len(candidate_rows)),
        "positive_count": int(labels.sum()),
        "split": {
            "train_fraction": float(args.train_fraction),
            "cutoff_batch": int(cutoff),
            "train_batches": int(len(train_batches)),
            "heldout_batches": int(len(test_batches)),
            "train_rows": int(train_y.shape[0]),
            "heldout_rows": int(test_y.shape[0]),
            "train_positive_count": int(train_y.sum()),
            "heldout_positive_count": int(test_y.sum()),
        },
        "row_classifier": {
            "logistic_train_auc": _auc(_predict(train_x_std, weights, bias), train_y),
            "logistic_train_ap": _average_precision(_predict(train_x_std, weights, bias), train_y),
            "logistic_heldout_auc": _auc(_predict(test_x_std, weights, bias), test_y),
            "logistic_heldout_ap": _average_precision(_predict(test_x_std, weights, bias), test_y),
            "weights": {name: float(weight) for name, weight in zip(feature_names, weights)},
            "bias": float(bias),
        },
        "baseline_train": _row_from_metrics("row_scan_baseline", "train", None, baseline_train),
        "baseline_heldout": _row_from_metrics("row_scan_baseline", "heldout", None, baseline_test),
        "baseline_full": _row_from_metrics("row_scan_baseline", "full", None, baseline_full),
        "oracle_keep_tp_train": _row_from_metrics("oracle_keep_tp_only", "train", 0.5, oracle_train),
        "oracle_keep_tp_heldout": _row_from_metrics("oracle_keep_tp_only", "heldout", 0.5, oracle_test),
        "oracle_keep_tp_full": _row_from_metrics("oracle_keep_tp_only", "full", 0.5, oracle_full),
        "logistic_train": _row_from_metrics("logistic_task_threshold", "train", logistic_threshold, logistic_train_metrics),
        "logistic_heldout": _row_from_metrics("logistic_task_threshold", "heldout", logistic_threshold, logistic_test_metrics),
        "best_heldout_variants": [row for row in variant_rows if row["split"] == "heldout"][:8],
        "interpretation": (
            "This is a read-only lane instance evidence audit. A held-out lane F1 gain may justify "
            "a learned instance-stability contract, but oracle TP filtering uses GT labels and is not "
            "itself a final lane-family F1 0.6 success."
        ),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
