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
from tools.probe_pv26_lane_temporal_neighbor_union import _build_scenario
from tools.probe_pv26_lane60_postprocess_thresholds import _detach_to_cpu
from tools.probe_pv26_stopline_angle_mask_extent import _as_2d_array, _row_from_metrics, _sample_tensor, _write_csv
from tools.probe_pv26_stopline_candidate_pool import (
    _fit_raw_patch_mlp,
    _nearest_gt,
    _predict_raw_patch_mlp,
    _standardize_from_train,
    _write_candidate_features_csv,
)
from tools.probe_pv26_stopline_raw_hough_candidates import _line_stats, _raw_points_to_output, _slice_raw_batch_sample
from tools.probe_pv26_stopline_temporal_candidates import _copy_stop_line, _line_length, _nearest_current_distance
from tools.pv26_train import cli as train_cli


SCORE_KEY = "crosswalk_edge_mlp_score"
CROSSWALK_EDGE_FEATURES = (
    "crosswalk_edge_score",
    "crosswalk_edge_rank_norm",
    "crosswalk_edge_crosswalk_rank_norm",
    "crosswalk_edge_side",
    "crosswalk_edge_offset_norm",
    "crosswalk_edge_length_scale",
    "crosswalk_edge_length_norm",
    "crosswalk_edge_crosswalk_length_norm",
    "crosswalk_edge_crosswalk_width_norm",
    "crosswalk_edge_crosswalk_aspect",
    "crosswalk_edge_current_stopline_count",
    "crosswalk_edge_nearest_current_distance_norm",
    "crosswalk_edge_stop_mask_mean",
    "crosswalk_edge_stop_mask_max",
    "crosswalk_edge_stop_center_mean",
    "crosswalk_edge_stop_center_max",
    "crosswalk_edge_stop_selector_mean",
    "crosswalk_edge_stop_selector_max",
    "crosswalk_edge_stop_proposal_mean",
    "crosswalk_edge_stop_proposal_max",
    "crosswalk_edge_crosswalk_mask_mean",
    "crosswalk_edge_crosswalk_mask_max",
    "crosswalk_edge_crosswalk_center_mean",
    "crosswalk_edge_crosswalk_center_max",
    "crosswalk_edge_center_x_norm",
    "crosswalk_edge_center_y_norm",
    "crosswalk_edge_abs_cos",
    "crosswalk_edge_abs_sin",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate stop-line candidates from predicted crosswalk hull long-edges, "
            "train a small no-GT verifier on canonical train batches, and replay it on validation."
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
    parser.add_argument("--crosswalk-edge-offsets", default="0,8,16,24,32,48")
    parser.add_argument("--crosswalk-edge-length-scales", default="0.85,1.0,1.15")
    parser.add_argument("--max-crosswalks", type=int, default=3)
    parser.add_argument("--crosswalk-edge-top-k", type=int, default=12)
    parser.add_argument("--max-crosswalk-edge-candidates", type=int, default=24)
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
    parser.add_argument("--crosswalk-obj-threshold", type=float, default=None)
    parser.add_argument("--crosswalk-mask-binary-threshold", type=float, default=None)
    parser.add_argument("--crosswalk-min-component-pixels", type=int, default=None)
    parser.add_argument("--crosswalk-max-components", type=int, default=None)
    parser.add_argument("--crosswalk-min-polygon-area-px", type=float, default=None)
    parser.add_argument("--crosswalk-min-bbox-aspect", type=float, default=None)
    parser.add_argument("--crosswalk-polygon-mode", choices=("rect", "hull"), default="hull")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def _parse_float_list(value: str, *, name: str, min_value: float = 0.0) -> tuple[float, ...]:
    parsed = tuple(float(part.strip()) for part in str(value).split(",") if part.strip())
    if not parsed:
        raise ValueError(f"{name} must contain at least one value")
    for item in parsed:
        if not math.isfinite(float(item)) or float(item) < float(min_value):
            raise ValueError(f"{name} values must be finite and >= {min_value}, got {item}")
    return parsed


def _crosswalk_points_array(crosswalk: dict[str, Any]) -> np.ndarray:
    points = np.asarray(crosswalk.get("points_xy", []), dtype=np.float32).reshape(-1, 2)
    if points.shape[0] < 3 or not bool(np.isfinite(points).all()):
        return np.zeros((0, 2), dtype=np.float32)
    keep = np.ones((points.shape[0],), dtype=bool)
    if points.shape[0] > 1:
        keep[1:] = np.linalg.norm(points[1:] - points[:-1], axis=1) > 1.0e-4
    points = points[keep]
    if points.shape[0] < 3:
        return np.zeros((0, 2), dtype=np.float32)
    return points.astype(np.float32)


def _clip_points(points: np.ndarray, meta: dict[str, Any]) -> np.ndarray:
    raw_h, raw_w = int(meta.get("raw_hw", (1, 1))[0]), int(meta.get("raw_hw", (1, 1))[1])
    clipped = np.asarray(points, dtype=np.float32).reshape(-1, 2).copy()
    clipped[:, 0] = np.clip(clipped[:, 0], 0.0, max(float(raw_w - 1), 0.0))
    clipped[:, 1] = np.clip(clipped[:, 1], 0.0, max(float(raw_h - 1), 0.0))
    return clipped.astype(np.float32)


def _crosswalk_axes(points: np.ndarray) -> dict[str, Any] | None:
    points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if points.shape[0] < 3:
        return None
    center = points.mean(axis=0)
    centered = points - center
    cov = np.cov(centered.T)
    if np.asarray(cov).shape != (2, 2) or not bool(np.isfinite(cov).all()):
        return None
    values, vectors = np.linalg.eigh(cov.astype(np.float64))
    order = np.argsort(values)[::-1]
    major = vectors[:, order[0]].astype(np.float32)
    norm = float(np.linalg.norm(major))
    if norm <= 1.0e-6:
        return None
    major = major / norm
    if float(major[0]) < 0.0:
        major = -major
    minor = np.asarray([-major[1], major[0]], dtype=np.float32)
    major_coord = centered @ major
    minor_coord = centered @ minor
    major_min, major_max = float(major_coord.min()), float(major_coord.max())
    minor_min, minor_max = float(minor_coord.min()), float(minor_coord.max())
    half_length = 0.5 * (major_max - major_min)
    half_width = 0.5 * (minor_max - minor_min)
    if half_length < 8.0 or half_width < 1.0:
        return None
    return {
        "center": center.astype(np.float32),
        "axis": major.astype(np.float32),
        "normal": minor.astype(np.float32),
        "major_min": major_min,
        "major_max": major_max,
        "minor_min": minor_min,
        "minor_max": minor_max,
        "half_length": float(half_length),
        "half_width": float(half_width),
    }


def _proposal_map(
    stop_center_probs: np.ndarray | None,
    stop_selector_probs: np.ndarray | None,
) -> np.ndarray | None:
    if stop_center_probs is not None and stop_selector_probs is not None:
        return np.maximum(stop_center_probs, stop_selector_probs)
    if stop_center_probs is not None:
        return stop_center_probs
    return stop_selector_probs


def _edge_line_stats(
    points_raw: np.ndarray,
    *,
    meta: dict[str, Any],
    map_array: np.ndarray | None,
) -> tuple[float, float]:
    if map_array is None:
        return 0.0, 0.0
    points = np.asarray(points_raw, dtype=np.float32).reshape(-1, 2)
    if points.shape[0] < 2:
        return 0.0, 0.0
    output_points = _raw_points_to_output(points, meta, (int(map_array.shape[0]), int(map_array.shape[1])))
    return _line_stats(map_array, output_points)


def _crosswalk_edge_features(
    candidate: dict[str, Any],
    *,
    meta: dict[str, Any],
    stop_mask_probs: np.ndarray | None,
    stop_center_probs: np.ndarray | None,
    stop_selector_probs: np.ndarray | None,
    crosswalk_mask_probs: np.ndarray | None,
    crosswalk_center_probs: np.ndarray | None,
    current_stop_lines: list[dict[str, Any]],
    candidate_rank: int,
) -> dict[str, float]:
    points_raw = np.asarray(candidate.get("points_xy", []), dtype=np.float32).reshape(-1, 2)
    if points_raw.shape[0] < 2:
        return {name: 0.0 for name in CROSSWALK_EDGE_FEATURES}
    proposal = _proposal_map(stop_center_probs, stop_selector_probs)
    stop_mask_mean, stop_mask_max = _edge_line_stats(points_raw, meta=meta, map_array=stop_mask_probs)
    stop_center_mean, stop_center_max = _edge_line_stats(points_raw, meta=meta, map_array=stop_center_probs)
    stop_selector_mean, stop_selector_max = _edge_line_stats(points_raw, meta=meta, map_array=stop_selector_probs)
    proposal_mean, proposal_max = _edge_line_stats(points_raw, meta=meta, map_array=proposal)
    cross_mask_mean, cross_mask_max = _edge_line_stats(points_raw, meta=meta, map_array=crosswalk_mask_probs)
    cross_center_mean, cross_center_max = _edge_line_stats(points_raw, meta=meta, map_array=crosswalk_center_probs)
    raw_h, raw_w = int(meta.get("raw_hw", (1, 1))[0]), int(meta.get("raw_hw", (1, 1))[1])
    length = _line_length(candidate)
    center = points_raw.mean(axis=0)
    delta = points_raw[-1] - points_raw[0]
    norm = float(np.linalg.norm(delta))
    axis = delta / max(norm, 1.0e-6)
    nearest_distance = _nearest_current_distance(candidate, current_stop_lines)
    crosswalk_length = float(candidate.get("crosswalk_edge_crosswalk_length_px", 0.0))
    crosswalk_width = float(candidate.get("crosswalk_edge_crosswalk_width_px", 0.0))
    features = {
        "crosswalk_edge_score": float(candidate.get("score", 0.0)),
        "crosswalk_edge_rank_norm": float(1.0 / max(int(candidate_rank), 1)),
        "crosswalk_edge_crosswalk_rank_norm": float(
            1.0 / max(int(candidate.get("crosswalk_edge_crosswalk_rank", 999)), 1)
        ),
        "crosswalk_edge_side": float(candidate.get("crosswalk_edge_side", 0.0)),
        "crosswalk_edge_offset_norm": float(min(float(candidate.get("crosswalk_edge_offset_px", 0.0)) / 96.0, 2.0)),
        "crosswalk_edge_length_scale": float(candidate.get("crosswalk_edge_length_scale", 1.0)),
        "crosswalk_edge_length_norm": float(min(length / max(float(raw_w), 1.0), 1.0)),
        "crosswalk_edge_crosswalk_length_norm": float(min(crosswalk_length / max(float(raw_w), 1.0), 1.0)),
        "crosswalk_edge_crosswalk_width_norm": float(min(crosswalk_width / max(float(raw_h), 1.0), 1.0)),
        "crosswalk_edge_crosswalk_aspect": float(
            min(crosswalk_length / max(crosswalk_width, 1.0e-6), 16.0)
        ),
        "crosswalk_edge_current_stopline_count": float(len(current_stop_lines)),
        "crosswalk_edge_nearest_current_distance_norm": float(min(nearest_distance / max(float(raw_w), 1.0), 4.0)),
        "crosswalk_edge_stop_mask_mean": float(stop_mask_mean),
        "crosswalk_edge_stop_mask_max": float(stop_mask_max),
        "crosswalk_edge_stop_center_mean": float(stop_center_mean),
        "crosswalk_edge_stop_center_max": float(stop_center_max),
        "crosswalk_edge_stop_selector_mean": float(stop_selector_mean),
        "crosswalk_edge_stop_selector_max": float(stop_selector_max),
        "crosswalk_edge_stop_proposal_mean": float(proposal_mean),
        "crosswalk_edge_stop_proposal_max": float(proposal_max),
        "crosswalk_edge_crosswalk_mask_mean": float(cross_mask_mean),
        "crosswalk_edge_crosswalk_mask_max": float(cross_mask_max),
        "crosswalk_edge_crosswalk_center_mean": float(cross_center_mean),
        "crosswalk_edge_crosswalk_center_max": float(cross_center_max),
        "crosswalk_edge_center_x_norm": float(np.clip(float(center[0]) / max(float(raw_w), 1.0), 0.0, 1.0)),
        "crosswalk_edge_center_y_norm": float(np.clip(float(center[1]) / max(float(raw_h), 1.0), 0.0, 1.0)),
        "crosswalk_edge_abs_cos": float(abs(float(axis[0]))),
        "crosswalk_edge_abs_sin": float(abs(float(axis[1]))),
    }
    return {key: 0.0 if not math.isfinite(float(value)) else float(value) for key, value in features.items()}


def _crosswalk_edge_candidates_from_crosswalks(
    crosswalks: list[dict[str, Any]],
    *,
    meta: dict[str, Any],
    offsets: tuple[float, ...],
    length_scales: tuple[float, ...],
    max_crosswalks: int,
    max_candidates: int,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for crosswalk_rank, crosswalk in enumerate(crosswalks[: max(1, int(max_crosswalks))], start=1):
        points = _crosswalk_points_array(crosswalk)
        axes = _crosswalk_axes(points)
        if axes is None:
            continue
        axis = np.asarray(axes["axis"], dtype=np.float32)
        normal = np.asarray(axes["normal"], dtype=np.float32)
        center = np.asarray(axes["center"], dtype=np.float32)
        half_length = float(axes["half_length"])
        minor_min = float(axes["minor_min"])
        minor_max = float(axes["minor_max"])
        crosswalk_width = float(axes["half_width"]) * 2.0
        crosswalk_length = half_length * 2.0
        for side in (-1.0, 1.0):
            edge_minor = minor_min if side < 0.0 else minor_max
            for offset in offsets:
                line_center = center + normal * float(edge_minor + side * float(offset))
                for length_scale in length_scales:
                    half = half_length * float(length_scale)
                    raw_points = np.stack([line_center - axis * half, line_center + axis * half], axis=0)
                    raw_points = _clip_points(raw_points, meta)
                    if float(np.linalg.norm(raw_points[1] - raw_points[0])) < 12.0:
                        continue
                    score = float(crosswalk.get("score", 0.0))
                    candidates.append(
                        {
                            "points_xy": [[float(x), float(y)] for x, y in raw_points.tolist()],
                            "score": score,
                            "crosswalk_edge_crosswalk_rank": int(crosswalk_rank),
                            "crosswalk_edge_side": float(side),
                            "crosswalk_edge_offset_px": float(offset),
                            "crosswalk_edge_length_scale": float(length_scale),
                            "crosswalk_edge_crosswalk_score": score,
                            "crosswalk_edge_crosswalk_length_px": crosswalk_length,
                            "crosswalk_edge_crosswalk_width_px": crosswalk_width,
                        }
                    )
    candidates.sort(
        key=lambda item: (
            float(item.get("score", 0.0)),
            -float(item.get("crosswalk_edge_offset_px", 0.0)),
            -abs(float(item.get("crosswalk_edge_length_scale", 1.0)) - 1.0),
            float(item.get("crosswalk_edge_crosswalk_length_px", 0.0)),
        ),
        reverse=True,
    )
    return candidates[: max(1, int(max_candidates))]


def _build_crosswalk_edge_candidates(
    *,
    meta: dict[str, Any],
    baseline_prediction: dict[str, Any],
    gt_stop_lines: list[dict[str, Any]],
    stop_mask_probs: np.ndarray | None,
    stop_center_probs: np.ndarray | None,
    stop_selector_probs: np.ndarray | None,
    crosswalk_mask_probs: np.ndarray | None,
    crosswalk_center_probs: np.ndarray | None,
    offsets: tuple[float, ...],
    length_scales: tuple[float, ...],
    max_crosswalks: int,
    max_candidates: int,
) -> list[dict[str, Any]]:
    current_stop_lines = list(baseline_prediction.get("stop_lines", []))
    candidates = _crosswalk_edge_candidates_from_crosswalks(
        list(baseline_prediction.get("crosswalks", [])),
        meta=meta,
        offsets=offsets,
        length_scales=length_scales,
        max_crosswalks=int(max_crosswalks),
        max_candidates=int(max_candidates),
    )
    output: list[dict[str, Any]] = []
    for rank, candidate in enumerate(candidates, start=1):
        distance, angle_error, gt_index = _nearest_gt(candidate, gt_stop_lines)
        candidate["nearest_gt_distance"] = float(distance)
        candidate["nearest_gt_angle_error"] = float(angle_error)
        candidate["nearest_gt_index"] = int(gt_index)
        candidate["is_oracle_positive"] = bool(float(distance) <= 40.0)
        candidate["crosswalk_edge_rank_score"] = float(1.0 / max(rank, 1))
        candidate.update(
            _crosswalk_edge_features(
                candidate,
                meta=meta,
                stop_mask_probs=stop_mask_probs,
                stop_center_probs=stop_center_probs,
                stop_selector_probs=stop_selector_probs,
                crosswalk_mask_probs=crosswalk_mask_probs,
                crosswalk_center_probs=crosswalk_center_probs,
                current_stop_lines=current_stop_lines,
                candidate_rank=rank,
            )
        )
        output.append(candidate)
    output.sort(
        key=lambda item: (
            float(item.get("crosswalk_edge_stop_proposal_max", 0.0)),
            float(item.get("crosswalk_edge_rank_score", 0.0)),
            float(item.get("crosswalk_edge_score", 0.0)),
            -float(item.get("crosswalk_edge_offset_px", 0.0)),
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
        "crosswalk_edge_crosswalk_rank": int(candidate.get("crosswalk_edge_crosswalk_rank", 0)),
        "crosswalk_edge_side": float(candidate.get("crosswalk_edge_side", 0.0)),
        "crosswalk_edge_offset_px": float(candidate.get("crosswalk_edge_offset_px", 0.0)),
        "crosswalk_edge_length_scale": float(candidate.get("crosswalk_edge_length_scale", 1.0)),
        "nearest_gt_distance": float(candidate.get("nearest_gt_distance", float("inf"))),
        "nearest_gt_angle_error": float(candidate.get("nearest_gt_angle_error", 180.0)),
        "nearest_gt_index": int(candidate.get("nearest_gt_index", -1)),
        "is_oracle_positive": int(bool(candidate.get("is_oracle_positive", False))),
    }
    for name in CROSSWALK_EDGE_FEATURES:
        row[name] = float(candidate.get(name, 0.0))
    if SCORE_KEY in candidate:
        row[SCORE_KEY] = float(candidate.get(SCORE_KEY, 0.0))
    return row


def _collect_records(
    *,
    loader: Any,
    evaluator: Any,
    postprocess_config: Any,
    max_batches: int,
    offsets: tuple[float, ...],
    length_scales: tuple[float, ...],
    max_crosswalks: int,
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
                print(f"[stopline_crosswalk_edge] collect {split_name} batch {batch_index}/{max_batches}", flush=True)
            raw_batch = raw_batch_for_metrics(batch)
            if raw_batch is None:
                raise ValueError("crosswalk-edge stop-line probe requires raw batches for metrics")
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
                crosswalk_mask_probs = _as_2d_array(
                    _sample_tensor(outputs, "crosswalk_mask_logits", sample_index),
                    sigmoid=True,
                )
                crosswalk_center_probs = _as_2d_array(
                    _sample_tensor(outputs, "crosswalk_center_logits", sample_index),
                    sigmoid=True,
                )
                candidates = _build_crosswalk_edge_candidates(
                    meta=meta,
                    baseline_prediction=baseline_prediction,
                    gt_stop_lines=list(gt_sample.get("stop_lines", [])),
                    stop_mask_probs=stop_mask_probs,
                    stop_center_probs=stop_center_probs,
                    stop_selector_probs=stop_selector_probs,
                    crosswalk_mask_probs=crosswalk_mask_probs,
                    crosswalk_center_probs=crosswalk_center_probs,
                    offsets=offsets,
                    length_scales=length_scales,
                    max_crosswalks=int(max_crosswalks),
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
                        "crosswalk_count": int(len(list(baseline_prediction.get("crosswalks", [])))),
                        "gt_stopline_count": int(len(list(gt_sample.get("stop_lines", [])))),
                        "crosswalk_edge_candidate_count": int(len(candidates)),
                        "crosswalk_edge_oracle_positive_count": int(
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
            features.append([float(candidate.get(name, 0.0)) for name in CROSSWALK_EDGE_FEATURES])
            labels.append(float(bool(candidate.get("is_oracle_positive", False))))
    if not features:
        return np.zeros((0, len(CROSSWALK_EDGE_FEATURES)), dtype=np.float32), np.zeros((0,), dtype=np.float32)
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
        raise ValueError("crosswalk-edge verifier requires non-empty train and validation candidates")
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


def _select_crosswalk_edge_stop_lines(
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
            float(item.get("crosswalk_edge_stop_proposal_max", 0.0)),
            float(item.get("crosswalk_edge_score", 0.0)),
            -float(item.get("crosswalk_edge_offset_px", 0.0)),
        ),
        reverse=True,
    )
    predictions = [
        _copy_stop_line(candidate, score=float(candidate.get(score_key, 0.0)), source="crosswalk_edge")
        for candidate in selected
    ]
    predictions.sort(key=_stopline_prediction_sort_key, reverse=True)
    return _dedupe_stop_line_predictions(predictions)[: max(1, int(max_components))]


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
            stop_lines = _select_crosswalk_edge_stop_lines(
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
    del max_components
    candidate_rows: list[tuple[float, int]] = []
    for record in records:
        for rank, candidate in enumerate(record.get("candidates", []), start=1):
            if rank > int(top_k):
                continue
            candidate_rows.append((float(candidate.get(score_key, 0.0)), int(bool(candidate.get("is_oracle_positive", False)))))
    if not candidate_rows:
        return 0.5
    total_positive = int(sum(label for _score, label in candidate_rows))
    if total_positive <= 0:
        return 1.0
    best_threshold = 0.5
    best_key: tuple[float, int, int] = (-1.0, -1, 0)
    for threshold in np.linspace(0.0, 1.0, max(2, int(grid_size))).tolist():
        tp = 0
        fp = 0
        for score, label in candidate_rows:
            if float(score) < float(threshold):
                continue
            if label:
                tp += 1
            else:
                fp += 1
        fn = max(0, total_positive - tp)
        f1 = 0.0 if tp <= 0 else (2.0 * float(tp)) / (2.0 * float(tp) + float(fp) + float(fn))
        key = (
            float(f1),
            int(tp),
            -int(fp),
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
    offsets = _parse_float_list(str(args.crosswalk_edge_offsets), name="crosswalk-edge-offsets", min_value=0.0)
    length_scales = _parse_float_list(
        str(args.crosswalk_edge_length_scales),
        name="crosswalk-edge-length-scales",
        min_value=0.1,
    )
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
        progress_callback=lambda message: print(f"[stopline_crosswalk_edge] {message}", flush=True),
    )
    train_loader, val_loader = train_cli._build_phase_train_loaders(dataset, train_config=train_config, phase=phase)
    if train_loader is None or val_loader is None:
        raise ValueError("crosswalk-edge stop-line probe requires train and validation loaders")
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
        offsets=offsets,
        length_scales=length_scales,
        max_crosswalks=int(args.max_crosswalks),
        max_candidates=int(args.max_crosswalk_edge_candidates),
        split_name="train",
    )
    val_records, val_candidate_rows, val_sample_rows = _collect_records(
        loader=val_loader,
        evaluator=evaluator,
        postprocess_config=postprocess_config,
        max_batches=int(args.max_val_batches),
        offsets=offsets,
        length_scales=length_scales,
        max_crosswalks=int(args.max_crosswalks),
        max_candidates=int(args.max_crosswalk_edge_candidates),
        split_name="val",
    )
    score_summary = _score_records(
        train_records=train_records,
        val_records=val_records,
        top_k=int(args.crosswalk_edge_top_k),
        epochs=int(args.verifier_epochs),
        lr=float(args.verifier_lr),
    )
    threshold = _best_threshold(
        train_records,
        score_key=SCORE_KEY,
        top_k=int(args.crosswalk_edge_top_k),
        max_components=int(args.max_components),
        grid_size=int(args.threshold_grid),
    )
    rows = [
        _metrics_row(train_records, name="baseline", split="train"),
        _metrics_row(val_records, name="baseline", split="val"),
        _metrics_row(
            train_records,
            name="crosswalk_edge_mlp",
            split="train",
            score_key=SCORE_KEY,
            threshold=threshold,
            top_k=int(args.crosswalk_edge_top_k),
            max_components=int(args.max_components),
        ),
        _metrics_row(
            val_records,
            name="crosswalk_edge_mlp",
            split="val",
            score_key=SCORE_KEY,
            threshold=threshold,
            top_k=int(args.crosswalk_edge_top_k),
            max_components=int(args.max_components),
        ),
        _metrics_row(
            train_records,
            name="baseline_plus_crosswalk_edge_mlp",
            split="train",
            score_key=SCORE_KEY,
            threshold=threshold,
            top_k=int(args.crosswalk_edge_top_k),
            max_components=int(args.max_components),
            union_baseline=True,
        ),
        _metrics_row(
            val_records,
            name="baseline_plus_crosswalk_edge_mlp",
            split="val",
            score_key=SCORE_KEY,
            threshold=threshold,
            top_k=int(args.crosswalk_edge_top_k),
            max_components=int(args.max_components),
            union_baseline=True,
        ),
        _metrics_row(
            val_records,
            name="oracle_crosswalk_edge",
            split="val",
            score_key="is_oracle_positive",
            threshold=0.5,
            top_k=int(args.crosswalk_edge_top_k),
            max_components=int(args.max_components),
        ),
        _metrics_row(
            val_records,
            name="baseline_plus_oracle_crosswalk_edge",
            split="val",
            score_key="is_oracle_positive",
            threshold=0.5,
            top_k=int(args.crosswalk_edge_top_k),
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
        "crosswalk_edge_offsets": [float(value) for value in offsets],
        "crosswalk_edge_length_scales": [float(value) for value in length_scales],
        "max_crosswalks": int(args.max_crosswalks),
        "crosswalk_edge_top_k": int(args.crosswalk_edge_top_k),
        "max_crosswalk_edge_candidates": int(args.max_crosswalk_edge_candidates),
        "max_components": int(args.max_components),
        "threshold": float(threshold),
        "threshold_selection": "train_candidate_oracle_label_f1",
        "score_summary": score_summary,
        "train_sample_count": int(len(train_records)),
        "val_sample_count": int(len(val_records)),
        "train_crosswalk_edge_candidate_count": int(len(train_candidate_rows)),
        "val_crosswalk_edge_candidate_count": int(len(val_candidate_rows)),
        "train_crosswalk_edge_oracle_positive_count": int(
            sum(int(row.get("is_oracle_positive", 0)) for row in train_candidate_rows)
        ),
        "val_crosswalk_edge_oracle_positive_count": int(
            sum(int(row.get("is_oracle_positive", 0)) for row in val_candidate_rows)
        ),
        "rows": rows,
        "interpretation": (
            "Crosswalk-edge stop-line candidate probe. Runtime candidates are generated from "
            "predicted crosswalk hull long-edges and current-frame dense maps; GT is used for train labels, "
            "oracle diagnostics, and final metrics only. This tests crosswalk-conditioned candidate generation, "
            "not layer-level crosswalk context fusion or another projection-comp threshold."
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
    _write_csv(output_dir / "crosswalk_edge_variants.csv", payload["rows"])
    _write_candidate_features_csv(output_dir / "train_candidate_features.csv", payload["train_candidate_rows"])
    _write_candidate_features_csv(output_dir / "val_candidate_features.csv", payload["val_candidate_rows"])
    _write_csv(output_dir / "crosswalk_edge_samples.csv", payload["sample_rows"])
    (output_dir / "summary.json").write_text(
        json.dumps(payload["summary"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
