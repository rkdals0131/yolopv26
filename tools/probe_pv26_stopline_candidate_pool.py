from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, replace
import json
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
from model.engine.metrics import (
    STOP_LINE_POINT_COUNT,
    _extract_gt_samples,
    _hungarian_from_cost,
    _mean_point_distance,
    _segment_angle_error,
    summarize_pv26_metrics,
)
from model.engine.postprocess import _dedupe_stop_line_predictions, _stopline_prediction_sort_key, postprocess_pv26_batch
from tools.evaluate_pv26_lane60_checkpoint import _advance_validation_sampler
from tools.probe_pv26_stopline_angle_mask_extent import (
    ExtentVariant,
    _as_2d_array,
    _detach_to_cpu,
    _mask_extent_line,
    _row_from_metrics,
    _sample_tensor,
    _write_csv,
)
from tools.probe_pv26_stopline_pred_angle_mask_extent import _proposal_map, _top_cells
from tools.pv26_train import cli as train_cli
from tools.pv26_train import config as train_config_api


SOURCE_RUN = (
    REPO_ROOT
    / "runs"
    / "pv26_exhaustive_od_lane_train"
    / "lane60_core_centerline_refine_cross_retain_from_exhaustive_od_lane_default_20260505_032217_default_20260510_003412"
)
DEFAULT_CHECKPOINT = SOURCE_RUN / "phase_4" / "checkpoints" / "best.pt"


@dataclass(frozen=True)
class CandidateVariant:
    name: str
    source: str
    top_k: int
    sort_mode: str
    min_score: float = 0.0
    min_length: float = 0.0
    oracle_positive_only: bool = False


VARIANTS = (
    CandidateVariant("max_top10_score", "max", top_k=10, sort_mode="score"),
    CandidateVariant("max_top10_longest", "max", top_k=10, sort_mode="length"),
    CandidateVariant("max_top10_score_len8", "max", top_k=10, sort_mode="score", min_length=8.0),
    CandidateVariant("max_top10_score_len12", "max", top_k=10, sort_mode="score", min_length=12.0),
    CandidateVariant("max_top10_score_s080", "max", top_k=10, sort_mode="score", min_score=0.80),
    CandidateVariant("max_top20_score_len8", "max", top_k=20, sort_mode="score", min_length=8.0),
    CandidateVariant("oracle_max_top10_positive", "max", top_k=10, sort_mode="distance", oracle_positive_only=True),
    CandidateVariant("oracle_max_top20_positive", "max", top_k=20, sort_mode="distance", oracle_positive_only=True),
)

BASE_VALIDATOR_FEATURES = (
    "score",
    "length",
    "proposal_rank",
    "inverse_rank",
    "score_x_length",
    "log1p_length",
)
RICH_VALIDATOR_EXTRA_FEATURES = (
    "proposal_row",
    "proposal_col",
    "decoded_center_row",
    "decoded_center_col",
    "center_prob",
    "selector_prob",
    "fused_center_selector_prob",
    "mask_prob",
    "decoded_center_mask_prob",
    "center_r1_max",
    "center_r1_mean",
    "center_r2_max",
    "center_r2_mean",
    "center_r4_max",
    "center_r4_mean",
    "selector_r1_max",
    "selector_r1_mean",
    "selector_r2_max",
    "selector_r2_mean",
    "selector_r4_max",
    "selector_r4_mean",
    "mask_r1_max",
    "mask_r1_mean",
    "mask_r2_max",
    "mask_r2_mean",
    "mask_r4_max",
    "mask_r4_mean",
)
RICH_VALIDATOR_FEATURES = BASE_VALIDATOR_FEATURES + RICH_VALIDATOR_EXTRA_FEATURES
RAW_BATCH_KEYS = ("det_targets", "tl_attr_targets", "lane_targets", "source_mask", "valid_mask", "meta")
CANDIDATE_FEATURE_FIELDNAMES = (
    "batch_index",
    "sample_index",
    "sample_id",
    "dataset_key",
    "image_path",
    "gt_stop_line_count",
    "gt_stop_line_points_json",
    "proposal_source",
    "proposal_min_gap",
    "proposal_rank",
    "proposal_row",
    "proposal_col",
    "decoded_center_row",
    "decoded_center_col",
    "candidate_points_json",
    "score",
    "length",
    "component_svd_length",
    "center_prob",
    "selector_prob",
    "fused_center_selector_prob",
    "mask_prob",
    "decoded_center_mask_prob",
    "center_r1_max",
    "center_r1_mean",
    "center_r2_max",
    "center_r2_mean",
    "center_r4_max",
    "center_r4_mean",
    "selector_r1_max",
    "selector_r1_mean",
    "selector_r2_max",
    "selector_r2_mean",
    "selector_r4_max",
    "selector_r4_mean",
    "mask_r1_max",
    "mask_r1_mean",
    "mask_r2_max",
    "mask_r2_mean",
    "mask_r4_max",
    "mask_r4_mean",
    "nearest_gt_distance",
    "nearest_gt_angle_error",
    "nearest_gt_index",
    "is_oracle_positive",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a top-k stop-line proposal candidate pool and audit whether non-GT feature "
            "filters or oracle candidate validation can recover stop-line F1."
        )
    )
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--preset", default="default")
    parser.add_argument("--phase-index", type=int, default=4)
    parser.add_argument("--max-val-batches", type=int, default=128)
    parser.add_argument("--validation-epoch", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dataset-root", default="", help="Override dataset root for detached worktrees.")
    parser.add_argument("--proposal-min-gap", type=float, default=4.0)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--rich-validator-replay", action="store_true")
    parser.add_argument("--rich-validator-train-fraction", type=float, default=0.5)
    parser.add_argument("--rich-validator-steps", type=int, default=2000)
    parser.add_argument("--rich-validator-lr", type=float, default=0.05)
    parser.add_argument("--rich-validator-top-k", type=int, default=20)
    parser.add_argument("--rich-validator-threshold-grid", type=int, default=101)
    return parser.parse_args()


def _resolve_device(requested: str, scenario_device: str) -> str:
    value = str(requested or "auto").strip().lower()
    candidate = str(scenario_device) if value == "auto" else str(requested)
    if candidate.startswith("cuda") and not torch.cuda.is_available():
        print("[stopline_candidate_pool] CUDA requested but unavailable; falling back to CPU", flush=True)
        return "cpu"
    return candidate


def _nearest_gt(candidate: dict[str, Any], gt_stop_lines: list[dict[str, Any]]) -> tuple[float, float, int]:
    points = candidate.get("points_xy", [])
    best_distance = float("inf")
    best_angle = 180.0
    best_index = -1
    for index, gt in enumerate(gt_stop_lines):
        gt_points = gt.get("points_xy", [])
        distance = _mean_point_distance(points, gt_points, STOP_LINE_POINT_COUNT)
        if distance < best_distance:
            best_distance = float(distance)
            best_angle = float(_segment_angle_error(points, gt_points, STOP_LINE_POINT_COUNT))
            best_index = int(index)
    return best_distance, best_angle, best_index


def _array_value(array: np.ndarray | None, row: int, col: int, default: float = 0.0) -> float:
    if array is None or array.ndim != 2:
        return float(default)
    clipped_row = max(0, min(int(array.shape[0]) - 1, int(row)))
    clipped_col = max(0, min(int(array.shape[1]) - 1, int(col)))
    return float(array[clipped_row, clipped_col])


def _window_stats(array: np.ndarray | None, row: int, col: int, radius: int) -> tuple[float, float]:
    if array is None or array.ndim != 2:
        return 0.0, 0.0
    clipped_row = max(0, min(int(array.shape[0]) - 1, int(row)))
    clipped_col = max(0, min(int(array.shape[1]) - 1, int(col)))
    radius = max(0, int(radius))
    row0 = max(0, clipped_row - radius)
    row1 = min(int(array.shape[0]), clipped_row + radius + 1)
    col0 = max(0, clipped_col - radius)
    col1 = min(int(array.shape[1]), clipped_col + radius + 1)
    window = array[row0:row1, col0:col1]
    if window.size == 0:
        return 0.0, 0.0
    return float(window.max(initial=0.0)), float(window.mean())


def _decode_candidates(
    *,
    outputs: dict[str, Any],
    sample_index: int,
    meta: dict[str, Any],
    gt_stop_lines: list[dict[str, Any]],
    source: str,
    top_k: int,
    min_gap: float,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    mask_probs = _as_2d_array(_sample_tensor(outputs, "stop_line_mask_logits", sample_index), sigmoid=True)
    angle_map = _sample_tensor(outputs, "stop_line_angle", sample_index)
    offset_map = _sample_tensor(outputs, "stop_line_center_offset", sample_index)
    proposal_map = _proposal_map(outputs, sample_index, source)
    center_probs = _as_2d_array(_sample_tensor(outputs, "stop_line_center_logits", sample_index), sigmoid=True)
    selector_probs = _as_2d_array(_sample_tensor(outputs, "stop_line_selector_map_logits", sample_index), sigmoid=True)
    if mask_probs is None or proposal_map is None or angle_map is None or offset_map is None:
        return [], {"missing_tensor": 1}
    if angle_map.ndim != 3 or offset_map.ndim != 3:
        return [], {"bad_tensor_shape": 1}
    output_hw = (int(mask_probs.shape[0]), int(mask_probs.shape[1]))
    extent_variant = ExtentVariant(
        name="candidate_pool",
        center_source="pred_center",
        angle_source="pred_angle",
        mask_threshold=0.50,
        normal_band=4.0,
    )
    stats: dict[str, int] = {}
    candidates: list[dict[str, Any]] = []
    top_cells = _top_cells(proposal_map, top_k=int(top_k), threshold=0.0, min_gap=float(min_gap))
    for proposal_rank, (row, col, score) in enumerate(top_cells, start=1):
        pred_offset = offset_map[:, row, col].numpy().astype(np.float32)
        center_xy = np.asarray([float(col) + pred_offset[0], float(row) + pred_offset[1]], dtype=np.float32)
        pred_angle = angle_map[:, row, col].numpy().astype(np.float32)
        candidate, reason = _mask_extent_line(
            mask_probs=mask_probs,
            center_xy=center_xy,
            angle_vec=pred_angle,
            meta=meta,
            output_hw=output_hw,
            variant=extent_variant,
            score=float(score),
        )
        stats[reason] = stats.get(reason, 0) + 1
        if candidate is None:
            continue
        nearest_distance, nearest_angle, nearest_index = _nearest_gt(candidate, gt_stop_lines)
        candidate = dict(candidate)
        candidate["proposal_rank"] = int(proposal_rank)
        candidate["proposal_source"] = str(source)
        candidate["proposal_min_gap"] = float(min_gap)
        candidate["proposal_row"] = int(row)
        candidate["proposal_col"] = int(col)
        center_row = int(round(float(center_xy[1])))
        center_col = int(round(float(center_xy[0])))
        candidate["decoded_center_row"] = int(center_row)
        candidate["decoded_center_col"] = int(center_col)
        candidate["center_prob"] = _array_value(center_probs, row, col)
        candidate["selector_prob"] = _array_value(selector_probs, row, col)
        candidate["mask_prob"] = _array_value(mask_probs, row, col)
        candidate["decoded_center_mask_prob"] = _array_value(mask_probs, center_row, center_col)
        candidate["fused_center_selector_prob"] = max(
            float(candidate["center_prob"]),
            float(candidate["selector_prob"]),
        )
        for radius in (1, 2, 4):
            center_max, center_mean = _window_stats(center_probs, row, col, radius)
            selector_max, selector_mean = _window_stats(selector_probs, row, col, radius)
            mask_max, mask_mean = _window_stats(mask_probs, row, col, radius)
            candidate[f"center_r{radius}_max"] = center_max
            candidate[f"center_r{radius}_mean"] = center_mean
            candidate[f"selector_r{radius}_max"] = selector_max
            candidate[f"selector_r{radius}_mean"] = selector_mean
            candidate[f"mask_r{radius}_max"] = mask_max
            candidate[f"mask_r{radius}_mean"] = mask_mean
        candidate["nearest_gt_distance"] = float(nearest_distance)
        candidate["nearest_gt_angle_error"] = float(nearest_angle)
        candidate["nearest_gt_index"] = int(nearest_index)
        candidate["is_oracle_positive"] = bool(nearest_distance <= 40.0)
        candidates.append(candidate)
    return candidates, stats


def _sort_key(candidate: dict[str, Any], mode: str) -> tuple[float, ...]:
    if mode == "score":
        return (float(candidate.get("score", 0.0)), float(candidate.get("length", 0.0)))
    if mode == "length":
        return (float(candidate.get("length", 0.0)), float(candidate.get("score", 0.0)))
    if mode == "distance":
        return (-float(candidate.get("nearest_gt_distance", float("inf"))), float(candidate.get("score", 0.0)))
    raise ValueError(f"unsupported sort mode: {mode}")


def _select_stop_lines(
    candidates: list[dict[str, Any]],
    *,
    variant: CandidateVariant,
    max_components: int,
) -> list[dict[str, Any]]:
    selected = [
        candidate
        for candidate in candidates
        if int(candidate.get("proposal_rank", 10**6)) <= int(variant.top_k)
        and float(candidate.get("score", 0.0)) >= float(variant.min_score)
        and float(candidate.get("length", 0.0)) >= float(variant.min_length)
        and (not bool(variant.oracle_positive_only) or bool(candidate.get("is_oracle_positive", False)))
    ]
    selected.sort(key=lambda candidate: _sort_key(candidate, variant.sort_mode), reverse=True)
    predictions = [
        {
            "score": float(candidate.get("score", 0.0)),
            "center_score": float(candidate.get("center_score", candidate.get("score", 0.0))),
            "length": float(candidate.get("length", 0.0)),
            "points_xy": candidate.get("points_xy", []),
        }
        for candidate in selected
    ]
    predictions.sort(key=_stopline_prediction_sort_key, reverse=True)
    predictions = _dedupe_stop_line_predictions(predictions)
    return predictions[: max(1, int(max_components))]


def _float_feature(candidate: dict[str, Any], name: str, default: float = 0.0) -> float:
    try:
        return float(candidate.get(name, default))
    except (TypeError, ValueError):
        return float(default)


def _rich_feature_vector(candidate: dict[str, Any]) -> list[float]:
    score = _float_feature(candidate, "score")
    length = _float_feature(candidate, "length")
    rank = max(1.0, _float_feature(candidate, "proposal_rank", 1.0))
    values = [
        score,
        length,
        rank,
        1.0 / rank,
        score * length,
        float(np.log1p(max(0.0, length))),
    ]
    values.extend(_float_feature(candidate, name) for name in RICH_VALIDATOR_EXTRA_FEATURES)
    return values


def _rich_feature_matrix(candidates: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    matrix = [_rich_feature_vector(candidate) for candidate in candidates]
    labels = [float(bool(candidate.get("is_oracle_positive", False))) for candidate in candidates]
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


def _threshold_metrics(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict[str, float | int]:
    predicted = scores >= float(threshold)
    actual = labels > 0.5
    tp = int(np.logical_and(predicted, actual).sum())
    fp = int(np.logical_and(predicted, ~actual).sum())
    fn = int(np.logical_and(~predicted, actual).sum())
    precision = float(tp / max(1, tp + fp))
    recall = float(tp / max(1, tp + fn))
    f1 = float(2.0 * precision * recall / max(1.0e-12, precision + recall))
    return {
        "threshold": float(threshold),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "predicted_positive_count": int(predicted.sum()),
    }


def _best_row_threshold(scores: np.ndarray, labels: np.ndarray, *, grid_size: int) -> dict[str, float | int]:
    if scores.size == 0:
        return _threshold_metrics(scores, labels, 1.0)
    thresholds = np.unique(np.quantile(scores, np.linspace(0.0, 1.0, max(2, int(grid_size)))))
    best = _threshold_metrics(scores, labels, float(thresholds[0]))
    for threshold in thresholds:
        metrics = _threshold_metrics(scores, labels, float(threshold))
        if (float(metrics["f1"]), float(metrics["precision"])) > (float(best["f1"]), float(best["precision"])):
            best = metrics
    return best


def _slice_raw_batch_sample(raw_batch: dict[str, Any], sample_index: int) -> dict[str, Any]:
    return {key: [raw_batch[key][sample_index]] for key in RAW_BATCH_KEYS}


def _validator_stop_lines(
    candidates: list[dict[str, Any]],
    *,
    score_key: str,
    threshold: float,
    top_k: int,
    max_components: int,
) -> list[dict[str, Any]]:
    selected = [
        candidate
        for candidate in candidates
        if int(candidate.get("proposal_rank", 10**6)) <= int(top_k)
        and _float_feature(candidate, score_key, 0.0) >= float(threshold)
    ]
    selected.sort(
        key=lambda candidate: (
            _float_feature(candidate, score_key, 0.0),
            _float_feature(candidate, "score", 0.0),
            _float_feature(candidate, "length", 0.0),
        ),
        reverse=True,
    )
    predictions = [
        {
            "score": _float_feature(candidate, score_key, 0.0),
            "center_score": _float_feature(candidate, score_key, 0.0),
            "length": _float_feature(candidate, "length", 0.0),
            "points_xy": candidate.get("points_xy", []),
        }
        for candidate in selected
    ]
    predictions.sort(key=_stopline_prediction_sort_key, reverse=True)
    predictions = _dedupe_stop_line_predictions(predictions)
    return predictions[: max(1, int(max_components))]


def _records_metrics_row(
    records: list[dict[str, Any]],
    *,
    name: str,
    split: str,
    score_key: str = "",
    threshold: float = 0.0,
    top_k: int = 0,
    max_components: int = 1,
) -> dict[str, Any]:
    if not records:
        raise ValueError(f"cannot evaluate empty {split} split")
    predictions: list[dict[str, Any]] = []
    raw_batches: list[dict[str, Any]] = []
    for record in records:
        raw_batches.append(record["raw_batch"])
        if score_key:
            stop_lines = _validator_stop_lines(
                list(record.get("candidates", [])),
                score_key=score_key,
                threshold=float(threshold),
                top_k=int(top_k),
                max_components=int(max_components),
            )
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
            "sample_count": int(len(records)),
        }
    )
    return row


def _stop_line_fast_summary(
    records: list[dict[str, Any]],
    *,
    score_key: str,
    threshold: float,
    top_k: int,
    max_components: int,
) -> dict[str, float | int]:
    total_tp = 0
    total_fp = 0
    total_fn = 0
    prediction_count = 0
    for record in records:
        pred_rows = _validator_stop_lines(
            list(record.get("candidates", [])),
            score_key=score_key,
            threshold=float(threshold),
            top_k=int(top_k),
            max_components=int(max_components),
        )
        gt_rows = list(record.get("gt_stop_lines", []))
        prediction_count += len(pred_rows)
        if pred_rows and gt_rows:
            cost_matrix = np.zeros((len(pred_rows), len(gt_rows)), dtype=np.float32)
            for pred_index, pred in enumerate(pred_rows):
                for gt_index, gt in enumerate(gt_rows):
                    cost_matrix[pred_index, gt_index] = _mean_point_distance(
                        pred["points_xy"],
                        gt["points_xy"],
                        STOP_LINE_POINT_COUNT,
                    )
            matches = _hungarian_from_cost(cost_matrix, max_cost=40.0)
        else:
            matches = []
        total_tp += len(matches)
        total_fp += len(pred_rows) - len(matches)
        total_fn += len(gt_rows) - len(matches)
    precision = float(total_tp / max(1, total_tp + total_fp))
    recall = float(total_tp / max(1, total_tp + total_fn))
    f1 = float(2.0 * precision * recall / max(1.0e-12, precision + recall))
    return {
        "stop_line_precision": precision,
        "stop_line_recall": recall,
        "stop_line_f1": f1,
        "stop_line_tp": int(total_tp),
        "stop_line_fp": int(total_fp),
        "stop_line_fn": int(total_fn),
        "pred_stop_line_count": int(prediction_count),
    }


def _best_task_threshold(
    records: list[dict[str, Any]],
    *,
    score_key: str,
    top_k: int,
    max_components: int,
    grid_size: int,
) -> dict[str, Any]:
    scores = [
        _float_feature(candidate, score_key, 0.0)
        for record in records
        for candidate in record.get("candidates", [])
        if int(candidate.get("proposal_rank", 10**6)) <= int(top_k)
    ]
    if not scores:
        raise ValueError(f"no candidates available for {score_key} threshold search")
    thresholds = np.unique(np.quantile(np.asarray(scores, dtype=np.float64), np.linspace(0.0, 1.0, max(2, int(grid_size)))))
    thresholds = np.unique(np.concatenate([thresholds, np.asarray([float(max(scores)) + 1.0e-6], dtype=np.float64)]))
    best_row: dict[str, Any] | None = None
    for threshold in thresholds:
        row = _stop_line_fast_summary(
            records,
            score_key=score_key,
            threshold=float(threshold),
            top_k=int(top_k),
            max_components=int(max_components),
        )
        key = (
            float(row.get("stop_line_f1", 0.0)),
            float(row.get("stop_line_precision", 0.0)),
            -float(row.get("pred_stop_line_count", 0.0)),
        )
        best_key = (
            float(best_row.get("stop_line_f1", 0.0)),
            float(best_row.get("stop_line_precision", 0.0)),
            -float(best_row.get("pred_stop_line_count", 0.0)),
        ) if best_row is not None else (-1.0, -1.0, 0.0)
        if key > best_key:
            best_row = {
                **row,
                "variant": f"{score_key}_task_threshold",
                "split": "train",
                "score_key": str(score_key),
                "threshold": float(threshold),
                "top_k": int(top_k),
                "sample_count": int(len(records)),
            }
    if best_row is None:
        raise ValueError(f"failed to select threshold for {score_key}")
    return best_row


def _run_rich_validator_replay(
    sample_records: list[dict[str, Any]],
    *,
    train_fraction: float,
    steps: int,
    lr: float,
    top_k: int,
    max_components: int,
    threshold_grid: int,
) -> dict[str, Any]:
    if not sample_records:
        raise ValueError("rich validator replay requires sample records")
    batch_indices = np.asarray([int(record["batch_index"]) for record in sample_records], dtype=np.int64)
    min_batch = int(batch_indices.min())
    max_batch = int(batch_indices.max())
    cutoff = min_batch + int(round((max_batch - min_batch + 1) * float(train_fraction))) - 1
    train_records = [record for record in sample_records if int(record["batch_index"]) <= cutoff]
    test_records = [record for record in sample_records if int(record["batch_index"]) > cutoff]
    if not train_records or not test_records:
        raise ValueError("rich validator replay requires non-empty train and held-out splits")

    train_candidates = [
        candidate
        for record in train_records
        for candidate in record.get("candidates", [])
        if int(candidate.get("proposal_rank", 10**6)) <= int(top_k)
    ]
    if not train_candidates:
        raise ValueError("rich validator replay found no train candidates")
    train_x, train_y = _rich_feature_matrix(train_candidates)
    train_x_std, mean, std = _standardize(train_x, train_x)
    weights, bias = _fit_logistic(train_x_std, train_y, steps=int(steps), lr=float(lr))

    all_candidates = [
        candidate
        for record in sample_records
        for candidate in record.get("candidates", [])
        if int(candidate.get("proposal_rank", 10**6)) <= int(top_k)
    ]
    all_x, _ = _rich_feature_matrix(all_candidates)
    all_scores = _predict((all_x - mean) / std, weights, bias)
    for candidate, score in zip(all_candidates, all_scores):
        candidate["rich_logistic_score"] = float(score)
        candidate["selector_r4_max_score"] = _float_feature(candidate, "selector_r4_max", 0.0)

    train_scores = np.asarray([_float_feature(candidate, "rich_logistic_score", 0.0) for candidate in train_candidates])
    row_threshold = float(_best_row_threshold(train_scores, train_y, grid_size=int(threshold_grid))["threshold"])
    task_threshold_row = _best_task_threshold(
        train_records,
        score_key="rich_logistic_score",
        top_k=int(top_k),
        max_components=int(max_components),
        grid_size=int(threshold_grid),
    )
    selector_task_threshold_row = _best_task_threshold(
        train_records,
        score_key="selector_r4_max_score",
        top_k=int(top_k),
        max_components=int(max_components),
        grid_size=int(threshold_grid),
    )
    task_threshold = float(task_threshold_row["threshold"])
    selector_task_threshold = float(selector_task_threshold_row["threshold"])

    rows: list[dict[str, Any]] = [
        _records_metrics_row(train_records, name="baseline", split="train"),
        _records_metrics_row(test_records, name="baseline", split="heldout"),
        _records_metrics_row(
            train_records,
            name="rich_logistic_task_threshold",
            split="train",
            score_key="rich_logistic_score",
            threshold=task_threshold,
            top_k=int(top_k),
            max_components=int(max_components),
        ),
        _records_metrics_row(
            test_records,
            name="rich_logistic_task_threshold",
            split="heldout",
            score_key="rich_logistic_score",
            threshold=task_threshold,
            top_k=int(top_k),
            max_components=int(max_components),
        ),
        _records_metrics_row(
            train_records,
            name="rich_logistic_row_threshold",
            split="train",
            score_key="rich_logistic_score",
            threshold=row_threshold,
            top_k=int(top_k),
            max_components=int(max_components),
        ),
        _records_metrics_row(
            test_records,
            name="rich_logistic_row_threshold",
            split="heldout",
            score_key="rich_logistic_score",
            threshold=row_threshold,
            top_k=int(top_k),
            max_components=int(max_components),
        ),
        _records_metrics_row(
            train_records,
            name="selector_r4_max_task_threshold",
            split="train",
            score_key="selector_r4_max_score",
            threshold=selector_task_threshold,
            top_k=int(top_k),
            max_components=int(max_components),
        ),
        _records_metrics_row(
            test_records,
            name="selector_r4_max_task_threshold",
            split="heldout",
            score_key="selector_r4_max_score",
            threshold=selector_task_threshold,
            top_k=int(top_k),
            max_components=int(max_components),
        ),
    ]
    return {
        "split": {
            "train_fraction": float(train_fraction),
            "cutoff_batch_index": int(cutoff),
            "train_samples": int(len(train_records)),
            "heldout_samples": int(len(test_records)),
            "train_candidate_count": int(len(train_candidates)),
            "train_positive_count": int(train_y.sum()),
        },
        "feature_names": list(RICH_VALIDATOR_FEATURES),
        "weights": {name: float(weight) for name, weight in zip(RICH_VALIDATOR_FEATURES, weights)},
        "bias": float(bias),
        "top_k": int(top_k),
        "max_components": int(max_components),
        "threshold_grid": int(threshold_grid),
        "rows": rows,
        "interpretation": (
            "Rich-validator replay is a held-out read-only task probe. It trains thresholds on the "
            "earlier validation half and evaluates task F1 on the later half; it is not a production decoder."
        ),
    }


def _points_json(points: Any) -> str:
    try:
        array = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    except (TypeError, ValueError):
        array = np.zeros((0, 2), dtype=np.float32)
    return json.dumps([[float(x), float(y)] for x, y in array.tolist()], separators=(",", ":"))


def _candidate_feature_rows(
    candidates: list[dict[str, Any]],
    *,
    batch_index: int,
    sample_index: int,
    meta: dict[str, Any] | None = None,
    gt_stop_lines: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    meta = meta or {}
    gt_stop_lines = gt_stop_lines or []
    gt_points_json = json.dumps(
        [json.loads(_points_json(gt.get("points_xy", []))) for gt in gt_stop_lines],
        separators=(",", ":"),
    )
    for candidate in candidates:
        rows.append(
            {
                "batch_index": int(batch_index),
                "sample_index": int(sample_index),
                "sample_id": str(meta.get("sample_id", "")),
                "dataset_key": str(meta.get("dataset_key", "")),
                "image_path": str(meta.get("image_path", "")),
                "gt_stop_line_count": int(len(gt_stop_lines)),
                "gt_stop_line_points_json": gt_points_json,
                "proposal_source": str(candidate.get("proposal_source", "")),
                "proposal_min_gap": float(candidate.get("proposal_min_gap", 4.0)),
                "proposal_rank": int(candidate.get("proposal_rank", 0)),
                "proposal_row": int(candidate.get("proposal_row", -1)),
                "proposal_col": int(candidate.get("proposal_col", -1)),
                "decoded_center_row": int(candidate.get("decoded_center_row", -1)),
                "decoded_center_col": int(candidate.get("decoded_center_col", -1)),
                "candidate_points_json": _points_json(candidate.get("points_xy", [])),
                "score": float(candidate.get("score", 0.0)),
                "length": float(candidate.get("length", 0.0)),
                "component_svd_length": float(candidate.get("component_svd_length", candidate.get("length", 0.0))),
                "center_prob": float(candidate.get("center_prob", 0.0)),
                "selector_prob": float(candidate.get("selector_prob", 0.0)),
                "fused_center_selector_prob": float(candidate.get("fused_center_selector_prob", 0.0)),
                "mask_prob": float(candidate.get("mask_prob", 0.0)),
                "decoded_center_mask_prob": float(candidate.get("decoded_center_mask_prob", 0.0)),
                "center_r1_max": float(candidate.get("center_r1_max", 0.0)),
                "center_r1_mean": float(candidate.get("center_r1_mean", 0.0)),
                "center_r2_max": float(candidate.get("center_r2_max", 0.0)),
                "center_r2_mean": float(candidate.get("center_r2_mean", 0.0)),
                "center_r4_max": float(candidate.get("center_r4_max", 0.0)),
                "center_r4_mean": float(candidate.get("center_r4_mean", 0.0)),
                "selector_r1_max": float(candidate.get("selector_r1_max", 0.0)),
                "selector_r1_mean": float(candidate.get("selector_r1_mean", 0.0)),
                "selector_r2_max": float(candidate.get("selector_r2_max", 0.0)),
                "selector_r2_mean": float(candidate.get("selector_r2_mean", 0.0)),
                "selector_r4_max": float(candidate.get("selector_r4_max", 0.0)),
                "selector_r4_mean": float(candidate.get("selector_r4_mean", 0.0)),
                "mask_r1_max": float(candidate.get("mask_r1_max", 0.0)),
                "mask_r1_mean": float(candidate.get("mask_r1_mean", 0.0)),
                "mask_r2_max": float(candidate.get("mask_r2_max", 0.0)),
                "mask_r2_mean": float(candidate.get("mask_r2_mean", 0.0)),
                "mask_r4_max": float(candidate.get("mask_r4_max", 0.0)),
                "mask_r4_mean": float(candidate.get("mask_r4_mean", 0.0)),
                "nearest_gt_distance": float(candidate.get("nearest_gt_distance", float("inf"))),
                "nearest_gt_angle_error": float(candidate.get("nearest_gt_angle_error", 180.0)),
                "nearest_gt_index": int(candidate.get("nearest_gt_index", -1)),
                "is_oracle_positive": bool(candidate.get("is_oracle_positive", False)),
            }
        )
    return rows


def _scenario_with_dataset_root(
    scenario: train_config_api.MetaTrainScenario,
    dataset_root: str,
) -> train_config_api.MetaTrainScenario:
    value = str(dataset_root or "").strip()
    if not value:
        return scenario
    dataset = train_config_api.DatasetConfig(root=Path(value).expanduser().resolve())
    return replace(scenario, dataset=dataset)


def _write_candidate_features_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = list(CANDIDATE_FEATURE_FIELDNAMES)
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")

    scenario = train_cli.load_meta_train_scenario(args.preset)
    scenario = _scenario_with_dataset_root(scenario, str(args.dataset_root))
    phase = scenario.phases[int(args.phase_index) - 1]
    train_config = train_config_api.scenario_phase_defaults(scenario.train_defaults, phase.overrides)
    train_config = replace(
        train_config,
        device=_resolve_device(args.device, train_config.device),
        val_batches=int(args.max_val_batches),
    )
    postprocess_config = train_cli._build_postprocess_config(train_config)
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if str(args.output_dir).strip()
        else checkpoint.parents[2] / "analysis_exports" / "stopline_candidate_pool_val128_epoch2"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    train_cli._configure_torch_multiprocessing()
    dataset = train_cli.PV26CanonicalDataset(
        train_cli._existing_dataset_roots(scenario),
        train_augmentation=False,
        progress_callback=lambda message: print(f"[stopline_candidate_pool] {message}", flush=True),
    )
    _, val_loader = train_cli._build_phase_train_loaders(dataset, train_config=train_config, phase=phase)
    if val_loader is None:
        raise ValueError("stop-line candidate pool probe requires validation batches")
    _advance_validation_sampler(val_loader, validation_epoch=int(args.validation_epoch))

    trainer = train_cli._build_phase_trainer(phase, train_config)
    trainer.load_model_weights(checkpoint, map_location=train_config.device)
    evaluator = trainer.build_evaluator()
    evaluator.adapter.raw_model.eval()
    evaluator.heads.eval()

    names = ("baseline",) + tuple(variant.name for variant in VARIANTS)
    predictions_by_variant: dict[str, list[dict[str, Any]]] = {name: [] for name in names}
    stats_by_variant: dict[str, dict[str, int]] = {name: {} for name in names}
    feature_rows: list[dict[str, Any]] = []
    raw_batches: list[dict[str, Any]] = []
    sample_records: list[dict[str, Any]] = []
    processed_batches = 0
    max_top_k = max(
        max(int(variant.top_k) for variant in VARIANTS),
        int(args.rich_validator_top_k) if bool(args.rich_validator_replay) else 0,
    )
    with torch.no_grad():
        for batch_index, batch in enumerate(val_loader, start=1):
            if batch_index > int(args.max_val_batches):
                break
            if batch_index == 1 or batch_index % 20 == 0:
                print(f"[stopline_candidate_pool] eval batch {batch_index}/{args.max_val_batches}", flush=True)
            raw_batch = raw_batch_for_metrics(batch)
            if raw_batch is None:
                raise ValueError("validation loader must provide raw batches for metrics")
            raw_batches.append(raw_batch)
            gt_samples = _extract_gt_samples(raw_batch)
            encoded = evaluator.prepare_batch(batch)
            outputs = _detach_to_cpu(evaluator.forward_encoded_batch(encoded))
            meta_rows = _detach_to_cpu(encoded["meta"])
            baseline_predictions = postprocess_pv26_batch(outputs, meta_rows, config=postprocess_config)
            predictions_by_variant["baseline"].extend(baseline_predictions)

            for sample_index, (meta, baseline_prediction, gt_sample) in enumerate(
                zip(meta_rows, baseline_predictions, gt_samples)
            ):
                candidates, stats = _decode_candidates(
                    outputs=outputs,
                    sample_index=sample_index,
                    meta=meta,
                    gt_stop_lines=list(gt_sample.get("stop_lines", [])),
                    source="max",
                    top_k=max_top_k,
                    min_gap=float(args.proposal_min_gap),
                )
                feature_rows.extend(
                    _candidate_feature_rows(
                        candidates,
                        batch_index=batch_index,
                        sample_index=sample_index,
                        meta=meta,
                        gt_stop_lines=list(gt_sample.get("stop_lines", [])),
                    )
                )
                sample_records.append(
                    {
                        "batch_index": int(batch_index),
                        "sample_index": int(sample_index),
                        "raw_batch": _slice_raw_batch_sample(raw_batch, sample_index),
                        "baseline_prediction": dict(baseline_prediction),
                        "gt_stop_lines": list(gt_sample.get("stop_lines", [])),
                        "candidates": candidates,
                    }
                )
                for variant in VARIANTS:
                    totals = stats_by_variant[variant.name]
                    for key, value in stats.items():
                        totals[key] = totals.get(key, 0) + int(value)
                    stop_lines = _select_stop_lines(
                        candidates,
                        variant=variant,
                        max_components=int(postprocess_config.stop_line_max_components),
                    )
                    if stop_lines:
                        totals["decoded_samples"] = totals.get("decoded_samples", 0) + 1
                        totals["decoded_lines"] = totals.get("decoded_lines", 0) + int(len(stop_lines))
                    predictions_by_variant[variant.name].append({**baseline_prediction, "stop_lines": stop_lines})
            processed_batches += 1

    merged_raw = _merge_raw_batches(raw_batches)
    rows: list[dict[str, Any]] = []
    for name in names:
        predictions = predictions_by_variant[name]
        metrics = augment_lane_family_metrics(summarize_pv26_metrics(predictions, merged_raw))
        prediction_count = sum(len(sample.get("stop_lines", [])) for sample in predictions)
        rows.append(
            _row_from_metrics(
                name,
                metrics,
                prediction_count=prediction_count,
                stats=stats_by_variant.get(name, {}),
            )
        )
    rows.sort(
        key=lambda row: (
            float(row.get("phase4_objective_proxy", 0.0)),
            float(row.get("stop_line_f1", 0.0)),
        ),
        reverse=True,
    )
    _write_csv(output_dir / "variants.csv", rows)
    _write_candidate_features_csv(output_dir / "candidate_features.csv", feature_rows)
    positives = sum(1 for row in feature_rows if bool(row.get("is_oracle_positive")))
    summary = {
        "checkpoint": str(checkpoint),
        "phase_name": phase.name,
        "phase_stage": phase.stage,
        "max_val_batches": int(args.max_val_batches),
        "validation_epoch": int(args.validation_epoch),
        "processed_batches": int(processed_batches),
        "postprocess_config": vars(postprocess_config),
        "candidate_count": int(len(feature_rows)),
        "candidate_oracle_positive_count": int(positives),
        "candidate_oracle_positive_rate": float(positives) / float(max(1, len(feature_rows))),
        "variants": rows,
        "interpretation": (
            "Candidate-pool variants are read-only probes. Oracle variants use GT distance to show "
            "candidate-pool headroom and are not production decoders."
        ),
    }
    if bool(args.rich_validator_replay):
        rich_validator = _run_rich_validator_replay(
            sample_records,
            train_fraction=float(args.rich_validator_train_fraction),
            steps=int(args.rich_validator_steps),
            lr=float(args.rich_validator_lr),
            top_k=int(args.rich_validator_top_k),
            max_components=int(postprocess_config.stop_line_max_components),
            threshold_grid=int(args.rich_validator_threshold_grid),
        )
        _write_csv(output_dir / "rich_validator_variants.csv", list(rich_validator["rows"]))
        summary["rich_validator_replay"] = rich_validator
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(rows, ensure_ascii=False, indent=2), flush=True)
    print(f"[stopline_candidate_pool] wrote {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
