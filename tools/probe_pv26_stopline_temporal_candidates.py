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

from model.data.dataset import collate_pv26_samples
from model.engine._trainer_epochs import _merge_raw_batches
from model.engine.batch import augment_lane_family_metrics, raw_batch_for_metrics
from model.engine.metrics import _extract_gt_samples, summarize_pv26_metrics
from model.engine.postprocess import _dedupe_stop_line_predictions, _stopline_prediction_sort_key, postprocess_pv26_batch
from tools.evaluate_pv26_lane60_checkpoint import _advance_validation_sampler, _postprocess_override_config
from tools.probe_pv26_lane60_postprocess_thresholds import _detach_to_cpu
from tools.probe_pv26_lane_temporal_neighbor_union import (
    _build_scenario,
    _neighbor_sample_id,
    _parse_neighbor_offsets,
)
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
from tools.probe_pv26_lane_flip_tta import _stop_line_distance
from tools.pv26_train import cli as train_cli


SCORE_KEY = "temporal_mlp_score"
TEMPORAL_FEATURES = (
    "temporal_neighbor_score",
    "temporal_neighbor_length",
    "temporal_neighbor_length_norm",
    "temporal_abs_offset",
    "temporal_offset_sign",
    "temporal_neighbor_rank_norm",
    "temporal_current_stopline_count",
    "temporal_nearest_current_distance_norm",
    "temporal_mask_mean",
    "temporal_mask_max",
    "temporal_center_mean",
    "temporal_center_max",
    "temporal_selector_mean",
    "temporal_selector_max",
    "temporal_proposal_mean",
    "temporal_proposal_max",
    "temporal_endpoint_proposal_mean",
    "temporal_center_x_norm",
    "temporal_center_y_norm",
    "temporal_abs_cos",
    "temporal_abs_sin",
)


class _StopLineNeighborPredictor:
    def __init__(self, *, dataset: Any, evaluator: Any, postprocess_config: Any) -> None:
        self.dataset = dataset
        self.evaluator = evaluator
        self.postprocess_config = postprocess_config
        self.cache: dict[int, dict[str, Any]] = {}

    def predict_index(self, dataset_index: int) -> dict[str, Any]:
        dataset_index = int(dataset_index)
        cached = self.cache.get(dataset_index)
        if cached is not None:
            return cached
        batch = collate_pv26_samples([self.dataset[dataset_index]])
        encoded = self.evaluator.prepare_batch(batch)
        outputs = _detach_to_cpu(self.evaluator.forward_encoded_batch(encoded))
        meta = _detach_to_cpu(encoded["meta"])
        prediction = postprocess_pv26_batch(outputs, meta, config=self.postprocess_config)[0]
        payload = {"prediction": prediction, "meta": meta[0]}
        self.cache[dataset_index] = payload
        return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a no-GT stop-line temporal candidate verifier. Neighbor-frame stop-line "
            "predictions become candidates, current-frame dense stop-line maps provide quality "
            "features, and a train-split MLP verifier is replayed on validation."
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
    parser.add_argument("--neighbor-offsets", default="-1,1")
    parser.add_argument("--temporal-top-k", type=int, default=8)
    parser.add_argument("--max-temporal-candidates", type=int, default=16)
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
    if points.shape[0] < 2:
        return 0.0
    return float(np.linalg.norm(points[-1] - points[0]))


def _copy_stop_line(line: dict[str, Any], *, score: float, source: str) -> dict[str, Any]:
    copied = dict(line)
    copied["points_xy"] = [[float(point[0]), float(point[1])] for point in line.get("points_xy", [])]
    copied["score"] = float(score)
    copied["center_score"] = float(score)
    copied["source"] = str(source)
    copied["proposal_source"] = str(source)
    return copied


def _nearest_current_distance(candidate: dict[str, Any], current_stop_lines: list[dict[str, Any]]) -> float:
    if not current_stop_lines:
        return 1.0e6
    return float(min(_stop_line_distance(candidate, current) for current in current_stop_lines))


def _stopline_temporal_features(
    candidate: dict[str, Any],
    *,
    meta: dict[str, Any],
    mask_probs: np.ndarray | None,
    center_probs: np.ndarray | None,
    selector_probs: np.ndarray | None,
    neighbor_offset: int,
    neighbor_rank: int,
    current_stop_lines: list[dict[str, Any]],
) -> dict[str, float]:
    points_raw = np.asarray(candidate.get("points_xy", []), dtype=np.float32).reshape(-1, 2)
    reference_map = next(
        (array for array in (mask_probs, center_probs, selector_probs) if isinstance(array, np.ndarray)),
        None,
    )
    if points_raw.shape[0] < 2 or reference_map is None:
        return {name: 0.0 for name in TEMPORAL_FEATURES}
    output_hw = (int(reference_map.shape[0]), int(reference_map.shape[1]))
    output_points = _raw_points_to_output(points_raw, meta, output_hw)
    proposal_map = None
    if center_probs is not None and selector_probs is not None:
        proposal_map = np.maximum(center_probs, selector_probs)
    elif center_probs is not None:
        proposal_map = center_probs
    elif selector_probs is not None:
        proposal_map = selector_probs
    mask_mean, mask_max = _line_stats(mask_probs, output_points)
    center_mean, center_max = _line_stats(center_probs, output_points)
    selector_mean, selector_max = _line_stats(selector_probs, output_points)
    proposal_mean, proposal_max = _line_stats(proposal_map, output_points)
    endpoint_mean = _endpoint_proposal_mean(proposal_map, output_points)
    length = _line_length(candidate)
    raw_h, raw_w = int(meta.get("raw_hw", (1, 1))[0]), int(meta.get("raw_hw", (1, 1))[1])
    center = points_raw.mean(axis=0)
    delta = points_raw[-1] - points_raw[0]
    norm = float(np.linalg.norm(delta))
    axis = delta / max(norm, 1.0e-6)
    nearest_distance = _nearest_current_distance(candidate, current_stop_lines)
    features = {
        "temporal_neighbor_score": float(_line_score(candidate)),
        "temporal_neighbor_length": float(length),
        "temporal_neighbor_length_norm": float(min(length / max(float(raw_w), 1.0), 1.0)),
        "temporal_abs_offset": float(abs(int(neighbor_offset))),
        "temporal_offset_sign": float(1.0 if int(neighbor_offset) > 0 else -1.0),
        "temporal_neighbor_rank_norm": float(1.0 / max(int(neighbor_rank), 1)),
        "temporal_current_stopline_count": float(len(current_stop_lines)),
        "temporal_nearest_current_distance_norm": float(min(nearest_distance / max(float(raw_w), 1.0), 4.0)),
        "temporal_mask_mean": float(mask_mean),
        "temporal_mask_max": float(mask_max),
        "temporal_center_mean": float(center_mean),
        "temporal_center_max": float(center_max),
        "temporal_selector_mean": float(selector_mean),
        "temporal_selector_max": float(selector_max),
        "temporal_proposal_mean": float(proposal_mean),
        "temporal_proposal_max": float(proposal_max),
        "temporal_endpoint_proposal_mean": float(endpoint_mean),
        "temporal_center_x_norm": float(np.clip(float(center[0]) / max(float(raw_w), 1.0), 0.0, 1.0)),
        "temporal_center_y_norm": float(np.clip(float(center[1]) / max(float(raw_h), 1.0), 0.0, 1.0)),
        "temporal_abs_cos": float(abs(float(axis[0]))),
        "temporal_abs_sin": float(abs(float(axis[1]))),
    }
    return {key: 0.0 if not math.isfinite(float(value)) else float(value) for key, value in features.items()}


def _candidate_row(candidate: dict[str, Any], *, batch_index: int, sample_index: int, meta: dict[str, Any]) -> dict[str, Any]:
    points = np.asarray(candidate.get("points_xy", []), dtype=np.float32).reshape(-1, 2)
    return {
        "batch_index": int(batch_index),
        "sample_index": int(sample_index),
        "sample_id": str(meta.get("sample_id", "")),
        "dataset_key": str(meta.get("dataset_key", "")),
        "image_path": str(meta.get("image_path", "")),
        "neighbor_offset": int(candidate.get("neighbor_offset", 0)),
        "neighbor_dataset_index": int(candidate.get("neighbor_dataset_index", -1)),
        "neighbor_rank": int(candidate.get("neighbor_rank", 0)),
        "candidate_points_json": json.dumps([[float(x), float(y)] for x, y in points.tolist()], separators=(",", ":")),
        "score": float(candidate.get("score", 0.0)),
        "length": float(candidate.get("length", 0.0)),
        "nearest_gt_distance": float(candidate.get("nearest_gt_distance", float("inf"))),
        "nearest_gt_angle_error": float(candidate.get("nearest_gt_angle_error", float("inf"))),
        "is_oracle_positive": int(bool(candidate.get("is_oracle_positive", False))),
        **{name: float(candidate.get(name, 0.0)) for name in TEMPORAL_FEATURES},
    }


def _build_temporal_candidates(
    *,
    meta: dict[str, Any],
    mask_probs: np.ndarray | None,
    center_probs: np.ndarray | None,
    selector_probs: np.ndarray | None,
    current_stop_lines: list[dict[str, Any]],
    gt_stop_lines: list[dict[str, Any]],
    neighbor_predictions: list[tuple[int, int, dict[str, Any]]],
    max_candidates: int,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for offset, neighbor_dataset_index, neighbor_prediction in neighbor_predictions:
        neighbor_lines = list(neighbor_prediction.get("stop_lines", []))
        neighbor_lines.sort(key=_stopline_prediction_sort_key, reverse=True)
        for neighbor_rank, line in enumerate(neighbor_lines, start=1):
            candidate = _copy_stop_line(line, score=_line_score(line), source="temporal_neighbor")
            candidate["neighbor_offset"] = int(offset)
            candidate["neighbor_dataset_index"] = int(neighbor_dataset_index)
            candidate["neighbor_rank"] = int(neighbor_rank)
            candidate["length"] = _line_length(candidate)
            candidate.update(
                _stopline_temporal_features(
                    candidate,
                    meta=meta,
                    mask_probs=mask_probs,
                    center_probs=center_probs,
                    selector_probs=selector_probs,
                    neighbor_offset=int(offset),
                    neighbor_rank=int(neighbor_rank),
                    current_stop_lines=current_stop_lines,
                )
            )
            nearest_distance, nearest_angle, nearest_index = _nearest_gt(candidate, gt_stop_lines)
            candidate["nearest_gt_distance"] = float(nearest_distance)
            candidate["nearest_gt_angle_error"] = float(nearest_angle)
            candidate["nearest_gt_index"] = int(nearest_index)
            candidate["is_oracle_positive"] = bool(nearest_distance <= 40.0)
            candidate["temporal_rank_score"] = (
                float(candidate.get("temporal_proposal_max", 0.0))
                + 0.5 * float(candidate.get("temporal_mask_mean", 0.0))
                + 0.25 * float(candidate.get("temporal_neighbor_score", 0.0))
                - 0.05 * float(abs(int(offset)))
            )
            candidates.append(candidate)
    candidates.sort(
        key=lambda item: (
            float(item.get("temporal_rank_score", 0.0)),
            float(item.get("temporal_neighbor_score", 0.0)),
            float(item.get("length", 0.0)),
        ),
        reverse=True,
    )
    return candidates[: max(1, int(max_candidates))]


def _collect_records(
    *,
    loader: Any,
    evaluator: Any,
    predictor: _StopLineNeighborPredictor,
    postprocess_config: Any,
    record_index_by_key: dict[tuple[str, str, str], int],
    max_batches: int,
    neighbor_offsets: tuple[int, ...],
    max_temporal_candidates: int,
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
                print(f"[stopline_temporal] collect {split_name} batch {batch_index}/{max_batches}", flush=True)
            raw_batch = raw_batch_for_metrics(batch)
            if raw_batch is None:
                raise ValueError("temporal stop-line probe requires raw batches for metrics")
            gt_samples = _extract_gt_samples(raw_batch)
            encoded = evaluator.prepare_batch(batch)
            outputs = _detach_to_cpu(evaluator.forward_encoded_batch(encoded))
            meta_rows = _detach_to_cpu(encoded["meta"])
            baseline_predictions = postprocess_pv26_batch(outputs, meta_rows, config=postprocess_config)
            for sample_index, (meta, baseline_prediction, gt_sample) in enumerate(
                zip(meta_rows, baseline_predictions, gt_samples)
            ):
                sample_id = str(meta.get("sample_id", ""))
                dataset_key = str(meta.get("dataset_key", ""))
                split = str(meta.get("split", ""))
                neighbor_payloads: list[tuple[int, int, dict[str, Any]]] = []
                missing_neighbors = 0
                for offset in neighbor_offsets:
                    neighbor_id = _neighbor_sample_id(sample_id, int(offset))
                    if neighbor_id is None:
                        missing_neighbors += 1
                        continue
                    neighbor_index = record_index_by_key.get((dataset_key, split, neighbor_id))
                    if neighbor_index is None:
                        missing_neighbors += 1
                        continue
                    neighbor_payload = predictor.predict_index(neighbor_index)
                    neighbor_payloads.append((int(offset), int(neighbor_index), dict(neighbor_payload["prediction"])))
                mask_probs = _as_2d_array(_sample_tensor(outputs, "stop_line_mask_logits", sample_index), sigmoid=True)
                center_probs = _as_2d_array(_sample_tensor(outputs, "stop_line_center_logits", sample_index), sigmoid=True)
                selector_probs = _as_2d_array(
                    _sample_tensor(outputs, "stop_line_selector_map_logits", sample_index),
                    sigmoid=True,
                )
                candidates = _build_temporal_candidates(
                    meta=meta,
                    mask_probs=mask_probs,
                    center_probs=center_probs,
                    selector_probs=selector_probs,
                    current_stop_lines=list(baseline_prediction.get("stop_lines", [])),
                    gt_stop_lines=list(gt_sample.get("stop_lines", [])),
                    neighbor_predictions=neighbor_payloads,
                    max_candidates=int(max_temporal_candidates),
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
                        "sample_id": sample_id,
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
                        "sample_id": sample_id,
                        "dataset_key": dataset_key,
                        "temporal_neighbor_count": int(len(neighbor_payloads)),
                        "missing_neighbor_count": int(missing_neighbors),
                        "baseline_stopline_count": int(len(list(baseline_prediction.get("stop_lines", [])))),
                        "gt_stopline_count": int(len(list(gt_sample.get("stop_lines", [])))),
                        "temporal_candidate_count": int(len(candidates)),
                        "temporal_oracle_positive_count": int(
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
            features.append([float(candidate.get(name, 0.0)) for name in TEMPORAL_FEATURES])
            labels.append(float(bool(candidate.get("is_oracle_positive", False))))
    if not features:
        return np.zeros((0, len(TEMPORAL_FEATURES)), dtype=np.float32), np.zeros((0,), dtype=np.float32)
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
        raise ValueError("temporal verifier requires non-empty train and validation candidates")
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


def _select_temporal_stop_lines(
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
            float(item.get("temporal_rank_score", 0.0)),
            float(item.get("length", 0.0)),
        ),
        reverse=True,
    )
    predictions = [
        _copy_stop_line(candidate, score=float(candidate.get(score_key, 0.0)), source="temporal_neighbor")
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
            stop_lines = _select_temporal_stop_lines(
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
    neighbor_offsets = _parse_neighbor_offsets(str(args.neighbor_offsets))
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
        progress_callback=lambda message: print(f"[stopline_temporal] {message}", flush=True),
    )
    train_loader, val_loader = train_cli._build_phase_train_loaders(dataset, train_config=train_config, phase=phase)
    if train_loader is None or val_loader is None:
        raise ValueError("temporal stop-line probe requires train and validation loaders")
    _advance_validation_sampler(val_loader, validation_epoch=int(args.validation_epoch))
    trainer = train_cli._build_phase_trainer(phase, train_config)
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    trainer.load_model_weights(checkpoint, map_location=train_config.device)
    evaluator = trainer.build_evaluator()
    evaluator.adapter.raw_model.eval()
    evaluator.heads.eval()
    postprocess_config = _postprocess_override_config(args, trainer) or train_cli._build_postprocess_config(train_config)
    predictor = _StopLineNeighborPredictor(
        dataset=dataset,
        evaluator=evaluator,
        postprocess_config=postprocess_config,
    )
    record_index_by_key = {
        (str(record.dataset_key), str(record.split), str(record.sample_id)): int(index)
        for index, record in enumerate(dataset.records)
    }
    train_records, train_candidate_rows, train_sample_rows = _collect_records(
        loader=train_loader,
        evaluator=evaluator,
        predictor=predictor,
        postprocess_config=postprocess_config,
        record_index_by_key=record_index_by_key,
        max_batches=int(args.train_record_batches),
        neighbor_offsets=neighbor_offsets,
        max_temporal_candidates=int(args.max_temporal_candidates),
        split_name="train",
    )
    val_records, val_candidate_rows, val_sample_rows = _collect_records(
        loader=val_loader,
        evaluator=evaluator,
        predictor=predictor,
        postprocess_config=postprocess_config,
        record_index_by_key=record_index_by_key,
        max_batches=int(args.max_val_batches),
        neighbor_offsets=neighbor_offsets,
        max_temporal_candidates=int(args.max_temporal_candidates),
        split_name="val",
    )
    score_summary = _score_records(
        train_records=train_records,
        val_records=val_records,
        top_k=int(args.temporal_top_k),
        epochs=int(args.verifier_epochs),
        lr=float(args.verifier_lr),
    )
    threshold = _best_threshold(
        train_records,
        score_key=SCORE_KEY,
        top_k=int(args.temporal_top_k),
        max_components=int(args.max_components),
        grid_size=int(args.threshold_grid),
    )
    rows = [
        _metrics_row(train_records, name="baseline", split="train"),
        _metrics_row(val_records, name="baseline", split="val"),
        _metrics_row(
            train_records,
            name="temporal_mlp",
            split="train",
            score_key=SCORE_KEY,
            threshold=threshold,
            top_k=int(args.temporal_top_k),
            max_components=int(args.max_components),
        ),
        _metrics_row(
            val_records,
            name="temporal_mlp",
            split="val",
            score_key=SCORE_KEY,
            threshold=threshold,
            top_k=int(args.temporal_top_k),
            max_components=int(args.max_components),
        ),
        _metrics_row(
            train_records,
            name="baseline_plus_temporal_mlp",
            split="train",
            score_key=SCORE_KEY,
            threshold=threshold,
            top_k=int(args.temporal_top_k),
            max_components=int(args.max_components),
            union_baseline=True,
        ),
        _metrics_row(
            val_records,
            name="baseline_plus_temporal_mlp",
            split="val",
            score_key=SCORE_KEY,
            threshold=threshold,
            top_k=int(args.temporal_top_k),
            max_components=int(args.max_components),
            union_baseline=True,
        ),
        _metrics_row(
            val_records,
            name="oracle_temporal",
            split="val",
            score_key="is_oracle_positive",
            threshold=0.5,
            top_k=int(args.temporal_top_k),
            max_components=int(args.max_components),
        ),
        _metrics_row(
            val_records,
            name="baseline_plus_oracle_temporal",
            split="val",
            score_key="is_oracle_positive",
            threshold=0.5,
            top_k=int(args.temporal_top_k),
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
        "neighbor_offsets": list(neighbor_offsets),
        "temporal_top_k": int(args.temporal_top_k),
        "max_temporal_candidates": int(args.max_temporal_candidates),
        "max_components": int(args.max_components),
        "threshold": float(threshold),
        "score_summary": score_summary,
        "train_sample_count": int(len(train_records)),
        "val_sample_count": int(len(val_records)),
        "train_temporal_candidate_count": int(len(train_candidate_rows)),
        "val_temporal_candidate_count": int(len(val_candidate_rows)),
        "train_temporal_oracle_positive_count": int(
            sum(int(row.get("is_oracle_positive", 0)) for row in train_candidate_rows)
        ),
        "val_temporal_oracle_positive_count": int(sum(int(row.get("is_oracle_positive", 0)) for row in val_candidate_rows)),
        "rows": rows,
        "interpretation": (
            "Temporal stop-line candidate probe. Runtime candidates come from neighboring frame predictions "
            "and current-frame dense stop-line map features; GT is used for train labels, oracle diagnostics, "
            "and final metrics only."
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
    _write_csv(output_dir / "temporal_variants.csv", payload["rows"])
    _write_candidate_features_csv(output_dir / "train_candidate_features.csv", payload["train_candidate_rows"])
    _write_candidate_features_csv(output_dir / "val_candidate_features.csv", payload["val_candidate_rows"])
    _write_csv(output_dir / "temporal_samples.csv", payload["sample_rows"])
    (output_dir / "summary.json").write_text(
        json.dumps(payload["summary"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
