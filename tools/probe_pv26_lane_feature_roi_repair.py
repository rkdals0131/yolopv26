from __future__ import annotations

import argparse
import copy
import csv
import json
import math
from dataclasses import dataclass
from dataclasses import replace
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
from model.engine.metrics import _extract_gt_samples, summarize_pv26_metrics
from model.engine.postprocess import postprocess_pv26_batch
from tools.evaluate_pv26_lane60_checkpoint import _advance_validation_sampler, _postprocess_override_config
from tools.probe_pv26_lane_flip_tta import _merge_lane_dense_predictions, _unflip_lane_dense_outputs
from tools.probe_pv26_lane_fn_recovery_audit import (
    SOURCE_RUN,
    _distance_bin,
    _gt_centerline_evidence,
    _match_predictions,
    _nearest_gt_lane,
    _points_json,
)
from tools.probe_pv26_lane_instance_evidence import (
    _as_channel,
    _resolve_dataset_root,
    _resolve_device,
    _safe_stat,
    _sample_polyline,
    _tangent_alignment,
    _values_at_points,
)
from tools.probe_pv26_lane_ranked_translate_repair import (
    DEFAULT_CHECKPOINT,
    _auto_repair_topk,
    _map_points_to_raw,
    _metric,
    _prediction_features,
    _raw_points_to_map,
)
from tools.probe_pv26_lane60_postprocess_thresholds import _detach_to_cpu
from tools.pv26_train import cli as train_cli
from tools.pv26_train import config as train_config_api
from tools.run_pv26_lane60_probe import _lane60_scenario


ROI_SAMPLE_COUNT = 16
ROI_RADIUS = 16
RIDGE_L2 = 0.05
LOGISTIC_EPOCHS = 400
LOGISTIC_LR = 0.08
LOGISTIC_L2 = 0.01


@dataclass
class FeatureScaler:
    mean: np.ndarray
    scale: np.ndarray


@dataclass
class LogisticModel:
    scaler: FeatureScaler
    weights: np.ndarray
    bias: float


@dataclass
class RidgeGeometryModel:
    scaler: FeatureScaler
    weights: np.ndarray
    bias: np.ndarray


@dataclass
class Candidate:
    sample_index: int
    sample_batch_index: int
    pred_index: int
    split_id: int
    feature: np.ndarray
    target: np.ndarray | None
    pred_points_map: np.ndarray
    meta: dict[str, Any]
    map_hw: tuple[int, int]
    row: dict[str, Any]
    score: float = math.nan
    selected: bool = False
    repaired_points_map: np.ndarray | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train/evaluate an out-of-fold no-GT lane feature-ROI repair replay. "
            "It samples dense lane maps along unmatched predicted lanes, trains a "
            "repairability classifier plus geometry ridge on opposite validation "
            "folds, then replaces selected lanes and recomputes metrics."
        )
    )
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--source-run", default=str(SOURCE_RUN))
    parser.add_argument("--lane60-experiment", default="core_centerline_refine_row_scan_tangent_link")
    parser.add_argument("--preset", default="default")
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
    parser.add_argument("--repair-topk", type=int, default=0, help="0 uses val-size-scaled top500/2048.")
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
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def _safe_sigmoid(value: np.ndarray | float) -> np.ndarray | float:
    array = np.asarray(value, dtype=np.float64)
    out = np.empty_like(array)
    positive = array >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-np.minimum(array[positive], 60.0)))
    exp_value = np.exp(np.maximum(array[~positive], -60.0))
    out[~positive] = exp_value / (1.0 + exp_value)
    if np.isscalar(value):
        return float(out.item())
    return out


def _fit_scaler(matrix: np.ndarray) -> FeatureScaler:
    if matrix.size == 0:
        return FeatureScaler(mean=np.zeros((0,), dtype=np.float32), scale=np.ones((0,), dtype=np.float32))
    mean = np.nanmean(matrix, axis=0).astype(np.float32)
    mean = np.where(np.isfinite(mean), mean, 0.0).astype(np.float32)
    filled = np.where(np.isfinite(matrix), matrix, mean.reshape(1, -1)).astype(np.float32)
    scale = filled.std(axis=0).astype(np.float32)
    scale = np.where(scale > 1.0e-6, scale, 1.0).astype(np.float32)
    return FeatureScaler(mean=mean, scale=scale)


def _transform(scaler: FeatureScaler, matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return matrix.astype(np.float32)
    filled = np.where(np.isfinite(matrix), matrix, scaler.mean.reshape(1, -1)).astype(np.float32)
    return (filled - scaler.mean.reshape(1, -1)) / scaler.scale.reshape(1, -1)


def _train_logistic(matrix: np.ndarray, labels: np.ndarray) -> LogisticModel:
    scaler = _fit_scaler(matrix)
    x = _transform(scaler, matrix)
    y = labels.astype(np.float32).reshape(-1)
    if x.shape[0] == 0:
        return LogisticModel(scaler=scaler, weights=np.zeros((x.shape[1],), dtype=np.float32), bias=0.0)
    pos_rate = float(np.clip(y.mean() if y.size else 0.01, 0.01, 0.99))
    bias = float(math.log(pos_rate / (1.0 - pos_rate)))
    weights = np.zeros((x.shape[1],), dtype=np.float32)
    for _ in range(LOGISTIC_EPOCHS):
        pred = np.asarray(_safe_sigmoid(x @ weights + bias), dtype=np.float32)
        error = pred - y
        grad_w = (x.T @ error) / max(1, x.shape[0]) + LOGISTIC_L2 * weights
        grad_b = float(error.mean())
        weights -= float(LOGISTIC_LR) * grad_w.astype(np.float32)
        bias -= float(LOGISTIC_LR) * grad_b
    return LogisticModel(scaler=scaler, weights=weights.astype(np.float32), bias=float(bias))


def _score_logistic(model: LogisticModel, matrix: np.ndarray) -> np.ndarray:
    x = _transform(model.scaler, matrix)
    return np.asarray(_safe_sigmoid(x @ model.weights + float(model.bias)), dtype=np.float32)


def _train_ridge(matrix: np.ndarray, targets: np.ndarray) -> RidgeGeometryModel:
    scaler = _fit_scaler(matrix)
    x = _transform(scaler, matrix)
    y = targets.astype(np.float32)
    if x.shape[0] == 0:
        return RidgeGeometryModel(
            scaler=scaler,
            weights=np.zeros((x.shape[1], targets.shape[1]), dtype=np.float32),
            bias=np.zeros((targets.shape[1],), dtype=np.float32),
        )
    bias = y.mean(axis=0).astype(np.float32)
    centered_y = y - bias.reshape(1, -1)
    lhs = x.T @ x
    lhs += np.eye(lhs.shape[0], dtype=np.float32) * float(RIDGE_L2)
    rhs = x.T @ centered_y
    try:
        weights = np.linalg.solve(lhs, rhs).astype(np.float32)
    except np.linalg.LinAlgError:
        weights = np.linalg.pinv(lhs) @ rhs
        weights = weights.astype(np.float32)
    return RidgeGeometryModel(scaler=scaler, weights=weights, bias=bias)


def _predict_ridge(model: RidgeGeometryModel, matrix: np.ndarray) -> np.ndarray:
    x = _transform(model.scaler, matrix)
    return (x @ model.weights + model.bias.reshape(1, -1)).astype(np.float32)


def _profile_delta_features(values: np.ndarray, points: np.ndarray, *, radius: int) -> tuple[np.ndarray, np.ndarray]:
    height, width = int(values.shape[0]), int(values.shape[1])
    deltas: list[float] = []
    masses: list[float] = []
    for x_value, y_value in points:
        row = min(max(int(round(float(y_value))), 0), height - 1)
        center_x = min(max(int(round(float(x_value))), 0), width - 1)
        start = max(0, center_x - int(radius))
        end = min(width, center_x + int(radius) + 1)
        xs = np.arange(start, end, dtype=np.float32)
        profile = np.clip(values[row, start:end].astype(np.float32), 0.0, None)
        mass = float(profile.sum())
        if mass > 1.0e-6:
            centroid = float((xs * profile).sum() / mass)
            deltas.append((centroid - float(x_value)) / max(float(radius), 1.0))
        else:
            deltas.append(0.0)
        masses.append(mass)
    return np.asarray(deltas, dtype=np.float32), np.asarray(masses, dtype=np.float32)


def _lane_target_label(distance: float, center_mean: float) -> bool:
    return float(distance) <= 120.0 and float(center_mean) >= 0.0


def _candidate_roi_feature(
    lane: dict[str, Any],
    *,
    maps: dict[str, torch.Tensor],
    meta: dict[str, Any],
    scalar_features: dict[str, float],
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    centerline = _as_channel(maps["centerline_core"])
    support = _as_channel(maps["support"])
    tangent = np.asarray(maps["tangent_axis"].detach().cpu().numpy() if isinstance(maps["tangent_axis"], torch.Tensor) else maps["tangent_axis"], dtype=np.float32)
    map_hw = (int(centerline.shape[0]), int(centerline.shape[1]))
    map_points = _raw_points_to_map(list(lane.get("points_xy", [])), meta, map_hw)
    sampled = _sample_polyline(map_points, count=ROI_SAMPLE_COUNT)
    if sampled.shape[0] != ROI_SAMPLE_COUNT:
        sampled = np.zeros((ROI_SAMPLE_COUNT, 2), dtype=np.float32)
    center_values = _values_at_points(centerline, sampled)
    support_values = _values_at_points(support, sampled)
    profile_dx, profile_mass = _profile_delta_features(centerline, sampled, radius=ROI_RADIUS)
    tangent_values = _tangent_alignment(tangent, sampled)
    tangent_mean = _safe_stat(tangent_values, "mean")
    scalar_vector = np.asarray(
        [
            float(scalar_features.get("pred_point_count", 0.0)),
            float(scalar_features.get("pred_polyline_length", 0.0)) / 1000.0,
            float(scalar_features.get("pred_bbox_width", 0.0)) / 800.0,
            float(scalar_features.get("pred_bbox_height", 0.0)) / 608.0,
            float(scalar_features.get("pred_bbox_aspect", 0.0)) / 10.0,
            float(scalar_features.get("pred_center_point_mean", 0.0)),
            float(scalar_features.get("pred_center_point_q10", 0.0)),
            float(scalar_features.get("pred_support_point_mean", 0.0)),
            float(scalar_features.get("nearest_other_pred_distance", 0.0)) / 400.0,
            float(scalar_features.get("sample_pred_lane_count", 0.0)) / 16.0,
            float(tangent_mean),
            _safe_stat(center_values, "mean"),
            _safe_stat(center_values, "q10"),
            _safe_stat(support_values, "mean"),
            _safe_stat(profile_dx, "mean"),
            _safe_stat(profile_dx, "q10"),
            _safe_stat(profile_mass, "mean"),
        ],
        dtype=np.float32,
    )
    point_features = np.stack(
        [
            sampled[:, 0] / max(float(map_hw[1] - 1), 1.0),
            sampled[:, 1] / max(float(map_hw[0] - 1), 1.0),
            center_values,
            support_values,
            profile_dx,
            np.clip(profile_mass, 0.0, 1.0),
        ],
        axis=1,
    ).reshape(-1)
    return np.concatenate([scalar_vector, point_features.astype(np.float32)], axis=0), sampled, map_hw


def _target_map_points(lane: dict[str, Any], *, meta: dict[str, Any], map_hw: tuple[int, int]) -> np.ndarray:
    gt_map = _raw_points_to_map(list(lane.get("points_xy", [])), meta, map_hw)
    sampled = _sample_polyline(gt_map, count=ROI_SAMPLE_COUNT)
    if sampled.shape[0] != ROI_SAMPLE_COUNT:
        return np.zeros((ROI_SAMPLE_COUNT, 2), dtype=np.float32)
    return sampled.astype(np.float32)


def _target_vector(points: np.ndarray, map_hw: tuple[int, int]) -> np.ndarray:
    out = np.asarray(points, dtype=np.float32).reshape(ROI_SAMPLE_COUNT, 2).copy()
    out[:, 0] = out[:, 0] / max(float(map_hw[1] - 1), 1.0)
    out[:, 1] = out[:, 1] / max(float(map_hw[0] - 1), 1.0)
    return out.reshape(-1)


def _prediction_from_vector(vector: np.ndarray, map_hw: tuple[int, int]) -> np.ndarray:
    points = np.asarray(vector, dtype=np.float32).reshape(ROI_SAMPLE_COUNT, 2).copy()
    points[:, 0] = np.clip(points[:, 0], 0.0, 1.0) * max(float(map_hw[1] - 1), 1.0)
    points[:, 1] = np.clip(points[:, 1], 0.0, 1.0) * max(float(map_hw[0] - 1), 1.0)
    return points


def _split_id(sample_index: int) -> int:
    return int(sample_index) % 2


def _metric_payload(metrics: dict[str, Any], task: str) -> dict[str, float]:
    return {
        "precision": _metric(metrics, task, "precision"),
        "recall": _metric(metrics, task, "recall"),
        "f1": _metric(metrics, task, "f1"),
        "tp": _metric(metrics, task, "tp"),
        "fp": _metric(metrics, task, "fp"),
        "fn": _metric(metrics, task, "fn"),
    }


def _task_delta(repaired: dict[str, float], baseline: dict[str, float]) -> dict[str, float]:
    return {
        "precision_delta": float(repaired["precision"] - baseline["precision"]),
        "recall_delta": float(repaired["recall"] - baseline["recall"]),
        "f1_delta": float(repaired["f1"] - baseline["f1"]),
        "tp_delta": float(repaired["tp"] - baseline["tp"]),
        "fp_delta": float(repaired["fp"] - baseline["fp"]),
        "fn_delta": float(repaired["fn"] - baseline["fn"]),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _build_scenario(args: argparse.Namespace) -> tuple[Any, Path, dict[str, Any], Any, Any]:
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    source_run = Path(args.source_run).expanduser().resolve()
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
    return scenario, scenario_path, options, phase, train_config


def _train_oof_models(candidates: list[Candidate]) -> dict[str, Any]:
    feature_matrix = np.stack([candidate.feature for candidate in candidates], axis=0).astype(np.float32)
    labels = np.asarray([1.0 if bool(candidate.row["repairable_le120_any_center"]) else 0.0 for candidate in candidates], dtype=np.float32)
    scores = np.full((len(candidates),), np.nan, dtype=np.float32)
    predictions: list[np.ndarray | None] = [None for _ in candidates]
    model_summaries: list[dict[str, Any]] = []

    for holdout_split in (0, 1):
        train_indices = [index for index, candidate in enumerate(candidates) if int(candidate.split_id) != holdout_split]
        holdout_indices = [index for index, candidate in enumerate(candidates) if int(candidate.split_id) == holdout_split]
        if not train_indices or not holdout_indices:
            continue
        train_x = feature_matrix[train_indices]
        train_y = labels[train_indices]
        logistic = _train_logistic(train_x, train_y)
        scores[holdout_indices] = _score_logistic(logistic, feature_matrix[holdout_indices])

        geometry_train_indices = [
            index
            for index in train_indices
            if bool(candidates[index].row["repairable_le120_any_center"]) and candidates[index].target is not None
        ]
        if geometry_train_indices:
            geometry_x = feature_matrix[geometry_train_indices]
            geometry_y = np.stack([candidates[index].target for index in geometry_train_indices], axis=0).astype(np.float32)
            ridge = _train_ridge(geometry_x, geometry_y)
            predicted = _predict_ridge(ridge, feature_matrix[holdout_indices])
            for local_index, candidate_index in enumerate(holdout_indices):
                predictions[candidate_index] = predicted[local_index]
        model_summaries.append(
            {
                "holdout_split": int(holdout_split),
                "train_count": int(len(train_indices)),
                "holdout_count": int(len(holdout_indices)),
                "train_positive_count": int(train_y.sum()),
                "geometry_train_count": int(len(geometry_train_indices)),
            }
        )

    for index, candidate in enumerate(candidates):
        candidate.score = float(scores[index]) if math.isfinite(float(scores[index])) else math.nan
        if predictions[index] is not None:
            candidate.repaired_points_map = _prediction_from_vector(predictions[index], candidate.map_hw)
    return {"model_summaries": model_summaries}


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    source_run = Path(args.source_run).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not source_run.is_dir():
        raise FileNotFoundError(source_run)

    scenario, scenario_path, options, phase, train_config = _build_scenario(args)
    train_cli._configure_torch_multiprocessing()
    dataset = train_cli.PV26CanonicalDataset(
        train_cli._existing_dataset_roots(scenario),
        train_augmentation=False,
        progress_callback=lambda message: print(f"[lane_feature_roi] {message}", flush=True),
    )
    _, val_loader = train_cli._build_phase_train_loaders(dataset, train_config=train_config, phase=phase)
    if val_loader is None:
        raise ValueError("lane feature ROI repair requires validation batches")
    _advance_validation_sampler(val_loader, validation_epoch=int(args.validation_epoch))

    trainer = train_cli._build_phase_trainer(phase, train_config)
    trainer.load_model_weights(checkpoint, map_location=train_config.device)
    postprocess_config = _postprocess_override_config(args, trainer) or train_cli._build_postprocess_config(train_config)
    evaluator = trainer.build_evaluator()
    evaluator.adapter.raw_model.eval()
    evaluator.heads.eval()

    raw_batches: list[dict[str, Any]] = []
    baseline_predictions_all: list[dict[str, Any]] = []
    candidates: list[Candidate] = []
    global_sample_index = 0

    with torch.no_grad():
        for batch_index, batch in enumerate(val_loader, start=1):
            if batch_index > int(args.max_val_batches):
                break
            if batch_index == 1 or batch_index % 20 == 0:
                print(f"[lane_feature_roi] eval batch {batch_index}/{args.max_val_batches}", flush=True)
            raw_batch = raw_batch_for_metrics(batch)
            if raw_batch is None:
                raise ValueError("validation loader must provide raw batches for metrics")
            encoded = evaluator.prepare_batch(batch)
            predictions = _detach_to_cpu(evaluator.forward_encoded_batch(encoded))
            flip_predictions = None
            if str(args.lane_flip_variant) != "baseline":
                flipped_encoded = dict(encoded)
                flipped_encoded["image"] = torch.flip(encoded["image"], dims=(-1,))
                flip_predictions = _unflip_lane_dense_outputs(_detach_to_cpu(evaluator.forward_encoded_batch(flipped_encoded)))
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
            raw_batches.append(raw_batch)
            baseline_predictions_all.extend(copy.deepcopy(batch_predictions))

            for sample_batch_index, (sample_pred, sample_gt, sample_meta) in enumerate(zip(batch_predictions, batch_gt, meta)):
                pred_lanes = list(sample_pred.get("lanes", []))
                gt_lanes = list(sample_gt.get("lanes", []))
                pred_to_gt, gt_to_pred = _match_predictions(pred_lanes, gt_lanes)
                unmatched_pred_indices = set(range(len(pred_lanes))) - set(pred_to_gt)
                unmatched_gt_indices = set(range(len(gt_lanes))) - set(gt_to_pred)
                maps = lane_segfirst_prediction_maps(postprocess_predictions, batch_index=sample_batch_index)
                fn_evidence_by_gt_index = {
                    int(gt_index): _gt_centerline_evidence(gt_lanes[gt_index], maps=maps, meta=sample_meta)
                    for gt_index in unmatched_gt_indices
                }
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
                    scalar_features = _prediction_features(pred_index, pred_lanes, maps=maps, meta=sample_meta)
                    feature_vector, pred_sampled_map, map_hw = _candidate_roi_feature(
                        pred_lane,
                        maps=maps,
                        meta=sample_meta,
                        scalar_features=scalar_features,
                    )
                    target_vector = None
                    if int(nearest_fn_gt_index) >= 0:
                        target_map = _target_map_points(gt_lanes[nearest_fn_gt_index], meta=sample_meta, map_hw=map_hw)
                        target_vector = _target_vector(target_map, map_hw)
                    repairable = _lane_target_label(float(nearest_fn_gt_distance), nearest_fn_center_mean)
                    row: dict[str, Any] = {
                        "batch_index": int(batch_index),
                        "sample_index": int(global_sample_index),
                        "sample_batch_index": int(sample_batch_index),
                        "pred_index": int(pred_index),
                        "split_id": int(_split_id(global_sample_index)),
                        "sample_gt_lane_count": int(len(gt_lanes)),
                        "sample_pred_lane_count": int(len(pred_lanes)),
                        "sample_matched_lane_count": int(len(gt_to_pred)),
                        "sample_unmatched_pred_count": int(len(unmatched_pred_indices)),
                        "nearest_fn_gt_index": int(nearest_fn_gt_index),
                        "nearest_fn_gt_distance": float(nearest_fn_gt_distance),
                        "nearest_fn_gt_distance_bin": _distance_bin(float(nearest_fn_gt_distance)),
                        "nearest_fn_gt_center_point_mean": nearest_fn_center_mean,
                        "repairable_le120_any_center": bool(repairable),
                        "repairable_le80_center050": bool(float(nearest_fn_gt_distance) <= 80.0 and nearest_fn_center_mean >= 0.50),
                        "pred_points_json": _points_json(pred_lane),
                        "nearest_fn_gt_points_json": _points_json(
                            gt_lanes[nearest_fn_gt_index] if nearest_fn_gt_index >= 0 else None
                        ),
                    }
                    for key, value in scalar_features.items():
                        row[str(key)] = float(value) if isinstance(value, (int, float)) and math.isfinite(float(value)) else value
                    candidates.append(
                        Candidate(
                            sample_index=int(global_sample_index),
                            sample_batch_index=int(sample_batch_index),
                            pred_index=int(pred_index),
                            split_id=int(_split_id(global_sample_index)),
                            feature=feature_vector.astype(np.float32),
                            target=target_vector.astype(np.float32) if isinstance(target_vector, np.ndarray) else None,
                            pred_points_map=pred_sampled_map.astype(np.float32),
                            meta=sample_meta,
                            map_hw=map_hw,
                            row=row,
                        )
                    )
                global_sample_index += 1

    if not raw_batches:
        raise ValueError("no validation batches were processed")

    model_payload = _train_oof_models(candidates) if candidates else {"model_summaries": []}
    repair_topk = int(args.repair_topk)
    if repair_topk <= 0:
        repair_topk = _auto_repair_topk(global_sample_index)
    eligible = [
        candidate
        for candidate in candidates
        if math.isfinite(float(candidate.score)) and isinstance(candidate.repaired_points_map, np.ndarray)
    ]
    selected = sorted(eligible, key=lambda candidate: float(candidate.score), reverse=True)[: max(0, repair_topk)]
    selected_ids = {id(candidate) for candidate in selected}
    repaired_predictions_all = copy.deepcopy(baseline_predictions_all)
    for candidate in candidates:
        candidate.selected = id(candidate) in selected_ids
        if not candidate.selected or candidate.repaired_points_map is None:
            continue
        repaired_points = _map_points_to_raw(candidate.repaired_points_map, candidate.meta, candidate.map_hw)
        sample_prediction = repaired_predictions_all[candidate.sample_index]
        if int(candidate.pred_index) < len(sample_prediction.get("lanes", [])):
            sample_prediction["lanes"][candidate.pred_index] = dict(sample_prediction["lanes"][candidate.pred_index])
            sample_prediction["lanes"][candidate.pred_index]["points_xy"] = repaired_points
        movement = float(np.linalg.norm(candidate.repaired_points_map - candidate.pred_points_map, axis=1).mean())
        candidate.row["repair_mean_move_map_px"] = movement

    merged_raw = _merge_raw_batches(raw_batches)
    baseline_metrics = augment_lane_family_metrics(summarize_pv26_metrics(baseline_predictions_all, merged_raw))
    repaired_metrics = augment_lane_family_metrics(summarize_pv26_metrics(repaired_predictions_all, merged_raw))
    baseline_tasks = {task: _metric_payload(baseline_metrics, task) for task in ("lane", "stop_line", "crosswalk")}
    repaired_tasks = {task: _metric_payload(repaired_metrics, task) for task in ("lane", "stop_line", "crosswalk")}
    deltas = {task: _task_delta(repaired_tasks[task], baseline_tasks[task]) for task in ("lane", "stop_line", "crosswalk")}

    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        row = dict(candidate.row)
        row["repair_score"] = float(candidate.score) if math.isfinite(float(candidate.score)) else ""
        row["selected"] = bool(candidate.selected)
        row["has_repair_prediction"] = isinstance(candidate.repaired_points_map, np.ndarray)
        row.setdefault("repair_mean_move_map_px", 0.0)
        rows.append(row)

    selected_rows = [row for row in rows if bool(row.get("selected"))]
    selected_positive = [row for row in selected_rows if bool(row.get("repairable_le120_any_center"))]
    summary = {
        "checkpoint": str(checkpoint),
        "source_run": str(source_run),
        "scenario_path": str(scenario_path),
        "lane60_experiment": str(args.lane60_experiment),
        "max_val_batches": int(args.max_val_batches),
        "validation_epoch": int(args.validation_epoch),
        "lane_flip_variant": str(args.lane_flip_variant),
        "repair_topk": int(repair_topk),
        "candidate_count": int(len(candidates)),
        "selected_count": int(len(selected_rows)),
        "selected_repairable_le120_any_center": int(len(selected_positive)),
        "selected_precision_le120_any_center": float(len(selected_positive) / max(1, len(selected_rows))),
        "baseline": baseline_tasks,
        "repaired": repaired_tasks,
        "delta": deltas,
        "models": model_payload["model_summaries"],
        "interpretation": (
            "Out-of-fold learned no-GT replay. GT labels train the opposite fold's "
            "repairability and geometry models; held-out candidates are selected by "
            "predicted score and repaired with dense lane ROI features before metrics "
            "are recomputed. This is stronger than an oracle replay but still a probe, "
            "not a deployed runtime contract."
        ),
    }
    return summary | {"candidate_rows": rows}


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = run_probe(args)
    candidate_rows = list(payload.pop("candidate_rows"))
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(output_dir / "lane_feature_roi_repair_candidates.csv", candidate_rows)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
