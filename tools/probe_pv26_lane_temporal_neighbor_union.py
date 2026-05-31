from __future__ import annotations

import argparse
import copy
from dataclasses import replace
import json
import math
from pathlib import Path
import re
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
from model.engine import raw_batch_for_metrics
from model.engine._trainer_epochs import _merge_raw_batches
from model.engine.batch import augment_lane_family_metrics
from model.engine.lane_segfirst_vectorizer import lane_segfirst_prediction_maps
from model.engine.metrics import _extract_gt_samples, summarize_pv26_metrics
from model.engine.postprocess import postprocess_pv26_batch
from tools.evaluate_pv26_lane60_checkpoint import _advance_validation_sampler, _postprocess_override_config
from tools.probe_pv26_lane_flip_tta import _merge_lane_dense_predictions, _unflip_lane_dense_outputs
from tools.probe_pv26_lane_fn_recovery_audit import (
    DEFAULT_CHECKPOINT,
    SOURCE_RUN,
    _distance_bin,
    _lane_distance,
    _match_predictions,
    _metric,
    _nearest_gt_lane,
    _points_json,
    _prediction_shape_features,
    _track_map_evidence,
)
from tools.probe_pv26_lane_instance_evidence import _resolve_dataset_root, _resolve_device, _write_csv
from tools.probe_pv26_lane60_postprocess_thresholds import _detach_to_cpu
from tools.probe_pv26_stopline_candidate_pool import (
    _fit_raw_patch_mlp,
    _predict_raw_patch_mlp,
    _standardize_from_train,
)
from tools.pv26_train import cli as train_cli
from tools.pv26_train import config as train_config_api
from tools.run_pv26_lane60_probe import _lane60_scenario


_TRAILING_FRAME_RE = re.compile(r"^(?P<prefix>.*?)(?P<frame>\d+)$")
SCORE_KEY = "lane_temporal_mlp_score"
LANE_TEMPORAL_FEATURES = (
    "temporal_abs_offset",
    "temporal_offset_sign",
    "temporal_neighbor_lane_count_norm",
    "temporal_base_lane_count_norm",
    "temporal_baseline_fn_count_norm",
    "temporal_nearest_existing_distance_norm",
    "current_frame_track_pixels_norm",
    "current_frame_center_mask_mean",
    "current_frame_center_mask_q10",
    "current_frame_center_mask_active05",
    "current_frame_center_point_mean",
    "current_frame_center_point_q10",
    "current_frame_center_point_active05",
    "current_frame_center_point_low_run05",
    "current_frame_support_point_mean",
    "current_frame_support_point_q10",
    "current_frame_support_mask_mean",
    "pred_point_count_norm",
    "pred_polyline_length_norm",
    "pred_center_x_norm",
    "pred_center_y_norm",
    "pred_bbox_width_norm",
    "pred_bbox_height_norm",
    "pred_bbox_aspect_norm",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Lane temporal-neighbor smoke. It keeps the checkpoint fixed, decodes the current "
            "validation sample plus immediate neighboring frame predictions, and adds only "
            "neighbor lane candidates supported by the current frame centerline map."
        )
    )
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--source-run", default=str(SOURCE_RUN))
    parser.add_argument("--lane60-experiment", default="core_centerline_refine_row_scan_tangent_link")
    parser.add_argument("--preset", default="default")
    parser.add_argument("--phase-index", type=int, default=4)
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
    parser.add_argument("--lane-flip-variant", choices=("baseline", "flip_centerline_avg"), default="flip_centerline_avg")
    parser.add_argument("--neighbor-offsets", default="-1,1")
    parser.add_argument(
        "--verifier-enabled",
        type=int,
        choices=(0, 1),
        default=0,
        help=(
            "Train a train-split no-GT MLP verifier for temporal lane candidates. "
            "Default 0 preserves the historical fixed temporal-neighbor smoke."
        ),
    )
    parser.add_argument("--train-record-batches", type=int, default=64)
    parser.add_argument("--verifier-epochs", type=int, default=60)
    parser.add_argument("--verifier-lr", type=float, default=1.0e-3)
    parser.add_argument("--threshold-grid", type=int, default=101)
    parser.add_argument("--verifier-top-k-per-sample", type=int, default=12)
    parser.add_argument("--current-center-mean-min", type=float, default=0.50)
    parser.add_argument("--dedupe-distance", type=float, default=40.0)
    parser.add_argument("--max-added-per-sample", type=int, default=2)
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


def _parse_neighbor_offsets(value: str) -> tuple[int, ...]:
    offsets: list[int] = []
    for item in str(value).split(","):
        item = item.strip()
        if not item:
            continue
        offset = int(item)
        if offset == 0:
            raise ValueError("neighbor offset 0 would duplicate the target frame")
        offsets.append(offset)
    if not offsets:
        raise ValueError("at least one neighbor offset is required")
    return tuple(dict.fromkeys(offsets))


def _split_temporal_sample_id(sample_id: str) -> tuple[str, int, int] | None:
    match = _TRAILING_FRAME_RE.match(str(sample_id))
    if match is None:
        return None
    digits = str(match.group("frame"))
    return str(match.group("prefix")), int(digits), len(digits)


def _neighbor_sample_id(sample_id: str, offset: int) -> str | None:
    parsed = _split_temporal_sample_id(sample_id)
    if parsed is None:
        return None
    prefix, frame, width = parsed
    neighbor_frame = frame + int(offset)
    if neighbor_frame < 0:
        return None
    return f"{prefix}{neighbor_frame:0{width}d}"


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


def _nearest_lane_distance(lane: dict[str, Any], lanes: list[dict[str, Any]]) -> float:
    best = math.inf
    for existing in lanes:
        best = min(best, _lane_distance(lane, existing))
    return float(best)


def _copy_lane(lane: dict[str, Any]) -> dict[str, Any]:
    copied = dict(lane)
    copied["points_xy"] = [[float(point[0]), float(point[1])] for point in lane.get("points_xy", [])]
    return copied


def _build_scenario(args: argparse.Namespace) -> tuple[Any, Path, dict[str, Any], Any, Any]:
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
        encode_train_batches_in_loader=False,
        encode_val_batches_in_loader=False,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        prefetch_factor=None,
    )
    return scenario, scenario_path, options, phase, train_config


class _Predictor:
    def __init__(
        self,
        *,
        dataset: Any,
        evaluator: Any,
        postprocess_config: Any,
        lane_flip_variant: str,
    ) -> None:
        self.dataset = dataset
        self.evaluator = evaluator
        self.postprocess_config = postprocess_config
        self.lane_flip_variant = str(lane_flip_variant)
        self.cache: dict[int, dict[str, Any]] = {}

    def predict_index(self, dataset_index: int) -> dict[str, Any]:
        dataset_index = int(dataset_index)
        cached = self.cache.get(dataset_index)
        if cached is not None:
            return cached
        batch = collate_pv26_samples([self.dataset[dataset_index]])
        encoded = self.evaluator.prepare_batch(batch)
        predictions = _detach_to_cpu(self.evaluator.forward_encoded_batch(encoded))
        flip_predictions = None
        if self.lane_flip_variant != "baseline":
            flipped_encoded = dict(encoded)
            flipped_encoded["image"] = torch.flip(encoded["image"], dims=(-1,))
            flip_predictions = _unflip_lane_dense_outputs(
                _detach_to_cpu(self.evaluator.forward_encoded_batch(flipped_encoded))
            )
        postprocess_predictions = (
            _merge_lane_dense_predictions(
                predictions,
                flip_predictions if flip_predictions is not None else {},
                variant=self.lane_flip_variant,
            )
            if self.lane_flip_variant != "baseline"
            else predictions
        )
        meta = _detach_to_cpu(encoded["meta"])
        batch_predictions = postprocess_pv26_batch(postprocess_predictions, meta, config=self.postprocess_config)
        payload = {
            "prediction": batch_predictions[0],
            "meta": meta[0],
        }
        self.cache[dataset_index] = payload
        return payload


def _candidate_audit_labels(
    lane: dict[str, Any],
    gt_lanes: list[dict[str, Any]],
    unmatched_gt_indices: set[int],
) -> dict[str, Any]:
    nearest_any_gt_index, nearest_any_gt_distance = _nearest_gt_lane(lane, gt_lanes)
    nearest_fn_gt_index, nearest_fn_gt_distance = _nearest_gt_lane(
        lane,
        gt_lanes,
        allowed_indices=unmatched_gt_indices,
    )
    return {
        "nearest_any_gt_index": int(nearest_any_gt_index),
        "nearest_any_gt_distance": float(nearest_any_gt_distance),
        "nearest_any_gt_distance_bin": _distance_bin(float(nearest_any_gt_distance)),
        "nearest_fn_gt_index": int(nearest_fn_gt_index),
        "nearest_fn_gt_distance": float(nearest_fn_gt_distance),
        "nearest_fn_gt_distance_bin": _distance_bin(float(nearest_fn_gt_distance)),
        "would_match_baseline_fn": bool(float(nearest_fn_gt_distance) <= 40.0),
    }


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if math.isfinite(out) else float(default)


def _lane_from_points_json(
    value: Any,
    *,
    score: float = 0.0,
    class_name: Any = "white_lane",
    lane_type: Any = "solid",
) -> dict[str, Any] | None:
    try:
        points = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    try:
        array = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    except (TypeError, ValueError):
        return None
    if array.shape[0] < 2:
        return None
    return {
        "points_xy": [[float(x), float(y)] for x, y in array.tolist()],
        "score": float(score),
        "class_name": str(class_name or "white_lane"),
        "lane_type": str(lane_type or "solid"),
        "temporal_mlp_score": float(score),
    }


def _candidate_shape_feature_block(candidate: dict[str, Any]) -> dict[str, float]:
    shape = _prediction_shape_features(candidate)
    return {
        "pred_point_count_norm": min(_finite_float(shape.get("pred_point_count")) / 96.0, 2.0),
        "pred_polyline_length_norm": min(_finite_float(shape.get("pred_polyline_length")) / 900.0, 2.0),
        "pred_center_x_norm": _finite_float(shape.get("pred_center_x")) / 1280.0,
        "pred_center_y_norm": _finite_float(shape.get("pred_center_y")) / 720.0,
        "pred_bbox_width_norm": min(_finite_float(shape.get("pred_bbox_width")) / 1280.0, 2.0),
        "pred_bbox_height_norm": min(_finite_float(shape.get("pred_bbox_height")) / 720.0, 2.0),
        "pred_bbox_aspect_norm": min(_finite_float(shape.get("pred_bbox_aspect")) / 20.0, 2.0),
    }


def _lane_temporal_feature_vector(row: dict[str, Any]) -> list[float]:
    return [_finite_float(row.get(name, 0.0)) for name in LANE_TEMPORAL_FEATURES]


def _lane_temporal_feature_matrix(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    if not rows:
        return np.zeros((0, len(LANE_TEMPORAL_FEATURES)), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    matrix = np.asarray([_lane_temporal_feature_vector(row) for row in rows], dtype=np.float32)
    labels = np.asarray(
        [1.0 if bool(row.get("would_match_baseline_fn", False)) else 0.0 for row in rows],
        dtype=np.float32,
    )
    return matrix, labels


def _score_temporal_candidate_rows(
    *,
    train_rows: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
    epochs: int,
    lr: float,
) -> dict[str, Any]:
    train_x, train_y = _lane_temporal_feature_matrix(train_rows)
    val_x, val_y = _lane_temporal_feature_matrix(val_rows)
    if train_x.shape[0] == 0 or val_x.shape[0] == 0:
        raise ValueError("lane temporal verifier requires non-empty train and validation candidates")
    combined_x = np.concatenate([train_x, val_x], axis=0)
    combined_std, mean, std = _standardize_from_train(train_x.astype(np.float64), combined_x.astype(np.float64))
    train_std = combined_std[: train_x.shape[0]].astype(np.float32)
    val_std = combined_std[train_x.shape[0] :].astype(np.float32)
    model = _fit_raw_patch_mlp(train_std, train_y.astype(np.float32), epochs=int(epochs), lr=float(lr))
    train_scores = _predict_raw_patch_mlp(model, train_std)
    val_scores = _predict_raw_patch_mlp(model, val_std)
    for row, score in zip(train_rows, train_scores.tolist()):
        row[SCORE_KEY] = float(score)
    for row, score in zip(val_rows, val_scores.tolist()):
        row[SCORE_KEY] = float(score)
    return {
        "train_candidate_count": int(train_x.shape[0]),
        "train_positive_count": int(train_y.sum()),
        "val_candidate_count": int(val_x.shape[0]),
        "val_positive_count": int(val_y.sum()),
        "feature_dim": int(train_x.shape[1]),
        "mean_dim": int(mean.shape[0]),
        "std_min": float(std.min()) if std.size else 0.0,
    }


def _best_score_threshold(
    rows: list[dict[str, Any]],
    *,
    score_key: str,
    top_k_per_sample: int,
    grid_size: int,
) -> float:
    by_sample: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_sample.setdefault(int(row.get("sample_index", -1)), []).append(row)
    scored_labels: list[tuple[float, int]] = []
    for sample_rows in by_sample.values():
        ranked = sorted(sample_rows, key=lambda item: float(item.get(score_key, 0.0)), reverse=True)[
            : max(1, int(top_k_per_sample))
        ]
        scored_labels.extend(
            (
                float(row.get(score_key, 0.0)),
                1 if bool(row.get("would_match_baseline_fn", False)) else 0,
            )
            for row in ranked
        )
    total_positive = int(sum(label for _score, label in scored_labels))
    if not scored_labels:
        return 0.5
    if total_positive <= 0:
        return 1.0
    best_threshold = 0.5
    best_key: tuple[float, int, int] = (-1.0, -1, 0)
    for threshold in np.linspace(0.0, 1.0, max(2, int(grid_size))).tolist():
        tp = 0
        fp = 0
        for score, label in scored_labels:
            if float(score) < float(threshold):
                continue
            if label:
                tp += 1
            else:
                fp += 1
        fn = max(0, total_positive - tp)
        f1 = 0.0 if tp <= 0 else (2.0 * float(tp)) / (2.0 * float(tp) + float(fp) + float(fn))
        key = (float(f1), int(tp), -int(fp))
        if key > best_key:
            best_key = key
            best_threshold = float(threshold)
    return float(best_threshold)


def _select_verified_temporal_lanes(
    rows: list[dict[str, Any]],
    *,
    threshold: float,
    top_k_per_sample: int,
    max_added_per_sample: int,
    dedupe_distance: float,
    baseline_lanes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    ranked = sorted(rows, key=lambda row: float(row.get(SCORE_KEY, 0.0)), reverse=True)
    for row in ranked[: max(1, int(top_k_per_sample))]:
        if len(selected) >= max(0, int(max_added_per_sample)):
            break
        score = float(row.get(SCORE_KEY, 0.0))
        if score < float(threshold):
            continue
        candidate = _lane_from_points_json(
            row.get("candidate_points_json", "[]"),
            score=score,
            class_name=row.get("candidate_class_name", "white_lane"),
            lane_type=row.get("candidate_lane_type", "solid"),
        )
        if candidate is None:
            continue
        if _nearest_lane_distance(candidate, [*baseline_lanes, *selected]) <= float(dedupe_distance):
            continue
        selected.append(candidate)
        row["verifier_selected"] = True
    return selected


def _temporal_candidates_for_sample(
    *,
    sample_index: int,
    sample_batch_index: int,
    target_meta: dict[str, Any],
    target_maps: dict[str, torch.Tensor],
    base_lanes: list[dict[str, Any]],
    gt_lanes: list[dict[str, Any]],
    unmatched_gt_indices: set[int],
    neighbor_predictions: list[tuple[int, int, dict[str, Any]]],
    center_mean_min: float,
    dedupe_distance: float,
    max_added_per_sample: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    proposed: list[tuple[float, dict[str, Any], dict[str, Any]]] = []
    for offset, neighbor_dataset_index, neighbor_prediction in neighbor_predictions:
        neighbor_lanes = list(neighbor_prediction.get("lanes", []))
        for neighbor_lane_index, lane in enumerate(neighbor_lanes):
            candidate = _copy_lane(lane)
            evidence = _track_map_evidence(
                list(candidate.get("points_xy", [])),
                maps=target_maps,
                meta=target_meta,
                prefix="current_frame",
            )
            nearest_existing_distance = _nearest_lane_distance(candidate, base_lanes)
            row: dict[str, Any] = {
                "sample_index": int(sample_index),
                "sample_batch_index": int(sample_batch_index),
                "sample_id": str(target_meta.get("sample_id", "")),
                "neighbor_offset": int(offset),
                "temporal_abs_offset": min(abs(float(offset)) / 8.0, 2.0),
                "temporal_offset_sign": -1.0 if int(offset) < 0 else 1.0,
                "neighbor_dataset_index": int(neighbor_dataset_index),
                "neighbor_lane_index": int(neighbor_lane_index),
                "base_lane_count": int(len(base_lanes)),
                "neighbor_lane_count": int(len(neighbor_lanes)),
                "temporal_base_lane_count_norm": min(float(len(base_lanes)) / 16.0, 2.0),
                "temporal_neighbor_lane_count_norm": min(float(len(neighbor_lanes)) / 16.0, 2.0),
                "temporal_baseline_fn_count_norm": min(float(len(unmatched_gt_indices)) / 16.0, 2.0),
                "nearest_existing_distance": float(nearest_existing_distance),
                "temporal_nearest_existing_distance_norm": min(float(nearest_existing_distance) / 240.0, 4.0),
                "nearest_existing_distance_bin": _distance_bin(float(nearest_existing_distance)),
                "candidate_points_json": _points_json(candidate),
                "candidate_class_name": str(candidate.get("class_name", "white_lane")),
                "candidate_lane_type": str(candidate.get("lane_type", "solid")),
                "selected": False,
                "verifier_selected": False,
                "blocked_reason": "",
            }
            row.update(evidence)
            row["current_frame_track_pixels_norm"] = min(
                float(row.get("current_frame_track_pixels", 0.0)) / 2000.0,
                4.0,
            )
            row.update(_candidate_shape_feature_block(candidate))
            row.update(_candidate_audit_labels(candidate, gt_lanes, unmatched_gt_indices))
            center_mean = float(evidence.get("current_frame_center_point_mean", 0.0))
            if center_mean < float(center_mean_min):
                row["blocked_reason"] = "low_current_center_mean"
            elif nearest_existing_distance <= float(dedupe_distance):
                row["blocked_reason"] = "duplicate_current_lane"
            else:
                proposed.append((center_mean, row, candidate))
            rows.append(row)

    proposed.sort(key=lambda item: float(item[0]), reverse=True)
    added: list[dict[str, Any]] = []
    for _score, row, candidate in proposed:
        if len(added) >= max(0, int(max_added_per_sample)):
            row["blocked_reason"] = "sample_add_cap"
            continue
        if _nearest_lane_distance(candidate, [*base_lanes, *added]) <= float(dedupe_distance):
            row["blocked_reason"] = "duplicate_selected_lane"
            continue
        row["selected"] = True
        row["blocked_reason"] = ""
        added.append(candidate)
    return added, rows


def _collect_temporal_records(
    *,
    loader: Any,
    evaluator: Any,
    predictor: _Predictor,
    postprocess_config: Any,
    record_index_by_key: dict[tuple[str, str, str], int],
    max_batches: int,
    neighbor_offsets: tuple[int, ...],
    args: argparse.Namespace,
    split_name: str,
) -> dict[str, Any]:
    baseline_predictions_all: list[dict[str, Any]] = []
    fixed_predictions_all: list[dict[str, Any]] = []
    raw_batches: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    global_sample_index = 0

    with torch.no_grad():
        for batch_index, batch in enumerate(loader, start=1):
            if batch_index > int(max_batches):
                break
            if batch_index == 1 or batch_index % 20 == 0:
                print(f"[lane_temporal] collect {split_name} batch {batch_index}/{max_batches}", flush=True)
            raw_batch = raw_batch_for_metrics(batch)
            if raw_batch is None:
                raise ValueError("loader must provide raw batches for metrics")
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
            raw_batches.append(raw_batch)
            baseline_predictions_all.extend(copy.deepcopy(batch_predictions))

            fixed_batch_predictions: list[dict[str, Any]] = []
            for sample_batch_index, (sample_pred, sample_gt, sample_meta) in enumerate(
                zip(batch_predictions, batch_gt, meta)
            ):
                sample_id = str(sample_meta.get("sample_id", ""))
                dataset_key = str(sample_meta.get("dataset_key", ""))
                split = str(sample_meta.get("split", ""))
                target_maps = lane_segfirst_prediction_maps(postprocess_predictions, batch_index=sample_batch_index)
                base_lanes = list(sample_pred.get("lanes", []))
                gt_lanes = list(sample_gt.get("lanes", []))
                _pred_to_gt, gt_to_pred = _match_predictions(base_lanes, gt_lanes)
                unmatched_gt_indices = set(range(len(gt_lanes))) - set(gt_to_pred)
                neighbor_payloads: list[tuple[int, int, dict[str, Any]]] = []
                missing_neighbors = 0
                for offset in neighbor_offsets:
                    neighbor_id = _neighbor_sample_id(sample_id, offset)
                    if neighbor_id is None:
                        missing_neighbors += 1
                        continue
                    neighbor_index = record_index_by_key.get((dataset_key, split, neighbor_id))
                    if neighbor_index is None:
                        missing_neighbors += 1
                        continue
                    neighbor_payload = predictor.predict_index(neighbor_index)
                    neighbor_payloads.append((int(offset), int(neighbor_index), dict(neighbor_payload["prediction"])))

                fixed_sample = copy.deepcopy(sample_pred)
                added_lanes, rows = _temporal_candidates_for_sample(
                    sample_index=global_sample_index,
                    sample_batch_index=sample_batch_index,
                    target_meta=sample_meta,
                    target_maps=target_maps,
                    base_lanes=base_lanes,
                    gt_lanes=gt_lanes,
                    unmatched_gt_indices=unmatched_gt_indices,
                    neighbor_predictions=neighbor_payloads,
                    center_mean_min=float(args.current_center_mean_min),
                    dedupe_distance=float(args.dedupe_distance),
                    max_added_per_sample=int(args.max_added_per_sample),
                )
                for row in rows:
                    row["split_name"] = str(split_name)
                    row["batch_index"] = int(batch_index)
                fixed_sample["lanes"] = [*list(fixed_sample.get("lanes", [])), *added_lanes]
                fixed_batch_predictions.append(fixed_sample)
                candidate_rows.extend(rows)
                sample_rows.append(
                    {
                        "split_name": str(split_name),
                        "batch_index": int(batch_index),
                        "sample_index": int(global_sample_index),
                        "sample_batch_index": int(sample_batch_index),
                        "sample_id": sample_id,
                        "dataset_key": dataset_key,
                        "split": split,
                        "base_lane_count": int(len(base_lanes)),
                        "gt_lane_count": int(len(gt_lanes)),
                        "baseline_fn_lane_count": int(len(unmatched_gt_indices)),
                        "temporal_neighbor_count": int(len(neighbor_payloads)),
                        "missing_neighbor_count": int(missing_neighbors),
                        "temporal_candidate_count": int(len(rows)),
                        "temporal_selected_count": int(len(added_lanes)),
                    }
                )
                global_sample_index += 1
            fixed_predictions_all.extend(fixed_batch_predictions)

    return {
        "baseline_predictions": baseline_predictions_all,
        "fixed_predictions": fixed_predictions_all,
        "raw_batches": raw_batches,
        "candidate_rows": candidate_rows,
        "sample_rows": sample_rows,
    }


def _metrics_for_predictions(
    *,
    predictions: list[dict[str, Any]],
    raw_batches: list[dict[str, Any]],
) -> dict[str, Any]:
    if not raw_batches:
        raise ValueError("no batches were processed")
    merged_raw = _merge_raw_batches(raw_batches)
    return augment_lane_family_metrics(summarize_pv26_metrics(predictions, merged_raw))


def _apply_verifier_to_predictions(
    *,
    baseline_predictions: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    threshold: float,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], int, int]:
    repaired = [copy.deepcopy(sample) for sample in baseline_predictions]
    by_sample: dict[int, list[dict[str, Any]]] = {}
    for row in candidate_rows:
        by_sample.setdefault(int(row.get("sample_index", -1)), []).append(row)
    selected_count = 0
    selected_positive_count = 0
    for sample_index, rows in by_sample.items():
        if sample_index < 0 or sample_index >= len(repaired):
            continue
        base_lanes = list(repaired[sample_index].get("lanes", []))
        added = _select_verified_temporal_lanes(
            rows,
            threshold=float(threshold),
            top_k_per_sample=int(args.verifier_top_k_per_sample),
            max_added_per_sample=int(args.max_added_per_sample),
            dedupe_distance=float(args.dedupe_distance),
            baseline_lanes=base_lanes,
        )
        if not added:
            continue
        for lane in added:
            repaired[sample_index].setdefault("lanes", []).append(lane)
        selected_count += int(len(added))
        selected_positive_count += int(
            sum(1 for row in rows if bool(row.get("verifier_selected")) and bool(row.get("would_match_baseline_fn")))
        )
    return repaired, int(selected_count), int(selected_positive_count)


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    source_run = Path(args.source_run).expanduser().resolve()
    neighbor_offsets = _parse_neighbor_offsets(str(args.neighbor_offsets))
    scenario, scenario_path, options, phase, train_config = _build_scenario(args)
    phase_index = int(tuple(options["selected_phase_indices"])[0])

    train_cli._configure_torch_multiprocessing()
    dataset = train_cli.PV26CanonicalDataset(
        train_cli._existing_dataset_roots(scenario),
        train_augmentation=False,
        progress_callback=lambda message: print(f"[lane_temporal] {message}", flush=True),
    )
    train_loader, val_loader = train_cli._build_phase_train_loaders(dataset, train_config=train_config, phase=phase)
    if bool(int(args.verifier_enabled)) and train_loader is None:
        raise ValueError("lane temporal verifier requires training batches")
    if val_loader is None:
        raise ValueError("lane temporal probe requires validation batches")
    _advance_validation_sampler(val_loader, validation_epoch=int(args.validation_epoch))

    trainer = train_cli._build_phase_trainer(phase, train_config)
    trainer.load_model_weights(checkpoint, map_location=train_config.device)
    postprocess_config = _postprocess_override_config(args, trainer) or train_cli._build_postprocess_config(train_config)
    evaluator = trainer.build_evaluator()
    evaluator.adapter.raw_model.eval()
    evaluator.heads.eval()
    predictor = _Predictor(
        dataset=dataset,
        evaluator=evaluator,
        postprocess_config=postprocess_config,
        lane_flip_variant=str(args.lane_flip_variant),
    )
    record_index_by_key = {
        (str(record.dataset_key), str(record.split), str(record.sample_id)): int(index)
        for index, record in enumerate(dataset.records)
    }

    val_records = _collect_temporal_records(
        loader=val_loader,
        evaluator=evaluator,
        predictor=predictor,
        postprocess_config=postprocess_config,
        record_index_by_key=record_index_by_key,
        max_batches=int(args.max_val_batches),
        neighbor_offsets=neighbor_offsets,
        args=args,
        split_name="val",
    )
    candidate_rows = list(val_records["candidate_rows"])
    sample_rows = list(val_records["sample_rows"])
    train_candidate_rows: list[dict[str, Any]] = []
    train_sample_rows: list[dict[str, Any]] = []
    verifier_threshold: float | None = None
    verifier_train_summary: dict[str, Any] | None = None
    verifier_selected_count = 0
    verifier_selected_would_match_count = 0

    if bool(int(args.verifier_enabled)):
        train_records = _collect_temporal_records(
            loader=train_loader,
            evaluator=evaluator,
            predictor=predictor,
            postprocess_config=postprocess_config,
            record_index_by_key=record_index_by_key,
            max_batches=int(args.train_record_batches),
            neighbor_offsets=neighbor_offsets,
            args=args,
            split_name="train",
        )
        train_candidate_rows = list(train_records["candidate_rows"])
        train_sample_rows = list(train_records["sample_rows"])
        verifier_train_summary = _score_temporal_candidate_rows(
            train_rows=train_candidate_rows,
            val_rows=candidate_rows,
            epochs=int(args.verifier_epochs),
            lr=float(args.verifier_lr),
        )
        verifier_threshold = _best_score_threshold(
            train_candidate_rows,
            score_key=SCORE_KEY,
            top_k_per_sample=int(args.verifier_top_k_per_sample),
            grid_size=int(args.threshold_grid),
        )
        repaired_predictions_all, verifier_selected_count, verifier_selected_would_match_count = (
            _apply_verifier_to_predictions(
                baseline_predictions=list(val_records["baseline_predictions"]),
                candidate_rows=candidate_rows,
                threshold=float(verifier_threshold),
                args=args,
            )
        )
    else:
        repaired_predictions_all = list(val_records["fixed_predictions"])

    baseline_metrics = _metrics_for_predictions(
        predictions=list(val_records["baseline_predictions"]),
        raw_batches=list(val_records["raw_batches"]),
    )
    repaired_metrics = _metrics_for_predictions(
        predictions=repaired_predictions_all,
        raw_batches=list(val_records["raw_batches"]),
    )
    baseline_tasks = {task: _metric_payload(baseline_metrics, task) for task in ("lane", "stop_line", "crosswalk")}
    repaired_tasks = {task: _metric_payload(repaired_metrics, task) for task in ("lane", "stop_line", "crosswalk")}
    deltas = {task: _task_delta(repaired_tasks[task], baseline_tasks[task]) for task in ("lane", "stop_line", "crosswalk")}
    selected_rows = [row for row in candidate_rows if bool(row.get("selected"))]
    selected_would_match = [row for row in selected_rows if bool(row.get("would_match_baseline_fn"))]
    if bool(int(args.verifier_enabled)):
        selected_count = int(verifier_selected_count)
        selected_would_match_count = int(verifier_selected_would_match_count)
        interpretation = (
            "This is a train-split no-GT temporal verifier probe. The MLP sees only temporal "
            "candidate geometry plus current-frame dense lane evidence; train GT labels select "
            "the verifier threshold, while validation GT is used only for audit labels and final metrics."
        )
    else:
        selected_count = int(len(selected_rows))
        selected_would_match_count = int(len(selected_would_match))
        interpretation = (
            "This is a no-GT temporal smoke, not a training result. Selection uses only immediate "
            "neighbor predictions plus current-frame centerline evidence and duplicate distance; "
            "GT is used only after selection for audit labels and final metrics."
        )
    summary = {
        "checkpoint": str(checkpoint),
        "source_run": str(source_run),
        "scenario_path": str(scenario_path),
        "lane60_experiment": str(args.lane60_experiment),
        "phase_index": int(phase_index),
        "max_val_batches": int(args.max_val_batches),
        "validation_epoch": int(args.validation_epoch),
        "lane_flip_variant": str(args.lane_flip_variant),
        "neighbor_offsets": list(neighbor_offsets),
        "verifier_enabled": bool(int(args.verifier_enabled)),
        "train_record_batches": int(args.train_record_batches),
        "verifier_epochs": int(args.verifier_epochs),
        "verifier_lr": float(args.verifier_lr),
        "threshold_grid": int(args.threshold_grid),
        "verifier_top_k_per_sample": int(args.verifier_top_k_per_sample),
        "verifier_threshold": None if verifier_threshold is None else float(verifier_threshold),
        "verifier_train_summary": verifier_train_summary,
        "current_center_mean_min": float(args.current_center_mean_min),
        "dedupe_distance": float(args.dedupe_distance),
        "max_added_per_sample": int(args.max_added_per_sample),
        "baseline": baseline_tasks,
        "repaired": repaired_tasks,
        "delta": deltas,
        "sample_count": int(len(sample_rows)),
        "sample_with_temporal_neighbor_count": int(
            sum(1 for row in sample_rows if int(row.get("temporal_neighbor_count", 0)) > 0)
        ),
        "candidate_count": int(len(candidate_rows)),
        "train_candidate_count": int(len(train_candidate_rows)),
        "train_sample_count": int(len(train_sample_rows)),
        "selected_count": selected_count,
        "selected_would_match_baseline_fn_count": selected_would_match_count,
        "fixed_rule_selected_count": int(len(selected_rows)),
        "fixed_rule_selected_would_match_baseline_fn_count": int(len(selected_would_match)),
        "interpretation": interpretation,
    }
    return summary | {
        "candidate_rows": [*train_candidate_rows, *candidate_rows],
        "sample_rows": [*train_sample_rows, *sample_rows],
    }


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = run_probe(args)
    candidate_rows = list(payload.pop("candidate_rows"))
    sample_rows = list(payload.pop("sample_rows"))
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(output_dir / "lane_temporal_neighbor_candidates.csv", candidate_rows)
    _write_csv(output_dir / "lane_temporal_neighbor_samples.csv", sample_rows)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
