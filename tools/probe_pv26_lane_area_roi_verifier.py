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
    _build_scenario,
    _forward_predictions,
    _json_ready,
    _lane_features,
    _nearest_gt,
    _task_delta,
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
    parser.add_argument(
        "--alignment-context-features",
        action="store_true",
        help=(
            "Append no-GT geometry context between each dropped candidate and "
            "the retained baseline lane predictions. This tests an "
            "instance-alignment FP-control signal, not a verifier threshold sweep."
        ),
    )
    parser.add_argument("--verifier-epochs", type=int, default=40)
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
                raw_candidates = _raw_lane_candidates(
                    maps=maps,
                    meta=sample_meta,
                    postprocess_config=postprocess_config,
                )
                sample_candidate_index = 0
                for candidate in raw_candidates:
                    if not _is_dropped_by_lane_filter(candidate, postprocess_config=postprocess_config):
                        continue
                    if _near_any_lane(
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
                    features, _, _ = _lane_features(
                        candidate,
                        predictions=sample_prediction_tensors,
                        maps=maps,
                        meta=sample_meta,
                    )
                    if bool(args.alignment_context_features):
                        features = np.concatenate(
                            [features, _alignment_context_features(candidate, baseline_lanes)]
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
) -> tuple[LaneAreaRoiVerifierNet, dict[str, Any]]:
    if not examples:
        raise ValueError("no area-ROI verifier training examples collected")
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
    generator.manual_seed(int(args.seed))
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
    }
    return model, summary


def _save_verifier_model(
    path: str,
    *,
    model: LaneAreaRoiVerifierNet,
    train_summary: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    if not path:
        return
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
    model: LaneAreaRoiVerifierNet,
    args: argparse.Namespace,
    device: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    repaired = [dict(sample, lanes=[dict(lane) for lane in sample.get("lanes", [])]) for sample in predictions_all]
    if not examples:
        return repaired, []
    features = torch.tensor(np.stack([row["features"] for row in examples]), dtype=torch.float32, device=device)
    features = (features - model.feature_mean) / model.feature_std
    with torch.no_grad():
        probabilities = torch.sigmoid(model(features)).detach().cpu().numpy().astype(np.float32)
    by_sample: dict[int, list[tuple[float, int]]] = {}
    for index, (example, probability) in enumerate(zip(examples, probabilities.tolist())):
        by_sample.setdefault(int(example["sample_index"]), []).append((float(probability), int(index)))
    selected_indices: set[int] = set()
    for sample_index, candidates in by_sample.items():
        candidates.sort(reverse=True)
        accepted: list[dict[str, Any]] = list(repaired[sample_index].get("lanes", [])) if 0 <= sample_index < len(repaired) else []
        for probability, index in candidates:
            if len(selected_indices) >= len(examples):
                break
            if probability < float(args.quality_threshold):
                continue
            candidate = dict(examples[index]["candidate"])
            if _near_any_lane(
                candidate,
                accepted,
                threshold_px=float(args.candidate_duplicate_distance_px),
            ):
                continue
            selected_indices.add(int(index))
            accepted.append(candidate)
            if sum(1 for item in selected_indices if int(examples[item]["sample_index"]) == int(sample_index)) >= int(
                args.max_appends_per_sample
            ):
                break
    rows: list[dict[str, Any]] = []
    for index, example in enumerate(examples):
        sample_index = int(example["sample_index"])
        selected = int(index) in selected_indices
        probability = float(probabilities[index])
        if selected and 0 <= sample_index < len(repaired):
            candidate = dict(example["candidate"])
            candidate["area_roi_verifier_score"] = probability
            repaired[sample_index].setdefault("lanes", []).append(candidate)
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
    scenario_args.max_val_batches = int(args.val_start_batch) + int(args.max_val_batches)
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
        model, train_summary = _train_verifier(train_examples, args=args, device=device)
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
        "max_val_batches": int(args.max_val_batches),
        "val_start_batch": int(args.val_start_batch),
        "eval_chunk_batches": int(args.eval_chunk_batches),
        "evaluated_val_batches": int(eval_payload["evaluated_val_batches"]),
        "validation_epoch": int(args.validation_epoch),
        "positive_distance_px": float(args.positive_distance_px),
        "negative_distance_px": float(args.negative_distance_px),
        "quality_threshold": float(args.quality_threshold),
        "max_appends_per_sample": int(args.max_appends_per_sample),
        "alignment_context_features": bool(args.alignment_context_features),
        "train_summary": train_summary,
        "val_candidate_count": int(eval_payload["val_candidate_count"]),
        "selected_candidate_count": int(eval_payload["selected_candidate_count"]),
        "selected_oracle_positive_count": int(eval_payload["selected_oracle_positive_count"]),
        "baseline": baseline_tasks,
        "repaired": repaired_tasks,
        "delta": deltas,
        "interpretation": (
            "No-GT runtime replay of a learned line-ROI verifier over raw seg-first "
            "lane candidates dropped by the default lane bbox/area filter. GT is used "
            "only for train labels and final audit metrics, not candidate selection."
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
