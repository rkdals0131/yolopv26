from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import site
import sys
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
repo_root = str(REPO_ROOT)
if repo_root not in sys.path:
    site.addsitedir(repo_root)

from model.data.transform import transform_from_meta, transform_points
from model.engine._trainer_epochs import _merge_raw_batches
from model.engine.batch import augment_lane_family_metrics, raw_batch_for_metrics
from model.engine.metrics import _extract_gt_samples, summarize_pv26_metrics
from model.engine.postprocess import postprocess_pv26_batch
from tools.evaluate_pv26_lane60_checkpoint import _advance_validation_sampler, _postprocess_override_config
from tools.probe_pv26_lane_feature_roi_repair import (
    DEFAULT_CHECKPOINT,
    SOURCE_RUN,
    _build_scenario,
    _forward_predictions,
    _json_ready,
    _metric_payload,
    _task_delta,
)
from tools.probe_pv26_lane_flip_tta import _stop_line_distance
from tools.probe_pv26_lane_instance_evidence import _write_csv
from tools.pv26_train import cli as train_cli


STOP_LINE_MATCH_THRESHOLD_PX = 40.0
LINE_SAMPLE_COUNT = 21
LINE_SIDE_OFFSET_PX = 3.0
FEATURE_MAP_KEYS = (
    "stop_line_mask_logits",
    "stop_line_center_logits",
    "stop_line_selector_map_logits",
    "stop_line_axis_valid_logits",
    "stop_line_row_logits",
    "stop_line_x_logits",
)


class StoplineRetainedSuppressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(input_dim), int(hidden_dim)),
            nn.ReLU(inplace=True),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.ReLU(inplace=True),
            nn.Linear(int(hidden_dim), 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a small no-GT suppress-only verifier for retained runtime stop-line "
            "outputs. The verifier reads line geometry plus frozen dense-map support, "
            "then removes low-confidence baseline stop-line candidates without emitting "
            "new segments."
        )
    )
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--source-run", default=str(SOURCE_RUN))
    parser.add_argument("--lane60-experiment", default="stopline_projection_comp_runtime")
    parser.add_argument("--preset", default="default")
    parser.add_argument("--suppressor-train-batches", type=int, default=64)
    parser.add_argument("--max-val-batches", type=int, default=4)
    parser.add_argument("--validation-epoch", type=int, default=2)
    parser.add_argument("--train-batches", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dataset-root", default="")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--suppressor-epochs", type=int, default=80)
    parser.add_argument("--suppressor-batch-size", type=int, default=128)
    parser.add_argument("--suppressor-lr", type=float, default=1.0e-3)
    parser.add_argument("--suppressor-weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--keep-threshold", type=float, default=0.50)
    parser.add_argument("--seed", type=int, default=20260531)
    parser.add_argument("--backbone-weights", default="")
    parser.add_argument(
        "--lane-flip-variant",
        choices=("baseline", "flip_centerline_avg", "flip_centerline_avg_lane_cross_comp050"),
        default="flip_centerline_avg_lane_cross_comp050",
    )
    parser.add_argument("--lane-obj-threshold", type=float, default=None)
    parser.add_argument("--lane-segfirst-track-mode", default=None)
    parser.add_argument("--lane-segfirst-max-row-gap", type=int, default=None)
    parser.add_argument("--lane-segfirst-max-link-dx", type=float, default=None)
    parser.add_argument("--lane-segfirst-max-turn-degrees", type=float, default=None)
    parser.add_argument("--stop-line-mask-binary-threshold", type=float, default=None)
    parser.add_argument("--stop-line-min-instance-score", type=float, default=None)
    parser.add_argument("--stop-line-presence-threshold", type=float, default=None)
    parser.add_argument("--stop-line-projection-comp-enabled", type=int, choices=(0, 1), default=None)
    parser.add_argument("--stop-line-projection-comp-topk", type=int, default=None)
    parser.add_argument("--stop-line-projection-comp-max-predictions", type=int, default=None)
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


def _line_score(line: dict[str, Any]) -> float:
    candidates = (
        line.get("score"),
        line.get("center_score"),
        line.get("instance_score"),
        line.get("orientation_score"),
        line.get("projection_comp_score"),
    )
    values = [float(value) for value in candidates if isinstance(value, (int, float))]
    return max(values) if values else 0.0


def _line_scalar_features(
    line: dict[str, Any],
    *,
    meta: dict[str, Any],
    rank: int,
    candidate_count: int,
) -> list[float]:
    transform = transform_from_meta(meta)
    network_h, network_w = int(transform.network_hw[0]), int(transform.network_hw[1])
    network_diag = math.hypot(float(network_w), float(network_h))
    raw_points = np.asarray(line.get("points_xy", []), dtype=np.float32).reshape(-1, 2)
    if raw_points.shape[0] < 2 or not bool(np.isfinite(raw_points).all()):
        return [0.0] * 26
    network_points = np.asarray(transform_points(raw_points.tolist(), transform), dtype=np.float32)
    start = network_points[0]
    end = network_points[-1]
    if (float(start[0]), float(start[1])) > (float(end[0]), float(end[1])):
        start, end = end.copy(), start.copy()
    vector = end - start
    length = float(np.linalg.norm(vector))
    if length <= 1.0e-6 or not math.isfinite(length):
        cos_angle = 1.0
        sin_angle = 0.0
    else:
        cos_angle = float(vector[0] / length)
        sin_angle = float(vector[1] / length)
    center = (start + end) * 0.5
    normalized = np.asarray(
        [
            start[0] / max(float(network_w - 1), 1.0),
            start[1] / max(float(network_h - 1), 1.0),
            end[0] / max(float(network_w - 1), 1.0),
            end[1] / max(float(network_h - 1), 1.0),
            center[0] / max(float(network_w - 1), 1.0),
            center[1] / max(float(network_h - 1), 1.0),
        ],
        dtype=np.float32,
    )
    scalar_values = [
        float(candidate_count),
        float(rank) / max(float(candidate_count - 1), 1.0),
        float(_line_score(line)),
        float(line.get("score", 0.0)) if isinstance(line.get("score"), (int, float)) else 0.0,
        float(line.get("center_score", 0.0)) if isinstance(line.get("center_score"), (int, float)) else 0.0,
        float(line.get("instance_score", 0.0)) if isinstance(line.get("instance_score"), (int, float)) else 0.0,
        float(line.get("orientation_score", 0.0)) if isinstance(line.get("orientation_score"), (int, float)) else 0.0,
        float(line.get("fragment_count", 0.0)) if isinstance(line.get("fragment_count"), (int, float)) else 0.0,
        float(line.get("length", length)) / max(network_diag, 1.0)
        if isinstance(line.get("length", length), (int, float))
        else float(length / max(network_diag, 1.0)),
        float(length / max(network_diag, 1.0)),
        cos_angle,
        sin_angle,
        float(abs(sin_angle)),
        float(abs(cos_angle)),
        float(np.clip(normalized[0], 0.0, 1.0)),
        float(np.clip(normalized[1], 0.0, 1.0)),
        float(np.clip(normalized[2], 0.0, 1.0)),
        float(np.clip(normalized[3], 0.0, 1.0)),
        float(np.clip(normalized[4], 0.0, 1.0)),
        float(np.clip(normalized[5], 0.0, 1.0)),
        float(abs(normalized[2] - normalized[0])),
        float(abs(normalized[3] - normalized[1])),
        float(raw_points[:, 0].std() / max(float(network_w), 1.0)),
        float(raw_points[:, 1].std() / max(float(network_h), 1.0)),
        float(raw_points.shape[0]),
        1.0,
    ]
    return [0.0 if not math.isfinite(float(value)) else float(value) for value in scalar_values]


def _as_prob_map(predictions: dict[str, Any], key: str, sample_index: int) -> np.ndarray | None:
    value = predictions.get(key)
    if not isinstance(value, torch.Tensor):
        return None
    if value.ndim < 3 or int(sample_index) >= int(value.shape[0]):
        return None
    sample = value[int(sample_index)].detach().float().cpu().sigmoid()
    while sample.ndim > 2 and int(sample.shape[0]) == 1:
        sample = sample.squeeze(0)
    if sample.ndim != 2:
        return None
    array = sample.numpy().astype(np.float32)
    if not bool(np.isfinite(array).all()):
        return None
    return array


def _resize_or_expand_map(array: np.ndarray | None, *, output_hw: tuple[int, int]) -> np.ndarray:
    output_h, output_w = int(output_hw[0]), int(output_hw[1])
    if array is None or array.size == 0:
        return np.zeros((output_h, output_w), dtype=np.float32)
    source_h, source_w = int(array.shape[0]), int(array.shape[1])
    if source_h == output_h and source_w == output_w:
        return array.astype(np.float32)
    if source_w == 1 and source_h == output_h:
        return np.repeat(array.astype(np.float32), output_w, axis=1)
    if source_h == 1 and source_w == output_w:
        return np.repeat(array.astype(np.float32), output_h, axis=0)
    tensor = torch.as_tensor(array, dtype=torch.float32).reshape(1, 1, source_h, source_w)
    resized = F.interpolate(tensor, size=(output_h, output_w), mode="bilinear", align_corners=False)
    return resized[0, 0].detach().cpu().numpy().astype(np.float32)


def _bilinear_values(map_array: np.ndarray, xy: np.ndarray) -> np.ndarray:
    if map_array.size == 0 or xy.size == 0:
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


def _line_points_in_map(
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
    weights = np.linspace(0.0, 1.0, num=max(3, int(sample_count)), dtype=np.float32)
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


def _line_dense_features(
    line: dict[str, Any],
    *,
    predictions: dict[str, Any],
    sample_index: int,
    meta: dict[str, Any],
) -> list[float]:
    maps = [_as_prob_map(predictions, key, sample_index) for key in FEATURE_MAP_KEYS]
    reference = next((array for array in maps if isinstance(array, np.ndarray)), None)
    if reference is None:
        return [0.0] * (len(FEATURE_MAP_KEYS) * 7)
    output_hw = (int(reference.shape[0]), int(reference.shape[1]))
    line_points = _line_points_in_map(
        line,
        meta,
        output_hw=output_hw,
        sample_count=LINE_SAMPLE_COUNT,
    )
    if line_points is None:
        return [0.0] * (len(FEATURE_MAP_KEYS) * 7)
    points, normal = line_points
    side_delta = normal[None, :] * float(LINE_SIDE_OFFSET_PX)
    side_points = np.concatenate([points + side_delta, points - side_delta], axis=0)
    center_index = points.shape[0] // 2
    edge_indices = np.unique(np.asarray([0, 1, points.shape[0] - 2, points.shape[0] - 1], dtype=np.int64))
    features: list[float] = []
    for map_array in maps:
        resized = _resize_or_expand_map(map_array, output_hw=output_hw)
        line_values = _bilinear_values(resized, points)
        side_values = _bilinear_values(resized, side_points)
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
    return [0.0 if not math.isfinite(float(value)) else float(value) for value in features]


def _line_feature_vector(
    line: dict[str, Any],
    *,
    predictions: dict[str, Any],
    sample_index: int,
    meta: dict[str, Any],
    rank: int,
    candidate_count: int,
) -> np.ndarray:
    values = [
        *_line_scalar_features(line, meta=meta, rank=rank, candidate_count=candidate_count),
        *_line_dense_features(line, predictions=predictions, sample_index=sample_index, meta=meta),
    ]
    array = np.asarray(values, dtype=np.float32).reshape(-1)
    return np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _assign_stopline_labels(
    lines: list[dict[str, Any]],
    gt_lines: list[dict[str, Any]],
    *,
    match_threshold: float = STOP_LINE_MATCH_THRESHOLD_PX,
) -> list[int]:
    labels = [0 for _ in lines]
    if not lines or not gt_lines:
        return labels
    candidates: list[tuple[float, int, int]] = []
    for line_index, line in enumerate(lines):
        for gt_index, gt_line in enumerate(gt_lines):
            distance = _stop_line_distance(line, gt_line)
            if math.isfinite(distance) and float(distance) <= float(match_threshold):
                candidates.append((float(distance), int(line_index), int(gt_index)))
    candidates.sort(key=lambda item: item[0])
    used_lines: set[int] = set()
    used_gt: set[int] = set()
    for _distance, line_index, gt_index in candidates:
        if line_index in used_lines or gt_index in used_gt:
            continue
        labels[line_index] = 1
        used_lines.add(line_index)
        used_gt.add(gt_index)
    return labels


def _collect_examples(
    *,
    loader: Any,
    evaluator: Any,
    postprocess_config: Any,
    args: argparse.Namespace,
    max_batches: int,
    training: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    examples: list[dict[str, Any]] = []
    predictions_all: list[dict[str, Any]] = []
    raw_batches: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    global_sample_index = 0
    with torch.no_grad():
        for batch_index, batch in enumerate(loader, start=1):
            if batch_index > int(max_batches):
                break
            if batch_index == 1 or batch_index % 20 == 0:
                split = "train" if training else "val"
                print(f"[stopline_retained_suppressor] collect {split} batch {batch_index}/{max_batches}", flush=True)
            raw_batch = raw_batch_for_metrics(batch)
            if raw_batch is None:
                raise ValueError("retained stop-line suppressor requires raw batches for metrics")
            encoded = evaluator.prepare_batch(batch)
            predictions = _forward_predictions(evaluator, encoded, lane_flip_variant=str(args.lane_flip_variant))
            meta = encoded["meta"]
            batch_predictions = postprocess_pv26_batch(predictions, meta, config=postprocess_config)
            batch_gt = _extract_gt_samples(raw_batch)
            predictions_all.extend(batch_predictions)
            raw_batches.append(raw_batch)
            for sample_batch_index, (sample_pred, sample_gt, sample_meta) in enumerate(
                zip(batch_predictions, batch_gt, meta)
            ):
                lines = [dict(line) for line in sample_pred.get("stop_lines", [])]
                labels = _assign_stopline_labels(lines, list(sample_gt.get("stop_lines", [])))
                for line_index, line in enumerate(lines):
                    feature = _line_feature_vector(
                        line,
                        predictions=predictions,
                        sample_index=int(sample_batch_index),
                        meta=sample_meta,
                        rank=int(line_index),
                        candidate_count=len(lines),
                    )
                    label = int(labels[line_index]) if line_index < len(labels) else 0
                    examples.append(
                        {
                            "features": feature,
                            "label": label,
                            "sample_index": int(global_sample_index),
                            "sample_batch_index": int(sample_batch_index),
                            "line_index": int(line_index),
                        }
                    )
                    rows.append(
                        {
                            "split": "train" if training else "val",
                            "batch_index": int(batch_index),
                            "sample_index": int(global_sample_index),
                            "sample_batch_index": int(sample_batch_index),
                            "line_index": int(line_index),
                            "label": int(label),
                            "candidate_count": int(len(lines)),
                            "gt_count": int(len(sample_gt.get("stop_lines", []))),
                            "score": float(_line_score(line)),
                        }
                    )
                if not lines:
                    rows.append(
                        {
                            "split": "train" if training else "val",
                            "batch_index": int(batch_index),
                            "sample_index": int(global_sample_index),
                            "sample_batch_index": int(sample_batch_index),
                            "line_index": -1,
                            "label": 0,
                            "candidate_count": 0,
                            "gt_count": int(len(sample_gt.get("stop_lines", []))),
                            "score": 0.0,
                        }
                    )
                global_sample_index += 1
    return examples, predictions_all, raw_batches, rows


def _train_suppressor(
    examples: list[dict[str, Any]],
    *,
    args: argparse.Namespace,
    device: str,
) -> tuple[StoplineRetainedSuppressor, dict[str, Any]]:
    if not examples:
        raise ValueError("retained stop-line suppressor requires non-empty candidate examples")
    features = torch.tensor(np.stack([row["features"] for row in examples]), dtype=torch.float32)
    labels = torch.tensor([float(row["label"]) for row in examples], dtype=torch.float32)
    positive_count = int(labels.sum().item())
    negative_count = int(labels.numel() - positive_count)
    mean = features.mean(dim=0)
    std = features.std(dim=0).clamp(min=1.0e-6)
    features = (features - mean) / std
    model = StoplineRetainedSuppressor(int(features.shape[1]), int(args.hidden_dim)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.suppressor_lr),
        weight_decay=float(args.suppressor_weight_decay),
    )
    features = features.to(device)
    labels = labels.to(device)
    pos_weight = max(float(negative_count) / max(float(positive_count), 1.0), 1.0)
    pos_weight_tensor = torch.tensor([pos_weight], dtype=torch.float32, device=device)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(args.seed))
    batch_size = max(1, int(args.suppressor_batch_size))
    history: list[dict[str, float]] = []
    for epoch in range(1, max(1, int(args.suppressor_epochs)) + 1):
        order = torch.randperm(int(features.shape[0]), generator=generator)
        losses: list[float] = []
        for start in range(0, int(order.numel()), batch_size):
            index = order[start : start + batch_size].to(device)
            logits = model(features[index])
            loss = F.binary_cross_entropy_with_logits(
                logits,
                labels[index],
                pos_weight=pos_weight_tensor,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        if epoch == 1 or epoch == int(args.suppressor_epochs) or epoch % 10 == 0:
            history.append({"epoch": float(epoch), "loss": float(np.mean(losses) if losses else 0.0)})
    model.register_buffer("feature_mean", mean.to(device), persistent=True)
    model.register_buffer("feature_std", std.to(device), persistent=True)
    summary = {
        "candidate_count": int(labels.numel()),
        "positive_count": int(positive_count),
        "negative_count": int(negative_count),
        "input_dim": int(features.shape[1]),
        "pos_weight": float(pos_weight),
        "keep_threshold": float(args.keep_threshold),
        "history": history,
    }
    return model, summary


def _apply_suppressor(
    *,
    examples: list[dict[str, Any]],
    baseline_predictions: list[dict[str, Any]],
    model: StoplineRetainedSuppressor,
    args: argparse.Namespace,
    device: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    suppressed = [dict(sample, stop_lines=[]) for sample in baseline_predictions]
    if not baseline_predictions:
        return suppressed, []
    grouped_examples: dict[tuple[int, int], dict[str, Any]] = {}
    if examples:
        features = torch.tensor(np.stack([row["features"] for row in examples]), dtype=torch.float32, device=device)
        features = (features - model.feature_mean) / model.feature_std
        with torch.no_grad():
            probabilities = model(features).sigmoid().detach().cpu().numpy().astype(np.float32)
        for row, probability in zip(examples, probabilities):
            key = (int(row["sample_index"]), int(row["line_index"]))
            grouped_examples[key] = {**row, "keep_probability": float(probability)}
    rows: list[dict[str, Any]] = []
    threshold = float(args.keep_threshold)
    for sample_index, sample in enumerate(baseline_predictions):
        kept: list[dict[str, Any]] = []
        for line_index, line in enumerate(sample.get("stop_lines", [])):
            row = grouped_examples.get((int(sample_index), int(line_index)))
            probability = float(row["keep_probability"]) if row is not None else 0.0
            keep = bool(probability >= threshold)
            if keep:
                kept_line = dict(line)
                kept_line["retained_suppressor_keep_probability"] = probability
                kept.append(kept_line)
            rows.append(
                {
                    "sample_index": int(sample_index),
                    "line_index": int(line_index),
                    "keep_probability": probability,
                    "keep": int(keep),
                    "label": int(row["label"]) if row is not None else 0,
                    "score": float(_line_score(line)),
                }
            )
        suppressed[sample_index]["stop_lines"] = kept
    return suppressed, rows


def _decision_audit(rows: list[dict[str, Any]]) -> dict[str, int]:
    kept_positive = kept_negative = dropped_positive = dropped_negative = 0
    for row in rows:
        keep = bool(int(row.get("keep", 0)))
        positive = bool(int(row.get("label", 0)))
        if keep and positive:
            kept_positive += 1
        elif keep and not positive:
            kept_negative += 1
        elif not keep and positive:
            dropped_positive += 1
        else:
            dropped_negative += 1
    return {
        "kept_positive": int(kept_positive),
        "kept_negative": int(kept_negative),
        "dropped_positive": int(dropped_positive),
        "dropped_negative": int(dropped_negative),
    }


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    source_run = Path(args.source_run).expanduser().resolve()
    scenario, scenario_path, options, phase, train_config = _build_scenario(args)
    phase_index = int(tuple(options["selected_phase_indices"])[0])
    train_cli._configure_torch_multiprocessing()
    dataset_roots = train_cli._existing_dataset_roots(scenario)
    dataset = train_cli.PV26CanonicalDataset(
        dataset_roots,
        train_augmentation=False,
        progress_callback=lambda message: print(f"[stopline_retained_suppressor] {message}", flush=True),
    )
    train_loader, val_loader = train_cli._build_phase_train_loaders(dataset, train_config=train_config, phase=phase)
    if train_loader is None or val_loader is None:
        raise ValueError("retained stop-line suppressor requires train and validation loaders")
    _advance_validation_sampler(val_loader, validation_epoch=int(args.validation_epoch))
    trainer = train_cli._build_phase_trainer(phase, train_config)
    trainer.load_model_weights(checkpoint, map_location=train_config.device)
    postprocess_config = _postprocess_override_config(args, trainer) or train_cli._build_postprocess_config(train_config)
    evaluator = trainer.build_evaluator()
    evaluator.adapter.raw_model.eval()
    evaluator.heads.eval()
    device = str(train_config.device)

    train_examples, _, _, train_rows = _collect_examples(
        loader=train_loader,
        evaluator=evaluator,
        postprocess_config=postprocess_config,
        args=args,
        max_batches=int(args.suppressor_train_batches),
        training=True,
    )
    model, train_summary = _train_suppressor(train_examples, args=args, device=device)
    val_examples, baseline_predictions, raw_batches, val_rows = _collect_examples(
        loader=val_loader,
        evaluator=evaluator,
        postprocess_config=postprocess_config,
        args=args,
        max_batches=int(args.max_val_batches),
        training=False,
    )
    suppressed_predictions, decision_rows = _apply_suppressor(
        examples=val_examples,
        baseline_predictions=baseline_predictions,
        model=model,
        args=args,
        device=device,
    )
    merged_raw = _merge_raw_batches(raw_batches)
    baseline_metrics = augment_lane_family_metrics(summarize_pv26_metrics(baseline_predictions, merged_raw))
    suppressed_metrics = augment_lane_family_metrics(summarize_pv26_metrics(suppressed_predictions, merged_raw))
    baseline_tasks = {task: _metric_payload(baseline_metrics, task) for task in ("lane", "stop_line", "crosswalk")}
    suppressed_tasks = {task: _metric_payload(suppressed_metrics, task) for task in ("lane", "stop_line", "crosswalk")}
    summary = {
        "checkpoint": str(checkpoint),
        "source_run": str(source_run),
        "scenario_path": str(scenario_path),
        "lane60_experiment": str(args.lane60_experiment),
        "phase_index": int(phase_index),
        "dataset_roots": [str(path) for path in dataset_roots],
        "dataset_record_count": int(len(dataset)),
        "suppressor_train_batches": int(args.suppressor_train_batches),
        "max_val_batches": int(args.max_val_batches),
        "validation_epoch": int(args.validation_epoch),
        "keep_threshold": float(args.keep_threshold),
        "train_summary": train_summary,
        "validation_candidate_count": int(len(decision_rows)),
        "kept_prediction_count": int(sum(int(row.get("keep", 0)) for row in decision_rows)),
        "decision_audit": _decision_audit(decision_rows),
        "baseline": baseline_tasks,
        "suppressed": suppressed_tasks,
        "delta": {task: _task_delta(suppressed_tasks[task], baseline_tasks[task]) for task in baseline_tasks},
        "interpretation": (
            "No-GT learned suppress-only verifier over retained runtime stop-line candidates. "
            "The verifier is trained on train-split metric labels, then validation uses only "
            "line geometry and frozen dense-map support to drop candidates; it never emits new stop-lines."
        ),
    }
    return {
        "summary": summary,
        "train_rows": train_rows,
        "val_rows": val_rows,
        "decision_rows": decision_rows,
    }


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = run_probe(args)
    summary = payload["summary"]
    (output_dir / "summary.json").write_text(json.dumps(_json_ready(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(output_dir / "train_candidates.csv", payload["train_rows"])
    _write_csv(output_dir / "val_candidates.csv", payload["val_rows"])
    _write_csv(output_dir / "suppressor_decisions.csv", payload["decision_rows"])
    print(json.dumps(_json_ready(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
