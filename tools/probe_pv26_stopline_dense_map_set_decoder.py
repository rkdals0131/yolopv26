from __future__ import annotations

import argparse
import itertools
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

from common.geometry import sample_stop_line_centerline
from model.data.transform import inverse_transform_points, transform_from_meta, transform_points
from model.engine._trainer_epochs import _merge_raw_batches
from model.engine.batch import augment_lane_family_metrics, raw_batch_for_metrics
from model.engine.metrics import STOP_LINE_POINT_COUNT, _extract_gt_samples, summarize_pv26_metrics
from model.engine.postprocess import (
    _decode_stopline_haf_consensus_segments,
    _dedupe_stop_line_predictions,
    postprocess_pv26_batch,
)
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


FEATURE_MAP_KEYS = (
    "stop_line_mask_logits",
    "stop_line_center_logits",
    "stop_line_selector_map_logits",
    "stop_line_axis_valid_logits",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a small no-GT stop-line set decoder on frozen dense stop-line maps. "
            "The decoder reads full dense maps, emits a small set of line segments, and "
            "is replayed on validation beside the projection-competition runtime baseline."
        )
    )
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--source-run", default=str(SOURCE_RUN))
    parser.add_argument("--lane60-experiment", default="stopline_projection_comp_runtime")
    parser.add_argument("--preset", default="default")
    parser.add_argument("--decoder-train-batches", type=int, default=64)
    parser.add_argument("--max-val-batches", type=int, default=4)
    parser.add_argument("--validation-epoch", type=int, default=2)
    parser.add_argument("--train-batches", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dataset-root", default="")
    parser.add_argument("--decoder-queries", type=int, default=2)
    parser.add_argument("--pool-height", type=int, default=12)
    parser.add_argument("--pool-width", type=int, default=20)
    parser.add_argument("--hidden-dim", type=int, default=384)
    parser.add_argument("--decoder-epochs", type=int, default=80)
    parser.add_argument("--decoder-batch-size", type=int, default=128)
    parser.add_argument("--decoder-lr", type=float, default=1.0e-3)
    parser.add_argument(
        "--metric-quality-objectness",
        action="store_true",
        help=(
            "Train matched query objectness from endpoint-distance quality instead of "
            "hard 1.0 positives. This keeps the dense-map decoder contract but makes "
            "the confidence target geometry-aware."
        ),
    )
    parser.add_argument(
        "--metric-quality-tau",
        type=float,
        default=0.06,
        help="Normalized endpoint-distance scale for metric-quality objectness.",
    )
    parser.add_argument("--object-threshold", type=float, default=0.55)
    parser.add_argument("--max-output-segments", type=int, default=2)
    parser.add_argument(
        "--candidate-verifier-enabled",
        action="store_true",
        help=(
            "Train a second-stage verifier on decoder-generated candidates, then "
            "add only verifier-kept decoder candidates to the preserved runtime baseline."
        ),
    )
    parser.add_argument(
        "--union-selector-enabled",
        action="store_true",
        help=(
            "Train a verifier over the union of retained projection-comp lines and "
            "dense-map decoder candidates, then emit only the selected fixed-size set. "
            "This lets the verifier suppress retained FP and add decoder TP in one contract."
        ),
    )
    parser.add_argument("--candidate-verifier-hidden-dim", type=int, default=128)
    parser.add_argument("--candidate-verifier-epochs", type=int, default=80)
    parser.add_argument("--candidate-verifier-batch-size", type=int, default=128)
    parser.add_argument("--candidate-verifier-lr", type=float, default=1.0e-3)
    parser.add_argument("--candidate-verifier-threshold", type=float, default=0.50)
    parser.add_argument(
        "--haf-candidate-verifier-enabled",
        action="store_true",
        help=(
            "Use decoded stop-line HAF consensus segments as the candidate generator, "
            "train the second-stage verifier on train split labels, and add only "
            "verified HAF candidates to the preserved projection-comp baseline."
        ),
    )
    parser.add_argument("--haf-candidate-valid-threshold", type=float, default=0.95)
    parser.add_argument("--haf-candidate-min-votes", type=int, default=4)
    parser.add_argument("--haf-candidate-cluster-endpoint-tolerance", type=float, default=3.0)
    parser.add_argument("--haf-candidate-max-endpoint-covariance", type=float, default=9.0)
    parser.add_argument("--haf-candidate-max-segments", type=int, default=8)
    parser.add_argument(
        "--baseline-slot-refiner-enabled",
        action="store_true",
        help=(
            "Train a baseline-aware slot refiner instead of the global set decoder. "
            "Slots are projection-comp baseline stop-lines plus dense top-support fallback "
            "anchors, so the candidate generator can replace retained geometry or add "
            "missing dense-supported lines without blindly appending every decoder query."
        ),
    )
    parser.add_argument("--slot-refiner-baseline-slots", type=int, default=2)
    parser.add_argument("--slot-refiner-fallback-slots", type=int, default=1)
    parser.add_argument("--slot-refiner-loose-positive-distance", type=float, default=120.0)
    parser.add_argument("--slot-refiner-fallback-length", type=float, default=0.62)
    parser.add_argument("--seed", type=int, default=26)
    parser.add_argument("--backbone-weights", default="")
    parser.add_argument("--lane-flip-variant", default="baseline")
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


class StoplineDenseMapSetDecoder(nn.Module):
    def __init__(self, input_dim: int, *, hidden_dim: int, queries: int) -> None:
        super().__init__()
        self.queries = int(queries)
        self.net = nn.Sequential(
            nn.Linear(int(input_dim), int(hidden_dim)),
            nn.SiLU(inplace=True),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.SiLU(inplace=True),
        )
        self.logits = nn.Linear(int(hidden_dim), self.queries)
        self.points = nn.Linear(int(hidden_dim), self.queries * 4)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.net(features)
        logits = self.logits(hidden)
        points = torch.sigmoid(self.points(hidden)).view(-1, self.queries, 2, 2)
        return logits, points


class StoplineDecoderCandidateVerifier(nn.Module):
    def __init__(self, input_dim: int, *, hidden_dim: int) -> None:
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


class StoplineBaselineSlotRefiner(nn.Module):
    def __init__(self, input_dim: int, *, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(input_dim), int(hidden_dim)),
            nn.SiLU(inplace=True),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.SiLU(inplace=True),
        )
        self.logit = nn.Linear(int(hidden_dim), 1)
        self.points = nn.Linear(int(hidden_dim), 4)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.net(features)
        logits = self.logit(hidden).squeeze(-1)
        points = torch.sigmoid(self.points(hidden)).view(-1, 2, 2)
        return logits, points


def _finite_array(values: np.ndarray | list[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32).reshape(-1)
    return np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _sigmoid_prediction_map(predictions: dict[str, Any], key: str, sample_index: int) -> torch.Tensor:
    value = predictions.get(key)
    if not isinstance(value, torch.Tensor):
        return torch.zeros((1, 1, 1), dtype=torch.float32)
    sample = value[sample_index].detach().float().cpu()
    if sample.ndim == 2:
        sample = sample.unsqueeze(0)
    if sample.ndim != 3:
        return torch.zeros((1, 1, 1), dtype=torch.float32)
    return sample.sigmoid()


def _expand_1d_map(value: torch.Tensor, *, height: int, width: int) -> torch.Tensor:
    if value.ndim != 3:
        return value
    if int(value.shape[-1]) == 1 and int(value.shape[-2]) == height:
        return value.expand(-1, height, width)
    if int(value.shape[-2]) == 1 and int(value.shape[-1]) == width:
        return value.expand(-1, height, width)
    return value


def _topk_features(array: torch.Tensor, *, top_k: int = 8) -> np.ndarray:
    flat = array.reshape(-1)
    if flat.numel() == 0:
        return np.zeros(top_k * 3, dtype=np.float32)
    count = min(int(top_k), int(flat.numel()))
    values, indices = torch.topk(flat, k=count)
    h, w = int(array.shape[-2]), int(array.shape[-1])
    rows = torch.div(indices, max(w, 1), rounding_mode="floor").float() / max(float(h - 1), 1.0)
    cols = (indices % max(w, 1)).float() / max(float(w - 1), 1.0)
    out = torch.zeros((int(top_k), 3), dtype=torch.float32)
    out[:count, 0] = values.float()
    out[:count, 1] = cols
    out[:count, 2] = rows
    return out.numpy().reshape(-1).astype(np.float32)


def _dense_feature_vector(
    predictions: dict[str, Any],
    *,
    sample_index: int,
    pool_hw: tuple[int, int],
) -> np.ndarray:
    maps = [_sigmoid_prediction_map(predictions, key, sample_index) for key in FEATURE_MAP_KEYS]
    height = max(int(item.shape[-2]) for item in maps)
    width = max(int(item.shape[-1]) for item in maps)
    row_map = _expand_1d_map(_sigmoid_prediction_map(predictions, "stop_line_row_logits", sample_index), height=height, width=width)
    x_map = _expand_1d_map(_sigmoid_prediction_map(predictions, "stop_line_x_logits", sample_index), height=height, width=width)
    maps.extend([row_map, x_map])
    pooled_parts: list[np.ndarray] = []
    for item in maps:
        if int(item.shape[-2]) != height or int(item.shape[-1]) != width:
            item = F.interpolate(item.unsqueeze(0), size=(height, width), mode="bilinear", align_corners=False).squeeze(0)
        avg = F.adaptive_avg_pool2d(item.unsqueeze(0), output_size=pool_hw).reshape(-1)
        max_values = F.adaptive_max_pool2d(item.unsqueeze(0), output_size=pool_hw).reshape(-1)
        pooled_parts.append(avg.numpy().astype(np.float32))
        pooled_parts.append(max_values.numpy().astype(np.float32))
    center = maps[1][0]
    selector = maps[2][0]
    pooled_parts.append(_topk_features(center))
    pooled_parts.append(_topk_features(selector))
    return _finite_array(np.concatenate(pooled_parts, axis=0))


def _canonical_segment(segment: np.ndarray) -> np.ndarray:
    points = np.asarray(segment, dtype=np.float32).reshape(2, 2).copy()
    if (float(points[0, 0]), float(points[0, 1])) > (float(points[1, 0]), float(points[1, 1])):
        points = points[[1, 0]]
    return np.clip(points, 0.0, 1.0).astype(np.float32)


def _gt_stopline_segments_norm(sample_gt: dict[str, Any], meta: dict[str, Any], *, max_count: int) -> np.ndarray:
    transform = transform_from_meta(meta)
    network_h, network_w = int(transform.network_hw[0]), int(transform.network_hw[1])
    segments: list[np.ndarray] = []
    for row in sample_gt.get("stop_lines", []):
        points = np.asarray(row.get("points_xy", []), dtype=np.float32).reshape(-1, 2)
        if points.shape[0] < 2:
            continue
        endpoints_raw = np.stack([points[0], points[-1]], axis=0)
        endpoints_network = np.asarray(transform_points(endpoints_raw.tolist(), transform), dtype=np.float32).reshape(2, 2)
        endpoints_network[:, 0] /= max(float(network_w - 1), 1.0)
        endpoints_network[:, 1] /= max(float(network_h - 1), 1.0)
        segments.append(_canonical_segment(endpoints_network))
    if not segments:
        return np.zeros((0, 2, 2), dtype=np.float32)
    segments.sort(key=lambda item: (float(item[:, 1].mean()), float(item[:, 0].mean())))
    return np.stack(segments[: max(1, int(max_count))]).astype(np.float32)


def _raw_stopline_segment_norm(line: dict[str, Any], meta: dict[str, Any]) -> np.ndarray | None:
    points = np.asarray(line.get("points_xy", []), dtype=np.float32).reshape(-1, 2)
    if points.shape[0] < 2:
        return None
    transform = transform_from_meta(meta)
    network_h, network_w = int(transform.network_hw[0]), int(transform.network_hw[1])
    endpoints_raw = np.stack([points[0], points[-1]], axis=0)
    endpoints_network = np.asarray(transform_points(endpoints_raw.tolist(), transform), dtype=np.float32).reshape(2, 2)
    endpoints_network[:, 0] /= max(float(network_w - 1), 1.0)
    endpoints_network[:, 1] /= max(float(network_h - 1), 1.0)
    return _canonical_segment(endpoints_network)


def _dense_anchor_segments_norm(
    predictions: dict[str, Any],
    *,
    sample_index: int,
    max_count: int,
    fallback_length: float,
) -> list[np.ndarray]:
    center = _sigmoid_prediction_map(predictions, "stop_line_center_logits", sample_index)
    selector = _sigmoid_prediction_map(predictions, "stop_line_selector_map_logits", sample_index)
    if center.ndim != 3 or selector.ndim != 3:
        return []
    if tuple(selector.shape[-2:]) != tuple(center.shape[-2:]):
        selector = F.interpolate(
            selector.unsqueeze(0),
            size=tuple(center.shape[-2:]),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
    support = (center[0] * torch.clamp(selector[0], min=0.0, max=1.0)).reshape(-1)
    if support.numel() == 0:
        return []
    count = min(max(0, int(max_count)), int(support.numel()))
    if count <= 0:
        return []
    values, indices = torch.topk(support, k=count)
    angle = predictions.get("stop_line_angle")
    angle_sample: torch.Tensor | None = None
    if isinstance(angle, torch.Tensor) and int(angle.ndim) == 4 and 0 <= int(sample_index) < int(angle.shape[0]):
        angle_sample = angle[sample_index].detach().float().cpu()
        if tuple(angle_sample.shape[-2:]) != tuple(center.shape[-2:]):
            angle_sample = F.interpolate(
                angle_sample.unsqueeze(0),
                size=tuple(center.shape[-2:]),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
    height, width = int(center.shape[-2]), int(center.shape[-1])
    anchors: list[np.ndarray] = []
    used: list[tuple[float, float]] = []
    for value, flat_index in zip(values.tolist(), indices.tolist()):
        if float(value) <= 0.0:
            continue
        row = int(flat_index) // max(width, 1)
        col = int(flat_index) % max(width, 1)
        center_xy = np.asarray(
            [
                float(col) / max(float(width - 1), 1.0),
                float(row) / max(float(height - 1), 1.0),
            ],
            dtype=np.float32,
        )
        if any(float(np.linalg.norm(center_xy - np.asarray(prev, dtype=np.float32))) < 0.04 for prev in used):
            continue
        used.append((float(center_xy[0]), float(center_xy[1])))
        axis = np.asarray([1.0, 0.0], dtype=np.float32)
        if angle_sample is not None and int(angle_sample.shape[0]) >= 2:
            raw_axis = angle_sample[:2, row, col].numpy().astype(np.float32)
            norm = float(np.linalg.norm(raw_axis))
            if norm > 1.0e-6 and math.isfinite(norm):
                axis = raw_axis / norm
        half_length = 0.5 * max(float(fallback_length), 1.0e-3)
        segment = np.stack([center_xy - axis * half_length, center_xy + axis * half_length], axis=0)
        anchors.append(_canonical_segment(segment))
        if len(anchors) >= int(max_count):
            break
    return anchors


def _slot_feature(sample_features: np.ndarray, *, anchor: np.ndarray, slot_kind: str, slot_rank: int) -> np.ndarray:
    kind_values = np.asarray(
        [
            1.0 if slot_kind == "baseline" else 0.0,
            1.0 if slot_kind == "fallback" else 0.0,
            1.0 / max(float(int(slot_rank) + 1), 1.0),
        ],
        dtype=np.float32,
    )
    return _finite_array(
        np.concatenate(
            [
                _decoder_candidate_feature(sample_features, segment=anchor, probability=1.0),
                kind_values,
            ],
            axis=0,
        )
    )


def _assign_baseline_slots_to_gt(
    baseline_lines: list[dict[str, Any]],
    gt_lines: list[dict[str, Any]],
    *,
    max_distance: float,
) -> dict[int, int]:
    pairs: list[tuple[float, int, int]] = []
    for baseline_index, line in enumerate(baseline_lines):
        for gt_index, gt_line in enumerate(gt_lines):
            distance = _stop_line_distance(line, gt_line)
            if math.isfinite(distance) and float(distance) <= float(max_distance):
                pairs.append((float(distance), int(baseline_index), int(gt_index)))
    pairs.sort(key=lambda item: item[0])
    assignments: dict[int, int] = {}
    used_gt: set[int] = set()
    for _distance, baseline_index, gt_index in pairs:
        if baseline_index in assignments or gt_index in used_gt:
            continue
        assignments[int(baseline_index)] = int(gt_index)
        used_gt.add(int(gt_index))
    return assignments


def _slot_examples_for_sample(
    *,
    sample_features: np.ndarray,
    baseline_lines: list[dict[str, Any]],
    gt_lines: list[dict[str, Any]],
    gt_segments: np.ndarray,
    baseline_anchors: list[np.ndarray],
    fallback_anchors: list[np.ndarray],
    loose_positive_distance: float,
    sample_meta: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    slots: list[dict[str, Any]] = []
    assignments = _assign_baseline_slots_to_gt(
        baseline_lines[: len(baseline_anchors)],
        gt_lines,
        max_distance=float(loose_positive_distance),
    )
    used_gt = set(assignments.values())
    for slot_index, anchor in enumerate(baseline_anchors):
        gt_index = assignments.get(int(slot_index))
        positive = gt_index is not None and 0 <= int(gt_index) < int(gt_segments.shape[0])
        target = gt_segments[int(gt_index)] if positive else anchor
        slots.append(
            {
                "features": _slot_feature(sample_features, anchor=anchor, slot_kind="baseline", slot_rank=slot_index),
                "anchor": _canonical_segment(anchor),
                "target": _canonical_segment(target),
                "positive": int(positive),
                "slot_kind": "baseline",
                "slot_rank": int(slot_index),
                "assigned_gt_index": -1 if gt_index is None else int(gt_index),
            }
        )
    remaining_gt = [gt_index for gt_index in range(int(gt_segments.shape[0])) if gt_index not in used_gt]
    for fallback_index, anchor in enumerate(fallback_anchors):
        gt_index: int | None = None
        if remaining_gt:
            pairs: list[tuple[float, int]] = []
            if sample_meta is not None and gt_lines:
                anchor_line = {"points_xy": _network_norm_segment_to_raw(anchor, sample_meta)}
                for candidate_gt_index in remaining_gt:
                    if 0 <= int(candidate_gt_index) < len(gt_lines):
                        distance = _stop_line_distance(anchor_line, gt_lines[int(candidate_gt_index)])
                        if math.isfinite(distance) and float(distance) <= float(loose_positive_distance):
                            pairs.append((float(distance), int(candidate_gt_index)))
            else:
                threshold = min(max(float(loose_positive_distance), 0.0), 1.0)
                for candidate_gt_index in remaining_gt:
                    target_segment = gt_segments[int(candidate_gt_index)]
                    direct = float(np.abs(_canonical_segment(anchor) - _canonical_segment(target_segment)).mean())
                    flipped = float(np.abs(_canonical_segment(anchor)[::-1] - _canonical_segment(target_segment)).mean())
                    distance = min(direct, flipped)
                    if math.isfinite(distance) and distance <= threshold:
                        pairs.append((distance, int(candidate_gt_index)))
            if pairs:
                pairs.sort(key=lambda item: item[0])
                gt_index = int(pairs[0][1])
                remaining_gt = [candidate for candidate in remaining_gt if int(candidate) != gt_index]
        positive = gt_index is not None and 0 <= int(gt_index) < int(gt_segments.shape[0])
        target = gt_segments[int(gt_index)] if positive else anchor
        slots.append(
            {
                "features": _slot_feature(sample_features, anchor=anchor, slot_kind="fallback", slot_rank=fallback_index),
                "anchor": _canonical_segment(anchor),
                "target": _canonical_segment(target),
                "positive": int(positive),
                "slot_kind": "fallback",
                "slot_rank": int(fallback_index),
                "assigned_gt_index": -1 if gt_index is None else int(gt_index),
            }
        )
    return slots


def _segment_cost(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    direct = torch.abs(pred - target).mean()
    flipped = torch.abs(pred.flip(dims=(0,)) - target).mean()
    return torch.minimum(direct, flipped)


def _best_assignment(cost: torch.Tensor, target_count: int) -> list[tuple[int, int]]:
    query_count = int(cost.shape[0])
    target_count = int(min(target_count, int(cost.shape[1]), query_count))
    if target_count <= 0:
        return []
    best_pairs: list[tuple[int, int]] = []
    best_cost = float("inf")
    for query_indices in itertools.permutations(range(query_count), target_count):
        total = 0.0
        for target_index, query_index in enumerate(query_indices):
            total += float(cost[query_index, target_index].detach().cpu())
        if total < best_cost:
            best_cost = total
            best_pairs = [(int(query_index), int(target_index)) for target_index, query_index in enumerate(query_indices)]
    return best_pairs


def _set_decoder_loss(
    logits: torch.Tensor,
    points: torch.Tensor,
    targets: list[torch.Tensor],
    *,
    pos_weight: float,
    metric_quality_objectness: bool = False,
    metric_quality_tau: float = 0.06,
) -> torch.Tensor:
    batch_losses: list[torch.Tensor] = []
    pos_weight_tensor = torch.tensor([max(float(pos_weight), 1.0)], dtype=logits.dtype, device=logits.device)
    for batch_index, target in enumerate(targets):
        query_logits = logits[batch_index]
        query_points = points[batch_index]
        object_target = torch.zeros_like(query_logits)
        point_loss = query_points.sum() * 0.0
        if target.numel() > 0:
            target = target.to(device=points.device, dtype=points.dtype)
            cost = torch.stack(
                [
                    torch.stack([_segment_cost(query_points[q], target[t]) for t in range(int(target.shape[0]))])
                    for q in range(int(query_points.shape[0]))
                ]
            )
            pairs = _best_assignment(cost, int(target.shape[0]))
            if pairs:
                query_indices = torch.tensor([query for query, _ in pairs], dtype=torch.long, device=points.device)
                target_indices = torch.tensor([target_index for _, target_index in pairs], dtype=torch.long, device=points.device)
                direct = F.smooth_l1_loss(query_points[query_indices], target[target_indices], reduction="none").mean(dim=(1, 2))
                flipped = F.smooth_l1_loss(
                    query_points[query_indices].flip(dims=(1,)),
                    target[target_indices],
                    reduction="none",
                ).mean(dim=(1, 2))
                matched_point_loss = torch.minimum(direct, flipped)
                point_loss = matched_point_loss.mean()
                if bool(metric_quality_objectness):
                    quality_tau = max(float(metric_quality_tau), 1.0e-6)
                    object_target[query_indices] = torch.exp(
                        -matched_point_loss.detach() / quality_tau
                    ).clamp(min=0.0, max=1.0)
                else:
                    object_target[query_indices] = 1.0
        object_loss = F.binary_cross_entropy_with_logits(query_logits, object_target, pos_weight=pos_weight_tensor)
        batch_losses.append(object_loss + 8.0 * point_loss)
    return torch.stack(batch_losses).mean()


def _train_decoder(
    examples: list[dict[str, Any]],
    *,
    args: argparse.Namespace,
    device: str,
) -> tuple[StoplineDenseMapSetDecoder, dict[str, Any]]:
    if not examples:
        raise ValueError("dense-map set decoder requires non-empty train examples")
    features = torch.tensor(np.stack([row["features"] for row in examples]), dtype=torch.float32)
    targets = [torch.tensor(row["targets"], dtype=torch.float32) for row in examples]
    target_count = sum(int(row.shape[0]) for row in targets)
    sample_positive_count = sum(1 for row in targets if int(row.shape[0]) > 0)
    mean = features.mean(dim=0)
    std = features.std(dim=0).clamp(min=1.0e-6)
    features = (features - mean) / std
    model = StoplineDenseMapSetDecoder(
        int(features.shape[1]),
        hidden_dim=int(args.hidden_dim),
        queries=int(args.decoder_queries),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.decoder_lr), weight_decay=1.0e-4)
    features = features.to(device)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(args.seed))
    batch_size = max(1, int(args.decoder_batch_size))
    query_slots = max(int(args.decoder_queries) * len(examples), 1)
    pos_weight = max(float(query_slots - target_count) / max(float(target_count), 1.0), 1.0)
    history: list[dict[str, float]] = []
    for epoch in range(1, max(1, int(args.decoder_epochs)) + 1):
        order = torch.randperm(int(features.shape[0]), generator=generator)
        losses: list[float] = []
        for start in range(0, int(order.numel()), batch_size):
            index = order[start : start + batch_size]
            batch_targets = [targets[int(i)] for i in index.tolist()]
            batch_features = features[index.to(device)]
            logits, points = model(batch_features)
            loss = _set_decoder_loss(
                logits,
                points,
                batch_targets,
                pos_weight=pos_weight,
                metric_quality_objectness=bool(args.metric_quality_objectness),
                metric_quality_tau=float(args.metric_quality_tau),
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        if epoch == 1 or epoch == int(args.decoder_epochs) or epoch % 10 == 0:
            history.append({"epoch": float(epoch), "loss": float(np.mean(losses) if losses else 0.0)})
    model.register_buffer("feature_mean", mean.to(device), persistent=True)
    model.register_buffer("feature_std", std.to(device), persistent=True)
    summary = {
        "example_count": int(len(examples)),
        "sample_positive_count": int(sample_positive_count),
        "target_segment_count": int(target_count),
        "input_dim": int(features.shape[1]),
        "pos_weight": float(pos_weight),
        "metric_quality_objectness": bool(args.metric_quality_objectness),
        "metric_quality_tau": float(args.metric_quality_tau),
        "history": history,
    }
    return model, summary


def _slot_refiner_loss(
    logits: torch.Tensor,
    points: torch.Tensor,
    targets: torch.Tensor,
    anchors: torch.Tensor,
    positive: torch.Tensor,
    *,
    pos_weight: float,
) -> torch.Tensor:
    pos_weight_tensor = torch.tensor([max(float(pos_weight), 1.0)], dtype=logits.dtype, device=logits.device)
    object_loss = F.binary_cross_entropy_with_logits(logits, positive, pos_weight=pos_weight_tensor)
    if bool((positive > 0.5).any()):
        pos_mask = positive > 0.5
        direct = F.smooth_l1_loss(points[pos_mask], targets[pos_mask], reduction="none").mean(dim=(1, 2))
        flipped = F.smooth_l1_loss(points[pos_mask].flip(dims=(1,)), targets[pos_mask], reduction="none").mean(dim=(1, 2))
        positive_point_loss = torch.minimum(direct, flipped).mean()
    else:
        positive_point_loss = points.sum() * 0.0
    if bool((positive <= 0.5).any()):
        neg_mask = positive <= 0.5
        negative_identity_loss = F.smooth_l1_loss(points[neg_mask], anchors[neg_mask], reduction="none").mean()
    else:
        negative_identity_loss = points.sum() * 0.0
    return object_loss + 8.0 * positive_point_loss + 0.5 * negative_identity_loss


def _train_slot_refiner(
    slot_examples: list[dict[str, Any]],
    *,
    args: argparse.Namespace,
    device: str,
) -> tuple[StoplineBaselineSlotRefiner, dict[str, Any]]:
    if not slot_examples:
        raise ValueError("baseline-slot refiner requires non-empty slot examples")
    features = torch.tensor(np.stack([row["features"] for row in slot_examples]), dtype=torch.float32)
    targets = torch.tensor(np.stack([row["target"] for row in slot_examples]), dtype=torch.float32)
    anchors = torch.tensor(np.stack([row["anchor"] for row in slot_examples]), dtype=torch.float32)
    positive = torch.tensor([float(row["positive"]) for row in slot_examples], dtype=torch.float32)
    positive_count = int(positive.sum().item())
    negative_count = int(positive.numel() - positive_count)
    mean = features.mean(dim=0)
    std = features.std(dim=0).clamp(min=1.0e-6)
    features = (features - mean) / std
    model = StoplineBaselineSlotRefiner(
        int(features.shape[1]),
        hidden_dim=int(args.hidden_dim),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.decoder_lr), weight_decay=1.0e-4)
    features = features.to(device)
    targets = targets.to(device)
    anchors = anchors.to(device)
    positive = positive.to(device)
    pos_weight = max(float(negative_count) / max(float(positive_count), 1.0), 1.0)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(args.seed) + 1703)
    batch_size = max(1, int(args.decoder_batch_size))
    history: list[dict[str, float]] = []
    for epoch in range(1, max(1, int(args.decoder_epochs)) + 1):
        order = torch.randperm(int(features.shape[0]), generator=generator)
        losses: list[float] = []
        for start in range(0, int(order.numel()), batch_size):
            index = order[start : start + batch_size].to(device)
            logits, points = model(features[index])
            loss = _slot_refiner_loss(
                logits,
                points,
                targets[index],
                anchors[index],
                positive[index],
                pos_weight=pos_weight,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        if epoch == 1 or epoch == int(args.decoder_epochs) or epoch % 10 == 0:
            history.append({"epoch": float(epoch), "loss": float(np.mean(losses) if losses else 0.0)})
    model.register_buffer("feature_mean", mean.to(device), persistent=True)
    model.register_buffer("feature_std", std.to(device), persistent=True)
    summary = {
        "slot_count": int(len(slot_examples)),
        "positive_count": int(positive_count),
        "negative_count": int(negative_count),
        "input_dim": int(features.shape[1]),
        "pos_weight": float(pos_weight),
        "baseline_slots": int(args.slot_refiner_baseline_slots),
        "fallback_slots": int(args.slot_refiner_fallback_slots),
        "loose_positive_distance": float(args.slot_refiner_loose_positive_distance),
        "fallback_length": float(args.slot_refiner_fallback_length),
        "history": history,
    }
    return model, summary


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
    pool_hw = (int(args.pool_height), int(args.pool_width))
    with torch.no_grad():
        for batch_index, batch in enumerate(loader, start=1):
            if batch_index > int(max_batches):
                break
            if batch_index == 1 or batch_index % 20 == 0:
                split = "train" if training else "val"
                print(f"[stopline_dense_map_set_decoder] collect {split} batch {batch_index}/{max_batches}", flush=True)
            raw_batch = raw_batch_for_metrics(batch)
            if raw_batch is None:
                raise ValueError("dense-map set decoder requires raw batches for metrics")
            encoded = evaluator.prepare_batch(batch)
            predictions = _forward_predictions(evaluator, encoded, lane_flip_variant=str(args.lane_flip_variant))
            meta = encoded["meta"]
            batch_predictions = postprocess_pv26_batch(predictions, meta, config=postprocess_config)
            batch_gt = _extract_gt_samples(raw_batch)
            predictions_all.extend(batch_predictions)
            raw_batches.append(raw_batch)
            for sample_batch_index, (sample_gt, sample_meta) in enumerate(zip(batch_gt, meta)):
                features = _dense_feature_vector(predictions, sample_index=sample_batch_index, pool_hw=pool_hw)
                target_max_count = max(
                    int(args.decoder_queries),
                    int(args.slot_refiner_baseline_slots) + int(args.slot_refiner_fallback_slots),
                )
                targets = _gt_stopline_segments_norm(sample_gt, sample_meta, max_count=target_max_count)
                baseline_lines = [
                    dict(item)
                    for item in batch_predictions[sample_batch_index].get("stop_lines", [])
                    if isinstance(item, dict)
                ][: max(0, int(args.slot_refiner_baseline_slots))]
                baseline_anchors = [
                    segment
                    for segment in (_raw_stopline_segment_norm(line, sample_meta) for line in baseline_lines)
                    if segment is not None
                ]
                fallback_anchors = _dense_anchor_segments_norm(
                    predictions,
                    sample_index=sample_batch_index,
                    max_count=int(args.slot_refiner_fallback_slots),
                    fallback_length=float(args.slot_refiner_fallback_length),
                )
                haf_candidates = (
                    _decode_haf_candidates_for_sample(
                        predictions,
                        sample_index=sample_batch_index,
                        meta=sample_meta,
                        args=args,
                    )
                    if bool(args.haf_candidate_verifier_enabled)
                    else []
                )
                slot_examples = _slot_examples_for_sample(
                    sample_features=features,
                    baseline_lines=baseline_lines,
                    gt_lines=[dict(item) for item in sample_gt.get("stop_lines", [])],
                    gt_segments=targets,
                    baseline_anchors=baseline_anchors,
                    fallback_anchors=fallback_anchors,
                    loose_positive_distance=float(args.slot_refiner_loose_positive_distance),
                    sample_meta=sample_meta,
                )
                examples.append(
                    {
                        "features": features,
                        "targets": targets,
                        "baseline_lines": baseline_lines,
                        "baseline_anchors": baseline_anchors,
                        "fallback_anchors": fallback_anchors,
                        "haf_candidates": haf_candidates,
                        "slot_examples": slot_examples,
                        "gt_stop_lines": [dict(item) for item in sample_gt.get("stop_lines", [])],
                        "sample_index": int(global_sample_index),
                        "sample_batch_index": int(sample_batch_index),
                        "meta": sample_meta,
                    }
                )
                rows.append(
                    {
                        "split": "train" if training else "val",
                        "batch_index": int(batch_index),
                        "sample_index": int(global_sample_index),
                        "sample_batch_index": int(sample_batch_index),
                        "target_count": int(targets.shape[0]),
                        "baseline_slot_count": int(len(baseline_anchors)),
                        "fallback_slot_count": int(len(fallback_anchors)),
                        "haf_candidate_count": int(len(haf_candidates)),
                        "slot_count": int(len(slot_examples)),
                        "positive_slot_count": int(sum(int(row["positive"]) for row in slot_examples)),
                    }
                )
                global_sample_index += 1
    return examples, predictions_all, raw_batches, rows


def _network_norm_segment_to_raw(segment: np.ndarray, meta: dict[str, Any]) -> list[list[float]]:
    transform = transform_from_meta(meta)
    points = np.asarray(segment, dtype=np.float32).reshape(2, 2).copy()
    points[:, 0] *= max(float(transform.network_hw[1] - 1), 1.0)
    points[:, 1] *= max(float(transform.network_hw[0] - 1), 1.0)
    network_points = sample_stop_line_centerline(points.tolist(), target_count=STOP_LINE_POINT_COUNT).tolist()
    raw_points = sample_stop_line_centerline(
        inverse_transform_points(network_points, transform),
        target_count=STOP_LINE_POINT_COUNT,
    ).tolist()
    return [[float(x), float(y)] for x, y in raw_points]


def _decoder_candidate_feature(
    sample_features: np.ndarray,
    *,
    segment: np.ndarray,
    probability: float,
) -> np.ndarray:
    points = _canonical_segment(segment).reshape(2, 2)
    vector = points[1] - points[0]
    length = float(np.linalg.norm(vector))
    if length <= 1.0e-6 or not math.isfinite(length):
        cos_angle = 1.0
        sin_angle = 0.0
    else:
        cos_angle = float(vector[0] / length)
        sin_angle = float(vector[1] / length)
    center = points.mean(axis=0)
    candidate_values = np.asarray(
        [
            float(probability),
            float(length),
            cos_angle,
            sin_angle,
            float(center[0]),
            float(center[1]),
            float(abs(vector[0])),
            float(abs(vector[1])),
            float(points[0, 0]),
            float(points[0, 1]),
            float(points[1, 0]),
            float(points[1, 1]),
        ],
        dtype=np.float32,
    )
    return _finite_array(np.concatenate([np.asarray(sample_features, dtype=np.float32).reshape(-1), candidate_values]))


def _haf_candidate_feature(
    sample_features: np.ndarray,
    *,
    segment: np.ndarray,
    line: dict[str, Any],
) -> np.ndarray:
    score = float(line.get("score", 0.0))
    base = _decoder_candidate_feature(sample_features, segment=segment, probability=score)
    vote_count = float(line.get("haf_vote_count", 0.0))
    covariance = float(line.get("haf_endpoint_covariance", 0.0))
    values = np.asarray(
        [
            float(line.get("center_score", score)),
            float(line.get("orientation_score", 0.0)),
            float(line.get("length", 0.0)),
            vote_count,
            math.log1p(max(vote_count, 0.0)),
            covariance,
            1.0 / (1.0 + max(covariance, 0.0)),
        ],
        dtype=np.float32,
    )
    return _finite_array(np.concatenate([base, values], axis=0))


def _source_flags(source: str) -> np.ndarray:
    normalized = str(source).strip().lower()
    return np.asarray(
        [
            1.0 if normalized == "retained_projection_comp" else 0.0,
            1.0 if normalized == "dense_map_set_decoder" else 0.0,
            1.0 if normalized == "haf_consensus_candidate" else 0.0,
        ],
        dtype=np.float32,
    )


def _union_candidate_feature(
    sample_features: np.ndarray,
    *,
    segment: np.ndarray,
    probability: float,
    source: str,
    rank: int,
    candidate_count: int,
) -> np.ndarray:
    base = _decoder_candidate_feature(sample_features, segment=segment, probability=probability)
    rank_features = np.asarray(
        [
            float(rank) / max(float(candidate_count - 1), 1.0),
            float(candidate_count),
        ],
        dtype=np.float32,
    )
    return _finite_array(np.concatenate([base, _source_flags(source), rank_features], axis=0))


def _decode_haf_candidates_for_sample(
    predictions: dict[str, Any],
    *,
    sample_index: int,
    meta: dict[str, Any],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    haf_endpoint = predictions.get("stop_line_haf_endpoint")
    haf_valid_logits = predictions.get("stop_line_haf_valid_logits")
    if not isinstance(haf_endpoint, torch.Tensor) or not isinstance(haf_valid_logits, torch.Tensor):
        return []
    if not (0 <= int(sample_index) < int(haf_endpoint.shape[0])) or not (
        0 <= int(sample_index) < int(haf_valid_logits.shape[0])
    ):
        return []
    lines = _decode_stopline_haf_consensus_segments(
        haf_endpoint=haf_endpoint[int(sample_index)],
        haf_valid_logits=haf_valid_logits[int(sample_index)],
        meta=meta,
        valid_threshold=float(args.haf_candidate_valid_threshold),
        min_votes=int(args.haf_candidate_min_votes),
        cluster_endpoint_tolerance=float(args.haf_candidate_cluster_endpoint_tolerance),
        max_endpoint_covariance=float(args.haf_candidate_max_endpoint_covariance),
        max_segments=int(args.haf_candidate_max_segments),
    )
    output: list[dict[str, Any]] = []
    for rank, line in enumerate(lines):
        payload = dict(line)
        payload["proposal_source"] = "haf_consensus_candidate"
        payload["haf_candidate_rank"] = int(rank)
        output.append(payload)
    return output


def _decoder_candidate_rows(
    *,
    examples: list[dict[str, Any]],
    model: StoplineDenseMapSetDecoder,
    device: str,
) -> list[list[dict[str, Any]]]:
    if not examples:
        return []
    features = torch.tensor(np.stack([row["features"] for row in examples]), dtype=torch.float32, device=device)
    features = (features - model.feature_mean) / model.feature_std
    with torch.no_grad():
        logits, points = model(features)
        probabilities = logits.sigmoid().detach().cpu().numpy().astype(np.float32)
        segments = points.detach().cpu().numpy().astype(np.float32)
    grouped: list[list[dict[str, Any]]] = []
    for sample_index, example in enumerate(examples):
        rows: list[dict[str, Any]] = []
        for query_index in range(int(probabilities.shape[1])):
            probability = float(probabilities[sample_index, query_index])
            segment = _canonical_segment(segments[sample_index, query_index])
            rows.append(
                {
                    "sample_index": int(example["sample_index"]),
                    "query_index": int(query_index),
                    "probability": probability,
                    "segment": segment,
                    "features": _decoder_candidate_feature(
                        example["features"],
                        segment=segment,
                        probability=probability,
                    ),
                    "line": {
                        "points_xy": _network_norm_segment_to_raw(segment, example["meta"]),
                        "score": probability,
                        "dense_map_set_decoder_score": probability,
                        "dense_map_set_decoder_query_index": int(query_index),
                        "proposal_source": "dense_map_set_decoder",
                        "allowed": True,
                    },
                }
            )
        grouped.append(rows)
    return grouped


def _haf_candidate_rows(*, examples: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    grouped: list[list[dict[str, Any]]] = []
    for example in examples:
        rows: list[dict[str, Any]] = []
        for query_index, line in enumerate(example.get("haf_candidates", [])):
            segment = _raw_stopline_segment_norm(line, example["meta"])
            if segment is None:
                continue
            probability = float(line.get("score", 0.0))
            payload = dict(line)
            payload["allowed"] = True
            payload["proposal_source"] = "haf_consensus_candidate"
            rows.append(
                {
                    "sample_index": int(example["sample_index"]),
                    "query_index": int(query_index),
                    "probability": probability,
                    "segment": segment,
                    "features": _haf_candidate_feature(
                        example["features"],
                        segment=segment,
                        line=payload,
                    ),
                    "line": payload,
                }
            )
        grouped.append(rows)
    return grouped


def _baseline_candidate_rows(*, examples: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    grouped: list[list[dict[str, Any]]] = []
    for example in examples:
        baseline_lines = [dict(item) for item in example.get("baseline_lines", []) if isinstance(item, dict)]
        rows: list[dict[str, Any]] = []
        candidate_count = len(baseline_lines)
        for line_index, line in enumerate(baseline_lines):
            segment = _raw_stopline_segment_norm(line, example["meta"])
            if segment is None:
                continue
            probability = float(
                max(
                    [
                        float(value)
                        for value in (
                            line.get("score", 0.0),
                            line.get("center_score", 0.0),
                            line.get("projection_comp_score", 0.0),
                            line.get("instance_score", 0.0),
                        )
                        if isinstance(value, (int, float)) and math.isfinite(float(value))
                    ],
                    default=0.0,
                )
            )
            payload = dict(line)
            payload["allowed"] = True
            payload["proposal_source"] = "retained_projection_comp"
            payload["retained_projection_comp_index"] = int(line_index)
            rows.append(
                {
                    "sample_index": int(example["sample_index"]),
                    "query_index": int(line_index),
                    "probability": probability,
                    "segment": segment,
                    "features": _union_candidate_feature(
                        example["features"],
                        segment=segment,
                        probability=probability,
                        source="retained_projection_comp",
                        rank=int(line_index),
                        candidate_count=candidate_count,
                    ),
                    "line": payload,
                }
            )
        grouped.append(rows)
    return grouped


def _union_candidate_rows(
    *,
    examples: list[dict[str, Any]],
    decoder_candidates: list[list[dict[str, Any]]],
) -> list[list[dict[str, Any]]]:
    baseline_rows = _baseline_candidate_rows(examples=examples)
    grouped: list[list[dict[str, Any]]] = []
    for sample_index, example in enumerate(examples):
        rows: list[dict[str, Any]] = []
        source_candidates = [
            *baseline_rows[sample_index],
            *(decoder_candidates[sample_index] if sample_index < len(decoder_candidates) else []),
        ]
        candidate_count = len(source_candidates)
        for union_index, candidate in enumerate(source_candidates):
            line = dict(candidate.get("line", {}))
            source = str(line.get("proposal_source", ""))
            segment = np.asarray(candidate.get("segment"), dtype=np.float32).reshape(2, 2)
            probability = float(candidate.get("probability", line.get("score", 0.0)))
            line["union_candidate_index"] = int(union_index)
            line["proposal_source"] = source
            rows.append(
                {
                    **candidate,
                    "query_index": int(union_index),
                    "source_query_index": int(candidate.get("query_index", -1)),
                    "probability": probability,
                    "features": _union_candidate_feature(
                        example["features"],
                        segment=segment,
                        probability=probability,
                        source=source,
                        rank=int(union_index),
                        candidate_count=candidate_count,
                    ),
                    "line": line,
                }
            )
        grouped.append(rows)
    return grouped


def _assign_candidate_labels(
    candidates: list[dict[str, Any]],
    gt_lines: list[dict[str, Any]],
    *,
    match_threshold: float = 40.0,
) -> list[int]:
    labels = [0 for _ in candidates]
    if not candidates or not gt_lines:
        return labels
    pairs: list[tuple[float, int, int]] = []
    for candidate_index, candidate in enumerate(candidates):
        line = candidate.get("line", {})
        for gt_index, gt_line in enumerate(gt_lines):
            distance = _stop_line_distance(line, gt_line)
            if math.isfinite(distance) and float(distance) <= float(match_threshold):
                pairs.append((float(distance), int(candidate_index), int(gt_index)))
    pairs.sort(key=lambda item: item[0])
    used_candidates: set[int] = set()
    used_gt: set[int] = set()
    for _distance, candidate_index, gt_index in pairs:
        if candidate_index in used_candidates or gt_index in used_gt:
            continue
        labels[candidate_index] = 1
        used_candidates.add(candidate_index)
        used_gt.add(gt_index)
    return labels


def _train_candidate_verifier(
    examples: list[dict[str, Any]],
    grouped_candidates: list[list[dict[str, Any]]],
    *,
    args: argparse.Namespace,
    device: str,
) -> tuple[StoplineDecoderCandidateVerifier, dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    feature_rows: list[np.ndarray] = []
    labels: list[float] = []
    for example, candidates in zip(examples, grouped_candidates):
        candidate_labels = _assign_candidate_labels(candidates, list(example.get("gt_stop_lines", [])))
        for candidate, label in zip(candidates, candidate_labels):
            feature_rows.append(np.asarray(candidate["features"], dtype=np.float32).reshape(-1))
            labels.append(float(label))
            rows.append(
                {
                    "split": "train",
                    "sample_index": int(candidate["sample_index"]),
                    "query_index": int(candidate["query_index"]),
                    "source_query_index": int(candidate.get("source_query_index", candidate["query_index"])),
                    "proposal_source": str(candidate.get("line", {}).get("proposal_source", "")),
                    "probability": float(candidate["probability"]),
                    "label": int(label),
                    "target_count": int(np.asarray(example.get("targets", [])).shape[0]),
                }
            )
    if not feature_rows:
        raise ValueError("candidate verifier requires non-empty decoder candidates")
    features = torch.tensor(np.stack(feature_rows), dtype=torch.float32)
    target = torch.tensor(labels, dtype=torch.float32)
    positive_count = int(target.sum().item())
    negative_count = int(target.numel() - positive_count)
    mean = features.mean(dim=0)
    std = features.std(dim=0).clamp(min=1.0e-6)
    features = (features - mean) / std
    model = StoplineDecoderCandidateVerifier(
        int(features.shape[1]),
        hidden_dim=int(args.candidate_verifier_hidden_dim),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.candidate_verifier_lr), weight_decay=1.0e-4)
    pos_weight = max(float(negative_count) / max(float(positive_count), 1.0), 1.0)
    pos_weight_tensor = torch.tensor([pos_weight], dtype=torch.float32, device=device)
    features = features.to(device)
    target = target.to(device)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(args.seed) + 911)
    batch_size = max(1, int(args.candidate_verifier_batch_size))
    history: list[dict[str, float]] = []
    for epoch in range(1, max(1, int(args.candidate_verifier_epochs)) + 1):
        order = torch.randperm(int(features.shape[0]), generator=generator)
        losses: list[float] = []
        for start in range(0, int(order.numel()), batch_size):
            index = order[start : start + batch_size].to(device)
            logits = model(features[index])
            loss = F.binary_cross_entropy_with_logits(logits, target[index], pos_weight=pos_weight_tensor)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        if epoch == 1 or epoch == int(args.candidate_verifier_epochs) or epoch % 10 == 0:
            history.append({"epoch": float(epoch), "loss": float(np.mean(losses) if losses else 0.0)})
    model.register_buffer("feature_mean", mean.to(device), persistent=True)
    model.register_buffer("feature_std", std.to(device), persistent=True)
    summary = {
        "candidate_count": int(target.numel()),
        "positive_count": int(positive_count),
        "negative_count": int(negative_count),
        "input_dim": int(features.shape[1]),
        "pos_weight": float(pos_weight),
        "threshold": float(args.candidate_verifier_threshold),
        "history": history,
    }
    return model, summary, rows


def _apply_candidate_verifier(
    *,
    examples: list[dict[str, Any]],
    baseline_predictions: list[dict[str, Any]],
    grouped_candidates: list[list[dict[str, Any]]],
    verifier: StoplineDecoderCandidateVerifier,
    args: argparse.Namespace,
    device: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not examples:
        return baseline_predictions, []
    feature_rows: list[np.ndarray] = []
    candidate_refs: list[tuple[int, dict[str, Any], int]] = []
    for sample_index, candidates in enumerate(grouped_candidates):
        labels = _assign_candidate_labels(candidates, list(examples[sample_index].get("gt_stop_lines", [])))
        for candidate, label in zip(candidates, labels):
            feature_rows.append(np.asarray(candidate["features"], dtype=np.float32).reshape(-1))
            candidate_refs.append((int(sample_index), candidate, int(label)))
    if not feature_rows:
        return baseline_predictions, []
    features = torch.tensor(np.stack(feature_rows), dtype=torch.float32, device=device)
    features = (features - verifier.feature_mean) / verifier.feature_std
    with torch.no_grad():
        keep_probabilities = verifier(features).sigmoid().detach().cpu().numpy().astype(np.float32)
    selected_by_sample: dict[int, list[dict[str, Any]]] = {}
    rows: list[dict[str, Any]] = []
    threshold = float(args.candidate_verifier_threshold)
    for probability, (sample_index, candidate, label) in zip(keep_probabilities, candidate_refs):
        selected = bool(float(probability) >= threshold)
        rows.append(
            {
                "sample_index": int(sample_index),
                "query_index": int(candidate["query_index"]),
                "source_query_index": int(candidate.get("source_query_index", candidate["query_index"])),
                "proposal_source": str(candidate.get("line", {}).get("proposal_source", "")),
                "decoder_probability": float(candidate["probability"]),
                "verifier_probability": float(probability),
                "selected": int(selected),
                "label": int(label),
            }
        )
        if selected:
            selected_by_sample.setdefault(int(sample_index), []).append(dict(candidate["line"]))
    merged_predictions = [
        dict(sample, stop_lines=[dict(line) for line in sample.get("stop_lines", [])])
        for sample in baseline_predictions
    ]
    max_segments = max(0, int(args.max_output_segments))
    for sample_index, selected_lines in selected_by_sample.items():
        if not (0 <= sample_index < len(merged_predictions)):
            continue
        merged = [dict(line) for line in merged_predictions[sample_index].get("stop_lines", [])] + selected_lines
        merged.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
        merged_predictions[sample_index]["stop_lines"] = _dedupe_stop_line_predictions(merged)[:max_segments]
    return merged_predictions, rows


def _apply_union_selector(
    *,
    examples: list[dict[str, Any]],
    baseline_predictions: list[dict[str, Any]],
    grouped_candidates: list[list[dict[str, Any]]],
    verifier: StoplineDecoderCandidateVerifier,
    args: argparse.Namespace,
    device: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected_predictions = [dict(sample, stop_lines=[]) for sample in baseline_predictions]
    feature_rows: list[np.ndarray] = []
    candidate_refs: list[tuple[int, dict[str, Any], int]] = []
    for sample_index, candidates in enumerate(grouped_candidates):
        labels = _assign_candidate_labels(candidates, list(examples[sample_index].get("gt_stop_lines", [])))
        for candidate, label in zip(candidates, labels):
            feature_rows.append(np.asarray(candidate["features"], dtype=np.float32).reshape(-1))
            candidate_refs.append((int(sample_index), candidate, int(label)))
    if not feature_rows:
        return selected_predictions, []

    features = torch.tensor(np.stack(feature_rows), dtype=torch.float32, device=device)
    features = (features - verifier.feature_mean) / verifier.feature_std
    with torch.no_grad():
        probabilities = verifier(features).sigmoid().detach().cpu().numpy().astype(np.float32)

    threshold = float(args.candidate_verifier_threshold)
    max_segments = max(0, int(args.max_output_segments))
    ranked_by_sample: dict[int, list[tuple[float, dict[str, Any], int]]] = {}
    rows: list[dict[str, Any]] = []
    for probability, (sample_index, candidate, label) in zip(probabilities, candidate_refs):
        selected = bool(float(probability) >= threshold)
        source = str(candidate.get("line", {}).get("proposal_source", ""))
        rows.append(
            {
                "sample_index": int(sample_index),
                "query_index": int(candidate["query_index"]),
                "source_query_index": int(candidate.get("source_query_index", candidate["query_index"])),
                "proposal_source": source,
                "decoder_probability": float(candidate["probability"]),
                "verifier_probability": float(probability),
                "selected": int(selected),
                "label": int(label),
            }
        )
        if selected:
            ranked_by_sample.setdefault(int(sample_index), []).append((float(probability), dict(candidate["line"]), int(label)))

    for sample_index, ranked in ranked_by_sample.items():
        if not (0 <= sample_index < len(selected_predictions)):
            continue
        ranked.sort(
            key=lambda item: (
                float(item[0]),
                float(item[1].get("score", 0.0)),
            ),
            reverse=True,
        )
        selected_predictions[sample_index]["stop_lines"] = _dedupe_stop_line_predictions(
            [dict(line) for _probability, line, _label in ranked[:max_segments]]
        )[:max_segments]
    return selected_predictions, rows


def _source_decision_audit(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    audit: dict[str, dict[str, int]] = {}
    for row in rows:
        source = str(row.get("proposal_source", ""))
        payload = audit.setdefault(
            source,
            {
                "selected_positive": 0,
                "selected_negative": 0,
                "rejected_positive": 0,
                "rejected_negative": 0,
            },
        )
        selected = bool(int(row.get("selected", 0)))
        positive = bool(int(row.get("label", 0)))
        if selected and positive:
            payload["selected_positive"] += 1
        elif selected and not positive:
            payload["selected_negative"] += 1
        elif not selected and positive:
            payload["rejected_positive"] += 1
        else:
            payload["rejected_negative"] += 1
    return audit


def _apply_grouped_candidate_lines(
    *,
    examples: list[dict[str, Any]],
    baseline_predictions: list[dict[str, Any]],
    grouped_candidates: list[list[dict[str, Any]]],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    candidate_only = [dict(sample, stop_lines=[]) for sample in baseline_predictions]
    baseline_plus = [
        dict(sample, stop_lines=[dict(item) for item in sample.get("stop_lines", [])])
        for sample in baseline_predictions
    ]
    max_segments = max(0, int(args.max_output_segments))
    rows: list[dict[str, Any]] = []
    for sample_index, candidates in enumerate(grouped_candidates):
        if not (0 <= int(sample_index) < len(examples)):
            continue
        ranked = sorted(
            [(candidate_index, candidate) for candidate_index, candidate in enumerate(candidates)],
            key=lambda item: float(item[1].get("line", {}).get("score", item[1].get("probability", 0.0))),
            reverse=True,
        )
        selected_indices = {int(candidate_index) for candidate_index, _candidate in ranked[:max_segments]}
        sample_lines = [dict(candidate["line"]) for _candidate_index, candidate in ranked[:max_segments]]
        for query_index, candidate in enumerate(candidates):
            rows.append(
                {
                    "sample_index": int(sample_index),
                    "query_index": int(candidate["query_index"]),
                    "probability": float(candidate["probability"]),
                    "selected": int(query_index in selected_indices),
                    "target_count": int(np.asarray(examples[sample_index].get("targets", [])).shape[0]),
                    "proposal_source": str(candidate.get("line", {}).get("proposal_source", "")),
                }
            )
        if 0 <= sample_index < len(candidate_only):
            candidate_only[sample_index]["stop_lines"] = [dict(item) for item in sample_lines]
        if 0 <= sample_index < len(baseline_plus):
            merged = [dict(item) for item in baseline_plus[sample_index].get("stop_lines", [])] + [
                dict(item) for item in sample_lines
            ]
            merged.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
            baseline_plus[sample_index]["stop_lines"] = _dedupe_stop_line_predictions(merged)[:max_segments]
    return candidate_only, baseline_plus, rows


def _apply_decoder(
    *,
    examples: list[dict[str, Any]],
    baseline_predictions: list[dict[str, Any]],
    model: StoplineDenseMapSetDecoder,
    args: argparse.Namespace,
    device: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if not examples:
        return [], baseline_predictions, []
    features = torch.tensor(np.stack([row["features"] for row in examples]), dtype=torch.float32, device=device)
    features = (features - model.feature_mean) / model.feature_std
    with torch.no_grad():
        logits, points = model(features)
        probabilities = logits.sigmoid().detach().cpu().numpy().astype(np.float32)
        segments = points.detach().cpu().numpy().astype(np.float32)
    decoder_only = [dict(sample, stop_lines=[]) for sample in baseline_predictions]
    baseline_plus = [dict(sample, stop_lines=[dict(item) for item in sample.get("stop_lines", [])]) for sample in baseline_predictions]
    rows: list[dict[str, Any]] = []
    threshold = float(args.object_threshold)
    max_segments = max(0, int(args.max_output_segments))
    for sample_index, example in enumerate(examples):
        sample_decoded: list[dict[str, Any]] = []
        for query_index in range(int(probabilities.shape[1])):
            probability = float(probabilities[sample_index, query_index])
            selected = bool(probability >= threshold)
            raw_points = _network_norm_segment_to_raw(segments[sample_index, query_index], example["meta"])
            rows.append(
                {
                    "sample_index": int(example["sample_index"]),
                    "query_index": int(query_index),
                    "probability": probability,
                    "selected": int(selected),
                    "target_count": int(np.asarray(example["targets"]).shape[0]),
                }
            )
            if not selected:
                continue
            sample_decoded.append(
                {
                    "points_xy": raw_points,
                    "score": probability,
                    "dense_map_set_decoder_score": probability,
                    "allowed": True,
                }
            )
        sample_decoded.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
        sample_decoded = sample_decoded[:max_segments]
        if 0 <= sample_index < len(decoder_only):
            decoder_only[sample_index]["stop_lines"] = [dict(item) for item in sample_decoded]
        if 0 <= sample_index < len(baseline_plus):
            merged = [dict(item) for item in baseline_plus[sample_index].get("stop_lines", [])] + [
                dict(item) for item in sample_decoded
            ]
            merged.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
            baseline_plus[sample_index]["stop_lines"] = _dedupe_stop_line_predictions(merged)[:max_segments]
    return decoder_only, baseline_plus, rows


def _flat_slot_examples(examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    slots: list[dict[str, Any]] = []
    for sample in examples:
        sample_index = int(sample.get("sample_index", len(slots)))
        for slot in sample.get("slot_examples", []):
            payload = dict(slot)
            payload["sample_index"] = sample_index
            slots.append(payload)
    return slots


def _apply_slot_refiner(
    *,
    examples: list[dict[str, Any]],
    baseline_predictions: list[dict[str, Any]],
    model: StoplineBaselineSlotRefiner,
    args: argparse.Namespace,
    device: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    slot_examples = _flat_slot_examples(examples)
    if not slot_examples:
        return [dict(sample, stop_lines=[]) for sample in baseline_predictions], baseline_predictions, []
    features = torch.tensor(np.stack([row["features"] for row in slot_examples]), dtype=torch.float32, device=device)
    features = (features - model.feature_mean) / model.feature_std
    with torch.no_grad():
        logits, points = model(features)
        probabilities = logits.sigmoid().detach().cpu().numpy().astype(np.float32)
        segments = points.detach().cpu().numpy().astype(np.float32)
    threshold = float(args.object_threshold)
    max_segments = max(0, int(args.max_output_segments))
    selected_by_sample: dict[int, list[dict[str, Any]]] = {}
    rows: list[dict[str, Any]] = []
    for row_index, (slot, probability) in enumerate(zip(slot_examples, probabilities.tolist())):
        selected = bool(float(probability) >= threshold)
        sample_index = int(slot.get("sample_index", -1))
        segment = _canonical_segment(segments[row_index])
        raw_points = _network_norm_segment_to_raw(segment, examples[sample_index]["meta"]) if 0 <= sample_index < len(examples) else []
        rows.append(
            {
                "sample_index": sample_index,
                "slot_kind": str(slot.get("slot_kind", "")),
                "slot_rank": int(slot.get("slot_rank", -1)),
                "probability": float(probability),
                "selected": int(selected),
                "train_label": int(slot.get("positive", 0)),
                "assigned_gt_index": int(slot.get("assigned_gt_index", -1)),
            }
        )
        if not selected or not raw_points:
            continue
        selected_by_sample.setdefault(sample_index, []).append(
            {
                "points_xy": raw_points,
                "score": float(probability),
                "baseline_slot_refiner_score": float(probability),
                "baseline_slot_kind": str(slot.get("slot_kind", "")),
                "baseline_slot_rank": int(slot.get("slot_rank", -1)),
                "proposal_source": "baseline_slot_refiner",
                "allowed": True,
            }
        )
    refiner_only = [dict(sample, stop_lines=[]) for sample in baseline_predictions]
    baseline_plus = [
        dict(sample, stop_lines=[dict(item) for item in sample.get("stop_lines", [])])
        for sample in baseline_predictions
    ]
    for sample_index, lines in selected_by_sample.items():
        if not (0 <= sample_index < len(baseline_plus)):
            continue
        lines = sorted(lines, key=lambda item: float(item.get("score", 0.0)), reverse=True)[:max_segments]
        refiner_only[sample_index]["stop_lines"] = [dict(item) for item in lines]
        merged = [dict(item) for item in baseline_plus[sample_index].get("stop_lines", [])] + [
            dict(item) for item in lines
        ]
        merged.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
        baseline_plus[sample_index]["stop_lines"] = _dedupe_stop_line_predictions(merged)[:max_segments]
    return refiner_only, baseline_plus, rows


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    source_run = Path(args.source_run).expanduser().resolve()
    scenario, scenario_path, options, phase, train_config = _build_scenario(args)
    phase_index = int(tuple(options["selected_phase_indices"])[0])
    train_cli._configure_torch_multiprocessing()
    dataset = train_cli.PV26CanonicalDataset(
        train_cli._existing_dataset_roots(scenario),
        train_augmentation=False,
        progress_callback=lambda message: print(f"[stopline_dense_map_set_decoder] {message}", flush=True),
    )
    train_loader, val_loader = train_cli._build_phase_train_loaders(dataset, train_config=train_config, phase=phase)
    if train_loader is None or val_loader is None:
        raise ValueError("dense-map set decoder requires train and validation loaders")
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
        max_batches=int(args.decoder_train_batches),
        training=True,
    )
    candidate_generator = "dense_map_set_decoder"
    slot_refiner_model: StoplineBaselineSlotRefiner | None = None
    slot_refiner_summary: dict[str, Any] | None = None
    model: StoplineDenseMapSetDecoder | None = None
    train_summary: dict[str, Any]
    if bool(args.haf_candidate_verifier_enabled):
        if (
            bool(args.baseline_slot_refiner_enabled)
            or bool(args.candidate_verifier_enabled)
            or bool(args.union_selector_enabled)
        ):
            raise ValueError(
                "haf-candidate-verifier-enabled is a standalone candidate-generator mode; "
                "do not combine it with baseline-slot-refiner-enabled, candidate-verifier-enabled, "
                "or union-selector-enabled"
            )
        candidate_generator = "haf_consensus_candidate"
        train_haf_candidates = _haf_candidate_rows(examples=train_examples)
        train_candidate_count = sum(len(row) for row in train_haf_candidates)
        if train_candidate_count <= 0:
            raise ValueError("HAF candidate verifier requires non-empty train HAF candidates")
        train_summary = {
            "candidate_generator": candidate_generator,
            "train_haf_candidate_count": int(train_candidate_count),
            "train_haf_candidate_samples": int(sum(1 for row in train_haf_candidates if row)),
        }
    elif bool(args.baseline_slot_refiner_enabled):
        if bool(args.union_selector_enabled):
            raise ValueError("union-selector-enabled currently uses dense-map decoder candidates, not slot refiner candidates")
        slot_refiner_model, slot_refiner_summary = _train_slot_refiner(
            _flat_slot_examples(train_examples),
            args=args,
            device=device,
        )
        train_summary = {"baseline_slot_refiner": slot_refiner_summary}
    else:
        model, train_summary = _train_decoder(train_examples, args=args, device=device)
    verifier_model: StoplineDecoderCandidateVerifier | None = None
    verifier_summary: dict[str, Any] | None = None
    verifier_train_rows: list[dict[str, Any]] = []
    if bool(args.haf_candidate_verifier_enabled):
        train_haf_candidates = _haf_candidate_rows(examples=train_examples)
        verifier_model, verifier_summary, verifier_train_rows = _train_candidate_verifier(
            train_examples,
            train_haf_candidates,
            args=args,
            device=device,
        )
        train_summary["candidate_verifier"] = verifier_summary
    elif bool(args.union_selector_enabled) and model is not None:
        train_decoder_candidates = _decoder_candidate_rows(
            examples=train_examples,
            model=model,
            device=device,
        )
        train_union_candidates = _union_candidate_rows(
            examples=train_examples,
            decoder_candidates=train_decoder_candidates,
        )
        verifier_model, verifier_summary, verifier_train_rows = _train_candidate_verifier(
            train_examples,
            train_union_candidates,
            args=args,
            device=device,
        )
        candidate_generator = "retained_projection_comp_plus_dense_map_set_decoder"
    elif bool(args.candidate_verifier_enabled) and model is not None:
        train_decoder_candidates = _decoder_candidate_rows(
            examples=train_examples,
            model=model,
            device=device,
        )
        verifier_model, verifier_summary, verifier_train_rows = _train_candidate_verifier(
            train_examples,
            train_decoder_candidates,
            args=args,
            device=device,
        )
    val_examples, baseline_predictions, raw_batches, val_rows = _collect_examples(
        loader=val_loader,
        evaluator=evaluator,
        postprocess_config=postprocess_config,
        args=args,
        max_batches=int(args.max_val_batches),
        training=False,
    )
    val_haf_candidates: list[list[dict[str, Any]]] | None = None
    if bool(args.haf_candidate_verifier_enabled):
        val_haf_candidates = _haf_candidate_rows(examples=val_examples)
        decoder_only_predictions, baseline_plus_predictions, decision_rows = _apply_grouped_candidate_lines(
            examples=val_examples,
            baseline_predictions=baseline_predictions,
            grouped_candidates=val_haf_candidates,
            args=args,
        )
    elif slot_refiner_model is not None:
        decoder_only_predictions, baseline_plus_predictions, decision_rows = _apply_slot_refiner(
            examples=val_examples,
            baseline_predictions=baseline_predictions,
            model=slot_refiner_model,
            args=args,
            device=device,
        )
    else:
        assert model is not None
        decoder_only_predictions, baseline_plus_predictions, decision_rows = _apply_decoder(
            examples=val_examples,
            baseline_predictions=baseline_predictions,
            model=model,
            args=args,
            device=device,
        )
    verified_plus_predictions: list[dict[str, Any]] | None = None
    union_selector_predictions: list[dict[str, Any]] | None = None
    verifier_decision_rows: list[dict[str, Any]] = []
    if bool(args.haf_candidate_verifier_enabled) and verifier_model is not None:
        assert val_haf_candidates is not None
        verified_plus_predictions, verifier_decision_rows = _apply_candidate_verifier(
            examples=val_examples,
            baseline_predictions=baseline_predictions,
            grouped_candidates=val_haf_candidates,
            verifier=verifier_model,
            args=args,
            device=device,
        )
    elif bool(args.union_selector_enabled) and verifier_model is not None:
        val_decoder_candidates = _decoder_candidate_rows(
            examples=val_examples,
            model=model,
            device=device,
        )
        val_union_candidates = _union_candidate_rows(
            examples=val_examples,
            decoder_candidates=val_decoder_candidates,
        )
        union_selector_predictions, verifier_decision_rows = _apply_union_selector(
            examples=val_examples,
            baseline_predictions=baseline_predictions,
            grouped_candidates=val_union_candidates,
            verifier=verifier_model,
            args=args,
            device=device,
        )
    elif verifier_model is not None:
        val_decoder_candidates = _decoder_candidate_rows(
            examples=val_examples,
            model=model,
            device=device,
        )
        verified_plus_predictions, verifier_decision_rows = _apply_candidate_verifier(
            examples=val_examples,
            baseline_predictions=baseline_predictions,
            grouped_candidates=val_decoder_candidates,
            verifier=verifier_model,
            args=args,
            device=device,
        )
    merged_raw = _merge_raw_batches(raw_batches)
    baseline_metrics = augment_lane_family_metrics(summarize_pv26_metrics(baseline_predictions, merged_raw))
    decoder_only_metrics = augment_lane_family_metrics(summarize_pv26_metrics(decoder_only_predictions, merged_raw))
    baseline_plus_metrics = augment_lane_family_metrics(summarize_pv26_metrics(baseline_plus_predictions, merged_raw))
    verified_plus_metrics = (
        augment_lane_family_metrics(summarize_pv26_metrics(verified_plus_predictions, merged_raw))
        if verified_plus_predictions is not None
        else None
    )
    union_selector_metrics = (
        augment_lane_family_metrics(summarize_pv26_metrics(union_selector_predictions, merged_raw))
        if union_selector_predictions is not None
        else None
    )
    baseline_tasks = {task: _metric_payload(baseline_metrics, task) for task in ("lane", "stop_line", "crosswalk")}
    decoder_only_tasks = {task: _metric_payload(decoder_only_metrics, task) for task in ("lane", "stop_line", "crosswalk")}
    baseline_plus_tasks = {task: _metric_payload(baseline_plus_metrics, task) for task in ("lane", "stop_line", "crosswalk")}
    verified_plus_tasks = (
        {task: _metric_payload(verified_plus_metrics, task) for task in ("lane", "stop_line", "crosswalk")}
        if verified_plus_metrics is not None
        else None
    )
    union_selector_tasks = (
        {task: _metric_payload(union_selector_metrics, task) for task in ("lane", "stop_line", "crosswalk")}
        if union_selector_metrics is not None
        else None
    )
    summary = {
        "checkpoint": str(checkpoint),
        "source_run": str(source_run),
        "scenario_path": str(scenario_path),
        "lane60_experiment": str(args.lane60_experiment),
        "phase_index": int(phase_index),
        "decoder_train_batches": int(args.decoder_train_batches),
        "max_val_batches": int(args.max_val_batches),
        "validation_epoch": int(args.validation_epoch),
        "decoder_queries": int(args.decoder_queries),
        "object_threshold": float(args.object_threshold),
        "candidate_generator": candidate_generator,
        "metric_quality_objectness": bool(args.metric_quality_objectness),
        "metric_quality_tau": float(args.metric_quality_tau),
        "candidate_verifier_enabled": bool(args.candidate_verifier_enabled),
        "union_selector_enabled": bool(args.union_selector_enabled),
        "haf_candidate_verifier_enabled": bool(args.haf_candidate_verifier_enabled),
        "haf_candidate_valid_threshold": float(args.haf_candidate_valid_threshold),
        "haf_candidate_min_votes": int(args.haf_candidate_min_votes),
        "haf_candidate_cluster_endpoint_tolerance": float(args.haf_candidate_cluster_endpoint_tolerance),
        "haf_candidate_max_endpoint_covariance": float(args.haf_candidate_max_endpoint_covariance),
        "haf_candidate_max_segments": int(args.haf_candidate_max_segments),
        "candidate_verifier_threshold": float(args.candidate_verifier_threshold),
        "baseline_slot_refiner_enabled": bool(args.baseline_slot_refiner_enabled),
        "slot_refiner_baseline_slots": int(args.slot_refiner_baseline_slots),
        "slot_refiner_fallback_slots": int(args.slot_refiner_fallback_slots),
        "slot_refiner_loose_positive_distance": float(args.slot_refiner_loose_positive_distance),
        "max_output_segments": int(args.max_output_segments),
        "pool_hw": [int(args.pool_height), int(args.pool_width)],
        "train_summary": train_summary,
        "slot_refiner_summary": slot_refiner_summary,
        "candidate_verifier_summary": verifier_summary,
        "selected_prediction_count": int(sum(int(row["selected"]) for row in decision_rows)),
        "verified_selected_prediction_count": int(sum(int(row["selected"]) for row in verifier_decision_rows)),
        "source_decision_audit": _source_decision_audit(verifier_decision_rows),
        "baseline": baseline_tasks,
        "decoder_only": decoder_only_tasks,
        "baseline_plus_decoder": baseline_plus_tasks,
        "slot_refiner_only": decoder_only_tasks if bool(args.baseline_slot_refiner_enabled) else None,
        "baseline_plus_slot_refiner": baseline_plus_tasks if bool(args.baseline_slot_refiner_enabled) else None,
        "haf_only": decoder_only_tasks if bool(args.haf_candidate_verifier_enabled) else None,
        "baseline_plus_haf": baseline_plus_tasks if bool(args.haf_candidate_verifier_enabled) else None,
        "baseline_plus_verified_haf": verified_plus_tasks if bool(args.haf_candidate_verifier_enabled) else None,
        "baseline_plus_verified_decoder": (
            verified_plus_tasks
            if verified_plus_tasks is not None
            and not bool(args.haf_candidate_verifier_enabled)
            and not bool(args.union_selector_enabled)
            else None
        ),
        "union_selector": union_selector_tasks,
        "delta": {
            "decoder_only": {task: _task_delta(decoder_only_tasks[task], baseline_tasks[task]) for task in baseline_tasks},
            "baseline_plus_decoder": {
                task: _task_delta(baseline_plus_tasks[task], baseline_tasks[task]) for task in baseline_tasks
            },
            "baseline_plus_verified_haf": (
                {task: _task_delta(verified_plus_tasks[task], baseline_tasks[task]) for task in baseline_tasks}
                if verified_plus_tasks is not None and bool(args.haf_candidate_verifier_enabled)
                else None
            ),
            "baseline_plus_verified_decoder": (
                {task: _task_delta(verified_plus_tasks[task], baseline_tasks[task]) for task in baseline_tasks}
                if verified_plus_tasks is not None
                and not bool(args.haf_candidate_verifier_enabled)
                and not bool(args.union_selector_enabled)
                else None
            ),
            "union_selector": (
                {task: _task_delta(union_selector_tasks[task], baseline_tasks[task]) for task in baseline_tasks}
                if union_selector_tasks is not None
                else None
            ),
        },
        "interpretation": (
            "No-GT learned structured decoder over frozen dense stop-line maps. "
            "If baseline_slot_refiner_enabled is true, decoder slots are projection-comp "
            "baseline stop-lines plus dense top-support fallback anchors and the model "
            "learns slot objectness plus endpoint refinement. If candidate_verifier_enabled "
            "is true, a second train-split verifier filters decoder candidates before "
            "adding them to the preserved runtime baseline. "
            "If union_selector_enabled is true, retained projection-comp lines and dense decoder candidates "
            "are scored together and only the selected fixed-size set is emitted. "
            "If haf_candidate_verifier_enabled is true, decoded HAF consensus segments "
            "replace the dense-map decoder as the candidate generator and are filtered "
            "by the same train-split no-GT verifier before being added to baseline. "
            "GT is used only to train the decoder/verifier on the train split and to audit validation metrics."
        ),
    }
    return {
        "summary": summary,
        "train_rows": train_rows,
        "val_rows": val_rows,
        "decision_rows": decision_rows,
        "verifier_train_rows": verifier_train_rows,
        "verifier_decision_rows": verifier_decision_rows,
    }


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = run_probe(args)
    summary = payload["summary"]
    (output_dir / "summary.json").write_text(json.dumps(_json_ready(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(output_dir / "train_samples.csv", payload["train_rows"])
    _write_csv(output_dir / "val_samples.csv", payload["val_rows"])
    _write_csv(output_dir / "decoder_decisions.csv", payload["decision_rows"])
    _write_csv(output_dir / "candidate_verifier_train.csv", payload["verifier_train_rows"])
    _write_csv(output_dir / "candidate_verifier_decisions.csv", payload["verifier_decision_rows"])
    print(json.dumps(_json_ready(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
