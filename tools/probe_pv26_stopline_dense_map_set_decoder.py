from __future__ import annotations

import argparse
import itertools
import json
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
from model.engine.postprocess import _dedupe_stop_line_predictions, postprocess_pv26_batch
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
    parser.add_argument("--object-threshold", type=float, default=0.55)
    parser.add_argument("--max-output-segments", type=int, default=2)
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
                object_target[query_indices] = 1.0
                direct = F.smooth_l1_loss(query_points[query_indices], target[target_indices], reduction="none").mean(dim=(1, 2))
                flipped = F.smooth_l1_loss(
                    query_points[query_indices].flip(dims=(1,)),
                    target[target_indices],
                    reduction="none",
                ).mean(dim=(1, 2))
                point_loss = torch.minimum(direct, flipped).mean()
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
            loss = _set_decoder_loss(logits, points, batch_targets, pos_weight=pos_weight)
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
                targets = _gt_stopline_segments_norm(sample_gt, sample_meta, max_count=int(args.decoder_queries))
                examples.append(
                    {
                        "features": features,
                        "targets": targets,
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
    model, train_summary = _train_decoder(train_examples, args=args, device=device)
    val_examples, baseline_predictions, raw_batches, val_rows = _collect_examples(
        loader=val_loader,
        evaluator=evaluator,
        postprocess_config=postprocess_config,
        args=args,
        max_batches=int(args.max_val_batches),
        training=False,
    )
    decoder_only_predictions, baseline_plus_predictions, decision_rows = _apply_decoder(
        examples=val_examples,
        baseline_predictions=baseline_predictions,
        model=model,
        args=args,
        device=device,
    )
    merged_raw = _merge_raw_batches(raw_batches)
    baseline_metrics = augment_lane_family_metrics(summarize_pv26_metrics(baseline_predictions, merged_raw))
    decoder_only_metrics = augment_lane_family_metrics(summarize_pv26_metrics(decoder_only_predictions, merged_raw))
    baseline_plus_metrics = augment_lane_family_metrics(summarize_pv26_metrics(baseline_plus_predictions, merged_raw))
    baseline_tasks = {task: _metric_payload(baseline_metrics, task) for task in ("lane", "stop_line", "crosswalk")}
    decoder_only_tasks = {task: _metric_payload(decoder_only_metrics, task) for task in ("lane", "stop_line", "crosswalk")}
    baseline_plus_tasks = {task: _metric_payload(baseline_plus_metrics, task) for task in ("lane", "stop_line", "crosswalk")}
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
        "max_output_segments": int(args.max_output_segments),
        "pool_hw": [int(args.pool_height), int(args.pool_width)],
        "train_summary": train_summary,
        "selected_prediction_count": int(sum(int(row["selected"]) for row in decision_rows)),
        "baseline": baseline_tasks,
        "decoder_only": decoder_only_tasks,
        "baseline_plus_decoder": baseline_plus_tasks,
        "delta": {
            "decoder_only": {task: _task_delta(decoder_only_tasks[task], baseline_tasks[task]) for task in baseline_tasks},
            "baseline_plus_decoder": {
                task: _task_delta(baseline_plus_tasks[task], baseline_tasks[task]) for task in baseline_tasks
            },
        },
        "interpretation": (
            "No-GT learned structured decoder over frozen dense stop-line maps. "
            "GT is used only to train the decoder on the train split and to audit validation metrics."
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
    _write_csv(output_dir / "train_samples.csv", payload["train_rows"])
    _write_csv(output_dir / "val_samples.csv", payload["val_rows"])
    _write_csv(output_dir / "decoder_decisions.csv", payload["decision_rows"])
    print(json.dumps(_json_ready(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
