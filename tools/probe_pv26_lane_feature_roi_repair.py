from __future__ import annotations

import argparse
from dataclasses import replace
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

from model.data.transform import inverse_transform_points, transform_from_meta, transform_points
from model.engine import raw_batch_for_metrics
from model.engine._trainer_epochs import _merge_raw_batches
from model.engine.batch import augment_lane_family_metrics
from model.engine.lane_segfirst_vectorizer import lane_segfirst_prediction_maps
from model.engine.metrics import _extract_gt_samples, _mean_point_distance, summarize_pv26_metrics
from model.engine.postprocess import postprocess_pv26_batch
from tools.evaluate_pv26_lane60_checkpoint import _advance_validation_sampler, _postprocess_override_config
from tools.probe_pv26_lane60_postprocess_thresholds import _detach_to_cpu
from tools.probe_pv26_lane_flip_tta import _merge_lane_dense_predictions, _unflip_lane_dense_outputs
from tools.probe_pv26_lane_instance_evidence import _resolve_dataset_root, _resolve_device, _write_csv
from tools.pv26_train import cli as train_cli
from tools.pv26_train import config as train_config_api
from tools.run_pv26_lane60_probe import _lane60_scenario


SOURCE_RUN = (
    REPO_ROOT
    / "runs"
    / "pv26_exhaustive_od_lane_train"
    / "lane60_lane_head_transplant_original_stop_pca_20260512"
)
DEFAULT_CHECKPOINT = SOURCE_RUN / "merged_lane_head.pt"
SAMPLE_COUNT = 16
LANE_MATCH_THRESHOLD = 40.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train and replay a small no-GT lane feature-ROI repair module. "
            "It samples dense lane features along decoded lane polylines, predicts a "
            "replacement polyline plus repair probability, and applies the repair in "
            "replace-only mode."
        )
    )
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--source-run", default=str(SOURCE_RUN))
    parser.add_argument("--lane60-experiment", default="stopline_projection_comp_runtime")
    parser.add_argument("--preset", default="default")
    parser.add_argument("--repair-train-batches", type=int, default=64)
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
    parser.add_argument("--positive-distance-px", type=float, default=120.0)
    parser.add_argument("--negative-distance-px", type=float, default=180.0)
    parser.add_argument("--repair-quality-threshold", type=float, default=0.70)
    parser.add_argument(
        "--repair-min-source-distance-px",
        type=float,
        default=LANE_MATCH_THRESHOLD,
        help=(
            "Only candidates farther than this from their nearest GT lane are "
            "positive repair examples; closer candidates become do-not-repair negatives."
        ),
    )
    parser.add_argument("--max-mean-move-px", type=float, default=90.0)
    parser.add_argument("--min-mean-move-px", type=float, default=0.0)
    parser.add_argument("--max-repairs-per-sample", type=int, default=3)
    parser.add_argument("--repair-epochs", type=int, default=40)
    parser.add_argument("--repair-batch-size", type=int, default=256)
    parser.add_argument("--repair-lr", type=float, default=1.0e-3)
    parser.add_argument("--point-loss-weight", type=float, default=1.0)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--max-repair-delta-norm", type=float, default=0.30)
    parser.add_argument(
        "--negative-identity-loss-weight",
        type=float,
        default=0.25,
        help="SmoothL1 weight that trains do-not-repair negatives to preserve their original polyline.",
    )
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
    parser.add_argument("--save-repair-model", action="store_true")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


class LaneFeatureRoiRepairNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        *,
        hidden_dim: int,
        sample_count: int = SAMPLE_COUNT,
        max_delta_norm: float = 0.30,
    ) -> None:
        super().__init__()
        self.sample_count = int(sample_count)
        self.max_delta_norm = float(max_delta_norm)
        self.net = nn.Sequential(
            nn.Linear(int(input_dim), int(hidden_dim)),
            nn.SiLU(inplace=True),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.SiLU(inplace=True),
        )
        self.points = nn.Linear(int(hidden_dim), self.sample_count * 2)
        self.quality = nn.Linear(int(hidden_dim), 1)
        nn.init.zeros_(self.points.weight)
        nn.init.zeros_(self.points.bias)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.net(features)
        point_delta_norm = torch.tanh(self.points(hidden)).view(-1, self.sample_count, 2) * self.max_delta_norm
        quality_logits = self.quality(hidden).squeeze(-1)
        return point_delta_norm, quality_logits


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


def _map_points_to_raw(points: np.ndarray, meta: dict[str, Any], map_hw: tuple[int, int]) -> list[list[float]]:
    transform = transform_from_meta(meta)
    out = np.asarray(points, dtype=np.float32).reshape(-1, 2).copy()
    if out.size == 0:
        return []
    map_h, map_w = int(map_hw[0]), int(map_hw[1])
    net_h, net_w = int(transform.network_hw[0]), int(transform.network_hw[1])
    out[:, 0] = np.clip(out[:, 0] * float(net_w) / max(float(map_w), 1.0), 0.0, float(net_w - 1))
    out[:, 1] = np.clip(out[:, 1] * float(net_h) / max(float(map_h), 1.0), 0.0, float(net_h - 1))
    return [[float(x), float(y)] for x, y in inverse_transform_points(out.tolist(), transform)]


def _sample_polyline(points: np.ndarray, *, count: int = SAMPLE_COUNT) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if points.shape[0] == 0:
        return np.zeros((int(count), 2), dtype=np.float32)
    if points.shape[0] == 1:
        return np.repeat(points[:1], repeats=int(count), axis=0)
    deltas = points[1:] - points[:-1]
    lengths = np.linalg.norm(deltas, axis=1)
    total = float(lengths.sum())
    if total <= 1.0e-6:
        return np.repeat(points[:1], repeats=int(count), axis=0)
    cumulative = np.concatenate([[0.0], np.cumsum(lengths)])
    targets = np.linspace(0.0, total, int(count), dtype=np.float32)
    sampled: list[np.ndarray] = []
    for target in targets:
        index = int(np.searchsorted(cumulative, float(target), side="right") - 1)
        index = max(0, min(index, len(lengths) - 1))
        interval = max(float(lengths[index]), 1.0e-6)
        ratio = (float(target) - float(cumulative[index])) / interval
        sampled.append(points[index] + ratio * (points[index + 1] - points[index]))
    return np.asarray(sampled, dtype=np.float32)


def _normalize_points(points: np.ndarray, map_hw: tuple[int, int]) -> np.ndarray:
    out = np.asarray(points, dtype=np.float32).reshape(-1, 2).copy()
    h, w = int(map_hw[0]), int(map_hw[1])
    out[:, 0] = out[:, 0] / max(float(w - 1), 1.0)
    out[:, 1] = out[:, 1] / max(float(h - 1), 1.0)
    return np.clip(out, 0.0, 1.0)


def _denormalize_points(points: np.ndarray, map_hw: tuple[int, int]) -> np.ndarray:
    out = np.asarray(points, dtype=np.float32).reshape(-1, 2).copy()
    h, w = int(map_hw[0]), int(map_hw[1])
    out[:, 0] = np.clip(out[:, 0] * max(float(w - 1), 1.0), 0.0, float(w - 1))
    out[:, 1] = np.clip(out[:, 1] * max(float(h - 1), 1.0), 0.0, float(h - 1))
    return out


def _values_at_points(array: np.ndarray, points: np.ndarray) -> np.ndarray:
    if points.shape[0] == 0:
        return np.zeros(0, dtype=np.float32)
    h, w = int(array.shape[-2]), int(array.shape[-1])
    xs = np.clip(np.rint(points[:, 0]).astype(np.int64), 0, w - 1)
    ys = np.clip(np.rint(points[:, 1]).astype(np.int64), 0, h - 1)
    if array.ndim == 2:
        return np.asarray(array[ys, xs], dtype=np.float32)
    return np.asarray(array[:, ys, xs].T, dtype=np.float32)


def _as_channel(value: torch.Tensor) -> np.ndarray:
    array = value.detach().cpu().numpy().astype(np.float32)
    if array.ndim == 3 and int(array.shape[0]) == 1:
        return array[0]
    return array


def _nearest_gt(
    lane: dict[str, Any],
    gt_lanes: list[dict[str, Any]],
) -> tuple[int, float]:
    best_index = -1
    best_distance = float("inf")
    for gt_index, gt_lane in enumerate(gt_lanes):
        distance = float(_mean_point_distance(lane.get("points_xy", []), gt_lane.get("points_xy", []), SAMPLE_COUNT))
        if distance < best_distance:
            best_index = int(gt_index)
            best_distance = float(distance)
    return best_index, best_distance


def _lane_features(
    lane: dict[str, Any],
    *,
    predictions: dict[str, Any],
    maps: dict[str, torch.Tensor],
    meta: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    centerline = _as_channel(maps["centerline_core"])
    support = _as_channel(maps["support"])
    tangent_axis = maps["tangent_axis"].detach().cpu().numpy().astype(np.float32)
    lane_feature = predictions.get("lane_feature")
    if not isinstance(lane_feature, torch.Tensor):
        raise KeyError("lane feature ROI repair requires lane_feature output")
    lane_feature_array = lane_feature.detach().cpu().numpy().astype(np.float32)
    if lane_feature_array.ndim != 3:
        raise ValueError(f"expected CHW lane_feature for one sample, got {lane_feature_array.shape}")
    map_hw = (int(centerline.shape[0]), int(centerline.shape[1]))
    map_points = _raw_points_to_map(list(lane.get("points_xy", [])), meta, map_hw)
    sampled = _sample_polyline(map_points, count=SAMPLE_COUNT)
    norm_points = _normalize_points(sampled, map_hw)
    center_values = _values_at_points(centerline, sampled).reshape(SAMPLE_COUNT, 1)
    support_values = _values_at_points(support, sampled).reshape(SAMPLE_COUNT, 1)
    tangent_values = _values_at_points(tangent_axis, sampled).reshape(SAMPLE_COUNT, 2)
    feature_values = _values_at_points(lane_feature_array, sampled)
    feature_mean = feature_values.mean(axis=0) if feature_values.size else np.zeros(lane_feature_array.shape[0], dtype=np.float32)
    feature_std = feature_values.std(axis=0) if feature_values.size else np.zeros(lane_feature_array.shape[0], dtype=np.float32)
    points = np.asarray(lane.get("points_xy", []), dtype=np.float32).reshape(-1, 2)
    if points.size:
        min_xy = points.min(axis=0)
        max_xy = points.max(axis=0)
        bbox = max_xy - min_xy
        shape = np.asarray(
            [
                float(bbox[0]) / 1280.0,
                float(bbox[1]) / 720.0,
                float(max(bbox[0] * bbox[1], 0.0)) / (1280.0 * 720.0),
                float(max(bbox[0], bbox[1]) / max(min(bbox[0], bbox[1]), 1.0)) / 20.0,
                float((min_xy[0] + max_xy[0]) * 0.5) / 1280.0,
                float((min_xy[1] + max_xy[1]) * 0.5) / 720.0,
            ],
            dtype=np.float32,
        )
    else:
        shape = np.zeros(6, dtype=np.float32)
    vector = np.concatenate(
        [
            norm_points.reshape(-1),
            center_values.reshape(-1),
            support_values.reshape(-1),
            tangent_values.reshape(-1),
            feature_mean.reshape(-1),
            feature_std.reshape(-1),
            shape,
        ]
    ).astype(np.float32)
    return vector, sampled, map_hw


def _json_ready(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return {"tensor_shape": list(value.shape), "tensor_dtype": str(value.dtype)}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _metric_payload(metrics: dict[str, Any], task: str) -> dict[str, float]:
    payload = metrics.get(task, {}) if isinstance(metrics.get(task), dict) else {}
    return {
        "precision": float(payload.get("precision", 0.0)),
        "recall": float(payload.get("recall", 0.0)),
        "f1": float(payload.get("f1", 0.0)),
        "tp": float(payload.get("tp", 0.0)),
        "fp": float(payload.get("fp", 0.0)),
        "fn": float(payload.get("fn", 0.0)),
    }


def _task_delta(repaired: dict[str, float], baseline: dict[str, float]) -> dict[str, float]:
    return {f"{key}_delta": float(repaired[key] - baseline[key]) for key in baseline}


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
    backbone_weights = str(args.backbone_weights).strip()
    if backbone_weights:
        scenario = replace(
            scenario,
            train_defaults=replace(
                scenario.train_defaults,
                backbone_weights=str(Path(backbone_weights).expanduser().resolve()),
            ),
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
        device=_resolve_device(str(args.device), train_config.device),
        val_batches=int(args.max_val_batches),
        batch_size=int(args.batch_size),
        encode_train_batches_in_loader=False,
        encode_val_batches_in_loader=False,
    )
    return scenario, scenario_path, options, phase, train_config


def _forward_predictions(
    evaluator: Any,
    encoded: dict[str, Any],
    *,
    lane_flip_variant: str,
) -> dict[str, Any]:
    predictions = _detach_to_cpu(evaluator.forward_encoded_batch(encoded))
    if str(lane_flip_variant) == "baseline":
        return predictions
    flipped_encoded = dict(encoded)
    flipped_encoded["image"] = torch.flip(encoded["image"], dims=(-1,))
    flipped_predictions = _unflip_lane_dense_outputs(_detach_to_cpu(evaluator.forward_encoded_batch(flipped_encoded)))
    return _merge_lane_dense_predictions(predictions, flipped_predictions, variant=str(lane_flip_variant))


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
                print(f"[lane_feature_roi_repair] collect {split} batch {batch_index}/{max_batches}", flush=True)
            raw_batch = raw_batch_for_metrics(batch)
            if raw_batch is None:
                raise ValueError("lane feature ROI repair requires raw batches for metrics")
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
                for pred_index, lane in enumerate(sample_pred.get("lanes", [])):
                    gt_index, distance = _nearest_gt(lane, gt_lanes)
                    positive = bool(
                        gt_index >= 0
                        and float(distance) > float(args.repair_min_source_distance_px)
                        and float(distance) <= float(args.positive_distance_px)
                    )
                    negative = bool(
                        gt_index < 0
                        or float(distance) <= float(args.repair_min_source_distance_px)
                        or float(distance) >= float(args.negative_distance_px)
                    )
                    if training and not positive and not negative:
                        continue
                    features, sampled_map_points, map_hw = _lane_features(
                        lane,
                        predictions=sample_prediction_tensors,
                        maps=maps,
                        meta=sample_meta,
                    )
                    target_points = sampled_map_points
                    if positive:
                        target_points = _sample_polyline(
                            _raw_points_to_map(list(gt_lanes[gt_index].get("points_xy", [])), sample_meta, map_hw),
                            count=SAMPLE_COUNT,
                        )
                    example = {
                        "features": features,
                        "target_points_norm": _normalize_points(target_points, map_hw).astype(np.float32),
                        "pred_points_norm": _normalize_points(sampled_map_points, map_hw).astype(np.float32),
                        "positive": float(1.0 if positive else 0.0),
                        "nearest_gt_index": int(gt_index),
                        "nearest_gt_distance": float(distance),
                        "sample_index": int(global_sample_index),
                        "sample_batch_index": int(sample_batch_index),
                        "pred_index": int(pred_index),
                        "map_hw": tuple(int(v) for v in map_hw),
                    }
                    examples.append(example)
                    rows.append(
                        {
                            "split": "train" if training else "val",
                            "batch_index": int(batch_index),
                            "sample_index": int(global_sample_index),
                            "sample_batch_index": int(sample_batch_index),
                            "pred_index": int(pred_index),
                            "nearest_gt_index": int(gt_index),
                            "nearest_gt_distance": float(distance),
                            "positive": int(positive),
                        }
                    )
                global_sample_index += 1
    return examples, predictions_all, raw_batches, rows


def _train_repair_model(
    examples: list[dict[str, Any]],
    *,
    args: argparse.Namespace,
    device: str,
) -> tuple[LaneFeatureRoiRepairNet, dict[str, Any]]:
    if not examples:
        raise ValueError("no repair training examples collected")
    features = torch.tensor(np.stack([row["features"] for row in examples]), dtype=torch.float32)
    targets = torch.tensor(np.stack([row["target_points_norm"] for row in examples]), dtype=torch.float32)
    origins = torch.tensor(np.stack([row["pred_points_norm"] for row in examples]), dtype=torch.float32)
    labels = torch.tensor([float(row["positive"]) for row in examples], dtype=torch.float32)
    mean = features.mean(dim=0)
    std = features.std(dim=0).clamp(min=1.0e-6)
    features = (features - mean) / std
    model = LaneFeatureRoiRepairNet(
        int(features.shape[1]),
        hidden_dim=int(args.hidden_dim),
        sample_count=SAMPLE_COUNT,
        max_delta_norm=float(args.max_repair_delta_norm),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.repair_lr), weight_decay=1.0e-4)
    features = features.to(device)
    targets = targets.to(device)
    origins = origins.to(device)
    labels = labels.to(device)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(args.seed))
    batch_size = max(1, int(args.repair_batch_size))
    positive_count = int(labels.sum().item())
    negative_count = int(labels.numel() - positive_count)
    pos_weight = torch.tensor(
        [max(float(negative_count) / max(float(positive_count), 1.0), 1.0)],
        dtype=torch.float32,
        device=device,
    )
    history: list[dict[str, float]] = []
    for epoch in range(1, max(1, int(args.repair_epochs)) + 1):
        order = torch.randperm(int(labels.numel()), generator=generator)
        epoch_losses: list[float] = []
        for start in range(0, int(order.numel()), batch_size):
            index = order[start : start + batch_size].to(device)
            pred_delta_norm, quality_logits = model(features[index])
            pred_points = torch.clamp(origins[index] + pred_delta_norm, min=0.0, max=1.0)
            batch_labels = labels[index]
            quality_loss = F.binary_cross_entropy_with_logits(
                quality_logits,
                batch_labels,
                pos_weight=pos_weight,
            )
            positive_mask = batch_labels > 0.5
            point_loss = pred_points.sum() * 0.0
            if bool(positive_mask.any()):
                point_loss = point_loss + F.smooth_l1_loss(
                    pred_points[positive_mask],
                    targets[index][positive_mask],
                    reduction="mean",
                )
            negative_mask = ~positive_mask
            negative_identity_weight = float(args.negative_identity_loss_weight)
            if negative_identity_weight > 0.0 and bool(negative_mask.any()):
                point_loss = point_loss + negative_identity_weight * F.smooth_l1_loss(
                    pred_points[negative_mask],
                    targets[index][negative_mask],
                    reduction="mean",
                )
            loss = quality_loss + float(args.point_loss_weight) * point_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        if epoch == 1 or epoch == int(args.repair_epochs) or epoch % 10 == 0:
            history.append({"epoch": float(epoch), "loss": float(np.mean(epoch_losses) if epoch_losses else 0.0)})
    model.register_buffer("feature_mean", mean.to(device), persistent=True)
    model.register_buffer("feature_std", std.to(device), persistent=True)
    summary = {
        "example_count": int(labels.numel()),
        "positive_count": positive_count,
        "negative_count": negative_count,
        "input_dim": int(features.shape[1]),
        "max_repair_delta_norm": float(args.max_repair_delta_norm),
        "point_loss_weight": float(args.point_loss_weight),
        "history": history,
    }
    return model, summary


def _apply_repairs(
    *,
    examples: list[dict[str, Any]],
    predictions_all: list[dict[str, Any]],
    model: LaneFeatureRoiRepairNet,
    args: argparse.Namespace,
    device: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    repaired = [dict(sample, lanes=[dict(lane) for lane in sample.get("lanes", [])]) for sample in predictions_all]
    if not examples:
        return repaired, []
    feature_tensor = torch.tensor(np.stack([row["features"] for row in examples]), dtype=torch.float32, device=device)
    feature_tensor = (feature_tensor - model.feature_mean) / model.feature_std
    original_points_norm = torch.tensor(
        np.stack([row["pred_points_norm"] for row in examples]),
        dtype=torch.float32,
        device=device,
    )
    with torch.no_grad():
        pred_delta_norm, quality_logits = model(feature_tensor)
        pred_points_norm = torch.clamp(original_points_norm + pred_delta_norm, min=0.0, max=1.0)
        probabilities = torch.sigmoid(quality_logits).detach().cpu().numpy().astype(np.float32)
        pred_points_norm_np = pred_points_norm.detach().cpu().numpy().astype(np.float32)
    rows: list[dict[str, Any]] = []
    selected_by_sample: dict[int, list[tuple[float, int]]] = {}
    for index, (example, probability) in enumerate(zip(examples, probabilities.tolist())):
        sample_index = int(example["sample_index"])
        selected_by_sample.setdefault(sample_index, []).append((float(probability), index))
    selected_indices: set[int] = set()
    for sample_index, candidates in selected_by_sample.items():
        candidates.sort(reverse=True)
        for probability, index in candidates[: max(0, int(args.max_repairs_per_sample))]:
            if probability >= float(args.repair_quality_threshold):
                selected_indices.add(int(index))
    for index, example in enumerate(examples):
        sample_index = int(example["sample_index"])
        pred_index = int(example["pred_index"])
        map_hw = tuple(int(v) for v in example["map_hw"])
        original_points = _denormalize_points(np.asarray(example["pred_points_norm"], dtype=np.float32), map_hw)
        predicted_points = _denormalize_points(pred_points_norm_np[index], map_hw)
        mean_move = float(np.linalg.norm(predicted_points - original_points, axis=1).mean())
        selected = (
            int(index) in selected_indices
            and mean_move >= float(args.min_mean_move_px)
            and mean_move <= float(args.max_mean_move_px)
        )
        if selected and 0 <= sample_index < len(repaired):
            lanes = repaired[sample_index].get("lanes", [])
            if 0 <= pred_index < len(lanes):
                lanes[pred_index]["points_xy"] = _map_points_to_raw(predicted_points, repaired[sample_index]["meta"], map_hw)
                lanes[pred_index]["roi_repair_score"] = float(probabilities[index])
                lanes[pred_index]["roi_repair_mean_move_px"] = mean_move
        rows.append(
            {
                "sample_index": sample_index,
                "pred_index": pred_index,
                "nearest_gt_index": int(example["nearest_gt_index"]),
                "nearest_gt_distance": float(example["nearest_gt_distance"]),
                "positive": int(float(example["positive"]) > 0.5),
                "repair_probability": float(probabilities[index]),
                "mean_move_px": mean_move,
                "selected": int(bool(selected)),
            }
        )
    return repaired, rows


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
        progress_callback=lambda message: print(f"[lane_feature_roi_repair] {message}", flush=True),
    )
    train_loader, val_loader = train_cli._build_phase_train_loaders(dataset, train_config=train_config, phase=phase)
    if train_loader is None or val_loader is None:
        raise ValueError("lane feature ROI repair requires train and validation loaders")
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
        max_batches=int(args.repair_train_batches),
        training=True,
    )
    model, train_summary = _train_repair_model(train_examples, args=args, device=device)

    val_examples, baseline_predictions, raw_batches, val_rows = _collect_examples(
        loader=val_loader,
        evaluator=evaluator,
        postprocess_config=postprocess_config,
        args=args,
        max_batches=int(args.max_val_batches),
        training=False,
    )
    repaired_predictions, repair_rows = _apply_repairs(
        examples=val_examples,
        predictions_all=baseline_predictions,
        model=model,
        args=args,
        device=device,
    )
    merged_raw = _merge_raw_batches(raw_batches)
    baseline_metrics = augment_lane_family_metrics(summarize_pv26_metrics(baseline_predictions, merged_raw))
    repaired_metrics = augment_lane_family_metrics(summarize_pv26_metrics(repaired_predictions, merged_raw))
    baseline_tasks = {task: _metric_payload(baseline_metrics, task) for task in ("lane", "stop_line", "crosswalk")}
    repaired_tasks = {task: _metric_payload(repaired_metrics, task) for task in ("lane", "stop_line", "crosswalk")}
    deltas = {task: _task_delta(repaired_tasks[task], baseline_tasks[task]) for task in baseline_tasks}
    selected = [row for row in repair_rows if int(row.get("selected", 0))]
    summary = {
        "checkpoint": str(checkpoint),
        "source_run": str(source_run),
        "scenario_path": str(scenario_path),
        "lane60_experiment": str(args.lane60_experiment),
        "phase_index": int(phase_index),
        "lane_flip_variant": str(args.lane_flip_variant),
        "repair_train_batches": int(args.repair_train_batches),
        "max_val_batches": int(args.max_val_batches),
        "validation_epoch": int(args.validation_epoch),
        "positive_distance_px": float(args.positive_distance_px),
        "negative_distance_px": float(args.negative_distance_px),
        "repair_min_source_distance_px": float(args.repair_min_source_distance_px),
        "max_repair_delta_norm": float(args.max_repair_delta_norm),
        "negative_identity_loss_weight": float(args.negative_identity_loss_weight),
        "repair_quality_threshold": float(args.repair_quality_threshold),
        "point_loss_weight": float(args.point_loss_weight),
        "max_mean_move_px": float(args.max_mean_move_px),
        "min_mean_move_px": float(args.min_mean_move_px),
        "max_repairs_per_sample": int(args.max_repairs_per_sample),
        "train_summary": train_summary,
        "val_example_count": int(len(val_examples)),
        "selected_repair_count": int(len(selected)),
        "baseline": baseline_tasks,
        "repaired": repaired_tasks,
        "delta": deltas,
        "interpretation": (
            "No-GT runtime replay of a learned lane feature-ROI repair module. "
            "The repair model is trained on train split decoded candidates and applied "
            "to validation predictions in replace-only mode."
        ),
    }
    return {
        "summary": summary,
        "train_rows": train_rows,
        "val_rows": val_rows,
        "repair_rows": repair_rows,
        "model_state": {
            key: value.detach().cpu()
            for key, value in model.state_dict().items()
        },
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
    _write_csv(output_dir / "repair_decisions.csv", payload["repair_rows"])
    if bool(args.save_repair_model):
        torch.save(payload["model_state"], output_dir / "lane_feature_roi_repair.pt")
    print(json.dumps(_json_ready(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
