from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, is_dataclass, replace as dataclasses_replace
import json
import math
from pathlib import Path
import random
import site
import sys
from typing import Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
repo_root = str(REPO_ROOT)
if repo_root not in sys.path:
    site.addsitedir(repo_root)

from model.engine._trainer_epochs import _merge_raw_batches
from model.engine.batch import augment_lane_family_metrics, raw_batch_for_metrics
from model.engine.metrics import (
    STOP_LINE_POINT_COUNT,
    _extract_gt_samples,
    _lane_family_metrics,
    _mean_point_distance,
    summarize_pv26_metrics,
)
from model.engine.postprocess import postprocess_pv26_batch
from tools.evaluate_pv26_lane60_checkpoint import _advance_validation_sampler, _postprocess_override_config
from tools.probe_pv26_lane_flip_tta import (
    DEFAULT_CHECKPOINT,
    SOURCE_RUN,
    _apply_lane_task_mask_competition,
    _dedupe_stop_lines_by_distance,
    _detach_to_cpu,
    _json_ready,
    _merge_stop_line_outputs,
    _resolve_device,
    _stop_line_distance,
    _unflip_lane_dense_outputs,
    _build_eval_contract,
)
from tools.pv26_train import cli as train_cli


DEFAULT_STOP_LINE_CHECKPOINT = (
    REPO_ROOT
    / "runs"
    / "pv26_exhaustive_od_lane_train"
    / "lane60_stopline_priority_positive_sampler_from_lane60_lane_head_transplant_original_stop_pca_20260512_default_20260529_101745"
    / "phase_4"
    / "checkpoints"
    / "best.pt"
)
ROUTER_MODES = ("primary", "specialist", "union_dedupe", "agreement", "empty")


class SourceRouterMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a small no-GT runtime router that chooses per-sample stop-line "
            "source outputs from the retained primary checkpoint, the stop-line "
            "specialist checkpoint, their deduped union/agreement, or an empty "
            "FP-suppression source."
        )
    )
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--stop-line-checkpoint", default=str(DEFAULT_STOP_LINE_CHECKPOINT))
    parser.add_argument("--source-run", default=str(SOURCE_RUN))
    parser.add_argument("--lane60-experiment", default="stopline_projection_comp_runtime")
    parser.add_argument("--stop-line-lane60-experiment", default="stopline_projection_comp_runtime")
    parser.add_argument("--preset", default="default")
    parser.add_argument("--router-train-batches", type=int, default=64)
    parser.add_argument("--max-val-batches", type=int, default=4)
    parser.add_argument("--validation-epoch", type=int, default=2)
    parser.add_argument("--train-batches", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--backbone-weights", default="")
    parser.add_argument("--dataset-root", default="")
    parser.add_argument("--router-hidden-dim", type=int, default=32)
    parser.add_argument("--router-epochs", type=int, default=80)
    parser.add_argument("--router-lr", type=float, default=3.0e-3)
    parser.add_argument("--router-weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--seed", type=int, default=20260530)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--stop-line-mask-binary-threshold", type=float, default=None)
    parser.add_argument("--stop-line-min-instance-score", type=float, default=None)
    parser.add_argument("--stop-line-presence-threshold", type=float, default=None)
    parser.add_argument(
        "--stop-line-component-gate-source",
        choices=("center", "selector", "max"),
        default=None,
    )
    parser.add_argument("--crosswalk-polygon-mode", choices=("rect", "hull"), default=None)
    parser.add_argument("--save-router-model", default="")
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
    if points.shape[0] < 2 or not bool(np.isfinite(points).all()):
        return 0.0
    deltas = points[1:] - points[:-1]
    return float(np.linalg.norm(deltas, axis=1).sum())


def _line_angle(line: dict[str, Any]) -> float:
    points = np.asarray(line.get("points_xy", []), dtype=np.float32).reshape(-1, 2)
    if points.shape[0] < 2 or not bool(np.isfinite(points).all()):
        return 0.0
    vector = points[-1] - points[0]
    if float(np.linalg.norm(vector)) <= 1.0e-6:
        return 0.0
    return float(math.atan2(float(vector[1]), float(vector[0])))


def _line_stats(lines: list[dict[str, Any]]) -> list[float]:
    if not lines:
        return [0.0] * 13
    scores = np.asarray([_line_score(line) for line in lines], dtype=np.float32)
    lengths = np.asarray([_line_length(line) for line in lines], dtype=np.float32)
    fragments = np.asarray(
        [
            float(line.get("fragment_count", 0.0))
            if isinstance(line.get("fragment_count"), (int, float))
            else 0.0
            for line in lines
        ],
        dtype=np.float32,
    )
    angles = np.asarray([_line_angle(line) for line in lines], dtype=np.float32)
    points = [
        np.asarray(line.get("points_xy", []), dtype=np.float32).reshape(-1, 2)
        for line in lines
        if len(line.get("points_xy", [])) > 0
    ]
    centers = np.asarray([point.mean(axis=0) for point in points if point.size], dtype=np.float32)
    if centers.size == 0:
        centers = np.zeros((1, 2), dtype=np.float32)
    return [
        float(len(lines)),
        float(scores.max(initial=0.0)),
        float(scores.mean()),
        float(scores.sum()),
        float(lengths.max(initial=0.0)),
        float(lengths.mean()),
        float(lengths.sum()),
        float(fragments.max(initial=0.0)),
        float(fragments.mean()),
        float(np.cos(angles).mean()),
        float(np.sin(angles).mean()),
        float(centers[:, 0].mean()),
        float(centers[:, 1].mean()),
    ]


def _pair_stats(primary_lines: list[dict[str, Any]], specialist_lines: list[dict[str, Any]]) -> list[float]:
    if not primary_lines or not specialist_lines:
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    distances = np.asarray(
        [
            _stop_line_distance(primary, specialist)
            for primary in primary_lines
            for specialist in specialist_lines
        ],
        dtype=np.float32,
    )
    finite = distances[np.isfinite(distances)]
    if finite.size == 0:
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    primary_best = max((_line_score(line) for line in primary_lines), default=0.0)
    specialist_best = max((_line_score(line) for line in specialist_lines), default=0.0)
    primary_len = max((_line_length(line) for line in primary_lines), default=0.0)
    specialist_len = max((_line_length(line) for line in specialist_lines), default=0.0)
    return [
        float(finite.min()),
        float(finite.mean()),
        float((finite <= 40.0).sum()),
        float((finite <= 64.0).sum()),
        float(specialist_best - primary_best),
        float(specialist_len - primary_len),
    ]


def _source_prediction(
    primary_sample: dict[str, Any],
    specialist_sample: dict[str, Any],
    mode: str,
) -> dict[str, Any]:
    primary_lines = [dict(line) for line in primary_sample.get("stop_lines", [])]
    specialist_lines = [dict(line) for line in specialist_sample.get("stop_lines", [])]
    if mode == "primary":
        chosen = primary_lines
    elif mode == "specialist":
        chosen = specialist_lines
    elif mode == "union_dedupe":
        chosen = _dedupe_stop_lines_by_distance([*primary_lines, *specialist_lines])
    elif mode == "agreement":
        chosen = []
        for line in primary_lines:
            if any(_stop_line_distance(line, other) <= 64.0 for other in specialist_lines):
                chosen.append(dict(line))
        for line in specialist_lines:
            if any(_stop_line_distance(line, other) <= 64.0 for other in primary_lines):
                chosen.append(dict(line))
        chosen = _dedupe_stop_lines_by_distance(chosen)
    elif mode == "empty":
        chosen = []
    else:
        raise ValueError(f"unknown router mode: {mode}")
    sample = dict(primary_sample)
    sample["stop_lines"] = chosen
    return sample


def _sample_stopline_metrics(prediction: dict[str, Any], gt_sample: dict[str, Any]) -> dict[str, Any]:
    return _lane_family_metrics(
        [prediction],
        [gt_sample],
        field_name="stop_lines",
        target_count=STOP_LINE_POINT_COUNT,
        match_threshold=40.0,
    )


def _label_for_sample(
    primary_sample: dict[str, Any],
    specialist_sample: dict[str, Any],
    gt_sample: dict[str, Any],
) -> int:
    gt_count = len(gt_sample.get("stop_lines", []))
    ranked: list[tuple[float, float, float, int]] = []
    for index, mode in enumerate(ROUTER_MODES):
        prediction = _source_prediction(primary_sample, specialist_sample, mode)
        metrics = _sample_stopline_metrics(prediction, gt_sample)
        tp = float(metrics.get("tp", 0.0))
        fp = float(metrics.get("fp", 0.0))
        f1 = float(metrics.get("f1", 0.0))
        if gt_count <= 0:
            ranked.append((-fp, 0.0, 0.0, index))
        else:
            # Prefer actual matches, but allow empty only when no source can match.
            ranked.append((f1, tp, -fp, index))
    ranked.sort(reverse=True)
    return int(ranked[0][3])


def _features_for_sample(primary_sample: dict[str, Any], specialist_sample: dict[str, Any]) -> list[float]:
    primary_lines = [dict(line) for line in primary_sample.get("stop_lines", [])]
    specialist_lines = [dict(line) for line in specialist_sample.get("stop_lines", [])]
    union_lines = _dedupe_stop_lines_by_distance([*primary_lines, *specialist_lines])
    features = [
        *_line_stats(primary_lines),
        *_line_stats(specialist_lines),
        *_line_stats(union_lines),
        *_pair_stats(primary_lines, specialist_lines),
    ]
    return [0.0 if not math.isfinite(float(value)) else float(value) for value in features]


def _build_predictions_for_loader(
    *,
    loader: Any,
    max_batches: int,
    evaluator: Any,
    stop_line_evaluator: Any,
    postprocess_config: Any,
    stop_line_postprocess_config: Any,
    split_name: str,
) -> tuple[list[dict[str, Any]], list[list[float]], list[int], dict[str, list[dict[str, Any]]]]:
    raw_batches: list[dict[str, Any]] = []
    features: list[list[float]] = []
    labels: list[int] = []
    predictions_by_mode: dict[str, list[dict[str, Any]]] = {mode: [] for mode in ROUTER_MODES}

    with torch.no_grad():
        for batch_index, batch in enumerate(loader, start=1):
            if batch_index > int(max_batches):
                break
            if batch_index == 1 or batch_index % 20 == 0:
                print(f"[stopline_source_router] {split_name} batch {batch_index}/{max_batches}", flush=True)
            raw_batch = raw_batch_for_metrics(batch)
            if raw_batch is None:
                raw_batch = batch
            raw_batches.append(raw_batch)
            gt_samples = _extract_gt_samples(raw_batch)
            encoded = evaluator.prepare_batch(batch)
            base_outputs = _detach_to_cpu(evaluator.forward_encoded_batch(encoded))
            flipped_encoded = dict(encoded)
            flipped_encoded["image"] = torch.flip(encoded["image"], dims=(-1,))
            flipped_outputs = _detach_to_cpu(evaluator.forward_encoded_batch(flipped_encoded))
            flipped_unflipped = _unflip_lane_dense_outputs(flipped_outputs)
            stop_line_outputs = _detach_to_cpu(stop_line_evaluator.forward_encoded_batch(encoded))
            lane_outputs = _apply_lane_task_mask_competition(
                {
                    **base_outputs,
                    "lane_seg_centerline_logits": 0.5 * base_outputs["lane_seg_centerline_logits"]
                    + 0.5 * flipped_unflipped["lane_seg_centerline_logits"],
                },
                mask_keys=("crosswalk_mask_logits",),
                strength=0.50,
            )
            primary_predictions = postprocess_pv26_batch(
                lane_outputs,
                _detach_to_cpu(encoded["meta"]),
                config=postprocess_config,
            )
            specialist_predictions = postprocess_pv26_batch(
                _merge_stop_line_outputs(lane_outputs, stop_line_outputs),
                _detach_to_cpu(encoded["meta"]),
                config=stop_line_postprocess_config,
            )
            for primary_sample, specialist_sample, gt_sample in zip(
                primary_predictions,
                specialist_predictions,
                gt_samples,
            ):
                features.append(_features_for_sample(primary_sample, specialist_sample))
                labels.append(_label_for_sample(primary_sample, specialist_sample, gt_sample))
                for mode in ROUTER_MODES:
                    predictions_by_mode[mode].append(_source_prediction(primary_sample, specialist_sample, mode))

    return raw_batches, features, labels, predictions_by_mode


def _train_router(
    features: list[list[float]],
    labels: list[int],
    *,
    hidden_dim: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    seed: int,
) -> tuple[SourceRouterMLP, torch.Tensor, torch.Tensor, dict[str, Any]]:
    if not features:
        raise ValueError("no router training features were collected")
    torch.manual_seed(int(seed))
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))
    x = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)
    mean = x.mean(dim=0)
    std = x.std(dim=0).clamp_min(1.0e-6)
    x_norm = (x - mean) / std
    model = SourceRouterMLP(x_norm.shape[1], int(hidden_dim), len(ROUTER_MODES))
    counts = torch.bincount(y, minlength=len(ROUTER_MODES)).to(dtype=torch.float32)
    weights = torch.where(counts > 0.0, float(y.numel()) / (counts * len(ROUTER_MODES)), torch.zeros_like(counts))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    losses: list[float] = []
    for _ in range(int(epochs)):
        optimizer.zero_grad(set_to_none=True)
        logits = model(x_norm)
        loss = F.cross_entropy(logits, y, weight=weights)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    with torch.no_grad():
        train_logits = model(x_norm)
        train_pred = train_logits.argmax(dim=1)
        train_accuracy = float((train_pred == y).float().mean().item())
    diagnostics = {
        "feature_count": int(x.shape[0]),
        "feature_dim": int(x.shape[1]),
        "label_counts": {mode: int(counts[index].item()) for index, mode in enumerate(ROUTER_MODES)},
        "class_weights": {mode: float(weights[index].item()) for index, mode in enumerate(ROUTER_MODES)},
        "train_accuracy": train_accuracy,
        "loss_first": losses[0] if losses else None,
        "loss_last": losses[-1] if losses else None,
    }
    return model, mean, std, diagnostics


def _predict_router_modes(
    model: SourceRouterMLP,
    mean: torch.Tensor,
    std: torch.Tensor,
    features: list[list[float]],
) -> list[int]:
    if not features:
        return []
    with torch.no_grad():
        x = torch.tensor(features, dtype=torch.float32)
        logits = model((x - mean) / std)
        return [int(value) for value in logits.argmax(dim=1).tolist()]


def _prediction_rows_from_choices(
    predictions_by_mode: dict[str, list[dict[str, Any]]],
    choices: list[int],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for sample_index, choice in enumerate(choices):
        mode = ROUTER_MODES[int(choice)]
        output.append(dict(predictions_by_mode[mode][sample_index]))
    return output


def _row_from_metrics(name: str, metrics: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {"variant": name}
    for task in ("lane", "stop_line", "crosswalk"):
        values = metrics.get(task, {}) if isinstance(metrics.get(task), dict) else {}
        for key in ("precision", "recall", "f1", "tp", "fp", "fn", "mean_point_distance", "mean_polygon_iou"):
            value = values.get(key)
            if isinstance(value, (int, float)):
                row[f"{task}_{key}"] = value
    lane_family = metrics.get("lane_family", {}) if isinstance(metrics.get("lane_family"), dict) else {}
    for key in ("mean_f1", "min_f1"):
        value = lane_family.get(key)
        if isinstance(value, (int, float)):
            row[f"lane_family_{key}"] = value
    row["phase4_objective_proxy"] = (
        0.50 * float(metrics.get("lane", {}).get("f1", 0.0))
        + 0.30 * float(metrics.get("stop_line", {}).get("f1", 0.0))
        + 0.20 * float(metrics.get("crosswalk", {}).get("f1", 0.0))
    )
    return row


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _json_safe(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _json_safe(asdict(value))
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return _json_ready(value)


def _choice_counts(choices: list[int]) -> dict[str, int]:
    return {mode: int(sum(1 for choice in choices if int(choice) == index)) for index, mode in enumerate(ROUTER_MODES)}


def _config_brief(config: Any) -> dict[str, Any]:
    return {
        "device": str(getattr(config, "device", "")),
        "batch_size": int(getattr(config, "batch_size", 0)),
        "train_batches": int(getattr(config, "train_batches", 0)),
        "val_batches": int(getattr(config, "val_batches", 0)),
        "task_mode": str(getattr(config, "task_mode", "")),
        "roadmark_architecture": str(getattr(config, "roadmark_architecture", "")),
        "lane_head_mode": str(getattr(config, "lane_head_mode", "")),
        "lane_segfirst_track_mode": str(getattr(config, "lane_segfirst_track_mode", "")),
        "crosswalk_polygon_mode": str(getattr(config, "crosswalk_polygon_mode", "")),
    }


def _postprocess_brief(config: Any) -> dict[str, Any]:
    fields = (
        "lane_segfirst_track_mode",
        "lane_segfirst_semantic_vote_mode",
        "stop_line_projection_comp_enabled",
        "stop_line_projection_comp_min_gap",
        "stop_line_projection_comp_topk",
        "stop_line_projection_comp_max_predictions",
        "crosswalk_polygon_mode",
    )
    return {field: _json_safe(getattr(config, field, None)) for field in fields}


def main() -> int:
    args = parse_args()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    stop_line_checkpoint = Path(args.stop_line_checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    if not stop_line_checkpoint.is_file():
        raise FileNotFoundError(f"stop-line checkpoint not found: {stop_line_checkpoint}")

    source_run = Path(args.source_run).expanduser().resolve()
    scenario, scenario_path, _options, phase, _phase_selection, train_config = _build_eval_contract(
        args,
        source_run=source_run,
        checkpoint=checkpoint,
        experiment=str(args.lane60_experiment),
    )
    (
        stop_line_scenario,
        stop_line_scenario_path,
        _stop_line_options,
        stop_line_phase,
        _stop_line_phase_selection,
        stop_line_train_config,
    ) = _build_eval_contract(
        args,
        source_run=source_run,
        checkpoint=stop_line_checkpoint,
        experiment=str(args.stop_line_lane60_experiment),
    )
    train_config = dataclasses_replace(
        train_config,
        device=_resolve_device(str(args.device), train_config.device),
        encode_train_batches_in_loader=False,
    )
    stop_line_train_config = dataclasses_replace(
        stop_line_train_config,
        device=_resolve_device(str(args.device), stop_line_train_config.device),
    )

    train_cli._configure_torch_multiprocessing()
    dataset = train_cli.PV26CanonicalDataset(
        train_cli._existing_dataset_roots(scenario),
        train_augmentation=False,
        progress_callback=lambda message: print(f"[stopline_source_router] {message}", flush=True),
    )
    train_loader, val_loader = train_cli._build_phase_train_loaders(dataset, train_config=train_config, phase=phase)
    if train_loader is None or val_loader is None:
        raise ValueError("source-router probe requires both train and validation loaders")
    _advance_validation_sampler(val_loader, validation_epoch=int(args.validation_epoch))

    trainer = train_cli._build_phase_trainer(phase, train_config)
    load_report = trainer.load_model_weights(checkpoint, map_location=train_config.device)
    evaluator = trainer.build_evaluator()
    evaluator.adapter.raw_model.eval()
    evaluator.heads.eval()
    postprocess_config = _postprocess_override_config(args, trainer) or train_cli._build_postprocess_config(train_config)

    stop_line_trainer = train_cli._build_phase_trainer(stop_line_phase, stop_line_train_config)
    stop_line_load_report = stop_line_trainer.load_model_weights(
        stop_line_checkpoint,
        map_location=stop_line_train_config.device,
    )
    stop_line_evaluator = stop_line_trainer.build_evaluator()
    stop_line_evaluator.adapter.raw_model.eval()
    stop_line_evaluator.heads.eval()
    stop_line_postprocess_config = (
        _postprocess_override_config(args, stop_line_trainer)
        or train_cli._build_postprocess_config(stop_line_train_config)
    )

    train_raw_batches, train_features, train_labels, _train_predictions = _build_predictions_for_loader(
        loader=train_loader,
        max_batches=int(args.router_train_batches),
        evaluator=evaluator,
        stop_line_evaluator=stop_line_evaluator,
        postprocess_config=postprocess_config,
        stop_line_postprocess_config=stop_line_postprocess_config,
        split_name="train",
    )
    router, feature_mean, feature_std, router_diagnostics = _train_router(
        train_features,
        train_labels,
        hidden_dim=int(args.router_hidden_dim),
        epochs=int(args.router_epochs),
        lr=float(args.router_lr),
        weight_decay=float(args.router_weight_decay),
        seed=int(args.seed),
    )
    val_raw_batches, val_features, val_labels, val_predictions_by_mode = _build_predictions_for_loader(
        loader=val_loader,
        max_batches=int(args.max_val_batches),
        evaluator=evaluator,
        stop_line_evaluator=stop_line_evaluator,
        postprocess_config=postprocess_config,
        stop_line_postprocess_config=stop_line_postprocess_config,
        split_name="val",
    )
    val_choices = _predict_router_modes(router, feature_mean, feature_std, val_features)
    val_oracle_choices = [int(value) for value in val_labels]

    merged_raw = _merge_raw_batches(val_raw_batches)
    predictions_by_variant: dict[str, list[dict[str, Any]]] = {
        mode: val_predictions_by_mode[mode] for mode in ROUTER_MODES
    }
    predictions_by_variant["learned_router"] = _prediction_rows_from_choices(val_predictions_by_mode, val_choices)
    predictions_by_variant["oracle_router"] = _prediction_rows_from_choices(val_predictions_by_mode, val_oracle_choices)
    metrics_by_variant = {
        name: augment_lane_family_metrics(summarize_pv26_metrics(predictions, merged_raw))
        for name, predictions in predictions_by_variant.items()
    }
    rows = [_row_from_metrics(name, metrics) for name, metrics in metrics_by_variant.items()]
    rows.sort(key=lambda row: float(row.get("phase4_objective_proxy", 0.0)), reverse=True)

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if str(args.output_dir).strip()
        else checkpoint.parent / "analysis_exports" / "stopline_source_router"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "metrics.csv", rows)
    if str(args.save_router_model).strip():
        torch.save(
            {
                "state_dict": router.state_dict(),
                "feature_mean": feature_mean,
                "feature_std": feature_std,
                "router_modes": ROUTER_MODES,
                "diagnostics": router_diagnostics,
            },
            Path(args.save_router_model).expanduser().resolve(),
        )
    payload = {
        "checkpoint": str(checkpoint),
        "stop_line_checkpoint": str(stop_line_checkpoint),
        "scenario_path": str(scenario_path),
        "stop_line_scenario_path": str(stop_line_scenario_path),
        "lane60_experiment": str(args.lane60_experiment),
        "stop_line_lane60_experiment": str(args.stop_line_lane60_experiment),
        "validation_epoch": int(args.validation_epoch),
        "router_train_batches": int(args.router_train_batches),
        "max_val_batches": int(args.max_val_batches),
        "processed_train_batches": int(len(train_raw_batches)),
        "processed_val_batches": int(len(val_raw_batches)),
        "router_modes": list(ROUTER_MODES),
        "router_diagnostics": router_diagnostics,
        "val_label_counts": _choice_counts(val_oracle_choices),
        "val_router_choice_counts": _choice_counts(val_choices),
        "train_config": _config_brief(train_config),
        "stop_line_train_config": _config_brief(stop_line_train_config),
        "postprocess_config": _postprocess_brief(postprocess_config),
        "stop_line_postprocess_config": _postprocess_brief(stop_line_postprocess_config),
        "load_report": {
            "missing_keys": len(load_report.get("missing_keys", [])) if isinstance(load_report, dict) else None,
            "unexpected_keys": len(load_report.get("unexpected_keys", [])) if isinstance(load_report, dict) else None,
        },
        "stop_line_load_report": {
            "missing_keys": len(stop_line_load_report.get("missing_keys", []))
            if isinstance(stop_line_load_report, dict)
            else None,
            "unexpected_keys": len(stop_line_load_report.get("unexpected_keys", []))
            if isinstance(stop_line_load_report, dict)
            else None,
        },
        "rows": rows,
        "metrics_by_variant": _json_safe(metrics_by_variant),
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(rows, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    print(f"[stopline_source_router] wrote {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
