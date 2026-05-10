from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
import math
from pathlib import Path
import site
import sys
import time
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
repo_root = str(REPO_ROOT)
if repo_root not in sys.path:
    site.addsitedir(repo_root)

from model.engine import raw_batch_for_metrics
from model.engine._trainer_epochs import _merge_raw_batches
from model.engine.metrics import (
    STOP_LINE_POINT_COUNT,
    _extract_gt_samples,
    _hungarian_from_cost,
    _hungarian_from_similarity,
    _mean_point_distance,
    _polygon_iou,
)
from model.engine.postprocess import postprocess_pv26_batch
from tools.probe_pv26_lane60_postprocess_thresholds import _advance_validation_sampler, _detach_to_cpu
from tools.probe_pv26_lane60_support_gate import _load_scenario
from tools.pv26_train import cli as train_cli
from tools.pv26_train import config as train_config_api


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export lane-family prediction TP/FP/FN filter features for a PV26 lane60 checkpoint."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--preset", default="default")
    parser.add_argument("--source-run", default="")
    parser.add_argument("--lane60-experiment", default="")
    parser.add_argument("--phase-index", type=int, default=4)
    parser.add_argument("--max-val-batches", type=int, default=128)
    parser.add_argument("--validation-epoch", type=int, default=2)
    parser.add_argument("--train-batches", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", default="")
    return parser.parse_args()


def _resolve_device(requested: str, scenario_device: str) -> str:
    value = str(requested or "auto").strip().lower()
    device = str(scenario_device) if value == "auto" else str(requested)
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[filters] CUDA requested but unavailable; falling back to CPU", flush=True)
        return "cpu"
    return device


def _points(points_xy: list[list[float]]) -> np.ndarray:
    return np.asarray(points_xy, dtype=np.float32).reshape(-1, 2)


def _polyline_length(points_xy: list[list[float]]) -> float:
    points = _points(points_xy)
    if points.shape[0] < 2 or not bool(np.isfinite(points).all()):
        return 0.0
    return float(np.linalg.norm(points[1:] - points[:-1], axis=1).sum())


def _polygon_area(points_xy: list[list[float]]) -> float:
    points = _points(points_xy)
    if points.shape[0] < 3 or not bool(np.isfinite(points).all()):
        return 0.0
    x = points[:, 0]
    y = points[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) * 0.5)


def _shape_features(points_xy: list[list[float]]) -> dict[str, float]:
    points = _points(points_xy)
    if points.size == 0 or not bool(np.isfinite(points).all()):
        return {
            "bbox_w": 0.0,
            "bbox_h": 0.0,
            "bbox_area": 0.0,
            "bbox_aspect": 0.0,
            "center_x": 0.0,
            "center_y": 0.0,
            "polyline_length": 0.0,
            "polygon_area": 0.0,
        }
    min_xy = points.min(axis=0)
    max_xy = points.max(axis=0)
    width = float(max_xy[0] - min_xy[0])
    height = float(max_xy[1] - min_xy[1])
    return {
        "bbox_w": width,
        "bbox_h": height,
        "bbox_area": max(width, 0.0) * max(height, 0.0),
        "bbox_aspect": max(width, height) / max(min(width, height), 1.0),
        "center_x": float((min_xy[0] + max_xy[0]) * 0.5),
        "center_y": float((min_xy[1] + max_xy[1]) * 0.5),
        "polyline_length": _polyline_length(points_xy),
        "polygon_area": _polygon_area(points_xy),
    }


def _record(
    *,
    sample_index: int,
    task: str,
    kind: str,
    pred: dict[str, Any] | None,
    gt: dict[str, Any] | None,
    match_quality: float = 0.0,
) -> dict[str, Any]:
    source = pred if pred is not None else gt if gt is not None else {"points_xy": []}
    points_xy = source.get("points_xy", [])
    features = _shape_features(points_xy)
    row = {
        "sample_index": int(sample_index),
        "task": task,
        "kind": kind,
        "score": float((pred or {}).get("score", 0.0)),
        "match_quality": float(match_quality),
    }
    row.update(features)
    return row


def _match_records(
    *,
    sample_index: int,
    task: str,
    pred_rows: list[dict[str, Any]],
    gt_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if task == "crosswalk":
        similarity = np.zeros((len(pred_rows), len(gt_rows)), dtype=np.float32)
        for pred_index, pred in enumerate(pred_rows):
            for gt_index, gt in enumerate(gt_rows):
                similarity[pred_index, gt_index] = _polygon_iou(pred["points_xy"], gt["points_xy"])
        matches = _hungarian_from_similarity(similarity, min_similarity=0.30)
        quality = lambda pred_index, gt_index: float(similarity[pred_index, gt_index])
    else:
        target_count = STOP_LINE_POINT_COUNT if task == "stop_line" else 16
        cost = np.zeros((len(pred_rows), len(gt_rows)), dtype=np.float32)
        for pred_index, pred in enumerate(pred_rows):
            for gt_index, gt in enumerate(gt_rows):
                cost[pred_index, gt_index] = _mean_point_distance(pred["points_xy"], gt["points_xy"], target_count)
        matches = _hungarian_from_cost(cost, max_cost=40.0)
        quality = lambda pred_index, gt_index: float(cost[pred_index, gt_index])

    matched_pred = {pred_index for pred_index, _ in matches}
    matched_gt = {gt_index for _, gt_index in matches}
    for pred_index, gt_index in matches:
        records.append(
            _record(
                sample_index=sample_index,
                task=task,
                kind="tp",
                pred=pred_rows[pred_index],
                gt=gt_rows[gt_index],
                match_quality=quality(pred_index, gt_index),
            )
        )
    for pred_index, pred in enumerate(pred_rows):
        if pred_index not in matched_pred:
            records.append(_record(sample_index=sample_index, task=task, kind="fp", pred=pred, gt=None))
    for gt_index, gt in enumerate(gt_rows):
        if gt_index not in matched_gt:
            records.append(_record(sample_index=sample_index, task=task, kind="fn", pred=None, gt=gt))
    return records


def _quantiles(values: list[float]) -> dict[str, float]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return {"count": 0, "min": 0.0, "q25": 0.0, "median": 0.0, "q75": 0.0, "max": 0.0}
    arr = np.asarray(finite, dtype=np.float32)
    return {
        "count": int(arr.size),
        "min": float(np.min(arr)),
        "q25": float(np.quantile(arr, 0.25)),
        "median": float(np.quantile(arr, 0.50)),
        "q75": float(np.quantile(arr, 0.75)),
        "max": float(np.max(arr)),
    }


def main() -> int:
    args = parse_args()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")

    scenario, scenario_path = _load_scenario(args)
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
        if args.output_dir
        else checkpoint.parents[2] / "analysis_exports" / f"lane60_filter_features_{int(time.time())}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    train_cli._configure_torch_multiprocessing()
    dataset = train_cli.PV26CanonicalDataset(
        train_cli._existing_dataset_roots(scenario),
        train_augmentation=False,
        progress_callback=lambda message: print(f"[filters] {message}", flush=True),
    )
    _, val_loader = train_cli._build_phase_train_loaders(dataset, train_config=train_config, phase=phase)
    if val_loader is None:
        raise ValueError("filter analysis requires validation batches")
    _advance_validation_sampler(val_loader, validation_epoch=int(args.validation_epoch))

    trainer = train_cli._build_phase_trainer(phase, train_config)
    trainer.load_model_weights(checkpoint, map_location=train_config.device)
    evaluator = trainer.build_evaluator()
    evaluator.adapter.raw_model.eval()
    evaluator.heads.eval()

    predictions: list[dict[str, Any]] = []
    raw_batches: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch_index, batch in enumerate(val_loader, start=1):
            if batch_index > int(args.max_val_batches):
                break
            print(f"[filters] eval batch {batch_index}/{args.max_val_batches}", flush=True)
            raw_batch = raw_batch_for_metrics(batch)
            if raw_batch is None:
                raise ValueError("validation loader must provide raw batches for metrics")
            encoded = evaluator.prepare_batch(batch)
            batch_predictions = _detach_to_cpu(evaluator.forward_encoded_batch(encoded))
            meta = _detach_to_cpu(encoded["meta"])
            predictions.extend(postprocess_pv26_batch(batch_predictions, meta, config=postprocess_config))
            raw_batches.append(raw_batch)

    gt_samples = _extract_gt_samples(_merge_raw_batches(raw_batches))
    rows: list[dict[str, Any]] = []
    for sample_index, (pred_sample, gt_sample) in enumerate(zip(predictions, gt_samples)):
        rows.extend(
            _match_records(
                sample_index=sample_index,
                task="lane",
                pred_rows=list(pred_sample["lanes"]),
                gt_rows=list(gt_sample["lanes"]),
            )
        )
        rows.extend(
            _match_records(
                sample_index=sample_index,
                task="stop_line",
                pred_rows=list(pred_sample["stop_lines"]),
                gt_rows=list(gt_sample["stop_lines"]),
            )
        )
        rows.extend(
            _match_records(
                sample_index=sample_index,
                task="crosswalk",
                pred_rows=list(pred_sample["crosswalks"]),
                gt_rows=list(gt_sample["crosswalks"]),
            )
        )

    fieldnames = [
        "sample_index",
        "task",
        "kind",
        "score",
        "match_quality",
        "bbox_w",
        "bbox_h",
        "bbox_area",
        "bbox_aspect",
        "center_x",
        "center_y",
        "polyline_length",
        "polygon_area",
    ]
    with (output_dir / "prediction_filter_features.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary: dict[str, Any] = {
        "checkpoint": str(checkpoint),
        "scenario_path": str(scenario_path),
        "phase_name": phase.name,
        "phase_stage": phase.stage,
        "max_val_batches": int(args.max_val_batches),
        "validation_epoch": int(args.validation_epoch),
        "postprocess_config": vars(postprocess_config),
        "tasks": {},
    }
    for task in ("lane", "stop_line", "crosswalk"):
        task_rows = [row for row in rows if row["task"] == task]
        task_summary: dict[str, Any] = {
            "counts": {
                kind: sum(1 for row in task_rows if row["kind"] == kind)
                for kind in ("tp", "fp", "fn")
            },
            "features": {},
        }
        for feature in ("score", "bbox_w", "bbox_h", "bbox_area", "bbox_aspect", "center_x", "center_y", "polyline_length", "polygon_area"):
            task_summary["features"][feature] = {
                kind: _quantiles([float(row[feature]) for row in task_rows if row["kind"] == kind])
                for kind in ("tp", "fp", "fn")
            }
        summary["tasks"][task] = task_summary

    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[filters] wrote {output_dir}", flush=True)
    for task, task_summary in summary["tasks"].items():
        print(f"[filters] {task} counts={task_summary['counts']}", flush=True)
        for feature in ("score", "bbox_area", "bbox_aspect", "center_y", "polygon_area"):
            print(f"[filters] {task} {feature}={task_summary['features'][feature]}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
