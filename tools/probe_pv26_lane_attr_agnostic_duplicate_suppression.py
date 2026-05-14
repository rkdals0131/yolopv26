from __future__ import annotations

import argparse
import copy
from dataclasses import replace
import json
import math
from pathlib import Path
import site
import sys
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
repo_root = str(REPO_ROOT)
if repo_root not in sys.path:
    site.addsitedir(repo_root)

from model.engine import raw_batch_for_metrics
from model.engine._trainer_epochs import _merge_raw_batches
from model.engine.batch import augment_lane_family_metrics
from model.engine.metrics import _extract_gt_samples, summarize_pv26_metrics
from model.engine.postprocess import postprocess_pv26_batch
from tools.evaluate_pv26_lane60_checkpoint import _advance_validation_sampler, _postprocess_override_config
from tools.probe_pv26_lane_flip_tta import _merge_lane_dense_predictions, _unflip_lane_dense_outputs
from tools.probe_pv26_lane_fn_recovery_audit import (
    DEFAULT_CHECKPOINT,
    SOURCE_RUN,
    _distance_bin,
    _lane_distance,
    _metric,
    _nearest_gt_lane,
    _points_json,
)
from tools.probe_pv26_lane_instance_evidence import _resolve_dataset_root, _resolve_device, _write_csv
from tools.probe_pv26_lane60_postprocess_thresholds import _detach_to_cpu
from tools.pv26_train import cli as train_cli
from tools.pv26_train import config as train_config_api
from tools.run_pv26_lane60_probe import _lane60_scenario


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay lane duplicate suppression while ignoring lane color/type attributes. "
            "Metrics match lanes by geometry only, so this checks whether cross-attribute "
            "near-duplicates are leaking FP after the same-schema dedupe audit."
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
    parser.add_argument("--dataset-root", default="")
    parser.add_argument("--lane-flip-variant", choices=("baseline", "flip_centerline_avg"), default="flip_centerline_avg")
    parser.add_argument("--duplicate-distance", type=float, default=24.0)
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


def _metric_payload(metrics: dict[str, Any], task: str) -> dict[str, float]:
    return {
        "precision": _metric(metrics, task, "precision"),
        "recall": _metric(metrics, task, "recall"),
        "f1": _metric(metrics, task, "f1"),
        "tp": _metric(metrics, task, "tp"),
        "fp": _metric(metrics, task, "fp"),
        "fn": _metric(metrics, task, "fn"),
    }


def _task_delta(replayed: dict[str, float], baseline: dict[str, float]) -> dict[str, float]:
    return {
        "precision_delta": float(replayed["precision"] - baseline["precision"]),
        "recall_delta": float(replayed["recall"] - baseline["recall"]),
        "f1_delta": float(replayed["f1"] - baseline["f1"]),
        "tp_delta": float(replayed["tp"] - baseline["tp"]),
        "fp_delta": float(replayed["fp"] - baseline["fp"]),
        "fn_delta": float(replayed["fn"] - baseline["fn"]),
    }


def _lane_score(lane: dict[str, Any]) -> float:
    try:
        return float(lane.get("score", 0.0))
    except (TypeError, ValueError):
        return 0.0


def _lane_has_same_schema(first: dict[str, Any], second: dict[str, Any]) -> bool:
    return str(first.get("class_name", "")) == str(second.get("class_name", "")) and str(
        first.get("lane_type", "")
    ) == str(second.get("lane_type", ""))


def suppress_duplicate_lanes(
    lanes: list[dict[str, Any]],
    *,
    duplicate_distance: float,
    require_same_schema: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ranked = sorted(enumerate(lanes), key=lambda item: (_lane_score(item[1]), -int(item[0])), reverse=True)
    kept_ranked: list[tuple[int, dict[str, Any]]] = []
    suppressed: list[dict[str, Any]] = []
    for original_index, lane in ranked:
        suppressor_index = -1
        suppressor_distance = math.inf
        for kept_index, kept_lane in kept_ranked:
            if require_same_schema and not _lane_has_same_schema(lane, kept_lane):
                continue
            distance = _lane_distance(lane, kept_lane)
            if distance <= float(duplicate_distance):
                suppressor_index = int(kept_index)
                suppressor_distance = float(distance)
                break
        if suppressor_index >= 0:
            suppressed.append(
                {
                    "original_index": int(original_index),
                    "suppressor_index": int(suppressor_index),
                    "suppressor_distance": float(suppressor_distance),
                    "suppressed_class_name": str(lane.get("class_name", "")),
                    "suppressed_lane_type": str(lane.get("lane_type", "")),
                    "suppressed_score": _lane_score(lane),
                    "suppressor_class_name": str(lanes[suppressor_index].get("class_name", "")),
                    "suppressor_lane_type": str(lanes[suppressor_index].get("lane_type", "")),
                    "suppressor_score": _lane_score(lanes[suppressor_index]),
                    "same_schema": _lane_has_same_schema(lane, lanes[suppressor_index]),
                    "suppressed_points_json": _points_json(lane),
                }
            )
            continue
        kept_ranked.append((int(original_index), lane))
    kept_indices = {index for index, _lane in kept_ranked}
    kept = [copy.deepcopy(lane) for index, lane in enumerate(lanes) if index in kept_indices]
    return kept, suppressed


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
    )
    return scenario, scenario_path, options, phase, train_config


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    source_run = Path(args.source_run).expanduser().resolve()
    scenario, scenario_path, options, phase, train_config = _build_scenario(args)
    phase_index = int(tuple(options["selected_phase_indices"])[0])

    train_cli._configure_torch_multiprocessing()
    dataset = train_cli.PV26CanonicalDataset(
        train_cli._existing_dataset_roots(scenario),
        train_augmentation=False,
        progress_callback=lambda message: print(f"[lane_attr_dedupe] {message}", flush=True),
    )
    _, val_loader = train_cli._build_phase_train_loaders(dataset, train_config=train_config, phase=phase)
    if val_loader is None:
        raise ValueError("lane attr-agnostic duplicate probe requires validation batches")
    _advance_validation_sampler(val_loader, validation_epoch=int(args.validation_epoch))

    trainer = train_cli._build_phase_trainer(phase, train_config)
    trainer.load_model_weights(checkpoint, map_location=train_config.device)
    postprocess_config = _postprocess_override_config(args, trainer) or train_cli._build_postprocess_config(train_config)
    evaluator = trainer.build_evaluator()
    evaluator.adapter.raw_model.eval()
    evaluator.heads.eval()

    baseline_predictions_all: list[dict[str, Any]] = []
    replay_predictions_all: list[dict[str, Any]] = []
    raw_batches: list[dict[str, Any]] = []
    suppressed_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    global_sample_index = 0

    with torch.no_grad():
        for batch_index, batch in enumerate(val_loader, start=1):
            if batch_index > int(args.max_val_batches):
                break
            if batch_index == 1 or batch_index % 20 == 0:
                print(f"[lane_attr_dedupe] eval batch {batch_index}/{args.max_val_batches}", flush=True)
            raw_batch = raw_batch_for_metrics(batch)
            if raw_batch is None:
                raise ValueError("validation loader must provide raw batches for metrics")
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

            replay_batch_predictions: list[dict[str, Any]] = []
            for sample_batch_index, (sample_pred, sample_gt, sample_meta) in enumerate(
                zip(batch_predictions, batch_gt, meta)
            ):
                replay_sample = copy.deepcopy(sample_pred)
                lanes = list(sample_pred.get("lanes", []))
                deduped_lanes, suppressed = suppress_duplicate_lanes(
                    lanes,
                    duplicate_distance=float(args.duplicate_distance),
                    require_same_schema=False,
                )
                replay_sample["lanes"] = deduped_lanes
                gt_lanes = list(sample_gt.get("lanes", []))
                for row in suppressed:
                    nearest_gt_index, nearest_gt_distance = _nearest_gt_lane(lanes[int(row["original_index"])], gt_lanes)
                    row.update(
                        {
                            "batch_index": int(batch_index),
                            "sample_index": int(global_sample_index),
                            "sample_batch_index": int(sample_batch_index),
                            "sample_id": str(sample_meta.get("sample_id", "")),
                            "nearest_gt_index": int(nearest_gt_index),
                            "nearest_gt_distance": float(nearest_gt_distance),
                            "nearest_gt_distance_bin": _distance_bin(float(nearest_gt_distance)),
                            "would_match_gt": bool(float(nearest_gt_distance) <= 40.0),
                        }
                    )
                    suppressed_rows.append(row)
                sample_rows.append(
                    {
                        "batch_index": int(batch_index),
                        "sample_index": int(global_sample_index),
                        "sample_batch_index": int(sample_batch_index),
                        "sample_id": str(sample_meta.get("sample_id", "")),
                        "baseline_lane_count": int(len(lanes)),
                        "deduped_lane_count": int(len(deduped_lanes)),
                        "suppressed_count": int(len(suppressed)),
                    }
                )
                replay_batch_predictions.append(replay_sample)
                global_sample_index += 1
            replay_predictions_all.extend(replay_batch_predictions)

    if not raw_batches:
        raise ValueError("no validation batches were processed")

    merged_raw = _merge_raw_batches(raw_batches)
    baseline_metrics = augment_lane_family_metrics(summarize_pv26_metrics(baseline_predictions_all, merged_raw))
    replay_metrics = augment_lane_family_metrics(summarize_pv26_metrics(replay_predictions_all, merged_raw))
    baseline_tasks = {task: _metric_payload(baseline_metrics, task) for task in ("lane", "stop_line", "crosswalk")}
    replay_tasks = {task: _metric_payload(replay_metrics, task) for task in ("lane", "stop_line", "crosswalk")}
    deltas = {task: _task_delta(replay_tasks[task], baseline_tasks[task]) for task in ("lane", "stop_line", "crosswalk")}
    suppressed_cross_schema = [row for row in suppressed_rows if not bool(row.get("same_schema"))]
    suppressed_would_match = [row for row in suppressed_rows if bool(row.get("would_match_gt"))]
    summary = {
        "checkpoint": str(checkpoint),
        "source_run": str(source_run),
        "scenario_path": str(scenario_path),
        "lane60_experiment": str(args.lane60_experiment),
        "phase_index": int(phase_index),
        "max_val_batches": int(args.max_val_batches),
        "validation_epoch": int(args.validation_epoch),
        "lane_flip_variant": str(args.lane_flip_variant),
        "duplicate_distance": float(args.duplicate_distance),
        "baseline": baseline_tasks,
        "replayed": replay_tasks,
        "delta": deltas,
        "sample_count": int(len(sample_rows)),
        "suppressed_count": int(len(suppressed_rows)),
        "suppressed_cross_schema_count": int(len(suppressed_cross_schema)),
        "suppressed_would_match_gt_count": int(len(suppressed_would_match)),
        "interpretation": (
            "This is a fixed replay only. It tests whether class/type-agnostic near-duplicate lane "
            "suppression can reduce FP under the geometry-only lane metric without dropping TP."
        ),
    }
    return summary | {"suppressed_rows": suppressed_rows, "sample_rows": sample_rows}


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = run_probe(args)
    suppressed_rows = list(payload.pop("suppressed_rows"))
    sample_rows = list(payload.pop("sample_rows"))
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(output_dir / "lane_attr_agnostic_suppressed_rows.csv", suppressed_rows)
    _write_csv(output_dir / "lane_attr_agnostic_samples.csv", sample_rows)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
