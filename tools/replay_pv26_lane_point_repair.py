from __future__ import annotations

import argparse
import copy
import json
import math
from dataclasses import replace
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
from model.engine.lane_segfirst_vectorizer import lane_segfirst_prediction_maps
from model.engine.metrics import _extract_gt_samples, summarize_pv26_metrics
from model.engine.postprocess import postprocess_pv26_batch
from tools.evaluate_pv26_lane60_checkpoint import _advance_validation_sampler, _postprocess_override_config
from tools.probe_pv26_lane_flip_tta import _merge_lane_dense_predictions, _unflip_lane_dense_outputs
from tools.probe_pv26_lane_fn_recovery_audit import (
    DEFAULT_CHECKPOINT,
    SOURCE_RUN,
    _distance_bin,
    _gt_centerline_evidence,
    _match_predictions,
    _metric,
    _nearest_gt_lane,
    _points_json,
    _prediction_shape_features,
    _track_map_evidence,
)
from tools.probe_pv26_lane_instance_evidence import _resolve_dataset_root, _resolve_device, _write_csv
from tools.probe_pv26_lane60_postprocess_thresholds import _detach_to_cpu
from tools.pv26_train import cli as train_cli
from tools.pv26_train import config as train_config_api
from tools.run_pv26_lane60_probe import _lane60_scenario


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay actual lane point repairs and recompute lane-family metrics. "
            "The default mode is explicitly oracle-only: it copies the nearest "
            "currently missed GT lane points into selected unmatched predictions."
        )
    )
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--source-run", default=str(SOURCE_RUN))
    parser.add_argument("--lane60-experiment", default="core_centerline_refine_row_scan_tangent_link")
    parser.add_argument("--preset", default="default")
    parser.add_argument("--phase-index", type=int, default=4)
    parser.add_argument("--max-val-batches", type=int, default=128)
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
    parser.add_argument(
        "--selection-mode",
        choices=("oracle_le120_any_center", "oracle_le80_center050", "all_unmatched_with_fn"),
        default="oracle_le120_any_center",
    )
    parser.add_argument(
        "--repair-mode",
        choices=("oracle_replace_nearest_fn",),
        default="oracle_replace_nearest_fn",
    )
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


def _task_delta(repaired: dict[str, float], baseline: dict[str, float]) -> dict[str, float]:
    return {
        "precision_delta": float(repaired["precision"] - baseline["precision"]),
        "recall_delta": float(repaired["recall"] - baseline["recall"]),
        "f1_delta": float(repaired["f1"] - baseline["f1"]),
        "tp_delta": float(repaired["tp"] - baseline["tp"]),
        "fp_delta": float(repaired["fp"] - baseline["fp"]),
        "fn_delta": float(repaired["fn"] - baseline["fn"]),
    }


def _candidate_is_selected(candidate: dict[str, Any], selection_mode: str) -> bool:
    if int(candidate.get("nearest_fn_gt_index", -1)) < 0:
        return False
    distance = float(candidate.get("nearest_fn_gt_distance", math.inf))
    center_mean = float(candidate.get("nearest_fn_gt_center_point_mean", 0.0))
    if selection_mode == "oracle_le120_any_center":
        return distance <= 120.0
    if selection_mode == "oracle_le80_center050":
        return distance <= 80.0 and center_mean >= 0.50
    if selection_mode == "all_unmatched_with_fn":
        return math.isfinite(distance)
    raise ValueError(f"unsupported selection mode: {selection_mode}")


def _replace_lane_points(lane: dict[str, Any], points_xy: list[Any]) -> dict[str, Any]:
    repaired = dict(lane)
    repaired["points_xy"] = [[float(point[0]), float(point[1])] for point in points_xy]
    return repaired


def _candidate_key(candidate: dict[str, Any]) -> tuple[int, int]:
    return (int(candidate.get("sample_index", -1)), int(candidate.get("nearest_fn_gt_index", -1)))


def _summarize_selected_targets(candidate_rows: list[dict[str, Any]]) -> dict[str, Any]:
    selected = [row for row in candidate_rows if bool(row.get("selected"))]
    unique_targets = {_candidate_key(row) for row in selected}
    return {
        "candidate_count": int(len(candidate_rows)),
        "selected_count": int(len(selected)),
        "selected_unique_target_count": int(len(unique_targets)),
        "selected_duplicate_target_count": int(len(selected) - len(unique_targets)),
        "samples_with_selected": int(len({int(row.get("sample_index", -1)) for row in selected})),
    }


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


def run_replay(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    source_run = Path(args.source_run).expanduser().resolve()
    scenario, scenario_path, options, phase, train_config = _build_scenario(args)
    phase_index = int(tuple(options["selected_phase_indices"])[0])

    train_cli._configure_torch_multiprocessing()
    dataset = train_cli.PV26CanonicalDataset(
        train_cli._existing_dataset_roots(scenario),
        train_augmentation=False,
        progress_callback=lambda message: print(f"[lane_point_repair] {message}", flush=True),
    )
    _, val_loader = train_cli._build_phase_train_loaders(dataset, train_config=train_config, phase=phase)
    if val_loader is None:
        raise ValueError("lane point repair replay requires validation batches")
    _advance_validation_sampler(val_loader, validation_epoch=int(args.validation_epoch))

    trainer = train_cli._build_phase_trainer(phase, train_config)
    trainer.load_model_weights(checkpoint, map_location=train_config.device)
    postprocess_config = _postprocess_override_config(args, trainer) or train_cli._build_postprocess_config(train_config)
    evaluator = trainer.build_evaluator()
    evaluator.adapter.raw_model.eval()
    evaluator.heads.eval()

    baseline_predictions_all: list[dict[str, Any]] = []
    repaired_predictions_all: list[dict[str, Any]] = []
    raw_batches: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    global_sample_index = 0

    with torch.no_grad():
        for batch_index, batch in enumerate(val_loader, start=1):
            if batch_index > int(args.max_val_batches):
                break
            if batch_index == 1 or batch_index % 20 == 0:
                print(f"[lane_point_repair] eval batch {batch_index}/{args.max_val_batches}", flush=True)
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

            repaired_batch_predictions: list[dict[str, Any]] = []
            for sample_batch_index, (sample_pred, sample_gt, sample_meta) in enumerate(
                zip(batch_predictions, batch_gt, meta)
            ):
                repaired_sample = copy.deepcopy(sample_pred)
                pred_lanes = list(sample_pred.get("lanes", []))
                gt_lanes = list(sample_gt.get("lanes", []))
                pred_to_gt, gt_to_pred = _match_predictions(pred_lanes, gt_lanes)
                unmatched_pred_indices = set(range(len(pred_lanes))) - set(pred_to_gt)
                unmatched_gt_indices = set(range(len(gt_lanes))) - set(gt_to_pred)
                maps = lane_segfirst_prediction_maps(postprocess_predictions, batch_index=sample_batch_index)
                fn_evidence_by_gt_index: dict[int, dict[str, float]] = {
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
                    row: dict[str, Any] = {
                        "batch_index": int(batch_index),
                        "sample_index": int(global_sample_index),
                        "sample_batch_index": int(sample_batch_index),
                        "pred_index": int(pred_index),
                        "sample_gt_lane_count": int(len(gt_lanes)),
                        "sample_pred_lane_count": int(len(pred_lanes)),
                        "sample_matched_lane_count": int(len(gt_to_pred)),
                        "sample_unmatched_pred_count": int(len(unmatched_pred_indices)),
                        "nearest_fn_gt_index": int(nearest_fn_gt_index),
                        "nearest_fn_gt_distance": float(nearest_fn_gt_distance),
                        "nearest_fn_gt_distance_bin": _distance_bin(float(nearest_fn_gt_distance)),
                        "nearest_fn_gt_center_point_mean": nearest_fn_center_mean,
                        "pred_points_json": _points_json(pred_lane),
                        "nearest_fn_gt_points_json": _points_json(
                            gt_lanes[nearest_fn_gt_index] if nearest_fn_gt_index >= 0 else None
                        ),
                    }
                    row.update(_prediction_shape_features(pred_lane))
                    row.update(
                        _track_map_evidence(
                            list(pred_lane.get("points_xy", [])),
                            maps=maps,
                            meta=sample_meta,
                            prefix="pred",
                        )
                    )
                    selected = _candidate_is_selected(row, str(args.selection_mode))
                    row["selected"] = bool(selected)
                    row["repair_mode"] = str(args.repair_mode)
                    if selected and int(nearest_fn_gt_index) >= 0:
                        repaired_sample["lanes"][pred_index] = _replace_lane_points(
                            repaired_sample["lanes"][pred_index],
                            list(gt_lanes[nearest_fn_gt_index].get("points_xy", [])),
                        )
                    candidate_rows.append(row)

                repaired_batch_predictions.append(repaired_sample)
                global_sample_index += 1

            repaired_predictions_all.extend(repaired_batch_predictions)

    if not raw_batches:
        raise ValueError("no validation batches were processed")

    merged_raw = _merge_raw_batches(raw_batches)
    baseline_metrics = augment_lane_family_metrics(summarize_pv26_metrics(baseline_predictions_all, merged_raw))
    repaired_metrics = augment_lane_family_metrics(summarize_pv26_metrics(repaired_predictions_all, merged_raw))
    baseline_tasks = {task: _metric_payload(baseline_metrics, task) for task in ("lane", "stop_line", "crosswalk")}
    repaired_tasks = {task: _metric_payload(repaired_metrics, task) for task in ("lane", "stop_line", "crosswalk")}
    deltas = {
        task: _task_delta(repaired_tasks[task], baseline_tasks[task])
        for task in ("lane", "stop_line", "crosswalk")
    }
    selected_summary = _summarize_selected_targets(candidate_rows)
    summary = {
        "checkpoint": str(checkpoint),
        "source_run": str(source_run),
        "scenario_path": str(scenario_path),
        "lane60_experiment": str(args.lane60_experiment),
        "phase_index": int(phase_index),
        "max_val_batches": int(args.max_val_batches),
        "validation_epoch": int(args.validation_epoch),
        "lane_flip_variant": str(args.lane_flip_variant),
        "selection_mode": str(args.selection_mode),
        "repair_mode": str(args.repair_mode),
        "baseline": baseline_tasks,
        "repaired": repaired_tasks,
        "delta": deltas,
        "selection": selected_summary,
        "interpretation": (
            "Oracle-only geometry replay. Selected unmatched predictions are assigned nearest missed GT "
            "lane points before metric recomputation. This proves the point-replay machinery and upper "
            "bound movement, not a production no-GT decoder."
        ),
    }
    return summary | {"candidate_rows": candidate_rows}


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = run_replay(args)
    candidate_rows = list(payload.pop("candidate_rows"))
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(output_dir / "lane_point_repair_candidates.csv", candidate_rows)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
