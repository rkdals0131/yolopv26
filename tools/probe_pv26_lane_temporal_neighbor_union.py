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
    _track_map_evidence,
)
from tools.probe_pv26_lane_instance_evidence import _resolve_dataset_root, _resolve_device, _write_csv
from tools.probe_pv26_lane60_postprocess_thresholds import _detach_to_cpu
from tools.pv26_train import cli as train_cli
from tools.pv26_train import config as train_config_api
from tools.run_pv26_lane60_probe import _lane60_scenario


_TRAILING_FRAME_RE = re.compile(r"^(?P<prefix>.*?)(?P<frame>\d+)$")


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
            "raw_batch": batch,
            "dense": postprocess_predictions,
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
                "neighbor_dataset_index": int(neighbor_dataset_index),
                "neighbor_lane_index": int(neighbor_lane_index),
                "base_lane_count": int(len(base_lanes)),
                "neighbor_lane_count": int(len(neighbor_lanes)),
                "nearest_existing_distance": float(nearest_existing_distance),
                "nearest_existing_distance_bin": _distance_bin(float(nearest_existing_distance)),
                "candidate_points_json": _points_json(candidate),
                "selected": False,
                "blocked_reason": "",
            }
            row.update(evidence)
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
    _, val_loader = train_cli._build_phase_train_loaders(dataset, train_config=train_config, phase=phase)
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

    baseline_predictions_all: list[dict[str, Any]] = []
    repaired_predictions_all: list[dict[str, Any]] = []
    raw_batches: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    global_sample_index = 0

    with torch.no_grad():
        for batch_index, batch in enumerate(val_loader, start=1):
            if batch_index > int(args.max_val_batches):
                break
            if batch_index == 1 or batch_index % 20 == 0:
                print(f"[lane_temporal] eval batch {batch_index}/{args.max_val_batches}", flush=True)
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

                repaired_sample = copy.deepcopy(sample_pred)
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
                repaired_sample["lanes"] = [*list(repaired_sample.get("lanes", [])), *added_lanes]
                repaired_batch_predictions.append(repaired_sample)
                candidate_rows.extend(rows)
                sample_rows.append(
                    {
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
            repaired_predictions_all.extend(repaired_batch_predictions)

    if not raw_batches:
        raise ValueError("no validation batches were processed")

    merged_raw = _merge_raw_batches(raw_batches)
    baseline_metrics = augment_lane_family_metrics(summarize_pv26_metrics(baseline_predictions_all, merged_raw))
    repaired_metrics = augment_lane_family_metrics(summarize_pv26_metrics(repaired_predictions_all, merged_raw))
    baseline_tasks = {task: _metric_payload(baseline_metrics, task) for task in ("lane", "stop_line", "crosswalk")}
    repaired_tasks = {task: _metric_payload(repaired_metrics, task) for task in ("lane", "stop_line", "crosswalk")}
    deltas = {task: _task_delta(repaired_tasks[task], baseline_tasks[task]) for task in ("lane", "stop_line", "crosswalk")}
    selected_rows = [row for row in candidate_rows if bool(row.get("selected"))]
    selected_would_match = [row for row in selected_rows if bool(row.get("would_match_baseline_fn"))]
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
        "selected_count": int(len(selected_rows)),
        "selected_would_match_baseline_fn_count": int(len(selected_would_match)),
        "interpretation": (
            "This is a no-GT temporal smoke, not a training result. Selection uses only immediate "
            "neighbor predictions plus current-frame centerline evidence and duplicate distance; "
            "GT is used only after selection for audit labels and final metrics."
        ),
    }
    return summary | {"candidate_rows": candidate_rows, "sample_rows": sample_rows}


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
