from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import site
import sys
import time
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
repo_root = str(REPO_ROOT)
if repo_root not in sys.path:
    site.addsitedir(repo_root)

import torch

from model.engine.batch import augment_lane_family_metrics, raw_batch_for_metrics
from model.engine.metrics import summarize_pv26_metrics
from model.engine.postprocess import PV26PostprocessConfig, postprocess_pv26_batch
from model.engine._trainer_epochs import _merge_raw_batches
from tools.pv26_train import cli as train_cli
from tools.pv26_train import config as train_config_api


DEFAULT_RUN_DIR = (
    REPO_ROOT
    / "runs"
    / "pv26_exhaustive_od_lane_train"
    / "exhaustive_od_lane_default_20260505_032217"
)
DEFAULT_CHECKPOINT = DEFAULT_RUN_DIR / "phase_4" / "checkpoints" / "best.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep lane-family postprocess thresholds for a PV26 checkpoint. "
            "The sweep runs one task threshold at a time, then verifies the combined task-best thresholds."
        )
    )
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT), help="Checkpoint to evaluate.")
    parser.add_argument("--preset", default="default", help="PV26 meta-train preset name.")
    parser.add_argument("--phase-index", type=int, default=4, help="Scenario phase index to evaluate.")
    parser.add_argument("--max-val-batches", type=int, default=128, help="Validation batches to evaluate.")
    parser.add_argument(
        "--device",
        default="auto",
        help="Evaluation device. Use 'auto' to keep the scenario device when available and fall back to CPU.",
    )
    parser.add_argument(
        "--thresholds",
        default="0.20,0.30,0.40,0.50,0.60,0.70,0.80",
        help="Comma-separated threshold values used for each one-axis sweep.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Output directory. Defaults to <run>/analysis_exports/lane_family_threshold_sweep_<timestamp>.",
    )
    return parser.parse_args()


def _thresholds(raw: str) -> list[float]:
    values: list[float] = []
    for item in str(raw).split(","):
        stripped = item.strip()
        if not stripped:
            continue
        value = float(stripped)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"threshold must be in [0, 1], got {value}")
        values.append(value)
    if not values:
        raise ValueError("at least one threshold is required")
    return sorted(dict.fromkeys(values))


def _metric_value(metrics: dict[str, Any], task_name: str, metric_name: str) -> float:
    task_metrics = metrics.get(task_name)
    if not isinstance(task_metrics, dict):
        return 0.0
    value = task_metrics.get(metric_name)
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


def _lane_family_row(
    *,
    config_name: str,
    varied_task: str,
    thresholds: dict[str, float],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "config_name": config_name,
        "varied_task": varied_task,
        "lane_threshold": float(thresholds["lane"]),
        "stop_line_threshold": float(thresholds["stop_line"]),
        "crosswalk_threshold": float(thresholds["crosswalk"]),
    }
    for task_name in ("lane", "stop_line", "crosswalk"):
        task_metrics = metrics.get(task_name, {}) if isinstance(metrics.get(task_name), dict) else {}
        for key in (
            "precision",
            "recall",
            "f1",
            "tp",
            "fp",
            "fn",
            "mean_point_distance",
            "mean_angle_error",
            "mean_polygon_iou",
            "mean_vertex_distance",
        ):
            value = task_metrics.get(key)
            if isinstance(value, (int, float)):
                row[f"{task_name}_{key}"] = value
    lane_family = metrics.get("lane_family", {}) if isinstance(metrics.get("lane_family"), dict) else {}
    for key in ("mean_f1", "min_f1"):
        value = lane_family.get(key)
        if isinstance(value, (int, float)):
            row[f"lane_family_{key}"] = value
    row["phase4_objective_proxy"] = (
        0.50 * _metric_value(metrics, "lane", "f1")
        + 0.30 * _metric_value(metrics, "stop_line", "f1")
        + 0.20 * _metric_value(metrics, "crosswalk", "f1")
    )
    return row


def _postprocess_config(base: PV26PostprocessConfig, thresholds: dict[str, float]) -> PV26PostprocessConfig:
    return replace(
        base,
        lane_obj_threshold=float(thresholds["lane"]),
        stop_line_obj_threshold=float(thresholds["stop_line"]),
        crosswalk_obj_threshold=float(thresholds["crosswalk"]),
    )


def _resolve_device(requested: str, scenario_device: str) -> str:
    value = str(requested or "auto").strip().lower()
    if value == "auto":
        candidate = str(scenario_device)
    else:
        candidate = str(requested)
    if candidate.startswith("cuda") and not torch.cuda.is_available():
        print("[threshold_sweep] CUDA requested but unavailable; falling back to CPU", flush=True)
        return "cpu"
    return candidate


def _detach_to_cpu(item: Any) -> Any:
    if isinstance(item, torch.Tensor):
        return item.detach().cpu()
    if isinstance(item, dict):
        return {key: _detach_to_cpu(value) for key, value in item.items()}
    if isinstance(item, list):
        return [_detach_to_cpu(value) for value in item]
    if isinstance(item, tuple):
        return tuple(_detach_to_cpu(value) for value in item)
    return item


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")

    sweep_values = _thresholds(args.thresholds)
    scenario = train_cli.load_meta_train_scenario(args.preset)
    scenario_path = train_cli._default_scenario_path(args.preset)
    phase = scenario.phases[int(args.phase_index) - 1]
    train_config = train_config_api.scenario_phase_defaults(scenario.train_defaults, phase.overrides)
    train_config = replace(
        train_config,
        device=_resolve_device(args.device, train_config.device),
        val_batches=int(args.max_val_batches),
    )
    base_postprocess = train_cli._build_postprocess_config(train_config)

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else checkpoint.parents[2]
        / "analysis_exports"
        / f"lane_family_threshold_sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    train_cli._configure_torch_multiprocessing()
    dataset_roots = train_cli._existing_dataset_roots(scenario)
    dataset = train_cli.PV26CanonicalDataset(
        dataset_roots,
        train_augmentation=False,
        progress_callback=lambda message: print(f"[threshold_sweep] {message}", flush=True),
    )
    _, val_loader = train_cli._build_phase_train_loaders(dataset, train_config=train_config, phase=phase)
    if val_loader is None:
        raise ValueError("threshold sweep requires validation batches")

    trainer = train_cli._build_phase_trainer(phase, train_config)
    trainer.load_model_weights(checkpoint, map_location=train_config.device)
    evaluator = trainer.build_evaluator()
    evaluator.adapter.raw_model.eval()
    evaluator.heads.eval()

    configs: list[tuple[str, str, dict[str, float]]] = []
    base_thresholds = {
        "lane": float(base_postprocess.lane_obj_threshold),
        "stop_line": float(base_postprocess.stop_line_obj_threshold),
        "crosswalk": float(base_postprocess.crosswalk_obj_threshold),
    }
    configs.append(("baseline", "none", dict(base_thresholds)))
    for task_name in ("lane", "stop_line", "crosswalk"):
        for threshold in sweep_values:
            thresholds = dict(base_thresholds)
            thresholds[task_name] = float(threshold)
            configs.append((f"{task_name}_{threshold:.2f}", task_name, thresholds))

    predictions_by_config: dict[str, list[dict[str, Any]]] = {name: [] for name, _, _ in configs}
    raw_batches: list[dict[str, Any]] = []
    evaluated_batches: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch_index, batch in enumerate(val_loader, start=1):
            if batch_index > int(args.max_val_batches):
                break
            print(f"[threshold_sweep] eval batch {batch_index}/{args.max_val_batches}", flush=True)
            raw_batch = raw_batch_for_metrics(batch)
            if raw_batch is None:
                raise ValueError("validation loader must provide raw batches for metrics")
            encoded = evaluator.prepare_batch(batch)
            predictions = _detach_to_cpu(evaluator.forward_encoded_batch(encoded))
            meta = _detach_to_cpu(encoded["meta"])
            raw_batches.append(raw_batch)
            evaluated_batches.append({"raw": raw_batch, "meta": meta, "predictions": predictions})
            for config_name, _, thresholds in configs:
                postprocess_config = _postprocess_config(base_postprocess, thresholds)
                predictions_by_config[config_name].extend(
                    postprocess_pv26_batch(predictions, meta, config=postprocess_config)
                )

    merged_raw = _merge_raw_batches(raw_batches)
    rows: list[dict[str, Any]] = []
    metrics_by_config: dict[str, Any] = {}
    thresholds_by_config = {name: thresholds for name, _, thresholds in configs}
    varied_by_config = {name: varied for name, varied, _ in configs}
    for config_name, _, _ in configs:
        metrics = augment_lane_family_metrics(
            summarize_pv26_metrics(predictions_by_config[config_name], merged_raw)
        )
        metrics_by_config[config_name] = metrics
        rows.append(
            _lane_family_row(
                config_name=config_name,
                varied_task=varied_by_config[config_name],
                thresholds=thresholds_by_config[config_name],
                metrics=metrics,
            )
        )

    best_by_task: dict[str, dict[str, Any]] = {}
    for task_name in ("lane", "stop_line", "crosswalk"):
        task_rows = [row for row in rows if row["varied_task"] == task_name]
        best_by_task[task_name] = max(task_rows, key=lambda row: float(row.get(f"{task_name}_f1", 0.0)))

    combined_thresholds = dict(base_thresholds)
    combined_thresholds["lane"] = float(best_by_task["lane"]["lane_threshold"])
    combined_thresholds["stop_line"] = float(best_by_task["stop_line"]["stop_line_threshold"])
    combined_thresholds["crosswalk"] = float(best_by_task["crosswalk"]["crosswalk_threshold"])
    combined_predictions: list[dict[str, Any]] = []
    postprocess_config = _postprocess_config(base_postprocess, combined_thresholds)
    for batch_index, evaluated_batch in enumerate(evaluated_batches, start=1):
        print(f"[threshold_sweep] combined eval batch {batch_index}/{len(evaluated_batches)}", flush=True)
        combined_predictions.extend(
            postprocess_pv26_batch(
                evaluated_batch["predictions"],
                evaluated_batch["meta"],
                config=postprocess_config,
            )
        )
    combined_metrics = augment_lane_family_metrics(
        summarize_pv26_metrics(combined_predictions, merged_raw)
    )
    rows.append(
        _lane_family_row(
            config_name="combined_task_best",
            varied_task="combined",
            thresholds=combined_thresholds,
            metrics=combined_metrics,
        )
    )
    metrics_by_config["combined_task_best"] = combined_metrics

    rows = sorted(rows, key=lambda row: (str(row["varied_task"]), -float(row.get("phase4_objective_proxy", 0.0))))
    _write_csv(output_dir / "threshold_sweep.csv", rows)
    summary = {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "duration_sec": max(0.0, time.perf_counter() - started),
        "checkpoint": str(checkpoint),
        "scenario_path": str(scenario_path),
        "phase_index": int(args.phase_index),
        "phase_name": phase.name,
        "phase_stage": phase.stage,
        "max_val_batches": int(args.max_val_batches),
        "thresholds": sweep_values,
        "base_postprocess": asdict(base_postprocess),
        "best_by_task": best_by_task,
        "combined_thresholds": combined_thresholds,
        "combined_metrics": combined_metrics,
        "output_files": {
            "csv": str(output_dir / "threshold_sweep.csv"),
            "summary": str(output_dir / "summary.json"),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary["best_by_task"], indent=2, sort_keys=True), flush=True)
    print(f"[threshold_sweep] wrote {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
