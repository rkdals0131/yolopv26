from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import gc
import json
import math
from pathlib import Path
import site
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
repo_root = str(REPO_ROOT)
if repo_root not in sys.path:
    site.addsitedir(repo_root)

import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


_TB_SIZE_GUIDANCE = {
    "compressedHistograms": 0,
    "histograms": 0,
    "images": 0,
    "scalars": 0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze a PV26 meta-train run and export CSV artifacts.")
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Path to the PV26 meta-train run directory.",
    )
    parser.add_argument(
        "--output-dir",
        help="Optional output directory. Defaults to <run-dir>/analysis_exports.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def iter_jsonl(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            yield json.loads(raw)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def phase_sort_key(path: Path) -> int:
    try:
        return int(path.name.split("_")[-1])
    except (IndexError, ValueError):
        return 0


def join_key(prefix: str, key: str) -> str:
    return f"{prefix}.{key}" if prefix else key


def should_skip(prefix: str, skip_prefixes: tuple[str, ...]) -> bool:
    return any(prefix == item or prefix.startswith(f"{item}.") for item in skip_prefixes)


def flatten_for_csv(
    payload: Any,
    *,
    prefix: str = "",
    out: dict[str, Any] | None = None,
    max_list_len: int = 16,
    skip_prefixes: tuple[str, ...] = (),
) -> dict[str, Any]:
    if out is None:
        out = {}
    if prefix and should_skip(prefix, skip_prefixes):
        return out
    if isinstance(payload, dict):
        for key, value in payload.items():
            flatten_for_csv(
                value,
                prefix=join_key(prefix, str(key)),
                out=out,
                max_list_len=max_list_len,
                skip_prefixes=skip_prefixes,
            )
        return out
    if isinstance(payload, (list, tuple)):
        if len(payload) > max_list_len:
            return out
        if not all(isinstance(item, (str, int, float, bool)) or item is None for item in payload):
            return out
        for index, value in enumerate(payload):
            flatten_for_csv(
                value,
                prefix=join_key(prefix, str(index)),
                out=out,
                max_list_len=max_list_len,
                skip_prefixes=skip_prefixes,
            )
        return out
    if payload is None:
        if prefix:
            out[prefix] = ""
        return out
    if isinstance(payload, bool):
        if prefix:
            out[prefix] = int(payload)
        return out
    if isinstance(payload, (int, float)):
        numeric = float(payload)
        if prefix and math.isfinite(numeric):
            out[prefix] = payload
        return out
    if isinstance(payload, str):
        if prefix:
            out[prefix] = payload
    return out


def nested_get(mapping: dict[str, Any], *parts: str, default: Any = "") -> Any:
    current: Any = mapping
    for part in parts:
        if not isinstance(current, dict):
            return default
        current = current.get(part)
    if current is None:
        return default
    return current


def iso_from_timestamp(timestamp: float) -> str:
    return datetime.fromtimestamp(float(timestamp), tz=timezone.utc).isoformat()


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * float(fraction)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    ratio = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * ratio


def summarize_numeric_values(values: list[float]) -> dict[str, Any]:
    finite_values = [float(value) for value in values if math.isfinite(float(value))]
    if not finite_values:
        return {
            "count": 0,
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "p50": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "max": 0.0,
        }
    count = len(finite_values)
    mean = sum(finite_values) / count
    variance = sum((value - mean) ** 2 for value in finite_values) / count
    return {
        "count": count,
        "mean": mean,
        "std": math.sqrt(variance),
        "min": min(finite_values),
        "p50": percentile(finite_values, 0.50),
        "p90": percentile(finite_values, 0.90),
        "p95": percentile(finite_values, 0.95),
        "p99": percentile(finite_values, 0.99),
        "max": max(finite_values),
    }


def iter_histogram_arrays(payload: Any, *, prefix: str = "") -> Any:
    if isinstance(payload, dict):
        for key, value in payload.items():
            next_prefix = join_key(prefix, str(key))
            yield from iter_histogram_arrays(value, prefix=next_prefix)
        return
    if isinstance(payload, list):
        if all(isinstance(item, (int, float)) and math.isfinite(float(item)) for item in payload):
            yield prefix, [float(item) for item in payload]


def write_csv(path: Path, rows: list[dict[str, Any]], *, fieldnames: list[str] | None = None) -> int:
    ensure_dir(path.parent)
    if fieldnames is None:
        ordered_fields: list[str] = []
        seen: set[str] = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    ordered_fields.append(key)
                    seen.add(key)
        fieldnames = ordered_fields
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return len(rows)


def write_combined_flattened_csv(
    *,
    sources: list[dict[str, Any]],
    output_path: Path,
    skip_prefixes: tuple[str, ...] = (),
    max_list_len: int = 16,
) -> int:
    fieldnames: list[str] = []
    seen: set[str] = set()
    row_count = 0
    for source in sources:
        for payload in iter_jsonl(source["path"]):
            row = dict(source["context"])
            row.update(
                flatten_for_csv(
                    payload,
                    max_list_len=max_list_len,
                    skip_prefixes=skip_prefixes,
                )
            )
            for key in row:
                if key not in seen:
                    fieldnames.append(key)
                    seen.add(key)
            row_count += 1
    ensure_dir(output_path.parent)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for source in sources:
            for payload in iter_jsonl(source["path"]):
                row = dict(source["context"])
                row.update(
                    flatten_for_csv(
                        payload,
                        max_list_len=max_list_len,
                        skip_prefixes=skip_prefixes,
                    )
                )
                writer.writerow(row)
    return row_count


def build_phase_infos(run_dir: Path, top_summary: dict[str, Any]) -> list[dict[str, Any]]:
    top_phase_entries = {
        Path(entry["run_dir"]).resolve(): entry
        for entry in top_summary.get("phases", [])
        if isinstance(entry, dict) and entry.get("run_dir")
    }
    phase_infos: list[dict[str, Any]] = []
    for phase_dir in sorted(run_dir.glob("phase_*"), key=phase_sort_key):
        phase_summary = read_json(phase_dir / "summary.json")
        top_entry = top_phase_entries.get(phase_dir.resolve(), {})
        phase_infos.append(
            {
                "phase_index": phase_sort_key(phase_dir),
                "phase_name": top_entry.get("name", phase_dir.name),
                "stage": top_entry.get("stage", phase_summary.get("stage", phase_dir.name)),
                "phase_dir": phase_dir,
                "top_entry": top_entry,
                "phase_summary": phase_summary,
            }
        )
    return phase_infos


def build_phase_overview_rows(run_dir: Path, phase_infos: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for phase in phase_infos:
        epochs = list(iter_jsonl(phase["phase_dir"] / "history" / "epochs.jsonl"))
        first_step = next(iter_jsonl(phase["phase_dir"] / "history" / "train_steps.jsonl"))
        first_epoch = epochs[0]
        last_epoch = epochs[-1]
        best_epoch = max(
            epochs,
            key=lambda item: float(nested_get(item, "selection_metrics", "phase_objective", default=0.0) or 0.0),
        )
        top_entry = phase["top_entry"]
        cfg = top_entry.get("phase_train_config", {})
        total_train_duration = sum(float(nested_get(item, "train", "duration_sec", default=0.0) or 0.0) for item in epochs)
        total_val_duration = sum(float(nested_get(item, "val", "duration_sec", default=0.0) or 0.0) for item in epochs)
        phase_source_totals: dict[str, int] = {}
        for item in epochs:
            for key, value in nested_get(item, "train", "source_counts", default={}).items():
                phase_source_totals[key] = phase_source_totals.get(key, 0) + int(value)
        row = {
            "phase_index": phase["phase_index"],
            "phase_name": phase["phase_name"],
            "stage": phase["stage"],
            "run_dir": str(phase["phase_dir"].relative_to(run_dir)),
            "source_file": str((phase["phase_dir"] / "summary.json").relative_to(run_dir)),
            "status": top_entry.get("status", ""),
            "completed_epochs": top_entry.get("completed_epochs", len(epochs)),
            "best_epoch": top_entry.get("best_epoch", best_epoch.get("epoch")),
            "best_metric_value": top_entry.get(
                "best_metric_value",
                nested_get(best_epoch, "selection_metrics", "phase_objective", default=0.0),
            ),
            "promotion_reason": top_entry.get("promotion_reason", ""),
            "selection_metric_path": nested_get(top_entry, "selection", "metric_path", default=""),
            "batch_size": cfg.get("batch_size", first_step.get("batch_size", "")),
            "train_batches": cfg.get("train_batches", ""),
            "val_batches": cfg.get("val_batches", ""),
            "trunk_lr_config": cfg.get("trunk_lr", ""),
            "head_lr_config": cfg.get("head_lr", ""),
            "schedule": cfg.get("schedule", ""),
            "freeze_policy": nested_get(first_step, "trainable", "freeze_policy", default=""),
            "head_training_policy": nested_get(first_step, "trainable", "head_training_policy", default=""),
            "trainable_trunk_params": nested_get(first_step, "trainable", "trainable_trunk_params", default=0),
            "trainable_head_params": nested_get(first_step, "trainable", "trainable_head_params", default=0),
            "trainable_det_head_params": nested_get(first_step, "trainable", "trainable_det_head_params", default=0),
            "trainable_tl_attr_head_params": nested_get(first_step, "trainable", "trainable_tl_attr_head_params", default=0),
            "trainable_lane_family_head_params": nested_get(first_step, "trainable", "trainable_lane_family_head_params", default=0),
            "phase_objective_first": nested_get(first_epoch, "selection_metrics", "phase_objective", default=0.0),
            "phase_objective_last": nested_get(last_epoch, "selection_metrics", "phase_objective", default=0.0),
            "phase_objective_best": nested_get(best_epoch, "selection_metrics", "phase_objective", default=0.0),
            "phase_objective_best_epoch": best_epoch.get("epoch", ""),
            "train_loss_total_mean_first": nested_get(first_epoch, "train", "losses", "total", "mean", default=0.0),
            "train_loss_total_mean_last": nested_get(last_epoch, "train", "losses", "total", "mean", default=0.0),
            "val_loss_total_mean_first": nested_get(first_epoch, "val", "losses", "total", "mean", default=0.0),
            "val_loss_total_mean_last": nested_get(last_epoch, "val", "losses", "total", "mean", default=0.0),
            "val_loss_total_mean_min": min(
                float(nested_get(item, "val", "losses", "total", "mean", default=0.0) or 0.0) for item in epochs
            ),
            "val_detector_map50_best": max(
                float(nested_get(item, "val", "metrics", "detector", "map50", default=0.0) or 0.0) for item in epochs
            ),
            "val_lane_f1_best": max(
                float(nested_get(item, "val", "metrics", "lane", "f1", default=0.0) or 0.0) for item in epochs
            ),
            "val_stop_line_f1_best": max(
                float(nested_get(item, "val", "metrics", "stop_line", "f1", default=0.0) or 0.0) for item in epochs
            ),
            "val_crosswalk_iou_best": max(
                float(nested_get(item, "val", "metrics", "crosswalk", "mean_polygon_iou", default=0.0) or 0.0)
                for item in epochs
            ),
            "lr_heads_first": nested_get(first_epoch, "train", "optimizer_lrs", "heads", default=""),
            "lr_heads_last": nested_get(last_epoch, "train", "optimizer_lrs", "heads", default=""),
            "lr_trunk_first": nested_get(first_epoch, "train", "optimizer_lrs", "trunk", default=""),
            "lr_trunk_last": nested_get(last_epoch, "train", "optimizer_lrs", "trunk", default=""),
            "train_duration_sec_total": total_train_duration,
            "val_duration_sec_total": total_val_duration,
            "train_batches_total": sum(int(nested_get(item, "train", "batches", default=0) or 0) for item in epochs),
            "val_batches_total": sum(int(nested_get(item, "val", "batches", default=0) or 0) for item in epochs),
            "global_step_end": nested_get(last_epoch, "train", "global_step_end", default=0),
            "best_checkpoint_path": top_entry.get("best_checkpoint_path", ""),
            "last_checkpoint_path": top_entry.get("last_checkpoint_path", ""),
        }
        for key, value in sorted(phase_source_totals.items()):
            row[f"phase_total_{key}"] = value
        rows.append(row)
    return rows


def build_epoch_kpi_rows(run_dir: Path, phase_infos: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for phase in phase_infos:
        path = phase["phase_dir"] / "history" / "epochs.jsonl"
        for payload in iter_jsonl(path):
            row = {
                "phase_index": phase["phase_index"],
                "phase_name": phase["phase_name"],
                "stage": phase["stage"],
                "source_file": str(path.relative_to(run_dir)),
                "epoch": payload.get("epoch", ""),
                "epoch_started_at": payload.get("epoch_started_at", ""),
                "global_step_start": nested_get(payload, "train", "global_step_start", default=0),
                "global_step_end": nested_get(payload, "train", "global_step_end", default=0),
                "selection_phase_objective": nested_get(payload, "selection_metrics", "phase_objective", default=0.0),
                "selection_detector_score": nested_get(
                    payload, "selection_metrics", "components", "detector", "score", default=0.0
                ),
                "selection_traffic_light_score": nested_get(
                    payload, "selection_metrics", "components", "traffic_light", "score", default=0.0
                ),
                "selection_lane_score": nested_get(payload, "selection_metrics", "components", "lane", "score", default=0.0),
                "selection_stop_line_score": nested_get(
                    payload, "selection_metrics", "components", "stop_line", "score", default=0.0
                ),
                "selection_crosswalk_score": nested_get(
                    payload, "selection_metrics", "components", "crosswalk", "score", default=0.0
                ),
                "selection_lane_support": nested_get(
                    payload, "selection_metrics", "components", "lane", "support", default=0
                ),
                "selection_stop_line_support": nested_get(
                    payload, "selection_metrics", "components", "stop_line", "support", default=0
                ),
                "selection_crosswalk_support": nested_get(
                    payload, "selection_metrics", "components", "crosswalk", "support", default=0
                ),
                "plateau_count": nested_get(payload, "phase_transition", "plateau_count", default=0),
                "transition_eligible": nested_get(payload, "phase_transition", "transition_eligible", default=0),
                "lr_heads": nested_get(payload, "train", "optimizer_lrs", "heads", default=""),
                "lr_trunk": nested_get(payload, "train", "optimizer_lrs", "trunk", default=""),
                "train_loss_total_mean": nested_get(payload, "train", "losses", "total", "mean", default=0.0),
                "train_loss_total_last": nested_get(payload, "train", "losses", "total", "last", default=0.0),
                "train_loss_weighted_lane_mean": nested_get(
                    payload, "train", "losses", "weighted", "lane", "mean", default=0.0
                ),
                "train_loss_weighted_stop_line_mean": nested_get(
                    payload, "train", "losses", "weighted", "stop_line", "mean", default=0.0
                ),
                "train_loss_weighted_crosswalk_mean": nested_get(
                    payload, "train", "losses", "weighted", "crosswalk", "mean", default=0.0
                ),
                "val_loss_total_mean": nested_get(payload, "val", "losses", "total", "mean", default=0.0),
                "val_loss_weighted_lane_mean": nested_get(
                    payload, "val", "losses", "weighted", "lane", "mean", default=0.0
                ),
                "val_loss_weighted_stop_line_mean": nested_get(
                    payload, "val", "losses", "weighted", "stop_line", "mean", default=0.0
                ),
                "val_loss_weighted_crosswalk_mean": nested_get(
                    payload, "val", "losses", "weighted", "crosswalk", "mean", default=0.0
                ),
                "val_detector_map50": nested_get(payload, "val", "metrics", "detector", "map50", default=0.0),
                "val_detector_map50_95": nested_get(payload, "val", "metrics", "detector", "map50_95", default=0.0),
                "val_detector_f1": nested_get(payload, "val", "metrics", "detector", "f1", default=0.0),
                "val_detector_tp": nested_get(payload, "val", "metrics", "detector", "tp", default=0),
                "val_detector_fp": nested_get(payload, "val", "metrics", "detector", "fp", default=0),
                "val_detector_fn": nested_get(payload, "val", "metrics", "detector", "fn", default=0),
                "val_tl_combo_accuracy": nested_get(
                    payload, "val", "metrics", "traffic_light", "combo_accuracy", default=0.0
                ),
                "val_tl_mean_f1": nested_get(payload, "val", "metrics", "traffic_light", "mean_f1", default=0.0),
                "val_lane_precision": nested_get(payload, "val", "metrics", "lane", "precision", default=0.0),
                "val_lane_recall": nested_get(payload, "val", "metrics", "lane", "recall", default=0.0),
                "val_lane_f1": nested_get(payload, "val", "metrics", "lane", "f1", default=0.0),
                "val_lane_tp": nested_get(payload, "val", "metrics", "lane", "tp", default=0),
                "val_lane_fp": nested_get(payload, "val", "metrics", "lane", "fp", default=0),
                "val_lane_fn": nested_get(payload, "val", "metrics", "lane", "fn", default=0),
                "val_lane_mean_point_distance": nested_get(
                    payload, "val", "metrics", "lane", "mean_point_distance", default=0.0
                ),
                "val_lane_color_accuracy": nested_get(
                    payload, "val", "metrics", "lane", "color_accuracy", default=0.0
                ),
                "val_lane_type_accuracy": nested_get(payload, "val", "metrics", "lane", "type_accuracy", default=0.0),
                "val_stop_line_precision": nested_get(
                    payload, "val", "metrics", "stop_line", "precision", default=0.0
                ),
                "val_stop_line_recall": nested_get(payload, "val", "metrics", "stop_line", "recall", default=0.0),
                "val_stop_line_f1": nested_get(payload, "val", "metrics", "stop_line", "f1", default=0.0),
                "val_stop_line_tp": nested_get(payload, "val", "metrics", "stop_line", "tp", default=0),
                "val_stop_line_fp": nested_get(payload, "val", "metrics", "stop_line", "fp", default=0),
                "val_stop_line_fn": nested_get(payload, "val", "metrics", "stop_line", "fn", default=0),
                "val_stop_line_mean_point_distance": nested_get(
                    payload, "val", "metrics", "stop_line", "mean_point_distance", default=0.0
                ),
                "val_stop_line_mean_angle_error": nested_get(
                    payload, "val", "metrics", "stop_line", "mean_angle_error", default=0.0
                ),
                "val_crosswalk_precision": nested_get(
                    payload, "val", "metrics", "crosswalk", "precision", default=0.0
                ),
                "val_crosswalk_recall": nested_get(payload, "val", "metrics", "crosswalk", "recall", default=0.0),
                "val_crosswalk_f1": nested_get(payload, "val", "metrics", "crosswalk", "f1", default=0.0),
                "val_crosswalk_tp": nested_get(payload, "val", "metrics", "crosswalk", "tp", default=0),
                "val_crosswalk_fp": nested_get(payload, "val", "metrics", "crosswalk", "fp", default=0),
                "val_crosswalk_fn": nested_get(payload, "val", "metrics", "crosswalk", "fn", default=0),
                "val_crosswalk_mean_polygon_iou": nested_get(
                    payload, "val", "metrics", "crosswalk", "mean_polygon_iou", default=0.0
                ),
                "val_crosswalk_mean_vertex_distance": nested_get(
                    payload, "val", "metrics", "crosswalk", "mean_vertex_distance", default=0.0
                ),
                "val_lane_family_mean_f1": nested_get(
                    payload, "val", "metrics", "lane_family", "mean_f1", default=0.0
                ),
                "val_lane_family_min_f1": nested_get(payload, "val", "metrics", "lane_family", "min_f1", default=0.0),
                "val_det_gt": nested_get(payload, "val", "counts", "det_gt", default=0),
                "val_tl_attr_gt": nested_get(payload, "val", "counts", "tl_attr_gt", default=0),
                "val_lane_rows": nested_get(payload, "val", "counts", "lane_rows", default=0),
                "val_stop_line_rows": nested_get(payload, "val", "counts", "stop_line_rows", default=0),
                "val_crosswalk_rows": nested_get(payload, "val", "counts", "crosswalk_rows", default=0),
                "train_det_source_samples": nested_get(
                    payload, "train", "source_counts", "det_source_samples", default=0
                ),
                "train_tl_attr_source_samples": nested_get(
                    payload, "train", "source_counts", "tl_attr_source_samples", default=0
                ),
                "train_lane_source_samples": nested_get(
                    payload, "train", "source_counts", "lane_source_samples", default=0
                ),
                "train_stop_line_source_samples": nested_get(
                    payload, "train", "source_counts", "stop_line_source_samples", default=0
                ),
                "train_crosswalk_source_samples": nested_get(
                    payload, "train", "source_counts", "crosswalk_source_samples", default=0
                ),
                "train_duration_sec": nested_get(payload, "train", "duration_sec", default=0.0),
                "val_duration_sec": nested_get(payload, "val", "duration_sec", default=0.0),
            }
            rows.append(row)
    return rows


def build_epoch_histogram_rows(run_dir: Path, phase_infos: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for phase in phase_infos:
        path = phase["phase_dir"] / "history" / "epochs.jsonl"
        for payload in iter_jsonl(path):
            histogram_payload = nested_get(payload, "val", "tensorboard_histograms", default={})
            for tag, values in iter_histogram_arrays(histogram_payload, prefix="val.tensorboard_histograms"):
                summary = summarize_numeric_values(values)
                row = {
                    "phase_index": phase["phase_index"],
                    "phase_name": phase["phase_name"],
                    "stage": phase["stage"],
                    "source_file": str(path.relative_to(run_dir)),
                    "epoch": payload.get("epoch", ""),
                    "tag": tag,
                }
                row.update(summary)
                rows.append(row)
    return rows


def export_tensorboard_scalars(run_dir: Path, phase_infos: list[dict[str, Any]], output_path: Path) -> int:
    ensure_dir(output_path.parent)
    fieldnames = [
        "phase_index",
        "phase_name",
        "stage",
        "source_file",
        "tag",
        "step",
        "wall_time",
        "wall_time_iso",
        "value",
    ]
    row_count = 0
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for phase in phase_infos:
            tensorboard_dir = phase["phase_dir"] / "tensorboard"
            if not tensorboard_dir.is_dir():
                continue
            accumulator = EventAccumulator(str(tensorboard_dir), size_guidance=_TB_SIZE_GUIDANCE)
            accumulator.Reload()
            relative_dir = str(tensorboard_dir.relative_to(run_dir))
            for tag in accumulator.Tags().get("scalars", []):
                for event in accumulator.Scalars(tag):
                    writer.writerow(
                        {
                            "phase_index": phase["phase_index"],
                            "phase_name": phase["phase_name"],
                            "stage": phase["stage"],
                            "source_file": relative_dir,
                            "tag": tag,
                            "step": event.step,
                            "wall_time": event.wall_time,
                            "wall_time_iso": iso_from_timestamp(event.wall_time),
                            "value": event.value,
                        }
                    )
                    row_count += 1
    return row_count


def export_tensorboard_histograms(run_dir: Path, phase_infos: list[dict[str, Any]], output_path: Path) -> int:
    ensure_dir(output_path.parent)
    fieldnames = [
        "phase_index",
        "phase_name",
        "stage",
        "source_file",
        "tag",
        "step",
        "wall_time",
        "wall_time_iso",
        "min",
        "max",
        "num",
        "sum",
        "sum_squares",
    ]
    row_count = 0
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for phase in phase_infos:
            tensorboard_dir = phase["phase_dir"] / "tensorboard"
            if not tensorboard_dir.is_dir():
                continue
            accumulator = EventAccumulator(str(tensorboard_dir), size_guidance=_TB_SIZE_GUIDANCE)
            accumulator.Reload()
            relative_dir = str(tensorboard_dir.relative_to(run_dir))
            for tag in accumulator.Tags().get("histograms", []):
                for event in accumulator.Histograms(tag):
                    histogram = event.histogram_value
                    writer.writerow(
                        {
                            "phase_index": phase["phase_index"],
                            "phase_name": phase["phase_name"],
                            "stage": phase["stage"],
                            "source_file": relative_dir,
                            "tag": tag,
                            "step": event.step,
                            "wall_time": event.wall_time,
                            "wall_time_iso": iso_from_timestamp(event.wall_time),
                            "min": histogram.min,
                            "max": histogram.max,
                            "num": histogram.num,
                            "sum": histogram.sum,
                            "sum_squares": histogram.sum_squares,
                        }
                    )
                    row_count += 1
    return row_count


def build_checkpoint_rows(run_dir: Path, phase_infos: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for phase in phase_infos:
        for checkpoint_path in sorted((phase["phase_dir"] / "checkpoints").glob("*.pt")):
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            metadata = checkpoint.get("checkpoint_metadata", {})
            stage_summary = checkpoint.get("stage_summary", {})
            row = {
                "phase_index": phase["phase_index"],
                "phase_name": phase["phase_name"],
                "stage": checkpoint.get("stage", phase["stage"]),
                "source_file": str(checkpoint_path.relative_to(run_dir)),
                "checkpoint_name": checkpoint_path.name,
                "global_step": checkpoint.get("global_step", 0),
                "history_len": len(checkpoint.get("history", [])),
                "epoch_history_len": len(checkpoint.get("epoch_history", [])),
            }
            row.update(flatten_for_csv(metadata, prefix="checkpoint_metadata"))
            row.update(flatten_for_csv(stage_summary, prefix="stage_summary"))
            rows.append(row)
            del checkpoint
            gc.collect()
    return rows


def build_analysis_manifest(
    *,
    run_dir: Path,
    output_dir: Path,
    generated_files: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    ai_input_order = [
        "phase_overview.csv",
        "epoch_kpis.csv",
        "checkpoints.csv",
        "tensorboard_scalars.csv",
        "epoch_histogram_summary.csv",
        "train_steps_scalar.csv",
    ]
    return {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "generated_files": generated_files,
        "ai_input_order": ai_input_order,
        "recommended_prompt": (
            "Attached are PV26 run analysis exports. First use phase_overview.csv and epoch_kpis.csv "
            "to identify where performance stalls or collapses. Then cross-check checkpoints.csv and "
            "tensorboard_scalars.csv to separate optimization progress from validation collapse. "
            "Finally use epoch_histogram_summary.csv and train_steps_scalar.csv to propose concrete "
            "fixes for data balance, loss weighting, stage schedule, and selection metric design."
        ),
    }


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else run_dir / "analysis_exports"
    ensure_dir(output_dir)

    top_summary = read_json(run_dir / "summary.json")
    phase_infos = build_phase_infos(run_dir, top_summary)
    if not phase_infos:
        raise SystemExit(f"No phase directories found under {run_dir}")

    phase_overview_rows = build_phase_overview_rows(run_dir, phase_infos)
    epoch_kpi_rows = build_epoch_kpi_rows(run_dir, phase_infos)
    epoch_histogram_rows = build_epoch_histogram_rows(run_dir, phase_infos)
    checkpoint_rows = build_checkpoint_rows(run_dir, phase_infos)

    generated_files: dict[str, dict[str, Any]] = {}

    phase_overview_path = output_dir / "phase_overview.csv"
    generated_files[phase_overview_path.name] = {
        "path": str(phase_overview_path),
        "row_count": write_csv(phase_overview_path, phase_overview_rows),
    }

    epoch_kpis_path = output_dir / "epoch_kpis.csv"
    generated_files[epoch_kpis_path.name] = {
        "path": str(epoch_kpis_path),
        "row_count": write_csv(epoch_kpis_path, epoch_kpi_rows),
    }

    epochs_scalar_path = output_dir / "epochs_scalar.csv"
    epoch_scalar_sources = [
        {
            "path": phase["phase_dir"] / "history" / "epochs.jsonl",
            "context": {
                "phase_index": phase["phase_index"],
                "phase_name": phase["phase_name"],
                "stage": phase["stage"],
                "source_file": str((phase["phase_dir"] / "history" / "epochs.jsonl").relative_to(run_dir)),
            },
        }
        for phase in phase_infos
    ]
    generated_files[epochs_scalar_path.name] = {
        "path": str(epochs_scalar_path),
        "row_count": write_combined_flattened_csv(
            sources=epoch_scalar_sources,
            output_path=epochs_scalar_path,
            skip_prefixes=("val.tensorboard_histograms",),
        ),
    }

    epoch_histogram_summary_path = output_dir / "epoch_histogram_summary.csv"
    generated_files[epoch_histogram_summary_path.name] = {
        "path": str(epoch_histogram_summary_path),
        "row_count": write_csv(epoch_histogram_summary_path, epoch_histogram_rows),
    }

    train_steps_scalar_path = output_dir / "train_steps_scalar.csv"
    step_sources = [
        {
            "path": phase["phase_dir"] / "history" / "train_steps.jsonl",
            "context": {
                "phase_index": phase["phase_index"],
                "phase_name": phase["phase_name"],
                "stage": phase["stage"],
                "source_file": str((phase["phase_dir"] / "history" / "train_steps.jsonl").relative_to(run_dir)),
            },
        }
        for phase in phase_infos
    ]
    generated_files[train_steps_scalar_path.name] = {
        "path": str(train_steps_scalar_path),
        "row_count": write_combined_flattened_csv(
            sources=step_sources,
            output_path=train_steps_scalar_path,
        ),
    }

    tensorboard_scalars_path = output_dir / "tensorboard_scalars.csv"
    generated_files[tensorboard_scalars_path.name] = {
        "path": str(tensorboard_scalars_path),
        "row_count": export_tensorboard_scalars(run_dir, phase_infos, tensorboard_scalars_path),
    }

    tensorboard_histograms_path = output_dir / "tensorboard_histograms.csv"
    generated_files[tensorboard_histograms_path.name] = {
        "path": str(tensorboard_histograms_path),
        "row_count": export_tensorboard_histograms(run_dir, phase_infos, tensorboard_histograms_path),
    }

    checkpoints_path = output_dir / "checkpoints.csv"
    generated_files[checkpoints_path.name] = {
        "path": str(checkpoints_path),
        "row_count": write_csv(checkpoints_path, checkpoint_rows),
    }

    manifest_path = output_dir / "analysis_manifest.json"
    generated_files[manifest_path.name] = {
        "path": str(manifest_path),
        "row_count": 1,
    }
    manifest = build_analysis_manifest(
        run_dir=run_dir,
        output_dir=output_dir,
        generated_files=generated_files,
    )
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
