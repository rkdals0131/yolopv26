from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import site
import sys
import time
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
repo_root = str(REPO_ROOT)
if repo_root not in sys.path:
    site.addsitedir(repo_root)

from model.engine import augment_lane_family_metrics, raw_batch_for_metrics, summarize_pv26_metrics
from model.engine._trainer_epochs import _merge_raw_batches
from model.engine.postprocess import PV26PostprocessConfig, postprocess_pv26_batch
from tools.probe_pv26_lane60_support_gate import _load_scenario, _selection_metrics, _write_csv
from tools.pv26_train import cli as train_cli
from tools.pv26_train import config as train_config_api


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate PV26 lane60 postprocess threshold variants for one checkpoint."
    )
    parser.add_argument("--checkpoint", required=True, help="Checkpoint to evaluate.")
    parser.add_argument("--preset", default="default", help="PV26 meta-train preset.")
    parser.add_argument("--source-run", default="", help="Optional source run used to reproduce lane60 probe validation config.")
    parser.add_argument("--lane60-experiment", default="", help="Optional run_pv26_lane60_probe experiment config to reuse.")
    parser.add_argument("--phase-index", type=int, default=4, help="1-based phase index used for validation config.")
    parser.add_argument("--max-val-batches", type=int, default=128)
    parser.add_argument(
        "--validation-epoch",
        type=int,
        default=1,
        help="One-based validation epoch subset to reproduce. Epoch 2 skips one sampler pass before evaluation.",
    )
    parser.add_argument("--train-batches", type=int, default=512, help="Only used with --lane60-experiment.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", default="")
    parser.add_argument(
        "--variants",
        default="",
        help="Optional comma-separated variant names to evaluate, preserving the provided order.",
    )
    return parser.parse_args()


def _resolve_device(requested: str, scenario_device: str) -> str:
    value = str(requested or "auto").strip().lower()
    device = str(scenario_device) if value == "auto" else str(requested)
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[thresholds] CUDA requested but unavailable; falling back to CPU", flush=True)
        return "cpu"
    return device


def _threshold_variants(base: PV26PostprocessConfig) -> list[tuple[str, PV26PostprocessConfig]]:
    variants: list[tuple[str, PV26PostprocessConfig]] = [("baseline", base)]
    for threshold in (0.30, 0.35, 0.40, 0.45, 0.55, 0.60, 0.65, 0.70):
        variants.append((f"lane_obj_{threshold:.2f}", replace(base, lane_obj_threshold=threshold)))
    for min_area in (0.0, 1024.0, 2048.0, 4096.0, 8192.0):
        variants.append((f"lane_bbox_area_{min_area:.0f}", replace(base, lane_segfirst_min_bbox_area_px=min_area)))
    for max_aspect in (0.0, 4.0, 6.0, 8.0, 12.0):
        variants.append((f"lane_max_aspect_{max_aspect:.1f}", replace(base, lane_segfirst_max_bbox_aspect=max_aspect)))
    for min_area in (2048.0, 4096.0, 8192.0):
        for max_aspect in (4.0, 6.0, 8.0):
            variants.append(
                (
                    f"lane_bbox_area_{min_area:.0f}__max_aspect_{max_aspect:.1f}",
                    replace(
                        base,
                        lane_segfirst_min_bbox_area_px=min_area,
                        lane_segfirst_max_bbox_aspect=max_aspect,
                    ),
                )
            )
    for threshold in (0.20, 0.30, 0.40, 0.60, 0.70):
        variants.append((f"stop_obj_{threshold:.2f}", replace(base, stop_line_obj_threshold=threshold)))
        variants.append((f"cross_obj_{threshold:.2f}", replace(base, crosswalk_obj_threshold=threshold)))
    for threshold in (0.20, 0.30, 0.40, 0.60, 0.70, 0.80):
        variants.append((f"stop_mask_{threshold:.2f}", replace(base, stop_line_mask_binary_threshold=threshold)))
        variants.append((f"cross_mask_{threshold:.2f}", replace(base, crosswalk_mask_binary_threshold=threshold)))
    for min_pixels in (4, 8, 12, 16, 24, 32):
        variants.append((f"stop_area_{min_pixels}", replace(base, stop_line_min_component_pixels=min_pixels)))
    for max_components in (1, 2, 3):
        variants.append((f"stop_topk_{max_components}", replace(base, stop_line_max_components=max_components)))
    for min_area in (0.0, 32.0, 64.0, 128.0, 256.0):
        variants.append((f"stop_bbox_area_{min_area:.0f}", replace(base, stop_line_min_bbox_area_px=min_area)))
    for min_aspect in (0.0, 4.0, 6.0, 8.0, 10.0):
        variants.append((f"stop_aspect_{min_aspect:.1f}", replace(base, stop_line_min_bbox_aspect=min_aspect)))
    for min_score in (0.0, 0.90, 0.92, 0.94, 0.96):
        variants.append((f"stop_score_{min_score:.2f}", replace(base, stop_line_min_instance_score=min_score)))
    for min_area in (0.0, 64.0, 128.0):
        for min_aspect in (4.0, 6.0, 8.0):
            for min_score in (0.90, 0.94, 0.96):
                variants.append(
                    (
                        f"stop_bbox_area_{min_area:.0f}__aspect_{min_aspect:.1f}__score_{min_score:.2f}",
                        replace(
                            base,
                            stop_line_min_bbox_area_px=min_area,
                            stop_line_min_bbox_aspect=min_aspect,
                            stop_line_min_instance_score=min_score,
                        ),
                    )
                )
    for min_pixels in (8, 12, 16, 24):
        for max_components in (1, 2):
            variants.append(
                (
                    f"stop_area_{min_pixels}__topk_{max_components}",
                    replace(
                        base,
                        stop_line_min_component_pixels=min_pixels,
                        stop_line_max_components=max_components,
                    ),
                )
            )
    for min_pixels in (8, 12, 16, 20, 24, 28, 32, 36, 40, 48, 64):
        variants.append((f"cross_area_{min_pixels}", replace(base, crosswalk_min_component_pixels=min_pixels)))
    for polygon_area in (0.0, 256.0, 384.0, 512.0, 640.0, 768.0, 1024.0):
        variants.append(
            (
                f"cross_poly_area_{polygon_area:.0f}",
                replace(base, crosswalk_min_polygon_area_px=polygon_area),
            )
        )
    for min_aspect in (0.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0):
        variants.append(
            (
                f"cross_aspect_{min_aspect:.1f}",
                replace(base, crosswalk_min_bbox_aspect=min_aspect),
            )
        )
    for polygon_area in (384.0, 512.0, 640.0):
        for min_aspect in (2.5, 3.0, 4.0):
            variants.append(
                (
                    f"cross_poly_area_{polygon_area:.0f}__aspect_{min_aspect:.1f}",
                    replace(
                        base,
                        crosswalk_min_polygon_area_px=polygon_area,
                        crosswalk_min_bbox_aspect=min_aspect,
                    ),
                )
            )
    for max_components in (1, 2, 3, 4, 6, 8):
        variants.append((f"cross_topk_{max_components}", replace(base, crosswalk_max_components=max_components)))
    for min_pixels in (8, 16, 32):
        for max_components in (1, 2, 3, 4):
            variants.append(
                (
                    f"cross_area_{min_pixels}__topk_{max_components}",
                    replace(
                        base,
                        crosswalk_min_component_pixels=min_pixels,
                        crosswalk_max_components=max_components,
                    ),
                )
            )
    for lane_threshold in (0.35, 0.40, 0.45, 0.50):
        for cross_threshold in (0.20, 0.30, 0.40):
            for min_pixels in (16, 20, 24, 28, 32):
                variants.append(
                    (
                        f"lane_obj_{lane_threshold:.2f}__cross_mask_{cross_threshold:.2f}__cross_area_{min_pixels}",
                        replace(
                            base,
                            lane_obj_threshold=lane_threshold,
                            crosswalk_mask_binary_threshold=cross_threshold,
                            crosswalk_min_component_pixels=min_pixels,
                        ),
                    )
                )
    for stop_threshold in (0.30, 0.40, 0.50, 0.60, 0.70):
        for cross_threshold in (0.30, 0.40, 0.50, 0.60, 0.70):
            variants.append(
                (
                    f"stop_mask_{stop_threshold:.2f}__cross_mask_{cross_threshold:.2f}",
                    replace(
                        base,
                        stop_line_mask_binary_threshold=stop_threshold,
                        crosswalk_mask_binary_threshold=cross_threshold,
                    ),
                )
            )
    return variants


def _filter_variants(
    variants: list[tuple[str, PV26PostprocessConfig]],
    requested_csv: str,
) -> list[tuple[str, PV26PostprocessConfig]]:
    requested = [name.strip() for name in str(requested_csv or "").split(",") if name.strip()]
    if not requested:
        return variants
    by_name = {name: (name, config) for name, config in variants}
    missing = [name for name in requested if name not in by_name]
    if missing:
        available = ", ".join(sorted(by_name)[:20])
        raise ValueError(
            f"unknown threshold variant(s): {', '.join(missing)}; "
            f"first available variants: {available}"
        )
    return [by_name[name] for name in requested]


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


def _row(name: str, config: PV26PostprocessConfig, metrics: dict[str, Any], selection: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "variant": name,
        "phase_objective": selection["phase_objective"],
        "lane_obj_threshold": config.lane_obj_threshold,
        "lane_segfirst_min_bbox_area_px": config.lane_segfirst_min_bbox_area_px,
        "lane_segfirst_max_bbox_aspect": config.lane_segfirst_max_bbox_aspect,
        "stop_line_obj_threshold": config.stop_line_obj_threshold,
        "stop_line_mask_binary_threshold": config.stop_line_mask_binary_threshold,
        "stop_line_min_component_pixels": config.stop_line_min_component_pixels,
        "stop_line_max_components": config.stop_line_max_components,
        "stop_line_min_bbox_area_px": config.stop_line_min_bbox_area_px,
        "stop_line_min_bbox_aspect": config.stop_line_min_bbox_aspect,
        "stop_line_min_instance_score": config.stop_line_min_instance_score,
        "crosswalk_obj_threshold": config.crosswalk_obj_threshold,
        "crosswalk_mask_binary_threshold": config.crosswalk_mask_binary_threshold,
        "crosswalk_min_component_pixels": config.crosswalk_min_component_pixels,
        "crosswalk_max_components": config.crosswalk_max_components,
        "crosswalk_min_polygon_area_px": config.crosswalk_min_polygon_area_px,
        "crosswalk_min_bbox_aspect": config.crosswalk_min_bbox_aspect,
    }
    for task in ("lane", "stop_line", "crosswalk"):
        task_metrics = metrics.get(task, {}) if isinstance(metrics.get(task), dict) else {}
        component = selection["components"].get(task, {})
        row[f"{task}_f1"] = task_metrics.get("f1", 0.0)
        row[f"{task}_precision"] = task_metrics.get("precision", 0.0)
        row[f"{task}_recall"] = task_metrics.get("recall", 0.0)
        row[f"{task}_tp"] = task_metrics.get("tp", 0)
        row[f"{task}_fp"] = task_metrics.get("fp", 0)
        row[f"{task}_fn"] = task_metrics.get("fn", 0)
        for metric_name in ("mean_point_distance", "mean_angle_error", "mean_polygon_iou", "mean_vertex_distance"):
            if metric_name in task_metrics:
                row[f"{task}_{metric_name}"] = task_metrics.get(metric_name, 0.0)
        row[f"{task}_score"] = component.get("score", 0.0)
        row[f"{task}_support"] = component.get("support", 0)
    return row


def _advance_validation_sampler(val_loader: Any, *, validation_epoch: int) -> None:
    skips = max(0, int(validation_epoch) - 1)
    if skips == 0:
        return
    batch_sampler = getattr(val_loader, "batch_sampler", None)
    if batch_sampler is None:
        raise ValueError("validation loader does not expose a batch_sampler to advance")
    for _ in range(skips):
        for _batch_indices in batch_sampler:
            pass


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
    base_postprocess = train_cli._build_postprocess_config(train_config)
    variants = _filter_variants(_threshold_variants(base_postprocess), args.variants)

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else checkpoint.parents[2] / "analysis_exports" / f"lane60_thresholds_{int(time.time())}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    train_cli._configure_torch_multiprocessing()
    dataset = train_cli.PV26CanonicalDataset(
        train_cli._existing_dataset_roots(scenario),
        train_augmentation=False,
        progress_callback=lambda message: print(f"[thresholds] {message}", flush=True),
    )
    _, val_loader = train_cli._build_phase_train_loaders(dataset, train_config=train_config, phase=phase)
    if val_loader is None:
        raise ValueError("threshold probe requires validation batches")
    _advance_validation_sampler(val_loader, validation_epoch=int(args.validation_epoch))

    trainer = train_cli._build_phase_trainer(phase, train_config)
    trainer.load_model_weights(checkpoint, map_location=train_config.device)
    evaluator = trainer.build_evaluator()
    evaluator.adapter.raw_model.eval()
    evaluator.heads.eval()

    predictions_by_variant: dict[str, list[dict[str, Any]]] = {name: [] for name, _ in variants}
    raw_batches: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch_index, batch in enumerate(val_loader, start=1):
            if batch_index > int(args.max_val_batches):
                break
            print(f"[thresholds] eval batch {batch_index}/{args.max_val_batches}", flush=True)
            raw_batch = raw_batch_for_metrics(batch)
            if raw_batch is None:
                raise ValueError("validation loader must provide raw batches for metrics")
            encoded = evaluator.prepare_batch(batch)
            predictions = _detach_to_cpu(evaluator.forward_encoded_batch(encoded))
            meta = _detach_to_cpu(encoded["meta"])
            raw_batches.append(raw_batch)
            for name, config in variants:
                predictions_by_variant[name].extend(postprocess_pv26_batch(predictions, meta, config=config))

    merged_raw = _merge_raw_batches(raw_batches)
    rows: list[dict[str, Any]] = []
    metrics_by_variant: dict[str, Any] = {}
    for name, config in variants:
        metrics = augment_lane_family_metrics(summarize_pv26_metrics(predictions_by_variant[name], merged_raw))
        selection = _selection_metrics(metrics, stage=phase.stage)
        metrics_by_variant[name] = {"metrics": metrics, "selection_metrics": selection}
        rows.append(_row(name, config, metrics, selection))
    rows = sorted(rows, key=lambda row: -float(row["phase_objective"]))
    _write_csv(output_dir / "thresholds.csv", rows)
    summary = {
        "checkpoint": str(checkpoint),
        "scenario_path": str(scenario_path),
        "source_run": str(args.source_run or ""),
        "lane60_experiment": str(args.lane60_experiment or ""),
        "phase_name": phase.name,
        "phase_stage": phase.stage,
        "max_val_batches": int(args.max_val_batches),
        "validation_epoch": int(args.validation_epoch),
        "base_postprocess": asdict(base_postprocess),
        "rows": rows,
        "metrics_by_variant": metrics_by_variant,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[thresholds] wrote {output_dir}", flush=True)
    for row in rows[:12]:
        print(
            "[thresholds] "
            f"{row['variant']} objective={float(row['phase_objective']):.6f} "
            f"lane={float(row['lane_f1']):.4f} stop={float(row['stop_line_f1']):.4f} "
            f"cross={float(row['crosswalk_f1']):.4f}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
