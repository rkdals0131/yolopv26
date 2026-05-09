from __future__ import annotations

import argparse
import csv
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
from model.engine.postprocess import postprocess_pv26_batch
from tools.pv26_train import cli as train_cli
from tools.pv26_train import config as train_config_api
from tools.pv26_train.runtime import (
    _LANE_FAMILY_SUPPORT_REFS,
    _PHASE_OBJECTIVE_WEIGHTS,
    _clamp01,
    _float_tree_value,
    _lane_family_reliability,
    _lane_family_support,
)


VARIANTS = (
    "baseline",
    "centerline_times_support",
    "centerline_min_support",
    "centerline_support_floor030",
    "centerline_support_floor050",
    "centerline_support_floor070",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate support-gated lane centerline decode variants for a PV26 checkpoint."
    )
    parser.add_argument("--checkpoint", required=True, help="Checkpoint to evaluate.")
    parser.add_argument("--preset", default="default", help="PV26 meta-train preset.")
    parser.add_argument("--source-run", default="", help="Optional source run used to reproduce lane60 probe validation config.")
    parser.add_argument("--lane60-experiment", default="", help="Optional run_pv26_lane60_probe experiment config to reuse.")
    parser.add_argument("--phase-index", type=int, default=4, help="1-based phase index used for validation config.")
    parser.add_argument("--max-val-batches", type=int, default=128)
    parser.add_argument("--train-batches", type=int, default=512, help="Only used with --lane60-experiment.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", default="")
    return parser.parse_args()


def _resolve_device(requested: str, scenario_device: str) -> str:
    value = str(requested or "auto").strip().lower()
    device = str(scenario_device) if value == "auto" else str(requested)
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[support_gate] CUDA requested but unavailable; falling back to CPU", flush=True)
        return "cpu"
    return device


def _load_scenario(args: argparse.Namespace) -> tuple[Any, Any]:
    if args.lane60_experiment:
        if not args.source_run:
            raise ValueError("--lane60-experiment requires --source-run")
        from tools.run_pv26_lane60_probe import _lane60_scenario

        source_run = Path(args.source_run).expanduser().resolve()
        checkpoint = Path(args.checkpoint).expanduser().resolve()
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
        return _lane60_scenario(scenario_args, source_run=source_run, seed_checkpoint=checkpoint)[:2]
    scenario = train_cli.load_meta_train_scenario(args.preset)
    return scenario, train_cli._default_scenario_path(args.preset)


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


def _logit(probability: torch.Tensor) -> torch.Tensor:
    return torch.logit(probability.clamp(min=1.0e-4, max=1.0 - 1.0e-4))


def _variant_predictions(predictions: dict[str, Any], variant: str) -> dict[str, Any]:
    if variant == "baseline":
        return predictions
    centerline_logits = predictions.get("lane_seg_centerline_logits")
    support_logits = predictions.get("lane_seg_support_logits")
    if not isinstance(centerline_logits, torch.Tensor) or not isinstance(support_logits, torch.Tensor):
        return predictions
    centerline = centerline_logits.sigmoid()
    support = support_logits.sigmoid()
    if variant == "centerline_times_support":
        probability = centerline * support
    elif variant == "centerline_min_support":
        probability = torch.minimum(centerline, support)
    elif variant.startswith("centerline_support_floor"):
        threshold = float(variant.removeprefix("centerline_support_floor")) / 100.0
        probability = torch.where(support >= threshold, centerline, torch.zeros_like(centerline))
    else:
        raise KeyError(f"unsupported support-gate variant: {variant}")
    output = dict(predictions)
    output["lane_seg_centerline_logits"] = _logit(probability)
    return output


def _selection_metrics(metrics: dict[str, Any], *, stage: str) -> dict[str, Any]:
    lane = metrics.get("lane", {}) if isinstance(metrics.get("lane"), dict) else {}
    stop_line = metrics.get("stop_line", {}) if isinstance(metrics.get("stop_line"), dict) else {}
    crosswalk = metrics.get("crosswalk", {}) if isinstance(metrics.get("crosswalk"), dict) else {}
    scores = {
        "lane": (
            0.55 * _float_tree_value(lane, "f1")
            + 0.20 * (1.0 - _clamp01(_float_tree_value(lane, "mean_point_distance") / 40.0))
            + 0.15 * _float_tree_value(lane, "color_accuracy")
            + 0.10 * _float_tree_value(lane, "type_accuracy")
        ),
        "stop_line": (
            0.55 * _float_tree_value(stop_line, "f1")
            + 0.25 * (1.0 - _clamp01(_float_tree_value(stop_line, "mean_point_distance") / 40.0))
            + 0.20 * (1.0 - _clamp01(_float_tree_value(stop_line, "mean_angle_error") / 30.0))
        ),
        "crosswalk": (
            0.45 * _float_tree_value(crosswalk, "f1")
            + 0.35 * _float_tree_value(crosswalk, "mean_polygon_iou")
            + 0.20 * (1.0 - _clamp01(_float_tree_value(crosswalk, "mean_vertex_distance") / 60.0))
        ),
    }
    weights = dict(_PHASE_OBJECTIVE_WEIGHTS.get(stage, {}))
    components: dict[str, dict[str, Any]] = {}
    objective_total = 0.0
    effective_weight_total = 0.0
    for name in ("lane", "stop_line", "crosswalk"):
        metric = metrics.get(name, {}) if isinstance(metrics.get(name), dict) else {}
        reliability = _lane_family_reliability(metric, ref=_LANE_FAMILY_SUPPORT_REFS[name])
        weight = float(weights.get(name, 0.0))
        effective_weight = weight * reliability if weight > 0.0 and reliability > 0.0 else 0.0
        score = float(scores[name])
        if effective_weight > 0.0:
            objective_total += score * effective_weight
            effective_weight_total += effective_weight
        components[name] = {
            "score": score,
            "weight": weight,
            "effective_weight": effective_weight,
            "reliability": reliability,
            "support": _lane_family_support(metric),
        }
    phase_objective = objective_total / effective_weight_total if effective_weight_total > 0.0 else 0.0
    return {"phase_objective": phase_objective, "components": components}


def _row(name: str, metrics: dict[str, Any], selection: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "variant": name,
        "phase_objective": selection["phase_objective"],
    }
    for task in ("lane", "stop_line", "crosswalk"):
        task_metrics = metrics.get(task, {}) if isinstance(metrics.get(task), dict) else {}
        component = selection["components"].get(task, {})
        row[f"{task}_f1"] = task_metrics.get("f1", 0.0)
        row[f"{task}_score"] = component.get("score", 0.0)
        row[f"{task}_support"] = component.get("support", 0)
    return row


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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
        writer.writerows(rows)


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
        else checkpoint.parents[2] / "analysis_exports" / f"lane60_support_gate_{int(time.time())}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    train_cli._configure_torch_multiprocessing()
    dataset = train_cli.PV26CanonicalDataset(
        train_cli._existing_dataset_roots(scenario),
        train_augmentation=False,
        progress_callback=lambda message: print(f"[support_gate] {message}", flush=True),
    )
    _, val_loader = train_cli._build_phase_train_loaders(dataset, train_config=train_config, phase=phase)
    if val_loader is None:
        raise ValueError("support-gate probe requires validation batches")

    trainer = train_cli._build_phase_trainer(phase, train_config)
    trainer.load_model_weights(checkpoint, map_location=train_config.device)
    evaluator = trainer.build_evaluator()
    evaluator.adapter.raw_model.eval()
    evaluator.heads.eval()

    predictions_by_variant: dict[str, list[dict[str, Any]]] = {name: [] for name in VARIANTS}
    raw_batches: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch_index, batch in enumerate(val_loader, start=1):
            if batch_index > int(args.max_val_batches):
                break
            print(f"[support_gate] eval batch {batch_index}/{args.max_val_batches}", flush=True)
            raw_batch = raw_batch_for_metrics(batch)
            if raw_batch is None:
                raise ValueError("validation loader must provide raw batches for metrics")
            encoded = evaluator.prepare_batch(batch)
            predictions = _detach_to_cpu(evaluator.forward_encoded_batch(encoded))
            meta = _detach_to_cpu(encoded["meta"])
            raw_batches.append(raw_batch)
            for variant in VARIANTS:
                predictions_by_variant[variant].extend(
                    postprocess_pv26_batch(
                        _variant_predictions(predictions, variant),
                        meta,
                        config=postprocess_config,
                    )
                )

    merged_raw = _merge_raw_batches(raw_batches)
    rows: list[dict[str, Any]] = []
    metrics_by_variant: dict[str, Any] = {}
    for variant, predictions in predictions_by_variant.items():
        metrics = augment_lane_family_metrics(summarize_pv26_metrics(predictions, merged_raw))
        selection = _selection_metrics(metrics, stage=phase.stage)
        metrics_by_variant[variant] = {"metrics": metrics, "selection_metrics": selection}
        rows.append(_row(variant, metrics, selection))
    rows = sorted(rows, key=lambda row: -float(row["phase_objective"]))
    _write_csv(output_dir / "support_gate.csv", rows)
    summary = {
        "checkpoint": str(checkpoint),
        "scenario_path": str(scenario_path),
        "source_run": str(args.source_run or ""),
        "lane60_experiment": str(args.lane60_experiment or ""),
        "phase_name": phase.name,
        "phase_stage": phase.stage,
        "max_val_batches": int(args.max_val_batches),
        "postprocess": asdict(postprocess_config),
        "rows": rows,
        "metrics_by_variant": metrics_by_variant,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[support_gate] wrote {output_dir}", flush=True)
    for row in rows:
        print(
            "[support_gate] "
            f"{row['variant']} objective={float(row['phase_objective']):.6f} "
            f"lane={float(row['lane_f1']):.4f} stop={float(row['stop_line_f1']):.4f} "
            f"cross={float(row['crosswalk_f1']):.4f}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
