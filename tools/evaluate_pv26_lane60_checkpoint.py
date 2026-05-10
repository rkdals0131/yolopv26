from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
import site
import sys
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
repo_root = str(REPO_ROOT)
if repo_root not in sys.path:
    site.addsitedir(repo_root)

from tools.pv26_train import cli as train_cli
from tools.pv26_train import config as train_config_api
from tools.pv26_train.runtime import PhaseTransitionController
from tools.run_pv26_lane60_probe import _lane60_scenario


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a PV26 lane60 checkpoint through the same trainer.validate_epoch "
            "and phase-selection path used by training."
        )
    )
    parser.add_argument("--checkpoint", required=True, help="Checkpoint to evaluate.")
    parser.add_argument("--source-run", required=True, help="Source PV26 meta-train run directory.")
    parser.add_argument("--lane60-experiment", required=True, help="run_pv26_lane60_probe experiment config to reuse.")
    parser.add_argument("--preset", default="default")
    parser.add_argument("--max-val-batches", type=int, default=128)
    parser.add_argument(
        "--validation-epoch",
        type=int,
        default=1,
        help="One-based validation epoch subset to reproduce. Epoch 2 skips one sampler pass before evaluation.",
    )
    parser.add_argument("--train-batches", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", default="")
    return parser.parse_args()


def _metric(metrics: dict[str, Any], task: str, name: str) -> float:
    payload = metrics.get(task, {}) if isinstance(metrics.get(task), dict) else {}
    value = payload.get(name, 0.0)
    return float(value) if isinstance(value, (int, float)) else 0.0


def _component(selection: dict[str, Any], task: str, name: str) -> float:
    components = selection.get("components", {}) if isinstance(selection.get("components"), dict) else {}
    payload = components.get(task, {}) if isinstance(components.get(task), dict) else {}
    value = payload.get(name, 0.0)
    return float(value) if isinstance(value, (int, float)) else 0.0


def _json_ready(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        value = asdict(value)
    if isinstance(value, torch.Tensor):
        detached = value.detach().cpu()
        if detached.numel() == 1:
            return detached.item()
        if detached.numel() <= 1024:
            return detached.tolist()
        return {"tensor_shape": list(detached.shape), "tensor_dtype": str(detached.dtype)}
    if isinstance(value, np.ndarray):
        if value.size == 1:
            return value.item()
        if value.size <= 1024:
            return value.tolist()
        return {"array_shape": list(value.shape), "array_dtype": str(value.dtype)}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return train_cli.train_artifacts.json_ready(value)


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
    source_run = Path(args.source_run).expanduser().resolve()
    if not source_run.is_dir():
        raise FileNotFoundError(f"source run not found: {source_run}")

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
    phase_index = int(tuple(options["selected_phase_indices"])[0])
    phase = scenario.phases[phase_index - 1]
    train_config = train_config_api.scenario_phase_defaults(scenario.train_defaults, phase.overrides)
    phase_selection = train_config_api.resolve_phase_selection(scenario.selection, phase)

    train_cli._configure_torch_multiprocessing()
    dataset = train_cli.PV26CanonicalDataset(
        train_cli._existing_dataset_roots(scenario),
        train_augmentation=False,
        progress_callback=lambda message: print(f"[lane60_eval] {message}", flush=True),
    )
    _, val_loader = train_cli._build_phase_train_loaders(dataset, train_config=train_config, phase=phase)
    if val_loader is None:
        raise ValueError("lane60 checkpoint evaluation requires validation batches")
    _advance_validation_sampler(val_loader, validation_epoch=int(args.validation_epoch))

    trainer = train_cli._build_phase_trainer(phase, train_config)
    load_report = trainer.load_model_weights(checkpoint, map_location=train_config.device)
    val_summary = trainer.validate_epoch(
        val_loader,
        epoch=int(args.validation_epoch),
        epoch_total=max(1, int(args.validation_epoch)),
        phase_index=phase_index,
        phase_count=len(scenario.phases),
        phase_name=phase.name,
        max_batches=train_config_api.resolve_val_batch_limit(train_config.val_batches),
        log_every_n_steps=20,
        profile_window=train_config.profile_window,
        profile_device_sync=train_config.profile_device_sync,
    )
    epoch_summary: dict[str, Any] = {"epoch": int(args.validation_epoch), "stage": phase.stage, "val": val_summary}
    controller = PhaseTransitionController(
        phase=phase,
        selection=phase_selection,
        resolve_summary_path=train_cli.resolve_summary_path,
    )
    controller.annotate_epoch(epoch_summary)
    selection = epoch_summary["selection_metrics"]
    metrics = val_summary.get("metrics", {}) if isinstance(val_summary.get("metrics"), dict) else {}
    row = {
        "checkpoint": str(checkpoint),
        "lane60_experiment": str(args.lane60_experiment),
        "validation_epoch": int(args.validation_epoch),
        "phase_objective": float(selection["phase_objective"]),
        "lane_f1": _metric(metrics, "lane", "f1"),
        "stop_line_f1": _metric(metrics, "stop_line", "f1"),
        "crosswalk_f1": _metric(metrics, "crosswalk", "f1"),
        "lane_score": _component(selection, "lane", "score"),
        "stop_line_score": _component(selection, "stop_line", "score"),
        "crosswalk_score": _component(selection, "crosswalk", "score"),
        "lane_support": int(_component(selection, "lane", "support")),
        "stop_line_support": int(_component(selection, "stop_line", "support")),
        "crosswalk_support": int(_component(selection, "crosswalk", "support")),
    }
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else checkpoint.parents[2] / "analysis_exports" / "lane60_exact_checkpoint_eval"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "metrics.csv", [row])
    payload = {
        "checkpoint": str(checkpoint),
        "scenario_path": str(scenario_path),
        "phase_index": phase_index,
        "phase_name": phase.name,
        "phase_stage": phase.stage,
        "validation_epoch": int(args.validation_epoch),
        "train_config": _json_ready(train_config),
        "load_report": _json_ready(load_report),
        "val_summary": _json_ready(val_summary),
        "selection_metrics": _json_ready(selection),
        "row": row,
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(row, ensure_ascii=False, indent=2), flush=True)
    print(f"[lane60_eval] wrote {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
