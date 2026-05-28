from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, is_dataclass, replace as dataclasses_replace
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
    parser.add_argument(
        "--backbone-weights",
        default="",
        help="Optional explicit YOLO26 backbone weights path to avoid implicit downloads during evaluation.",
    )
    parser.add_argument(
        "--dataset-root",
        default="",
        help="Optional canonical dataset root. If omitted, use the scenario dataset root when it exists.",
    )
    parser.add_argument("--lane-obj-threshold", type=float, default=None)
    parser.add_argument("--lane-segfirst-track-mode", default=None)
    parser.add_argument("--lane-segfirst-max-row-gap", type=int, default=None)
    parser.add_argument("--lane-segfirst-max-link-dx", type=float, default=None)
    parser.add_argument("--lane-segfirst-max-turn-degrees", type=float, default=None)
    parser.add_argument("--lane-conditional-row-enabled", action="store_true", default=None)
    parser.add_argument("--lane-conditional-row-disabled", action="store_false", dest="lane_conditional_row_enabled")
    parser.add_argument("--stop-line-mask-binary-threshold", type=float, default=None)
    parser.add_argument("--stop-line-min-instance-score", type=float, default=None)
    parser.add_argument("--stop-line-presence-threshold", type=float, default=None)
    parser.add_argument("--stop-line-haf-enabled", action="store_true", default=None)
    parser.add_argument("--stop-line-haf-disabled", action="store_false", dest="stop_line_haf_enabled")
    parser.add_argument("--stop-line-haf-valid-threshold", type=float, default=None)
    parser.add_argument("--stop-line-haf-min-votes", type=int, default=None)
    parser.add_argument("--stop-line-haf-cluster-endpoint-tolerance", type=float, default=None)
    parser.add_argument("--stop-line-haf-max-endpoint-covariance", type=float, default=None)
    parser.add_argument("--stop-line-haf-max-segments", type=int, default=None)
    parser.add_argument("--stop-line-segment-set-enabled", action="store_true", default=None)
    parser.add_argument("--stop-line-segment-set-disabled", action="store_false", dest="stop_line_segment_set_enabled")
    parser.add_argument("--stop-line-segment-set-score-threshold", type=float, default=None)
    parser.add_argument("--stop-line-segment-set-max-segments", type=int, default=None)
    parser.add_argument("--stop-line-segment-verifier-score-weight", type=float, default=None)
    parser.add_argument("--stop-line-endpoint-pair-segment-enabled", action="store_true", default=None)
    parser.add_argument(
        "--stop-line-endpoint-pair-segment-disabled",
        action="store_false",
        dest="stop_line_endpoint_pair_segment_enabled",
    )
    parser.add_argument("--stop-line-endpoint-pair-segment-score-threshold", type=float, default=None)
    parser.add_argument("--stop-line-endpoint-pair-segment-max-segments", type=int, default=None)
    parser.add_argument("--stop-line-endpoint-pair-verifier-score-weight", type=float, default=None)
    parser.add_argument(
        "--stop-line-component-gate-source",
        choices=("center", "selector", "max"),
        default=None,
    )
    parser.add_argument("--crosswalk-obj-threshold", type=float, default=None)
    parser.add_argument("--crosswalk-mask-binary-threshold", type=float, default=None)
    parser.add_argument("--crosswalk-min-component-pixels", type=int, default=None)
    parser.add_argument("--crosswalk-max-components", type=int, default=None)
    parser.add_argument("--crosswalk-min-polygon-area-px", type=float, default=None)
    parser.add_argument("--crosswalk-min-bbox-aspect", type=float, default=None)
    parser.add_argument("--crosswalk-polygon-mode", choices=("rect", "hull"), default=None)
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


def _resolve_dataset_root(args: argparse.Namespace, source_run: Path, scenario_root: Path) -> Path:
    explicit = Path(str(args.dataset_root)).expanduser().resolve() if str(args.dataset_root).strip() else None
    if explicit is not None:
        return explicit
    if Path(scenario_root).is_dir():
        return Path(scenario_root).resolve()
    for parent in (source_run, *source_run.parents):
        candidate = parent / "seg_dataset" / "pv26_exhaustive_od_lane_dataset"
        if candidate.is_dir():
            return candidate.resolve()
    return Path(scenario_root).resolve()


def _postprocess_override_config(args: argparse.Namespace, trainer: Any) -> Any | None:
    replacements: dict[str, Any] = {}
    if getattr(args, "lane_obj_threshold", None) is not None:
        replacements["lane_obj_threshold"] = float(args.lane_obj_threshold)
    if getattr(args, "lane_segfirst_track_mode", None) is not None:
        replacements["lane_segfirst_track_mode"] = str(args.lane_segfirst_track_mode)
    if getattr(args, "lane_segfirst_max_row_gap", None) is not None:
        replacements["lane_segfirst_max_row_gap"] = int(args.lane_segfirst_max_row_gap)
    if getattr(args, "lane_segfirst_max_link_dx", None) is not None:
        replacements["lane_segfirst_max_link_dx"] = float(args.lane_segfirst_max_link_dx)
    if getattr(args, "lane_segfirst_max_turn_degrees", None) is not None:
        replacements["lane_segfirst_max_turn_degrees"] = float(args.lane_segfirst_max_turn_degrees)
    if getattr(args, "lane_conditional_row_enabled", None) is not None:
        replacements["lane_conditional_row_enabled"] = bool(args.lane_conditional_row_enabled)
    if getattr(args, "stop_line_mask_binary_threshold", None) is not None:
        replacements["stop_line_mask_binary_threshold"] = float(args.stop_line_mask_binary_threshold)
    if getattr(args, "stop_line_min_instance_score", None) is not None:
        replacements["stop_line_min_instance_score"] = float(args.stop_line_min_instance_score)
    if getattr(args, "stop_line_presence_threshold", None) is not None:
        replacements["stop_line_presence_threshold"] = float(args.stop_line_presence_threshold)
    if getattr(args, "stop_line_haf_enabled", None) is not None:
        replacements["stop_line_haf_enabled"] = bool(args.stop_line_haf_enabled)
    if getattr(args, "stop_line_haf_valid_threshold", None) is not None:
        replacements["stop_line_haf_valid_threshold"] = float(args.stop_line_haf_valid_threshold)
    if getattr(args, "stop_line_haf_min_votes", None) is not None:
        replacements["stop_line_haf_min_votes"] = int(args.stop_line_haf_min_votes)
    if getattr(args, "stop_line_haf_cluster_endpoint_tolerance", None) is not None:
        replacements["stop_line_haf_cluster_endpoint_tolerance"] = float(args.stop_line_haf_cluster_endpoint_tolerance)
    if getattr(args, "stop_line_haf_max_endpoint_covariance", None) is not None:
        replacements["stop_line_haf_max_endpoint_covariance"] = float(args.stop_line_haf_max_endpoint_covariance)
    if getattr(args, "stop_line_haf_max_segments", None) is not None:
        replacements["stop_line_haf_max_segments"] = int(args.stop_line_haf_max_segments)
    if getattr(args, "stop_line_segment_set_enabled", None) is not None:
        replacements["stop_line_segment_set_enabled"] = bool(args.stop_line_segment_set_enabled)
    if getattr(args, "stop_line_segment_set_score_threshold", None) is not None:
        replacements["stop_line_segment_set_score_threshold"] = float(args.stop_line_segment_set_score_threshold)
    if getattr(args, "stop_line_segment_set_max_segments", None) is not None:
        replacements["stop_line_segment_set_max_segments"] = int(args.stop_line_segment_set_max_segments)
    if getattr(args, "stop_line_segment_verifier_score_weight", None) is not None:
        replacements["stop_line_segment_verifier_score_weight"] = float(args.stop_line_segment_verifier_score_weight)
    if getattr(args, "stop_line_endpoint_pair_segment_enabled", None) is not None:
        replacements["stop_line_endpoint_pair_segment_enabled"] = bool(args.stop_line_endpoint_pair_segment_enabled)
    if getattr(args, "stop_line_endpoint_pair_segment_score_threshold", None) is not None:
        replacements["stop_line_endpoint_pair_segment_score_threshold"] = float(
            args.stop_line_endpoint_pair_segment_score_threshold
        )
    if getattr(args, "stop_line_endpoint_pair_segment_max_segments", None) is not None:
        replacements["stop_line_endpoint_pair_segment_max_segments"] = int(
            args.stop_line_endpoint_pair_segment_max_segments
        )
    if getattr(args, "stop_line_endpoint_pair_verifier_score_weight", None) is not None:
        replacements["stop_line_endpoint_pair_verifier_score_weight"] = float(
            args.stop_line_endpoint_pair_verifier_score_weight
        )
    if getattr(args, "stop_line_component_gate_source", None) is not None:
        replacements["stop_line_component_gate_source"] = str(args.stop_line_component_gate_source)
    if getattr(args, "crosswalk_obj_threshold", None) is not None:
        replacements["crosswalk_obj_threshold"] = float(args.crosswalk_obj_threshold)
    if getattr(args, "crosswalk_mask_binary_threshold", None) is not None:
        replacements["crosswalk_mask_binary_threshold"] = float(args.crosswalk_mask_binary_threshold)
    if getattr(args, "crosswalk_min_component_pixels", None) is not None:
        replacements["crosswalk_min_component_pixels"] = int(args.crosswalk_min_component_pixels)
    if getattr(args, "crosswalk_max_components", None) is not None:
        replacements["crosswalk_max_components"] = int(args.crosswalk_max_components)
    if getattr(args, "crosswalk_min_polygon_area_px", None) is not None:
        replacements["crosswalk_min_polygon_area_px"] = float(args.crosswalk_min_polygon_area_px)
    if getattr(args, "crosswalk_min_bbox_aspect", None) is not None:
        replacements["crosswalk_min_bbox_aspect"] = float(args.crosswalk_min_bbox_aspect)
    if getattr(args, "crosswalk_polygon_mode", None) is not None:
        replacements["crosswalk_polygon_mode"] = str(args.crosswalk_polygon_mode)
    if not replacements:
        return None
    postprocess_config = dataclasses_replace(getattr(trainer, "postprocess_config"), **replacements)
    setattr(trainer, "postprocess_config", postprocess_config)
    return postprocess_config


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
    backbone_weights = str(args.backbone_weights).strip()
    if backbone_weights:
        scenario = dataclasses_replace(
            scenario,
            train_defaults=dataclasses_replace(
                scenario.train_defaults,
                backbone_weights=str(Path(backbone_weights).expanduser().resolve()),
            ),
        )
    dataset_root = _resolve_dataset_root(args, source_run, scenario.dataset.root)
    scenario = dataclasses_replace(
        scenario,
        dataset=train_config_api.DatasetConfig(
            root=dataset_root,
            additional_roots=tuple(scenario.dataset.additional_roots),
        ),
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
    postprocess_config = _postprocess_override_config(args, trainer)
    evaluator = None
    if postprocess_config is not None:
        evaluator = train_cli._wrap_evaluator_postprocess_config(
            trainer.build_evaluator(),
            postprocess_config=postprocess_config,
        )
    val_summary = trainer.validate_epoch(
        val_loader,
        epoch=int(args.validation_epoch),
        epoch_total=max(1, int(args.validation_epoch)),
        phase_index=phase_index,
        phase_count=len(scenario.phases),
        phase_name=phase.name,
        evaluator=evaluator,
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
        "lane_tp": int(_metric(metrics, "lane", "tp")),
        "lane_fp": int(_metric(metrics, "lane", "fp")),
        "lane_fn": int(_metric(metrics, "lane", "fn")),
        "stop_line_tp": int(_metric(metrics, "stop_line", "tp")),
        "stop_line_fp": int(_metric(metrics, "stop_line", "fp")),
        "stop_line_fn": int(_metric(metrics, "stop_line", "fn")),
        "crosswalk_tp": int(_metric(metrics, "crosswalk", "tp")),
        "crosswalk_fp": int(_metric(metrics, "crosswalk", "fp")),
        "crosswalk_fn": int(_metric(metrics, "crosswalk", "fn")),
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
