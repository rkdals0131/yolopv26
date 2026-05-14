from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass, replace as dataclasses_replace
import csv
import json
from pathlib import Path
import site
import sys
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
repo_root = str(REPO_ROOT)
if repo_root not in sys.path:
    site.addsitedir(repo_root)

from model.engine._trainer_epochs import _merge_raw_batches
from model.engine.batch import augment_lane_family_metrics, raw_batch_for_metrics
from model.engine.metrics import summarize_pv26_metrics
from model.engine.postprocess import postprocess_pv26_batch
from tools.evaluate_pv26_lane60_checkpoint import (
    _advance_validation_sampler,
    _postprocess_override_config,
    _resolve_dataset_root,
)
from tools.pv26_train import cli as train_cli
from tools.pv26_train import config as train_config_api
from tools.pv26_train.runtime import PhaseTransitionController
from tools.run_pv26_lane60_probe import _lane60_scenario


SOURCE_RUN = (
    REPO_ROOT
    / "runs"
    / "pv26_exhaustive_od_lane_train"
    / "lane60_lane_head_transplant_original_stop_pca_20260512"
)
DEFAULT_CHECKPOINT = SOURCE_RUN / "merged_lane_head.pt"
LANE_LOGIT_KEYS = (
    "lane_seg_centerline_logits",
    "lane_seg_support_logits",
    "lane_seg_color_logits",
    "lane_seg_type_logits",
)
LANE_TANGENT_KEY = "lane_seg_tangent_axis"
DEFAULT_VARIANTS = (
    "baseline",
    "flip_centerline_avg",
    "flip_centerline_avg_lane_cross_comp050",
)
SEMANTIC_VOTE_MODES = (
    "component",
    "centerline",
    "centerline_excess",
    "component_core",
)
TASK_MASK_COMPETITION_VARIANTS: dict[str, tuple[str, tuple[str, ...], float]] = {
    "flip_centerline_avg_lane_cross_comp050": ("flip_centerline_avg", ("crosswalk_mask_logits",), 0.50),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Probe whether horizontal flip centerline averaging and the fixed "
            "crosswalk-mask lane competition gate reproduce the retained PV26 "
            "lane-family runtime composite."
        )
    )
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--source-run", default=str(SOURCE_RUN))
    parser.add_argument("--lane60-experiment", required=True)
    parser.add_argument("--preset", default="default")
    parser.add_argument("--max-val-batches", type=int, default=128)
    parser.add_argument("--validation-epoch", type=int, default=1)
    parser.add_argument("--train-batches", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--backbone-weights",
        default="",
        help="Optional explicit YOLO26 backbone weights path to avoid implicit downloads during evaluation.",
    )
    parser.add_argument("--dataset-root", default="")
    parser.add_argument(
        "--variants",
        default=",".join(DEFAULT_VARIANTS),
        help=f"Comma-separated variants. Available: {', '.join(DEFAULT_VARIANTS)}",
    )
    parser.add_argument(
        "--lane-semantic-vote-modes",
        default="component",
        help=f"Comma-separated lane semantic vote modes. Available: {', '.join(SEMANTIC_VOTE_MODES)}",
    )
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--stop-line-mask-binary-threshold", type=float, default=None)
    parser.add_argument("--stop-line-min-instance-score", type=float, default=None)
    parser.add_argument("--stop-line-presence-threshold", type=float, default=None)
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
    return parser.parse_args()


def _resolve_device(requested: str, scenario_device: str) -> str:
    value = str(requested or "auto").strip().lower()
    scenario_value = str(scenario_device or "auto").strip()
    if value == "auto" and scenario_value.lower() == "auto":
        candidate = "cuda:0" if torch.cuda.is_available() else "cpu"
    else:
        candidate = scenario_value if value == "auto" else str(requested)
    if candidate.startswith("cuda") and not torch.cuda.is_available():
        print("[lane_flip_tta] CUDA requested but unavailable; falling back to CPU", flush=True)
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


def _unflip_spatial_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return torch.flip(tensor, dims=(-1,))


def _unflip_lane_tangent_axis(tensor: torch.Tensor) -> torch.Tensor:
    unflipped = _unflip_spatial_tensor(tensor).clone()
    if unflipped.ndim == 4 and int(unflipped.shape[1]) >= 1:
        unflipped[:, 0, :, :] = -unflipped[:, 0, :, :]
    elif unflipped.ndim == 3 and int(unflipped.shape[0]) >= 1:
        unflipped[0, :, :] = -unflipped[0, :, :]
    else:
        raise ValueError(f"lane tangent axis must be CHW or BCHW, got {tuple(unflipped.shape)}")
    return unflipped


def _unflip_lane_dense_outputs(predictions: dict[str, Any]) -> dict[str, Any]:
    output = dict(predictions)
    for key in LANE_LOGIT_KEYS:
        value = predictions.get(key)
        if isinstance(value, torch.Tensor):
            output[key] = _unflip_spatial_tensor(value)
    tangent = predictions.get(LANE_TANGENT_KEY)
    if isinstance(tangent, torch.Tensor):
        output[LANE_TANGENT_KEY] = _unflip_lane_tangent_axis(tangent)
    return output


def _merge_required_tensor(base: dict[str, Any], flipped: dict[str, Any], key: str) -> tuple[torch.Tensor, torch.Tensor]:
    base_value = base.get(key)
    flip_value = flipped.get(key)
    if not isinstance(base_value, torch.Tensor) or not isinstance(flip_value, torch.Tensor):
        raise KeyError(f"flip TTA variant requires tensor key: {key}")
    if tuple(base_value.shape) != tuple(flip_value.shape):
        raise ValueError(f"shape mismatch for {key}: {tuple(base_value.shape)} vs {tuple(flip_value.shape)}")
    return base_value, flip_value


def _resize_spatial_like(value: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if tuple(value.shape) == tuple(reference.shape):
        return value
    if value.ndim != reference.ndim or value.ndim < 3:
        raise ValueError(f"cannot resize tensor shape {tuple(value.shape)} to {tuple(reference.shape)}")
    if tuple(value.shape[:-2]) != tuple(reference.shape[:-2]):
        raise ValueError(f"non-spatial shape mismatch: {tuple(value.shape)} vs {tuple(reference.shape)}")
    return F.interpolate(
        value.to(dtype=reference.dtype),
        size=tuple(int(dim) for dim in reference.shape[-2:]),
        mode="bilinear",
        align_corners=False,
    )


def _logit_from_probability(probability: torch.Tensor) -> torch.Tensor:
    clipped = probability.clamp(min=1.0e-4, max=1.0 - 1.0e-4)
    return torch.log(clipped / (1.0 - clipped))


def _apply_lane_task_mask_competition(
    predictions: dict[str, Any],
    *,
    mask_keys: tuple[str, ...],
    strength: float,
) -> dict[str, Any]:
    centerline_logits = predictions.get("lane_seg_centerline_logits")
    if not isinstance(centerline_logits, torch.Tensor):
        raise KeyError("lane task-mask competition requires lane_seg_centerline_logits")
    competition = torch.zeros_like(centerline_logits.sigmoid())
    for key in mask_keys:
        mask_logits = predictions.get(key)
        if not isinstance(mask_logits, torch.Tensor):
            raise KeyError(f"lane task-mask competition requires tensor key: {key}")
        mask_probability = _resize_spatial_like(mask_logits.sigmoid(), centerline_logits)
        competition = torch.maximum(competition, mask_probability.to(dtype=competition.dtype))
    keep_probability = (1.0 - float(strength) * competition).clamp(min=0.0, max=1.0)
    output = dict(predictions)
    output["lane_seg_centerline_logits"] = _logit_from_probability(centerline_logits.sigmoid() * keep_probability)
    return output


def _merge_lane_dense_predictions(
    base: dict[str, Any],
    flipped_unflipped: dict[str, Any],
    *,
    variant: str,
) -> dict[str, Any]:
    normalized = str(variant).strip()
    task_mask_spec = TASK_MASK_COMPETITION_VARIANTS.get(normalized)
    if task_mask_spec is not None:
        base_variant, mask_keys, strength = task_mask_spec
        merged = _merge_lane_dense_predictions(base, flipped_unflipped, variant=base_variant)
        return _apply_lane_task_mask_competition(merged, mask_keys=mask_keys, strength=strength)
    if normalized == "baseline":
        return dict(base)
    output = dict(base)
    if normalized == "flip_centerline_avg":
        base_value, flip_value = _merge_required_tensor(base, flipped_unflipped, "lane_seg_centerline_logits")
        output["lane_seg_centerline_logits"] = 0.5 * base_value + 0.5 * flip_value
        return output
    raise ValueError(f"unknown flip TTA variant: {variant}")


def _variant_names(raw: str) -> tuple[str, ...]:
    names = tuple(name.strip() for name in str(raw).split(",") if name.strip())
    if not names:
        raise ValueError("at least one variant is required")
    unknown = sorted(set(names) - set(DEFAULT_VARIANTS))
    if unknown:
        raise ValueError(f"unknown flip TTA variants: {', '.join(unknown)}")
    return names


def _semantic_vote_modes(raw: str) -> tuple[str, ...]:
    modes = tuple(mode.strip() for mode in str(raw).split(",") if mode.strip())
    if not modes:
        raise ValueError("at least one lane semantic vote mode is required")
    unknown = sorted(set(modes) - set(SEMANTIC_VOTE_MODES))
    if unknown:
        raise ValueError(f"unknown lane semantic vote modes: {', '.join(unknown)}")
    return modes


def _variant_label(*, flip_variant: str, semantic_vote_mode: str, multi_mode: bool) -> str:
    if not multi_mode and semantic_vote_mode == "component":
        return str(flip_variant)
    return f"{flip_variant}__semantic_{semantic_vote_mode}"


def _metric(metrics: dict[str, Any], task: str, name: str) -> float:
    payload = metrics.get(task, {}) if isinstance(metrics.get(task), dict) else {}
    value = payload.get(name, 0.0)
    return float(value) if isinstance(value, (int, float)) else 0.0


def _row_from_metrics(name: str, metrics: dict[str, Any], selection: dict[str, Any] | None = None) -> dict[str, Any]:
    row: dict[str, Any] = {"variant": name}
    for task in ("lane", "stop_line", "crosswalk"):
        values = metrics.get(task, {}) if isinstance(metrics.get(task), dict) else {}
        for key in ("precision", "recall", "f1", "tp", "fp", "fn", "mean_point_distance", "mean_polygon_iou"):
            value = values.get(key)
            if isinstance(value, (int, float)):
                row[f"{task}_{key}"] = value
    lane_family = metrics.get("lane_family", {}) if isinstance(metrics.get("lane_family"), dict) else {}
    for key in ("mean_f1", "min_f1"):
        value = lane_family.get(key)
        if isinstance(value, (int, float)):
            row[f"lane_family_{key}"] = value
    row["phase4_objective_proxy"] = (
        0.50 * _metric(metrics, "lane", "f1")
        + 0.30 * _metric(metrics, "stop_line", "f1")
        + 0.20 * _metric(metrics, "crosswalk", "f1")
    )
    if isinstance(selection, dict):
        phase_objective = selection.get("phase_objective")
        if isinstance(phase_objective, (int, float)):
            row["phase_objective"] = float(phase_objective)
        components = selection.get("components", {}) if isinstance(selection.get("components"), dict) else {}
        for task in ("lane", "stop_line", "crosswalk"):
            payload = components.get(task, {}) if isinstance(components.get(task), dict) else {}
            score = payload.get("score")
            support = payload.get("support")
            if isinstance(score, (int, float)):
                row[f"{task}_score"] = float(score)
            if isinstance(support, (int, float)):
                row[f"{task}_support"] = int(support)
    return row


def _json_ready(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        value = asdict(value)
    if isinstance(value, torch.Tensor):
        detached = value.detach().cpu()
        if detached.numel() == 1:
            return detached.item()
        return {"tensor_shape": list(detached.shape), "tensor_dtype": str(detached.dtype)}
    if isinstance(value, np.ndarray):
        if value.size == 1:
            return value.item()
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


def main() -> int:
    args = parse_args()
    variants = _variant_names(args.variants)
    semantic_vote_modes = _semantic_vote_modes(args.lane_semantic_vote_modes)
    variant_matrix = [
        (
            _variant_label(
                flip_variant=variant,
                semantic_vote_mode=semantic_vote_mode,
                multi_mode=len(semantic_vote_modes) > 1 or semantic_vote_mode != "component",
            ),
            variant,
            semantic_vote_mode,
        )
        for variant in variants
        for semantic_vote_mode in semantic_vote_modes
    ]
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
    train_config = dataclasses_replace(
        train_config,
        device=_resolve_device(str(args.device), train_config.device),
        val_batches=int(args.max_val_batches),
        batch_size=int(args.batch_size),
    )

    train_cli._configure_torch_multiprocessing()
    dataset = train_cli.PV26CanonicalDataset(
        train_cli._existing_dataset_roots(scenario),
        train_augmentation=False,
        progress_callback=lambda message: print(f"[lane_flip_tta] {message}", flush=True),
    )
    _, val_loader = train_cli._build_phase_train_loaders(dataset, train_config=train_config, phase=phase)
    if val_loader is None:
        raise ValueError("lane flip TTA probe requires validation batches")
    _advance_validation_sampler(val_loader, validation_epoch=int(args.validation_epoch))

    trainer = train_cli._build_phase_trainer(phase, train_config)
    load_report = trainer.load_model_weights(checkpoint, map_location=train_config.device)
    postprocess_config = _postprocess_override_config(args, trainer) or train_cli._build_postprocess_config(train_config)
    evaluator = trainer.build_evaluator()
    evaluator.adapter.raw_model.eval()
    evaluator.heads.eval()

    predictions_by_variant: dict[str, list[dict[str, Any]]] = {label: [] for label, _, _ in variant_matrix}
    raw_batches: list[dict[str, Any]] = []
    processed_batches = 0
    with torch.no_grad():
        for batch_index, batch in enumerate(val_loader, start=1):
            if batch_index > int(args.max_val_batches):
                break
            if batch_index == 1 or batch_index % 20 == 0:
                print(f"[lane_flip_tta] eval batch {batch_index}/{args.max_val_batches}", flush=True)
            raw_batch = raw_batch_for_metrics(batch)
            if raw_batch is None:
                raise ValueError("validation loader must provide raw batches for metrics")
            raw_batches.append(raw_batch)
            encoded = evaluator.prepare_batch(batch)
            base_outputs = _detach_to_cpu(evaluator.forward_encoded_batch(encoded))
            flipped_encoded = dict(encoded)
            flipped_encoded["image"] = torch.flip(encoded["image"], dims=(-1,))
            flipped_outputs = _detach_to_cpu(evaluator.forward_encoded_batch(flipped_encoded))
            flipped_unflipped = _unflip_lane_dense_outputs(flipped_outputs)
            meta_rows = _detach_to_cpu(encoded["meta"])
            outputs_by_variant: dict[str, dict[str, Any]] = {}
            for variant in variants:
                outputs_by_variant[variant] = _merge_lane_dense_predictions(
                    base_outputs,
                    flipped_unflipped,
                    variant=variant,
                )
            for label, variant, semantic_vote_mode in variant_matrix:
                variant_postprocess_config = dataclasses_replace(
                    postprocess_config,
                    lane_segfirst_semantic_vote_mode=str(semantic_vote_mode),
                )
                predictions_by_variant[label].extend(
                    postprocess_pv26_batch(
                        outputs_by_variant[variant],
                        meta_rows,
                        config=variant_postprocess_config,
                    )
                )
            processed_batches += 1

    merged_raw = _merge_raw_batches(raw_batches)
    metrics_by_variant = {
        label: augment_lane_family_metrics(summarize_pv26_metrics(predictions_by_variant[label], merged_raw))
        for label, _, _ in variant_matrix
    }
    selection_by_variant: dict[str, dict[str, Any]] = {}
    for name, _, _ in variant_matrix:
        epoch_summary: dict[str, Any] = {
            "epoch": int(args.validation_epoch),
            "stage": phase.stage,
            "val": {"metrics": metrics_by_variant[name]},
        }
        PhaseTransitionController(
            phase=phase,
            selection=phase_selection,
            resolve_summary_path=train_cli.resolve_summary_path,
        ).annotate_epoch(epoch_summary)
        selection = epoch_summary.get("selection_metrics")
        selection_by_variant[name] = dict(selection) if isinstance(selection, dict) else {}
    rows = [_row_from_metrics(name, metrics_by_variant[name], selection_by_variant.get(name)) for name, _, _ in variant_matrix]
    rows.sort(key=lambda row: float(row.get("phase_objective", row.get("phase4_objective_proxy", 0.0))), reverse=True)

    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else checkpoint.parent / "analysis_exports" / "lane_flip_tta"
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "metrics.csv", rows)
    payload = {
        "checkpoint": str(checkpoint),
        "scenario_path": str(scenario_path),
        "lane60_experiment": str(args.lane60_experiment),
        "phase_index": phase_index,
        "phase_name": phase.name,
        "validation_epoch": int(args.validation_epoch),
        "processed_batches": int(processed_batches),
        "variants": list(variants),
        "semantic_vote_modes": list(semantic_vote_modes),
        "variant_matrix": [
            {
                "label": label,
                "flip_variant": variant,
                "semantic_vote_mode": semantic_vote_mode,
            }
            for label, variant, semantic_vote_mode in variant_matrix
        ],
        "train_config": _json_ready(train_config),
        "postprocess_config": _json_ready(postprocess_config),
        "load_report": _json_ready(load_report),
        "rows": rows,
        "metrics_by_variant": _json_ready(metrics_by_variant),
        "selection_by_variant": _json_ready(selection_by_variant),
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(rows, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    print(f"[lane_flip_tta] wrote {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
