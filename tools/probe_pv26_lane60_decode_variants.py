from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import site
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
repo_root = str(REPO_ROOT)
if repo_root not in sys.path:
    site.addsitedir(repo_root)

import torch

from model.engine._trainer_epochs import _merge_raw_batches
from model.engine.batch import augment_lane_family_metrics, raw_batch_for_metrics
from model.engine.metrics import summarize_pv26_metrics
from model.engine.postprocess import PV26PostprocessConfig, postprocess_pv26_batch
from tools.pv26_train import cli as train_cli
from tools.pv26_train import config as train_config_api


SOURCE_RUN = (
    REPO_ROOT
    / "runs"
    / "pv26_exhaustive_od_lane_train"
    / "exhaustive_od_lane_default_20260505_032217"
)
DEFAULT_CHECKPOINT = SOURCE_RUN / "phase_4" / "checkpoints" / "best.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe PV26 lane-family decode variants against an existing checkpoint."
    )
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--preset", default="default")
    parser.add_argument("--phase-index", type=int, default=4)
    parser.add_argument("--max-val-batches", type=int, default=128)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-json", default="")
    return parser.parse_args()


def _resolve_device(requested: str, scenario_device: str) -> str:
    value = str(requested or "auto").strip().lower()
    candidate = str(scenario_device) if value == "auto" else str(requested)
    if candidate.startswith("cuda") and not torch.cuda.is_available():
        print("[decode_probe] CUDA requested but unavailable; falling back to CPU", flush=True)
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


def _clone_predictions(predictions: dict[str, Any]) -> dict[str, Any]:
    return dict(predictions)


def _variant_predictions(predictions: dict[str, Any], variant: str) -> dict[str, Any]:
    candidate = _clone_predictions(predictions)
    if "lane_support_as_centerline" in variant:
        support = candidate.get("lane_seg_support_logits")
        if isinstance(support, torch.Tensor):
            candidate["lane_seg_centerline_logits"] = support
    if "lane_centerline_support_blend" in variant:
        center = candidate.get("lane_seg_centerline_logits")
        support = candidate.get("lane_seg_support_logits")
        if isinstance(center, torch.Tensor) and isinstance(support, torch.Tensor):
            candidate["lane_seg_centerline_logits"] = 0.5 * center + 0.5 * support
    if "stop_mask_only" in variant:
        for key in (
            "stop_line_center_logits",
            "stop_line_selector_map_logits",
            "stop_line_row_logits",
            "stop_line_x_logits",
            "stop_line_center_offset",
            "stop_line_angle",
            "stop_line_half_length",
        ):
            candidate.pop(key, None)
    return candidate


def _variant_postprocess_config(base: PV26PostprocessConfig, variant: str) -> PV26PostprocessConfig:
    if variant == "baseline":
        return base
    overrides: dict[str, Any] = {}
    if "lane_t080" in variant:
        overrides["lane_obj_threshold"] = 0.80
    if "lane_t090" in variant:
        overrides["lane_obj_threshold"] = 0.90
    if "lane_row_scan" in variant:
        overrides["lane_segfirst_track_mode"] = "row_scan"
    if "row_gap24" in variant:
        overrides["lane_segfirst_max_row_gap"] = 24
    if "row_gap36" in variant:
        overrides["lane_segfirst_max_row_gap"] = 36
    if "row_dx12" in variant:
        overrides["lane_segfirst_max_link_dx"] = 12.0
    if "row_dx16" in variant:
        overrides["lane_segfirst_max_link_dx"] = 16.0
    if "lane_len40" in variant:
        overrides["lane_segfirst_min_polyline_length_px"] = 40.0
    if "lane_len80" in variant:
        overrides["lane_segfirst_min_polyline_length_px"] = 80.0
    if "lane_bottom30" in variant:
        overrides["lane_segfirst_min_polyline_bottom_y_fraction"] = 0.30
    if "lane_bottom50" in variant:
        overrides["lane_segfirst_min_polyline_bottom_y_fraction"] = 0.50
    if "stop_obj030" in variant:
        overrides["stop_line_obj_threshold"] = 0.30
    if "stop_obj070" in variant:
        overrides["stop_line_obj_threshold"] = 0.70
    if "stop_mask030" in variant:
        overrides["stop_line_mask_binary_threshold"] = 0.30
    if "stop_mask070" in variant:
        overrides["stop_line_mask_binary_threshold"] = 0.70
    if "cross_obj030" in variant:
        overrides["crosswalk_obj_threshold"] = 0.30
    if "cross_obj070" in variant:
        overrides["crosswalk_obj_threshold"] = 0.70
    if "cross_mask030" in variant:
        overrides["crosswalk_mask_binary_threshold"] = 0.30
    if "cross_mask070" in variant:
        overrides["crosswalk_mask_binary_threshold"] = 0.70
    return replace(base, **overrides) if overrides else base


def _row_from_metrics(name: str, metrics: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {"variant": name}
    for task in ("lane", "stop_line", "crosswalk"):
        values = metrics.get(task, {}) if isinstance(metrics.get(task), dict) else {}
        for key in ("precision", "recall", "f1", "tp", "fp", "fn", "mean_point_distance", "mean_angle_error", "mean_polygon_iou"):
            value = values.get(key)
            if isinstance(value, (int, float)):
                row[f"{task}_{key}"] = value
    lane_family = metrics.get("lane_family", {}) if isinstance(metrics.get("lane_family"), dict) else {}
    for key in ("mean_f1", "min_f1"):
        value = lane_family.get(key)
        if isinstance(value, (int, float)):
            row[f"lane_family_{key}"] = value
    row["phase4_objective_proxy"] = (
        0.50 * float(row.get("lane_f1", 0.0))
        + 0.30 * float(row.get("stop_line_f1", 0.0))
        + 0.20 * float(row.get("crosswalk_f1", 0.0))
    )
    return row


def main() -> int:
    args = parse_args()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    scenario = train_cli.load_meta_train_scenario(args.preset)
    phase = scenario.phases[int(args.phase_index) - 1]
    train_config = train_config_api.scenario_phase_defaults(scenario.train_defaults, phase.overrides)
    train_config = replace(
        train_config,
        device=_resolve_device(args.device, train_config.device),
        val_batches=int(args.max_val_batches),
    )
    postprocess_config = train_cli._build_postprocess_config(train_config)

    train_cli._configure_torch_multiprocessing()
    dataset = train_cli.PV26CanonicalDataset(
        train_cli._existing_dataset_roots(scenario),
        train_augmentation=False,
        progress_callback=lambda message: print(f"[decode_probe] {message}", flush=True),
    )
    _, val_loader = train_cli._build_phase_train_loaders(dataset, train_config=train_config, phase=phase)
    if val_loader is None:
        raise ValueError("decode variant probe requires validation batches")

    trainer = train_cli._build_phase_trainer(phase, train_config)
    trainer.load_model_weights(checkpoint, map_location=train_config.device)
    evaluator = trainer.build_evaluator()
    evaluator.adapter.raw_model.eval()
    evaluator.heads.eval()

    variants = (
        "baseline",
        "stop_mask_only",
        "lane_t080",
        "lane_t090",
        "lane_t090_stop_mask_only",
        "lane_t090_len40",
        "lane_t090_len40_stop_mask_only",
        "lane_t090_len80",
        "lane_t090_bottom30",
        "lane_t090_bottom50",
        "stop_obj030",
        "stop_obj070",
        "stop_mask030",
        "stop_mask070",
        "cross_obj030",
        "cross_obj070",
        "cross_mask030",
        "cross_mask070",
        "lane_t090_stop_mask_only_cross_obj030",
        "lane_t090_stop_mask_only_cross_obj070",
        "lane_t090_stop_mask_only_cross_mask030",
        "lane_t090_stop_mask_only_cross_mask070",
        "lane_t090_stop_mask_only_stop_obj030",
        "lane_t090_stop_mask_only_stop_obj070",
        "lane_t090_stop_mask_only_stop_mask030",
        "lane_t090_stop_mask_only_stop_mask070",
        "lane_support_as_centerline",
        "lane_centerline_support_blend",
        "lane_support_stop_mask_only",
        "lane_centerline_support_blend_stop_mask_only",
        "lane_row_scan",
        "lane_row_scan_row_gap24",
        "lane_row_scan_row_gap24_row_dx12",
        "lane_row_scan_row_gap36_row_dx12",
        "lane_row_scan_row_gap36_row_dx16",
        "lane_t080_lane_row_scan",
        "lane_t090_lane_row_scan",
    )
    predictions_by_variant: dict[str, list[dict[str, Any]]] = {name: [] for name in variants}
    raw_batches: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch_index, batch in enumerate(val_loader, start=1):
            if batch_index > int(args.max_val_batches):
                break
            if batch_index == 1 or batch_index % 20 == 0:
                print(f"[decode_probe] eval batch {batch_index}/{args.max_val_batches}", flush=True)
            raw_batch = raw_batch_for_metrics(batch)
            if raw_batch is None:
                raise ValueError("validation loader must provide raw batches for metrics")
            raw_batches.append(raw_batch)
            encoded = evaluator.prepare_batch(batch)
            predictions = _detach_to_cpu(evaluator.forward_encoded_batch(encoded))
            meta = _detach_to_cpu(encoded["meta"])
            for variant in variants:
                variant_predictions = predictions if variant == "baseline" else _variant_predictions(predictions, variant)
                variant_postprocess = _variant_postprocess_config(postprocess_config, variant)
                predictions_by_variant[variant].extend(
                    postprocess_pv26_batch(variant_predictions, meta, config=variant_postprocess)
                )

    merged_raw = _merge_raw_batches(raw_batches)
    rows = []
    for variant in variants:
        metrics = augment_lane_family_metrics(
            summarize_pv26_metrics(predictions_by_variant[variant], merged_raw)
        )
        rows.append(_row_from_metrics(variant, metrics))
    rows.sort(key=lambda row: float(row.get("phase4_objective_proxy", 0.0)), reverse=True)
    if str(args.output_json).strip():
        output_path = Path(args.output_json).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
        print(f"[decode_probe] wrote {output_path}", flush=True)
    print(json.dumps(rows, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
