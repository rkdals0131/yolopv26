from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import json
from pathlib import Path
import site
import sys
from typing import Any

import numpy as np
import torch
from scipy import ndimage

REPO_ROOT = Path(__file__).resolve().parents[1]
repo_root = str(REPO_ROOT)
if repo_root not in sys.path:
    site.addsitedir(repo_root)

from common.geometry import sample_stop_line_centerline
from model.data.transform import inverse_transform_points, transform_from_meta, unique_point_count
from model.engine import augment_lane_family_metrics, raw_batch_for_metrics, summarize_pv26_metrics
from model.engine._trainer_epochs import _merge_raw_batches
from model.engine.postprocess import (
    STOP_LINE_POINT_COUNT,
    STOPLINE_MIN_ASPECT_RATIO,
    STOPLINE_MIN_COMPONENT_LENGTH,
    STOPLINE_MIN_COMPONENT_PIXELS,
    PV26PostprocessConfig,
    _dedupe_stop_line_predictions,
    _filter_stop_line_predictions,
    _fit_stopline_segment,
    _prepare_stopline_binary_mask,
    _promote_stop_line_endpoint_floor_backup,
    _promote_stop_line_structured_fallback,
    _stopline_allowed_labels,
    _stopline_orientation_score,
    _stopline_prediction_sort_key,
    _suppress_stop_line_fragments,
    _tensor_all_finite,
    postprocess_pv26_batch,
)
from tools.probe_pv26_lane60_postprocess_thresholds import _advance_validation_sampler, _detach_to_cpu
from tools.probe_pv26_lane60_support_gate import _load_scenario, _selection_metrics
from tools.pv26_train import cli as train_cli
from tools.pv26_train import config as train_config_api


SOURCE_RUN = (
    REPO_ROOT
    / "runs"
    / "pv26_exhaustive_od_lane_train"
    / "lane60_core_centerline_refine_cross_retain_from_exhaustive_od_lane_default_20260505_032217_default_20260510_003412"
)
DEFAULT_CHECKPOINT = SOURCE_RUN / "phase_4" / "checkpoints" / "best.pt"


@dataclass(frozen=True)
class LocalExtractionVariant:
    name: str
    score_source: str
    row_band: int
    normal_band: float
    score_quantile: float
    max_components: int = 1
    merge_mode: str = "replace"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Probe stop-line component-conditioned local extraction variants against "
            "an existing lane60 checkpoint."
        )
    )
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--preset", default="default")
    parser.add_argument("--source-run", default=str(SOURCE_RUN))
    parser.add_argument("--lane60-experiment", default="core_centerline_refine_cross_retain")
    parser.add_argument("--phase-index", type=int, default=4)
    parser.add_argument("--max-val-batches", type=int, default=128)
    parser.add_argument("--validation-epoch", type=int, default=2)
    parser.add_argument("--train-batches", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output-json",
        default=str(SOURCE_RUN / "analysis_exports" / "stopline_local_component_extraction_val128_epoch2.json"),
    )
    return parser.parse_args()


def _resolve_device(requested: str, scenario_device: str) -> str:
    value = str(requested or "auto").strip().lower()
    device = str(scenario_device) if value == "auto" else str(requested)
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[stopline_local_extract] CUDA requested but unavailable; falling back to CPU", flush=True)
        return "cpu"
    return device


def _variants() -> tuple[LocalExtractionVariant, ...]:
    return (
        LocalExtractionVariant("local_center_r2_n2_q60", "center", 2, 2.0, 0.60),
        LocalExtractionVariant("local_center_r4_n2_q60", "center", 4, 2.0, 0.60),
        LocalExtractionVariant("local_center_r6_n3_q50", "center", 6, 3.0, 0.50),
        LocalExtractionVariant("local_selector_r2_n2_q60", "selector", 2, 2.0, 0.60),
        LocalExtractionVariant("local_selector_r4_n2_q60", "selector", 4, 2.0, 0.60),
        LocalExtractionVariant("local_fused_r2_n2_q60", "fused", 2, 2.0, 0.60),
        LocalExtractionVariant("local_fused_r4_n2_q60", "fused", 4, 2.0, 0.60),
        LocalExtractionVariant("local_fused_r6_n3_q50", "fused", 6, 3.0, 0.50),
        LocalExtractionVariant("local_fused_r4_n2_q60_top2", "fused", 4, 2.0, 0.60, max_components=2),
        LocalExtractionVariant("local_center_r6_n3_q50_append_top2", "center", 6, 3.0, 0.50, max_components=2, merge_mode="append"),
        LocalExtractionVariant("local_selector_r4_n2_q60_append_top2", "selector", 4, 2.0, 0.60, max_components=2, merge_mode="append"),
        LocalExtractionVariant("local_fused_r4_n2_q60_append_top2", "fused", 4, 2.0, 0.60, max_components=2, merge_mode="append"),
        LocalExtractionVariant("local_fused_r6_n3_q50_append_top2", "fused", 6, 3.0, 0.50, max_components=2, merge_mode="append"),
    )


def _as_numpy_map(tensor: torch.Tensor | None) -> np.ndarray | None:
    if not isinstance(tensor, torch.Tensor) or not _tensor_all_finite(tensor):
        return None
    array = tensor.sigmoid().detach().cpu().numpy()
    while array.ndim > 2 and array.shape[0] == 1:
        array = array.squeeze(0)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array.squeeze(0)
    if array.ndim != 2:
        return None
    return np.asarray(array, dtype=np.float32)


def _row_probs(row_logits: torch.Tensor | None) -> np.ndarray | None:
    array = _as_numpy_map(row_logits)
    if array is None:
        return None
    if array.ndim == 2:
        return np.asarray(array.max(axis=-1), dtype=np.float32)
    return None


def _score_map(
    *,
    source: str,
    mask_probs: np.ndarray,
    center_probs: np.ndarray | None,
    selector_probs: np.ndarray | None,
) -> np.ndarray:
    center = center_probs if center_probs is not None and center_probs.shape == mask_probs.shape else None
    selector = selector_probs if selector_probs is not None and selector_probs.shape == mask_probs.shape else None
    if source == "center" and center is not None:
        return np.asarray(center, dtype=np.float32)
    if source == "selector" and selector is not None:
        return np.asarray(selector, dtype=np.float32)
    if center is not None and selector is not None:
        return np.asarray(0.5 * center + 0.5 * selector, dtype=np.float32)
    if center is not None:
        return np.asarray(center, dtype=np.float32)
    if selector is not None:
        return np.asarray(selector, dtype=np.float32)
    return np.asarray(mask_probs, dtype=np.float32)


def _segment_to_raw_points(
    segment: tuple[np.ndarray, np.ndarray, float, float],
    *,
    meta: dict[str, Any],
    map_hw: tuple[int, int],
) -> list[list[float]]:
    start, end, _length, _thickness = segment
    output_h, output_w = map_hw
    network_segment = np.stack([start, end], axis=0).astype(np.float32)
    network_segment[:, 0] = (network_segment[:, 0] + 0.5) * (float(meta["network_hw"][1]) / float(output_w))
    network_segment[:, 1] = (network_segment[:, 1] + 0.5) * (float(meta["network_hw"][0]) / float(output_h))
    transform = transform_from_meta(meta)
    network_points = sample_stop_line_centerline(network_segment.tolist(), target_count=STOP_LINE_POINT_COUNT).tolist()
    if unique_point_count(network_points) < 2:
        return []
    raw_points = sample_stop_line_centerline(
        inverse_transform_points(network_points, transform),
        target_count=STOP_LINE_POINT_COUNT,
    ).tolist()
    if unique_point_count(raw_points) < 2:
        return []
    return [[float(x), float(y)] for x, y in raw_points]


def _fit_local_component(
    *,
    rows: np.ndarray,
    cols: np.ndarray,
    mask_values: np.ndarray,
    score_values: np.ndarray,
    variant: LocalExtractionVariant,
) -> tuple[tuple[np.ndarray, np.ndarray, float, float] | None, np.ndarray, float]:
    points = np.stack([cols.astype(np.float32), rows.astype(np.float32)], axis=1)
    if points.shape[0] < STOPLINE_MIN_COMPONENT_PIXELS:
        return None, np.zeros((0,), dtype=bool), 0.0
    combined = np.asarray(score_values, dtype=np.float32) * np.asarray(mask_values, dtype=np.float32)
    if not bool(np.isfinite(combined).all()) or float(combined.max(initial=0.0)) <= 0.0:
        combined = np.asarray(mask_values, dtype=np.float32)
    anchor_index = int(np.argmax(combined))
    anchor = points[anchor_index].astype(np.float32)

    floor = float(np.quantile(combined, float(variant.score_quantile))) if combined.size else 0.0
    row_mask = np.abs(rows.astype(np.float32) - float(anchor[1])) <= float(variant.row_band)
    seed_mask = row_mask & (combined >= max(0.0, floor))
    if int(seed_mask.sum()) < STOPLINE_MIN_COMPONENT_PIXELS:
        top_count = min(max(STOPLINE_MIN_COMPONENT_PIXELS, 12), int(points.shape[0]))
        top_indices = np.argsort(combined)[-top_count:]
        seed_mask = np.zeros((points.shape[0],), dtype=bool)
        seed_mask[top_indices] = True
    if int(seed_mask.sum()) < STOPLINE_MIN_COMPONENT_PIXELS:
        return None, seed_mask, float(combined[anchor_index])

    seed_points = points[seed_mask]
    seed_center = seed_points.mean(axis=0, keepdims=True)
    try:
        _, _, seed_vh = np.linalg.svd(seed_points - seed_center, full_matrices=False)
    except np.linalg.LinAlgError:
        return None, seed_mask, float(combined[anchor_index])
    axis = seed_vh[0]
    if abs(float(axis[0])) < abs(float(axis[1])) and seed_vh.shape[0] > 1:
        axis = seed_vh[1]
    axis = axis / max(float(np.linalg.norm(axis)), 1.0e-6)
    normal = np.asarray([-axis[1], axis[0]], dtype=np.float32)
    normal_distance = np.abs((points - anchor[None, :]) @ normal)
    refined_floor = float(np.quantile(combined, max(0.0, float(variant.score_quantile) - 0.10)))
    refined_mask = (normal_distance <= float(variant.normal_band)) & (combined >= max(0.0, refined_floor))
    if int(refined_mask.sum()) < STOPLINE_MIN_COMPONENT_PIXELS:
        refined_mask = seed_mask

    local_points = points[refined_mask]
    local_values = mask_values[refined_mask]
    fitted = _fit_stopline_segment(local_points, mask_values=local_values, center_anchor=anchor)
    if fitted is None:
        return None, refined_mask, float(combined[anchor_index])
    return fitted, refined_mask, float(combined[anchor_index])


def _decode_local_stop_lines(
    outputs: dict[str, Any],
    sample_index: int,
    meta: dict[str, Any],
    *,
    variant: LocalExtractionVariant,
    config: PV26PostprocessConfig,
) -> list[dict[str, Any]]:
    mask_logits = outputs.get("stop_line_mask_logits")
    if not isinstance(mask_logits, torch.Tensor):
        return []
    sample_mask = mask_logits[sample_index]
    if not _tensor_all_finite(sample_mask):
        return []
    mask_probs = sample_mask.sigmoid().squeeze(0).detach().cpu().numpy()
    if mask_probs.ndim == 3:
        mask_probs = mask_probs.squeeze(0)
    if mask_probs.ndim != 2:
        return []
    if float(mask_probs.max(initial=0.0)) <= float(config.stop_line_obj_threshold):
        return []
    binary = _prepare_stopline_binary_mask(mask_probs, threshold=float(config.stop_line_mask_binary_threshold))
    if not bool(binary.any()):
        return []
    labels, component_count = ndimage.label(binary)
    if component_count <= 0:
        return []
    component_scores = ndimage.maximum(mask_probs, labels, index=np.arange(1, int(component_count) + 1))

    center_logits = outputs.get("stop_line_center_logits")
    selector_logits = outputs.get("stop_line_selector_map_logits")
    row_logits = outputs.get("stop_line_row_logits")
    center_probs = _as_numpy_map(center_logits[sample_index]) if isinstance(center_logits, torch.Tensor) else None
    selector_probs = _as_numpy_map(selector_logits[sample_index]) if isinstance(selector_logits, torch.Tensor) else None
    sample_row_probs = _row_probs(row_logits[sample_index]) if isinstance(row_logits, torch.Tensor) else None
    score_map = _score_map(
        source=str(variant.score_source),
        mask_probs=mask_probs,
        center_probs=center_probs,
        selector_probs=selector_probs,
    )
    allowed_labels = _stopline_allowed_labels(labels, center_probs, row_probs=sample_row_probs)

    predictions: list[dict[str, Any]] = []
    for label_index in range(1, int(component_count) + 1):
        rows, cols = np.nonzero(labels == label_index)
        if len(rows) < max(STOPLINE_MIN_COMPONENT_PIXELS, int(config.stop_line_min_component_pixels)):
            continue
        if allowed_labels is not None and label_index not in allowed_labels:
            continue
        mask_values = mask_probs[rows, cols]
        score_values = score_map[rows, cols]
        fitted, support_mask, peak_score = _fit_local_component(
            rows=rows,
            cols=cols,
            mask_values=mask_values,
            score_values=score_values,
            variant=variant,
        )
        if fitted is None:
            continue
        _start, _end, length, thickness = fitted
        if length < STOPLINE_MIN_COMPONENT_LENGTH:
            continue
        if thickness > 12.0:
            continue
        if length / max(thickness, 1.0) < STOPLINE_MIN_ASPECT_RATIO:
            continue
        raw_points = _segment_to_raw_points(fitted, meta=meta, map_hw=mask_probs.shape)
        if len(raw_points) < 2:
            continue
        component_score = float(component_scores[label_index - 1])
        local_mask_score = float(mask_values[support_mask].mean()) if bool(support_mask.any()) else float(mask_values.mean())
        instance_score = 0.72 * component_score + 0.18 * float(peak_score) + 0.10 * local_mask_score
        if allowed_labels is not None and label_index in allowed_labels:
            instance_score += 0.02
        if instance_score <= float(config.stop_line_obj_threshold):
            continue
        predictions.append(
            {
                "allowed": bool(allowed_labels is None or label_index in allowed_labels),
                "score": float(instance_score),
                "center_score": float(peak_score),
                "orientation_score": float(_stopline_orientation_score(raw_points)),
                "length": float(length),
                "thickness": float(thickness),
                "points_xy": raw_points,
            }
        )

    predictions.sort(key=_stopline_prediction_sort_key, reverse=True)
    predictions = _dedupe_stop_line_predictions(predictions)
    predictions = _suppress_stop_line_fragments(predictions)
    predictions = _promote_stop_line_structured_fallback(predictions)
    predictions = _promote_stop_line_endpoint_floor_backup(predictions, meta=meta)
    predictions.sort(key=_stopline_prediction_sort_key, reverse=True)
    predictions = _dedupe_stop_line_predictions(predictions)
    predictions = _filter_stop_line_predictions(
        predictions,
        min_bbox_area_px=float(config.stop_line_min_bbox_area_px),
        min_bbox_aspect=float(config.stop_line_min_bbox_aspect),
        min_instance_score=float(config.stop_line_min_instance_score),
    )
    return predictions[: max(1, int(variant.max_components))]


def _with_stop_lines(prediction: dict[str, Any], stop_lines: list[dict[str, Any]]) -> dict[str, Any]:
    return {**prediction, "stop_lines": stop_lines}


def _merge_stop_lines(
    baseline_stop_lines: list[dict[str, Any]],
    local_stop_lines: list[dict[str, Any]],
    *,
    variant: LocalExtractionVariant,
    config: PV26PostprocessConfig,
) -> list[dict[str, Any]]:
    if variant.merge_mode == "replace":
        return local_stop_lines
    merged = [*baseline_stop_lines, *local_stop_lines]
    merged.sort(key=_stopline_prediction_sort_key, reverse=True)
    merged = _dedupe_stop_line_predictions(merged)
    merged = _suppress_stop_line_fragments(merged)
    merged.sort(key=_stopline_prediction_sort_key, reverse=True)
    merged = _dedupe_stop_line_predictions(merged)
    merged = _filter_stop_line_predictions(
        merged,
        min_bbox_area_px=float(config.stop_line_min_bbox_area_px),
        min_bbox_aspect=float(config.stop_line_min_bbox_aspect),
        min_instance_score=float(config.stop_line_min_instance_score),
    )
    return merged[: max(1, int(variant.max_components))]


def _row_from_metrics(name: str, metrics: dict[str, Any], selection: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {"variant": name, "phase_objective": selection["phase_objective"]}
    for task in ("lane", "stop_line", "crosswalk"):
        values = metrics.get(task, {}) if isinstance(metrics.get(task), dict) else {}
        component = selection["components"].get(task, {})
        for key in ("precision", "recall", "f1", "tp", "fp", "fn", "mean_point_distance", "mean_angle_error", "mean_polygon_iou"):
            value = values.get(key)
            if isinstance(value, (int, float)):
                row[f"{task}_{key}"] = value
        row[f"{task}_score"] = component.get("score", 0.0)
        row[f"{task}_support"] = component.get("support", 0)
    lane_family = metrics.get("lane_family", {}) if isinstance(metrics.get("lane_family"), dict) else {}
    for key in ("mean_f1", "min_f1"):
        value = lane_family.get(key)
        if isinstance(value, (int, float)):
            row[f"lane_family_{key}"] = value
    return row


def main() -> int:
    args = parse_args()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")

    scenario, _scenario_path = _load_scenario(args)
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
        progress_callback=lambda message: print(f"[stopline_local_extract] {message}", flush=True),
    )
    _, val_loader = train_cli._build_phase_train_loaders(dataset, train_config=train_config, phase=phase)
    if val_loader is None:
        raise ValueError("stop-line local extraction probe requires validation batches")
    _advance_validation_sampler(val_loader, validation_epoch=int(args.validation_epoch))

    trainer = train_cli._build_phase_trainer(phase, train_config)
    trainer.load_model_weights(checkpoint, map_location=train_config.device)
    evaluator = trainer.build_evaluator()
    evaluator.adapter.raw_model.eval()
    evaluator.heads.eval()

    variants = _variants()
    predictions_by_variant: dict[str, list[dict[str, Any]]] = {"baseline": []}
    predictions_by_variant.update({variant.name: [] for variant in variants})
    raw_batches: list[dict[str, Any]] = []

    with torch.no_grad():
        for batch_index, batch in enumerate(val_loader, start=1):
            if batch_index > int(args.max_val_batches):
                break
            if batch_index == 1 or batch_index % 20 == 0:
                print(f"[stopline_local_extract] eval batch {batch_index}/{args.max_val_batches}", flush=True)
            raw_batch = raw_batch_for_metrics(batch)
            if raw_batch is None:
                raise ValueError("validation loader must provide raw batches for metrics")
            raw_batches.append(raw_batch)
            encoded = evaluator.prepare_batch(batch)
            outputs = _detach_to_cpu(evaluator.forward_encoded_batch(encoded))
            meta_rows = _detach_to_cpu(encoded["meta"])
            baseline_predictions = postprocess_pv26_batch(outputs, meta_rows, config=postprocess_config)
            predictions_by_variant["baseline"].extend(baseline_predictions)
            for variant in variants:
                for sample_index, (baseline_prediction, meta) in enumerate(zip(baseline_predictions, meta_rows)):
                    stop_lines = _decode_local_stop_lines(
                        outputs,
                        sample_index,
                        meta,
                        variant=variant,
                        config=postprocess_config,
                    )
                    merged_stop_lines = _merge_stop_lines(
                        list(baseline_prediction.get("stop_lines", [])),
                        stop_lines,
                        variant=variant,
                        config=postprocess_config,
                    )
                    predictions_by_variant[variant.name].append(_with_stop_lines(baseline_prediction, merged_stop_lines))

    merged_raw = _merge_raw_batches(raw_batches)
    rows: list[dict[str, Any]] = []
    for name, predictions in predictions_by_variant.items():
        metrics = augment_lane_family_metrics(summarize_pv26_metrics(predictions, merged_raw))
        selection = _selection_metrics(metrics, stage=phase.stage)
        rows.append(_row_from_metrics(name, metrics, selection))
    rows.sort(key=lambda row: float(row.get("phase_objective", 0.0)), reverse=True)

    payload = {
        "checkpoint": str(checkpoint),
        "source_run": str(Path(args.source_run).expanduser().resolve()) if args.source_run else "",
        "lane60_experiment": str(args.lane60_experiment),
        "phase_index": int(args.phase_index),
        "max_val_batches": int(args.max_val_batches),
        "validation_epoch": int(args.validation_epoch),
        "postprocess_config": vars(postprocess_config),
        "rows": rows,
    }
    output_path = Path(args.output_json).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(rows, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    print(f"[stopline_local_extract] wrote {output_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
