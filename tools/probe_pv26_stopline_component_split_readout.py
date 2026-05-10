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

from model.engine import augment_lane_family_metrics, raw_batch_for_metrics, summarize_pv26_metrics
from model.engine._trainer_epochs import _merge_raw_batches
from model.engine.postprocess import (
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
from tools.probe_pv26_stopline_local_component_extraction import (
    DEFAULT_CHECKPOINT,
    SOURCE_RUN,
    _as_numpy_map,
    _merge_stop_lines,
    _resolve_device,
    _row_from_metrics,
    _row_probs,
    _score_map,
    _segment_to_raw_points,
    _with_stop_lines,
)
from tools.pv26_train import cli as train_cli
from tools.pv26_train import config as train_config_api


@dataclass(frozen=True)
class ComponentSplitVariant:
    name: str
    score_source: str
    top_points: int
    pair_stride: int
    normal_band: float
    min_pair_distance: float
    max_components: int = 1
    merge_mode: str = "replace"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Probe stop-line component split readout variants against an existing "
            "lane60 checkpoint without retraining."
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
        default=str(SOURCE_RUN / "analysis_exports" / "stopline_component_split_readout_val128_epoch2.json"),
    )
    return parser.parse_args()


def _variants() -> tuple[ComponentSplitVariant, ...]:
    return (
        ComponentSplitVariant("split_mask_top24_b2p0", "mask", 24, 3, 2.0, 8.0),
        ComponentSplitVariant("split_mask_top32_b2p5", "mask", 32, 4, 2.5, 8.0),
        ComponentSplitVariant("split_fused_top24_b2p0", "fused", 24, 3, 2.0, 8.0),
        ComponentSplitVariant("split_fused_top32_b2p5", "fused", 32, 4, 2.5, 8.0),
        ComponentSplitVariant("split_fused_top48_b3p5", "fused", 48, 6, 3.5, 10.0),
        ComponentSplitVariant("split_fused_top32_b2p5_top2", "fused", 32, 4, 2.5, 8.0, max_components=2),
        ComponentSplitVariant(
            "split_fused_top32_b2p5_append_top2",
            "fused",
            32,
            4,
            2.5,
            8.0,
            max_components=2,
            merge_mode="append",
        ),
    )


def _line_inlier_mask(points: np.ndarray, start: np.ndarray, end: np.ndarray, *, normal_band: float) -> np.ndarray:
    axis = end.astype(np.float32) - start.astype(np.float32)
    length = float(np.linalg.norm(axis))
    if length <= 1.0e-6:
        return np.zeros((points.shape[0],), dtype=bool)
    axis = axis / length
    normal = np.asarray([-axis[1], axis[0]], dtype=np.float32)
    offsets = points - start[None, :]
    projection = offsets @ axis
    normal_distance = np.abs(offsets @ normal)
    return (
        (normal_distance <= float(normal_band))
        & (projection >= -0.10 * length)
        & (projection <= 1.10 * length)
    )


def _candidate_segments_for_component(
    *,
    rows: np.ndarray,
    cols: np.ndarray,
    mask_values: np.ndarray,
    score_values: np.ndarray,
    variant: ComponentSplitVariant,
) -> list[tuple[float, tuple[np.ndarray, np.ndarray, float, float], np.ndarray, float]]:
    points = np.stack([cols.astype(np.float32), rows.astype(np.float32)], axis=1)
    if points.shape[0] < STOPLINE_MIN_COMPONENT_PIXELS:
        return []
    combined = np.asarray(score_values, dtype=np.float32) * np.asarray(mask_values, dtype=np.float32)
    if not bool(np.isfinite(combined).all()) or float(combined.max(initial=0.0)) <= 0.0:
        combined = np.asarray(mask_values, dtype=np.float32)
    order = np.argsort(combined)[::-1]
    top_count = min(max(4, int(variant.top_points)), int(order.shape[0]))
    top_indices = order[:top_count]
    stride = max(1, int(variant.pair_stride))

    candidate_rows: list[tuple[float, tuple[np.ndarray, np.ndarray, float, float], np.ndarray, float]] = []
    sampled = top_indices[::stride]
    if sampled.shape[0] < 2:
        sampled = top_indices
    for left_pos, left_index in enumerate(sampled.tolist()):
        left = points[left_index]
        for right_index in sampled[left_pos + 1 :].tolist():
            right = points[right_index]
            pair_distance = float(np.linalg.norm(right - left))
            if pair_distance < float(variant.min_pair_distance):
                continue
            axis = right - left
            axis_norm = float(np.linalg.norm(axis))
            if axis_norm <= 1.0e-6:
                continue
            axis = axis / axis_norm
            if abs(float(axis[0])) < 0.70:
                continue
            inliers = _line_inlier_mask(points, left, right, normal_band=float(variant.normal_band))
            if int(inliers.sum()) < STOPLINE_MIN_COMPONENT_PIXELS:
                continue
            fitted = _fit_stopline_segment(points[inliers], mask_values=mask_values[inliers], center_anchor=left)
            if fitted is None:
                continue
            _start, _end, length, thickness = fitted
            if length < STOPLINE_MIN_COMPONENT_LENGTH or thickness > 12.0:
                continue
            if length / max(thickness, 1.0) < STOPLINE_MIN_ASPECT_RATIO:
                continue
            inlier_mask_score = float(mask_values[inliers].mean())
            inlier_signal = float(combined[inliers].mean())
            support = float(inliers.sum()) / max(float(points.shape[0]), 1.0)
            score = (
                0.45 * inlier_mask_score
                + 0.35 * inlier_signal
                + 0.15 * min(float(length) / 32.0, 1.0)
                + 0.05 * support
            )
            candidate_rows.append((score, fitted, inliers, float(combined[left_index])))

    candidate_rows.sort(key=lambda item: item[0], reverse=True)
    kept: list[tuple[float, tuple[np.ndarray, np.ndarray, float, float], np.ndarray, float]] = []
    for candidate in candidate_rows:
        start, end, _length, _thickness = candidate[1]
        candidate_mid = 0.5 * (start + end)
        duplicate = False
        for existing in kept:
            prev_start, prev_end, _prev_length, _prev_thickness = existing[1]
            prev_mid = 0.5 * (prev_start + prev_end)
            if float(np.linalg.norm(candidate_mid - prev_mid)) <= 3.0:
                duplicate = True
                break
        if duplicate:
            continue
        kept.append(candidate)
        if len(kept) >= max(1, int(variant.max_components)):
            break
    return kept


def _decode_component_split_stop_lines(
    outputs: dict[str, Any],
    sample_index: int,
    meta: dict[str, Any],
    *,
    variant: ComponentSplitVariant,
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
        for split_score, fitted, support_mask, peak_score in _candidate_segments_for_component(
            rows=rows,
            cols=cols,
            mask_values=mask_values,
            score_values=score_values,
            variant=variant,
        ):
            raw_points = _segment_to_raw_points(fitted, meta=meta, map_hw=mask_probs.shape)
            if len(raw_points) < 2:
                continue
            _start, _end, length, thickness = fitted
            component_score = float(component_scores[label_index - 1])
            local_mask_score = float(mask_values[support_mask].mean()) if bool(support_mask.any()) else float(mask_values.mean())
            instance_score = 0.64 * component_score + 0.18 * float(split_score) + 0.12 * local_mask_score + 0.06 * float(peak_score)
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
        progress_callback=lambda message: print(f"[stopline_split] {message}", flush=True),
    )
    _, val_loader = train_cli._build_phase_train_loaders(dataset, train_config=train_config, phase=phase)
    if val_loader is None:
        raise ValueError("stop-line component split probe requires validation batches")
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
                print(f"[stopline_split] eval batch {batch_index}/{args.max_val_batches}", flush=True)
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
                    stop_lines = _decode_component_split_stop_lines(
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
    print(f"[stopline_split] wrote {output_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
