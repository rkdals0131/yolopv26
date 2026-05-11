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

REPO_ROOT = Path(__file__).resolve().parents[1]
repo_root = str(REPO_ROOT)
if repo_root not in sys.path:
    site.addsitedir(repo_root)

from common.geometry import sample_stop_line_centerline
from model.data.transform import inverse_transform_points, transform_from_meta
from model.engine._trainer_epochs import _merge_raw_batches
from model.engine.batch import augment_lane_family_metrics, raw_batch_for_metrics
from model.engine.metrics import STOP_LINE_POINT_COUNT, summarize_pv26_metrics
from model.engine.postprocess import (
    PV26PostprocessConfig,
    STOPLINE_MIN_ASPECT_RATIO,
    STOPLINE_MIN_COMPONENT_LENGTH,
    STOPLINE_MIN_COMPONENT_PIXELS,
    _dedupe_stop_line_predictions,
    _filter_stop_line_predictions,
    _fit_stopline_segment,
    _prepare_stopline_binary_mask,
    _stopline_prediction_sort_key,
    _suppress_stop_line_fragments,
    postprocess_pv26_batch,
)
from tools.evaluate_pv26_lane60_checkpoint import _advance_validation_sampler
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
class RowXVariant:
    name: str
    row_band: int
    x_threshold: float
    max_rows: int = 1
    min_row_score: float = 0.20
    fallback_baseline: bool = False


VARIANTS = (
    RowXVariant("rowx_r2_x030_replace", row_band=2, x_threshold=0.30),
    RowXVariant("rowx_r3_x030_replace", row_band=3, x_threshold=0.30),
    RowXVariant("rowx_r4_x020_replace", row_band=4, x_threshold=0.20),
    RowXVariant("rowx_r4_x040_replace", row_band=4, x_threshold=0.40),
    RowXVariant("rowx_r3_x030_fallback", row_band=3, x_threshold=0.30, fallback_baseline=True),
    RowXVariant("rowx_r4_x020_fallback", row_band=4, x_threshold=0.20, fallback_baseline=True),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe stop-line row/x projection proposals against an existing PV26 checkpoint."
    )
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--preset", default="default")
    parser.add_argument("--phase-index", type=int, default=4)
    parser.add_argument("--max-val-batches", type=int, default=128)
    parser.add_argument("--validation-epoch", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-json", default="")
    return parser.parse_args()


def _resolve_device(requested: str, scenario_device: str) -> str:
    value = str(requested or "auto").strip().lower()
    candidate = str(scenario_device) if value == "auto" else str(requested)
    if candidate.startswith("cuda") and not torch.cuda.is_available():
        print("[stopline_rowx] CUDA requested but unavailable; falling back to CPU", flush=True)
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


def _as_numpy_2d(tensor: torch.Tensor | None) -> np.ndarray | None:
    if not isinstance(tensor, torch.Tensor):
        return None
    array = tensor.detach().cpu().numpy()
    while array.ndim > 2 and array.shape[0] == 1:
        array = array.squeeze(0)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array.squeeze(0)
    if array.ndim != 2:
        return None
    return np.asarray(array, dtype=np.float32)


def _row_probs(row_logits: torch.Tensor | None) -> np.ndarray | None:
    array = _as_numpy_2d(row_logits.sigmoid() if isinstance(row_logits, torch.Tensor) else None)
    if array is None:
        return None
    return np.asarray(array.max(axis=-1), dtype=np.float32)


def _x_probs(x_logits: torch.Tensor | None) -> np.ndarray | None:
    array = _as_numpy_2d(x_logits.sigmoid() if isinstance(x_logits, torch.Tensor) else None)
    if array is None:
        return None
    return np.asarray(array.max(axis=0), dtype=np.float32)


def _raw_points_from_map_segment(segment: np.ndarray, meta: dict[str, Any], map_hw: tuple[int, int]) -> list[list[float]]:
    transform = transform_from_meta(meta)
    map_h, map_w = map_hw
    network_h, network_w = transform.network_hw
    network_segment = np.asarray(segment, dtype=np.float32).reshape(-1, 2).copy()
    network_segment[:, 0] = (network_segment[:, 0] + 0.5) * (float(network_w) / float(map_w))
    network_segment[:, 1] = (network_segment[:, 1] + 0.5) * (float(network_h) / float(map_h))
    raw_points = inverse_transform_points(network_segment.tolist(), transform)
    return sample_stop_line_centerline(raw_points, target_count=STOP_LINE_POINT_COUNT).tolist()


def _orientation_score(points_xy: list[list[float]]) -> float:
    points = np.asarray(points_xy, dtype=np.float32)
    if points.ndim != 2 or points.shape[0] < 2:
        return 0.0
    delta = points[-1] - points[0]
    norm = float(np.linalg.norm(delta))
    if norm <= 1.0e-6:
        return 0.0
    return float(min(1.0, abs(float(delta[0])) / norm))


def _connected_spans(mask: np.ndarray, *, min_length: int) -> list[tuple[int, int]]:
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        return []
    spans: list[tuple[int, int]] = []
    start = int(indices[0])
    previous = int(indices[0])
    for value in indices[1:]:
        current = int(value)
        if current != previous + 1:
            if previous - start + 1 >= int(min_length):
                spans.append((start, previous))
            start = current
        previous = current
    if previous - start + 1 >= int(min_length):
        spans.append((start, previous))
    return spans


def _top_rows(row_scores: np.ndarray, variant: RowXVariant) -> list[int]:
    order = np.argsort(-row_scores)
    selected: list[int] = []
    min_gap = max(2, int(variant.row_band) * 2 + 1)
    for value in order.tolist():
        row_index = int(value)
        if float(row_scores[row_index]) < float(variant.min_row_score):
            break
        if any(abs(row_index - existing) < min_gap for existing in selected):
            continue
        selected.append(row_index)
        if len(selected) >= int(variant.max_rows):
            break
    return selected


def _segment_from_fit_or_span(
    *,
    points: np.ndarray,
    mask_values: np.ndarray,
    anchor: np.ndarray,
    row_index: int,
    span: tuple[int, int],
) -> tuple[np.ndarray, float, float] | None:
    fitted = _fit_stopline_segment(points, mask_values=mask_values, center_anchor=anchor)
    if fitted is not None:
        start, end, length, thickness = fitted
        if float(length) >= STOPLINE_MIN_COMPONENT_LENGTH and float(thickness) <= 12.0:
            aspect = float(length) / max(float(thickness), 1.0)
            if aspect >= STOPLINE_MIN_ASPECT_RATIO:
                return np.stack([start, end], axis=0).astype(np.float32), float(length), float(thickness)
    start_col, end_col = span
    length = float(end_col - start_col + 1)
    if length < STOPLINE_MIN_COMPONENT_LENGTH:
        return None
    segment = np.asarray([[float(start_col), float(row_index)], [float(end_col), float(row_index)]], dtype=np.float32)
    return segment, length, 1.0


def _decode_rowx_stop_lines(
    *,
    outputs: dict[str, Any],
    sample_index: int,
    meta: dict[str, Any],
    config: PV26PostprocessConfig,
    variant: RowXVariant,
) -> list[dict[str, Any]]:
    mask_batch = outputs.get("stop_line_mask_logits")
    row_batch = outputs.get("stop_line_row_logits")
    x_batch = outputs.get("stop_line_x_logits")
    if not all(isinstance(value, torch.Tensor) for value in (mask_batch, row_batch, x_batch)):
        return []

    mask_probs = _as_numpy_2d(mask_batch[sample_index].sigmoid())
    rows_1d = _row_probs(row_batch[sample_index])
    cols_1d = _x_probs(x_batch[sample_index])
    if mask_probs is None or rows_1d is None or cols_1d is None:
        return []
    if float(mask_probs.max(initial=0.0)) <= float(config.stop_line_obj_threshold):
        return []

    binary = _prepare_stopline_binary_mask(mask_probs, threshold=float(config.stop_line_mask_binary_threshold))
    if not bool(binary.any()):
        return []
    map_h, map_w = mask_probs.shape
    predictions: list[dict[str, Any]] = []
    for row_index in _top_rows(rows_1d, variant):
        lower = max(0, int(row_index) - int(variant.row_band))
        upper = min(map_h - 1, int(row_index) + int(variant.row_band))
        row_band_mask = np.zeros_like(binary, dtype=bool)
        row_band_mask[lower : upper + 1, :] = True
        band_binary = binary & row_band_mask
        if not bool(band_binary.any()):
            continue
        col_mask = (cols_1d >= float(variant.x_threshold)) & band_binary.any(axis=0)
        spans = _connected_spans(col_mask, min_length=max(STOPLINE_MIN_COMPONENT_LENGTH // 2, 8))
        for span in spans:
            start_col, end_col = span
            span_mask = np.zeros_like(binary, dtype=bool)
            span_mask[:, start_col : end_col + 1] = True
            local_mask = band_binary & span_mask
            local_rows, local_cols = np.nonzero(local_mask)
            if local_rows.size < max(8, STOPLINE_MIN_COMPONENT_PIXELS // 2):
                continue
            points = np.stack([local_cols.astype(np.float32), local_rows.astype(np.float32)], axis=1)
            mask_values = mask_probs[local_rows, local_cols].astype(np.float32)
            col_weights = mask_values * cols_1d[local_cols].astype(np.float32)
            if float(col_weights.sum()) > 1.0e-6:
                anchor_x = float(np.average(local_cols.astype(np.float32), weights=col_weights))
            else:
                anchor_x = 0.5 * float(start_col + end_col)
            anchor = np.asarray([anchor_x, float(row_index)], dtype=np.float32)
            fitted = _segment_from_fit_or_span(
                points=points,
                mask_values=mask_values,
                anchor=anchor,
                row_index=int(row_index),
                span=span,
            )
            if fitted is None:
                continue
            segment, length, thickness = fitted
            segment[:, 0] = np.clip(segment[:, 0], 0.0, float(map_w - 1))
            segment[:, 1] = np.clip(segment[:, 1], 0.0, float(map_h - 1))
            raw_points = _raw_points_from_map_segment(segment, meta, mask_probs.shape)
            if len(raw_points) < 2:
                continue
            local_score = float(mask_values.max(initial=0.0))
            x_score = float(cols_1d[start_col : end_col + 1].max(initial=0.0))
            row_score = float(rows_1d[row_index])
            score = 0.45 * local_score + 0.30 * row_score + 0.25 * x_score
            if score <= float(config.stop_line_obj_threshold):
                continue
            predictions.append(
                {
                    "allowed": True,
                    "score": float(score),
                    "center_score": float(row_score),
                    "orientation_score": _orientation_score(raw_points),
                    "length": float(length),
                    "thickness": float(thickness),
                    "points_xy": [[float(x), float(y)] for x, y in raw_points],
                }
            )
    predictions.sort(key=_stopline_prediction_sort_key, reverse=True)
    predictions = _dedupe_stop_line_predictions(predictions)
    predictions = _suppress_stop_line_fragments(predictions)
    predictions = _filter_stop_line_predictions(
        predictions,
        min_bbox_area_px=float(config.stop_line_min_bbox_area_px),
        min_bbox_aspect=float(config.stop_line_min_bbox_aspect),
        min_instance_score=float(config.stop_line_min_instance_score),
    )
    return predictions[: max(1, int(config.stop_line_max_components))]


def _with_stop_lines(prediction: dict[str, Any], stop_lines: list[dict[str, Any]]) -> dict[str, Any]:
    return {**prediction, "stop_lines": stop_lines}


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
        progress_callback=lambda message: print(f"[stopline_rowx] {message}", flush=True),
    )
    _, val_loader = train_cli._build_phase_train_loaders(dataset, train_config=train_config, phase=phase)
    if val_loader is None:
        raise ValueError("stop-line row/x probe requires validation batches")
    _advance_validation_sampler(val_loader, validation_epoch=int(args.validation_epoch))

    trainer = train_cli._build_phase_trainer(phase, train_config)
    trainer.load_model_weights(checkpoint, map_location=train_config.device)
    evaluator = trainer.build_evaluator()
    evaluator.adapter.raw_model.eval()
    evaluator.heads.eval()

    names = ("baseline",) + tuple(variant.name for variant in VARIANTS)
    predictions_by_variant: dict[str, list[dict[str, Any]]] = {name: [] for name in names}
    variant_stats: dict[str, dict[str, int]] = {
        variant.name: {"decoded_samples": 0, "decoded_lines": 0, "fallback_samples": 0}
        for variant in VARIANTS
    }
    raw_batches: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch_index, batch in enumerate(val_loader, start=1):
            if batch_index > int(args.max_val_batches):
                break
            if batch_index == 1 or batch_index % 20 == 0:
                print(f"[stopline_rowx] eval batch {batch_index}/{args.max_val_batches}", flush=True)
            raw_batch = raw_batch_for_metrics(batch)
            if raw_batch is None:
                raise ValueError("validation loader must provide raw batches for metrics")
            raw_batches.append(raw_batch)

            encoded = evaluator.prepare_batch(batch)
            outputs = _detach_to_cpu(evaluator.forward_encoded_batch(encoded))
            meta_rows = _detach_to_cpu(encoded["meta"])
            baseline_predictions = postprocess_pv26_batch(outputs, meta_rows, config=postprocess_config)
            predictions_by_variant["baseline"].extend(baseline_predictions)

            for sample_index, (meta, baseline_prediction) in enumerate(zip(meta_rows, baseline_predictions)):
                for variant in VARIANTS:
                    stop_lines = _decode_rowx_stop_lines(
                        outputs=outputs,
                        sample_index=sample_index,
                        meta=meta,
                        config=postprocess_config,
                        variant=variant,
                    )
                    if stop_lines:
                        variant_stats[variant.name]["decoded_samples"] += 1
                        variant_stats[variant.name]["decoded_lines"] += len(stop_lines)
                    elif variant.fallback_baseline:
                        variant_stats[variant.name]["fallback_samples"] += 1
                        stop_lines = list(baseline_prediction.get("stop_lines", []))
                    predictions_by_variant[variant.name].append(_with_stop_lines(baseline_prediction, stop_lines))

    merged_raw = _merge_raw_batches(raw_batches)
    rows = []
    for name in names:
        metrics = augment_lane_family_metrics(summarize_pv26_metrics(predictions_by_variant[name], merged_raw))
        rows.append(_row_from_metrics(name, metrics))
    rows.sort(key=lambda row: float(row.get("phase4_objective_proxy", 0.0)), reverse=True)

    output = {
        "checkpoint": str(checkpoint),
        "processed_batches": int(min(len(raw_batches), int(args.max_val_batches))),
        "validation_epoch": int(args.validation_epoch),
        "variant_stats": variant_stats,
        "variants": rows,
    }
    if str(args.output_json).strip():
        output_path = Path(args.output_json).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"[stopline_rowx] wrote {output_path}", flush=True)
    print(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
