from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
import math
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
from model.data.transform import inverse_transform_points, transform_from_meta, transform_points
from model.engine import raw_batch_for_metrics
from model.engine.metrics import (
    STOP_LINE_POINT_COUNT,
    _extract_gt_samples,
    _hungarian_from_cost,
    _mean_point_distance,
    _segment_angle_error,
)
from model.engine.postprocess import (
    STOPLINE_MIN_ASPECT_RATIO,
    STOPLINE_MIN_COMPONENT_LENGTH,
    STOPLINE_MIN_COMPONENT_PIXELS,
    _fit_stopline_segment,
    _prepare_stopline_binary_mask,
    _stopline_component_anchor,
    postprocess_pv26_batch,
)
from tools.probe_pv26_lane60_postprocess_thresholds import _advance_validation_sampler, _detach_to_cpu
from tools.probe_pv26_lane60_support_gate import _load_scenario
from tools.pv26_train import cli as train_cli
from tools.pv26_train import config as train_config_api


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit whether stop-line GTs have usable predicted mask components before "
            "production center/anchor/readout filtering."
        )
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--preset", default="default")
    parser.add_argument("--source-run", default="")
    parser.add_argument("--lane60-experiment", default="")
    parser.add_argument("--phase-index", type=int, default=4)
    parser.add_argument("--max-val-batches", type=int, default=128)
    parser.add_argument("--validation-epoch", type=int, default=2)
    parser.add_argument("--train-batches", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gt-radius", type=float, default=2.5)
    parser.add_argument("--output-dir", default="")
    return parser.parse_args()


def _resolve_device(requested: str, scenario_device: str) -> str:
    value = str(requested or "auto").strip().lower()
    device = str(scenario_device) if value == "auto" else str(requested)
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[stopline_readout] CUDA requested but unavailable; falling back to CPU", flush=True)
        return "cpu"
    return device


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
    if array.ndim == 2:
        return np.asarray(array.max(axis=-1), dtype=np.float32)
    return None


def _x_probs(x_logits: torch.Tensor | None) -> np.ndarray | None:
    array = _as_numpy_2d(x_logits.sigmoid() if isinstance(x_logits, torch.Tensor) else None)
    if array is None:
        return None
    if array.ndim == 2:
        return np.asarray(array.max(axis=0), dtype=np.float32)
    return None


def _point_segment_distance(points: np.ndarray, segment: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    start = segment[0].astype(np.float32)
    end = segment[-1].astype(np.float32)
    axis = end - start
    length_sq = max(float(axis @ axis), 1.0e-6)
    raw_t = ((points - start[None, :]) @ axis) / length_sq
    t = np.clip(raw_t, 0.0, 1.0)
    closest = start[None, :] + t[:, None] * axis[None, :]
    distances = np.linalg.norm(points - closest, axis=1)
    return distances.astype(np.float32), raw_t.astype(np.float32)


def _line_tube_mask(shape_hw: tuple[int, int], segment: np.ndarray, *, radius: float) -> np.ndarray:
    height, width = shape_hw
    min_x = max(0, int(math.floor(float(segment[:, 0].min()) - radius - 1.0)))
    max_x = min(width - 1, int(math.ceil(float(segment[:, 0].max()) + radius + 1.0)))
    min_y = max(0, int(math.floor(float(segment[:, 1].min()) - radius - 1.0)))
    max_y = min(height - 1, int(math.ceil(float(segment[:, 1].max()) + radius + 1.0)))
    mask = np.zeros((height, width), dtype=bool)
    if max_x < min_x or max_y < min_y:
        return mask
    yy, xx = np.mgrid[min_y : max_y + 1, min_x : max_x + 1]
    points = np.stack([xx.astype(np.float32), yy.astype(np.float32)], axis=-1).reshape(-1, 2)
    distances, raw_t = _point_segment_distance(points, segment)
    local = (distances <= float(radius)) & (raw_t >= -0.05) & (raw_t <= 1.05)
    mask[min_y : max_y + 1, min_x : max_x + 1] = local.reshape((max_y - min_y + 1, max_x - min_x + 1))
    return mask


def _map_points_from_raw(points_xy: list[list[float]], meta: dict[str, Any], map_hw: tuple[int, int]) -> np.ndarray:
    transform = transform_from_meta(meta)
    network_points = np.asarray(transform_points(points_xy, transform), dtype=np.float32).reshape(-1, 2)
    map_h, map_w = map_hw
    network_h, network_w = transform.network_hw
    scaled = network_points.copy()
    scaled[:, 0] = scaled[:, 0] * (float(map_w) / float(network_w))
    scaled[:, 1] = scaled[:, 1] * (float(map_h) / float(network_h))
    return sample_stop_line_centerline(scaled, target_count=STOP_LINE_POINT_COUNT)


def _raw_points_from_map_segment(segment: np.ndarray, meta: dict[str, Any], map_hw: tuple[int, int]) -> list[list[float]]:
    transform = transform_from_meta(meta)
    map_h, map_w = map_hw
    network_h, network_w = transform.network_hw
    network_segment = np.asarray(segment, dtype=np.float32).reshape(-1, 2).copy()
    network_segment[:, 0] = (network_segment[:, 0] + 0.5) * (float(network_w) / float(map_w))
    network_segment[:, 1] = (network_segment[:, 1] + 0.5) * (float(network_h) / float(map_h))
    raw_points = inverse_transform_points(network_segment.tolist(), transform)
    return sample_stop_line_centerline(raw_points, target_count=STOP_LINE_POINT_COUNT).tolist()


def _fit_to_row(
    *,
    fitted: tuple[np.ndarray, np.ndarray, float, float] | None,
    gt_points: list[list[float]],
    meta: dict[str, Any],
    map_hw: tuple[int, int],
    prefix: str,
) -> dict[str, float | int]:
    if fitted is None:
        return {
            f"{prefix}_valid": 0,
            f"{prefix}_passes_geometry": 0,
            f"{prefix}_length": 0.0,
            f"{prefix}_thickness": 0.0,
            f"{prefix}_aspect": 0.0,
            f"{prefix}_mean_distance": 9999.0,
            f"{prefix}_angle_error": 180.0,
        }
    start, end, length, thickness = fitted
    segment = np.stack([start, end], axis=0).astype(np.float32)
    raw_points = _raw_points_from_map_segment(segment, meta, map_hw)
    aspect = float(length) / max(float(thickness), 1.0)
    return {
        f"{prefix}_valid": 1,
        f"{prefix}_passes_geometry": int(
            float(length) >= STOPLINE_MIN_COMPONENT_LENGTH
            and float(thickness) <= 12.0
            and aspect >= STOPLINE_MIN_ASPECT_RATIO
        ),
        f"{prefix}_length": float(length),
        f"{prefix}_thickness": float(thickness),
        f"{prefix}_aspect": aspect,
        f"{prefix}_mean_distance": float(_mean_point_distance(raw_points, gt_points, target_count=STOP_LINE_POINT_COUNT)),
        f"{prefix}_angle_error": float(_segment_angle_error(raw_points, gt_points, target_count=STOP_LINE_POINT_COUNT)),
    }


def _production_gt_matches(pred_stop_lines: list[dict[str, Any]], gt_stop_lines: list[dict[str, Any]]) -> dict[int, float]:
    if not pred_stop_lines or not gt_stop_lines:
        return {}
    cost = np.zeros((len(pred_stop_lines), len(gt_stop_lines)), dtype=np.float32)
    for pred_index, pred in enumerate(pred_stop_lines):
        for gt_index, gt in enumerate(gt_stop_lines):
            cost[pred_index, gt_index] = _mean_point_distance(
                pred["points_xy"],
                gt["points_xy"],
                target_count=STOP_LINE_POINT_COUNT,
            )
    matches = _hungarian_from_cost(cost, max_cost=40.0)
    return {int(gt_index): float(cost[pred_index, gt_index]) for pred_index, gt_index in matches}


def _best_component_for_gt(
    *,
    labels: np.ndarray,
    component_scores: np.ndarray,
    mask_probs: np.ndarray,
    gt_segment_map: np.ndarray,
    gt_tube: np.ndarray,
    radius: float,
) -> dict[str, Any] | None:
    component_count = int(component_scores.shape[0])
    best: dict[str, Any] | None = None
    for label_index in range(1, component_count + 1):
        rows, cols = np.nonzero(labels == label_index)
        if rows.size == 0:
            continue
        points = np.stack([cols.astype(np.float32), rows.astype(np.float32)], axis=1)
        distances, raw_t = _point_segment_distance(points, gt_segment_map)
        near_mask = (distances <= float(radius)) & (raw_t >= -0.05) & (raw_t <= 1.05)
        near_count = int(near_mask.sum())
        min_distance = float(distances.min()) if distances.size else 9999.0
        tube_hits = int(((labels == label_index) & gt_tube).sum())
        score_tuple = (
            near_count,
            tube_hits,
            -min_distance,
            float(component_scores[label_index - 1]),
            int(rows.size),
        )
        if best is None or score_tuple > best["score_tuple"]:
            best = {
                "label_index": int(label_index),
                "rows": rows,
                "cols": cols,
                "points": points,
                "score_tuple": score_tuple,
                "near_count": near_count,
                "tube_hits": tube_hits,
                "min_distance_map": min_distance,
                "pixel_count": int(rows.size),
                "component_score": float(component_scores[label_index - 1]),
                "mask_values": mask_probs[rows, cols],
            }
    return best


def _quantile(values: list[float], q: float) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return 0.0
    return float(np.quantile(np.asarray(finite, dtype=np.float32), q))


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    gt_count = len(rows)
    if gt_count == 0:
        return {"gt_count": 0}

    def count_if(name: str, predicate: Any) -> int:
        return sum(1 for row in rows if predicate(float(row.get(name, 0.0))))

    production_tp = sum(1 for row in rows if int(row.get("production_matched", 0)) == 1)
    anchorless_close = sum(1 for row in rows if float(row.get("no_anchor_mean_distance", 9999.0)) <= 40.0)
    anchored_close = sum(1 for row in rows if float(row.get("anchored_mean_distance", 9999.0)) <= 40.0)
    return {
        "gt_count": int(gt_count),
        "production_tp": int(production_tp),
        "production_recall": production_tp / gt_count,
        "gt_with_tube_mask_hit_frac_ge_0_10": count_if("gt_tube_hit_fraction", lambda value: value >= 0.10),
        "gt_with_tube_mask_hit_frac_ge_0_30": count_if("gt_tube_hit_fraction", lambda value: value >= 0.30),
        "gt_with_mask_max_ge_0_50": count_if("gt_tube_mask_max", lambda value: value >= 0.50),
        "gt_with_center_max_ge_0_50": count_if("gt_tube_center_max", lambda value: value >= 0.50),
        "anchorless_fit_close_count": int(anchorless_close),
        "anchorless_fit_close_recall": anchorless_close / gt_count,
        "anchored_fit_close_count": int(anchored_close),
        "anchored_fit_close_recall": anchored_close / gt_count,
        "median_gt_tube_mask_max": _quantile([float(row.get("gt_tube_mask_max", 0.0)) for row in rows], 0.5),
        "median_gt_tube_center_max": _quantile([float(row.get("gt_tube_center_max", 0.0)) for row in rows], 0.5),
        "median_anchor_error_map": _quantile([float(row.get("anchor_error_map", 9999.0)) for row in rows], 0.5),
        "median_no_anchor_distance": _quantile([float(row.get("no_anchor_mean_distance", 9999.0)) for row in rows], 0.5),
        "median_anchored_distance": _quantile([float(row.get("anchored_mean_distance", 9999.0)) for row in rows], 0.5),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
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
        else checkpoint.parents[2] / "analysis_exports" / "stopline_readout_component_audit"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    train_cli._configure_torch_multiprocessing()
    dataset = train_cli.PV26CanonicalDataset(
        train_cli._existing_dataset_roots(scenario),
        train_augmentation=False,
        progress_callback=lambda message: print(f"[stopline_readout] {message}", flush=True),
    )
    _, val_loader = train_cli._build_phase_train_loaders(dataset, train_config=train_config, phase=phase)
    if val_loader is None:
        raise ValueError("stop-line readout audit requires validation batches")
    _advance_validation_sampler(val_loader, validation_epoch=int(args.validation_epoch))

    trainer = train_cli._build_phase_trainer(phase, train_config)
    trainer.load_model_weights(checkpoint, map_location=train_config.device)
    evaluator = trainer.build_evaluator()
    evaluator.adapter.raw_model.eval()
    evaluator.heads.eval()

    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch_index, batch in enumerate(val_loader, start=1):
            if batch_index > int(args.max_val_batches):
                break
            if batch_index == 1 or batch_index % 20 == 0:
                print(f"[stopline_readout] eval batch {batch_index}/{args.max_val_batches}", flush=True)
            raw_batch = raw_batch_for_metrics(batch)
            if raw_batch is None:
                raise ValueError("validation loader must provide raw batches for metrics")

            encoded = evaluator.prepare_batch(batch)
            outputs = _detach_to_cpu(evaluator.forward_encoded_batch(encoded))
            meta_rows = _detach_to_cpu(encoded["meta"])
            batch_predictions = postprocess_pv26_batch(outputs, meta_rows, config=postprocess_config)

            gt_samples = _extract_gt_samples(raw_batch)
            mask_batch = outputs.get("stop_line_mask_logits")
            if not isinstance(mask_batch, torch.Tensor):
                continue
            for sample_offset, (meta, gt_sample, pred_sample) in enumerate(zip(meta_rows, gt_samples, batch_predictions)):
                sample_global_index = (batch_index - 1) * int(train_config.batch_size) + sample_offset
                mask_logits = mask_batch[sample_offset]
                mask_probs = _as_numpy_2d(mask_logits.sigmoid())
                if mask_probs is None:
                    continue
                binary = _prepare_stopline_binary_mask(
                    mask_probs,
                    threshold=float(postprocess_config.stop_line_mask_binary_threshold),
                )
                labels, component_count = ndimage.label(binary)
                component_scores = (
                    ndimage.maximum(mask_probs, labels, index=np.arange(1, int(component_count) + 1))
                    if int(component_count) > 0
                    else np.zeros((0,), dtype=np.float32)
                )
                center_logits = outputs.get("stop_line_center_logits")
                selector_logits = outputs.get("stop_line_selector_map_logits")
                row_logits = outputs.get("stop_line_row_logits")
                x_logits = outputs.get("stop_line_x_logits")
                center_offset = outputs.get("stop_line_center_offset")
                center_probs = (
                    _as_numpy_2d(center_logits[sample_offset].sigmoid())
                    if isinstance(center_logits, torch.Tensor)
                    else None
                )
                selector_probs = (
                    _as_numpy_2d(selector_logits[sample_offset].sigmoid())
                    if isinstance(selector_logits, torch.Tensor)
                    else None
                )
                sample_row_probs = _row_probs(row_logits[sample_offset]) if isinstance(row_logits, torch.Tensor) else None
                sample_x_probs = _x_probs(x_logits[sample_offset]) if isinstance(x_logits, torch.Tensor) else None
                sample_center_offset = center_offset[sample_offset] if isinstance(center_offset, torch.Tensor) else None
                production_matches = _production_gt_matches(
                    list(pred_sample.get("stop_lines", [])),
                    list(gt_sample.get("stop_lines", [])),
                )

                for gt_index, gt_stop_line in enumerate(gt_sample.get("stop_lines", [])):
                    gt_points = [[float(x), float(y)] for x, y in gt_stop_line.get("points_xy", [])]
                    gt_map = _map_points_from_raw(gt_points, meta, mask_probs.shape)
                    gt_segment = np.stack([gt_map[0], gt_map[-1]], axis=0).astype(np.float32)
                    gt_tube = _line_tube_mask(mask_probs.shape, gt_segment, radius=float(args.gt_radius))
                    tube_values = mask_probs[gt_tube] if bool(gt_tube.any()) else np.asarray([], dtype=np.float32)
                    center_values = (
                        center_probs[gt_tube]
                        if center_probs is not None and center_probs.shape == mask_probs.shape and bool(gt_tube.any())
                        else np.asarray([], dtype=np.float32)
                    )
                    selector_values = (
                        selector_probs[gt_tube]
                        if selector_probs is not None and selector_probs.shape == mask_probs.shape and bool(gt_tube.any())
                        else np.asarray([], dtype=np.float32)
                    )
                    best = _best_component_for_gt(
                        labels=labels,
                        component_scores=component_scores,
                        mask_probs=mask_probs,
                        gt_segment_map=gt_segment,
                        gt_tube=gt_tube,
                        radius=float(args.gt_radius),
                    )
                    row: dict[str, Any] = {
                        "sample_index": int(sample_global_index),
                        "batch_index": int(batch_index),
                        "sample_offset": int(sample_offset),
                        "sample_id": str(meta.get("sample_id", "")),
                        "gt_index": int(gt_index),
                        "component_count": int(component_count),
                        "production_matched": int(gt_index in production_matches),
                        "production_match_distance": float(production_matches.get(gt_index, 9999.0)),
                        "gt_tube_pixels": int(gt_tube.sum()),
                        "gt_tube_mask_max": float(tube_values.max()) if tube_values.size else 0.0,
                        "gt_tube_mask_mean": float(tube_values.mean()) if tube_values.size else 0.0,
                        "gt_tube_mask_frac_ge_binary": float(
                            (tube_values >= float(postprocess_config.stop_line_mask_binary_threshold)).mean()
                        )
                        if tube_values.size
                        else 0.0,
                        "gt_tube_center_max": float(center_values.max()) if center_values.size else 0.0,
                        "gt_tube_selector_max": float(selector_values.max()) if selector_values.size else 0.0,
                    }
                    if best is None:
                        row.update(
                            {
                                "best_label": 0,
                                "best_component_pixels": 0,
                                "best_component_score": 0.0,
                                "best_component_near_pixels": 0,
                                "best_component_min_distance_map": 9999.0,
                                "gt_tube_component_hits": 0,
                                "gt_tube_hit_fraction": 0.0,
                                "anchor_error_map": 9999.0,
                                "anchor_row_prob": 0.0,
                                "anchor_x_prob": 0.0,
                                "anchor_center_prob": 0.0,
                            }
                        )
                        row.update(_fit_to_row(fitted=None, gt_points=gt_points, meta=meta, map_hw=mask_probs.shape, prefix="no_anchor"))
                        row.update(_fit_to_row(fitted=None, gt_points=gt_points, meta=meta, map_hw=mask_probs.shape, prefix="anchored"))
                        rows.append(row)
                        continue

                    rows_np = best["rows"]
                    cols_np = best["cols"]
                    mask_values = best["mask_values"]
                    anchor = None
                    if center_probs is not None:
                        anchor = _stopline_component_anchor(
                            rows_np,
                            cols_np,
                            mask_values=mask_values,
                            center_probs=center_probs,
                            center_offset=sample_center_offset,
                            row_probs=sample_row_probs,
                            x_probs=sample_x_probs,
                        )
                    gt_midpoint = gt_segment.mean(axis=0)
                    anchor_error = float(np.linalg.norm(anchor - gt_midpoint)) if anchor is not None else 9999.0
                    anchor_row = int(round(float(anchor[1]))) if anchor is not None else -1
                    anchor_col = int(round(float(anchor[0]))) if anchor is not None else -1
                    anchor_row_prob = (
                        float(sample_row_probs[anchor_row])
                        if sample_row_probs is not None and 0 <= anchor_row < int(sample_row_probs.shape[0])
                        else 0.0
                    )
                    anchor_x_prob = (
                        float(sample_x_probs[anchor_col])
                        if sample_x_probs is not None and 0 <= anchor_col < int(sample_x_probs.shape[0])
                        else 0.0
                    )
                    anchor_center_prob = (
                        float(center_probs[anchor_row, anchor_col])
                        if center_probs is not None
                        and 0 <= anchor_row < int(center_probs.shape[0])
                        and 0 <= anchor_col < int(center_probs.shape[1])
                        else 0.0
                    )
                    row.update(
                        {
                            "best_label": int(best["label_index"]),
                            "best_component_pixels": int(best["pixel_count"]),
                            "best_component_score": float(best["component_score"]),
                            "best_component_near_pixels": int(best["near_count"]),
                            "best_component_min_distance_map": float(best["min_distance_map"]),
                            "gt_tube_component_hits": int(best["tube_hits"]),
                            "gt_tube_hit_fraction": float(best["tube_hits"]) / max(float(gt_tube.sum()), 1.0),
                            "anchor_error_map": anchor_error,
                            "anchor_row_prob": anchor_row_prob,
                            "anchor_x_prob": anchor_x_prob,
                            "anchor_center_prob": anchor_center_prob,
                        }
                    )
                    no_anchor_fit = _fit_stopline_segment(
                        best["points"],
                        mask_values=mask_values,
                        center_anchor=None,
                    )
                    anchored_fit = _fit_stopline_segment(
                        best["points"],
                        mask_values=mask_values,
                        center_anchor=anchor,
                    )
                    row.update(
                        _fit_to_row(
                            fitted=no_anchor_fit,
                            gt_points=gt_points,
                            meta=meta,
                            map_hw=mask_probs.shape,
                            prefix="no_anchor",
                        )
                    )
                    row.update(
                        _fit_to_row(
                            fitted=anchored_fit,
                            gt_points=gt_points,
                            meta=meta,
                            map_hw=mask_probs.shape,
                            prefix="anchored",
                        )
                    )
                    rows.append(row)

    summary = {
        "checkpoint": str(checkpoint),
        "scenario_path": str(scenario_path),
        "phase_name": phase.name,
        "phase_stage": phase.stage,
        "max_val_batches": int(args.max_val_batches),
        "validation_epoch": int(args.validation_epoch),
        "gt_radius": float(args.gt_radius),
        "postprocess_config": vars(postprocess_config),
        "summary": _summarize(rows),
    }
    _write_csv(output_dir / "stopline_readout_gt_rows.csv", rows)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary["summary"], ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    print(f"[stopline_readout] wrote {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
