from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, replace
import json
from pathlib import Path
import site
import sys
from typing import Any

import numpy as np
from scipy import ndimage
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
repo_root = str(REPO_ROOT)
if repo_root not in sys.path:
    site.addsitedir(repo_root)

from common.geometry import sample_stop_line_centerline
from model.data.transform import inverse_transform_points, transform_from_meta, transform_points
from model.engine._trainer_epochs import _merge_raw_batches
from model.engine.batch import augment_lane_family_metrics, raw_batch_for_metrics
from model.engine.metrics import STOP_LINE_POINT_COUNT, _extract_gt_samples, summarize_pv26_metrics
from model.engine.postprocess import _prepare_stopline_binary_mask, postprocess_pv26_batch
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
class ExtentVariant:
    name: str
    center_source: str
    angle_source: str
    mask_threshold: float
    normal_band: float
    search_radius: float = 8.0


VARIANTS = (
    ExtentVariant(
        "gt_center_gt_angle_mask_thr050_band4",
        center_source="gt_center",
        angle_source="gt_angle",
        mask_threshold=0.50,
        normal_band=4.0,
    ),
    ExtentVariant(
        "gt_center_pred_angle_mask_thr050_band4",
        center_source="gt_center",
        angle_source="pred_angle",
        mask_threshold=0.50,
        normal_band=4.0,
    ),
    ExtentVariant(
        "gt_cell_pred_offset_pred_angle_mask_thr050_band4",
        center_source="pred_offset",
        angle_source="pred_angle",
        mask_threshold=0.50,
        normal_band=4.0,
    ),
    ExtentVariant(
        "gt_center_pred_angle_mask_thr080_band4",
        center_source="gt_center",
        angle_source="pred_angle",
        mask_threshold=0.80,
        normal_band=4.0,
    ),
    ExtentVariant(
        "gt_center_pred_angle_mask_thr050_full",
        center_source="gt_center",
        angle_source="pred_angle",
        mask_threshold=0.50,
        normal_band=0.0,
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Probe whether predicted stop-line masks contain enough extent to recover line length "
            "when GT center and predicted/GT angle are supplied as a read-only diagnostic."
        )
    )
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--preset", default="default")
    parser.add_argument("--phase-index", type=int, default=4)
    parser.add_argument("--max-val-batches", type=int, default=128)
    parser.add_argument("--validation-epoch", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", default="")
    return parser.parse_args()


def _resolve_device(requested: str, scenario_device: str) -> str:
    value = str(requested or "auto").strip().lower()
    candidate = str(scenario_device) if value == "auto" else str(requested)
    if candidate.startswith("cuda") and not torch.cuda.is_available():
        print("[stopline_angle_mask_extent] CUDA requested but unavailable; falling back to CPU", flush=True)
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


def _sample_tensor(predictions: dict[str, Any], key: str, sample_index: int) -> torch.Tensor | None:
    value = predictions.get(key)
    if not isinstance(value, torch.Tensor):
        return None
    sample = value[sample_index].detach().cpu()
    if not bool(torch.isfinite(sample).all()):
        return None
    return sample


def _as_2d_array(tensor: torch.Tensor | None, *, sigmoid: bool = False) -> np.ndarray | None:
    if not isinstance(tensor, torch.Tensor):
        return None
    value = tensor.sigmoid() if sigmoid else tensor
    array = value.detach().cpu().numpy()
    while array.ndim > 2 and array.shape[0] == 1:
        array = array.squeeze(0)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array.squeeze(0)
    if array.ndim != 2:
        return None
    return np.asarray(array, dtype=np.float32)


def _gt_descriptor(stop_line: dict[str, Any], meta: dict[str, Any], *, output_hw: tuple[int, int]) -> dict[str, Any] | None:
    raw_points = np.asarray(stop_line.get("points_xy", []), dtype=np.float32)
    if raw_points.ndim != 2 or raw_points.shape[0] < 2 or raw_points.shape[1] != 2:
        return None
    transform = transform_from_meta(meta)
    network_points = np.asarray(
        sample_stop_line_centerline(
            transform_points(raw_points.tolist(), transform),
            target_count=STOP_LINE_POINT_COUNT,
        ),
        dtype=np.float32,
    )
    if network_points.shape[0] < 2 or not bool(np.isfinite(network_points).all()):
        return None
    output_h, output_w = int(output_hw[0]), int(output_hw[1])
    network_h, network_w = int(meta["network_hw"][0]), int(meta["network_hw"][1])
    scaled = network_points.copy()
    scaled[:, 0] *= float(output_w) / float(network_w)
    scaled[:, 1] *= float(output_h) / float(network_h)
    start = scaled[0]
    end = scaled[-1]
    center = (start + end) * 0.5
    col = max(0, min(output_w - 1, int(np.floor(float(center[0])))))
    row = max(0, min(output_h - 1, int(np.floor(float(center[1])))))
    delta = end - start
    norm = float(np.linalg.norm(delta))
    if norm <= 1.0e-6:
        return None
    return {
        "center": center.astype(np.float32),
        "row": int(row),
        "col": int(col),
        "angle": (delta / norm).astype(np.float32),
    }


def _raw_line_from_map_extent(
    *,
    center_xy: np.ndarray,
    angle_vec: np.ndarray,
    start_proj: float,
    end_proj: float,
    meta: dict[str, Any],
    output_hw: tuple[int, int],
    score: float,
) -> dict[str, Any] | None:
    angle_vec = np.asarray(angle_vec, dtype=np.float32).reshape(2)
    norm = float(np.linalg.norm(angle_vec))
    if norm <= 1.0e-6 or not np.isfinite(norm):
        return None
    angle_vec = angle_vec / norm
    start_proj = float(start_proj)
    end_proj = float(end_proj)
    if not np.isfinite(start_proj) or not np.isfinite(end_proj) or end_proj - start_proj <= 1.0:
        return None
    center_xy = np.asarray(center_xy, dtype=np.float32).reshape(2)
    start = center_xy + angle_vec * start_proj
    end = center_xy + angle_vec * end_proj
    output_h, output_w = int(output_hw[0]), int(output_hw[1])
    network_segment = np.stack([start, end], axis=0).astype(np.float32)
    network_segment[:, 0] = (network_segment[:, 0] + 0.5) * (float(meta["network_hw"][1]) / float(output_w))
    network_segment[:, 1] = (network_segment[:, 1] + 0.5) * (float(meta["network_hw"][0]) / float(output_h))
    transform = transform_from_meta(meta)
    network_points = sample_stop_line_centerline(network_segment.tolist(), target_count=STOP_LINE_POINT_COUNT)
    raw_points = sample_stop_line_centerline(
        inverse_transform_points(network_points.tolist(), transform),
        target_count=STOP_LINE_POINT_COUNT,
    )
    if raw_points.shape[0] < 2 or not bool(np.isfinite(raw_points).all()):
        return None
    return {
        "score": float(score),
        "center_score": float(score),
        "length": float(end_proj - start_proj),
        "points_xy": [[float(x), float(y)] for x, y in raw_points.tolist()],
    }


def _nearest_component_label(
    labels: np.ndarray,
    *,
    center_xy: np.ndarray,
    search_radius: float,
) -> int:
    output_h, output_w = labels.shape
    row = max(0, min(output_h - 1, int(round(float(center_xy[1])))))
    col = max(0, min(output_w - 1, int(round(float(center_xy[0])))))
    center_label = int(labels[row, col])
    if center_label > 0:
        return center_label
    rows, cols = np.nonzero(labels > 0)
    if rows.size == 0:
        return 0
    distances = np.sqrt((cols.astype(np.float32) - float(center_xy[0])) ** 2 + (rows.astype(np.float32) - float(center_xy[1])) ** 2)
    best_index = int(np.argmin(distances))
    if float(distances[best_index]) > float(search_radius):
        return 0
    return int(labels[int(rows[best_index]), int(cols[best_index])])


def _mask_extent_line(
    *,
    mask_probs: np.ndarray,
    center_xy: np.ndarray,
    angle_vec: np.ndarray,
    meta: dict[str, Any],
    output_hw: tuple[int, int],
    variant: ExtentVariant,
    score: float,
) -> tuple[dict[str, Any] | None, str]:
    binary = _prepare_stopline_binary_mask(mask_probs, threshold=float(variant.mask_threshold))
    labels, _ = ndimage.label(binary)
    label = _nearest_component_label(labels, center_xy=center_xy, search_radius=float(variant.search_radius))
    if label <= 0:
        return None, "component_miss"
    rows, cols = np.nonzero(labels == label)
    if rows.size < 2:
        return None, "too_few_component_pixels"
    points = np.stack([cols.astype(np.float32), rows.astype(np.float32)], axis=1)
    angle_vec = np.asarray(angle_vec, dtype=np.float32).reshape(2)
    norm = float(np.linalg.norm(angle_vec))
    if norm <= 1.0e-6 or not np.isfinite(norm):
        return None, "bad_angle"
    axis = angle_vec / norm
    normal = np.array([-axis[1], axis[0]], dtype=np.float32)
    offsets = points - np.asarray(center_xy, dtype=np.float32).reshape(1, 2)
    if float(variant.normal_band) > 0.0:
        local_mask = np.abs(offsets @ normal) <= float(variant.normal_band)
        if int(local_mask.sum()) >= 2:
            offsets = offsets[local_mask]
    if offsets.shape[0] < 2:
        return None, "too_few_extent_pixels"
    projections = offsets @ axis
    if projections.size < 2:
        return None, "empty_projection"
    if projections.size >= 8:
        start_proj = float(np.quantile(projections, 0.05))
        end_proj = float(np.quantile(projections, 0.95))
    else:
        start_proj = float(projections.min())
        end_proj = float(projections.max())
    candidate = _raw_line_from_map_extent(
        center_xy=center_xy,
        angle_vec=axis,
        start_proj=start_proj,
        end_proj=end_proj,
        meta=meta,
        output_hw=output_hw,
        score=score,
    )
    if candidate is None:
        return None, "invalid_line"
    return candidate, "ok"


def _variant_stop_lines(
    gt_sample: dict[str, Any],
    sample_predictions: dict[str, Any],
    *,
    sample_index: int,
    variant: ExtentVariant,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    mask_probs = _as_2d_array(_sample_tensor(sample_predictions, "stop_line_mask_logits", sample_index), sigmoid=True)
    angle_map = _sample_tensor(sample_predictions, "stop_line_angle", sample_index)
    offset_map = _sample_tensor(sample_predictions, "stop_line_center_offset", sample_index)
    center_logits = _sample_tensor(sample_predictions, "stop_line_center_logits", sample_index)
    if mask_probs is None or angle_map is None or offset_map is None:
        return [], {"missing_tensor": int(len(gt_sample.get("stop_lines", [])))}
    if angle_map.ndim != 3 or offset_map.ndim != 3:
        return [], {"bad_tensor_shape": int(len(gt_sample.get("stop_lines", [])))}
    output_hw = (int(mask_probs.shape[0]), int(mask_probs.shape[1]))
    meta = dict(gt_sample.get("meta", {}))
    stats: dict[str, int] = {}
    output: list[dict[str, Any]] = []
    for stop_line in gt_sample.get("stop_lines", []):
        gt = _gt_descriptor(stop_line, meta, output_hw=output_hw)
        if gt is None:
            stats["bad_gt"] = stats.get("bad_gt", 0) + 1
            continue
        row = int(gt["row"])
        col = int(gt["col"])
        pred_offset = offset_map[:, row, col].numpy().astype(np.float32)
        pred_angle = angle_map[:, row, col].numpy().astype(np.float32)
        if variant.center_source == "gt_center":
            center_xy = np.asarray(gt["center"], dtype=np.float32)
        elif variant.center_source == "pred_offset":
            center_xy = np.asarray([float(col) + pred_offset[0], float(row) + pred_offset[1]], dtype=np.float32)
        else:
            raise ValueError(f"unsupported center source: {variant.center_source}")
        if variant.angle_source == "gt_angle":
            angle_vec = np.asarray(gt["angle"], dtype=np.float32)
        elif variant.angle_source == "pred_angle":
            angle_vec = pred_angle
        else:
            raise ValueError(f"unsupported angle source: {variant.angle_source}")
        score = 1.0
        if isinstance(center_logits, torch.Tensor) and center_logits.ndim == 3:
            score = float(center_logits[0, row, col].sigmoid().item())
        candidate, reason = _mask_extent_line(
            mask_probs=mask_probs,
            center_xy=center_xy,
            angle_vec=angle_vec,
            meta=meta,
            output_hw=output_hw,
            variant=variant,
            score=score,
        )
        stats[reason] = stats.get(reason, 0) + 1
        if candidate is not None:
            output.append(candidate)
    return output, stats


def _replace_stop_lines(
    base_samples: list[dict[str, Any]],
    gt_samples: list[dict[str, Any]],
    raw_predictions_by_batch: list[dict[str, Any]],
    batch_sizes: list[int],
    *,
    variant: ExtentVariant,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    if len(base_samples) != len(gt_samples):
        raise ValueError(f"prediction/GT length mismatch: {len(base_samples)} vs {len(gt_samples)}")
    output: list[dict[str, Any]] = []
    totals: dict[str, int] = {}
    global_index = 0
    for batch_predictions, batch_size in zip(raw_predictions_by_batch, batch_sizes):
        for sample_index in range(int(batch_size)):
            base_sample = base_samples[global_index]
            gt_sample = gt_samples[global_index]
            stop_lines, stats = _variant_stop_lines(
                gt_sample,
                batch_predictions,
                sample_index=sample_index,
                variant=variant,
            )
            for key, value in stats.items():
                totals[key] = totals.get(key, 0) + int(value)
            candidate = dict(base_sample)
            candidate["stop_lines"] = stop_lines
            output.append(candidate)
            global_index += 1
    return output, totals


def _replace_stop_lines_for_batch(
    base_samples: list[dict[str, Any]],
    gt_samples: list[dict[str, Any]],
    batch_predictions: dict[str, Any],
    *,
    variant: ExtentVariant,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    if len(base_samples) != len(gt_samples):
        raise ValueError(f"batch prediction/GT length mismatch: {len(base_samples)} vs {len(gt_samples)}")
    output: list[dict[str, Any]] = []
    totals: dict[str, int] = {}
    for sample_index, (base_sample, gt_sample) in enumerate(zip(base_samples, gt_samples)):
        stop_lines, stats = _variant_stop_lines(
            gt_sample,
            batch_predictions,
            sample_index=sample_index,
            variant=variant,
        )
        for key, value in stats.items():
            totals[key] = totals.get(key, 0) + int(value)
        candidate = dict(base_sample)
        candidate["stop_lines"] = stop_lines
        output.append(candidate)
    return output, totals


def _row_from_metrics(
    variant: str,
    metrics: dict[str, Any],
    *,
    prediction_count: int,
    stats: dict[str, int] | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "variant": variant,
        "pred_stop_line_count": int(prediction_count),
    }
    if stats:
        for key, value in sorted(stats.items()):
            row[f"stat_{key}"] = int(value)
    for task in ("lane", "stop_line", "crosswalk"):
        values = metrics.get(task, {}) if isinstance(metrics.get(task), dict) else {}
        for key in (
            "precision",
            "recall",
            "f1",
            "tp",
            "fp",
            "fn",
            "mean_point_distance",
            "mean_angle_error",
            "mean_polygon_iou",
        ):
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


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if str(args.output_dir).strip()
        else checkpoint.parents[2] / "analysis_exports" / "stopline_angle_mask_extent_val128_epoch2"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    train_cli._configure_torch_multiprocessing()
    dataset = train_cli.PV26CanonicalDataset(
        train_cli._existing_dataset_roots(scenario),
        train_augmentation=False,
        progress_callback=lambda message: print(f"[stopline_angle_mask_extent] {message}", flush=True),
    )
    _, val_loader = train_cli._build_phase_train_loaders(dataset, train_config=train_config, phase=phase)
    if val_loader is None:
        raise ValueError("stop-line angle/mask extent probe requires validation batches")
    _advance_validation_sampler(val_loader, validation_epoch=int(args.validation_epoch))

    trainer = train_cli._build_phase_trainer(phase, train_config)
    trainer.load_model_weights(checkpoint, map_location=train_config.device)
    evaluator = trainer.build_evaluator()
    evaluator.adapter.raw_model.eval()
    evaluator.heads.eval()

    predictions_by_variant: dict[str, list[dict[str, Any]]] = {"baseline": []}
    stats_by_variant: dict[str, dict[str, int]] = {"baseline": {}}
    for variant in VARIANTS:
        predictions_by_variant[variant.name] = []
        stats_by_variant[variant.name] = {}
    raw_batches: list[dict[str, Any]] = []
    processed_batches = 0
    with torch.no_grad():
        for batch_index, batch in enumerate(val_loader, start=1):
            if batch_index > int(args.max_val_batches):
                break
            if batch_index == 1 or batch_index % 20 == 0:
                print(f"[stopline_angle_mask_extent] eval batch {batch_index}/{args.max_val_batches}", flush=True)
            raw_batch = raw_batch_for_metrics(batch)
            if raw_batch is None:
                raise ValueError("validation loader must provide raw batches for metrics")
            raw_batches.append(raw_batch)
            gt_samples = _extract_gt_samples(raw_batch)
            encoded = evaluator.prepare_batch(batch)
            raw_predictions = _detach_to_cpu(evaluator.forward_encoded_batch(encoded))
            meta_batch = _detach_to_cpu(encoded["meta"])
            baseline_predictions = postprocess_pv26_batch(raw_predictions, meta_batch, config=postprocess_config)
            predictions_by_variant["baseline"].extend(baseline_predictions)
            for variant in VARIANTS:
                variant_predictions, stats = _replace_stop_lines_for_batch(
                    baseline_predictions,
                    gt_samples,
                    raw_predictions,
                    variant=variant,
                )
                predictions_by_variant[variant.name].extend(variant_predictions)
                totals = stats_by_variant[variant.name]
                for key, value in stats.items():
                    totals[key] = totals.get(key, 0) + int(value)
            processed_batches += 1

    merged_raw = _merge_raw_batches(raw_batches)
    rows: list[dict[str, Any]] = []
    for name, predictions in predictions_by_variant.items():
        metrics = augment_lane_family_metrics(summarize_pv26_metrics(predictions, merged_raw))
        prediction_count = sum(len(sample.get("stop_lines", [])) for sample in predictions)
        rows.append(
            _row_from_metrics(
                name,
                metrics,
                prediction_count=prediction_count,
                stats=stats_by_variant.get(name, {}),
            )
        )
    rows.sort(
        key=lambda row: (
            float(row.get("stop_line_f1", 0.0)),
            float(row.get("phase4_objective_proxy", 0.0)),
        ),
        reverse=True,
    )
    _write_csv(output_dir / "variants.csv", rows)
    summary = {
        "checkpoint": str(checkpoint),
        "phase_name": phase.name,
        "phase_stage": phase.stage,
        "max_val_batches": int(args.max_val_batches),
        "validation_epoch": int(args.validation_epoch),
        "processed_batches": int(processed_batches),
        "postprocess_config": vars(postprocess_config),
        "variants": rows,
        "interpretation": (
            "GT-center mask-extent variants are read-only diagnostics. They use GT stop-line centers "
            "to isolate whether predicted masks contain metric-compatible length along predicted or GT angle."
        ),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(rows, ensure_ascii=False, indent=2), flush=True)
    print(f"[stopline_angle_mask_extent] wrote {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
