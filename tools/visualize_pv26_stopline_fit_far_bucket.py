from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import replace
from pathlib import Path
import site
import sys
from typing import Any

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:  # pragma: no cover
    Image = None
    ImageDraw = None
    ImageFont = None

import numpy as np
import torch
from scipy import ndimage

REPO_ROOT = Path(__file__).resolve().parents[1]
repo_root = str(REPO_ROOT)
if repo_root not in sys.path:
    site.addsitedir(repo_root)

from common.geometry import sample_stop_line_centerline
from common.overlay import render_overlay
from model.engine import raw_batch_for_metrics
from model.engine.metrics import _extract_gt_samples
from model.engine.postprocess import (
    _fit_stopline_segment,
    _prepare_stopline_binary_mask,
    _stopline_component_anchor,
    postprocess_pv26_batch,
)
from tools.probe_pv26_lane60_postprocess_thresholds import _advance_validation_sampler, _detach_to_cpu
from tools.probe_pv26_stopline_readout_components import (
    _as_numpy_2d,
    _best_component_for_gt,
    _line_tube_mask,
    _map_points_from_raw,
    _raw_points_from_map_segment,
    _row_probs,
    _x_probs,
)
from tools.probe_pv26_lane60_support_gate import _load_scenario
from tools.pv26_train import cli as train_cli
from tools.pv26_train import config as train_config_api
from tools.pv26_train.epoch_visualization import _gt_scene_from_sample, _prediction_to_overlay_scene


SOURCE_RUN = (
    REPO_ROOT
    / "runs"
    / "pv26_exhaustive_od_lane_train"
    / "lane60_core_centerline_refine_cross_retain_from_exhaustive_od_lane_default_20260505_032217_default_20260510_003412"
)
DEFAULT_CHECKPOINT = SOURCE_RUN / "phase_4" / "checkpoints" / "best.pt"
DEFAULT_AUDIT_DIR = SOURCE_RUN / "analysis_exports" / "stopline_readout_component_audit_val512_epoch2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render stop-line FN samples where dense mask/center signal exists but "
            "component fitting remains far from the GT line."
        )
    )
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--source-run", default=str(SOURCE_RUN))
    parser.add_argument("--audit-csv", default=str(DEFAULT_AUDIT_DIR / "stopline_readout_gt_rows.csv"))
    parser.add_argument("--preset", default="default")
    parser.add_argument("--lane60-experiment", default="core_centerline_refine_cross_retain")
    parser.add_argument("--phase-index", type=int, default=4)
    parser.add_argument("--max-val-batches", type=int, default=512)
    parser.add_argument("--validation-epoch", type=int, default=2)
    parser.add_argument("--train-batches", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sample-count", type=int, default=18)
    parser.add_argument("--columns", type=int, default=3)
    parser.add_argument("--mask-min", type=float, default=0.50)
    parser.add_argument("--center-min", type=float, default=0.50)
    parser.add_argument("--distance-min", type=float, default=40.0)
    parser.add_argument("--sort", choices=("hit-distance", "distance", "sample-index"), default="hit-distance")
    parser.add_argument(
        "--output-dir",
        default=str(SOURCE_RUN / "analysis_exports" / "stopline_fit_far_visual_audit_val512_epoch2"),
    )
    return parser.parse_args()


def _resolve_device(requested: str, scenario_device: str) -> str:
    value = str(requested or "auto").strip().lower()
    device = str(scenario_device) if value == "auto" else str(requested)
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[stopline_fit_far_visual] CUDA requested but unavailable; falling back to CPU", flush=True)
        return "cpu"
    return device


def _as_float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return float(default)


def _as_int(row: dict[str, str], key: str, default: int = 0) -> int:
    try:
        return int(float(row.get(key, default)))
    except (TypeError, ValueError):
        return int(default)


def _load_bucket_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    audit_csv = Path(args.audit_csv).expanduser().resolve()
    if not audit_csv.is_file():
        raise FileNotFoundError(f"audit CSV not found: {audit_csv}")
    rows: list[dict[str, Any]] = []
    with audit_csv.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if _as_int(row, "production_matched") != 0:
                continue
            if _as_float(row, "gt_tube_mask_max") < float(args.mask_min):
                continue
            if _as_float(row, "gt_tube_center_max") < float(args.center_min):
                continue
            if _as_float(row, "no_anchor_mean_distance", 9999.0) <= float(args.distance_min):
                continue
            rows.append({**row})

    if args.sort == "hit-distance":
        rows.sort(
            key=lambda row: (
                -_as_float(row, "gt_tube_hit_fraction"),
                -_as_float(row, "no_anchor_mean_distance"),
                _as_int(row, "sample_index"),
                _as_int(row, "gt_index"),
            )
        )
    elif args.sort == "distance":
        rows.sort(
            key=lambda row: (
                -_as_float(row, "no_anchor_mean_distance"),
                -_as_float(row, "gt_tube_hit_fraction"),
                _as_int(row, "sample_index"),
                _as_int(row, "gt_index"),
            )
        )
    else:
        rows.sort(key=lambda row: (_as_int(row, "sample_index"), _as_int(row, "gt_index")))

    limited = rows[: max(0, int(args.sample_count))]
    return [
        {
            **row,
            "sample_index_int": _as_int(row, "sample_index"),
            "gt_index_int": _as_int(row, "gt_index"),
            "batch_index_int": _as_int(row, "batch_index"),
            "sample_offset_int": _as_int(row, "sample_offset"),
            "gt_tube_hit_fraction_float": _as_float(row, "gt_tube_hit_fraction"),
            "no_anchor_mean_distance_float": _as_float(row, "no_anchor_mean_distance", 9999.0),
            "anchored_mean_distance_float": _as_float(row, "anchored_mean_distance", 9999.0),
        }
        for row in limited
    ]


def _sample_from_raw(raw_batch: dict[str, Any], index: int) -> dict[str, Any]:
    sample = {
        "det_targets": raw_batch["det_targets"][index],
        "tl_attr_targets": raw_batch["tl_attr_targets"][index],
        "lane_targets": raw_batch["lane_targets"][index],
        "source_mask": raw_batch["source_mask"][index],
        "valid_mask": raw_batch["valid_mask"][index],
        "meta": raw_batch["meta"][index],
    }
    if "image" in raw_batch and isinstance(raw_batch["image"], torch.Tensor):
        sample["image"] = raw_batch["image"][index]
    return sample


def _safe_id(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)[:120]


def _draw_polyline(draw: Any, points: list[list[float]], *, fill: tuple[int, int, int], width: int) -> None:
    if len(points) < 2:
        return
    draw.line([(float(x), float(y)) for x, y in points], fill=fill, width=max(1, int(width)))


def _label(draw: Any, xy: tuple[float, float], text: str, *, fill: tuple[int, int, int], font: Any) -> None:
    x_value, y_value = xy
    draw.text((max(0.0, float(x_value)), max(0.0, float(y_value))), text, fill=fill, font=font)


def _heat_overlay(base: Any, channels: list[tuple[np.ndarray, tuple[int, int, int], float]]) -> Any:
    if Image is None:
        raise RuntimeError("Pillow is required for stop-line fit-far visuals")
    base_rgb = base.convert("RGB")
    width, height = base_rgb.size
    composite = np.asarray(base_rgb).astype(np.float32)
    for values, color, alpha in channels:
        heat = np.clip(np.asarray(values, dtype=np.float32), 0.0, 1.0)
        heat_img = Image.fromarray(np.uint8(np.round(heat * 255.0)), mode="L").resize((width, height))
        heat_arr = np.asarray(heat_img).astype(np.float32) / 255.0
        color_arr = np.asarray(color, dtype=np.float32)
        local_alpha = heat_arr[..., None] * float(alpha)
        composite = composite * (1.0 - local_alpha) + color_arr[None, None, :] * local_alpha
    return Image.fromarray(np.uint8(np.clip(np.round(composite), 0, 255)), mode="RGB")


def _component_overlay(base: Any, component_mask: np.ndarray) -> Any:
    return _heat_overlay(base, [(component_mask.astype(np.float32), (255, 0, 255), 0.55)])


def _map_fit_to_raw(fitted: tuple[np.ndarray, np.ndarray, float, float] | None, meta: dict[str, Any], map_hw: tuple[int, int]) -> list[list[float]]:
    if fitted is None:
        return []
    start, end, _length, _thickness = fitted
    segment = np.stack([start, end], axis=0).astype(np.float32)
    return _raw_points_from_map_segment(segment, meta, map_hw)


def _compose_tile(
    *,
    sample_id: str,
    row: dict[str, Any],
    gt_path: Path,
    production_path: Path,
    fit_path: Path,
    heat_path: Path,
    output_path: Path,
) -> None:
    if Image is None or ImageDraw is None or ImageFont is None:  # pragma: no cover
        raise RuntimeError("Pillow is required for stop-line fit-far visuals")
    images = [Image.open(path).convert("RGB") for path in (gt_path, production_path, fit_path, heat_path)]
    max_width = max(image.width for image in images)
    max_height = max(image.height for image in images)
    header_height = 62
    canvas = Image.new("RGB", (max_width * 4, max_height + header_height), color=(10, 10, 10))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    labels = ("ground_truth", "production", "selected_gt_fit", "mask_red_center_green")
    title = (
        f"{sample_id} gt={row['gt_index_int']} "
        f"hit={row['gt_tube_hit_fraction_float']:.2f} "
        f"noA={row['no_anchor_mean_distance_float']:.1f} "
        f"anch={row['anchored_mean_distance_float']:.1f}"
    )
    draw.text((8, 6), title, fill=(255, 255, 255), font=font)
    legend = "selected GT yellow | production blue | no-anchor green | anchored cyan | best component magenta"
    draw.text((8, 24), legend, fill=(220, 220, 220), font=font)
    for column, (label, image) in enumerate(zip(labels, images)):
        canvas.paste(image, (column * max_width, header_height))
        draw.text((column * max_width + 8, 44), label, fill=(255, 255, 255), font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def _compose_grid(tile_paths: list[Path], *, columns: int, output_path: Path) -> None:
    if Image is None:  # pragma: no cover
        raise RuntimeError("Pillow is required for stop-line fit-far visuals")
    if not tile_paths:
        raise ValueError("cannot compose stop-line fit-far grid with zero tiles")
    tiles = [Image.open(path).convert("RGB") for path in tile_paths]
    tile_width = max(tile.width for tile in tiles)
    tile_height = max(tile.height for tile in tiles)
    safe_columns = max(1, int(columns))
    rows = (len(tiles) + safe_columns - 1) // safe_columns
    canvas = Image.new("RGB", (tile_width * safe_columns, tile_height * rows), color=(8, 8, 8))
    for index, tile in enumerate(tiles):
        row = index // safe_columns
        column = index % safe_columns
        canvas.paste(tile, (column * tile_width, row * tile_height))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def _render_fit_panels(
    *,
    row: dict[str, Any],
    sample: dict[str, Any],
    gt_sample: dict[str, Any],
    pred_sample: dict[str, Any],
    outputs: dict[str, Any],
    sample_offset: int,
    postprocess_threshold: float,
    sample_dir: Path,
) -> dict[str, Any]:
    if Image is None or ImageDraw is None or ImageFont is None:  # pragma: no cover
        raise RuntimeError("Pillow is required for stop-line fit-far visuals")
    meta = sample["meta"]
    image_path = Path(str(meta["image_path"]))
    base = Image.open(image_path).convert("RGB")
    font = ImageFont.load_default()

    mask_batch = outputs.get("stop_line_mask_logits")
    if not isinstance(mask_batch, torch.Tensor):
        raise ValueError("model output does not contain stop_line_mask_logits")
    mask_probs = _as_numpy_2d(mask_batch[sample_offset].sigmoid())
    if mask_probs is None:
        raise ValueError("stop_line_mask_logits did not produce a 2D map")
    binary = _prepare_stopline_binary_mask(mask_probs, threshold=float(postprocess_threshold))
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
        _as_numpy_2d(center_logits[sample_offset].sigmoid()) if isinstance(center_logits, torch.Tensor) else None
    )
    selector_probs = (
        _as_numpy_2d(selector_logits[sample_offset].sigmoid()) if isinstance(selector_logits, torch.Tensor) else None
    )
    sample_row_probs = _row_probs(row_logits[sample_offset]) if isinstance(row_logits, torch.Tensor) else None
    sample_x_probs = _x_probs(x_logits[sample_offset]) if isinstance(x_logits, torch.Tensor) else None
    sample_center_offset = center_offset[sample_offset] if isinstance(center_offset, torch.Tensor) else None

    gt_index = int(row["gt_index_int"])
    stop_lines = list(gt_sample.get("stop_lines", []))
    if not (0 <= gt_index < len(stop_lines)):
        raise IndexError(f"GT stop-line index out of range: {gt_index} / {len(stop_lines)}")
    gt_points = [[float(x), float(y)] for x, y in stop_lines[gt_index].get("points_xy", [])]
    gt_map = _map_points_from_raw(gt_points, meta, mask_probs.shape)
    gt_segment = np.stack([gt_map[0], gt_map[-1]], axis=0).astype(np.float32)
    gt_tube = _line_tube_mask(mask_probs.shape, gt_segment, radius=2.5)
    best = _best_component_for_gt(
        labels=labels,
        component_scores=component_scores,
        mask_probs=mask_probs,
        gt_segment_map=gt_segment,
        gt_tube=gt_tube,
        radius=2.5,
    )

    component_mask = np.zeros_like(mask_probs, dtype=np.float32)
    no_anchor_points: list[list[float]] = []
    anchored_points: list[list[float]] = []
    best_label = 0
    if best is not None:
        rows_np = best["rows"]
        cols_np = best["cols"]
        mask_values = best["mask_values"]
        component_mask[rows_np, cols_np] = 1.0
        best_label = int(best["label_index"])
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
        no_anchor_fit = _fit_stopline_segment(best["points"], mask_values=mask_values, center_anchor=None)
        anchored_fit = _fit_stopline_segment(best["points"], mask_values=mask_values, center_anchor=anchor)
        no_anchor_points = _map_fit_to_raw(no_anchor_fit, meta, mask_probs.shape)
        anchored_points = _map_fit_to_raw(anchored_fit, meta, mask_probs.shape)

    gt_path = sample_dir / "ground_truth.png"
    production_path = sample_dir / "production.png"
    render_overlay(_gt_scene_from_sample(sample), gt_path)
    render_overlay(_prediction_to_overlay_scene(pred_sample, sample), production_path)

    fit_image = _component_overlay(base, component_mask)
    fit_draw = ImageDraw.Draw(fit_image)
    for stop_line in pred_sample.get("stop_lines", []):
        points = [[float(x), float(y)] for x, y in stop_line.get("points_xy", [])]
        _draw_polyline(fit_draw, points, fill=(70, 130, 255), width=3)
    _draw_polyline(fit_draw, gt_points, fill=(255, 230, 0), width=5)
    _draw_polyline(fit_draw, no_anchor_points, fill=(0, 255, 80), width=4)
    _draw_polyline(fit_draw, anchored_points, fill=(0, 255, 255), width=4)
    if gt_points:
        _label(fit_draw, (gt_points[0][0], gt_points[0][1] - 20), f"gt {gt_index}", fill=(255, 230, 0), font=font)
    fit_path = sample_dir / "selected_gt_fit.png"
    fit_image.save(fit_path)

    heat_channels: list[tuple[np.ndarray, tuple[int, int, int], float]] = [(mask_probs, (255, 0, 0), 0.45)]
    if center_probs is not None:
        heat_channels.append((center_probs, (0, 255, 0), 0.55))
    if selector_probs is not None:
        heat_channels.append((selector_probs, (0, 64, 255), 0.25))
    heat_image = _heat_overlay(base, heat_channels)
    heat_draw = ImageDraw.Draw(heat_image)
    _draw_polyline(heat_draw, gt_points, fill=(255, 230, 0), width=5)
    heat_path = sample_dir / "mask_center_heat.png"
    heat_image.save(heat_path)

    tile_path = sample_dir / "comparison.png"
    _compose_tile(
        sample_id=str(meta.get("sample_id", "")),
        row=row,
        gt_path=gt_path,
        production_path=production_path,
        fit_path=fit_path,
        heat_path=heat_path,
        output_path=tile_path,
    )
    return {
        "sample_id": str(meta.get("sample_id", "")),
        "dataset_key": str(meta.get("dataset_key", "")),
        "image_path": str(meta.get("image_path", "")),
        "sample_index": int(row["sample_index_int"]),
        "gt_index": int(gt_index),
        "component_count": int(component_count),
        "best_label": int(best_label),
        "production_stop_line_count": len(pred_sample.get("stop_lines", [])),
        "gt_tube_hit_fraction": float(row["gt_tube_hit_fraction_float"]),
        "no_anchor_mean_distance": float(row["no_anchor_mean_distance_float"]),
        "anchored_mean_distance": float(row["anchored_mean_distance_float"]),
        "comparison_path": str(tile_path),
    }


def main() -> int:
    args = parse_args()
    if Image is None or ImageDraw is None or ImageFont is None:
        raise RuntimeError("Pillow is required for stop-line fit-far visuals")

    checkpoint = Path(args.checkpoint).expanduser().resolve()
    source_run = Path(args.source_run).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")

    selected_rows = _load_bucket_rows(args)
    if not selected_rows:
        raise ValueError("no rows matched the requested fit-far bucket")
    selected_by_sample: dict[int, list[dict[str, Any]]] = {}
    for row in selected_rows:
        selected_by_sample.setdefault(int(row["sample_index_int"]), []).append(row)
    max_required_sample = max(selected_by_sample)
    max_required_batch = int(math.floor(max_required_sample / max(1, int(args.batch_size)))) + 1
    max_val_batches = max(int(args.max_val_batches), max_required_batch)

    scenario_args = argparse.Namespace(
        checkpoint=str(checkpoint),
        preset=str(args.preset),
        source_run=str(source_run),
        lane60_experiment=str(args.lane60_experiment),
        phase_index=int(args.phase_index),
        max_val_batches=max_val_batches,
        validation_epoch=int(args.validation_epoch),
        train_batches=int(args.train_batches),
        batch_size=int(args.batch_size),
        device=str(args.device),
        output_dir=str(output_dir),
    )
    scenario, _scenario_path = _load_scenario(scenario_args)
    phase = scenario.phases[int(args.phase_index) - 1]
    train_config = train_config_api.scenario_phase_defaults(scenario.train_defaults, phase.overrides)
    train_config = replace(
        train_config,
        device=_resolve_device(args.device, train_config.device),
        val_batches=max_val_batches,
    )
    postprocess_config = train_cli._build_postprocess_config(train_config)

    train_cli._configure_torch_multiprocessing()
    dataset = train_cli.PV26CanonicalDataset(
        train_cli._existing_dataset_roots(scenario),
        train_augmentation=False,
        progress_callback=lambda message: print(f"[stopline_fit_far_visual] {message}", flush=True),
    )
    _, val_loader = train_cli._build_phase_train_loaders(dataset, train_config=train_config, phase=phase)
    if val_loader is None:
        raise ValueError("stop-line fit-far visual audit requires validation batches")
    _advance_validation_sampler(val_loader, validation_epoch=int(args.validation_epoch))

    trainer = train_cli._build_phase_trainer(phase, train_config)
    trainer.load_model_weights(checkpoint, map_location=train_config.device)
    evaluator = trainer.build_evaluator()
    evaluator.adapter.raw_model.eval()
    evaluator.heads.eval()

    tile_paths: list[Path] = []
    entries: list[dict[str, Any]] = []
    remaining = set(selected_by_sample)
    with torch.no_grad():
        for batch_index, batch in enumerate(val_loader, start=1):
            if batch_index > max_val_batches or not remaining:
                break
            if batch_index == 1 or batch_index % 20 == 0:
                print(f"[stopline_fit_far_visual] eval batch {batch_index}/{max_val_batches}", flush=True)
            raw_batch = raw_batch_for_metrics(batch)
            if raw_batch is None:
                raise ValueError("validation loader must provide raw batches")
            encoded = evaluator.prepare_batch(batch)
            outputs = _detach_to_cpu(evaluator.forward_encoded_batch(encoded))
            meta_rows = _detach_to_cpu(encoded["meta"])
            predictions = postprocess_pv26_batch(outputs, meta_rows, config=postprocess_config)
            gt_samples = _extract_gt_samples(raw_batch)
            for sample_offset, (meta, gt_sample, pred_sample) in enumerate(zip(meta_rows, gt_samples, predictions)):
                sample_index = (batch_index - 1) * int(train_config.batch_size) + sample_offset
                rows_for_sample = selected_by_sample.get(sample_index)
                if not rows_for_sample:
                    continue
                sample = _sample_from_raw(raw_batch, sample_offset)
                sample_id = str(meta.get("sample_id", f"sample_{sample_index}"))
                for row in rows_for_sample:
                    sample_dir = output_dir / f"{len(entries) + 1:02d}__s{sample_index}_gt{row['gt_index_int']}__{_safe_id(sample_id)}"
                    entry = _render_fit_panels(
                        row=row,
                        sample=sample,
                        gt_sample=gt_sample,
                        pred_sample=pred_sample,
                        outputs=outputs,
                        sample_offset=sample_offset,
                        postprocess_threshold=float(postprocess_config.stop_line_mask_binary_threshold),
                        sample_dir=sample_dir,
                    )
                    tile_paths.append(Path(entry["comparison_path"]))
                    entries.append(entry)
                remaining.discard(sample_index)

    if remaining:
        raise RuntimeError(f"failed to render selected sample indexes: {sorted(remaining)}")

    grid_path = output_dir / "stopline_fit_far_bucket_grid.png"
    _compose_grid(tile_paths, columns=int(args.columns), output_path=grid_path)
    manifest = {
        "checkpoint": str(checkpoint),
        "audit_csv": str(Path(args.audit_csv).expanduser().resolve()),
        "lane60_experiment": str(args.lane60_experiment),
        "phase_index": int(args.phase_index),
        "validation_epoch": int(args.validation_epoch),
        "max_val_batches": int(max_val_batches),
        "filter": {
            "production_matched": 0,
            "gt_tube_mask_max_min": float(args.mask_min),
            "gt_tube_center_max_min": float(args.center_min),
            "no_anchor_mean_distance_min": float(args.distance_min),
            "sort": str(args.sort),
        },
        "sample_count": len(entries),
        "grid_path": str(grid_path),
        "samples": entries,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"grid_path": str(grid_path), "sample_count": len(entries)}, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
