from __future__ import annotations

import argparse
import csv
from dataclasses import replace
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

from model.engine.batch import raw_batch_for_metrics
from model.engine.metrics import _extract_gt_samples
from tools.evaluate_pv26_lane60_checkpoint import _advance_validation_sampler
from tools.probe_pv26_stopline_angle_mask_extent import _detach_to_cpu, _gt_descriptor
from tools.probe_pv26_stopline_pred_angle_mask_extent import _proposal_map, _top_cells
from tools.pv26_train import cli as train_cli
from tools.pv26_train import config as train_config_api


SOURCE_RUN = (
    REPO_ROOT
    / "runs"
    / "pv26_exhaustive_od_lane_train"
    / "lane60_core_centerline_refine_cross_retain_from_exhaustive_od_lane_default_20260505_032217_default_20260510_003412"
)
DEFAULT_CHECKPOINT = SOURCE_RUN / "phase_4" / "checkpoints" / "best.pt"

SOURCES = ("center", "selector", "max")
WINDOW_RADII = (2, 4, 8)
SCORE_THRESHOLDS = (0.20, 0.40, 0.60, 0.80)
TOP_KS = (1, 3, 5, 10, 20)
HIT_RADII = (4, 8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit whether predicted stop-line center/selector proposal maps rank GT stop-line "
            "centers highly enough to support a production readout."
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
        print("[stopline_proposal_recall] CUDA requested but unavailable; falling back to CPU", flush=True)
        return "cpu"
    return candidate


def _window_max(map_scores: np.ndarray, row: int, col: int, radius: int) -> float:
    output_h, output_w = map_scores.shape
    y0 = max(0, int(row) - int(radius))
    y1 = min(output_h, int(row) + int(radius) + 1)
    x0 = max(0, int(col) - int(radius))
    x1 = min(output_w, int(col) + int(radius) + 1)
    window = map_scores[y0:y1, x0:x1]
    if window.size == 0:
        return 0.0
    return float(np.max(window))


def _rank_of_cell(map_scores: np.ndarray, row: int, col: int) -> int:
    score = float(map_scores[int(row), int(col)])
    return int(np.count_nonzero(map_scores.reshape(-1) > score) + 1)


def _nearest_top_distance(top_cells: list[tuple[int, int, float]], center_xy: np.ndarray) -> float | None:
    if not top_cells:
        return None
    center_x = float(center_xy[0])
    center_y = float(center_xy[1])
    distances = [
        float(np.hypot(float(col) - center_x, float(row) - center_y))
        for row, col, _ in top_cells
    ]
    return float(min(distances)) if distances else None


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float32), q))


def _aggregate_source(source: str, details: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [row for row in details if row["source"] == source]
    gt_count = len(rows)
    output: dict[str, Any] = {"source": source, "gt_count": gt_count}
    if gt_count == 0:
        return output
    for key in ("score_at_gt_cell", "rank_at_gt_cell"):
        values = [float(row[key]) for row in rows]
        output[f"{key}_mean"] = float(sum(values) / len(values))
        output[f"{key}_median"] = _percentile(values, 50)
        output[f"{key}_p10"] = _percentile(values, 10)
        output[f"{key}_p90"] = _percentile(values, 90)
    for radius in WINDOW_RADII:
        key = f"max_r{radius}"
        values = [float(row[key]) for row in rows]
        output[f"{key}_mean"] = float(sum(values) / len(values))
        output[f"{key}_median"] = _percentile(values, 50)
        for threshold in SCORE_THRESHOLDS:
            count = sum(1 for value in values if value >= float(threshold))
            label = str(threshold).replace(".", "")
            output[f"{key}_ge_{label}"] = int(count)
            output[f"{key}_ge_{label}_rate"] = float(count) / float(gt_count)
    ranks = [int(row["rank_at_gt_cell"]) for row in rows]
    for top_k in TOP_KS:
        count = sum(1 for rank in ranks if rank <= int(top_k))
        output[f"raw_rank_le_top{top_k}"] = int(count)
        output[f"raw_rank_le_top{top_k}_rate"] = float(count) / float(gt_count)
    for top_k in TOP_KS:
        for radius in HIT_RADII:
            key = f"top{top_k}_hit_r{radius}"
            count = sum(1 for row in rows if bool(row[key]))
            output[key] = int(count)
            output[f"{key}_rate"] = float(count) / float(gt_count)
    return output


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
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if str(args.output_dir).strip()
        else checkpoint.parents[2] / "analysis_exports" / "stopline_proposal_recall_val128_epoch2"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    train_cli._configure_torch_multiprocessing()
    dataset = train_cli.PV26CanonicalDataset(
        train_cli._existing_dataset_roots(scenario),
        train_augmentation=False,
        progress_callback=lambda message: print(f"[stopline_proposal_recall] {message}", flush=True),
    )
    _, val_loader = train_cli._build_phase_train_loaders(dataset, train_config=train_config, phase=phase)
    if val_loader is None:
        raise ValueError("stop-line proposal recall audit requires validation batches")
    _advance_validation_sampler(val_loader, validation_epoch=int(args.validation_epoch))

    trainer = train_cli._build_phase_trainer(phase, train_config)
    trainer.load_model_weights(checkpoint, map_location=train_config.device)
    evaluator = trainer.build_evaluator()
    evaluator.adapter.raw_model.eval()
    evaluator.heads.eval()

    details: list[dict[str, Any]] = []
    processed_batches = 0
    bad_gt = 0
    missing_maps = 0
    with torch.no_grad():
        for batch_index, batch in enumerate(val_loader, start=1):
            if batch_index > int(args.max_val_batches):
                break
            if batch_index == 1 or batch_index % 20 == 0:
                print(f"[stopline_proposal_recall] eval batch {batch_index}/{args.max_val_batches}", flush=True)
            raw_batch = raw_batch_for_metrics(batch)
            if raw_batch is None:
                raise ValueError("validation loader must provide raw batches for proposal recall audit")
            gt_samples = _extract_gt_samples(raw_batch)
            encoded = evaluator.prepare_batch(batch)
            outputs = _detach_to_cpu(evaluator.forward_encoded_batch(encoded))
            meta_rows = _detach_to_cpu(encoded["meta"])
            for sample_index, (gt_sample, meta) in enumerate(zip(gt_samples, meta_rows)):
                source_maps = {
                    source: _proposal_map(outputs, sample_index, source)
                    for source in SOURCES
                }
                valid_maps = [value for value in source_maps.values() if isinstance(value, np.ndarray)]
                if not valid_maps:
                    missing_maps += int(len(gt_sample.get("stop_lines", [])))
                    continue
                output_hw = (int(valid_maps[0].shape[0]), int(valid_maps[0].shape[1]))
                top_cells_by_source: dict[str, dict[int, list[tuple[int, int, float]]]] = {}
                for source, map_scores in source_maps.items():
                    if not isinstance(map_scores, np.ndarray):
                        continue
                    top_cells_by_source[source] = {
                        top_k: _top_cells(map_scores, top_k=int(top_k), threshold=0.0, min_gap=10.0)
                        for top_k in TOP_KS
                    }
                for gt_index, stop_line in enumerate(gt_sample.get("stop_lines", [])):
                    gt = _gt_descriptor(stop_line, meta, output_hw=output_hw)
                    if gt is None:
                        bad_gt += 1
                        continue
                    row = int(gt["row"])
                    col = int(gt["col"])
                    center_xy = np.asarray(gt["center"], dtype=np.float32)
                    for source, map_scores in source_maps.items():
                        if not isinstance(map_scores, np.ndarray):
                            continue
                        detail: dict[str, Any] = {
                            "batch_index": int(batch_index),
                            "sample_index": int(sample_index),
                            "gt_index": int(gt_index),
                            "source": source,
                            "gt_row": row,
                            "gt_col": col,
                            "score_at_gt_cell": float(map_scores[row, col]),
                            "rank_at_gt_cell": _rank_of_cell(map_scores, row, col),
                        }
                        for radius in WINDOW_RADII:
                            detail[f"max_r{radius}"] = _window_max(map_scores, row, col, int(radius))
                        for top_k in TOP_KS:
                            nearest = _nearest_top_distance(top_cells_by_source[source][top_k], center_xy)
                            detail[f"top{top_k}_nearest_distance"] = "" if nearest is None else float(nearest)
                            for radius in HIT_RADII:
                                detail[f"top{top_k}_hit_r{radius}"] = bool(nearest is not None and nearest <= float(radius))
                        details.append(detail)
            processed_batches += 1

    aggregate_rows = [_aggregate_source(source, details) for source in SOURCES]
    _write_csv(output_dir / "proposal_recall.csv", aggregate_rows)
    _write_csv(output_dir / "per_gt.csv", details)
    summary = {
        "checkpoint": str(checkpoint),
        "phase_name": phase.name,
        "phase_stage": phase.stage,
        "max_val_batches": int(args.max_val_batches),
        "validation_epoch": int(args.validation_epoch),
        "processed_batches": int(processed_batches),
        "bad_gt": int(bad_gt),
        "missing_maps": int(missing_maps),
        "sources": list(SOURCES),
        "window_radii": list(WINDOW_RADII),
        "score_thresholds": list(SCORE_THRESHOLDS),
        "top_ks": list(TOP_KS),
        "hit_radii": list(HIT_RADII),
        "aggregate": aggregate_rows,
        "interpretation": (
            "Read-only proposal reliability audit. A high max_r4/max_r8 score with low top-k hit "
            "means the proposal signal exists but is not ranked competitively; low local scores mean "
            "the center/selector maps themselves miss the GT center neighborhood."
        ),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(aggregate_rows, ensure_ascii=False, indent=2), flush=True)
    print(f"[stopline_proposal_recall] wrote {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
