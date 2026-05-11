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
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
repo_root = str(REPO_ROOT)
if repo_root not in sys.path:
    site.addsitedir(repo_root)

from model.engine._trainer_epochs import _merge_raw_batches
from model.engine.batch import augment_lane_family_metrics, raw_batch_for_metrics
from model.engine.metrics import (
    STOP_LINE_POINT_COUNT,
    _extract_gt_samples,
    _mean_point_distance,
    _segment_angle_error,
    summarize_pv26_metrics,
)
from model.engine.postprocess import _dedupe_stop_line_predictions, _stopline_prediction_sort_key, postprocess_pv26_batch
from tools.evaluate_pv26_lane60_checkpoint import _advance_validation_sampler
from tools.probe_pv26_stopline_angle_mask_extent import (
    ExtentVariant,
    _as_2d_array,
    _detach_to_cpu,
    _mask_extent_line,
    _row_from_metrics,
    _sample_tensor,
    _write_csv,
)
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


@dataclass(frozen=True)
class CandidateVariant:
    name: str
    source: str
    top_k: int
    sort_mode: str
    min_score: float = 0.0
    min_length: float = 0.0
    oracle_positive_only: bool = False


VARIANTS = (
    CandidateVariant("max_top10_score", "max", top_k=10, sort_mode="score"),
    CandidateVariant("max_top10_longest", "max", top_k=10, sort_mode="length"),
    CandidateVariant("max_top10_score_len8", "max", top_k=10, sort_mode="score", min_length=8.0),
    CandidateVariant("max_top10_score_len12", "max", top_k=10, sort_mode="score", min_length=12.0),
    CandidateVariant("max_top10_score_s080", "max", top_k=10, sort_mode="score", min_score=0.80),
    CandidateVariant("max_top20_score_len8", "max", top_k=20, sort_mode="score", min_length=8.0),
    CandidateVariant("oracle_max_top10_positive", "max", top_k=10, sort_mode="distance", oracle_positive_only=True),
    CandidateVariant("oracle_max_top20_positive", "max", top_k=20, sort_mode="distance", oracle_positive_only=True),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a top-k stop-line proposal candidate pool and audit whether non-GT feature "
            "filters or oracle candidate validation can recover stop-line F1."
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
        print("[stopline_candidate_pool] CUDA requested but unavailable; falling back to CPU", flush=True)
        return "cpu"
    return candidate


def _nearest_gt(candidate: dict[str, Any], gt_stop_lines: list[dict[str, Any]]) -> tuple[float, float, int]:
    points = candidate.get("points_xy", [])
    best_distance = float("inf")
    best_angle = 180.0
    best_index = -1
    for index, gt in enumerate(gt_stop_lines):
        gt_points = gt.get("points_xy", [])
        distance = _mean_point_distance(points, gt_points, STOP_LINE_POINT_COUNT)
        if distance < best_distance:
            best_distance = float(distance)
            best_angle = float(_segment_angle_error(points, gt_points, STOP_LINE_POINT_COUNT))
            best_index = int(index)
    return best_distance, best_angle, best_index


def _decode_candidates(
    *,
    outputs: dict[str, Any],
    sample_index: int,
    meta: dict[str, Any],
    gt_stop_lines: list[dict[str, Any]],
    source: str,
    top_k: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    mask_probs = _as_2d_array(_sample_tensor(outputs, "stop_line_mask_logits", sample_index), sigmoid=True)
    angle_map = _sample_tensor(outputs, "stop_line_angle", sample_index)
    offset_map = _sample_tensor(outputs, "stop_line_center_offset", sample_index)
    proposal_map = _proposal_map(outputs, sample_index, source)
    if mask_probs is None or proposal_map is None or angle_map is None or offset_map is None:
        return [], {"missing_tensor": 1}
    if angle_map.ndim != 3 or offset_map.ndim != 3:
        return [], {"bad_tensor_shape": 1}
    output_hw = (int(mask_probs.shape[0]), int(mask_probs.shape[1]))
    extent_variant = ExtentVariant(
        name="candidate_pool",
        center_source="pred_center",
        angle_source="pred_angle",
        mask_threshold=0.50,
        normal_band=4.0,
    )
    stats: dict[str, int] = {}
    candidates: list[dict[str, Any]] = []
    top_cells = _top_cells(proposal_map, top_k=int(top_k), threshold=0.0, min_gap=10.0)
    for proposal_rank, (row, col, score) in enumerate(top_cells, start=1):
        pred_offset = offset_map[:, row, col].numpy().astype(np.float32)
        center_xy = np.asarray([float(col) + pred_offset[0], float(row) + pred_offset[1]], dtype=np.float32)
        pred_angle = angle_map[:, row, col].numpy().astype(np.float32)
        candidate, reason = _mask_extent_line(
            mask_probs=mask_probs,
            center_xy=center_xy,
            angle_vec=pred_angle,
            meta=meta,
            output_hw=output_hw,
            variant=extent_variant,
            score=float(score),
        )
        stats[reason] = stats.get(reason, 0) + 1
        if candidate is None:
            continue
        nearest_distance, nearest_angle, nearest_index = _nearest_gt(candidate, gt_stop_lines)
        candidate = dict(candidate)
        candidate["proposal_rank"] = int(proposal_rank)
        candidate["proposal_source"] = str(source)
        candidate["nearest_gt_distance"] = float(nearest_distance)
        candidate["nearest_gt_angle_error"] = float(nearest_angle)
        candidate["nearest_gt_index"] = int(nearest_index)
        candidate["is_oracle_positive"] = bool(nearest_distance <= 40.0)
        candidates.append(candidate)
    return candidates, stats


def _sort_key(candidate: dict[str, Any], mode: str) -> tuple[float, ...]:
    if mode == "score":
        return (float(candidate.get("score", 0.0)), float(candidate.get("length", 0.0)))
    if mode == "length":
        return (float(candidate.get("length", 0.0)), float(candidate.get("score", 0.0)))
    if mode == "distance":
        return (-float(candidate.get("nearest_gt_distance", float("inf"))), float(candidate.get("score", 0.0)))
    raise ValueError(f"unsupported sort mode: {mode}")


def _select_stop_lines(
    candidates: list[dict[str, Any]],
    *,
    variant: CandidateVariant,
    max_components: int,
) -> list[dict[str, Any]]:
    selected = [
        candidate
        for candidate in candidates
        if int(candidate.get("proposal_rank", 10**6)) <= int(variant.top_k)
        and float(candidate.get("score", 0.0)) >= float(variant.min_score)
        and float(candidate.get("length", 0.0)) >= float(variant.min_length)
        and (not bool(variant.oracle_positive_only) or bool(candidate.get("is_oracle_positive", False)))
    ]
    selected.sort(key=lambda candidate: _sort_key(candidate, variant.sort_mode), reverse=True)
    predictions = [
        {
            "score": float(candidate.get("score", 0.0)),
            "center_score": float(candidate.get("center_score", candidate.get("score", 0.0))),
            "length": float(candidate.get("length", 0.0)),
            "points_xy": candidate.get("points_xy", []),
        }
        for candidate in selected
    ]
    predictions.sort(key=_stopline_prediction_sort_key, reverse=True)
    predictions = _dedupe_stop_line_predictions(predictions)
    return predictions[: max(1, int(max_components))]


def _candidate_feature_rows(candidates: list[dict[str, Any]], *, batch_index: int, sample_index: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        rows.append(
            {
                "batch_index": int(batch_index),
                "sample_index": int(sample_index),
                "proposal_source": str(candidate.get("proposal_source", "")),
                "proposal_rank": int(candidate.get("proposal_rank", 0)),
                "score": float(candidate.get("score", 0.0)),
                "length": float(candidate.get("length", 0.0)),
                "nearest_gt_distance": float(candidate.get("nearest_gt_distance", float("inf"))),
                "nearest_gt_angle_error": float(candidate.get("nearest_gt_angle_error", 180.0)),
                "nearest_gt_index": int(candidate.get("nearest_gt_index", -1)),
                "is_oracle_positive": bool(candidate.get("is_oracle_positive", False)),
            }
        )
    return rows


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
        else checkpoint.parents[2] / "analysis_exports" / "stopline_candidate_pool_val128_epoch2"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    train_cli._configure_torch_multiprocessing()
    dataset = train_cli.PV26CanonicalDataset(
        train_cli._existing_dataset_roots(scenario),
        train_augmentation=False,
        progress_callback=lambda message: print(f"[stopline_candidate_pool] {message}", flush=True),
    )
    _, val_loader = train_cli._build_phase_train_loaders(dataset, train_config=train_config, phase=phase)
    if val_loader is None:
        raise ValueError("stop-line candidate pool probe requires validation batches")
    _advance_validation_sampler(val_loader, validation_epoch=int(args.validation_epoch))

    trainer = train_cli._build_phase_trainer(phase, train_config)
    trainer.load_model_weights(checkpoint, map_location=train_config.device)
    evaluator = trainer.build_evaluator()
    evaluator.adapter.raw_model.eval()
    evaluator.heads.eval()

    names = ("baseline",) + tuple(variant.name for variant in VARIANTS)
    predictions_by_variant: dict[str, list[dict[str, Any]]] = {name: [] for name in names}
    stats_by_variant: dict[str, dict[str, int]] = {name: {} for name in names}
    feature_rows: list[dict[str, Any]] = []
    raw_batches: list[dict[str, Any]] = []
    processed_batches = 0
    max_top_k = max(int(variant.top_k) for variant in VARIANTS)
    with torch.no_grad():
        for batch_index, batch in enumerate(val_loader, start=1):
            if batch_index > int(args.max_val_batches):
                break
            if batch_index == 1 or batch_index % 20 == 0:
                print(f"[stopline_candidate_pool] eval batch {batch_index}/{args.max_val_batches}", flush=True)
            raw_batch = raw_batch_for_metrics(batch)
            if raw_batch is None:
                raise ValueError("validation loader must provide raw batches for metrics")
            raw_batches.append(raw_batch)
            gt_samples = _extract_gt_samples(raw_batch)
            encoded = evaluator.prepare_batch(batch)
            outputs = _detach_to_cpu(evaluator.forward_encoded_batch(encoded))
            meta_rows = _detach_to_cpu(encoded["meta"])
            baseline_predictions = postprocess_pv26_batch(outputs, meta_rows, config=postprocess_config)
            predictions_by_variant["baseline"].extend(baseline_predictions)

            for sample_index, (meta, baseline_prediction, gt_sample) in enumerate(
                zip(meta_rows, baseline_predictions, gt_samples)
            ):
                candidates, stats = _decode_candidates(
                    outputs=outputs,
                    sample_index=sample_index,
                    meta=meta,
                    gt_stop_lines=list(gt_sample.get("stop_lines", [])),
                    source="max",
                    top_k=max_top_k,
                )
                feature_rows.extend(_candidate_feature_rows(candidates, batch_index=batch_index, sample_index=sample_index))
                for variant in VARIANTS:
                    totals = stats_by_variant[variant.name]
                    for key, value in stats.items():
                        totals[key] = totals.get(key, 0) + int(value)
                    stop_lines = _select_stop_lines(
                        candidates,
                        variant=variant,
                        max_components=int(postprocess_config.stop_line_max_components),
                    )
                    if stop_lines:
                        totals["decoded_samples"] = totals.get("decoded_samples", 0) + 1
                        totals["decoded_lines"] = totals.get("decoded_lines", 0) + int(len(stop_lines))
                    predictions_by_variant[variant.name].append({**baseline_prediction, "stop_lines": stop_lines})
            processed_batches += 1

    merged_raw = _merge_raw_batches(raw_batches)
    rows: list[dict[str, Any]] = []
    for name in names:
        predictions = predictions_by_variant[name]
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
            float(row.get("phase4_objective_proxy", 0.0)),
            float(row.get("stop_line_f1", 0.0)),
        ),
        reverse=True,
    )
    _write_csv(output_dir / "variants.csv", rows)
    _write_csv(output_dir / "candidate_features.csv", feature_rows)
    positives = sum(1 for row in feature_rows if bool(row.get("is_oracle_positive")))
    summary = {
        "checkpoint": str(checkpoint),
        "phase_name": phase.name,
        "phase_stage": phase.stage,
        "max_val_batches": int(args.max_val_batches),
        "validation_epoch": int(args.validation_epoch),
        "processed_batches": int(processed_batches),
        "postprocess_config": vars(postprocess_config),
        "candidate_count": int(len(feature_rows)),
        "candidate_oracle_positive_count": int(positives),
        "candidate_oracle_positive_rate": float(positives) / float(max(1, len(feature_rows))),
        "variants": rows,
        "interpretation": (
            "Candidate-pool variants are read-only probes. Oracle variants use GT distance to show "
            "candidate-pool headroom and are not production decoders."
        ),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(rows, ensure_ascii=False, indent=2), flush=True)
    print(f"[stopline_candidate_pool] wrote {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
