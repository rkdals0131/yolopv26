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

from model.engine._trainer_epochs import _merge_raw_batches
from model.engine.batch import augment_lane_family_metrics, raw_batch_for_metrics
from model.engine.metrics import summarize_pv26_metrics
from model.engine.postprocess import (
    _dedupe_stop_line_predictions,
    _stopline_prediction_sort_key,
    postprocess_pv26_batch,
)
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
class PredExtentVariant:
    name: str
    proposal_source: str
    top_k: int
    proposal_threshold: float
    mask_threshold: float = 0.50
    normal_band: float = 4.0
    min_gap: float = 10.0
    fallback_baseline: bool = False


VARIANTS = (
    PredExtentVariant("pred_center_top1_s020_mask050_band4", "center", top_k=1, proposal_threshold=0.20),
    PredExtentVariant("pred_center_top3_s020_mask050_band4", "center", top_k=3, proposal_threshold=0.20),
    PredExtentVariant("pred_max_top3_s020_mask050_band4", "max", top_k=3, proposal_threshold=0.20),
    PredExtentVariant("pred_selector_top3_s020_mask050_band4", "selector", top_k=3, proposal_threshold=0.20),
    PredExtentVariant("pred_selector_top1_s040_mask050_band4", "selector", top_k=1, proposal_threshold=0.40),
    PredExtentVariant("pred_selector_top1_s060_mask050_band4", "selector", top_k=1, proposal_threshold=0.60),
    PredExtentVariant(
        "pred_selector_top1_s040_mask050_band4_fallback",
        "selector",
        top_k=1,
        proposal_threshold=0.40,
        fallback_baseline=True,
    ),
    PredExtentVariant(
        "pred_selector_top1_s060_mask050_band4_fallback",
        "selector",
        top_k=1,
        proposal_threshold=0.60,
        fallback_baseline=True,
    ),
    PredExtentVariant(
        "pred_max_top3_s020_mask050_band4_fallback",
        "max",
        top_k=3,
        proposal_threshold=0.20,
        fallback_baseline=True,
    ),
    PredExtentVariant(
        "pred_max_top3_s040_mask050_band4_fallback",
        "max",
        top_k=3,
        proposal_threshold=0.40,
        fallback_baseline=True,
    ),
    PredExtentVariant(
        "pred_max_top3_s020_mask080_band4_fallback",
        "max",
        top_k=3,
        proposal_threshold=0.20,
        mask_threshold=0.80,
        fallback_baseline=True,
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay predicted stop-line center/selector proposals with angle-anchored mask extent "
            "readout against an existing PV26 checkpoint."
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
        print("[stopline_pred_angle_mask_extent] CUDA requested but unavailable; falling back to CPU", flush=True)
        return "cpu"
    return candidate


def _proposal_map(outputs: dict[str, Any], sample_index: int, source: str) -> np.ndarray | None:
    center_map = _as_2d_array(_sample_tensor(outputs, "stop_line_center_logits", sample_index), sigmoid=True)
    selector_map = _as_2d_array(_sample_tensor(outputs, "stop_line_selector_map_logits", sample_index), sigmoid=True)
    source_key = str(source).strip().lower()
    if source_key == "center":
        return center_map
    if source_key == "selector":
        return selector_map
    if source_key == "max":
        if center_map is None:
            return selector_map
        if selector_map is None:
            return center_map
        return np.maximum(center_map, selector_map)
    raise ValueError(f"unsupported proposal source: {source}")


def _top_cells(map_scores: np.ndarray, *, top_k: int, threshold: float, min_gap: float) -> list[tuple[int, int, float]]:
    if map_scores.ndim != 2:
        return []
    flat_order = np.argsort(-map_scores.reshape(-1))
    output_h, output_w = map_scores.shape
    selected: list[tuple[int, int, float]] = []
    for flat_index in flat_order.tolist():
        score = float(map_scores.reshape(-1)[flat_index])
        if score < float(threshold):
            break
        row, col = divmod(int(flat_index), int(output_w))
        too_close = False
        for prev_row, prev_col, _ in selected:
            distance = float(np.hypot(float(col - prev_col), float(row - prev_row)))
            if distance < float(min_gap):
                too_close = True
                break
        if too_close:
            continue
        selected.append((int(row), int(col), score))
        if len(selected) >= int(top_k):
            break
    return selected


def _decode_predicted_stop_lines(
    *,
    outputs: dict[str, Any],
    sample_index: int,
    meta: dict[str, Any],
    variant: PredExtentVariant,
    max_components: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    mask_probs = _as_2d_array(_sample_tensor(outputs, "stop_line_mask_logits", sample_index), sigmoid=True)
    angle_map = _sample_tensor(outputs, "stop_line_angle", sample_index)
    offset_map = _sample_tensor(outputs, "stop_line_center_offset", sample_index)
    proposal_map = _proposal_map(outputs, sample_index, variant.proposal_source)
    if mask_probs is None or proposal_map is None or angle_map is None or offset_map is None:
        return [], {"missing_tensor": 1}
    if angle_map.ndim != 3 or offset_map.ndim != 3:
        return [], {"bad_tensor_shape": 1}
    output_hw = (int(mask_probs.shape[0]), int(mask_probs.shape[1]))
    extent_variant = ExtentVariant(
        name=variant.name,
        center_source="pred_center",
        angle_source="pred_angle",
        mask_threshold=float(variant.mask_threshold),
        normal_band=float(variant.normal_band),
    )
    stats: dict[str, int] = {}
    candidates: list[dict[str, Any]] = []
    for row, col, score in _top_cells(
        proposal_map,
        top_k=int(variant.top_k),
        threshold=float(variant.proposal_threshold),
        min_gap=float(variant.min_gap),
    ):
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
        if candidate is not None:
            candidates.append(candidate)
    if not candidates:
        return [], stats
    candidates.sort(key=_stopline_prediction_sort_key, reverse=True)
    candidates = _dedupe_stop_line_predictions(candidates)
    limit = max(1, int(max_components))
    return candidates[:limit], stats


def _with_stop_lines(prediction: dict[str, Any], stop_lines: list[dict[str, Any]]) -> dict[str, Any]:
    return {**prediction, "stop_lines": stop_lines}


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
        else checkpoint.parents[2] / "analysis_exports" / "stopline_pred_angle_mask_extent_val128_epoch2"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    train_cli._configure_torch_multiprocessing()
    dataset = train_cli.PV26CanonicalDataset(
        train_cli._existing_dataset_roots(scenario),
        train_augmentation=False,
        progress_callback=lambda message: print(f"[stopline_pred_angle_mask_extent] {message}", flush=True),
    )
    _, val_loader = train_cli._build_phase_train_loaders(dataset, train_config=train_config, phase=phase)
    if val_loader is None:
        raise ValueError("stop-line predicted angle/mask extent probe requires validation batches")
    _advance_validation_sampler(val_loader, validation_epoch=int(args.validation_epoch))

    trainer = train_cli._build_phase_trainer(phase, train_config)
    trainer.load_model_weights(checkpoint, map_location=train_config.device)
    evaluator = trainer.build_evaluator()
    evaluator.adapter.raw_model.eval()
    evaluator.heads.eval()

    names = ("baseline",) + tuple(variant.name for variant in VARIANTS)
    predictions_by_variant: dict[str, list[dict[str, Any]]] = {name: [] for name in names}
    stats_by_variant: dict[str, dict[str, int]] = {name: {} for name in names}
    raw_batches: list[dict[str, Any]] = []
    processed_batches = 0
    with torch.no_grad():
        for batch_index, batch in enumerate(val_loader, start=1):
            if batch_index > int(args.max_val_batches):
                break
            if batch_index == 1 or batch_index % 20 == 0:
                print(f"[stopline_pred_angle_mask_extent] eval batch {batch_index}/{args.max_val_batches}", flush=True)
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
                    stop_lines, stats = _decode_predicted_stop_lines(
                        outputs=outputs,
                        sample_index=sample_index,
                        meta=meta,
                        variant=variant,
                        max_components=int(postprocess_config.stop_line_max_components),
                    )
                    totals = stats_by_variant[variant.name]
                    for key, value in stats.items():
                        totals[key] = totals.get(key, 0) + int(value)
                    if stop_lines:
                        totals["decoded_samples"] = totals.get("decoded_samples", 0) + 1
                        totals["decoded_lines"] = totals.get("decoded_lines", 0) + int(len(stop_lines))
                    elif bool(variant.fallback_baseline):
                        stop_lines = list(baseline_prediction.get("stop_lines", []))
                        totals["fallback_samples"] = totals.get("fallback_samples", 0) + 1
                    predictions_by_variant[variant.name].append(_with_stop_lines(baseline_prediction, stop_lines))
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
            "Predicted-proposal variants are production-style decode probes: no GT centers are used. "
            "They replace only stop_lines while preserving lane and crosswalk predictions."
        ),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(rows, ensure_ascii=False, indent=2), flush=True)
    print(f"[stopline_pred_angle_mask_extent] wrote {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
