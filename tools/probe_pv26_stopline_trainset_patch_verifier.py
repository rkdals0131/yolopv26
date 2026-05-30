from __future__ import annotations

import argparse
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
from model.engine.postprocess import postprocess_pv26_batch
from tools.evaluate_pv26_lane60_checkpoint import _advance_validation_sampler
from tools.probe_pv26_lane_feature_roi_repair import DEFAULT_CHECKPOINT
from tools.probe_pv26_stopline_angle_mask_extent import _detach_to_cpu, _write_csv
from tools.probe_pv26_stopline_candidate_pool import (
    _best_task_threshold,
    _candidate_feature_rows,
    _decode_candidates,
    _fit_raw_patch_cnn,
    _raw_patch_cnn_candidate_arrays,
    _records_metrics_row,
    _scenario_with_dataset_root,
    _standardize_from_train,
    _write_candidate_features_csv,
)
from tools.pv26_train import cli as train_cli
from tools.pv26_train import config as train_config_api


SCORE_KEY = "trainset_raw_patch_cnn_score"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a stop-line raw-patch CNN verifier on canonical train-split "
            "candidate rows, then replay the fixed train-selected threshold on "
            "validation records. This uses existing data in place."
        )
    )
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--preset", default="default")
    parser.add_argument("--phase-index", type=int, default=4)
    parser.add_argument("--train-record-batches", type=int, default=256)
    parser.add_argument("--max-val-batches", type=int, default=128)
    parser.add_argument("--validation-epoch", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dataset-root", default="", help="Override dataset root for detached worktrees.")
    parser.add_argument("--proposal-min-gap", type=float, default=4.0)
    parser.add_argument("--candidate-top-k", type=int, default=50)
    parser.add_argument("--threshold-grid", type=int, default=101)
    parser.add_argument("--verifier-epochs", type=int, default=80)
    parser.add_argument("--verifier-lr", type=float, default=0.001)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def _resolve_device(requested: str, scenario_device: str) -> str:
    value = str(requested or "auto").strip().lower()
    candidate = str(scenario_device) if value == "auto" else str(requested)
    if candidate.startswith("cuda") and not torch.cuda.is_available():
        print("[stopline_trainset_patch_verifier] CUDA requested but unavailable; falling back to CPU", flush=True)
        return "cpu"
    return candidate


def _train_config_with_runtime_defaults(train_config: Any, *, device: str, max_val_batches: int) -> Any:
    return replace(
        train_config,
        device=str(device),
        val_batches=int(max_val_batches),
        encode_train_batches_in_loader=False,
        encode_val_batches_in_loader=False,
        lane_segfirst_track_mode="row_scan_tangent",
        stop_line_projection_comp_enabled=True,
        stop_line_projection_comp_min_gap=4.0,
        stop_line_projection_comp_topk=50,
        stop_line_projection_comp_union_min_score=0.80,
        stop_line_projection_comp_single_min_score=0.90,
        stop_line_projection_comp_angle_threshold_deg=16.0,
        stop_line_projection_comp_offset_threshold_px=48.0,
        stop_line_projection_comp_min_cluster_count=2,
        stop_line_projection_comp_projection_gap_px=320.0,
        stop_line_projection_comp_max_predictions=2,
        stop_line_projection_comp_second_min_fragment_count=5,
        crosswalk_polygon_mode="hull",
    )


def _slice_raw_batch_sample(raw_batch: dict[str, Any], sample_index: int) -> dict[str, Any]:
    keys = ("det_targets", "tl_attr_targets", "lane_targets", "source_mask", "valid_mask", "meta")
    return {key: [raw_batch[key][sample_index]] for key in keys}


def _collect_candidate_records(
    *,
    loader: Any,
    evaluator: Any,
    postprocess_config: Any,
    max_batches: int,
    proposal_min_gap: float,
    candidate_top_k: int,
    split: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch_index, batch in enumerate(loader, start=1):
            if batch_index > int(max_batches):
                break
            if batch_index == 1 or batch_index % 20 == 0:
                print(
                    f"[stopline_trainset_patch_verifier] collect {split} batch {batch_index}/{max_batches}",
                    flush=True,
                )
            raw_batch = raw_batch_for_metrics(batch)
            if raw_batch is None:
                raise ValueError("stop-line trainset patch verifier requires raw batches for metrics")
            gt_samples = _extract_gt_samples(raw_batch)
            encoded = evaluator.prepare_batch(batch)
            outputs = _detach_to_cpu(evaluator.forward_encoded_batch(encoded))
            meta_rows = _detach_to_cpu(encoded["meta"])
            baseline_predictions = postprocess_pv26_batch(outputs, meta_rows, config=postprocess_config)
            for sample_index, (meta, baseline_prediction, gt_sample) in enumerate(
                zip(meta_rows, baseline_predictions, gt_samples)
            ):
                gt_stop_lines = list(gt_sample.get("stop_lines", []))
                candidates, _stats = _decode_candidates(
                    outputs=outputs,
                    sample_index=sample_index,
                    meta=meta,
                    gt_stop_lines=gt_stop_lines,
                    source="max",
                    top_k=int(candidate_top_k),
                    min_gap=float(proposal_min_gap),
                )
                candidate_rows = _candidate_feature_rows(
                    candidates,
                    batch_index=batch_index,
                    sample_index=sample_index,
                    meta=meta,
                    gt_stop_lines=gt_stop_lines,
                )
                for row in candidate_rows:
                    row["split"] = str(split)
                rows.extend(candidate_rows)
                records.append(
                    {
                        "batch_index": int(batch_index),
                        "sample_index": int(sample_index),
                        "sample_id": str(meta.get("sample_id", "")),
                        "raw_batch": _slice_raw_batch_sample(raw_batch, sample_index),
                        "baseline_prediction": dict(baseline_prediction),
                        "gt_stop_lines": gt_stop_lines,
                        "gt_stop_line_count": int(len(gt_stop_lines)),
                        "candidates": candidates,
                        "candidate_feature_rows": candidate_rows,
                    }
                )
    return records, rows


def _attach_candidate_scores(candidates: list[dict[str, Any]], scores: np.ndarray, *, score_key: str = SCORE_KEY) -> None:
    if len(candidates) != int(scores.shape[0]):
        raise ValueError(f"candidate/score length mismatch: {len(candidates)} != {int(scores.shape[0])}")
    for candidate, score in zip(candidates, scores.tolist()):
        candidate[str(score_key)] = float(score)


def _attach_record_candidate_scores(
    records: list[dict[str, Any]],
    scores: np.ndarray,
    *,
    top_k: int,
    score_key: str = SCORE_KEY,
) -> None:
    index = 0
    for record in records:
        for candidate, row in zip(record.get("candidates", []), record.get("candidate_feature_rows", [])):
            if int(candidate.get("proposal_rank", 10**6)) > int(top_k):
                continue
            score = float(scores[index])
            candidate[str(score_key)] = score
            row[str(score_key)] = score
            index += 1
    if index != int(scores.shape[0]):
        raise ValueError(f"record/score length mismatch: {index} != {int(scores.shape[0])}")


def _score_records_with_trainset_cnn(
    *,
    train_records: list[dict[str, Any]],
    val_records: list[dict[str, Any]],
    top_k: int,
    epochs: int,
    lr: float,
    device: str,
) -> dict[str, Any]:
    image_cache: dict[str, np.ndarray] = {}
    train_candidates, train_patches, train_tabular, train_labels = _raw_patch_cnn_candidate_arrays(
        train_records,
        top_k=int(top_k),
        image_cache=image_cache,
    )
    val_candidates, val_patches, val_tabular, val_labels = _raw_patch_cnn_candidate_arrays(
        val_records,
        top_k=int(top_k),
        image_cache=image_cache,
    )
    if not train_candidates or train_patches.shape[0] == 0:
        raise ValueError("no train candidates collected for stop-line trainset raw-patch CNN")
    if not val_candidates or val_patches.shape[0] == 0:
        raise ValueError("no validation candidates collected for stop-line trainset raw-patch CNN")
    combined_tabular = np.concatenate([train_tabular, val_tabular], axis=0)
    combined_std, mean, std = _standardize_from_train(train_tabular, combined_tabular)
    train_tabular_std = combined_std[: train_tabular.shape[0]]
    val_tabular_std = combined_std[train_tabular.shape[0] :]
    model = _fit_raw_patch_cnn(
        train_patches.astype(np.float32),
        train_tabular_std.astype(np.float32),
        train_labels.astype(np.float32),
        epochs=int(epochs),
        lr=float(lr),
        device=str(device),
    )
    train_scores = _predict_trainset_cnn(model, train_patches, train_tabular_std, device=str(device))
    val_scores = _predict_trainset_cnn(model, val_patches, val_tabular_std, device=str(device))
    _attach_candidate_scores(train_candidates, train_scores)
    _attach_candidate_scores(val_candidates, val_scores)
    _attach_record_candidate_scores(train_records, train_scores, top_k=int(top_k))
    _attach_record_candidate_scores(val_records, val_scores, top_k=int(top_k))
    return {
        "train_candidate_count": int(len(train_candidates)),
        "train_positive_count": int(train_labels.sum()),
        "val_candidate_count": int(len(val_candidates)),
        "val_positive_count": int(val_labels.sum()),
        "image_cache_count": int(len(image_cache)),
        "tabular_feature_dim": int(train_tabular.shape[1]),
        "patch_shape": [int(value) for value in train_patches.shape[1:]],
        "mean_dim": int(mean.shape[0]),
        "std_min": float(std.min()) if std.size else 0.0,
    }


def _predict_trainset_cnn(model: torch.nn.Module, patches: np.ndarray, tabular: np.ndarray, *, device: str) -> np.ndarray:
    from tools.probe_pv26_stopline_candidate_pool import _predict_raw_patch_cnn

    return _predict_raw_patch_cnn(model, patches.astype(np.float32), tabular.astype(np.float32), device=str(device))


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    scenario = train_cli.load_meta_train_scenario(args.preset)
    scenario = _scenario_with_dataset_root(scenario, str(args.dataset_root))
    phase = scenario.phases[int(args.phase_index) - 1]
    base_train_config = train_config_api.scenario_phase_defaults(scenario.train_defaults, phase.overrides)
    device = _resolve_device(str(args.device), str(base_train_config.device))
    train_config = _train_config_with_runtime_defaults(
        base_train_config,
        device=device,
        max_val_batches=int(args.max_val_batches),
    )
    train_cli._configure_torch_multiprocessing()
    dataset = train_cli.PV26CanonicalDataset(
        train_cli._existing_dataset_roots(scenario),
        train_augmentation=False,
        progress_callback=lambda message: print(f"[stopline_trainset_patch_verifier] {message}", flush=True),
    )
    train_loader, val_loader = train_cli._build_phase_train_loaders(dataset, train_config=train_config, phase=phase)
    if train_loader is None or val_loader is None:
        raise ValueError("stop-line trainset patch verifier requires train and validation loaders")
    _advance_validation_sampler(val_loader, validation_epoch=int(args.validation_epoch))
    trainer = train_cli._build_phase_trainer(phase, train_config)
    trainer.load_model_weights(checkpoint, map_location=train_config.device)
    evaluator = trainer.build_evaluator()
    evaluator.adapter.raw_model.eval()
    evaluator.heads.eval()
    postprocess_config = train_cli._build_postprocess_config(train_config)

    train_records, train_rows = _collect_candidate_records(
        loader=train_loader,
        evaluator=evaluator,
        postprocess_config=postprocess_config,
        max_batches=int(args.train_record_batches),
        proposal_min_gap=float(args.proposal_min_gap),
        candidate_top_k=int(args.candidate_top_k),
        split="train",
    )
    val_records, val_rows = _collect_candidate_records(
        loader=val_loader,
        evaluator=evaluator,
        postprocess_config=postprocess_config,
        max_batches=int(args.max_val_batches),
        proposal_min_gap=float(args.proposal_min_gap),
        candidate_top_k=int(args.candidate_top_k),
        split="val",
    )
    score_summary = _score_records_with_trainset_cnn(
        train_records=train_records,
        val_records=val_records,
        top_k=int(args.candidate_top_k),
        epochs=int(args.verifier_epochs),
        lr=float(args.verifier_lr),
        device=str(train_config.device),
    )
    threshold_row = _best_task_threshold(
        train_records,
        score_key=SCORE_KEY,
        top_k=int(args.candidate_top_k),
        max_components=int(postprocess_config.stop_line_max_components),
        grid_size=int(args.threshold_grid),
    )
    threshold = float(threshold_row["threshold"])
    rows = [
        _records_metrics_row(train_records, name="baseline", split="train"),
        _records_metrics_row(val_records, name="baseline", split="val"),
        _records_metrics_row(
            train_records,
            name="trainset_raw_patch_cnn_task_threshold",
            split="train",
            score_key=SCORE_KEY,
            threshold=threshold,
            top_k=int(args.candidate_top_k),
            max_components=int(postprocess_config.stop_line_max_components),
        ),
        _records_metrics_row(
            val_records,
            name="trainset_raw_patch_cnn_task_threshold",
            split="val",
            score_key=SCORE_KEY,
            threshold=threshold,
            top_k=int(args.candidate_top_k),
            max_components=int(postprocess_config.stop_line_max_components),
        ),
    ]
    summary = {
        "checkpoint": str(checkpoint),
        "phase_name": phase.name,
        "phase_stage": phase.stage,
        "train_record_batches": int(args.train_record_batches),
        "max_val_batches": int(args.max_val_batches),
        "validation_epoch": int(args.validation_epoch),
        "proposal_min_gap": float(args.proposal_min_gap),
        "candidate_top_k": int(args.candidate_top_k),
        "threshold_grid": int(args.threshold_grid),
        "verifier_epochs": int(args.verifier_epochs),
        "verifier_lr": float(args.verifier_lr),
        "threshold": threshold,
        "score_summary": score_summary,
        "rows": rows,
        "interpretation": (
            "Train-split stop-line raw-patch CNN verifier replay. The verifier is trained on "
            "candidate rows collected from canonical train batches and replayed on validation "
            "candidate rows with one train-selected task threshold. Runtime selection uses no GT; "
            "GT is used only for verifier labels and final audit metrics."
        ),
    }
    return {
        "summary": summary,
        "rows": rows,
        "train_rows": train_rows,
        "val_rows": val_rows,
    }


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = run_probe(args)
    _write_csv(output_dir / "trainset_patch_verifier_variants.csv", payload["rows"])
    _write_candidate_features_csv(output_dir / "train_candidate_features.csv", payload["train_rows"])
    _write_candidate_features_csv(output_dir / "val_candidate_features.csv", payload["val_rows"])
    (output_dir / "summary.json").write_text(json.dumps(payload["summary"], ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
