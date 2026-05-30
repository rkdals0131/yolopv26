from __future__ import annotations

import argparse
import json
from pathlib import Path
import site
import sys
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

REPO_ROOT = Path(__file__).resolve().parents[1]
repo_root = str(REPO_ROOT)
if repo_root not in sys.path:
    site.addsitedir(repo_root)

from model.engine import raw_batch_for_metrics
from model.engine._trainer_epochs import _merge_raw_batches
from model.engine.batch import augment_lane_family_metrics
from model.engine.lane_segfirst_vectorizer import LaneSegFirstVectorizerConfig, vectorize_lane_segfirst_maps
from model.engine.metrics import _extract_gt_samples, _mean_point_distance, summarize_pv26_metrics
from model.engine.postprocess import _filter_lane_predictions, postprocess_pv26_batch
from tools.evaluate_pv26_lane60_checkpoint import _postprocess_override_config
from tools.probe_pv26_lane_feature_roi_repair import (
    DEFAULT_CHECKPOINT,
    LANE_MATCH_THRESHOLD,
    SOURCE_RUN,
    _build_scenario,
    _forward_predictions,
    _json_ready,
    _lane_features,
    _metric_payload,
    _nearest_gt,
    _task_delta,
)
from tools.probe_pv26_lane_flip_tta import _detach_to_cpu
from tools.probe_pv26_lane_instance_evidence import _resolve_device, _write_csv
from tools.pv26_train import cli as train_cli


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a no-GT line-ROI verifier over raw seg-first lane candidates "
            "that the default bbox/area filter drops, then append selected "
            "candidates and report actual lane-family TP/FP/FN."
        )
    )
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--source-run", default=str(SOURCE_RUN))
    parser.add_argument("--lane60-experiment", default="stopline_projection_comp_runtime")
    parser.add_argument("--preset", default="default")
    parser.add_argument("--verifier-train-batches", type=int, default=64)
    parser.add_argument("--max-val-batches", type=int, default=4)
    parser.add_argument("--validation-epoch", type=int, default=2)
    parser.add_argument("--train-batches", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dataset-root", default="")
    parser.add_argument(
        "--lane-flip-variant",
        choices=("baseline", "flip_centerline_avg", "flip_centerline_avg_lane_cross_comp050"),
        default="flip_centerline_avg_lane_cross_comp050",
    )
    parser.add_argument("--positive-distance-px", type=float, default=LANE_MATCH_THRESHOLD)
    parser.add_argument("--negative-distance-px", type=float, default=60.0)
    parser.add_argument("--baseline-duplicate-distance-px", type=float, default=LANE_MATCH_THRESHOLD)
    parser.add_argument("--candidate-duplicate-distance-px", type=float, default=LANE_MATCH_THRESHOLD)
    parser.add_argument("--quality-threshold", type=float, default=0.80)
    parser.add_argument("--max-appends-per-sample", type=int, default=2)
    parser.add_argument("--verifier-epochs", type=int, default=40)
    parser.add_argument("--verifier-batch-size", type=int, default=256)
    parser.add_argument("--verifier-lr", type=float, default=1.0e-3)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--seed", type=int, default=26)
    parser.add_argument("--backbone-weights", default="")
    parser.add_argument("--lane-obj-threshold", type=float, default=None)
    parser.add_argument("--lane-segfirst-track-mode", default=None)
    parser.add_argument("--lane-segfirst-max-row-gap", type=int, default=None)
    parser.add_argument("--lane-segfirst-max-link-dx", type=float, default=None)
    parser.add_argument("--lane-segfirst-max-turn-degrees", type=float, default=None)
    parser.add_argument("--stop-line-mask-binary-threshold", type=float, default=None)
    parser.add_argument("--stop-line-min-instance-score", type=float, default=None)
    parser.add_argument("--stop-line-presence-threshold", type=float, default=None)
    parser.add_argument(
        "--stop-line-component-gate-source",
        choices=("center", "selector", "max", "validator", "max_validator", "product_validator"),
        default=None,
    )
    parser.add_argument("--crosswalk-obj-threshold", type=float, default=None)
    parser.add_argument("--crosswalk-mask-binary-threshold", type=float, default=None)
    parser.add_argument("--crosswalk-min-component-pixels", type=int, default=None)
    parser.add_argument("--crosswalk-max-components", type=int, default=None)
    parser.add_argument("--crosswalk-min-polygon-area-px", type=float, default=None)
    parser.add_argument("--crosswalk-min-bbox-aspect", type=float, default=None)
    parser.add_argument("--crosswalk-polygon-mode", choices=("rect", "hull"), default="hull")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


class LaneAreaRoiVerifierNet(nn.Module):
    def __init__(self, input_dim: int, *, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(input_dim), int(hidden_dim)),
            nn.SiLU(inplace=True),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.SiLU(inplace=True),
            nn.Linear(int(hidden_dim), 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


def _lane_distance(a: dict[str, Any], b: dict[str, Any]) -> float:
    try:
        return float(_mean_point_distance(a.get("points_xy", []), b.get("points_xy", []), target_count=20))
    except Exception:
        return float("inf")


def _near_any_lane(candidate: dict[str, Any], lanes: list[dict[str, Any]], *, threshold_px: float) -> bool:
    return any(_lane_distance(candidate, lane) <= float(threshold_px) for lane in lanes)


def _baseline_matched_gt_indices(
    baseline_lanes: list[dict[str, Any]],
    gt_lanes: list[dict[str, Any]],
    *,
    threshold_px: float = LANE_MATCH_THRESHOLD,
) -> set[int]:
    if not baseline_lanes or not gt_lanes:
        return set()
    cost = np.zeros((len(baseline_lanes), len(gt_lanes)), dtype=np.float32)
    for pred_index, prediction in enumerate(baseline_lanes):
        for gt_index, gt_lane in enumerate(gt_lanes):
            cost[pred_index, gt_index] = float(_lane_distance(prediction, gt_lane))
    pred_indices, gt_indices = linear_sum_assignment(cost)
    matched: set[int] = set()
    for pred_index, gt_index in zip(pred_indices.tolist(), gt_indices.tolist()):
        if float(cost[pred_index, gt_index]) <= float(threshold_px):
            matched.add(int(gt_index))
    return matched


def _is_dropped_by_lane_filter(candidate: dict[str, Any], *, postprocess_config: Any) -> bool:
    kept = _filter_lane_predictions(
        [candidate],
        min_bbox_area_px=float(postprocess_config.lane_segfirst_min_bbox_area_px),
        max_bbox_aspect=float(postprocess_config.lane_segfirst_max_bbox_aspect),
    )
    return len(kept) == 0


def _raw_lane_candidates(
    *,
    maps: dict[str, torch.Tensor],
    meta: dict[str, Any],
    postprocess_config: Any,
) -> list[dict[str, Any]]:
    return vectorize_lane_segfirst_maps(
        maps,
        meta=meta,
        config=LaneSegFirstVectorizerConfig(
            track_mode=str(postprocess_config.lane_segfirst_track_mode),
            centerline_threshold=float(postprocess_config.lane_obj_threshold),
            min_polyline_length_px=float(postprocess_config.lane_segfirst_min_polyline_length_px),
            min_polyline_bottom_y_fraction=float(postprocess_config.lane_segfirst_min_polyline_bottom_y_fraction),
            semantic_vote_mode=str(postprocess_config.lane_segfirst_semantic_vote_mode),
            max_row_gap=int(postprocess_config.lane_segfirst_max_row_gap),
            max_link_dx=float(postprocess_config.lane_segfirst_max_link_dx),
            max_turn_degrees=float(postprocess_config.lane_segfirst_max_turn_degrees),
            seed_threshold=float(postprocess_config.lane_segfirst_seed_threshold),
            seed_trace_max_seeds=int(postprocess_config.lane_segfirst_seed_trace_max_seeds),
            center_offset_enabled=bool(postprocess_config.lane_segfirst_center_offset_enabled),
            center_offset_max_shift_px=float(postprocess_config.lane_segfirst_center_offset_max_shift_px),
            center_offset_min_support_score=float(postprocess_config.lane_segfirst_center_offset_min_support_score),
        ),
    )


def _candidate_label(
    *,
    candidate: dict[str, Any],
    gt_lanes: list[dict[str, Any]],
    baseline_matched_gt: set[int],
    positive_distance_px: float,
    negative_distance_px: float,
) -> tuple[bool, bool, int, float]:
    gt_index, distance = _nearest_gt(candidate, gt_lanes)
    positive = bool(
        gt_index >= 0
        and int(gt_index) not in baseline_matched_gt
        and float(distance) <= float(positive_distance_px)
    )
    negative = bool(
        gt_index < 0
        or int(gt_index) in baseline_matched_gt
        or float(distance) >= float(negative_distance_px)
    )
    return positive, negative, int(gt_index), float(distance)


def _collect_examples(
    *,
    loader: Any,
    evaluator: Any,
    postprocess_config: Any,
    args: argparse.Namespace,
    max_batches: int,
    training: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    from model.engine.lane_segfirst_vectorizer import lane_segfirst_prediction_maps

    examples: list[dict[str, Any]] = []
    predictions_all: list[dict[str, Any]] = []
    raw_batches: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    global_sample_index = 0
    split = "train" if training else "val"
    with torch.no_grad():
        for batch_index, batch in enumerate(loader, start=1):
            if batch_index > int(max_batches):
                break
            if batch_index == 1 or batch_index % 20 == 0:
                print(f"[lane_area_roi_verifier] collect {split} batch {batch_index}/{max_batches}", flush=True)
            raw_batch = raw_batch_for_metrics(batch)
            if raw_batch is None:
                raise ValueError("lane area ROI verifier requires raw batches for metrics")
            encoded = evaluator.prepare_batch(batch)
            predictions = _forward_predictions(evaluator, encoded, lane_flip_variant=str(args.lane_flip_variant))
            meta = _detach_to_cpu(encoded["meta"])
            batch_predictions = postprocess_pv26_batch(predictions, meta, config=postprocess_config)
            batch_gt = _extract_gt_samples(raw_batch)
            predictions_all.extend(batch_predictions)
            raw_batches.append(raw_batch)
            for sample_batch_index, (sample_pred, sample_gt, sample_meta) in enumerate(
                zip(batch_predictions, batch_gt, meta)
            ):
                maps = lane_segfirst_prediction_maps(predictions, batch_index=sample_batch_index)
                sample_prediction_tensors = {
                    key: value[sample_batch_index]
                    for key, value in predictions.items()
                    if isinstance(value, torch.Tensor) and int(value.shape[0]) > sample_batch_index
                }
                gt_lanes = list(sample_gt.get("lanes", []))
                baseline_lanes = list(sample_pred.get("lanes", []))
                baseline_matched_gt = _baseline_matched_gt_indices(baseline_lanes, gt_lanes)
                raw_candidates = _raw_lane_candidates(
                    maps=maps,
                    meta=sample_meta,
                    postprocess_config=postprocess_config,
                )
                sample_candidate_index = 0
                for candidate in raw_candidates:
                    if not _is_dropped_by_lane_filter(candidate, postprocess_config=postprocess_config):
                        continue
                    if _near_any_lane(
                        candidate,
                        baseline_lanes,
                        threshold_px=float(args.baseline_duplicate_distance_px),
                    ):
                        continue
                    positive, negative, gt_index, distance = _candidate_label(
                        candidate=candidate,
                        gt_lanes=gt_lanes,
                        baseline_matched_gt=baseline_matched_gt,
                        positive_distance_px=float(args.positive_distance_px),
                        negative_distance_px=float(args.negative_distance_px),
                    )
                    if training and not positive and not negative:
                        continue
                    features, _, _ = _lane_features(
                        candidate,
                        predictions=sample_prediction_tensors,
                        maps=maps,
                        meta=sample_meta,
                    )
                    example = {
                        "features": features.astype(np.float32),
                        "positive": float(1.0 if positive else 0.0),
                        "negative": float(1.0 if negative else 0.0),
                        "nearest_gt_index": int(gt_index),
                        "nearest_gt_distance": float(distance),
                        "sample_index": int(global_sample_index),
                        "sample_batch_index": int(sample_batch_index),
                        "candidate_index": int(sample_candidate_index),
                        "candidate": dict(candidate),
                    }
                    examples.append(example)
                    rows.append(
                        {
                            "split": split,
                            "batch_index": int(batch_index),
                            "sample_index": int(global_sample_index),
                            "sample_batch_index": int(sample_batch_index),
                            "candidate_index": int(sample_candidate_index),
                            "nearest_gt_index": int(gt_index),
                            "nearest_gt_distance": float(distance),
                            "positive": int(positive),
                            "negative": int(negative),
                            "baseline_matched_gt_count": int(len(baseline_matched_gt)),
                        }
                    )
                    sample_candidate_index += 1
                global_sample_index += 1
    return examples, predictions_all, raw_batches, rows


def _train_verifier(
    examples: list[dict[str, Any]],
    *,
    args: argparse.Namespace,
    device: str,
) -> tuple[LaneAreaRoiVerifierNet, dict[str, Any]]:
    if not examples:
        raise ValueError("no area-ROI verifier training examples collected")
    features = torch.tensor(np.stack([row["features"] for row in examples]), dtype=torch.float32)
    labels = torch.tensor([float(row["positive"]) for row in examples], dtype=torch.float32)
    mean = features.mean(dim=0)
    std = features.std(dim=0).clamp(min=1.0e-6)
    features = (features - mean) / std
    model = LaneAreaRoiVerifierNet(int(features.shape[1]), hidden_dim=int(args.hidden_dim)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.verifier_lr), weight_decay=1.0e-4)
    features = features.to(device)
    labels = labels.to(device)
    positive_count = int(labels.sum().item())
    negative_count = int(labels.numel() - positive_count)
    pos_weight = torch.tensor(
        [max(float(negative_count) / max(float(positive_count), 1.0), 1.0)],
        dtype=torch.float32,
        device=device,
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(args.seed))
    batch_size = max(1, int(args.verifier_batch_size))
    history: list[dict[str, float]] = []
    for epoch in range(1, max(1, int(args.verifier_epochs)) + 1):
        order = torch.randperm(int(labels.numel()), generator=generator)
        losses: list[float] = []
        for start in range(0, int(order.numel()), batch_size):
            index = order[start : start + batch_size].to(device)
            logits = model(features[index])
            loss = F.binary_cross_entropy_with_logits(logits, labels[index], pos_weight=pos_weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        if epoch == 1 or epoch == int(args.verifier_epochs) or epoch % 10 == 0:
            history.append({"epoch": float(epoch), "loss": float(np.mean(losses) if losses else 0.0)})
    model.register_buffer("feature_mean", mean.to(device), persistent=True)
    model.register_buffer("feature_std", std.to(device), persistent=True)
    summary = {
        "example_count": int(labels.numel()),
        "positive_count": positive_count,
        "negative_count": negative_count,
        "input_dim": int(features.shape[1]),
        "history": history,
    }
    return model, summary


def _apply_verifier(
    *,
    examples: list[dict[str, Any]],
    predictions_all: list[dict[str, Any]],
    model: LaneAreaRoiVerifierNet,
    args: argparse.Namespace,
    device: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    repaired = [dict(sample, lanes=[dict(lane) for lane in sample.get("lanes", [])]) for sample in predictions_all]
    if not examples:
        return repaired, []
    features = torch.tensor(np.stack([row["features"] for row in examples]), dtype=torch.float32, device=device)
    features = (features - model.feature_mean) / model.feature_std
    with torch.no_grad():
        probabilities = torch.sigmoid(model(features)).detach().cpu().numpy().astype(np.float32)
    by_sample: dict[int, list[tuple[float, int]]] = {}
    for index, (example, probability) in enumerate(zip(examples, probabilities.tolist())):
        by_sample.setdefault(int(example["sample_index"]), []).append((float(probability), int(index)))
    selected_indices: set[int] = set()
    for sample_index, candidates in by_sample.items():
        candidates.sort(reverse=True)
        accepted: list[dict[str, Any]] = list(repaired[sample_index].get("lanes", [])) if 0 <= sample_index < len(repaired) else []
        for probability, index in candidates:
            if len(selected_indices) >= len(examples):
                break
            if probability < float(args.quality_threshold):
                continue
            candidate = dict(examples[index]["candidate"])
            if _near_any_lane(
                candidate,
                accepted,
                threshold_px=float(args.candidate_duplicate_distance_px),
            ):
                continue
            selected_indices.add(int(index))
            accepted.append(candidate)
            if sum(1 for item in selected_indices if int(examples[item]["sample_index"]) == int(sample_index)) >= int(
                args.max_appends_per_sample
            ):
                break
    rows: list[dict[str, Any]] = []
    for index, example in enumerate(examples):
        sample_index = int(example["sample_index"])
        selected = int(index) in selected_indices
        probability = float(probabilities[index])
        if selected and 0 <= sample_index < len(repaired):
            candidate = dict(example["candidate"])
            candidate["area_roi_verifier_score"] = probability
            repaired[sample_index].setdefault("lanes", []).append(candidate)
        rows.append(
            {
                "sample_index": sample_index,
                "candidate_index": int(example["candidate_index"]),
                "nearest_gt_index": int(example["nearest_gt_index"]),
                "nearest_gt_distance": float(example["nearest_gt_distance"]),
                "positive": int(float(example["positive"]) > 0.5),
                "negative": int(float(example["negative"]) > 0.5),
                "verifier_probability": probability,
                "selected": int(bool(selected)),
            }
        )
    return repaired, rows


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    scenario, scenario_path, options, phase, train_config = _build_scenario(args)
    phase_index = int(tuple(options["selected_phase_indices"])[0])
    train_cli._configure_torch_multiprocessing()
    dataset = train_cli.PV26CanonicalDataset(
        train_cli._existing_dataset_roots(scenario),
        train_augmentation=False,
        progress_callback=lambda message: print(f"[lane_area_roi_verifier] {message}", flush=True),
    )
    train_loader, val_loader = train_cli._build_phase_train_loaders(dataset, train_config=train_config, phase=phase)
    if train_loader is None or val_loader is None:
        raise ValueError("lane area ROI verifier requires train and validation loaders")
    from tools.evaluate_pv26_lane60_checkpoint import _advance_validation_sampler

    _advance_validation_sampler(val_loader, validation_epoch=int(args.validation_epoch))
    trainer = train_cli._build_phase_trainer(phase, train_config)
    trainer.load_model_weights(Path(args.checkpoint).expanduser().resolve(), map_location=train_config.device)
    postprocess_config = _postprocess_override_config(args, trainer) or train_cli._build_postprocess_config(train_config)
    evaluator = trainer.build_evaluator()
    evaluator.adapter.raw_model.eval()
    evaluator.heads.eval()
    device = _resolve_device(str(args.device), str(train_config.device))

    train_examples, _, _, train_rows = _collect_examples(
        loader=train_loader,
        evaluator=evaluator,
        postprocess_config=postprocess_config,
        args=args,
        max_batches=int(args.verifier_train_batches),
        training=True,
    )
    model, train_summary = _train_verifier(train_examples, args=args, device=device)
    val_examples, baseline_predictions, raw_batches, val_rows = _collect_examples(
        loader=val_loader,
        evaluator=evaluator,
        postprocess_config=postprocess_config,
        args=args,
        max_batches=int(args.max_val_batches),
        training=False,
    )
    repaired_predictions, verifier_rows = _apply_verifier(
        examples=val_examples,
        predictions_all=baseline_predictions,
        model=model,
        args=args,
        device=device,
    )
    merged_raw = _merge_raw_batches(raw_batches)
    baseline_metrics = augment_lane_family_metrics(summarize_pv26_metrics(baseline_predictions, merged_raw))
    repaired_metrics = augment_lane_family_metrics(summarize_pv26_metrics(repaired_predictions, merged_raw))
    baseline_tasks = {task: _metric_payload(baseline_metrics, task) for task in ("lane", "stop_line", "crosswalk")}
    repaired_tasks = {task: _metric_payload(repaired_metrics, task) for task in ("lane", "stop_line", "crosswalk")}
    deltas = {task: _task_delta(repaired_tasks[task], baseline_tasks[task]) for task in baseline_tasks}
    selected = [row for row in verifier_rows if int(row.get("selected", 0))]
    summary = {
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "source_run": str(Path(args.source_run).expanduser().resolve()),
        "scenario_path": str(scenario_path),
        "lane60_experiment": str(args.lane60_experiment),
        "phase_index": int(phase_index),
        "lane_flip_variant": str(args.lane_flip_variant),
        "verifier_train_batches": int(args.verifier_train_batches),
        "max_val_batches": int(args.max_val_batches),
        "validation_epoch": int(args.validation_epoch),
        "positive_distance_px": float(args.positive_distance_px),
        "negative_distance_px": float(args.negative_distance_px),
        "quality_threshold": float(args.quality_threshold),
        "max_appends_per_sample": int(args.max_appends_per_sample),
        "train_summary": train_summary,
        "val_candidate_count": int(len(val_examples)),
        "selected_candidate_count": int(len(selected)),
        "selected_oracle_positive_count": int(sum(int(row.get("positive", 0)) for row in selected)),
        "baseline": baseline_tasks,
        "repaired": repaired_tasks,
        "delta": deltas,
        "interpretation": (
            "No-GT runtime replay of a learned line-ROI verifier over raw seg-first "
            "lane candidates dropped by the default lane bbox/area filter. GT is used "
            "only for train labels and final audit metrics, not candidate selection."
        ),
    }
    return {
        "summary": summary,
        "train_rows": train_rows,
        "val_rows": val_rows,
        "verifier_rows": verifier_rows,
    }


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = run_probe(args)
    (output_dir / "summary.json").write_text(json.dumps(_json_ready(payload["summary"]), indent=2), encoding="utf-8")
    _write_csv(output_dir / "train_candidates.csv", payload["train_rows"])
    _write_csv(output_dir / "val_candidates.csv", payload["val_rows"])
    _write_csv(output_dir / "verifier_replay_rows.csv", payload["verifier_rows"])
    print(json.dumps(_json_ready({"summary": payload["summary"]}), indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
