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

REPO_ROOT = Path(__file__).resolve().parents[1]
repo_root = str(REPO_ROOT)
if repo_root not in sys.path:
    site.addsitedir(repo_root)

from model.engine import raw_batch_for_metrics
from model.engine._trainer_epochs import _merge_raw_batches
from model.engine.lane_segfirst_vectorizer import lane_segfirst_prediction_maps
from model.engine.metrics import _extract_gt_samples, summarize_pv26_metrics
from model.engine.postprocess import postprocess_pv26_batch
from tools.evaluate_pv26_lane60_checkpoint import _advance_validation_sampler, _postprocess_override_config
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
from tools.probe_pv26_lane_ranked_translate_repair import (
    affine_snap_points_to_local_centerline,
    component_project_points_to_centerline,
    row_profile_project_points_to_centerline,
    snap_points_to_local_centerline,
    translate_points_to_centerline,
)
from tools.pv26_train import cli as train_cli


REPAIR_MODES = (
    "translate_x",
    "local_2d_snap",
    "affine_2d_snap",
    "component_row_project",
    "row_profile_softargmax",
)
STAT_KEYS = (
    "dx",
    "moved_points",
    "mean_move",
    "max_move",
    "affine_residual",
    "snap_moved_points",
    "snap_mean_move",
    "snap_max_move",
    "component_count",
    "component_size",
    "component_median_distance",
    "mean_profile_mass",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a no-GT selector over a fixed bank of deterministic lane repair "
            "hypotheses, then replay replace-only repairs and report TP/FP/FN."
        )
    )
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--source-run", default=str(SOURCE_RUN))
    parser.add_argument("--lane60-experiment", default="stopline_projection_comp_runtime")
    parser.add_argument("--preset", default="default")
    parser.add_argument("--selector-train-batches", type=int, default=64)
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
    parser.add_argument("--repair-modes", default=",".join(REPAIR_MODES))
    parser.add_argument("--translation-radius", type=int, default=4)
    parser.add_argument("--selector-quality-threshold", type=float, default=0.80)
    parser.add_argument("--max-repairs-per-sample", type=int, default=2)
    parser.add_argument("--min-mean-move-px", type=float, default=0.25)
    parser.add_argument("--max-mean-move-px", type=float, default=90.0)
    parser.add_argument("--selector-epochs", type=int, default=40)
    parser.add_argument("--selector-batch-size", type=int, default=256)
    parser.add_argument("--selector-lr", type=float, default=1.0e-3)
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


class LaneRepairHypothesisSelectorNet(nn.Module):
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


def _parse_repair_modes(value: str) -> tuple[str, ...]:
    modes = tuple(mode.strip() for mode in str(value).split(",") if mode.strip())
    unknown = sorted(set(modes) - set(REPAIR_MODES))
    if unknown:
        raise ValueError(f"unsupported repair modes: {', '.join(unknown)}")
    if not modes:
        raise ValueError("at least one repair mode is required")
    return modes


def _finite_array(values: list[float] | np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32).reshape(-1)
    return np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _repair_points(
    mode: str,
    lane: dict[str, Any],
    *,
    centerline: np.ndarray,
    meta: dict[str, Any],
    radius: int,
) -> tuple[list[list[float]], dict[str, float]]:
    points_xy = list(lane.get("points_xy", []))
    if mode == "local_2d_snap":
        return snap_points_to_local_centerline(points_xy, centerline=centerline, meta=meta, radius=int(radius))
    if mode == "affine_2d_snap":
        return affine_snap_points_to_local_centerline(points_xy, centerline=centerline, meta=meta, radius=int(radius))
    if mode == "component_row_project":
        return component_project_points_to_centerline(points_xy, centerline=centerline, meta=meta)
    if mode == "row_profile_softargmax":
        return row_profile_project_points_to_centerline(points_xy, centerline=centerline, meta=meta, radius=int(radius))
    repaired_points, dx = translate_points_to_centerline(points_xy, centerline=centerline, meta=meta, radius=int(radius))
    moved = [
        float(np.linalg.norm(np.asarray([x, y], dtype=np.float32) - np.asarray([tx, ty], dtype=np.float32)))
        for (x, y), (tx, ty) in zip(points_xy, repaired_points)
    ]
    move_values = np.asarray(moved, dtype=np.float32)
    return repaired_points, {
        "dx": float(dx),
        "moved_points": float((move_values > 1.0e-3).sum()) if move_values.size else 0.0,
        "mean_move": float(move_values.mean()) if move_values.size else 0.0,
        "max_move": float(move_values.max()) if move_values.size else 0.0,
    }


def _mean_move(points_xy: list[list[float]], repaired_points: list[list[float]]) -> float:
    original = np.asarray(points_xy, dtype=np.float32).reshape(-1, 2)
    repaired = np.asarray(repaired_points, dtype=np.float32).reshape(-1, 2)
    if original.shape != repaired.shape or original.shape[0] == 0:
        return 0.0
    return float(np.linalg.norm(repaired - original, axis=1).mean())


def _feature_vector(
    *,
    mode: str,
    mode_index: int,
    modes: tuple[str, ...],
    original_features: np.ndarray,
    repaired_features: np.ndarray,
    stats: dict[str, float],
    mean_move: float,
) -> np.ndarray:
    stats_vector = [float(stats.get(key, 0.0)) for key in STAT_KEYS]
    stats_vector.append(float(mean_move))
    mode_vector = [1.0 if int(index) == int(mode_index) else 0.0 for index, _ in enumerate(modes)]
    diff = np.asarray(repaired_features, dtype=np.float32) - np.asarray(original_features, dtype=np.float32)
    return _finite_array(
        np.concatenate(
            [
                np.asarray(original_features, dtype=np.float32).reshape(-1),
                np.asarray(repaired_features, dtype=np.float32).reshape(-1),
                diff.reshape(-1),
                np.asarray(stats_vector, dtype=np.float32),
                np.asarray(mode_vector, dtype=np.float32),
            ],
            axis=0,
        )
    )


def _collect_examples(
    *,
    loader: Any,
    evaluator: Any,
    postprocess_config: Any,
    args: argparse.Namespace,
    max_batches: int,
    training: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    modes = _parse_repair_modes(str(args.repair_modes))
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
                print(f"[lane_repair_hypothesis_selector] collect {split} batch {batch_index}/{max_batches}", flush=True)
            raw_batch = raw_batch_for_metrics(batch)
            if raw_batch is None:
                raise ValueError("lane repair hypothesis selector requires raw batches for metrics")
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
                centerline = maps["centerline_core"].detach().cpu().numpy().astype(np.float32)
                if centerline.ndim == 3 and int(centerline.shape[0]) == 1:
                    centerline = centerline[0]
                gt_lanes = list(sample_gt.get("lanes", []))
                for pred_index, lane in enumerate(sample_pred.get("lanes", [])):
                    original_gt_index, original_distance = _nearest_gt(lane, gt_lanes)
                    original_features, _, _ = _lane_features(
                        lane,
                        predictions=sample_prediction_tensors,
                        maps=maps,
                        meta=sample_meta,
                    )
                    for mode_index, mode in enumerate(modes):
                        repaired_points, stats = _repair_points(
                            mode,
                            lane,
                            centerline=centerline,
                            meta=sample_meta,
                            radius=int(args.translation_radius),
                        )
                        repaired_lane = dict(lane)
                        repaired_lane["points_xy"] = [[float(x), float(y)] for x, y in repaired_points]
                        repaired_gt_index, repaired_distance = _nearest_gt(repaired_lane, gt_lanes)
                        repaired_features, _, _ = _lane_features(
                            repaired_lane,
                            predictions=sample_prediction_tensors,
                            maps=maps,
                            meta=sample_meta,
                        )
                        mean_move = _mean_move(list(lane.get("points_xy", [])), repaired_points)
                        positive = bool(
                            original_gt_index >= 0
                            and int(repaired_gt_index) == int(original_gt_index)
                            and float(original_distance) > LANE_MATCH_THRESHOLD
                            and float(repaired_distance) <= LANE_MATCH_THRESHOLD
                        )
                        feature_vector = _feature_vector(
                            mode=mode,
                            mode_index=mode_index,
                            modes=modes,
                            original_features=original_features,
                            repaired_features=repaired_features,
                            stats=stats,
                            mean_move=mean_move,
                        )
                        examples.append(
                            {
                                "features": feature_vector,
                                "positive": float(1.0 if positive else 0.0),
                                "sample_index": int(global_sample_index),
                                "sample_batch_index": int(sample_batch_index),
                                "pred_index": int(pred_index),
                                "mode": str(mode),
                                "original_gt_index": int(original_gt_index),
                                "repaired_gt_index": int(repaired_gt_index),
                                "original_distance": float(original_distance),
                                "repaired_distance": float(repaired_distance),
                                "mean_move": float(mean_move),
                                "repaired_points": repaired_lane["points_xy"],
                            }
                        )
                        rows.append(
                            {
                                "split": split,
                                "batch_index": int(batch_index),
                                "sample_index": int(global_sample_index),
                                "sample_batch_index": int(sample_batch_index),
                                "pred_index": int(pred_index),
                                "mode": str(mode),
                                "positive": int(positive),
                                "original_gt_index": int(original_gt_index),
                                "repaired_gt_index": int(repaired_gt_index),
                                "original_distance": float(original_distance),
                                "repaired_distance": float(repaired_distance),
                                "mean_move": float(mean_move),
                            }
                        )
                global_sample_index += 1
    return examples, predictions_all, raw_batches, rows


def _train_selector(
    examples: list[dict[str, Any]],
    *,
    args: argparse.Namespace,
    device: str,
) -> tuple[LaneRepairHypothesisSelectorNet, dict[str, Any]]:
    if not examples:
        raise ValueError("no selector training examples collected")
    features = torch.tensor(np.stack([row["features"] for row in examples]), dtype=torch.float32)
    labels = torch.tensor([float(row["positive"]) for row in examples], dtype=torch.float32)
    positive_count = int(labels.sum().item())
    negative_count = int(labels.numel() - positive_count)
    if positive_count <= 0:
        raise ValueError("no positive repair hypotheses collected")
    mean = features.mean(dim=0)
    std = features.std(dim=0).clamp(min=1.0e-6)
    features = (features - mean) / std
    model = LaneRepairHypothesisSelectorNet(int(features.shape[1]), hidden_dim=int(args.hidden_dim)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.selector_lr), weight_decay=1.0e-4)
    features = features.to(device)
    labels = labels.to(device)
    pos_weight = torch.tensor(
        [max(float(negative_count) / max(float(positive_count), 1.0), 1.0)],
        dtype=torch.float32,
        device=device,
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(args.seed))
    history: list[dict[str, float]] = []
    batch_size = max(1, int(args.selector_batch_size))
    for epoch in range(1, max(1, int(args.selector_epochs)) + 1):
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
        if epoch == 1 or epoch == int(args.selector_epochs) or epoch % 10 == 0:
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


def select_hypothesis_indices(
    examples: list[dict[str, Any]],
    probabilities: np.ndarray,
    *,
    quality_threshold: float,
    max_repairs_per_sample: int,
    min_mean_move_px: float,
    max_mean_move_px: float,
) -> set[int]:
    by_sample: dict[int, list[tuple[float, int]]] = {}
    for index, (example, probability) in enumerate(zip(examples, probabilities.tolist())):
        mean_move = float(example.get("mean_move", 0.0))
        if float(probability) < float(quality_threshold):
            continue
        if mean_move < float(min_mean_move_px) or mean_move > float(max_mean_move_px):
            continue
        by_sample.setdefault(int(example["sample_index"]), []).append((float(probability), int(index)))
    selected: set[int] = set()
    replaced: set[tuple[int, int]] = set()
    for sample_index, candidates in by_sample.items():
        accepted_for_sample = 0
        for _, index in sorted(candidates, reverse=True):
            key = (int(sample_index), int(examples[index]["pred_index"]))
            if key in replaced:
                continue
            selected.add(int(index))
            replaced.add(key)
            accepted_for_sample += 1
            if accepted_for_sample >= int(max_repairs_per_sample):
                break
    return selected


def _apply_selector(
    *,
    examples: list[dict[str, Any]],
    predictions_all: list[dict[str, Any]],
    model: LaneRepairHypothesisSelectorNet,
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
    selected = select_hypothesis_indices(
        examples,
        probabilities,
        quality_threshold=float(args.selector_quality_threshold),
        max_repairs_per_sample=int(args.max_repairs_per_sample),
        min_mean_move_px=float(args.min_mean_move_px),
        max_mean_move_px=float(args.max_mean_move_px),
    )
    rows: list[dict[str, Any]] = []
    for index, example in enumerate(examples):
        sample_index = int(example["sample_index"])
        pred_index = int(example["pred_index"])
        is_selected = int(index) in selected
        if is_selected and 0 <= sample_index < len(repaired):
            lanes = repaired[sample_index].get("lanes", [])
            if 0 <= pred_index < len(lanes):
                lanes[pred_index]["points_xy"] = [
                    [float(x), float(y)] for x, y in example.get("repaired_points", [])
                ]
                lanes[pred_index]["repair_hypothesis_score"] = float(probabilities[index])
                lanes[pred_index]["repair_hypothesis_mode"] = str(example["mode"])
        rows.append(
            {
                "sample_index": sample_index,
                "pred_index": pred_index,
                "mode": str(example["mode"]),
                "positive": int(float(example["positive"]) > 0.5),
                "original_gt_index": int(example["original_gt_index"]),
                "repaired_gt_index": int(example["repaired_gt_index"]),
                "original_distance": float(example["original_distance"]),
                "repaired_distance": float(example["repaired_distance"]),
                "mean_move": float(example["mean_move"]),
                "selector_probability": float(probabilities[index]),
                "selected": int(bool(is_selected)),
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
        progress_callback=lambda message: print(f"[lane_repair_hypothesis_selector] {message}", flush=True),
    )
    train_loader, val_loader = train_cli._build_phase_train_loaders(dataset, train_config=train_config, phase=phase)
    if train_loader is None or val_loader is None:
        raise ValueError("lane repair hypothesis selector requires train and validation loaders")
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
        max_batches=int(args.selector_train_batches),
        training=True,
    )
    model, train_summary = _train_selector(train_examples, args=args, device=device)
    val_examples, baseline_predictions, raw_batches, val_rows = _collect_examples(
        loader=val_loader,
        evaluator=evaluator,
        postprocess_config=postprocess_config,
        args=args,
        max_batches=int(args.max_val_batches),
        training=False,
    )
    repaired_predictions, selector_rows = _apply_selector(
        examples=val_examples,
        predictions_all=baseline_predictions,
        model=model,
        args=args,
        device=device,
    )
    raw_all = _merge_raw_batches(raw_batches)
    baseline_tasks = {
        task: _metric_payload(summarize_pv26_metrics(baseline_predictions, raw_all), task)
        for task in ("lane", "stop_line", "crosswalk")
    }
    repaired_tasks = {
        task: _metric_payload(summarize_pv26_metrics(repaired_predictions, raw_all), task)
        for task in ("lane", "stop_line", "crosswalk")
    }
    selected = [row for row in selector_rows if int(row.get("selected", 0))]
    selected_positive = [row for row in selected if int(row.get("positive", 0))]
    summary = {
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "source_run": str(Path(args.source_run).expanduser().resolve()),
        "scenario_path": str(scenario_path),
        "phase_index": int(phase_index),
        "lane60_experiment": str(args.lane60_experiment),
        "lane_flip_variant": str(args.lane_flip_variant),
        "repair_modes": list(_parse_repair_modes(str(args.repair_modes))),
        "selector_train_batches": int(args.selector_train_batches),
        "max_val_batches": int(args.max_val_batches),
        "validation_epoch": int(args.validation_epoch),
        "selector_quality_threshold": float(args.selector_quality_threshold),
        "max_repairs_per_sample": int(args.max_repairs_per_sample),
        "min_mean_move_px": float(args.min_mean_move_px),
        "max_mean_move_px": float(args.max_mean_move_px),
        "train_summary": train_summary,
        "train_candidate_count": int(len(train_examples)),
        "val_candidate_count": int(len(val_examples)),
        "selected_candidate_count": int(len(selected)),
        "selected_oracle_positive_count": int(len(selected_positive)),
        "baseline": baseline_tasks,
        "repaired": repaired_tasks,
        "delta": {task: _task_delta(repaired_tasks[task], baseline_tasks[task]) for task in baseline_tasks},
        "interpretation": (
            "No-GT runtime replay of a learned selector over a bank of deterministic "
            "lane repair hypotheses. GT is used only for train labels and audit metrics."
        ),
    }
    return {
        "summary": summary,
        "train_rows": train_rows,
        "val_rows": val_rows,
        "selector_rows": selector_rows,
    }


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = run_probe(args)
    (output_dir / "summary.json").write_text(json.dumps(_json_ready(payload["summary"]), indent=2), encoding="utf-8")
    _write_csv(output_dir / "train_candidates.csv", payload["train_rows"])
    _write_csv(output_dir / "val_candidates.csv", payload["val_rows"])
    _write_csv(output_dir / "selector_replay_rows.csv", payload["selector_rows"])
    print(json.dumps(_json_ready({"summary": payload["summary"]}), indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
