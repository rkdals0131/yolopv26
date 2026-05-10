from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import site
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
repo_root = str(REPO_ROOT)
if repo_root not in sys.path:
    site.addsitedir(repo_root)

import torch

from tools.pv26_train import cli as train_cli
from tools.pv26_train import config as train_config_api


SOURCE_RUN = (
    REPO_ROOT
    / "runs"
    / "pv26_exhaustive_od_lane_train"
    / "exhaustive_od_lane_default_20260505_032217"
)
DEFAULT_CHECKPOINT = SOURCE_RUN / "phase_4" / "checkpoints" / "best.pt"
THRESHOLDS = tuple(round(0.1 * value, 1) for value in range(1, 10))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe PV26 lane-family dense map precision/recall before vectorization."
    )
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--preset", default="default")
    parser.add_argument("--phase-index", type=int, default=4)
    parser.add_argument("--max-val-batches", type=int, default=128)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def _resolve_device(requested: str, scenario_device: str) -> str:
    value = str(requested or "auto").strip().lower()
    candidate = str(scenario_device) if value == "auto" else str(requested)
    if candidate.startswith("cuda") and not torch.cuda.is_available():
        print("[dense_probe] CUDA requested but unavailable; falling back to CPU", flush=True)
        return "cpu"
    return candidate


def _empty_counts() -> dict[str, int]:
    return {"tp": 0, "fp": 0, "fn": 0, "tn": 0}


def _add_binary_counts(
    counts: dict[str, int],
    *,
    prob: torch.Tensor,
    target: torch.Tensor,
    threshold: float,
    target_threshold: float,
    valid: torch.Tensor | None = None,
) -> None:
    pred_mask = prob >= float(threshold)
    target_mask = target > float(target_threshold)
    if valid is None:
        valid_mask = torch.ones_like(target_mask, dtype=torch.bool)
    else:
        valid_mask = valid.to(device=target_mask.device, dtype=torch.bool)
    pred_mask = pred_mask & valid_mask
    target_mask = target_mask & valid_mask
    counts["tp"] += int((pred_mask & target_mask).sum().item())
    counts["fp"] += int((pred_mask & ~target_mask).sum().item())
    counts["fn"] += int((~pred_mask & target_mask).sum().item())
    counts["tn"] += int((~pred_mask & ~target_mask & valid_mask).sum().item())


def _score_counts(counts: dict[str, int]) -> dict[str, float | int]:
    tp = int(counts["tp"])
    fp = int(counts["fp"])
    fn = int(counts["fn"])
    tn = int(counts["tn"])
    precision = float(tp / (tp + fp)) if (tp + fp) else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) else 0.0
    f1 = float(2.0 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _best_rows(raw_counts: dict[str, dict[float, dict[str, int]]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for name, threshold_counts in raw_counts.items():
        rows = []
        for threshold, counts in threshold_counts.items():
            rows.append({"threshold": float(threshold), **_score_counts(counts)})
        rows.sort(key=lambda row: (float(row["f1"]), float(row["precision"])), reverse=True)
        output[name] = {
            "best": rows[0] if rows else {},
            "thresholds": sorted(rows, key=lambda row: float(row["threshold"])),
        }
    return output


def _map_specs(predictions: dict[str, torch.Tensor], targets: dict[str, torch.Tensor]) -> dict[str, tuple[torch.Tensor, torch.Tensor, float, torch.Tensor | None]]:
    lane_valid = ~(targets["lane_seg_ignore"].to(dtype=torch.bool))
    return {
        "lane_centerline_core": (
            predictions["lane_seg_centerline_logits"].sigmoid(),
            targets["lane_seg_centerline_core"],
            0.5,
            lane_valid,
        ),
        "lane_centerline_soft": (
            predictions["lane_seg_centerline_logits"].sigmoid(),
            targets["lane_seg_centerline_soft"],
            0.05,
            lane_valid,
        ),
        "lane_support": (
            predictions["lane_seg_support_logits"].sigmoid(),
            targets["lane_seg_support"],
            0.5,
            lane_valid,
        ),
        "stop_line_mask": (
            predictions["stop_line_mask_logits"].sigmoid(),
            targets["stop_line_mask"],
            0.5,
            None,
        ),
        "stop_line_center": (
            predictions["stop_line_center_logits"].sigmoid(),
            targets["stop_line_center_heatmap"],
            0.05,
            None,
        ),
        "crosswalk_mask": (
            predictions["crosswalk_mask_logits"].sigmoid(),
            targets["crosswalk_mask"],
            0.5,
            None,
        ),
        "crosswalk_center": (
            predictions["crosswalk_center_logits"].sigmoid(),
            targets["crosswalk_center"],
            0.5,
            None,
        ),
    }


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

    train_cli._configure_torch_multiprocessing()
    dataset = train_cli.PV26CanonicalDataset(
        train_cli._existing_dataset_roots(scenario),
        train_augmentation=False,
        progress_callback=lambda message: print(f"[dense_probe] {message}", flush=True),
    )
    _, val_loader = train_cli._build_phase_train_loaders(dataset, train_config=train_config, phase=phase)
    if val_loader is None:
        raise ValueError("dense map probe requires validation batches")

    trainer = train_cli._build_phase_trainer(phase, train_config)
    trainer.load_model_weights(checkpoint, map_location=train_config.device)
    evaluator = trainer.build_evaluator()
    evaluator.adapter.raw_model.eval()
    evaluator.heads.eval()

    counts: dict[str, dict[float, dict[str, int]]] = {}
    processed_batches = 0
    with torch.no_grad():
        for batch_index, batch in enumerate(val_loader, start=1):
            if batch_index > int(args.max_val_batches):
                break
            if batch_index == 1 or batch_index % 20 == 0:
                print(f"[dense_probe] eval batch {batch_index}/{args.max_val_batches}", flush=True)
            encoded = evaluator.prepare_batch(batch)
            predictions = evaluator.forward_encoded_batch(encoded)
            targets = encoded["roadmark_v2"]
            for name, (prob, target, target_threshold, valid) in _map_specs(predictions, targets).items():
                threshold_counts = counts.setdefault(name, {threshold: _empty_counts() for threshold in THRESHOLDS})
                for threshold in THRESHOLDS:
                    _add_binary_counts(
                        threshold_counts[threshold],
                        prob=prob,
                        target=target,
                        threshold=threshold,
                        target_threshold=target_threshold,
                        valid=valid,
                    )
            processed_batches += 1

    output = {
        "checkpoint": str(checkpoint),
        "processed_batches": int(processed_batches),
        "batch_size": int(train_config.batch_size),
        "maps": _best_rows(counts),
    }
    print(json.dumps(output, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
