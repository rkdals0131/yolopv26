#!/usr/bin/env python3
"""Estimate PV26 gradient noise scale from real balanced training batches."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import random
import site
from statistics import median

import numpy as np
import torch

site.addsitedir(str(Path(__file__).resolve().parents[1]))

from common.io import write_json
from model.data.dataset import FocusedDataset, FocusedSource, LogicalBatchSampler, collate_focused
from model.engine.loss import PV26FocusedLoss
from model.engine.trainer import FocusedBatchAdapter
from model.net.pv26 import PV26FocusedModel
from tools.pv26_train.cli import _optimizer_and_scheduler


def _device_batch(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def estimate(run_dir: Path, *, logical_batches: int, small_batch_size: int) -> dict:
    cfg = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    checkpoint = torch.load(run_dir / "checkpoints/latest.pt", map_location="cpu", weights_only=False)
    train_cfg, data_cfg = cfg["train"], cfg["data"]
    logical_batch_size = int(train_cfg["logical_batch_size"])
    if logical_batches <= 0 or small_batch_size <= 0:
        raise ValueError("logical batch count and small batch size must be positive")
    if logical_batch_size % small_batch_size:
        raise ValueError("logical batch size must be divisible by the small batch size")
    device = torch.device(train_cfg["device"])
    seed = int(data_cfg["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    sources = [
        FocusedSource(**{**item, "root": Path(item["root"])})
        for item in data_cfg["sources"]
    ]
    dataset = FocusedDataset(
        sources, split="train", image_hw=tuple(cfg["model"]["image_hw"]), seed=seed,
        augment=bool(data_cfg.get("augment", False)), index_path=run_dir / "train_samples.jsonl",
    )
    sampler_position = int((checkpoint.get("sampler") or {}).get("position", 0))
    sampler = LogicalBatchSampler(
        dataset, batch_size=logical_batch_size, seed=seed,
        start_position=sampler_position,
        strategy=str(data_cfg.get("sampling_strategy", "random_with_replacement")),
    )
    iterator = iter(sampler)
    model = PV26FocusedModel(
        weights=None, variant=cfg["model"]["variant"],
        roadmark_width=int(cfg["model"]["roadmark_width"]),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    stage = str(checkpoint["stage"])
    model.set_train_stage(stage)
    if str(train_cfg.get("optimizer", "adamw")) == "schedulefree_adamw":
        optimizer, _ = _optimizer_and_scheduler(
            model, train_cfg, int(checkpoint["planned_steps"])
        )
        optimizer.load_state_dict(checkpoint["optimizer"])
        optimizer.train()
        del optimizer
    model.train()
    criterion = PV26FocusedLoss(
        model, det_weight=float(train_cfg.get("det_loss_weight", 1.0)),
        roadmark_weight=float(train_cfg.get("roadmark_loss_weight", 1.0)),
    ).to(device)
    criterion.set_progress(int(checkpoint["global_step"]), int(checkpoint["planned_steps"]))
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    adapter = FocusedBatchAdapter()
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(train_cfg["amp_dtype"])
    estimates = []
    for logical_index in range(logical_batches):
        keys = next(iterator)
        logical_batch = collate_focused([dataset[key] for key in keys])
        total_counts = adapter.term_counts(logical_batch)
        chunks_per_logical_batch = logical_batch_size // small_batch_size
        gradient_sums: list[torch.Tensor | None] = [None] * len(parameters)
        small_norm_sum = torch.zeros((), device=device)
        small_batches = 0
        for start in range(0, logical_batch_size, small_batch_size):
            cpu_batch = adapter.slice_batch(
                logical_batch, start, start + small_batch_size
            )
            micro_counts = adapter.term_counts(cpu_batch)
            batch = _device_batch(cpu_batch, device)
            amp = (
                torch.autocast(device_type=device.type, dtype=dtype)
                if dtype is not None else nullcontext()
            )
            with amp:
                outputs = model.forward_for_loss(batch["image"])
            losses = criterion(outputs, batch)
            weighted_terms = adapter.weighted_terms(losses, criterion)
            contribution = sum(
                term * (micro_counts[name] / total_counts[name] if total_counts[name] else 0.0)
                for name, term in weighted_terms.items()
            )
            # Scale each contribution into a small-batch gradient estimator whose
            # mean is the exact trainer-normalized logical-batch gradient.
            loss = contribution * chunks_per_logical_batch
            gradients = torch.autograd.grad(loss, parameters, allow_unused=True)
            norm = torch.zeros((), device=device)
            for index, gradient in enumerate(gradients):
                if gradient is None:
                    continue
                gradient = gradient.detach().float()
                norm += gradient.square().sum()
                gradient_sums[index] = (
                    gradient if gradient_sums[index] is None
                    else gradient_sums[index] + gradient
                )
            small_norm_sum += norm
            small_batches += 1
        small_norm_sq = small_norm_sum / small_batches
        large_norm_sq = torch.zeros((), device=device)
        for gradient_sum in gradient_sums:
            if gradient_sum is not None:
                large_norm_sq += (gradient_sum / small_batches).square().sum()
        denominator = 1.0 / small_batch_size - 1.0 / logical_batch_size
        noise_trace = ((small_norm_sq - large_norm_sq) / denominator).clamp_min(0)
        signal_sq = large_norm_sq - noise_trace / logical_batch_size
        noise_scale = float(noise_trace / signal_sq) if float(signal_sq) > 0 else None
        estimates.append({
            "logical_batch": logical_index,
            "small_gradient_norm_sq": float(small_norm_sq),
            "large_gradient_norm_sq": float(large_norm_sq),
            "noise_trace": float(noise_trace),
            "signal_norm_sq": float(signal_sq),
            "noise_scale": noise_scale,
        })
    small_mean = sum(row["small_gradient_norm_sq"] for row in estimates) / len(estimates)
    large_mean = sum(row["large_gradient_norm_sq"] for row in estimates) / len(estimates)
    aggregate_noise = max((small_mean - large_mean) / denominator, 0.0)
    aggregate_signal = large_mean - aggregate_noise / logical_batch_size
    scales = [row["noise_scale"] for row in estimates if row["noise_scale"] is not None]
    return {
        "run_dir": str(run_dir),
        "checkpoint_step": int(checkpoint["global_step"]),
        "small_batch_size": small_batch_size,
        "large_batch_size": logical_batch_size,
        "logical_batches": logical_batches,
        "noise_scale_aggregate": (
            aggregate_noise / aggregate_signal if aggregate_signal > 0 else None
        ),
        "aggregate_noise_trace": aggregate_noise,
        "aggregate_signal_norm_sq": aggregate_signal,
        "valid_per_batch_estimates": len(scales),
        "noise_scale_median": median(scales) if scales else None,
        "estimates": estimates,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--logical-batches", type=int, default=6)
    parser.add_argument("--small-batch-size", type=int, default=4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = estimate(
        args.run_dir.resolve(), logical_batches=args.logical_batches,
        small_batch_size=args.small_batch_size,
    )
    if args.output is not None:
        write_json(args.output.resolve(), result, ensure_ascii=False)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
