from __future__ import annotations

from collections import Counter
import time
from typing import Any

import torch

from ._trainer_io import _append_jsonl, _now_iso
from ._trainer_reporting import (
    _aggregate_assignment_modes,
    _aggregate_count_tree,
    _format_train_progress_log,
    _is_successful_summary,
    _loss_stats_from_summaries,
    _mean_metric_tree,
    _successful_summaries,
    _sum_counts,
    _timing_profile,
    _zero_successful_batches_error,
)


def _safe_len(loader: Any) -> int | None:
    try:
        return int(len(loader))
    except Exception:
        return None


def _merge_raw_batches(batches: list[dict[str, Any]]) -> dict[str, Any]:
    if not batches:
        raise ValueError("cannot merge zero raw batches")
    merged = {
        "det_targets": [item for batch in batches for item in batch["det_targets"]],
        "tl_attr_targets": [item for batch in batches for item in batch["tl_attr_targets"]],
        "lane_targets": [item for batch in batches for item in batch["lane_targets"]],
        "source_mask": [item for batch in batches for item in batch["source_mask"]],
        "valid_mask": [item for batch in batches for item in batch["valid_mask"]],
        "meta": [item for batch in batches for item in batch["meta"]],
    }
    if all("image" in batch for batch in batches):
        merged["image"] = torch.cat([batch["image"] for batch in batches], dim=0)
    return merged


def _raw_batch_for_metrics(batch: dict[str, Any]) -> dict[str, Any] | None:
    raw_batch = batch.get("_raw_batch")
    if isinstance(raw_batch, dict):
        return raw_batch
    if "det_targets" in batch:
        return batch
    return None


def run_train_epoch(
    trainer: Any,
    loader: Any,
    *,
    epoch: int,
    epoch_total: int | None = None,
    phase_index: int | None = None,
    phase_count: int | None = None,
    phase_name: str | None = None,
    max_batches: int | None = None,
    step_log_path: str | None = None,
    log_every_n_steps: int = 1,
    profile_window: int = 20,
    profile_device_sync: bool = False,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    epoch_started_at_iso = _now_iso()
    step_summaries: list[dict[str, Any]] = []
    timing_window: list[dict[str, Any]] = []
    start_step = trainer.global_step
    total_batches = max_batches if max_batches is not None else _safe_len(loader)
    loader_iter = iter(loader)
    batch_index = 0
    while max_batches is None or batch_index < max_batches:
        fetch_started_at = time.perf_counter()
        try:
            batch = next(loader_iter)
        except StopIteration:
            break
        fetch_ended_at = time.perf_counter()
        batch_index += 1
        step_summary = trainer.train_step(
            batch,
            wait_sec=max(0.0, fetch_ended_at - fetch_started_at),
            profile_device_sync=profile_device_sync,
        )
        timing_window.append(step_summary)
        timing_window = timing_window[-max(1, int(profile_window)) :]
        elapsed_sec = max(0.0, time.perf_counter() - started_at)
        remaining_batches = None
        if total_batches is not None:
            remaining_batches = max(0, int(total_batches) - batch_index)
        profile_summary = _timing_profile(list(timing_window))
        eta_sec = None
        if remaining_batches is not None:
            eta_sec = float(profile_summary["iteration_sec"]["mean"]) * float(remaining_batches)
        step_summary["progress"] = {
            "epoch": int(epoch),
            "iteration": int(batch_index),
            "total_iterations": int(total_batches) if total_batches is not None else None,
            "elapsed_sec": elapsed_sec,
            "eta_sec": eta_sec,
            "epoch_started_at": epoch_started_at_iso,
        }
        step_summary["profile"] = profile_summary
        step_summaries.append(step_summary)
        if step_log_path is not None:
            _append_jsonl(step_log_path, step_summary)
        should_log = batch_index % max(1, int(log_every_n_steps)) == 0
        if total_batches is not None and batch_index == total_batches:
            should_log = True
        if should_log:
            print(
                _format_train_progress_log(
                    stage=trainer.stage,
                    phase_index=phase_index,
                    phase_count=phase_count,
                    phase_name=phase_name,
                    epoch=epoch,
                    epoch_total=epoch_total,
                    batch_index=batch_index,
                    total_batches=total_batches,
                    global_step=step_summary["global_step"],
                    epoch_started_at_iso=epoch_started_at_iso,
                    elapsed_sec=elapsed_sec,
                    eta_sec=eta_sec,
                    losses=step_summary["losses"],
                    profile_summary=step_summary["profile"],
                ),
                flush=True,
            )
    if not step_summaries:
        raise ValueError("train_epoch received zero batches")
    successful_summaries = _successful_summaries(step_summaries)
    if not successful_summaries:
        raise ValueError(_zero_successful_batches_error(step_summaries))
    skipped_summaries = [item for item in step_summaries if not _is_successful_summary(item)]
    ended_at = time.perf_counter()
    return {
        "epoch": int(epoch),
        "epoch_started_at": epoch_started_at_iso,
        "epoch_ended_at": _now_iso(),
        "batches": len(successful_summaries),
        "attempted_batches": len(step_summaries),
        "successful_batches": len(successful_summaries),
        "skipped_batches": len(skipped_summaries),
        "skipped_reasons": dict(
            Counter(str(item.get("skipped_reason") or "unknown") for item in skipped_summaries)
        ),
        "global_step_start": int(start_step),
        "global_step_end": int(trainer.global_step),
        "duration_sec": ended_at - started_at,
        "timing_profile": _timing_profile(step_summaries),
        "losses": _loss_stats_from_summaries(successful_summaries),
        "optimizer_lrs": dict(successful_summaries[-1]["optimizer_lrs"]),
        "assignment": _aggregate_assignment_modes(successful_summaries),
        "attempted_source_counts": _aggregate_count_tree(step_summaries, "source_counts"),
        "skipped_source_counts": _aggregate_count_tree(skipped_summaries, "source_counts"),
        "source_counts": _aggregate_count_tree(successful_summaries, "source_counts"),
        "det_supervision": _aggregate_count_tree(successful_summaries, "det_supervision"),
        "det_components": _aggregate_count_tree(successful_summaries, "det_components"),
    }


def run_validate_epoch(
    trainer: Any,
    loader: Any,
    *,
    epoch: int,
    evaluator: Any = None,
    max_batches: int | None = None,
) -> dict[str, Any]:
    from .metrics import summarize_pv26_metrics

    evaluator = evaluator or trainer.build_evaluator()
    started_at = time.perf_counter()
    batch_summaries: list[dict[str, Any]] = []
    raw_batches: list[dict[str, Any]] = []
    epoch_predictions: list[dict[str, Any]] = []
    for batch_index, batch in enumerate(loader, start=1):
        if max_batches is not None and batch_index > max_batches:
            break
        raw_batch = _raw_batch_for_metrics(batch)
        needs_predictions = raw_batch is not None
        batch_summary = evaluator.evaluate_batch(batch, include_predictions=needs_predictions)
        batch_summaries.append(batch_summary)
        if raw_batch is not None:
            raw_batches.append(raw_batch)
            epoch_predictions.extend(batch_summary.get("predictions", []))
    if not batch_summaries:
        raise ValueError("validate_epoch received zero batches")
    ended_at = time.perf_counter()
    metrics = {}
    if raw_batches:
        metrics = summarize_pv26_metrics(epoch_predictions, _merge_raw_batches(raw_batches))
    else:
        metric_summaries = [item["metrics"] for item in batch_summaries if item.get("metrics")]
        metrics = _mean_metric_tree(metric_summaries) if metric_summaries else {}
    return {
        "epoch": int(epoch),
        "batches": len(batch_summaries),
        "duration_sec": ended_at - started_at,
        "losses": _loss_stats_from_summaries(batch_summaries),
        "counts": _sum_counts(batch_summaries),
        "metrics": metrics,
    }
