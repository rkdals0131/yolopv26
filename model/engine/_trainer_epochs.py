from __future__ import annotations

from collections import Counter
import math
import time
from typing import Any

import torch

from .batch import augment_lane_family_metrics, raw_batch_for_metrics
from ._trainer_io import _append_jsonl, _now_iso
from .metrics import PV26MetricConfig, summarize_pv26_metrics, summarize_pv26_tensorboard_histograms
from .trainer_reporting import (
    _aggregate_assignment_modes,
    _aggregate_count_tree,
    _format_fraction,
    _format_train_live_detail,
    _format_train_progress_log,
    _format_validate_progress_log,
    _format_validate_live_detail,
    _is_successful_summary,
    _loss_stats_from_summaries,
    _mean_metric_tree,
    _phase_label,
    _successful_summaries,
    _sum_counts,
    _timing_profile,
    _train_progress_desc,
    _train_progress_postfix,
    _validate_progress_desc,
    _validate_progress_postfix,
    _validation_timing_profile,
    _zero_successful_batches_error,
)
from .trainer_progress import (
    build_progress_bar,
    emit_progress_message,
    next_loader_batch,
    safe_len,
    should_log_progress,
    summarize_progress,
    sync_profile_device,
    update_timing_window,
)

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


def _is_step_anomaly(summary: dict[str, Any]) -> bool:
    if not _is_successful_summary(summary):
        return True
    if summary.get("skipped_reason"):
        return True
    total_loss = summary.get("losses", {}).get("total")
    return isinstance(total_loss, float) and not math.isfinite(total_loss)


def _slim_step_summary(summary: dict[str, Any], *, include_grad_details: bool) -> dict[str, Any]:
    keys = (
        "history_index",
        "global_step",
        "stage",
        "batch_size",
        "successful",
        "losses",
        "det_components",
        "optimizer_step",
        "micro_step",
        "accumulate_steps",
        "skipped_reason",
        "skipped_reason_detail",
        "skipped_steps",
        "amp_enabled",
        "gradient_scale",
        "optimizer_lrs",
        "assignment",
        "timing",
        "source_counts",
        "det_supervision",
        "progress",
        "profile",
    )
    slim = {key: summary[key] for key in keys if key in summary}
    if include_grad_details and "multitask_conflict" in summary:
        slim["multitask_conflict"] = summary["multitask_conflict"]
    return slim


def _should_log_step_summary(
    *,
    batch_index: int,
    total_batches: int | None,
    every_n_steps: int,
    enabled: bool,
    summary: dict[str, Any],
) -> bool:
    if _is_step_anomaly(summary):
        return True
    if not enabled:
        return False
    if batch_index == 1:
        return True
    if total_batches is not None and batch_index == total_batches:
        return True
    return every_n_steps > 0 and batch_index % every_n_steps == 0


def _mean_nested_values(items: list[dict[str, Any]], key: str) -> dict[str, float]:
    sums: Counter[str] = Counter()
    counts: Counter[str] = Counter()
    for item in items:
        payload = item.get(key)
        if not isinstance(payload, dict):
            continue
        for name, value in payload.items():
            if isinstance(value, (int, float)):
                sums[str(name)] += float(value)
                counts[str(name)] += 1
    return {name: float(sums[name]) / float(counts[name]) for name in sorted(sums) if counts[name]}


def _aggregate_pcgrad_window(*, epoch: int, window: list[dict[str, Any]]) -> dict[str, Any]:
    conflict_counts: Counter[str] = Counter()
    task_conflict_counts: Counter[str] = Counter()
    pairwise_sums: Counter[str] = Counter()
    pairwise_counts: Counter[str] = Counter()
    param_groups: set[str] = set()
    param_count_sum = 0
    param_count_seen = 0
    enabled_steps = 0
    for summary in window:
        pcgrad = summary.get("multitask_conflict")
        if not isinstance(pcgrad, dict) or not pcgrad.get("enabled"):
            continue
        enabled_steps += 1
        for group in pcgrad.get("param_groups", []):
            param_groups.add(str(group))
        if isinstance(pcgrad.get("param_count"), int):
            param_count_sum += int(pcgrad["param_count"])
            param_count_seen += 1
        for pair in pcgrad.get("conflict_pairs", []):
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                continue
            left, right = str(pair[0]), str(pair[1])
            conflict_counts[f"{left}_vs_{right}"] += 1
            task_conflict_counts[left] += 1
        pairwise = pcgrad.get("pairwise_dots")
        if isinstance(pairwise, dict):
            for left, nested in pairwise.items():
                if not isinstance(nested, dict):
                    continue
                for right, value in nested.items():
                    if isinstance(value, (int, float)):
                        key = f"{left}_vs_{right}"
                        pairwise_sums[key] += float(value)
                        pairwise_counts[key] += 1
    denominator = max(enabled_steps, 1)
    return {
        "epoch": int(epoch),
        "window_started_history_index": int(window[0].get("history_index", 0)) if window else 0,
        "window_ended_history_index": int(window[-1].get("history_index", 0)) if window else 0,
        "steps": len(window),
        "pcgrad_enabled_steps": int(enabled_steps),
        "param_groups": sorted(param_groups),
        "mean_param_count": float(param_count_sum) / float(param_count_seen) if param_count_seen else 0.0,
        "conflict_rate": {
            name: float(count) / float(denominator)
            for name, count in sorted(conflict_counts.items())
        },
        "projection_rate_by_task": {
            name: float(count) / float(denominator)
            for name, count in sorted(task_conflict_counts.items())
        },
        "mean_pairwise_dot": {
            name: float(pairwise_sums[name]) / float(pairwise_counts[name])
            for name in sorted(pairwise_sums)
            if pairwise_counts[name]
        },
        "mean_raw_grad_norm": _mean_nested_values(
            [summary["multitask_conflict"] for summary in window if isinstance(summary.get("multitask_conflict"), dict)],
            "raw_grad_norms",
        ),
        "mean_projected_grad_norm": _mean_nested_values(
            [summary["multitask_conflict"] for summary in window if isinstance(summary.get("multitask_conflict"), dict)],
            "projected_grad_norms",
        ),
    }


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
    step_history_enabled: bool = True,
    step_history_every_n_steps: int = 100,
    step_history_include_grad_details: bool = False,
    pcgrad_diagnostics_path: str | None = None,
    pcgrad_diagnostics_enabled: bool = True,
    pcgrad_aggregate_every_n_steps: int = 100,
    pcgrad_keep_raw_every_n_steps: int = 1000,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    epoch_started_at_iso = _now_iso()
    step_summaries: list[dict[str, Any]] = []
    logged_step_count = 0
    pcgrad_window: list[dict[str, Any]] = []
    pcgrad_window_count = 0
    timing_window: list[dict[str, Any]] = []
    start_step = trainer.global_step
    total_batches = max_batches if max_batches is not None else safe_len(loader)
    loader_iter = iter(loader)
    batch_index = 0
    progress_bar = build_progress_bar(
        total=total_batches,
        desc=_train_progress_desc(
            stage=trainer.stage,
            phase_index=phase_index,
            phase_count=phase_count,
            phase_name=phase_name,
            epoch=epoch,
            epoch_total=epoch_total,
            epoch_started_at_iso=epoch_started_at_iso,
        ),
    )
    try:
        while max_batches is None or batch_index < max_batches:
            has_batch, batch, wait_sec = next_loader_batch(loader_iter)
            if not has_batch:
                break
            batch_index += 1
            step_summary = trainer.train_step(
                batch,
                wait_sec=wait_sec,
                profile_device_sync=profile_device_sync,
                store_history=False,
            )
            timing_window = update_timing_window(timing_window, step_summary, profile_window=profile_window)
            profile_summary, elapsed_sec, eta_sec = summarize_progress(
                started_at=started_at,
                batch_index=batch_index,
                total_batches=total_batches,
                timing_window=timing_window,
                profile_builder=_timing_profile,
            )
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
            should_log_step = _should_log_step_summary(
                batch_index=batch_index,
                total_batches=total_batches,
                every_n_steps=step_history_every_n_steps,
                enabled=step_history_enabled,
                summary=step_summary,
            )
            include_grad_details = bool(step_history_include_grad_details)
            if (
                pcgrad_keep_raw_every_n_steps > 0
                and batch_index % pcgrad_keep_raw_every_n_steps == 0
            ) or _is_step_anomaly(step_summary):
                include_grad_details = True
            if should_log_step:
                logged_summary = _slim_step_summary(
                    step_summary,
                    include_grad_details=include_grad_details,
                )
                trainer.history.append(logged_summary)
                logged_step_count += 1
                if step_log_path is not None:
                    _append_jsonl(step_log_path, logged_summary)
            if pcgrad_diagnostics_enabled:
                pcgrad_window.append(step_summary)
                if pcgrad_aggregate_every_n_steps > 0 and len(pcgrad_window) >= pcgrad_aggregate_every_n_steps:
                    pcgrad_window_count += 1
                    if pcgrad_diagnostics_path is not None:
                        _append_jsonl(
                            pcgrad_diagnostics_path,
                            _aggregate_pcgrad_window(epoch=epoch, window=pcgrad_window),
                        )
                    pcgrad_window = []
            should_log = should_log_progress(
                batch_index=batch_index,
                total_batches=total_batches,
                log_every_n_steps=log_every_n_steps,
            )
            if progress_bar is not None:
                progress_bar.update(1)
                progress_bar.set_postfix_str(
                    _train_progress_postfix(
                        batch_index=batch_index,
                        total_batches=total_batches,
                        elapsed_sec=elapsed_sec,
                        eta_sec=eta_sec,
                        losses=step_summary["losses"],
                        profile_summary=step_summary["profile"],
                    ),
                    refresh=True,
                )
            if should_log:
                detail_message = _format_train_live_detail(
                    losses=step_summary["losses"],
                    profile_summary=step_summary["profile"],
                )
                emit_progress_message(
                    detail_message if progress_bar is not None else _format_train_progress_log(
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
                    progress_bar=progress_bar,
                )
    finally:
        if progress_bar is not None:
            progress_bar.close()
    if not step_summaries:
        raise ValueError("train_epoch received zero batches")
    successful_summaries = _successful_summaries(step_summaries)
    if not successful_summaries:
        raise ValueError(_zero_successful_batches_error(step_summaries))
    skipped_summaries = [item for item in step_summaries if not _is_successful_summary(item)]
    if pcgrad_diagnostics_enabled and pcgrad_window:
        pcgrad_window_count += 1
        if pcgrad_diagnostics_path is not None:
            _append_jsonl(
                pcgrad_diagnostics_path,
                _aggregate_pcgrad_window(epoch=epoch, window=pcgrad_window),
            )
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
        "loss_weights": dict(successful_summaries[-1].get("loss_weights", {})),
        "losses": _loss_stats_from_summaries(successful_summaries),
        "optimizer_lrs": dict(successful_summaries[-1]["optimizer_lrs"]),
        "assignment": _aggregate_assignment_modes(successful_summaries),
        "attempted_source_counts": _aggregate_count_tree(step_summaries, "source_counts"),
        "skipped_source_counts": _aggregate_count_tree(skipped_summaries, "source_counts"),
        "source_counts": _aggregate_count_tree(successful_summaries, "source_counts"),
        "det_supervision": _aggregate_count_tree(successful_summaries, "det_supervision"),
        "det_components": _aggregate_count_tree(successful_summaries, "det_components"),
        "step_history": {
            "enabled": bool(step_history_enabled),
            "logged_steps": int(logged_step_count),
            "every_n_steps": int(step_history_every_n_steps),
            "include_grad_details": bool(step_history_include_grad_details),
        },
        "pcgrad_diagnostics": {
            "enabled": bool(pcgrad_diagnostics_enabled),
            "windows": int(pcgrad_window_count),
            "aggregate_every_n_steps": int(pcgrad_aggregate_every_n_steps),
            "keep_raw_every_n_steps": int(pcgrad_keep_raw_every_n_steps),
        },
    }


def run_validate_epoch(
    trainer: Any,
    loader: Any,
    *,
    epoch: int,
    epoch_total: int | None = None,
    phase_index: int | None = None,
    phase_count: int | None = None,
    phase_name: str | None = None,
    evaluator: Any = None,
    max_batches: int | None = None,
    log_every_n_steps: int = 1,
    profile_window: int = 20,
    profile_device_sync: bool = False,
) -> dict[str, Any]:
    from .metrics import summarize_pv26_metrics

    evaluator = evaluator or trainer.build_evaluator()
    started_at = time.perf_counter()
    epoch_started_at_iso = _now_iso()
    batch_summaries: list[dict[str, Any]] = []
    raw_batches: list[dict[str, Any]] = []
    epoch_predictions: list[dict[str, Any]] = []
    timing_window: list[dict[str, float]] = []
    total_batches = max_batches if max_batches is not None else safe_len(loader)
    loader_iter = iter(loader)
    batch_index = 0
    progress_bar = build_progress_bar(
        total=total_batches,
        desc=_validate_progress_desc(
            stage=trainer.stage,
            phase_index=phase_index,
            phase_count=phase_count,
            phase_name=phase_name,
            epoch=epoch,
            epoch_total=epoch_total,
            epoch_started_at_iso=epoch_started_at_iso,
        ),
    )
    try:
        while max_batches is None or batch_index < max_batches:
            has_batch, batch, wait_sec = next_loader_batch(loader_iter)
            if not has_batch:
                break
            batch_index += 1
            raw_batch = raw_batch_for_metrics(batch)
            needs_predictions = raw_batch is not None
            sync_profile_device(trainer.device, profile_device_sync)
            evaluate_started_at = time.perf_counter()
            batch_summary = evaluator.evaluate_batch(batch, include_predictions=needs_predictions)
            sync_profile_device(trainer.device, profile_device_sync)
            evaluate_ended_at = time.perf_counter()
            batch_summary = dict(batch_summary)
            batch_summary["loss_weights"] = {
                str(name): float(value)
                for name, value in dict(getattr(trainer.criterion, "loss_weights", {})).items()
            }
            batch_timing = {
                "wait_sec": wait_sec,
                "evaluate_sec": max(0.0, evaluate_ended_at - evaluate_started_at),
            }
            batch_timing["iteration_sec"] = float(batch_timing["wait_sec"]) + float(batch_timing["evaluate_sec"])
            batch_summary["timing"] = batch_timing
            batch_summaries.append(batch_summary)
            timing_window = update_timing_window(timing_window, batch_timing, profile_window=profile_window)
            profile_summary, elapsed_sec, eta_sec = summarize_progress(
                started_at=started_at,
                batch_index=batch_index,
                total_batches=total_batches,
                timing_window=timing_window,
                profile_builder=_validation_timing_profile,
            )
            should_log = should_log_progress(
                batch_index=batch_index,
                total_batches=total_batches,
                log_every_n_steps=log_every_n_steps,
            )
            if progress_bar is not None:
                progress_bar.update(1)
                progress_bar.set_postfix_str(
                    _validate_progress_postfix(
                        batch_index=batch_index,
                        total_batches=total_batches,
                        elapsed_sec=elapsed_sec,
                        eta_sec=eta_sec,
                        profile_summary=profile_summary,
                        batch_summary=batch_summary,
                    ),
                    refresh=True,
                )
            if should_log:
                detail_message = _format_validate_live_detail(
                    elapsed_sec=elapsed_sec,
                    eta_sec=eta_sec,
                    batch_summary=batch_summary,
                    profile_summary=profile_summary,
                )
                emit_progress_message(
                    detail_message if progress_bar is not None else _format_validate_progress_log(
                        stage=trainer.stage,
                        phase_index=phase_index,
                        phase_count=phase_count,
                        phase_name=phase_name,
                        epoch=epoch,
                        epoch_total=epoch_total,
                        batch_index=batch_index,
                        total_batches=total_batches,
                        epoch_started_at_iso=epoch_started_at_iso,
                        elapsed_sec=elapsed_sec,
                        eta_sec=eta_sec,
                        batch_summary=batch_summary,
                        profile_summary=profile_summary,
                    ),
                    progress_bar=progress_bar,
                )
            if raw_batch is not None:
                raw_batches.append(raw_batch)
                epoch_predictions.extend(batch_summary.get("predictions", []))
        if not batch_summaries:
            raise ValueError("validate_epoch received zero batches")
        metric_summary_started_at = time.perf_counter()
        phase_label = _phase_label(
            phase_index=phase_index,
            phase_count=phase_count,
            phase_name=phase_name,
        )
        phase_prefix = f"{phase_label} " if phase_label else ""
        emit_progress_message(
            (
                "[val] "
                f"{phase_prefix}"
                f"stage={trainer.stage} "
                f"epoch={_format_fraction(int(epoch), int(epoch_total) if epoch_total is not None else None)} "
                f"summarizing_metrics batches={len(batch_summaries)} "
                f"epoch_start={epoch_started_at_iso}"
            ).strip(),
            progress_bar=progress_bar,
        )
        metrics = {}
        merged_raw_batch: dict[str, Any] | None = None
        if raw_batches:
            merged_raw_batch = _merge_raw_batches(raw_batches)
            metrics = summarize_pv26_metrics(epoch_predictions, merged_raw_batch)
        else:
            metric_summaries = [item["metrics"] for item in batch_summaries if item.get("metrics")]
            metrics = _mean_metric_tree(metric_summaries) if metric_summaries else {}
        metrics = augment_lane_family_metrics(metrics)
        tensorboard_histograms = {}
        if merged_raw_batch is not None:
            tensorboard_histograms = summarize_pv26_tensorboard_histograms(
                epoch_predictions,
                merged_raw_batch,
                config=PV26MetricConfig(),
            )
        metric_summary_ended_at = time.perf_counter()
        emit_progress_message(
            (
                "[val] "
                f"{phase_prefix}"
                f"stage={trainer.stage} "
                f"epoch={_format_fraction(int(epoch), int(epoch_total) if epoch_total is not None else None)} "
                f"metrics_ready metric_summary_sec={max(0.0, metric_summary_ended_at - metric_summary_started_at):.3f}"
            ).strip(),
            progress_bar=progress_bar,
        )
        return {
            "epoch": int(epoch),
            "batches": len(batch_summaries),
            "epoch_started_at": epoch_started_at_iso,
            "duration_sec": metric_summary_ended_at - started_at,
            "metric_summary_sec": max(0.0, metric_summary_ended_at - metric_summary_started_at),
            "timing_profile": _validation_timing_profile([dict(item["timing"]) for item in batch_summaries]),
            "loss_weights": dict(batch_summaries[-1].get("loss_weights", {})),
            "losses": _loss_stats_from_summaries(batch_summaries),
            "counts": _sum_counts(batch_summaries),
            "metrics": metrics,
            "tensorboard_histograms": tensorboard_histograms,
        }
    finally:
        if progress_bar is not None:
            progress_bar.close()
