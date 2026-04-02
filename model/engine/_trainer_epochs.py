from __future__ import annotations

from collections import Counter
import time
from typing import Any

import torch
try:
    from rich.console import Console
    from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn
except Exception:  # pragma: no cover - optional dependency fallback.
    Console = None
    Progress = None
    BarColumn = None
    TaskProgressColumn = None
    TextColumn = None

from ._trainer_io import _append_jsonl, _now_iso
from ._trainer_reporting import (
    _aggregate_assignment_modes,
    _aggregate_count_tree,
    _format_duration,
    _format_fraction,
    _format_train_progress_log,
    _is_successful_summary,
    _join_segments,
    _loss_stats_from_summaries,
    _mean_metric_tree,
    _percentile,
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


def _augment_lane_family_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(metrics, dict):
        return {}
    lane_family = [
        metrics.get("lane", {}),
        metrics.get("stop_line", {}),
        metrics.get("crosswalk", {}),
    ]
    f1_values = [
        float(item["f1"])
        for item in lane_family
        if isinstance(item, dict) and isinstance(item.get("f1"), (int, float))
    ]
    output = dict(metrics)
    if f1_values:
        output["lane_family"] = {
            "mean_f1": sum(f1_values) / len(f1_values),
            "min_f1": min(f1_values),
        }
    return output


def _progress_console() -> Any:
    if Console is None:
        return None
    return Console(stderr=True)


def _should_use_rich_progress() -> bool:
    console = _progress_console()
    return bool(console is not None and getattr(console, "is_terminal", False) and Progress is not None)


class _RichProgressBar:
    def __init__(self, *, total: int | None, desc: str) -> None:
        if Progress is None:
            raise RuntimeError("rich progress backend is unavailable")
        console = _progress_console()
        if console is None:
            raise RuntimeError("rich console backend is unavailable")
        self.console = console
        self._progress = Progress(
            TextColumn("{task.description}", markup=False),
            BarColumn(bar_width=10),
            TaskProgressColumn(),
            TextColumn("  |  "),
            TextColumn("{task.fields[status]}", markup=False),
            console=self.console,
            transient=False,
            auto_refresh=True,
        )
        self._progress.start()
        self._task_id = self._progress.add_task(desc, total=total, status="")
        self._closed = False

    def update(self, advance: int = 1) -> None:
        self._progress.update(self._task_id, advance=advance)

    def set_postfix_str(self, value: str, refresh: bool = True) -> None:
        self._progress.update(self._task_id, status=str(value), refresh=refresh)

    def write(self, message: str) -> None:
        self.console.print(message, soft_wrap=True, markup=False)

    def close(self) -> None:
        if self._closed:
            return
        self._progress.stop()
        self._closed = True


def _sync_profile_device(device: torch.device, enabled: bool) -> None:
    if enabled and device.type == "cuda":
        torch.cuda.synchronize(device)


def _phase_label(
    *,
    phase_index: int | None,
    phase_count: int | None,
    phase_name: str | None,
) -> str:
    return _join_segments(
        f"phase={_format_fraction(int(phase_index), int(phase_count))}" if phase_index is not None and phase_count is not None else None,
        phase_name,
    )


def _train_progress_desc(
    *,
    stage: str,
    phase_index: int | None,
    phase_count: int | None,
    phase_name: str | None,
    epoch: int,
    epoch_total: int | None,
    epoch_started_at_iso: str,
) -> str:
    return _join_segments(
        "[train]",
        _phase_label(
            phase_index=phase_index,
            phase_count=phase_count,
            phase_name=phase_name,
        ),
        f"epoch={_format_fraction(int(epoch), int(epoch_total) if epoch_total is not None else None)}",
    )


def _train_progress_postfix(
    *,
    batch_index: int,
    total_batches: int | None,
    elapsed_sec: float,
    eta_sec: float | None,
    losses: dict[str, Any],
    profile_summary: dict[str, Any],
) -> str:
    return _join_segments(
        f"iter={_format_fraction(int(batch_index), int(total_batches) if total_batches is not None else None)}",
        f"loss={float(losses.get('total', float('nan'))):.4f}",
        f"eta={_format_duration(eta_sec)}",
    )


def _validation_timing_profile(summaries: list[dict[str, float]]) -> dict[str, Any]:
    if not summaries:
        return {"window_size": 0}
    profile: dict[str, Any] = {"window_size": len(summaries)}
    for key in ("wait_sec", "evaluate_sec", "iteration_sec"):
        values = [float(item.get(key, 0.0)) for item in summaries]
        profile[key] = {
            "mean": sum(values) / len(values),
            "p50": _percentile(values, 0.50),
            "p99": _percentile(values, 0.99),
        }
    return profile


def _validate_progress_desc(
    *,
    stage: str,
    phase_index: int | None,
    phase_count: int | None,
    phase_name: str | None,
    epoch: int,
    epoch_total: int | None,
    epoch_started_at_iso: str,
) -> str:
    return _join_segments(
        "[val]",
        _phase_label(
            phase_index=phase_index,
            phase_count=phase_count,
            phase_name=phase_name,
        ),
        f"epoch={_format_fraction(int(epoch), int(epoch_total) if epoch_total is not None else None)}",
    )


def _validate_progress_postfix(
    *,
    batch_index: int,
    total_batches: int | None,
    elapsed_sec: float,
    eta_sec: float | None,
    profile_summary: dict[str, Any],
    batch_summary: dict[str, Any],
) -> str:
    losses = dict(batch_summary.get("losses", {}))
    return _join_segments(
        f"iter={_format_fraction(int(batch_index), int(total_batches) if total_batches is not None else None)}",
        f"loss={float(losses.get('total', float('nan'))):.4f}",
        f"eta={_format_duration(eta_sec)}",
    )


def _format_validate_progress_log(
    *,
    stage: str,
    phase_index: int | None,
    phase_count: int | None,
    phase_name: str | None,
    epoch: int,
    epoch_total: int | None,
    batch_index: int,
    total_batches: int | None,
    epoch_started_at_iso: str,
    elapsed_sec: float,
    eta_sec: float | None,
    batch_summary: dict[str, Any],
    profile_summary: dict[str, Any],
) -> str:
    header = _join_segments(
        "[val]",
        _phase_label(
            phase_index=phase_index,
            phase_count=phase_count,
            phase_name=phase_name,
        ),
        f"epoch={_format_fraction(int(epoch), int(epoch_total) if epoch_total is not None else None)}",
        f"iter={_format_fraction(int(batch_index), int(total_batches) if total_batches is not None else None)}",
    )
    iteration_profile = profile_summary["iteration_sec"]
    losses = dict(batch_summary.get("losses", {}))
    return "\n".join(
        [
            header,
            "  "
            + _join_segments(
                f"elapsed={_format_duration(elapsed_sec)}",
                f"eta={_format_duration(eta_sec)}",
                f"total={float(losses.get('total', float('nan'))):.4f}",
            ),
            "  "
            + _join_segments(
                "loss",
                f"det={float(losses.get('det', float('nan'))):.4f}",
                f"tl={float(losses.get('tl_attr', float('nan'))):.4f}",
                f"lane={float(losses.get('lane', float('nan'))):.4f}",
                f"stop={float(losses.get('stop_line', float('nan'))):.4f}",
                f"cross={float(losses.get('crosswalk', float('nan'))):.4f}",
            ),
            "  "
            + _join_segments(
                "timing_ms",
                f"eval={profile_summary['evaluate_sec']['mean'] * 1000.0:.3f}",
                f"total={iteration_profile['mean'] * 1000.0:.3f}",
            ),
        ]
    )


def _emit_progress_message(message: str, *, progress_bar: Any = None) -> None:
    if progress_bar is not None and hasattr(progress_bar, "write"):
        progress_bar.write(message)
        return
    print(message, flush=True)


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
    progress_bar = None
    if _should_use_rich_progress():
        progress_bar = _RichProgressBar(
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
            _emit_progress_message(
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
                progress_bar=progress_bar,
            )
    if progress_bar is not None:
        progress_bar.close()
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
    total_batches = max_batches if max_batches is not None else _safe_len(loader)
    loader_iter = iter(loader)
    batch_index = 0
    progress_bar = None
    if _should_use_rich_progress():
        progress_bar = _RichProgressBar(
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
    while max_batches is None or batch_index < max_batches:
        fetch_started_at = time.perf_counter()
        try:
            batch = next(loader_iter)
        except StopIteration:
            break
        fetch_ended_at = time.perf_counter()
        batch_index += 1
        raw_batch = _raw_batch_for_metrics(batch)
        needs_predictions = raw_batch is not None
        _sync_profile_device(trainer.device, profile_device_sync)
        evaluate_started_at = time.perf_counter()
        batch_summary = evaluator.evaluate_batch(batch, include_predictions=needs_predictions)
        _sync_profile_device(trainer.device, profile_device_sync)
        evaluate_ended_at = time.perf_counter()
        batch_summary = dict(batch_summary)
        batch_timing = {
            "wait_sec": max(0.0, fetch_ended_at - fetch_started_at),
            "evaluate_sec": max(0.0, evaluate_ended_at - evaluate_started_at),
        }
        batch_timing["iteration_sec"] = float(batch_timing["wait_sec"]) + float(batch_timing["evaluate_sec"])
        batch_summary["timing"] = batch_timing
        batch_summaries.append(batch_summary)
        timing_window.append(batch_timing)
        timing_window = timing_window[-max(1, int(profile_window)) :]
        elapsed_sec = max(0.0, time.perf_counter() - started_at)
        remaining_batches = None
        if total_batches is not None:
            remaining_batches = max(0, int(total_batches) - batch_index)
        profile_summary = _validation_timing_profile(list(timing_window))
        eta_sec = None
        if remaining_batches is not None:
            eta_sec = float(profile_summary["iteration_sec"]["mean"]) * float(remaining_batches)
        should_log = batch_index % max(1, int(log_every_n_steps)) == 0
        if total_batches is not None and batch_index == total_batches:
            should_log = True
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
            _emit_progress_message(
                _format_validate_progress_log(
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
        if progress_bar is not None:
            progress_bar.close()
        raise ValueError("validate_epoch received zero batches")
    metric_summary_started_at = time.perf_counter()
    phase_label = _phase_label(
        phase_index=phase_index,
        phase_count=phase_count,
        phase_name=phase_name,
    )
    phase_prefix = f"{phase_label} " if phase_label else ""
    _emit_progress_message(
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
    if raw_batches:
        metrics = summarize_pv26_metrics(epoch_predictions, _merge_raw_batches(raw_batches))
    else:
        metric_summaries = [item["metrics"] for item in batch_summaries if item.get("metrics")]
        metrics = _mean_metric_tree(metric_summaries) if metric_summaries else {}
    metrics = _augment_lane_family_metrics(metrics)
    metric_summary_ended_at = time.perf_counter()
    _emit_progress_message(
        (
            "[val] "
            f"{phase_prefix}"
            f"stage={trainer.stage} "
            f"epoch={_format_fraction(int(epoch), int(epoch_total) if epoch_total is not None else None)} "
            f"metrics_ready metric_summary_sec={max(0.0, metric_summary_ended_at - metric_summary_started_at):.3f}"
        ).strip(),
        progress_bar=progress_bar,
    )
    if progress_bar is not None:
        progress_bar.close()
    return {
        "epoch": int(epoch),
        "batches": len(batch_summaries),
        "epoch_started_at": epoch_started_at_iso,
        "duration_sec": metric_summary_ended_at - started_at,
        "metric_summary_sec": max(0.0, metric_summary_ended_at - metric_summary_started_at),
        "timing_profile": _validation_timing_profile([dict(item["timing"]) for item in batch_summaries]),
        "losses": _loss_stats_from_summaries(batch_summaries),
        "counts": _sum_counts(batch_summaries),
        "metrics": metrics,
    }
