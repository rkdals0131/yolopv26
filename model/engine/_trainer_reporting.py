from __future__ import annotations

from collections import Counter
import math
from typing import Any

from common.train_runtime import build_progress_status as _common_build_progress_status
from common.train_runtime import format_duration as _common_format_duration
from common.train_runtime import join_status_segments as _common_join_status_segments
from common.train_runtime import progress_meter as _common_progress_meter
from common.train_runtime import quantile as _common_quantile
from common.train_runtime import timing_profile as _common_timing_profile
from common.train_runtime import write_tensorboard_scalars as _common_write_tensorboard_scalars
from .spec import build_loss_spec


TIMING_KEYS = (
    "wait_sec",
    "load_sec",
    "forward_sec",
    "loss_sec",
    "backward_sec",
    "iteration_sec",
)
TENSORBOARD_LOSS_KEYS = (
    "total",
    "det",
    "tl_attr",
    "lane",
    "stop_line",
    "crosswalk",
)
STAGE_LOSS_WEIGHTS = {
    str(stage["name"]): {str(name): float(value) for name, value in dict(stage["loss_weights"]).items()}
    for stage in build_loss_spec()["training_schedule"]
}


def _is_successful_summary(summary: dict[str, Any]) -> bool:
    return bool(summary.get("successful", summary.get("skipped_reason") is None))


def _successful_summaries(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [item for item in summaries if _is_successful_summary(item)]


def _truncate_debug_text(value: Any, *, max_length: int = 240) -> str | None:
    if value is None:
        return None
    text = " ".join(str(value).split())
    if not text:
        return None
    if len(text) <= max_length:
        return text
    return text[: max_length - 3].rstrip() + "..."


def _zero_successful_batches_error(step_summaries: list[dict[str, Any]]) -> str:
    skipped_summaries = [item for item in step_summaries if not _is_successful_summary(item)]
    skipped_reasons = Counter(str(item.get("skipped_reason") or "unknown") for item in skipped_summaries)
    dominant_reason = skipped_reasons.most_common(1)[0][0] if skipped_reasons else "unknown"
    dominant_detail = None
    for item in reversed(skipped_summaries):
        if str(item.get("skipped_reason") or "unknown") != dominant_reason:
            continue
        dominant_detail = _truncate_debug_text(item.get("skipped_reason_detail"))
        if dominant_detail is not None:
            break
    message = (
        "train_epoch completed with zero successful batches "
        f"(attempted={len(step_summaries)}, skipped={len(skipped_summaries)}, "
        f"dominant_reason={dominant_reason}, reasons={dict(skipped_reasons)})"
    )
    if dominant_detail is not None:
        message += f"; dominant_detail={dominant_detail}"
    return message


def _loss_summary(history: list[dict[str, Any]], name: str) -> dict[str, float]:
    successful = _successful_summaries(history)
    if not successful:
        raise ValueError("loss summary requires at least one successful step")
    values = [float(item["losses"][name]) for item in successful]
    return {
        "last": values[-1],
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
    }


def _aggregate_count_tree(summaries: list[dict[str, Any]], key: str) -> dict[str, Any]:
    if not summaries:
        return {}
    first = summaries[0].get(key, {})
    output: dict[str, Any] = {}
    for name, value in first.items():
        values = [item.get(key, {}).get(name) for item in summaries]
        if isinstance(value, dict):
            output[name] = _aggregate_count_tree(
                [{key: item.get(key, {}).get(name, {})} for item in summaries],
                key,
            )
        elif isinstance(value, (int, bool)):
            output[name] = int(sum(int(item) for item in values))
        elif isinstance(value, float):
            output[name] = sum(float(item) for item in values) / len(values)
    return output


def _write_tensorboard_scalars(writer: Any, prefix: str, payload: dict[str, Any], step: int) -> None:
    _common_write_tensorboard_scalars(writer, prefix, payload, step)


def _select_numeric_scalars(payload: dict[str, Any], keys: tuple[str, ...]) -> dict[str, float]:
    output: dict[str, float] = {}
    for key in keys:
        value = payload.get(key)
        if isinstance(value, bool):
            output[key] = 1.0 if value else 0.0
            continue
        if isinstance(value, (int, float)):
            numeric = float(value)
            if math.isfinite(numeric):
                output[key] = numeric
    return output


def _loss_mean_scalars(payload: dict[str, Any]) -> dict[str, float]:
    output: dict[str, float] = {}
    for key in TENSORBOARD_LOSS_KEYS:
        value = payload.get(key)
        if isinstance(value, dict):
            value = value.get("mean")
        if isinstance(value, (int, float)):
            numeric = float(value)
            if math.isfinite(numeric):
                output[key] = numeric
    return output


def _weighted_loss_mean_scalars(payload: dict[str, Any]) -> dict[str, float]:
    weighted = payload.get("weighted", {})
    if not isinstance(weighted, dict):
        return {}
    output: dict[str, float] = {}
    for key in TENSORBOARD_LOSS_KEYS:
        if key == "total":
            continue
        value = weighted.get(key)
        if isinstance(value, dict):
            value = value.get("mean")
        if isinstance(value, (int, float)):
            numeric = float(value)
            if math.isfinite(numeric):
                output[key] = numeric
    return output


def _stage_weighted_losses(stage: str | None, losses: dict[str, Any]) -> dict[str, float]:
    weights = STAGE_LOSS_WEIGHTS.get(str(stage), {})
    output: dict[str, float] = {}
    for key in TENSORBOARD_LOSS_KEYS:
        if key == "total":
            continue
        value = losses.get(key)
        if isinstance(value, dict):
            value = value.get("mean")
        if isinstance(value, (int, float)) and key in weights:
            numeric = float(value) * float(weights[key])
            if math.isfinite(numeric):
                output[key] = numeric
    return output


def _tensorboard_val_metric_scalars(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "detector": _select_numeric_scalars(
            metrics.get("detector", {}),
            ("precision", "recall", "f1", "map50"),
        ),
        "detector_size_buckets": {
            bucket_name: _select_numeric_scalars(payload, ("precision", "recall", "f1", "ap50"))
            for bucket_name, payload in metrics.get("detector", {}).get("size_buckets", {}).items()
            if isinstance(payload, dict)
        },
        "traffic_light": _select_numeric_scalars(
            metrics.get("traffic_light", {}),
            ("combo_accuracy", "mean_f1"),
        ),
        "lane": _select_numeric_scalars(
            metrics.get("lane", {}),
            ("precision", "recall", "f1", "mean_point_distance", "color_accuracy", "type_accuracy"),
        ),
        "stop_line": _select_numeric_scalars(
            metrics.get("stop_line", {}),
            ("precision", "recall", "f1", "mean_point_distance", "mean_angle_error"),
        ),
        "crosswalk": _select_numeric_scalars(
            metrics.get("crosswalk", {}),
            ("precision", "recall", "f1", "mean_polygon_iou", "mean_vertex_distance"),
        ),
        "lane_family": _select_numeric_scalars(
            metrics.get("lane_family", {}),
            ("mean_f1", "min_f1"),
        ),
    }


def _timing_profile_mean_scalars(profile: dict[str, Any]) -> dict[str, float]:
    output: dict[str, float] = {}
    for key in TIMING_KEYS:
        raw = profile.get(key)
        if not isinstance(raw, dict):
            continue
        value = raw.get("mean")
        if isinstance(value, (int, float)):
            numeric = float(value)
            if math.isfinite(numeric):
                output[key] = numeric
    return output


def _tensorboard_train_step_payload(summary: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "profile_sec": _select_numeric_scalars(summary["timing"], TIMING_KEYS),
    }
    if summary["successful"]:
        payload["loss"] = _select_numeric_scalars(summary["losses"], TENSORBOARD_LOSS_KEYS)
        weighted = _stage_weighted_losses(str(summary.get("stage")), summary["losses"])
        if weighted:
            payload["loss_weighted"] = weighted
    return payload


def _tensorboard_progress_payload(
    step_summary: dict[str, Any],
    *,
    epoch: int,
    batch_index: int,
    total_batches: int | None,
    elapsed_sec: float,
    eta_sec: float | None,
    profile_summary: dict[str, Any],
) -> dict[str, Any]:
    return {
        "progress": {
            "epoch": int(epoch),
            "elapsed_sec": elapsed_sec,
            "eta_sec": eta_sec if eta_sec is not None else 0.0,
            "global_step": int(step_summary["global_step"]),
            "iteration_in_epoch": int(batch_index),
            "total_iterations_in_epoch": int(total_batches) if total_batches is not None else 0,
        },
        "profile_sec": _timing_profile_mean_scalars(profile_summary),
    }


def _tensorboard_epoch_payload(epoch_summary: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "train": {
            "loss_mean": _loss_mean_scalars(epoch_summary["train"]["losses"]),
        },
        "lr": _select_numeric_scalars(epoch_summary["train"].get("optimizer_lrs", {}), ("trunk", "heads")),
    }
    weighted_train = _weighted_loss_mean_scalars(epoch_summary["train"]["losses"])
    if weighted_train:
        payload["train"]["loss_weighted_mean"] = weighted_train
    val_summary = epoch_summary.get("val")
    if isinstance(val_summary, dict):
        payload["val"] = {
            "loss_mean": _loss_mean_scalars(val_summary.get("losses", {})),
        }
        weighted_val = _weighted_loss_mean_scalars(val_summary.get("losses", {}))
        if weighted_val:
            payload["val"]["loss_weighted_mean"] = weighted_val
        val_metrics = _tensorboard_val_metric_scalars(val_summary.get("metrics", {}))
        if any(val_metrics.values()):
            payload["val"]["metrics"] = val_metrics
    return payload


def _percentile(values: list[float], quantile: float) -> float:
    return _common_quantile(values, quantile)


def _timing_profile(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    return _common_timing_profile(
        summaries,
        keys=TIMING_KEYS,
        value_resolver=lambda item, key: item.get("timing", {}).get(key, 0.0),
    )


def _format_duration(seconds: float | None) -> str:
    return _common_format_duration(seconds, unavailable="n/a")


def _format_fraction(current: int, total: int | None) -> str:
    if total is None:
        return f"{current}/?"
    return f"{current}/{total}"


def _phase_label(
    *,
    phase_index: int | None,
    phase_count: int | None,
    phase_name: str | None,
) -> str:
    tokens: list[str] = []
    if phase_index is not None and phase_count is not None:
        tokens.append(f"phase={_format_fraction(int(phase_index), int(phase_count))}")
    if phase_name:
        tokens.append(f"phase_name={phase_name}")
    return " ".join(tokens)


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
    tokens = ["[train]"]
    phase_label = _phase_label(
        phase_index=phase_index,
        phase_count=phase_count,
        phase_name=phase_name,
    )
    if phase_label:
        tokens.append(phase_label)
    tokens.extend(
        [
            f"stage={stage}",
            f"epoch={_format_fraction(int(epoch), int(epoch_total) if epoch_total is not None else None)}",
            f"epoch_start={epoch_started_at_iso}",
        ]
    )
    return " ".join(tokens)


def _train_progress_postfix(
    *,
    batch_index: int,
    total_batches: int | None,
    elapsed_sec: float,
    eta_sec: float | None,
    losses: dict[str, Any],
    profile_summary: dict[str, Any],
) -> str:
    iteration_profile = profile_summary["iteration_sec"]
    return " ".join(
        [
            f"iter={_format_fraction(int(batch_index), int(total_batches) if total_batches is not None else None)}",
            f"elapsed={_format_duration(elapsed_sec)}",
            f"eta={_format_duration(eta_sec)}",
            f"loss={float(losses.get('total', float('nan'))):.4f}",
            (
                "iter_ms="
                f"{iteration_profile['mean'] * 1000.0:.1f}/"
                f"{iteration_profile['p50'] * 1000.0:.1f}/"
                f"{iteration_profile['p99'] * 1000.0:.1f}"
            ),
            (
                "timing_ms="
                f"wait:{profile_summary['wait_sec']['mean'] * 1000.0:.1f},"
                f"load:{profile_summary['load_sec']['mean'] * 1000.0:.1f},"
                f"fwd:{profile_summary['forward_sec']['mean'] * 1000.0:.1f},"
                f"loss:{profile_summary['loss_sec']['mean'] * 1000.0:.1f},"
                f"bwd:{profile_summary['backward_sec']['mean'] * 1000.0:.1f}"
            ),
        ]
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
    tokens = ["[val]"]
    phase_label = _phase_label(
        phase_index=phase_index,
        phase_count=phase_count,
        phase_name=phase_name,
    )
    if phase_label:
        tokens.append(phase_label)
    tokens.extend(
        [
            f"stage={stage}",
            f"epoch={_format_fraction(int(epoch), int(epoch_total) if epoch_total is not None else None)}",
            f"epoch_start={epoch_started_at_iso}",
        ]
    )
    return " ".join(tokens)


def _validate_progress_postfix(
    *,
    batch_index: int,
    total_batches: int | None,
    elapsed_sec: float,
    eta_sec: float | None,
    profile_summary: dict[str, Any],
    batch_summary: dict[str, Any],
) -> str:
    iteration_profile = profile_summary["iteration_sec"]
    losses = dict(batch_summary.get("losses", {}))
    return " ".join(
        [
            f"iter={_format_fraction(int(batch_index), int(total_batches) if total_batches is not None else None)}",
            f"elapsed={_format_duration(elapsed_sec)}",
            f"eta={_format_duration(eta_sec)}",
            f"loss={float(losses.get('total', float('nan'))):.4f}",
            (
                "iter_ms="
                f"{iteration_profile['mean'] * 1000.0:.1f}/"
                f"{iteration_profile['p50'] * 1000.0:.1f}/"
                f"{iteration_profile['p99'] * 1000.0:.1f}"
            ),
            (
                "timing_ms="
                f"wait:{profile_summary['wait_sec']['mean'] * 1000.0:.1f},"
                f"eval:{profile_summary['evaluate_sec']['mean'] * 1000.0:.1f}"
            ),
        ]
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
    del stage, epoch_started_at_iso
    header = _join_segments(
        "[val]",
        f"phase={_format_fraction(int(phase_index), int(phase_count))}" if phase_index is not None and phase_count is not None else None,
        phase_name,
        f"epoch={_format_fraction(int(epoch), int(epoch_total) if epoch_total is not None else None)}",
        f"iter={_format_fraction(int(batch_index), int(total_batches) if total_batches is not None else None)}",
    )
    iteration_profile = profile_summary["iteration_sec"]
    losses = dict(batch_summary.get("losses", {}))
    progress_line = _common_build_progress_status(
        current=int(batch_index),
        total=int(total_batches) if total_batches is not None else None,
        segments=(
            f"elapsed={_format_duration(elapsed_sec)}",
            f"eta={_format_duration(eta_sec)}",
        ),
    )
    loss_primary = _join_segments(
        "loss",
        f"det={float(losses.get('det', float('nan'))):.4f}",
        f"tl={float(losses.get('tl_attr', float('nan'))):.4f}",
        f"lane={float(losses.get('lane', float('nan'))):.4f}",
        f"total={float(losses.get('total', float('nan'))):.4f}",
    )
    loss_secondary = _join_segments(
        f"stop={float(losses.get('stop_line', float('nan'))):.4f}",
        f"cross={float(losses.get('crosswalk', float('nan'))):.4f}",
    )
    return "\n".join(        [
            header,
            f"  {progress_line}",
            f"  {loss_primary}",
            f"  {loss_secondary}",
            "  "
            + _join_segments(
                "timing_ms",
                f"eval={profile_summary['evaluate_sec']['mean'] * 1000.0:.3f}",
                f"total={iteration_profile['mean'] * 1000.0:.3f}",
            ),
        ]
    )


def _join_segments(*segments: Any) -> str:
    return _common_join_status_segments(*segments)


def _progress_meter(current: int, total: int | None, *, width: int = 8) -> str | None:
    return _common_progress_meter(current, total, width=width)


def _format_train_progress_log(
    *,
    stage: str,
    phase_index: int | None,
    phase_count: int | None,
    phase_name: str | None,
    epoch: int,
    epoch_total: int | None,
    batch_index: int,
    total_batches: int | None,
    global_step: int,
    epoch_started_at_iso: str,
    elapsed_sec: float,
    eta_sec: float | None,
    losses: dict[str, Any],
    profile_summary: dict[str, Any],
) -> str:
    header = _join_segments(
        "[train]",
        f"phase={_format_fraction(int(phase_index), int(phase_count))}" if phase_index is not None and phase_count is not None else None,
        phase_name,
        f"epoch={_format_fraction(int(epoch), int(epoch_total) if epoch_total is not None else None)}",
        f"iter={_format_fraction(int(batch_index), int(total_batches) if total_batches is not None else None)}",
    )
    iteration_profile = profile_summary["iteration_sec"]
    progress_line = _common_build_progress_status(
        current=int(batch_index),
        total=int(total_batches) if total_batches is not None else None,
        segments=(
            f"elapsed={_format_duration(elapsed_sec)}",
            f"eta={_format_duration(eta_sec)}",
            f"step={int(global_step)}",
        ),
    )
    loss_primary = _join_segments(
        "loss",
        f"total={float(losses.get('total', float('nan'))):.4f}",
        f"det={float(losses.get('det', float('nan'))):.4f}",
        f"tl={float(losses.get('tl_attr', float('nan'))):.4f}",
        f"lane={float(losses.get('lane', float('nan'))):.4f}",
    )
    loss_secondary = _join_segments(
        f"stop={float(losses.get('stop_line', float('nan'))):.4f}",
        f"cross={float(losses.get('crosswalk', float('nan'))):.4f}",
    )
    return "\n".join(
        [
            header,
            f"  {progress_line}",
            f"  {loss_primary}",
            f"  {loss_secondary}",
            "  "
            + _join_segments(
                "timing_ms",
                f"load={profile_summary['load_sec']['mean'] * 1000.0:.3f}",
                f"fwd={profile_summary['forward_sec']['mean'] * 1000.0:.3f}",
                f"loss={profile_summary['loss_sec']['mean'] * 1000.0:.3f}",
                f"bwd={profile_summary['backward_sec']['mean'] * 1000.0:.3f}",
                f"total={iteration_profile['mean'] * 1000.0:.3f}",
            ),
        ]
    )


def _format_train_live_detail(
    *,
    losses: dict[str, Any],
    profile_summary: dict[str, Any],
) -> str:
    del losses
    iteration_profile = profile_summary["iteration_sec"]
    return _join_segments(
        "timing_ms",
        f"load={profile_summary['load_sec']['mean'] * 1000.0:.3f}",
        f"fwd={profile_summary['forward_sec']['mean'] * 1000.0:.3f}",
        f"loss={profile_summary['loss_sec']['mean'] * 1000.0:.3f}",
        f"bwd={profile_summary['backward_sec']['mean'] * 1000.0:.3f}",
        f"total={iteration_profile['mean'] * 1000.0:.3f}",
    )


def _format_validate_live_detail(
    *,
    elapsed_sec: float,
    eta_sec: float | None,
    batch_summary: dict[str, Any],
    profile_summary: dict[str, Any],
) -> str:
    del elapsed_sec, eta_sec, batch_summary
    iteration_profile = profile_summary["iteration_sec"]
    return _join_segments(
        "timing_ms",
        f"eval={profile_summary['evaluate_sec']['mean'] * 1000.0:.3f}",
        f"total={iteration_profile['mean'] * 1000.0:.3f}",
    )


def _loss_mean_for_log(summary: dict[str, Any] | None) -> float | None:
    if not isinstance(summary, dict):
        return None
    losses = summary.get("losses", {})
    if not isinstance(losses, dict):
        return None
    total = losses.get("total")
    if isinstance(total, dict):
        total = total.get("mean", total.get("last"))
    if isinstance(total, (int, float)):
        numeric = float(total)
        if math.isfinite(numeric):
            return numeric
    return None


def _format_epoch_completion_log(
    *,
    phase_index: int | None,
    phase_count: int | None,
    phase_name: str | None,
    epoch: int,
    epoch_total: int | None,
    train_summary: dict[str, Any],
    val_summary: dict[str, Any] | None,
    best_metric_value: float | None,
    best_epoch: int | None,
    is_best: bool,
) -> str:
    checkpoint_label = "last,best" if is_best else "last"
    segments: list[Any] = [
        "[epoch]",
        f"phase={_format_fraction(int(phase_index), int(phase_count))}" if phase_index is not None and phase_count is not None else None,
        phase_name,
        f"epoch={_format_fraction(int(epoch), int(epoch_total) if epoch_total is not None else None)}",
    ]
    train_loss = _loss_mean_for_log(train_summary)
    if train_loss is not None:
        segments.append(f"train={train_loss:.4f}")
    val_loss = _loss_mean_for_log(val_summary)
    if val_loss is not None:
        segments.append(f"val={val_loss:.4f}")
    if isinstance(best_metric_value, (int, float)) and math.isfinite(float(best_metric_value)) and best_epoch is not None:
        segments.append(f"best={float(best_metric_value):.4f}@{int(best_epoch)}")
    segments.append(f"checkpoint={checkpoint_label}")
    return _join_segments(*segments)


def _loss_stats_from_summaries(summaries: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    if not summaries:
        return {}
    names = list(summaries[0]["losses"].keys())
    output = {
        name: {
            "mean": sum(float(item["losses"][name]) for item in summaries) / len(summaries),
            "min": min(float(item["losses"][name]) for item in summaries),
            "max": max(float(item["losses"][name]) for item in summaries),
            "last": float(summaries[-1]["losses"][name]),
        }
        for name in names
    }
    stage = str(summaries[-1].get("stage", ""))
    weights = STAGE_LOSS_WEIGHTS.get(stage, {})
    weighted: dict[str, dict[str, float]] = {}
    for name in names:
        if name == "total" or name not in weights:
            continue
        weighted[name] = {
            "mean": output[name]["mean"] * float(weights[name]),
            "min": output[name]["min"] * float(weights[name]),
            "max": output[name]["max"] * float(weights[name]),
            "last": output[name]["last"] * float(weights[name]),
            "weight": float(weights[name]),
        }
    if weighted:
        output["weighted"] = weighted
    return output


def _sum_counts(summaries: list[dict[str, Any]]) -> dict[str, int]:
    if not summaries:
        return {}
    keys = list(summaries[0]["counts"].keys())
    return {
        key: int(sum(int(item["counts"].get(key, 0)) for item in summaries))
        for key in keys
    }


def _aggregate_assignment_modes(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    det_modes = Counter(str(item["assignment"]["det"]) for item in summaries)
    lane_tasks = list(summaries[0]["assignment"]["lane"].keys()) if summaries else []
    lane_modes = {
        task: dict(Counter(str(item["assignment"]["lane"].get(task, "unknown")) for item in summaries))
        for task in lane_tasks
    }
    return {
        "det_modes": dict(det_modes),
        "lane_modes": lane_modes,
    }


def _mean_metric_tree(metric_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    if not metric_summaries:
        return {}
    first = metric_summaries[0]
    output: dict[str, Any] = {}
    count_keys = {"tp", "fp", "fn", "gt_count", "matched_pairs"}
    for key, value in first.items():
        values = [item[key] for item in metric_summaries if key in item]
        if isinstance(value, dict):
            output[key] = _mean_metric_tree(values)
        elif key in count_keys:
            output[key] = int(sum(int(item) for item in values))
        else:
            output[key] = sum(float(item) for item in values) / len(values)
    return output
