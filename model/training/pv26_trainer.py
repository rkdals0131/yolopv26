from __future__ import annotations

from collections import Counter, deque
from contextlib import nullcontext
from datetime import datetime
import math
import json
from pathlib import Path
import time
from typing import Any, Callable

import torch

from ..encoding import encode_pv26_batch
from ..loss import PV26DetAssignmentUnavailable, PV26MultiTaskLoss
from ..loss.spec import build_loss_spec
from ..trunk import forward_pyramid_features


STAGE_NAMES = (
    "stage_0_smoke",
    "stage_1_frozen_trunk_warmup",
    "stage_2_partial_unfreeze",
    "stage_3_end_to_end_finetune",
)
STAGE_ALIASES = {
    "stage_1_head_warmup": "stage_1_frozen_trunk_warmup",
}
RUN_MANIFEST_VERSION = "pv26-train-run-v1"
OD_CLASSES = tuple(build_loss_spec()["model_contract"]["od_classes"])
TIMING_KEYS = (
    "wait_sec",
    "load_sec",
    "forward_sec",
    "loss_sec",
    "backward_sec",
    "iteration_sec",
)
TENSORBOARD_MODES = (
    "curated",
    "full",
)
TENSORBOARD_LOSS_KEYS = (
    "total",
    "det",
    "tl_attr",
    "lane",
    "stop_line",
    "crosswalk",
)
TENSORBOARD_SOURCE_KEYS = (
    "det_source_samples",
    "tl_attr_source_samples",
    "lane_source_samples",
    "stop_line_source_samples",
    "crosswalk_source_samples",
)


def _canonical_stage(stage: str) -> str:
    return STAGE_ALIASES.get(stage, stage)


def _move_to_device(item: Any, device: torch.device, *, non_blocking: bool = False) -> Any:
    if isinstance(item, torch.Tensor):
        return item.to(device, non_blocking=non_blocking)
    if isinstance(item, dict):
        return {key: _move_to_device(value, device, non_blocking=non_blocking) for key, value in item.items()}
    if isinstance(item, list):
        return [_move_to_device(value, device, non_blocking=non_blocking) for value in item]
    if isinstance(item, tuple):
        return tuple(_move_to_device(value, device, non_blocking=non_blocking) for value in item)
    return item


def _count_parameters(parameters: list[torch.nn.Parameter]) -> int:
    return sum(parameter.numel() for parameter in parameters)


def _trainable_parameters(module: torch.nn.Module) -> list[torch.nn.Parameter]:
    return [parameter for parameter in module.parameters() if parameter.requires_grad]


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


def _default_det_components() -> dict[str, float | int]:
    return {
        "det_obj_loss": 0.0,
        "det_cls_matched_loss": 0.0,
        "det_cls_unmatched_neg_loss": 0.0,
        "det_iou_loss": 0.0,
        "det_l1_loss": 0.0,
        "det_cls_matched_count": 0,
        "det_cls_unmatched_neg_count": 0,
    }


def _criterion_config_from_instance(criterion: torch.nn.Module, stage: str) -> dict[str, Any] | None:
    export_config = getattr(criterion, "export_config", None)
    if not callable(export_config):
        return None
    config = dict(export_config())
    config["stage"] = _canonical_stage(str(config.get("stage", stage)))
    return config


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


def _optimizer_group_hparams(optimizer: torch.optim.Optimizer) -> dict[str, float]:
    values = {
        "trunk_lr": 1e-4,
        "head_lr": 1e-3,
        "weight_decay": 1e-4,
    }
    for index, group in enumerate(optimizer.param_groups):
        group_name = str(group.get("group_name", f"group_{index}"))
        if group_name == "trunk":
            values["trunk_lr"] = float(group.get("lr", values["trunk_lr"]))
            values["weight_decay"] = float(group.get("weight_decay", values["weight_decay"]))
        if group_name == "heads":
            values["head_lr"] = float(group.get("lr", values["head_lr"]))
            values["weight_decay"] = float(group.get("weight_decay", values["weight_decay"]))
    return values


def _default_run_dir() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("runs") / "pv26_train" / f"pv26_fit_{timestamp}"


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _canonical_tensorboard_mode(mode: str) -> str:
    normalized = str(mode).strip().lower()
    aliases = {
        "default": "curated",
        "recommended": "curated",
    }
    resolved = aliases.get(normalized, normalized)
    if resolved not in TENSORBOARD_MODES:
        raise ValueError(f"tensorboard_mode must be one of {TENSORBOARD_MODES}, got {mode!r}")
    return resolved


def _write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return output_path


def _append_jsonl(path: str | Path, payload: dict[str, Any]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True, sort_keys=True) + "\n")
    return output_path


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _flatten_scalar_tree(prefix: str, payload: Any) -> list[tuple[str, float]]:
    scalars: list[tuple[str, float]] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            next_prefix = f"{prefix}/{key}" if prefix else str(key)
            scalars.extend(_flatten_scalar_tree(next_prefix, value))
        return scalars
    if isinstance(payload, (list, tuple)):
        for index, value in enumerate(payload):
            next_prefix = f"{prefix}/{index}" if prefix else str(index)
            scalars.extend(_flatten_scalar_tree(next_prefix, value))
        return scalars
    if isinstance(payload, bool):
        return [(prefix, 1.0 if payload else 0.0)]
    if isinstance(payload, (int, float)):
        numeric = float(payload)
        if math.isfinite(numeric):
            return [(prefix, numeric)]
    return scalars


def _maybe_build_summary_writer(log_dir: Path, *, purge_step: int | None = None):
    try:
        from torch.utils.tensorboard import SummaryWriter
    except Exception as exc:  # pragma: no cover - optional dependency.
        return None, {
            "enabled": False,
            "status": "unavailable",
            "error": str(exc),
            "log_dir": str(log_dir),
            "purge_step": purge_step,
        }
    try:
        writer_kwargs: dict[str, Any] = {"log_dir": str(log_dir)}
        if purge_step is not None:
            writer_kwargs["purge_step"] = int(purge_step)
        writer = SummaryWriter(**writer_kwargs)
    except Exception as exc:  # pragma: no cover - filesystem or environment issue.
        return None, {
            "enabled": False,
            "status": "init_failed",
            "error": str(exc),
            "log_dir": str(log_dir),
            "purge_step": purge_step,
        }
    return writer, {
        "enabled": True,
        "status": "active",
        "error": None,
        "log_dir": str(log_dir),
        "purge_step": purge_step,
    }


def _true_count(value: torch.Tensor) -> int:
    return int(value.to(dtype=torch.int64).sum().item())


def _source_counts(encoded: dict[str, Any]) -> dict[str, int]:
    mask = encoded["mask"]
    return {
        "det_source_samples": _true_count(mask["det_source"]),
        "tl_attr_source_samples": _true_count(mask["tl_attr_source"]),
        "lane_source_samples": _true_count(mask["lane_source"]),
        "stop_line_source_samples": _true_count(mask["stop_line_source"]),
        "crosswalk_source_samples": _true_count(mask["crosswalk_source"]),
    }


def _det_supervision_summary(encoded: dict[str, Any]) -> dict[str, Any]:
    det_source = encoded["mask"]["det_source"].to(dtype=torch.bool)
    allow_objectness = encoded["mask"].get("det_allow_objectness_negatives")
    if allow_objectness is None:
        allow_objectness = torch.ones_like(det_source)
    else:
        allow_objectness = allow_objectness.to(dtype=torch.bool)
    allow_unmatched_class = encoded["mask"].get("det_allow_unmatched_class_negatives")
    if allow_unmatched_class is None:
        allow_unmatched_class = torch.ones_like(det_source)
    else:
        allow_unmatched_class = allow_unmatched_class.to(dtype=torch.bool)
    class_mask = encoded["mask"].get("det_supervised_class_mask")
    if class_mask is None:
        class_mask = torch.ones((det_source.shape[0], len(OD_CLASSES)), dtype=torch.bool, device=det_source.device)
    else:
        class_mask = class_mask.to(dtype=torch.bool)

    partial_det = det_source & (~allow_objectness | ~allow_unmatched_class)
    det_valid = encoded["det_gt"]["valid_mask"].to(dtype=torch.bool)
    det_classes = encoded["det_gt"]["classes"].to(dtype=torch.long)
    batch_size = max(1, int(det_source.shape[0]))

    supervised_class_sample_counts: dict[str, int] = {}
    gt_class_counts: dict[str, int] = {}
    for class_index, class_name in enumerate(OD_CLASSES):
        supervised_class_sample_counts[class_name] = _true_count(class_mask[:, class_index])
        gt_class_counts[class_name] = int(((det_classes == class_index) & det_valid).sum().item())

    return {
        "det_source_samples": _true_count(det_source),
        "partial_det_samples": _true_count(partial_det),
        "objectness_negative_enabled_samples": _true_count(det_source & allow_objectness),
        "class_negative_enabled_samples": _true_count(det_source & allow_unmatched_class),
        "partial_det_ratio": float(_true_count(partial_det)) / float(batch_size),
        "supervised_class_sample_counts": supervised_class_sample_counts,
        "gt_class_counts": gt_class_counts,
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
    for name, value in _flatten_scalar_tree(prefix, payload):
        writer.add_scalar(name, value, global_step=step)


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


def _curated_val_metric_scalars(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "detector": _select_numeric_scalars(
            metrics.get("detector", {}),
            ("precision", "recall", "f1", "map50"),
        ),
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


def _tensorboard_train_step_payload(summary: dict[str, Any], mode: str) -> dict[str, Any]:
    resolved_mode = _canonical_tensorboard_mode(mode)
    if resolved_mode == "full":
        payload = {
            "lr": summary["optimizer_lrs"],
            "timing": summary["timing"],
            "source": summary["source_counts"],
            "det_supervision": summary["det_supervision"],
            "state": {
                "optimizer_step": summary["optimizer_step"],
                "micro_step": summary["micro_step"],
                "skipped_steps": summary["skipped_steps"],
                "amp_enabled": summary["amp_enabled"],
                "gradient_scale": summary["gradient_scale"],
                "successful": summary["successful"],
                "skipped_non_finite_loss": summary["skipped_reason"] == "non_finite_loss",
                "skipped_oom_recovered": summary["skipped_reason"] == "oom_recovered",
                "skipped_det_assignment_unavailable": summary["skipped_reason"] == "det_assignment_unavailable",
            },
        }
        if summary["successful"]:
            payload["loss"] = summary["losses"]
            payload["det_components"] = summary["det_components"]
        return payload
    payload = {
        "lr": _select_numeric_scalars(summary["optimizer_lrs"], ("trunk", "heads")),
        "timing": _select_numeric_scalars(summary["timing"], TIMING_KEYS),
        "health": {
            "gradient_scale": float(summary["gradient_scale"]),
            "skipped_steps": float(summary["skipped_steps"]),
            "successful": bool(summary["successful"]),
        },
    }
    if summary["successful"]:
        payload["loss"] = _select_numeric_scalars(summary["losses"], TENSORBOARD_LOSS_KEYS)
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
    mode: str,
) -> dict[str, Any]:
    resolved_mode = _canonical_tensorboard_mode(mode)
    if resolved_mode == "full":
        return {
            "timing": step_summary["timing"],
            "progress": {
                "epoch": int(epoch),
                "elapsed_sec": elapsed_sec,
                "eta_sec": eta_sec if eta_sec is not None else 0.0,
                "global_step": int(step_summary["global_step"]),
                "iteration_in_epoch": int(batch_index),
                "total_iterations_in_epoch": int(total_batches) if total_batches is not None else 0,
            },
            "profile": profile_summary,
        }
    return {
        "progress": {
            "epoch": int(epoch),
            "elapsed_sec": elapsed_sec,
            "eta_sec": eta_sec if eta_sec is not None else 0.0,
            "global_step": int(step_summary["global_step"]),
            "iteration_in_epoch": int(batch_index),
            "total_iterations_in_epoch": int(total_batches) if total_batches is not None else 0,
        },
        "profile_mean": _timing_profile_mean_scalars(profile_summary),
    }


def _tensorboard_epoch_payload(epoch_summary: dict[str, Any], mode: str) -> dict[str, Any]:
    resolved_mode = _canonical_tensorboard_mode(mode)
    if resolved_mode == "full":
        return {
            "train": {
                "loss": epoch_summary["train"]["losses"],
                "source": epoch_summary["train"]["source_counts"],
                "det_supervision": epoch_summary["train"]["det_supervision"],
                "det_components": epoch_summary["train"].get("det_components", {}),
            },
            "val": epoch_summary.get("val", {}),
            "scheduler": {"lr": epoch_summary.get("scheduler_lrs", [])},
        }
    payload = {
        "train": {
            "loss_mean": _loss_mean_scalars(epoch_summary["train"]["losses"]),
            "duration_sec": float(epoch_summary["train"]["duration_sec"]),
            "skipped_batches": float(epoch_summary["train"]["skipped_batches"]),
            "source": _select_numeric_scalars(epoch_summary["train"]["source_counts"], TENSORBOARD_SOURCE_KEYS),
            "det_supervision": _select_numeric_scalars(
                epoch_summary["train"]["det_supervision"],
                ("partial_det_ratio",),
            ),
        },
        "scheduler": {"lr": epoch_summary.get("scheduler_lrs", [])},
    }
    val_summary = epoch_summary.get("val")
    if isinstance(val_summary, dict):
        payload["val"] = {
            "loss_mean": _loss_mean_scalars(val_summary.get("losses", {})),
            "duration_sec": float(val_summary.get("duration_sec", 0.0)),
        }
        val_metrics = _curated_val_metric_scalars(val_summary.get("metrics", {}))
        if any(val_metrics.values()):
            payload["val"]["metrics"] = val_metrics
    return payload


def _safe_len(loader: Any) -> int | None:
    try:
        return int(len(loader))
    except Exception:
        return None


def _sync_timing_device(device: torch.device, enabled: bool) -> None:
    if enabled and device.type == "cuda":
        torch.cuda.synchronize(device)


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * float(quantile)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _timing_profile(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    if not summaries:
        return {"window_size": 0}
    profile: dict[str, Any] = {"window_size": len(summaries)}
    for key in TIMING_KEYS:
        values = [float(item.get("timing", {}).get(key, 0.0)) for item in summaries]
        profile[key] = {
            "mean": sum(values) / len(values),
            "p50": _percentile(values, 0.50),
            "p99": _percentile(values, 0.99),
        }
    return profile


def _format_duration(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(float(seconds)):
        return "n/a"
    total_seconds = max(0, int(round(float(seconds))))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _format_fraction(current: int, total: int | None) -> str:
    if total is None:
        return f"{current}/?"
    return f"{current}/{total}"


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
    header_tokens = ["[train]"]
    if phase_index is not None and phase_count is not None:
        header_tokens.append(f"phase={_format_fraction(int(phase_index), int(phase_count))}")
    if phase_name:
        header_tokens.append(f"phase_name={phase_name}")
    header_tokens.extend(
        [
            f"stage={stage}",
            f"epoch={_format_fraction(int(epoch), int(epoch_total) if epoch_total is not None else None)}",
            f"iter={_format_fraction(int(batch_index), int(total_batches) if total_batches is not None else None)}",
            f"global_step={int(global_step)}",
        ]
    )
    iteration_profile = profile_summary["iteration_sec"]
    return "\n".join(
        [
            " ".join(header_tokens),
            (
                "  progress: "
                f"epoch_start={epoch_started_at_iso} "
                f"elapsed={_format_duration(elapsed_sec)} "
                f"eta={_format_duration(eta_sec)}"
            ),
            (
                "  loss: "
                f"total={float(losses.get('total', float('nan'))):.4f} "
                f"det={float(losses.get('det', float('nan'))):.4f} "
                f"tl={float(losses.get('tl_attr', float('nan'))):.4f} "
                f"lane={float(losses.get('lane', float('nan'))):.4f} "
                f"stop={float(losses.get('stop_line', float('nan'))):.4f} "
                f"cross={float(losses.get('crosswalk', float('nan'))):.4f}"
            ),
            (
                "  iter_ms: "
                f"mean={iteration_profile['mean'] * 1000.0:.3f} "
                f"p50={iteration_profile['p50'] * 1000.0:.3f} "
                f"p99={iteration_profile['p99'] * 1000.0:.3f}"
            ),
            (
                "  timing_ms: "
                f"wait={profile_summary['wait_sec']['mean'] * 1000.0:.3f} "
                f"load={profile_summary['load_sec']['mean'] * 1000.0:.3f} "
                f"fwd={profile_summary['forward_sec']['mean'] * 1000.0:.3f} "
                f"loss={profile_summary['loss_sec']['mean'] * 1000.0:.3f} "
                f"bwd={profile_summary['backward_sec']['mean'] * 1000.0:.3f}"
            ),
        ]
    )


def _loss_stats_from_summaries(summaries: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    if not summaries:
        return {}
    names = list(summaries[0]["losses"].keys())
    return {
        name: {
            "mean": sum(float(item["losses"][name]) for item in summaries) / len(summaries),
            "min": min(float(item["losses"][name]) for item in summaries),
            "max": max(float(item["losses"][name]) for item in summaries),
            "last": float(summaries[-1]["losses"][name]),
        }
        for name in names
    }


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


def _resolve_summary_path(summary: dict[str, Any], path: str) -> float:
    current: Any = summary
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            raise KeyError(f"summary path not found: {path}")
        current = current[part]
    return float(current)


def _is_better(candidate: float, current_best: float | None, mode: str) -> bool:
    if current_best is None:
        return True
    if mode == "min":
        return candidate < current_best
    if mode == "max":
        return candidate > current_best
    raise KeyError(f"unsupported comparison mode: {mode}")


def _is_oom_error(exc: RuntimeError) -> bool:
    message = str(exc).lower()
    return "out of memory" in message or "cuda error: out of memory" in message


def build_pv26_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    epochs: int,
    schedule: str = "cosine",
    min_lr_ratio: float = 0.1,
):
    if epochs <= 0:
        raise ValueError("scheduler epochs must be > 0")
    if schedule == "none":
        return None
    if schedule == "cosine":
        base_lrs = [float(group.get("lr", 0.0)) for group in optimizer.param_groups]
        eta_min = min(base_lrs) * float(min_lr_ratio) if base_lrs else 0.0
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, epochs),
            eta_min=eta_min,
        )
    raise KeyError(f"unsupported PV26 scheduler: {schedule}")


def configure_pv26_train_stage(adapter: Any, heads: torch.nn.Module, stage: str) -> dict[str, int | str]:
    stage = _canonical_stage(stage)
    if stage not in STAGE_NAMES:
        raise KeyError(f"unsupported PV26 train stage: {stage}")

    for parameter in heads.parameters():
        parameter.requires_grad = True

    trunk_layers = list(adapter.trunk.children())
    if stage == "stage_1_frozen_trunk_warmup":
        adapter.freeze_trunk()
    elif stage == "stage_2_partial_unfreeze":
        adapter.freeze_trunk()
        partial_count = max(1, len(trunk_layers) // 3)
        for layer in trunk_layers[-partial_count:]:
            for parameter in layer.parameters():
                parameter.requires_grad = True
    else:
        adapter.unfreeze_trunk()

    trainable_trunk = _trainable_parameters(adapter.trunk)
    trainable_heads = _trainable_parameters(heads)
    return {
        "stage": stage,
        "trainable_trunk_params": _count_parameters(trainable_trunk),
        "trainable_head_params": _count_parameters(trainable_heads),
    }


def build_pv26_optimizer(
    adapter: Any,
    heads: torch.nn.Module,
    *,
    trunk_lr: float = 1e-4,
    head_lr: float = 1e-3,
    weight_decay: float = 1e-4,
) -> torch.optim.Optimizer:
    param_groups: list[dict[str, Any]] = []
    trunk_params = _trainable_parameters(adapter.trunk)
    head_params = _trainable_parameters(heads)

    if trunk_params:
        param_groups.append(
            {
                "params": trunk_params,
                "lr": trunk_lr,
                "weight_decay": weight_decay,
                "group_name": "trunk",
            }
        )
    if head_params:
        param_groups.append(
            {
                "params": head_params,
                "lr": head_lr,
                "weight_decay": weight_decay,
                "group_name": "heads",
            }
        )
    if not param_groups:
        raise ValueError("no trainable parameters are available for optimizer construction")
    return torch.optim.AdamW(param_groups, betas=(0.9, 0.999))


class PV26Trainer:
    def __init__(
        self,
        adapter: Any,
        heads: torch.nn.Module,
        *,
        stage: str = "stage_0_smoke",
        device: str | torch.device = "cpu",
        criterion: torch.nn.Module | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        trunk_lr: float = 1e-4,
        head_lr: float = 1e-3,
        weight_decay: float = 1e-4,
        amp: bool = False,
        accumulate_steps: int = 1,
        grad_clip_norm: float | None = None,
        skip_non_finite_loss: bool = True,
        oom_guard: bool = True,
    ) -> None:
        if accumulate_steps <= 0:
            raise ValueError("accumulate_steps must be > 0")
        self.adapter = adapter
        self.heads = heads
        self.device = torch.device(device)
        self.stage = _canonical_stage(stage)
        self.stage_summary = configure_pv26_train_stage(adapter, heads, self.stage)
        self.adapter.raw_model.to(self.device)
        self.heads.to(self.device)
        self.criterion = (criterion or PV26MultiTaskLoss(stage=self.stage)).to(self.device)
        self.optimizer = optimizer or build_pv26_optimizer(
            adapter,
            heads,
            trunk_lr=trunk_lr,
            head_lr=head_lr,
            weight_decay=weight_decay,
        )
        self.scheduler = scheduler
        self.accumulate_steps = int(accumulate_steps)
        self.grad_clip_norm = float(grad_clip_norm) if grad_clip_norm is not None else None
        self.skip_non_finite_loss = bool(skip_non_finite_loss)
        self.oom_guard = bool(oom_guard)
        self.amp_enabled = bool(amp) and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp_enabled)
        self.micro_step = 0
        self.skipped_steps = 0
        self.global_step = 0
        self.history: list[dict[str, Any]] = []
        self.epoch_history: list[dict[str, Any]] = []
        self.tensorboard_writer = None
        self.tensorboard_mode = "curated"
        self.tensorboard_status: dict[str, Any] = {
            "enabled": False,
            "status": "inactive",
            "error": None,
            "log_dir": None,
            "purge_step": None,
            "mode": self.tensorboard_mode,
        }
        self._tensorboard_train_step = 0

    def prepare_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        encoded = batch if "det_gt" in batch else encode_pv26_batch(batch)
        return _move_to_device(encoded, self.device, non_blocking=self.device.type == "cuda")

    def forward_encoded_batch(self, encoded: dict[str, Any]) -> dict[str, torch.Tensor]:
        features = forward_pyramid_features(self.adapter, encoded["image"])
        return self.heads(features)

    def _autocast_context(self):
        if not self.amp_enabled:
            return nullcontext()
        return torch.autocast(device_type=self.device.type, dtype=torch.float16)

    def train_step(
        self,
        batch: dict[str, Any],
        *,
        wait_sec: float = 0.0,
        profile_device_sync: bool = False,
    ) -> dict[str, Any]:
        self.adapter.raw_model.train()
        self.heads.train()
        load_started_at = time.perf_counter()
        encoded = self.prepare_batch(batch)
        _sync_timing_device(self.device, profile_device_sync)
        load_ended_at = time.perf_counter()
        if self.micro_step == 0:
            self.optimizer.zero_grad(set_to_none=True)

        def _nan_losses() -> dict[str, torch.Tensor]:
            return {
                "total": torch.full((), float("nan"), device=self.device),
                "det": torch.full((), float("nan"), device=self.device),
                "tl_attr": torch.full((), float("nan"), device=self.device),
                "lane": torch.full((), float("nan"), device=self.device),
                "stop_line": torch.full((), float("nan"), device=self.device),
                "crosswalk": torch.full((), float("nan"), device=self.device),
            }

        def _summary_det_components(payload: Any) -> dict[str, float | int]:
            raw = payload if isinstance(payload, dict) else {}
            summary = dict(_default_det_components())
            for key in summary:
                value = raw.get(key, summary[key])
                if isinstance(value, torch.Tensor):
                    value = float(value.detach().cpu())
                if key.endswith("_count"):
                    summary[key] = int(value)
                else:
                    summary[key] = float(value)
            return summary

        skipped_reason: str | None = None
        skipped_reason_detail: str | None = None
        optimizer_step = False
        successful = False
        losses: dict[str, torch.Tensor] = _nan_losses()
        det_components = _summary_det_components({})
        assignment_det_mode = "unknown"
        assignment_lane_modes = dict(getattr(self.criterion, "last_lane_assignment_modes", {}))
        forward_started_at = load_ended_at
        forward_ended_at = load_ended_at
        loss_started_at = load_ended_at
        loss_ended_at = load_ended_at
        backward_started_at = load_ended_at
        backward_ended_at = load_ended_at
        try:
            _sync_timing_device(self.device, profile_device_sync)
            forward_started_at = time.perf_counter()
            with self._autocast_context():
                predictions = self.forward_encoded_batch(encoded)
            _sync_timing_device(self.device, profile_device_sync)
            forward_ended_at = time.perf_counter()
            loss_started_at = forward_ended_at
            losses = self.criterion(predictions, encoded)
            assignment_det_mode = str(getattr(self.criterion, "last_det_assignment_mode", "unknown"))
            assignment_lane_modes = dict(getattr(self.criterion, "last_lane_assignment_modes", {}))
            det_components = _summary_det_components(getattr(self.criterion, "last_det_loss_breakdown", {}))
            _sync_timing_device(self.device, profile_device_sync)
            loss_ended_at = time.perf_counter()
            total_loss = losses["total"]
            backward_started_at = loss_ended_at
            if not torch.isfinite(total_loss):
                skipped_reason = "non_finite_loss"
                if not self.skip_non_finite_loss:
                    raise FloatingPointError("non-finite PV26 total loss encountered")
                self.optimizer.zero_grad(set_to_none=True)
                self.micro_step = 0
                self.skipped_steps += 1
            else:
                scaled_total = total_loss / float(self.accumulate_steps)
                if self.amp_enabled:
                    self.scaler.scale(scaled_total).backward()
                else:
                    scaled_total.backward()
                self.micro_step += 1
                if self.micro_step >= self.accumulate_steps:
                    if self.grad_clip_norm is not None:
                        if self.amp_enabled:
                            self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            list(self.adapter.raw_model.parameters()) + list(self.heads.parameters()),
                            max_norm=self.grad_clip_norm,
                        )
                    if self.amp_enabled:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.global_step += 1
                    self.micro_step = 0
                    optimizer_step = True
                successful = True
            _sync_timing_device(self.device, profile_device_sync)
            backward_ended_at = time.perf_counter()
        except PV26DetAssignmentUnavailable as exc:
            skipped_reason = "det_assignment_unavailable"
            skipped_reason_detail = str(exc)
            self.optimizer.zero_grad(set_to_none=True)
            self.micro_step = 0
            self.skipped_steps += 1
            losses = _nan_losses()
            det_components = _summary_det_components({})
            assignment_det_mode = "det_assignment_unavailable"
            assignment_lane_modes = {}
            _sync_timing_device(self.device, profile_device_sync)
            backward_ended_at = time.perf_counter()
        except RuntimeError as exc:
            if not self.oom_guard or not _is_oom_error(exc):
                raise
            skipped_reason = "oom_recovered"
            skipped_reason_detail = str(exc)
            self.optimizer.zero_grad(set_to_none=True)
            self.micro_step = 0
            self.skipped_steps += 1
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            losses = _nan_losses()
            det_components = _summary_det_components({})
            assignment_det_mode = "oom_recovered"
            assignment_lane_modes = {}
            _sync_timing_device(self.device, profile_device_sync)
            backward_ended_at = time.perf_counter()
        timing = {
            "wait_sec": float(wait_sec),
            "load_sec": max(0.0, load_ended_at - load_started_at),
            "forward_sec": max(0.0, forward_ended_at - forward_started_at),
            "loss_sec": max(0.0, loss_ended_at - loss_started_at),
            "backward_sec": max(0.0, backward_ended_at - backward_started_at),
        }
        timing["iteration_sec"] = sum(float(timing[key]) for key in TIMING_KEYS if key != "iteration_sec")
        summary = {
            "history_index": len(self.history) + 1,
            "global_step": self.global_step,
            "stage": self.stage,
            "batch_size": int(encoded["image"].shape[0]),
            "successful": successful,
            "losses": {
                name: float(value.detach().cpu())
                for name, value in losses.items()
            },
            "det_components": det_components,
            "optimizer_step": optimizer_step,
            "micro_step": int(self.micro_step),
            "accumulate_steps": int(self.accumulate_steps),
            "skipped_reason": skipped_reason,
            "skipped_reason_detail": skipped_reason_detail,
            "skipped_steps": int(self.skipped_steps),
            "amp_enabled": bool(self.amp_enabled),
            "gradient_scale": float(self.scaler.get_scale()) if self.amp_enabled else 1.0,
            "optimizer_lrs": {
                str(group.get("group_name", f"group_{index}")): float(group["lr"])
                for index, group in enumerate(self.optimizer.param_groups)
            },
            "trainable": dict(self.stage_summary),
            "assignment": {
                "det": assignment_det_mode,
                "lane": assignment_lane_modes,
            },
            "timing": timing,
            "source_counts": _source_counts(encoded),
            "det_supervision": _det_supervision_summary(encoded),
        }
        self.history.append(summary)
        if self.tensorboard_writer is not None:
            self._tensorboard_train_step += 1
            _write_tensorboard_scalars(
                self.tensorboard_writer,
                "train_step",
                _tensorboard_train_step_payload(summary, self.tensorboard_mode),
                self._tensorboard_train_step,
            )
        return summary

    def summarize_history(self, *, last_n: int | None = None) -> dict[str, Any]:
        if not self.history:
            raise ValueError("trainer history is empty")
        window = self.history[-last_n:] if last_n is not None else self.history
        successful_window = _successful_summaries(window)
        anchor = successful_window[-1] if successful_window else window[-1]
        summary: dict[str, Any] = {
            "steps": len(window),
            "successful_steps": len(successful_window),
            "global_step": int(anchor["global_step"]),
            "stage": str(anchor["stage"]),
            "assignment": {
                "det": str(anchor["assignment"]["det"]),
                "lane": dict(anchor["assignment"]["lane"]),
            },
            "losses": {},
        }
        if not successful_window:
            return summary
        for name in anchor["losses"]:
            summary["losses"][name] = _loss_summary(window, name)
        return summary

    def save_epoch_history_jsonl(self, path: str | Path) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = "".join(json.dumps(item, ensure_ascii=True, sort_keys=True) + "\n" for item in self.epoch_history)
        output_path.write_text(payload, encoding="utf-8")
        return output_path

    def save_history_jsonl(self, path: str | Path) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = "".join(json.dumps(item, ensure_ascii=True, sort_keys=True) + "\n" for item in self.history)
        output_path.write_text(payload, encoding="utf-8")
        return output_path

    def checkpoint_state(self) -> dict[str, Any]:
        checkpoint = {
            "stage": self.stage,
            "global_step": self.global_step,
            "stage_summary": dict(self.stage_summary),
            "adapter_state_dict": self.adapter.raw_model.state_dict(),
            "heads_state_dict": self.heads.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "criterion_stage": str(getattr(self.criterion, "stage", self.stage)),
            "history": list(self.history),
            "epoch_history": list(self.epoch_history),
            "micro_step": int(self.micro_step),
            "skipped_steps": int(self.skipped_steps),
            "accumulate_steps": int(self.accumulate_steps),
            "grad_clip_norm": self.grad_clip_norm,
            "amp_enabled": bool(self.amp_enabled),
        }
        criterion_config = _criterion_config_from_instance(self.criterion, self.stage)
        if criterion_config is not None:
            checkpoint["criterion_config"] = criterion_config
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        if self.amp_enabled:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()
        return checkpoint

    def save_checkpoint(self, path: str | Path, *, extra_state: dict[str, Any] | None = None) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = self.checkpoint_state()
        if extra_state:
            checkpoint["extra_state"] = dict(extra_state)
        torch.save(checkpoint, output_path)
        return output_path

    def load_checkpoint(
        self,
        path: str | Path,
        *,
        map_location: str | torch.device | None = None,
    ) -> dict[str, Any]:
        checkpoint = torch.load(path, map_location=map_location or self.device)
        checkpoint_stage = _canonical_stage(str(checkpoint.get("stage", self.stage)))
        optimizer_hparams = _optimizer_group_hparams(self.optimizer)
        current_criterion_config = _criterion_config_from_instance(self.criterion, self.stage)
        checkpoint_criterion_config = checkpoint.get("criterion_config")
        if isinstance(checkpoint_criterion_config, dict):
            criterion_config = dict(checkpoint_criterion_config)
            criterion_config["stage"] = checkpoint_stage
        elif current_criterion_config is not None:
            criterion_config = dict(current_criterion_config)
            criterion_config["stage"] = checkpoint_stage
        else:
            criterion_config = None

        if checkpoint_stage != self.stage:
            self.stage = checkpoint_stage
            self.stage_summary = configure_pv26_train_stage(self.adapter, self.heads, self.stage)
            self.optimizer = build_pv26_optimizer(
                self.adapter,
                self.heads,
                trunk_lr=optimizer_hparams["trunk_lr"],
                head_lr=optimizer_hparams["head_lr"],
                weight_decay=optimizer_hparams["weight_decay"],
            )
        if criterion_config is not None:
            self.criterion = PV26MultiTaskLoss(**criterion_config).to(self.device)
        self.adapter.raw_model.load_state_dict(checkpoint["adapter_state_dict"])
        self.heads.load_state_dict(checkpoint["heads_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if self.amp_enabled and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        self.global_step = int(checkpoint.get("global_step", 0))
        self.stage_summary = dict(checkpoint.get("stage_summary", self.stage_summary))
        self.history = list(checkpoint.get("history", []))
        self.epoch_history = list(checkpoint.get("epoch_history", []))
        self.micro_step = int(checkpoint.get("micro_step", 0))
        self.skipped_steps = int(checkpoint.get("skipped_steps", 0))
        return checkpoint

    def load_model_weights(
        self,
        path: str | Path,
        *,
        map_location: str | torch.device | None = None,
    ) -> dict[str, Any]:
        checkpoint = torch.load(path, map_location=map_location or self.device)
        self.adapter.raw_model.load_state_dict(checkpoint["adapter_state_dict"])
        self.heads.load_state_dict(checkpoint["heads_state_dict"])
        return checkpoint

    def build_evaluator(self):
        from ..eval import PV26Evaluator

        criterion = self.criterion
        criterion_config = _criterion_config_from_instance(self.criterion, self.stage)
        if criterion_config is not None:
            criterion = PV26MultiTaskLoss(**criterion_config).to(self.device)
        return PV26Evaluator(
            self.adapter,
            self.heads,
            stage=self.stage,
            device=self.device,
            criterion=criterion,
        )

    def train_epoch(
        self,
        loader,
        *,
        epoch: int,
        epoch_total: int | None = None,
        phase_index: int | None = None,
        phase_count: int | None = None,
        phase_name: str | None = None,
        max_batches: int | None = None,
        step_log_path: str | Path | None = None,
        log_every_n_steps: int = 1,
        profile_window: int = 20,
        profile_device_sync: bool = False,
    ) -> dict[str, Any]:
        started_at = time.perf_counter()
        epoch_started_at_iso = _now_iso()
        step_summaries: list[dict[str, Any]] = []
        timing_window: deque[dict[str, Any]] = deque(maxlen=max(1, int(profile_window)))
        start_step = self.global_step
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
            step_summary = self.train_step(
                batch,
                wait_sec=max(0.0, fetch_ended_at - fetch_started_at),
                profile_device_sync=profile_device_sync,
            )
            timing_window.append(step_summary)
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
            if self.tensorboard_writer is not None:
                _write_tensorboard_scalars(
                    self.tensorboard_writer,
                    "train_progress",
                    _tensorboard_progress_payload(
                        step_summary,
                        epoch=epoch,
                        batch_index=batch_index,
                        total_batches=total_batches,
                        elapsed_sec=elapsed_sec,
                        eta_sec=eta_sec,
                        profile_summary=profile_summary,
                        mode=self.tensorboard_mode,
                    ),
                    max(1, self._tensorboard_train_step),
                )
            should_log = batch_index % max(1, int(log_every_n_steps)) == 0
            if total_batches is not None and batch_index == total_batches:
                should_log = True
            if should_log:
                print(
                    _format_train_progress_log(
                        stage=self.stage,
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
            "global_step_end": int(self.global_step),
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

    def validate_epoch(
        self,
        loader,
        *,
        epoch: int,
        evaluator=None,
        max_batches: int | None = None,
    ) -> dict[str, Any]:
        from ..eval import summarize_pv26_metrics

        evaluator = evaluator or self.build_evaluator()
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

    def fit(
        self,
        train_loader,
        *,
        epochs: int,
        phase_index: int | None = None,
        phase_count: int | None = None,
        phase_name: str | None = None,
        val_loader=None,
        run_dir: str | Path | None = None,
        val_every: int = 1,
        checkpoint_every: int = 1,
        max_train_batches: int | None = None,
        max_val_batches: int | None = None,
        best_metric: str | None = None,
        best_mode: str = "min",
        auto_resume: bool = False,
        resume_path: str | Path | None = None,
        enable_tensorboard: bool = True,
        tensorboard_mode: str = "curated",
        early_exit_callback: Callable[[dict[str, Any]], dict[str, Any] | None] | None = None,
        run_manifest_extra: dict[str, Any] | None = None,
        log_every_n_steps: int = 1,
        profile_window: int = 20,
        profile_device_sync: bool = False,
    ) -> dict[str, Any]:
        if epochs <= 0:
            raise ValueError("fit requires epochs > 0")
        if log_every_n_steps <= 0:
            raise ValueError("log_every_n_steps must be > 0")
        if profile_window <= 0:
            raise ValueError("profile_window must be > 0")

        output_dir = Path(run_dir) if run_dir is not None else _default_run_dir()
        history_dir = output_dir / "history"
        checkpoint_dir = output_dir / "checkpoints"
        tensorboard_dir = output_dir / "tensorboard"
        history_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        tensorboard_dir.mkdir(parents=True, exist_ok=True)

        best_metric_path = best_metric or (
            "val.losses.total.mean" if val_loader is not None else "train.losses.total.mean"
        )
        best_metric_value: float | None = None
        best_epoch: int | None = None
        best_checkpoint_path: Path | None = None
        run_started_at = time.perf_counter()
        run_started_at_iso = _now_iso()
        run_summary: dict[str, Any] = {}
        early_exit_state: dict[str, Any] | None = None
        start_epoch = 1
        manifest_path = output_dir / "run_manifest.json"
        previous_writer = self.tensorboard_writer
        previous_mode = self.tensorboard_mode
        previous_status = dict(self.tensorboard_status)
        previous_tb_step = int(self._tensorboard_train_step)
        tensorboard_purge_step: int | None = None
        resumed_from_checkpoint = False
        self.tensorboard_mode = _canonical_tensorboard_mode(tensorboard_mode)
        if auto_resume:
            resume_candidate = Path(resume_path) if resume_path is not None else checkpoint_dir / "last.pt"
            if resume_candidate.is_file():
                checkpoint = self.load_checkpoint(resume_candidate, map_location=self.device)
                resumed_from_checkpoint = True
                extra_state = checkpoint.get("extra_state", {}) if isinstance(checkpoint.get("extra_state"), dict) else {}
                restored_epoch = int(extra_state.get("epoch", len(self.epoch_history)))
                start_epoch = restored_epoch + 1
                summary_path = output_dir / "summary.json"
                if summary_path.is_file():
                    prior_summary = json.loads(summary_path.read_text(encoding="utf-8"))
                    if prior_summary.get("best_metric_value") is not None:
                        best_metric_value = float(prior_summary["best_metric_value"])
                    if prior_summary.get("best_epoch") is not None:
                        best_epoch = int(prior_summary["best_epoch"])
                if best_metric_value is None and self.epoch_history:
                    best_metric_value = _resolve_summary_path(
                        self.epoch_history[-1],
                        best_metric or ("val.losses.total.mean" if val_loader is not None else "train.losses.total.mean"),
                    )
                    best_epoch = int(self.epoch_history[-1]["epoch"])
                if (checkpoint_dir / "best.pt").is_file():
                    best_checkpoint_path = checkpoint_dir / "best.pt"
        self._tensorboard_train_step = len(self.history)
        if resumed_from_checkpoint:
            tensorboard_purge_step = max(1, self._tensorboard_train_step + 1)
        if enable_tensorboard:
            self.tensorboard_writer, self.tensorboard_status = _maybe_build_summary_writer(
                tensorboard_dir,
                purge_step=tensorboard_purge_step,
            )
            self.tensorboard_status["mode"] = self.tensorboard_mode
        else:
            self.tensorboard_writer = None
            self.tensorboard_status = {
                "enabled": False,
                "status": "disabled_by_config",
                "error": None,
                "log_dir": str(tensorboard_dir),
                "purge_step": tensorboard_purge_step,
                "mode": self.tensorboard_mode,
            }
        evaluator = self.build_evaluator() if val_loader is not None else None
        try:
            if start_epoch > epochs:
                run_summary = {
                    "stage": self.stage,
                    "epochs": int(epochs),
                    "completed_epochs": len(self.epoch_history),
                    "global_step": int(self.global_step),
                    "run_dir": str(output_dir),
                    "best_metric_path": best_metric or (
                        "val.losses.total.mean" if val_loader is not None else "train.losses.total.mean"
                    ),
                    "best_metric_value": best_metric_value,
                    "best_epoch": best_epoch,
                    "last_epoch": self.epoch_history[-1] if self.epoch_history else None,
                    "history_paths": {
                        "train_steps": str(history_dir / "train_steps.jsonl"),
                        "epochs": str(history_dir / "epochs.jsonl"),
                    },
                    "checkpoint_paths": {
                        "last": str(checkpoint_dir / "last.pt"),
                        "best": str(best_checkpoint_path) if best_checkpoint_path is not None else None,
                    },
                    "tensorboard": dict(self.tensorboard_status),
                    "manifest_path": str(manifest_path),
                    "duration_sec": time.perf_counter() - run_started_at,
                    "auto_resumed": True,
                    "resume_start_epoch": int(start_epoch),
                }
                _write_json(output_dir / "summary.json", run_summary)
                _write_json(
                    manifest_path,
                    {
                        "version": RUN_MANIFEST_VERSION,
                        "created_at": run_started_at_iso,
                        "updated_at": _now_iso(),
                        "stage": self.stage,
                        "device": str(self.device),
                        "optimizer": _optimizer_group_hparams(self.optimizer),
                        "trainer": {
                            "amp_enabled": bool(self.amp_enabled),
                            "accumulate_steps": int(self.accumulate_steps),
                            "grad_clip_norm": self.grad_clip_norm,
                            "skip_non_finite_loss": bool(self.skip_non_finite_loss),
                            "oom_guard": bool(self.oom_guard),
                            "tensorboard_mode": self.tensorboard_mode,
                            "log_every_n_steps": int(log_every_n_steps),
                            "profile_window": int(profile_window),
                            "profile_device_sync": bool(profile_device_sync),
                        },
                        "artifacts": {
                            "summary": str(output_dir / "summary.json"),
                            "history": run_summary["history_paths"],
                            "checkpoints": run_summary["checkpoint_paths"],
                            "tensorboard": dict(self.tensorboard_status),
                        },
                        "run_state": _json_ready(run_summary),
                        "extra": _json_ready(run_manifest_extra or {}),
                    },
                )
                return run_summary

            for epoch in range(start_epoch, epochs + 1):
                epoch_summary: dict[str, Any] = {
                    "epoch": int(epoch),
                    "stage": self.stage,
                    "epoch_started_at": _now_iso(),
                    "train": self.train_epoch(
                        train_loader,
                        epoch=epoch,
                        epoch_total=epochs,
                        phase_index=phase_index,
                        phase_count=phase_count,
                        phase_name=phase_name,
                        max_batches=max_train_batches,
                        step_log_path=history_dir / "train_steps.jsonl",
                        log_every_n_steps=log_every_n_steps,
                        profile_window=profile_window,
                        profile_device_sync=profile_device_sync,
                    ),
                }
                if val_loader is not None and epoch % val_every == 0:
                    epoch_summary["val"] = self.validate_epoch(
                        val_loader,
                        epoch=epoch,
                        evaluator=evaluator,
                        max_batches=max_val_batches,
                    )
                if self.scheduler is not None:
                    self.scheduler.step()
                    epoch_summary["scheduler_lrs"] = [
                        float(group["lr"]) for group in self.optimizer.param_groups
                    ]

                metric_value = _resolve_summary_path(epoch_summary, best_metric_path)
                epoch_summary["selection"] = {
                    "best_metric_path": best_metric_path,
                    "best_metric_value": metric_value,
                    "best_mode": best_mode,
                }
                is_best = _is_better(metric_value, best_metric_value, best_mode)
                if is_best:
                    best_metric_value = metric_value
                    best_epoch = epoch

                early_exit_state = None
                if early_exit_callback is not None:
                    callback_result = early_exit_callback(epoch_summary)
                    if callback_result is not None:
                        if not isinstance(callback_result, dict):
                            raise TypeError("early_exit_callback must return dict[str, Any] | None")
                        early_exit_state = dict(callback_result)
                        early_exit_state.setdefault("should_stop", True)

                self.epoch_history.append(epoch_summary)
                last_checkpoint_path = self.save_checkpoint(
                    checkpoint_dir / "last.pt",
                    extra_state={"epoch": epoch, "epoch_summary": epoch_summary},
                )
                epoch_summary["checkpoint_last"] = str(last_checkpoint_path)
                if epoch % checkpoint_every == 0:
                    epoch_checkpoint_path = self.save_checkpoint(
                        checkpoint_dir / f"epoch_{epoch:03d}.pt",
                        extra_state={"epoch": epoch, "epoch_summary": epoch_summary},
                    )
                    epoch_summary["checkpoint_epoch"] = str(epoch_checkpoint_path)
                if is_best:
                    best_checkpoint_path = self.save_checkpoint(
                        checkpoint_dir / "best.pt",
                        extra_state={"epoch": epoch, "epoch_summary": epoch_summary},
                    )
                    epoch_summary["checkpoint_best"] = str(best_checkpoint_path)

                _append_jsonl(history_dir / "epochs.jsonl", epoch_summary)
                self.save_history_jsonl(history_dir / "train_steps.jsonl")
                self.save_epoch_history_jsonl(history_dir / "epochs.jsonl")

                if self.tensorboard_writer is not None:
                    _write_tensorboard_scalars(
                        self.tensorboard_writer,
                        "epoch",
                        _tensorboard_epoch_payload(epoch_summary, self.tensorboard_mode),
                        epoch,
                    )
                    self.tensorboard_writer.flush()

                run_summary = {
                    "stage": self.stage,
                    "epochs": int(epochs),
                    "completed_epochs": len(self.epoch_history),
                    "global_step": int(self.global_step),
                    "run_dir": str(output_dir),
                    "best_metric_path": best_metric_path,
                    "best_metric_value": best_metric_value,
                    "best_epoch": best_epoch,
                    "last_epoch": self.epoch_history[-1],
                    "skipped_steps": int(self.skipped_steps),
                    "history_paths": {
                        "train_steps": str(history_dir / "train_steps.jsonl"),
                        "epochs": str(history_dir / "epochs.jsonl"),
                    },
                    "checkpoint_paths": {
                        "last": str(checkpoint_dir / "last.pt"),
                        "best": str(best_checkpoint_path) if best_checkpoint_path is not None else None,
                    },
                    "tensorboard": dict(self.tensorboard_status),
                    "manifest_path": str(manifest_path),
                    "duration_sec": time.perf_counter() - run_started_at,
                    "auto_resumed": bool(auto_resume and start_epoch > 1),
                    "resume_start_epoch": int(start_epoch),
                }
                if early_exit_state is not None:
                    run_summary["early_exit"] = _json_ready(early_exit_state)
                _write_json(output_dir / "summary.json", run_summary)
                _write_json(
                    manifest_path,
                    {
                        "version": RUN_MANIFEST_VERSION,
                        "created_at": run_started_at_iso,
                        "updated_at": _now_iso(),
                        "stage": self.stage,
                        "device": str(self.device),
                        "optimizer": _optimizer_group_hparams(self.optimizer),
                        "trainer": {
                            "amp_enabled": bool(self.amp_enabled),
                            "accumulate_steps": int(self.accumulate_steps),
                            "grad_clip_norm": self.grad_clip_norm,
                            "skip_non_finite_loss": bool(self.skip_non_finite_loss),
                            "oom_guard": bool(self.oom_guard),
                            "tensorboard_mode": self.tensorboard_mode,
                            "log_every_n_steps": int(log_every_n_steps),
                            "profile_window": int(profile_window),
                            "profile_device_sync": bool(profile_device_sync),
                        },
                        "artifacts": {
                            "summary": str(output_dir / "summary.json"),
                            "history": run_summary["history_paths"],
                            "checkpoints": run_summary["checkpoint_paths"],
                            "tensorboard": dict(self.tensorboard_status),
                        },
                        "run_state": _json_ready(run_summary),
                        "extra": _json_ready(run_manifest_extra or {}),
                    },
                )
                if early_exit_state is not None and bool(early_exit_state.get("should_stop", True)):
                    break

            return run_summary
        finally:
            if self.tensorboard_writer is not None:
                self.tensorboard_writer.flush()
                self.tensorboard_writer.close()
            self.tensorboard_writer = previous_writer
            self.tensorboard_mode = previous_mode
            self.tensorboard_status = previous_status
            self._tensorboard_train_step = previous_tb_step


def run_pv26_tiny_overfit(
    trainer: PV26Trainer,
    batch: dict[str, Any],
    *,
    steps: int = 8,
) -> dict[str, Any]:
    if steps <= 0:
        raise ValueError("tiny overfit smoke requires steps > 0")

    sample_ids = [str(item.get("sample_id", "unknown")) for item in batch.get("meta", [])]
    history: list[dict[str, Any]] = []
    for _ in range(steps):
        history.append(trainer.train_step(batch))

    total_history = [float(item["losses"]["total"]) for item in history]
    first_total = total_history[0]
    final_total = total_history[-1]
    best_total = min(total_history)
    best_step = total_history.index(best_total) + 1
    return {
        "stage": trainer.stage,
        "steps": steps,
        "sample_ids": sample_ids,
        "history": history,
        "first_total": first_total,
        "final_total": final_total,
        "best_total": best_total,
        "best_step": best_step,
        "improvement": first_total - best_total,
        "improvement_ratio": (first_total - best_total) / max(first_total, 1e-12),
    }


__all__ = [
    "PV26Trainer",
    "STAGE_NAMES",
    "build_pv26_optimizer",
    "configure_pv26_train_stage",
    "run_pv26_tiny_overfit",
]
