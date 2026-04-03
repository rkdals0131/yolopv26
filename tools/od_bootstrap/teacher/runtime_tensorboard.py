from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from common.train_runtime import maybe_build_summary_writer as _common_maybe_build_summary_writer
from common.train_runtime import write_tensorboard_scalars as _common_write_tensorboard_scalars

__all__ = [
    "build_epoch_tensorboard_payload",
    "build_train_step_tensorboard_payload",
    "maybe_build_summary_writer",
    "write_tensorboard_scalars",
]


def maybe_build_summary_writer(log_dir: Path):
    writer, status = _common_maybe_build_summary_writer(log_dir)
    return writer, {key: value for key, value in status.items() if key != "purge_step"}


def write_tensorboard_scalars(writer: Any, prefix: str, payload: dict[str, Any], step: int) -> int:
    return _common_write_tensorboard_scalars(writer, prefix, payload, step)


def _coerce_scalar(value: Any) -> float | None:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        numeric = float(value)
        if math.isfinite(numeric):
            return numeric
    return None


def _first_scalar(source: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        if key in source:
            numeric = _coerce_scalar(source[key])
            if numeric is not None:
                return numeric
    return None


def _train_loss_payload(losses: dict[str, Any]) -> dict[str, float]:
    payload: dict[str, float] = {}
    for source_key, target_key in (
        ("train/box_loss", "box_loss"),
        ("train/cls_loss", "cls_loss"),
        ("train/dfl_loss", "dfl_loss"),
    ):
        value = _first_scalar(losses, source_key, target_key)
        if value is not None:
            payload[target_key] = value
    return payload


def _epoch_lr_payload(lr_values: dict[str, Any]) -> dict[str, float]:
    value = _first_scalar(lr_values, "lr/pg0", "pg0")
    if value is None:
        return {}
    return {"pg0": value}


def _epoch_metric_payload(metrics: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    precision = None
    recall = None
    for target_key, candidates in (
        ("precision", ("metrics/precision(B)", "metrics/precision")),
        ("recall", ("metrics/recall(B)", "metrics/recall")),
        ("mAP50", ("metrics/mAP50(B)", "metrics/mAP50")),
        ("mAP50_95", ("metrics/mAP50-95(B)", "metrics/mAP50-95")),
    ):
        value = _first_scalar(metrics, *candidates)
        if value is not None:
            payload[target_key] = value
            if target_key == "precision":
                precision = value
            elif target_key == "recall":
                recall = value

    if precision is not None and recall is not None and (precision + recall) > 0.0:
        payload["f1"] = (2.0 * precision * recall) / (precision + recall)

    val_payload: dict[str, float] = {}
    for source_key, target_key in (
        ("val/box_loss", "box_loss"),
        ("val/cls_loss", "cls_loss"),
        ("val/dfl_loss", "dfl_loss"),
    ):
        value = _first_scalar(metrics, source_key, target_key)
        if value is not None:
            val_payload[target_key] = value
    if val_payload:
        payload["val"] = val_payload
    return payload


def _epoch_profile_payload(profile_summary: dict[str, Any]) -> dict[str, float]:
    payload: dict[str, float] = {}
    for target_key, source_group in (
        ("iteration_mean", "iteration_sec"),
        ("wait_mean", "wait_sec"),
        ("compute_mean", "compute_sec"),
    ):
        if isinstance(profile_summary.get(source_group), dict):
            value = _first_scalar(profile_summary[source_group], "mean")
            if value is not None:
                payload[target_key] = value
    return payload


def _train_step_profile_payload(profile_summary: dict[str, Any]) -> dict[str, float]:
    payload: dict[str, float] = {}
    for target_key, source_group, stat_key in (
        ("iteration_mean", "iteration_sec", "mean"),
        ("iteration_p50", "iteration_sec", "p50"),
        ("iteration_p99", "iteration_sec", "p99"),
        ("wait_mean", "wait_sec", "mean"),
        ("compute_mean", "compute_sec", "mean"),
    ):
        if isinstance(profile_summary.get(source_group), dict):
            value = _first_scalar(profile_summary[source_group], stat_key)
            if value is not None:
                payload[target_key] = value
    return payload


def build_epoch_tensorboard_payload(
    *,
    losses: dict[str, Any],
    profile_summary: dict[str, Any],
    lr_values: dict[str, Any],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    payload: dict[str, Any] = {}

    train_loss = _train_loss_payload(losses)
    if train_loss:
        payload["train"] = train_loss

    lr_payload = _epoch_lr_payload(lr_values)
    if lr_payload:
        payload["lr"] = lr_payload

    profile_payload = _epoch_profile_payload(profile_summary)
    if profile_payload:
        payload["profile_sec"] = profile_payload

    payload.update(_epoch_metric_payload(metrics))
    return payload


def build_train_step_tensorboard_payload(
    *,
    losses: dict[str, Any],
    profile_summary: dict[str, Any],
    elapsed_sec: float,
) -> dict[str, Any]:
    payload: dict[str, Any] = {}

    loss_payload = _train_loss_payload(losses)
    if loss_payload:
        payload["loss"] = loss_payload

    profile_payload = _train_step_profile_payload(profile_summary)
    if profile_payload:
        payload["profile_sec"] = profile_payload

    payload["elapsed_sec"] = float(elapsed_sec)
    return payload


_build_epoch_tensorboard_payload = build_epoch_tensorboard_payload
_build_train_step_tensorboard_payload = build_train_step_tensorboard_payload
_maybe_build_summary_writer = maybe_build_summary_writer
_write_tensorboard_scalars = write_tensorboard_scalars
