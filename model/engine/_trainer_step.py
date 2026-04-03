from __future__ import annotations

import time
from typing import Any

import torch

from common.train_runtime import sync_timing_device as _common_sync_timing_device
from .trainer_reporting import _tensorboard_train_step_payload, _write_tensorboard_scalars
from .loss import PV26DetAssignmentUnavailable


TIMING_KEYS = (
    "wait_sec",
    "load_sec",
    "forward_sec",
    "loss_sec",
    "backward_sec",
    "iteration_sec",
)


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


def _true_count(value: torch.Tensor) -> int:
    return int(value.to(dtype=torch.int64).sum().item())


def _source_counts(encoded: dict[str, Any], *, od_classes: tuple[str, ...]) -> dict[str, int]:
    del od_classes
    mask = encoded["mask"]
    return {
        "det_source_samples": _true_count(mask["det_source"]),
        "tl_attr_source_samples": _true_count(mask["tl_attr_source"]),
        "lane_source_samples": _true_count(mask["lane_source"]),
        "stop_line_source_samples": _true_count(mask["stop_line_source"]),
        "crosswalk_source_samples": _true_count(mask["crosswalk_source"]),
    }


def _det_supervision_summary(encoded: dict[str, Any], *, od_classes: tuple[str, ...]) -> dict[str, Any]:
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
        class_mask = torch.ones((det_source.shape[0], len(od_classes)), dtype=torch.bool, device=det_source.device)
    else:
        class_mask = class_mask.to(dtype=torch.bool)

    partial_det = det_source & (~allow_objectness | ~allow_unmatched_class)
    det_valid = encoded["det_gt"]["valid_mask"].to(dtype=torch.bool)
    det_classes = encoded["det_gt"]["classes"].to(dtype=torch.long)
    batch_size = max(1, int(det_source.shape[0]))

    supervised_class_sample_counts: dict[str, int] = {}
    gt_class_counts: dict[str, int] = {}
    for class_index, class_name in enumerate(od_classes):
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


def sync_timing_device(device: torch.device, enabled: bool) -> None:
    _common_sync_timing_device(torch, device, enabled)


def _nan_losses(device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "total": torch.full((), float("nan"), device=device),
        "det": torch.full((), float("nan"), device=device),
        "tl_attr": torch.full((), float("nan"), device=device),
        "lane": torch.full((), float("nan"), device=device),
        "stop_line": torch.full((), float("nan"), device=device),
        "crosswalk": torch.full((), float("nan"), device=device),
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


def run_train_step(
    trainer: Any,
    batch: dict[str, Any],
    *,
    wait_sec: float = 0.0,
    profile_device_sync: bool = False,
    od_classes: tuple[str, ...],
    is_oom_error_fn: Any,
) -> dict[str, Any]:
    trainer.adapter.raw_model.train()
    trainer.heads.train()
    load_started_at = time.perf_counter()
    encoded = trainer.prepare_batch(batch)
    sync_timing_device(trainer.device, profile_device_sync)
    load_ended_at = time.perf_counter()
    if trainer.micro_step == 0:
        trainer.optimizer.zero_grad(set_to_none=True)

    skipped_reason: str | None = None
    skipped_reason_detail: str | None = None
    optimizer_step = False
    successful = False
    losses: dict[str, torch.Tensor] = _nan_losses(trainer.device)
    det_components = _summary_det_components({})
    assignment_det_mode = "unknown"
    assignment_lane_modes = dict(getattr(trainer.criterion, "last_lane_assignment_modes", {}))
    forward_started_at = load_ended_at
    forward_ended_at = load_ended_at
    loss_started_at = load_ended_at
    loss_ended_at = load_ended_at
    backward_started_at = load_ended_at
    backward_ended_at = load_ended_at
    try:
        sync_timing_device(trainer.device, profile_device_sync)
        forward_started_at = time.perf_counter()
        with trainer._autocast_context():
            predictions = trainer.forward_encoded_batch(encoded)
        sync_timing_device(trainer.device, profile_device_sync)
        forward_ended_at = time.perf_counter()
        loss_started_at = forward_ended_at
        losses = trainer.criterion(predictions, encoded)
        assignment_det_mode = str(getattr(trainer.criterion, "last_det_assignment_mode", "unknown"))
        assignment_lane_modes = dict(getattr(trainer.criterion, "last_lane_assignment_modes", {}))
        det_components = _summary_det_components(getattr(trainer.criterion, "last_det_loss_breakdown", {}))
        sync_timing_device(trainer.device, profile_device_sync)
        loss_ended_at = time.perf_counter()
        total_loss = losses["total"]
        backward_started_at = loss_ended_at
        if not torch.isfinite(total_loss):
            skipped_reason = "non_finite_loss"
            if not trainer.skip_non_finite_loss:
                raise FloatingPointError("non-finite PV26 total loss encountered")
            trainer.optimizer.zero_grad(set_to_none=True)
            trainer.micro_step = 0
            trainer.skipped_steps += 1
        else:
            scaled_total = total_loss / float(trainer.accumulate_steps)
            if trainer.amp_enabled:
                trainer.scaler.scale(scaled_total).backward()
            else:
                scaled_total.backward()
            trainer.micro_step += 1
            if trainer.micro_step >= trainer.accumulate_steps:
                if trainer.grad_clip_norm is not None:
                    if trainer.amp_enabled:
                        trainer.scaler.unscale_(trainer.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        list(trainer.adapter.raw_model.parameters()) + list(trainer.heads.parameters()),
                        max_norm=trainer.grad_clip_norm,
                    )
                if trainer.amp_enabled:
                    trainer.scaler.step(trainer.optimizer)
                    trainer.scaler.update()
                else:
                    trainer.optimizer.step()
                trainer.optimizer.zero_grad(set_to_none=True)
                trainer.global_step += 1
                trainer.micro_step = 0
                optimizer_step = True
            successful = True
        sync_timing_device(trainer.device, profile_device_sync)
        backward_ended_at = time.perf_counter()
    except PV26DetAssignmentUnavailable as exc:
        skipped_reason = "det_assignment_unavailable"
        skipped_reason_detail = str(exc)
        trainer.optimizer.zero_grad(set_to_none=True)
        trainer.micro_step = 0
        trainer.skipped_steps += 1
        losses = _nan_losses(trainer.device)
        det_components = _summary_det_components({})
        assignment_det_mode = "det_assignment_unavailable"
        assignment_lane_modes = {}
        sync_timing_device(trainer.device, profile_device_sync)
        backward_ended_at = time.perf_counter()
    except RuntimeError as exc:
        if not trainer.oom_guard or not is_oom_error_fn(exc):
            raise
        skipped_reason = "oom_recovered"
        skipped_reason_detail = str(exc)
        trainer.optimizer.zero_grad(set_to_none=True)
        trainer.micro_step = 0
        trainer.skipped_steps += 1
        if trainer.device.type == "cuda":
            torch.cuda.empty_cache()
        losses = _nan_losses(trainer.device)
        det_components = _summary_det_components({})
        assignment_det_mode = "oom_recovered"
        assignment_lane_modes = {}
        sync_timing_device(trainer.device, profile_device_sync)
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
        "history_index": len(trainer.history) + 1,
        "global_step": trainer.global_step,
        "stage": trainer.stage,
        "batch_size": int(encoded["image"].shape[0]),
        "successful": successful,
        "losses": {name: float(value.detach().cpu()) for name, value in losses.items()},
        "det_components": det_components,
        "optimizer_step": optimizer_step,
        "micro_step": int(trainer.micro_step),
        "accumulate_steps": int(trainer.accumulate_steps),
        "skipped_reason": skipped_reason,
        "skipped_reason_detail": skipped_reason_detail,
        "skipped_steps": int(trainer.skipped_steps),
        "amp_enabled": bool(trainer.amp_enabled),
        "gradient_scale": float(trainer.scaler.get_scale()) if trainer.amp_enabled else 1.0,
        "optimizer_lrs": {
            str(group.get("group_name", f"group_{index}")): float(group["lr"])
            for index, group in enumerate(trainer.optimizer.param_groups)
        },
        "trainable": dict(trainer.stage_summary),
        "assignment": {
            "det": assignment_det_mode,
            "lane": assignment_lane_modes,
        },
        "timing": timing,
        "source_counts": _source_counts(encoded, od_classes=od_classes),
        "det_supervision": _det_supervision_summary(encoded, od_classes=od_classes),
    }
    trainer.history.append(summary)
    if trainer.tensorboard_writer is not None:
        trainer._tensorboard_train_step += 1
        _write_tensorboard_scalars(
            trainer.tensorboard_writer,
            "train_step",
            _tensorboard_train_step_payload(summary),
            trainer._tensorboard_train_step,
        )
    return summary
