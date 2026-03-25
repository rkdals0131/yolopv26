from __future__ import annotations

from collections import Counter, deque
from contextlib import nullcontext
from datetime import datetime
import math
import json
from pathlib import Path
import time
from typing import Any

import torch

from ..encoding import encode_pv26_batch
from ..loss import PV26MultiTaskLoss
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


def _canonical_stage(stage: str) -> str:
    return STAGE_ALIASES.get(stage, stage)


def _move_to_device(item: Any, device: torch.device) -> Any:
    if isinstance(item, torch.Tensor):
        return item.to(device)
    if isinstance(item, dict):
        return {key: _move_to_device(value, device) for key, value in item.items()}
    if isinstance(item, list):
        return [_move_to_device(value, device) for value in item]
    if isinstance(item, tuple):
        return tuple(_move_to_device(value, device) for value in item)
    return item


def _count_parameters(parameters: list[torch.nn.Parameter]) -> int:
    return sum(parameter.numel() for parameter in parameters)


def _trainable_parameters(module: torch.nn.Module) -> list[torch.nn.Parameter]:
    return [parameter for parameter in module.parameters() if parameter.requires_grad]


def _loss_summary(history: list[dict[str, Any]], name: str) -> dict[str, float]:
    values = [float(item["losses"][name]) for item in history]
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


def _maybe_build_summary_writer(log_dir: Path):
    try:
        from torch.utils.tensorboard import SummaryWriter
    except Exception as exc:  # pragma: no cover - optional dependency.
        return None, {"enabled": False, "status": "unavailable", "error": str(exc), "log_dir": str(log_dir)}
    try:
        writer = SummaryWriter(log_dir=str(log_dir))
    except Exception as exc:  # pragma: no cover - filesystem or environment issue.
        return None, {"enabled": False, "status": "init_failed", "error": str(exc), "log_dir": str(log_dir)}
    return writer, {"enabled": True, "status": "active", "error": None, "log_dir": str(log_dir)}


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
    allow_background = encoded["mask"].get("det_allow_background_negatives")
    if allow_background is None:
        allow_background = torch.ones_like(det_source)
    else:
        allow_background = allow_background.to(dtype=torch.bool)
    class_mask = encoded["mask"].get("det_supervised_class_mask")
    if class_mask is None:
        class_mask = torch.ones((det_source.shape[0], len(OD_CLASSES)), dtype=torch.bool, device=det_source.device)
    else:
        class_mask = class_mask.to(dtype=torch.bool)

    partial_det = det_source & ~allow_background
    full_det = det_source & allow_background
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
        "full_det_samples": _true_count(full_det),
        "background_negative_enabled_samples": _true_count(full_det),
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
    return {
        "image": torch.cat([batch["image"] for batch in batches], dim=0),
        "det_targets": [item for batch in batches for item in batch["det_targets"]],
        "tl_attr_targets": [item for batch in batches for item in batch["tl_attr_targets"]],
        "lane_targets": [item for batch in batches for item in batch["lane_targets"]],
        "source_mask": [item for batch in batches for item in batch["source_mask"]],
        "valid_mask": [item for batch in batches for item in batch["valid_mask"]],
        "meta": [item for batch in batches for item in batch["meta"]],
    }


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
        self.tensorboard_status: dict[str, Any] = {
            "enabled": False,
            "status": "inactive",
            "error": None,
            "log_dir": None,
        }
        self._tensorboard_train_step = 0

    def prepare_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        encoded = batch if "det_gt" in batch else encode_pv26_batch(batch)
        return _move_to_device(encoded, self.device)

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

        skipped_reason: str | None = None
        optimizer_step = False
        losses: dict[str, torch.Tensor]
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
            _sync_timing_device(self.device, profile_device_sync)
            backward_ended_at = time.perf_counter()
        except RuntimeError as exc:
            if not self.oom_guard or not _is_oom_error(exc):
                raise
            skipped_reason = "oom_recovered"
            self.optimizer.zero_grad(set_to_none=True)
            self.micro_step = 0
            self.skipped_steps += 1
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            losses = {
                "total": torch.full((), float("nan"), device=self.device),
                "det": torch.full((), float("nan"), device=self.device),
                "tl_attr": torch.full((), float("nan"), device=self.device),
                "lane": torch.full((), float("nan"), device=self.device),
                "stop_line": torch.full((), float("nan"), device=self.device),
                "crosswalk": torch.full((), float("nan"), device=self.device),
            }
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
            "losses": {
                name: float(value.detach().cpu())
                for name, value in losses.items()
            },
            "optimizer_step": optimizer_step,
            "micro_step": int(self.micro_step),
            "accumulate_steps": int(self.accumulate_steps),
            "skipped_reason": skipped_reason,
            "skipped_steps": int(self.skipped_steps),
            "amp_enabled": bool(self.amp_enabled),
            "gradient_scale": float(self.scaler.get_scale()) if self.amp_enabled else 1.0,
            "optimizer_lrs": {
                str(group.get("group_name", f"group_{index}")): float(group["lr"])
                for index, group in enumerate(self.optimizer.param_groups)
            },
            "trainable": dict(self.stage_summary),
            "assignment": {
                "det": str(getattr(self.criterion, "last_det_assignment_mode", "unknown")),
                "lane": dict(getattr(self.criterion, "last_lane_assignment_modes", {})),
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
                {
                    "loss": summary["losses"],
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
                    },
                },
                self._tensorboard_train_step,
            )
        return summary

    def summarize_history(self, *, last_n: int | None = None) -> dict[str, Any]:
        if not self.history:
            raise ValueError("trainer history is empty")
        window = self.history[-last_n:] if last_n is not None else self.history
        summary: dict[str, Any] = {
            "steps": len(window),
            "global_step": int(window[-1]["global_step"]),
            "stage": str(window[-1]["stage"]),
            "assignment": {
                "det": str(window[-1]["assignment"]["det"]),
                "lane": dict(window[-1]["assignment"]["lane"]),
            },
            "losses": {},
        }
        for name in window[-1]["losses"]:
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
        if checkpoint_stage != self.stage:
            optimizer_hparams = _optimizer_group_hparams(self.optimizer)
            self.stage = checkpoint_stage
            self.stage_summary = configure_pv26_train_stage(self.adapter, self.heads, self.stage)
            self.criterion = PV26MultiTaskLoss(stage=self.stage).to(self.device)
            self.optimizer = build_pv26_optimizer(
                self.adapter,
                self.heads,
                trunk_lr=optimizer_hparams["trunk_lr"],
                head_lr=optimizer_hparams["head_lr"],
                weight_decay=optimizer_hparams["weight_decay"],
            )
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

    def build_evaluator(self):
        from ..eval import PV26Evaluator

        return PV26Evaluator(
            self.adapter,
            self.heads,
            stage=self.stage,
            device=self.device,
        )

    def train_epoch(
        self,
        loader,
        *,
        epoch: int,
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
                    {
                        "timing": step_summary["timing"],
                        "progress": {
                            "elapsed_sec": elapsed_sec,
                            "eta_sec": eta_sec if eta_sec is not None else 0.0,
                            "iteration": int(batch_index),
                            "total_iterations": int(total_batches) if total_batches is not None else 0,
                        },
                        "profile": profile_summary,
                    },
                    max(1, self._tensorboard_train_step),
                )
            should_log = batch_index % max(1, int(log_every_n_steps)) == 0
            if total_batches is not None and batch_index == total_batches:
                should_log = True
            if should_log:
                profile_iteration = step_summary["profile"]["iteration_sec"]
                print(
                    "[train] "
                    f"epoch={epoch} iter={batch_index}/{total_batches if total_batches is not None else '?'} "
                    f"global_step={step_summary['global_step']} "
                    f"epoch_start={epoch_started_at_iso} "
                    f"elapsed={_format_duration(elapsed_sec)} "
                    f"eta={_format_duration(eta_sec)} "
                    f"iter_mean={profile_iteration['mean']:.3f}s "
                    f"iter_p50={profile_iteration['p50']:.3f}s "
                    f"iter_p99={profile_iteration['p99']:.3f}s "
                    f"wait={step_summary['profile']['wait_sec']['mean']:.3f}s "
                    f"load={step_summary['profile']['load_sec']['mean']:.3f}s "
                    f"fwd={step_summary['profile']['forward_sec']['mean']:.3f}s "
                    f"loss={step_summary['profile']['loss_sec']['mean']:.3f}s "
                    f"bwd={step_summary['profile']['backward_sec']['mean']:.3f}s",
                    flush=True,
                )
        if not step_summaries:
            raise ValueError("train_epoch received zero batches")
        ended_at = time.perf_counter()
        return {
            "epoch": int(epoch),
            "epoch_started_at": epoch_started_at_iso,
            "epoch_ended_at": _now_iso(),
            "batches": len(step_summaries),
            "global_step_start": int(start_step),
            "global_step_end": int(self.global_step),
            "duration_sec": ended_at - started_at,
            "timing_profile": _timing_profile(step_summaries),
            "losses": _loss_stats_from_summaries(step_summaries),
            "optimizer_lrs": dict(step_summaries[-1]["optimizer_lrs"]),
            "assignment": _aggregate_assignment_modes(step_summaries),
            "source_counts": _aggregate_count_tree(step_summaries, "source_counts"),
            "det_supervision": _aggregate_count_tree(step_summaries, "det_supervision"),
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
            needs_predictions = "det_targets" in batch
            batch_summary = evaluator.evaluate_batch(batch, include_predictions=needs_predictions)
            batch_summaries.append(batch_summary)
            if needs_predictions:
                raw_batches.append(batch)
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
        start_epoch = 1
        manifest_path = output_dir / "run_manifest.json"
        previous_writer = self.tensorboard_writer
        previous_status = dict(self.tensorboard_status)
        previous_tb_step = int(self._tensorboard_train_step)
        if enable_tensorboard:
            self.tensorboard_writer, self.tensorboard_status = _maybe_build_summary_writer(tensorboard_dir)
        else:
            self.tensorboard_writer = None
            self.tensorboard_status = {
                "enabled": False,
                "status": "disabled_by_config",
                "error": None,
                "log_dir": str(tensorboard_dir),
            }
        self._tensorboard_train_step = len(self.history)
        if auto_resume:
            resume_candidate = Path(resume_path) if resume_path is not None else checkpoint_dir / "last.pt"
            if resume_candidate.is_file():
                checkpoint = self.load_checkpoint(resume_candidate, map_location=self.device)
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
                        {
                            "train": {
                                "loss": epoch_summary["train"]["losses"],
                                "source": epoch_summary["train"]["source_counts"],
                                "det_supervision": epoch_summary["train"]["det_supervision"],
                            },
                            "val": epoch_summary.get("val", {}),
                            "scheduler": {"lr": epoch_summary.get("scheduler_lrs", [])},
                        },
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
        finally:
            if self.tensorboard_writer is not None:
                self.tensorboard_writer.flush()
                self.tensorboard_writer.close()
            self.tensorboard_writer = previous_writer
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
