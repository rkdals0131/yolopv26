from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable

import torch

from ..data.target_encoder import encode_pv26_batch
from common.task_mode import LANE_FAMILY_TASK_MODE
from . import _trainer_checkpoint as _checkpoint
from . import _trainer_epochs as _epochs
from . import _trainer_fit as _fit
from . import _trainer_io as _io
from . import _trainer_step as _step
from . import trainer_reporting as _reporting
from .batch import move_batch_to_device
from .loss import PV26MultiTaskLoss
from .multitask_conflict import init_multitask_conflict_state, normalize_multitask_conflict
from .spec import build_loss_spec
from .train_summary import resolve_summary_path
from ..net.trunk import forward_pyramid_features


STAGE_NAMES = (
    "stage_1_frozen_trunk_warmup",
    "stage_2_partial_unfreeze",
    "stage_3_end_to_end_finetune",
    "stage_4_lane_family_finetune",
)
STAGE_ALIASES = {
    "stage_1_head_warmup": "stage_1_frozen_trunk_warmup",
}
RUN_MANIFEST_VERSION = "pv26-train-run-v1"
OD_CLASSES = tuple(build_loss_spec()["model_contract"]["od_classes"])
TIMING_KEYS = _reporting.TIMING_KEYS
TENSORBOARD_LOSS_KEYS = _reporting.TENSORBOARD_LOSS_KEYS
DISTILL_TEACHER_CACHE_KEYS = (
    "lane_row_logits",
    "lane_exist_logits",
    "lane_row_col_expectation",
    "lane_seg_centerline_logits",
    "lane_seg_support_logits",
    "lane_seg_center_offset",
    "lane_seg_tangent_axis",
    "lane_seg_color_logits",
    "lane_seg_type_logits",
    "lane_feature",
    "stop_line_mask_logits",
    "stop_line_center_logits",
    "stop_line_center_offset",
    "stop_line_angle",
    "stop_line_half_length",
    "stop_line_feature",
    "crosswalk_mask_logits",
    "crosswalk_boundary_logits",
    "crosswalk_center_logits",
    "crosswalk_feature",
)


def _canonical_stage(stage: str) -> str:
    return STAGE_ALIASES.get(stage, stage)


def _count_parameters(parameters: list[torch.nn.Parameter]) -> int:
    return sum(parameter.numel() for parameter in parameters)


class PV26DistillTeacher:
    def __init__(self, adapter: Any, heads: torch.nn.Module) -> None:
        self.adapter = adapter
        self.heads = heads
        self.eval()
        for parameter in self.adapter.raw_model.parameters():
            parameter.requires_grad = False
        for parameter in self.heads.parameters():
            parameter.requires_grad = False

    def to(self, device: str | torch.device) -> "PV26DistillTeacher":
        resolved_device = torch.device(device)
        self.adapter.raw_model.to(resolved_device)
        self.heads.to(resolved_device)
        return self

    def eval(self) -> "PV26DistillTeacher":
        self.adapter.raw_model.eval()
        self.heads.eval()
        return self

    @torch.no_grad()
    def build_cache(self, encoded: dict[str, Any]) -> dict[str, torch.Tensor]:
        self.eval()
        features = forward_pyramid_features(self.adapter, encoded["image"])
        outputs = self.heads(features, encoded=encoded) if getattr(self.heads, "supports_encoded_context", False) else self.heads(features)
        return {
            key: value.detach()
            for key, value in outputs.items()
            if key in DISTILL_TEACHER_CACHE_KEYS and isinstance(value, torch.Tensor)
        }


_DISTILL_TASK_CACHE_PREFIXES = {
    "lane": ("lane_",),
    "stop_line": ("stop_line_",),
    "crosswalk": ("crosswalk_",),
}


class PV26TaskRoutedDistillTeacher:
    def __init__(
        self,
        default_teacher: PV26DistillTeacher,
        task_teachers: dict[str, PV26DistillTeacher],
    ) -> None:
        self.default_teacher = default_teacher
        self.task_teachers = dict(task_teachers)

    def to(self, device: str | torch.device) -> "PV26TaskRoutedDistillTeacher":
        self.default_teacher.to(device)
        for teacher in self.task_teachers.values():
            teacher.to(device)
        return self

    def eval(self) -> "PV26TaskRoutedDistillTeacher":
        self.default_teacher.eval()
        for teacher in self.task_teachers.values():
            teacher.eval()
        return self

    @torch.no_grad()
    def build_cache(self, encoded: dict[str, Any]) -> dict[str, torch.Tensor]:
        cache = dict(self.default_teacher.build_cache(encoded))
        for task_name, teacher in self.task_teachers.items():
            prefixes = _DISTILL_TASK_CACHE_PREFIXES.get(str(task_name), ())
            if not prefixes:
                continue
            task_cache = teacher.build_cache(encoded)
            for key, value in task_cache.items():
                if key.startswith(prefixes):
                    cache[key] = value
        return cache


def _trainable_parameters(module: torch.nn.Module) -> list[torch.nn.Parameter]:
    return [parameter for parameter in module.parameters() if parameter.requires_grad]


def _trainable_parameters_from_modules(modules: list[torch.nn.Module]) -> list[torch.nn.Parameter]:
    parameters: list[torch.nn.Parameter] = []
    seen: set[int] = set()
    for module in modules:
        for parameter in module.parameters():
            if not parameter.requires_grad:
                continue
            parameter_id = id(parameter)
            if parameter_id in seen:
                continue
            seen.add(parameter_id)
            parameters.append(parameter)
    return parameters


def _set_module_requires_grad(module: torch.nn.Module, requires_grad: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = requires_grad


def _lane_family_modules(heads: torch.nn.Module) -> list[torch.nn.Module]:
    lane_family_getter = getattr(heads, "lane_family_modules", None)
    if callable(lane_family_getter):
        modules = [module for module in lane_family_getter() if isinstance(module, torch.nn.Module)]
        if modules:
            return modules
    return [
        module
        for module in (
            getattr(heads, "lane_head", None),
            getattr(heads, "stop_line_head", None),
            getattr(heads, "crosswalk_head", None),
        )
        if isinstance(module, torch.nn.Module)
    ]


def _lane_family_adapter_modules(heads: torch.nn.Module) -> list[torch.nn.Module]:
    adapter_getter = getattr(heads, "lane_family_adapter_modules", None)
    if callable(adapter_getter):
        return [module for module in adapter_getter() if isinstance(module, torch.nn.Module)]
    roadmark_heads = getattr(heads, "roadmark_heads", None)
    adapter_getter = getattr(roadmark_heads, "lane_family_adapter_modules", None)
    if callable(adapter_getter):
        return [module for module in adapter_getter() if isinstance(module, torch.nn.Module)]
    return []


def _stop_line_modules(heads: torch.nn.Module) -> list[torch.nn.Module]:
    getter = getattr(heads, "stop_line_modules", None)
    if callable(getter):
        modules = [module for module in getter() if isinstance(module, torch.nn.Module)]
        if modules:
            return modules
    roadmark_heads = getattr(heads, "roadmark_heads", None)
    getter = getattr(roadmark_heads, "stop_line_modules", None)
    if callable(getter):
        modules = [module for module in getter() if isinstance(module, torch.nn.Module)]
        if modules:
            return modules
    stop_line_head = getattr(heads, "stop_line_head", None)
    return [stop_line_head] if isinstance(stop_line_head, torch.nn.Module) else []


def _parameter_ids_from_modules(modules: list[torch.nn.Module]) -> set[int]:
    parameter_ids: set[int] = set()
    for module in modules:
        for parameter in module.parameters():
            parameter_ids.add(id(parameter))
    return parameter_ids


def _require_lane_family_modules(heads: torch.nn.Module, *, policy: str) -> list[torch.nn.Module]:
    modules = _lane_family_modules(heads)
    if not modules:
        raise RuntimeError(f"{policy} requires lane_head, stop_line_head, and crosswalk_head modules")
    return modules


def _require_named_head_modules(heads: torch.nn.Module, names: tuple[str, ...], *, policy: str) -> list[torch.nn.Module]:
    modules: list[torch.nn.Module] = []
    missing: list[str] = []
    for name in names:
        module = getattr(heads, name, None)
        if isinstance(module, torch.nn.Module):
            modules.append(module)
        else:
            missing.append(name)
    if missing:
        raise RuntimeError(f"{policy} requires {', '.join(missing)} modules")
    return modules


def _require_stop_line_modules(heads: torch.nn.Module, *, policy: str) -> list[torch.nn.Module]:
    modules = _stop_line_modules(heads)
    if not modules:
        raise RuntimeError(f"{policy} requires stop_line_head or stop_line_modules")
    return modules


def _criterion_config_from_instance(criterion: torch.nn.Module, stage: str) -> dict[str, Any] | None:
    export_config = getattr(criterion, "export_config", None)
    if not callable(export_config):
        return None
    config = dict(export_config())
    config["stage"] = _canonical_stage(str(config.get("stage", stage)))
    return config


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


def configure_pv26_train_stage(
    adapter: Any,
    heads: torch.nn.Module,
    stage: str,
    *,
    freeze_policy: str | None = None,
) -> dict[str, int | str]:
    stage = _canonical_stage(stage)
    if stage not in STAGE_NAMES:
        raise KeyError(f"unsupported PV26 train stage: {stage}")

    for parameter in heads.parameters():
        parameter.requires_grad = True

    trunk_layers = list(adapter.trunk.children())
    policy = freeze_policy
    if policy is None:
        if stage == "stage_1_frozen_trunk_warmup":
            policy = "backbone_and_neck"
        elif stage == "stage_2_partial_unfreeze":
            policy = "lower_backbone_only"
        elif stage == "stage_4_lane_family_finetune":
            policy = "lane_family_heads_only"
        else:
            policy = "none"

    head_policy = "all_heads"
    if policy == "backbone_and_neck":
        adapter.freeze_trunk()
    elif policy == "lower_backbone_only":
        adapter.freeze_trunk()
        partial_count = max(1, len(trunk_layers) // 3)
        for layer in trunk_layers[-partial_count:]:
            for parameter in layer.parameters():
                parameter.requires_grad = True
    elif policy == "lane_family_plus_upper_trunk":
        adapter.freeze_trunk()
        partial_count = max(1, len(trunk_layers) // 3)
        for layer in trunk_layers[-partial_count:]:
            for parameter in layer.parameters():
                parameter.requires_grad = True
        _set_module_requires_grad(heads, False)
        for module in _require_lane_family_modules(heads, policy=policy):
            _set_module_requires_grad(module, True)
        head_policy = "lane_family_only"
    elif policy == "lane_family_heads_only":
        adapter.freeze_trunk()
        _set_module_requires_grad(heads, False)
        for module in _require_lane_family_modules(heads, policy=policy):
            _set_module_requires_grad(module, True)
        head_policy = "lane_family_only"
    elif policy == "lane_family_stop_cross_heads_only":
        adapter.freeze_trunk()
        _set_module_requires_grad(heads, False)
        for module in _require_named_head_modules(heads, ("stop_line_head", "crosswalk_head"), policy=policy):
            _set_module_requires_grad(module, True)
        head_policy = "stop_cross_only"
    elif policy == "lane_family_stopline_only":
        adapter.freeze_trunk()
        _set_module_requires_grad(heads, False)
        for module in _require_stop_line_modules(heads, policy=policy):
            _set_module_requires_grad(module, True)
        head_policy = "stopline_only"
    elif policy == "none":
        adapter.unfreeze_trunk()
    else:
        raise KeyError(f"unsupported PV26 freeze policy: {policy}")

    trainable_trunk = _trainable_parameters(adapter.trunk)
    trainable_heads = _trainable_parameters(heads)
    stage_summary: dict[str, int | str] = {
        "stage": stage,
        "freeze_policy": policy,
        "trainable_trunk_params": _count_parameters(trainable_trunk),
        "trainable_head_params": _count_parameters(trainable_heads),
    }
    if hasattr(heads, "det_heads"):
        det_heads = getattr(heads, "det_heads")
        if isinstance(det_heads, torch.nn.Module):
            stage_summary["trainable_det_head_params"] = _count_parameters(_trainable_parameters(det_heads))
    if hasattr(heads, "tl_attr_heads"):
        tl_attr_heads = getattr(heads, "tl_attr_heads")
        if isinstance(tl_attr_heads, torch.nn.Module):
            stage_summary["trainable_tl_attr_head_params"] = _count_parameters(_trainable_parameters(tl_attr_heads))
    lane_family_modules_for_summary = _lane_family_modules(heads)
    lane_family_trainable = _count_parameters(_trainable_parameters_from_modules(lane_family_modules_for_summary))
    if lane_family_trainable:
        stage_summary["trainable_lane_family_head_params"] = lane_family_trainable
    if policy in {
        "lane_family_heads_only",
        "lane_family_plus_upper_trunk",
        "lane_family_stop_cross_heads_only",
        "lane_family_stopline_only",
    }:
        stage_summary["head_training_policy"] = head_policy
    return stage_summary


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
    adapter_param_ids = _parameter_ids_from_modules(_lane_family_adapter_modules(heads))
    adapter_params = [
        parameter
        for parameter in heads.parameters()
        if parameter.requires_grad and id(parameter) in adapter_param_ids
    ]
    head_params = [
        parameter
        for parameter in heads.parameters()
        if parameter.requires_grad and id(parameter) not in adapter_param_ids
    ]

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
    if adapter_params:
        param_groups.append(
            {
                "params": adapter_params,
                "lr": head_lr,
                "weight_decay": weight_decay,
                "group_name": "lane_family_adapters",
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
        stage: str = "stage_1_frozen_trunk_warmup",
        device: str | torch.device = "cpu",
        criterion: torch.nn.Module | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        trunk_lr: float = 1e-4,
        head_lr: float = 1e-3,
        weight_decay: float = 1e-4,
        loss_weights: dict[str, float] | None = None,
        freeze_policy: str | None = None,
        amp: bool = False,
        amp_init_scale: float = 65536.0,
        accumulate_steps: int = 1,
        grad_clip_norm: float | None = None,
        skip_non_finite_loss: bool = False,
        oom_guard: bool = False,
        multitask_conflict: dict[str, Any] | None = None,
        distill_teacher: Any | None = None,
    ) -> None:
        if accumulate_steps <= 0:
            raise ValueError("accumulate_steps must be > 0")
        if amp_init_scale <= 0.0:
            raise ValueError("amp_init_scale must be > 0")
        self.adapter = adapter
        self.heads = heads
        self.device = torch.device(device)
        self.distill_teacher = distill_teacher
        self.stage = _canonical_stage(stage)
        self.freeze_policy = freeze_policy
        self.stage_summary = configure_pv26_train_stage(
            adapter,
            heads,
            self.stage,
            freeze_policy=self.freeze_policy,
        )
        self.adapter.raw_model.to(self.device)
        self.heads.to(self.device)
        teacher_to_device = getattr(self.distill_teacher, "to", None)
        if callable(teacher_to_device):
            teacher_to_device(self.device)
        self.criterion = (criterion or PV26MultiTaskLoss(stage=self.stage, loss_weights=loss_weights)).to(self.device)
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
        self.multitask_conflict = normalize_multitask_conflict(multitask_conflict)
        self.multitask_conflict_state = init_multitask_conflict_state(self.multitask_conflict)
        self.amp_enabled = bool(amp) and self.device.type == "cuda"
        self.amp_init_scale = float(amp_init_scale)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp_enabled, init_scale=self.amp_init_scale)
        self.micro_step = 0
        self.skipped_steps = 0
        self.global_step = 0
        self.train_step_count = 0
        self.history: list[dict[str, Any]] = []
        self.epoch_history: list[dict[str, Any]] = []
        self.tensorboard_writer = None
        self.tensorboard_status: dict[str, Any] = {
            "enabled": False,
            "status": "inactive",
            "error": None,
            "log_dir": None,
            "purge_step": None,
        }
        self._tensorboard_train_step = 0

    def prepare_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        task_mode = str(getattr(self.criterion, "task_mode", LANE_FAMILY_TASK_MODE))
        include_segfirst = str(getattr(self.heads, "lane_head_mode", "row_native")) == "seg_first"
        if "det_gt" in batch:
            encoded = batch
            raw_batch = encoded.get("_raw_batch") if isinstance(encoded.get("_raw_batch"), dict) else None
            has_segfirst = isinstance(encoded.get("roadmark_v2"), dict) and "lane_seg_centerline_core" in encoded["roadmark_v2"]
            if include_segfirst and not has_segfirst and raw_batch is not None:
                encoded = encode_pv26_batch(
                    {"image": encoded["image"], **raw_batch},
                    task_mode=task_mode,
                    include_lane_segfirst_targets=True,
                )
                encoded["_raw_batch"] = raw_batch
        else:
            encoded = encode_pv26_batch(
                batch,
                task_mode=task_mode,
                include_lane_segfirst_targets=include_segfirst,
            )
        return move_batch_to_device(encoded, self.device, non_blocking=self.device.type == "cuda")

    def forward_encoded_batch(self, encoded: dict[str, Any]) -> dict[str, torch.Tensor]:
        features = forward_pyramid_features(self.adapter, encoded["image"])
        return self.heads(features, encoded=encoded) if getattr(self.heads, "supports_encoded_context", False) else self.heads(features)

    def attach_teacher_cache(self, encoded: dict[str, Any], *, phase: str) -> dict[str, Any]:
        teacher = self.distill_teacher
        if teacher is None:
            return encoded
        build_cache = getattr(teacher, "build_cache", None)
        if not callable(build_cache):
            raise TypeError("distill_teacher must provide build_cache(encoded)")
        cache = build_cache(encoded)
        if not isinstance(cache, dict):
            raise TypeError("distill_teacher.build_cache(encoded) must return a dict")
        encoded["teacher_cache"] = cache
        encoded["_distill_phase"] = str(phase)
        return encoded

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
        store_history: bool = True,
    ) -> dict[str, Any]:
        return _step.run_train_step(
            self,
            batch,
            wait_sec=wait_sec,
            profile_device_sync=profile_device_sync,
            store_history=store_history,
            od_classes=OD_CLASSES,
            is_oom_error_fn=_is_oom_error,
        )

    def summarize_history(self, *, last_n: int | None = None) -> dict[str, Any]:
        if not self.history:
            raise ValueError("trainer history is empty")
        window = self.history[-last_n:] if last_n is not None else self.history
        successful_window = _reporting.successful_summaries(window)
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
            summary["losses"][name] = _reporting.loss_summary(window, name)
        return summary

    def save_epoch_history_jsonl(self, path: str | Path) -> Path:
        return _io._write_jsonl_rows(path, self.epoch_history)

    def save_history_jsonl(self, path: str | Path) -> Path:
        return _io._write_jsonl_rows(path, self.history)

    def checkpoint_state(self) -> dict[str, Any]:
        return _checkpoint.checkpoint_state(
            self,
            criterion_config_from_instance_fn=_criterion_config_from_instance,
        )

    def save_checkpoint(self, path: str | Path, *, extra_state: dict[str, Any] | None = None) -> Path:
        return _checkpoint.save_checkpoint(
            self,
            path,
            extra_state=extra_state,
            checkpoint_state_fn=self.checkpoint_state,
        )

    def load_checkpoint(
        self,
        path: str | Path,
        *,
        map_location: str | torch.device | None = None,
    ) -> dict[str, Any]:
        return _checkpoint.load_checkpoint(
            self,
            path,
            map_location=map_location,
            canonical_stage_fn=_canonical_stage,
            optimizer_group_hparams_fn=_optimizer_group_hparams,
            criterion_config_from_instance_fn=_criterion_config_from_instance,
            configure_stage_fn=configure_pv26_train_stage,
            build_optimizer_fn=build_pv26_optimizer,
        )

    def load_model_weights(
        self,
        path: str | Path,
        *,
        map_location: str | torch.device | None = None,
    ) -> dict[str, Any]:
        return _checkpoint.load_model_weights(self, path, map_location=map_location)

    def build_evaluator(self):
        from .evaluator import PV26Evaluator

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
        step_history_enabled: bool = True,
        step_history_every_n_steps: int = 100,
        step_history_include_grad_details: bool = False,
        pcgrad_diagnostics_path: str | Path | None = None,
        pcgrad_diagnostics_enabled: bool = True,
        pcgrad_aggregate_every_n_steps: int = 100,
        pcgrad_keep_raw_every_n_steps: int = 1000,
    ) -> dict[str, Any]:
        return _epochs.run_train_epoch(
            self,
            loader,
            epoch=epoch,
            epoch_total=epoch_total,
            phase_index=phase_index,
            phase_count=phase_count,
            phase_name=phase_name,
            max_batches=max_batches,
            step_log_path=str(step_log_path) if step_log_path is not None else None,
            log_every_n_steps=log_every_n_steps,
            profile_window=profile_window,
            profile_device_sync=profile_device_sync,
            step_history_enabled=step_history_enabled,
            step_history_every_n_steps=step_history_every_n_steps,
            step_history_include_grad_details=step_history_include_grad_details,
            pcgrad_diagnostics_path=str(pcgrad_diagnostics_path) if pcgrad_diagnostics_path is not None else None,
            pcgrad_diagnostics_enabled=pcgrad_diagnostics_enabled,
            pcgrad_aggregate_every_n_steps=pcgrad_aggregate_every_n_steps,
            pcgrad_keep_raw_every_n_steps=pcgrad_keep_raw_every_n_steps,
        )

    def validate_epoch(
        self,
        loader,
        *,
        epoch: int,
        epoch_total: int | None = None,
        phase_index: int | None = None,
        phase_count: int | None = None,
        phase_name: str | None = None,
        evaluator=None,
        max_batches: int | None = None,
        log_every_n_steps: int = 1,
        profile_window: int = 20,
        profile_device_sync: bool = False,
    ) -> dict[str, Any]:
        return _epochs.run_validate_epoch(
            self,
            loader,
            epoch=epoch,
            epoch_total=epoch_total,
            phase_index=phase_index,
            phase_count=phase_count,
            phase_name=phase_name,
            evaluator=evaluator,
            max_batches=max_batches,
            log_every_n_steps=log_every_n_steps,
            profile_window=profile_window,
            profile_device_sync=profile_device_sync,
        )

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
        checkpoint_every: int = 1,
        max_train_batches: int | None = None,
        max_val_batches: int | None = None,
        best_metric: str | None = None,
        best_mode: str = "min",
        auto_resume: bool = False,
        resume_path: str | Path | None = None,
        enable_tensorboard: bool = True,
        selection_metric_callback: Callable[[dict[str, Any]], None] | None = None,
        early_exit_callback: Callable[[dict[str, Any]], dict[str, Any] | None] | None = None,
        epoch_end_callback: Callable[[dict[str, Any]], None] | None = None,
        run_manifest_extra: dict[str, Any] | None = None,
        log_every_n_steps: int = 1,
        profile_window: int = 20,
        profile_device_sync: bool = False,
        step_history_enabled: bool = True,
        step_history_every_n_steps: int = 100,
        step_history_include_grad_details: bool = False,
        pcgrad_diagnostics_enabled: bool = True,
        pcgrad_aggregate_every_n_steps: int = 100,
        pcgrad_keep_raw_every_n_steps: int = 1000,
    ) -> dict[str, Any]:
        return _fit.run_fit(
            self,
            train_loader,
            epochs=epochs,
            phase_index=phase_index,
            phase_count=phase_count,
            phase_name=phase_name,
            val_loader=val_loader,
            run_dir=run_dir,
            checkpoint_every=checkpoint_every,
            max_train_batches=max_train_batches,
            max_val_batches=max_val_batches,
            best_metric=best_metric,
            best_mode=best_mode,
            auto_resume=auto_resume,
            resume_path=resume_path,
            enable_tensorboard=enable_tensorboard,
            selection_metric_callback=selection_metric_callback,
            early_exit_callback=early_exit_callback,
            epoch_end_callback=epoch_end_callback,
            run_manifest_extra=run_manifest_extra,
            log_every_n_steps=log_every_n_steps,
            profile_window=profile_window,
            profile_device_sync=profile_device_sync,
            step_history_enabled=step_history_enabled,
            step_history_every_n_steps=step_history_every_n_steps,
            step_history_include_grad_details=step_history_include_grad_details,
            pcgrad_diagnostics_enabled=pcgrad_diagnostics_enabled,
            pcgrad_aggregate_every_n_steps=pcgrad_aggregate_every_n_steps,
            pcgrad_keep_raw_every_n_steps=pcgrad_keep_raw_every_n_steps,
            default_run_dir_fn=_io._default_run_dir,
            now_iso_fn=_io._now_iso,
            write_json_fn=_io._write_json,
            json_ready_fn=_io._json_ready,
            maybe_build_summary_writer_fn=_io._maybe_build_summary_writer,
            optimizer_group_hparams_fn=_optimizer_group_hparams,
            resolve_summary_path_fn=resolve_summary_path,
            is_better_fn=_is_better,
            write_tensorboard_scalars_fn=_reporting._write_tensorboard_scalars,
            write_tensorboard_histograms_fn=_reporting._write_tensorboard_histograms,
            tensorboard_epoch_payload_fn=_reporting._tensorboard_epoch_payload,
            run_manifest_version=RUN_MANIFEST_VERSION,
        )


def run_pv26_tiny_overfit(
    trainer: PV26Trainer,
    batch: dict[str, Any],
    *,
    steps: int = 8,
) -> dict[str, Any]:
    if steps <= 0:
        raise ValueError("tiny overfit requires steps > 0")

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
    "TIMING_KEYS",
    "TENSORBOARD_LOSS_KEYS",
    "build_pv26_optimizer",
    "build_pv26_scheduler",
    "configure_pv26_train_stage",
    "run_pv26_tiny_overfit",
]
