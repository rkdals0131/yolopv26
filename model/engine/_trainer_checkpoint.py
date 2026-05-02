from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import torch

from .loss import PV26MultiTaskLoss
from .spec import build_loss_spec
from ..net import load_matching_state_dict


CHECKPOINT_FORMAT_VERSION = 2
ARCHITECTURE_GENERATION = "pv26-road-marking-v3"
SPEC_VERSION = str(build_loss_spec()["version"])


def _checkpoint_metadata(trainer: Any) -> dict[str, Any]:
    describe = getattr(trainer.heads, "describe", None)
    head_summary = describe() if callable(describe) else None
    return {
        "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
        "architecture_generation": ARCHITECTURE_GENERATION,
        "spec_version": SPEC_VERSION,
        "head_summary": head_summary,
    }


def _require_exact_resume_compatible(checkpoint: dict[str, Any], path: str | Path) -> None:
    metadata = checkpoint.get("checkpoint_metadata")
    if not isinstance(metadata, dict):
        raise RuntimeError(
            "exact resume unsupported for checkpoints without checkpoint_metadata; "
            f"use weights-only migration instead: {path}"
        )
    generation = str(metadata.get("architecture_generation") or "")
    if generation != ARCHITECTURE_GENERATION:
        raise RuntimeError(
            "exact resume unsupported for checkpoint architecture generation "
            f"{generation or 'unknown'}; expected {ARCHITECTURE_GENERATION}. "
            f"Use weights-only migration instead: {path}"
        )


def checkpoint_state(
    trainer: Any,
    *,
    criterion_config_from_instance_fn: Callable[[torch.nn.Module, str], dict[str, Any] | None],
) -> dict[str, Any]:
    checkpoint = {
        "stage": trainer.stage,
        "global_step": trainer.global_step,
        "stage_summary": dict(trainer.stage_summary),
        "adapter_state_dict": trainer.adapter.raw_model.state_dict(),
        "heads_state_dict": trainer.heads.state_dict(),
        "optimizer_state_dict": trainer.optimizer.state_dict(),
        "criterion_stage": str(getattr(trainer.criterion, "stage", trainer.stage)),
        "history": list(trainer.history),
        "epoch_history": list(trainer.epoch_history),
        "micro_step": int(trainer.micro_step),
        "skipped_steps": int(trainer.skipped_steps),
        "accumulate_steps": int(trainer.accumulate_steps),
        "grad_clip_norm": trainer.grad_clip_norm,
        "amp_enabled": bool(trainer.amp_enabled),
        "multitask_conflict": dict(getattr(trainer, "multitask_conflict", {})),
        "multitask_conflict_state": dict(getattr(trainer, "multitask_conflict_state", {})),
        "checkpoint_metadata": _checkpoint_metadata(trainer),
    }
    criterion_config = criterion_config_from_instance_fn(trainer.criterion, trainer.stage)
    if criterion_config is not None:
        checkpoint["criterion_config"] = criterion_config
    if trainer.scheduler is not None:
        checkpoint["scheduler_state_dict"] = trainer.scheduler.state_dict()
    if trainer.amp_enabled:
        checkpoint["scaler_state_dict"] = trainer.scaler.state_dict()
    return checkpoint


def save_checkpoint(
    trainer: Any,
    path: str | Path,
    *,
    extra_state: dict[str, Any] | None = None,
    checkpoint_state_fn: Callable[[], dict[str, Any]],
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint_state_fn()
    if extra_state:
        checkpoint["extra_state"] = dict(extra_state)
    torch.save(checkpoint, output_path)
    return output_path


def load_checkpoint(
    trainer: Any,
    path: str | Path,
    *,
    map_location: str | torch.device | None = None,
    canonical_stage_fn: Callable[[str], str],
    optimizer_group_hparams_fn: Callable[[torch.optim.Optimizer], dict[str, float]],
    criterion_config_from_instance_fn: Callable[[torch.nn.Module, str], dict[str, Any] | None],
    configure_stage_fn: Callable[[Any, torch.nn.Module, str], dict[str, int | str]],
    build_optimizer_fn: Callable[..., torch.optim.Optimizer],
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=map_location or trainer.device)
    _require_exact_resume_compatible(checkpoint, path)
    checkpoint_stage = canonical_stage_fn(str(checkpoint.get("stage", trainer.stage)))
    optimizer_hparams = optimizer_group_hparams_fn(trainer.optimizer)
    current_criterion_config = criterion_config_from_instance_fn(trainer.criterion, trainer.stage)
    checkpoint_criterion_config = checkpoint.get("criterion_config")
    if isinstance(checkpoint_criterion_config, dict):
        criterion_config = dict(checkpoint_criterion_config)
        criterion_config["stage"] = checkpoint_stage
    elif current_criterion_config is not None:
        criterion_config = dict(current_criterion_config)
        criterion_config["stage"] = checkpoint_stage
    else:
        criterion_config = None

    if checkpoint_stage != trainer.stage:
        trainer.stage = checkpoint_stage
        trainer.stage_summary = configure_stage_fn(trainer.adapter, trainer.heads, trainer.stage)
        trainer.optimizer = build_optimizer_fn(
            trainer.adapter,
            trainer.heads,
            trunk_lr=optimizer_hparams["trunk_lr"],
            head_lr=optimizer_hparams["head_lr"],
            weight_decay=optimizer_hparams["weight_decay"],
        )
    if criterion_config is not None:
        trainer.criterion = PV26MultiTaskLoss(**criterion_config).to(trainer.device)
    trainer.adapter.raw_model.load_state_dict(checkpoint["adapter_state_dict"])
    trainer.heads.load_state_dict(checkpoint["heads_state_dict"])
    trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if trainer.scheduler is not None and "scheduler_state_dict" in checkpoint:
        trainer.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if trainer.amp_enabled and "scaler_state_dict" in checkpoint:
        trainer.scaler.load_state_dict(checkpoint["scaler_state_dict"])
    trainer.global_step = int(checkpoint.get("global_step", 0))
    trainer.stage_summary = dict(checkpoint.get("stage_summary", trainer.stage_summary))
    trainer.history = list(checkpoint.get("history", []))
    trainer.epoch_history = list(checkpoint.get("epoch_history", []))
    trainer.micro_step = int(checkpoint.get("micro_step", 0))
    trainer.skipped_steps = int(checkpoint.get("skipped_steps", 0))
    if isinstance(checkpoint.get("multitask_conflict"), dict):
        trainer.multitask_conflict = dict(checkpoint["multitask_conflict"])
    if isinstance(checkpoint.get("multitask_conflict_state"), dict):
        trainer.multitask_conflict_state = dict(checkpoint["multitask_conflict_state"])
    return checkpoint


def load_model_weights(
    trainer: Any,
    path: str | Path,
    *,
    map_location: str | torch.device | None = None,
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=map_location or trainer.device)
    adapter_report = load_matching_state_dict(trainer.adapter.raw_model, checkpoint["adapter_state_dict"])
    heads_report = load_matching_state_dict(trainer.heads, checkpoint["heads_state_dict"])
    checkpoint["load_policy"] = "shape_aware_partial"
    checkpoint["adapter_load_report"] = adapter_report
    checkpoint["heads_load_report"] = heads_report
    return checkpoint
