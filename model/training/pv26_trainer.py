from __future__ import annotations

from typing import Any

import torch

from ..encoding import encode_pv26_batch
from ..loss import PV26MultiTaskLoss
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
        trunk_lr: float = 1e-4,
        head_lr: float = 1e-3,
        weight_decay: float = 1e-4,
    ) -> None:
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
        self.global_step = 0

    def prepare_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        encoded = batch if "det_gt" in batch else encode_pv26_batch(batch)
        return _move_to_device(encoded, self.device)

    def forward_encoded_batch(self, encoded: dict[str, Any]) -> dict[str, torch.Tensor]:
        features = forward_pyramid_features(self.adapter, encoded["image"])
        return self.heads(features)

    def train_step(self, batch: dict[str, Any]) -> dict[str, Any]:
        self.adapter.raw_model.train()
        self.heads.train()
        encoded = self.prepare_batch(batch)
        self.optimizer.zero_grad(set_to_none=True)
        predictions = self.forward_encoded_batch(encoded)
        losses = self.criterion(predictions, encoded)
        losses["total"].backward()
        self.optimizer.step()
        self.global_step += 1
        return {
            "global_step": self.global_step,
            "stage": self.stage,
            "batch_size": int(encoded["image"].shape[0]),
            "losses": {
                name: float(value.detach().cpu())
                for name, value in losses.items()
            },
            "optimizer_lrs": {
                str(group.get("group_name", f"group_{index}")): float(group["lr"])
                for index, group in enumerate(self.optimizer.param_groups)
            },
            "trainable": dict(self.stage_summary),
        }


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
