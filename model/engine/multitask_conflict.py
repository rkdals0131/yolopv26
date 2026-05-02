from __future__ import annotations

from typing import Any

import torch


CONFLICT_TASKS = ("det", "tl_attr", "lane", "stop_line", "crosswalk")


def normalize_multitask_conflict(raw: dict[str, Any] | None) -> dict[str, Any]:
    payload = dict(raw or {})
    enabled = bool(payload.get("enabled", False))
    mode = str(payload.get("mode", "none"))
    tasks_raw = payload.get("tasks", CONFLICT_TASKS)
    if not isinstance(tasks_raw, (list, tuple)):
        raise TypeError("multitask_conflict.tasks must be a list")
    tasks = tuple(str(task) for task in tasks_raw if str(task) in CONFLICT_TASKS)
    if mode not in {"none", "pcgrad_style"}:
        raise ValueError("multitask_conflict.mode must be one of: none, pcgrad_style")
    if not tasks:
        tasks = CONFLICT_TASKS
    return {
        "enabled": enabled,
        "mode": mode,
        "tasks": tasks,
    }


def init_multitask_conflict_state(config: dict[str, Any] | None) -> dict[str, Any]:
    normalized = normalize_multitask_conflict(config)
    return {
        "enabled": bool(normalized.get("enabled", False)),
        "mode": str(normalized.get("mode", "none")),
        "tasks": tuple(str(task) for task in normalized.get("tasks", CONFLICT_TASKS)),
        "accumulated_trunk_grads": [],
        "accumulated_micro_steps": 0,
    }


def reset_multitask_conflict_state(state: dict[str, Any] | None) -> dict[str, Any]:
    payload = dict(state or {})
    payload["accumulated_trunk_grads"] = []
    payload["accumulated_micro_steps"] = 0
    return payload


def accumulate_pcgrad_trunk_update(
    state: dict[str, Any] | None,
    grads: list[torch.Tensor | None] | None,
) -> dict[str, Any]:
    payload = dict(state or {})
    current = payload.get("accumulated_trunk_grads", [])
    if not isinstance(current, list):
        current = []
    if grads is None:
        payload["accumulated_trunk_grads"] = current
        return payload
    if not current:
        payload["accumulated_trunk_grads"] = [
            None if grad is None else grad.detach().clone()
            for grad in grads
        ]
    else:
        combined: list[torch.Tensor | None] = []
        max_len = max(len(current), len(grads))
        for index in range(max_len):
            left = current[index] if index < len(current) else None
            right = grads[index] if index < len(grads) else None
            if left is None and right is None:
                combined.append(None)
            elif left is None:
                combined.append(None if right is None else right.detach().clone())
            elif right is None:
                combined.append(left)
            else:
                combined.append(left + right.detach())
        payload["accumulated_trunk_grads"] = combined
    payload["accumulated_micro_steps"] = int(payload.get("accumulated_micro_steps", 0)) + 1
    return payload


def current_pcgrad_trunk_update(state: dict[str, Any] | None) -> list[torch.Tensor | None] | None:
    payload = dict(state or {})
    grads = payload.get("accumulated_trunk_grads", [])
    if not isinstance(grads, list) or not grads:
        return None
    return grads


def _grad_norm(grad_list: list[torch.Tensor | None]) -> float:
    total = 0.0
    for grad in grad_list:
        if grad is None:
            continue
        total += float(grad.detach().pow(2).sum().item())
    return total ** 0.5 if total > 0.0 else 0.0


def _dot(left: list[torch.Tensor | None], right: list[torch.Tensor | None]) -> torch.Tensor:
    result = None
    for lhs, rhs in zip(left, right):
        if lhs is None or rhs is None:
            continue
        value = (lhs * rhs).sum()
        result = value if result is None else result + value
    if result is None:
        return torch.zeros((), dtype=torch.float32)
    return result


def _norm_sq(values: list[torch.Tensor | None]) -> torch.Tensor:
    result = None
    for value in values:
        if value is None:
            continue
        item = (value * value).sum()
        result = item if result is None else result + item
    if result is None:
        return torch.zeros((), dtype=torch.float32)
    return result


def _clone_grads(grads: list[torch.Tensor | None]) -> list[torch.Tensor | None]:
    return [None if grad is None else grad.detach().clone() for grad in grads]


def compute_pcgrad_trunk_update(
    losses: dict[str, torch.Tensor],
    *,
    params: list[torch.nn.Parameter],
    config: dict[str, Any],
) -> tuple[list[torch.Tensor | None] | None, dict[str, Any]]:
    if not bool(config.get("enabled", False)) or str(config.get("mode")) != "pcgrad_style":
        return None, {"enabled": False, "mode": "none"}
    if not params:
        return None, {"enabled": False, "mode": "pcgrad_style", "reason": "no_trunk_params"}

    tasks = tuple(
        str(task)
        for task in config["tasks"]
        if task in losses and isinstance(losses[task], torch.Tensor) and bool(losses[task].requires_grad)
    )
    if not tasks:
        return None, {"enabled": False, "mode": "pcgrad_style", "reason": "no_differentiable_task_losses"}

    raw_grads = {
        task: [
            None if grad is None else grad.detach()
            for grad in torch.autograd.grad(losses[task], params, retain_graph=True, allow_unused=True)
        ]
        for task in tasks
    }
    projected = {task: _clone_grads(raw_grads[task]) for task in tasks}
    pairwise_dots: dict[str, dict[str, float]] = {task: {} for task in tasks}
    conflict_pairs: list[tuple[str, str]] = []

    for task in tasks:
        for other in tasks:
            if task == other:
                continue
            dot = _dot(projected[task], raw_grads[other])
            pairwise_dots[task][other] = float(dot.detach().cpu())
            if float(dot.detach().cpu()) < 0.0:
                norm_sq = _norm_sq(raw_grads[other]).clamp(min=1.0e-12)
                coeff = dot / norm_sq
                updated: list[torch.Tensor | None] = []
                for grad, ref in zip(projected[task], raw_grads[other]):
                    if grad is None:
                        updated.append(None)
                    elif ref is None:
                        updated.append(grad)
                    else:
                        updated.append(grad - coeff * ref)
                projected[task] = updated
                conflict_pairs.append((task, other))

    combined: list[torch.Tensor | None] = []
    for param_index in range(len(params)):
        items = [projected[task][param_index] for task in tasks if projected[task][param_index] is not None]
        if not items:
            combined.append(None)
            continue
        total = items[0]
        for item in items[1:]:
            total = total + item
        combined.append(total / float(len(items)))

    snapshot = {
        "enabled": True,
        "mode": "pcgrad_style",
        "tasks": list(tasks),
        "conflict_pairs": [[left, right] for left, right in conflict_pairs],
        "pairwise_dots": pairwise_dots,
        "raw_grad_norms": {task: _grad_norm(raw_grads[task]) for task in tasks},
        "projected_grad_norms": {task: _grad_norm(projected[task]) for task in tasks},
        "combined_grad_norm": _grad_norm(combined),
    }
    return combined, snapshot


__all__ = [
    "accumulate_pcgrad_trunk_update",
    "CONFLICT_TASKS",
    "compute_pcgrad_trunk_update",
    "current_pcgrad_trunk_update",
    "init_multitask_conflict_state",
    "normalize_multitask_conflict",
    "reset_multitask_conflict_state",
]
