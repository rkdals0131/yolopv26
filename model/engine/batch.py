from __future__ import annotations

from typing import Any

import torch


def move_batch_to_device(item: Any, device: torch.device, *, non_blocking: bool = False) -> Any:
    if isinstance(item, torch.Tensor):
        return item.to(device, non_blocking=non_blocking)
    if isinstance(item, dict):
        return {key: move_batch_to_device(value, device, non_blocking=non_blocking) for key, value in item.items()}
    if isinstance(item, list):
        return [move_batch_to_device(value, device, non_blocking=non_blocking) for value in item]
    if isinstance(item, tuple):
        return tuple(move_batch_to_device(value, device, non_blocking=non_blocking) for value in item)
    return item


def raw_batch_for_metrics(batch: dict[str, Any]) -> dict[str, Any] | None:
    raw_batch = batch.get("_raw_batch")
    if isinstance(raw_batch, dict):
        return raw_batch
    if "det_targets" in batch:
        return batch
    return None


def augment_lane_family_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
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


__all__ = [
    "augment_lane_family_metrics",
    "move_batch_to_device",
    "raw_batch_for_metrics",
]
