from __future__ import annotations

from typing import Any

import torch


RAW_BATCH_FIELDS = ("det_targets", "tl_attr_targets", "lane_targets", "source_mask", "valid_mask", "meta")


def _image_batch_size(image: Any, *, context: str) -> int:
    if not isinstance(image, torch.Tensor) or image.ndim != 4:
        shape = tuple(image.shape) if isinstance(image, torch.Tensor) else type(image).__name__
        raise ValueError(f"{context} image must be a 4D tensor batch: shape={shape}")
    return int(image.shape[0])


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


def validate_raw_batch_matches_image(
    raw_batch: dict[str, Any],
    image: torch.Tensor,
    *,
    context: str,
) -> None:
    expected_length = _image_batch_size(image, context=context)
    missing = [field for field in RAW_BATCH_FIELDS if field not in raw_batch]
    if missing:
        raise ValueError(f"{context} _raw_batch missing required fields: {missing}")
    field_lengths = {field: len(raw_batch[field]) for field in RAW_BATCH_FIELDS}
    mismatched = {field: length for field, length in field_lengths.items() if length != expected_length}
    if mismatched:
        raise ValueError(
            f"{context} _raw_batch length must match image batch size: "
            f"batch_size={expected_length} mismatched={mismatched}"
        )


def validate_prediction_batch_matches_image(
    predictions: dict[str, Any],
    image: torch.Tensor,
) -> None:
    expected_length = _image_batch_size(image, context="prediction")
    mismatched = {
        name: int(value.shape[0])
        for name, value in predictions.items()
        if isinstance(value, torch.Tensor) and value.ndim > 0 and int(value.shape[0]) != expected_length
    }
    if mismatched:
        raise ValueError(
            "prediction batch size must match image batch size: "
            f"batch_size={expected_length} mismatched={mismatched}"
        )


def merge_raw_batches(batches: list[dict[str, Any]]) -> dict[str, Any]:
    if not batches:
        raise ValueError("cannot merge zero raw batches")
    for batch_index, batch in enumerate(batches):
        field_lengths = {field: len(batch[field]) for field in RAW_BATCH_FIELDS}
        expected_length = field_lengths["meta"]
        mismatched = {field: length for field, length in field_lengths.items() if length != expected_length}
        if mismatched:
            raise ValueError(
                "raw batch field lengths must match before merge: "
                f"batch_index={batch_index} meta={expected_length} mismatched={mismatched}"
            )
        if "image" in batch:
            image_length = _image_batch_size(batch["image"], context=f"raw batch {batch_index}")
            if image_length != expected_length:
                raise ValueError(
                    "raw batch image batch size must match meta length before merge: "
                    f"batch_index={batch_index} image={image_length} meta={expected_length}"
                )
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
    "merge_raw_batches",
    "move_batch_to_device",
    "raw_batch_for_metrics",
    "validate_prediction_batch_matches_image",
    "validate_raw_batch_matches_image",
]
