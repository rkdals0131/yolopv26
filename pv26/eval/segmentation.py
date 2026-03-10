"""Segmentation evaluation helpers."""

from __future__ import annotations

import torch
from torch import Tensor

from ..training.runtime import update_binary_iou


def lane_subclass_eval_valid_mask(
    *,
    rm_mask: Tensor,
    rm_lane_subclass_mask: Tensor,
    has_rm: Tensor,
    has_rm_lane_subclass: Tensor,
) -> tuple[Tensor, Tensor]:
    """
    Restrict lane-subclass evaluation to GT lane-marker pixels with valid subclass labels.
    """
    if rm_mask.ndim != 4 or rm_mask.shape[1] < 1:
        raise ValueError(f"rm_mask must be [B,C,H,W] with lane_marker at channel 0, got {tuple(rm_mask.shape)}")
    if rm_lane_subclass_mask.ndim != 3:
        raise ValueError(f"rm_lane_subclass_mask must be [B,H,W], got {tuple(rm_lane_subclass_mask.shape)}")
    if rm_mask.shape[0] != rm_lane_subclass_mask.shape[0] or rm_mask.shape[-2:] != rm_lane_subclass_mask.shape[-2:]:
        raise ValueError(
            f"rm_mask / lane_subclass shape mismatch: rm={tuple(rm_mask.shape)} lane={tuple(rm_lane_subclass_mask.shape)}"
        )

    has_rm_lane_marker = has_rm[:, 0].to(dtype=torch.bool).view(-1, 1, 1)
    has_lane_subclass = has_rm_lane_subclass.to(dtype=torch.bool).view(-1, 1, 1)
    gt_lane_marker = (rm_mask[:, 0] == 1) & has_rm_lane_marker
    valid = (rm_lane_subclass_mask != 255) & has_lane_subclass & gt_lane_marker
    supervised = valid.reshape(valid.shape[0], -1).any(dim=1)
    return valid, supervised


__all__ = ["lane_subclass_eval_valid_mask", "update_binary_iou"]
