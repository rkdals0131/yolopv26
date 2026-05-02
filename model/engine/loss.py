from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from scipy.optimize import linear_sum_assignment
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from ultralytics.utils.metrics import bbox_iou as ultralytics_bbox_iou
    from ultralytics.utils.tal import TaskAlignedAssigner
except ImportError:  # pragma: no cover - depends on external environment.
    ultralytics_bbox_iou = None
    TaskAlignedAssigner = None

from .det_geometry import decode_anchor_relative_boxes, make_anchor_grid
from .spec import build_loss_spec, render_loss_spec_markdown
from common.pv26_schema import LANE_CLASSES
from common.task_mode import (
    CROSSWALK_ONLY_TASK_MODE,
    LANE_FAMILY_TASK_MODE,
    LANE_ONLY_TASK_MODE,
    ROADMARK_JOINT_TASK_MODE,
    STOPLINE_ONLY_TASK_MODE,
    canonicalize_task_mode,
    filter_loss_weights_for_task_mode,
)


SPEC = build_loss_spec()
TL_BITS = tuple(SPEC["model_contract"]["tl_bits"])
OD_CLASSES = tuple(SPEC["model_contract"]["od_classes"])
TL_CLASS_ID = OD_CLASSES.index("traffic_light")
LANE_COLOR_DIM = int(SPEC["heads"]["lane"]["target_encoding"]["color_logits"])
LANE_TYPE_DIM = int(SPEC["heads"]["lane"]["target_encoding"]["type_logits"])
LANE_ANCHOR_COUNT = int(SPEC["heads"]["lane"]["target_encoding"]["anchor_rows"])
LANE_COLOR_SLICE = slice(1, 1 + LANE_COLOR_DIM)
LANE_TYPE_SLICE = slice(LANE_COLOR_SLICE.stop, LANE_COLOR_SLICE.stop + LANE_TYPE_DIM)
LANE_X_SLICE = slice(LANE_TYPE_SLICE.stop, LANE_TYPE_SLICE.stop + LANE_ANCHOR_COUNT)
LANE_VIS_SLICE = slice(LANE_X_SLICE.stop, LANE_X_SLICE.stop + LANE_ANCHOR_COUNT)
STOP_LINE_POINT_COUNT = int(SPEC["heads"]["stop_line"]["target_encoding"]["polyline_points"])
CROSSWALK_POINT_COUNT = int(SPEC["heads"]["crosswalk"]["target_encoding"]["sequence_points"])
STAGE_LOSS_WEIGHTS = {
    stage["name"]: dict(stage["loss_weights"]) for stage in SPEC["training_schedule"]
}
SEG_FIRST_LANE_LOSS_WEIGHTS = {
    "centerline_bce": 0.75,
    "centerline_dice": 0.75,
    "support_bce": 0.25,
    "tangent": 0.25,
    "color": 0.50,
    "type": 0.25,
}
STAGE_ALIASES = {
    "stage_1_head_warmup": "stage_1_frozen_trunk_warmup",
}
DEFAULT_DET_FEATURE_STRIDES = (8, 16, 32)


class PV26DetAssignmentUnavailable(RuntimeError):
    pass


def _canonical_stage(stage: str) -> str:
    return STAGE_ALIASES.get(stage, stage)


def _zero_graph(*tensors: torch.Tensor) -> torch.Tensor:
    zero = None
    for tensor in tensors:
        if not isinstance(tensor, torch.Tensor):
            continue
        safe_tensor = torch.where(torch.isfinite(tensor), tensor, torch.zeros_like(tensor))
        term = safe_tensor.sum() * 0.0
        zero = term if zero is None else zero + term
    if zero is None:
        zero = torch.tensor(0.0)
    return zero


def _mean_or_zero(values: list[torch.Tensor], *graph_tensors: torch.Tensor) -> torch.Tensor:
    if values:
        return torch.stack(values).mean()
    return _zero_graph(*graph_tensors)


def _logit_kl_divergence(student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
    student_log_probs = F.log_softmax(student_logits, dim=-1)
    teacher_probs = F.softmax(teacher_logits, dim=-1)
    return F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")


def _binary_logit_distill(student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
    teacher_probs = torch.sigmoid(teacher_logits)
    return F.binary_cross_entropy_with_logits(student_logits, teacher_probs, reduction="mean")


def _soft_dice_distill(student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
    student_probs = torch.sigmoid(student_logits)
    teacher_probs = torch.sigmoid(teacher_logits)
    numerator = 2.0 * (student_probs * teacher_probs).sum()
    denominator = student_probs.square().sum() + teacher_probs.square().sum()
    if not bool(torch.isfinite(denominator)):
        return _zero_graph(student_logits, teacher_logits)
    return 1.0 - ((numerator + 1.0) / (denominator + 1.0))


def _feature_cosine_similarity(student_feature: torch.Tensor, teacher_feature: torch.Tensor) -> torch.Tensor:
    if student_feature.shape != teacher_feature.shape and student_feature.ndim == teacher_feature.ndim == 4:
        teacher_feature = F.interpolate(
            teacher_feature,
            size=tuple(student_feature.shape[-2:]),
            mode="bilinear",
            align_corners=False,
        )
    return F.cosine_similarity(student_feature.flatten(1), teacher_feature.flatten(1), dim=1).mean()


def _prediction_reference_tensor(predictions: dict[str, torch.Tensor]) -> torch.Tensor:
    for value in predictions.values():
        if isinstance(value, torch.Tensor):
            return value
    raise KeyError("predictions must include at least one tensor payload")


def _loss_precision_predictions(predictions: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(predictions)
    for key in (
        "det",
        "tl_attr",
        "lane",
        "stop_line",
        "crosswalk",
        "lane_row_logits",
        "lane_exist_logits",
        "lane_slot_obj_logits",
        "lane_slot_color_logits",
        "lane_slot_type_logits",
        "lane_row_col_expectation",
        "lane_seed_logits",
        "lane_seed_offset",
        "lane_centerline_logits",
        "lane_seed_color_prior",
        "lane_seed_scores",
        "lane_proposal",
        "lane_refine_delta",
        "lane_seg_centerline_logits",
        "lane_seg_support_logits",
        "lane_seg_tangent_axis",
        "lane_seg_color_logits",
        "lane_seg_type_logits",
        "stop_line_mask_logits",
        "stop_line_row_logits",
        "stop_line_x_logits",
        "stop_line_selector_map_logits",
        "stop_line_center_logits",
        "stop_line_center_offset",
        "stop_line_angle",
        "stop_line_half_length",
        "crosswalk_mask_logits",
        "crosswalk_boundary_logits",
        "crosswalk_center_logits",
    ):
        value = normalized.get(key)
        if isinstance(value, torch.Tensor) and value.dtype != torch.float32:
            # Keep AMP/autocast in the model forward path, but evaluate loss geometry
            # and matching in float32 for numerical stability.
            normalized[key] = value.to(dtype=torch.float32)
            value = normalized[key]
        if isinstance(value, torch.Tensor) and not bool(torch.isfinite(value).all()):
            normalized[key] = torch.where(torch.isfinite(value), value, torch.zeros_like(value))
    return normalized


def _smoothness_regularizer(
    points: torch.Tensor,
    visible_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if points.shape[1] < 3:
        return points.sum() * 0.0
    second = points[:, 2:] - 2.0 * points[:, 1:-1] + points[:, :-2]
    triplet_penalty = second.abs().mean(dim=-1)
    if visible_mask is None:
        return triplet_penalty.mean()
    triplet_mask = visible_mask[:, :-2] & visible_mask[:, 1:-1] & visible_mask[:, 2:]
    if not bool(triplet_mask.any()):
        return points.sum() * 0.0
    weights = triplet_mask.to(dtype=triplet_penalty.dtype)
    return (triplet_penalty * weights).sum() / weights.sum().clamp(min=1.0)


def _visibility_total_variation(
    logits: torch.Tensor,
    targets: torch.Tensor | None = None,
) -> torch.Tensor:
    if logits.shape[1] < 2:
        return logits.sum() * 0.0
    probabilities = logits.sigmoid()
    prediction_tv = (probabilities[:, 1:] - probabilities[:, :-1]).abs().mean(dim=-1)
    if targets is None:
        return prediction_tv.mean()
    target_tv = (targets[:, 1:] - targets[:, :-1]).abs().mean(dim=-1)
    return (prediction_tv - target_tv).clamp(min=0.0).mean()


def _masked_binary_ce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    sample_mask: torch.Tensor,
) -> torch.Tensor:
    weights = sample_mask.to(dtype=logits.dtype)
    if not bool(weights.any()):
        return _zero_graph(logits, targets)
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    return (loss * weights).sum() / weights.sum().clamp(min=1.0)


def _masked_focal_bce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    sample_mask: torch.Tensor,
    *,
    alpha: float = 0.75,
    gamma: float = 2.0,
) -> torch.Tensor:
    weights = sample_mask.to(dtype=logits.dtype)
    if not bool(weights.any()):
        return _zero_graph(logits, targets)
    loss = _focal_bce_with_logits(logits, targets, alpha=float(alpha), gamma=float(gamma))
    return (loss * weights).sum() / weights.sum().clamp(min=1.0)


def _masked_binary_ce_balanced(
    logits: torch.Tensor,
    targets: torch.Tensor,
    sample_mask: torch.Tensor,
    *,
    positive_threshold: float = 0.5,
    max_positive_weight: float = 16.0,
) -> torch.Tensor:
    weights = sample_mask.to(dtype=logits.dtype)
    if not bool(weights.any()):
        return _zero_graph(logits, targets)
    positive_mask = (targets > float(positive_threshold)) & sample_mask
    positive_count = positive_mask.to(dtype=logits.dtype).sum()
    total_count = weights.sum()
    if float(positive_count.item()) > 0.0:
        negative_count = (total_count - positive_count).clamp(min=0.0)
        positive_weight = (negative_count / positive_count.clamp(min=1.0)).clamp(min=1.0, max=float(max_positive_weight))
        weights = torch.where(positive_mask, weights * positive_weight, weights)
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    return (loss * weights).sum() / weights.sum().clamp(min=1.0)


def _masked_smooth_l1(
    pred: torch.Tensor,
    target: torch.Tensor,
    positive_mask: torch.Tensor,
) -> torch.Tensor:
    weights = positive_mask.to(dtype=pred.dtype)
    if not bool(weights.any()):
        return _zero_graph(pred, target)
    loss = F.smooth_l1_loss(pred, target, reduction="none")
    return (loss * weights).sum() / weights.sum().clamp(min=1.0)


def _masked_dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    sample_mask: torch.Tensor,
) -> torch.Tensor:
    weights = sample_mask.to(dtype=logits.dtype)
    if not bool(weights.any()):
        return _zero_graph(logits, targets)
    probs = logits.sigmoid() * weights
    masked_targets = targets * weights
    numerator = 2.0 * (probs * masked_targets).sum()
    denominator = probs.sum() + masked_targets.sum()
    return 1.0 - (numerator + 1.0) / (denominator + 1.0)


def _stop_line_row_support_loss(
    center_logits: torch.Tensor,
    center_target: torch.Tensor,
    source_mask: torch.Tensor,
) -> torch.Tensor:
    row_logits = center_logits.amax(dim=-1, keepdim=True)
    row_target = center_target.amax(dim=-1, keepdim=True)
    row_mask = source_mask[:, None, None, None].expand_as(row_logits)
    return _masked_binary_ce_balanced(row_logits, row_target, row_mask, max_positive_weight=16.0)


def _stop_line_row_distribution_loss(
    center_logits: torch.Tensor,
    center_target: torch.Tensor,
    source_mask: torch.Tensor,
) -> torch.Tensor:
    row_logits = center_logits.amax(dim=-1).squeeze(1)
    row_target = center_target.amax(dim=-1).squeeze(1)
    sample_mask = source_mask.to(dtype=torch.bool).reshape(-1)
    if row_logits.ndim != 2 or row_target.ndim != 2 or not bool(sample_mask.any()):
        return _zero_graph(center_logits, center_target)
    valid_logits = row_logits[sample_mask]
    valid_target = row_target[sample_mask]
    positive_rows = valid_target.sum(dim=-1) > 0.0
    if not bool(positive_rows.any()):
        return _zero_graph(center_logits, center_target)
    valid_logits = valid_logits[positive_rows]
    valid_target = valid_target[positive_rows]
    target_distribution = valid_target / valid_target.sum(dim=-1, keepdim=True).clamp(min=1.0e-6)
    log_probs = F.log_softmax(valid_logits, dim=-1)
    return F.kl_div(log_probs, target_distribution, reduction="batchmean", log_target=False)


def _stop_line_row_expectation_loss(
    row_logits: torch.Tensor,
    center_target: torch.Tensor,
    source_mask: torch.Tensor,
) -> torch.Tensor:
    logits_1d = row_logits.amax(dim=-1).squeeze(1)
    target_1d = center_target.amax(dim=-1).squeeze(1)
    sample_mask = source_mask.to(dtype=torch.bool).reshape(-1)
    if logits_1d.ndim != 2 or target_1d.ndim != 2 or not bool(sample_mask.any()):
        return _zero_graph(row_logits, center_target)
    valid_logits = logits_1d[sample_mask]
    valid_target = target_1d[sample_mask]
    positive_rows = valid_target.sum(dim=-1) > 0.0
    if not bool(positive_rows.any()):
        return _zero_graph(row_logits, center_target)
    valid_logits = valid_logits[positive_rows]
    valid_target = valid_target[positive_rows]
    row_positions = torch.arange(valid_logits.shape[-1], device=valid_logits.device, dtype=valid_logits.dtype)
    pred_distribution = valid_logits.softmax(dim=-1)
    pred_expectation = (pred_distribution * row_positions[None, :]).sum(dim=-1)
    target_distribution = valid_target / valid_target.sum(dim=-1, keepdim=True).clamp(min=1.0e-6)
    target_expectation = (target_distribution * row_positions[None, :]).sum(dim=-1)
    return F.smooth_l1_loss(pred_expectation, target_expectation, reduction="mean")


def _stop_line_col_support_loss(
    x_logits: torch.Tensor,
    mask_target: torch.Tensor,
    source_mask: torch.Tensor,
) -> torch.Tensor:
    col_logits = x_logits.amax(dim=-2, keepdim=True)
    col_target = mask_target.amax(dim=-2, keepdim=True)
    col_mask = source_mask[:, None, None, None].expand_as(col_logits)
    return _masked_binary_ce_balanced(col_logits, col_target, col_mask, max_positive_weight=16.0)


def _stop_line_col_distribution_loss(
    x_logits: torch.Tensor,
    mask_target: torch.Tensor,
    source_mask: torch.Tensor,
) -> torch.Tensor:
    logits_1d = x_logits.amax(dim=-2).squeeze(1)
    target_1d = mask_target.amax(dim=-2).squeeze(1)
    sample_mask = source_mask.to(dtype=torch.bool).reshape(-1)
    if logits_1d.ndim != 2 or target_1d.ndim != 2 or not bool(sample_mask.any()):
        return _zero_graph(x_logits, mask_target)
    valid_logits = logits_1d[sample_mask]
    valid_target = target_1d[sample_mask]
    positive_cols = valid_target.sum(dim=-1) > 0.0
    if not bool(positive_cols.any()):
        return _zero_graph(x_logits, mask_target)
    valid_logits = valid_logits[positive_cols]
    valid_target = valid_target[positive_cols]
    target_distribution = valid_target / valid_target.sum(dim=-1, keepdim=True).clamp(min=1.0e-6)
    log_probs = F.log_softmax(valid_logits, dim=-1)
    return F.kl_div(log_probs, target_distribution, reduction="batchmean", log_target=False)


def _stop_line_selector_map_loss(
    row_logits: torch.Tensor,
    x_logits: torch.Tensor,
    mask_target: torch.Tensor,
    source_mask: torch.Tensor,
) -> torch.Tensor:
    row_map = row_logits
    if row_map.ndim == 4 and row_map.shape[-1] == 1:
        row_map = row_map.expand(-1, -1, -1, mask_target.shape[-1])
    x_map = x_logits
    if x_map.ndim == 4 and x_map.shape[-2] == 1:
        x_map = x_map.expand(-1, -1, mask_target.shape[-2], -1)
    if row_map.shape != mask_target.shape or x_map.shape != mask_target.shape:
        return _zero_graph(row_logits, x_logits, mask_target)
    selector_logits = 0.5 * (row_map + x_map)
    sample_mask = source_mask[:, None, None, None].expand_as(mask_target)
    selector_ce = _masked_binary_ce_balanced(selector_logits, mask_target, sample_mask, max_positive_weight=16.0)
    selector_dice = _masked_dice_loss(selector_logits, mask_target, sample_mask)
    return 0.5 * selector_ce + 0.5 * selector_dice


def _stop_line_dense_selector_loss(
    selector_map_logits: torch.Tensor,
    selector_target: torch.Tensor,
    source_mask: torch.Tensor,
) -> torch.Tensor:
    sample_mask = source_mask[:, None, None, None].expand_as(selector_map_logits)
    selector_ce = _masked_binary_ce_balanced(selector_map_logits, selector_target, sample_mask, max_positive_weight=16.0)
    selector_dice = _masked_dice_loss(selector_map_logits, selector_target, sample_mask)
    return 0.5 * selector_ce + 0.5 * selector_dice


def _lane_v2_auxiliary_loss(predictions: dict[str, torch.Tensor], encoded: dict[str, Any]) -> torch.Tensor:
    seed_logits = predictions.get("lane_seed_logits")
    if not isinstance(seed_logits, torch.Tensor):
        return _zero_graph(predictions["lane"])
    seed_offset = predictions.get("lane_seed_offset")
    centerline_logits = predictions.get("lane_centerline_logits")
    aux = encoded.get("roadmark_v2")
    if not isinstance(aux, dict):
        return _zero_graph(seed_logits)
    source = encoded["mask"]["lane_source"].to(device=seed_logits.device, dtype=torch.bool)
    heatmap_target = aux["lane_seed_heatmap"].to(device=seed_logits.device, dtype=seed_logits.dtype)
    offset_target = aux["lane_seed_offset"].to(device=seed_logits.device, dtype=seed_logits.dtype)
    sample_mask = source[:, None, None].expand_as(seed_logits)
    heatmap_loss = _masked_binary_ce(seed_logits, heatmap_target, sample_mask)
    aux_total = 0.5 * heatmap_loss
    positive_mask = (heatmap_target > 0.5) & sample_mask
    if isinstance(seed_offset, torch.Tensor):
        offset_loss = _masked_smooth_l1(seed_offset, offset_target, positive_mask)
        aux_total = aux_total + 0.25 * offset_loss
    if isinstance(centerline_logits, torch.Tensor):
        centerline_target = aux.get("lane_centerline")
        if isinstance(centerline_target, torch.Tensor):
            centerline_target = centerline_target.to(device=centerline_logits.device, dtype=centerline_logits.dtype)
            centerline_mask = source[:, None, None, None].expand_as(centerline_logits)
            centerline_focal = _masked_focal_bce(
                centerline_logits,
                centerline_target,
                centerline_mask,
            )
            centerline_dice = _masked_dice_loss(centerline_logits, centerline_target, centerline_mask)
            # Keep the dense lane anchor modest; this is a spatial stabilizer, not the primary decode objective.
            aux_total = aux_total + 0.20 * centerline_focal + 0.20 * centerline_dice
    return aux_total


def _lane_segfirst_loss(
    predictions: dict[str, torch.Tensor],
    encoded: dict[str, Any],
    *,
    loss_weights: dict[str, float] | None = None,
    color_class_weights: dict[str, float] | None = None,
    breakdown: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    weights = dict(SEG_FIRST_LANE_LOSS_WEIGHTS)
    if loss_weights:
        weights.update({str(name): float(value) for name, value in loss_weights.items()})
    centerline_logits = predictions.get("lane_seg_centerline_logits")
    support_logits = predictions.get("lane_seg_support_logits")
    tangent_axis = predictions.get("lane_seg_tangent_axis")
    color_logits = predictions.get("lane_seg_color_logits")
    type_logits = predictions.get("lane_seg_type_logits")
    if not all(isinstance(value, torch.Tensor) for value in (centerline_logits, support_logits, tangent_axis, color_logits, type_logits)):
        raise KeyError("seg-first lane loss requires centerline/support/tangent/color/type predictions")
    aux = encoded.get("roadmark_v2")
    if not isinstance(aux, dict) or "lane_seg_centerline_core" not in aux:
        raise KeyError("seg-first lane loss requires include_lane_segfirst_targets=True encoded targets")

    source = encoded["mask"]["lane_source"].to(device=centerline_logits.device, dtype=torch.bool)
    centerline_soft = aux["lane_seg_centerline_soft"].to(device=centerline_logits.device, dtype=centerline_logits.dtype)
    support_target = aux["lane_seg_support"].to(device=centerline_logits.device, dtype=centerline_logits.dtype)
    ignore = aux["lane_seg_ignore"].to(device=centerline_logits.device, dtype=torch.bool)
    valid_mask = source[:, None, None, None].expand_as(centerline_logits) & (~ignore)

    centerline_bce = _masked_binary_ce_balanced(
        centerline_logits,
        centerline_soft,
        valid_mask,
        max_positive_weight=32.0,
    )
    centerline_dice = _masked_dice_loss(centerline_logits, centerline_soft, valid_mask)
    support_bce = _masked_binary_ce_balanced(
        support_logits,
        support_target,
        valid_mask.expand_as(support_logits),
        max_positive_weight=16.0,
    )

    tangent_target = aux["lane_seg_tangent_axis"].to(device=tangent_axis.device, dtype=tangent_axis.dtype)
    tangent_count = aux["lane_seg_tangent_count"].to(device=tangent_axis.device, dtype=tangent_axis.dtype)
    tangent_mask = (
        source[:, None, None, None].expand_as(tangent_count).to(dtype=torch.bool)
        & (tangent_count > 0.0)
        & (~ignore.to(device=tangent_axis.device))
    )
    if bool(tangent_mask.any()):
        pred_axis = F.normalize(tangent_axis, dim=1, eps=1.0e-6)
        target_axis = F.normalize(tangent_target, dim=1, eps=1.0e-6)
        axis_dot = (pred_axis * target_axis).sum(dim=1, keepdim=True).abs().clamp(0.0, 1.0)
        tangent_loss = (1.0 - axis_dot)[tangent_mask].mean()
    else:
        tangent_loss = _zero_graph(tangent_axis)

    semantic_mask = tangent_mask.squeeze(1)
    if bool(semantic_mask.any()):
        color_target = aux["lane_seg_color"].to(device=color_logits.device, dtype=color_logits.dtype).argmax(dim=1)
        type_target = aux["lane_seg_type"].to(device=type_logits.device, dtype=type_logits.dtype).argmax(dim=1)
        color_weight_tensor = None
        if color_class_weights:
            color_weight_tensor = color_logits.new_tensor(
                [float(color_class_weights.get(class_name, 1.0)) for class_name in LANE_CLASSES],
                dtype=color_logits.dtype,
            )
        color_loss = F.cross_entropy(
            color_logits.permute(0, 2, 3, 1)[semantic_mask],
            color_target[semantic_mask],
            weight=color_weight_tensor,
            reduction="mean",
        )
        type_loss = F.cross_entropy(type_logits.permute(0, 2, 3, 1)[semantic_mask], type_target[semantic_mask], reduction="mean")
    else:
        color_loss = _zero_graph(color_logits)
        type_loss = _zero_graph(type_logits)

    total = (
        weights["centerline_bce"] * centerline_bce
        + weights["centerline_dice"] * centerline_dice
        + weights["support_bce"] * support_bce
        + weights["tangent"] * tangent_loss
        + weights["color"] * color_loss
        + weights["type"] * type_loss
    )
    if breakdown is not None:
        breakdown.update(
            {
                "seg_centerline_bce": centerline_bce,
                "seg_centerline_dice": centerline_dice,
                "seg_support_bce": support_bce,
                "seg_tangent_axis": tangent_loss,
                "seg_color": color_loss,
                "seg_type": type_loss,
            }
        )
    return total


def _stop_line_v2_auxiliary_loss(predictions: dict[str, torch.Tensor], encoded: dict[str, Any]) -> torch.Tensor:
    center_logits = predictions.get("stop_line_center_logits")
    center_offset = predictions.get("stop_line_center_offset")
    angle = predictions.get("stop_line_angle")
    half_length = predictions.get("stop_line_half_length")
    if not all(isinstance(value, torch.Tensor) for value in (center_logits, center_offset, angle, half_length)):
        return _zero_graph(predictions["stop_line"])
    aux = encoded.get("roadmark_v2")
    if not isinstance(aux, dict):
        return _zero_graph(center_logits)
    source = encoded["mask"]["stop_line_source"].to(device=center_logits.device, dtype=torch.bool)
    center_target = aux["stop_line_center_heatmap"].to(device=center_logits.device, dtype=center_logits.dtype)
    offset_target = aux["stop_line_center_offset"].to(device=center_logits.device, dtype=center_logits.dtype)
    angle_target = aux["stop_line_angle"].to(device=center_logits.device, dtype=center_logits.dtype)
    half_length_target = aux["stop_line_half_length"].to(device=center_logits.device, dtype=center_logits.dtype)
    sample_mask = source[:, None, None, None].expand_as(center_logits)
    center_loss = _masked_binary_ce(center_logits, center_target, sample_mask)
    positive_mask = (center_target > 0.5) & sample_mask
    offset_loss = _masked_smooth_l1(center_offset, offset_target, positive_mask.expand_as(center_offset))
    angle_loss = _masked_smooth_l1(angle, angle_target, positive_mask.expand_as(angle))
    length_loss = _masked_smooth_l1(half_length, half_length_target, positive_mask.expand_as(half_length))
    return 0.5 * center_loss + 0.25 * offset_loss + 0.25 * angle_loss + 0.25 * length_loss


def _crosswalk_v2_auxiliary_loss(predictions: dict[str, torch.Tensor], encoded: dict[str, Any]) -> torch.Tensor:
    mask_logits = predictions.get("crosswalk_mask_logits")
    if not isinstance(mask_logits, torch.Tensor):
        return _zero_graph(predictions["crosswalk"])
    boundary_logits = predictions.get("crosswalk_boundary_logits")
    center_logits = predictions.get("crosswalk_center_logits")
    aux = encoded.get("roadmark_v2")
    if not isinstance(aux, dict):
        return _zero_graph(mask_logits)
    source = encoded["mask"]["crosswalk_source"].to(device=mask_logits.device, dtype=torch.bool)
    mask_target = aux["crosswalk_mask"].to(device=mask_logits.device, dtype=mask_logits.dtype)
    boundary_target = aux.get("crosswalk_boundary")
    center_target = aux.get("crosswalk_center")
    sample_mask = source[:, None, None, None].expand_as(mask_logits)
    mask_ce = _masked_binary_ce(mask_logits, mask_target, sample_mask)
    mask_dice = _masked_dice_loss(mask_logits, mask_target, sample_mask)
    if isinstance(boundary_logits, torch.Tensor) and isinstance(boundary_target, torch.Tensor):
        boundary_target = boundary_target.to(device=mask_logits.device, dtype=mask_logits.dtype)
        boundary_mask = source[:, None, None, None].expand_as(boundary_logits)
        boundary_loss = _masked_binary_ce(boundary_logits, boundary_target, boundary_mask)
    else:
        boundary_loss = _zero_graph(mask_logits)
    if isinstance(center_logits, torch.Tensor) and isinstance(center_target, torch.Tensor):
        center_target = center_target.to(device=mask_logits.device, dtype=mask_logits.dtype)
        center_mask = source[:, None, None, None].expand_as(center_logits)
        center_loss = _masked_binary_ce(center_logits, center_target, center_mask)
    else:
        center_loss = _zero_graph(mask_logits)
    return 0.5 * mask_ce + 0.5 * mask_dice + 0.2 * boundary_loss + 0.2 * center_loss


def _lane_row_cost_matrix(
    *,
    pred_cols: torch.Tensor,
    pred_exist_logits: torch.Tensor,
    pred_obj_logits: torch.Tensor,
    gt_cols: torch.Tensor,
    gt_exists: torch.Tensor,
    col_normalizer: float,
) -> torch.Tensor:
    slot_count = int(pred_cols.shape[0])
    gt_count = int(gt_cols.shape[0])
    cost = pred_cols.new_zeros((slot_count, gt_count))
    pred_exist_prob = pred_exist_logits.sigmoid()
    pred_obj_prob = pred_obj_logits.sigmoid()
    for slot_index in range(slot_count):
        slot_cols = pred_cols[slot_index]
        slot_exist_logits = pred_exist_logits[slot_index]
        slot_exist_prob = pred_exist_prob[slot_index]
        slot_obj_prob = pred_obj_prob[slot_index]
        for gt_index in range(gt_count):
            gt_visible = gt_exists[gt_index] > 0.5
            if not bool(gt_visible.any()):
                cost[slot_index, gt_index] = 1.0e6
                continue
            visible_float = gt_visible.to(dtype=slot_cols.dtype)
            geom_cost = (
                (slot_cols[gt_visible] - gt_cols[gt_index, gt_visible]).abs().mean()
                / max(float(col_normalizer), 1.0)
            )
            exist_target = gt_exists[gt_index]
            exist_bce = F.binary_cross_entropy_with_logits(
                slot_exist_logits,
                exist_target,
                reduction="none",
            )
            exist_cost = exist_bce.mean()
            match_quality = (slot_exist_prob * visible_float).sum() / visible_float.sum().clamp(min=1.0)
            obj_cost = (1.0 - slot_obj_prob) * (0.5 + 0.5 * match_quality)
            bottom_visible_indices = torch.nonzero(gt_visible, as_tuple=False).flatten()
            bottom_index = int(bottom_visible_indices[0].item())
            bottom_cost = (
                (slot_cols[bottom_index] - gt_cols[gt_index, bottom_index]).abs()
                / max(float(col_normalizer), 1.0)
            )
            cost[slot_index, gt_index] = geom_cost + 0.35 * exist_cost + 0.15 * obj_cost + 0.15 * bottom_cost
    return cost


def _dynamic_lane_row_targets(
    predictions: dict[str, torch.Tensor],
    encoded: dict[str, Any],
) -> dict[str, torch.Tensor] | None:
    row_logits = predictions.get("lane_row_logits")
    exist_logits = predictions.get("lane_exist_logits")
    slot_obj_logits = predictions.get("lane_slot_obj_logits")
    row_col_expectation = predictions.get("lane_row_col_expectation")
    if not all(
        isinstance(value, torch.Tensor)
        for value in (row_logits, exist_logits, slot_obj_logits, row_col_expectation)
    ):
        return None
    lane_target = encoded.get("lane")
    mask_payload = encoded.get("mask")
    if not isinstance(lane_target, torch.Tensor) or not isinstance(mask_payload, dict):
        return None
    lane_valid = mask_payload.get("lane_valid")
    lane_source = mask_payload.get("lane_source")
    meta = encoded.get("meta")
    if not isinstance(lane_valid, torch.Tensor) or not isinstance(lane_source, torch.Tensor) or not isinstance(meta, list):
        return None

    batch_size, slot_count, row_count, output_cols = row_logits.shape
    device = row_logits.device
    dtype = row_logits.dtype
    slot_valid = torch.zeros((batch_size, slot_count), dtype=torch.bool, device=device)
    row_exists = torch.zeros((batch_size, slot_count, row_count), dtype=dtype, device=device)
    row_col_index = torch.full((batch_size, slot_count, row_count), -1, dtype=torch.long, device=device)
    row_col_target = torch.full((batch_size, slot_count, row_count), -1.0, dtype=dtype, device=device)
    row_soft_target = torch.zeros((batch_size, slot_count, row_count, output_cols), dtype=dtype, device=device)
    slot_color = torch.zeros((batch_size, slot_count), dtype=torch.long, device=device)
    slot_type = torch.zeros((batch_size, slot_count), dtype=torch.long, device=device)
    col_positions = torch.arange(output_cols, dtype=dtype, device=device)
    soft_sigma = 2.0

    lane_target = lane_target.to(device=device, dtype=dtype)
    lane_valid = lane_valid.to(device=device, dtype=torch.bool)
    lane_source = lane_source.to(device=device, dtype=torch.bool)
    pred_cols = row_col_expectation.to(device=device, dtype=dtype)
    pred_exist_logits = exist_logits.to(device=device, dtype=dtype)
    pred_obj_logits = slot_obj_logits.to(device=device, dtype=dtype)

    for batch_index in range(batch_size):
        if not bool(lane_source[batch_index]):
            continue
        sample_valid = lane_valid[batch_index]
        if not bool(sample_valid.any()):
            continue
        sample_meta = meta[batch_index] if batch_index < len(meta) and isinstance(meta[batch_index], dict) else {}
        network_hw = sample_meta.get("network_hw", ())
        network_w = int(network_hw[1]) if isinstance(network_hw, (list, tuple)) and len(network_hw) >= 2 else None
        if network_w is None or network_w <= 1:
            continue
        gt_rows = lane_target[batch_index, sample_valid]
        gt_exists = gt_rows[:, LANE_VIS_SLICE]
        gt_cols = gt_rows[:, LANE_X_SLICE] * (float(output_cols - 1) / float(network_w - 1))
        cost = _lane_row_cost_matrix(
            pred_cols=pred_cols[batch_index],
            pred_exist_logits=pred_exist_logits[batch_index],
            pred_obj_logits=pred_obj_logits[batch_index],
            gt_cols=gt_cols,
            gt_exists=gt_exists,
            col_normalizer=max(float(output_cols - 1), 1.0),
        )
        if cost.numel() == 0:
            continue
        slot_indices, gt_indices = linear_sum_assignment(cost.detach().cpu().numpy())
        for slot_index, gt_index in zip(slot_indices.tolist(), gt_indices.tolist()):
            gt_row = gt_rows[gt_index]
            gt_visible = gt_exists[gt_index] > 0.5
            if int(gt_visible.sum().item()) < 2:
                continue
            slot_valid[batch_index, slot_index] = True
            row_exists[batch_index, slot_index] = gt_visible.to(dtype=dtype)
            row_col_target[batch_index, slot_index, gt_visible] = gt_cols[gt_index, gt_visible]
            row_col_index[batch_index, slot_index, gt_visible] = gt_cols[gt_index, gt_visible].round().to(dtype=torch.long)
            for row_index in torch.nonzero(gt_visible, as_tuple=False).flatten().tolist():
                target_col = float(gt_cols[gt_index, row_index].item())
                soft = torch.exp(-0.5 * ((col_positions - target_col) / soft_sigma) ** 2)
                row_soft_target[batch_index, slot_index, row_index] = soft / soft.sum().clamp(min=1.0e-6)
            slot_color[batch_index, slot_index] = int(gt_row[LANE_COLOR_SLICE].argmax().item())
            slot_type[batch_index, slot_index] = int(gt_row[LANE_TYPE_SLICE].argmax().item())
    return {
        "lane_slot_valid": slot_valid,
        "lane_row_exists": row_exists,
        "lane_row_col_index": row_col_index,
        "lane_row_col_target": row_col_target,
        "lane_row_soft_target": row_soft_target,
        "lane_slot_color": slot_color,
        "lane_slot_type": slot_type,
    }


def _lane_row_classification_loss(
    predictions: dict[str, torch.Tensor],
    encoded: dict[str, Any],
    *,
    assignment_mode: str = "fixed_slot",
    objectness_target_mode: str = "binary",
    objectness_quality_min: float = 0.25,
    objectness_quality_tau: float = 10.0,
    centerline_focal_weight: float = 0.0,
    centerline_dice_weight: float = 0.0,
    dynamic_coverage_weight: float = 0.0,
    breakdown: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    row_logits = predictions.get("lane_row_logits")
    exist_logits = predictions.get("lane_exist_logits")
    slot_obj_logits = predictions.get("lane_slot_obj_logits")
    color_logits = predictions.get("lane_slot_color_logits")
    type_logits = predictions.get("lane_slot_type_logits")
    if not all(isinstance(value, torch.Tensor) for value in (row_logits, exist_logits, slot_obj_logits, color_logits, type_logits)):
        return _zero_graph(predictions["lane"])
    aux = encoded.get("roadmark_v2")
    if str(assignment_mode) == "dynamic_match":
        aux = _dynamic_lane_row_targets(predictions, encoded)
    if not isinstance(aux, dict):
        return _zero_graph(row_logits)
    source = encoded["mask"]["lane_source"].to(device=row_logits.device, dtype=torch.bool)
    slot_valid = aux["lane_slot_valid"].to(device=row_logits.device, dtype=torch.bool)
    row_exists = aux["lane_row_exists"].to(device=row_logits.device, dtype=row_logits.dtype)
    row_col_index = aux["lane_row_col_index"].to(device=row_logits.device, dtype=torch.long)
    row_col_target = aux["lane_row_col_target"].to(device=row_logits.device, dtype=row_logits.dtype)
    row_soft_target = aux["lane_row_soft_target"].to(device=row_logits.device, dtype=row_logits.dtype)
    slot_color = aux["lane_slot_color"].to(device=row_logits.device, dtype=torch.long)
    slot_type = aux["lane_slot_type"].to(device=row_logits.device, dtype=torch.long)

    exist_mask = source[:, None, None].expand_as(exist_logits)
    exist_loss = _masked_binary_ce(exist_logits, row_exists, exist_mask)

    # Change C: KL-div against Gaussian-smoothed soft targets instead of hard CE.
    row_mask = source[:, None, None] & slot_valid[:, :, None] & (row_col_index >= 0)
    if bool(row_mask.any()):
        log_probs = F.log_softmax(row_logits, dim=-1)  # [B, S, R, C]
        # row_mask is [B, S, R]; expand to [B, S, R, C] for gathering, then use
        # the flat mask for KL-div.
        location_loss = F.kl_div(
            log_probs[row_mask],       # [N_valid, C]
            row_soft_target[row_mask],  # [N_valid, C]
            reduction="batchmean",
            log_target=False,
        )
    else:
        location_loss = _zero_graph(row_logits)

    row_col_expectation = predictions.get("lane_row_col_expectation")
    if isinstance(row_col_expectation, torch.Tensor) and bool(row_mask.any()):
        refine_loss = F.smooth_l1_loss(
            row_col_expectation[row_mask],
            row_col_target[row_mask],
            reduction="mean",
        )
    else:
        refine_loss = _zero_graph(row_logits)

    # Change B: Focal objectness with positive/negative weighting.
    slot_mask = source[:, None] & slot_valid
    negative_slot_mask = source[:, None] & (~slot_valid)
    if bool(slot_mask.any()):
        slot_obj_target = torch.ones_like(slot_obj_logits[slot_mask])
        if str(objectness_target_mode) == "quality_ramp":
            pred_cols = row_col_expectation[slot_mask]
            target_cols = row_col_target[slot_mask]
            target_vis = row_exists[slot_mask] > 0.5
            pred_vis_prob = exist_logits[slot_mask].sigmoid()
            quality_targets = torch.zeros_like(slot_obj_target)
            for index in range(pred_cols.shape[0]):
                visible = target_vis[index]
                if not bool(visible.any()):
                    continue
                mae_cols = (pred_cols[index, visible] - target_cols[index, visible]).abs().mean()
                geom_quality = torch.exp(-mae_cols / max(float(objectness_quality_tau), 1.0e-6))
                gt_vis = row_exists[slot_mask][index]
                vis_prob = pred_vis_prob[index]
                vis_intersection = (vis_prob * gt_vis).sum()
                vis_union = (vis_prob + gt_vis - vis_prob * gt_vis).sum().clamp(min=1.0e-6)
                visibility_quality = vis_intersection / vis_union
                q_raw = geom_quality * visibility_quality
                q_target = ((q_raw - float(objectness_quality_min)) / max(1.0 - float(objectness_quality_min), 1.0e-6)).clamp(0.0, 1.0)
                quality_targets[index] = q_target.detach()
            slot_obj_target = quality_targets
        pos_bce = F.binary_cross_entropy_with_logits(
            slot_obj_logits[slot_mask],
            slot_obj_target,
            reduction="none",
        )
        pos_prob = slot_obj_logits[slot_mask].sigmoid()
        pos_focal = ((1.0 - pos_prob) ** 2.0) * pos_bce
        slot_obj_loss = pos_focal.sum() / max(float(slot_mask.sum().item()), 1.0)
        color_loss = F.cross_entropy(color_logits[slot_mask], slot_color[slot_mask], reduction="mean")
        type_loss = F.cross_entropy(type_logits[slot_mask], slot_type[slot_mask], reduction="mean")
    else:
        slot_obj_loss = _zero_graph(slot_obj_logits)
        color_loss = _zero_graph(color_logits)
        type_loss = _zero_graph(type_logits)
    if bool(negative_slot_mask.any()):
        neg_bce = F.binary_cross_entropy_with_logits(
            slot_obj_logits[negative_slot_mask],
            torch.zeros_like(slot_obj_logits[negative_slot_mask]),
            reduction="none",
        )
        neg_prob = slot_obj_logits[negative_slot_mask].sigmoid()
        neg_focal = (neg_prob ** 2.0) * neg_bce
        neg_weight = 0.25
        slot_obj_loss = slot_obj_loss + neg_weight * neg_focal.sum() / max(float(negative_slot_mask.sum().item()), 1.0)

    centerline_focal = _zero_graph(row_logits)
    centerline_dice = _zero_graph(row_logits)
    centerline_loss = _zero_graph(row_logits)
    centerline_logits = predictions.get("lane_centerline_logits")
    centerline_target = aux.get("lane_centerline")
    if (
        (float(centerline_focal_weight) > 0.0 or float(centerline_dice_weight) > 0.0)
        and isinstance(centerline_logits, torch.Tensor)
        and isinstance(centerline_target, torch.Tensor)
    ):
        centerline_target = centerline_target.to(device=centerline_logits.device, dtype=centerline_logits.dtype)
        centerline_source = source.to(device=centerline_logits.device, dtype=torch.bool)
        centerline_mask = centerline_source[:, None, None, None].expand_as(centerline_logits)
        centerline_focal = _masked_focal_bce(centerline_logits, centerline_target, centerline_mask)
        centerline_dice = _masked_dice_loss(centerline_logits, centerline_target, centerline_mask)
        centerline_loss = (
            float(centerline_focal_weight) * centerline_focal
            + float(centerline_dice_weight) * centerline_dice
        )

    dynamic_coverage_loss = _zero_graph(row_logits)
    if float(dynamic_coverage_weight) > 0.0 and str(assignment_mode) != "dynamic_match":
        dynamic_aux = _dynamic_lane_row_targets(predictions, encoded)
        if isinstance(dynamic_aux, dict):
            dynamic_slot_valid = dynamic_aux["lane_slot_valid"].to(device=row_logits.device, dtype=torch.bool)
            dynamic_row_exists = dynamic_aux["lane_row_exists"].to(device=row_logits.device, dtype=row_logits.dtype)
            dynamic_row_col_index = dynamic_aux["lane_row_col_index"].to(device=row_logits.device, dtype=torch.long)
            dynamic_row_col_target = dynamic_aux["lane_row_col_target"].to(device=row_logits.device, dtype=row_logits.dtype)
            dynamic_row_soft_target = dynamic_aux["lane_row_soft_target"].to(device=row_logits.device, dtype=row_logits.dtype)

            dynamic_slot_mask = source[:, None] & dynamic_slot_valid
            dynamic_exist_mask = source[:, None, None] & dynamic_slot_valid[:, :, None]
            dynamic_row_mask = dynamic_exist_mask & (dynamic_row_col_index >= 0)

            if bool(dynamic_row_mask.any()):
                dynamic_location_loss = F.kl_div(
                    F.log_softmax(row_logits, dim=-1)[dynamic_row_mask],
                    dynamic_row_soft_target[dynamic_row_mask],
                    reduction="batchmean",
                    log_target=False,
                )
                if isinstance(row_col_expectation, torch.Tensor):
                    dynamic_refine_loss = F.smooth_l1_loss(
                        row_col_expectation[dynamic_row_mask],
                        dynamic_row_col_target[dynamic_row_mask],
                        reduction="mean",
                    )
                else:
                    dynamic_refine_loss = _zero_graph(row_logits)
            else:
                dynamic_location_loss = _zero_graph(row_logits)
                dynamic_refine_loss = _zero_graph(row_logits)

            if bool(dynamic_exist_mask.any()):
                dynamic_exist_loss = _masked_binary_ce(exist_logits, dynamic_row_exists, dynamic_exist_mask)
            else:
                dynamic_exist_loss = _zero_graph(row_logits)

            if bool(dynamic_slot_mask.any()):
                dynamic_obj_target = torch.ones_like(slot_obj_logits[dynamic_slot_mask])
                dynamic_obj_bce = F.binary_cross_entropy_with_logits(
                    slot_obj_logits[dynamic_slot_mask],
                    dynamic_obj_target,
                    reduction="none",
                )
                dynamic_obj_prob = slot_obj_logits[dynamic_slot_mask].sigmoid()
                dynamic_obj_loss = (((1.0 - dynamic_obj_prob) ** 2.0) * dynamic_obj_bce).sum() / max(float(dynamic_slot_mask.sum().item()), 1.0)
            else:
                dynamic_obj_loss = _zero_graph(slot_obj_logits)

            dynamic_coverage_loss = (
                0.25 * dynamic_obj_loss
                + dynamic_exist_loss
                + 3.0 * dynamic_location_loss
                + 0.2 * dynamic_refine_loss
            )
            dynamic_coverage_slot_count = dynamic_slot_valid.to(dtype=row_logits.dtype).sum()
            dynamic_coverage_row_count = (dynamic_row_col_index >= 0).to(dtype=row_logits.dtype).sum()
            dynamic_coverage_conflict_count = (
                (dynamic_slot_valid != slot_valid.to(device=row_logits.device, dtype=torch.bool))
                .to(dtype=row_logits.dtype)
                .sum()
            )
        else:
            dynamic_obj_loss = _zero_graph(slot_obj_logits)
            dynamic_exist_loss = _zero_graph(row_logits)
            dynamic_location_loss = _zero_graph(row_logits)
            dynamic_refine_loss = _zero_graph(row_logits)
            dynamic_coverage_slot_count = _zero_graph(row_logits)
            dynamic_coverage_row_count = _zero_graph(row_logits)
            dynamic_coverage_conflict_count = _zero_graph(row_logits)
    else:
        dynamic_obj_loss = _zero_graph(slot_obj_logits)
        dynamic_exist_loss = _zero_graph(row_logits)
        dynamic_location_loss = _zero_graph(row_logits)
        dynamic_refine_loss = _zero_graph(row_logits)
        dynamic_coverage_slot_count = _zero_graph(row_logits)
        dynamic_coverage_row_count = _zero_graph(row_logits)
        dynamic_coverage_conflict_count = _zero_graph(row_logits)

    total = (
        slot_obj_loss
        + exist_loss
        + 3.0 * location_loss
        + 0.2 * refine_loss
        + color_loss
        + 0.5 * type_loss
        + centerline_loss
        + float(dynamic_coverage_weight) * dynamic_coverage_loss
    )
    if breakdown is not None:
        breakdown.update(
            {
                "slot_obj_loss": slot_obj_loss.detach(),
                "exist_loss": exist_loss.detach(),
                "location_loss": location_loss.detach(),
                "refine_loss": refine_loss.detach(),
                "color_loss": color_loss.detach(),
                "type_loss": type_loss.detach(),
                "centerline_focal_loss": centerline_focal.detach(),
                "centerline_dice_loss": centerline_dice.detach(),
                "centerline_loss": centerline_loss.detach(),
                "dynamic_coverage_obj_loss": dynamic_obj_loss.detach(),
                "dynamic_coverage_exist_loss": dynamic_exist_loss.detach(),
                "dynamic_coverage_location_loss": dynamic_location_loss.detach(),
                "dynamic_coverage_refine_loss": dynamic_refine_loss.detach(),
                "dynamic_coverage_loss": dynamic_coverage_loss.detach(),
                "dynamic_coverage_weighted_loss": (float(dynamic_coverage_weight) * dynamic_coverage_loss).detach(),
                "dynamic_coverage_slot_count": dynamic_coverage_slot_count.detach(),
                "dynamic_coverage_row_count": dynamic_coverage_row_count.detach(),
                "dynamic_coverage_conflict_count": dynamic_coverage_conflict_count.detach(),
                "total": total.detach(),
            }
        )
    return total


def _stop_line_mask_loss(predictions: dict[str, torch.Tensor], encoded: dict[str, Any]) -> torch.Tensor:
    return _stop_line_mask_loss_with_selector_weight(
        predictions,
        encoded,
        local_x_aux_weight=0.0,
        selector_aux_weight=1.0,
        geometry_aux_weight=1.0,
        center_target_mode="union",
        centerline_target_weight=1.0,
    )


def _stop_line_mask_loss_with_selector_weight(
    predictions: dict[str, torch.Tensor],
    encoded: dict[str, Any],
    *,
    local_x_aux_weight: float = 0.0,
    selector_aux_weight: float,
    geometry_aux_weight: float = 1.0,
    center_target_mode: str = "union",
    centerline_target_weight: float = 1.0,
) -> torch.Tensor:
    mask_logits = predictions.get("stop_line_mask_logits")
    center_logits = predictions.get("stop_line_center_logits")
    center_offset = predictions.get("stop_line_center_offset")
    angle = predictions.get("stop_line_angle")
    half_length = predictions.get("stop_line_half_length")
    if not isinstance(mask_logits, torch.Tensor):
        return _zero_graph(predictions["stop_line"])
    aux = encoded.get("roadmark_v2")
    if not isinstance(aux, dict):
        return _zero_graph(mask_logits)
    source = encoded["mask"]["stop_line_source"].to(device=mask_logits.device, dtype=torch.bool)
    mask_target = aux["stop_line_mask"].to(device=mask_logits.device, dtype=mask_logits.dtype)
    center_heatmap_target = aux["stop_line_center_heatmap"].to(device=mask_logits.device, dtype=mask_logits.dtype)
    centerline_target = aux.get("stop_line_centerline")
    if isinstance(centerline_target, torch.Tensor):
        centerline_target = centerline_target.to(device=mask_logits.device, dtype=mask_logits.dtype)
    resolved_center_target_mode = str(center_target_mode or "union").strip().lower()
    weighted_centerline = None
    local_centerline_target = None
    if isinstance(centerline_target, torch.Tensor):
        local_centerline_support = F.max_pool2d(center_heatmap_target, kernel_size=5, stride=1, padding=2)
        local_centerline_target = centerline_target * local_centerline_support
        weighted_centerline = centerline_target * float(centerline_target_weight)
    if resolved_center_target_mode == "mask":
        center_target = mask_target
    elif resolved_center_target_mode == "heatmap":
        center_target = center_heatmap_target
    elif resolved_center_target_mode == "weighted_union" and isinstance(weighted_centerline, torch.Tensor):
        weighted_local_centerline = weighted_centerline
        if isinstance(local_centerline_target, torch.Tensor):
            weighted_local_centerline = local_centerline_target * float(centerline_target_weight)
        center_target = torch.maximum(center_heatmap_target, weighted_local_centerline)
    elif resolved_center_target_mode == "local_union" and isinstance(local_centerline_target, torch.Tensor):
        center_target = torch.maximum(center_heatmap_target, local_centerline_target)
    elif resolved_center_target_mode == "centerline" and isinstance(centerline_target, torch.Tensor):
        center_target = centerline_target
    elif isinstance(centerline_target, torch.Tensor):
        center_target = torch.maximum(center_heatmap_target, centerline_target)
    else:
        center_target = center_heatmap_target
    sample_mask = source[:, None, None, None].expand_as(mask_logits)
    mask_ce = _masked_binary_ce_balanced(mask_logits, mask_target, sample_mask, max_positive_weight=16.0)
    mask_dice = _masked_dice_loss(mask_logits, mask_target, sample_mask)
    row_selector_logits = predictions.get("stop_line_row_logits")
    if not isinstance(row_selector_logits, torch.Tensor):
        row_selector_logits = center_logits
    x_selector_logits = predictions.get("stop_line_x_logits")
    if not isinstance(x_selector_logits, torch.Tensor):
        x_selector_logits = center_logits
    selector_map_logits = predictions.get("stop_line_selector_map_logits")
    row_target = mask_target
    selector_target = centerline_target if isinstance(centerline_target, torch.Tensor) else mask_target
    local_x_target = local_centerline_target if isinstance(local_centerline_target, torch.Tensor) else centerline_target
    if isinstance(center_logits, torch.Tensor):
        center_mask = source[:, None, None, None].expand_as(center_logits)
        center_loss = _masked_binary_ce_balanced(center_logits, center_target, center_mask, max_positive_weight=64.0)
        row_support_loss = _stop_line_row_support_loss(row_selector_logits, row_target, source)
        row_distribution_loss = _stop_line_row_distribution_loss(row_selector_logits, row_target, source)
        row_expectation_loss = _stop_line_row_expectation_loss(row_selector_logits, row_target, source)
        col_support_loss = _stop_line_col_support_loss(x_selector_logits, mask_target, source)
        col_distribution_loss = _stop_line_col_distribution_loss(x_selector_logits, mask_target, source)
        if isinstance(local_x_target, torch.Tensor):
            local_col_support_loss = _stop_line_col_support_loss(x_selector_logits, local_x_target, source)
            local_col_distribution_loss = _stop_line_col_distribution_loss(x_selector_logits, local_x_target, source)
        else:
            local_col_support_loss = _zero_graph(mask_logits)
            local_col_distribution_loss = _zero_graph(mask_logits)
        selector_map_loss = _stop_line_selector_map_loss(row_selector_logits, x_selector_logits, mask_target, source)
        if isinstance(selector_map_logits, torch.Tensor):
            dense_selector_loss = _stop_line_dense_selector_loss(selector_map_logits, selector_target, source)
        else:
            dense_selector_loss = _zero_graph(mask_logits)
    else:
        center_loss = _zero_graph(mask_logits)
        row_support_loss = _zero_graph(mask_logits)
        row_distribution_loss = _zero_graph(mask_logits)
        row_expectation_loss = _zero_graph(mask_logits)
        col_support_loss = _zero_graph(mask_logits)
        col_distribution_loss = _zero_graph(mask_logits)
        local_col_support_loss = _zero_graph(mask_logits)
        local_col_distribution_loss = _zero_graph(mask_logits)
        selector_map_loss = _zero_graph(mask_logits)
        dense_selector_loss = _zero_graph(mask_logits)
    positive_mask = (center_heatmap_target > 0.5) & sample_mask
    if (
        isinstance(center_offset, torch.Tensor)
        and isinstance(angle, torch.Tensor)
        and isinstance(half_length, torch.Tensor)
    ):
        offset_target = aux["stop_line_center_offset"].to(device=mask_logits.device, dtype=mask_logits.dtype)
        angle_target = aux["stop_line_angle"].to(device=mask_logits.device, dtype=mask_logits.dtype)
        half_length_target = aux["stop_line_half_length"].to(device=mask_logits.device, dtype=mask_logits.dtype)
        offset_loss = _masked_smooth_l1(center_offset, offset_target, positive_mask.expand_as(center_offset))
        angle_loss = _masked_smooth_l1(angle, angle_target, positive_mask.expand_as(angle))
        length_loss = _masked_smooth_l1(half_length, half_length_target, positive_mask.expand_as(half_length))
    else:
        offset_loss = _zero_graph(mask_logits)
        angle_loss = _zero_graph(mask_logits)
        length_loss = _zero_graph(mask_logits)
    return (
        0.5 * mask_ce
        + 0.5 * mask_dice
        + 2.0 * center_loss
        + float(local_x_aux_weight) * (0.5 * local_col_support_loss + 0.5 * local_col_distribution_loss)
        + float(selector_aux_weight)
        * (
            0.5 * row_support_loss
            + 0.75 * row_distribution_loss
            + 0.5 * row_expectation_loss
            + 0.25 * col_support_loss
            + 0.25 * col_distribution_loss
            + 0.5 * selector_map_loss
            + 0.5 * dense_selector_loss
        )
        + 0.5 * offset_loss
        + float(geometry_aux_weight) * angle_loss
        + float(geometry_aux_weight) * length_loss
    )


def _polygon_shape_loss(pred_points: torch.Tensor, target_points: torch.Tensor) -> torch.Tensor:
    pred_edges = torch.linalg.norm(torch.roll(pred_points, shifts=-1, dims=1) - pred_points, dim=-1)
    target_edges = torch.linalg.norm(torch.roll(target_points, shifts=-1, dims=1) - target_points, dim=-1)
    pred_ratios = pred_edges / pred_edges.sum(dim=1, keepdim=True).clamp(min=1.0e-6)
    target_ratios = target_edges / target_edges.sum(dim=1, keepdim=True).clamp(min=1.0e-6)
    return F.smooth_l1_loss(pred_ratios, target_ratios, reduction="mean")


def _polygon_signed_area(points: torch.Tensor) -> torch.Tensor:
    shifted = torch.roll(points, shifts=-1, dims=1)
    cross = points[..., 0] * shifted[..., 1] - points[..., 1] * shifted[..., 0]
    return 0.5 * cross.sum(dim=1)


def _polygon_area_loss(pred_points: torch.Tensor, target_points: torch.Tensor) -> torch.Tensor:
    pred_area = _polygon_signed_area(pred_points).abs()
    target_area = _polygon_signed_area(target_points).abs()
    return F.smooth_l1_loss(pred_area, target_area, reduction="mean")


def _cyclic_contour_loss(pred_points: torch.Tensor, target_points: torch.Tensor) -> torch.Tensor:
    if pred_points.shape[1] == 0:
        return _zero_graph(pred_points, target_points)
    candidate_losses: list[torch.Tensor] = []
    reversed_target = torch.flip(target_points, dims=(1,))
    for shift in range(target_points.shape[1]):
        candidate_losses.append(
            F.smooth_l1_loss(
                pred_points,
                torch.roll(target_points, shifts=shift, dims=1),
                reduction="none",
            ).mean(dim=(1, 2))
        )
        candidate_losses.append(
            F.smooth_l1_loss(
                pred_points,
                torch.roll(reversed_target, shifts=shift, dims=1),
                reduction="none",
            ).mean(dim=(1, 2))
        )
    return torch.stack(candidate_losses, dim=1).min(dim=1).values.mean()


def _soft_convex_polygon_occupancy(
    points: torch.Tensor,
    sample_points: torch.Tensor,
    *,
    sharpness: float = 12.0,
) -> torch.Tensor:
    next_points = torch.roll(points, shifts=-1, dims=1)
    edges = next_points - points
    edge_norm = torch.linalg.norm(edges, dim=-1).clamp(min=1.0e-6)
    relative = sample_points[:, None, :, :] - points[:, :, None, :]
    cross = edges[:, :, None, 0] * relative[..., 1] - edges[:, :, None, 1] * relative[..., 0]
    orientation = torch.where(
        _polygon_signed_area(points) >= 0.0,
        torch.ones(points.shape[0], device=points.device, dtype=points.dtype),
        -torch.ones(points.shape[0], device=points.device, dtype=points.dtype),
    )
    signed_distance = (cross * orientation[:, None, None]) / edge_norm[:, :, None]
    return torch.sigmoid(sharpness * signed_distance.min(dim=1).values)


def _soft_polygon_iou_loss(
    pred_points: torch.Tensor,
    target_points: torch.Tensor,
    *,
    grid_size: int = 12,
    sharpness: float = 12.0,
) -> torch.Tensor:
    if pred_points.shape[0] == 0:
        return _zero_graph(pred_points, target_points)
    all_points = torch.cat((pred_points, target_points), dim=1)
    min_xy = all_points.amin(dim=1)
    max_xy = all_points.amax(dim=1)
    padding = (max_xy - min_xy).clamp(min=4.0) * 0.1 + 1.0
    min_xy = min_xy - padding
    max_xy = max_xy + padding
    axis = torch.linspace(0.0, 1.0, grid_size, device=pred_points.device, dtype=pred_points.dtype)
    xs = min_xy[:, 0:1] * (1.0 - axis[None, :]) + max_xy[:, 0:1] * axis[None, :]
    ys = min_xy[:, 1:2] * (1.0 - axis[None, :]) + max_xy[:, 1:2] * axis[None, :]
    grid_x = xs[:, None, :].expand(-1, grid_size, -1)
    grid_y = ys[:, :, None].expand(-1, -1, grid_size)
    samples = torch.stack((grid_x, grid_y), dim=-1).reshape(pred_points.shape[0], grid_size * grid_size, 2)
    pred_occupancy = _soft_convex_polygon_occupancy(pred_points, samples, sharpness=sharpness)
    target_occupancy = _soft_convex_polygon_occupancy(target_points, samples, sharpness=sharpness)
    intersection = torch.minimum(pred_occupancy, target_occupancy).sum(dim=1)
    union = torch.maximum(pred_occupancy, target_occupancy).sum(dim=1).clamp(min=1.0e-6)
    return (1.0 - intersection / union).mean()


def _polygon_iou(points_a: torch.Tensor, points_b: torch.Tensor) -> float:
    polygon_a = points_a.detach().cpu().numpy().astype(np.float32).reshape(-1, 2)
    polygon_b = points_b.detach().cpu().numpy().astype(np.float32).reshape(-1, 2)
    if polygon_a.shape[0] < 3 or polygon_b.shape[0] < 3:
        return 0.0
    if not bool(np.isfinite(polygon_a).all()) or not bool(np.isfinite(polygon_b).all()):
        return 0.0
    min_x = int(np.floor(min(float(polygon_a[:, 0].min()), float(polygon_b[:, 0].min()))))
    min_y = int(np.floor(min(float(polygon_a[:, 1].min()), float(polygon_b[:, 1].min()))))
    max_x = int(np.ceil(max(float(polygon_a[:, 0].max()), float(polygon_b[:, 0].max()))))
    max_y = int(np.ceil(max(float(polygon_a[:, 1].max()), float(polygon_b[:, 1].max()))))
    width = max(1, max_x - min_x + 3)
    height = max(1, max_y - min_y + 3)

    def rasterize(points: np.ndarray) -> np.ndarray:
        canvas = Image.new("1", (width, height), 0)
        shifted = [(float(x - min_x + 1.0), float(y - min_y + 1.0)) for x, y in points]
        ImageDraw.Draw(canvas).polygon(shifted, outline=1, fill=1)
        return np.asarray(canvas, dtype=bool)

    mask_a = rasterize(polygon_a)
    mask_b = rasterize(polygon_b)
    intersection = float(np.logical_and(mask_a, mask_b).sum())
    union = float(np.logical_or(mask_a, mask_b).sum())
    if union <= 0.0:
        return 0.0
    return intersection / union


def _bbox_iou(pred_boxes: torch.Tensor, target_boxes: torch.Tensor, *, ciou: bool = False) -> torch.Tensor:
    if ultralytics_bbox_iou is not None:
        return ultralytics_bbox_iou(pred_boxes, target_boxes, xywh=False, CIoU=ciou)

    top_left = torch.maximum(pred_boxes[..., :2], target_boxes[..., :2])
    bottom_right = torch.minimum(pred_boxes[..., 2:], target_boxes[..., 2:])
    wh = (bottom_right - top_left).clamp(min=0.0)
    inter = wh[..., 0] * wh[..., 1]

    pred_wh = (pred_boxes[..., 2:] - pred_boxes[..., :2]).clamp(min=0.0)
    target_wh = (target_boxes[..., 2:] - target_boxes[..., :2]).clamp(min=0.0)
    pred_area = pred_wh[..., 0] * pred_wh[..., 1]
    target_area = target_wh[..., 0] * target_wh[..., 1]
    union = pred_area + target_area - inter
    iou = inter / union.clamp(min=1e-6)
    if not ciou:
        return iou.unsqueeze(-1)

    pred_center = (pred_boxes[..., :2] + pred_boxes[..., 2:]) / 2.0
    target_center = (target_boxes[..., :2] + target_boxes[..., 2:]) / 2.0
    center_dist = ((pred_center - target_center) ** 2).sum(dim=-1)

    enclosure_top_left = torch.minimum(pred_boxes[..., :2], target_boxes[..., :2])
    enclosure_bottom_right = torch.maximum(pred_boxes[..., 2:], target_boxes[..., 2:])
    enclosure_wh = (enclosure_bottom_right - enclosure_top_left).clamp(min=0.0)
    enclosure_diag = (enclosure_wh[..., 0] ** 2 + enclosure_wh[..., 1] ** 2).clamp(min=1e-6)

    pred_w = pred_wh[..., 0].clamp(min=1e-6)
    pred_h = pred_wh[..., 1].clamp(min=1e-6)
    target_w = target_wh[..., 0].clamp(min=1e-6)
    target_h = target_wh[..., 1].clamp(min=1e-6)
    v = (4.0 / torch.pi**2) * (torch.atan(target_w / target_h) - torch.atan(pred_w / pred_h)) ** 2
    alpha = v / (1.0 - iou + v).clamp(min=1e-6)
    ciou_value = iou - (center_dist / enclosure_diag) - alpha * v
    return ciou_value.unsqueeze(-1)


def _batched_box_from_points(points: torch.Tensor) -> torch.Tensor:
    x = points[..., 0]
    y = points[..., 1]
    return torch.stack((x.min(dim=-1).values, y.min(dim=-1).values, x.max(dim=-1).values, y.max(dim=-1).values), dim=-1)


def _batched_angle_length_cost(pred_points: torch.Tensor, target_points: torch.Tensor) -> torch.Tensor:
    pred_vec = pred_points[..., -1, :] - pred_points[..., 0, :]
    target_vec = target_points[..., -1, :] - target_points[..., 0, :]
    pred_len = torch.linalg.norm(pred_vec, dim=-1)
    target_len = torch.linalg.norm(target_vec, dim=-1)
    length_cost = (pred_len - target_len).abs() / target_len.clamp(min=1.0)
    cosine = (pred_vec * target_vec).sum(dim=-1) / (pred_len * target_len).clamp(min=1e-6)
    angle_cost = 1.0 - cosine.clamp(min=-1.0, max=1.0)
    return length_cost + angle_cost


def _focal_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    alpha: float = 0.75,
    gamma: float = 2.0,
) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probabilities = logits.sigmoid()
    pt = probabilities * targets + (1.0 - probabilities) * (1.0 - targets)
    alpha_factor = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    modulating = (1.0 - pt).pow(gamma)
    return bce * alpha_factor * modulating


def _quality_to_objectness_target(quality: torch.Tensor, *, floor: float = 0.35) -> torch.Tensor:
    bounded = quality.clamp(min=0.0, max=1.0)
    return floor + (1.0 - floor) * bounded


def _objectness_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    *,
    positive_weight: float = 4.0,
) -> torch.Tensor:
    loss = _focal_bce_with_logits(logits, targets)
    weights = torch.where(
        targets > 0.0,
        torch.full_like(targets, positive_weight),
        torch.ones_like(targets),
    )
    masked_weights = weights * mask.to(dtype=logits.dtype)
    return (loss * masked_weights).sum() / masked_weights.sum().clamp(min=1.0)


def _hungarian_match(cost_matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if cost_matrix.numel() == 0 or cost_matrix.shape[0] == 0 or cost_matrix.shape[1] == 0:
        empty = torch.empty(0, dtype=torch.long, device=cost_matrix.device)
        return empty, empty
    row_ind, col_ind = linear_sum_assignment(cost_matrix.detach().cpu().numpy())
    return (
        torch.as_tensor(row_ind, dtype=torch.long, device=cost_matrix.device),
        torch.as_tensor(col_ind, dtype=torch.long, device=cost_matrix.device),
    )


def _sanitize_hungarian_cost_matrix(cost_matrix: torch.Tensor) -> tuple[torch.Tensor, str]:
    finite_mask = torch.isfinite(cost_matrix)
    if bool(finite_mask.all()):
        return cost_matrix, "clean"
    if not bool(finite_mask.any()):
        return cost_matrix, "all_invalid"

    finite_values = cost_matrix[finite_mask]
    replacement = float((finite_values.max() + finite_values.abs().max() + 1.0).item())
    sanitized = torch.where(
        finite_mask,
        cost_matrix,
        torch.full_like(cost_matrix, replacement),
    )
    return sanitized, "sanitized"


def _lane_cost_matrix(pred_rows: torch.Tensor, gt_rows: torch.Tensor) -> torch.Tensor:
    pred_x = pred_rows[:, LANE_X_SLICE]
    gt_x = gt_rows[:, LANE_X_SLICE]
    gt_visibility = gt_rows[:, LANE_VIS_SLICE] > 0.5
    x_diff = (pred_x[:, None] - gt_x[None]).abs()
    x_mask = gt_visibility[None].to(dtype=pred_rows.dtype)
    points_cost = (x_diff * x_mask).sum(dim=-1) / x_mask.sum(dim=-1).clamp(min=1.0)

    color_targets = gt_rows[:, LANE_COLOR_SLICE].argmax(dim=-1)
    type_targets = gt_rows[:, LANE_TYPE_SLICE].argmax(dim=-1)
    color_cost = -F.log_softmax(pred_rows[:, LANE_COLOR_SLICE], dim=-1)[:, color_targets]
    type_cost = -F.log_softmax(pred_rows[:, LANE_TYPE_SLICE], dim=-1)[:, type_targets]
    vis_cost = F.binary_cross_entropy_with_logits(
        pred_rows[:, None, LANE_VIS_SLICE].expand(-1, gt_rows.shape[0], -1),
        gt_rows[None, :, LANE_VIS_SLICE].expand(pred_rows.shape[0], -1, -1),
        reduction="none",
    ).mean(dim=-1)
    return 3.0 * points_cost + 1.0 * color_cost + 0.5 * type_cost + 0.5 * vis_cost


def _stop_line_cost_matrix(pred_rows: torch.Tensor, gt_rows: torch.Tensor) -> torch.Tensor:
    pred_points = pred_rows[:, 1:].view(pred_rows.shape[0], STOP_LINE_POINT_COUNT, 2)
    gt_points = gt_rows[:, 1:].view(gt_rows.shape[0], STOP_LINE_POINT_COUNT, 2)
    points_cost = (pred_points[:, None] - gt_points[None]).abs().mean(dim=(2, 3))
    angle_length_cost = _batched_angle_length_cost(pred_points[:, None], gt_points[None])
    return 4.0 * points_cost + 0.5 * angle_length_cost


def _crosswalk_cost_matrix(pred_rows: torch.Tensor, gt_rows: torch.Tensor) -> torch.Tensor:
    pred_points = pred_rows[:, 1:].view(pred_rows.shape[0], CROSSWALK_POINT_COUNT, 2)
    gt_points = gt_rows[:, 1:].view(gt_rows.shape[0], CROSSWALK_POINT_COUNT, 2)
    points_cost = (pred_points[:, None] - gt_points[None]).abs().mean(dim=(2, 3))
    overlap = torch.zeros(
        (pred_rows.shape[0], gt_rows.shape[0]),
        device=pred_rows.device,
        dtype=pred_rows.dtype,
    )
    for pred_index in range(pred_rows.shape[0]):
        for gt_index in range(gt_rows.shape[0]):
            overlap[pred_index, gt_index] = float(_polygon_iou(pred_points[pred_index], gt_points[gt_index]))
    return 3.0 * points_cost + (1.0 - overlap)


def _lane_match_quality(pred_rows: torch.Tensor, target_rows: torch.Tensor) -> torch.Tensor:
    target_visibility = target_rows[:, LANE_VIS_SLICE] > 0.5
    point_error = (pred_rows[:, LANE_X_SLICE] - target_rows[:, LANE_X_SLICE]).abs()
    point_quality = torch.exp(
        -(point_error * target_visibility.to(dtype=pred_rows.dtype)).sum(dim=-1)
        / target_visibility.to(dtype=pred_rows.dtype).sum(dim=-1).clamp(min=1.0)
        / 20.0
    )
    visibility_quality = torch.exp(
        -(pred_rows[:, LANE_VIS_SLICE].sigmoid() - target_rows[:, LANE_VIS_SLICE]).abs().mean(dim=-1)
    )
    return point_quality * visibility_quality


def _stop_line_match_quality(pred_rows: torch.Tensor, target_rows: torch.Tensor) -> torch.Tensor:
    pred_points = pred_rows[:, 1:].view(pred_rows.shape[0], STOP_LINE_POINT_COUNT, 2)
    target_points = target_rows[:, 1:].view(target_rows.shape[0], STOP_LINE_POINT_COUNT, 2)
    distance = torch.linalg.norm(pred_points - target_points, dim=-1).mean(dim=-1)
    pred_vec = pred_points[:, -1] - pred_points[:, 0]
    target_vec = target_points[:, -1] - target_points[:, 0]
    pred_norm = torch.linalg.norm(pred_vec, dim=-1).clamp(min=1.0e-6)
    target_norm = torch.linalg.norm(target_vec, dim=-1).clamp(min=1.0e-6)
    direction = ((pred_vec * target_vec).sum(dim=-1) / (pred_norm * target_norm)).clamp(min=-1.0, max=1.0)
    direction_quality = (direction + 1.0) * 0.5
    return torch.exp(-distance / 20.0) * direction_quality


def _crosswalk_match_quality(pred_rows: torch.Tensor, target_rows: torch.Tensor) -> torch.Tensor:
    quality = torch.zeros(pred_rows.shape[0], device=pred_rows.device, dtype=pred_rows.dtype)
    pred_points = pred_rows[:, 1:].view(pred_rows.shape[0], CROSSWALK_POINT_COUNT, 2)
    target_points = target_rows[:, 1:].view(target_rows.shape[0], CROSSWALK_POINT_COUNT, 2)
    for index in range(pred_rows.shape[0]):
        quality[index] = float(_polygon_iou(pred_points[index], target_points[index]))
    return quality


class PV26MultiTaskLoss(nn.Module):
    def __init__(
        self,
        stage: str = "stage_1_frozen_trunk_warmup",
        *,
        det_cls_negative_weight: float = 0.1,
        loss_weights: dict[str, float] | None = None,
        task_mode: str = LANE_FAMILY_TASK_MODE,
        lane_assignment_mode: str = "fixed_slot",
        lane_objectness_target_mode: str = "binary",
        lane_objectness_quality_min: float = 0.25,
        lane_objectness_quality_tau: float = 10.0,
        lane_centerline_focal_weight: float = 0.0,
        lane_centerline_dice_weight: float = 0.0,
        lane_dynamic_coverage_weight: float = 0.0,
        lane_segfirst_loss_weights: dict[str, float] | None = None,
        lane_segfirst_color_class_weights: dict[str, float] | None = None,
        stopline_local_x_aux_weight: float = 0.0,
        stopline_selector_aux_weight: float = 1.0,
        stopline_geometry_aux_weight: float = 1.0,
        stopline_center_target_mode: str = "union",
        stopline_centerline_target_weight: float = 1.0,
        distill_enabled: bool = False,
        distill_teacher_mode: str = "cache",
        distill_loss_weights: dict[str, float] | None = None,
        distill_normalize_mode: str = "none",
        distill_ema_decay: float = 0.95,
        distill_ema_warmup_steps: int = 4,
        distill_ema_eps: float = 1.0e-6,
    ) -> None:
        super().__init__()
        stage = _canonical_stage(stage)
        try:
            self.loss_weights = dict(STAGE_LOSS_WEIGHTS[stage])
        except KeyError as exc:
            raise KeyError(f"unsupported PV26 loss stage: {stage}") from exc
        if loss_weights:
            self.loss_weights.update({str(name): float(value) for name, value in loss_weights.items()})
        self.task_mode = canonicalize_task_mode(task_mode)
        self.loss_weights = filter_loss_weights_for_task_mode(self.loss_weights, self.task_mode)
        self.stage = stage
        self.det_cls_negative_weight = float(det_cls_negative_weight)
        self.lane_assignment_mode = str(lane_assignment_mode)
        self.lane_objectness_target_mode = str(lane_objectness_target_mode)
        self.lane_objectness_quality_min = float(lane_objectness_quality_min)
        self.lane_objectness_quality_tau = float(lane_objectness_quality_tau)
        self.lane_centerline_focal_weight = float(lane_centerline_focal_weight)
        self.lane_centerline_dice_weight = float(lane_centerline_dice_weight)
        self.lane_dynamic_coverage_weight = float(lane_dynamic_coverage_weight)
        self.lane_segfirst_loss_weights = dict(SEG_FIRST_LANE_LOSS_WEIGHTS)
        if lane_segfirst_loss_weights:
            self.lane_segfirst_loss_weights.update({str(name): float(value) for name, value in lane_segfirst_loss_weights.items()})
        self.lane_segfirst_color_class_weights = (
            {str(name): float(value) for name, value in lane_segfirst_color_class_weights.items()}
            if lane_segfirst_color_class_weights
            else {}
        )
        self.stopline_local_x_aux_weight = float(stopline_local_x_aux_weight)
        self.stopline_selector_aux_weight = float(stopline_selector_aux_weight)
        self.stopline_geometry_aux_weight = float(stopline_geometry_aux_weight)
        self.stopline_center_target_mode = str(stopline_center_target_mode)
        self.stopline_centerline_target_weight = float(stopline_centerline_target_weight)
        self.distill_enabled = bool(distill_enabled)
        self.distill_teacher_mode = str(distill_teacher_mode)
        self.distill_normalize_mode = str(distill_normalize_mode)
        self.distill_ema_decay = float(distill_ema_decay)
        self.distill_ema_warmup_steps = int(distill_ema_warmup_steps)
        self.distill_ema_eps = float(distill_ema_eps)
        self.distill_loss_weights = {
            "lane": 1.0,
            "stop_line": 1.0,
            "crosswalk": 1.0,
        }
        if distill_loss_weights:
            self.distill_loss_weights.update({str(name): float(value) for name, value in distill_loss_weights.items()})
        self._distill_ema_state = {
            "lane": None,
            "stop_line": None,
            "crosswalk": None,
        }
        self._distill_ema_steps = {
            "lane": 0,
            "stop_line": 0,
            "crosswalk": 0,
        }
        self.register_buffer(
            "tl_bit_weights",
            torch.tensor(
                [1.0, 2.5, 1.0, 1.8],
                dtype=torch.float32,
            ),
            persistent=False,
        )
        self.assigner = None
        if TaskAlignedAssigner is not None:
            self.assigner = TaskAlignedAssigner(
                topk=10,
                num_classes=len(OD_CLASSES),
                alpha=0.5,
                beta=6.0,
                stride=list(DEFAULT_DET_FEATURE_STRIDES),
            )
        self.last_det_assignment_mode = "uninitialized"
        self.last_det_positive_count = 0
        self.last_lane_assignment_modes = {
            "lane": "uninitialized",
            "stop_line": "uninitialized",
            "crosswalk": "uninitialized",
        }
        self.last_det_loss_breakdown = {
            "det_obj_loss": 0.0,
            "det_cls_matched_loss": 0.0,
            "det_cls_unmatched_neg_loss": 0.0,
            "det_iou_loss": 0.0,
            "det_l1_loss": 0.0,
            "det_cls_matched_count": 0,
            "det_cls_unmatched_neg_count": 0,
        }
        self.last_lane_loss_breakdown: dict[str, float] = {}
        self.last_teacher_agreement: dict[str, dict[str, float | None]] = {}
        self.last_distill_breakdown: dict[str, dict[str, float | None]] = {}

    def export_config(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "det_cls_negative_weight": float(self.det_cls_negative_weight),
            "task_mode": self.task_mode,
            "lane_assignment_mode": self.lane_assignment_mode,
            "lane_objectness_target_mode": self.lane_objectness_target_mode,
            "lane_objectness_quality_min": float(self.lane_objectness_quality_min),
            "lane_objectness_quality_tau": float(self.lane_objectness_quality_tau),
            "lane_centerline_focal_weight": float(self.lane_centerline_focal_weight),
            "lane_centerline_dice_weight": float(self.lane_centerline_dice_weight),
            "lane_dynamic_coverage_weight": float(self.lane_dynamic_coverage_weight),
            "lane_segfirst_loss_weights": dict(self.lane_segfirst_loss_weights),
            "lane_segfirst_color_class_weights": dict(self.lane_segfirst_color_class_weights),
            "stopline_local_x_aux_weight": float(self.stopline_local_x_aux_weight),
            "stopline_selector_aux_weight": float(self.stopline_selector_aux_weight),
            "stopline_geometry_aux_weight": float(self.stopline_geometry_aux_weight),
            "stopline_center_target_mode": self.stopline_center_target_mode,
            "stopline_centerline_target_weight": float(self.stopline_centerline_target_weight),
            "loss_weights": dict(self.loss_weights),
            "distill_enabled": bool(self.distill_enabled),
            "distill_teacher_mode": self.distill_teacher_mode,
            "distill_normalize_mode": self.distill_normalize_mode,
            "distill_ema_decay": float(self.distill_ema_decay),
            "distill_ema_warmup_steps": int(self.distill_ema_warmup_steps),
            "distill_ema_eps": float(self.distill_ema_eps),
            "distill_loss_weights": dict(self.distill_loss_weights),
        }

    def _detector_losses_enabled(self) -> bool:
        return bool(
            float(self.loss_weights.get("det", 0.0)) > 0.0
            or float(self.loss_weights.get("tl_attr", 0.0)) > 0.0
        )

    def _task_loss_enabled(self, task_name: str) -> bool:
        return bool(float(self.loss_weights.get(task_name, 0.0)) > 0.0)

    def _teacher_cache(self, encoded: dict[str, Any]) -> dict[str, torch.Tensor]:
        teacher_cache = encoded.get("teacher_cache")
        return teacher_cache if isinstance(teacher_cache, dict) else {}

    def _normalize_distill_loss(
        self,
        task_name: str,
        raw_loss: torch.Tensor,
        encoded: dict[str, Any],
    ) -> tuple[torch.Tensor, dict[str, float | None]]:
        raw_value = float(raw_loss.detach().cpu())
        scale = 1.0
        ema_mean = self._distill_ema_state.get(task_name)
        phase = str(encoded.get("_distill_phase", "train"))
        if self.distill_normalize_mode == "ema":
            if phase == "train":
                previous = self._distill_ema_state.get(task_name)
                if previous is None:
                    updated = raw_value
                else:
                    updated = float(self.distill_ema_decay) * float(previous) + (1.0 - float(self.distill_ema_decay)) * raw_value
                self._distill_ema_state[task_name] = updated
                self._distill_ema_steps[task_name] = int(self._distill_ema_steps.get(task_name, 0)) + 1
                ema_mean = updated
            warmup_steps = int(self.distill_ema_warmup_steps)
            if ema_mean is not None and int(self._distill_ema_steps.get(task_name, 0)) > warmup_steps:
                scale = 1.0 / max(float(ema_mean), float(self.distill_ema_eps))
        scaled_loss = raw_loss * float(scale)
        return scaled_loss, {
            "pre_scale_loss": raw_value,
            "post_scale_loss": float(scaled_loss.detach().cpu()),
            "scale": float(scale),
            "ema_mean": None if ema_mean is None else float(ema_mean),
        }

    def _lane_distill_loss(
        self,
        predictions: dict[str, torch.Tensor],
        encoded: dict[str, Any],
    ) -> tuple[torch.Tensor, dict[str, float | None]]:
        teacher_cache = self._teacher_cache(encoded)
        required = ("lane_row_logits", "lane_exist_logits", "lane_row_col_expectation", "lane_feature")
        if not self.distill_enabled or any(key not in teacher_cache for key in required):
            return _zero_graph(predictions["lane_row_logits"]), {"logit_kl": None, "feature_cosine": None}
        row_kl = _logit_kl_divergence(predictions["lane_row_logits"], teacher_cache["lane_row_logits"])
        exist_bce = _binary_logit_distill(predictions["lane_exist_logits"], teacher_cache["lane_exist_logits"])
        row_expectation = F.smooth_l1_loss(
            predictions["lane_row_col_expectation"],
            teacher_cache["lane_row_col_expectation"],
            reduction="mean",
        )
        feature_cosine = _feature_cosine_similarity(predictions["lane_feature"], teacher_cache["lane_feature"])
        raw_loss = row_kl + exist_bce + row_expectation
        scaled_loss, scaling = self._normalize_distill_loss("lane", raw_loss, encoded)
        return scaled_loss, {
            "logit_kl": float(row_kl.detach().cpu()),
            "feature_cosine": float(feature_cosine.detach().cpu()),
            **scaling,
        }

    def _stop_line_distill_loss(
        self,
        predictions: dict[str, torch.Tensor],
        encoded: dict[str, Any],
    ) -> tuple[torch.Tensor, dict[str, float | None]]:
        teacher_cache = self._teacher_cache(encoded)
        required = (
            "stop_line_mask_logits",
            "stop_line_center_logits",
            "stop_line_center_offset",
            "stop_line_angle",
            "stop_line_half_length",
            "stop_line_feature",
        )
        if not self.distill_enabled or any(key not in teacher_cache for key in required):
            return _zero_graph(predictions["stop_line_mask_logits"]), {"logit_kl": None, "feature_cosine": None}
        mask_bce = _binary_logit_distill(predictions["stop_line_mask_logits"], teacher_cache["stop_line_mask_logits"])
        mask_dice = _soft_dice_distill(predictions["stop_line_mask_logits"], teacher_cache["stop_line_mask_logits"])
        center_bce = _binary_logit_distill(predictions["stop_line_center_logits"], teacher_cache["stop_line_center_logits"])
        center_offset = F.smooth_l1_loss(
            predictions["stop_line_center_offset"],
            teacher_cache["stop_line_center_offset"],
            reduction="mean",
        )
        angle = F.smooth_l1_loss(predictions["stop_line_angle"], teacher_cache["stop_line_angle"], reduction="mean")
        half_length = F.smooth_l1_loss(
            predictions["stop_line_half_length"],
            teacher_cache["stop_line_half_length"],
            reduction="mean",
        )
        feature_cosine = _feature_cosine_similarity(predictions["stop_line_feature"], teacher_cache["stop_line_feature"])
        raw_loss = mask_bce + mask_dice + center_bce + center_offset + angle + half_length
        scaled_loss, scaling = self._normalize_distill_loss("stop_line", raw_loss, encoded)
        return scaled_loss, {
            "logit_kl": float((mask_bce + center_bce).detach().cpu()),
            "feature_cosine": float(feature_cosine.detach().cpu()),
            **scaling,
        }

    def _crosswalk_distill_loss(
        self,
        predictions: dict[str, torch.Tensor],
        encoded: dict[str, Any],
    ) -> tuple[torch.Tensor, dict[str, float | None]]:
        teacher_cache = self._teacher_cache(encoded)
        required = (
            "crosswalk_mask_logits",
            "crosswalk_boundary_logits",
            "crosswalk_center_logits",
            "crosswalk_feature",
        )
        if not self.distill_enabled or any(key not in teacher_cache for key in required):
            return _zero_graph(predictions["crosswalk_mask_logits"]), {"logit_kl": None, "feature_cosine": None}
        mask_bce = _binary_logit_distill(predictions["crosswalk_mask_logits"], teacher_cache["crosswalk_mask_logits"])
        mask_dice = _soft_dice_distill(predictions["crosswalk_mask_logits"], teacher_cache["crosswalk_mask_logits"])
        boundary_bce = _binary_logit_distill(
            predictions["crosswalk_boundary_logits"],
            teacher_cache["crosswalk_boundary_logits"],
        )
        center_bce = _binary_logit_distill(
            predictions["crosswalk_center_logits"],
            teacher_cache["crosswalk_center_logits"],
        )
        feature_cosine = _feature_cosine_similarity(predictions["crosswalk_feature"], teacher_cache["crosswalk_feature"])
        raw_loss = mask_bce + mask_dice + boundary_bce + center_bce
        scaled_loss, scaling = self._normalize_distill_loss("crosswalk", raw_loss, encoded)
        return scaled_loss, {
            "logit_kl": float((mask_bce + boundary_bce + center_bce).detach().cpu()),
            "feature_cosine": float(feature_cosine.detach().cpu()),
            **scaling,
        }

    def _run_task_aligned_assigner(
        self,
        pred_scores: torch.Tensor,
        pred_bboxes: torch.Tensor,
        anchor_points: torch.Tensor,
        gt_labels: torch.Tensor,
        gt_bboxes: torch.Tensor,
        mask_gt: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.assigner is None:
            raise RuntimeError("task_aligned_assigner_missing")

        original_topk = getattr(self.assigner, "topk", None)
        original_topk2 = getattr(self.assigner, "topk2", None)
        if original_topk is None:
            return self.assigner(pred_scores, pred_bboxes, anchor_points, gt_labels, gt_bboxes, mask_gt)

        effective_topk = max(1, min(int(original_topk), int(pred_scores.shape[1])))
        if effective_topk != int(original_topk):
            self.assigner.topk = effective_topk
        if original_topk2 is not None and effective_topk != int(original_topk2):
            self.assigner.topk2 = effective_topk
        try:
            return self.assigner(pred_scores, pred_bboxes, anchor_points, gt_labels, gt_bboxes, mask_gt)
        finally:
            if effective_topk != int(original_topk):
                self.assigner.topk = original_topk
            if original_topk2 is not None and effective_topk != int(original_topk2):
                self.assigner.topk2 = original_topk2

    def _det_contract_label(self, encoded: dict[str, Any], batch_index: int) -> str:
        meta = encoded.get("meta", [])
        if isinstance(meta, list) and batch_index < len(meta) and isinstance(meta[batch_index], dict):
            dataset_key = str(meta[batch_index].get("dataset_key") or "unknown_dataset")
            sample_id = str(meta[batch_index].get("sample_id") or f"batch_{batch_index}")
            return f"dataset={dataset_key} sample_id={sample_id}"
        return f"batch_index={batch_index}"

    def _validate_det_supervision_contract(
        self,
        encoded: dict[str, Any],
        *,
        device: torch.device,
        batch_size: int,
        num_classes: int,
    ) -> tuple[torch.BoolTensor, torch.BoolTensor, torch.BoolTensor, torch.BoolTensor]:
        mask_payload = encoded.get("mask")
        if not isinstance(mask_payload, dict):
            raise ValueError("encoded det supervision contract violation: mask payload must be a dict")

        det_source = mask_payload.get("det_source")
        class_mask = mask_payload.get("det_supervised_class_mask")
        allow_objectness = mask_payload.get("det_allow_objectness_negatives")
        allow_unmatched_class = mask_payload.get("det_allow_unmatched_class_negatives")
        missing = [
            name
            for name, value in (
                ("det_source", det_source),
                ("det_supervised_class_mask", class_mask),
                ("det_allow_objectness_negatives", allow_objectness),
                ("det_allow_unmatched_class_negatives", allow_unmatched_class),
            )
            if value is None
        ]
        if missing:
            raise ValueError(
                "encoded det supervision contract violation: missing mask fields "
                + ", ".join(missing)
            )

        det_source = det_source.to(device=device, dtype=torch.bool)
        class_mask = class_mask.to(device=device, dtype=torch.bool)
        allow_objectness = allow_objectness.to(device=device, dtype=torch.bool)
        allow_unmatched_class = allow_unmatched_class.to(device=device, dtype=torch.bool)
        expected_shapes = {
            "det_source": (batch_size,),
            "det_supervised_class_mask": (batch_size, num_classes),
            "det_allow_objectness_negatives": (batch_size,),
            "det_allow_unmatched_class_negatives": (batch_size,),
        }
        actual_shapes = {
            "det_source": tuple(det_source.shape),
            "det_supervised_class_mask": tuple(class_mask.shape),
            "det_allow_objectness_negatives": tuple(allow_objectness.shape),
            "det_allow_unmatched_class_negatives": tuple(allow_unmatched_class.shape),
        }
        for name, expected in expected_shapes.items():
            if actual_shapes[name] != expected:
                raise ValueError(
                    f"encoded det supervision contract violation: {name} shape {actual_shapes[name]} != {expected}"
                )
        invalid_rows = torch.nonzero(det_source & ~class_mask.any(dim=1), as_tuple=False).flatten()
        if invalid_rows.numel() > 0:
            label = self._det_contract_label(encoded, int(invalid_rows[0].item()))
            raise ValueError(
                f"encoded det supervision contract violation for {label}: "
                "det_source rows require at least one supervised detector class"
            )
        return det_source, class_mask, allow_objectness, allow_unmatched_class

    def forward(self, predictions: dict[str, torch.Tensor], encoded: dict[str, Any]) -> dict[str, torch.Tensor]:
        predictions = _loss_precision_predictions(predictions)
        reference_tensor = _prediction_reference_tensor(predictions)
        self.last_det_loss_breakdown = {
            "det_obj_loss": 0.0,
            "det_cls_matched_loss": 0.0,
            "det_cls_unmatched_neg_loss": 0.0,
            "det_iou_loss": 0.0,
            "det_l1_loss": 0.0,
            "det_cls_matched_count": 0,
            "det_cls_unmatched_neg_count": 0,
        }
        det_assignment: dict[str, torch.Tensor | str] | None = None
        self.last_lane_assignment_modes = {
            "lane": "disabled" if not self._task_loss_enabled("lane") else "uninitialized",
            "stop_line": "disabled" if not self._task_loss_enabled("stop_line") else "uninitialized",
            "crosswalk": "disabled" if not self._task_loss_enabled("crosswalk") else "uninitialized",
        }
        self.last_lane_loss_breakdown = {}
        self.last_teacher_agreement = {
            "lane": {"logit_kl": None, "feature_cosine": None},
            "stop_line": {"logit_kl": None, "feature_cosine": None},
            "crosswalk": {"logit_kl": None, "feature_cosine": None},
        }
        self.last_distill_breakdown = {
            "lane": {"loss": None},
            "stop_line": {"loss": None},
            "crosswalk": {"loss": None},
        }
        if self._detector_losses_enabled():
            det_assignment = self._build_det_assignment(predictions, encoded)
            self.last_det_assignment_mode = str(det_assignment["mode"])
            self.last_det_positive_count = int(det_assignment["fg_mask"].sum().item())
            det = self._det_loss(predictions["det"], encoded, det_assignment)
            tl_attr = self._tl_attr_loss(predictions["tl_attr"], encoded, det_assignment)
        else:
            self.last_det_assignment_mode = "disabled"
            self.last_det_positive_count = 0
            det = _zero_graph(reference_tensor)
            tl_attr = _zero_graph(reference_tensor)
        lane = self._lane_loss(predictions, encoded) if self._task_loss_enabled("lane") else _zero_graph(reference_tensor)
        stop_line = (
            self._stop_line_loss(predictions, encoded)
            if self._task_loss_enabled("stop_line")
            else _zero_graph(reference_tensor)
        )
        crosswalk = (
            self._crosswalk_loss(predictions, encoded)
            if self._task_loss_enabled("crosswalk")
            else _zero_graph(reference_tensor)
        )
        if self.distill_enabled:
            lane_distill, lane_agreement = self._lane_distill_loss(predictions, encoded)
            stop_line_distill, stop_line_agreement = self._stop_line_distill_loss(predictions, encoded)
            crosswalk_distill, crosswalk_agreement = self._crosswalk_distill_loss(predictions, encoded)
            lane = lane + self.distill_loss_weights["lane"] * lane_distill
            stop_line = stop_line + self.distill_loss_weights["stop_line"] * stop_line_distill
            crosswalk = crosswalk + self.distill_loss_weights["crosswalk"] * crosswalk_distill
            self.last_teacher_agreement = {
                "lane": lane_agreement,
                "stop_line": stop_line_agreement,
                "crosswalk": crosswalk_agreement,
            }
            self.last_distill_breakdown = {
                "lane": {
                    "loss": float(lane_distill.detach().cpu()),
                    **lane_agreement,
                },
                "stop_line": {
                    "loss": float(stop_line_distill.detach().cpu()),
                    **stop_line_agreement,
                },
                "crosswalk": {
                    "loss": float(crosswalk_distill.detach().cpu()),
                    **crosswalk_agreement,
                },
            }

        total = (
            self.loss_weights["det"] * det
            + self.loss_weights["tl_attr"] * tl_attr
            + self.loss_weights["lane"] * lane
            + self.loss_weights["stop_line"] * stop_line
            + self.loss_weights["crosswalk"] * crosswalk
        )
        return {
            "total": total,
            "det": det,
            "tl_attr": tl_attr,
            "lane": lane,
            "stop_line": stop_line,
            "crosswalk": crosswalk,
        }

    def _build_det_assignment(
        self,
        predictions: dict[str, torch.Tensor],
        encoded: dict[str, Any],
    ) -> dict[str, torch.Tensor | str]:
        det_gt = encoded["det_gt"]
        det_pred = predictions["det"]
        det_valid = det_gt["valid_mask"].to(device=det_pred.device, dtype=torch.bool)
        obj_logits = det_pred[..., 4]
        cls_logits = det_pred[..., 5:]
        batch_size, query_count, _ = det_pred.shape
        num_classes = cls_logits.shape[-1]
        det_source, _, _, _ = self._validate_det_supervision_contract(
            encoded,
            device=det_pred.device,
            batch_size=batch_size,
            num_classes=num_classes,
        )
        positive_required = det_source & det_valid.any(dim=1)

        target_bboxes = torch.zeros((batch_size, query_count, 4), device=det_pred.device, dtype=torch.float32)
        target_scores = torch.zeros((batch_size, query_count, num_classes), device=det_pred.device, dtype=torch.float32)
        target_gt_idx = torch.full((batch_size, query_count), -1, device=det_pred.device, dtype=torch.long)
        fg_mask = torch.zeros((batch_size, query_count), device=det_pred.device, dtype=torch.bool)
        obj_target = torch.zeros((batch_size, query_count), device=det_pred.device, dtype=torch.float32)

        def _empty_assignment(mode: str) -> dict[str, torch.Tensor | str]:
            return {
                "mode": mode,
                "pred_boxes": det_pred[..., :4],
                "obj_logits": obj_logits,
                "cls_logits": cls_logits,
                "target_bboxes": target_bboxes,
                "target_scores": target_scores,
                "target_gt_idx": target_gt_idx,
                "fg_mask": fg_mask,
                "obj_target": obj_target,
                "target_score_sum": target_scores.sum().clamp(min=1.0),
                "anchor_points": torch.zeros((query_count, 2), device=det_pred.device, dtype=det_pred.dtype),
                "stride_tensor": torch.ones((query_count, 1), device=det_pred.device, dtype=det_pred.dtype),
                "det_source": det_source,
            }

        if not bool(positive_required.any()):
            return _empty_assignment("zero_positive")

        feature_shapes = predictions.get("det_feature_shapes")
        feature_strides = predictions.get("det_feature_strides")
        feature_meta_valid = (
            isinstance(feature_shapes, list)
            and isinstance(feature_strides, list)
            and len(feature_shapes) == len(feature_strides)
            and sum(int(height) * int(width) for height, width in feature_shapes) == query_count
        )
        if self.assigner is not None and feature_meta_valid:
            anchor_points, stride_tensor = make_anchor_grid(
                [(int(height), int(width)) for height, width in feature_shapes],
                [int(value) for value in feature_strides],
                dtype=det_pred.dtype,
                device=det_pred.device,
            )
            pred_boxes = decode_anchor_relative_boxes(det_pred[..., :4], anchor_points, stride_tensor)
            gt_labels = det_gt["classes"].to(device=det_pred.device, dtype=torch.long).clamp(min=0).unsqueeze(-1)
            gt_bboxes = det_gt["boxes_xyxy"].to(device=det_pred.device, dtype=torch.float32)
            mask_gt = det_valid.unsqueeze(-1) & det_source[:, None, None]
            # Ultralytics task-aligned assigner is not consistently AMP-safe.
            # Keep the training graph in the model dtype, but run assigner inputs in float32.
            _, assigned_bboxes, assigned_scores, assigned_fg, assigned_gt_idx = self._run_task_aligned_assigner(
                cls_logits.detach().to(dtype=torch.float32).sigmoid(),
                pred_boxes.detach().to(dtype=torch.float32),
                anchor_points.to(dtype=torch.float32),
                gt_labels,
                gt_bboxes,
                mask_gt,
            )
            assigned_fg = assigned_fg.to(dtype=torch.bool)
            assigned_scores = assigned_scores * det_source[:, None, None].to(dtype=det_pred.dtype)
            assigned_fg = assigned_fg & det_source[:, None]
            assigned_gt_idx = assigned_gt_idx.to(dtype=torch.long)

            target_bboxes = assigned_bboxes.to(dtype=torch.float32)
            target_scores = assigned_scores.to(dtype=torch.float32)
            fg_mask = assigned_fg
            target_gt_idx = torch.where(
                fg_mask,
                assigned_gt_idx,
                torch.full_like(assigned_gt_idx, -1),
            )
            obj_target = target_scores.max(dim=-1).values
            return {
                "mode": "task_aligned",
                "pred_boxes": pred_boxes,
                "obj_logits": obj_logits,
                "cls_logits": cls_logits,
                "target_bboxes": target_bboxes,
                "target_scores": target_scores,
                "target_gt_idx": target_gt_idx,
                "fg_mask": fg_mask,
                "obj_target": obj_target,
                "target_score_sum": target_scores.sum().clamp(min=1.0),
                "anchor_points": anchor_points,
                "stride_tensor": stride_tensor,
                "det_source": det_source,
            }

        if self.assigner is None:
            raise RuntimeError("task_aligned_assigner_unavailable")
        if not feature_meta_valid:
            raise ValueError("det_feature_metadata_invalid")
        raise RuntimeError("task_aligned_assigner_unavailable")

    def _det_supervision_masks(
        self,
        encoded: dict[str, Any],
        *,
        device: torch.device,
        batch_size: int,
        num_classes: int,
    ) -> tuple[torch.BoolTensor, torch.BoolTensor, torch.BoolTensor]:
        _, class_mask, allow_objectness, allow_unmatched_class = self._validate_det_supervision_contract(
            encoded,
            device=device,
            batch_size=batch_size,
            num_classes=num_classes,
        )
        return class_mask, allow_objectness, allow_unmatched_class

    def _det_loss(
        self,
        det_pred: torch.Tensor,
        encoded: dict[str, Any],
        assignment: dict[str, torch.Tensor | str],
    ) -> torch.Tensor:
        obj_logits = assignment["obj_logits"]
        cls_logits = assignment["cls_logits"]
        pred_boxes = assignment["pred_boxes"]
        target_bboxes = assignment["target_bboxes"]
        target_scores = assignment["target_scores"]
        fg_mask = assignment["fg_mask"]
        obj_target = assignment["obj_target"]
        det_source = assignment["det_source"]
        det_class_mask, det_allow_objectness, det_allow_unmatched_class = self._det_supervision_masks(
            encoded,
            device=det_pred.device,
            batch_size=det_pred.shape[0],
            num_classes=cls_logits.shape[-1],
        )

        obj_mask = torch.where(
            det_allow_objectness[:, None],
            det_source[:, None].expand_as(obj_logits),
            fg_mask & det_source[:, None],
        )
        obj_loss = F.binary_cross_entropy_with_logits(obj_logits, obj_target, reduction="none")
        obj_loss = _objectness_loss(obj_logits, obj_target, obj_mask)

        cls_bce = F.binary_cross_entropy_with_logits(cls_logits, target_scores, reduction="none")
        cls_pos_mask = fg_mask[:, :, None] & det_class_mask[:, None, :]
        cls_neg_mask = (
            (~fg_mask)[:, :, None]
            & det_source[:, None, None]
            & det_allow_unmatched_class[:, None, None]
            & det_class_mask[:, None, :]
        )
        cls_pos_loss = (cls_bce * cls_pos_mask.to(dtype=det_pred.dtype)).sum() / cls_pos_mask.sum().clamp(min=1)
        cls_neg_loss = (cls_bce * cls_neg_mask.to(dtype=det_pred.dtype)).sum() / cls_neg_mask.sum().clamp(min=1)
        cls_loss = cls_pos_loss + self.det_cls_negative_weight * cls_neg_loss

        iou_loss = _zero_graph(det_pred)
        l1_loss = _zero_graph(det_pred)
        if bool(fg_mask.any()):
            target_score_sum = target_scores.sum().clamp(min=1.0)
            weights = target_scores.sum(dim=-1)[fg_mask]
            iou = _bbox_iou(pred_boxes[fg_mask], target_bboxes[fg_mask], ciou=True).squeeze(-1)
            iou_loss = ((1.0 - iou) * weights).sum() / target_score_sum
            l1_loss = (
                F.l1_loss(pred_boxes[fg_mask], target_bboxes[fg_mask], reduction="none").mean(dim=-1) * weights
            ).sum() / target_score_sum

        self.last_det_loss_breakdown = {
            "det_obj_loss": float(obj_loss.detach().cpu()),
            "det_cls_matched_loss": float(cls_pos_loss.detach().cpu()),
            "det_cls_unmatched_neg_loss": float(cls_neg_loss.detach().cpu()),
            "det_iou_loss": float(iou_loss.detach().cpu()),
            "det_l1_loss": float(l1_loss.detach().cpu()),
            "det_cls_matched_count": int(cls_pos_mask.sum().item()),
            "det_cls_unmatched_neg_count": int(cls_neg_mask.sum().item()),
        }
        return obj_loss + cls_loss + iou_loss + l1_loss

    def _tl_attr_loss(
        self,
        tl_pred: torch.Tensor,
        encoded: dict[str, Any],
        assignment: dict[str, torch.Tensor | str],
    ) -> torch.Tensor:
        tl_source = encoded["mask"]["tl_attr_source"].to(device=tl_pred.device, dtype=torch.bool)
        tl_bits = encoded["tl_attr_gt_bits"].to(device=tl_pred.device, dtype=torch.float32)
        tl_mask = encoded["tl_attr_gt_mask"].to(device=tl_pred.device, dtype=torch.bool)
        det_classes = encoded["det_gt"]["classes"].to(device=tl_pred.device, dtype=torch.long)
        fg_mask = assignment["fg_mask"]
        target_gt_idx = assignment["target_gt_idx"]
        per_sample: list[torch.Tensor] = []

        for batch_index in range(tl_pred.shape[0]):
            if not bool(tl_source[batch_index]):
                continue
            assigned = fg_mask[batch_index] & (target_gt_idx[batch_index] >= 0)
            if not bool(assigned.any()):
                continue
            matched_gt = target_gt_idx[batch_index, assigned]
            valid = tl_mask[batch_index, matched_gt] & det_classes[batch_index, matched_gt].eq(TL_CLASS_ID)
            if not bool(valid.any()):
                continue
            logits = tl_pred[batch_index, assigned][valid]
            targets = tl_bits[batch_index, matched_gt][valid]
            loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
            weighted = loss * self.tl_bit_weights.to(device=tl_pred.device)
            per_sample.append(weighted.mean())

        return _mean_or_zero(per_sample, tl_pred)

    def _lane_loss(self, predictions: dict[str, torch.Tensor] | torch.Tensor, encoded: dict[str, Any]) -> torch.Tensor:
        prediction_dict = predictions if isinstance(predictions, dict) else {"lane": predictions}
        if "lane_seg_centerline_logits" in prediction_dict:
            self.last_lane_assignment_modes["lane"] = "seg_first_dense"
            lane_breakdown: dict[str, torch.Tensor] = {}
            lane_loss = _lane_segfirst_loss(
                prediction_dict,
                encoded,
                loss_weights=self.lane_segfirst_loss_weights,
                color_class_weights=self.lane_segfirst_color_class_weights,
                breakdown=lane_breakdown,
            )
            self.last_lane_loss_breakdown = {
                name: float(value.detach().cpu())
                for name, value in lane_breakdown.items()
                if isinstance(value, torch.Tensor)
            }
            return lane_loss
        if self.task_mode == ROADMARK_JOINT_TASK_MODE and "lane_row_logits" not in prediction_dict:
            raise KeyError("roadmark_joint requires lane_row_logits for native lane loss dispatch")
        if self.task_mode in {LANE_ONLY_TASK_MODE, ROADMARK_JOINT_TASK_MODE} and "lane_row_logits" in prediction_dict:
            assignment_mode = self.lane_assignment_mode
            if assignment_mode == "dynamic_match_phase2":
                assignment_mode = "dynamic_match" if _canonical_stage(self.stage) != "stage_4_lane_family_finetune" else "fixed_slot"
            self.last_lane_assignment_modes["lane"] = f"row_classification_{assignment_mode}"
            lane_breakdown: dict[str, torch.Tensor] = {}
            lane_loss = _lane_row_classification_loss(
                prediction_dict,
                encoded,
                assignment_mode=assignment_mode,
                objectness_target_mode=self.lane_objectness_target_mode,
                objectness_quality_min=self.lane_objectness_quality_min,
                objectness_quality_tau=self.lane_objectness_quality_tau,
                centerline_focal_weight=self.lane_centerline_focal_weight,
                centerline_dice_weight=self.lane_centerline_dice_weight,
                dynamic_coverage_weight=self.lane_dynamic_coverage_weight,
                breakdown=lane_breakdown,
            )
            self.last_lane_loss_breakdown = {
                name: float(value.detach().cpu())
                for name, value in lane_breakdown.items()
                if isinstance(value, torch.Tensor)
            }
            return lane_loss
        lane_pred = prediction_dict["lane"]
        lane_target = encoded["lane"].to(device=lane_pred.device, dtype=torch.float32)
        lane_source = encoded["mask"]["lane_source"].to(device=lane_pred.device, dtype=torch.bool)
        lane_valid = encoded["mask"]["lane_valid"].to(device=lane_pred.device, dtype=torch.bool)
        assignment = self._build_query_assignment(
            lane_pred,
            lane_target,
            lane_valid,
            lane_source,
            task_name="lane",
            cost_builder=_lane_cost_matrix,
            quality_builder=_lane_match_quality,
        )

        source_mask = lane_source[:, None].expand_as(lane_pred[..., 0])
        obj_loss = _objectness_loss(lane_pred[..., 0], assignment["obj_target"], source_mask)
        aux_loss = _lane_v2_auxiliary_loss(prediction_dict, encoded)

        valid = assignment["fg_mask"]
        if not bool(valid.any()):
            return obj_loss + aux_loss

        assigned_target = assignment["assigned_target"]
        color_target = assigned_target[..., LANE_COLOR_SLICE].argmax(dim=-1)
        type_target = assigned_target[..., LANE_TYPE_SLICE].argmax(dim=-1)
        color_loss = F.cross_entropy(lane_pred[..., LANE_COLOR_SLICE][valid], color_target[valid], reduction="mean")
        type_loss = F.cross_entropy(lane_pred[..., LANE_TYPE_SLICE][valid], type_target[valid], reduction="mean")
        pred_x = lane_pred[..., LANE_X_SLICE][valid]
        target_x = assigned_target[..., LANE_X_SLICE][valid]
        target_vis = assigned_target[..., LANE_VIS_SLICE][valid]
        x_mask = target_vis > 0.5
        if bool(x_mask.any()):
            x_loss_matrix = F.smooth_l1_loss(pred_x, target_x, reduction="none")
            points_loss = (x_loss_matrix * x_mask.to(dtype=lane_pred.dtype)).sum() / x_mask.to(dtype=lane_pred.dtype).sum().clamp(min=1.0)
        else:
            points_loss = _zero_graph(pred_x, target_x)
        vis_loss = F.binary_cross_entropy_with_logits(
            lane_pred[..., LANE_VIS_SLICE][valid],
            target_vis,
            reduction="mean",
        )
        smoothness = _smoothness_regularizer(pred_x.unsqueeze(-1), x_mask)
        visibility_tv = _visibility_total_variation(
            lane_pred[..., LANE_VIS_SLICE][valid],
            target_vis,
        )
        return (
            obj_loss
            + color_loss
            + 0.5 * type_loss
            + 5.0 * points_loss
            + vis_loss
            + 0.25 * smoothness
            + 0.1 * visibility_tv
            + aux_loss
        )

    def _stop_line_loss(self, predictions: dict[str, torch.Tensor] | torch.Tensor, encoded: dict[str, Any]) -> torch.Tensor:
        prediction_dict = predictions if isinstance(predictions, dict) else {"stop_line": predictions}
        if self.task_mode == ROADMARK_JOINT_TASK_MODE and "stop_line_mask_logits" not in prediction_dict:
            raise KeyError("roadmark_joint requires stop_line_mask_logits for native stop-line loss dispatch")
        if self.task_mode in {STOPLINE_ONLY_TASK_MODE, ROADMARK_JOINT_TASK_MODE} and "stop_line_mask_logits" in prediction_dict:
            self.last_lane_assignment_modes["stop_line"] = "dense_mask_only"
            if float(self.stopline_selector_aux_weight) == 1.0:
                return _stop_line_mask_loss(prediction_dict, encoded)
            return _stop_line_mask_loss_with_selector_weight(
                prediction_dict,
                encoded,
                local_x_aux_weight=float(self.stopline_local_x_aux_weight),
                selector_aux_weight=float(self.stopline_selector_aux_weight),
                geometry_aux_weight=float(self.stopline_geometry_aux_weight),
                center_target_mode=self.stopline_center_target_mode,
                centerline_target_weight=float(self.stopline_centerline_target_weight),
            )
        stop_pred = prediction_dict["stop_line"]
        stop_target = encoded["stop_line"].to(device=stop_pred.device, dtype=torch.float32)
        stop_source = encoded["mask"]["stop_line_source"].to(device=stop_pred.device, dtype=torch.bool)
        stop_valid = encoded["mask"]["stop_line_valid"].to(device=stop_pred.device, dtype=torch.bool)
        assignment = self._build_query_assignment(
            stop_pred,
            stop_target,
            stop_valid,
            stop_source,
            task_name="stop_line",
            cost_builder=_stop_line_cost_matrix,
            quality_builder=_stop_line_match_quality,
        )

        source_mask = stop_source[:, None].expand_as(stop_pred[..., 0])
        obj_loss = _objectness_loss(stop_pred[..., 0], assignment["obj_target"], source_mask)
        aux_loss = _stop_line_v2_auxiliary_loss(prediction_dict, encoded)

        valid = assignment["fg_mask"]
        if not bool(valid.any()):
            return obj_loss + aux_loss

        assigned_target = assignment["assigned_target"]
        points_pred = stop_pred[..., 1:][valid].view(-1, STOP_LINE_POINT_COUNT, 2)
        points_target = assigned_target[..., 1:][valid].view(-1, STOP_LINE_POINT_COUNT, 2)
        points_loss = F.smooth_l1_loss(points_pred, points_target, reduction="mean")
        angle_length = _batched_angle_length_cost(points_pred, points_target).mean()
        return obj_loss + 6.0 * points_loss + 0.5 * angle_length + aux_loss

    def _crosswalk_loss(self, predictions: dict[str, torch.Tensor] | torch.Tensor, encoded: dict[str, Any]) -> torch.Tensor:
        prediction_dict = predictions if isinstance(predictions, dict) else {"crosswalk": predictions}
        if self.task_mode == ROADMARK_JOINT_TASK_MODE and "crosswalk_mask_logits" not in prediction_dict:
            raise KeyError("roadmark_joint requires crosswalk_mask_logits for native crosswalk loss dispatch")
        cross_pred = prediction_dict["crosswalk"]
        aux_loss = _crosswalk_v2_auxiliary_loss(prediction_dict, encoded)
        if self.task_mode in {CROSSWALK_ONLY_TASK_MODE, ROADMARK_JOINT_TASK_MODE} and "crosswalk_mask_logits" in prediction_dict:
            self.last_lane_assignment_modes["crosswalk"] = "dense_mask_only"
            return aux_loss
        cross_target = encoded["crosswalk"].to(device=cross_pred.device, dtype=torch.float32)
        cross_source = encoded["mask"]["crosswalk_source"].to(device=cross_pred.device, dtype=torch.bool)
        cross_valid = encoded["mask"]["crosswalk_valid"].to(device=cross_pred.device, dtype=torch.bool)
        assignment = self._build_query_assignment(
            cross_pred,
            cross_target,
            cross_valid,
            cross_source,
            task_name="crosswalk",
            cost_builder=_crosswalk_cost_matrix,
            quality_builder=_crosswalk_match_quality,
        )

        source_mask = cross_source[:, None].expand_as(cross_pred[..., 0])
        obj_loss = _objectness_loss(cross_pred[..., 0], assignment["obj_target"], source_mask)

        valid = assignment["fg_mask"]
        if not bool(valid.any()):
            return obj_loss + aux_loss

        assigned_target = assignment["assigned_target"]
        points_pred = cross_pred[..., 1:][valid].view(-1, CROSSWALK_POINT_COUNT, 2)
        points_target = assigned_target[..., 1:][valid].view(-1, CROSSWALK_POINT_COUNT, 2)
        points_loss = _cyclic_contour_loss(points_pred, points_target)
        shape = _polygon_shape_loss(points_pred, points_target)
        area = _polygon_area_loss(points_pred, points_target)
        overlap = _soft_polygon_iou_loss(points_pred, points_target)
        return obj_loss + 3.0 * points_loss + 0.5 * shape + 0.5 * area + overlap + aux_loss

    def _build_query_assignment(
        self,
        pred_rows: torch.Tensor,
        target_rows: torch.Tensor,
        valid_mask: torch.Tensor,
        source_mask: torch.Tensor,
        *,
        task_name: str,
        cost_builder,
        quality_builder,
    ) -> dict[str, torch.Tensor]:
        batch_size, query_count, vector_dim = pred_rows.shape
        assigned_target = torch.zeros((batch_size, query_count, vector_dim), device=pred_rows.device, dtype=pred_rows.dtype)
        obj_target = torch.zeros((batch_size, query_count), device=pred_rows.device, dtype=pred_rows.dtype)
        fg_mask = torch.zeros((batch_size, query_count), device=pred_rows.device, dtype=torch.bool)
        matched_target_idx = torch.full((batch_size, query_count), -1, device=pred_rows.device, dtype=torch.long)
        mode = "hungarian"

        for batch_index in range(batch_size):
            if not bool(source_mask[batch_index]):
                continue
            valid_indices = torch.nonzero(valid_mask[batch_index], as_tuple=False).flatten()
            if valid_indices.numel() == 0:
                continue
            sample_pred = pred_rows[batch_index].detach()
            sample_gt = target_rows[batch_index, valid_indices].detach()
            cost = cost_builder(sample_pred, sample_gt)
            cost, cost_state = _sanitize_hungarian_cost_matrix(cost)
            if cost_state == "all_invalid":
                mode = "hungarian_all_invalid_cost"
                continue
            if cost_state == "sanitized" and mode == "hungarian":
                mode = "hungarian_sanitized"
            query_idx, gt_local_idx = _hungarian_match(cost)
            if query_idx.numel() == 0:
                continue
            gt_idx = valid_indices[gt_local_idx]
            assigned_target[batch_index, query_idx] = target_rows[batch_index, gt_idx].to(dtype=pred_rows.dtype)
            quality = quality_builder(
                sample_pred[query_idx],
                target_rows[batch_index, gt_idx].detach(),
            ).to(device=pred_rows.device, dtype=pred_rows.dtype)
            obj_target[batch_index, query_idx] = _quality_to_objectness_target(quality)
            fg_mask[batch_index, query_idx] = True
            matched_target_idx[batch_index, query_idx] = gt_idx

        self.last_lane_assignment_modes[task_name] = mode
        return {
            "assigned_target": assigned_target,
            "obj_target": obj_target,
            "fg_mask": fg_mask,
            "matched_target_idx": matched_target_idx,
        }


__all__ = ["PV26DetAssignmentUnavailable", "PV26MultiTaskLoss"]
