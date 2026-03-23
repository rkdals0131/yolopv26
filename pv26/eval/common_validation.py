"""Shared validation metric summaries for PV26/PV2 eval scripts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from torch import Tensor


@dataclass(frozen=True)
class BinaryMetricSummary:
    supervised_samples: int
    valid_pixels: int
    true_positive: int
    false_positive: int
    false_negative: int
    true_negative: int
    iou: Optional[float]
    precision: Optional[float]
    recall: Optional[float]
    f1: Optional[float]


def binary_metric_summary_from_confusion(
    *,
    supervised_samples: int,
    valid_pixels: int,
    true_positive: int,
    false_positive: int,
    false_negative: int,
    true_negative: int,
) -> BinaryMetricSummary:
    if supervised_samples <= 0 or valid_pixels <= 0:
        return BinaryMetricSummary(
            supervised_samples=int(supervised_samples),
            valid_pixels=int(valid_pixels),
            true_positive=int(true_positive),
            false_positive=int(false_positive),
            false_negative=int(false_negative),
            true_negative=int(true_negative),
            iou=None,
            precision=None,
            recall=None,
            f1=None,
        )

    denom_iou = true_positive + false_positive + false_negative
    denom_precision = true_positive + false_positive
    denom_recall = true_positive + false_negative
    precision = None if denom_precision <= 0 else float(true_positive) / float(denom_precision)
    recall = None if denom_recall <= 0 else float(true_positive) / float(denom_recall)
    f1 = None
    if precision is not None and recall is not None and (precision + recall) > 0.0:
        f1 = 2.0 * precision * recall / (precision + recall)

    return BinaryMetricSummary(
        supervised_samples=int(supervised_samples),
        valid_pixels=int(valid_pixels),
        true_positive=int(true_positive),
        false_positive=int(false_positive),
        false_negative=int(false_negative),
        true_negative=int(true_negative),
        iou=(0.0 if denom_iou <= 0 else float(true_positive) / float(denom_iou)),
        precision=precision,
        recall=recall,
        f1=f1,
    )


def accumulate_binary_confusion(
    stats: Dict[str, int],
    *,
    pred_mask: Tensor,
    target_mask: Tensor,
    valid_mask: Tensor,
) -> None:
    valid_any = valid_mask.reshape(valid_mask.shape[0], -1).any(dim=1)
    stats["supervised_samples"] += int(valid_any.sum().item())
    stats["valid_pixels"] += int(valid_mask.sum().item())
    if not bool(valid_mask.any()):
        return

    tgt = target_mask == 1
    pred_valid = pred_mask[valid_mask]
    tgt_valid = tgt[valid_mask]
    stats["tp"] += int((pred_valid & tgt_valid).sum().item())
    stats["fp"] += int((pred_valid & ~tgt_valid).sum().item())
    stats["fn"] += int((~pred_valid & tgt_valid).sum().item())
    stats["tn"] += int((~pred_valid & ~tgt_valid).sum().item())


def binary_metric_summary_to_dict(metric: BinaryMetricSummary) -> dict[str, object]:
    return {
        "supervised_samples": int(metric.supervised_samples),
        "valid_pixels": int(metric.valid_pixels),
        "true_positive": int(metric.true_positive),
        "false_positive": int(metric.false_positive),
        "false_negative": int(metric.false_negative),
        "true_negative": int(metric.true_negative),
        "iou": metric.iou,
        "precision": metric.precision,
        "recall": metric.recall,
        "f1": metric.f1,
    }
