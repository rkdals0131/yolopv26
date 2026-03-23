"""PV26 checkpoint validation helpers on the PV26 val dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

from ..dataset.labels import DET_CLASSES_CANONICAL
from ..dataset.loading.manifest_dataset import Pv26Sample
from ..eval.segmentation import lane_subclass_eval_valid_mask
from ..model.multitask_yolo26 import build_pv26_inference_model_from_state_dict
from .common_validation import (
    BinaryMetricSummary,
    accumulate_binary_confusion,
    binary_metric_summary_from_confusion,
    binary_metric_summary_to_dict,
)
from .detection import compute_map50, cxcywh_to_xyxy


LANE_SUBCLASS_CLASS_IDS: dict[str, int] = {
    "white_solid": 1,
    "white_dashed": 2,
    "yellow_solid": 3,
    "yellow_dashed": 4,
}

LANE_SUBCLASS_GROUPS: dict[str, tuple[int, ...]] = {
    "white_all": (1, 2),
    "yellow_all": (3, 4),
    "solid_all": (1, 3),
    "dashed_all": (2, 4),
}


@dataclass(frozen=True)
class Pv26EvalBatch:
    images: Tensor
    sample_ids: list[str]
    det_yolo: list[Tensor]
    has_det: Tensor
    det_label_scope: list[str]
    da_mask: Tensor
    has_da: Tensor
    rm_lane_mask: Tensor
    has_rm_lane_marker: Tensor
    rm_road_mask: Tensor
    has_rm_road_marker_non_lane: Tensor
    rm_stop_mask: Tensor
    has_rm_stop_line: Tensor
    rm_lane_subclass_mask: Tensor
    has_rm_lane_subclass: Tensor


@dataclass(frozen=True)
class Pv26ValidationSummary:
    weights_path: str
    dataset_root: str
    checkpoint_layout: str
    num_samples: int
    num_batches: int
    input_height: int
    input_width: int
    det_map50: Optional[float]
    det_eval_images: int
    det_gt_boxes: int
    det_predictions: int
    da: BinaryMetricSummary
    lane: BinaryMetricSummary
    rm_road_marker_non_lane: BinaryMetricSummary
    rm_stop_line: BinaryMetricSummary
    lane_subclass_per_class: dict[str, BinaryMetricSummary]
    lane_subclass_groups: dict[str, BinaryMetricSummary]
    lane_subclass_miou4: Optional[float]
    lane_subclass_miou4_present: Optional[float]


def collate_pv26_eval(samples: Sequence[Pv26Sample]) -> Pv26EvalBatch:
    return Pv26EvalBatch(
        images=torch.stack([sample.image for sample in samples], dim=0),
        sample_ids=[str(sample.sample_id) for sample in samples],
        det_yolo=[sample.det_yolo for sample in samples],
        has_det=torch.tensor([int(sample.has_det) for sample in samples], dtype=torch.long),
        det_label_scope=[str(sample.det_label_scope) for sample in samples],
        da_mask=torch.stack([sample.da_mask for sample in samples], dim=0),
        has_da=torch.tensor([int(sample.has_da) for sample in samples], dtype=torch.long),
        rm_lane_mask=torch.stack([sample.rm_mask[0] for sample in samples], dim=0),
        has_rm_lane_marker=torch.tensor([int(sample.has_rm_lane_marker) for sample in samples], dtype=torch.long),
        rm_road_mask=torch.stack([sample.rm_mask[1] for sample in samples], dim=0),
        has_rm_road_marker_non_lane=torch.tensor(
            [int(sample.has_rm_road_marker_non_lane) for sample in samples],
            dtype=torch.long,
        ),
        rm_stop_mask=torch.stack([sample.rm_mask[2] for sample in samples], dim=0),
        has_rm_stop_line=torch.tensor([int(sample.has_rm_stop_line) for sample in samples], dtype=torch.long),
        rm_lane_subclass_mask=torch.stack([sample.rm_lane_subclass_mask for sample in samples], dim=0),
        has_rm_lane_subclass=torch.tensor([int(sample.has_rm_lane_subclass) for sample in samples], dtype=torch.long),
    )


def load_pv26_checkpoint(weights_path: Path, *, device: torch.device) -> tuple[torch.nn.Module, str]:
    ckpt = torch.load(str(weights_path), map_location=device, weights_only=False)
    if "model_state" not in ckpt:
        raise RuntimeError(f"unsupported PV26 checkpoint payload: expected model_state in {weights_path}")
    model, layout = build_pv26_inference_model_from_state_dict(
        ckpt["model_state"],
        num_det_classes=len(DET_CLASSES_CANONICAL),
        yolo26_cfg="yolo26n.yaml",
    )
    model = model.to(device).eval()
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    return model, layout


def _prepare_images_for_inference(images: Tensor, *, device: torch.device) -> Tensor:
    images = images.to(device=device, non_blocking=True)
    if images.dtype != torch.float32:
        images = images.to(dtype=torch.float32)
        images.mul_(1.0 / 255.0)
    if device.type == "cuda":
        images = images.contiguous(memory_format=torch.channels_last)
    return images


def _resize_if_needed(logits: Tensor, *, out_hw: tuple[int, int]) -> Tensor:
    if tuple(logits.shape[-2:]) == tuple(out_hw):
        return logits
    return F.interpolate(logits, size=out_hw, mode="bilinear", align_corners=False)


def _new_confusion_stats() -> Dict[str, int]:
    return {"supervised_samples": 0, "valid_pixels": 0, "tp": 0, "fp": 0, "fn": 0, "tn": 0}


def _accumulate_lane_subclass_stats(
    stats: Dict[str, Dict[str, int]],
    *,
    pred_class: Tensor,
    target_class: Tensor,
    valid_mask: Tensor,
) -> None:
    for name, cls_id in LANE_SUBCLASS_CLASS_IDS.items():
        accumulate_binary_confusion(
            stats[name],
            pred_mask=(pred_class == int(cls_id)),
            target_mask=(target_class == int(cls_id)).to(dtype=torch.uint8),
            valid_mask=valid_mask,
        )

    for name, cls_ids in LANE_SUBCLASS_GROUPS.items():
        pred_mask = torch.zeros_like(pred_class, dtype=torch.bool)
        target_mask = torch.zeros_like(target_class, dtype=torch.uint8)
        for cls_id in cls_ids:
            pred_mask |= pred_class == int(cls_id)
            target_mask = torch.where(target_class == int(cls_id), torch.ones_like(target_mask), target_mask)
        accumulate_binary_confusion(
            stats[name],
            pred_mask=pred_mask,
            target_mask=target_mask,
            valid_mask=valid_mask,
        )


def validate_pv26(
    *,
    model: torch.nn.Module,
    loader,
    device: torch.device,
    weights_path: Path,
    dataset_root: Path,
    checkpoint_layout: str,
    conf_thres: float = 0.01,
    max_batches: int = 0,
    progress_hook: Optional[Callable[[int, int], None]] = None,
) -> Pv26ValidationSummary:
    model.eval()
    num_det_classes = len(DET_CLASSES_CANONICAL)
    preds_by_class: Dict[int, list[tuple[float, str, Tensor]]] = {c: [] for c in range(num_det_classes)}
    gt_by_img_class: Dict[tuple[str, int], Tensor] = {}

    da_stats = _new_confusion_stats()
    lane_stats = _new_confusion_stats()
    road_stats = _new_confusion_stats()
    stop_stats = _new_confusion_stats()
    lane_subclass_stats: Dict[str, Dict[str, int]] = {
        **{name: _new_confusion_stats() for name in LANE_SUBCLASS_CLASS_IDS},
        **{name: _new_confusion_stats() for name in LANE_SUBCLASS_GROUPS},
    }

    det_eval_images = 0
    det_gt_boxes = 0
    det_predictions = 0
    num_batches = 0
    num_samples = 0
    total_batches = len(loader)
    if max_batches > 0:
        total_batches = min(total_batches, int(max_batches))
    input_height = 0
    input_width = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if max_batches > 0 and batch_idx >= max_batches:
                break
            if not isinstance(batch, Pv26EvalBatch):
                raise TypeError(f"unexpected batch type: {type(batch)!r}")

            images = _prepare_images_for_inference(batch.images, device=device)
            input_height = int(images.shape[-2])
            input_width = int(images.shape[-1])

            out = model(images)
            da_logits = _resize_if_needed(out.da, out_hw=(batch.da_mask.shape[-2], batch.da_mask.shape[-1]))
            rm_logits = _resize_if_needed(out.rm, out_hw=(batch.rm_lane_mask.shape[-2], batch.rm_lane_mask.shape[-1]))
            lane_subclass_logits = _resize_if_needed(
                out.rm_lane_subclass,
                out_hw=(batch.rm_lane_subclass_mask.shape[-2], batch.rm_lane_subclass_mask.shape[-1]),
            )

            da_target = batch.da_mask.to(device=device)
            lane_target = batch.rm_lane_mask.to(device=device)
            road_target = batch.rm_road_mask.to(device=device)
            stop_target = batch.rm_stop_mask.to(device=device)
            lane_subclass_target = batch.rm_lane_subclass_mask.to(device=device)

            pred_da = da_logits[:, 0] > 0
            pred_lane = rm_logits[:, 0] > 0
            pred_road = rm_logits[:, 1] > 0
            pred_stop = rm_logits[:, 2] > 0

            accumulate_binary_confusion(
                da_stats,
                pred_mask=pred_da,
                target_mask=da_target,
                valid_mask=(da_target != 255) & batch.has_da.to(device=device, dtype=torch.bool).view(-1, 1, 1),
            )
            accumulate_binary_confusion(
                lane_stats,
                pred_mask=pred_lane,
                target_mask=lane_target,
                valid_mask=(lane_target != 255)
                & batch.has_rm_lane_marker.to(device=device, dtype=torch.bool).view(-1, 1, 1),
            )
            accumulate_binary_confusion(
                road_stats,
                pred_mask=pred_road,
                target_mask=road_target,
                valid_mask=(road_target != 255)
                & batch.has_rm_road_marker_non_lane.to(device=device, dtype=torch.bool).view(-1, 1, 1),
            )
            accumulate_binary_confusion(
                stop_stats,
                pred_mask=pred_stop,
                target_mask=stop_target,
                valid_mask=(stop_target != 255)
                & batch.has_rm_stop_line.to(device=device, dtype=torch.bool).view(-1, 1, 1),
            )

            pred_lane_subclass = torch.argmax(lane_subclass_logits, dim=1)
            rm_mask_full = torch.stack([lane_target, road_target, stop_target], dim=1)
            has_rm = torch.stack(
                [
                    batch.has_rm_lane_marker.to(device=device),
                    batch.has_rm_road_marker_non_lane.to(device=device),
                    batch.has_rm_stop_line.to(device=device),
                ],
                dim=1,
            )
            valid_lane_subclass, _ = lane_subclass_eval_valid_mask(
                rm_mask=rm_mask_full,
                rm_lane_subclass_mask=lane_subclass_target,
                has_rm=has_rm,
                has_rm_lane_subclass=batch.has_rm_lane_subclass.to(device=device),
            )
            _accumulate_lane_subclass_stats(
                lane_subclass_stats,
                pred_class=pred_lane_subclass,
                target_class=lane_subclass_target,
                valid_mask=valid_lane_subclass,
            )

            det_out = out.det
            if not isinstance(det_out, tuple) or len(det_out) < 1 or not torch.is_tensor(det_out[0]):
                raise RuntimeError("unexpected PV26 detection output")
            det_tensor = det_out[0].detach().cpu()

            for sample_idx, sample_id in enumerate(batch.sample_ids):
                num_samples += 1
                sample_rows = det_tensor[sample_idx] if sample_idx < det_tensor.shape[0] else det_tensor.new_zeros((0, 6))
                sample_rows = sample_rows[sample_rows[:, 4] > float(conf_thres)]
                det_predictions += int(sample_rows.shape[0])

                has_det = int(batch.has_det[sample_idx].item()) != 0
                det_scope = str(batch.det_label_scope[sample_idx]).strip().lower()
                if not has_det or det_scope != "full":
                    continue

                det_eval_images += 1
                gt = batch.det_yolo[sample_idx]
                if gt.numel() > 0:
                    gt_classes = gt[:, 0].to(dtype=torch.long)
                    gt_boxes = cxcywh_to_xyxy(gt[:, 1:5].to(dtype=torch.float32))
                    gt_boxes[:, 0::2] *= float(input_width)
                    gt_boxes[:, 1::2] *= float(input_height)
                    det_gt_boxes += int(gt_boxes.shape[0])
                    for cls_id in range(num_det_classes):
                        match = gt_classes == cls_id
                        if bool(match.any()):
                            gt_by_img_class[(sample_id, cls_id)] = gt_boxes[match].detach().cpu()

                for pred_idx in range(int(sample_rows.shape[0])):
                    cls_id = int(sample_rows[pred_idx, 5].item())
                    if 0 <= cls_id < num_det_classes:
                        preds_by_class[cls_id].append(
                            (
                                float(sample_rows[pred_idx, 4].item()),
                                sample_id,
                                sample_rows[pred_idx, 0:4].detach().cpu(),
                            )
                        )

            num_batches += 1
            if progress_hook is not None:
                progress_hook(num_batches, total_batches)

    det_map50, _ = compute_map50(
        preds_by_class=preds_by_class,
        gt_by_img_class=gt_by_img_class,
        num_classes=num_det_classes,
        iou_thres=0.5,
    )

    lane_subclass_per_class = {
        name: binary_metric_summary_from_confusion(
            supervised_samples=stats["supervised_samples"],
            valid_pixels=stats["valid_pixels"],
            true_positive=stats["tp"],
            false_positive=stats["fp"],
            false_negative=stats["fn"],
            true_negative=stats["tn"],
        )
        for name, stats in lane_subclass_stats.items()
        if name in LANE_SUBCLASS_CLASS_IDS
    }
    lane_subclass_groups = {
        name: binary_metric_summary_from_confusion(
            supervised_samples=stats["supervised_samples"],
            valid_pixels=stats["valid_pixels"],
            true_positive=stats["tp"],
            false_positive=stats["fp"],
            false_negative=stats["fn"],
            true_negative=stats["tn"],
        )
        for name, stats in lane_subclass_stats.items()
        if name in LANE_SUBCLASS_GROUPS
    }
    lane_subclass_ious = [metric.iou for metric in lane_subclass_per_class.values() if metric.iou is not None]
    lane_subclass_ious_present = [
        metric.iou
        for metric in lane_subclass_per_class.values()
        if metric.iou is not None and (metric.true_positive + metric.false_positive + metric.false_negative) > 0
    ]

    return Pv26ValidationSummary(
        weights_path=str(Path(weights_path)),
        dataset_root=str(Path(dataset_root)),
        checkpoint_layout=str(checkpoint_layout),
        num_samples=int(num_samples),
        num_batches=int(num_batches),
        input_height=int(input_height),
        input_width=int(input_width),
        det_map50=det_map50,
        det_eval_images=int(det_eval_images),
        det_gt_boxes=int(det_gt_boxes),
        det_predictions=int(det_predictions),
        da=binary_metric_summary_from_confusion(
            supervised_samples=da_stats["supervised_samples"],
            valid_pixels=da_stats["valid_pixels"],
            true_positive=da_stats["tp"],
            false_positive=da_stats["fp"],
            false_negative=da_stats["fn"],
            true_negative=da_stats["tn"],
        ),
        lane=binary_metric_summary_from_confusion(
            supervised_samples=lane_stats["supervised_samples"],
            valid_pixels=lane_stats["valid_pixels"],
            true_positive=lane_stats["tp"],
            false_positive=lane_stats["fp"],
            false_negative=lane_stats["fn"],
            true_negative=lane_stats["tn"],
        ),
        rm_road_marker_non_lane=binary_metric_summary_from_confusion(
            supervised_samples=road_stats["supervised_samples"],
            valid_pixels=road_stats["valid_pixels"],
            true_positive=road_stats["tp"],
            false_positive=road_stats["fp"],
            false_negative=road_stats["fn"],
            true_negative=road_stats["tn"],
        ),
        rm_stop_line=binary_metric_summary_from_confusion(
            supervised_samples=stop_stats["supervised_samples"],
            valid_pixels=stop_stats["valid_pixels"],
            true_positive=stop_stats["tp"],
            false_positive=stop_stats["fp"],
            false_negative=stop_stats["fn"],
            true_negative=stop_stats["tn"],
        ),
        lane_subclass_per_class=lane_subclass_per_class,
        lane_subclass_groups=lane_subclass_groups,
        lane_subclass_miou4=(
            None if not lane_subclass_ious else float(sum(float(v) for v in lane_subclass_ious) / float(len(lane_subclass_ious)))
        ),
        lane_subclass_miou4_present=(
            None
            if not lane_subclass_ious_present
            else float(sum(float(v) for v in lane_subclass_ious_present) / float(len(lane_subclass_ious_present)))
        ),
    )


def validation_summary_to_dict(summary: Pv26ValidationSummary) -> dict[str, object]:
    return {
        "model_type": "pv26",
        "weights_path": summary.weights_path,
        "dataset_root": summary.dataset_root,
        "checkpoint_layout": summary.checkpoint_layout,
        "num_samples": int(summary.num_samples),
        "num_batches": int(summary.num_batches),
        "input_height": int(summary.input_height),
        "input_width": int(summary.input_width),
        "det_map50": summary.det_map50,
        "det_eval_images": int(summary.det_eval_images),
        "det_gt_boxes": int(summary.det_gt_boxes),
        "det_predictions": int(summary.det_predictions),
        "da": binary_metric_summary_to_dict(summary.da),
        "lane": binary_metric_summary_to_dict(summary.lane),
        "rm_road_marker_non_lane": binary_metric_summary_to_dict(summary.rm_road_marker_non_lane),
        "rm_stop_line": binary_metric_summary_to_dict(summary.rm_stop_line),
        "lane_subclass_per_class": {
            name: binary_metric_summary_to_dict(metric) for name, metric in summary.lane_subclass_per_class.items()
        },
        "lane_subclass_groups": {
            name: binary_metric_summary_to_dict(metric) for name, metric in summary.lane_subclass_groups.items()
        },
        "lane_subclass_miou4": summary.lane_subclass_miou4,
        "lane_subclass_miou4_present": summary.lane_subclass_miou4_present,
    }


__all__ = [
    "Pv26EvalBatch",
    "Pv26ValidationSummary",
    "LANE_SUBCLASS_CLASS_IDS",
    "LANE_SUBCLASS_GROUPS",
    "collate_pv26_eval",
    "load_pv26_checkpoint",
    "validate_pv26",
    "validation_summary_to_dict",
]
