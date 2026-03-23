"""YOLOPv2 TorchScript validation helpers for PV26 val datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor
from torchvision.ops import nms

from ..dataset.labels import DET_CLASSES_CANONICAL
from ..dataset.loading.manifest_dataset import Pv26Sample
from .common_validation import (
    BinaryMetricSummary,
    accumulate_binary_confusion,
    binary_metric_summary_from_confusion,
    binary_metric_summary_to_dict,
)
from .detection import compute_map50, cxcywh_to_xyxy


@dataclass(frozen=True)
class YoloPv2EvalBatch:
    images: Tensor
    sample_ids: list[str]
    det_yolo: list[Tensor]
    has_det: Tensor
    det_label_scope: list[str]
    da_mask: Tensor
    has_da: Tensor
    rm_lane_mask: Tensor
    has_rm_lane_marker: Tensor


@dataclass(frozen=True)
class YoloPv2ValidationSummary:
    weights_path: str
    dataset_root: str
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
    unsupported_metrics: tuple[str, ...]


def collate_yolopv2_eval(samples: Sequence[Pv26Sample]) -> YoloPv2EvalBatch:
    images = torch.stack([sample.image for sample in samples], dim=0)
    da_mask = torch.stack([sample.da_mask for sample in samples], dim=0)
    rm_lane_mask = torch.stack([sample.rm_mask[0] for sample in samples], dim=0)
    has_det = torch.tensor([int(sample.has_det) for sample in samples], dtype=torch.long)
    has_da = torch.tensor([int(sample.has_da) for sample in samples], dtype=torch.long)
    has_rm_lane_marker = torch.tensor([int(sample.has_rm_lane_marker) for sample in samples], dtype=torch.long)
    return YoloPv2EvalBatch(
        images=images,
        sample_ids=[str(sample.sample_id) for sample in samples],
        det_yolo=[sample.det_yolo for sample in samples],
        has_det=has_det,
        det_label_scope=[str(sample.det_label_scope) for sample in samples],
        da_mask=da_mask,
        has_da=has_da,
        rm_lane_mask=rm_lane_mask,
        has_rm_lane_marker=has_rm_lane_marker,
    )


def load_yolopv2_torchscript(weights_path: Path, *, device: torch.device) -> torch.jit.ScriptModule:
    model = torch.jit.load(str(weights_path), map_location=device)
    model.eval()
    return model


def decode_yolopv2_predictions(
    raw_preds: Sequence[Tensor],
    anchor_grid: Sequence[Tensor],
    *,
    input_h: int,
    input_w: int,
) -> Tensor:
    decoded: list[Tensor] = []
    for pred, anchors in zip(raw_preds, anchor_grid):
        if pred.ndim != 4:
            raise ValueError(f"pred must be [B,C,H,W], got {tuple(pred.shape)}")
        if anchors.ndim != 5:
            raise ValueError(f"anchor_grid must be [1,na,1,1,2], got {tuple(anchors.shape)}")
        bs, ch, ny, nx = pred.shape
        na = int(anchors.shape[1])
        if na <= 0 or ch % na != 0:
            raise ValueError(f"invalid detection layout: ch={ch} na={na}")
        no = ch // na
        pred = pred.view(bs, na, no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
        y = pred.sigmoid()

        grid_y, grid_x = torch.meshgrid(
            torch.arange(ny, device=pred.device),
            torch.arange(nx, device=pred.device),
            indexing="ij",
        )
        grid = torch.stack((grid_x, grid_y), dim=-1).view(1, 1, ny, nx, 2).to(dtype=y.dtype)
        stride_x = float(input_w) / float(nx)
        stride_y = float(input_h) / float(ny)

        y[..., 0] = (y[..., 0] * 2.0 - 0.5 + grid[..., 0]) * stride_x
        y[..., 1] = (y[..., 1] * 2.0 - 0.5 + grid[..., 1]) * stride_y
        y[..., 2:4] = (y[..., 2:4] * 2.0).pow(2.0) * anchors.to(dtype=y.dtype)
        decoded.append(y.view(bs, -1, no))

    if not decoded:
        raise ValueError("raw_preds/anchor_grid must be non-empty")
    return torch.cat(decoded, dim=1)


def nms_yolopv2_predictions(
    pred: Tensor,
    *,
    conf_thres: float,
    iou_thres: float,
    max_det: int,
) -> tuple[Tensor, Tensor, Tensor]:
    if pred.ndim != 2:
        raise ValueError(f"pred must be [N,5+nc], got {tuple(pred.shape)}")
    if pred.numel() == 0:
        z = pred.new_zeros((0,))
        return pred.new_zeros((0, 4)), z, z.to(dtype=torch.long)

    obj_conf = pred[:, 4]
    cls_scores, cls_idx = pred[:, 5:].max(dim=1)
    scores = obj_conf * cls_scores
    keep = scores > float(conf_thres)
    if not bool(keep.any()):
        z = pred.new_zeros((0,))
        return pred.new_zeros((0, 4)), z, z.to(dtype=torch.long)

    boxes_xyxy = cxcywh_to_xyxy(pred[keep, :4])
    scores = scores[keep]
    cls_idx = cls_idx[keep]
    max_wh = 4096.0
    boxes_for_nms = boxes_xyxy + cls_idx.to(dtype=boxes_xyxy.dtype).unsqueeze(1) * max_wh
    keep_idx = nms(boxes_for_nms, scores, float(iou_thres))[: int(max_det)]
    return boxes_xyxy[keep_idx], scores[keep_idx], cls_idx[keep_idx]


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


def validate_yolopv2(
    *,
    model: torch.nn.Module,
    loader,
    device: torch.device,
    weights_path: Path,
    dataset_root: Path,
    conf_thres: float,
    iou_thres: float,
    max_det: int,
    max_batches: int = 0,
    progress_hook: Optional[Callable[[int, int], None]] = None,
) -> YoloPv2ValidationSummary:
    model.eval()
    num_det_classes = len(DET_CLASSES_CANONICAL)
    preds_by_class: Dict[int, list[tuple[float, str, Tensor]]] = {c: [] for c in range(num_det_classes)}
    gt_by_img_class: Dict[tuple[str, int], Tensor] = {}
    da_stats = {"supervised_samples": 0, "valid_pixels": 0, "tp": 0, "fp": 0, "fn": 0, "tn": 0}
    lane_stats = {"supervised_samples": 0, "valid_pixels": 0, "tp": 0, "fp": 0, "fn": 0, "tn": 0}
    total_predictions = 0
    det_eval_images = 0
    det_gt_boxes = 0
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

            images = _prepare_images_for_inference(batch.images, device=device)
            input_height = int(images.shape[-2])
            input_width = int(images.shape[-1])
            out = model(images)
            if not isinstance(out, tuple) or len(out) != 3:
                raise RuntimeError("unexpected YOLOPv2 TorchScript output: expected ((pred,anchors), seg, ll)")

            det_out, da_logits, lane_logits = out
            if not isinstance(det_out, tuple) or len(det_out) != 2:
                raise RuntimeError("unexpected YOLOPv2 detection output: expected (pred_layers, anchor_grid)")

            raw_preds, anchor_grid = det_out
            decoded = decode_yolopv2_predictions(
                raw_preds,
                anchor_grid,
                input_h=input_height,
                input_w=input_width,
            )

            da_logits = _resize_if_needed(da_logits, out_hw=(batch.da_mask.shape[-2], batch.da_mask.shape[-1]))
            lane_logits = _resize_if_needed(lane_logits, out_hw=(batch.rm_lane_mask.shape[-2], batch.rm_lane_mask.shape[-1]))

            if int(da_logits.shape[1]) == 2:
                pred_da = da_logits[:, 1] > da_logits[:, 0]
            else:
                pred_da = torch.sigmoid(da_logits[:, 0]) > 0.5

            lane_prob = lane_logits[:, 0]
            if float(lane_prob.min().item()) < 0.0 or float(lane_prob.max().item()) > 1.0:
                lane_prob = torch.sigmoid(lane_prob)
            pred_lane = lane_prob > 0.5

            accumulate_binary_confusion(
                da_stats,
                pred_mask=pred_da,
                target_mask=batch.da_mask.to(device=device),
                valid_mask=(batch.da_mask.to(device=device) != 255)
                & batch.has_da.to(device=device, dtype=torch.bool).view(-1, 1, 1),
            )
            accumulate_binary_confusion(
                lane_stats,
                pred_mask=pred_lane,
                target_mask=batch.rm_lane_mask.to(device=device),
                valid_mask=(batch.rm_lane_mask.to(device=device) != 255)
                & batch.has_rm_lane_marker.to(device=device, dtype=torch.bool).view(-1, 1, 1),
            )

            for sample_idx, sample_id in enumerate(batch.sample_ids):
                num_samples += 1
                sample_preds = decoded[sample_idx]
                boxes_xyxy, scores, classes = nms_yolopv2_predictions(
                    sample_preds,
                    conf_thres=conf_thres,
                    iou_thres=iou_thres,
                    max_det=max_det,
                )
                total_predictions += int(scores.shape[0])

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

                for pred_idx in range(int(scores.shape[0])):
                    cls_id = int(classes[pred_idx].item())
                    if 0 <= cls_id < num_det_classes:
                        preds_by_class[cls_id].append(
                            (
                                float(scores[pred_idx].item()),
                                sample_id,
                                boxes_xyxy[pred_idx].detach().cpu(),
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
    return YoloPv2ValidationSummary(
        weights_path=str(Path(weights_path)),
        dataset_root=str(Path(dataset_root)),
        num_samples=int(num_samples),
        num_batches=int(num_batches),
        input_height=int(input_height),
        input_width=int(input_width),
        det_map50=det_map50,
        det_eval_images=int(det_eval_images),
        det_gt_boxes=int(det_gt_boxes),
        det_predictions=int(total_predictions),
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
        unsupported_metrics=(
            "rm_road_marker_non_lane",
            "rm_stop_line",
            "rm_lane_subclass",
        ),
    )


def validation_summary_to_dict(summary: YoloPv2ValidationSummary) -> dict[str, object]:
    return {
        "model_type": "yolopv2",
        "weights_path": summary.weights_path,
        "dataset_root": summary.dataset_root,
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
        "unsupported_metrics": list(summary.unsupported_metrics),
    }
