from __future__ import annotations

import random
from typing import Dict, List, Optional

import torch


def _run_nms(boxes_xyxy: torch.Tensor, scores: torch.Tensor, nms_iou: float) -> torch.Tensor:
    from torchvision.ops import nms

    return nms(boxes_xyxy, scores, float(nms_iou))


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def seed_everything(seed: int, *, device: torch.device) -> None:
    seed = int(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def mean_losses(losses_sum: Dict[str, float], steps: int) -> Dict[str, float]:
    if steps <= 0:
        return {"total": 0.0, "od": 0.0, "da": 0.0, "rm": 0.0, "rm_lane_subclass": 0.0}
    return {k: v / float(steps) for k, v in losses_sum.items()}


def update_binary_iou(stats: Dict[str, int], logits: torch.Tensor, target_mask: torch.Tensor) -> None:
    valid = target_mask != 255
    if not bool(valid.any()):
        return
    pred = logits > 0
    tgt = target_mask == 1
    pred_valid = pred[valid]
    tgt_valid = tgt[valid]
    inter = int((pred_valid & tgt_valid).sum().item())
    union = int((pred_valid | tgt_valid).sum().item())
    stats["inter"] += inter
    stats["union"] += union
    stats["supervised"] += 1


def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    return torch.stack([x1, y1, x2, y2], dim=-1)


def _box_iou_one_to_many(box_xyxy: torch.Tensor, boxes_xyxy: torch.Tensor) -> torch.Tensor:
    x1 = torch.maximum(box_xyxy[0], boxes_xyxy[:, 0])
    y1 = torch.maximum(box_xyxy[1], boxes_xyxy[:, 1])
    x2 = torch.minimum(box_xyxy[2], boxes_xyxy[:, 2])
    y2 = torch.minimum(box_xyxy[3], boxes_xyxy[:, 3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    area1 = (box_xyxy[2] - box_xyxy[0]).clamp(min=0) * (box_xyxy[3] - box_xyxy[1]).clamp(min=0)
    area2 = (boxes_xyxy[:, 2] - boxes_xyxy[:, 0]).clamp(min=0) * (boxes_xyxy[:, 3] - boxes_xyxy[:, 1]).clamp(min=0)
    union = area1 + area2 - inter
    return inter / (union + 1e-9)


def _compute_ap(recalls: List[float], precisions: List[float]) -> float:
    if not recalls:
        return 0.0
    mrec = [0.0] + [float(r) for r in recalls] + [1.0]
    mpre = [0.0] + [float(p) for p in precisions] + [0.0]
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    ap = 0.0
    for i in range(len(mrec) - 1):
        ap += (mrec[i + 1] - mrec[i]) * mpre[i + 1]
    return float(ap)


def decode_det_predictions(
    det_logits: torch.Tensor,
    *,
    conf_thres: float,
    nms_iou: float,
    max_det: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    out_ch, _, _ = det_logits.shape
    num_classes = out_ch - 5
    if num_classes <= 0:
        return det_logits.new_zeros((0, 4)), det_logits.new_zeros((0,)), det_logits.new_zeros((0,), dtype=torch.long)

    flat = det_logits.permute(1, 2, 0).reshape(-1, 5 + num_classes)
    boxes = torch.sigmoid(flat[:, 0:4])
    obj = torch.sigmoid(flat[:, 4])
    cls_prob = torch.softmax(flat[:, 5:], dim=1)
    scores, cls_idx = (obj[:, None] * cls_prob).max(dim=1)

    keep = scores > float(conf_thres)
    if not bool(keep.any()):
        return det_logits.new_zeros((0, 4)), det_logits.new_zeros((0,)), det_logits.new_zeros((0,), dtype=torch.long)

    boxes = boxes[keep]
    scores = scores[keep]
    cls_idx = cls_idx[keep]
    boxes_xyxy = cxcywh_to_xyxy(boxes).clamp(0.0, 1.0)

    kept_all: List[torch.Tensor] = []
    for c in range(num_classes):
        idx = (cls_idx == c).nonzero(as_tuple=False).squeeze(1)
        if idx.numel() == 0:
            continue
        keep_idx = _run_nms(boxes_xyxy[idx], scores[idx], float(nms_iou))
        kept_all.append(idx[keep_idx])

    if not kept_all:
        return det_logits.new_zeros((0, 4)), det_logits.new_zeros((0,)), det_logits.new_zeros((0,), dtype=torch.long)

    kept = torch.cat(kept_all, dim=0)
    kept = kept[torch.argsort(scores[kept], descending=True)]
    kept = kept[: int(max_det)]
    return boxes_xyxy[kept], scores[kept], cls_idx[kept]


def compute_map50(
    *,
    preds_by_class: Dict[int, List[tuple[float, str, torch.Tensor]]],
    gt_by_img_class: Dict[tuple[str, int], torch.Tensor],
    num_classes: int,
    iou_thres: float = 0.5,
) -> tuple[Optional[float], Dict[int, Optional[float]]]:
    ap_by_class: Dict[int, Optional[float]] = {}
    gt_imgs_for_class: Dict[int, List[str]] = {c: [] for c in range(num_classes)}
    for (img_id, c), gt_boxes in gt_by_img_class.items():
        if gt_boxes.numel() > 0:
            gt_imgs_for_class[c].append(img_id)

    for c in range(num_classes):
        img_ids = gt_imgs_for_class.get(c, [])
        total_gt = int(sum(int(gt_by_img_class[(img_id, c)].shape[0]) for img_id in img_ids)) if img_ids else 0
        if total_gt == 0:
            ap_by_class[c] = None
            continue

        matched: Dict[str, torch.Tensor] = {
            img_id: torch.zeros((int(gt_by_img_class[(img_id, c)].shape[0]),), dtype=torch.bool) for img_id in img_ids
        }
        preds = preds_by_class.get(c, [])
        preds_sorted = sorted(preds, key=lambda t: float(t[0]), reverse=True)
        tp_cum = 0
        fp_cum = 0
        recalls: List[float] = []
        precisions: List[float] = []

        for _score, img_id, box_xyxy in preds_sorted:
            gt_boxes = gt_by_img_class.get((img_id, c))
            if gt_boxes is None or gt_boxes.numel() == 0:
                fp_cum += 1
            else:
                ious = _box_iou_one_to_many(box_xyxy, gt_boxes)
                max_iou, max_idx = torch.max(ious, dim=0)
                j = int(max_idx.item())
                if float(max_iou.item()) >= float(iou_thres) and not bool(matched[img_id][j].item()):
                    matched[img_id][j] = True
                    tp_cum += 1
                else:
                    fp_cum += 1
            recalls.append(tp_cum / float(total_gt))
            precisions.append(tp_cum / float(max(1, tp_cum + fp_cum)))

        ap_by_class[c] = _compute_ap(recalls, precisions)

    aps = [ap for ap in ap_by_class.values() if ap is not None]
    map50 = float(sum(aps) / float(len(aps))) if aps else None
    return map50, ap_by_class


def format_loss_line(prefix: str, losses: Dict[str, float]) -> str:
    line = (
        f"{prefix} total={losses['total']:.4f} od={losses['od']:.4f} "
        f"da={losses['da']:.4f} rm={losses['rm']:.4f}"
    )
    if "rm_lane_subclass" in losses:
        line += f" rm_lane_subclass={losses['rm_lane_subclass']:.4f}"
    return line
