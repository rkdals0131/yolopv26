#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import random
import sys
from typing import Dict, List, Optional
import warnings

import torch
from torch.utils.data import DataLoader
from torchvision.ops import nms

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pv26.constants import DET_CLASSES_CANONICAL
from pv26.criterion import PV26Criterion
from pv26.multitask_model import PV26MultiHead
from pv26.torch_dataset import Pv26ManifestDataset, Pv26Sample


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="PV26 minimal train/val smoke loop")
    p.add_argument("--dataset-root", type=Path, required=True, help="PV26 dataset root with meta/split_manifest.csv")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", type=str, default="auto", help="auto|cpu|cuda|cuda:N")
    p.add_argument("--out-dir", type=Path, default=Path("runs/pv26_smoke"))
    p.add_argument("--max-train-batches", type=int, default=50)
    p.add_argument("--max-val-batches", type=int, default=50)
    p.add_argument("--log-every", type=int, default=10)
    return p


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


def _mean_losses(losses_sum: Dict[str, float], steps: int) -> Dict[str, float]:
    if steps <= 0:
        return {"total": 0.0, "od": 0.0, "da": 0.0, "rm": 0.0}
    return {k: v / float(steps) for k, v in losses_sum.items()}


def _update_binary_iou(stats: Dict[str, int], logits: torch.Tensor, target_mask: torch.Tensor) -> None:
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


def _cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    return torch.stack([x1, y1, x2, y2], dim=-1)


def _box_iou_one_to_many(box_xyxy: torch.Tensor, boxes_xyxy: torch.Tensor) -> torch.Tensor:
    """
    Args:
      box_xyxy: [4]
      boxes_xyxy: [N,4]
    Returns:
      iou: [N]
    """
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
    # VOC-style AP with precision envelope.
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


def _decode_det_predictions(
    det_logits: torch.Tensor,
    *,
    conf_thres: float,
    nms_iou: float,
    max_det: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Decode dense det head to a list of (xyxy, score, class_id).

    Expected det_logits shape: [5 + C, Hg, Wg]
      - box logits are sigmoid'ed into normalized cx/cy/w/h in [0,1]
      - objectness uses sigmoid
      - class uses softmax
      - score = obj * max_class_prob
    """
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
    boxes_xyxy = _cxcywh_to_xyxy(boxes).clamp(0.0, 1.0)

    kept_all: List[torch.Tensor] = []
    for c in range(num_classes):
        idx = (cls_idx == c).nonzero(as_tuple=False).squeeze(1)
        if idx.numel() == 0:
            continue
        keep_idx = nms(boxes_xyxy[idx], scores[idx], float(nms_iou))
        kept_all.append(idx[keep_idx])

    if not kept_all:
        return det_logits.new_zeros((0, 4)), det_logits.new_zeros((0,)), det_logits.new_zeros((0,), dtype=torch.long)

    kept = torch.cat(kept_all, dim=0)
    kept = kept[torch.argsort(scores[kept], descending=True)]
    kept = kept[: int(max_det)]
    return boxes_xyxy[kept], scores[kept], cls_idx[kept]


def _compute_map50(
    *,
    preds_by_class: Dict[int, List[tuple[float, str, torch.Tensor]]],
    gt_by_img_class: Dict[tuple[str, int], torch.Tensor],
    num_classes: int,
    iou_thres: float = 0.5,
) -> tuple[Optional[float], Dict[int, Optional[float]]]:
    ap_by_class: Dict[int, Optional[float]] = {}

    # Pre-build image lists per class for faster access.
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

        for score, img_id, box_xyxy in preds_sorted:
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


def _format_loss_line(prefix: str, losses: Dict[str, float]) -> str:
    return (
        f"{prefix} total={losses['total']:.4f} od={losses['od']:.4f} "
        f"da={losses['da']:.4f} rm={losses['rm']:.4f}"
    )


def train_one_epoch(
    *,
    model: PV26MultiHead,
    criterion: PV26Criterion,
    loader: DataLoader[List[Pv26Sample]],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_batches: int,
    log_every: int,
) -> Dict[str, float]:
    model.train()
    losses_sum = {"total": 0.0, "od": 0.0, "da": 0.0, "rm": 0.0}
    steps = 0

    for batch_idx, batch in enumerate(loader):
        if batch_idx >= max_batches:
            break

        images = torch.stack([s.image for s in batch], dim=0).to(device=device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        preds = model(images)
        losses = criterion(preds, batch)
        losses["total"].backward()
        optimizer.step()

        steps += 1
        for key in losses_sum:
            losses_sum[key] += float(losses[key].detach().item())

        if (batch_idx + 1) % max(1, log_every) == 0 or batch_idx == 0:
            step_losses = {k: float(v.detach().item()) for k, v in losses.items()}
            print(_format_loss_line(f"[train] step={batch_idx + 1}", step_losses), flush=True)

    return _mean_losses(losses_sum, steps)


def validate(
    *,
    model: PV26MultiHead,
    criterion: PV26Criterion,
    loader: DataLoader[List[Pv26Sample]],
    device: torch.device,
    max_batches: int,
    log_every: int,
) -> tuple[Dict[str, float], Dict[str, Optional[float]], Optional[float]]:
    model.eval()
    losses_sum = {"total": 0.0, "od": 0.0, "da": 0.0, "rm": 0.0}
    steps = 0

    metric_stats: Dict[str, Dict[str, int]] = {
        "da": {"inter": 0, "union": 0, "supervised": 0},
        "rm_lane_marker": {"inter": 0, "union": 0, "supervised": 0},
        "rm_road_marker_non_lane": {"inter": 0, "union": 0, "supervised": 0},
        "rm_stop_line": {"inter": 0, "union": 0, "supervised": 0},
    }

    num_det_classes = len(DET_CLASSES_CANONICAL)
    det_preds_by_class: Dict[int, List[tuple[float, str, torch.Tensor]]] = {c: [] for c in range(num_det_classes)}
    gt_by_img_class: Dict[tuple[str, int], torch.Tensor] = {}
    det_conf_thres = 0.01
    det_nms_iou = 0.5
    det_max_per_image = 200

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= max_batches:
                break

            images = torch.stack([s.image for s in batch], dim=0).to(device=device, non_blocking=True)
            preds = model(images)
            losses = criterion(preds, batch)

            steps += 1
            for key in losses_sum:
                losses_sum[key] += float(losses[key].detach().item())

            if (batch_idx + 1) % max(1, log_every) == 0 or batch_idx == 0:
                step_losses = {k: float(v.detach().item()) for k, v in losses.items()}
                print(_format_loss_line(f"[val] step={batch_idx + 1}", step_losses), flush=True)

            for i, sample in enumerate(batch):
                if sample.has_da:
                    _update_binary_iou(metric_stats["da"], preds.da[i, 0], sample.da_mask.to(device=device))

                rm_channels = [
                    ("rm_lane_marker", sample.has_rm_lane_marker),
                    ("rm_road_marker_non_lane", sample.has_rm_road_marker_non_lane),
                    ("rm_stop_line", sample.has_rm_stop_line),
                ]
                for c_idx, (name, has_flag) in enumerate(rm_channels):
                    if has_flag:
                        _update_binary_iou(metric_stats[name], preds.rm[i, c_idx], sample.rm_mask[c_idx].to(device=device))

                # Detection mAP@0.5 (only when fully labeled)
                if sample.has_det and str(sample.det_label_scope) == "full":
                    gt = sample.det_yolo
                    if gt.numel():
                        gt_cls = gt[:, 0].to(dtype=torch.long)
                        gt_boxes = _cxcywh_to_xyxy(gt[:, 1:5].to(dtype=torch.float32)).clamp(0.0, 1.0)
                        for c in range(num_det_classes):
                            m = gt_cls == c
                            if bool(m.any()):
                                gt_by_img_class[(sample.sample_id, c)] = gt_boxes[m].detach().cpu()

                    det_boxes, det_scores, det_cls = _decode_det_predictions(
                        preds.det[i].detach(),
                        conf_thres=det_conf_thres,
                        nms_iou=det_nms_iou,
                        max_det=det_max_per_image,
                    )
                    det_boxes = det_boxes.detach().cpu()
                    det_scores = det_scores.detach().cpu()
                    det_cls = det_cls.detach().cpu()
                    for j in range(int(det_scores.shape[0])):
                        c = int(det_cls[j].item())
                        if 0 <= c < num_det_classes:
                            det_preds_by_class[c].append((float(det_scores[j].item()), sample.sample_id, det_boxes[j]))

    mean_losses = _mean_losses(losses_sum, steps)
    iou_metrics: Dict[str, Optional[float]] = {}
    for name, stats in metric_stats.items():
        if stats["supervised"] == 0:
            iou_metrics[name] = None
        elif stats["union"] == 0:
            iou_metrics[name] = 0.0
        else:
            iou_metrics[name] = float(stats["inter"]) / float(stats["union"])

    map50, _ap_by_class = _compute_map50(
        preds_by_class=det_preds_by_class,
        gt_by_img_class=gt_by_img_class,
        num_classes=num_det_classes,
        iou_thres=0.5,
    )
    return mean_losses, iou_metrics, map50


def main() -> int:
    args = build_argparser().parse_args()
    device = resolve_device(args.device)
    if device.type == "cpu":
        warnings.filterwarnings(
            "ignore",
            message=r"CUDA initialization: Unexpected error from cudaGetDeviceCount.*",
        )
    seed_everything(args.seed, device=device)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = Pv26ManifestDataset(dataset_root=args.dataset_root, splits=("train",))
    val_ds = Pv26ManifestDataset(dataset_root=args.dataset_root, splits=("val",))

    if len(train_ds) == 0:
        raise RuntimeError("train split is empty")
    if len(val_ds) == 0:
        raise RuntimeError("val split is empty")

    dl_gen = torch.Generator().manual_seed(int(args.seed))
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=lambda x: x,
        generator=dl_gen,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda x: x,
    )

    model = PV26MultiHead(num_det_classes=len(DET_CLASSES_CANONICAL)).to(device)
    criterion = PV26Criterion(num_det_classes=len(DET_CLASSES_CANONICAL)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(f"[pv26] device={device} seed={int(args.seed)}")
    print(f"[pv26] train_samples={len(train_ds)} val_samples={len(val_ds)}")

    for epoch in range(args.epochs):
        print(f"\n[pv26] epoch {epoch + 1}/{args.epochs}")
        train_losses = train_one_epoch(
            model=model,
            criterion=criterion,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            max_batches=args.max_train_batches,
            log_every=args.log_every,
        )
        print(_format_loss_line("[train] mean", train_losses))

        val_losses, ious, map50 = validate(
            model=model,
            criterion=criterion,
            loader=val_loader,
            device=device,
            max_batches=args.max_val_batches,
            log_every=args.log_every,
        )
        print(_format_loss_line("[val] mean", val_losses))

        for name, iou in ious.items():
            if iou is None:
                print(f"[val] iou_{name}=skipped(no_supervision)")
            else:
                print(f"[val] iou_{name}={iou:.4f}")

        if map50 is None:
            print("[val] map50=skipped(no_det_gt)")
        else:
            print(f"[val] map50={map50:.4f}")

        ckpt_path = out_dir / f"epoch_{epoch + 1:03d}.pt"
        torch.save(model.state_dict(), ckpt_path)
        print(f"[pv26] saved checkpoint: {ckpt_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
