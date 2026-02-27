#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
import sys
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
from torchvision.ops import nms

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pv26.constants import DET_CLASSES_CANONICAL
from pv26.multitask_model import PV26MultiHeadYOLO26


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render pv2/pv26 outputs on one image")
    p.add_argument("--image", type=Path, default=Path("datasets/weights/example.jpg"))
    p.add_argument("--pv2-weight", type=Path, default=Path("datasets/weights/yolopv2.pt"))
    p.add_argument("--pv26-weight", type=Path, default=Path("datasets/weights/best.pt"))
    p.add_argument("--imgsz", type=int, default=640, help="Legacy square input size when --input-width/--input-height are not set")
    p.add_argument("--input-width", type=int, default=None, help="Model input width (recommend: 960)")
    p.add_argument("--input-height", type=int, default=None, help="Model input height (recommend: 544)")
    p.add_argument("--conf-thres", type=float, default=0.3)
    p.add_argument("--iou-thres", type=float, default=0.45)
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "auto"])
    p.add_argument("--da-alpha", type=float, default=0.30, help="Max alpha for DA overlay")
    p.add_argument("--lane-alpha", type=float, default=0.55, help="Max alpha for lane overlay")
    p.add_argument("--pv26-lane-thres", type=float, default=0.0, help="Optional floor for PV26 lane prob before overlay")
    p.add_argument("--out-pv2", type=Path, default=Path("datasets/weights/result_pv2.jpg"))
    p.add_argument("--out-pv26", type=Path, default=Path("datasets/weights/result_pv26.jpg"))
    return p.parse_args()


@dataclass(frozen=True)
class LetterboxMeta:
    orig_h: int
    orig_w: int
    in_h: int
    in_w: int
    new_h: int
    new_w: int
    pad_y: int
    pad_x: int
    scale: float


def pick_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _letterbox_image_for_model(image_bgr: np.ndarray, *, in_h: int, in_w: int, pad_value: int = 114) -> tuple[np.ndarray, LetterboxMeta]:
    orig_h, orig_w = image_bgr.shape[:2]
    if orig_h <= 0 or orig_w <= 0:
        raise ValueError("invalid input image size")
    if in_h <= 0 or in_w <= 0:
        raise ValueError("invalid model input size")

    scale = min(float(in_w) / float(orig_w), float(in_h) / float(orig_h))
    new_w = int(round(orig_w * scale))
    new_h = int(round(orig_h * scale))
    pad_x = (in_w - new_w) // 2
    pad_y = (in_h - new_h) // 2

    resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((in_h, in_w, 3), int(pad_value), dtype=np.uint8)
    canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized
    meta = LetterboxMeta(
        orig_h=orig_h,
        orig_w=orig_w,
        in_h=in_h,
        in_w=in_w,
        new_h=new_h,
        new_w=new_w,
        pad_y=pad_y,
        pad_x=pad_x,
        scale=scale,
    )
    return canvas, meta


def read_image(path: Path, *, in_h: int, in_w: int) -> tuple[np.ndarray, torch.Tensor, LetterboxMeta]:
    orig_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if orig_bgr is None:
        raise FileNotFoundError(f"failed to load image: {path}")
    model_bgr, meta = _letterbox_image_for_model(orig_bgr, in_h=in_h, in_w=in_w, pad_value=114)
    rgb = cv2.cvtColor(model_bgr, cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    return orig_bgr, x, meta


def overlay_mask(base: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float = 0.35) -> np.ndarray:
    out = base.copy()
    overlay = np.zeros_like(out, dtype=np.uint8)
    overlay[:, :] = np.array(color, dtype=np.uint8)
    sel = mask.astype(bool)
    out[sel] = cv2.addWeighted(base[sel], 1.0 - alpha, overlay[sel], alpha, 0)
    return out


def overlay_prob(base: np.ndarray, prob: np.ndarray, color: tuple[int, int, int], max_alpha: float) -> np.ndarray:
    p = np.clip(prob.astype(np.float32), 0.0, 1.0)
    a = np.clip(p * float(max_alpha), 0.0, 1.0)[..., None]
    color_arr = np.array(color, dtype=np.float32).reshape(1, 1, 3)
    out = base.astype(np.float32) * (1.0 - a) + color_arr * a
    return np.clip(out, 0.0, 255.0).astype(np.uint8)


def draw_boxes(
    image: np.ndarray,
    boxes_xyxy: torch.Tensor,
    scores: torch.Tensor,
    classes: torch.Tensor,
    names: Iterable[str] | None = None,
) -> np.ndarray:
    out = image.copy()
    name_list = list(names) if names is not None else None
    for i in range(boxes_xyxy.shape[0]):
        x1, y1, x2, y2 = [int(v) for v in boxes_xyxy[i].tolist()]
        conf = float(scores[i].item())
        cls = int(classes[i].item())
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 165, 255), 2)
        label = f"{cls}:{conf:.2f}"
        if name_list is not None and 0 <= cls < len(name_list):
            label = f"{name_list[cls]}:{conf:.2f}"
        cv2.putText(out, label, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def xywh_to_xyxy(x: torch.Tensor) -> torch.Tensor:
    y = x.clone()
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def decode_pv2_preds(raw_preds: list[torch.Tensor], anchor_grid: list[torch.Tensor], input_h: int, input_w: int) -> torch.Tensor:
    decoded = []
    for p, ag in zip(raw_preds, anchor_grid):
        bs, ch, ny, nx = p.shape
        na = ag.shape[1]
        no = ch // na
        p = p.view(bs, na, no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
        y = p.sigmoid()

        yv, xv = torch.meshgrid(
            torch.arange(ny, device=p.device),
            torch.arange(nx, device=p.device),
            indexing="ij",
        )
        grid = torch.stack((xv, yv), dim=-1).view(1, 1, ny, nx, 2).float()

        stride_x = float(input_w) / float(nx)
        stride_y = float(input_h) / float(ny)
        y[..., 0] = (y[..., 0] * 2.0 - 0.5 + grid[..., 0]) * stride_x
        y[..., 1] = (y[..., 1] * 2.0 - 0.5 + grid[..., 1]) * stride_y
        y[..., 2:4] = (y[..., 2:4] * 2.0) ** 2 * ag
        decoded.append(y.view(bs, -1, no))
    return torch.cat(decoded, dim=1)


def nms_yolo(pred: torch.Tensor, conf_thres: float, iou_thres: float, max_det: int = 200) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # pred: [N, 5+nc] with xywh.
    if pred.numel() == 0:
        z = pred.new_zeros((0,))
        return pred.new_zeros((0, 4)), z, z.long()

    obj_conf = pred[:, 4]
    cls_scores, cls_idx = pred[:, 5:].max(dim=1)
    scores = obj_conf * cls_scores
    keep = scores > conf_thres
    if not torch.any(keep):
        z = pred.new_zeros((0,))
        return pred.new_zeros((0, 4)), z, z.long()

    boxes = xywh_to_xyxy(pred[keep, :4])
    scores = scores[keep]
    cls_idx = cls_idx[keep]

    max_wh = 4096
    boxes_for_nms = boxes + cls_idx.float().unsqueeze(1) * max_wh
    keep_idx = nms(boxes_for_nms, scores, iou_thres)
    keep_idx = keep_idx[:max_det]
    return boxes[keep_idx], scores[keep_idx], cls_idx[keep_idx]


def _unletterbox_prob_map(prob_map: np.ndarray, meta: LetterboxMeta) -> np.ndarray:
    crop = prob_map[meta.pad_y : meta.pad_y + meta.new_h, meta.pad_x : meta.pad_x + meta.new_w]
    return cv2.resize(crop, (meta.orig_w, meta.orig_h), interpolation=cv2.INTER_LINEAR)


def _unletterbox_binary_mask(mask: np.ndarray, meta: LetterboxMeta) -> np.ndarray:
    m = mask.astype(np.uint8)
    crop = m[meta.pad_y : meta.pad_y + meta.new_h, meta.pad_x : meta.pad_x + meta.new_w]
    out = cv2.resize(crop, (meta.orig_w, meta.orig_h), interpolation=cv2.INTER_NEAREST)
    return out.astype(bool)


def _unletterbox_boxes_xyxy(boxes_xyxy: torch.Tensor, meta: LetterboxMeta) -> torch.Tensor:
    if boxes_xyxy.numel() == 0:
        return boxes_xyxy
    out = boxes_xyxy.clone()
    out[:, 0] = (out[:, 0] - float(meta.pad_x)) / float(meta.scale)
    out[:, 2] = (out[:, 2] - float(meta.pad_x)) / float(meta.scale)
    out[:, 1] = (out[:, 1] - float(meta.pad_y)) / float(meta.scale)
    out[:, 3] = (out[:, 3] - float(meta.pad_y)) / float(meta.scale)
    out[:, [0, 2]] = out[:, [0, 2]].clamp(0.0, float(meta.orig_w - 1))
    out[:, [1, 3]] = out[:, [1, 3]].clamp(0.0, float(meta.orig_h - 1))
    return out


def render_pv2(
    model_path: Path,
    input_tensor: torch.Tensor,
    base_bgr: np.ndarray,
    meta: LetterboxMeta,
    conf_thres: float,
    iou_thres: float,
    da_alpha: float,
    lane_alpha: float,
    device: torch.device,
) -> np.ndarray:
    model = torch.jit.load(str(model_path), map_location=device).eval()
    with torch.no_grad():
        out = model(input_tensor.to(device))

    (pred_raw, anchor_grid), seg, ll = out
    in_h = int(input_tensor.shape[-2])
    in_w = int(input_tensor.shape[-1])
    pred = decode_pv2_preds(list(pred_raw), list(anchor_grid), input_h=in_h, input_w=in_w)[0]
    boxes, scores, classes = nms_yolo(pred, conf_thres=conf_thres, iou_thres=iou_thres)
    boxes = _unletterbox_boxes_xyxy(boxes, meta)

    # Match official PV2 demo style: binary masks, then overlay.
    if seg.shape[1] == 2:
        da_mask = (seg[0, 1] > seg[0, 0]).detach().cpu().numpy()
    else:
        da_mask = (torch.sigmoid(seg[0, 0]) > 0.5).detach().cpu().numpy()
    ll_prob = ll[0, 0]
    if float(ll_prob.min()) < 0.0 or float(ll_prob.max()) > 1.0:
        ll_prob = torch.sigmoid(ll_prob)
    ll_mask = (ll_prob > 0.5).detach().cpu().numpy()

    da_mask_np = _unletterbox_binary_mask(da_mask, meta)
    ll_mask_np = _unletterbox_binary_mask(ll_mask, meta)

    vis = overlay_mask(base_bgr, da_mask_np, color=(0, 200, 0), alpha=da_alpha)
    vis = overlay_mask(vis, ll_mask_np, color=(0, 0, 255), alpha=lane_alpha)
    vis = draw_boxes(vis, boxes.cpu(), scores.cpu(), classes.cpu(), names=None)
    cv2.putText(vis, "PV2: DA(green) + Lane(red)", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    return vis


def render_pv26(
    ckpt_path: Path,
    input_tensor: torch.Tensor,
    base_bgr: np.ndarray,
    meta: LetterboxMeta,
    conf_thres: float,
    da_alpha: float,
    lane_alpha: float,
    lane_thres: float,
    device: torch.device,
) -> np.ndarray:
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    model = PV26MultiHeadYOLO26(num_det_classes=len(DET_CLASSES_CANONICAL), yolo26_cfg="yolo26n.yaml").to(device).eval()
    model.load_state_dict(ckpt["model_state"], strict=True)

    with torch.no_grad():
        out = model(input_tensor.to(device))

    det = out.det[0][0] if isinstance(out.det, tuple) else out.det[0]
    if det.numel() > 0:
        conf = det[:, 4]
        keep = conf > conf_thres
        det = det[keep]
    if det.numel() > 0:
        boxes = det[:, :4].detach().cpu()
        scores = det[:, 4].detach().cpu()
        classes = det[:, 5].long().detach().cpu()
    else:
        boxes = torch.zeros((0, 4), dtype=torch.float32)
        scores = torch.zeros((0,), dtype=torch.float32)
        classes = torch.zeros((0,), dtype=torch.long)

    boxes = _unletterbox_boxes_xyxy(boxes, meta)
    da_prob = torch.sigmoid(out.da[0, 0]).detach().cpu().numpy()
    lane_prob = torch.sigmoid(out.rm[0, 0])
    lane_prob_np = lane_prob.detach().cpu().numpy()
    if float(lane_thres) > 0.0:
        lane_prob_np = np.where(lane_prob_np > float(lane_thres), lane_prob_np, 0.0)

    da_prob_np = _unletterbox_prob_map(da_prob, meta)
    lane_prob_np = _unletterbox_prob_map(lane_prob_np, meta)

    vis = overlay_prob(base_bgr, da_prob_np, color=(0, 180, 0), max_alpha=da_alpha)
    vis = overlay_prob(vis, lane_prob_np, color=(0, 0, 255), max_alpha=lane_alpha)
    vis = draw_boxes(vis, boxes, scores, classes, names=[c.name for c in DET_CLASSES_CANONICAL])
    cv2.putText(
        vis,
        f"PV26: DA(green) + Lane(red,t>{float(lane_thres):.2f})",
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return vis


def main() -> int:
    args = parse_args()
    os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp")
    device = pick_device(args.device)

    in_h = int(args.input_height) if args.input_height is not None else int(args.imgsz)
    in_w = int(args.input_width) if args.input_width is not None else int(args.imgsz)

    base_bgr, x, meta = read_image(args.image, in_h=in_h, in_w=in_w)
    pv2_img = render_pv2(args.pv2_weight, x, base_bgr, meta, args.conf_thres, args.iou_thres, args.da_alpha, args.lane_alpha, device)
    pv26_img = render_pv26(
        args.pv26_weight,
        x,
        base_bgr,
        meta,
        args.conf_thres,
        args.da_alpha,
        args.lane_alpha,
        args.pv26_lane_thres,
        device,
    )

    args.out_pv2.parent.mkdir(parents=True, exist_ok=True)
    args.out_pv26.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out_pv2), pv2_img)
    cv2.imwrite(str(args.out_pv26), pv26_img)
    print(f"saved: {args.out_pv2}")
    print(f"saved: {args.out_pv26}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
