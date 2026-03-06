#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
import logging
import os
import sys
import time
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
from torchvision.ops import nms

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pv26.dataset.labels import DET_CLASSES_CANONICAL
from pv26.model.multitask_yolo26 import PV26MultiHeadYOLO26

LOGGER = logging.getLogger("render_weights_example")


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
    p.add_argument(
        "--pv26-lane-thres",
        type=float,
        default=0.0,
        help="Optional floor for PV26 lane-subclass confidence before overlay",
    )
    p.add_argument(
        "--pv26-rm-lane-thres",
        type=float,
        default=0.5,
        help="RM lane_marker threshold used to gate lane-subclass (default: 0.5)",
    )
    p.add_argument("--out-pv2", type=Path, default=Path("datasets/weights/result_pv2.jpg"))
    p.add_argument("--out-pv26", type=Path, default=Path("datasets/weights/result_pv26.jpg"))
    p.add_argument("--out-pv26-verbose", type=Path, default=Path("datasets/weights/result_pv26_verbose.jpg"))
    p.add_argument("--out-pv26-rm-lane", type=Path, default=Path("datasets/weights/result_pv26_rm_lane.jpg"))
    p.add_argument("--quiet", action="store_true", help="상세 진행 로그 출력 비활성화")
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


@dataclass(frozen=True)
class PV26RenderData:
    boxes: torch.Tensor
    scores: torch.Tensor
    classes: torch.Tensor
    da_prob_np: np.ndarray
    rm_lane_prob_np: np.ndarray
    lane_cls_np: np.ndarray
    lane_conf_np: np.ndarray
    det_names: list[str]
    det_colors: list[tuple[int, int, int]]


def pick_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def setup_logging(*, quiet: bool = False) -> None:
    level = logging.WARNING if quiet else logging.INFO
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    if not quiet:
        LOGGER.info("상세 진행 로그를 활성화했습니다.")


def log_step_start(step_name: str) -> float:
    LOGGER.info("▶ %s", step_name)
    return time.perf_counter()


def log_step_done(step_name: str, started_at: float) -> None:
    elapsed = time.perf_counter() - started_at
    LOGGER.info("✓ %s (%.2f초)", step_name, elapsed)


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
    t0 = log_step_start(f"입력 이미지 로드 및 전처리: {path}")
    orig_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if orig_bgr is None:
        raise FileNotFoundError(f"failed to load image: {path}")
    model_bgr, meta = _letterbox_image_for_model(orig_bgr, in_h=in_h, in_w=in_w, pad_value=114)
    rgb = cv2.cvtColor(model_bgr, cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    LOGGER.info(
        "입력 크기 원본=%dx%d, 모델입력=%dx%d, resize=%dx%d, pad=(y:%d, x:%d), scale=%.6f",
        meta.orig_w,
        meta.orig_h,
        meta.in_w,
        meta.in_h,
        meta.new_w,
        meta.new_h,
        meta.pad_y,
        meta.pad_x,
        meta.scale,
    )
    log_step_done("입력 이미지 로드 및 전처리", t0)
    return orig_bgr, x, meta


def overlay_mask(base: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float = 0.35) -> np.ndarray:
    out = base.copy()
    overlay = np.zeros_like(out, dtype=np.uint8)
    overlay[:, :] = np.array(color, dtype=np.uint8)
    sel = mask.astype(bool)
    if not np.any(sel):
        return out

    blended = cv2.addWeighted(base[sel], 1.0 - alpha, overlay[sel], alpha, 0)
    if blended is None:
        # OpenCV returns None for zero-length slices on some builds.
        blended = (
            base[sel].astype(np.float32) * (1.0 - alpha)
            + overlay[sel].astype(np.float32) * alpha
        ).astype(np.uint8)
    out[sel] = blended
    return out


def overlay_prob(base: np.ndarray, prob: np.ndarray, color: tuple[int, int, int], max_alpha: float) -> np.ndarray:
    p = np.clip(prob.astype(np.float32), 0.0, 1.0)
    a = np.clip(p * float(max_alpha), 0.0, 1.0)[..., None]
    color_arr = np.array(color, dtype=np.float32).reshape(1, 1, 3)
    out = base.astype(np.float32) * (1.0 - a) + color_arr * a
    return np.clip(out, 0.0, 255.0).astype(np.uint8)


def overlay_alpha_map(base: np.ndarray, alpha_map: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    a = np.clip(alpha_map.astype(np.float32), 0.0, 1.0)[..., None]
    color_arr = np.array(color, dtype=np.float32).reshape(1, 1, 3)
    out = base.astype(np.float32) * (1.0 - a) + color_arr * a
    return np.clip(out, 0.0, 255.0).astype(np.uint8)


def _hsv_to_bgr(h: float, s: float, v: float) -> tuple[int, int, int]:
    rgb = np.array(cv2.cvtColor(np.uint8([[[int(h * 179.0), int(s * 255.0), int(v * 255.0)]]]), cv2.COLOR_HSV2BGR)[0, 0])
    return int(rgb[0]), int(rgb[1]), int(rgb[2])


def build_det_class_colors(num_classes: int) -> list[tuple[int, int, int]]:
    # Deterministic distinct-ish palette in BGR.
    out: list[tuple[int, int, int]] = []
    for i in range(max(1, int(num_classes))):
        h = ((i * 0.1618033989) % 1.0)
        out.append(_hsv_to_bgr(h, 0.75, 0.95))
    return out


LANE_SUBCLASS_INFO: list[tuple[int, str, tuple[int, int, int]]] = [
    (1, "lane_white_solid", (255, 170, 60)),
    (2, "lane_white_dashed", (255, 235, 140)),
    (3, "lane_yellow_solid", (30, 210, 255)),
    (4, "lane_yellow_dashed", (0, 255, 255)),
]
DA_COLOR_BGR: tuple[int, int, int] = (40, 180, 40)


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


def _unletterbox_class_map(class_map: np.ndarray, meta: LetterboxMeta) -> np.ndarray:
    m = class_map.astype(np.uint8)
    crop = m[meta.pad_y : meta.pad_y + meta.new_h, meta.pad_x : meta.pad_x + meta.new_w]
    out = cv2.resize(crop, (meta.orig_w, meta.orig_h), interpolation=cv2.INTER_NEAREST)
    return out.astype(np.uint8)


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


def draw_od_boxes_colored(
    image: np.ndarray,
    boxes_xyxy: torch.Tensor,
    scores: torch.Tensor,
    classes: torch.Tensor,
    names: list[str],
    class_colors: list[tuple[int, int, int]],
) -> np.ndarray:
    out = image.copy()
    for i in range(int(boxes_xyxy.shape[0])):
        x1, y1, x2, y2 = [int(v) for v in boxes_xyxy[i].tolist()]
        conf = float(scores[i].item())
        cls = int(classes[i].item())
        color = class_colors[cls % len(class_colors)] if class_colors else (0, 165, 255)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cls_name = names[cls] if 0 <= cls < len(names) else str(cls)
        label = f"{cls_name} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        ty1 = max(0, y1 - th - 6)
        ty2 = max(0, y1)
        tx2 = min(out.shape[1] - 1, x1 + tw + 4)
        cv2.rectangle(out, (x1, ty1), (tx2, ty2), color, -1)
        cv2.putText(out, label, (x1 + 2, ty2 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (20, 20, 20), 1, cv2.LINE_AA)
    return out


def draw_legend(
    image: np.ndarray,
    *,
    da_color: tuple[int, int, int],
    lane_info: list[tuple[int, str, tuple[int, int, int]]],
    det_names: list[str],
    det_colors: list[tuple[int, int, int]],
    lane_thres: float,
) -> np.ndarray:
    out = image.copy()
    entries: list[tuple[str, tuple[int, int, int] | None]] = [
        ("DA overlay", da_color),
        ("Lane subclasses", None),
    ]
    for _cid, lname, lcolor in lane_info:
        entries.append((f"  {lname}", lcolor))
    entries.append((f"Lane prob thres: {lane_thres:.2f}", None))
    entries.append(("OD boxes (class color)", None))
    for i, n in enumerate(det_names):
        entries.append((f"  {n}", det_colors[i % len(det_colors)] if det_colors else None))

    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = 0.38
    th = 1
    pad = 8
    line_h = 14
    sw = 10
    gap = 6
    text_w = 0
    for text, _c in entries:
        (tw, _th), _ = cv2.getTextSize(text, font, fs, th)
        if tw > text_w:
            text_w = tw
    panel_w = pad * 2 + sw + gap + text_w
    panel_h = pad * 2 + line_h * len(entries)
    x0, y0 = 8, 8
    x1, y1 = min(out.shape[1] - 1, x0 + panel_w), min(out.shape[0] - 1, y0 + panel_h)

    panel = out.copy()
    cv2.rectangle(panel, (x0, y0), (x1, y1), (24, 24, 24), -1)
    out = cv2.addWeighted(panel, 0.65, out, 0.35, 0)

    y = y0 + pad + 11
    for text, c in entries:
        if c is not None:
            cv2.rectangle(out, (x0 + pad, y - 8), (x0 + pad + sw, y + 2), c, -1)
        cv2.putText(out, text, (x0 + pad + sw + gap, y), font, fs, (235, 235, 235), th, cv2.LINE_AA)
        y += line_h
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
    t_total = log_step_start(f"PV2 렌더링 시작 (weight={model_path})")
    t_load = log_step_start("PV2 TorchScript 모델 로드")
    model = torch.jit.load(str(model_path), map_location=device).eval()
    log_step_done("PV2 TorchScript 모델 로드", t_load)
    t_infer = log_step_start("PV2 추론 실행")
    with torch.no_grad():
        out = model(input_tensor.to(device))
    log_step_done("PV2 추론 실행", t_infer)

    (pred_raw, anchor_grid), seg, ll = out
    in_h = int(input_tensor.shape[-2])
    in_w = int(input_tensor.shape[-1])
    LOGGER.info(
        "PV2 출력 텐서: pred_layers=%d, seg=%s, lane=%s",
        len(pred_raw),
        tuple(seg.shape),
        tuple(ll.shape),
    )

    t_post = log_step_start("PV2 후처리 (decode + NMS + 마스크)")
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
    LOGGER.info(
        "PV2 탐지 결과: boxes=%d, DA pixels=%d, Lane pixels=%d",
        int(boxes.shape[0]),
        int(np.count_nonzero(da_mask_np)),
        int(np.count_nonzero(ll_mask_np)),
    )
    log_step_done("PV2 후처리 (decode + NMS + 마스크)", t_post)

    t_vis = log_step_start("PV2 시각화 합성")
    vis = overlay_mask(base_bgr, da_mask_np, color=(0, 200, 0), alpha=da_alpha)
    vis = overlay_mask(vis, ll_mask_np, color=(0, 0, 255), alpha=lane_alpha)
    vis = draw_boxes(vis, boxes.cpu(), scores.cpu(), classes.cpu(), names=None)
    cv2.putText(vis, "PV2: DA(green) + Lane(red)", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    log_step_done("PV2 시각화 합성", t_vis)
    log_step_done("PV2 렌더링 완료", t_total)
    return vis


def infer_pv26(
    ckpt_path: Path,
    input_tensor: torch.Tensor,
    meta: LetterboxMeta,
    conf_thres: float,
    rm_lane_thres: float,
    device: torch.device,
) -> PV26RenderData:
    t_total = log_step_start(f"PV26 추론/후처리 시작 (checkpoint={ckpt_path})")
    t_load = log_step_start("PV26 체크포인트 및 모델 로드")
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    model = PV26MultiHeadYOLO26(num_det_classes=len(DET_CLASSES_CANONICAL), yolo26_cfg="yolo26n.yaml").to(device).eval()
    try:
        model.load_state_dict(ckpt["model_state"], strict=True)
    except RuntimeError as ex:
        raise RuntimeError(
            "PV26 checkpoint/model mismatch. "
            "This renderer requires a checkpoint trained with the current PV26 model "
            "(including rm_lane_subclass_head)."
        ) from ex
    log_step_done("PV26 체크포인트 및 모델 로드", t_load)

    t_infer = log_step_start("PV26 추론 실행")
    with torch.no_grad():
        out = model(input_tensor.to(device))
    log_step_done("PV26 추론 실행", t_infer)
    LOGGER.info(
        "PV26 출력 텐서: da=%s, lane_subclass=%s",
        tuple(out.da.shape),
        tuple(out.rm_lane_subclass.shape),
    )

    det: torch.Tensor
    if isinstance(out.det, tuple) and len(out.det) >= 1 and isinstance(out.det[0], torch.Tensor):
        det = out.det[0][0]
    elif isinstance(out.det, torch.Tensor):
        det = out.det[0]
    else:
        det = torch.zeros((0, 6), dtype=torch.float32, device=device)

    if det.numel() > 0:
        conf = det[:, 4].to(torch.float32)
        keep = conf > float(conf_thres)
        det = det[keep]
    if det.numel() > 0:
        boxes = det[:, :4].detach().cpu()
        scores = det[:, 4].to(torch.float32).detach().cpu()
        classes = det[:, 5].to(torch.long).detach().cpu()
    else:
        boxes = torch.zeros((0, 4), dtype=torch.float32)
        scores = torch.zeros((0,), dtype=torch.float32)
        classes = torch.zeros((0,), dtype=torch.long)
    LOGGER.info("PV26 탐지 결과: conf_thres=%.3f, boxes=%d", float(conf_thres), int(boxes.shape[0]))

    boxes = _unletterbox_boxes_xyxy(boxes, meta)
    det_names = [c.name for c in DET_CLASSES_CANONICAL]
    det_colors = build_det_class_colors(len(det_names))

    da_prob_in = torch.sigmoid(out.da[0, 0]).detach().cpu().numpy()
    da_prob_np = _unletterbox_prob_map(da_prob_in, meta)
    rm_lane_prob_in = torch.sigmoid(out.rm[0, 0]).detach().cpu().numpy()
    rm_lane_prob_np = _unletterbox_prob_map(rm_lane_prob_in, meta)

    lane_logits = out.rm_lane_subclass[0]
    lane_soft = torch.softmax(lane_logits, dim=0).detach().cpu().numpy()
    lane_cls_in = np.argmax(lane_soft, axis=0).astype(np.uint8)
    lane_conf_in = np.max(lane_soft, axis=0).astype(np.float32)

    # Option-A policy: lane-subclass is meaningful only where RM lane_marker is positive.
    rm_lane_thres = float(rm_lane_thres)
    if not (0.0 <= rm_lane_thres <= 1.0):
        raise ValueError(f"--pv26-rm-lane-thres must be in [0,1], got {rm_lane_thres}")
    lane_gate_in = rm_lane_prob_in >= rm_lane_thres
    lane_cls_in = np.where(lane_gate_in, lane_cls_in, 0).astype(np.uint8)
    lane_conf_in = np.where(lane_gate_in, lane_conf_in, 0.0).astype(np.float32)

    lane_cls_np = _unletterbox_class_map(lane_cls_in, meta)
    lane_conf_np = _unletterbox_prob_map(lane_conf_in, meta)
    lane_summary: list[str] = []
    for cls_id, _name, _cls_color in LANE_SUBCLASS_INFO:
        m = lane_cls_np == int(cls_id)
        if not np.any(m):
            lane_summary.append(f"{_name}=0")
            continue
        lane_summary.append(f"{_name}={int(np.count_nonzero(m))}")
    LOGGER.info(
        "PV26 lane 픽셀 분포: %s",
        ", ".join(lane_summary) if lane_summary else "없음",
    )
    log_step_done("PV26 추론/후처리 완료", t_total)
    return PV26RenderData(
        boxes=boxes,
        scores=scores,
        classes=classes,
        da_prob_np=da_prob_np,
        rm_lane_prob_np=rm_lane_prob_np,
        lane_cls_np=lane_cls_np,
        lane_conf_np=lane_conf_np,
        det_names=det_names,
        det_colors=det_colors,
    )


def render_pv26_simple(
    *,
    base_bgr: np.ndarray,
    data: PV26RenderData,
    da_alpha: float,
    lane_alpha: float,
    lane_thres: float,
) -> np.ndarray:
    t_vis = log_step_start("PV26 단순 시각화 합성")
    da_mask_np = data.da_prob_np > 0.5
    lane_conf_floor = max(0.5, float(lane_thres))
    lane_mask_np = (data.lane_cls_np != 0) & (data.lane_conf_np >= lane_conf_floor)
    LOGGER.info(
        "PV26 단순 시각화: boxes=%d, DA pixels=%d, Lane pixels=%d, lane_thres=%.2f",
        int(data.boxes.shape[0]),
        int(np.count_nonzero(da_mask_np)),
        int(np.count_nonzero(lane_mask_np)),
        lane_conf_floor,
    )
    vis = overlay_mask(base_bgr, da_mask_np, color=(0, 200, 0), alpha=da_alpha)
    vis = overlay_mask(vis, lane_mask_np, color=(0, 0, 255), alpha=lane_alpha)
    vis = draw_boxes(vis, data.boxes, data.scores, data.classes, names=None)
    cv2.putText(vis, "PV26(simple): OD + DA(green) + Lane(red)", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    log_step_done("PV26 단순 시각화 합성", t_vis)
    return vis


def render_pv26_rm_lane(
    *,
    base_bgr: np.ndarray,
    data: PV26RenderData,
    da_alpha: float,
    lane_alpha: float,
    lane_thres: float,
) -> np.ndarray:
    t_vis = log_step_start("PV26 RM-lane 시각화 합성")
    da_mask_np = data.da_prob_np > 0.5
    lane_floor = max(0.5, float(lane_thres))
    lane_mask_np = data.rm_lane_prob_np >= lane_floor
    LOGGER.info(
        "PV26 RM-lane 시각화: boxes=%d, DA pixels=%d, Lane pixels=%d, lane_thres=%.2f",
        int(data.boxes.shape[0]),
        int(np.count_nonzero(da_mask_np)),
        int(np.count_nonzero(lane_mask_np)),
        lane_floor,
    )
    vis = overlay_mask(base_bgr, da_mask_np, color=(0, 200, 0), alpha=da_alpha)
    vis = overlay_mask(vis, lane_mask_np, color=(0, 0, 255), alpha=lane_alpha)
    vis = draw_boxes(vis, data.boxes, data.scores, data.classes, names=None)
    cv2.putText(vis, "PV26(rm-lane): OD + DA(green) + Lane(red)", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    log_step_done("PV26 RM-lane 시각화 합성", t_vis)
    return vis


def render_pv26_verbose(
    *,
    base_bgr: np.ndarray,
    data: PV26RenderData,
    da_alpha: float,
    lane_alpha: float,
    lane_thres: float,
) -> np.ndarray:
    t_vis = log_step_start("PV26 상세 시각화 합성")
    vis = overlay_prob(base_bgr, data.da_prob_np, color=DA_COLOR_BGR, max_alpha=da_alpha)
    for cls_id, _name, cls_color in LANE_SUBCLASS_INFO:
        m = data.lane_cls_np == int(cls_id)
        if not np.any(m):
            continue
        alpha_map = np.zeros_like(data.lane_conf_np, dtype=np.float32)
        alpha_map[m] = data.lane_conf_np[m] * float(lane_alpha)
        if float(lane_thres) > 0.0:
            alpha_map[m & (data.lane_conf_np < float(lane_thres))] = 0.0
        vis = overlay_alpha_map(vis, alpha_map, color=cls_color)

    vis = draw_od_boxes_colored(
        vis,
        data.boxes,
        data.scores,
        data.classes,
        names=data.det_names,
        class_colors=data.det_colors,
    )
    vis = draw_legend(
        vis,
        da_color=DA_COLOR_BGR,
        lane_info=LANE_SUBCLASS_INFO,
        det_names=data.det_names,
        det_colors=data.det_colors,
        lane_thres=float(lane_thres),
    )
    log_step_done("PV26 상세 시각화 합성", t_vis)
    return vis


def main() -> int:
    args = parse_args()
    setup_logging(quiet=bool(args.quiet))
    os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp")
    LOGGER.info("YOLO_CONFIG_DIR=%s", os.environ.get("YOLO_CONFIG_DIR"))
    LOGGER.info("입력 인자: %s", args)

    t_main = log_step_start("전체 파이프라인 시작")
    device = pick_device(args.device)
    LOGGER.info("사용 디바이스: %s", device)

    in_h = int(args.input_height) if args.input_height is not None else int(args.imgsz)
    in_w = int(args.input_width) if args.input_width is not None else int(args.imgsz)
    LOGGER.info("모델 입력 크기: width=%d, height=%d", in_w, in_h)

    base_bgr, x, meta = read_image(args.image, in_h=in_h, in_w=in_w)
    pv2_img = render_pv2(args.pv2_weight, x, base_bgr, meta, args.conf_thres, args.iou_thres, args.da_alpha, args.lane_alpha, device)
    pv26_data = infer_pv26(
        args.pv26_weight,
        x,
        meta,
        args.conf_thres,
        args.pv26_rm_lane_thres,
        device,
    )
    pv26_img = render_pv26_simple(
        base_bgr=base_bgr,
        data=pv26_data,
        da_alpha=args.da_alpha,
        lane_alpha=args.lane_alpha,
        lane_thres=args.pv26_lane_thres,
    )
    pv26_rm_lane_img = render_pv26_rm_lane(
        base_bgr=base_bgr,
        data=pv26_data,
        da_alpha=args.da_alpha,
        lane_alpha=args.lane_alpha,
        lane_thres=args.pv26_lane_thres,
    )
    pv26_verbose_img = render_pv26_verbose(
        base_bgr=base_bgr,
        data=pv26_data,
        da_alpha=args.da_alpha,
        lane_alpha=args.lane_alpha,
        lane_thres=args.pv26_lane_thres,
    )

    t_save = log_step_start("결과 이미지 저장")
    args.out_pv2.parent.mkdir(parents=True, exist_ok=True)
    args.out_pv26.parent.mkdir(parents=True, exist_ok=True)
    args.out_pv26_verbose.parent.mkdir(parents=True, exist_ok=True)
    args.out_pv26_rm_lane.parent.mkdir(parents=True, exist_ok=True)
    ok_pv2 = cv2.imwrite(str(args.out_pv2), pv2_img)
    ok_pv26 = cv2.imwrite(str(args.out_pv26), pv26_img)
    ok_pv26_rm_lane = cv2.imwrite(str(args.out_pv26_rm_lane), pv26_rm_lane_img)
    ok_pv26_verbose = cv2.imwrite(str(args.out_pv26_verbose), pv26_verbose_img)
    if not ok_pv2:
        raise RuntimeError(f"failed to save output image: {args.out_pv2}")
    if not ok_pv26:
        raise RuntimeError(f"failed to save output image: {args.out_pv26}")
    if not ok_pv26_rm_lane:
        raise RuntimeError(f"failed to save output image: {args.out_pv26_rm_lane}")
    if not ok_pv26_verbose:
        raise RuntimeError(f"failed to save output image: {args.out_pv26_verbose}")
    log_step_done("결과 이미지 저장", t_save)
    log_step_done("전체 파이프라인 완료", t_main)

    print(f"saved: {args.out_pv2}")
    print(f"saved: {args.out_pv26}")
    print(f"saved: {args.out_pv26_rm_lane}")
    print(f"saved: {args.out_pv26_verbose}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
