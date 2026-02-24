from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

import torch
from torch import Tensor
from torch.utils.data import Dataset

from .masks import IGNORE_VALUE, make_all_ignore_mask, validate_binary_mask_u8
from .manifest import MANIFEST_COLUMNS, validate_manifest_row_basic


@dataclass(frozen=True)
class LetterboxSpec:
    """
    Deterministic letterbox spec.

    PRD default:
      - input: 960x544
      - BDD images commonly: 1280x720 (scale=0.75 -> 960x540, pad_y=2+2)
    """

    out_width: int = 960
    out_height: int = 544
    image_pad_value: int = 114
    mask_pad_value: int = IGNORE_VALUE


@dataclass(frozen=True)
class Pv26Sample:
    sample_id: str
    split: str
    # Image: float32 [3, H, W], 0..1 (letterboxed)
    image: Tensor
    # Detection targets: float32 [N, 5] with YOLO format (cls, cx, cy, w, h) normalized to letterboxed image.
    det_yolo: Tensor
    # Seg masks: uint8 in {0,1,255(ignore)} (letterboxed)
    da_mask: Tensor  # [H, W]
    rm_mask: Tensor  # [3, H, W] channels: lane_marker, road_marker_non_lane, stop_line
    # Availability flags
    has_det: int
    has_da: int
    has_rm_lane_marker: int
    has_rm_road_marker_non_lane: int
    has_rm_stop_line: int
    det_label_scope: str
    det_annotated_class_ids: str


def _parse_bool01(v: str) -> int:
    return 1 if str(v).strip() == "1" else 0


def _read_manifest_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        rows = list(r)

    # Light schema validation (fast, no filesystem touch).
    for idx, row in enumerate(rows[:1000]):  # avoid O(N) string building on huge manifests
        errs = validate_manifest_row_basic(row)
        if errs:
            raise ValueError(f"manifest schema error at row={idx}: {errs[:10]}")
    return rows


def _load_yolo_txt(path: Path) -> np.ndarray:
    """
    Returns:
      float32 array [N, 5] as (cls, cx, cy, w, h), normalized.
    """
    if not path.exists():
        return np.zeros((0, 5), dtype=np.float32)
    txt = path.read_text(encoding="utf-8").strip()
    if not txt:
        return np.zeros((0, 5), dtype=np.float32)
    out: List[List[float]] = []
    for line in txt.splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        try:
            cls = int(parts[0])
            cx, cy, w, h = (float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))
        except Exception:
            continue
        out.append([float(cls), cx, cy, w, h])
    if not out:
        return np.zeros((0, 5), dtype=np.float32)
    return np.asarray(out, dtype=np.float32)


def _letterbox_params(in_w: int, in_h: int, out_w: int, out_h: int) -> Tuple[float, int, int, int, int]:
    """
    Returns:
      scale, new_w, new_h, pad_x, pad_y (pad applied on left/top; right/bottom is inferred)
    """
    if in_w <= 0 or in_h <= 0:
        raise ValueError("in_w/in_h must be positive")
    scale = min(out_w / float(in_w), out_h / float(in_h))
    new_w = int(round(in_w * scale))
    new_h = int(round(in_h * scale))
    pad_x = (out_w - new_w) // 2
    pad_y = (out_h - new_h) // 2
    return scale, new_w, new_h, pad_x, pad_y


def _letterbox_image(
    image: Image.Image,
    *,
    out_w: int,
    out_h: int,
    pad_value: int,
) -> Image.Image:
    in_w, in_h = image.size
    scale, new_w, new_h, pad_x, pad_y = _letterbox_params(in_w, in_h, out_w, out_h)
    resized = image.resize((new_w, new_h), resample=Image.BILINEAR)
    canvas = Image.new("RGB", (out_w, out_h), color=(pad_value, pad_value, pad_value))
    canvas.paste(resized, (pad_x, pad_y))
    return canvas


def _letterbox_mask_u8(
    mask_u8: np.ndarray,
    *,
    in_w: int,
    in_h: int,
    out_w: int,
    out_h: int,
    pad_value: int,
) -> np.ndarray:
    if mask_u8.shape != (in_h, in_w):
        raise ValueError(f"mask size mismatch: expected={(in_h, in_w)} got={mask_u8.shape}")
    validate_binary_mask_u8(mask_u8, allow_ignore=True, name="mask")

    scale, new_w, new_h, pad_x, pad_y = _letterbox_params(in_w, in_h, out_w, out_h)
    # uint8 index mask: nearest interpolation
    im = Image.fromarray(mask_u8, mode="L")
    resized = im.resize((new_w, new_h), resample=Image.NEAREST)
    canvas = Image.new("L", (out_w, out_h), color=int(pad_value))
    canvas.paste(resized, (pad_x, pad_y))
    out = np.array(canvas, dtype=np.uint8)
    validate_binary_mask_u8(out, allow_ignore=True, name="mask_letterboxed")
    return out


def _letterbox_det_yolo(
    det_yolo: np.ndarray,
    *,
    in_w: int,
    in_h: int,
    out_w: int,
    out_h: int,
) -> np.ndarray:
    if det_yolo.size == 0:
        return det_yolo.astype(np.float32, copy=False)
    scale, new_w, new_h, pad_x, pad_y = _letterbox_params(in_w, in_h, out_w, out_h)

    out = det_yolo.copy().astype(np.float32, copy=False)
    cls = out[:, 0:1]
    cx = out[:, 1] * in_w
    cy = out[:, 2] * in_h
    bw = out[:, 3] * in_w
    bh = out[:, 4] * in_h

    # center in letterboxed canvas
    cx = cx * scale + pad_x
    cy = cy * scale + pad_y
    bw = bw * scale
    bh = bh * scale

    out[:, 0:1] = cls
    out[:, 1] = np.clip(cx / float(out_w), 0.0, 1.0)
    out[:, 2] = np.clip(cy / float(out_h), 0.0, 1.0)
    out[:, 3] = np.clip(bw / float(out_w), 0.0, 1.0)
    out[:, 4] = np.clip(bh / float(out_h), 0.0, 1.0)
    return out


class Pv26ManifestDataset(Dataset[Pv26Sample]):
    """
    Minimal manifest-driven PyTorch Dataset.

    Notes:
    - Always uses meta/split_manifest.csv as the source of truth.
    - Applies deterministic letterbox by default (PRD: 960x544).
    - Handles partial-label policy by returning masks even when unsupervised
      (training-time loss masking is expected to use has_* flags).
    """

    def __init__(
        self,
        *,
        dataset_root: Path,
        splits: Sequence[str] = ("train",),
        letterbox: Optional[LetterboxSpec] = LetterboxSpec(),
    ):
        self.dataset_root = Path(dataset_root)
        self.splits = tuple(s.strip().lower() for s in splits if s.strip())
        if not self.splits:
            raise ValueError("splits must be non-empty")
        bad = [s for s in self.splits if s not in {"train", "val", "test"}]
        if bad:
            raise ValueError(f"invalid splits: {bad}")

        manifest_path = self.dataset_root / "meta" / "split_manifest.csv"
        self.rows = [r for r in _read_manifest_rows(manifest_path) if r.get("split", "") in set(self.splits)]
        self.letterbox = letterbox

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Pv26Sample:
        row = self.rows[int(idx)]
        # Ensure required columns exist (defensive)
        for k in MANIFEST_COLUMNS:
            if k not in row:
                raise KeyError(f"manifest missing column: {k}")

        sample_id = row["sample_id"]
        split = row["split"]
        in_w = int(row["width"])
        in_h = int(row["height"])

        img_path = self.dataset_root / row["image_relpath"]
        with Image.open(img_path) as im:
            image = im.convert("RGB")

        det_path = self.dataset_root / row["det_relpath"]
        det_yolo = _load_yolo_txt(det_path)

        has_det = _parse_bool01(row["has_det"])
        has_da = _parse_bool01(row["has_da"])
        has_rm_lane = _parse_bool01(row["has_rm_lane_marker"])
        has_rm_road = _parse_bool01(row["has_rm_road_marker_non_lane"])
        has_rm_stop = _parse_bool01(row["has_rm_stop_line"])
        det_scope = row["det_label_scope"]
        det_annotated = row["det_annotated_class_ids"]

        # Masks: if unsupervised, we still return an all-ignore mask of the right size.
        if has_da:
            da_path = self.dataset_root / row["da_relpath"]
            da_u8 = np.array(Image.open(da_path), dtype=np.uint8)
        else:
            da_u8 = make_all_ignore_mask(in_h, in_w)

        rm_lane_path = self.dataset_root / row["rm_lane_marker_relpath"]
        rm_road_path = self.dataset_root / row["rm_road_marker_non_lane_relpath"]
        rm_stop_path = self.dataset_root / row["rm_stop_line_relpath"]

        rm_lane_u8 = np.array(Image.open(rm_lane_path), dtype=np.uint8) if has_rm_lane else make_all_ignore_mask(in_h, in_w)
        rm_road_u8 = np.array(Image.open(rm_road_path), dtype=np.uint8) if has_rm_road else make_all_ignore_mask(in_h, in_w)
        rm_stop_u8 = np.array(Image.open(rm_stop_path), dtype=np.uint8) if has_rm_stop else make_all_ignore_mask(in_h, in_w)

        # Apply letterbox (default PRD sizes)
        if self.letterbox is not None:
            lb = self.letterbox
            image = _letterbox_image(image, out_w=lb.out_width, out_h=lb.out_height, pad_value=int(lb.image_pad_value))
            da_u8 = _letterbox_mask_u8(
                da_u8,
                in_w=in_w,
                in_h=in_h,
                out_w=lb.out_width,
                out_h=lb.out_height,
                pad_value=int(lb.mask_pad_value),
            )
            rm_lane_u8 = _letterbox_mask_u8(
                rm_lane_u8,
                in_w=in_w,
                in_h=in_h,
                out_w=lb.out_width,
                out_h=lb.out_height,
                pad_value=int(lb.mask_pad_value),
            )
            rm_road_u8 = _letterbox_mask_u8(
                rm_road_u8,
                in_w=in_w,
                in_h=in_h,
                out_w=lb.out_width,
                out_h=lb.out_height,
                pad_value=int(lb.mask_pad_value),
            )
            rm_stop_u8 = _letterbox_mask_u8(
                rm_stop_u8,
                in_w=in_w,
                in_h=in_h,
                out_w=lb.out_width,
                out_h=lb.out_height,
                pad_value=int(lb.mask_pad_value),
            )
            det_yolo = _letterbox_det_yolo(det_yolo, in_w=in_w, in_h=in_h, out_w=lb.out_width, out_h=lb.out_height)

        # Convert to torch
        img_np = np.array(image, dtype=np.float32) / 255.0  # [H,W,3]
        img_t = torch.from_numpy(img_np).permute(2, 0, 1).contiguous()  # [3,H,W]

        det_t = torch.from_numpy(det_yolo).to(torch.float32) if has_det else torch.zeros((0, 5), dtype=torch.float32)
        da_t = torch.from_numpy(da_u8).to(torch.uint8)
        rm_t = torch.stack(
            [
                torch.from_numpy(rm_lane_u8).to(torch.uint8),
                torch.from_numpy(rm_road_u8).to(torch.uint8),
                torch.from_numpy(rm_stop_u8).to(torch.uint8),
            ],
            dim=0,
        )

        return Pv26Sample(
            sample_id=sample_id,
            split=split,
            image=img_t,
            det_yolo=det_t,
            da_mask=da_t,
            rm_mask=rm_t,
            has_det=has_det,
            has_da=has_da,
            has_rm_lane_marker=has_rm_lane,
            has_rm_road_marker_non_lane=has_rm_road,
            has_rm_stop_line=has_rm_stop,
            det_label_scope=det_scope,
            det_annotated_class_ids=det_annotated,
        )
