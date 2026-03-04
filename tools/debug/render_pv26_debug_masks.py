#!/usr/bin/env python3
"""
PV26 debug mask visualizer.

Renders index-PNG labels (uint8) as colorized PNGs + image overlays so humans can
quickly inspect PV26 segmentation masks.

Expected dataset structure (converted PV26 dataset):
  <dataset_root>/
    meta/split_manifest.csv
    images/<split>/*.jpg
    labels_seg_*/<split>/*.png

Dependencies: stdlib + numpy + Pillow.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

IGNORE_VALUE = 255
IGNORE_COLOR = (255, 0, 255)  # magenta


CHANNELS: Dict[str, Dict[str, str]] = {
    "da": {"mask_col": "da_relpath", "has_col": "has_da"},
    "rm_lane_marker": {"mask_col": "rm_lane_marker_relpath", "has_col": "has_rm_lane_marker"},
    "rm_lane_subclass": {"mask_col": "rm_lane_subclass_relpath", "has_col": "has_rm_lane_subclass"},
    "rm_road_marker_non_lane": {
        "mask_col": "rm_road_marker_non_lane_relpath",
        "has_col": "has_rm_road_marker_non_lane",
    },
    "rm_stop_line": {"mask_col": "rm_stop_line_relpath", "has_col": "has_rm_stop_line"},
}


@dataclass(frozen=True)
class ManifestRow:
    sample_id: str
    split: str
    image_relpath: str
    mask_relpath: str
    has_label: bool


def _hsv_to_rgb_u8(h: float, s: float, v: float) -> Tuple[int, int, int]:
    """Minimal HSV->RGB conversion (all channels in [0,1])."""
    if s <= 0.0:
        g = int(round(v * 255.0))
        return g, g, g
    h = h % 1.0
    i = int(h * 6.0)
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    i = i % 6
    if i == 0:
        r, g, b = v, t, p
    elif i == 1:
        r, g, b = q, v, p
    elif i == 2:
        r, g, b = p, v, t
    elif i == 3:
        r, g, b = p, q, v
    elif i == 4:
        r, g, b = t, p, v
    else:
        r, g, b = v, p, q
    return int(round(r * 255.0)), int(round(g * 255.0)), int(round(b * 255.0))


def build_color_lut() -> np.ndarray:
    """
    Build deterministic LUT for values 0..255 (uint8) -> RGB.

    - value 0 is black (background)
    - value 255 is magenta (ignore)
    - other values get a bright deterministic hue from their index
    """
    lut = np.zeros((256, 3), dtype=np.uint8)
    lut[0] = (0, 0, 0)
    lut[IGNORE_VALUE] = IGNORE_COLOR

    # Golden-ratio hue stepping spreads colors fairly uniformly.
    phi = 0.6180339887498949
    for idx in range(1, 255):
        h = (idx * phi) % 1.0
        r, g, b = _hsv_to_rgb_u8(h, s=0.85, v=0.95)
        # Avoid accidental collision with ignore magenta.
        if (r, g, b) == IGNORE_COLOR:
            r = (r + 1) % 256
        lut[idx] = (r, g, b)
    return lut


def write_lut_csv(out_path: Path, lut: np.ndarray) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["value", "r", "g", "b", "hex", "note"])
        for v in range(256):
            r, g, b = map(int, lut[v])
            note = ""
            if v == 0:
                note = "background"
            elif v == IGNORE_VALUE:
                note = "ignore"
            w.writerow([v, r, g, b, f"#{r:02x}{g:02x}{b:02x}", note])


def _load_manifest_rows(dataset_root: Path, split: str, channel: str) -> List[ManifestRow]:
    meta = dataset_root / "meta"
    manifest_path = meta / "split_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing manifest: {manifest_path}")

    if channel not in CHANNELS:
        raise KeyError(f"unknown channel: {channel}")
    mask_col = CHANNELS[channel]["mask_col"]
    has_col = CHANNELS[channel]["has_col"]

    rows: List[ManifestRow] = []
    with manifest_path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            if row.get("split", "") != split:
                continue
            sample_id = row.get("sample_id", "")
            img_rel = row.get("image_relpath", "")
            mask_rel = row.get(mask_col, "")
            has_label = row.get(has_col, "0") == "1"
            if not sample_id or not mask_rel:
                continue
            rows.append(
                ManifestRow(
                    sample_id=sample_id,
                    split=split,
                    image_relpath=img_rel,
                    mask_relpath=mask_rel,
                    has_label=has_label,
                )
            )
    return rows


def _load_u8_mask(path: Path) -> np.ndarray:
    arr = np.array(Image.open(path))
    if arr.ndim == 3:
        # Some PNG writers may store paletted/expanded images; keep first channel.
        arr = arr[..., 0]
    if arr.ndim != 2:
        raise ValueError(f"mask not single-channel: {path} shape={arr.shape}")
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8, copy=False)
    return arr


def _colorize_mask(mask_u8: np.ndarray, lut: np.ndarray) -> Image.Image:
    rgb = lut[mask_u8]  # (H,W,3) uint8
    return Image.fromarray(rgb, mode="RGB")


def _make_overlay(image_rgb: Image.Image, mask_u8: np.ndarray, lut: np.ndarray) -> Image.Image:
    """
    Overlay labels on top of the source image using a simple alpha rule:
      - value == 0: alpha 0
      - value == 255 (ignore): alpha 160
      - otherwise: alpha 140
    """
    if image_rgb.mode != "RGB":
        image_rgb = image_rgb.convert("RGB")
    w, h = image_rgb.size
    if mask_u8.shape != (h, w):
        raise ValueError(f"image/mask size mismatch image={(w,h)} mask={mask_u8.shape[::-1]}")

    rgb = lut[mask_u8]
    alpha = np.zeros(mask_u8.shape, dtype=np.uint8)
    alpha[(mask_u8 != 0) & (mask_u8 != IGNORE_VALUE)] = 140
    alpha[mask_u8 == IGNORE_VALUE] = 160
    rgba = np.dstack([rgb, alpha])

    base = image_rgb.convert("RGBA")
    over = Image.fromarray(rgba, mode="RGBA")
    return Image.alpha_composite(base, over)


def _stable_seed(base_seed: int, channel: str) -> int:
    # Stable across runs/interpreters (unlike built-in hash()).
    h = 0
    for b in channel.encode("utf-8"):
        h = (h * 131 + b) & 0xFFFFFFFF
    return (base_seed + int(h)) & 0xFFFFFFFF


def render_channel(
    *,
    dataset_root: Path,
    split: str,
    channel: str,
    num_samples: int,
    out_root: Path,
    seed: int,
    lut: np.ndarray,
) -> List[Path]:
    rows = _load_manifest_rows(dataset_root, split, channel)
    eligible = [r for r in rows if r.has_label]
    if not eligible:
        # Fallback: still allow rendering masks even when has_* is all 0
        eligible = rows

    rng = random.Random(_stable_seed(seed, channel))
    rng.shuffle(eligible)
    selected = eligible[: max(0, int(num_samples))]

    written: List[Path] = []
    for r in selected:
        mask_path = dataset_root / r.mask_relpath
        if not mask_path.exists():
            print(f"[pv26][warn] missing mask: {mask_path}", file=sys.stderr)
            continue

        mask_u8 = _load_u8_mask(mask_path)
        mask_vis = _colorize_mask(mask_u8, lut)

        out_dir = out_root / split / channel
        out_dir.mkdir(parents=True, exist_ok=True)
        mask_out = out_dir / f"{r.sample_id}__mask.png"
        mask_vis.save(mask_out)
        written.append(mask_out)

        if r.image_relpath:
            img_path = dataset_root / r.image_relpath
            if img_path.exists():
                with Image.open(img_path) as im:
                    overlay = _make_overlay(im.convert("RGB"), mask_u8, lut)
                overlay_out = out_dir / f"{r.sample_id}__overlay.png"
                overlay.save(overlay_out)
                written.append(overlay_out)
            else:
                print(f"[pv26][warn] missing image (skip overlay): {img_path}", file=sys.stderr)
        else:
            print(f"[pv26][warn] missing image_relpath in manifest (skip overlay) sample_id={r.sample_id}", file=sys.stderr)
    return written


def _parse_channels(s: str) -> List[str]:
    chans = [c.strip() for c in (s or "").split(",") if c.strip()]
    if not chans:
        raise ValueError("empty --channels")
    unknown = [c for c in chans if c not in CHANNELS]
    if unknown:
        raise ValueError(f"unknown channels: {unknown}. Allowed: {sorted(CHANNELS)}")
    return chans


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Render PV26 uint8 mask PNGs as debug visualizations.")
    p.add_argument("--dataset-root", type=Path, required=True, help="PV26 dataset root (converted format).")
    p.add_argument("--split", type=str, required=True, choices=["train", "val", "test"], help="Split to sample from.")
    p.add_argument(
        "--channels",
        type=str,
        required=True,
        help="Comma-separated list: da,rm_lane_marker,rm_road_marker_non_lane,rm_stop_line",
    )
    p.add_argument("--num-samples", type=int, default=10, help="Number of samples per channel (default: 10)")
    p.add_argument("--out-root", type=Path, default=Path("/tmp/pv26_mask_vis"), help="Output root (default: /tmp/pv26_mask_vis)")
    p.add_argument("--seed", type=int, default=42, help="RNG seed (default: 42)")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_argparser().parse_args(argv)
    dataset_root: Path = args.dataset_root
    out_root: Path = args.out_root
    split: str = args.split
    num_samples: int = int(args.num_samples)

    try:
        channels = _parse_channels(args.channels)
    except Exception as ex:
        print(f"[pv26] invalid --channels: {ex}", file=sys.stderr)
        return 2

    lut = build_color_lut()
    lut_path = out_root / "pv26_mask_lut.csv"
    write_lut_csv(lut_path, lut)

    all_written: List[Path] = [lut_path]
    for ch in channels:
        try:
            written = render_channel(
                dataset_root=dataset_root,
                split=split,
                channel=ch,
                num_samples=num_samples,
                out_root=out_root,
                seed=int(args.seed),
                lut=lut,
            )
        except Exception as ex:
            print(f"[pv26] failed channel={ch}: {ex}", file=sys.stderr)
            return 2
        all_written.extend(written)

    print(f"[pv26] wrote {len(all_written)} files under: {out_root}")
    for p in all_written[:2000]:
        print(p)
    if len(all_written) > 2000:
        print(f"[pv26] ... truncated ({len(all_written) - 2000} more)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
