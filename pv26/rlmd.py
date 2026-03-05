from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from .masks import (
    IGNORE_VALUE,
    LANE_SUBCLASS_WHITE_DASHED,
    LANE_SUBCLASS_WHITE_SOLID,
    LANE_SUBCLASS_YELLOW_DASHED,
    LANE_SUBCLASS_YELLOW_SOLID,
)


@dataclass(frozen=True)
class RlmdRgbClass:
    class_id: int
    name: str
    rgb: Tuple[int, int, int]

    @property
    def code(self) -> int:
        r, g, b = self.rgb
        return (int(r) << 16) | (int(g) << 8) | int(b)


RLMD_STOP_LINE_NAME = "stop line"

RLMD_LANE_MARKER_NAMES = {
    "solid single white",
    "solid single yellow",
    "solid single red",
    "solid double white",
    "solid double yellow",
    "dashed single white",
    "dashed single yellow",
    "channelizing line",
}

RLMD_LANE_SUBCLASS_BY_NAME: Dict[str, int] = {
    "solid single white": LANE_SUBCLASS_WHITE_SOLID,
    "solid double white": LANE_SUBCLASS_WHITE_SOLID,
    "dashed single white": LANE_SUBCLASS_WHITE_DASHED,
    "solid single yellow": LANE_SUBCLASS_YELLOW_SOLID,
    "solid double yellow": LANE_SUBCLASS_YELLOW_SOLID,
    "dashed single yellow": LANE_SUBCLASS_YELLOW_DASHED,
}


def load_rlmd_palette(csv_path: Path) -> Dict[int, RlmdRgbClass]:
    """
    Load RLMD palette CSV (id,name,r,g,b) as code->class mapping.
    """
    palette: Dict[int, RlmdRgbClass] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            class_id = int(row[0])
            name = str(row[1]).strip()
            r, g, b = int(row[2]), int(row[3]), int(row[4])
            rc = RlmdRgbClass(class_id=class_id, name=name, rgb=(r, g, b))
            palette[rc.code] = rc
    return palette


def rgb_to_code_u32(img_rgb_u8: np.ndarray) -> np.ndarray:
    """
    (H,W,3) uint8 RGB -> (H,W) uint32 packed code (r<<16|g<<8|b).
    """
    if img_rgb_u8.ndim != 3 or img_rgb_u8.shape[2] != 3:
        raise ValueError(f"expected RGB mask shape (H,W,3), got {img_rgb_u8.shape}")
    r = img_rgb_u8[..., 0].astype(np.uint32)
    g = img_rgb_u8[..., 1].astype(np.uint32)
    b = img_rgb_u8[..., 2].astype(np.uint32)
    return (r << 16) | (g << 8) | b


def rlmd_code_mask_to_pv26_rm_masks(
    code_u32: np.ndarray, *, palette: Dict[int, RlmdRgbClass]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Convert RLMD RGB-code mask into PV26 Type-A road-marking masks.

    Returns:
      rm_lane_marker, rm_road_marker_non_lane, rm_stop_line, rm_lane_subclass, unknown_pixels

    Unknown colors are mapped to ignore(255) in all output masks.
    """
    if code_u32.ndim != 2:
        raise ValueError(f"expected code mask shape (H,W), got {code_u32.shape}")

    h, w = code_u32.shape
    rm_lane = np.zeros((h, w), dtype=np.uint8)
    rm_road = np.zeros((h, w), dtype=np.uint8)
    rm_stop = np.zeros((h, w), dtype=np.uint8)
    rm_lane_sub = np.zeros((h, w), dtype=np.uint8)

    known = np.zeros((h, w), dtype=bool)

    for code, cls in palette.items():
        mask = code_u32 == np.uint32(code)
        if not np.any(mask):
            continue
        known |= mask
        name = cls.name.strip().lower()
        if name == "background":
            continue
        if name == RLMD_STOP_LINE_NAME:
            rm_stop[mask] = 1
            rm_road[mask] = 1
            continue
        if name in RLMD_LANE_MARKER_NAMES:
            rm_lane[mask] = 1
            sub = RLMD_LANE_SUBCLASS_BY_NAME.get(name)
            if sub is not None:
                rm_lane_sub[mask] = int(sub)
            continue

        rm_road[mask] = 1

    rm_lane_sub[(rm_lane == 1) & (rm_lane_sub == 0)] = IGNORE_VALUE

    unknown = ~known
    unknown_pixels = int(np.count_nonzero(unknown))
    if unknown_pixels > 0:
        rm_lane[unknown] = IGNORE_VALUE
        rm_road[unknown] = IGNORE_VALUE
        rm_stop[unknown] = IGNORE_VALUE
        rm_lane_sub[unknown] = IGNORE_VALUE

    return rm_lane, rm_road, rm_stop, rm_lane_sub, unknown_pixels

