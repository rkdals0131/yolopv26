from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw

from .masks import (
    IGNORE_VALUE,
    LANE_SUBCLASS_WHITE_DASHED,
    LANE_SUBCLASS_WHITE_SOLID,
    LANE_SUBCLASS_YELLOW_DASHED,
    LANE_SUBCLASS_YELLOW_SOLID,
)


ETRI_LABEL_OUT_OF_ROI = "out of roi"

ETRI_DA_LABELS = {"road"}

ETRI_LANE_SUBCLASS_BY_LABEL: Dict[str, int] = {
    "whsol": LANE_SUBCLASS_WHITE_SOLID,
    "whdot": LANE_SUBCLASS_WHITE_DASHED,
    "yesol": LANE_SUBCLASS_YELLOW_SOLID,
    "yedot": LANE_SUBCLASS_YELLOW_DASHED,
}

ETRI_LANE_LIKE_RE = re.compile(
    r"(whdot|whsol|yedot|yesol|bldot|blsol|guidance line|lane divider)",
    re.IGNORECASE,
)

ETRI_ROAD_MARK_NON_LANE_RE = re.compile(
    r"(general road mark|crosswalk|stop line|arrow|prohibition|number|slow|motor|bike icon|box junction|parking|speed bump|channelizing line|left|right|forward|straight|leftu|protection zone)",
    re.IGNORECASE,
)

ETRI_STOP_LINE_LABELS = {"stop line"}


@dataclass(frozen=True)
class EtriPolygonObject:
    label: str
    polygon: Any
    deleted: int = 0


def read_etri_polygon_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _clamp_point(x: float, y: float, *, width: int, height: int) -> Tuple[float, float]:
    x2 = min(max(float(x), 0.0), float(max(0, width - 1)))
    y2 = min(max(float(y), 0.0), float(max(0, height - 1)))
    return x2, y2


def _parse_polygon_points(poly: Any, *, width: int, height: int) -> List[Tuple[float, float]]:
    if not isinstance(poly, list) or not poly:
        return []
    out: List[Tuple[float, float]] = []
    for p in poly:
        if not isinstance(p, list) or len(p) < 2:
            continue
        try:
            x, y = float(p[0]), float(p[1])
        except Exception:
            continue
        out.append(_clamp_point(x, y, width=width, height=height))
    # PIL requires at least 2 points to render something; polygon fill needs >= 3.
    return out


def iter_etri_objects(data: Mapping[str, Any]) -> Iterable[EtriPolygonObject]:
    objs = data.get("objects")
    if not isinstance(objs, list):
        return
    for o in objs:
        if not isinstance(o, dict):
            continue
        label = o.get("label")
        if not isinstance(label, str) or not label.strip():
            continue
        poly = o.get("polygon")
        deleted = int(o.get("deleted", 0) or 0)
        yield EtriPolygonObject(label=label.strip(), polygon=poly, deleted=deleted)


def rasterize_etri_type_a_masks(
    data: Mapping[str, Any],
    *,
    width: int,
    height: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Rasterize a Cityscapes-like ETRI polygon JSON into PV26 Type-A masks.

    Returns:
      da_mask, rm_lane_marker, rm_road_marker_non_lane, rm_stop_line, rm_lane_subclass

    Notes:
    - `rm_lane_subclass` uses 255(ignore) for lane-like labels that cannot be mapped.
    - If an "out of roi" polygon exists, it is applied as ignore(255) to all masks.
    """
    if width <= 0 or height <= 0:
        raise ValueError("width/height must be positive")

    da = Image.new("L", (width, height), 0)
    lane = Image.new("L", (width, height), 0)
    road_mark = Image.new("L", (width, height), 0)
    stop = Image.new("L", (width, height), 0)
    lane_sub = Image.new("L", (width, height), 0)
    lane_sub_ign = Image.new("L", (width, height), 0)
    ignore = Image.new("L", (width, height), 0)

    da_draw = ImageDraw.Draw(da)
    lane_draw = ImageDraw.Draw(lane)
    road_draw = ImageDraw.Draw(road_mark)
    stop_draw = ImageDraw.Draw(stop)
    lane_sub_draw = ImageDraw.Draw(lane_sub)
    lane_sub_ign_draw = ImageDraw.Draw(lane_sub_ign)
    ign_draw = ImageDraw.Draw(ignore)

    for obj in iter_etri_objects(data):
        label_raw = obj.label
        label = label_raw.strip().lower()
        if label == ETRI_LABEL_OUT_OF_ROI:
            pts = _parse_polygon_points(obj.polygon, width=width, height=height)
            if len(pts) >= 3:
                ign_draw.polygon(pts, fill=1)
            continue
        if obj.deleted:
            continue

        pts = _parse_polygon_points(obj.polygon, width=width, height=height)
        if len(pts) < 3:
            continue

        if label in ETRI_DA_LABELS:
            da_draw.polygon(pts, fill=1)

        if label in ETRI_STOP_LINE_LABELS:
            stop_draw.polygon(pts, fill=1)
            road_draw.polygon(pts, fill=1)

        if ETRI_LANE_LIKE_RE.search(label):
            lane_draw.polygon(pts, fill=1)
            sub_id = ETRI_LANE_SUBCLASS_BY_LABEL.get(label)
            if sub_id is None:
                lane_sub_ign_draw.polygon(pts, fill=1)
            else:
                lane_sub_draw.polygon(pts, fill=int(sub_id))
            continue

        if ETRI_ROAD_MARK_NON_LANE_RE.search(label):
            road_draw.polygon(pts, fill=1)

    da_u8 = np.array(da, dtype=np.uint8)
    lane_u8 = np.array(lane, dtype=np.uint8)
    road_u8 = np.array(road_mark, dtype=np.uint8)
    stop_u8 = np.array(stop, dtype=np.uint8)

    lane_sub_u8 = np.array(lane_sub, dtype=np.uint8)
    lane_sub_ign_u8 = np.array(lane_sub_ign, dtype=np.uint8)
    lane_sub_u8[lane_sub_ign_u8 == 1] = IGNORE_VALUE

    ign_u8 = np.array(ignore, dtype=np.uint8)
    if np.any(ign_u8 == 1):
        da_u8[ign_u8 == 1] = IGNORE_VALUE
        lane_u8[ign_u8 == 1] = IGNORE_VALUE
        road_u8[ign_u8 == 1] = IGNORE_VALUE
        stop_u8[ign_u8 == 1] = IGNORE_VALUE
        lane_sub_u8[ign_u8 == 1] = IGNORE_VALUE

    return da_u8, lane_u8, road_u8, stop_u8, lane_sub_u8
