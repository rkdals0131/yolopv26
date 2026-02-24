from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw

from .constants import DET_NAME_TO_ID
from .masks import make_all_ignore_mask
from .yolo import BoxXYXY, format_yolo_line, xyxy_to_yolo_normalized


BDD_TO_CANONICAL_DET: Dict[str, str] = {
    # direct matches
    "car": "car",
    "bus": "bus",
    "truck": "truck",
    "motorcycle": "motorcycle",
    "bicycle": "bicycle",
    "bike": "bicycle",
    "person": "pedestrian",
    "pedestrian": "pedestrian",
    # map "rider" to pedestrian (Cityscapes policy equivalent, pragmatic)
    "rider": "pedestrian",
    # fixtures -> sign_pole bucket per spec
    "traffic sign": "sign_pole",
    "traffic light": "sign_pole",
    "pole": "sign_pole",
    "pole-like roadside fixture": "sign_pole",
    "sign": "sign_pole",
    "light": "sign_pole",
    # obstacle-like classes
    "traffic cone": "traffic_cone",
    "construction cone": "traffic_cone",
    "cone": "traffic_cone",
    "barrier": "barrier",
    "bollard": "bollard",
    # hazards (best-effort)
    "other vehicle": "road_obstacle",
    "train": "road_obstacle",
    "motor": "motorcycle",
}


BDD_LANE_MARKER_CATEGORIES = {
    "lane/single white",
    "lane/double white",
    "lane/single yellow",
    "lane/double yellow",
    "lane/single other",
}

BDD_ROAD_MARKER_NON_LANE_CATEGORIES = {
    "lane/crosswalk",
    "lane/road curb",
}

# BDD100K 100k labels commonly do not expose explicit stop-line class in this schema.
BDD_STOP_LINE_CATEGORIES = {
    "lane/stop line",
}


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def iter_bdd_label_records(labels_path_or_dir: Path) -> Iterator[Dict[str, Any]]:
    """
    Yield per-image label records.

    Supports:
      - A directory of per-image JSON files (each a dict)
      - A single JSON file containing a list[dict]
    """
    if labels_path_or_dir.is_dir():
        for p in sorted(labels_path_or_dir.rglob("*.json")):
            rec = _read_json(p)
            if not isinstance(rec, dict):
                raise ValueError(f"expected dict JSON in {p}, got {type(rec)}")
            rec["__json_path__"] = str(p)
            yield rec
        return

    recs = _read_json(labels_path_or_dir)
    if isinstance(recs, list):
        for r in recs:
            if not isinstance(r, dict):
                continue
            yield r
        return

    if isinstance(recs, dict):
        # Some schemas may store under a key.
        for k in ("frames", "images", "data"):
            v = recs.get(k)
            if isinstance(v, list):
                for r in v:
                    if isinstance(r, dict):
                        yield r
                return
        raise ValueError(f"unrecognized labels json schema in {labels_path_or_dir}")

    raise ValueError(f"unrecognized labels json type in {labels_path_or_dir}: {type(recs)}")


def parse_bdd_filename_for_sequence_and_frame(name: str) -> Tuple[str, str]:
    """
    Best-effort parsing for BDD-like filenames.

    Examples:
      day_city_001__000123.jpg -> (day_city_001, 000123)
      day_city_001_000123.jpg  -> (day_city_001, 000123)
      00a0f008-3c67908e.jpg    -> (00a0f008-3c67908e, 000000)
    """
    stem = Path(name).stem
    m = re.match(r"^(.*?)[_]{1,2}(\d+)$", stem)
    if m:
        seq, frame = m.group(1), m.group(2).zfill(6)
        return seq, frame
    return stem, "000000"


def bdd_record_to_image_name(rec: Mapping[str, Any]) -> Optional[str]:
    # Common keys in BDD-style schemas.
    for k in ("name", "image_name", "image", "file_name"):
        v = rec.get(k)
        if isinstance(v, str) and v:
            if Path(v).suffix:
                return v
            return f"{v}.jpg"
    return None


def _bdd_record_objects(rec: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    # BDD per-image JSON commonly stores object annotations under frames[0].objects.
    frames = rec.get("frames")
    if isinstance(frames, list) and frames:
        first = frames[0]
        if isinstance(first, dict):
            objs = first.get("objects")
            if isinstance(objs, list):
                return [o for o in objs if isinstance(o, dict)]

    # Some schemas expose labels directly.
    labels = rec.get("labels")
    if isinstance(labels, list):
        return [o for o in labels if isinstance(o, dict)]

    return []


def bdd_record_tags(rec: Mapping[str, Any]) -> Tuple[str, str, str]:
    """
    Map BDD attributes to (weather_tag, time_tag, scene_tag) in PV26 vocabulary.
    """
    attrs = rec.get("attributes") if isinstance(rec.get("attributes"), dict) else {}
    weather = str(attrs.get("weather", "")).lower()
    timeofday = str(attrs.get("timeofday", attrs.get("timeOfDay", ""))).lower()
    scene = str(attrs.get("scene", "")).lower()

    # weather_tag: dry|rain|snow|unknown
    if weather in {"clear", "overcast", "partly cloudy", "cloudy", "sunny"}:
        weather_tag = "dry"
    elif "rain" in weather:
        weather_tag = "rain"
    elif "snow" in weather:
        weather_tag = "snow"
    else:
        weather_tag = "unknown"

    # time_tag: day|night|dawn_dusk|unknown
    if timeofday in {"daytime", "day"}:
        time_tag = "day"
    elif timeofday in {"night"}:
        time_tag = "night"
    elif timeofday in {"dawn", "dusk", "dawn/dusk", "dawn_dusk"}:
        time_tag = "dawn_dusk"
    else:
        time_tag = "unknown"

    # scene_tag: open|tunnel|shadow|unknown
    if "tunnel" in scene:
        scene_tag = "tunnel"
    elif scene:
        scene_tag = "open"
    else:
        scene_tag = "unknown"

    return weather_tag, time_tag, scene_tag


def bdd_record_to_yolo_lines(
    rec: Mapping[str, Any],
    *,
    width: int,
    height: int,
    min_box_area_px: int = 0,
) -> List[str]:
    """
    Convert a BDD label record into YOLO txt lines using canonical det IDs.

    Unmapped categories are skipped.
    """
    labels = _bdd_record_objects(rec)
    if not labels:
        return []

    out: List[str] = []
    for lab in labels:
        if not isinstance(lab, dict):
            continue
        cat = lab.get("category")
        if not isinstance(cat, str):
            continue

        canonical_name = BDD_TO_CANONICAL_DET.get(cat.strip().lower())
        if canonical_name is None:
            # Try to normalize some common variants.
            c2 = cat.strip().lower().replace("_", " ")
            canonical_name = BDD_TO_CANONICAL_DET.get(c2)
        if canonical_name is None:
            continue

        class_id = DET_NAME_TO_ID[canonical_name]

        box2d = lab.get("box2d")
        if not isinstance(box2d, dict):
            continue
        try:
            x1 = float(box2d["x1"])
            y1 = float(box2d["y1"])
            x2 = float(box2d["x2"])
            y2 = float(box2d["y2"])
        except Exception:
            continue

        box = BoxXYXY(x1=x1, y1=y1, x2=x2, y2=y2).clip(width=width, height=height)
        if box.area_px() <= 0:
            continue
        if min_box_area_px > 0 and box.area_px() < float(min_box_area_px):
            # MVP policy exceptions (small-object-critical classes) can be added later.
            continue

        cx, cy, w, h = xyxy_to_yolo_normalized(box, width=width, height=height)
        # Hard clamp to [0,1] range due to rounding and clip edge cases.
        cx = min(1.0, max(0.0, cx))
        cy = min(1.0, max(0.0, cy))
        w = min(1.0, max(0.0, w))
        h = min(1.0, max(0.0, h))
        out.append(format_yolo_line(class_id, cx, cy, w, h))

    return out


def _poly2d_points(poly2d: Any) -> List[Tuple[float, float]]:
    """
    Parse BDD poly2d variants into a point list.
    Common form in BDD100K:
      [[x, y, type], [x, y, type], ...]
    """
    pts: List[Tuple[float, float]] = []
    if not isinstance(poly2d, list):
        return pts
    for p in poly2d:
        if isinstance(p, (list, tuple)) and len(p) >= 2:
            try:
                x = float(p[0])
                y = float(p[1])
            except Exception:
                continue
            pts.append((x, y))
        elif isinstance(p, dict):
            vs = p.get("vertices")
            if isinstance(vs, list):
                for v in vs:
                    if isinstance(v, (list, tuple)) and len(v) >= 2:
                        try:
                            pts.append((float(v[0]), float(v[1])))
                        except Exception:
                            continue
    return pts


def bdd_record_to_rm_masks(
    rec: Mapping[str, Any],
    *,
    width: int,
    height: int,
    line_width: int = 8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int, int]:
    """
    Rasterize BDD lane annotations into PV26 RM masks.

    Returns:
      (rm_lane_marker, rm_road_marker_non_lane, rm_stop_line, has_lane, has_road, has_stop)

    Policy for current Type-A:
    - lane/non-lane channels are supervised when record exists (has_lane=has_road=1)
    - stop_line class is treated as unavailable in BDD100K 100k schema by default
      (has_stop=0, all-255 mask), unless explicit stop-line category is present.
    """
    if width <= 0 or height <= 0:
        raise ValueError("width/height must be positive")

    lane_img = Image.new("L", (width, height), 0)
    road_img = Image.new("L", (width, height), 0)
    stop_img = Image.new("L", (width, height), 0)
    lane_draw = ImageDraw.Draw(lane_img)
    road_draw = ImageDraw.Draw(road_img)
    stop_draw = ImageDraw.Draw(stop_img)

    objs = _bdd_record_objects(rec)
    saw_explicit_stop = False
    for o in objs:
        cat = o.get("category")
        if not isinstance(cat, str):
            continue
        c = cat.strip().lower()
        if not c.startswith("lane/"):
            continue
        pts = _poly2d_points(o.get("poly2d"))
        if len(pts) < 2:
            continue

        if c in BDD_LANE_MARKER_CATEGORIES:
            lane_draw.line(pts, fill=1, width=max(1, int(line_width)))
            continue

        if c in BDD_ROAD_MARKER_NON_LANE_CATEGORIES:
            if c == "lane/crosswalk" and len(pts) >= 3:
                road_draw.polygon(pts, fill=1)
            else:
                road_draw.line(pts, fill=1, width=max(1, int(line_width)))
            continue

        if c in BDD_STOP_LINE_CATEGORIES:
            saw_explicit_stop = True
            stop_draw.line(pts, fill=1, width=max(1, int(line_width)))
            road_draw.line(pts, fill=1, width=max(1, int(line_width)))
            continue

    rm_lane = np.array(lane_img, dtype=np.uint8)
    rm_road = np.array(road_img, dtype=np.uint8)

    # Current Type-A default: stop line supervision unavailable -> ignore channel.
    if saw_explicit_stop:
        rm_stop = np.array(stop_img, dtype=np.uint8)
        has_stop = 1
    else:
        rm_stop = make_all_ignore_mask(height, width)
        has_stop = 0

    has_lane = 1
    has_road = 1
    return rm_lane, rm_road, rm_stop, has_lane, has_road, has_stop
