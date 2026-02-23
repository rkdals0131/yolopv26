#!/usr/bin/env python3
"""
Dataset label inventory for class-unification meetings.

Goal: give quick, concrete counts of what labels actually exist per dataset
without forcing every dataset to be converted first.

This script is intentionally pragmatic:
- For huge datasets, it defaults to counting files (supervision availability).
- For RLMD/ETRI (manageable sizes), it also summarizes label distributions.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
from PIL import Image


def _utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def _count_glob(path: Path, pattern: str) -> int:
    return sum(1 for _ in path.glob(pattern))


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _polygon_area_xy(points: List[List[float]]) -> float:
    # Shoelace formula. Points may be non-closed; close implicitly.
    if len(points) < 3:
        return 0.0
    xs = np.array([p[0] for p in points], dtype=np.float64)
    ys = np.array([p[1] for p in points], dtype=np.float64)
    xs2 = np.concatenate([xs, xs[:1]])
    ys2 = np.concatenate([ys, ys[:1]])
    return float(0.5 * abs(np.dot(xs2[:-1], ys2[1:]) - np.dot(xs2[1:], ys2[:-1])))


@dataclass(frozen=True)
class RgbClass:
    class_id: int
    name: str
    rgb: Tuple[int, int, int]

    @property
    def code(self) -> int:
        r, g, b = self.rgb
        return (r << 16) | (g << 8) | b


def _load_rlmd_palette(csv_path: Path) -> Dict[int, RgbClass]:
    # rlmd.csv: id,name,r,g,b
    palette: Dict[int, RgbClass] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            class_id = int(row[0])
            name = row[1].strip()
            r, g, b = int(row[2]), int(row[3]), int(row[4])
            rc = RgbClass(class_id=class_id, name=name, rgb=(r, g, b))
            palette[rc.code] = rc
    return palette


def _rgb_code_image(img_rgb_u8: np.ndarray) -> np.ndarray:
    # (H,W,3) uint8 -> (H,W) uint32 code.
    r = img_rgb_u8[..., 0].astype(np.uint32)
    g = img_rgb_u8[..., 1].astype(np.uint32)
    b = img_rgb_u8[..., 2].astype(np.uint32)
    return (r << 16) | (g << 8) | b


def _rlmd_stats(rlmd_root: Path) -> Dict[str, Any]:
    base = rlmd_root / "RLMD_1080p"
    labels_root = base / "labels"
    csv_path = base / "rlmd.csv"
    if not labels_root.exists() or not csv_path.exists():
        return {"present": False, "reason": "RLMD_1080p/labels or rlmd.csv not found"}

    palette = _load_rlmd_palette(csv_path)
    split_dirs = [(labels_root / "train"), (labels_root / "val")]
    masks: List[Path] = []
    for d in split_dirs:
        if d.exists():
            masks.extend(sorted(d.glob("*.png")))

    pixel_counts = Counter()
    image_counts = Counter()
    unknown_codes = Counter()
    canonical = Counter()

    # Conservative lane-boundary set for meetings. Adjust as needed.
    lane_boundary_names = {
        "solid single white",
        "solid single yellow",
        "solid single red",
        "solid double white",
        "solid double yellow",
        "dashed single white",
        "dashed single yellow",
        "channelizing line",
    }
    stop_line_name = "stop line"

    for p in masks:
        arr = np.array(Image.open(p))
        if arr.ndim != 3 or arr.shape[2] != 3:
            unknown_codes["__non_rgb__"] += 1
            continue
        code = _rgb_code_image(arr)
        vals, counts = np.unique(code, return_counts=True)
        seen_names = set()
        for v, c in zip(vals.tolist(), counts.tolist()):
            cls = palette.get(int(v))
            if cls is None:
                unknown_codes[int(v)] += int(c)
                continue
            pixel_counts[cls.name] += int(c)
            if cls.name not in seen_names:
                image_counts[cls.name] += 1
                seen_names.add(cls.name)

        # Canonical group presence per mask.
        has_stop = stop_line_name in seen_names
        has_lane_boundary = any(n in seen_names for n in lane_boundary_names)
        has_non_lane_marker = any(
            (n != "background") and (n not in lane_boundary_names) for n in seen_names
        )
        if has_stop:
            canonical["frames_with_stop_line"] += 1
        if has_lane_boundary:
            canonical["frames_with_lane_boundary"] += 1
        if has_non_lane_marker:
            canonical["frames_with_road_marker_non_lane"] += 1
        if has_stop and has_lane_boundary:
            canonical["frames_with_both"] += 1

    def _top(counter: Counter, n: int = 25) -> List[Tuple[str, int]]:
        return [(k, int(v)) for k, v in counter.most_common(n)]

    return {
        "present": True,
        "num_masks": len(masks),
        "palette_classes": len(palette),
        "image_counts_top": _top(image_counts, 50),
        "pixel_counts_top": _top(pixel_counts, 50),
        "stop_line": {
            "name": stop_line_name,
            "masks_with": int(image_counts.get(stop_line_name, 0)),
            "pixels": int(pixel_counts.get(stop_line_name, 0)),
        },
        "canonical_groups": {
            "lane_boundary_names": sorted(lane_boundary_names),
            "road_marker_non_lane_definition": "any label other than background and lane_boundary_names",
            **{k: int(v) for k, v in canonical.items()},
        },
        "canonical_pixels": {
            "pixels_lane_boundary": int(
                sum(int(pixel_counts.get(n, 0)) for n in lane_boundary_names)
            ),
            "pixels_road_marker_non_lane": int(
                sum(
                    int(v)
                    for k, v in pixel_counts.items()
                    if (k != "background") and (k not in lane_boundary_names)
                )
            ),
        },
        "unknown_codes_top": _top(unknown_codes, 10),
    }


def _etri_stats(etri_root: Path) -> Dict[str, Any]:
    # Cityscapes-like polygon JSON: {"imgHeight","imgWidth","objects":[{"label":..., "polygon":[[x,y],...]}]}
    mono = etri_root / "MonoCameraSemanticSegmentation" / "labels"
    multi_train = etri_root / "Multi Camera Semantic Segmentation" / "labels" / "train"
    multi_val = etri_root / "Multi Camera Semantic Segmentation" / "labels" / "val"

    json_paths: List[Path] = []
    for base in [mono, multi_train, multi_val]:
        if base.exists():
            json_paths.extend(sorted(base.rglob("*_gtFine_polygons.json")))

    if not json_paths:
        return {"present": False, "reason": "No *_gtFine_polygons.json found"}

    obj_counts = Counter()
    frame_counts = Counter()
    area_sums = Counter()
    canonical = Counter()

    for p in json_paths:
        data = _read_json(p)
        objects = data.get("objects") or []
        seen = set()
        for obj in objects:
            label = obj.get("label")
            if not label:
                continue
            obj_counts[label] += 1
            seen.add(label)
            poly = obj.get("polygon")
            if isinstance(poly, list) and poly and isinstance(poly[0], list) and len(poly[0]) >= 2:
                area_sums[label] += _polygon_area_xy(poly)
        for label in seen:
            frame_counts[label] += 1

    # Heuristic grouping for meetings: lane-boundary-ish labels in ETRI are often short codes or "guidance line".
    lane_like_pat = re.compile(r"(whdot|whsol|yedot|yesol|bldot|blsol|guidance line|lane divider)", re.IGNORECASE)
    stop_line_key = "stop line"
    other_road_mark_pat = re.compile(
        r"(general road mark|crosswalk|stop line|arrow|prohibition|number|slow|motor|bike icon|box junction|parking|speed bump|channelizing line|left|right|forward|straight|leftu)",
        re.IGNORECASE,
    )

    lane_like_objs = 0
    for k, v in obj_counts.items():
        if lane_like_pat.search(k):
            lane_like_objs += int(v)

    # Re-scan just for frame-level presence (avoid double-counting multiple lane-like labels per frame).
    for p in json_paths:
        data = _read_json(p)
        objects = data.get("objects") or []
        any_lane_like = False
        any_stop_line = False
        any_other_road_mark = False
        for obj in objects:
            label = obj.get("label") or ""
            if lane_like_pat.search(label):
                any_lane_like = True
            if label == stop_line_key:
                any_stop_line = True
            if other_road_mark_pat.search(label):
                any_other_road_mark = True
            if any_lane_like and any_stop_line:
                break
        if any_lane_like:
            canonical["frames_with_lane_boundary"] += 1
        if any_stop_line:
            canonical["frames_with_stop_line"] += 1
        if any_lane_like and any_stop_line:
            canonical["frames_with_both"] += 1
        if any_other_road_mark:
            canonical["frames_with_road_marker_non_lane"] += 1

    def _top(counter: Counter, n: int = 25) -> List[Tuple[str, float]]:
        return [(k, float(v)) for k, v in counter.most_common(n)]

    return {
        "present": True,
        "num_json": len(json_paths),
        "frame_counts_top": [(k, int(v)) for k, v in frame_counts.most_common(50)],
        "obj_counts_top": [(k, int(v)) for k, v in obj_counts.most_common(50)],
        "area_sums_top": _top(area_sums, 50),
        "stop_line": {
            "label": stop_line_key,
            "frames_with": int(frame_counts.get(stop_line_key, 0)),
            "objects": int(obj_counts.get(stop_line_key, 0)),
            "area_sum_px2": float(area_sums.get(stop_line_key, 0.0)),
        },
        "lane_like": {
            "pattern": lane_like_pat.pattern,
            "frames_with_any": int(canonical.get("frames_with_lane_boundary", 0)),
            "objects_sum": int(lane_like_objs),
        },
        "canonical_groups": {k: int(v) for k, v in canonical.items()},
        "heuristics": {
            "other_road_mark_pattern": other_road_mark_pat.pattern,
            "road_marker_non_lane_definition": "any label matching other_road_mark_pattern (includes stop line)",
        },
    }


def _waymo_decoded_manifest_stats(wod_root: Path) -> Dict[str, Any]:
    decoded = wod_root / "wod_pv2_minimal_1ctx" / "decoded"
    manifest = decoded / "meta" / "manifest.csv"
    if not manifest.exists():
        return {"present": False, "reason": "decoded/meta/manifest.csv not found"}

    sums = defaultdict(int)
    rows = 0
    with manifest.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows += 1
            for k in ["has_det", "has_da", "has_lane", "has_semantic_id"]:
                try:
                    sums[k] += int(r.get(k, "0") or "0")
                except ValueError:
                    pass
    sums["num_rows"] = rows
    return {"present": True, **dict(sums)}


def _waymo_training_join_stats(wod_root: Path) -> Dict[str, Any]:
    """Counts supervision availability by joining camera_image and camera_segmentation keys.

    This avoids the 'decoded' sampling bias (e.g., exporting first N rows might miss
    the few frames that actually have segmentation labels).
    """

    import io

    import pyarrow.parquet as pq  # local import: not everyone has pyarrow installed

    training = wod_root / "wod_pv2_minimal_1ctx" / "training"
    img_dir = training / "camera_image"
    seg_dir = training / "camera_segmentation"
    if not img_dir.exists() or not seg_dir.exists():
        return {"present": False, "reason": "training/camera_image or training/camera_segmentation not found"}

    img_parquets = sorted(img_dir.glob("*.parquet"))
    seg_parquets = sorted(seg_dir.glob("*.parquet"))
    if not img_parquets or not seg_parquets:
        return {"present": False, "reason": "no *.parquet under camera_image or camera_segmentation"}

    # Minimal sample typically has exactly one context parquet per component.
    img_pf = pq.ParquetFile(str(img_parquets[0]))
    seg_pf = pq.ParquetFile(str(seg_parquets[0]))
    cols = ["key.segment_context_name", "key.frame_timestamp_micros", "key.camera_name"]

    img_keys = set()
    for batch in img_pf.iter_batches(columns=cols, batch_size=2048):
        for r in batch.to_pylist():
            img_keys.add(
                (
                    r["key.segment_context_name"],
                    int(r["key.frame_timestamp_micros"]),
                    int(r["key.camera_name"]),
                )
            )

    seg_keys = set()
    for batch in seg_pf.iter_batches(columns=cols, batch_size=2048):
        for r in batch.to_pylist():
            seg_keys.add(
                (
                    r["key.segment_context_name"],
                    int(r["key.frame_timestamp_micros"]),
                    int(r["key.camera_name"]),
                )
            )

    inter = len(img_keys & seg_keys)

    # For the minimal sample, it's small enough to inspect semantic presence
    # by decoding a bounded number of panoptic labels.
    sem_cols = cols + [
        "[CameraSegmentationLabelComponent].panoptic_label_divisor",
        "[CameraSegmentationLabelComponent].panoptic_label",
    ]
    present = Counter()
    inspected = 0
    max_inspect = 200  # keep this fast; full WOD would be enormous.
    for batch in seg_pf.iter_batches(columns=sem_cols, batch_size=16):
        for r in batch.to_pylist():
            divisor = int(r["[CameraSegmentationLabelComponent].panoptic_label_divisor"])
            png_bytes = r["[CameraSegmentationLabelComponent].panoptic_label"]
            if divisor <= 0 or not png_bytes:
                continue
            pan = np.array(Image.open(io.BytesIO(png_bytes)), dtype=np.uint16)
            sem = (pan // divisor).astype(np.uint16)
            # IDs per WOD camera_segmentation.proto: ROAD=20, LANE_MARKER=21, ROAD_MARKER=22, SIDEWALK=23.
            if np.any(sem == 20):
                present["frames_with_road"] += 1
            if np.any(sem == 21):
                present["frames_with_lane_marker"] += 1
            if np.any(sem == 22):
                present["frames_with_road_marker"] += 1
            if np.any(sem == 23):
                present["frames_with_sidewalk"] += 1
            inspected += 1
            if inspected >= max_inspect:
                break
        if inspected >= max_inspect:
            break

    return {
        "present": True,
        "image_rows": int(img_pf.metadata.num_rows),
        "seg_rows": int(seg_pf.metadata.num_rows),
        "image_unique_keys": len(img_keys),
        "seg_unique_keys": len(seg_keys),
        "image_keys_with_seg": inter,
        "seg_inspected_rows": inspected,
        "seg_presence": {k: int(v) for k, v in present.items()},
    }


def _basic_file_counts(datasets_root: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    bdd = datasets_root / "BDD100K"
    if bdd.exists():
        img100k = bdd / "bdd100k_images_100k" / "100k"
        img10k = bdd / "bdd100k_images_10k" / "10k"
        out["BDD100K"] = {
            "images_100k_train_jpg": _count_glob(img100k / "train", "*.jpg"),
            "images_100k_val_jpg": _count_glob(img100k / "val", "*.jpg"),
            "images_100k_test_jpg": _count_glob(img100k / "test", "*.jpg"),
            "images_10k_train_jpg": _count_glob(img10k / "train", "*.jpg"),
            "images_10k_val_jpg": _count_glob(img10k / "val", "*.jpg"),
            "images_10k_test_jpg": _count_glob(img10k / "test", "*.jpg"),
            "drivable_label_png_train": _count_glob(bdd / "bdd100k_drivable_maps" / "labels" / "train", "*.png"),
            "drivable_label_png_val": _count_glob(bdd / "bdd100k_drivable_maps" / "labels" / "val", "*.png"),
            "seg_label_png_train": _count_glob(bdd / "bdd100k_seg_maps" / "labels" / "train", "*.png"),
            "seg_label_png_val": _count_glob(bdd / "bdd100k_seg_maps" / "labels" / "val", "*.png"),
            "det_json_train": _count_glob(bdd / "bdd100k_labels" / "100k" / "train", "*.json"),
            "det_json_val": _count_glob(bdd / "bdd100k_labels" / "100k" / "val", "*.json"),
        }

    cs = datasets_root / "Cityscapes"
    if cs.exists():
        li = cs / "leftImg8bit"
        gt = cs / "gtFine"
        out["Cityscapes"] = {
            "leftImg8bit_train_png": _count_glob(li / "train", "*/*_leftImg8bit.png"),
            "leftImg8bit_val_png": _count_glob(li / "val", "*/*_leftImg8bit.png"),
            "leftImg8bit_test_png": _count_glob(li / "test", "*/*_leftImg8bit.png"),
            "gtFine_train_labelIds_png": _count_glob(gt / "train", "*/*_gtFine_labelIds.png"),
            "gtFine_val_labelIds_png": _count_glob(gt / "val", "*/*_gtFine_labelIds.png"),
            "gtFine_train_polygons_json": _count_glob(gt / "train", "*/*_gtFine_polygons.json"),
            "gtFine_val_polygons_json": _count_glob(gt / "val", "*/*_gtFine_polygons.json"),
        }

    k360 = datasets_root / "KITTI-360"
    if k360.exists():
        sem = (
            k360
            / "data_2d_semantics_image_01"
            / "data_2d_semantics"
            / "train"
        )
        out["KITTI-360"] = {
            "semantic_png": sum(1 for _ in sem.rglob("image_01/semantic/*.png")) if sem.exists() else 0,
        }

    rlmd = datasets_root / "RLMD"
    if rlmd.exists():
        base = rlmd / "RLMD_1080p"
        out["RLMD"] = {
            "label_png_train": _count_glob(base / "labels" / "train", "*.png"),
            "label_png_val": _count_glob(base / "labels" / "val", "*.png"),
        }

    etri = datasets_root / "ETRI"
    if etri.exists():
        out["ETRI"] = {
            "mono_polygons_json": sum(1 for _ in (etri / "MonoCameraSemanticSegmentation" / "labels").glob("*_gtFine_polygons.json"))
            if (etri / "MonoCameraSemanticSegmentation" / "labels").exists()
            else 0,
            "multi_train_polygons_json": sum(1 for _ in (etri / "Multi Camera Semantic Segmentation" / "labels" / "train").rglob("*_gtFine_polygons.json"))
            if (etri / "Multi Camera Semantic Segmentation" / "labels" / "train").exists()
            else 0,
            "multi_val_polygons_json": sum(1 for _ in (etri / "Multi Camera Semantic Segmentation" / "labels" / "val").rglob("*_gtFine_polygons.json"))
            if (etri / "Multi Camera Semantic Segmentation" / "labels" / "val").exists()
            else 0,
        }

    wod = datasets_root / "WaymoOpenDataset"
    if wod.exists():
        out["WaymoOpenDataset"] = {
            "decoded_manifest_csv": int((wod / "wod_pv2_minimal_1ctx" / "decoded" / "meta" / "manifest.csv").exists()),
        }

    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets-root", type=Path, default=Path("datasets"))
    ap.add_argument("--out", type=Path, default=None, help="Write JSON report to this path.")
    args = ap.parse_args()

    datasets_root: Path = args.datasets_root
    report: Dict[str, Any] = {
        "generated_at": _utc_now_iso(),
        "datasets_root": str(datasets_root.resolve()) if datasets_root.exists() else str(datasets_root),
        "basic_counts": _basic_file_counts(datasets_root),
        "rlmd": _rlmd_stats(datasets_root / "RLMD"),
        "etri": _etri_stats(datasets_root / "ETRI"),
        "waymo_decoded": _waymo_decoded_manifest_stats(datasets_root / "WaymoOpenDataset"),
        "waymo_training": _waymo_training_join_stats(datasets_root / "WaymoOpenDataset"),
    }

    # Meeting-friendly summary for proposed PV26 sigmoid channels.
    pv26 = {}
    basic = report["basic_counts"]
    if "BDD100K" in basic:
        pv26["BDD100K"] = {
            "drivable_frames": int(basic["BDD100K"]["drivable_label_png_train"] + basic["BDD100K"]["drivable_label_png_val"]),
            "lane_marker_frames": 0,
            "road_marker_non_lane_frames": 0,
            "stop_line_frames": 0,
        }
    if "Cityscapes" in basic:
        cs_labels = int(basic["Cityscapes"]["gtFine_train_labelIds_png"] + basic["Cityscapes"]["gtFine_val_labelIds_png"])
        cs_images = int(basic["Cityscapes"]["leftImg8bit_train_png"] + basic["Cityscapes"]["leftImg8bit_val_png"])
        pv26["Cityscapes"] = {
            "drivable_frames": int(min(cs_labels, cs_images)),
            "lane_marker_frames": 0,
            "road_marker_non_lane_frames": 0,
            "stop_line_frames": 0,
            **(
                {"note": f"mismatch: labelIds={cs_labels} leftImg8bit(train+val)={cs_images}"}
                if cs_labels != cs_images
                else {}
            ),
        }
    if "KITTI-360" in basic:
        pv26["KITTI-360"] = {
            "semantic_frames": int(basic["KITTI-360"]["semantic_png"]),
            "note": "semantic->drivable/lane mapping not computed here",
        }
    if report["rlmd"].get("present"):
        cg = report["rlmd"].get("canonical_groups", {})
        pv26["RLMD"] = {
            "lane_marker_frames": int(cg.get("frames_with_lane_boundary", 0)),
            "road_marker_non_lane_frames": int(cg.get("frames_with_road_marker_non_lane", 0)),
            "stop_line_frames": int(cg.get("frames_with_stop_line", 0)),
        }
    if report["etri"].get("present"):
        cg = report["etri"].get("canonical_groups", {})
        pv26["ETRI"] = {
            "lane_marker_frames": int(cg.get("frames_with_lane_boundary", 0)),
            "road_marker_non_lane_frames": int(cg.get("frames_with_road_marker_non_lane", 0)),
            "stop_line_frames": int(cg.get("frames_with_stop_line", 0)),
        }
    if report["waymo_training"].get("present"):
        wt = report["waymo_training"]
        sp = wt.get("seg_presence", {})
        pv26["WaymoOpenDataset(minimal_1ctx)"] = {
            "seg_rows": int(wt.get("seg_rows", 0)),
            "drivable_frames": int(sp.get("frames_with_road", 0)),
            "lane_marker_frames": int(sp.get("frames_with_lane_marker", 0)),
            "road_marker_non_lane_frames": int(sp.get("frames_with_road_marker", 0)),
            "sidewalk_frames": int(sp.get("frames_with_sidewalk", 0)),
            "note": "counts are over inspected seg rows in the minimal sample context",
        }

    report["pv26_channels_summary"] = pv26

    out_path: Optional[Path] = args.out
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # Human-readable summary (small and stable).
    print("== Dataset Label Inventory ==")
    print(f"generated_at: {report['generated_at']}")
    print(f"datasets_root: {report['datasets_root']}")
    print("")

    basic = report["basic_counts"]
    for ds_name in ["BDD100K", "Cityscapes", "KITTI-360", "RLMD", "ETRI", "WaymoOpenDataset"]:
        if ds_name in basic:
            print(f"- {ds_name}: {basic[ds_name]}")
    print("")

    if report["rlmd"].get("present"):
        sl = report["rlmd"]["stop_line"]
        print(f"- RLMD stop line: masks_with={sl['masks_with']} pixels={sl['pixels']}")
    else:
        print(f"- RLMD: not present ({report['rlmd'].get('reason')})")

    if report["etri"].get("present"):
        sl = report["etri"]["stop_line"]
        lane = report["etri"]["lane_like"]
        print(f"- ETRI stop line: frames_with={sl['frames_with']} objects={sl['objects']}")
        print(f"- ETRI lane-like (heuristic): frames_with_any={lane['frames_with_any']} objects_sum={lane['objects_sum']}")
    else:
        print(f"- ETRI: not present ({report['etri'].get('reason')})")

    if report["waymo_training"].get("present"):
        wt = report["waymo_training"]
        msg = (
            f"- Waymo training join: image_rows={wt['image_rows']} seg_rows={wt['seg_rows']} "
            f"image_keys_with_seg={wt['image_keys_with_seg']}"
        )
        if wt.get("seg_inspected_rows", 0) > 0:
            msg += f" seg_presence={wt.get('seg_presence', {})}"
        print(msg)
    else:
        print(f"- Waymo training: not present ({report['waymo_training'].get('reason')})")

    if report["waymo_decoded"].get("present"):
        wd = report["waymo_decoded"]
        print(
            f"- Waymo decoded manifest (decoded sample only): rows={wd['num_rows']} has_det={wd['has_det']} has_da={wd['has_da']} has_lane={wd['has_lane']} has_semantic_id={wd['has_semantic_id']}"
        )

    print("")
    print("== PV26 Proposed Sigmoid Channels (Frame Counts) ==")
    for name, row in report["pv26_channels_summary"].items():
        print(f"- {name}: {row}")

    if out_path is not None:
        print("")
        print(f"wrote: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
