#!/usr/bin/env python3
"""Extract images / masks / boxes from Waymo Perception v2 parquet files.

Usage example:
  python tools/data_analysis/wod/extract_wod_v2_sample.py \
    --training-root /home/user1/Python_Workspace/YOLOPv26/datasets/WaymoOpenDataset/wod_pv2_minimal_1ctx/training \
    --out-root /home/user1/Python_Workspace/YOLOPv26/datasets/WaymoOpenDataset/wod_pv2_minimal_1ctx/decoded \
    --max-rows 50
"""

from __future__ import annotations

import argparse
import csv
import io
import os
from dataclasses import dataclass
from glob import glob
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pyarrow.parquet as pq
from PIL import Image


WAYMO_SEG_ROAD = 20
WAYMO_SEG_LANE_MARKER = 21
WAYMO_SEG_ROAD_MARKER = 22


CAMERA_NAME_MAP = {
    0: "unknown",
    1: "front",
    2: "front_left",
    3: "front_right",
    4: "side_left",
    5: "side_right",
}


WAYMO_TYPE_TO_PV26 = {
    1: 0,   # VEHICLE -> car
    2: 5,   # PEDESTRIAN -> pedestrian
    3: 10,  # SIGN -> sign_pole
    4: 4,   # CYCLIST -> bicycle
}


@dataclass(frozen=True)
class Key:
    context: str
    timestamp_micros: int
    camera_name: int


@dataclass
class SegEntry:
    divisor: int
    panoptic_png: bytes


@dataclass
class BoxEntry:
    box_type: int
    cx: float
    cy: float
    w: float
    h: float


def _iter_rows(path: str, columns: Optional[Sequence[str]] = None) -> Iterable[dict]:
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(columns=columns, batch_size=256):
        for row in batch.to_pylist():
            yield row


def _pick_parquet(component_dir: str, context: Optional[str]) -> str:
    if context:
        path = os.path.join(component_dir, f"{context}.parquet")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing parquet: {path}")
        return path
    files = sorted(glob(os.path.join(component_dir, "*.parquet")))
    if not files:
        raise FileNotFoundError(f"No parquet found under: {component_dir}")
    if len(files) > 1:
        raise ValueError(
            f"Multiple parquets in {component_dir}; pass --context-name to pick one."
        )
    return files[0]


def _key_from_row(row: dict) -> Key:
    return Key(
        context=row["key.segment_context_name"],
        timestamp_micros=int(row["key.frame_timestamp_micros"]),
        camera_name=int(row["key.camera_name"]),
    )


def _decode_panoptic_to_semantic(panoptic_png: bytes, divisor: int) -> np.ndarray:
    if divisor <= 0:
        raise ValueError(f"Invalid panoptic divisor: {divisor}")
    panoptic = np.array(Image.open(io.BytesIO(panoptic_png)), dtype=np.uint16)
    semantic = panoptic // divisor
    return semantic.astype(np.int32)


def _save_png_uint8(array_u8: np.ndarray, path: str) -> None:
    Image.fromarray(array_u8.astype(np.uint8), mode="L").save(path)


def _ensure_dirs(out_root: str) -> Dict[str, str]:
    dirs = {
        "images": os.path.join(out_root, "images"),
        "det": os.path.join(out_root, "labels_det"),
        "da": os.path.join(out_root, "labels_seg_da"),
        "lane": os.path.join(out_root, "labels_seg_lane"),
        "road_marker": os.path.join(out_root, "labels_seg_road_marker"),
        "stop_line": os.path.join(out_root, "labels_seg_stop_line"),
        "semantic": os.path.join(out_root, "labels_semantic_id"),
        "meta": os.path.join(out_root, "meta"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs


def _norm(v: float, denom: float) -> float:
    if denom <= 0:
        return 0.0
    return max(0.0, min(1.0, v / denom))


def _build_sample_id(key: Key) -> str:
    cam = CAMERA_NAME_MAP.get(key.camera_name, f"cam{key.camera_name}")
    return f"waymo__{key.context}__{key.timestamp_micros}__{cam}"


def extract(
    training_root: str,
    out_root: str,
    context_name: Optional[str],
    max_rows: Optional[int],
    drivable_ids: Sequence[int],
    lane_ids: Sequence[int],
    road_marker_ids: Sequence[int],
    stop_line_ids: Sequence[int],
    require_seg: bool,
) -> None:
    image_parquet = _pick_parquet(os.path.join(training_root, "camera_image"), context_name)
    box_parquet = _pick_parquet(os.path.join(training_root, "camera_box"), context_name)
    seg_parquet = _pick_parquet(
        os.path.join(training_root, "camera_segmentation"), context_name
    )

    # If context is omitted, infer it from selected image parquet.
    inferred_context = os.path.basename(image_parquet).replace(".parquet", "")
    dirs = _ensure_dirs(out_root)
    manifest_path = os.path.join(dirs["meta"], "manifest.csv")

    seg_cols = [
        "key.segment_context_name",
        "key.frame_timestamp_micros",
        "key.camera_name",
        "[CameraSegmentationLabelComponent].panoptic_label_divisor",
        "[CameraSegmentationLabelComponent].panoptic_label",
    ]
    seg_by_key: Dict[Key, SegEntry] = {}
    for row in _iter_rows(seg_parquet, columns=seg_cols):
        key = _key_from_row(row)
        seg_by_key[key] = SegEntry(
            divisor=int(row["[CameraSegmentationLabelComponent].panoptic_label_divisor"]),
            panoptic_png=row["[CameraSegmentationLabelComponent].panoptic_label"],
        )

    box_cols = [
        "key.segment_context_name",
        "key.frame_timestamp_micros",
        "key.camera_name",
        "[CameraBoxComponent].type",
        "[CameraBoxComponent].box.center.x",
        "[CameraBoxComponent].box.center.y",
        "[CameraBoxComponent].box.size.x",
        "[CameraBoxComponent].box.size.y",
    ]
    boxes_by_key: Dict[Key, List[BoxEntry]] = {}
    for row in _iter_rows(box_parquet, columns=box_cols):
        key = _key_from_row(row)
        boxes_by_key.setdefault(key, []).append(
            BoxEntry(
                box_type=int(row["[CameraBoxComponent].type"]),
                cx=float(row["[CameraBoxComponent].box.center.x"]),
                cy=float(row["[CameraBoxComponent].box.center.y"]),
                w=float(row["[CameraBoxComponent].box.size.x"]),
                h=float(row["[CameraBoxComponent].box.size.y"]),
            )
        )

    img_cols = [
        "key.segment_context_name",
        "key.frame_timestamp_micros",
        "key.camera_name",
        "[CameraImageComponent].image",
    ]

    manifest_rows = []
    count = 0
    for row in _iter_rows(image_parquet, columns=img_cols):
        key = _key_from_row(row)
        sample_id = _build_sample_id(key)

        seg = seg_by_key.get(key)
        if require_seg and seg is None:
            continue

        img = Image.open(io.BytesIO(row["[CameraImageComponent].image"]))
        width, height = img.size
        image_path = os.path.join(dirs["images"], f"{sample_id}.jpg")
        img.save(image_path, format="JPEG", quality=95)

        # Detection labels (YOLO txt); empty file when no labels.
        det_path = os.path.join(dirs["det"], f"{sample_id}.txt")
        with open(det_path, "w", encoding="utf-8") as f:
            for b in boxes_by_key.get(key, []):
                cls = WAYMO_TYPE_TO_PV26.get(b.box_type)
                if cls is None:
                    continue
                cx_n = _norm(b.cx, width)
                cy_n = _norm(b.cy, height)
                w_n = _norm(b.w, width)
                h_n = _norm(b.h, height)
                f.write(f"{cls} {cx_n:.6f} {cy_n:.6f} {w_n:.6f} {h_n:.6f}\n")

        da_path = os.path.join(dirs["da"], f"{sample_id}.png")
        lane_path = os.path.join(dirs["lane"], f"{sample_id}.png")
        road_marker_path = os.path.join(dirs["road_marker"], f"{sample_id}.png")
        stop_line_path = os.path.join(dirs["stop_line"], f"{sample_id}.png")
        semantic_path = os.path.join(dirs["semantic"], f"{sample_id}.png")

        if seg is None:
            # Missing segmentation supervision.
            ignore_mask = np.full((height, width), 255, dtype=np.uint8)
            _save_png_uint8(ignore_mask, da_path)
            _save_png_uint8(ignore_mask, lane_path)
            _save_png_uint8(ignore_mask, road_marker_path)
            _save_png_uint8(ignore_mask, stop_line_path)
            has_da, has_lane, has_road_marker, has_stop_line, has_semantic = 0, 0, 0, 0, 0
        else:
            semantic = _decode_panoptic_to_semantic(seg.panoptic_png, seg.divisor)
            da = np.isin(semantic, drivable_ids).astype(np.uint8)
            lane = np.isin(semantic, lane_ids).astype(np.uint8)
            road_marker = np.isin(semantic, road_marker_ids).astype(np.uint8)
            if stop_line_ids:
                stop_line = np.isin(semantic, stop_line_ids).astype(np.uint8)
                has_stop_line = 1
            else:
                stop_line = np.full_like(da, 255, dtype=np.uint8)  # unknown
                has_stop_line = 0
            semantic_id = np.zeros_like(da, dtype=np.uint8)
            semantic_id[da == 1] = 1
            semantic_id[lane == 1] = 2  # lane priority
            _save_png_uint8(da, da_path)
            _save_png_uint8(lane, lane_path)
            _save_png_uint8(road_marker, road_marker_path)
            _save_png_uint8(stop_line, stop_line_path)
            _save_png_uint8(semantic_id, semantic_path)
            has_da, has_lane, has_road_marker, has_semantic = 1, 1, 1, 1

        manifest_rows.append(
            {
                "sample_id": sample_id,
                "context_name": key.context,
                "timestamp_micros": key.timestamp_micros,
                "camera_name": key.camera_name,
                "width": width,
                "height": height,
                "has_det": 1 if boxes_by_key.get(key) else 0,
                "has_da": has_da,
                "has_lane": has_lane,
                "has_road_marker": has_road_marker,
                "has_stop_line": has_stop_line,
                "has_semantic_id": has_semantic,
                "image_path": image_path,
                "det_path": det_path,
                "da_path": da_path,
                "lane_path": lane_path,
                "road_marker_path": road_marker_path,
                "stop_line_path": stop_line_path,
                "semantic_id_path": semantic_path if has_semantic else "",
            }
        )

        count += 1
        if max_rows is not None and count >= max_rows:
            break

    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_id",
                "context_name",
                "timestamp_micros",
                "camera_name",
                "width",
                "height",
                "has_det",
                "has_da",
                "has_lane",
                "has_road_marker",
                "has_stop_line",
                "has_semantic_id",
                "image_path",
                "det_path",
                "da_path",
                "lane_path",
                "road_marker_path",
                "stop_line_path",
                "semantic_id_path",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    print("Done.")
    print(f"context: {inferred_context}")
    print(f"rows exported: {count}")
    print(f"output root: {out_root}")
    print(f"manifest: {manifest_path}")


def _parse_ids(arg: str) -> List[int]:
    ids = []
    for x in arg.split(","):
        x = x.strip()
        if not x:
            continue
        ids.append(int(x))
    if not ids:
        raise ValueError("Expected at least one id")
    return ids


def _parse_ids_allow_empty(arg: str) -> List[int]:
    arg = arg.strip()
    if not arg:
        return []
    return _parse_ids(arg)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--training-root",
        required=True,
        help="Path like .../wod_pv2_minimal_1ctx/training",
    )
    parser.add_argument(
        "--out-root",
        required=True,
        help="Where decoded images/masks/labels will be written",
    )
    parser.add_argument(
        "--context-name",
        default=None,
        help="Context name without .parquet. Optional if each component has exactly one file.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Limit number of rows to export (default: all rows).",
    )
    parser.add_argument(
        "--drivable-ids",
        type=_parse_ids,
        default=[WAYMO_SEG_ROAD],
        help="Comma-separated Waymo semantic IDs mapped to drivable (default: 20).",
    )
    parser.add_argument(
        "--lane-ids",
        type=_parse_ids,
        default=[WAYMO_SEG_LANE_MARKER],
        help="Comma-separated Waymo semantic IDs mapped to lane (default: 21).",
    )
    parser.add_argument(
        "--road-marker-ids",
        type=_parse_ids,
        default=[WAYMO_SEG_ROAD_MARKER],
        help="Comma-separated Waymo semantic IDs mapped to non-lane road markers (default: 22).",
    )
    parser.add_argument(
        "--stop-line-ids",
        type=_parse_ids_allow_empty,
        default=[],
        help="Comma-separated Waymo semantic IDs mapped to stop line. Empty means unknown (default).",
    )
    parser.add_argument(
        "--require-seg",
        action="store_true",
        help="If set, only export frames that have camera segmentation labels.",
    )
    args = parser.parse_args()

    extract(
        training_root=args.training_root,
        out_root=args.out_root,
        context_name=args.context_name,
        max_rows=args.max_rows,
        drivable_ids=args.drivable_ids,
        lane_ids=args.lane_ids,
        road_marker_ids=args.road_marker_ids,
        stop_line_ids=args.stop_line_ids,
        require_seg=args.require_seg,
    )


if __name__ == "__main__":
    main()
