#!/usr/bin/env python3
"""
Convert Waymo Open Dataset (Perception v2 parquet) into PV26 Type-A dataset layout.

This implementation targets the minimal sample first (single context parquet per component)
but keeps the I/O contract compatible with the PV26 conversion spec.

Related docs:
- docs/PV26_PRD.md
- docs/PV26_DATASET_CONVERSION_SPEC.md
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import sys
import time
from collections import Counter
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pv26.class_map import render_class_map_yaml
from pv26.constants import CLASSMAP_VERSION_V3
from pv26.dataset_layout import Pv26Layout, SPLITS
from pv26.manifest import ManifestRow, write_manifest_csv
from pv26.masks import make_all_ignore_mask
from pv26.utils import list_files_recursive, sha256_file, stable_split_for_group_key, utc_now_iso, write_json
from pv26.wod import semantic_to_pv26_da_rm_masks


WAYMO_TYPE_TO_PV26 = {
    1: 0,   # VEHICLE -> car
    2: 5,   # PEDESTRIAN -> pedestrian
    3: 10,  # SIGN -> sign_pole
    4: 4,   # CYCLIST -> bicycle
}

CAMERA_NAME_MAP = {
    0: "unknown",
    1: "front",
    2: "front_left",
    3: "front_right",
    4: "side_left",
    5: "side_right",
}


@dataclass(frozen=True)
class Key:
    context: str
    timestamp_micros: int
    camera_name: int


@dataclass(frozen=True)
class SegEntry:
    divisor: int
    panoptic_png: bytes


@dataclass(frozen=True)
class BoxEntry:
    box_type: int
    cx: float
    cy: float
    w: float
    h: float


def _fmt_duration(seconds: Optional[float]) -> str:
    if seconds is None:
        return "계산중"
    sec = max(0, int(round(float(seconds))))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _count_glob(path: Path, pattern: str) -> int:
    return sum(1 for _ in path.glob(pattern))


def _iter_rows(path: str, columns: Optional[Sequence[str]] = None) -> Iterable[dict]:
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(columns=columns, batch_size=256):
        for row in batch.to_pylist():
            yield row


def _pick_parquet(component_dir: Path, context: Optional[str]) -> Path:
    if context:
        p = component_dir / f"{context}.parquet"
        if not p.exists():
            raise FileNotFoundError(f"missing parquet: {p}")
        return p
    files = sorted(Path(p).resolve() for p in glob(str(component_dir / "*.parquet")))
    if not files:
        raise FileNotFoundError(f"no parquet under: {component_dir}")
    if len(files) > 1:
        raise ValueError(f"multiple parquets in {component_dir}; pass --context-name")
    return Path(files[0])


def _key_from_row(row: dict) -> Key:
    return Key(
        context=str(row["key.segment_context_name"]),
        timestamp_micros=int(row["key.frame_timestamp_micros"]),
        camera_name=int(row["key.camera_name"]),
    )


def _decode_panoptic_to_semantic(panoptic_png: bytes, divisor: int) -> np.ndarray:
    if divisor <= 0:
        raise ValueError(f"invalid panoptic divisor: {divisor}")
    panoptic = np.array(Image.open(io.BytesIO(panoptic_png)), dtype=np.uint16)
    semantic = panoptic // np.uint16(divisor)
    return semantic.astype(np.int32)


def _save_u8_mask(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr.astype(np.uint8), mode="L").save(path, format="PNG", optimize=False)


def _save_det_txt(path: Path, lines: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for ln in lines:
            f.write(ln.rstrip() + "\n")


def _norm(v: float, denom: int) -> float:
    if denom <= 0:
        return 0.0
    return float(v) / float(denom)


def _write_checksums_parallel(*, out_root: Path, files: Sequence[Path], out_path: Path) -> None:
    if not files:
        out_path.write_text("", encoding="utf-8")
        return
    rows: List[Tuple[str, str]] = []
    for p in files:
        relp = p.relative_to(out_root).as_posix()
        rows.append((sha256_file(p), relp))
    rows.sort(key=lambda x: x[1])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for h, relp in rows:
            f.write(f"{h}  {relp}\n")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--training-root",
        type=Path,
        default=Path("datasets/WaymoOpenDataset/wod_pv2_minimal_1ctx/training"),
        help="Waymo training root containing camera_image/camera_box/camera_segmentation",
    )
    p.add_argument(
        "--out-root",
        type=Path,
        default=Path("datasets/pv26_v1_waymo_minimal_1ctx"),
        help="Output root (default: datasets/pv26_v1_waymo_minimal_1ctx)",
    )
    p.add_argument("--context-name", type=str, default=None, help="Context name (parquet basename without .parquet)")
    p.add_argument("--seed", type=int, default=0, help="Split seed (deterministic)")
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument(
        "--split-policy",
        type=str,
        default="all_train",
        choices=["all_train", "stable_by_context"],
        help="Split policy (default: all_train for minimal contexts)",
    )
    p.add_argument(
        "--splits",
        type=str,
        default="train,val,test",
        help="Comma-separated split filter (default: train,val,test)",
    )
    p.add_argument("--max-rows", type=int, default=0, help="Max camera_image rows to process (0=all)")
    p.add_argument(
        "--require-seg",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Export only rows that have camera_segmentation (default: false)",
    )
    p.add_argument("--run-id", type=str, default="", help="Optional run id for conversion report")
    return p


def main() -> int:
    args = build_argparser().parse_args()
    training_root: Path = args.training_root
    out_root: Path = args.out_root
    split_filter = {s.strip().lower() for s in str(args.splits).split(",") if s.strip()}
    if not split_filter:
        split_filter = set(SPLITS)

    layout = Pv26Layout(out_root=out_root)
    layout.ensure_dirs()

    img_dir = training_root / "camera_image"
    seg_dir = training_root / "camera_segmentation"
    box_dir = training_root / "camera_box"
    if not img_dir.exists() or not seg_dir.exists() or not box_dir.exists():
        print(f"[pv26][waymo] missing component dirs under: {training_root}", flush=True)
        return 2

    ctx = args.context_name
    image_parquet = _pick_parquet(img_dir, ctx)
    seg_parquet = _pick_parquet(seg_dir, ctx)
    box_parquet = _pick_parquet(box_dir, ctx)

    seg_cols = [
        "key.segment_context_name",
        "key.frame_timestamp_micros",
        "key.camera_name",
        "[CameraSegmentationLabelComponent].panoptic_label_divisor",
        "[CameraSegmentationLabelComponent].panoptic_label",
    ]
    seg_by_key: Dict[Key, SegEntry] = {}
    for row in _iter_rows(str(seg_parquet), columns=seg_cols):
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
    for row in _iter_rows(str(box_parquet), columns=box_cols):
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

    rows: List[ManifestRow] = []
    counts = Counter()

    started = time.monotonic()
    done = 0
    split_by_context: Dict[str, str] = {}

    for row in _iter_rows(str(image_parquet), columns=img_cols):
        key = _key_from_row(row)

        seg = seg_by_key.get(key)
        if args.require_seg and seg is None:
            continue

        camera_name = CAMERA_NAME_MAP.get(int(key.camera_name), f"cam{int(key.camera_name)}")
        source = "waymo"
        sequence = key.context
        frame = str(int(key.timestamp_micros))
        camera_id = camera_name
        sample_id = f"{source}__{sequence}__{frame}__{camera_id}"
        source_group_key = f"{source}::{sequence}"

        if args.split_policy == "all_train":
            split = "train"
        else:
            split = split_by_context.get(sequence)
            if split is None:
                split = stable_split_for_group_key(
                    source_group_key,
                    seed=int(args.seed),
                    train_ratio=float(args.train_ratio),
                    val_ratio=float(args.val_ratio),
                )
                split_by_context[sequence] = split
        if split not in SPLITS:
            raise RuntimeError(f"invalid split computed: {split}")
        if split not in split_filter:
            continue

        img = Image.open(io.BytesIO(row["[CameraImageComponent].image"])).convert("RGB")
        w, h = img.size
        out_img_rel = str(Path("images") / split / f"{sample_id}.jpg")
        out_img_path = out_root / out_img_rel
        out_img_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(out_img_path, format="JPEG", quality=95, optimize=True)

        det_lines: List[str] = []
        for b in boxes_by_key.get(key, []):
            cls = WAYMO_TYPE_TO_PV26.get(int(b.box_type))
            if cls is None:
                continue
            cx_n = _norm(b.cx, w)
            cy_n = _norm(b.cy, h)
            w_n = _norm(b.w, w)
            h_n = _norm(b.h, h)
            det_lines.append(f"{cls} {cx_n:.6f} {cy_n:.6f} {w_n:.6f} {h_n:.6f}")

        out_det_rel = str(Path("labels_det") / split / f"{sample_id}.txt")
        _save_det_txt(out_root / out_det_rel, det_lines)

        out_da_rel = str(Path("labels_seg_da") / split / f"{sample_id}.png")
        out_rm_lane_rel = str(Path("labels_seg_rm_lane_marker") / split / f"{sample_id}.png")
        out_rm_road_rel = str(Path("labels_seg_rm_road_marker_non_lane") / split / f"{sample_id}.png")
        out_rm_stop_rel = str(Path("labels_seg_rm_stop_line") / split / f"{sample_id}.png")
        out_rm_lane_sub_rel = str(Path("labels_seg_rm_lane_subclass") / split / f"{sample_id}.png")

        if seg is None:
            da = make_all_ignore_mask(h, w)
            rm_lane = make_all_ignore_mask(h, w)
            rm_road = make_all_ignore_mask(h, w)
            has_da = 0
            has_lane = 0
            has_road = 0
        else:
            sem = _decode_panoptic_to_semantic(seg.panoptic_png, seg.divisor)
            da, rm_lane, rm_road = semantic_to_pv26_da_rm_masks(sem)
            has_da = 1
            has_lane = 1
            has_road = 1

        rm_stop = make_all_ignore_mask(h, w)
        rm_lane_sub = make_all_ignore_mask(h, w)
        has_stop = 0
        has_lane_sub = 0

        _save_u8_mask(out_root / out_da_rel, da)
        _save_u8_mask(out_root / out_rm_lane_rel, rm_lane)
        _save_u8_mask(out_root / out_rm_road_rel, rm_road)
        _save_u8_mask(out_root / out_rm_stop_rel, rm_stop)
        _save_u8_mask(out_root / out_rm_lane_sub_rel, rm_lane_sub)

        manifest_row = ManifestRow(
            sample_id=sample_id,
            split=split,
            source=source,
            sequence=sequence,
            frame=frame,
            camera_id=camera_id,
            timestamp_ns=str(int(key.timestamp_micros) * 1000),
            has_det=1,
            has_da=has_da,
            has_rm_lane_marker=has_lane,
            has_rm_road_marker_non_lane=has_road,
            has_rm_stop_line=has_stop,
            has_rm_lane_subclass=has_lane_sub,
            has_semantic_id=0,
            det_label_scope="subset",
            det_annotated_class_ids="0,4,5,10",
            image_relpath=out_img_rel,
            det_relpath=out_det_rel,
            da_relpath=out_da_rel,
            rm_lane_marker_relpath=out_rm_lane_rel,
            rm_road_marker_non_lane_relpath=out_rm_road_rel,
            rm_stop_line_relpath=out_rm_stop_rel,
            rm_lane_subclass_relpath=out_rm_lane_sub_rel,
            semantic_relpath="",
            width=int(w),
            height=int(h),
            weather_tag="unknown",
            time_tag="unknown",
            scene_tag="unknown",
            source_group_key=source_group_key,
        )
        rows.append(manifest_row)

        counts["total"] += 1
        counts[f"split:{split}"] += 1
        counts[f"has_det:{int(manifest_row.has_det)}"] += 1
        counts[f"has_da:{int(manifest_row.has_da)}"] += 1
        counts[f"has_rm_lane_marker:{int(manifest_row.has_rm_lane_marker)}"] += 1

        done += 1
        if done % 500 == 0:
            elapsed = time.monotonic() - started
            rate = float(done) / max(1e-9, elapsed)
            print(
                f"[pv26][waymo] {done:,} rows | {rate:,.2f} row/s | elapsed {_fmt_duration(elapsed)}",
                flush=True,
            )

        if args.max_rows and int(args.max_rows) > 0 and done >= int(args.max_rows):
            break

    if not rows:
        print("[pv26][waymo] no rows exported (check filters / parquet inputs)", flush=True)
        return 2

    layout.meta_dir().mkdir(parents=True, exist_ok=True)
    layout.class_map_path().write_text(render_class_map_yaml(classmap_version=CLASSMAP_VERSION_V3), encoding="utf-8")
    write_manifest_csv(layout.manifest_path(), rows)

    with layout.source_stats_path().open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["source", "num_samples"])
        writer.writerow(["waymo", counts["total"]])

    exported_files = list_files_recursive(out_root)
    _write_checksums_parallel(out_root=out_root, files=exported_files, out_path=layout.checksums_path())

    report = {
        "converter": "convert_wod_type_a.py",
        "converter_version": "0.1.0",
        "spec": "docs/PV26_DATASET_CONVERSION_SPEC.md v1.5",
        "timestamp_utc": utc_now_iso(),
        "run_id": args.run_id,
        "config": {
            "training_root": str(training_root),
            "out_root": str(out_root),
            "context_name": ctx or "",
            "seed": int(args.seed),
            "train_ratio": float(args.train_ratio),
            "val_ratio": float(args.val_ratio),
            "split_policy": str(args.split_policy),
            "splits": str(args.splits),
            "max_rows": int(args.max_rows),
            "require_seg": bool(args.require_seg),
            "classmap_version": CLASSMAP_VERSION_V3,
        },
        "counts": dict(counts),
        "notes": [
            "Waymo Perception v2 semantic IDs are mapped: ROAD->DA, LANE_MARKER->rm_lane_marker, ROAD_MARKER->rm_road_marker_non_lane.",
            "Stop line and lane subclass supervision are unavailable; masks are all-255 with has_rm_stop_line=0 and has_rm_lane_subclass=0.",
            "Detection labels cover only a subset of PV26 classes; det_label_scope=subset with det_annotated_class_ids='0,4,5,10'.",
        ],
    }
    write_json(layout.report_path(), report)

    print(f"[pv26][waymo] wrote: {out_root}")
    print(f"[pv26][waymo] manifest: {layout.manifest_path()}")
    print(f"[pv26][waymo] report:   {layout.report_path()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
