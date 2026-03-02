#!/usr/bin/env python3
"""
Convert BDD100K assets into PV26 Type-A dataset layout (BDD-only, BDD adapter).

Implements the first executable slice described in:
- docs/PRD.md
- docs/DATASET_CONVERSION_SPEC.md
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pv26.bdd import (
    bdd_record_to_rm_masks,
    bdd_record_tags,
    bdd_record_to_image_name,
    bdd_record_to_yolo_lines,
    iter_bdd_label_records,
    parse_bdd_filename_for_sequence_and_frame,
)
from pv26.class_map import render_class_map_yaml
from pv26.constants import CLASSMAP_VERSION_V2
from pv26.dataset_layout import Pv26Layout, SPLITS
from pv26.manifest import ManifestRow, write_manifest_csv
from pv26.masks import (
    IGNORE_VALUE,
    SemanticComposeResult,
    compose_semantic_id_v2,
    convert_bdd_drivable_id_to_da_mask_u8,
    make_all_ignore_mask,
)
from pv26.utils import list_files_recursive, sha256_file, stable_split_for_group_key, utc_now_iso, write_json


def _load_u8_mask(path: Path) -> np.ndarray:
    arr = np.array(Image.open(path))
    if arr.ndim != 2:
        raise ValueError(f"expected single-channel mask: {path} shape={arr.shape}")
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8, copy=False)
    return arr


def _save_u8_mask(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr, mode="L").save(path, format="PNG", optimize=False)


def _save_det_txt(path: Path, lines: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for ln in lines:
            f.write(ln.rstrip() + "\n")


def _save_image_as_jpg(dst_path: Path, src_path: Path) -> Tuple[int, int]:
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src_path) as im:
        im = im.convert("RGB")
        w, h = im.size
        im.save(dst_path, format="JPEG", quality=95, optimize=True)
    return int(w), int(h)


def _index_records_by_image_name(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for r in records:
        name = bdd_record_to_image_name(r)
        if not name:
            # For per-image json mode, allow using the json filename stem.
            p = r.get("__json_path__")
            if isinstance(p, str):
                name = Path(p).stem + ".jpg"
        if name:
            out[name] = r
            out[Path(name).stem] = r
    return out


def _should_include_sample(
    *,
    weather_tag: str,
    time_tag: str,
    scene_tag: str,
    include_rain: bool,
    include_night: bool,
    allow_unknown_tags: bool,
) -> bool:
    if not include_rain and weather_tag != "dry":
        if allow_unknown_tags and weather_tag == "unknown":
            pass
        else:
            return False
    if not include_night and time_tag != "day":
        if allow_unknown_tags and time_tag == "unknown":
            pass
        else:
            return False
    return True


def _infer_split_from_relpath(relpath: Path) -> Optional[str]:
    if not relpath.parts:
        return None
    first = relpath.parts[0].lower()
    if first in SPLITS:
        return first
    return None


def _index_drivable_masks(drivable_root: Optional[Path]) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    if drivable_root is None or not drivable_root.exists():
        return out
    for p in drivable_root.rglob("*.png"):
        stem = p.stem
        if stem.endswith("_drivable_id"):
            stem = stem[: -len("_drivable_id")]
        out.setdefault(stem, p)
    return out


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--images-root", type=Path, required=True, help="BDD images root directory")
    p.add_argument(
        "--labels",
        type=Path,
        required=False,
        help="BDD labels (dir of per-image json, or a single json file). If omitted, has_det=0 for all.",
    )
    p.add_argument(
        "--drivable-root",
        type=Path,
        required=False,
        help="BDD drivable id mask directory (png). If omitted, has_da=0 for all.",
    )
    p.add_argument(
        "--out-root",
        type=Path,
        default=Path("datasets/pv26_v1"),
        help="Output root (default: datasets/pv26_v1)",
    )
    p.add_argument("--seed", type=int, default=0, help="Split seed (deterministic)")
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--min-box-area-px", type=int, default=0)
    p.add_argument("--limit", type=int, default=0, help="Limit number of images processed (0=all)")
    p.add_argument(
        "--splits",
        type=str,
        default="train,val,test",
        help="Comma-separated split filter when images_root contains split folders (default: train,val,test)",
    )

    p.add_argument("--include-rain", action="store_true", help="Include rainy samples (default: excluded)")
    p.add_argument("--include-night", action="store_true", help="Include night samples (default: excluded)")
    p.add_argument(
        "--allow-unknown-tags",
        action="store_true",
        help="Include samples with unknown weather/time tags (default: excluded by MVP policy)",
    )
    p.add_argument("--run-id", type=str, default="", help="Optional run id for conversion report")
    return p


def main() -> int:
    args = build_argparser().parse_args()

    images_root: Path = args.images_root
    if not images_root.exists():
        raise SystemExit(f"--images-root not found: {images_root}")

    labels_path: Optional[Path] = args.labels
    drivable_root: Optional[Path] = args.drivable_root

    out_root: Path = args.out_root
    split_filter = {s.strip() for s in str(args.splits).split(",") if s.strip()}
    invalid = split_filter - set(SPLITS)
    if invalid:
        raise SystemExit(f"--splits has invalid values: {sorted(invalid)} (allowed: {list(SPLITS)})")
    layout = Pv26Layout(out_root=out_root)
    layout.ensure_dirs()

    # Load label records (optional).
    records: List[Dict[str, Any]] = []
    if labels_path is not None:
        if not labels_path.exists():
            raise SystemExit(f"--labels not found: {labels_path}")
        records = list(iter_bdd_label_records(labels_path))
    rec_by_name = _index_records_by_image_name(records) if records else {}
    drivable_by_stem = _index_drivable_masks(drivable_root)

    # Discover images.
    image_paths = sorted(
        [p for p in images_root.rglob("*") if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    )
    rows: List[ManifestRow] = []
    counts = Counter()
    skipped = Counter()

    for img_path in image_paths:
        rel = img_path.relative_to(images_root)
        img_name = img_path.name
        rec = rec_by_name.get(img_name) or rec_by_name.get(Path(img_name).stem)
        weather_tag, time_tag, scene_tag = ("unknown", "unknown", "unknown")
        if rec is not None:
            weather_tag, time_tag, scene_tag = bdd_record_tags(rec)

        if rec is not None:
            if not _should_include_sample(
                weather_tag=weather_tag,
                time_tag=time_tag,
                scene_tag=scene_tag,
                include_rain=bool(args.include_rain),
                include_night=bool(args.include_night),
                allow_unknown_tags=bool(args.allow_unknown_tags),
            ):
                skipped["domain_filter"] += 1
                continue

        sequence, frame = parse_bdd_filename_for_sequence_and_frame(img_name)
        camera_id = "cam0"
        source = "bdd100k"
        sample_id = f"{source}__{sequence}__{frame}__{camera_id}"

        # Group key for split leakage prevention: {source, sequence}
        source_group_key = f"{source}::{sequence}"
        split = _infer_split_from_relpath(rel)
        if split is None:
            split = stable_split_for_group_key(
                source_group_key,
                seed=int(args.seed),
                train_ratio=float(args.train_ratio),
                val_ratio=float(args.val_ratio),
            )
        if split not in SPLITS:
            raise RuntimeError(f"invalid split computed: {split}")
        if split not in split_filter:
            skipped["split_filter"] += 1
            continue

        if args.limit and args.limit > 0 and counts["total"] >= int(args.limit):
            break

        out_img_rel = str(Path("images") / split / f"{sample_id}.jpg")
        out_det_rel = str(Path("labels_det") / split / f"{sample_id}.txt")
        out_da_rel = str(Path("labels_seg_da") / split / f"{sample_id}.png")
        out_rm_lane_rel = str(Path("labels_seg_rm_lane_marker") / split / f"{sample_id}.png")
        out_rm_road_rel = str(Path("labels_seg_rm_road_marker_non_lane") / split / f"{sample_id}.png")
        out_rm_stop_rel = str(Path("labels_seg_rm_stop_line") / split / f"{sample_id}.png")
        out_sem_rel = str(Path("labels_semantic_id") / split / f"{sample_id}.png")

        # Write image (re-encode to jpg for consistent contract).
        w, h = _save_image_as_jpg(out_root / out_img_rel, img_path)

        # Detection (optional).
        if rec is None:
            has_det = 0
            det_scope = "none"
            det_annotated = ""
            det_lines: List[str] = []
        else:
            has_det = 1
            det_scope = "full"
            det_annotated = ""
            det_lines = bdd_record_to_yolo_lines(
                rec,
                width=w,
                height=h,
                min_box_area_px=int(args.min_box_area_px),
            )
        _save_det_txt(out_root / out_det_rel, det_lines)

        # DA mask (optional).
        da_mask: np.ndarray
        if drivable_root is None:
            has_da = 0
            da_mask = make_all_ignore_mask(h, w)
        else:
            stem = Path(img_name).stem
            da_src = drivable_by_stem.get(stem)
            if da_src is None or not da_src.exists():
                has_da = 0
                da_mask = make_all_ignore_mask(h, w)
            else:
                has_da = 1
                drivable_id = _load_u8_mask(da_src)
                if drivable_id.shape != (h, w):
                    raise SystemExit(f"drivable mask size mismatch: {da_src} mask={drivable_id.shape} image={(h,w)}")
                da_mask = convert_bdd_drivable_id_to_da_mask_u8(drivable_id)
        _save_u8_mask(out_root / out_da_rel, da_mask)

        # RM masks: not available for Type-A BDD in this slice; emit all-255.
        if rec is None:
            has_rm_lane = 0
            has_rm_road = 0
            has_rm_stop = 0
            rm_lane = make_all_ignore_mask(h, w)
            rm_road = make_all_ignore_mask(h, w)
            rm_stop = make_all_ignore_mask(h, w)
        else:
            rm_lane, rm_road, rm_stop, has_rm_lane, has_rm_road, has_rm_stop = bdd_record_to_rm_masks(
                rec,
                width=w,
                height=h,
                line_width=8,
            )
        _save_u8_mask(out_root / out_rm_lane_rel, rm_lane)
        _save_u8_mask(out_root / out_rm_road_rel, rm_road)
        _save_u8_mask(out_root / out_rm_stop_rel, rm_stop)

        # Semantic ID: only when all channels are supervised and contain no ignore(255).
        sem_ok = False
        if has_da and has_rm_lane and has_rm_road and has_rm_stop:
            sem: SemanticComposeResult = compose_semantic_id_v2(da_mask, rm_lane, rm_road, rm_stop)
            if sem.ok:
                sem_ok = True
                _save_u8_mask(out_root / out_sem_rel, sem.semantic_id)
        has_sem = 1 if sem_ok else 0
        semantic_relpath = out_sem_rel if sem_ok else ""

        row = ManifestRow(
            sample_id=sample_id,
            split=split,
            source=source,
            sequence=sequence,
            frame=frame,
            camera_id=camera_id,
            timestamp_ns="",
            has_det=has_det,
            has_da=has_da,
            has_rm_lane_marker=has_rm_lane,
            has_rm_road_marker_non_lane=has_rm_road,
            has_rm_stop_line=has_rm_stop,
            has_semantic_id=has_sem,
            det_label_scope=det_scope,
            det_annotated_class_ids=det_annotated,
            image_relpath=out_img_rel,
            det_relpath=out_det_rel,
            da_relpath=out_da_rel,
            rm_lane_marker_relpath=out_rm_lane_rel,
            rm_road_marker_non_lane_relpath=out_rm_road_rel,
            rm_stop_line_relpath=out_rm_stop_rel,
            semantic_relpath=semantic_relpath,
            width=w,
            height=h,
            weather_tag=weather_tag,
            time_tag=time_tag,
            scene_tag=scene_tag,
            source_group_key=source_group_key,
        )
        rows.append(row)
        counts[f"split:{split}"] += 1
        counts["total"] += 1
        counts[f"has_det:{has_det}"] += 1
        counts[f"has_da:{has_da}"] += 1

    # meta outputs
    (layout.meta_dir()).mkdir(parents=True, exist_ok=True)
    (layout.class_map_path()).write_text(render_class_map_yaml(classmap_version=CLASSMAP_VERSION_V2), encoding="utf-8")
    write_manifest_csv(layout.manifest_path(), rows)

    # source_stats.csv (minimal)
    with layout.source_stats_path().open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source", "num_samples"])
        w.writerow(["bdd100k", counts["total"]])

    # checksums.sha256 (includes everything under out_root except itself, since it doesn't exist yet)
    exported_files = list_files_recursive(out_root)
    with layout.checksums_path().open("w", encoding="utf-8") as f:
        for p in exported_files:
            relp = p.relative_to(out_root).as_posix()
            f.write(f"{sha256_file(p)}  {relp}\n")

    report = {
        "converter": "convert_bdd_type_a.py",
        "converter_version": "0.1.0",
        "spec": "docs/DATASET_CONVERSION_SPEC.md v1.4",
        "timestamp_utc": utc_now_iso(),
        "run_id": args.run_id,
        "config": {
            "images_root": str(images_root),
            "labels": str(labels_path) if labels_path else "",
            "drivable_root": str(drivable_root) if drivable_root else "",
            "out_root": str(out_root),
            "seed": int(args.seed),
            "train_ratio": float(args.train_ratio),
            "val_ratio": float(args.val_ratio),
            "min_box_area_px": int(args.min_box_area_px),
            "include_rain": bool(args.include_rain),
            "include_night": bool(args.include_night),
            "allow_unknown_tags": bool(args.allow_unknown_tags),
            "limit": int(args.limit),
            "classmap_version": CLASSMAP_VERSION_V2,
        },
        "counts": dict(counts),
        "skipped": dict(skipped),
        "notes": [
            "Type-A BDD-only slice: lane/non-lane RM masks are rasterized from BDD lane/* poly2d labels.",
            "stop_line channel defaults to ignore(255) unless explicit stop-line category is present.",
            "semantic_id is exported only when DA+RM supervision is fully available without ignore(255).",
        ],
    }
    write_json(layout.report_path(), report)

    print(f"[pv26] wrote {len(rows)} samples to: {out_root}")
    print(f"[pv26] manifest: {layout.manifest_path()}")
    print(f"[pv26] report:   {layout.report_path()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
