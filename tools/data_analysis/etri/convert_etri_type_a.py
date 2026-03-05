#!/usr/bin/env python3
"""
Convert ETRI (Mono+Multi polygon JSON) into PV26 Type-A dataset layout.

Implements the adapter contract described in:
- docs/PV26_PRD.md
- docs/PV26_DATASET_CONVERSION_SPEC.md
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pv26.class_map import render_class_map_yaml
from pv26.constants import CLASSMAP_VERSION_V3
from pv26.dataset_layout import Pv26Layout, SPLITS
from pv26.etri import read_etri_polygon_json, rasterize_etri_type_a_masks
from pv26.manifest import ManifestRow, write_manifest_csv
from pv26.utils import list_files_recursive, sha256_file, stable_split_for_group_key, utc_now_iso, write_json


@dataclass(frozen=True)
class ConvertTask:
    index: int
    image_path: str
    label_json: str
    sample_id: str
    split: str
    source: str
    sequence: str
    frame: str
    camera_id: str
    source_group_key: str


@dataclass(frozen=True)
class ConvertTaskResult:
    index: int
    row: ManifestRow


def _fmt_duration(seconds: Optional[float]) -> str:
    if seconds is None:
        return "계산중"
    sec = max(0, int(round(float(seconds))))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _log_progress(*, stage: str, done: int, total: int, started_at: float) -> None:
    elapsed = max(1e-9, time.monotonic() - started_at)
    rate = float(done) / elapsed
    eta: Optional[float]
    if done <= 0 or rate <= 1e-9:
        eta = None
    else:
        eta = max(0.0, float(total - done) / rate)
    pct = 100.0 if total <= 0 else (100.0 * float(done) / float(total))
    print(
        f"[pv26][etri][{stage}] {done:,}/{total:,} ({pct:6.2f}%) | "
        f"속도 {rate:,.2f} sample/초 | 남은 시간 {_fmt_duration(eta)}",
        flush=True,
    )


def _progress_interval(total: int, *, target_updates: int = 120, min_interval: int = 200) -> int:
    if total <= 0:
        return 1
    return max(min_interval, max(1, total // max(1, target_updates)))


def _determine_default_workers() -> int:
    cpu = os.cpu_count() or 1
    return max(1, min(12, cpu - 2 if cpu > 2 else cpu))


def _infer_split_from_relpath(relpath: Path) -> Optional[str]:
    for p in relpath.parts:
        q = p.lower()
        if q in SPLITS:
            return q
    return None


def _save_u8_mask(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr.astype(np.uint8), mode="L").save(path, format="PNG", optimize=False)


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


def _process_convert_task(task: ConvertTask, *, out_root: Path) -> ConvertTaskResult:

    out_img_rel = str(Path("images") / task.split / f"{task.sample_id}.jpg")
    out_det_rel = str(Path("labels_det") / task.split / f"{task.sample_id}.txt")
    out_da_rel = str(Path("labels_seg_da") / task.split / f"{task.sample_id}.png")
    out_rm_lane_rel = str(Path("labels_seg_rm_lane_marker") / task.split / f"{task.sample_id}.png")
    out_rm_road_rel = str(Path("labels_seg_rm_road_marker_non_lane") / task.split / f"{task.sample_id}.png")
    out_rm_stop_rel = str(Path("labels_seg_rm_stop_line") / task.split / f"{task.sample_id}.png")
    out_rm_lane_sub_rel = str(Path("labels_seg_rm_lane_subclass") / task.split / f"{task.sample_id}.png")

    w, h = _save_image_as_jpg(out_root / out_img_rel, Path(task.image_path))

    data = read_etri_polygon_json(Path(task.label_json))
    da, rm_lane, rm_road, rm_stop, rm_lane_sub = rasterize_etri_type_a_masks(data, width=w, height=h)

    _save_det_txt(out_root / out_det_rel, [])
    _save_u8_mask(out_root / out_da_rel, da)
    _save_u8_mask(out_root / out_rm_lane_rel, rm_lane)
    _save_u8_mask(out_root / out_rm_road_rel, rm_road)
    _save_u8_mask(out_root / out_rm_stop_rel, rm_stop)
    _save_u8_mask(out_root / out_rm_lane_sub_rel, rm_lane_sub)

    semantic_rel = ""
    row = ManifestRow(
        sample_id=task.sample_id,
        split=task.split,
        source=task.source,
        sequence=task.sequence,
        frame=task.frame,
        camera_id=task.camera_id,
        timestamp_ns="",
        has_det=0,
        has_da=1,
        has_rm_lane_marker=1,
        has_rm_road_marker_non_lane=1,
        has_rm_stop_line=1,
        has_rm_lane_subclass=1,
        has_semantic_id=0,
        det_label_scope="none",
        det_annotated_class_ids="",
        image_relpath=out_img_rel,
        det_relpath=out_det_rel,
        da_relpath=out_da_rel,
        rm_lane_marker_relpath=out_rm_lane_rel,
        rm_road_marker_non_lane_relpath=out_rm_road_rel,
        rm_stop_line_relpath=out_rm_stop_rel,
        rm_lane_subclass_relpath=out_rm_lane_sub_rel,
        semantic_relpath=semantic_rel,
        width=w,
        height=h,
        weather_tag="unknown",
        time_tag="unknown",
        scene_tag="unknown",
        source_group_key=task.source_group_key,
    )
    return ConvertTaskResult(index=task.index, row=row)


def _process_tasks_serial(tasks: Sequence[ConvertTask], *, out_root: Path) -> List[ConvertTaskResult]:
    total = len(tasks)
    results: List[ConvertTaskResult] = []
    started = time.monotonic()
    interval = _progress_interval(total=total)
    for i, task in enumerate(tasks, start=1):
        results.append(_process_convert_task(task, out_root=out_root))
        if i % interval == 0 or i == total:
            _log_progress(stage="convert", done=i, total=total, started_at=started)
    return results


def _process_tasks_parallel(tasks: Sequence[ConvertTask], *, out_root: Path, workers: int) -> List[ConvertTaskResult]:
    total = len(tasks)
    if total == 0:
        return []

    started = time.monotonic()
    interval = _progress_interval(total=total)
    max_inflight = max(1, int(workers) * 4)
    results: List[ConvertTaskResult] = []

    with ThreadPoolExecutor(max_workers=int(workers)) as ex:
        futs = []
        done = 0
        for t in tasks:
            futs.append(ex.submit(_process_convert_task, t, out_root=out_root))
            if len(futs) >= max_inflight:
                done_set, pending = wait(futs, return_when=FIRST_COMPLETED)
                for fut in done_set:
                    results.append(fut.result())
                futs = list(pending)
                done += len(done_set)
                if done % interval == 0:
                    _log_progress(stage="convert", done=done, total=total, started_at=started)

        done_set, _pending = wait(futs)
        for fut in done_set:
            results.append(fut.result())
        _log_progress(stage="convert", done=len(results), total=total, started_at=started)

    return results


def _write_checksums_parallel(*, out_root: Path, files: Sequence[Path], workers: int, out_path: Path) -> None:
    if not files:
        out_path.write_text("", encoding="utf-8")
        return

    def _worker(path: Path) -> Tuple[str, str]:
        relp = path.relative_to(out_root).as_posix()
        return sha256_file(path), relp

    started = time.monotonic()
    results: List[Tuple[str, str]] = []
    done = 0
    total = len(files)
    interval = _progress_interval(total=total, target_updates=80, min_interval=500)
    max_inflight = max(1, int(workers) * 8)

    with ThreadPoolExecutor(max_workers=int(workers)) as ex:
        futs = []
        for p in files:
            futs.append(ex.submit(_worker, p))
            if len(futs) >= max_inflight:
                done_set, pending = wait(futs, return_when=FIRST_COMPLETED)
                for fut in done_set:
                    results.append(fut.result())
                futs = list(pending)
                done += len(done_set)
                if done % interval == 0:
                    _log_progress(stage="checksums", done=done, total=total, started_at=started)

        done_set, _pending = wait(futs)
        for fut in done_set:
            results.append(fut.result())
        done = len(results)
        _log_progress(stage="checksums", done=done, total=total, started_at=started)

    results.sort(key=lambda x: x[1])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for h, relp in results:
            f.write(f"{h}  {relp}\n")


def _parse_etri_mono_name(name: str) -> Optional[Tuple[str, str]]:
    stem = Path(name).stem
    if stem.endswith("_leftImg8bit"):
        stem = stem[: -len("_leftImg8bit")]
    parts = stem.split("_")
    if len(parts) < 2:
        return None
    place = "_".join(parts[:-1])
    frame = parts[-1].zfill(6)
    return place, frame


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--etri-root",
        type=Path,
        default=Path("datasets/ETRI"),
        help="ETRI dataset root (default: datasets/ETRI)",
    )
    p.add_argument(
        "--out-root",
        type=Path,
        default=Path("datasets/pv26_v1_etri"),
        help="Output root (default: datasets/pv26_v1_etri)",
    )
    p.add_argument("--seed", type=int, default=0, help="Split seed for mono set (deterministic)")
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument(
        "--splits",
        type=str,
        default="train,val,test",
        help="Comma-separated split filter (default: train,val,test)",
    )
    p.add_argument("--limit", type=int, default=0, help="Limit number of samples processed (0=all)")
    p.add_argument(
        "--workers",
        type=int,
        default=_determine_default_workers(),
        help="Number of parallel workers for per-sample conversion (default: min(12, cpu-2))",
    )
    p.add_argument("--run-id", type=str, default="", help="Optional run id for conversion report")
    return p


def main() -> int:
    args = build_argparser().parse_args()
    etri_root: Path = args.etri_root
    out_root: Path = args.out_root
    workers = max(1, int(args.workers))
    split_filter = {s.strip().lower() for s in str(args.splits).split(",") if s.strip()}
    if not split_filter:
        split_filter = set(SPLITS)

    layout = Pv26Layout(out_root=out_root)
    layout.ensure_dirs()

    tasks: List[ConvertTask] = []
    source = "etri"

    # Mono (no split folders)
    mono_images = etri_root / "MonoCameraSemanticSegmentation" / "JPEGImages_mosaic"
    mono_labels = etri_root / "MonoCameraSemanticSegmentation" / "labels"
    if mono_images.exists() and mono_labels.exists():
        for img_path in sorted(mono_images.glob("*_leftImg8bit.*")):
            if img_path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                continue
            pair = _parse_etri_mono_name(img_path.name)
            if pair is None:
                continue
            place, frame = pair
            label_path = mono_labels / f"{place}_{int(frame):06d}_gtFine_polygons.json"
            if not label_path.exists():
                continue
            camera_id = "cam0"
            sequence = f"mono_{place}"
            sample_id = f"{source}__{sequence}__{frame}__{camera_id}"
            source_group_key = f"{source}::{sequence}"
            split = stable_split_for_group_key(
                source_group_key,
                seed=int(args.seed),
                train_ratio=float(args.train_ratio),
                val_ratio=float(args.val_ratio),
            )
            if split not in split_filter:
                continue
            tasks.append(
                ConvertTask(
                    index=len(tasks),
                    image_path=str(img_path),
                    label_json=str(label_path),
                    sample_id=sample_id,
                    split=split,
                    source=source,
                    sequence=sequence,
                    frame=frame,
                    camera_id=camera_id,
                    source_group_key=source_group_key,
                )
            )
            if args.limit and int(args.limit) > 0 and len(tasks) >= int(args.limit):
                break

    # Multi (train/val folders, left/right)
    multi_root = etri_root / "Multi Camera Semantic Segmentation"
    multi_labels = multi_root / "labels"
    multi_left = multi_root / "leftImg"
    multi_right = multi_root / "rightImg"
    if multi_labels.exists() and multi_left.exists() and multi_right.exists():
        for label_path in sorted(multi_labels.rglob("*_gtFine_polygons.json")):
            rel = label_path.relative_to(multi_labels)
            split = _infer_split_from_relpath(rel)
            if split is None:
                split = "train"
            if split not in split_filter:
                continue

            # rel example: train/<sequence>/000044_gtFine_polygons.json
            if len(rel.parts) < 3:
                continue
            seq = rel.parts[1]
            frame = Path(rel.parts[-1]).stem.replace("_gtFine_polygons", "")
            if not frame.isdigit():
                continue
            frame = frame.zfill(6)

            left_img = multi_left / rel.parent / f"{frame}_leftImg8bit.png"
            right_img = multi_right / rel.parent / f"{frame}_rightImg8bit.png"

            for cam_id, img_path in [("camL", left_img), ("camR", right_img)]:
                if not img_path.exists():
                    continue
                sequence = f"multi_{seq}"
                sample_id = f"{source}__{sequence}__{frame}__{cam_id}"
                source_group_key = f"{source}::{sequence}"
                tasks.append(
                    ConvertTask(
                        index=len(tasks),
                        image_path=str(img_path),
                        label_json=str(label_path),
                        sample_id=sample_id,
                        split=split,
                        source=source,
                        sequence=sequence,
                        frame=frame,
                        camera_id=cam_id,
                        source_group_key=source_group_key,
                    )
                )
                if args.limit and int(args.limit) > 0 and len(tasks) >= int(args.limit):
                    break
            if args.limit and int(args.limit) > 0 and len(tasks) >= int(args.limit):
                break

    print(f"[pv26][etri] tasks: {len(tasks):,} samples | out_root={out_root}", flush=True)
    if not tasks:
        print("[pv26][etri] no tasks found (check --etri-root paths)", flush=True)
        return 2

    conv_started = time.monotonic()
    if workers <= 1:
        results = _process_tasks_serial(tasks, out_root=out_root)
    else:
        results = _process_tasks_parallel(tasks, out_root=out_root, workers=workers)
    results.sort(key=lambda x: x.index)
    rows = [r.row for r in results]
    print(f"[pv26][etri] converted: {len(rows):,} samples in {_fmt_duration(time.monotonic() - conv_started)}", flush=True)

    counts = Counter()
    for row in rows:
        counts["total"] += 1
        counts[f"split:{row.split}"] += 1
        counts[f"has_da:{int(row.has_da)}"] += 1

    meta_started = time.monotonic()
    layout.meta_dir().mkdir(parents=True, exist_ok=True)
    layout.class_map_path().write_text(render_class_map_yaml(classmap_version=CLASSMAP_VERSION_V3), encoding="utf-8")
    write_manifest_csv(layout.manifest_path(), rows)

    with layout.source_stats_path().open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["source", "num_samples"])
        writer.writerow([source, counts["total"]])

    exported_files = list_files_recursive(out_root)
    _write_checksums_parallel(out_root=out_root, files=exported_files, workers=workers, out_path=layout.checksums_path())

    report = {
        "converter": "convert_etri_type_a.py",
        "converter_version": "0.1.0",
        "spec": "docs/PV26_DATASET_CONVERSION_SPEC.md v1.5",
        "timestamp_utc": utc_now_iso(),
        "run_id": args.run_id,
        "config": {
            "etri_root": str(etri_root),
            "out_root": str(out_root),
            "seed": int(args.seed),
            "train_ratio": float(args.train_ratio),
            "val_ratio": float(args.val_ratio),
            "splits": str(args.splits),
            "limit": int(args.limit),
            "workers": int(workers),
            "classmap_version": CLASSMAP_VERSION_V3,
        },
        "counts": dict(counts),
        "notes": [
            "ETRI polygon JSON (Mono+Multi) is rasterized into DA/RM masks.",
            "Lane subclass mapping: whsol/whdot/yesol/yedot -> {1..4}; other lane-like pixels are ignored(255) in rm_lane_subclass.",
            "out of roi polygons are applied as ignore(255) to all masks when present.",
        ],
    }
    write_json(layout.report_path(), report)

    print(f"[pv26][etri] wrote: {out_root}")
    print(f"[pv26][etri] manifest: {layout.manifest_path()}")
    print(f"[pv26][etri] report:   {layout.report_path()}")
    print(f"[pv26][etri] meta:     {_fmt_duration(time.monotonic() - meta_started)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
