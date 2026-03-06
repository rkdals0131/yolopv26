#!/usr/bin/env python3
"""
Convert RLMD (1080p + RLMD-AC labeled splits) into PV26 Type-A dataset layout.

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
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pv26.class_map import render_class_map_yaml
from pv26.constants import CLASSMAP_VERSION_V3
from pv26.dataset_layout import Pv26Layout
from pv26.manifest import ManifestRow, write_manifest_csv
from pv26.masks import make_all_ignore_mask
from pv26.rlmd import RlmdRgbClass, load_rlmd_palette, rgb_to_code_u32, rlmd_code_mask_to_pv26_rm_masks
from pv26.utils import list_files_recursive, sha256_file, utc_now_iso, write_json


@dataclass(frozen=True)
class ConvertTask:
    index: int
    image_path: str
    label_path: str
    sample_id: str
    split: str
    source: str
    sequence: str
    frame: str
    camera_id: str
    weather_tag: str
    time_tag: str
    scene_tag: str
    source_group_key: str


@dataclass(frozen=True)
class ConvertTaskResult:
    index: int
    row: ManifestRow
    unknown_pixels: int


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
        f"[pv26][rlmd][{stage}] {done:,}/{total:,} ({pct:6.2f}%) | "
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

def _process_convert_task(
    task: ConvertTask, *, out_root: Path, palette: Dict[int, RlmdRgbClass]
) -> ConvertTaskResult:

    out_img_rel = str(Path("images") / task.split / f"{task.sample_id}.jpg")
    out_det_rel = str(Path("labels_det") / task.split / f"{task.sample_id}.txt")
    out_da_rel = str(Path("labels_seg_da") / task.split / f"{task.sample_id}.png")
    out_rm_lane_rel = str(Path("labels_seg_rm_lane_marker") / task.split / f"{task.sample_id}.png")
    out_rm_road_rel = str(Path("labels_seg_rm_road_marker_non_lane") / task.split / f"{task.sample_id}.png")
    out_rm_stop_rel = str(Path("labels_seg_rm_stop_line") / task.split / f"{task.sample_id}.png")
    out_rm_lane_sub_rel = str(Path("labels_seg_rm_lane_subclass") / task.split / f"{task.sample_id}.png")

    w, h = _save_image_as_jpg(out_root / out_img_rel, Path(task.image_path))

    with Image.open(task.label_path) as im:
        im = im.convert("RGB")
        if im.size != (w, h):
            im = im.resize((w, h), resample=Image.NEAREST)
        rgb = np.array(im, dtype=np.uint8)

    code = rgb_to_code_u32(rgb)
    rm_lane, rm_road, rm_stop, rm_lane_sub, unknown_pixels = rlmd_code_mask_to_pv26_rm_masks(code, palette=palette)

    da = make_all_ignore_mask(h, w)

    _save_det_txt(out_root / out_det_rel, [])
    _save_u8_mask(out_root / out_da_rel, da)
    _save_u8_mask(out_root / out_rm_lane_rel, rm_lane)
    _save_u8_mask(out_root / out_rm_road_rel, rm_road)
    _save_u8_mask(out_root / out_rm_stop_rel, rm_stop)
    _save_u8_mask(out_root / out_rm_lane_sub_rel, rm_lane_sub)

    row = ManifestRow(
        sample_id=task.sample_id,
        split=task.split,
        source=task.source,
        sequence=task.sequence,
        frame=task.frame,
        camera_id=task.camera_id,
        timestamp_ns="",
        has_det=0,
        has_da=0,
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
        semantic_relpath="",
        width=w,
        height=h,
        weather_tag=task.weather_tag,
        time_tag=task.time_tag,
        scene_tag=task.scene_tag,
        source_group_key=task.source_group_key,
    )
    return ConvertTaskResult(index=task.index, row=row, unknown_pixels=int(unknown_pixels))


def _process_tasks_serial(
    tasks: Sequence[ConvertTask], *, out_root: Path, palette: Dict[int, RlmdRgbClass]
) -> List[ConvertTaskResult]:
    total = len(tasks)
    if total == 0:
        return []
    results: List[ConvertTaskResult] = []
    started = time.monotonic()
    interval = _progress_interval(total=total)
    for i, t in enumerate(tasks, start=1):
        results.append(_process_convert_task(t, out_root=out_root, palette=palette))
        if i % interval == 0 or i == total:
            _log_progress(stage="convert", done=i, total=total, started_at=started)
    return results


def _process_tasks_parallel(
    tasks: Sequence[ConvertTask], *, out_root: Path, palette: Dict[int, RlmdRgbClass], workers: int
) -> List[ConvertTaskResult]:
    total = len(tasks)
    if total == 0:
        return []

    started = time.monotonic()
    interval = _progress_interval(total=total)
    max_inflight = max(1, int(workers) * 4)
    results: List[ConvertTaskResult] = []
    done = 0

    with ThreadPoolExecutor(max_workers=int(workers)) as ex:
        futs = []
        for t in tasks:
            futs.append(ex.submit(_process_convert_task, t, out_root=out_root, palette=palette))
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
        _log_progress(stage="checksums", done=len(results), total=total, started_at=started)

    results.sort(key=lambda x: x[1])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for h, relp in results:
            f.write(f"{h}  {relp}\n")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--rlmd-root",
        type=Path,
        default=Path("datasets/RLMD"),
        help="RLMD dataset root (default: datasets/RLMD)",
    )
    p.add_argument(
        "--out-root",
        type=Path,
        default=Path("datasets/pv26_v1_rlmd"),
        help="Output root (default: datasets/pv26_v1_rlmd)",
    )
    p.add_argument(
        "--include-ac",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include RLMD-AC labeled splits when labels exist (default: enabled)",
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


def _iter_pairs(images_dir: Path, labels_dir: Path) -> List[Tuple[Path, Path]]:
    pairs: List[Tuple[Path, Path]] = []
    for lab in sorted(labels_dir.glob("*.png")):
        img = images_dir / f"{lab.stem}.jpg"
        if img.exists():
            pairs.append((img, lab))
    return pairs


def main() -> int:
    args = build_argparser().parse_args()
    rlmd_root: Path = args.rlmd_root
    out_root: Path = args.out_root
    workers = max(1, int(args.workers))

    layout = Pv26Layout(out_root=out_root)
    layout.ensure_dirs()

    palette_csv = rlmd_root / "RLMD_1080p" / "rlmd.csv"
    if not palette_csv.exists():
        print(f"[pv26][rlmd] missing palette csv: {palette_csv}", flush=True)
        return 2
    palette = load_rlmd_palette(palette_csv)

    tasks: List[ConvertTask] = []
    source = "rlmd"

    # RLMD_1080p
    base = rlmd_root / "RLMD_1080p"
    for split in ["train", "val"]:
        pairs = _iter_pairs(base / "images" / split, base / "labels" / split)
        for img_path, lab_path in pairs:
            sample_id_part = lab_path.stem
            sequence = f"1080p_{split}_{sample_id_part}"
            frame = "000000"
            camera_id = "cam0"
            sample_id = f"{source}__{sequence}__{frame}__{camera_id}"
            source_group_key = f"{source}::{sequence}"
            tasks.append(
                ConvertTask(
                    index=len(tasks),
                    image_path=str(img_path),
                    label_path=str(lab_path),
                    sample_id=sample_id,
                    split=split,
                    source=source,
                    sequence=sequence,
                    frame=frame,
                    camera_id=camera_id,
                    weather_tag="unknown",
                    time_tag="unknown",
                    scene_tag="unknown",
                    source_group_key=source_group_key,
                )
            )
            if args.limit and int(args.limit) > 0 and len(tasks) >= int(args.limit):
                break
        if args.limit and int(args.limit) > 0 and len(tasks) >= int(args.limit):
            break

    # RLMD-AC (labeled splits only)
    if args.include_ac:
        ac = rlmd_root / "RLMD-AC"
        for cond, wt, tt in [
            ("clear", "dry", "day"),
            ("night", "dry", "night"),
            ("rainy", "rain", "day"),
        ]:
            for split in ["train", "val"]:
                labels_dir = ac / cond / split / "labels"
                images_dir = ac / cond / split / "images"
                if not labels_dir.exists():
                    continue
                pairs = _iter_pairs(images_dir, labels_dir)
                if not pairs:
                    continue
                for img_path, lab_path in pairs:
                    sample_id_part = lab_path.stem
                    sequence = f"ac_{cond}_{split}_{sample_id_part}"
                    frame = "000000"
                    camera_id = "cam0"
                    sample_id = f"{source}__{sequence}__{frame}__{camera_id}"
                    source_group_key = f"{source}::{sequence}"
                    tasks.append(
                        ConvertTask(
                            index=len(tasks),
                            image_path=str(img_path),
                            label_path=str(lab_path),
                            sample_id=sample_id,
                            split=split,
                            source=source,
                            sequence=sequence,
                            frame=frame,
                            camera_id=camera_id,
                            weather_tag=wt,
                            time_tag=tt,
                            scene_tag="unknown",
                            source_group_key=source_group_key,
                        )
                    )
                    if args.limit and int(args.limit) > 0 and len(tasks) >= int(args.limit):
                        break
                if args.limit and int(args.limit) > 0 and len(tasks) >= int(args.limit):
                    break
            if args.limit and int(args.limit) > 0 and len(tasks) >= int(args.limit):
                break

    print(f"[pv26][rlmd] tasks: {len(tasks):,} samples | out_root={out_root}", flush=True)
    if not tasks:
        print("[pv26][rlmd] no tasks found (check --rlmd-root paths)", flush=True)
        return 2

    conv_started = time.monotonic()
    if workers <= 1:
        results = _process_tasks_serial(tasks, out_root=out_root, palette=palette)
    else:
        results = _process_tasks_parallel(tasks, out_root=out_root, palette=palette, workers=workers)
    results.sort(key=lambda x: x.index)
    rows = [r.row for r in results]

    counts = Counter()
    unknown_total = 0
    for r in results:
        counts["total"] += 1
        counts[f"split:{r.row.split}"] += 1
        unknown_total += int(r.unknown_pixels)
    counts["unknown_pixels_total"] = int(unknown_total)

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
        "converter": "convert_rlmd_type_a.py",
        "converter_version": "0.1.0",
        "spec": "docs/PV26_DATASET_CONVERSION_SPEC.md v1.5",
        "timestamp_utc": utc_now_iso(),
        "run_id": args.run_id,
        "config": {
            "rlmd_root": str(rlmd_root),
            "out_root": str(out_root),
            "include_ac": bool(args.include_ac),
            "limit": int(args.limit),
            "workers": int(workers),
            "palette_csv": str(palette_csv),
            "classmap_version": CLASSMAP_VERSION_V3,
        },
        "counts": dict(counts),
        "notes": [
            "RLMD uses palette RGB masks. Output DA is not available (all-255, has_da=0).",
            "Lane subclass mapping supports white/yellow solid/dashed; other lane-marker pixels are ignored(255) in rm_lane_subclass.",
            "RLMD-AC is included only when labels exist for the split.",
        ],
    }
    write_json(layout.report_path(), report)

    print(f"[pv26][rlmd] converted: {len(rows):,} samples in {_fmt_duration(time.monotonic() - conv_started)}", flush=True)
    print(f"[pv26][rlmd] wrote: {out_root}")
    print(f"[pv26][rlmd] manifest: {layout.manifest_path()}")
    print(f"[pv26][rlmd] report:   {layout.report_path()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
    palette = load_rlmd_palette(palette_csv)
