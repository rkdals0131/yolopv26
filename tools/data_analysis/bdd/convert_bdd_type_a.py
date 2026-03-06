#!/usr/bin/env python3
"""
Convert BDD100K assets into PV26 Type-A dataset layout (BDD-only, BDD adapter).

Implements the first executable slice described in:
- docs/PV26_PRD.md
- docs/PV26_DATASET_CONVERSION_SPEC.md
"""

from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import os
import sys
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pv26.bdd import (
    bdd_record_to_rm_masks_with_lane_subclass,
    bdd_record_tags,
    bdd_record_to_image_name,
    bdd_record_to_yolo_lines,
    iter_bdd_label_records,
    parse_bdd_filename_for_sequence_and_frame,
)
from pv26.class_map import render_class_map_yaml
from pv26.constants import CLASSMAP_VERSION_V3
from pv26.dataset_layout import Pv26Layout, SPLITS
from pv26.manifest import ManifestRow, write_manifest_csv
from pv26.masks import (
    SemanticComposeResult,
    compose_semantic_id_v3,
    convert_bdd_drivable_id_to_da_mask_u8,
    make_all_ignore_mask,
)
from pv26.utils import list_files_recursive, sha256_file, stable_split_for_group_key, utc_now_iso, write_json


# Global worker context. With fork-based workers this is inherited by child processes.
_WORKER_REC_BY_NAME: Dict[str, Dict[str, Any]] = {}
_WORKER_DRIVABLE_BY_STEM: Dict[str, Path] = {}
_WORKER_OUT_ROOT: Optional[Path] = None
_WORKER_MIN_BOX_AREA_PX: int = 0
_WORKER_HAS_DRIVABLE_ROOT: bool = False


@dataclass(frozen=True)
class ConvertTask:
    index: int
    img_path: str
    img_name: str
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


def _fmt_duration(seconds: Optional[float]) -> str:
    if seconds is None:
        return "계산중"
    sec = max(0, int(round(float(seconds))))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _log_progress(
    *,
    stage: str,
    meaning: str,
    done: int,
    total: int,
    started_at: float,
    unit: str,
) -> None:
    elapsed = max(1e-9, time.monotonic() - started_at)
    rate = float(done) / elapsed
    eta: Optional[float]
    if done <= 0 or rate <= 1e-9:
        eta = None
    else:
        eta = max(0.0, float(total - done) / rate)
    pct = 100.0 if total <= 0 else (100.0 * float(done) / float(total))
    print(
        f"[pv26][로딩][{stage}] {done:,}/{total:,} ({pct:6.2f}%) | "
        f"속도 {rate:,.2f} {unit}/초 | 남은 시간 {_fmt_duration(eta)} | 의미: {meaning}",
        flush=True,
    )


def _progress_interval(total: int, *, target_updates: int = 100, min_interval: int = 500) -> int:
    if total <= 0:
        return 1
    return max(min_interval, max(1, total // max(1, target_updates)))


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


def _set_worker_context(
    *,
    rec_by_name: Dict[str, Dict[str, Any]],
    drivable_by_stem: Dict[str, Path],
    out_root: Path,
    min_box_area_px: int,
    has_drivable_root: bool,
) -> None:
    global _WORKER_REC_BY_NAME
    global _WORKER_DRIVABLE_BY_STEM
    global _WORKER_OUT_ROOT
    global _WORKER_MIN_BOX_AREA_PX
    global _WORKER_HAS_DRIVABLE_ROOT

    _WORKER_REC_BY_NAME = rec_by_name
    _WORKER_DRIVABLE_BY_STEM = drivable_by_stem
    _WORKER_OUT_ROOT = out_root
    _WORKER_MIN_BOX_AREA_PX = int(min_box_area_px)
    _WORKER_HAS_DRIVABLE_ROOT = bool(has_drivable_root)


def _process_convert_task(task: ConvertTask) -> ConvertTaskResult:
    if _WORKER_OUT_ROOT is None:
        raise RuntimeError("worker context is not initialized")

    out_root = _WORKER_OUT_ROOT
    img_path = Path(task.img_path)
    img_name = task.img_name
    rec = _WORKER_REC_BY_NAME.get(img_name) or _WORKER_REC_BY_NAME.get(Path(img_name).stem)

    out_img_rel = str(Path("images") / task.split / f"{task.sample_id}.jpg")
    out_det_rel = str(Path("labels_det") / task.split / f"{task.sample_id}.txt")
    out_da_rel = str(Path("labels_seg_da") / task.split / f"{task.sample_id}.png")
    out_rm_lane_rel = str(Path("labels_seg_rm_lane_marker") / task.split / f"{task.sample_id}.png")
    out_rm_road_rel = str(Path("labels_seg_rm_road_marker_non_lane") / task.split / f"{task.sample_id}.png")
    out_rm_stop_rel = str(Path("labels_seg_rm_stop_line") / task.split / f"{task.sample_id}.png")
    out_rm_lane_sub_rel = str(Path("labels_seg_rm_lane_subclass") / task.split / f"{task.sample_id}.png")
    out_sem_rel = str(Path("labels_semantic_id") / task.split / f"{task.sample_id}.png")

    # Write image (re-encode to jpg for consistent contract).
    w, h = _save_image_as_jpg(out_root / out_img_rel, img_path)

    # Detection.
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
            min_box_area_px=int(_WORKER_MIN_BOX_AREA_PX),
        )
    _save_det_txt(out_root / out_det_rel, det_lines)

    # DA mask.
    da_mask: np.ndarray
    if not _WORKER_HAS_DRIVABLE_ROOT:
        has_da = 0
        da_mask = make_all_ignore_mask(h, w)
    else:
        stem = Path(img_name).stem
        da_src = _WORKER_DRIVABLE_BY_STEM.get(stem)
        if da_src is None or not da_src.exists():
            has_da = 0
            da_mask = make_all_ignore_mask(h, w)
        else:
            has_da = 1
            drivable_id = _load_u8_mask(da_src)
            if drivable_id.shape != (h, w):
                raise RuntimeError(f"drivable mask size mismatch: {da_src} mask={drivable_id.shape} image={(h, w)}")
            da_mask = convert_bdd_drivable_id_to_da_mask_u8(drivable_id)
    _save_u8_mask(out_root / out_da_rel, da_mask)

    # RM masks.
    if rec is None:
        has_rm_lane = 0
        has_rm_road = 0
        has_rm_stop = 0
        has_rm_lane_subclass = 0
        rm_lane = make_all_ignore_mask(h, w)
        rm_road = make_all_ignore_mask(h, w)
        rm_stop = make_all_ignore_mask(h, w)
        rm_lane_subclass = make_all_ignore_mask(h, w)
    else:
        (
            rm_lane,
            rm_road,
            rm_stop,
            rm_lane_subclass,
            has_rm_lane,
            has_rm_road,
            has_rm_stop,
            has_rm_lane_subclass,
        ) = bdd_record_to_rm_masks_with_lane_subclass(
            rec,
            width=w,
            height=h,
            line_width=8,
        )
    _save_u8_mask(out_root / out_rm_lane_rel, rm_lane)
    _save_u8_mask(out_root / out_rm_road_rel, rm_road)
    _save_u8_mask(out_root / out_rm_stop_rel, rm_stop)
    _save_u8_mask(out_root / out_rm_lane_sub_rel, rm_lane_subclass)

    # Semantic ID: only when all channels are supervised and contain no ignore(255).
    sem_ok = False
    if has_da and has_rm_lane_subclass and has_rm_road and has_rm_stop:
        sem: SemanticComposeResult = compose_semantic_id_v3(da_mask, rm_lane_subclass, rm_road, rm_stop)
        if sem.ok:
            sem_ok = True
            _save_u8_mask(out_root / out_sem_rel, sem.semantic_id)
    has_sem = 1 if sem_ok else 0
    semantic_relpath = out_sem_rel if sem_ok else ""

    row = ManifestRow(
        sample_id=task.sample_id,
        split=task.split,
        source=task.source,
        sequence=task.sequence,
        frame=task.frame,
        camera_id=task.camera_id,
        timestamp_ns="",
        has_det=has_det,
        has_da=has_da,
        has_rm_lane_marker=has_rm_lane,
        has_rm_road_marker_non_lane=has_rm_road,
        has_rm_stop_line=has_rm_stop,
        has_rm_lane_subclass=has_rm_lane_subclass,
        has_semantic_id=has_sem,
        det_label_scope=det_scope,
        det_annotated_class_ids=det_annotated,
        image_relpath=out_img_rel,
        det_relpath=out_det_rel,
        da_relpath=out_da_rel,
        rm_lane_marker_relpath=out_rm_lane_rel,
        rm_road_marker_non_lane_relpath=out_rm_road_rel,
        rm_stop_line_relpath=out_rm_stop_rel,
        rm_lane_subclass_relpath=out_rm_lane_sub_rel,
        semantic_relpath=semantic_relpath,
        width=w,
        height=h,
        weather_tag=task.weather_tag,
        time_tag=task.time_tag,
        scene_tag=task.scene_tag,
        source_group_key=task.source_group_key,
    )
    return ConvertTaskResult(index=task.index, row=row)


def _determine_default_workers() -> int:
    cpu = os.cpu_count() or 1
    # Reserve a couple of cores for the OS/IO and keep default stable for reproducible runs.
    return max(1, min(12, cpu - 2 if cpu > 2 else cpu))


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
    p.add_argument(
        "--workers",
        type=int,
        default=_determine_default_workers(),
        help="Number of parallel workers for per-image conversion (default: min(12, cpu-2))",
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


def _process_tasks_serial(tasks: Sequence[ConvertTask]) -> List[ConvertTaskResult]:
    total = len(tasks)
    results: List[ConvertTaskResult] = []
    started = time.monotonic()
    interval = _progress_interval(total=total, target_updates=120, min_interval=500)

    for i, task in enumerate(tasks, start=1):
        results.append(_process_convert_task(task))
        if i % interval == 0 or i == total:
            _log_progress(
                stage="샘플 변환",
                meaning="이미지 재인코딩 + 검출 라벨 + DA/RM 마스크를 생성해 학습 데이터 계약을 맞추는 중",
                done=i,
                total=total,
                started_at=started,
                unit="샘플",
            )
    return results


def _process_tasks_parallel(tasks: Sequence[ConvertTask], *, workers: int) -> List[ConvertTaskResult]:
    total = len(tasks)
    if total == 0:
        return []

    try:
        mp_ctx = mp.get_context("fork")
    except ValueError:
        print(
            "[pv26][로딩][샘플 변환] fork 멀티프로세스를 사용할 수 없어 단일 프로세스로 폴백합니다.",
            flush=True,
        )
        return _process_tasks_serial(tasks)

    started = time.monotonic()
    interval = _progress_interval(total=total, target_updates=120, min_interval=500)
    max_inflight = max(1, int(workers) * 4)
    results: List[ConvertTaskResult] = []

    task_iter = iter(tasks)
    done = 0
    inflight: Dict[Any, ConvertTask] = {}

    try:
        executor = ProcessPoolExecutor(max_workers=int(workers), mp_context=mp_ctx)
    except (OSError, PermissionError) as ex:
        print(
            "[pv26][warn] 멀티프로세스 초기화 실패로 단일 프로세스로 폴백합니다: "
            f"{type(ex).__name__}: {ex}",
            flush=True,
        )
        return _process_tasks_serial(tasks)

    with executor as ex:

        def _submit_next() -> bool:
            try:
                task = next(task_iter)
            except StopIteration:
                return False
            fut = ex.submit(_process_convert_task, task)
            inflight[fut] = task
            return True

        for _ in range(min(max_inflight, total)):
            _submit_next()

        while inflight:
            done_set, _ = wait(tuple(inflight.keys()), return_when=FIRST_COMPLETED)
            for fut in done_set:
                _task = inflight.pop(fut)
                results.append(fut.result())
                done += 1
                if done % interval == 0 or done == total:
                    _log_progress(
                        stage="샘플 변환",
                        meaning="이미지 재인코딩 + 검출 라벨 + DA/RM 마스크를 생성해 학습 데이터 계약을 맞추는 중",
                        done=done,
                        total=total,
                        started_at=started,
                        unit="샘플",
                    )
                _submit_next()

    return results


def _write_checksums_parallel(*, out_root: Path, files: Sequence[Path], workers: int, out_path: Path) -> None:
    total = len(files)
    if total == 0:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("", encoding="utf-8")
        return

    checksum_workers = max(1, min(16, int(workers)))
    max_inflight = max(1, checksum_workers * 8)
    digests: List[str] = [""] * total

    started = time.monotonic()
    interval = _progress_interval(total=total, target_updates=120, min_interval=1000)

    print(
        f"[pv26][로딩][체크섬] SHA256 계산 시작: 파일 {total:,}개, 워커 {checksum_workers}개",
        flush=True,
    )

    next_idx = 0
    done = 0
    inflight: Dict[Any, int] = {}

    with ThreadPoolExecutor(max_workers=checksum_workers) as ex:

        def _submit_next() -> bool:
            nonlocal next_idx
            if next_idx >= total:
                return False
            idx = next_idx
            next_idx += 1
            fut = ex.submit(sha256_file, files[idx])
            inflight[fut] = idx
            return True

        for _ in range(min(max_inflight, total)):
            _submit_next()

        while inflight:
            done_set, _ = wait(tuple(inflight.keys()), return_when=FIRST_COMPLETED)
            for fut in done_set:
                idx = inflight.pop(fut)
                digests[idx] = fut.result()
                done += 1
                if done % interval == 0 or done == total:
                    _log_progress(
                        stage="체크섬",
                        meaning="산출 파일의 무결성(SHA256)을 기록해 배포/학습 재현성을 보장하는 중",
                        done=done,
                        total=total,
                        started_at=started,
                        unit="파일",
                    )
                _submit_next()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for p, digest in zip(files, digests):
            relp = p.relative_to(out_root).as_posix()
            f.write(f"{digest}  {relp}\n")


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

    workers = max(1, int(args.workers))
    cpu = os.cpu_count() or 1
    if workers > cpu:
        print(f"[pv26][warn] workers={workers} > cpu={cpu}; cpu에 맞춰 {cpu}으로 조정합니다.", flush=True)
        workers = cpu

    print("[pv26][로딩][초기화] 변환 설정을 확인하고 출력 디렉터리 구조를 준비합니다.", flush=True)
    print(
        f"[pv26][로딩][초기화] splits={sorted(split_filter)} limit={int(args.limit)} seed={int(args.seed)} "
        f"workers={workers} include_rain={bool(args.include_rain)} include_night={bool(args.include_night)} "
        f"allow_unknown_tags={bool(args.allow_unknown_tags)}",
        flush=True,
    )

    layout = Pv26Layout(out_root=out_root)
    layout.ensure_dirs()

    # Load label records (optional).
    load_started = time.monotonic()
    records: List[Dict[str, Any]] = []
    if labels_path is not None:
        if not labels_path.exists():
            raise SystemExit(f"--labels not found: {labels_path}")
        print("[pv26][로딩][라벨 로드] BDD 라벨 JSON을 메모리로 읽어 빠른 이름 인덱스를 준비합니다.", flush=True)
        records = list(iter_bdd_label_records(labels_path))
    rec_by_name = _index_records_by_image_name(records) if records else {}
    print(
        f"[pv26][로딩][라벨 로드] 완료: 원본 레코드 {len(records):,}개, 인덱스 키 {len(rec_by_name):,}개, "
        f"소요 {_fmt_duration(time.monotonic() - load_started)}",
        flush=True,
    )

    idx_started = time.monotonic()
    print("[pv26][로딩][드라이버블 인덱스] drivable_id PNG를 역참조하기 위한 stem 인덱스를 구성합니다.", flush=True)
    drivable_by_stem = _index_drivable_masks(drivable_root)
    print(
        f"[pv26][로딩][드라이버블 인덱스] 완료: {len(drivable_by_stem):,}개, "
        f"소요 {_fmt_duration(time.monotonic() - idx_started)}",
        flush=True,
    )

    # Discover images.
    discover_started = time.monotonic()
    print("[pv26][로딩][이미지 탐색] 입력 루트 전체를 스캔해 변환 대상 이미지를 정렬합니다.", flush=True)
    image_paths = sorted(
        [p for p in images_root.rglob("*") if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    )
    print(
        f"[pv26][로딩][이미지 탐색] 완료: 후보 이미지 {len(image_paths):,}개, "
        f"소요 {_fmt_duration(time.monotonic() - discover_started)}",
        flush=True,
    )

    # Build task list with deterministic ordering and filtering.
    prep_started = time.monotonic()
    print("[pv26][로딩][작업 계획] 도메인 필터/스플릿 규칙을 적용해 실제 변환 작업 큐를 생성합니다.", flush=True)
    tasks: List[ConvertTask] = []
    skipped = Counter()
    for img_path in image_paths:
        rel = img_path.relative_to(images_root)
        img_name = img_path.name
        rec = rec_by_name.get(img_name) or rec_by_name.get(Path(img_name).stem)

        weather_tag, time_tag, scene_tag = ("unknown", "unknown", "unknown")
        if rec is not None:
            weather_tag, time_tag, scene_tag = bdd_record_tags(rec)
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

        if args.limit and int(args.limit) > 0 and len(tasks) >= int(args.limit):
            break

        tasks.append(
            ConvertTask(
                index=len(tasks),
                img_path=str(img_path),
                img_name=img_name,
                sample_id=sample_id,
                split=split,
                source=source,
                sequence=sequence,
                frame=frame,
                camera_id=camera_id,
                weather_tag=weather_tag,
                time_tag=time_tag,
                scene_tag=scene_tag,
                source_group_key=source_group_key,
            )
        )

    print(
        f"[pv26][로딩][작업 계획] 완료: 변환 대상 {len(tasks):,}개, "
        f"스킵(domain_filter={skipped.get('domain_filter', 0):,}, split_filter={skipped.get('split_filter', 0):,}), "
        f"소요 {_fmt_duration(time.monotonic() - prep_started)}",
        flush=True,
    )

    _set_worker_context(
        rec_by_name=rec_by_name,
        drivable_by_stem=drivable_by_stem,
        out_root=out_root,
        min_box_area_px=int(args.min_box_area_px),
        has_drivable_root=drivable_root is not None,
    )

    # Convert tasks.
    conv_started = time.monotonic()
    print(
        "[pv26][로딩][샘플 변환] 변환 본작업 시작: 멀티프로세스로 이미지/검출/세그멘테이션 아티팩트를 생성합니다.",
        flush=True,
    )

    if workers <= 1:
        results = _process_tasks_serial(tasks)
    else:
        results = _process_tasks_parallel(tasks, workers=workers)

    results.sort(key=lambda x: x.index)
    rows = [r.row for r in results]

    counts = Counter()
    for row in rows:
        counts[f"split:{row.split}"] += 1
        counts["total"] += 1
        counts[f"has_det:{int(row.has_det)}"] += 1
        counts[f"has_da:{int(row.has_da)}"] += 1
        counts[f"has_rm_lane_subclass:{int(row.has_rm_lane_subclass)}"] += 1

    print(
        f"[pv26][로딩][샘플 변환] 완료: {len(rows):,}개 샘플 생성, "
        f"소요 {_fmt_duration(time.monotonic() - conv_started)}",
        flush=True,
    )

    # meta outputs
    meta_started = time.monotonic()
    print("[pv26][로딩][메타 기록] class_map, manifest, source 통계를 기록합니다.", flush=True)
    layout.meta_dir().mkdir(parents=True, exist_ok=True)
    layout.class_map_path().write_text(render_class_map_yaml(classmap_version=CLASSMAP_VERSION_V3), encoding="utf-8")
    write_manifest_csv(layout.manifest_path(), rows)

    with layout.source_stats_path().open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["source", "num_samples"])
        writer.writerow(["bdd100k", counts["total"]])
    print(
        f"[pv26][로딩][메타 기록] 완료: manifest={layout.manifest_path()} "
        f"소요 {_fmt_duration(time.monotonic() - meta_started)}",
        flush=True,
    )

    # checksums.sha256
    scan_started = time.monotonic()
    print("[pv26][로딩][체크섬 준비] 출력 디렉터리 파일 목록을 수집합니다.", flush=True)
    exported_files = list_files_recursive(out_root)
    print(
        f"[pv26][로딩][체크섬 준비] 완료: 대상 파일 {len(exported_files):,}개, "
        f"소요 {_fmt_duration(time.monotonic() - scan_started)}",
        flush=True,
    )
    _write_checksums_parallel(
        out_root=out_root,
        files=exported_files,
        workers=workers,
        out_path=layout.checksums_path(),
    )

    report = {
        "converter": "convert_bdd_type_a.py",
        "converter_version": "0.2.0",
        "spec": "docs/PV26_DATASET_CONVERSION_SPEC.md v1.5",
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
            "workers": int(workers),
            "include_rain": bool(args.include_rain),
            "include_night": bool(args.include_night),
            "allow_unknown_tags": bool(args.allow_unknown_tags),
            "limit": int(args.limit),
            "classmap_version": CLASSMAP_VERSION_V3,
        },
        "counts": dict(counts),
        "skipped": dict(skipped),
        "notes": [
            "Type-A BDD-only slice: lane/non-lane RM masks are rasterized from BDD lane/* poly2d labels.",
            "rm_lane_subclass mono8 mask is exported: "
            "0=background, 1=white_solid, 2=white_dashed, 3=yellow_solid, 4=yellow_dashed, 255=ignore.",
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
