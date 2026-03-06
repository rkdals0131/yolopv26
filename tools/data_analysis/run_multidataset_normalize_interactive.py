#!/usr/bin/env python3
"""
Interactive orchestrator for PV26 Type-A normalization across BDD/ETRI/RLMD/WOD.

Pipeline per selected dataset:
  1) convert_<dataset>_type_a.py
  2) validate_pv26_dataset.py (optional)
  3) pv26_qc_report.py (optional)
  4) render_pv26_debug_masks.py (optional)

Each dataset writes:
  <out_root>/meta/run_manifest_interactive.json
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pv26.io import utc_now_iso, write_json


ALLOWED_SPLITS: Tuple[str, ...] = ("train", "val", "test")
ALLOWED_DEBUG_CHANNELS: Tuple[str, ...] = (
    "da",
    "rm_lane_marker",
    "rm_lane_subclass",
    "rm_road_marker_non_lane",
    "rm_stop_line",
)


@dataclass
class StageResult:
    name: str
    argv: List[str]
    cwd: str
    started_at: str
    completed_at: str
    returncode: int
    stdout_tail: str
    stderr_tail: str


@dataclass
class DatasetRunConfig:
    key: str
    label: str
    source_root: Path
    out_root: Path
    convert_argv: List[str]


@dataclass
class PostConfig:
    run_validate: bool
    run_qc: bool
    run_debug: bool
    workers: int
    debug_split: str
    debug_channels: str
    debug_num_samples: int
    debug_seed: int


def _default_workers() -> int:
    cpu = os.cpu_count() or 1
    return max(1, min(12, cpu - 2 if cpu > 2 else cpu))


def _prompt(msg: str) -> str:
    try:
        return input(msg)
    except EOFError:
        print("[pv26] 오류: 대화형 입력이 예상보다 일찍 종료되었습니다.", file=sys.stderr)
        raise SystemExit(2)


def _ask_yes_no(prompt: str, *, default: Optional[bool]) -> bool:
    if default is True:
        suffix = " [Y/n] "
    elif default is False:
        suffix = " [y/N] "
    else:
        suffix = " [y/n] "

    yes_set = {"y", "yes"}
    no_set = {"n", "no"}

    while True:
        s = _prompt(prompt + suffix).strip().lower()
        if not s:
            if default is None:
                print("Please answer with 'y' or 'n'.", file=sys.stderr)
                continue
            return bool(default)
        if s in yes_set:
            return True
        if s in no_set:
            return False
        print("Invalid input. Please answer with 'y' or 'n'.", file=sys.stderr)


def _ask_int(prompt: str, *, default: int, min_value: Optional[int] = None) -> int:
    while True:
        s = _prompt(f"{prompt} [기본={default}] ").strip()
        if not s:
            v = int(default)
        else:
            try:
                v = int(s)
            except ValueError:
                print("정수를 입력해 주세요.", file=sys.stderr)
                continue
        if min_value is not None and v < min_value:
            print(f"{min_value} 이상 값을 입력해 주세요.", file=sys.stderr)
            continue
        return v


def _ask_path(prompt: str, *, default: Path) -> Path:
    s = _prompt(f"{prompt} [기본={default}] ").strip()
    if not s:
        return default
    return Path(s).expanduser()


def _ask_optional_text(prompt: str, *, default: str = "") -> str:
    shown = default if default else "빈값"
    s = _prompt(f"{prompt} [기본={shown}] ").strip()
    if not s:
        return default
    return s


def _ask_choice(
    prompt: str,
    *,
    options: Dict[str, str],
    default_key: str,
    aliases: Optional[Dict[str, str]] = None,
) -> str:
    while True:
        print(prompt)
        for k, v in options.items():
            marker = "(기본)" if k == default_key else ""
            print(f"  {k}) {v} {marker}".rstrip())
        s = _prompt("선택 번호를 입력해 주세요: ").strip()
        if not s:
            s = default_key
        s = s.lower()
        if aliases and s in aliases:
            s = aliases[s]
        if s in options:
            return s
        print(
            f"잘못된 선택입니다. 가능한 값: {', '.join(options.keys())}"
            + (" 또는 별칭 입력" if aliases else ""),
            file=sys.stderr,
        )


def _ask_splits(prompt: str, *, default_csv: str = "train,val,test") -> str:
    default_norm = ",".join([p.strip() for p in default_csv.split(",") if p.strip()])
    allowed = set(ALLOWED_SPLITS)
    while True:
        s = _prompt(f"{prompt} [기본={default_norm}] ").strip()
        if not s:
            s = default_norm
        parts = [p.strip().lower() for p in s.split(",") if p.strip()]
        if not parts:
            print("최소 1개 split을 입력해 주세요.", file=sys.stderr)
            continue
        bad = [p for p in parts if p not in allowed]
        if bad:
            print(f"지원하지 않는 split: {bad}. 허용값: {list(ALLOWED_SPLITS)}", file=sys.stderr)
            continue
        uniq: List[str] = []
        for p in parts:
            if p not in uniq:
                uniq.append(p)
        return ",".join(uniq)


def _ask_channels(prompt: str, *, default_csv: str = "da,rm_lane_marker") -> str:
    default_norm = ",".join([p.strip() for p in default_csv.split(",") if p.strip()])
    allowed = set(ALLOWED_DEBUG_CHANNELS)
    while True:
        s = _prompt(f"{prompt} [기본={default_norm}] ").strip()
        if not s:
            s = default_norm
        parts = [p.strip() for p in s.split(",") if p.strip()]
        if not parts:
            print("최소 1개 채널을 입력해 주세요.", file=sys.stderr)
            continue
        bad = [p for p in parts if p not in allowed]
        if bad:
            print(
                f"지원하지 않는 채널: {bad}. 허용값: {list(ALLOWED_DEBUG_CHANNELS)}",
                file=sys.stderr,
            )
            continue
        uniq: List[str] = []
        for p in parts:
            if p not in uniq:
                uniq.append(p)
        return ",".join(uniq)


def _ask_debug_split(prompt: str, *, default: str = "val") -> str:
    default = default.strip().lower()
    if default not in set(ALLOWED_SPLITS):
        default = "val"
    while True:
        s = _prompt(f"{prompt} [기본={default}] ").strip().lower()
        if not s:
            s = default
        if s in set(ALLOWED_SPLITS):
            return s
        print(f"지원하지 않는 split입니다. 허용값: {list(ALLOWED_SPLITS)}", file=sys.stderr)


def _ask_dataset_selection() -> List[str]:
    mapping = {
        "1": "bdd",
        "2": "etri",
        "3": "rlmd",
        "4": "wod",
        "5": "core_all",
        "6": "all",
        "bdd": "bdd",
        "etri": "etri",
        "rlmd": "rlmd",
        "wod": "wod",
        "waymo": "wod",
        "all": "core_all",
        "전체": "all",
        "모두": "all",
    }

    print("\n[1단계] 실행할 데이터셋을 선택하세요.")
    print("  1) BDD100K")
    print("  2) ETRI (Mono+Multi polygon JSON)")
    print("  3) RLMD (RLMD_1080p + RLMD-AC labeled)")
    print("  4) WOD (Waymo v2 parquet)")
    print("  5) 전체 실행(권장): ETRI + RLMD + WOD")
    print("  6) 전체 실행(확장): BDD + ETRI + RLMD + WOD")

    while True:
        s = _prompt("입력 (예: 2,4 또는 5) [필수]: ").strip().lower()
        if not s:
            print("빈 입력은 허용되지 않습니다. 실행할 데이터셋 번호를 명시해 주세요.", file=sys.stderr)
            continue
        parts = [p.strip() for p in s.split(",") if p.strip()]
        if not parts:
            print("최소 1개를 선택해 주세요.", file=sys.stderr)
            continue

        out: List[str] = []
        bad: List[str] = []
        for p in parts:
            v = mapping.get(p)
            if v is None:
                bad.append(p)
                continue
            if v == "core_all":
                return ["etri", "rlmd", "wod"]
            if v == "all":
                return ["bdd", "etri", "rlmd", "wod"]
            if v not in out:
                out.append(v)
        if bad:
            print(f"잘못된 입력: {bad}", file=sys.stderr)
            continue
        if out:
            return out
        print("선택 결과가 비어 있습니다. 다시 입력해 주세요.", file=sys.stderr)


def _tail(s: str, *, max_chars: int = 8000) -> str:
    s = s or ""
    if len(s) <= max_chars:
        return s
    return s[-max_chars:]


def _stream_pipe(
    *,
    src,
    dst,
    tail_chunks: Deque[str],
    tail_size_ref: List[int],
    max_tail_chars: int,
) -> None:
    for line in iter(src.readline, ""):
        dst.write(line)
        dst.flush()
        if not line:
            continue
        tail_chunks.append(line)
        tail_size_ref[0] += len(line)
        while tail_size_ref[0] > max_tail_chars and tail_chunks:
            removed = tail_chunks.popleft()
            tail_size_ref[0] -= len(removed)


def _run_stage(name: str, argv: Sequence[str], *, cwd: Path) -> StageResult:
    started = utc_now_iso()
    t0 = time.monotonic()
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")

    p = subprocess.Popen(
        list(argv),
        cwd=str(cwd),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=1,
        env=env,
    )
    if p.stdout is None or p.stderr is None:
        raise RuntimeError(f"stage failed to open pipes: {name}")

    out_chunks: Deque[str] = deque()
    err_chunks: Deque[str] = deque()
    out_size_ref = [0]
    err_size_ref = [0]

    tout = threading.Thread(
        target=_stream_pipe,
        kwargs={
            "src": p.stdout,
            "dst": sys.stdout,
            "tail_chunks": out_chunks,
            "tail_size_ref": out_size_ref,
            "max_tail_chars": 8000,
        },
        daemon=True,
    )
    terr = threading.Thread(
        target=_stream_pipe,
        kwargs={
            "src": p.stderr,
            "dst": sys.stderr,
            "tail_chunks": err_chunks,
            "tail_size_ref": err_size_ref,
            "max_tail_chars": 8000,
        },
        daemon=True,
    )

    tout.start()
    terr.start()
    rc = p.wait()
    tout.join()
    terr.join()

    completed = utc_now_iso()
    elapsed = time.monotonic() - t0
    print(f"[pv26][단계완료] {name} | rc={rc} | 소요={elapsed:.1f}초", flush=True)

    return StageResult(
        name=name,
        argv=list(argv),
        cwd=str(cwd),
        started_at=started,
        completed_at=completed,
        returncode=int(rc),
        stdout_tail=_tail("".join(out_chunks)),
        stderr_tail=_tail("".join(err_chunks)),
    )


def _ask_existing_out_policy(out_root: Path) -> str:
    print(f"\n출력 폴더가 이미 존재합니다: {out_root}")
    options = {
        "1": "삭제 후 다시 생성하고 진행",
        "2": "이 데이터셋은 건너뛰고 다음으로 진행",
        "3": "전체 작업 중단",
    }
    choice = _ask_choice("처리 방식을 선택하세요.", options=options, default_key="2")
    if choice == "1":
        return "overwrite"
    if choice == "2":
        return "skip"
    return "abort"


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Interactive PV26 multi-dataset normalization runner.")
    p.add_argument("--bdd-root", type=Path, default=Path("datasets/BDD100K"), help="BDD100K root")
    p.add_argument("--etri-root", type=Path, default=Path("datasets/ETRI"), help="ETRI root")
    p.add_argument("--rlmd-root", type=Path, default=Path("datasets/RLMD"), help="RLMD root")
    p.add_argument(
        "--wod-training-root",
        type=Path,
        default=Path("datasets/WaymoOpenDataset/wod_pv2_minimal_1ctx/training"),
        help="Waymo training root (camera_image/camera_box/camera_segmentation)",
    )
    return p.parse_args(argv)


def _build_dataset_configs(args: argparse.Namespace, *, run_id: str) -> List[DatasetRunConfig]:
    selected = _ask_dataset_selection()
    configs: List[DatasetRunConfig] = []

    print("\n[2단계] 데이터셋별 변환 옵션을 확인합니다.")

    if "bdd" in selected:
        print("\n[BDD100K 설정]")
        print("  - 기대 구조:")
        print("    * <bdd_root>/bdd100k_images_100k/100k")
        print("    * <bdd_root>/bdd100k_labels/100k")
        print("    * <bdd_root>/bdd100k_drivable_maps/labels")
        bdd_root = _ask_path("BDD100K 원본 루트 경로", default=args.bdd_root)
        bdd_out = _ask_path("BDD100K 출력 루트 경로", default=Path("datasets/pv26_v1_bdd_full"))
        bdd_splits = _ask_splits("BDD100K split 필터(train,val,test)", default_csv="train,val,test")
        bdd_limit = _ask_int("BDD100K 처리 수 제한(0=전체)", default=0, min_value=0)
        bdd_seed = _ask_int("BDD100K split seed", default=0, min_value=0)
        bdd_workers = _ask_int("BDD100K 변환 workers", default=_default_workers(), min_value=1)
        bdd_min_box = _ask_int("BDD100K min_box_area_px", default=0, min_value=0)
        bdd_include_rain = _ask_yes_no("비(rain) 조건 샘플을 포함할까요?", default=False)
        bdd_include_night = _ask_yes_no("야간(night) 조건 샘플을 포함할까요?", default=False)
        bdd_allow_unknown = _ask_yes_no("알 수 없는 태그(weather/time)를 허용할까요?", default=False)

        images_root = bdd_root / "bdd100k_images_100k" / "100k"
        labels_root = bdd_root / "bdd100k_labels" / "100k"
        drivable_root = bdd_root / "bdd100k_drivable_maps" / "labels"
        convert_argv = [
            sys.executable,
            "-u",
            str(REPO_ROOT / "tools" / "data_analysis" / "bdd" / "convert_bdd_type_a.py"),
            "--images-root",
            str(images_root),
            "--labels",
            str(labels_root),
            "--drivable-root",
            str(drivable_root),
            "--out-root",
            str(bdd_out),
            "--seed",
            str(bdd_seed),
            "--workers",
            str(bdd_workers),
            "--min-box-area-px",
            str(bdd_min_box),
            "--limit",
            str(bdd_limit),
            "--splits",
            str(bdd_splits),
            "--run-id",
            str(run_id),
        ]
        if bdd_include_rain:
            convert_argv.append("--include-rain")
        if bdd_include_night:
            convert_argv.append("--include-night")
        if bdd_allow_unknown:
            convert_argv.append("--allow-unknown-tags")

        configs.append(
            DatasetRunConfig(
                key="bdd",
                label="BDD100K",
                source_root=bdd_root,
                out_root=bdd_out,
                convert_argv=convert_argv,
            )
        )

    if "etri" in selected:
        print("\n[ETRI 설정]")
        print("  - 기대 구조(둘 중 하나 이상 존재):")
        print("    * <etri_root>/MonoCameraSemanticSegmentation/JPEGImages_mosaic")
        print("      + <etri_root>/MonoCameraSemanticSegmentation/labels")
        print("    * <etri_root>/Multi Camera Semantic Segmentation/(leftImg,rightImg,labels)")
        etri_root = _ask_path("ETRI 원본 루트 경로", default=args.etri_root)
        etri_out = _ask_path("ETRI 출력 루트 경로", default=Path("datasets/pv26_v1_etri"))
        etri_splits = _ask_splits("ETRI split 필터(train,val,test)", default_csv="train,val,test")
        etri_limit = _ask_int("ETRI 처리 수 제한(0=전체)", default=0, min_value=0)
        etri_seed = _ask_int("ETRI split seed", default=0, min_value=0)
        etri_workers = _ask_int("ETRI 변환 workers", default=_default_workers(), min_value=1)

        convert_argv = [
            sys.executable,
            "-u",
            str(REPO_ROOT / "tools" / "data_analysis" / "etri" / "convert_etri_type_a.py"),
            "--etri-root",
            str(etri_root),
            "--out-root",
            str(etri_out),
            "--splits",
            str(etri_splits),
            "--limit",
            str(etri_limit),
            "--seed",
            str(etri_seed),
            "--workers",
            str(etri_workers),
            "--run-id",
            str(run_id),
        ]
        configs.append(
            DatasetRunConfig(
                key="etri",
                label="ETRI",
                source_root=etri_root,
                out_root=etri_out,
                convert_argv=convert_argv,
            )
        )

    if "rlmd" in selected:
        print("\n[RLMD 설정]")
        print("  - 기대 구조:")
        print("    * <rlmd_root>/RLMD_1080p/rlmd.csv")
        print("    * <rlmd_root>/RLMD_1080p/images/{train,val}")
        print("    * <rlmd_root>/RLMD_1080p/labels/{train,val}")
        print("    * (선택) <rlmd_root>/RLMD-AC/.../labels")
        rlmd_root = _ask_path("RLMD 원본 루트 경로", default=args.rlmd_root)
        rlmd_out = _ask_path("RLMD 출력 루트 경로", default=Path("datasets/pv26_v1_rlmd"))
        include_ac = _ask_yes_no("RLMD-AC 라벨 가능한 split도 포함할까요?", default=True)
        rlmd_limit = _ask_int("RLMD 처리 수 제한(0=전체)", default=0, min_value=0)
        rlmd_workers = _ask_int("RLMD 변환 workers", default=_default_workers(), min_value=1)

        convert_argv = [
            sys.executable,
            "-u",
            str(REPO_ROOT / "tools" / "data_analysis" / "rlmd" / "convert_rlmd_type_a.py"),
            "--rlmd-root",
            str(rlmd_root),
            "--out-root",
            str(rlmd_out),
            "--limit",
            str(rlmd_limit),
            "--workers",
            str(rlmd_workers),
            "--run-id",
            str(run_id),
            "--include-ac" if include_ac else "--no-include-ac",
        ]
        configs.append(
            DatasetRunConfig(
                key="rlmd",
                label="RLMD",
                source_root=rlmd_root,
                out_root=rlmd_out,
                convert_argv=convert_argv,
            )
        )

    if "wod" in selected:
        print("\n[WOD 설정]")
        print("  - 기대 구조:")
        print("    * <training_root>/camera_image/*.parquet")
        print("    * <training_root>/camera_box/*.parquet")
        print("    * <training_root>/camera_segmentation/*.parquet")
        wod_root = _ask_path("WOD training 루트 경로", default=args.wod_training_root)
        wod_out = _ask_path("WOD 출력 루트 경로", default=Path("datasets/pv26_v1_waymo_minimal_1ctx"))
        context_name = _ask_optional_text("WOD context_name(비우면 parquet 1개일 때 자동)", default="")
        split_choice = _ask_choice(
            "WOD split 정책을 선택하세요. [기본=1(all_train)]",
            options={
                "1": "all_train (권장: minimal context에서 누수/불균형 회피)",
                "2": "stable_by_context (context 기준 train/val/test 고정 분할)",
            },
            default_key="1",
            aliases={
                "all_train": "1",
                "all-train": "1",
                "stable_by_context": "2",
                "stable-by-context": "2",
                "stable": "2",
                "context": "2",
            },
        )
        split_policy = "all_train" if split_choice == "1" else "stable_by_context"
        split_default = "train" if split_policy == "all_train" else "train,val,test"
        wod_splits = _ask_splits("WOD split 필터(train,val,test)", default_csv=split_default)
        wod_seed = _ask_int("WOD split seed", default=0, min_value=0)
        wod_max_rows = _ask_int("WOD 최대 row 수(0=전체)", default=0, min_value=0)
        wod_require_seg = _ask_yes_no("세그멘테이션 있는 row만 내보낼까요?", default=True)

        convert_argv = [
            sys.executable,
            "-u",
            str(REPO_ROOT / "tools" / "data_analysis" / "wod" / "convert_wod_type_a.py"),
            "--training-root",
            str(wod_root),
            "--out-root",
            str(wod_out),
            "--split-policy",
            str(split_policy),
            "--splits",
            str(wod_splits),
            "--seed",
            str(wod_seed),
            "--max-rows",
            str(wod_max_rows),
            "--run-id",
            str(run_id),
            "--require-seg" if wod_require_seg else "--no-require-seg",
        ]
        if context_name:
            convert_argv.extend(["--context-name", context_name])

        configs.append(
            DatasetRunConfig(
                key="wod",
                label="WOD",
                source_root=wod_root,
                out_root=wod_out,
                convert_argv=convert_argv,
            )
        )

    return configs


def _validate_source_layout(ds: DatasetRunConfig) -> Tuple[bool, str]:
    if not ds.source_root.exists():
        return False, f"원본 루트가 존재하지 않습니다: {ds.source_root}"

    if ds.key == "bdd":
        images_root = ds.source_root / "bdd100k_images_100k" / "100k"
        labels_root = ds.source_root / "bdd100k_labels" / "100k"
        drivable_root = ds.source_root / "bdd100k_drivable_maps" / "labels"
        missing = [str(p) for p in [images_root, labels_root, drivable_root] if not p.exists()]
        if missing:
            return False, "BDD100K 필수 하위 경로 누락: " + ", ".join(missing)
        return True, "BDD100K 표준 하위 경로 확인 완료"

    if ds.key == "etri":
        mono_images = ds.source_root / "MonoCameraSemanticSegmentation" / "JPEGImages_mosaic"
        mono_labels = ds.source_root / "MonoCameraSemanticSegmentation" / "labels"
        multi_left = ds.source_root / "Multi Camera Semantic Segmentation" / "leftImg"
        multi_right = ds.source_root / "Multi Camera Semantic Segmentation" / "rightImg"
        multi_labels = ds.source_root / "Multi Camera Semantic Segmentation" / "labels"
        has_mono = mono_images.exists() and mono_labels.exists()
        has_multi = multi_left.exists() and multi_right.exists() and multi_labels.exists()
        if not has_mono and not has_multi:
            return (
                False,
                "ETRI 구조 미일치: Mono(이미지+labels) 또는 Multi(leftImg/rightImg/labels) 중 하나가 필요합니다.",
            )
        return True, "ETRI Mono/Multi 구조 확인 완료"

    if ds.key == "rlmd":
        palette_csv = ds.source_root / "RLMD_1080p" / "rlmd.csv"
        if not palette_csv.exists():
            return False, f"RLMD 팔레트 파일 누락: {palette_csv}"
        return True, "RLMD_1080p 팔레트/루트 구조 확인 완료"

    if ds.key == "wod":
        camera_image = ds.source_root / "camera_image"
        camera_box = ds.source_root / "camera_box"
        camera_seg = ds.source_root / "camera_segmentation"
        missing = [str(p) for p in [camera_image, camera_box, camera_seg] if not p.exists()]
        if missing:
            return False, "WOD 필수 component 디렉터리 누락: " + ", ".join(missing)
        return True, "WOD training component 구조 확인 완료"

    return True, "구조 검증 로직이 정의되지 않은 데이터셋입니다."


def _ask_post_config() -> PostConfig:
    print("\n[3단계] 변환 이후 후처리(검증/QC/디버그)를 설정합니다.")
    workers = _ask_int("검증/QC/디버그 공통 workers", default=_default_workers(), min_value=1)
    run_validate = _ask_yes_no("변환 후 validate를 실행할까요?", default=True)
    run_qc = _ask_yes_no("변환 후 QC 리포트를 생성할까요?", default=True)
    run_debug = _ask_yes_no("변환 후 디버그 시각화를 생성할까요?", default=False)

    debug_split = "val"
    debug_channels = "da,rm_lane_marker"
    debug_num_samples = 10
    debug_seed = 42
    if run_debug:
        debug_split = _ask_debug_split("디버그 시각화 split", default="val")
        debug_channels = _ask_channels(
            "디버그 채널(쉼표 구분)",
            default_csv="da,rm_lane_marker",
        )
        debug_num_samples = _ask_int("채널별 샘플 수", default=10, min_value=1)
        debug_seed = _ask_int("디버그 샘플링 seed", default=42, min_value=0)

    return PostConfig(
        run_validate=run_validate,
        run_qc=run_qc,
        run_debug=run_debug,
        workers=workers,
        debug_split=debug_split,
        debug_channels=debug_channels,
        debug_num_samples=debug_num_samples,
        debug_seed=debug_seed,
    )


def _build_post_stage_args(dataset_out_root: Path, cfg: PostConfig) -> List[Tuple[str, List[str], str]]:
    stages: List[Tuple[str, List[str], str]] = []

    if cfg.run_validate:
        validate_argv = [
            sys.executable,
            "-u",
            str(REPO_ROOT / "tools" / "data_analysis" / "bdd" / "validate_pv26_dataset.py"),
            "--out-root",
            str(dataset_out_root),
            "--workers",
            str(cfg.workers),
        ]
        stages.append(
            (
                "validate_pv26_dataset",
                validate_argv,
                "생성물의 파일/해상도/값 도메인/부분라벨 계약을 검사합니다.",
            )
        )

    if cfg.run_qc:
        qc_out = dataset_out_root / "meta" / "qc_report.json"
        qc_argv = [
            sys.executable,
            "-u",
            str(REPO_ROOT / "tools" / "data_analysis" / "bdd" / "pv26_qc_report.py"),
            "--dataset-root",
            str(dataset_out_root),
            "--workers",
            str(cfg.workers),
            "--out-json",
            str(qc_out),
        ]
        stages.append(
            (
                "pv26_qc_report",
                qc_argv,
                "split/라벨 가용성/마스크 non-empty 통계를 산출합니다.",
            )
        )

    if cfg.run_debug:
        debug_out = dataset_out_root / "meta" / "debug_vis"
        dbg_argv = [
            sys.executable,
            "-u",
            str(REPO_ROOT / "tools" / "debug" / "render_pv26_debug_masks.py"),
            "--dataset-root",
            str(dataset_out_root),
            "--split",
            str(cfg.debug_split),
            "--channels",
            str(cfg.debug_channels),
            "--num-samples",
            str(cfg.debug_num_samples),
            "--out-root",
            str(debug_out),
            "--workers",
            str(cfg.workers),
            "--seed",
            str(cfg.debug_seed),
        ]
        stages.append(
            (
                "render_pv26_debug_masks",
                dbg_argv,
                "마스크 컬러맵/오버레이 시각화를 생성해 육안 점검합니다.",
            )
        )

    return stages


def _print_summary(configs: List[DatasetRunConfig], post_cfg: PostConfig, run_id: str) -> None:
    print("\n[pv26] 실행 계획 요약")
    print(f"  - run_id: {run_id}")
    print(f"  - 대상 데이터셋 수: {len(configs)}")
    for i, ds in enumerate(configs, start=1):
        print(f"    {i}. {ds.label}")
        print(f"       - source_root: {ds.source_root}")
        print(f"       - out_root:    {ds.out_root}")
    print(f"  - validate: {post_cfg.run_validate}")
    print(f"  - qc:       {post_cfg.run_qc}")
    print(f"  - debug:    {post_cfg.run_debug}")
    if post_cfg.run_debug:
        print(f"    - debug_split:       {post_cfg.debug_split}")
        print(f"    - debug_channels:    {post_cfg.debug_channels}")
        print(f"    - debug_num_samples: {post_cfg.debug_num_samples}")


def _run_single_dataset(ds: DatasetRunConfig, post_cfg: PostConfig, run_id: str) -> int:
    ok, layout_msg = _validate_source_layout(ds)
    if not ok:
        print(f"[pv26][{ds.key}] 오류: {layout_msg}", file=sys.stderr)
        return 2
    print(f"[pv26][{ds.key}] 입력 구조 확인 완료: {layout_msg}", flush=True)

    if ds.out_root.exists():
        policy = _ask_existing_out_policy(ds.out_root)
        if policy == "abort":
            print("[pv26] 사용자 요청으로 전체 작업을 중단합니다.", file=sys.stderr)
            return 130
        if policy == "skip":
            print(f"[pv26][{ds.key}] 사용자 선택으로 건너뜁니다.", flush=True)
            return 0
        shutil.rmtree(ds.out_root)
        print(f"[pv26][{ds.key}] 기존 출력 폴더를 삭제했습니다: {ds.out_root}", flush=True)

    meta_dir = ds.out_root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = meta_dir / "run_manifest_interactive.json"

    manifest: Dict[str, Any] = {
        "started_at": utc_now_iso(),
        "completed_at": "",
        "run_id": run_id,
        "dataset": ds.key,
        "dataset_label": ds.label,
        "source_root": str(ds.source_root),
        "output_root": str(ds.out_root),
        "convert_command": ds.convert_argv,
        "post_config": {
            "run_validate": bool(post_cfg.run_validate),
            "run_qc": bool(post_cfg.run_qc),
            "run_debug": bool(post_cfg.run_debug),
            "workers": int(post_cfg.workers),
            "debug_split": post_cfg.debug_split,
            "debug_channels": post_cfg.debug_channels,
            "debug_num_samples": int(post_cfg.debug_num_samples),
            "debug_seed": int(post_cfg.debug_seed),
        },
        "commands_invoked": [],
        "status": "running",
        "failed_stage": "",
        "failed_returncode": None,
    }
    write_json(manifest_path, manifest)

    stage_results: List[StageResult] = []
    failed_stage = ""
    failed_rc: Optional[int] = None

    try:
        print(f"\n[pv26][{ds.key}] [1단계/변환] {ds.label} 변환 시작", flush=True)
        print("  - 의미: 원본 데이터를 PV26 Type-A 학습 구조로 표준화합니다.", flush=True)
        r = _run_stage(f"{ds.key}:convert", ds.convert_argv, cwd=REPO_ROOT)
        stage_results.append(r)
        if r.returncode != 0:
            failed_stage = r.name
            failed_rc = r.returncode
            raise RuntimeError(f"stage failed: {r.name} rc={r.returncode}")

        post_stages = _build_post_stage_args(ds.out_root, post_cfg)
        for idx, (stage_name, argv, meaning) in enumerate(post_stages, start=2):
            print(f"\n[pv26][{ds.key}] [{idx}단계/후처리] {stage_name}", flush=True)
            print(f"  - 의미: {meaning}", flush=True)
            r = _run_stage(f"{ds.key}:{stage_name}", argv, cwd=REPO_ROOT)
            stage_results.append(r)
            if r.returncode != 0:
                failed_stage = r.name
                failed_rc = r.returncode
                raise RuntimeError(f"stage failed: {r.name} rc={r.returncode}")

    except KeyboardInterrupt:
        failed_stage = failed_stage or "keyboard_interrupt"
        failed_rc = failed_rc if failed_rc is not None else 130
        print(f"\n[pv26][{ds.key}] 중단됨: KeyboardInterrupt", file=sys.stderr)
        manifest["status"] = "failed"
    except Exception as ex:  # noqa: BLE001 - CLI runner
        if not failed_stage:
            failed_stage = "unknown"
        if failed_rc is None:
            failed_rc = 2
        print(f"[pv26][{ds.key}] 오류: {type(ex).__name__}: {ex}", file=sys.stderr)
        manifest["status"] = "failed"
    else:
        manifest["status"] = "success"
    finally:
        manifest["completed_at"] = utc_now_iso()
        manifest["failed_stage"] = failed_stage
        manifest["failed_returncode"] = failed_rc
        manifest["commands_invoked"] = [
            {
                "name": r.name,
                "argv": r.argv,
                "cwd": r.cwd,
                "started_at": r.started_at,
                "completed_at": r.completed_at,
                "returncode": r.returncode,
                "stdout_tail": r.stdout_tail,
                "stderr_tail": r.stderr_tail,
            }
            for r in stage_results
        ]
        write_json(manifest_path, manifest)
        sys.stdout.write(
            json.dumps(
                {
                    "dataset": ds.key,
                    "status": manifest["status"],
                    "out_root": str(ds.out_root),
                    "failed_stage": failed_stage,
                    "failed_returncode": failed_rc,
                    "manifest": str(manifest_path),
                }
            )
            + "\n"
        )

    if manifest["status"] != "success":
        return 2
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)

    run_id_default = f"multi-{time.strftime('%Y%m%d-%H%M%S')}"
    print("[pv26] 멀티 데이터셋 인터랙티브 변환기를 시작합니다.")
    run_id = _ask_optional_text("run_id(리포트 식별용)", default=run_id_default)

    configs = _build_dataset_configs(args, run_id=run_id)
    if not configs:
        print("[pv26] 실행할 데이터셋이 없습니다. 종료합니다.")
        return 0

    post_cfg = _ask_post_config()
    _print_summary(configs, post_cfg, run_id)

    proceed = _ask_yes_no("위 계획대로 실행할까요?", default=True)
    if not proceed:
        print("[pv26] 사용자 요청으로 실행을 취소했습니다.", file=sys.stderr)
        return 0

    start_all = time.monotonic()
    results: List[Tuple[str, int]] = []

    for ds in configs:
        rc = _run_single_dataset(ds, post_cfg, run_id)
        results.append((ds.key, rc))
        if rc == 130:
            print("[pv26] 사용자 요청 중단 신호를 받아 전체 작업을 종료합니다.", file=sys.stderr)
            return 130
        if rc != 0:
            print(f"[pv26] 실패: dataset={ds.key} rc={rc}", file=sys.stderr)
            return rc

    elapsed = time.monotonic() - start_all
    print("\n[pv26] 전체 작업 완료")
    print(f"  - 소요 시간: {elapsed:.1f}초")
    for key, rc in results:
        print(f"  - {key}: rc={rc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
