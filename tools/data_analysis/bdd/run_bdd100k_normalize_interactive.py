#!/usr/bin/env python3
"""
Interactive BDD100K -> PV26 normalization orchestrator.

This script is intentionally BDD-only and hardcodes:
  - output root: datasets/pv26_v1_bdd_full
  - input subpaths under --bdd-root:
      - bdd100k_images_100k/100k
      - bdd100k_labels/100k
      - bdd100k_drivable_maps/labels

It runs the pipeline in order:
  1) tools/data_analysis/bdd/convert_bdd_pv26.py
  2) tools/data_analysis/bdd/validate_pv26_dataset.py (optional)
  3) tools/data_analysis/bdd/pv26_qc_report.py (always; writes meta/qc_report.json)
  4) tools/debug/render_pv26_debug_masks.py (optional)

And writes a run manifest to:
  <out_root>/meta/run_manifest.json
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

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pv26.io import utc_now_iso, write_json


OUT_ROOT = Path("datasets/pv26_v1_bdd_full")
ALLOWED_SPLITS: Tuple[str, ...] = ("train", "val", "test")
ALLOWED_DEBUG_CHANNELS: Tuple[str, ...] = (
    "da",
    "rm_lane_marker",
    "rm_lane_subclass",
    "rm_road_marker_non_lane",
    "rm_stop_line",
)


def _prompt(msg: str) -> str:
    try:
        return input(msg)
    except EOFError:
        # Common when running with piped answers and not providing enough lines.
        print("[pv26] error: unexpected end-of-input while reading interactive answers.", file=sys.stderr)
        raise SystemExit(2)


def _ask_yes_no(prompt: str, *, default: Optional[bool]) -> bool:
    if default is True:
        suffix = " [Y/n] "
    elif default is False:
        suffix = " [y/N] "
    else:
        suffix = " [y/n] "

    while True:
        s = _prompt(prompt + suffix).strip().lower()
        if not s:
            if default is None:
                continue
            return bool(default)
        if s in {"y", "yes"}:
            return True
        if s in {"n", "no"}:
            return False
        print("Please answer y/n.", file=sys.stderr)


def _ask_int(prompt: str, *, default: int, min_value: Optional[int] = None) -> int:
    while True:
        s = _prompt(f"{prompt} [{default}] ").strip()
        if not s:
            v = int(default)
        else:
            try:
                v = int(s)
            except ValueError:
                print("Please enter an integer.", file=sys.stderr)
                continue
        if min_value is not None and v < min_value:
            print(f"Please enter >= {min_value}.", file=sys.stderr)
            continue
        return v


def _ask_splits(prompt: str, *, default_csv: str = "train,val,test") -> str:
    default_norm = ",".join([p.strip() for p in default_csv.split(",") if p.strip()])
    while True:
        s = _prompt(f"{prompt} [{default_norm}] ").strip()
        if not s:
            s = default_norm
        parts = [p.strip().lower() for p in s.split(",") if p.strip()]
        if not parts:
            print("Please enter at least one split.", file=sys.stderr)
            continue
        bad = [p for p in parts if p not in set(ALLOWED_SPLITS)]
        if bad:
            print(f"Invalid split(s): {bad}. Allowed: {list(ALLOWED_SPLITS)}", file=sys.stderr)
            continue
        # Keep user order, but remove duplicates.
        uniq: List[str] = []
        for p in parts:
            if p not in uniq:
                uniq.append(p)
        return ",".join(uniq)


def _ask_channels(prompt: str, *, default_csv: str = "da,rm_lane_marker") -> str:
    default_norm = ",".join([p.strip() for p in default_csv.split(",") if p.strip()])
    while True:
        s = _prompt(f"{prompt} [{default_norm}] ").strip()
        if not s:
            s = default_norm
        parts = [p.strip() for p in s.split(",") if p.strip()]
        if not parts:
            print("Please enter at least one channel.", file=sys.stderr)
            continue
        bad = [p for p in parts if p not in set(ALLOWED_DEBUG_CHANNELS)]
        if bad:
            print(f"Invalid channel(s): {bad}. Allowed: {list(ALLOWED_DEBUG_CHANNELS)}", file=sys.stderr)
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
        s = _prompt(f"{prompt} [{default}] ").strip().lower()
        if not s:
            s = default
        if s in set(ALLOWED_SPLITS):
            return s
        print(f"Invalid split: {s}. Allowed: {list(ALLOWED_SPLITS)}", file=sys.stderr)


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
    print(
        f"[pv26][로딩][{name}] 완료: rc={rc} 소요={elapsed:.1f}초",
        flush=True,
    )

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


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Interactive BDD100K -> PV26 dataset normalization.")
    p.add_argument("--bdd-root", type=Path, required=True, help="BDD100K root directory.")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_argparser().parse_args(argv)
    bdd_root: Path = args.bdd_root

    images_root = bdd_root / "bdd100k_images_100k" / "100k"
    labels_root = bdd_root / "bdd100k_labels" / "100k"
    drivable_root = bdd_root / "bdd100k_drivable_maps" / "labels"
    out_root = REPO_ROOT / OUT_ROOT

    if not bdd_root.exists():
        print(f"[pv26] error: --bdd-root not found: {bdd_root}", file=sys.stderr)
        return 2

    missing: List[str] = []
    if not images_root.exists():
        missing.append(str(images_root))
    if not labels_root.exists():
        missing.append(str(labels_root))
    if not drivable_root.exists():
        missing.append(str(drivable_root))
    if missing:
        print("[pv26] error: missing required BDD inputs:", file=sys.stderr)
        for m in missing:
            print(f"  - {m}", file=sys.stderr)
        return 2

    print("[pv26] resolved inputs:")
    print(f"  - bdd_root:      {bdd_root}")
    print(f"  - images_root:   {images_root}")
    print(f"  - labels_root:   {labels_root}")
    print(f"  - drivable_root: {drivable_root}")
    print(f"[pv26] output root (fixed): {out_root}")

    if out_root.exists():
        overwrite = _ask_yes_no(f"Output root exists: {out_root}\nOverwrite (DELETE) it?", default=False)
        if not overwrite:
            print("[pv26] abort: output root exists and overwrite declined.", file=sys.stderr)
            return 0
        shutil.rmtree(out_root)
        print(f"[pv26] removed: {out_root}")

    # Interactive options (all run options are asked here).
    splits = _ask_splits("Splits to process (comma-separated)", default_csv="train,val,test")
    limit = _ask_int("Limit (0=all)", default=0, min_value=0)
    seed = _ask_int("Seed", default=0, min_value=0)
    workers = _ask_int("convert workers (권장 12, 고성능 16)", default=12, min_value=1)
    min_box_area_px = _ask_int("min_box_area_px", default=0, min_value=0)
    include_rain = _ask_yes_no("Include rain?", default=False)
    include_night = _ask_yes_no("Include night?", default=False)
    allow_unknown_tags = _ask_yes_no("Allow unknown weather/time tags?", default=False)
    run_validate = _ask_yes_no("Run full validate stage?", default=True)
    run_debug = _ask_yes_no("Run debug visualization stage?", default=False)

    debug_channels = ""
    debug_num_samples = 0
    debug_split = ""
    if run_debug:
        debug_channels = _ask_channels("Debug channels (comma-separated)", default_csv="da,rm_lane_marker")
        debug_num_samples = _ask_int("Debug num_samples", default=10, min_value=1)
        debug_split = _ask_debug_split("Debug split", default="val")

    print("\n[pv26] run configuration:")
    print(f"  - out_root:            {out_root}")
    print(f"  - splits:              {splits}")
    print(f"  - limit:               {limit}")
    print(f"  - seed:                {seed}")
    print(f"  - workers:             {workers}")
    print(f"  - min_box_area_px:     {min_box_area_px}")
    print(f"  - include_rain:        {include_rain}")
    print(f"  - include_night:       {include_night}")
    print(f"  - allow_unknown_tags:  {allow_unknown_tags}")
    print(f"  - run_validate:        {run_validate}")
    print(f"  - run_debug:           {run_debug}")
    if run_debug:
        print(f"  - debug_channels:      {debug_channels}")
        print(f"  - debug_num_samples:   {debug_num_samples}")
        print(f"  - debug_split:         {debug_split}")

    proceed = _ask_yes_no("Proceed to run conversion pipeline now?", default=True)
    if not proceed:
        print("[pv26] abort: user declined to proceed.", file=sys.stderr)
        return 0

    meta_dir = out_root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = meta_dir / "run_manifest.json"

    manifest: Dict[str, Any] = {
        "started_at": utc_now_iso(),
        "completed_at": "",
        "bdd_root": str(bdd_root),
        "resolved_input_paths": {
            "images_root": str(images_root),
            "labels_root": str(labels_root),
            "drivable_root": str(drivable_root),
        },
        "output_root": str(out_root),
        "interactive_answers": {
            "splits": splits,
            "limit": int(limit),
            "seed": int(seed),
            "workers": int(workers),
            "min_box_area_px": int(min_box_area_px),
            "include_rain": bool(include_rain),
            "include_night": bool(include_night),
            "allow_unknown_tags": bool(allow_unknown_tags),
            "run_validate": bool(run_validate),
            "run_debug": bool(run_debug),
            "debug": {
                "channels": debug_channels,
                "num_samples": int(debug_num_samples),
                "split": debug_split,
            }
            if run_debug
            else None,
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
        # 1) Convert.
        convert_py = REPO_ROOT / "tools" / "data_analysis" / "bdd" / "convert_bdd_pv26.py"
        convert_argv: List[str] = [
            sys.executable,
            "-u",
            str(convert_py),
            "--images-root",
            str(images_root),
            "--labels",
            str(labels_root),
            "--drivable-root",
            str(drivable_root),
            "--out-root",
            str(out_root),
            "--seed",
            str(seed),
            "--workers",
            str(workers),
            "--min-box-area-px",
            str(min_box_area_px),
            "--limit",
            str(limit),
            "--splits",
            str(splits),
        ]
        if include_rain:
            convert_argv.append("--include-rain")
        if include_night:
            convert_argv.append("--include-night")
        if allow_unknown_tags:
            convert_argv.append("--allow-unknown-tags")

        print(
            "[pv26][로딩][1/4 변환] 원본 이미지/라벨을 학습 포맷으로 변환하고 "
            "검출 라벨+주행가능영역+도로마커 마스크를 생성합니다.",
            flush=True,
        )
        r = _run_stage("convert_bdd_pv26", convert_argv, cwd=REPO_ROOT)
        stage_results.append(r)
        if r.returncode != 0:
            failed_stage = r.name
            failed_rc = r.returncode
            raise RuntimeError(f"stage failed: {r.name} rc={r.returncode}")

        # 2) Validate (optional).
        if run_validate:
            validate_py = REPO_ROOT / "tools" / "data_analysis" / "bdd" / "validate_pv26_dataset.py"
            validate_argv = [
                sys.executable,
                "-u",
                str(validate_py),
                "--out-root",
                str(out_root),
                "--workers",
                str(workers),
            ]
            print(
                "[pv26][로딩][2/4 검증] 생성된 파일의 존재/해상도/마스크 값 도메인/부분라벨 계약을 전수 검사합니다.",
                flush=True,
            )
            r = _run_stage("validate_pv26_dataset", validate_argv, cwd=REPO_ROOT)
            stage_results.append(r)
            if r.returncode != 0:
                failed_stage = r.name
                failed_rc = r.returncode
                raise RuntimeError(f"stage failed: {r.name} rc={r.returncode}")

        # 3) QC report (always).
        qc_py = REPO_ROOT / "tools" / "data_analysis" / "bdd" / "pv26_qc_report.py"
        qc_out = out_root / "meta" / "qc_report.json"
        qc_argv = [
            sys.executable,
            "-u",
            str(qc_py),
            "--dataset-root",
            str(out_root),
            "--workers",
            str(workers),
            "--out-json",
            str(qc_out),
        ]
        print(
            "[pv26][로딩][3/4 QC] split/태그/라벨 존재 분포와 마스크 non-empty 통계를 산출합니다.",
            flush=True,
        )
        r = _run_stage("pv26_qc_report", qc_argv, cwd=REPO_ROOT)
        stage_results.append(r)
        if r.returncode != 0:
            failed_stage = r.name
            failed_rc = r.returncode
            raise RuntimeError(f"stage failed: {r.name} rc={r.returncode}")

        # 4) Debug visualization (optional).
        if run_debug:
            debug_py = REPO_ROOT / "tools" / "debug" / "render_pv26_debug_masks.py"
            debug_out_root = out_root / "meta" / "debug_vis"
            dbg_argv = [
                sys.executable,
                "-u",
                str(debug_py),
                "--dataset-root",
                str(out_root),
                "--split",
                str(debug_split),
                "--channels",
                str(debug_channels),
                "--num-samples",
                str(debug_num_samples),
                "--out-root",
                str(debug_out_root),
                "--workers",
                str(workers),
                "--seed",
                str(seed),
            ]
            print(
                "[pv26][로딩][4/4 디버그 시각화] 마스크를 컬러맵/오버레이 이미지로 렌더링해 사람이 빠르게 품질 확인할 수 있게 만듭니다.",
                flush=True,
            )
            r = _run_stage("render_pv26_debug_masks", dbg_argv, cwd=REPO_ROOT)
            stage_results.append(r)
            if r.returncode != 0:
                failed_stage = r.name
                failed_rc = r.returncode
                raise RuntimeError(f"stage failed: {r.name} rc={r.returncode}")

    except KeyboardInterrupt:
        failed_stage = failed_stage or "keyboard_interrupt"
        failed_rc = failed_rc if failed_rc is not None else 130
        print("\n[pv26] aborted: KeyboardInterrupt", file=sys.stderr)
        manifest["status"] = "failed"
    except Exception as ex:  # noqa: BLE001 - orchestrator CLI
        if not failed_stage:
            failed_stage = "unknown"
        if failed_rc is None:
            failed_rc = 2
        print(f"[pv26] error: {type(ex).__name__}: {ex}", file=sys.stderr)
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
        # Ensure JSON is stable and readable.
        write_json(manifest_path, manifest)
        # Also emit a compact line for external orchestration.
        sys.stdout.write(
            json.dumps(
                {
                    "status": manifest["status"],
                    "out_root": str(out_root),
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


if __name__ == "__main__":
    raise SystemExit(main())
