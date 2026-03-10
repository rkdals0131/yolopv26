#!/usr/bin/env python3
"""
한국어 대화형 WOD -> PV26 벌크 처리 실행기.

이 스크립트는 이미 다운로드된 WOD parquet를 대상으로 다음 단계를 한 번에 안내합니다.
  1) scan  : 현재 training_root 아래 parquet 인벤토리를 스캔
  2) run   : 처리 가능한 context를 shard 단위 PV26로 변환
  3) merge : 완료된 shard를 최종 PV26 root로 병합

다운로드 자동화는 아직 포함하지 않습니다.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Deque, Dict, Iterable, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pv26.dataset.wod_acquire import (
    WOD_DOWNLOAD_STATUS_DOWNLOADED,
    WOD_DOWNLOAD_STATUS_RAW_DELETED,
    iter_contexts_for_download,
    load_wod_acquire_state,
    reconcile_wod_acquire_state,
    summarize_wod_acquire_state,
)
from pv26.dataset.wod_bulk import (
    iter_contexts_for_processing,
    load_wod_bulk_state,
    reconcile_wod_bulk_state,
    summarize_wod_bulk_state,
)
from pv26.io import utc_now_iso, write_json


DEFAULT_TRAINING_ROOT = Path("datasets/WaymoOpenDataset/wod_perception_v2/training")
DEFAULT_SHARDS_ROOT = Path("datasets/pv26_wod_shards")
DEFAULT_MERGED_OUT_ROOT = Path("datasets/pv26_waymo_merged")
WOD_ACQUIRE_SCRIPT = REPO_ROOT / "tools" / "data_analysis" / "wod" / "process_wod_pv26_acquire.py"
WOD_BULK_SCRIPT = REPO_ROOT / "tools" / "data_analysis" / "wod" / "process_wod_pv26_bulk.py"


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


@dataclass(frozen=True)
class ProgressSnapshot:
    total: int
    pending: int
    blocked: int
    in_progress: int
    completed: int
    skipped: int
    failed: int
    output_rows: int
    det_rows: int
    in_progress_names: List[str]


@dataclass(frozen=True)
class AcquireProgressSnapshot:
    total: int
    remote_only: int
    downloading: int
    downloaded: int
    raw_deleted: int
    failed: int
    downloaded_rows: int
    in_progress_names: List[str]


@dataclass(frozen=True)
class RunTargetPlan:
    context_names: List[str]

    @property
    def total(self) -> int:
        return len(self.context_names)


def _default_workers() -> int:
    cpu = os.cpu_count() or 1
    return max(1, min(8, cpu))


def _prompt(msg: str) -> str:
    try:
        return input(msg)
    except EOFError:
        print("[pv26][오류] 대화형 입력이 예상보다 일찍 종료되었습니다.", file=sys.stderr)
        raise SystemExit(2)


def _ask_yes_no(prompt: str, *, default: Optional[bool]) -> bool:
    if default is True:
        suffix = " [Y/n] "
    elif default is False:
        suffix = " [y/N] "
    else:
        suffix = " [y/n] "

    yes_set = {"y", "yes", "1", "true"}
    no_set = {"n", "no", "0", "false"}

    while True:
        s = _prompt(prompt + suffix).strip().lower()
        if not s:
            if default is None:
                print("y 또는 n으로 입력해 주세요.", file=sys.stderr)
                continue
            return bool(default)
        if s in yes_set:
            return True
        if s in no_set:
            return False
        print("입력이 올바르지 않습니다. y/n, yes/no, 1/0 중 하나로 입력해 주세요.", file=sys.stderr)


def _ask_int(prompt: str, *, default: int, min_value: Optional[int] = None) -> int:
    while True:
        s = _prompt(f"{prompt} [기본={default}] ").strip()
        if not s:
            value = int(default)
        else:
            try:
                value = int(s)
            except ValueError:
                print("정수를 입력해 주세요.", file=sys.stderr)
                continue
        if min_value is not None and value < min_value:
            print(f"{min_value} 이상 값을 입력해 주세요.", file=sys.stderr)
            continue
        return value


def _ask_path(prompt: str, *, default: Path) -> Path:
    s = _prompt(f"{prompt} [기본={default}] ").strip()
    return Path(s).expanduser() if s else default


def _ask_text(prompt: str, *, default: str = "") -> str:
    shown = default if default else "빈값"
    s = _prompt(f"{prompt} [기본={shown}] ").strip()
    return s if s else default


def _ask_choice(prompt: str, *, options: Dict[str, str], default_key: str, aliases: Dict[str, str]) -> str:
    while True:
        print(prompt)
        for key, label in options.items():
            marker = " (기본)" if key == default_key else ""
            print(f"  {key}) {label}{marker}")
        chosen = _canonical_choice(_prompt("선택 입력: "), aliases, default=default_key)
        if chosen in options:
            return chosen
        print("지원하지 않는 선택입니다. 숫자나 별칭을 다시 입력해 주세요.", file=sys.stderr)


def _tail(text: str, *, max_chars: int = 8000) -> str:
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def _stream_pipe(*, src, dst, tail_chunks: Deque[str], tail_size_ref: List[int], max_tail_chars: int) -> None:
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


def _format_bytes(num_bytes: int) -> str:
    size = float(max(0, int(num_bytes)))
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            if unit == "B":
                return f"{int(size)} {unit}"
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{int(num_bytes)} B"


def _canonical_choice(text: str, aliases: Dict[str, str], default: str = "") -> str:
    raw = str(text).strip().lower()
    if not raw:
        return str(default)
    return str(aliases.get(raw, raw))


def _print_state_summary(state: Dict[str, Any]) -> None:
    summary = summarize_wod_bulk_state(state)
    total_bytes = 0
    for ctx in state.get("contexts", []):
        total_bytes += int(ctx.get("image_bytes", 0) or 0)
        total_bytes += int(ctx.get("segmentation_bytes", 0) or 0)
        total_bytes += int(ctx.get("box_bytes", 0) or 0)
    print("[pv26][상태요약] 현재 WOD 인벤토리")
    print(f"  - 전체 context:     {int(summary.get('total_contexts', 0)):,}")
    print(f"  - 지금 처리 가능:   {int(summary.get('processable_now', 0)):,}")
    print(f"  - box 포함:         {int(summary.get('with_box', 0)):,}")
    print(f"  - 대기 중:          {int(summary.get('status_pending', 0)):,}")
    print(f"  - 구성 부족:        {int(summary.get('status_blocked_missing_components', 0)):,}")
    print(f"  - 진행 중:          {int(summary.get('status_in_progress', 0)):,}")
    print(f"  - 완료:             {int(summary.get('status_completed', 0)):,}")
    print(f"  - 건너뜀:           {int(summary.get('status_skipped_empty_export', 0)):,}")
    print(f"  - 실패:             {int(summary.get('status_failed', 0)):,}")
    print(f"  - 원본 총용량:      {_format_bytes(total_bytes)}")


def _print_acquire_state_summary(state: Dict[str, Any]) -> None:
    summary = summarize_wod_acquire_state(state)
    total_remote_bytes = 0
    total_local_bytes = 0
    for ctx in state.get("contexts", []):
        total_remote_bytes += int(ctx.get("camera_image_bytes", 0) or 0)
        total_remote_bytes += int(ctx.get("camera_segmentation_bytes", 0) or 0)
        total_remote_bytes += int(ctx.get("camera_box_bytes", 0) or 0)
        total_local_bytes += int(ctx.get("local_image_bytes", 0) or 0)
        total_local_bytes += int(ctx.get("local_segmentation_bytes", 0) or 0)
        total_local_bytes += int(ctx.get("local_box_bytes", 0) or 0)
    print("[pv26][상태요약] 현재 WOD 원격/다운로드 인벤토리")
    print(f"  - 전체 context:        {int(summary.get('total_contexts', 0)):,}")
    print(f"  - 원격 처리 가능:      {int(summary.get('remote_processable', 0)):,}")
    print(f"  - 원격 box 포함:       {int(summary.get('remote_with_box', 0)):,}")
    print(f"  - remote_only:         {int(summary.get('download_status_remote_only', 0)):,}")
    print(f"  - downloading:         {int(summary.get('download_status_downloading', 0)):,}")
    print(f"  - downloaded:          {int(summary.get('download_status_downloaded', 0)):,}")
    print(f"  - raw_deleted:         {int(summary.get('download_status_raw_deleted', 0)):,}")
    print(f"  - failed_download:     {int(summary.get('download_status_failed', 0)):,}")
    print(f"  - bulk completed:      {int(summary.get('bulk_status_completed', 0)):,}")
    print(f"  - 원격 총용량:         {_format_bytes(total_remote_bytes)}")
    print(f"  - 로컬 총용량:         {_format_bytes(total_local_bytes)}")


def _build_run_target_plan(
    *,
    training_root: Path,
    shards_root: Path,
    state_path: Path,
    selected_contexts_csv: str,
    retry_failed: bool,
    max_contexts: int,
) -> RunTargetPlan:
    prior = load_wod_bulk_state(state_path)
    state = reconcile_wod_bulk_state(training_root=training_root, shards_root=shards_root, prior_state=prior)
    selected_contexts = [s.strip() for s in selected_contexts_csv.split(",") if s.strip()]
    candidates = iter_contexts_for_processing(
        state,
        include_failed=bool(retry_failed),
        selected_contexts=selected_contexts,
    )
    if max_contexts > 0:
        candidates = candidates[: int(max_contexts)]
    return RunTargetPlan(context_names=[str(ctx.get("context_name", "")).strip() for ctx in candidates])


def _build_download_target_plan(
    *,
    training_root: Path,
    acquire_state_path: Path,
    selected_contexts_csv: str,
    retry_failed: bool,
    max_contexts: int,
) -> RunTargetPlan:
    prior = load_wod_acquire_state(acquire_state_path)
    state = reconcile_wod_acquire_state(training_root=training_root, prior_state=prior)
    selected_contexts = [s.strip() for s in selected_contexts_csv.split(",") if s.strip()]
    candidates = iter_contexts_for_download(
        state,
        include_failed=bool(retry_failed),
        selected_contexts=selected_contexts,
    )
    if max_contexts > 0:
        candidates = candidates[: int(max_contexts)]
    return RunTargetPlan(context_names=[str(ctx.get("context_name", "")).strip() for ctx in candidates])


def _build_dynamic_bulk_progress_line(
    *,
    training_root: Path,
    shards_root: Path,
    state_path: Path,
    selected_contexts_csv: str,
    retry_failed: bool,
    max_contexts: int,
) -> str:
    target_plan = _build_run_target_plan(
        training_root=training_root,
        shards_root=shards_root,
        state_path=state_path,
        selected_contexts_csv=selected_contexts_csv,
        retry_failed=retry_failed,
        max_contexts=max_contexts,
    )
    return _build_bulk_progress_line(state_path, target_plan.context_names)


def _build_dynamic_acquire_progress_line(
    *,
    training_root: Path,
    acquire_state_path: Path,
    selected_contexts_csv: str,
    retry_failed: bool,
    max_contexts: int,
) -> str:
    target_plan = _build_download_target_plan(
        training_root=training_root,
        acquire_state_path=acquire_state_path,
        selected_contexts_csv=selected_contexts_csv,
        retry_failed=retry_failed,
        max_contexts=max_contexts,
    )
    return _build_acquire_progress_line(acquire_state_path, target_plan.context_names)


def _summarize_target_progress(state: Dict[str, Any], target_contexts: Iterable[str]) -> Dict[str, Any]:
    target_set = {str(name).strip() for name in target_contexts if str(name).strip()}
    summary: Dict[str, Any] = {
        "target_total": len(target_set),
        "pending": 0,
        "blocked": 0,
        "in_progress": 0,
        "completed": 0,
        "skipped": 0,
        "failed": 0,
        "output_num_rows": 0,
        "output_has_det_rows": 0,
        "in_progress_names": [],
    }
    if not target_set:
        return summary

    for ctx in state.get("contexts", []):
        name = str(ctx.get("context_name", "")).strip()
        if name not in target_set:
            continue
        status = str(ctx.get("status", "")).strip()
        if status == "pending":
            summary["pending"] += 1
        elif status == "blocked_missing_components":
            summary["blocked"] += 1
        elif status == "in_progress":
            summary["in_progress"] += 1
            summary["in_progress_names"].append(name)
        elif status == "completed":
            summary["completed"] += 1
        elif status == "skipped_empty_export":
            summary["skipped"] += 1
        elif status == "failed":
            summary["failed"] += 1
        summary["output_num_rows"] += int(ctx.get("output_num_rows", 0) or 0)
        summary["output_has_det_rows"] += int(ctx.get("output_has_det_rows", 0) or 0)
    return summary


def _load_progress_snapshot(state_path: Path, *, target_contexts: Iterable[str]) -> ProgressSnapshot:
    state = load_wod_bulk_state(state_path)
    summary = _summarize_target_progress(state, target_contexts)
    return ProgressSnapshot(
        total=int(summary.get("target_total", 0) or 0),
        pending=int(summary.get("pending", 0) or 0),
        blocked=int(summary.get("blocked", 0) or 0),
        in_progress=int(summary.get("in_progress", 0) or 0),
        completed=int(summary.get("completed", 0) or 0),
        skipped=int(summary.get("skipped", 0) or 0),
        failed=int(summary.get("failed", 0) or 0),
        output_rows=int(summary.get("output_num_rows", 0) or 0),
        det_rows=int(summary.get("output_has_det_rows", 0) or 0),
        in_progress_names=[str(x) for x in summary.get("in_progress_names", []) if str(x).strip()],
    )


def _summarize_acquire_target_progress(state: Dict[str, Any], target_contexts: Iterable[str]) -> Dict[str, Any]:
    target_set = {str(name).strip() for name in target_contexts if str(name).strip()}
    summary: Dict[str, Any] = {
        "target_total": len(target_set),
        "remote_only": 0,
        "downloading": 0,
        "downloaded": 0,
        "raw_deleted": 0,
        "failed": 0,
        "downloaded_rows": 0,
        "in_progress_names": [],
    }
    if not target_set:
        return summary
    for ctx in state.get("contexts", []):
        name = str(ctx.get("context_name", "")).strip()
        if name not in target_set:
            continue
        status = str(ctx.get("download_status", "")).strip()
        if status == "remote_only":
            summary["remote_only"] += 1
        elif status == "downloading":
            summary["downloading"] += 1
            summary["in_progress_names"].append(name)
        elif status == "downloaded":
            summary["downloaded"] += 1
        elif status == "raw_deleted":
            summary["raw_deleted"] += 1
        elif status == "failed":
            summary["failed"] += 1
        if status in {WOD_DOWNLOAD_STATUS_DOWNLOADED, WOD_DOWNLOAD_STATUS_RAW_DELETED}:
            summary["downloaded_rows"] += 1
    return summary


def _load_acquire_progress_snapshot(state_path: Path, *, target_contexts: Iterable[str]) -> AcquireProgressSnapshot:
    state = load_wod_acquire_state(state_path)
    summary = _summarize_acquire_target_progress(state, target_contexts)
    return AcquireProgressSnapshot(
        total=int(summary.get("target_total", 0) or 0),
        remote_only=int(summary.get("remote_only", 0) or 0),
        downloading=int(summary.get("downloading", 0) or 0),
        downloaded=int(summary.get("downloaded", 0) or 0),
        raw_deleted=int(summary.get("raw_deleted", 0) or 0),
        failed=int(summary.get("failed", 0) or 0),
        downloaded_rows=int(summary.get("downloaded_rows", 0) or 0),
        in_progress_names=[str(x) for x in summary.get("in_progress_names", []) if str(x).strip()],
    )


def _render_target_progress_line(summary: Dict[str, Any]) -> str:
    total = int(summary.get("target_total", 0) or 0)
    if total <= 0:
        return "[pv26][진행] 이번 run에서 처리할 대상 context가 없습니다."
    completed = int(summary.get("completed", 0) or 0)
    skipped = int(summary.get("skipped", 0) or 0)
    failed = int(summary.get("failed", 0) or 0)
    in_progress = int(summary.get("in_progress", 0) or 0)
    pending = int(summary.get("pending", 0) or 0)
    blocked = int(summary.get("blocked", 0) or 0)
    done = completed + skipped + failed
    percent = (100.0 * done / total) if total > 0 else 0.0
    pieces = [
        f"[pv26][진행] {done}/{total} ({percent:.1f}%)",
        f"완료 {completed}",
        f"건너뜀 {skipped}",
        f"실패 {failed}",
        f"진행중 {in_progress}",
        f"대기 {pending}",
    ]
    if blocked > 0:
        pieces.append(f"구성부족 {blocked}")
    pieces.append(f"출력행 {int(summary.get('output_num_rows', 0) or 0):,}")
    pieces.append(f"det행 {int(summary.get('output_has_det_rows', 0) or 0):,}")
    active = [str(x) for x in summary.get("in_progress_names", []) if str(x).strip()]
    if active:
        shown = ", ".join(active[:2])
        if len(active) > 2:
            shown += f" 외 {len(active) - 2}개"
        pieces.append(f"현재 {shown}")
    return " | ".join(pieces)


def _render_acquire_progress_line(summary: Dict[str, Any]) -> str:
    total = int(summary.get("target_total", 0) or 0)
    if total <= 0:
        return "[pv26][진행] 이번 download 대상 context가 없습니다."
    downloaded = int(summary.get("downloaded", 0) or 0)
    failed = int(summary.get("failed", 0) or 0)
    downloading = int(summary.get("downloading", 0) or 0)
    remote_only = int(summary.get("remote_only", 0) or 0)
    raw_deleted = int(summary.get("raw_deleted", 0) or 0)
    done = downloaded + failed
    percent = (100.0 * done / total) if total > 0 else 0.0
    pieces = [
        f"[pv26][다운로드진행] {done}/{total} ({percent:.1f}%)",
        f"완료 {downloaded}",
        f"실패 {failed}",
        f"진행중 {downloading}",
        f"대기 {remote_only}",
    ]
    if raw_deleted > 0:
        pieces.append(f"raw_deleted {raw_deleted}")
    pieces.append(f"로컬완료표시 {int(summary.get('downloaded_rows', 0) or 0):,}")
    active = [str(x) for x in summary.get("in_progress_names", []) if str(x).strip()]
    if active:
        shown = ", ".join(active[:2])
        if len(active) > 2:
            shown += f" 외 {len(active) - 2}개"
        pieces.append(f"현재 {shown}")
    return " | ".join(pieces)


def _build_bulk_progress_line(state_path: Path, target_contexts: Iterable[str]) -> str:
    snapshot = _load_progress_snapshot(state_path, target_contexts=target_contexts)
    return _render_target_progress_line(
        {
            "target_total": snapshot.total,
            "pending": snapshot.pending,
            "blocked": snapshot.blocked,
            "in_progress": snapshot.in_progress,
            "completed": snapshot.completed,
            "skipped": snapshot.skipped,
            "failed": snapshot.failed,
            "output_num_rows": snapshot.output_rows,
            "output_has_det_rows": snapshot.det_rows,
            "in_progress_names": snapshot.in_progress_names,
        }
    )


def _build_acquire_progress_line(state_path: Path, target_contexts: Iterable[str]) -> str:
    snapshot = _load_acquire_progress_snapshot(state_path, target_contexts=target_contexts)
    return _render_acquire_progress_line(
        {
            "target_total": snapshot.total,
            "remote_only": snapshot.remote_only,
            "downloading": snapshot.downloading,
            "downloaded": snapshot.downloaded,
            "raw_deleted": snapshot.raw_deleted,
            "failed": snapshot.failed,
            "downloaded_rows": snapshot.downloaded_rows,
            "in_progress_names": snapshot.in_progress_names,
        }
    )


def _build_bulk_command(
    *,
    subcommand: str,
    training_root: Optional[Path] = None,
    shards_root: Optional[Path] = None,
    state_path: Optional[Path] = None,
    summary_csv_path: Optional[Path] = None,
    contexts: str = "",
    retry_failed: bool = False,
    overwrite_shard: bool = False,
    delete_raw_on_success: bool = False,
    max_contexts: int = 0,
    jobs: int = 1,
    seed: int = 0,
    split_policy: str = "stable_by_context",
    splits: str = "train,val,test",
    run_id: str = "",
    out_root: Optional[Path] = None,
    materialize_mode: str = "auto",
    workers: int = 8,
    validate: bool = False,
    validate_workers: int = 4,
) -> List[str]:
    cmd = [sys.executable, "-u", str(WOD_BULK_SCRIPT), str(subcommand)]
    if subcommand in {"scan", "run"}:
        if training_root is None or shards_root is None or state_path is None or summary_csv_path is None:
            raise ValueError("scan/run requires training_root, shards_root, state_path, summary_csv_path")
        cmd.extend(
            [
                "--training-root",
                str(training_root),
                "--shards-root",
                str(shards_root),
                "--state-path",
                str(state_path),
                "--summary-csv-path",
                str(summary_csv_path),
            ]
        )
    if subcommand == "run":
        cmd.extend(
            [
                "--contexts",
                str(contexts),
                "--max-contexts",
                str(int(max_contexts)),
                "--jobs",
                str(int(jobs)),
                "--seed",
                str(int(seed)),
                "--split-policy",
                str(split_policy),
                "--splits",
                str(splits),
                "--run-id",
                str(run_id),
            ]
        )
        if retry_failed:
            cmd.append("--retry-failed")
        if overwrite_shard:
            cmd.append("--overwrite-shard")
        if delete_raw_on_success:
            cmd.append("--delete-raw-on-success")
        return cmd
    if subcommand == "scan":
        return cmd
    if subcommand == "merge":
        if state_path is None or out_root is None:
            raise ValueError("merge requires state_path and out_root")
        cmd.extend(
            [
                "--state-path",
                str(state_path),
                "--out-root",
                str(out_root),
                "--materialize-mode",
                str(materialize_mode),
                "--workers",
                str(int(workers)),
                "--validate-workers",
                str(int(validate_workers)),
            ]
        )
        if validate:
            cmd.append("--validate")
        return cmd
    raise ValueError(f"unsupported subcommand: {subcommand}")


def _build_acquire_command(
    *,
    subcommand: str,
    training_root: Path,
    shards_root: Path,
    state_path: Path,
    summary_csv_path: Path,
    contexts: str = "",
    retry_failed: bool = False,
    max_contexts: int = 0,
    jobs: int = 4,
    include_box: bool = False,
) -> List[str]:
    cmd = [
        sys.executable,
        "-u",
        str(WOD_ACQUIRE_SCRIPT),
        str(subcommand),
        "--training-root",
        str(training_root),
        "--shards-root",
        str(shards_root),
        "--state-path",
        str(state_path),
        "--summary-csv-path",
        str(summary_csv_path),
    ]
    if subcommand == "sync-index":
        return cmd
    if subcommand == "download":
        cmd.extend(
            [
                "--contexts",
                str(contexts),
                "--max-contexts",
                str(int(max_contexts)),
                "--jobs",
                str(int(jobs)),
            ]
        )
        if retry_failed:
            cmd.append("--retry-failed")
        if include_box:
            cmd.append("--include-box")
        return cmd
    raise ValueError(f"unsupported acquire subcommand: {subcommand}")


def _progress_monitor_loop(
    *,
    stop_event: threading.Event,
    build_line: Callable[[], str],
    poll_interval_s: float = 2.0,
) -> None:
    last_line = ""
    last_error = ""
    while not stop_event.wait(poll_interval_s):
        try:
            line = build_line()
            if line != last_line:
                print(line, flush=True)
                last_line = line
                last_error = ""
        except Exception as exc:  # pragma: no cover - best effort monitor
            err = f"[pv26][진행] 상태 파일 읽기 대기 중: {exc}"
            if err != last_error:
                print(err, flush=True)
                last_error = err


def _run_stage(
    name: str,
    argv: Sequence[str],
    *,
    cwd: Path,
    progress_line_fn: Optional[Callable[[], str]] = None,
) -> StageResult:
    started = utc_now_iso()
    t0 = time.monotonic()
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    proc = subprocess.Popen(
        list(argv),
        cwd=str(cwd),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=1,
        env=env,
    )
    if proc.stdout is None or proc.stderr is None:
        raise RuntimeError(f"stage failed to open pipes: {name}")

    out_chunks: Deque[str] = deque()
    err_chunks: Deque[str] = deque()
    out_size_ref = [0]
    err_size_ref = [0]

    stdout_thread = threading.Thread(
        target=_stream_pipe,
        kwargs={
            "src": proc.stdout,
            "dst": sys.stdout,
            "tail_chunks": out_chunks,
            "tail_size_ref": out_size_ref,
            "max_tail_chars": 8000,
        },
        daemon=True,
    )
    stderr_thread = threading.Thread(
        target=_stream_pipe,
        kwargs={
            "src": proc.stderr,
            "dst": sys.stderr,
            "tail_chunks": err_chunks,
            "tail_size_ref": err_size_ref,
            "max_tail_chars": 8000,
        },
        daemon=True,
    )

    stop_event = threading.Event()
    monitor_thread: Optional[threading.Thread] = None
    if progress_line_fn is not None:
        monitor_thread = threading.Thread(
            target=_progress_monitor_loop,
            kwargs={
                "stop_event": stop_event,
                "build_line": progress_line_fn,
            },
            daemon=True,
        )

    stdout_thread.start()
    stderr_thread.start()
    if monitor_thread is not None:
        monitor_thread.start()

    rc = proc.wait()
    stop_event.set()
    stdout_thread.join()
    stderr_thread.join()
    if monitor_thread is not None:
        monitor_thread.join()
    completed = utc_now_iso()
    elapsed = time.monotonic() - t0
    print(f"[pv26][단계완료][{name}] rc={rc} 소요={elapsed:.1f}초", flush=True)

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


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="한국어 대화형 WOD -> PV26 벌크 처리 실행기")
    parser.add_argument("--training-root", type=Path, default=DEFAULT_TRAINING_ROOT)
    parser.add_argument("--shards-root", type=Path, default=DEFAULT_SHARDS_ROOT)
    parser.add_argument("--merged-out-root", type=Path, default=DEFAULT_MERGED_OUT_ROOT)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)

    print("\n[pv26][안내] 이 스크립트는 WOD를 원격 인덱스 동기화 -> 다운로드 -> 변환 -> 병합까지 안내합니다.")
    print("[pv26][안내] WOD는 전용 파이프라인으로 처리하고, merge는 필요할 때만 마지막에 수행합니다.")

    training_root = _ask_path("WOD training_root 경로를 입력해 주세요", default=args.training_root)
    shards_root = _ask_path("중간 shard/output 루트 경로를 입력해 주세요", default=args.shards_root)
    merged_out_root = _ask_path("최종 병합 out_root 경로를 입력해 주세요", default=args.merged_out_root)
    acquire_state_path = shards_root / "meta" / "wod_acquire_state.json"
    acquire_summary_csv_path = shards_root / "meta" / "wod_acquire_state.csv"
    state_path = shards_root / "meta" / "wod_bulk_state.json"
    summary_csv_path = shards_root / "meta" / "wod_bulk_state.csv"
    manifest_path = shards_root / "meta" / "run_manifest_interactive.json"

    action = _ask_choice(
        "\n[1단계] 실행 작업을 선택해 주세요.",
        options={
            "1": "원격 인덱스만 동기화",
            "2": "원격 인덱스 -> 다운로드",
            "3": "원격 인덱스 -> 다운로드 -> 변환",
            "4": "원격 인덱스 -> 다운로드 -> 변환 -> merge",
            "5": "로컬 scan만 실행",
            "6": "로컬 run만 실행",
            "7": "로컬 merge만 실행",
            "8": "로컬 scan -> run 실행",
            "9": "로컬 scan -> run -> merge 실행",
            "10": "현재 상태만 요약해서 보기",
        },
        default_key="3",
        aliases={
            "remote_sync": "1",
            "remote_download": "2",
            "remote_run": "3",
            "remote_full": "4",
            "scan": "5",
            "run": "6",
            "merge": "7",
            "scanrun": "8",
            "local_full": "9",
            "full": "4",
            "status": "10",
            "요약": "10",
        },
    )

    is_remote_action = action in {"1", "2", "3", "4"}
    run_actions = {"3", "4", "6", "8", "9"}
    merge_actions = {"4", "7", "9"}
    local_scan_actions = {"5", "8", "9"}
    local_only_actions = {"5", "6", "7", "8", "9"}

    if action in local_only_actions and not training_root.exists():
        print(f"[pv26][오류] training_root를 찾을 수 없습니다: {training_root}", file=sys.stderr)
        return 2

    contexts_csv = ""
    download_retry_failed = False
    download_jobs = max(2, min(4, _default_workers()))
    include_box = False
    retry_failed = False
    overwrite_shard = False
    delete_raw_on_success = False
    max_contexts = 0
    jobs = max(2, min(4, _default_workers()))
    seed = 0
    split_policy = "stable_by_context"
    splits = "train,val,test"
    run_id = ""
    materialize_mode = "auto"
    merge_workers = _default_workers()
    validate_merge = True
    validate_workers = max(1, min(4, merge_workers))

    if action in {"2", "3", "4"}:
        print("\n[2단계] 원격 다운로드 옵션을 입력합니다.")
        contexts_csv = _ask_text(
            "특정 context만 다운로드하려면 쉼표로 입력하세요. 전체면 비워두세요",
            default="",
        )
        download_retry_failed = _ask_yes_no("이전에 다운로드 실패한 context도 재시도할까요?", default=True)
        max_contexts = _ask_int("이번 다운로드에서 처리할 최대 context 수 (0=전부)", default=0, min_value=0)
        download_jobs = _ask_int(
            "동시에 다운로드할 jobs 수",
            default=max(2, min(4, _default_workers())),
            min_value=1,
        )
        include_box = _ask_yes_no("camera_box parquet도 같이 받을까요?", default=False)

    if action in run_actions:
        print("\n[3단계] 변환(run) 옵션을 입력합니다.")
        if not contexts_csv:
            contexts_csv = _ask_text(
                "특정 context만 변환하려면 쉼표로 입력하세요. 전체면 비워두세요",
                default="",
            )
        retry_failed = _ask_yes_no("이전에 변환 실패한 context도 재시도할까요?", default=True)
        overwrite_shard = _ask_yes_no("기존 shard_root가 있으면 삭제 후 다시 만들까요?", default=False)
        delete_raw_on_success = _ask_yes_no(
            "context 처리 성공 직후 원본 parquet를 삭제할까요?",
            default=False,
        )
        if action not in {"2"}:
            max_contexts = _ask_int("이번 변환에서 처리할 최대 context 수 (0=전부)", default=max_contexts, min_value=0)
        jobs = _ask_int("동시에 처리할 run jobs 수", default=max(2, min(4, _default_workers())), min_value=1)
        seed = _ask_int("split seed", default=0, min_value=0)
        split_policy_choice = _ask_choice(
            "split 정책을 선택해 주세요.",
            options={
                "1": "stable_by_context (권장)",
                "2": "all_train",
            },
            default_key="1",
            aliases={
                "stable": "1",
                "stable_by_context": "1",
                "all_train": "2",
                "train": "2",
            },
        )
        split_policy = "stable_by_context" if split_policy_choice == "1" else "all_train"
        splits = _ask_text("허용 split CSV", default="train,val,test")
        run_id = _ask_text("run-id 접두어", default="")

    if action in {"6", "8", "9"}:
        print("\n[3단계] 로컬 run 옵션을 입력합니다.")
        contexts_csv = _ask_text(
            "특정 context만 처리하려면 쉼표로 입력하세요. 전체면 비워두세요",
            default="",
        )
        retry_failed = _ask_yes_no("이전에 실패한 context도 재시도할까요?", default=True)
        overwrite_shard = _ask_yes_no("기존 shard_root가 있으면 삭제 후 다시 만들까요?", default=False)
        delete_raw_on_success = _ask_yes_no(
            "context 처리 성공 직후 원본 parquet를 삭제할까요?",
            default=False,
        )
        max_contexts = _ask_int("이번 run에서 처리할 최대 context 수 (0=전부)", default=0, min_value=0)
        jobs = _ask_int("동시에 처리할 jobs 수", default=max(2, min(4, _default_workers())), min_value=1)
        seed = _ask_int("split seed", default=0, min_value=0)
        split_policy_choice = _ask_choice(
            "split 정책을 선택해 주세요.",
            options={
                "1": "stable_by_context (권장)",
                "2": "all_train",
            },
            default_key="1",
            aliases={
                "stable": "1",
                "stable_by_context": "1",
                "all_train": "2",
                "train": "2",
            },
        )
        split_policy = "stable_by_context" if split_policy_choice == "1" else "all_train"
        splits = _ask_text("허용 split CSV", default="train,val,test")
        run_id = _ask_text("run-id 접두어", default="")

    if action in merge_actions:
        print("\n[4단계] merge 옵션을 입력합니다.")
        materialize_choice = _ask_choice(
            "파일 materialize 방식을 선택해 주세요.",
            options={
                "1": "auto (권장)",
                "2": "hardlink",
                "3": "copy",
                "4": "symlink",
            },
            default_key="1",
            aliases={
                "auto": "1",
                "hardlink": "2",
                "copy": "3",
                "symlink": "4",
            },
        )
        materialize_mode = {
            "1": "auto",
            "2": "hardlink",
            "3": "copy",
            "4": "symlink",
        }[materialize_choice]
        merge_workers = _ask_int("merge workers", default=_default_workers(), min_value=1)
        validate_merge = _ask_yes_no("merge 후 validate를 실행할까요?", default=True)
        validate_workers = _ask_int("validate workers", default=max(1, min(4, merge_workers)), min_value=1)

    if action == "10":
        acquire_state = {}
        if acquire_state_path.exists():
            acquire_state = load_wod_acquire_state(acquire_state_path)
        elif training_root.exists():
            acquire_state = reconcile_wod_acquire_state(training_root=training_root)
        if acquire_state:
            _print_acquire_state_summary(acquire_state)
            print(f"[pv26][상태파일] {acquire_state_path}")

        bulk_state = {}
        if training_root.exists():
            bulk_state = reconcile_wod_bulk_state(
                training_root=training_root,
                shards_root=shards_root,
                prior_state=load_wod_bulk_state(state_path),
            )
        elif state_path.exists():
            bulk_state = load_wod_bulk_state(state_path)
        if bulk_state:
            _print_state_summary(bulk_state)
            print(f"[pv26][상태파일] {state_path}")
        if not acquire_state and not bulk_state:
            print("[pv26][오류] 보여줄 WOD acquire/bulk 상태 파일이 없습니다.", file=sys.stderr)
            return 2
        return 0

    print("\n[pv26][설정요약]")
    print(f"  - training_root:       {training_root}")
    print(f"  - shards_root:         {shards_root}")
    print(f"  - acquire_state_path:  {acquire_state_path}")
    print(f"  - acquire_summary_csv: {acquire_summary_csv_path}")
    print(f"  - state_path:          {state_path}")
    print(f"  - merged_out_root:     {merged_out_root}")
    print(f"  - action:              {action}")
    if action in {"2", "3", "4"}:
        print(f"  - download_contexts:   {contexts_csv or '(전체)'}")
        print(f"  - retry_failed_dl:     {download_retry_failed}")
        print(f"  - download_max_ctx:    {max_contexts}")
        print(f"  - download_jobs:       {download_jobs}")
        print(f"  - include_box:         {include_box}")
    if action in run_actions:
        print(f"  - contexts_csv:        {contexts_csv or '(전체)'}")
        print(f"  - retry_failed:        {retry_failed}")
        print(f"  - overwrite_shard:     {overwrite_shard}")
        print(f"  - delete_raw_success:  {delete_raw_on_success}")
        print(f"  - max_contexts:        {max_contexts}")
        print(f"  - jobs:                {jobs}")
        print(f"  - seed:                {seed}")
        print(f"  - split_policy:        {split_policy}")
        print(f"  - splits:              {splits}")
        print(f"  - run_id:              {run_id or '(자동)'}")
    if action in merge_actions:
        print(f"  - materialize_mode:    {materialize_mode}")
        print(f"  - merge_workers:       {merge_workers}")
        print(f"  - validate_merge:      {validate_merge}")
        print(f"  - validate_workers:    {validate_workers}")

    proceed = _ask_yes_no("\n위 설정대로 실행할까요?", default=True)
    if not proceed:
        print("[pv26][중단] 사용자가 실행을 취소했습니다.", file=sys.stderr)
        return 0

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest: Dict[str, Any] = {
        "started_at": utc_now_iso(),
        "completed_at": "",
        "status": "running",
        "failed_stage": "",
        "failed_returncode": None,
        "paths": {
            "training_root": str(training_root),
            "shards_root": str(shards_root),
            "acquire_state_path": str(acquire_state_path),
            "acquire_summary_csv_path": str(acquire_summary_csv_path),
            "state_path": str(state_path),
            "summary_csv_path": str(summary_csv_path),
            "merged_out_root": str(merged_out_root),
        },
        "interactive_answers": {
            "action": action,
            "contexts_csv": contexts_csv,
            "download_retry_failed": bool(download_retry_failed),
            "download_jobs": int(download_jobs),
            "include_box": bool(include_box),
            "retry_failed": bool(retry_failed),
            "overwrite_shard": bool(overwrite_shard),
            "delete_raw_on_success": bool(delete_raw_on_success),
            "max_contexts": int(max_contexts),
            "jobs": int(jobs),
            "seed": int(seed),
            "split_policy": split_policy,
            "splits": splits,
            "run_id": run_id,
            "materialize_mode": materialize_mode,
            "merge_workers": int(merge_workers),
            "validate_merge": bool(validate_merge),
            "validate_workers": int(validate_workers),
        },
        "commands_invoked": [],
        "stage_results": [],
    }
    write_json(manifest_path, manifest)

    stages: List[tuple[str, str, List[str], Optional[Callable[[], str]]]] = []

    if action in {"1", "2", "3", "4"}:
        acquire_sync_argv = _build_acquire_command(
            subcommand="sync-index",
            training_root=training_root,
            shards_root=shards_root,
            state_path=acquire_state_path,
            summary_csv_path=acquire_summary_csv_path,
        )
        stages.append(
            (
                "sync-index",
                "인증된 gcloud listing으로 WOD 원격 object 목록을 조회하고 context 단위 캐시를 갱신합니다.",
                acquire_sync_argv,
                None,
            )
        )

    if action in {"2", "3", "4"}:
        acquire_download_argv = _build_acquire_command(
            subcommand="download",
            training_root=training_root,
            shards_root=shards_root,
            state_path=acquire_state_path,
            summary_csv_path=acquire_summary_csv_path,
            contexts=contexts_csv,
            retry_failed=download_retry_failed,
            max_contexts=max_contexts,
            jobs=download_jobs,
            include_box=include_box,
        )
        stages.append(
            (
                "download",
                "원격 index에서 선택한 context를 병렬 다운로드합니다.",
                acquire_download_argv,
                lambda tr=training_root, sp=acquire_state_path, ctx=contexts_csv, rf=download_retry_failed, mc=max_contexts: _build_dynamic_acquire_progress_line(
                    training_root=tr,
                    acquire_state_path=sp,
                    selected_contexts_csv=ctx,
                    retry_failed=rf,
                    max_contexts=mc,
                ),
            )
        )

    if action in local_scan_actions:
        scan_argv = _build_bulk_command(
            subcommand="scan",
            training_root=training_root,
            shards_root=shards_root,
            state_path=state_path,
            summary_csv_path=summary_csv_path,
        )
        stages.append(
            (
                "scan",
                "현재 training_root 아래 parquet 인벤토리를 스캔하고 context별 상태 JSON/CSV를 갱신합니다.",
                scan_argv,
                None,
            )
        )

    if action in run_actions:
        run_argv = _build_bulk_command(
            subcommand="run",
            training_root=training_root,
            shards_root=shards_root,
            state_path=state_path,
            summary_csv_path=summary_csv_path,
            contexts=contexts_csv,
            retry_failed=retry_failed,
            overwrite_shard=overwrite_shard,
            delete_raw_on_success=delete_raw_on_success,
            max_contexts=max_contexts,
            jobs=jobs,
            seed=seed,
            split_policy=split_policy,
            splits=splits,
            run_id=run_id,
        )
        stages.append(
            (
                "run",
                "처리 가능한 context를 하나씩 PV26 shard로 변환합니다. 실시간 로그와 진행도가 함께 표시됩니다.",
                run_argv,
                lambda tr=training_root, sr=shards_root, sp=state_path, ctx=contexts_csv, rf=retry_failed, mc=max_contexts: _build_dynamic_bulk_progress_line(
                    training_root=tr,
                    shards_root=sr,
                    state_path=sp,
                    selected_contexts_csv=ctx,
                    retry_failed=rf,
                    max_contexts=mc,
                ),
            )
        )

    if action in merge_actions:
        if merged_out_root.exists():
            overwrite_merge = _ask_yes_no(
                f"merge out_root가 이미 존재합니다: {merged_out_root}\n삭제 후 다시 만들까요?",
                default=False,
            )
            if not overwrite_merge:
                print("[pv26][중단] 기존 merge out_root 덮어쓰기를 사용자가 거절했습니다.", file=sys.stderr)
                return 0
            shutil.rmtree(merged_out_root)
            print(f"[pv26][정리] 기존 merge out_root 삭제: {merged_out_root}")

        merge_argv = _build_bulk_command(
            subcommand="merge",
            state_path=state_path,
            out_root=merged_out_root,
            materialize_mode=materialize_mode,
            workers=merge_workers,
            validate=validate_merge,
            validate_workers=validate_workers,
        )
        stages.append(
            (
                "merge",
                "완료된 shard만 모아 최종 PV26 dataset root로 병합합니다.",
                merge_argv,
                None,
            )
        )

    stage_results: List[StageResult] = []
    failed_stage = ""
    failed_rc: Optional[int] = None

    try:
        total_stages = len(stages)
        for index, (name, description, argv2, progress_line_fn) in enumerate(stages, start=1):
            print(f"\n[pv26][단계 {index}/{total_stages}] {name}")
            print(f"[pv26][설명] {description}")
            if name == "download":
                plan = _build_download_target_plan(
                    training_root=training_root,
                    acquire_state_path=acquire_state_path,
                    selected_contexts_csv=contexts_csv,
                    retry_failed=download_retry_failed,
                    max_contexts=max_contexts,
                )
                print(f"[pv26][설명] 이번 download 대상 context 수: {plan.total:,}")
            if name == "run":
                plan = _build_run_target_plan(
                    training_root=training_root,
                    shards_root=shards_root,
                    state_path=state_path,
                    selected_contexts_csv=contexts_csv,
                    retry_failed=retry_failed,
                    max_contexts=max_contexts,
                )
                print(f"[pv26][설명] 이번 run 대상 context 수: {plan.total:,}")
            result = _run_stage(
                name,
                argv2,
                cwd=REPO_ROOT,
                progress_line_fn=progress_line_fn,
            )
            stage_results.append(result)
            manifest["commands_invoked"].append(result.argv)
            manifest["stage_results"] = [
                {
                    "name": sr.name,
                    "argv": sr.argv,
                    "cwd": sr.cwd,
                    "started_at": sr.started_at,
                    "completed_at": sr.completed_at,
                    "returncode": sr.returncode,
                    "stdout_tail": sr.stdout_tail,
                    "stderr_tail": sr.stderr_tail,
                }
                for sr in stage_results
            ]
            write_json(manifest_path, manifest)

            if result.returncode != 0:
                failed_stage = result.name
                failed_rc = result.returncode
                raise RuntimeError(f"stage failed: {result.name} rc={result.returncode}")

            if name in {"sync-index", "download"}:
                acquire_after = load_wod_acquire_state(acquire_state_path)
                if acquire_after:
                    _print_acquire_state_summary(acquire_after)
                    print(f"[pv26][상태파일] {acquire_state_path}")
                    print(f"[pv26][요약CSV]  {acquire_summary_csv_path}")

            if name in {"scan", "run"}:
                state_after = load_wod_bulk_state(state_path)
                if state_after:
                    _print_state_summary(state_after)
                    print(f"[pv26][상태파일] {state_path}")
                    print(f"[pv26][요약CSV]  {summary_csv_path}")

    except Exception as exc:
        manifest["status"] = "failed"
        manifest["failed_stage"] = failed_stage
        manifest["failed_returncode"] = failed_rc
        manifest["completed_at"] = utc_now_iso()
        manifest["error"] = str(exc)
        write_json(manifest_path, manifest)
        if failed_stage:
            print(f"[pv26][실패] 단계 {failed_stage} 에서 중단되었습니다. rc={failed_rc}", file=sys.stderr)
        else:
            print(f"[pv26][실패] {exc}", file=sys.stderr)
        return int(failed_rc or 1)

    manifest["status"] = "completed"
    manifest["completed_at"] = utc_now_iso()
    write_json(manifest_path, manifest)
    print(f"\n[pv26][완료] 모든 요청 단계가 끝났습니다.")
    print(f"[pv26][기록] run manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
