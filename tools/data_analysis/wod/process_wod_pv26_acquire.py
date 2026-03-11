#!/usr/bin/env python3
"""
Acquire Waymo/WOD parquet from GCS for the PV26 WOD pipeline.

Workflow:
  1. `sync-index` lists remote objects and caches a context-level ledger.
  2. `download` downloads selected contexts into training_root.

Conversion into PV26 shards stays in `process_wod_pv26_bulk.py`.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pv26.dataset.wod_acquire import (
    WOD_DOWNLOAD_STATUS_DOWNLOADED,
    WOD_DOWNLOAD_STATUS_DOWNLOADING,
    WOD_DOWNLOAD_STATUS_FAILED,
    WOD_REMOTE_COMPONENTS,
    iter_contexts_for_download,
    load_wod_acquire_state,
    parse_gcloud_storage_ls_long,
    reconcile_wod_acquire_state,
    summarize_wod_acquire_state,
    write_wod_acquire_state,
    write_wod_acquire_state_csv,
)
from pv26.dataset.wod_bulk import load_wod_bulk_state
from pv26.io import utc_now_iso


BUCKET_URL = "gs://waymo_open_dataset_v_2_0_1"


def _default_state_path(shards_root: Path) -> Path:
    return shards_root / "meta" / "wod_acquire_state.json"


def _default_summary_csv_path(shards_root: Path) -> Path:
    return shards_root / "meta" / "wod_acquire_state.csv"


def _default_bulk_state_path(shards_root: Path) -> Path:
    return shards_root / "meta" / "wod_bulk_state.json"


def _parse_contexts_csv(csv_text: str) -> list[str]:
    return [s.strip() for s in str(csv_text).split(",") if s.strip()]


def _persist_state(*, state_path: Path, summary_csv_path: Path, state: dict) -> None:
    state["updated_at"] = utc_now_iso()
    state["summary"] = summarize_wod_acquire_state(state)
    write_wod_acquire_state(state_path, state)
    write_wod_acquire_state_csv(summary_csv_path, state)


def _print_summary(state: dict) -> None:
    summary = state.get("summary", {})
    print("[pv26][wod-acquire] summary:", flush=True)
    for key in [
        "total_contexts",
        "remote_processable",
        "remote_with_box",
        "download_status_remote_only",
        "download_status_downloading",
        "download_status_downloaded",
        "download_status_raw_deleted",
        "download_status_failed",
        "bulk_status_completed",
    ]:
        print(f"  - {key}: {int(summary.get(key, 0))}", flush=True)


def _build_gcloud_ls_command(*, bucket_url: str, component: str) -> list[str]:
    return [
        "gcloud",
        "storage",
        "ls",
        "--long",
        f"{bucket_url}/training/{component}/*",
    ]


def _build_gcloud_cp_command(*, remote_url: str, local_path: Path) -> list[str]:
    return [
        "gcloud",
        "storage",
        "cp",
        remote_url,
        str(local_path),
    ]


def _list_remote_component(*, bucket_url: str, component: str) -> list:
    cmd = _build_gcloud_ls_command(bucket_url=bucket_url, component=component)
    print(f"[pv26][wod-acquire] remote list component={component}", flush=True)
    proc = subprocess.run(cmd, cwd=REPO_ROOT, check=False, text=True, capture_output=True)
    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        raise RuntimeError(f"gcloud ls failed for {component}: rc={proc.returncode} {stderr}")
    items = parse_gcloud_storage_ls_long(proc.stdout, component=component)
    print(f"[pv26][wod-acquire] component={component} objects={len(items):,}", flush=True)
    return items


def _sync_remote_index(
    *,
    training_root: Path,
    shards_root: Path,
    state_path: Path,
    summary_csv_path: Path,
    bucket_url: str,
) -> dict:
    prior = load_wod_acquire_state(state_path)
    bulk_state = load_wod_bulk_state(_default_bulk_state_path(shards_root))
    remote_objects_by_component = {}
    with ThreadPoolExecutor(max_workers=len(WOD_REMOTE_COMPONENTS), thread_name_prefix="wod-index") as executor:
        futures = {
            executor.submit(_list_remote_component, bucket_url=bucket_url, component=component): component
            for component in WOD_REMOTE_COMPONENTS
        }
        for future in as_completed(futures):
            component = futures[future]
            remote_objects_by_component[component] = future.result()
    state = reconcile_wod_acquire_state(
        training_root=training_root,
        prior_state=prior,
        remote_objects_by_component=remote_objects_by_component,
        bulk_state=bulk_state,
    )
    state["bucket_url"] = bucket_url
    state["shards_root"] = str(shards_root)
    state["remote_synced_at"] = utc_now_iso()
    _persist_state(state_path=state_path, summary_csv_path=summary_csv_path, state=state)
    return state


def _download_one_object(*, remote_url: str, local_path: Path) -> None:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    if local_path.exists() and local_path.stat().st_size > 0:
        return
    partial_path = local_path.with_name(local_path.name + ".partial")
    if partial_path.exists():
        partial_path.unlink()
    cmd = _build_gcloud_cp_command(remote_url=remote_url, local_path=partial_path)
    proc = subprocess.run(cmd, cwd=REPO_ROOT, check=False, text=True, capture_output=True)
    if proc.returncode != 0:
        if partial_path.exists():
            partial_path.unlink()
        stderr = (proc.stderr or "").strip()
        raise RuntimeError(f"gcloud cp failed rc={proc.returncode}: {stderr}")
    partial_path.replace(local_path)


def _download_context_entry(
    *,
    entry: dict,
    state: dict,
    state_lock: threading.Lock,
    training_root: Path,
    state_path: Path,
    summary_csv_path: Path,
    include_box: bool,
) -> None:
    context_name = str(entry.get("context_name", "")).strip()
    try:
        with state_lock:
            entry["download_status"] = WOD_DOWNLOAD_STATUS_DOWNLOADING
            entry["attempt_count"] = int(entry.get("attempt_count", 0) or 0) + 1
            entry["last_started_at"] = utc_now_iso()
            entry["last_error"] = ""
            _persist_state(state_path=state_path, summary_csv_path=summary_csv_path, state=state)

        targets = [
            ("camera_image", "image"),
            ("camera_segmentation", "segmentation"),
        ]
        if include_box and bool(entry.get("remote_has_box", False)):
            targets.append(("camera_box", "box"))

        for component, suffix in targets:
            remote_url = str(entry.get(f"{component}_url", "")).strip()
            if not remote_url:
                continue
            local_relpath = str(entry.get(f"local_{suffix}_relpath", "")).strip() or f"{component}/{context_name}.parquet"
            local_path = training_root / local_relpath
            print(f"[pv26][wod-acquire] download context={context_name} component={component}", flush=True)
            _download_one_object(remote_url=remote_url, local_path=local_path)

        with state_lock:
            for component, suffix in [("camera_image", "image"), ("camera_segmentation", "segmentation"), ("camera_box", "box")]:
                local_path = training_root / str(entry.get(f"local_{suffix}_relpath", "")).strip()
                entry[f"local_has_{suffix}"] = bool(local_path.exists())
                entry[f"local_{suffix}_bytes"] = int(local_path.stat().st_size) if local_path.exists() else 0
            entry["download_status"] = WOD_DOWNLOAD_STATUS_DOWNLOADED
            entry["last_completed_at"] = utc_now_iso()
            _persist_state(state_path=state_path, summary_csv_path=summary_csv_path, state=state)
    except Exception as ex:  # noqa: BLE001
        with state_lock:
            entry["download_status"] = WOD_DOWNLOAD_STATUS_FAILED
            entry["last_error"] = f"{type(ex).__name__}:{ex}"
            _persist_state(state_path=state_path, summary_csv_path=summary_csv_path, state=state)


def _run_download(args: argparse.Namespace) -> int:
    training_root = Path(args.training_root).expanduser().resolve()
    shards_root = Path(args.shards_root).expanduser().resolve()
    state_path = Path(args.state_path).expanduser().resolve()
    summary_csv_path = Path(args.summary_csv_path).expanduser().resolve()

    state = load_wod_acquire_state(state_path)
    if not state:
        raise SystemExit(f"missing acquire state file: {state_path}; run sync-index first")

    selected_contexts = _parse_contexts_csv(args.contexts)
    candidates = iter_contexts_for_download(
        state,
        include_failed=bool(args.retry_failed),
        selected_contexts=selected_contexts,
    )
    if args.max_contexts and int(args.max_contexts) > 0:
        candidates = candidates[: int(args.max_contexts)]
    if not candidates:
        _print_summary(state)
        print("[pv26][wod-acquire] no contexts selected for download", flush=True)
        return 0

    jobs = max(1, int(args.jobs))
    training_root.mkdir(parents=True, exist_ok=True)
    state_lock = threading.Lock()
    print(f"[pv26][wod-acquire] selected_contexts={len(candidates)} jobs={jobs}", flush=True)
    with ThreadPoolExecutor(max_workers=jobs, thread_name_prefix="wod-download") as executor:
        futures = [
            executor.submit(
                _download_context_entry,
                entry=entry,
                state=state,
                state_lock=state_lock,
                training_root=training_root,
                state_path=state_path,
                summary_csv_path=summary_csv_path,
                include_box=bool(args.include_box),
            )
            for entry in candidates
        ]
        for future in as_completed(futures):
            future.result()

    refreshed = reconcile_wod_acquire_state(
        training_root=training_root,
        prior_state=state,
        bulk_state=load_wod_bulk_state(_default_bulk_state_path(shards_root)),
    )
    refreshed["bucket_url"] = str(state.get("bucket_url", args.bucket_url))
    refreshed["shards_root"] = str(shards_root)
    refreshed["remote_synced_at"] = str(state.get("remote_synced_at", ""))
    _persist_state(state_path=state_path, summary_csv_path=summary_csv_path, state=refreshed)
    _print_summary(refreshed)
    return 0


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Acquire WOD parquet from GCS for the PV26 pipeline.")
    sub = p.add_subparsers(dest="cmd", required=True)

    def _add_common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument(
            "--training-root",
            type=Path,
            default=Path("datasets/WaymoOpenDataset/wod_perception_v2/training"),
            help="Local training root where camera_image/camera_segmentation parquet files are stored.",
        )
        sp.add_argument(
            "--shards-root",
            type=Path,
            default=Path("datasets/pv26_wod_shards"),
            help="Shard/meta root used by the WOD PV26 pipeline.",
        )
        sp.add_argument(
            "--state-path",
            type=Path,
            default=None,
            help="Optional explicit acquire-state JSON path (default: <shards-root>/meta/wod_acquire_state.json).",
        )
        sp.add_argument(
            "--summary-csv-path",
            type=Path,
            default=None,
            help="Optional explicit acquire-state CSV path (default: <shards-root>/meta/wod_acquire_state.csv).",
        )
        sp.add_argument(
            "--bucket-url",
            type=str,
            default=BUCKET_URL,
            help="Waymo GCS bucket base URL.",
        )

    sync_p = sub.add_parser("sync-index", help="List remote Waymo objects and refresh the cached acquire state.")
    _add_common(sync_p)

    dl_p = sub.add_parser("download", help="Download selected WOD contexts from the cached remote index.")
    _add_common(dl_p)
    dl_p.add_argument("--contexts", type=str, default="", help="Optional comma-separated context names to download.")
    dl_p.add_argument("--retry-failed", action="store_true", help="Retry contexts marked as failed.")
    dl_p.add_argument("--max-contexts", type=int, default=0, help="Max contexts to download this run (0=all).")
    dl_p.add_argument("--jobs", type=int, default=4, help="Parallel download worker count.")
    dl_p.add_argument("--include-box", action="store_true", help="Also download camera_box parquet when available.")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_argparser()
    args = parser.parse_args(argv)
    shards_root = Path(args.shards_root).expanduser().resolve()
    state_path = Path(args.state_path).expanduser().resolve() if args.state_path else _default_state_path(shards_root)
    summary_csv_path = (
        Path(args.summary_csv_path).expanduser().resolve()
        if args.summary_csv_path
        else _default_summary_csv_path(shards_root)
    )
    args.state_path = state_path
    args.summary_csv_path = summary_csv_path

    if args.cmd == "sync-index":
        state = _sync_remote_index(
            training_root=Path(args.training_root).expanduser().resolve(),
            shards_root=shards_root,
            state_path=state_path,
            summary_csv_path=summary_csv_path,
            bucket_url=str(args.bucket_url),
        )
        _print_summary(state)
        print(f"[pv26][wod-acquire] state={state_path}", flush=True)
        print(f"[pv26][wod-acquire] summary_csv={summary_csv_path}", flush=True)
        return 0

    if args.cmd == "download":
        return _run_download(args)

    raise RuntimeError(f"unsupported command: {args.cmd}")


if __name__ == "__main__":
    raise SystemExit(main())
