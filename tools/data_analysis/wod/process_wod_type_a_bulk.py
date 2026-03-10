#!/usr/bin/env python3
"""
Scan and process many Waymo/WOD parquet contexts into per-context PV26 Type-A shards.

Workflow:
  1. `scan` records what is currently downloaded under training_root.
  2. `run` converts pending contexts one-by-one into shard roots.
  3. `merge` merges completed shard roots into one final PV26 dataset.

This keeps progress/state in JSON+CSV so raw parquet files can be downloaded,
converted, and deleted incrementally without losing the processing ledger.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pv26.dataset.manifest import read_manifest_csv
from pv26.dataset.wod_bulk import (
    WOD_STATUS_COMPLETED,
    WOD_STATUS_FAILED,
    WOD_STATUS_IN_PROGRESS,
    completed_shard_roots_from_state,
    iter_contexts_for_processing,
    load_wod_bulk_state,
    reconcile_wod_bulk_state,
    summarize_wod_bulk_state,
    write_wod_bulk_state,
    write_wod_bulk_state_csv,
)
from pv26.io import utc_now_iso
from tools.data_analysis.merge_pv26_type_a import merge_pv26_datasets


CONVERTER_SCRIPT = REPO_ROOT / "tools" / "data_analysis" / "wod" / "convert_wod_type_a.py"


def _default_state_path(shards_root: Path) -> Path:
    return shards_root / "meta" / "wod_bulk_state.json"


def _default_summary_csv_path(shards_root: Path) -> Path:
    return shards_root / "meta" / "wod_bulk_state.csv"


def _parse_contexts_csv(csv_text: str) -> list[str]:
    return [s.strip() for s in str(csv_text).split(",") if s.strip()]


def _persist_state(*, state_path: Path, summary_csv_path: Path, state: dict) -> None:
    state["updated_at"] = utc_now_iso()
    state["summary"] = summarize_wod_bulk_state(state)
    write_wod_bulk_state(state_path, state)
    write_wod_bulk_state_csv(summary_csv_path, state)


def _scan_and_persist(
    *,
    training_root: Path,
    shards_root: Path,
    state_path: Path,
    summary_csv_path: Path,
) -> dict:
    prior = load_wod_bulk_state(state_path)
    state = reconcile_wod_bulk_state(training_root=training_root, shards_root=shards_root, prior_state=prior)
    _persist_state(state_path=state_path, summary_csv_path=summary_csv_path, state=state)
    return state


def _print_summary(state: dict) -> None:
    summary = state.get("summary", {})
    print("[pv26][wod-bulk] summary:", flush=True)
    for key in [
        "total_contexts",
        "processable_now",
        "with_box",
        "status_pending",
        "status_blocked_missing_components",
        "status_in_progress",
        "status_completed",
        "status_failed",
    ]:
        print(f"  - {key}: {int(summary.get(key, 0))}", flush=True)


def _build_convert_command(
    *,
    training_root: Path,
    shard_root: Path,
    context_name: str,
    split_policy: str,
    splits: str,
    seed: int,
    run_id: str,
) -> list[str]:
    cmd = [
        sys.executable,
        str(CONVERTER_SCRIPT),
        "--training-root",
        str(training_root),
        "--out-root",
        str(shard_root),
        "--context-name",
        str(context_name),
        "--split-policy",
        str(split_policy),
        "--splits",
        str(splits),
        "--seed",
        str(int(seed)),
    ]
    if run_id:
        cmd.extend(["--run-id", run_id])
    return cmd


def _read_shard_stats(shard_root: Path) -> tuple[int, dict[str, int], int]:
    rows = read_manifest_csv(shard_root / "meta" / "split_manifest.csv")
    rows_by_split = Counter((row.get("split", "") or "").strip() for row in rows)
    has_det_rows = sum(1 for row in rows if (row.get("has_det", "") or "").strip() == "1")
    return len(rows), {k: int(v) for k, v in sorted(rows_by_split.items(), key=lambda kv: kv[0])}, int(has_det_rows)


def _delete_raw_components(*, training_root: Path, entry: dict) -> None:
    for key in ("image_relpath", "segmentation_relpath", "box_relpath"):
        rel = str(entry.get(key, "")).strip()
        if not rel:
            continue
        p = training_root / rel
        if p.exists():
            p.unlink()


def _run_process(args: argparse.Namespace) -> int:
    training_root = Path(args.training_root).expanduser().resolve()
    shards_root = Path(args.shards_root).expanduser().resolve()
    state_path = Path(args.state_path).expanduser().resolve()
    summary_csv_path = Path(args.summary_csv_path).expanduser().resolve()

    state = _scan_and_persist(
        training_root=training_root,
        shards_root=shards_root,
        state_path=state_path,
        summary_csv_path=summary_csv_path,
    )

    selected_contexts = _parse_contexts_csv(args.contexts)
    candidates = iter_contexts_for_processing(
        state,
        include_failed=bool(args.retry_failed),
        selected_contexts=selected_contexts,
    )
    if args.max_contexts and int(args.max_contexts) > 0:
        candidates = candidates[: int(args.max_contexts)]

    if not candidates:
        _print_summary(state)
        print("[pv26][wod-bulk] no contexts selected for processing", flush=True)
        return 0

    for entry in candidates:
        context_name = str(entry["context_name"])
        shard_root = Path(str(entry["shard_root"])).expanduser().resolve()
        run_id = str(args.run_id).strip() or f"wod-bulk:{context_name}"

        if shard_root.exists():
            if bool(args.overwrite_shard):
                shutil.rmtree(shard_root)
            else:
                entry["status"] = WOD_STATUS_FAILED
                entry["last_error"] = f"shard_root_exists:{shard_root}"
                _persist_state(state_path=state_path, summary_csv_path=summary_csv_path, state=state)
                continue

        entry["status"] = WOD_STATUS_IN_PROGRESS
        entry["attempt_count"] = int(entry.get("attempt_count", 0) or 0) + 1
        entry["last_started_at"] = utc_now_iso()
        entry["last_error"] = ""
        _persist_state(state_path=state_path, summary_csv_path=summary_csv_path, state=state)

        cmd = _build_convert_command(
            training_root=training_root,
            shard_root=shard_root,
            context_name=context_name,
            split_policy=str(args.split_policy),
            splits=str(args.splits),
            seed=int(args.seed),
            run_id=run_id,
        )
        print(f"[pv26][wod-bulk] convert context={context_name}", flush=True)
        print(f"[pv26][wod-bulk] cmd={' '.join(cmd)}", flush=True)
        proc = subprocess.run(cmd, cwd=REPO_ROOT, check=False)
        if proc.returncode != 0:
            entry["status"] = WOD_STATUS_FAILED
            entry["last_error"] = f"converter_exit_code:{int(proc.returncode)}"
            _persist_state(state_path=state_path, summary_csv_path=summary_csv_path, state=state)
            continue

        num_rows, rows_by_split, has_det_rows = _read_shard_stats(shard_root)
        entry["status"] = WOD_STATUS_COMPLETED
        entry["last_completed_at"] = utc_now_iso()
        entry["output_num_rows"] = int(num_rows)
        entry["output_rows_by_split"] = rows_by_split
        entry["output_has_det_rows"] = int(has_det_rows)

        if bool(args.delete_raw_on_success):
            _delete_raw_components(training_root=training_root, entry=entry)
            entry["raw_deleted_at"] = utc_now_iso()
            entry["has_image"] = False
            entry["has_segmentation"] = False
            entry["has_box"] = False
            entry["processable_now"] = False

        _persist_state(state_path=state_path, summary_csv_path=summary_csv_path, state=state)

    _print_summary(state)
    return 0


def _run_merge(args: argparse.Namespace) -> int:
    state_path = Path(args.state_path).expanduser().resolve()
    state = load_wod_bulk_state(state_path)
    if not state:
        raise SystemExit(f"missing state file: {state_path}")

    selected_contexts = set(_parse_contexts_csv(args.contexts))
    shard_roots = completed_shard_roots_from_state(state)
    if selected_contexts:
        filtered = []
        for ctx in state.get("contexts", []):
            if str(ctx.get("context_name", "")).strip() not in selected_contexts:
                continue
            if str(ctx.get("status", "")).strip() != WOD_STATUS_COMPLETED:
                continue
            shard_root = str(ctx.get("shard_root", "")).strip()
            if shard_root:
                filtered.append(Path(shard_root).expanduser())
        shard_roots = filtered

    if not shard_roots:
        raise SystemExit("no completed shard roots to merge")

    report = merge_pv26_datasets(
        input_roots=shard_roots,
        out_root=Path(args.out_root).expanduser(),
        materialize_mode=str(args.materialize_mode),
        workers=max(1, int(args.workers)),
        validate=bool(args.validate),
        validate_workers=max(1, int(args.validate_workers)),
        argv=sys.argv,
    )
    print(
        f"[pv26][wod-bulk] merged shards={len(shard_roots)} rows={int(report['num_rows']):,} "
        f"out_root={Path(args.out_root).expanduser().resolve()}",
        flush=True,
    )
    return 0


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Bulk WOD scan/process/merge pipeline for PV26 Type-A shards.")
    sub = p.add_subparsers(dest="cmd", required=True)

    def _add_common_scan_args(sp: argparse.ArgumentParser) -> None:
        sp.add_argument(
            "--training-root",
            type=Path,
            default=Path("datasets/WaymoOpenDataset/wod_perception_v2/training"),
            help="Waymo training root containing camera_image/camera_segmentation parquet directories.",
        )
        sp.add_argument(
            "--shards-root",
            type=Path,
            default=Path("datasets/pv26_wod_shards"),
            help="Root where per-context PV26 shard outputs and state metadata are stored.",
        )
        sp.add_argument(
            "--state-path",
            type=Path,
            default=None,
            help="Optional explicit state JSON path (default: <shards-root>/meta/wod_bulk_state.json).",
        )
        sp.add_argument(
            "--summary-csv-path",
            type=Path,
            default=None,
            help="Optional explicit CSV summary path (default: <shards-root>/meta/wod_bulk_state.csv).",
        )

    scan_p = sub.add_parser("scan", help="Scan currently downloaded WOD parquet contexts and write state metadata.")
    _add_common_scan_args(scan_p)

    run_p = sub.add_parser("run", help="Convert pending contexts into per-context PV26 shard roots.")
    _add_common_scan_args(run_p)
    run_p.add_argument(
        "--contexts",
        type=str,
        default="",
        help="Optional comma-separated context names to process. Default: all pending contexts.",
    )
    run_p.add_argument("--retry-failed", action="store_true", help="Retry contexts currently marked as failed.")
    run_p.add_argument("--overwrite-shard", action="store_true", help="Delete an existing shard root before retrying.")
    run_p.add_argument("--delete-raw-on-success", action="store_true", help="Delete raw parquet files after success.")
    run_p.add_argument("--max-contexts", type=int, default=0, help="Max contexts to process in this run (0=all).")
    run_p.add_argument("--seed", type=int, default=0, help="Split seed passed to the per-context converter.")
    run_p.add_argument(
        "--split-policy",
        type=str,
        default="stable_by_context",
        choices=["all_train", "stable_by_context"],
        help="Split policy for per-context conversion (default: stable_by_context).",
    )
    run_p.add_argument(
        "--splits",
        type=str,
        default="train,val,test",
        help="Comma-separated split filter passed to the per-context converter.",
    )
    run_p.add_argument("--run-id", type=str, default="", help="Optional run id prefix passed to the converter.")

    merge_p = sub.add_parser("merge", help="Merge completed shard roots into one final PV26 dataset root.")
    merge_p.add_argument(
        "--state-path",
        type=Path,
        required=True,
        help="State JSON created by scan/run.",
    )
    merge_p.add_argument(
        "--out-root",
        type=Path,
        required=True,
        help="Final merged PV26 output root. Must not already contain files.",
    )
    merge_p.add_argument(
        "--contexts",
        type=str,
        default="",
        help="Optional comma-separated completed context names to merge. Default: all completed shards.",
    )
    merge_p.add_argument(
        "--materialize-mode",
        type=str,
        default="auto",
        choices=["auto", "hardlink", "copy", "symlink"],
        help="How to materialize files into the merged root.",
    )
    merge_p.add_argument("--workers", type=int, default=8, help="Parallel workers for merge file materialization.")
    merge_p.add_argument("--validate", action="store_true", help="Run PV26 validator on the merged output.")
    merge_p.add_argument("--validate-workers", type=int, default=4, help="Workers for merge validation.")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_argparser()
    args = parser.parse_args(argv)

    if args.cmd == "merge":
        return _run_merge(args)

    shards_root = Path(args.shards_root).expanduser().resolve()
    state_path = Path(args.state_path).expanduser().resolve() if args.state_path else _default_state_path(shards_root)
    summary_csv_path = (
        Path(args.summary_csv_path).expanduser().resolve()
        if args.summary_csv_path
        else _default_summary_csv_path(shards_root)
    )
    args.state_path = state_path
    args.summary_csv_path = summary_csv_path

    if args.cmd == "scan":
        state = _scan_and_persist(
            training_root=Path(args.training_root).expanduser().resolve(),
            shards_root=shards_root,
            state_path=state_path,
            summary_csv_path=summary_csv_path,
        )
        _print_summary(state)
        print(f"[pv26][wod-bulk] state={state_path}", flush=True)
        print(f"[pv26][wod-bulk] summary_csv={summary_csv_path}", flush=True)
        return 0

    if args.cmd == "run":
        return _run_process(args)

    raise RuntimeError(f"unsupported command: {args.cmd}")


if __name__ == "__main__":
    raise SystemExit(main())
