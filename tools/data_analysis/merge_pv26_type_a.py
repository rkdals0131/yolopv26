#!/usr/bin/env python3
"""
Merge multiple PV26 Type-A dataset roots into one manifest-compatible root.

The merged dataset keeps the standard PV26 directory layout and writes:
  - meta/split_manifest.csv
  - meta/class_map.yaml
  - meta/source_stats.csv
  - meta/merge_report.json
  - meta/input_datasets/<dataset_name>/*  (selected source metadata snapshots)

By default it discovers input roots under `datasets/` matching `pv26_v1*`,
skipping previous merged outputs that already contain `meta/merge_report.json`.
"""

from __future__ import annotations

import argparse
import csv
import errno
import json
import os
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
import shutil
import sys
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pv26.dataset.manifest import MANIFEST_COLUMNS, read_manifest_csv, validate_manifest_row_basic
from pv26.io import utc_now_iso, write_json
from pv26.dataset.validation import validate_pv26_dataset


FILE_RELPATH_KEYS: Tuple[str, ...] = (
    "image_relpath",
    "det_relpath",
    "da_relpath",
    "rm_lane_marker_relpath",
    "rm_road_marker_non_lane_relpath",
    "rm_stop_line_relpath",
    "rm_lane_subclass_relpath",
    "semantic_relpath",
)

SOURCE_META_FILENAMES: Tuple[str, ...] = (
    "checksums.sha256",
    "class_map.yaml",
    "conversion_report.json",
    "qc_report.json",
    "run_manifest.json",
    "run_manifest_interactive.json",
    "source_stats.csv",
)

SPLIT_ORDER: Mapping[str, int] = {"train": 0, "val": 1, "test": 2}


@dataclass(frozen=True)
class InputDataset:
    root: Path
    rows: List[Dict[str, str]]
    class_map_text: str
    source_names: Tuple[str, ...]
    meta_snapshot_files: Tuple[str, ...]


def _default_workers() -> int:
    cpu = os.cpu_count() or 1
    return max(1, min(16, cpu if cpu <= 4 else cpu - 2))


def _normalize_root(path: Path) -> Path:
    return path.expanduser().resolve()


def _safe_dataset_name(root: Path) -> str:
    return root.name.replace(os.sep, "_")


def _discover_input_roots(*, dataset_parent: Path, pattern: str, out_root: Path) -> List[Path]:
    roots: List[Path] = []
    parent = _normalize_root(dataset_parent)
    out_norm = _normalize_root(out_root)
    if not parent.exists():
        raise FileNotFoundError(f"dataset parent not found: {parent}")
    for cand in sorted(parent.glob(pattern)):
        if not cand.is_dir():
            continue
        cand_norm = _normalize_root(cand)
        meta = cand_norm / "meta"
        if cand_norm == out_norm:
            continue
        if not (meta / "split_manifest.csv").exists():
            continue
        if (meta / "merge_report.json").exists():
            continue
        roots.append(cand_norm)
    return roots


def _read_json_if_exists(path: Path) -> Optional[Any]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_input_dataset(root: Path) -> InputDataset:
    root = _normalize_root(root)
    meta = root / "meta"
    manifest_path = meta / "split_manifest.csv"
    class_map_path = meta / "class_map.yaml"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing manifest: {manifest_path}")
    if not class_map_path.exists():
        raise FileNotFoundError(f"missing class map: {class_map_path}")

    rows = read_manifest_csv(manifest_path)
    errors: List[str] = []
    for idx, row in enumerate(rows):
        row_errors = validate_manifest_row_basic(row)
        if row_errors:
            errors.extend([f"{root.name}: row[{idx}] {msg}" for msg in row_errors[:10]])
        if len(errors) >= 50:
            break
    if errors:
        raise ValueError("invalid input manifest rows:\n" + "\n".join(errors))

    sources = tuple(sorted({(row.get("source", "") or "").strip() for row in rows if row.get("source", "").strip()}))
    snap_files = tuple(name for name in SOURCE_META_FILENAMES if (meta / name).exists())
    return InputDataset(
        root=root,
        rows=rows,
        class_map_text=class_map_path.read_text(encoding="utf-8"),
        source_names=sources,
        meta_snapshot_files=snap_files,
    )


def _ensure_clean_out_root(out_root: Path) -> None:
    out_root = _normalize_root(out_root)
    if not out_root.exists():
        return
    if not out_root.is_dir():
        raise NotADirectoryError(f"out_root exists and is not a directory: {out_root}")
    contents = list(out_root.iterdir())
    if contents:
        raise FileExistsError(f"out_root is not empty: {out_root}")


def _collect_merged_rows_and_files(datasets: Sequence[InputDataset]) -> Tuple[List[Dict[str, str]], Dict[str, Path]]:
    if not datasets:
        raise ValueError("no input datasets provided")

    class_map0 = datasets[0].class_map_text
    for ds in datasets[1:]:
        if ds.class_map_text != class_map0:
            raise ValueError(f"class_map.yaml mismatch between inputs: {datasets[0].root.name} vs {ds.root.name}")

    merged_rows: List[Dict[str, str]] = []
    file_map: Dict[str, Path] = {}
    seen_sample_ids: Dict[str, Path] = {}

    for ds in datasets:
        for row in ds.rows:
            sample_id = (row.get("sample_id", "") or "").strip()
            if not sample_id:
                raise ValueError(f"missing sample_id in manifest: {ds.root}")
            prev = seen_sample_ids.get(sample_id)
            if prev is not None:
                raise ValueError(f"duplicate sample_id across inputs: {sample_id} ({prev} vs {ds.root})")
            seen_sample_ids[sample_id] = ds.root

            out_row = {k: row.get(k, "") for k in MANIFEST_COLUMNS}
            merged_rows.append(out_row)

            for key in FILE_RELPATH_KEYS:
                rel = (row.get(key, "") or "").strip()
                if not rel:
                    continue
                src = ds.root / rel
                if not src.exists():
                    raise FileNotFoundError(f"missing source file for merge: key={key} path={src}")
                prev_src = file_map.get(rel)
                if prev_src is not None and prev_src != src:
                    raise ValueError(f"relpath collision across inputs: {rel} ({prev_src} vs {src})")
                file_map[rel] = src

    merged_rows.sort(key=lambda row: (SPLIT_ORDER.get(row.get("split", ""), 99), row.get("source", ""), row.get("sample_id", "")))
    return merged_rows, file_map


def _materialize_one_file(*, src: Path, dst: Path, mode: str) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        try:
            if dst.samefile(src):
                return "existing"
        except FileNotFoundError:
            pass
        raise FileExistsError(f"destination already exists with different content: {dst}")

    if mode == "hardlink":
        os.link(src, dst)
        return "hardlink"
    if mode == "symlink":
        dst.symlink_to(src)
        return "symlink"
    if mode == "copy":
        shutil.copy2(src, dst)
        return "copy"
    if mode == "auto":
        try:
            os.link(src, dst)
            return "hardlink"
        except OSError as exc:
            if exc.errno not in {errno.EXDEV, errno.EPERM, errno.EACCES, errno.ENOTSUP, errno.EMLINK, errno.EINVAL}:
                raise
            shutil.copy2(src, dst)
            return "copy"
    raise ValueError(f"unsupported materialize mode: {mode}")


def _materialize_files(*, out_root: Path, file_map: Mapping[str, Path], mode: str, workers: int) -> Dict[str, int]:
    out_root = _normalize_root(out_root)
    counts: Counter[str] = Counter()
    items = sorted(file_map.items())
    total = len(items)
    if total == 0:
        return dict(counts)

    def _job(item: Tuple[str, Path]) -> str:
        rel, src = item
        return _materialize_one_file(src=src, dst=out_root / rel, mode=mode)

    if workers <= 1 or total <= 1:
        for idx, item in enumerate(items, start=1):
            op = _job(item)
            counts[op] += 1
            if idx % 2000 == 0 or idx == total:
                print(f"[pv26][merge] files {idx:,}/{total:,}", flush=True)
        return dict(counts)

    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as ex:
        futures = {ex.submit(_job, item): item[0] for item in items}
        done = 0
        for fut in as_completed(futures):
            op = fut.result()
            counts[op] += 1
            done += 1
            if done % 2000 == 0 or done == total:
                print(f"[pv26][merge] files {done:,}/{total:,}", flush=True)
    return dict(counts)


def _write_manifest_csv(path: Path, rows: Iterable[Mapping[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in MANIFEST_COLUMNS})


def _write_source_stats_csv(path: Path, rows: Sequence[Mapping[str, str]]) -> None:
    counts = Counter((row.get("source", "") or "").strip() for row in rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["source", "num_samples"])
        writer.writeheader()
        for source, num_samples in sorted(counts.items(), key=lambda kv: kv[0]):
            writer.writerow({"source": source, "num_samples": str(int(num_samples))})


def _copy_input_meta_snapshots(*, datasets: Sequence[InputDataset], out_root: Path) -> List[Dict[str, Any]]:
    snapshots: List[Dict[str, Any]] = []
    base = out_root / "meta" / "input_datasets"
    for ds in datasets:
        dst_dir = base / _safe_dataset_name(ds.root)
        dst_dir.mkdir(parents=True, exist_ok=True)
        copied: List[str] = []
        for name in ds.meta_snapshot_files:
            src = ds.root / "meta" / name
            if not src.exists():
                continue
            shutil.copy2(src, dst_dir / name)
            copied.append(name)
        snapshots.append(
            {
                "dataset_root": str(ds.root),
                "dataset_name": ds.root.name,
                "source_names": list(ds.source_names),
                "num_rows": len(ds.rows),
                "copied_meta_files": copied,
            }
        )
    return snapshots


def _build_merge_report(
    *,
    out_root: Path,
    datasets: Sequence[InputDataset],
    rows: Sequence[Mapping[str, str]],
    file_map: Mapping[str, Path],
    materialize_counts: Mapping[str, int],
    mode: str,
    argv: Sequence[str],
    validation_errors: Optional[List[str]] = None,
) -> Dict[str, Any]:
    rows_by_source = Counter((row.get("source", "") or "").strip() for row in rows)
    rows_by_split = Counter((row.get("split", "") or "").strip() for row in rows)
    rows_by_source_split: Dict[str, Dict[str, int]] = defaultdict(dict)
    for row in rows:
        source = (row.get("source", "") or "").strip()
        split = (row.get("split", "") or "").strip()
        rows_by_source_split[source][split] = rows_by_source_split[source].get(split, 0) + 1

    flag_keys = [
        "has_det",
        "has_da",
        "has_rm_lane_marker",
        "has_rm_road_marker_non_lane",
        "has_rm_stop_line",
        "has_rm_lane_subclass",
        "has_semantic_id",
    ]
    flag_counts: Dict[str, Dict[str, int]] = {}
    for key in flag_keys:
        flag_counter = Counter((row.get(key, "") or "").strip() for row in rows)
        flag_counts[key] = {k: int(v) for k, v in sorted(flag_counter.items(), key=lambda kv: kv[0])}

    input_reports: List[Dict[str, Any]] = []
    for ds in datasets:
        meta = ds.root / "meta"
        input_reports.append(
            {
                "dataset_root": str(ds.root),
                "dataset_name": ds.root.name,
                "num_rows": len(ds.rows),
                "source_names": list(ds.source_names),
                "row_counts_by_split": dict(
                    sorted(Counter((row.get("split", "") or "").strip() for row in ds.rows).items(), key=lambda kv: kv[0])
                ),
                "meta_files_present": list(ds.meta_snapshot_files),
                "conversion_report": _read_json_if_exists(meta / "conversion_report.json"),
                "qc_report": _read_json_if_exists(meta / "qc_report.json"),
            }
        )

    report: Dict[str, Any] = {
        "created_at": utc_now_iso(),
        "out_root": str(out_root),
        "argv": list(argv),
        "materialize_mode": mode,
        "num_input_roots": len(datasets),
        "num_rows": len(rows),
        "num_unique_files": len(file_map),
        "materialize_counts": {k: int(v) for k, v in materialize_counts.items()},
        "rows_by_source": {k: int(v) for k, v in sorted(rows_by_source.items(), key=lambda kv: kv[0])},
        "rows_by_split": {k: int(v) for k, v in sorted(rows_by_split.items(), key=lambda kv: kv[0])},
        "rows_by_source_split": {
            source: {k: int(v) for k, v in sorted(split_counts.items(), key=lambda kv: SPLIT_ORDER.get(kv[0], 99))}
            for source, split_counts in sorted(rows_by_source_split.items(), key=lambda kv: kv[0])
        },
        "flag_counts": flag_counts,
        "inputs": input_reports,
    }
    if validation_errors is not None:
        report["validation"] = {
            "ok": len(validation_errors) == 0,
            "num_errors": len(validation_errors),
            "errors_head": validation_errors[:50],
        }
    return report


def merge_pv26_datasets(
    *,
    input_roots: Sequence[Path],
    out_root: Path,
    materialize_mode: str = "auto",
    workers: int = 1,
    validate: bool = False,
    validate_workers: int = 1,
    argv: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    datasets = [_load_input_dataset(root) for root in input_roots]
    if not datasets:
        raise ValueError("no input datasets selected for merge")

    out_root = _normalize_root(out_root)
    _ensure_clean_out_root(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    merged_rows, file_map = _collect_merged_rows_and_files(datasets)
    print(
        f"[pv26][merge] inputs={len(datasets)} rows={len(merged_rows):,} unique_files={len(file_map):,} "
        f"mode={materialize_mode}",
        flush=True,
    )
    materialize_counts = _materialize_files(
        out_root=out_root,
        file_map=file_map,
        mode=materialize_mode,
        workers=max(1, int(workers)),
    )

    meta_dir = out_root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    (meta_dir / "class_map.yaml").write_text(datasets[0].class_map_text, encoding="utf-8")
    _write_manifest_csv(meta_dir / "split_manifest.csv", merged_rows)
    _write_source_stats_csv(meta_dir / "source_stats.csv", merged_rows)
    input_snapshots = _copy_input_meta_snapshots(datasets=datasets, out_root=out_root)
    write_json(meta_dir / "input_datasets.json", {"created_at": utc_now_iso(), "datasets": input_snapshots})

    validation_errors: Optional[List[str]] = None
    if validate:
        summary = validate_pv26_dataset(out_root, workers=max(1, int(validate_workers)))
        validation_errors = list(summary.errors)
        if summary.errors:
            raise RuntimeError(
                "merged dataset validation failed:\n" + "\n".join(summary.errors[:50])
            )

    report = _build_merge_report(
        out_root=out_root,
        datasets=datasets,
        rows=merged_rows,
        file_map=file_map,
        materialize_counts=materialize_counts,
        mode=materialize_mode,
        argv=list(argv or []),
        validation_errors=validation_errors,
    )
    write_json(meta_dir / "merge_report.json", report)
    return report


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Merge multiple PV26 Type-A dataset roots into one merged root.")
    p.add_argument(
        "--input-roots",
        type=Path,
        nargs="*",
        default=None,
        help="Explicit PV26 dataset roots to merge. If omitted, discover under --dataset-parent.",
    )
    p.add_argument(
        "--dataset-parent",
        type=Path,
        default=Path("datasets"),
        help="Parent directory used for auto-discovery when --input-roots is omitted (default: datasets).",
    )
    p.add_argument(
        "--discover-glob",
        type=str,
        default="pv26_v1*",
        help="Glob used for auto-discovery under --dataset-parent (default: pv26_v1*).",
    )
    p.add_argument(
        "--out-root",
        type=Path,
        required=True,
        help="Output merged PV26 dataset root.",
    )
    p.add_argument(
        "--materialize-mode",
        type=str,
        default="auto",
        choices=["auto", "hardlink", "copy", "symlink"],
        help="How to materialize files into the merged root (default: auto=hardlink then copy fallback).",
    )
    p.add_argument("--workers", type=int, default=_default_workers(), help="Parallel workers for file materialization.")
    p.add_argument("--validate", action="store_true", help="Run PV26 validator on the merged output.")
    p.add_argument(
        "--validate-workers",
        type=int,
        default=max(1, min(12, _default_workers())),
        help="Workers for validation when --validate is set.",
    )
    return p


def main() -> int:
    args = build_argparser().parse_args()
    out_root = Path(args.out_root).expanduser()

    if args.input_roots:
        input_roots = [Path(p).expanduser() for p in args.input_roots]
    else:
        input_roots = _discover_input_roots(
            dataset_parent=Path(args.dataset_parent),
            pattern=str(args.discover_glob),
            out_root=out_root,
        )

    if not input_roots:
        raise SystemExit("no input datasets found for merge")

    print("[pv26][merge] input roots:", flush=True)
    for root in input_roots:
        print(f"  - {Path(root).expanduser().resolve()}", flush=True)

    report = merge_pv26_datasets(
        input_roots=input_roots,
        out_root=out_root,
        materialize_mode=str(args.materialize_mode),
        workers=max(1, int(args.workers)),
        validate=bool(args.validate),
        validate_workers=max(1, int(args.validate_workers)),
        argv=sys.argv,
    )
    print(
        f"[pv26][merge] done: out_root={out_root} rows={int(report['num_rows']):,} "
        f"files={int(report['num_unique_files']):,}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
