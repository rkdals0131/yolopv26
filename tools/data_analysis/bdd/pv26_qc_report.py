#!/usr/bin/env python3
"""
QC report CLI for converted PV26 datasets.

Reads:  <dataset_root>/meta/split_manifest.csv
Optionally writes: --out-json <path>

Dependencies: stdlib + numpy + pillow
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image


MANIFEST_REL = Path("meta") / "split_manifest.csv"


FLAG_COLUMNS: Tuple[str, ...] = (
    "has_det",
    "has_da",
    "has_rm_lane_marker",
    "has_rm_road_marker_non_lane",
    "has_rm_stop_line",
    "has_rm_lane_subclass",
    "has_semantic_id",
)

TAG_COLUMNS: Tuple[str, ...] = ("weather_tag", "time_tag", "scene_tag")


@dataclass(frozen=True)
class SegChannelSpec:
    name: str
    flag_key: str
    relpath_key: str
    is_semantic_id: bool = False


SEG_CHANNELS: Tuple[SegChannelSpec, ...] = (
    SegChannelSpec(name="da", flag_key="has_da", relpath_key="da_relpath"),
    SegChannelSpec(name="rm_lane_marker", flag_key="has_rm_lane_marker", relpath_key="rm_lane_marker_relpath"),
    SegChannelSpec(
        name="rm_road_marker_non_lane",
        flag_key="has_rm_road_marker_non_lane",
        relpath_key="rm_road_marker_non_lane_relpath",
    ),
    SegChannelSpec(name="rm_stop_line", flag_key="has_rm_stop_line", relpath_key="rm_stop_line_relpath"),
    SegChannelSpec(name="rm_lane_subclass", flag_key="has_rm_lane_subclass", relpath_key="rm_lane_subclass_relpath"),
    SegChannelSpec(name="semantic_id", flag_key="has_semantic_id", relpath_key="semantic_relpath", is_semantic_id=True),
)


def _parse_splits_arg(split_arg: Optional[str]) -> Optional[Tuple[str, ...]]:
    """
    Returns:
      None -> all splits
      ("train",) / ("val",) / ("test",) / ("train","val") ...
    """
    if split_arg is None:
        return None
    s = split_arg.strip().lower()
    if not s or s == "all":
        return None
    parts = tuple(p.strip() for p in s.split(",") if p.strip())
    bad = [p for p in parts if p not in {"train", "val", "test"}]
    if bad:
        raise ValueError(f"--split: invalid value(s) {bad}; expected train/val/test or all")
    return parts


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate a QC report for a converted PV26 dataset.")
    p.add_argument("--dataset-root", type=Path, required=True, help="Converted PV26 dataset root.")
    p.add_argument(
        "--split",
        type=str,
        default=None,
        help="Comma-separated split(s) to analyze: train,val,test (default: all).",
    )
    p.add_argument("--out-json", type=Path, default=None, help="Optional path to write JSON report.")
    return p


def _read_manifest_rows(manifest_csv: Path) -> List[Dict[str, str]]:
    with open(manifest_csv, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _safe_tag(v: str) -> str:
    v = (v or "").strip()
    return v if v else "<empty>"


def _open_mask_u8(mask_path: Path) -> np.ndarray:
    # Use PIL and force single-channel.
    with Image.open(mask_path) as im:
        im = im.convert("L")
        arr = np.array(im, dtype=np.uint8)
    if arr.ndim != 2:
        raise ValueError(f"mask is not 2D after L-conversion: shape={arr.shape}")
    return arr


def _mask_has_positive(mask: np.ndarray, *, is_semantic_id: bool) -> bool:
    """
    For binary masks: positive means any pixel == 1.
    For semantic_id: positive means any non-background pixel != 0.
    """
    if is_semantic_id:
        return bool(np.any(mask != 0))
    return bool(np.any(mask == 1))


def _pct(n: int, d: int) -> float:
    return 0.0 if d <= 0 else (100.0 * float(n) / float(d))


def compute_qc_report(dataset_root: Path, *, splits: Optional[Tuple[str, ...]]) -> Dict[str, Any]:
    manifest_csv = dataset_root / MANIFEST_REL
    if not manifest_csv.exists():
        raise FileNotFoundError(f"manifest not found: {manifest_csv}")

    rows_all = _read_manifest_rows(manifest_csv)
    if splits is None:
        rows = rows_all
        splits_used = ("train", "val", "test")
    else:
        rows = [r for r in rows_all if r.get("split", "").strip() in set(splits)]
        splits_used = tuple(sorted(set(splits), key=lambda x: {"train": 0, "val": 1, "test": 2}[x]))

    row_count_per_split = Counter()
    for r in rows:
        row_count_per_split[r.get("split", "<missing>")] += 1

    flag_counts: Dict[str, Dict[str, int]] = {}
    for k in FLAG_COLUMNS:
        c = Counter()
        for r in rows:
            c[(r.get(k, "") or "<missing>").strip()] += 1
        # Normalize output to {0,1,other...} for stability.
        flag_counts[k] = dict(sorted(c.items(), key=lambda kv: kv[0]))

    tag_counts: Dict[str, Dict[str, int]] = {}
    for k in TAG_COLUMNS:
        c = Counter()
        for r in rows:
            c[_safe_tag(r.get(k, ""))] += 1
        tag_counts[k] = dict(c.most_common())

    seg_nonempty: Dict[str, Any] = {}
    seg_read_errors: Dict[str, List[str]] = defaultdict(list)
    for spec in SEG_CHANNELS:
        supervised_rows: List[Dict[str, str]] = []
        for r in rows:
            if (r.get(spec.flag_key, "0") or "0").strip() != "1":
                continue
            rel = (r.get(spec.relpath_key, "") or "").strip()
            if not rel:
                continue
            supervised_rows.append(r)

        n_supervised = len(supervised_rows)
        n_nonempty = 0
        for r in supervised_rows:
            rel = (r.get(spec.relpath_key, "") or "").strip()
            p = dataset_root / rel
            try:
                m = _open_mask_u8(p)
                if _mask_has_positive(m, is_semantic_id=spec.is_semantic_id):
                    n_nonempty += 1
            except Exception as e:  # noqa: BLE001 - CLI tool: capture and report.
                seg_read_errors[spec.name].append(f"{rel} :: {type(e).__name__}: {e}")

        seg_nonempty[spec.name] = {
            "n_supervised": n_supervised,
            "n_nonempty": n_nonempty,
            "nonempty_ratio": (None if n_supervised == 0 else float(n_nonempty) / float(n_supervised)),
        }

    report: Dict[str, Any] = {
        "dataset_root": str(dataset_root),
        "splits": list(splits_used) if splits is None else list(splits_used),
        "num_rows": len(rows),
        "row_count_per_split": {k: int(v) for k, v in sorted(row_count_per_split.items())},
        "flag_distributions": flag_counts,
        "tag_distributions": tag_counts,
        "seg_nonempty": seg_nonempty,
        "seg_read_errors": {k: v[:50] for k, v in seg_read_errors.items()},
        "seg_read_error_counts": {k: len(v) for k, v in seg_read_errors.items()},
    }
    return report


def _print_report_human(report: Dict[str, Any]) -> None:
    ds = report["dataset_root"]
    print(f"[pv26-qc] dataset_root: {ds}")
    print(f"[pv26-qc] splits: {','.join(report['splits'])}  rows: {report['num_rows']}")

    print("\n[pv26-qc] rows per split:")
    rps = report["row_count_per_split"]
    for split in ("train", "val", "test"):
        if split in rps:
            print(f"  - {split}: {rps[split]}")

    print("\n[pv26-qc] label availability (counts):")
    for k, dist in report["flag_distributions"].items():
        total = sum(int(v) for v in dist.values())
        n1 = int(dist.get("1", 0))
        n0 = int(dist.get("0", 0))
        print(f"  - {k}: 1={n1} ({_pct(n1,total):.2f}%)  0={n0} ({_pct(n0,total):.2f}%)")

    print("\n[pv26-qc] tags (top 20):")
    for k, dist in report["tag_distributions"].items():
        items = list(dist.items())[:20]
        joined = ", ".join([f"{name}={cnt}" for name, cnt in items])
        print(f"  - {k}: {joined}")

    print("\n[pv26-qc] seg non-empty ratios (among supervised masks):")
    for name, s in report["seg_nonempty"].items():
        n_sup = int(s["n_supervised"])
        n_ne = int(s["n_nonempty"])
        ratio = s["nonempty_ratio"]
        ratio_s = "n/a" if ratio is None else f"{ratio:.4f}"
        print(f"  - {name}: nonempty={n_ne}/{n_sup}  ratio={ratio_s}")

    err_counts = report.get("seg_read_error_counts", {})
    total_err = sum(int(v) for v in err_counts.values())
    if total_err:
        print(f"\n[pv26-qc] WARNING: {total_err} mask read/parse errors (showing up to 3 per channel):")
        errors = report.get("seg_read_errors", {})
        for ch, n in sorted(err_counts.items(), key=lambda kv: (-kv[1], kv[0])):
            if n <= 0:
                continue
            print(f"  - {ch}: {n}")
            for line in errors.get(ch, [])[:3]:
                print(f"      {line}")


def main() -> int:
    args = build_argparser().parse_args()
    dataset_root: Path = args.dataset_root
    out_json: Optional[Path] = args.out_json
    try:
        splits = _parse_splits_arg(args.split)
    except ValueError as e:
        print(f"[pv26-qc] error: {e}")
        return 2

    try:
        report = compute_qc_report(dataset_root, splits=splits)
    except Exception as e:  # noqa: BLE001 - CLI
        print(f"[pv26-qc] error: {type(e).__name__}: {e}")
        return 2

    _print_report_human(report)

    if out_json is not None:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, sort_keys=True)
            f.write("\n")
        print(f"\n[pv26-qc] wrote: {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
