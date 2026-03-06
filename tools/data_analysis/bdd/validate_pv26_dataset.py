#!/usr/bin/env python3
"""
Validate a converted PV26 dataset (manifest + files + basic label contracts).

Hard-fail on violations with non-zero exit code.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pv26.dataset.validation import validate_pv26_dataset


def _fmt_duration(seconds: Optional[float]) -> str:
    if seconds is None:
        return "계산중"
    sec = max(0, int(round(float(seconds))))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--out-root",
        type=Path,
        default=Path("datasets/pv26_v1"),
        help="Converted dataset root (default: datasets/pv26_v1)",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=max(1, min(12, (os.cpu_count() or 1) - 2 if (os.cpu_count() or 1) > 2 else (os.cpu_count() or 1))),
        help="Number of parallel workers for validation row checks",
    )
    return p


def main() -> int:
    args = build_argparser().parse_args()
    out_root: Path = args.out_root
    workers = max(1, int(args.workers))
    print(
        f"[pv26][로딩][검증] 시작: workers={workers} | 의미: 파일 존재/해상도/마스크 도메인/부분라벨 계약 전수 검증",
        flush=True,
    )

    def _progress_hook(
        done: int,
        total: int,
        err_count: int,
        warn_count: int,
        rate: float,
        eta_seconds: Optional[float],
    ) -> None:
        pct = 100.0 if total <= 0 else (100.0 * float(done) / float(total))
        print(
            f"[pv26][로딩][검증] {done:,}/{total:,} ({pct:6.2f}%) | "
            f"속도 {rate:,.2f} row/초 | 남은 시간 {_fmt_duration(eta_seconds)} | "
            f"누적 오류 {err_count:,} 경고 {warn_count:,} | "
            "의미: manifest 행별 파일 존재/해상도/마스크 값 도메인/부분라벨 계약을 검사 중",
            flush=True,
        )

    summary = validate_pv26_dataset(
        out_root,
        progress_hook=_progress_hook,
        progress_every=1000,
        workers=workers,
    )

    print(f"[pv26] validated: {out_root}")
    print(f"[pv26] rows: {summary.num_rows}")
    print(f"[pv26] errors: {len(summary.errors)}")
    print(f"[pv26] warnings: {len(summary.warnings)}")

    if summary.errors:
        print("\n--- Errors ---")
        for e in summary.errors[:2000]:
            print(e)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
