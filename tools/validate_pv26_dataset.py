#!/usr/bin/env python3
"""
Validate a converted PV26 dataset (manifest + files + basic label contracts).

Hard-fail on violations with non-zero exit code.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pv26.validate_dataset import validate_pv26_dataset


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--out-root",
        type=Path,
        default=Path("datasets/pv26_v1"),
        help="Converted dataset root (default: datasets/pv26_v1)",
    )
    return p


def main() -> int:
    args = build_argparser().parse_args()
    out_root: Path = args.out_root
    summary = validate_pv26_dataset(out_root)

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
