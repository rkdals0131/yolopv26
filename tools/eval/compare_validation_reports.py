#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional


COMMON_METRICS = ("det_map50", "da.iou", "da.f1", "lane.iou", "lane.f1")
PV26_EXTRA_METRICS = (
    "rm_road_marker_non_lane.iou",
    "rm_stop_line.iou",
    "lane_subclass_miou4",
    "lane_subclass_miou4_present",
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _get_path(obj: dict[str, Any], path: str) -> Optional[float]:
    cur: Any = obj
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    if cur is None:
        return None
    return float(cur)


def _fmt(val: Optional[float]) -> str:
    return "n/a" if val is None else f"{val:.4f}"


def main() -> int:
    p = argparse.ArgumentParser(description="Compare PV2 and PV26 validation JSON reports.")
    p.add_argument("--pv2-json", type=Path, required=True)
    p.add_argument("--pv26-json", type=Path, required=True)
    args = p.parse_args()

    pv2 = _load_json(args.pv2_json)
    pv26 = _load_json(args.pv26_json)

    print(f"[compare] pv2={args.pv2_json}")
    print(f"[compare] pv26={args.pv26_json}")
    print("[compare] common_metrics")
    for key in COMMON_METRICS:
        pv2_val = _get_path(pv2, key)
        pv26_val = _get_path(pv26, key)
        delta = None if pv2_val is None or pv26_val is None else pv26_val - pv2_val
        print(f"[compare] {key}: pv2={_fmt(pv2_val)} pv26={_fmt(pv26_val)} delta={_fmt(delta)}")

    print("[compare] pv26_extra_metrics")
    for key in PV26_EXTRA_METRICS:
        print(f"[compare] {key}: pv26={_fmt(_get_path(pv26, key))}")

    lane_groups = pv26.get("lane_subclass_groups", {})
    if isinstance(lane_groups, dict) and lane_groups:
        print("[compare] pv26_lane_subclass_groups")
        for name in sorted(lane_groups.keys()):
            iou = _get_path(pv26, f"lane_subclass_groups.{name}.iou")
            f1 = _get_path(pv26, f"lane_subclass_groups.{name}.f1")
            print(f"[compare] lane_subclass_groups.{name}: iou={_fmt(iou)} f1={_fmt(f1)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
