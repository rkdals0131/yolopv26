from __future__ import annotations

import csv
from dataclasses import dataclass, fields
from typing import Dict, Iterable, List, Optional


MANIFEST_COLUMNS: List[str] = [
    "sample_id",
    "split",
    "source",
    "sequence",
    "frame",
    "camera_id",
    "timestamp_ns",
    "has_det",
    "has_da",
    "has_rm_lane_marker",
    "has_rm_road_marker_non_lane",
    "has_rm_stop_line",
    "has_semantic_id",
    "det_label_scope",
    "det_annotated_class_ids",
    "image_relpath",
    "det_relpath",
    "da_relpath",
    "rm_lane_marker_relpath",
    "rm_road_marker_non_lane_relpath",
    "rm_stop_line_relpath",
    "semantic_relpath",
    "width",
    "height",
    "weather_tag",
    "time_tag",
    "scene_tag",
    "source_group_key",
]


@dataclass(frozen=True)
class ManifestRow:
    sample_id: str
    split: str
    source: str
    sequence: str
    frame: str
    camera_id: str
    timestamp_ns: str
    has_det: int
    has_da: int
    has_rm_lane_marker: int
    has_rm_road_marker_non_lane: int
    has_rm_stop_line: int
    has_semantic_id: int
    det_label_scope: str
    det_annotated_class_ids: str
    image_relpath: str
    det_relpath: str
    da_relpath: str
    rm_lane_marker_relpath: str
    rm_road_marker_non_lane_relpath: str
    rm_stop_line_relpath: str
    semantic_relpath: str
    width: int
    height: int
    weather_tag: str
    time_tag: str
    scene_tag: str
    source_group_key: str

    def as_csv_row(self) -> Dict[str, str]:
        out: Dict[str, str] = {}
        for f in fields(self):
            v = getattr(self, f.name)
            out[f.name] = "" if v is None else str(v)
        return out


def write_manifest_csv(path, rows: Iterable[ManifestRow]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow(r.as_csv_row())


def read_manifest_csv(path) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def validate_manifest_row_basic(row: Dict[str, str]) -> List[str]:
    """
    Light schema checks that can run without touching the filesystem.
    Returns a list of error strings (empty when ok).
    """
    errs: List[str] = []
    for k in MANIFEST_COLUMNS:
        if k not in row:
            errs.append(f"missing_column:{k}")

    split = row.get("split", "")
    if split not in {"train", "val", "test"}:
        errs.append(f"invalid_split:{split}")

    for k in [
        "has_det",
        "has_da",
        "has_rm_lane_marker",
        "has_rm_road_marker_non_lane",
        "has_rm_stop_line",
        "has_semantic_id",
    ]:
        v = row.get(k, "")
        if v not in {"0", "1"}:
            errs.append(f"invalid_flag:{k}={v}")

    scope = row.get("det_label_scope", "")
    if scope not in {"full", "subset", "none"}:
        errs.append(f"invalid_det_label_scope:{scope}")

    if scope == "subset" and not row.get("det_annotated_class_ids", "").strip():
        errs.append("subset_missing_det_annotated_class_ids")

    if scope in {"full", "none"} and row.get("det_annotated_class_ids", "").strip():
        errs.append("det_annotated_class_ids_must_be_empty_for_full_or_none")

    return errs

