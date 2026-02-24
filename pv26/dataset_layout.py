from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List


SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class Pv26Layout:
    out_root: Path

    def images_dir(self, split: str) -> Path:
        return self.out_root / "images" / split

    def labels_det_dir(self, split: str) -> Path:
        return self.out_root / "labels_det" / split

    def labels_seg_da_dir(self, split: str) -> Path:
        return self.out_root / "labels_seg_da" / split

    def labels_seg_rm_lane_marker_dir(self, split: str) -> Path:
        return self.out_root / "labels_seg_rm_lane_marker" / split

    def labels_seg_rm_road_marker_non_lane_dir(self, split: str) -> Path:
        return self.out_root / "labels_seg_rm_road_marker_non_lane" / split

    def labels_seg_rm_stop_line_dir(self, split: str) -> Path:
        return self.out_root / "labels_seg_rm_stop_line" / split

    def labels_semantic_id_dir(self, split: str) -> Path:
        return self.out_root / "labels_semantic_id" / split

    def meta_dir(self) -> Path:
        return self.out_root / "meta"

    def class_map_path(self) -> Path:
        return self.meta_dir() / "class_map.yaml"

    def manifest_path(self) -> Path:
        return self.meta_dir() / "split_manifest.csv"

    def report_path(self) -> Path:
        return self.meta_dir() / "conversion_report.json"

    def source_stats_path(self) -> Path:
        return self.meta_dir() / "source_stats.csv"

    def checksums_path(self) -> Path:
        return self.meta_dir() / "checksums.sha256"

    def ensure_dirs(self) -> None:
        for split in SPLITS:
            self.images_dir(split).mkdir(parents=True, exist_ok=True)
            self.labels_det_dir(split).mkdir(parents=True, exist_ok=True)
            self.labels_seg_da_dir(split).mkdir(parents=True, exist_ok=True)
            self.labels_seg_rm_lane_marker_dir(split).mkdir(parents=True, exist_ok=True)
            self.labels_seg_rm_road_marker_non_lane_dir(split).mkdir(parents=True, exist_ok=True)
            self.labels_seg_rm_stop_line_dir(split).mkdir(parents=True, exist_ok=True)
            self.labels_semantic_id_dir(split).mkdir(parents=True, exist_ok=True)
        self.meta_dir().mkdir(parents=True, exist_ok=True)

