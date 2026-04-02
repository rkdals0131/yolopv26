from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from tools.od_bootstrap.build.sweep_types import ClassPolicy


@dataclass(frozen=True)
class CalibrationRunConfig:
    output_root: Path
    device: str = "cuda:0"
    imgsz: int = 640
    batch_size: int = 8
    predict_conf: float = 0.001
    predict_iou: float = 0.99


@dataclass(frozen=True)
class CalibrationSearchConfig:
    match_iou: float = 0.5
    min_precision: float = 0.90
    min_precision_by_class: dict[str, float] = field(default_factory=dict)
    score_thresholds: tuple[float, ...] = ()
    nms_iou_thresholds: tuple[float, ...] = ()
    min_box_sizes: tuple[int, ...] = ()


@dataclass(frozen=True)
class CalibrationDatasetConfig:
    root: Path
    source_dataset_key: str = ""
    image_dir: str = "images"
    label_dir: str = "labels"
    split: str = "val"


@dataclass(frozen=True)
class CalibrationTeacherConfig:
    name: str
    checkpoint_path: Path
    model_version: str
    dataset: CalibrationDatasetConfig
    classes: tuple[str, ...]
    imgsz: int | None = None


@dataclass(frozen=True)
class HardNegativeConfig:
    manifest_path: Path | None = None
    top_k_per_class: int = 25
    focus_classes: tuple[str, ...] = ()


@dataclass(frozen=True)
class CalibrationScenario:
    run: CalibrationRunConfig
    search: CalibrationSearchConfig
    teachers: tuple[CalibrationTeacherConfig, ...]
    policy_template_path: Path | None = None
    policy_template: dict[str, ClassPolicy] | None = None
    hard_negative: HardNegativeConfig | None = None
