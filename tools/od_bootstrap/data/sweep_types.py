from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


REQUIRED_TEACHER_ORDER = ("mobility", "signal", "obstacle")


@dataclass(frozen=True)
class RunConfig:
    output_root: Path
    execution_mode: str = "model-centric"
    device: str = "cuda:0"
    imgsz: int = 640
    batch_size: int = 8
    predict_conf: float = 0.001
    predict_iou: float = 0.99


@dataclass(frozen=True)
class ImageListConfig:
    manifest_path: Path


@dataclass(frozen=True)
class MaterializationConfig:
    output_root: Path
    copy_images: bool = False


@dataclass(frozen=True)
class TeacherConfig:
    name: str
    base_model: str
    checkpoint_path: Path
    model_version: str
    classes: tuple[str, ...]


@dataclass(frozen=True)
class ClassPolicy:
    score_threshold: float
    nms_iou_threshold: float
    min_box_size: int
    allowed_source_datasets: tuple[str, ...] = ()
    suppress_with_classes: tuple[str, ...] = ()
    cross_class_iou_threshold: float | None = None
    center_x_range: tuple[float, float] | None = None
    center_y_range: tuple[float, float] | None = None
    aspect_ratio_range: tuple[float, float] | None = None
    area_ratio_range: tuple[float, float] | None = None


@dataclass(frozen=True)
class BootstrapSweepScenario:
    run: RunConfig
    image_list: ImageListConfig
    materialization: MaterializationConfig
    teachers: tuple[TeacherConfig, ...]
    class_policy_path: Path
    class_policy: dict[str, ClassPolicy]
