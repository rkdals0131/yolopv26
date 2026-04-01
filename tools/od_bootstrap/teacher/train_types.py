from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TeacherRunConfig:
    output_root: Path
    exist_ok: bool = True


@dataclass(frozen=True)
class TeacherDatasetConfig:
    root: Path
    image_dir: str = "images"
    label_dir: str = "labels"
    train_split: str = "train"
    val_split: str = "val"


@dataclass(frozen=True)
class TeacherModelConfig:
    model_size: str = "n"
    weights: str | None = None
    class_names: tuple[str, ...] = ()


@dataclass(frozen=True)
class TeacherTrainParams:
    epochs: int = 100
    imgsz: int = 640
    batch: int = 16
    device: str = "cuda:0"
    workers: int = 8
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int | None = 4
    patience: int = 50
    cache: bool = False
    amp: bool = True
    optimizer: str = "auto"
    seed: int = 0
    resume: bool = False
    val: bool = True
    save_period: int = 10
    log_every_n_steps: int = 20
    profile_window: int = 20
    profile_device_sync: bool = True


@dataclass(frozen=True)
class TeacherTrainScenario:
    teacher_name: str
    run: TeacherRunConfig
    dataset: TeacherDatasetConfig
    model: TeacherModelConfig
    train: TeacherTrainParams
