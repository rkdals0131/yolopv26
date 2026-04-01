from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CheckpointEvalRunConfig:
    output_root: Path
    exist_ok: bool = True


@dataclass(frozen=True)
class CheckpointEvalDatasetConfig:
    root: Path
    image_dir: str = "images"
    label_dir: str = "labels"
    split: str = "val"
    sample_limit: int = 8


@dataclass(frozen=True)
class CheckpointEvalModelConfig:
    checkpoint_path: Path
    class_names: tuple[str, ...] = ()
    model_size: str = "n"


@dataclass(frozen=True)
class CheckpointEvalParams:
    imgsz: int = 640
    batch: int = 1
    device: str = "cuda:0"
    conf: float = 0.25
    iou: float = 0.7
    predict: bool = True
    val: bool = True
    save_conf: bool = False
    verbose: bool = False


@dataclass(frozen=True)
class CheckpointEvalScenario:
    teacher_name: str
    run: CheckpointEvalRunConfig
    dataset: CheckpointEvalDatasetConfig
    model: CheckpointEvalModelConfig
    eval: CheckpointEvalParams
