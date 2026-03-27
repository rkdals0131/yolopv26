from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from tools.od_bootstrap.common import resolve_path


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
    patience: int = 50
    cache: bool = False
    amp: bool = True
    optimizer: str = "auto"
    seed: int = 0
    resume: bool = False
    val: bool = True
    save_period: int = 10


@dataclass(frozen=True)
class TeacherTrainScenario:
    teacher_name: str
    run: TeacherRunConfig
    dataset: TeacherDatasetConfig
    model: TeacherModelConfig
    train: TeacherTrainParams

def _coerce_mapping(value: Any, *, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError(f"{field_name} must be a mapping")
    return dict(value)


def _coerce_sequence(value: Any, *, field_name: str) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise TypeError(f"{field_name} must be a sequence")
    return list(value)


def _coerce_str(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be empty")
    return normalized


def _coerce_bool(value: Any, *, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    raise TypeError(f"{field_name} must be a boolean")


def _coerce_int(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be an integer")
    return int(value)


def _run_config_from_mapping(payload: dict[str, Any], *, base_dir: Path) -> TeacherRunConfig:
    data = _coerce_mapping(payload, field_name="run")
    return TeacherRunConfig(
        output_root=resolve_path(data.get("output_root", "../runs/od_bootstrap/train"), base_dir=base_dir),
        exist_ok=_coerce_bool(data.get("exist_ok", True), field_name="run.exist_ok"),
    )


def _dataset_config_from_mapping(payload: dict[str, Any], *, base_dir: Path) -> TeacherDatasetConfig:
    data = _coerce_mapping(payload, field_name="dataset")
    return TeacherDatasetConfig(
        root=resolve_path(data.get("root"), base_dir=base_dir),
        image_dir=_coerce_str(data.get("image_dir", "images"), field_name="dataset.image_dir"),
        label_dir=_coerce_str(data.get("label_dir", "labels"), field_name="dataset.label_dir"),
        train_split=_coerce_str(data.get("train_split", "train"), field_name="dataset.train_split"),
        val_split=_coerce_str(data.get("val_split", "val"), field_name="dataset.val_split"),
    )


def _model_config_from_mapping(payload: dict[str, Any]) -> TeacherModelConfig:
    data = _coerce_mapping(payload, field_name="model")
    class_names = tuple(
        _coerce_str(item, field_name="model.class_names[]")
        for item in _coerce_sequence(data.get("class_names"), field_name="model.class_names")
    )
    return TeacherModelConfig(
        model_size=_coerce_str(data.get("model_size", "n"), field_name="model.model_size"),
        weights=(
            _coerce_str(data.get("weights"), field_name="model.weights")
            if data.get("weights") is not None
            else None
        ),
        class_names=class_names,
    )


def _train_config_from_mapping(payload: dict[str, Any]) -> TeacherTrainParams:
    data = _coerce_mapping(payload, field_name="train")
    defaults = TeacherTrainParams()
    return TeacherTrainParams(
        epochs=_coerce_int(data.get("epochs", defaults.epochs), field_name="train.epochs"),
        imgsz=_coerce_int(data.get("imgsz", defaults.imgsz), field_name="train.imgsz"),
        batch=_coerce_int(data.get("batch", defaults.batch), field_name="train.batch"),
        device=_coerce_str(data.get("device", defaults.device), field_name="train.device"),
        workers=_coerce_int(data.get("workers", defaults.workers), field_name="train.workers"),
        patience=_coerce_int(data.get("patience", defaults.patience), field_name="train.patience"),
        cache=_coerce_bool(data.get("cache", defaults.cache), field_name="train.cache"),
        amp=_coerce_bool(data.get("amp", defaults.amp), field_name="train.amp"),
        optimizer=_coerce_str(data.get("optimizer", defaults.optimizer), field_name="train.optimizer"),
        seed=_coerce_int(data.get("seed", defaults.seed), field_name="train.seed"),
        resume=_coerce_bool(data.get("resume", defaults.resume), field_name="train.resume"),
        val=_coerce_bool(data.get("val", defaults.val), field_name="train.val"),
        save_period=_coerce_int(data.get("save_period", defaults.save_period), field_name="train.save_period"),
    )


def load_teacher_train_scenario(path: str | Path) -> TeacherTrainScenario:
    scenario_path = Path(path).resolve()
    payload = yaml.safe_load(scenario_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise TypeError("scenario root must be a mapping")
    base_dir = scenario_path.parent

    teacher_name = _coerce_str(payload.get("teacher_name"), field_name="teacher_name")
    scenario = TeacherTrainScenario(
        teacher_name=teacher_name,
        run=_run_config_from_mapping(payload.get("run"), base_dir=base_dir),
        dataset=_dataset_config_from_mapping(payload.get("dataset"), base_dir=base_dir),
        model=_model_config_from_mapping(payload.get("model")),
        train=_train_config_from_mapping(payload.get("train")),
    )
    if not scenario.model.class_names:
        raise ValueError("model.class_names must not be empty")
    return scenario
