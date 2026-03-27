from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from tools.od_bootstrap.common import resolve_path


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
    score_thresholds: tuple[float, ...] = ()
    nms_iou_thresholds: tuple[float, ...] = ()
    min_box_sizes: tuple[int, ...] = ()


@dataclass(frozen=True)
class CalibrationDatasetConfig:
    root: Path
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


@dataclass(frozen=True)
class CalibrationScenario:
    run: CalibrationRunConfig
    search: CalibrationSearchConfig
    teachers: tuple[CalibrationTeacherConfig, ...]


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


def _coerce_int(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be an integer")
    return int(value)


def _coerce_float(value: Any, *, field_name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be a float")
    return float(value)


def _load_run_config(data: Any, *, base_dir: Path) -> CalibrationRunConfig:
    payload = _coerce_mapping(data, field_name="run")
    return CalibrationRunConfig(
        output_root=resolve_path(payload.get("output_root", "../runs/od_bootstrap/calibration/default"), base_dir=base_dir),
        device=_coerce_str(payload.get("device", "cuda:0"), field_name="run.device"),
        imgsz=_coerce_int(payload.get("imgsz", 640), field_name="run.imgsz"),
        batch_size=_coerce_int(payload.get("batch_size", 8), field_name="run.batch_size"),
        predict_conf=_coerce_float(payload.get("predict_conf", 0.001), field_name="run.predict_conf"),
        predict_iou=_coerce_float(payload.get("predict_iou", 0.99), field_name="run.predict_iou"),
    )


def _load_search_config(data: Any) -> CalibrationSearchConfig:
    payload = _coerce_mapping(data, field_name="search")
    score_thresholds = tuple(
        _coerce_float(item, field_name=f"search.score_thresholds[{index}]")
        for index, item in enumerate(_coerce_sequence(payload.get("score_thresholds"), field_name="search.score_thresholds"))
    )
    nms_iou_thresholds = tuple(
        _coerce_float(item, field_name=f"search.nms_iou_thresholds[{index}]")
        for index, item in enumerate(_coerce_sequence(payload.get("nms_iou_thresholds"), field_name="search.nms_iou_thresholds"))
    )
    min_box_sizes = tuple(
        _coerce_int(item, field_name=f"search.min_box_sizes[{index}]")
        for index, item in enumerate(_coerce_sequence(payload.get("min_box_sizes"), field_name="search.min_box_sizes"))
    )
    if not score_thresholds or not nms_iou_thresholds or not min_box_sizes:
        raise ValueError("search grid must define score_thresholds, nms_iou_thresholds, and min_box_sizes")
    return CalibrationSearchConfig(
        match_iou=_coerce_float(payload.get("match_iou", 0.5), field_name="search.match_iou"),
        min_precision=_coerce_float(payload.get("min_precision", 0.90), field_name="search.min_precision"),
        score_thresholds=score_thresholds,
        nms_iou_thresholds=nms_iou_thresholds,
        min_box_sizes=min_box_sizes,
    )


def _load_dataset_config(data: Any, *, base_dir: Path, field_name: str) -> CalibrationDatasetConfig:
    payload = _coerce_mapping(data, field_name=field_name)
    return CalibrationDatasetConfig(
        root=resolve_path(_coerce_str(payload.get("root"), field_name=f"{field_name}.root"), base_dir=base_dir),
        image_dir=_coerce_str(payload.get("image_dir", "images"), field_name=f"{field_name}.image_dir"),
        label_dir=_coerce_str(payload.get("label_dir", "labels"), field_name=f"{field_name}.label_dir"),
        split=_coerce_str(payload.get("split", "val"), field_name=f"{field_name}.split"),
    )


def _load_teacher_config(data: Any, *, base_dir: Path, index: int) -> CalibrationTeacherConfig:
    payload = _coerce_mapping(data, field_name=f"teachers[{index}]")
    classes = tuple(
        _coerce_str(item, field_name=f"teachers[{index}].classes[{class_index}]")
        for class_index, item in enumerate(_coerce_sequence(payload.get("classes"), field_name=f"teachers[{index}].classes"))
    )
    if not classes:
        raise ValueError(f"teachers[{index}].classes must not be empty")
    return CalibrationTeacherConfig(
        name=_coerce_str(payload.get("name"), field_name=f"teachers[{index}].name"),
        checkpoint_path=resolve_path(
            _coerce_str(payload.get("checkpoint_path"), field_name=f"teachers[{index}].checkpoint_path"),
            base_dir=base_dir,
        ),
        model_version=_coerce_str(payload.get("model_version"), field_name=f"teachers[{index}].model_version"),
        dataset=_load_dataset_config(payload.get("dataset"), base_dir=base_dir, field_name=f"teachers[{index}].dataset"),
        classes=classes,
    )


def load_calibration_scenario(path: str | Path) -> CalibrationScenario:
    scenario_path = Path(path).resolve()
    payload = yaml.safe_load(scenario_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise TypeError("scenario root must be a mapping")
    base_dir = scenario_path.parent
    scenario = CalibrationScenario(
        run=_load_run_config(payload.get("run"), base_dir=base_dir),
        search=_load_search_config(payload.get("search")),
        teachers=tuple(
            _load_teacher_config(item, base_dir=base_dir, index=index)
            for index, item in enumerate(_coerce_sequence(payload.get("teachers"), field_name="teachers"))
        ),
    )
    if not scenario.teachers:
        raise ValueError("teachers must not be empty")
    return scenario
