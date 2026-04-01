from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from tools.od_bootstrap.common import resolve_path
from tools.od_bootstrap.sweep.scenario import ClassPolicy, load_class_policy


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
    min_precision_by_class_payload = _coerce_mapping(
        payload.get("min_precision_by_class"),
        field_name="search.min_precision_by_class",
    )
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
        min_precision_by_class={
            _coerce_str(class_name, field_name=f"search.min_precision_by_class[{index}].key"): _coerce_float(
                class_min_precision,
                field_name=f"search.min_precision_by_class[{index}].value",
            )
            for index, (class_name, class_min_precision) in enumerate(min_precision_by_class_payload.items())
        },
        score_thresholds=score_thresholds,
        nms_iou_thresholds=nms_iou_thresholds,
        min_box_sizes=min_box_sizes,
    )


def _load_dataset_config(data: Any, *, base_dir: Path, field_name: str) -> CalibrationDatasetConfig:
    payload = _coerce_mapping(data, field_name=field_name)
    return CalibrationDatasetConfig(
        root=resolve_path(_coerce_str(payload.get("root"), field_name=f"{field_name}.root"), base_dir=base_dir),
        source_dataset_key=_coerce_str(
            payload.get("source_dataset_key", Path(str(payload.get("root", ""))).name or "dataset"),
            field_name=f"{field_name}.source_dataset_key",
        ),
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


def _load_hard_negative_config(data: Any, *, base_dir: Path) -> HardNegativeConfig | None:
    if data is None:
        return None
    payload = _coerce_mapping(data, field_name="hard_negative")
    manifest_value = payload.get("manifest_path")
    manifest_path = (
        resolve_path(_coerce_str(manifest_value, field_name="hard_negative.manifest_path"), base_dir=base_dir)
        if manifest_value is not None
        else None
    )
    focus_classes = tuple(
        _coerce_str(item, field_name=f"hard_negative.focus_classes[{index}]")
        for index, item in enumerate(_coerce_sequence(payload.get("focus_classes"), field_name="hard_negative.focus_classes"))
    )
    return HardNegativeConfig(
        manifest_path=manifest_path,
        top_k_per_class=_coerce_int(payload.get("top_k_per_class", 25), field_name="hard_negative.top_k_per_class"),
        focus_classes=focus_classes,
    )


def load_calibration_scenario(path: str | Path) -> CalibrationScenario:
    scenario_path = Path(path).resolve()
    payload = yaml.safe_load(scenario_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise TypeError("scenario root must be a mapping")
    base_dir = scenario_path.parent
    policy_template_path = (
        resolve_path(_coerce_str(payload.get("policy_template_path"), field_name="policy_template_path"), base_dir=base_dir)
        if payload.get("policy_template_path") is not None
        else None
    )
    scenario = CalibrationScenario(
        run=_load_run_config(payload.get("run"), base_dir=base_dir),
        search=_load_search_config(payload.get("search")),
        teachers=tuple(
            _load_teacher_config(item, base_dir=base_dir, index=index)
            for index, item in enumerate(_coerce_sequence(payload.get("teachers"), field_name="teachers"))
        ),
        policy_template_path=policy_template_path,
        policy_template=load_class_policy(policy_template_path) if policy_template_path is not None else None,
        hard_negative=_load_hard_negative_config(payload.get("hard_negative"), base_dir=base_dir),
    )
    if not scenario.teachers:
        raise ValueError("teachers must not be empty")
    known_classes = {
        class_name
        for teacher in scenario.teachers
        for class_name in teacher.classes
    }
    invalid_min_precision_overrides = sorted(set(scenario.search.min_precision_by_class).difference(known_classes))
    if invalid_min_precision_overrides:
        raise ValueError(
            "search.min_precision_by_class contain unknown classes: "
            + ", ".join(invalid_min_precision_overrides)
        )
    if scenario.hard_negative is not None:
        invalid_focus = sorted(set(scenario.hard_negative.focus_classes).difference(known_classes))
        if invalid_focus:
            raise ValueError(f"hard_negative.focus_classes contain unknown classes: {', '.join(invalid_focus)}")
    return scenario
