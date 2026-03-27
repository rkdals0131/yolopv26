from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from tools.od_bootstrap.common import resolve_path


REQUIRED_TEACHER_ORDER = ("mobility", "signal", "obstacle")


@dataclass(frozen=True)
class RunConfig:
    output_root: Path
    execution_mode: str = "model-centric"
    dry_run: bool = False
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


@dataclass(frozen=True)
class BootstrapSweepScenario:
    run: RunConfig
    image_list: ImageListConfig
    materialization: MaterializationConfig
    teachers: tuple[TeacherConfig, ...]
    class_policy_path: Path
    class_policy: dict[str, ClassPolicy]


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


def _coerce_float(value: Any, *, field_name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be a float")
    return float(value)


def _load_run_config(data: Any, *, base_dir: Path) -> RunConfig:
    payload = _coerce_mapping(data, field_name="run")
    return RunConfig(
        output_root=resolve_path(payload.get("output_root", "../runs/od_bootstrap"), base_dir=base_dir),
        execution_mode=_coerce_str(payload.get("execution_mode", "model-centric"), field_name="run.execution_mode"),
        dry_run=_coerce_bool(payload.get("dry_run", False), field_name="run.dry_run"),
        device=_coerce_str(payload.get("device", "cuda:0"), field_name="run.device"),
        imgsz=_coerce_int(payload.get("imgsz", 640), field_name="run.imgsz"),
        batch_size=_coerce_int(payload.get("batch_size", 8), field_name="run.batch_size"),
        predict_conf=_coerce_float(payload.get("predict_conf", 0.001), field_name="run.predict_conf"),
        predict_iou=_coerce_float(payload.get("predict_iou", 0.99), field_name="run.predict_iou"),
    )


def _load_image_list_config(data: Any, *, base_dir: Path) -> ImageListConfig:
    payload = _coerce_mapping(data, field_name="image_list")
    return ImageListConfig(
        manifest_path=resolve_path(
            _coerce_str(payload.get("manifest_path"), field_name="image_list.manifest_path"),
            base_dir=base_dir,
        )
    )


def _load_materialization_config(data: Any, *, base_dir: Path) -> MaterializationConfig:
    payload = _coerce_mapping(data, field_name="materialization")
    return MaterializationConfig(
        output_root=resolve_path(
            payload.get("output_root", "../seg_dataset/pv26_od_bootstrap/exhaustive_od"),
            base_dir=base_dir,
        ),
        copy_images=_coerce_bool(payload.get("copy_images", False), field_name="materialization.copy_images"),
    )


def _load_teacher_config(data: Any, *, base_dir: Path, index: int) -> TeacherConfig:
    payload = _coerce_mapping(data, field_name=f"teachers[{index}]")
    classes = tuple(
        _coerce_str(item, field_name=f"teachers[{index}].classes[{class_index}]")
        for class_index, item in enumerate(_coerce_sequence(payload.get("classes"), field_name=f"teachers[{index}].classes"))
    )
    if not classes:
        raise ValueError(f"teachers[{index}].classes must not be empty")
    return TeacherConfig(
        name=_coerce_str(payload.get("name"), field_name=f"teachers[{index}].name"),
        base_model=_coerce_str(payload.get("base_model"), field_name=f"teachers[{index}].base_model"),
        checkpoint_path=resolve_path(
            _coerce_str(payload.get("checkpoint_path"), field_name=f"teachers[{index}].checkpoint_path"),
            base_dir=base_dir,
        ),
        model_version=_coerce_str(payload.get("model_version"), field_name=f"teachers[{index}].model_version"),
        classes=classes,
    )


def _load_class_policy_mapping(data: Any) -> dict[str, ClassPolicy]:
    payload = _coerce_mapping(data, field_name="class_policy")
    resolved: dict[str, ClassPolicy] = {}
    for class_name, raw_policy in payload.items():
        class_key = _coerce_str(class_name, field_name="class_policy.<key>")
        policy = _coerce_mapping(raw_policy, field_name=f"class_policy.{class_key}")
        resolved[class_key] = ClassPolicy(
            score_threshold=_coerce_float(policy.get("score_threshold"), field_name=f"class_policy.{class_key}.score_threshold"),
            nms_iou_threshold=_coerce_float(
                policy.get("nms_iou_threshold"),
                field_name=f"class_policy.{class_key}.nms_iou_threshold",
            ),
            min_box_size=_coerce_int(policy.get("min_box_size"), field_name=f"class_policy.{class_key}.min_box_size"),
        )
    return resolved


def load_class_policy(path: str | Path) -> dict[str, ClassPolicy]:
    policy_path = Path(path).resolve()
    payload = yaml.safe_load(policy_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise TypeError("class policy file root must be a mapping")
    return _load_class_policy_mapping(payload)


def _validate_scenario(scenario: BootstrapSweepScenario) -> None:
    if scenario.run.execution_mode != "model-centric":
        raise ValueError("run.execution_mode must be 'model-centric'")

    teacher_names = tuple(teacher.name for teacher in scenario.teachers)
    if teacher_names != REQUIRED_TEACHER_ORDER:
        raise ValueError(f"teachers must be ordered as {REQUIRED_TEACHER_ORDER}")

    seen_classes: dict[str, str] = {}
    required_policy: set[str] = set()
    for teacher in scenario.teachers:
        for class_name in teacher.classes:
            if class_name in seen_classes:
                raise ValueError(f"class '{class_name}' is assigned to both {seen_classes[class_name]} and {teacher.name}")
            seen_classes[class_name] = teacher.name
            required_policy.add(class_name)

    missing_policy = sorted(required_policy.difference(scenario.class_policy))
    if missing_policy:
        raise ValueError(f"class_policy missing entries for: {', '.join(missing_policy)}")


def load_sweep_scenario(path: str | Path) -> BootstrapSweepScenario:
    scenario_path = Path(path).resolve()
    payload = yaml.safe_load(scenario_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise TypeError("scenario root must be a mapping")
    base_dir = scenario_path.parent
    class_policy_path = resolve_path(
        _coerce_str(payload.get("class_policy_path"), field_name="class_policy_path"),
        base_dir=base_dir,
    )
    scenario = BootstrapSweepScenario(
        run=_load_run_config(payload.get("run"), base_dir=base_dir),
        image_list=_load_image_list_config(payload.get("image_list"), base_dir=base_dir),
        materialization=_load_materialization_config(payload.get("materialization"), base_dir=base_dir),
        teachers=tuple(
            _load_teacher_config(item, base_dir=base_dir, index=index)
            for index, item in enumerate(_coerce_sequence(payload.get("teachers"), field_name="teachers"))
        ),
        class_policy_path=class_policy_path,
        class_policy=load_class_policy(class_policy_path),
    )
    _validate_scenario(scenario)
    return scenario
