from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from tools.od_bootstrap.common import resolve_path


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


def _run_config_from_mapping(payload: dict[str, Any], *, base_dir: Path) -> CheckpointEvalRunConfig:
    data = _coerce_mapping(payload, field_name="run")
    return CheckpointEvalRunConfig(
        output_root=resolve_path(data.get("output_root", "../runs/od_bootstrap/eval"), base_dir=base_dir),
        exist_ok=_coerce_bool(data.get("exist_ok", True), field_name="run.exist_ok"),
    )


def _dataset_config_from_mapping(payload: dict[str, Any], *, base_dir: Path) -> CheckpointEvalDatasetConfig:
    data = _coerce_mapping(payload, field_name="dataset")
    return CheckpointEvalDatasetConfig(
        root=resolve_path(data.get("root"), base_dir=base_dir),
        image_dir=_coerce_str(data.get("image_dir", "images"), field_name="dataset.image_dir"),
        label_dir=_coerce_str(data.get("label_dir", "labels"), field_name="dataset.label_dir"),
        split=_coerce_str(data.get("split", "val"), field_name="dataset.split"),
        sample_limit=_coerce_int(data.get("sample_limit", 8), field_name="dataset.sample_limit"),
    )


def _eval_config_from_mapping(payload: dict[str, Any]) -> CheckpointEvalParams:
    data = _coerce_mapping(payload, field_name="eval")
    defaults = CheckpointEvalParams()
    return CheckpointEvalParams(
        imgsz=_coerce_int(data.get("imgsz", defaults.imgsz), field_name="eval.imgsz"),
        batch=_coerce_int(data.get("batch", defaults.batch), field_name="eval.batch"),
        device=_coerce_str(data.get("device", defaults.device), field_name="eval.device"),
        conf=float(data.get("conf", defaults.conf)),
        iou=float(data.get("iou", defaults.iou)),
        predict=_coerce_bool(data.get("predict", defaults.predict), field_name="eval.predict"),
        val=_coerce_bool(data.get("val", defaults.val), field_name="eval.val"),
        save_conf=_coerce_bool(data.get("save_conf", defaults.save_conf), field_name="eval.save_conf"),
        verbose=_coerce_bool(data.get("verbose", defaults.verbose), field_name="eval.verbose"),
    )


def load_teacher_checkpoint_eval_scenario(path: str | Path) -> CheckpointEvalScenario:
    scenario_path = Path(path).resolve()
    payload = yaml.safe_load(scenario_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise TypeError("scenario root must be a mapping")
    base_dir = scenario_path.parent
    model_payload = _coerce_mapping(payload.get("model"), field_name="model")

    scenario = CheckpointEvalScenario(
        teacher_name=_coerce_str(payload.get("teacher_name"), field_name="teacher_name"),
        run=_run_config_from_mapping(payload.get("run"), base_dir=base_dir),
        dataset=_dataset_config_from_mapping(payload.get("dataset"), base_dir=base_dir),
        model=CheckpointEvalModelConfig(
            checkpoint_path=resolve_path(
                _coerce_str(model_payload.get("checkpoint_path"), field_name="model.checkpoint_path"),
                base_dir=base_dir,
            ),
            class_names=tuple(
                _coerce_str(item, field_name="model.class_names[]")
                for item in _coerce_sequence(model_payload.get("class_names"), field_name="model.class_names")
            ),
            model_size=_coerce_str(model_payload.get("model_size", "n"), field_name="model.model_size"),
        ),
        eval=_eval_config_from_mapping(payload.get("eval")),
    )
    if not scenario.model.class_names:
        raise ValueError("model.class_names must not be empty")
    return scenario
