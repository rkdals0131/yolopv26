from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


TeacherKind = Literal["od", "signal_attr"]
TeacherStage = Literal["dataset", "train", "eval"]


@dataclass(frozen=True)
class TeacherDefinition:
    name: str
    display_name: str
    kind: TeacherKind
    train_action_key: str
    eval_action_key: str
    supports_calibration: bool = False
    supports_torchscript_export: bool = False


TEACHER_DEFINITIONS: dict[str, TeacherDefinition] = {
    "mobility": TeacherDefinition(
        name="mobility",
        display_name="Mobility",
        kind="od",
        train_action_key="3",
        eval_action_key="6",
        supports_calibration=True,
        supports_torchscript_export=True,
    ),
    "signal": TeacherDefinition(
        name="signal",
        display_name="Signal",
        kind="od",
        train_action_key="4",
        eval_action_key="7",
        supports_calibration=True,
        supports_torchscript_export=True,
    ),
    "signal_attr": TeacherDefinition(
        name="signal_attr",
        display_name="Signal attr",
        kind="signal_attr",
        train_action_key="4A",
        eval_action_key="7A",
    ),
    "obstacle": TeacherDefinition(
        name="obstacle",
        display_name="Obstacle",
        kind="od",
        train_action_key="5",
        eval_action_key="8",
        supports_calibration=True,
        supports_torchscript_export=True,
    ),
}

ALL_TEACHER_NAMES = tuple(TEACHER_DEFINITIONS)
OD_TEACHER_NAMES = tuple(
    name for name, definition in TEACHER_DEFINITIONS.items() if definition.kind == "od"
)
SIGNAL_ATTR_TEACHER_NAME = "signal_attr"


def teacher_definition(teacher_name: str) -> TeacherDefinition:
    try:
        return TEACHER_DEFINITIONS[str(teacher_name)]
    except KeyError as exc:
        raise KeyError(f"unknown teacher: {teacher_name}") from exc


def teacher_choices(*, include_signal_attr: bool = True) -> tuple[str, ...]:
    return ALL_TEACHER_NAMES if include_signal_attr else OD_TEACHER_NAMES


def teacher_dataset_root(base_root: Path, teacher_name: str) -> Path:
    return Path(base_root) / teacher_definition(teacher_name).name


def teacher_train_root(base_root: Path, teacher_name: str) -> Path:
    return Path(base_root) / teacher_definition(teacher_name).name


def teacher_eval_root(base_root: Path, teacher_name: str) -> Path:
    return Path(base_root) / teacher_definition(teacher_name).name


def teacher_dataset_summary_path(base_root: Path, teacher_name: str) -> Path:
    root = teacher_dataset_root(base_root, teacher_name)
    definition = teacher_definition(teacher_name)
    if definition.kind == "signal_attr":
        return root / "meta" / "signal_attr_dataset_manifest.json"
    return root / "meta" / "teacher_dataset_summary.json"


def teacher_train_summary_path(base_root: Path, teacher_name: str) -> Path:
    root = teacher_train_root(base_root, teacher_name)
    definition = teacher_definition(teacher_name)
    if definition.kind == "signal_attr":
        return root / "train_summary.json"
    return root / "run_summary.json"


def teacher_checkpoint_path(base_root: Path, teacher_name: str) -> Path:
    root = teacher_train_root(base_root, teacher_name)
    definition = teacher_definition(teacher_name)
    if definition.kind == "signal_attr":
        return root / "best_signal_attr.pt"
    return root / "weights" / "best.pt"


def teacher_eval_summary_path(base_root: Path, teacher_name: str) -> Path:
    root = teacher_eval_root(base_root, teacher_name)
    definition = teacher_definition(teacher_name)
    if definition.kind == "signal_attr":
        return root / "signal_attr_eval_report.json"
    return root / "checkpoint_eval_summary.json"


def legacy_stage_flag(stage: TeacherStage, teacher_name: str) -> str | None:
    definition = teacher_definition(teacher_name)
    if definition.kind != "signal_attr":
        return None
    return {
        "dataset": "signal_attr_dataset",
        "train": "signal_attr_train",
        "eval": "signal_attr_eval",
    }[stage]


__all__ = [
    "ALL_TEACHER_NAMES",
    "OD_TEACHER_NAMES",
    "SIGNAL_ATTR_TEACHER_NAME",
    "TEACHER_DEFINITIONS",
    "TeacherDefinition",
    "TeacherKind",
    "TeacherStage",
    "legacy_stage_flag",
    "teacher_checkpoint_path",
    "teacher_choices",
    "teacher_dataset_root",
    "teacher_dataset_summary_path",
    "teacher_definition",
    "teacher_eval_root",
    "teacher_eval_summary_path",
    "teacher_train_root",
    "teacher_train_summary_path",
]
