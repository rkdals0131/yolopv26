from __future__ import annotations

from . import _sweep_impl as _impl
from ._sweep_impl import run_model_centric_sweep_scenario as _run_model_centric_sweep_scenario
from .sweep_types import (
    BootstrapSweepScenario,
    ClassPolicy,
    ImageListConfig,
    MaterializationConfig,
    REQUIRED_TEACHER_ORDER,
    RunConfig,
    TeacherConfig,
)

YOLO = _impl.YOLO


def run_model_centric_sweep_scenario(*args, **kwargs):
    _impl.YOLO = YOLO
    return _run_model_centric_sweep_scenario(*args, **kwargs)


__all__ = [
    "BootstrapSweepScenario",
    "ClassPolicy",
    "ImageListConfig",
    "MaterializationConfig",
    "REQUIRED_TEACHER_ORDER",
    "RunConfig",
    "TeacherConfig",
    "YOLO",
    "run_model_centric_sweep_scenario",
]
