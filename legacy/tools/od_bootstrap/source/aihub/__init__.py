from __future__ import annotations

from importlib import import_module as _import_module
from pathlib import Path
from typing import Any

_PIPELINE = _import_module("tools.od_bootstrap.source.aihub.pipeline")
__file__ = str(Path(__file__).resolve().parents[1] / "aihub.py")

__all__ = [
    "TL_BITS",
    "PIPELINE_VERSION",
    "main",
    "run_standardization",
    "_prepare_debug_scene_for_overlay",
    "_select_debug_vis_summaries",
]


def __getattr__(name: str) -> Any:
    try:
        return getattr(_PIPELINE, name)
    except AttributeError as exc:
        raise AttributeError(name) from exc
