"""Ultralytics trunk adapters."""

from .ultralytics_yolo26 import (
    MIN_YOLO26_VERSION,
    UltralyticsYOLO26TrunkAdapter,
    build_yolo26n_trunk,
    forward_pyramid_features,
    load_matching_state_dict,
    summarize_trunk_adapter,
)

__all__ = [
    "MIN_YOLO26_VERSION",
    "UltralyticsYOLO26TrunkAdapter",
    "build_yolo26n_trunk",
    "forward_pyramid_features",
    "load_matching_state_dict",
    "summarize_trunk_adapter",
]
