"""Ultralytics trunk adapters."""

from importlib import import_module

__all__ = [
    "MIN_YOLO26_VERSION",
    "UltralyticsYOLO26TrunkAdapter",
    "build_yolo26n_trunk",
    "forward_pyramid_features",
    "load_matching_state_dict",
    "summarize_trunk_adapter",
]


def __getattr__(name: str):
    if name in {
        "MIN_YOLO26_VERSION",
        "UltralyticsYOLO26TrunkAdapter",
        "build_yolo26n_trunk",
        "forward_pyramid_features",
        "load_matching_state_dict",
        "summarize_trunk_adapter",
    }:
        module = import_module(".ultralytics_yolo26", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
