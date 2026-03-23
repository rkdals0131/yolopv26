"""PV26 evaluation runtime."""

from importlib import import_module

__all__ = [
    "PV26MetricConfig",
    "PV26Evaluator",
    "PV26PostprocessConfig",
    "postprocess_pv26_batch",
    "summarize_pv26_metrics",
]


def __getattr__(name: str):
    if name in {"PV26MetricConfig", "summarize_pv26_metrics"}:
        module = import_module(".metrics", __name__)
        return getattr(module, name)
    if name in {"PV26PostprocessConfig", "postprocess_pv26_batch"}:
        module = import_module(".postprocess", __name__)
        return getattr(module, name)
    if name == "PV26Evaluator":
        module = import_module(".pv26_evaluator", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
