"""PV26 training and evaluation engine.

The public package surface intentionally re-exports only the stable engine
entry points; underscore-prefixed helper modules remain internal.
"""

from __future__ import annotations

from importlib import import_module


__all__ = [
    "PV26DetAssignmentUnavailable",
    "PV26Evaluator",
    "PV26MetricConfig",
    "PV26MultiTaskLoss",
    "PV26PostprocessConfig",
    "PV26Trainer",
    "SPEC_VERSION",
    "STAGE_NAMES",
    "TIMING_KEYS",
    "TENSORBOARD_LOSS_KEYS",
    "augment_lane_family_metrics",
    "build_loss_spec",
    "build_pv26_optimizer",
    "build_pv26_scheduler",
    "configure_pv26_train_stage",
    "move_batch_to_device",
    "postprocess_pv26_batch",
    "raw_batch_for_metrics",
    "render_loss_spec_markdown",
    "run_pv26_tiny_overfit",
    "summarize_pv26_metrics",
]

_EXPORTS = {
    "augment_lane_family_metrics": ("batch", "augment_lane_family_metrics"),
    "move_batch_to_device": ("batch", "move_batch_to_device"),
    "raw_batch_for_metrics": ("batch", "raw_batch_for_metrics"),
    "PV26Evaluator": ("evaluator", "PV26Evaluator"),
    "PV26DetAssignmentUnavailable": ("loss", "PV26DetAssignmentUnavailable"),
    "PV26MultiTaskLoss": ("loss", "PV26MultiTaskLoss"),
    "PV26MetricConfig": ("metrics", "PV26MetricConfig"),
    "summarize_pv26_metrics": ("metrics", "summarize_pv26_metrics"),
    "PV26PostprocessConfig": ("postprocess", "PV26PostprocessConfig"),
    "postprocess_pv26_batch": ("postprocess", "postprocess_pv26_batch"),
    "SPEC_VERSION": ("spec", "SPEC_VERSION"),
    "build_loss_spec": ("spec", "build_loss_spec"),
    "render_loss_spec_markdown": ("spec", "render_loss_spec_markdown"),
    "PV26Trainer": ("trainer", "PV26Trainer"),
    "STAGE_NAMES": ("trainer", "STAGE_NAMES"),
    "TIMING_KEYS": ("trainer", "TIMING_KEYS"),
    "TENSORBOARD_LOSS_KEYS": ("trainer", "TENSORBOARD_LOSS_KEYS"),
    "build_pv26_optimizer": ("trainer", "build_pv26_optimizer"),
    "build_pv26_scheduler": ("trainer", "build_pv26_scheduler"),
    "configure_pv26_train_stage": ("trainer", "configure_pv26_train_stage"),
    "run_pv26_tiny_overfit": ("trainer", "run_pv26_tiny_overfit"),
}


def __getattr__(name: str):
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(f".{module_name}", __name__), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
