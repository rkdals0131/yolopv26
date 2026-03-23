"""PV26 training runtime."""

from importlib import import_module

__all__ = [
    "PV26Trainer",
    "build_pv26_scheduler",
    "build_pv26_optimizer",
    "configure_pv26_train_stage",
    "run_pv26_tiny_overfit",
]


def __getattr__(name: str):
    if name in {
        "PV26Trainer",
        "build_pv26_scheduler",
        "build_pv26_optimizer",
        "configure_pv26_train_stage",
        "run_pv26_tiny_overfit",
    }:
        module = import_module(".pv26_trainer", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
