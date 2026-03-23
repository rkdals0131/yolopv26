"""PV26 project package."""

from importlib import import_module

__all__ = [
    "build_loss_spec",
    "render_loss_spec_markdown",
    "PV26CanonicalDataset",
    "collate_pv26_samples",
    "run_standardization",
]


def __getattr__(name: str):
    if name == "run_standardization":
        return import_module(".preprocess.aihub_standardize", __name__).run_standardization
    if name == "build_loss_spec":
        return import_module(".loss.spec", __name__).build_loss_spec
    if name == "render_loss_spec_markdown":
        return import_module(".loss.spec", __name__).render_loss_spec_markdown
    if name in {"PV26CanonicalDataset", "collate_pv26_samples"}:
        module = import_module(".loading", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
