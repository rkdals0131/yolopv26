from importlib import import_module

__all__ = [
    "PV26DetAssignmentUnavailable",
    "PV26MultiTaskLoss",
    "build_loss_spec",
    "render_loss_spec_markdown",
]


def __getattr__(name: str):
    if name == "PV26DetAssignmentUnavailable":
        return import_module(".runtime", __name__).PV26DetAssignmentUnavailable
    if name == "PV26MultiTaskLoss":
        return import_module(".runtime", __name__).PV26MultiTaskLoss
    if name == "build_loss_spec":
        return import_module(".spec", __name__).build_loss_spec
    if name == "render_loss_spec_markdown":
        return import_module(".spec", __name__).render_loss_spec_markdown
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
