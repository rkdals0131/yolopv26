from importlib import import_module

__all__ = ["build_loss_spec", "render_loss_spec_markdown"]


def __getattr__(name: str):
    if name == "build_loss_spec":
        return import_module(".spec", __name__).build_loss_spec
    if name == "render_loss_spec_markdown":
        return import_module(".spec", __name__).render_loss_spec_markdown
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
