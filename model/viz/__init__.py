from importlib import import_module

__all__ = ["render_overlay"]


def __getattr__(name: str):
    if name == "render_overlay":
        return import_module(".overlay", __name__).render_overlay
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
