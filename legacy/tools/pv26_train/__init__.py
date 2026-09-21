from importlib import import_module

__all__ = ["artifacts", "config", "runtime", "scenario", "scenarios"]


def __getattr__(name):
    if name in __all__ or name in {"cli", "focused"}:
        value = import_module(f".{name}", __name__)
    else:
        value = getattr(import_module(".cli", __name__), name)
    globals()[name] = value
    return value
