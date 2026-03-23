from importlib import import_module

__all__ = ["run_standardization", "run_bdd100k_standardization"]


def __getattr__(name: str):
    if name == "run_standardization":
        return import_module(".aihub_standardize", __name__).run_standardization
    if name == "run_bdd100k_standardization":
        return import_module(".bdd100k_standardize", __name__).run_standardization
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
