from importlib import import_module

__all__ = [
    "PV26CanonicalDataset",
    "collate_pv26_samples",
    "compute_letterbox_transform",
    "load_letterboxed_image",
]


def __getattr__(name: str):
    if name in {"PV26CanonicalDataset", "collate_pv26_samples"}:
        module = import_module(".pv26_loader", __name__)
        return getattr(module, name)
    if name in {"compute_letterbox_transform", "load_letterboxed_image"}:
        module = import_module(".transform", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
