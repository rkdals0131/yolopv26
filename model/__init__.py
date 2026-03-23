"""PV26 project package."""

from importlib import import_module

__all__ = [
    "build_loss_spec",
    "render_loss_spec_markdown",
    "PV26MultiTaskLoss",
    "PV26CanonicalDataset",
    "collate_pv26_samples",
    "encode_pv26_batch",
    "PV26Heads",
    "build_yolo26n_trunk",
    "load_matching_state_dict",
    "PV26Trainer",
    "build_pv26_optimizer",
    "configure_pv26_train_stage",
    "run_pv26_tiny_overfit",
    "PV26Evaluator",
    "run_standardization",
]


def __getattr__(name: str):
    if name == "run_standardization":
        return import_module(".preprocess.aihub_standardize", __name__).run_standardization
    if name == "build_loss_spec":
        return import_module(".loss.spec", __name__).build_loss_spec
    if name == "render_loss_spec_markdown":
        return import_module(".loss.spec", __name__).render_loss_spec_markdown
    if name == "PV26MultiTaskLoss":
        return import_module(".loss", __name__).PV26MultiTaskLoss
    if name in {"PV26CanonicalDataset", "collate_pv26_samples"}:
        module = import_module(".loading", __name__)
        return getattr(module, name)
    if name == "encode_pv26_batch":
        return import_module(".encoding", __name__).encode_pv26_batch
    if name == "PV26Heads":
        return import_module(".heads", __name__).PV26Heads
    if name in {"build_yolo26n_trunk", "load_matching_state_dict"}:
        module = import_module(".trunk", __name__)
        return getattr(module, name)
    if name in {"PV26Trainer", "build_pv26_optimizer", "configure_pv26_train_stage", "run_pv26_tiny_overfit"}:
        module = import_module(".training", __name__)
        return getattr(module, name)
    if name == "PV26Evaluator":
        return import_module(".eval", __name__).PV26Evaluator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
