from __future__ import annotations

import time

from . import _ultralytics_runner_impl as _impl
from ._ultralytics_runner_impl import (
    _build_epoch_tensorboard_payload,
    _build_train_step_tensorboard_payload,
    _flatten_scalar_tree,
    _install_ultralytics_postfix_renderer,
    _make_teacher_trainer,
    _resolve_resume_argument,
    _coerce_weights_name,
    _load_checkpoint_payload,
    _checkpoint_resume_metadata,
    _infer_teacher_root_from_checkpoint,
    _resume_candidate_sort_key,
    _find_latest_resumable_checkpoint,
    _resolve_resume_checkpoint_path,
    YOLO,
)

def train_teacher_with_ultralytics(*args, **kwargs):
    _impl.YOLO = YOLO
    return _impl.train_teacher_with_ultralytics(*args, **kwargs)

__all__ = [
    "_build_epoch_tensorboard_payload",
    "_build_train_step_tensorboard_payload",
    "_checkpoint_resume_metadata",
    "_coerce_weights_name",
    "_find_latest_resumable_checkpoint",
    "_flatten_scalar_tree",
    "_infer_teacher_root_from_checkpoint",
    "_install_ultralytics_postfix_renderer",
    "_load_checkpoint_payload",
    "_make_teacher_trainer",
    "_resolve_resume_argument",
    "_resolve_resume_checkpoint_path",
    "_resume_candidate_sort_key",
    "train_teacher_with_ultralytics",
    "YOLO",
]
