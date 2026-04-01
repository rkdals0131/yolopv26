from __future__ import annotations

from . import _eval_impl as _impl
from ._eval_impl import *  # noqa: F401,F403

YOLO = _impl.YOLO


def eval_teacher_checkpoint(*args, **kwargs):
    _impl.YOLO = YOLO
    return _impl.eval_teacher_checkpoint(*args, **kwargs)
