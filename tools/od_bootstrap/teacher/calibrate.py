from __future__ import annotations

from . import _calibrate_impl as _impl
from ._calibrate_impl import *  # noqa: F401,F403

YOLO = _impl.YOLO


def calibrate_class_policy_scenario(*args, **kwargs):
    _impl.YOLO = YOLO
    return _impl.calibrate_class_policy_scenario(*args, **kwargs)
