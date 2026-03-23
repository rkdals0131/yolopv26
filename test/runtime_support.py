from __future__ import annotations


def has_yolo26_runtime() -> bool:
    try:
        from model.trunk.ultralytics_yolo26 import YOLO, ensure_yolo26_support

        ensure_yolo26_support()
        return YOLO is not None
    except Exception:
        return False
