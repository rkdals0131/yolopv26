from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class BoxXYXY:
    x1: float
    y1: float
    x2: float
    y2: float

    def clip(self, width: int, height: int) -> "BoxXYXY":
        x1 = max(0.0, min(float(width), self.x1))
        y1 = max(0.0, min(float(height), self.y1))
        x2 = max(0.0, min(float(width), self.x2))
        y2 = max(0.0, min(float(height), self.y2))
        # Ensure proper ordering after clip.
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        return BoxXYXY(x1=x1, y1=y1, x2=x2, y2=y2)

    def area_px(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)


def xyxy_to_yolo_normalized(box: BoxXYXY, width: int, height: int) -> Tuple[float, float, float, float]:
    if width <= 0 or height <= 0:
        raise ValueError("width/height must be positive")
    cx = ((box.x1 + box.x2) / 2.0) / float(width)
    cy = ((box.y1 + box.y2) / 2.0) / float(height)
    w = (box.x2 - box.x1) / float(width)
    h = (box.y2 - box.y1) / float(height)
    return cx, cy, w, h


def format_yolo_line(class_id: int, cx: float, cy: float, w: float, h: float) -> str:
    # Spec: 6 decimals.
    return f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"

