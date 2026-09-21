from __future__ import annotations

from .det_geometry import decode_anchor_relative_boxes, make_anchor_grid


_make_anchor_grid = make_anchor_grid
_decode_anchor_relative_boxes = decode_anchor_relative_boxes


__all__ = ["_decode_anchor_relative_boxes", "_make_anchor_grid"]
