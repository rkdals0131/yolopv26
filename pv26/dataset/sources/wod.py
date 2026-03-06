from __future__ import annotations

from typing import Tuple

import numpy as np


WAYMO_SEM_ROAD = 20
WAYMO_SEM_LANE_MARKER = 21
WAYMO_SEM_ROAD_MARKER = 22
WAYMO_SEM_SIDEWALK = 23


def semantic_to_pv26_da_rm_masks(semantic_id: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert Waymo Perception v2 camera semantic ids into PV26 masks.

    Returns:
      da_mask, rm_lane_marker, rm_road_marker_non_lane
    """
    if semantic_id.ndim != 2:
        raise ValueError(f"expected 2D semantic id mask, got shape={semantic_id.shape}")
    da = (semantic_id == WAYMO_SEM_ROAD).astype(np.uint8)
    lane = (semantic_id == WAYMO_SEM_LANE_MARKER).astype(np.uint8)
    road = (semantic_id == WAYMO_SEM_ROAD_MARKER).astype(np.uint8)
    return da, lane, road

