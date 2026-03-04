"""
PV26 dataset conversion utilities.

This package implements the first executable slice for PV26 Type-A (BDD-only)
dataset conversion, following:
- docs/PRD.md
- docs/DATASET_CONVERSION_SPEC.md
"""

from .constants import (  # noqa: F401
    CLASSMAP_VERSION_V2,
    CLASSMAP_VERSION_V3,
    DEFAULT_CLASSMAP_VERSION,
    DET_CLASSES_CANONICAL,
    DET_NAME_TO_ID,
    SEG_ID_BACKGROUND,
    SEG_ID_DRIVABLE,
    SEG_ID_LANE_MARKING,
    SEG_ID_STOP_LINE,
    SEG3_ID_BACKGROUND,
    SEG3_ID_DRIVABLE,
    SEG3_ID_LANE_WHITE_SOLID,
    SEG3_ID_LANE_WHITE_DASHED,
    SEG3_ID_LANE_YELLOW_SOLID,
    SEG3_ID_LANE_YELLOW_DASHED,
    SEG3_ID_ROAD_MARKER_NON_LANE,
    SEG3_ID_STOP_LINE,
)
from .multitask_model import PV26MultiHead, PV26MultiHeadOutput  # noqa: F401

from .criterion import PV26Criterion, PV26LossBreakdown  # noqa: F401
