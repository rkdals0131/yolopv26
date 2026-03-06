from __future__ import annotations

from dataclasses import dataclass
from typing import Set

import numpy as np

IGNORE_VALUE = 255

# lane subclass ids in mono8 rm_lane_subclass mask
LANE_SUBCLASS_BG = 0
LANE_SUBCLASS_WHITE_SOLID = 1
LANE_SUBCLASS_WHITE_DASHED = 2
LANE_SUBCLASS_YELLOW_SOLID = 3
LANE_SUBCLASS_YELLOW_DASHED = 4


def validate_binary_mask_u8(mask: np.ndarray, *, allow_ignore: bool = True, name: str = "mask") -> None:
    """
    Validate a uint8 mask with domain {0,1} (+ optional {255} ignore).
    Raises ValueError on violations.
    """
    if mask.dtype != np.uint8:
        raise ValueError(f"{name}: expected dtype=uint8, got {mask.dtype}")
    if mask.ndim != 2:
        raise ValueError(f"{name}: expected 2D mask, got shape={mask.shape}")

    allowed = {0, 1, IGNORE_VALUE} if allow_ignore else {0, 1}
    vals = set(np.unique(mask).tolist())
    bad = vals - allowed
    if bad:
        raise ValueError(f"{name}: invalid values {sorted(bad)} (allowed {sorted(allowed)})")


def validate_lane_subclass_mask_u8(mask: np.ndarray, *, allow_ignore: bool = True, name: str = "rm_lane_subclass") -> None:
    """
    Validate lane-subclass mono8 mask.

    Domain:
      0 background
      1 white_solid
      2 white_dashed
      3 yellow_solid
      4 yellow_dashed
      255 ignore (optional)
    """
    if mask.dtype != np.uint8:
        raise ValueError(f"{name}: expected dtype=uint8, got {mask.dtype}")
    if mask.ndim != 2:
        raise ValueError(f"{name}: expected 2D mask, got shape={mask.shape}")

    allowed = {
        LANE_SUBCLASS_BG,
        LANE_SUBCLASS_WHITE_SOLID,
        LANE_SUBCLASS_WHITE_DASHED,
        LANE_SUBCLASS_YELLOW_SOLID,
        LANE_SUBCLASS_YELLOW_DASHED,
    }
    if allow_ignore:
        allowed = set(allowed)
        allowed.add(IGNORE_VALUE)

    vals = set(np.unique(mask).tolist())
    bad = vals - allowed
    if bad:
        raise ValueError(f"{name}: invalid values {sorted(bad)} (allowed {sorted(allowed)})")


def make_all_ignore_mask(height: int, width: int) -> np.ndarray:
    return np.full((height, width), IGNORE_VALUE, dtype=np.uint8)


def convert_bdd_drivable_id_to_da_mask_u8(drivable_id: np.ndarray) -> np.ndarray:
    """
    Convert a BDD drivable *id mask* to PV26 DA mask {0,1,255}.

    Expected BDD ids (common):
      0: background
      1: direct drivable
      2: alternative drivable
    Any other value is mapped to ignore(255).
    """
    if drivable_id.dtype != np.uint8:
        drivable_id = drivable_id.astype(np.uint8, copy=False)
    if drivable_id.ndim != 2:
        raise ValueError(f"drivable_id: expected 2D, got shape={drivable_id.shape}")

    out = np.full(drivable_id.shape, IGNORE_VALUE, dtype=np.uint8)
    out[drivable_id == 0] = 0
    out[(drivable_id == 1) | (drivable_id == 2)] = 1
    return out


@dataclass(frozen=True)
class SemanticComposeResult:
    semantic_id: np.ndarray
    # True only when composition was performed without unknown(255) inputs.
    ok: bool
    reason: str = ""


def _has_ignore(mask: np.ndarray) -> bool:
    return bool(np.any(mask == IGNORE_VALUE))


def compose_semantic_id_v2(
    da_mask: np.ndarray,
    rm_lane_marker: np.ndarray,
    rm_road_marker_non_lane: np.ndarray,
    rm_stop_line: np.ndarray,
) -> SemanticComposeResult:
    """
    Compose classmap-v2 semantic_id from DA + RM channels.

    Contract:
      - semantic_id must contain no 255.
      - If any input contains ignore(255), composition is refused.
    """
    for n, m in [
        ("da_mask", da_mask),
        ("rm_lane_marker", rm_lane_marker),
        ("rm_road_marker_non_lane", rm_road_marker_non_lane),
        ("rm_stop_line", rm_stop_line),
    ]:
        validate_binary_mask_u8(m, allow_ignore=True, name=n)

    if not (da_mask.shape == rm_lane_marker.shape == rm_road_marker_non_lane.shape == rm_stop_line.shape):
        return SemanticComposeResult(
            semantic_id=np.zeros((1, 1), dtype=np.uint8),
            ok=False,
            reason="shape_mismatch",
        )

    if _has_ignore(da_mask) or _has_ignore(rm_lane_marker) or _has_ignore(rm_road_marker_non_lane) or _has_ignore(rm_stop_line):
        return SemanticComposeResult(
            semantic_id=np.zeros((1, 1), dtype=np.uint8),
            ok=False,
            reason="has_ignore_255_in_inputs",
        )

    # Import here to avoid circular import at module load time.
    from .constants import SEG_ID_BACKGROUND, SEG_ID_DRIVABLE, SEG_ID_LANE_MARKING, SEG_ID_STOP_LINE

    sem = np.full(da_mask.shape, SEG_ID_BACKGROUND, dtype=np.uint8)
    sem[da_mask == 1] = SEG_ID_DRIVABLE

    # Road-marker-non-lane is not a dedicated semantic class in v2; treat as lane_marking in semantic_id.
    sem[rm_road_marker_non_lane == 1] = SEG_ID_LANE_MARKING
    sem[rm_lane_marker == 1] = SEG_ID_LANE_MARKING
    sem[rm_stop_line == 1] = SEG_ID_STOP_LINE

    return SemanticComposeResult(semantic_id=sem, ok=True)


def compose_semantic_id_v3(
    da_mask: np.ndarray,
    rm_lane_subclass: np.ndarray,
    rm_road_marker_non_lane: np.ndarray,
    rm_stop_line: np.ndarray,
) -> SemanticComposeResult:
    """
    Compose classmap-v3 semantic_id from DA + lane-subclass + road-marker channels.

    Priority:
      stop_line > lane_subclass > road_marker_non_lane > drivable > background

    Contract:
      - semantic_id must contain no 255.
      - If any input contains ignore(255), composition is refused.
    """
    validate_binary_mask_u8(da_mask, allow_ignore=True, name="da_mask")
    validate_lane_subclass_mask_u8(rm_lane_subclass, allow_ignore=True, name="rm_lane_subclass")
    validate_binary_mask_u8(rm_road_marker_non_lane, allow_ignore=True, name="rm_road_marker_non_lane")
    validate_binary_mask_u8(rm_stop_line, allow_ignore=True, name="rm_stop_line")

    if not (da_mask.shape == rm_lane_subclass.shape == rm_road_marker_non_lane.shape == rm_stop_line.shape):
        return SemanticComposeResult(
            semantic_id=np.zeros((1, 1), dtype=np.uint8),
            ok=False,
            reason="shape_mismatch",
        )

    if _has_ignore(da_mask) or _has_ignore(rm_lane_subclass) or _has_ignore(rm_road_marker_non_lane) or _has_ignore(rm_stop_line):
        return SemanticComposeResult(
            semantic_id=np.zeros((1, 1), dtype=np.uint8),
            ok=False,
            reason="has_ignore_255_in_inputs",
        )

    from .constants import (
        SEG3_ID_BACKGROUND,
        SEG3_ID_DRIVABLE,
        SEG3_ID_LANE_WHITE_DASHED,
        SEG3_ID_LANE_WHITE_SOLID,
        SEG3_ID_LANE_YELLOW_DASHED,
        SEG3_ID_LANE_YELLOW_SOLID,
        SEG3_ID_ROAD_MARKER_NON_LANE,
        SEG3_ID_STOP_LINE,
    )

    sem = np.full(da_mask.shape, SEG3_ID_BACKGROUND, dtype=np.uint8)
    sem[da_mask == 1] = SEG3_ID_DRIVABLE
    sem[rm_road_marker_non_lane == 1] = SEG3_ID_ROAD_MARKER_NON_LANE

    sem[rm_lane_subclass == LANE_SUBCLASS_WHITE_SOLID] = SEG3_ID_LANE_WHITE_SOLID
    sem[rm_lane_subclass == LANE_SUBCLASS_WHITE_DASHED] = SEG3_ID_LANE_WHITE_DASHED
    sem[rm_lane_subclass == LANE_SUBCLASS_YELLOW_SOLID] = SEG3_ID_LANE_YELLOW_SOLID
    sem[rm_lane_subclass == LANE_SUBCLASS_YELLOW_DASHED] = SEG3_ID_LANE_YELLOW_DASHED

    sem[rm_stop_line == 1] = SEG3_ID_STOP_LINE
    return SemanticComposeResult(semantic_id=sem, ok=True)


def validate_semantic_id_u8(semantic_id: np.ndarray, *, allowed_ids: Set[int], name: str = "semantic_id") -> None:
    if semantic_id.dtype != np.uint8:
        raise ValueError(f"{name}: expected dtype=uint8, got {semantic_id.dtype}")
    if semantic_id.ndim != 2:
        raise ValueError(f"{name}: expected 2D mask, got shape={semantic_id.shape}")
    vals = set(np.unique(semantic_id).tolist())
    if IGNORE_VALUE in vals:
        raise ValueError(f"{name}: contains forbidden ignore value 255")
    bad = vals - set(allowed_ids)
    if bad:
        raise ValueError(f"{name}: contains invalid class ids {sorted(bad)} (allowed {sorted(allowed_ids)})")
