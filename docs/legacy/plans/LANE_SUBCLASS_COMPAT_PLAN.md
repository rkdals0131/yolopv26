> **Archived (2026-03-05)** — kept for history; may be outdated.  
> Canonical docs: `docs/PV26_PRD.md`, `docs/PV26_DATASET_CONVERSION_SPEC.md`, `docs/PV26_DATASET_SOURCES_AND_MAPPING.md`, `docs/PV26_IMPLEMENTATION_STATUS.md`.

# PV26 Lane Subclass Compatibility & Mapping Plan

## 1. Scope
- Goal: add lane subclass supervision (`white_solid`, `white_dashed`, `yellow_solid`, `yellow_dashed`) to PV26.
- Current implementation scope: **BDD100K first**.
- Existing RM binary channels stay unchanged:
  - `rm_lane_marker`
  - `rm_road_marker_non_lane`
  - `rm_stop_line`
- Added sidecar mask:
  - `labels_seg_rm_lane_subclass/*/*.png` (mono8)

## 2. Backward Compatibility
- Kept stable:
  - RM binary 3-channel training path and loss
  - Existing manifest fields for RM binary channels
- Added (non-overwriting):
  - `has_rm_lane_subclass`
  - `rm_lane_subclass_relpath`
  - model output `rm_lane_subclass` logits
- Operational impact:
  - New dataset conversion emits `classmap-v3` by default.
  - Older `classmap-v2` datasets should be re-converted for subclass training.

## 3. Dataset Mapping Status

### 3.1 BDD100K (implemented)
- Source signal:
  - lane category (`lane/single white`, `lane/double yellow`, ...)
  - `attributes.style` (`solid|dashed`)
- Mapping:
  - white + solid -> 1
  - white + dashed -> 2
  - yellow + solid -> 3
  - yellow + dashed -> 4
- `lane/single other`, `lane/double other`:
  - kept in `rm_lane_marker`
  - mapped to `255(ignore)` in `rm_lane_subclass`

### 3.2 RLMD (planned)
- RLMD palette already contains direct classes:
  - `solid single white`, `dashed single white`, `solid single yellow`, `dashed single yellow`
- Planned mapping policy:
  - direct 1:1 remap to lane subclass ids
  - double-line classes initially merged by color/style policy

### 3.3 ETRI (planned)
- ETRI polygon labels include lane-like codes:
  - `whsol`, `whdot`, `yesol`, `yedot`
- Planned mapping policy:
  - `whsol -> 1`, `whdot -> 2`, `yesol -> 3`, `yedot -> 4`
  - remaining lane-like labels (`bldot`, `blsol`, guidance variants) initially `255(ignore)` unless policy extension approved

### 3.4 Waymo (planned, limited)
- Waymo camera segmentation exposes coarse classes (`TYPE_LANE_MARKER`, `TYPE_ROAD_MARKER`) without explicit yellow/white x solid/dashed IDs.
- Planned policy:
  - keep contributing to RM binary channels
  - do not use as direct supervised source for lane subclass classes
  - evaluate weak-supervision or pseudo-label path separately

## 4. Semantic Contract (current)
- `classmap-v3` semantic IDs:
  - `0 background`
  - `1 drivable_area`
  - `2 lane_white_solid`
  - `3 lane_white_dashed`
  - `4 lane_yellow_solid`
  - `5 lane_yellow_dashed`
  - `6 road_marker_non_lane`
  - `7 stop_line`
- Compose priority:
  - `stop_line > lane_subclass > road_marker_non_lane > drivable > background`

## 5. Follow-up Work Queue
1. RLMD converter adapter for subclass direct mapping
2. ETRI converter adapter for polygon-label subclass mapping
3. Waymo coarse-to-subclass strategy proposal (weak supervision)
4. class-frequency balancing policy for subclass loss weighting
