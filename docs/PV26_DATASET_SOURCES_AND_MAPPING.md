# PV26 Dataset Sources & Mapping

- Doc version: `v1.2`
- Last reviewed: `2026-03-10`
- Scope: local `datasets/` inventory + source label inventory + mapping review notes for the PV26 unified dataset format
- See also:
  - Product / runtime contract: `docs/PV26_PRD.md`
  - Conversion output contract: `docs/PV26_DATASET_CONVERSION_SPEC.md`
  - Historical inventory snapshot: `docs/legacy/snapshots/Dataset_Profile_2026-02-19.md`

## 1) Why this doc

PV26는 여러 데이터셋을 섞어 학습해야 한다. 그런데 데이터셋마다:
- OD taxonomy가 다르다
- lane / road marker label-space가 다르다
- 어떤 태스크는 원천에 없고, 어떤 태스크는 coarse only다

그래서 이 문서는 두 가지를 명확히 분리해서 관리한다.
- 현재 코드가 실제로 구현한 mapping
- 앞으로 검토할 새로운 coarse taxonomy 제안

핵심 원칙은 다음과 같다.
1. 원천 라벨을 학습 시점에 직접 파싱하지 않는다.
2. source별 class inventory를 먼저 문서화한다.
3. target taxonomy를 잠근 뒤에 converter / loss / eval을 바꾼다.
4. active converter/runtime는 detection coverage를 `full|none`으로만 다루고, coverage 문제는 source별 class remap/skip 정책으로 upstream 해결한다.

## 2) Two Tracks In This Doc

### 2.1 Current implementation track

현재 repo 코드가 실제로 사용 중인 canonical contract다.
- OD: coarse 7-class
- DA: binary
- RM: `lane_marker`, `road_marker_non_lane`, `stop_line`
- lane subclass: `white/yellow x solid/dashed`

### 2.2 Provisional review track

이번 문서 업데이트의 목적은 모든 source를 함께 놓고 coarse 7-class OD taxonomy를 실제 구현과 문서에 일치시키는 것이다.

이 문서에서는 아래 draft를 검토 대상으로 둔다.
- `vehicle`
- `bike`
- `pedestrian`
- `traffic_cone`
- `obstacle`
- `traffic_light`
- `sign_pole`

주의:
- 이 7-class taxonomy는 현재 active implementation이다.
- 다만 source별 raw inventory와 merge risk는 계속 문서화해 유지한다.
- `others`는 training class에 넣지 않는다. source마다 의미가 너무 달라 잡탕 클래스가 되기 쉽기 때문이다.

## 3) Current Active Canonical Tasks

### 3.1 Current OD classes in code

현재 구현은 아래 7-class OD를 사용한다.

| det_id | class_name |
|---|---|
| 0 | vehicle |
| 1 | bike |
| 2 | pedestrian |
| 3 | traffic_cone |
| 4 | obstacle |
| 5 | traffic_light |
| 6 | sign_pole |

Source of truth:
- `pv26/dataset/labels.py`
- `docs/PV26_DATASET_CONVERSION_SPEC.md`

### 3.2 Current RM / lane contract

- `rm_lane_marker`: binary `{0,1,255}`
- `rm_road_marker_non_lane`: binary `{0,1,255}`
- `rm_stop_line`: binary `{0,1,255}`
- `rm_lane_subclass`: mono8 `{0,1,2,3,4,255}`
  - `1 white_solid`
  - `2 white_dashed`
  - `3 yellow_solid`
  - `4 yellow_dashed`

## 4) Active 7-Class OD Target

이 section은 현재 구현 기준의 canonical detection taxonomy다.

| draft_id | draft_class | Intended merges |
|---|---|---|
| 0 | vehicle | `car`, `bus`, `truck`, source-level `vehicle` |
| 1 | bike | `bicycle`, `motorcycle`, source-level `cyclist` |
| 2 | pedestrian | `person`, `pedestrian`, `rider` |
| 3 | traffic_cone | `traffic cone`, `construction cone`, `rubber cone` |
| 4 | obstacle | `barrier`, `bollard`, `road_obstacle`, unknown static/movable roadside hazard |
| 5 | traffic_light | raw `traffic light` |
| 6 | sign_pole | raw `traffic sign`, `pole`, `sign`, pole-like roadside fixture |

Design notes:
- `traffic_light`는 `sign_pole`과 분리해서 유지한다.
- `traffic_cone`는 `obstacle`과 분리해서 유지한다.
- `others`는 active training class에 넣지 않는다.
- source별 raw inventory는 이 7-class에 대한 remap 근거로 계속 유지한다.

## 5) Source Capability Matrix

상태 표기:
- `implemented`: PV26 adapter가 있고 현재 계약으로 검증 가능
- `planned`: source inventory는 있으나 adapter 또는 mapping policy가 미완성
- `not-possible`: 원천 supervision이 현재 목적과 맞지 않거나 실질적으로 없음

| Dataset | OD | DA | RM lane_marker | RM road_marker_non_lane | RM stop_line | lane_subclass | Notes |
|---|---|---|---|---|---|---|---|
| BDD100K | implemented | implemented | implemented | implemented | implemented | implemented | 현재 OD base source |
| Cityscapes | planned | planned | not-possible | not-possible | not-possible | not-possible | OD는 semantic/instance 기반 재구성이 필요 |
| ETRI | raw labels exist, adapter not using them | implemented | implemented | implemented | implemented | implemented | 현재는 RM/DA 전용으로 사용 중 |
| KITTI-360 | planned | planned | planned | planned | planned | not-possible | Cityscapes-equivalent review 필요 |
| RLMD | not-possible | not-possible | implemented | implemented | implemented | implemented | RM 전용 |
| Waymo (Perception v2) | implemented | implemented | implemented | implemented | not-possible | not-possible | active OD export는 panoptic segmentation-backed full only |

## 6) Per-Source Inventory And Proposed Merge Review

이 section은 source별로 아래 네 가지를 기록한다.
1. raw label inventory
2. current implementation usage
3. provisional 7-class OD proposal
4. risk / open question

### 6.1 BDD100K

Assets:
- Image (RGB)
- Detection JSON
- Drivable area index mask
- Lane poly2d / semantic-related labels

#### Raw OD inventory

From checked-in legacy snapshot and current adapter support:
- standard det labels seen in BDD inventories:
  - `pedestrian`
  - `rider`
  - `car`
  - `truck`
  - `bus`
  - `train`
  - `motorcycle`
  - `bicycle`
  - `traffic light`
  - `traffic sign`
- additional strings accepted by current adapter when present in source variants:
  - `traffic cone`
  - `construction cone`
  - `cone`
  - `barrier`
  - `bollard`
  - `other vehicle`
  - `pole`
  - `sign`
  - `light`

#### Current implemented mapping

Current adapter (`pv26/dataset/sources/bdd.py`):
- `car|bus|truck|other vehicle -> vehicle`
- `motorcycle|bicycle|bike -> bike`
- `person|pedestrian|rider -> pedestrian`
- `traffic light|light -> traffic_light`
- `traffic sign|pole|sign -> sign_pole`
- `traffic cone|construction cone|cone -> traffic_cone`
- `barrier|bollard|road obstacle|train -> obstacle`

#### Active 7-class merge

| Raw BDD label | Proposed 7-class |
|---|---|
| `car`, `bus`, `truck` | `vehicle` |
| `motorcycle`, `bicycle`, `bike` | `bike` |
| `person`, `pedestrian`, `rider` | `pedestrian` |
| `traffic light` | `traffic_light` |
| `traffic sign`, `pole`, `sign`, `light` | `sign_pole` |
| `traffic cone`, `construction cone`, `cone` | `traffic_cone` |
| `barrier`, `bollard`, `other vehicle`, `train` | `obstacle` |

#### Raw RM / lane inventory

Current source code explicitly recognizes:
- lane-marker-like:
  - `lane/single white`
  - `lane/double white`
  - `lane/single yellow`
  - `lane/double yellow`
  - `lane/single other`
  - `lane/double other`
- road-marker-non-lane:
  - `lane/crosswalk`
  - `lane/road curb`
- stop line:
  - `lane/stop line`
- lane subclass cues:
  - white/yellow x solid/dashed from lane attributes

#### RM notes

- `lane/single|double other`는 `rm_lane_marker`에는 포함하고 `lane_subclass`에서는 `255(ignore)` 처리한다.
- BDD lane schema는 RM/lane_subclass 기준점으로 계속 가치가 높다.

### 6.2 Cityscapes

Assets:
- Image (RGB)
- Semantic labelIds / trainId
- Instance ids
- Polygon JSON

#### Raw OD-relevant inventory

From checked-in legacy snapshot:
- `person`
- `rider`
- `car`
- `truck`
- `bus`
- `train`
- `motorcycle`
- `bicycle`
- `traffic light`
- `traffic sign`
- `pole`

#### Current implementation usage

- current repo policy는 Cityscapes OD adapter를 아직 구현하지 않았다
- current repo policy는 RM을 기본 미제공으로 본다

#### Provisional 7-class proposal

| Raw Cityscapes label | Proposed 7-class |
|---|---|
| `car`, `truck`, `bus` | `vehicle` |
| `motorcycle`, `bicycle` | `bike` |
| `person`, `rider` | `pedestrian` |
| `traffic light` | `traffic_light` |
| `traffic sign`, `pole` | `sign_pole` |
| `train` | `obstacle` |

#### Risk / open question

- Cityscapes는 bbox가 아니라 semantic / instance 기반이라 OD export policy를 따로 정해야 한다.
- `traffic_cone`에 직접 해당하는 raw class는 없다.

### 6.3 ETRI

Assets:
- Polygon JSON
- RGB images

#### Raw OD-relevant inventory

From checked-in legacy snapshot for ETRI polygon labels:
- `car`
- `bus`
- `truck`
- `motorcycle`
- `bicycle`
- `person`
- `rider`
- `traffic light`
- `traffic sign`
- `rubber cone`
- `pole`
- `polegroup`

Current converter does not export OD from these labels.

#### Current implementation usage

Current adapter (`pv26/dataset/sources/etri.py`) only uses ETRI for:
- `road -> DA`
- lane-like labels -> `rm_lane_marker`
- non-lane road-mark labels -> `rm_road_marker_non_lane`
- `stop line -> rm_stop_line`
- `whsol/whdot/yesol/yedot -> lane_subclass`

Current train-time manifest records:
- `has_det=0`
- `det_label_scope=none`

#### Provisional 7-class proposal if OD extraction is added later

| Raw ETRI label | Proposed 7-class |
|---|---|
| `car`, `bus`, `truck` | `vehicle` |
| `motorcycle`, `bicycle` | `bike` |
| `person`, `rider` | `pedestrian` |
| `traffic light` | `traffic_light` |
| `traffic sign`, `pole`, `polegroup` | `sign_pole` |
| `rubber cone` | `traffic_cone` |

#### Raw RM / lane inventory

Current adapter groups:
- lane-subclass-direct:
  - `whsol`
  - `whdot`
  - `yesol`
  - `yedot`
- lane-like but not subclass-direct:
  - `bldot`
  - `blsol`
  - `guidance line`
  - `lane divider`
- road-marker-non-lane regex bucket:
  - `general road mark`
  - `crosswalk`
  - `stop line`
  - `arrow`
  - `prohibition`
  - `number`
  - `slow`
  - `motor`
  - `bike icon`
  - `box junction`
  - `parking`
  - `speed bump`
  - `channelizing line`
  - `left`
  - `right`
  - `forward`
  - `straight`
  - `leftu`
  - `protection zone`

#### RM notes

- ETRI는 RM inventory가 rich하다.
- ETRI는 현재도 RM/lane_subclass 보강 source로 가치가 높다.
- ETRI raw labels에는 OD-relevant objects도 보이므로, future OD source 후보로 재검토 가치가 있다.

### 6.4 KITTI-360

Assets:
- Semantic / instance labels

#### Raw OD-relevant inventory

From checked-in legacy snapshot:
- `person`
- `rider`
- `car`
- `truck`
- `bus`
- `caravan`
- `trailer`
- `train`
- `motorcycle`
- `bicycle`
- `traffic light`
- `traffic sign`
- `pole`

#### Current implementation usage

- adapter is still planned
- current repo has no active normalized KITTI-360 PV26 builder

#### Provisional 7-class proposal

| Raw KITTI-360 label | Proposed 7-class |
|---|---|
| `car`, `truck`, `bus`, `caravan`, `trailer` | `vehicle` |
| `motorcycle`, `bicycle` | `bike` |
| `person`, `rider` | `pedestrian` |
| `traffic light` | `traffic_light` |
| `traffic sign`, `pole` | `sign_pole` |
| `train` | `obstacle` |

#### Risk / open question

- `caravan` / `trailer`를 `vehicle`로 합치는 것이 충분한지 검토 필요
- RM usable labels는 별도 inventory review가 더 필요하다

### 6.5 RLMD

Assets:
- RGB palette road-marking masks

#### Raw OD inventory

- 없음
- current RLMD source는 road marking 전용으로 간주한다

#### Current implementation usage

Current adapter (`pv26/dataset/sources/rlmd.py`) only exports RM channels and lane_subclass.

#### Raw RM inventory

From current source code and checked-in legacy snapshot:
- stop line:
  - `stop line`
- lane-marker-like:
  - `solid single white`
  - `solid single yellow`
  - `solid single red`
  - `solid double white`
  - `solid double yellow`
  - `dashed single white`
  - `dashed single yellow`
  - `channelizing line`
- non-lane road markers:
  - `box junction`
  - `crosswalk`
  - `left arrow`
  - `straight arrow`
  - `right arrow`
  - `left straight arrow`
  - `right straight arrow`
  - `motor prohibited`
  - `slow`
  - `motor priority lane`
  - `motor waiting zone`
  - `left turn box`
  - `motor icon`
  - `bike icon`
  - `parking lot`

#### Current implemented mapping

- `stop line -> rm_stop_line` and also `rm_road_marker_non_lane`
- lane-marker names -> `rm_lane_marker`
- lane-marker names with white/yellow + solid/dashed -> `lane_subclass`
- remaining known palette classes -> `rm_road_marker_non_lane`

#### RM notes

- RLMD는 OD source가 아니라 RM enrichment source다.
- `solid single red`는 현재 lane_marker로 취급되지만 lane_subclass direct class에는 들어가지 않는다.

### 6.6 Waymo Open Dataset (Perception v2)

Assets:
- camera image parquet
- camera box parquet
- camera segmentation parquet

#### Raw OD inventory

Current camera box type inventory in checked-in snapshot:
- `vehicle`
- `pedestrian`
- `sign`
- `cyclist`

#### Raw semantic inventory relevant to future review

Checked-in snapshot also shows semantic / panoptic types such as:
- `TYPE_CAR`
- `TYPE_TRUCK`
- `TYPE_BUS`
- `TYPE_OTHER_LARGE_VEHICLE`
- `TYPE_BICYCLE`
- `TYPE_MOTORCYCLE`
- `TYPE_PEDESTRIAN`
- `TYPE_CYCLIST`
- `TYPE_MOTORCYCLIST`
- `TYPE_CONSTRUCTION_CONE_POLE`
- `TYPE_POLE`
- `TYPE_SIGN`
- `TYPE_TRAFFIC_LIGHT`
- `TYPE_ROAD`
- `TYPE_LANE_MARKER`
- `TYPE_ROAD_MARKER`

#### Current implementation usage

Current converter (`tools/data_analysis/wod/convert_wod_pv26.py`):
- rows without `camera_segmentation` are skipped for OD export
- segmentation-backed rows:
  - decode panoptic into `semantic_id + instance_id`
  - derive minimal axis-aligned 2D boxes per instance for selected thing classes
  - current 7-class remap:
    - `TYPE_CAR|TYPE_TRUCK|TYPE_BUS|TYPE_OTHER_LARGE_VEHICLE|TYPE_TRAILER -> vehicle`
    - `TYPE_BICYCLE|TYPE_MOTORCYCLE|TYPE_CYCLIST|TYPE_MOTORCYCLIST -> bike`
    - `TYPE_PEDESTRIAN -> pedestrian`
    - `TYPE_CONSTRUCTION_CONE_POLE -> traffic_cone`
    - `TYPE_PEDESTRIAN_OBJECT -> obstacle`
    - `TYPE_TRAFFIC_LIGHT -> traffic_light`
    - `TYPE_SIGN|TYPE_POLE -> sign_pole`
- manifest marks exported WOD OD as `det_label_scope=full`
- RM:
  - `TYPE_ROAD -> DA`
  - `TYPE_LANE_MARKER -> rm_lane_marker`
  - `TYPE_ROAD_MARKER -> rm_road_marker_non_lane`
- current converter still does not provide:
  - `rm_stop_line`
  - `lane_subclass`

#### Active 7-class merge

| Raw Waymo source label | Proposed 7-class |
|---|---|
| `vehicle`, `TYPE_CAR`, `TYPE_TRUCK`, `TYPE_BUS`, `TYPE_OTHER_LARGE_VEHICLE` | `vehicle` |
| `pedestrian` | `pedestrian` |
| `cyclist`, `TYPE_BICYCLE`, `TYPE_MOTORCYCLE`, `TYPE_CYCLIST`, `TYPE_MOTORCYCLIST` | `bike` |
| `TYPE_CONSTRUCTION_CONE_POLE` | `traffic_cone` |
| `TYPE_TRAFFIC_LIGHT` | `traffic_light` |
| `sign`, `TYPE_SIGN`, `TYPE_POLE` | `sign_pole` |

#### Risk / open question

- `TYPE_CONSTRUCTION_CONE_POLE`는 cone/pole 혼합 이름이라 `traffic_cone` 매핑 품질 확인이 필요하다.
- `TYPE_POLE`를 `sign_pole`로 넣으면 일반 pole도 함께 들어올 수 있다.
- WOD는 `vehicle / bike / pedestrian / traffic_cone / traffic_light / sign_pole` coverage가 강하지만, generic `obstacle` coverage는 BDD보다 약하다.

## 7) Road-Marking Harmonization Summary

This section records how each source distinguishes lane / road-marker semantics today.

| Dataset | lane_marker signal | road_marker_non_lane signal | stop_line signal | lane_subclass signal | Notes |
|---|---|---|---|---|---|
| BDD100K | lane white/yellow single/double + other | crosswalk, curb | explicit if present | white/yellow x solid/dashed from attributes | current base source |
| ETRI | `wh*`, `ye*`, `bl*`, guidance line, lane divider | crosswalk, arrows, numbers, speed bump, box junction, parking, etc. | explicit `stop line` | `whsol`, `whdot`, `yesol`, `yedot` | rich polygon inventory |
| RLMD | solid/dashed white/yellow/red, channelizing line | crosswalk, arrows, icons, parking lot, motor zones, etc. | explicit `stop line` | white/yellow x solid/dashed only | palette mask source |
| Waymo | `TYPE_LANE_MARKER` | `TYPE_ROAD_MARKER` | unavailable in current converter | unavailable | coarse RM only |
| Cityscapes | unavailable in current policy | unavailable in current policy | unavailable | unavailable | separate source needed |
| KITTI-360 | pending review | pending review | pending review | unavailable | planned |

## 8) Current Active Decisions vs Review Decisions

### 8.1 Current active implementation decisions

These remain true until code changes land.
1. Current code uses coarse 7-class OD.
2. Current code still exports `classmap-v3`.
3. Current BDD / ETRI / RLMD / WOD converters keep their present behavior.
4. WOD OD export is segmentation-backed and `det_label_scope=full`.

### 8.2 Provisional review decisions

These are still subject to ongoing review even though the baseline has moved to 7-class.
1. Keep `traffic_light` separate from `sign_pole` unless source quality proves it is not sustainable.
2. Keep `traffic_cone` separate from generic `obstacle` unless source quality proves it is not sustainable.
3. Do not create an `others` class unless source inventory review proves it is necessary and well-defined.
4. Revisit every source, including BDD100K, as a remap candidate rather than assuming any single source taxonomy should stay canonical.

## 9) Read-Only Raw -> Unified Output Mapping

원본 데이터셋은 read-only로 취급하고, 변환 결과만 `out_root`에 생성한다.
입력 구조가 달라도 출력 구조는 PV26로 동일하다.

공통 결과 디렉터리:
- `images/<split>/<sample_id>.jpg`
- `labels_det/<split>/<sample_id>.txt`
- `labels_seg_da/<split>/<sample_id>.png`
- `labels_seg_rm_lane_marker/<split>/<sample_id>.png`
- `labels_seg_rm_road_marker_non_lane/<split>/<sample_id>.png`
- `labels_seg_rm_stop_line/<split>/<sample_id>.png`
- `labels_seg_rm_lane_subclass/<split>/<sample_id>.png`
- `labels_semantic_id/<split>/<sample_id>.png` (optional)
- `meta/split_manifest.csv`
- `meta/conversion_report.json`
- `meta/source_stats.csv`
- `meta/checksums.sha256`

## 10) What Must Be Decided Before Code Changes

1. Is the provisional 7-class OD taxonomy sufficient after reviewing all raw source inventories?
2. Does any dataset require an extra class beyond the draft 7?
3. Should `train`, `trailer`, `caravan`, `other vehicle`, and unknown construction-like labels be merged into `vehicle` or `obstacle`?
4. Can any current source supply `traffic_light` OD reliably enough to justify keeping it separate?
5. Does any source besides BDD provide trustworthy `traffic_cone` or cone-equivalent OD?
6. Once the target taxonomy is locked, converters, labels, loss, eval, and checkpoint metrics must be updated together.
