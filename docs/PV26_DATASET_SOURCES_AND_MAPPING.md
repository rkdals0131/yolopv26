# PV26 Dataset Sources & Mapping

- Doc version: `v1.1`
- Last reviewed: `2026-03-05`
- Scope: local `datasets/` inventory + per-source mapping notes for PV26 Type‑A
- See also:
  - Product / runtime contract: `docs/PV26_PRD.md`
  - Conversion output contract: `docs/PV26_DATASET_CONVERSION_SPEC.md`

## 1) Why this doc

PV26는 여러 데이터셋을 섞어 학습해야 한다. 데이터셋마다 라벨 스키마와 클래스 정의가 달라서,
학습 단계에서 원천 포맷을 직접 파싱하면 유지보수/재현성이 깨지고 “없는 라벨을 background로 오해”하는 문제가 발생한다.

이 문서는 다음을 **정성(qualitative) 기준**으로 유지한다.
1. 로컬에 무엇이 있는지(인벤토리 refresh 루틴)
2. 각 데이터셋이 PV26 태스크(OD/DA/RM/lanesub/semantic_id)에 무엇을 기여할 수 있는지
3. 데이터셋별 매핑/어댑터의 구현 상태(implemented/planned/not-possible)

> 커버리지 숫자/표는 시간이 지나면 바로 구버전이 되므로, 문서에 하드코딩하지 않는다.

## 2) PV26 canonical tasks (Type‑A)

- **OD (Detection)**: YOLO txt (`labels_det/*/*.txt`)
- **DA (Drivable Area)**: binary mask `{0,1,255}` (`labels_seg_da/*/*.png`)
- **RM (Road Marking)**: 3× binary masks `{0,1,255}`
  - `rm_lane_marker`
  - `rm_road_marker_non_lane`
  - `rm_stop_line` (일부 데이터셋은 supervision 부재 가능)
- **Lane subclass (sidecar)**: mono8 `{0,1,2,3,4,255}` (`labels_seg_rm_lane_subclass/*/*.png`)
- **semantic_id (optional)**: mono8 class-id map (`labels_semantic_id/*/*.png`, `255` 금지)

## 3) Offline normalization policy (must)

- 학습 로더는 원천 라벨(JSON/polygon/parquet 등)을 직접 파싱하지 않는다.
- 모든 데이터셋은 학습 전, Section 2의 형식(`jpg + txt + png`)으로 **오프라인 정규화**한다.
- Partial label 정책:
  - 라벨이 없는 태스크/채널은 절대 `0`으로 채우지 않는다.
  - 라벨 부재는 “전 픽셀 `255`” + `has_* = 0`으로 표현하고, 학습 시 loss에서 제외한다.
- polygon/json 계열은 “룩업 테이블(원천 클래스 → PV26 채널/ignore)”을 먼저 잠근 뒤 rasterization 한다.

## 4) Inventory refresh (no hardcoded counts)

`datasets/`는 보통 외부 스토리지로 연결된 심볼릭 링크다(환경에 따라 다름).

인벤토리 리포트 생성(회의 전 표준 루틴):
```bash
python tools/data_analysis/bdd/dataset_label_inventory.py --out /tmp/dataset_label_inventory.json
```

과거 스냅샷(참고용):
- `docs/legacy/snapshots/Dataset_Profile_2026-02-19.md`

## 5) Source → PV26 mapping status (qualitative)

상태 표기:
- `implemented`: PV26 adapter 스크립트가 있고, `validate_pv26_dataset.py` 계약으로 검증 가능
- `planned`: 어댑터가 아직 없거나(또는 룩업테이블 미확정), 구현이 필요
- `not-possible`: 원천 데이터가 해당 supervision을 제공하지 않음(또는 coarse하여 목적 불일치)

|Dataset|OD|DA|RM lane_marker|RM road_marker_non_lane|RM stop_line|lane_subclass|semantic_id|
|---|---|---|---|---|---|---|---|
|BDD100K|implemented|implemented|implemented|implemented|implemented|implemented|implemented|
|Cityscapes|planned|planned|not-possible|not-possible|not-possible|not-possible|planned|
|ETRI|not-possible|implemented|implemented|implemented|implemented|implemented|not-possible|
|KITTI-360|planned|planned|planned|planned|planned|not-possible|planned|
|RLMD|not-possible|not-possible|implemented|implemented|implemented|implemented|not-possible|
|Waymo (Perception v2)|implemented|implemented|implemented|implemented|not-possible|not-possible|planned|

Notes:
- Waymo OD는 subset class만 매핑한다(`det_label_scope=subset`, `det_annotated_class_ids=0,4,5,10`).
- BDD100K `semantic_id`는 모든 선행 채널이 supervised이고 compose 가능할 때만 export된다.
- supervision이 없는 채널은 all-`255` + `has_*=0`을 유지한다.

## 6) Dataset notes (adapter tips)

### 6.1 BDD100K

- 자산: 이미지 + det JSON + drivable id mask + lane poly2d
- PV26 변환: `tools/data_analysis/bdd/convert_bdd_type_a.py` (implemented)
- RoadMarking:
  - lane poly2d를 rasterize하여 `rm_*` 마스크를 만든다.
  - stop_line은 명시 클래스가 없는 경우가 많아, 채널이 all-`255`로 남을 수 있다.

### 6.2 Cityscapes

- 자산: semantic/instance mask + polygon JSON
- 정책(초기):
  - DA는 semantic에서 `road/parking` 기반으로 파생 가능(planned)
  - RoadMarking은 기본 미제공으로 보고 `has_rm_*=0`(not-possible)

### 6.3 ETRI

- 자산: Cityscapes-like polygon JSON
- PV26 변환: `tools/data_analysis/etri/convert_etri_type_a.py` (implemented)
- 정책(현재):
  - polygon 라벨을 오프라인 rasterization으로 DA/RM/lane_subclass 마스크로 변환한다.
  - 검출 라벨은 원천 미제공이므로 `has_det=0`, `labels_det/*.txt`는 빈 파일로 기록한다.
  - `whsol/whdot/yesol/yedot`만 lane_subclass(1..4)로 매핑하고, 기타 lane-like는 subclass에서 `255(ignore)` 처리한다.

### 6.4 KITTI-360

- 자산: semantic mask가 대량 존재
- 정책(초기):
  - semantic id → (DA/RM) 매핑 룩업 테이블을 먼저 잠근 뒤 변환(planned)

### 6.5 RLMD

- 자산: RGB palette road marking mask(+ palette csv)
- PV26 변환: `tools/data_analysis/rlmd/convert_rlmd_type_a.py` (implemented)
- 정책(현재):
  - palette 기반 remap으로 RM 채널/stop_line/lane_subclass를 생성한다.
  - DA/OD는 원천 미제공으로 `has_da=0`, `has_det=0`을 기록한다.
  - RLMD-AC는 라벨이 존재하는 split만 포함한다.

### 6.6 Waymo Open Dataset (Perception v2)

- 자산: parquet 기반(디코딩 필요)
- PV26 변환: `tools/data_analysis/wod/convert_wod_type_a.py` (implemented, minimal-first)
- coarse class 정책(현재):
  - `TYPE_ROAD` → DA
  - `TYPE_LANE_MARKER` → `rm_lane_marker`
  - `TYPE_ROAD_MARKER` → `rm_road_marker_non_lane`
  - stop line과 lane_subclass는 dedicated class가 없어 `not-possible`
  - OD는 subset class만 매핑(`vehicle/pedestrian/cyclist/sign`)
  - parquet는 스크립트가 `pyarrow.parquet`로 직접 읽는다(수동 decode 필요 없음).

샘플 디코딩(세그 있는 프레임만):
```bash
python tools/data_analysis/wod/extract_wod_v2_sample.py \
  --training-root datasets/WaymoOpenDataset/wod_pv2_minimal_1ctx/training \
  --out-root /tmp/wod_decoded \
  --require-seg
```

## 7) Lane subclass mapping rules (canonical)

- **BDD100K (implemented)**:
  - white + solid → 1
  - white + dashed → 2
  - yellow + solid → 3
  - yellow + dashed → 4
  - 기타 lane-like는 `rm_lane_marker`에는 포함하되 `lane_subclass`에서는 `255(ignore)`
- **RLMD (implemented)**: palette 클래스 기반 direct remap (double-line 처리 정책은 어댑터에서 고정)
- **ETRI (implemented)**: `whsol/whdot/yesol/yedot` → (1..4) remap, 나머지 lane-like는 `255(ignore)`
- **Waymo (not-possible)**: coarse class로는 (white/yellow × solid/dashed) supervision 불가

## 8) Immediate policy decisions (locked)

1. RoadMarking binary 채널은 `rm_lane_marker`, `rm_road_marker_non_lane`, `rm_stop_line` 3채널을 유지한다.
2. lane 세분 supervision은 `rm_lane_subclass`(mono8: white/yellow × solid/dashed)로 sidecar 운영한다.
3. `rm_stop_line ⊂ rm_road_marker_non_lane`을 전제로 한다.
4. json/polygon 라벨은 “룩업 테이블 + 오프라인 rasterization” 후에만 학습에 넣는다.
5. 데이터셋에 없는 태스크/채널은 `255(ignore)` + `has_*=0`으로 loss에서 제외한다.

## 9) Read-only raw → unified output mapping (implemented)

원본 데이터셋은 read-only로 취급하고, 변환 결과만 `out_root`에 생성한다.
입력 구조가 달라도 출력 구조는 PV26 Type-A로 동일하다.

공통 결과 디렉터리(모든 데이터셋):
- `images/<split>/<sample_id>.jpg`
- `labels_det/<split>/<sample_id>.txt`
- `labels_seg_da/<split>/<sample_id>.png`
- `labels_seg_rm_lane_marker/<split>/<sample_id>.png`
- `labels_seg_rm_road_marker_non_lane/<split>/<sample_id>.png`
- `labels_seg_rm_stop_line/<split>/<sample_id>.png`
- `labels_seg_rm_lane_subclass/<split>/<sample_id>.png`
- `labels_semantic_id/<split>/<sample_id>.png` (optional)
- `meta/split_manifest.csv`, `meta/conversion_report.json`, `meta/source_stats.csv`, `meta/checksums.sha256`

|Dataset|raw image source|OD source → `labels_det`|seg/rm source → `labels_seg_*`|remarks|
|---|---|---|---|---|
|BDD100K|`bdd100k_images_100k/100k`|det JSON|drivable id + lane poly2d|`semantic_id`는 조건 충족 시에만 export|
|ETRI|Mono/Multi 이미지 폴더|없음(빈 txt + `has_det=0`)|polygon JSON rasterization|`out of roi`는 mask ignore(255) 반영|
|RLMD|`RLMD_1080p/images/*` (+ RLMD-AC)|없음(빈 txt + `has_det=0`)|palette RGB label png remap|DA 없음(`has_da=0`, all-255)|
|Waymo v2|`camera_image/*.parquet`|`camera_box/*.parquet`(subset class)|`camera_segmentation/*.parquet`|WOD parquet는 스크립트가 직접 로딩/디코딩|

## 10) WOD parquet handling note

`tools/data_analysis/wod/convert_wod_type_a.py`는 WOD parquet를 직접 읽는다.

- `pyarrow.parquet.ParquetFile(...).iter_batches(...)`로 row batch 순회
- 컴포넌트별 parquet를 key(`context`, `timestamp`, `camera`)로 조인
- 이미지 바이트는 JPEG로 저장, 세그 panoptic은 semantic id로 환산해 PV26 채널로 매핑

즉, 사용자는 parquet를 필요한 범위만 내려받으면 되고, 별도 수동 decode 파이프라인은 필수 아니다.

## 11) WOD external references (official)

WOD v2 내부 구조/형식 확인 시 우선 보는 공식 레퍼런스:

- Waymo Perception 페이지(공식): modular format이 Apache Parquet 기반이며 component 분리 구조임을 명시  
  https://waymo.com/intl/zh-cn/open/data/perception
- Waymo Download 페이지(공식): Perception v2.0.1(modular, without maps) 배포 정보  
  https://waymo.com/intl/fil/open/download
- Waymo Open Dataset GitHub(공식): 저장소 루트 및 릴리즈 히스토리  
  https://github.com/waymo-research/waymo-open-dataset  
  https://github.com/waymo-research/waymo-open-dataset/releases
- v2 tutorial notebook(공식): parquet component를 직접 로딩하고(`camera_image`, `camera_box` 등) join하는 예제  
  https://github.com/waymo-research/waymo-open-dataset/blob/master/tutorial/tutorial_v2.ipynb  
  https://raw.githubusercontent.com/waymo-research/waymo-open-dataset/refs/heads/master/tutorial/tutorial_v2.ipynb

운영 메모:
- 현재 우리 컨버터는 위 v2 tutorial의 접근 방식과 동일하게 parquet를 직접 읽어서 key(`context`, `timestamp`, `camera`) 기준으로 결합한다.
- 공식 사이트의 기본 `/open/...` 경로는 환경에 따라 로그인 리다이렉트가 걸릴 수 있어, 접근 가능한 로케일 경로를 함께 기록했다.
