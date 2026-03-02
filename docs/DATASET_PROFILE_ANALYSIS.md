# YOLO PV26 Dataset Profile Analysis (2026-02-19)

- Doc version: `v0.3`
- Date: `2026-02-19`
- Scope: local `datasets/` inventory + label-space reasoning + offline normalization policy
- Non-goals: conversion pipeline implementation details (converter/adapter code), model architecture code
 
Changelog:
- `v0.3`: Cityscapes 원본 이미지(`leftImg8bit`) 로컬 확보 반영 + 커버리지 스냅샷 섹션 추가

## 1. Why This Doc

YOLO PV26는 여러 공개/자체 데이터셋을 섞어 학습해야 한다. 문제는 데이터셋마다:
1. 라벨 스키마가 다르고(`png mask`, `json polygons`, `parquet panoptic` 등),
2. 클래스 정의가 다르고(특히 road marking),
3. “없는 라벨”을 배경으로 채우면 오학습이 발생한다.

이 문서는 “지금 로컬에 실제로 무엇이 있는지”와 “학습에 쓰기 위해 어떤 오프라인 정규화가 필요한지”를 결정한다.

## 2. Canonical Tasks (MVP)

MVP 학습 태스크는 3개로 고정한다.
1. `OD` (detection)
2. `Drivable` (binary mask)
3. `RoadMarking` (multi-channel mask, sigmoid)

RoadMarking 채널(회의 확정, 최소 세트):
1. `rm_lane_marker`
2. `rm_road_marker_non_lane`
3. `rm_stop_line`

채널 관계:
- `rm_stop_line ⊂ rm_road_marker_non_lane` (멀티라벨이므로 픽셀에서 동시 활성 허용)

## 3. Offline Normalization Policy (핵심)

학습 로더는 원천 라벨 스키마를 직접 파싱하지 않는다.
모든 데이터셋은 학습 전에 “원본 이미지 + 픽셀 마스크(PNG)” 형태로 오프라인 정규화한다.

정규화 산출물(채널별 mask):
- `labels_seg_da`: `{0,1,255}` (`255=ignore`)
- `labels_seg_rm_lane_marker`: `{0,1,255}`
- `labels_seg_rm_road_marker_non_lane`: `{0,1,255}`
- `labels_seg_rm_stop_line`: `{0,1,255}`

부분 라벨 정책:
1. 라벨이 없는 태스크/채널은 절대 `0`으로 채우지 않는다.
2. 라벨 부재는 “전 픽셀 `255`” + `has_* = 0`으로 표현하고, 학습 시 loss에서 제외한다.

### 3.1 JSON/Polygon 계열 라벨 정책

`json polygons`처럼 픽셀 마스크가 아닌 라벨은 다음을 포함하는 “룩업 테이블”을 먼저 만든 뒤 변환한다.
1. 원천 클래스(문자열/ID) -> PV26 채널 매핑
2. ignore/void 처리 규칙(`255`)
3. 폴리곤 rasterization 규칙(thickness, fill, antialias 금지 등)

변환 결과는 “원본 이미지 파일 + 채널별 pixel mask PNG” 쌍으로만 유지한다.

## 4. Local Dataset Inventory (Observed)

`datasets/`는 심볼릭 링크이며 실제 경로는 `/home/user1/Storage/seg_dataset`(= `/mnt/data/data/seg_dataset`)이다.

로컬에 존재하는 데이터셋:
1. `BDD100K`
2. `Cityscapes`
3. `ETRI`
4. `KITTI-360`
5. `RLMD`
6. `WaymoOpenDataset`

로컬 인벤토리/커버리지 리포트 생성:
```bash
python tools/data_analysis/bdd/dataset_label_inventory.py --out /tmp/dataset_label_inventory.json
```

## 5. Coverage Snapshot (회의용, 2026-02-19)

아래 표는 `tools/data_analysis/bdd/dataset_label_inventory.py` 실행 결과를 문서에 옮긴 “스냅샷”이다.
숫자는 데이터셋 다운로드/정규화 진행에 따라 계속 변하므로, 회의 전에는 리포트를 다시 뽑고 표도 갱신한다.

|Dataset|Drivable(frames)|rm_lane_marker(frames)|rm_road_marker_non_lane(frames)|rm_stop_line(frames)|비고|
|---|---:|---:|---:|---:|---|
|BDD100K|80000|0|0|0|RoadMarking은 별도 소스/정규화 필요|
|Cityscapes|3475|0|0|0|train/val만 라벨 존재(gtFine)|
|KITTI-360|N/A|N/A|N/A|N/A|semantic->(drivable/road-marking) 매핑 룩업테이블 먼저 필요|
|RLMD|N/A|2120|1479|732|RGB palette 기반 변환 가능|
|ETRI|N/A|908|644|278|polygon rasterization 필요(룩업테이블 필수)|
|Waymo(minimal_1ctx)|66|22|35|0|stop line은 dedicated class가 없어 unknown(`has_rm_stop_line=0`)|

## 6. Dataset Notes (What It Can Provide)

### 6.1 BDD100K

로컬 자산:
- 이미지(`bdd100k_images_100k`) + drivable mask + semantic seg mask + OD json

기여 가능:
- `OD`, `Drivable`은 바로 사용 가능
- RoadMarking은 BDD lane-mark 소스가 mask가 아닌 경우가 많아(형식 다양) 정규화 룰을 먼저 고정해야 한다

### 6.2 Cityscapes

로컬 자산:
- `leftImg8bit` 원본 RGB 이미지 있음(train/val)
- `gtFine` 라벨 있음(train/val)
- test split은 로컬에 없으므로 학습에는 train/val만 사용 가능

정책:
- 드라이버블(road/parking) 보강 용도 후보(원본 이미지까지 갖춰져서 학습 샘플 생성 가능)
- RoadMarking은 기본 미제공으로 보고 채널은 `has_rm_*=0` 처리

### 6.3 ETRI

로컬 자산:
- Cityscapes-like polygon JSON: `*_gtFine_polygons.json`
- 일부 이미지(예: mosaic) 존재

정책:
- polygon 라벨은 반드시 오프라인 rasterization으로 PNG mask를 만든다
- `stop line`이 문자열 라벨로 명시되어 있어 `rm_stop_line` supervision 소스로 유용

### 6.4 KITTI-360

로컬 자산:
- `image_01/semantic` 및 `semantic_rgb` 마스크가 대량 존재
- raw perspective download는 로컬에 없음(현재 상태 기준)

정책:
- semantic id -> (drivable/road-marking) 매핑 룩업 테이블을 먼저 만들어야 한다

### 6.5 RLMD

로컬 자산:
- 이미지(`jpg`) + RGB 마스크(`png`) + palette(`rlmd.csv`)

정책:
- `rlmd.csv` 기반으로 RGB->클래스 룩업 테이블이 이미 존재하므로 채널 변환이 비교적 쉽다
- lane boundary류와 arrow/text/box-junction 등을 분리해서:
  - lane boundary -> `rm_lane_marker`
  - 나머지 road marking -> `rm_road_marker_non_lane`
  - stop line -> `rm_stop_line` (+ `rm_road_marker_non_lane`)

### 6.6 Waymo Open Dataset (Perception v2)

로컬 자산:
- SDK/문서: `WaymoOpenDataset/wod-gh`
- minimal context(parquet): `WaymoOpenDataset/wod_pv2_minimal_1ctx/training/*/*.parquet`

camera segmentation class 관점(중요):
- `TYPE_ROAD=20` -> `drivable`
- `TYPE_LANE_MARKER=21` -> `rm_lane_marker`
- `TYPE_ROAD_MARKER=22` -> `rm_road_marker_non_lane`
- `TYPE_SIDEWALK=23`은 MVP에서 선택사항(현재는 채널로 운영하지 않음)
- Waymo는 stop line을 dedicated class로 제공하지 않으므로 `rm_stop_line`은 기본 unknown으로 둔다(`255`, `has_rm_stop_line=0`)

샘플 디코딩(세그 있는 프레임만 뽑기; 샘플링 바이어스 방지):
```bash
python tools/data_analysis/wod/extract_wod_v2_sample.py \
  --training-root /home/user1/Storage/seg_dataset/WaymoOpenDataset/wod_pv2_minimal_1ctx/training \
  --out-root /tmp/wod_decoded \
  --require-seg
```

## 7. What To Keep Updated

회의/의사결정에 필요한 숫자는 “문서에 하드코딩”하지 말고 리포트를 갱신한다.
1. `/tmp/dataset_label_inventory.json` (채널별 커버리지)
2. WOD 디코딩 산출물(샘플 확인용)

## 8. Immediate Policy Decisions (Locked)

1. RoadMarking은 `rm_lane_marker`, `rm_road_marker_non_lane`, `rm_stop_line` 3채널로 시작한다.
2. `rm_stop_line ⊂ rm_road_marker_non_lane`을 전제로 한다.
3. json/polygon 라벨은 “룩업 테이블 + 오프라인 rasterization” 후에만 학습에 넣는다.
4. 데이터셋에 없는 태스크/채널은 `255(ignore)` + `has_*=0`으로 loss에서 제외한다.
