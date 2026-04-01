# Data And Standardization

원본 AIHUB/BDD 디렉터리 구조, 현재 로컬 subset, 파일명 규약은 [3A_RAW_DATASET_LAYOUTS.md](3A_RAW_DATASET_LAYOUTS.md)를 먼저 본다.

## 데이터 소스

- AIHUB
  - `차선-횡단보도 인지 영상(수도권)`
  - `신호등-도로표지판 인지 영상(수도권)`
  - `도로장애물·표면 인지 영상(수도권)` for `traffic_cone / obstacle` only
- BDD100K
  - 7-class OD 보강용
- future sources
  - AIHUB OD source 추가 가능

## AIHUB bootstrap canonicalization 원칙

- 원본 dataset는 intact 유지
- output은 `seg_dataset/pv26_od_bootstrap/canonical/aihub_standardized` 아래 별도 생성
- 원본 좌표계 유지
- image는 hardlink 우선, 실패 시 copy
- det label과 scene label을 분리 저장
- meta 리포트를 함께 저장
- 기존 output이 있으면 resume scan으로 재사용하고, 필요 시 `force_reprocess`로 전체 재생성한다
- partial failure는 즉시 중단 대신 failure manifest와 QA summary로 남긴다
- 기본 경로는 repo-relative로 계산하고 `PV26_REPO_ROOT`, `PV26_SEG_DATASET_ROOT`, `PV26_AIHUB_ROOT`, `PV26_AIHUB_OUTPUT_ROOT` env override를 허용한다
- image size probing은 PIL 우선, 실패 시 `identify` fallback을 사용한다

## AIHUB obstacle source policy

- source: `도로장애물·표면 인지 영상(수도권)`
- V1 목적은 `traffic_cone / obstacle` supervision 보강이다.
- detector remap
  - `Traffic cone -> traffic_cone`
  - `Animals(Dolls) / Garbage bag & sacks / Box / Stones on road / Construction signs & Parking prohibited board -> obstacle`
- current exclusion
  - `Person`
  - `Manhole`
  - `Pothole on road`
  - `Filled pothole`
- 즉 V1에서는 road-surface defect task로 확장하지 않고 detector의 missing obstacle classes만 채운다.

## BDD100K bootstrap canonicalization 원칙

- 원본 dataset는 intact 유지
- output은 `seg_dataset/pv26_od_bootstrap/canonical/bdd100k_det_100k` 아래 별도 생성
- detector-only canonical source로 사용
- BDD는 `vehicle / bike / pedestrian` 보강 source로만 사용하고 `traffic_light / sign`는 canonical output에서 제외한다
- image는 hardlink 우선, 실패 시 copy
- det label과 minimal scene label을 함께 저장
- BDD context metadata는 scene JSON에 보존하되 TL supervision은 끈다
- 기존 output이 있으면 resume scan으로 재사용하고, 필요 시 `force_reprocess`로 전체 재생성한다
- partial failure는 failure manifest와 QA summary로 남긴다
- 기본 경로는 repo-relative로 계산하고 `PV26_REPO_ROOT`, `PV26_SEG_DATASET_ROOT`, `PV26_BDD_ROOT`, `PV26_BDD_OUTPUT_ROOT` env override를 허용한다
- image size probing은 host tool 의존도를 줄이기 위해 PIL 우선 probing을 사용한다

## 현재 canonical output

```text
seg_dataset/pv26_od_bootstrap/canonical/aihub_standardized/
  images/<split>/*
  labels_det/<split>/*.txt
  labels_scene/<split>/*.json
  meta/
    class_map_det.yaml
    class_map_scene.yaml
    conversion_report.json
    conversion_report.md
    failure_manifest.json
    failure_manifest.md
    qa_summary.json
    qa_summary.md
    source_inventory.json
    source_inventory.md
    debug_vis/
      <split>/*.png
      index.json
```

```text
seg_dataset/pv26_od_bootstrap/canonical/bdd100k_det_100k/
  images/<split>/*
  labels_det/<split>/*.txt
  labels_scene/<split>/*.json
  meta/
    class_map_det.yaml
    class_map_scene.yaml
    conversion_report.json
    conversion_report.md
    failure_manifest.json
    failure_manifest.md
    qa_summary.json
    qa_summary.md
    source_inventory.json
    source_inventory.md
    debug_vis/
      <split>/*.png
      index.json
```

## detector taxonomy

- `vehicle`
- `bike`
- `pedestrian`
- `traffic_cone`
- `obstacle`
- `traffic_light`
- `sign`

## traffic light attribute policy

- detector class는 generic `traffic_light`
- attribute bits는 `red / yellow / green / arrow`
- `left_arrow`, `others_arrow`는 `arrow=1`
- `off`는 4 bit 모두 0
- non-car, `x_light`, multi-color는 attr loss mask
- BDD100K는 signal class 자체를 canonical output에서 제외하고 TL attr supervision source로도 쓰지 않는다

## lane geometry policy

- lane color
  - `white_lane`
  - `yellow_lane`
  - `blue_lane`
- lane type metadata
  - `solid`
  - `dotted`
- extra geometry
  - `stop_line`
  - `crosswalk`

## resize 정책

- standardization 단계에서는 image를 `800x608`로 굽지 않는다.
- training/inference loader에서만 online transform을 적용한다.
- 이유
  - raw-space canonical dataset 유지
  - 입력 크기 변경 시 dataset 재생성 방지
  - 라벨 좌표계 이중 관리 방지

## training input policy

- dataset raw: variable resolution
- vehicle camera reference raw: `800x600`
- network input: `800x608`
- online policy: standardized dataset raw에서 `800x608`으로 직접 resize + pad를 적용
- vehicle camera가 이미 `800x600`이면 같은 contract가 `800x600 -> 800x608` pad-only로 축약된다
- 학습과 추론은 같은 preprocessing contract를 공유해야 한다.

## AIHUB 문서/실데이터 운영 규칙

- source dataset README는 각 원본 루트에 유지
- standardization output meta가 dataset understanding의 1차 레퍼런스다
- debug overlay는 사람이 conversion 품질을 빠르게 보는 QA 수단이다
- full run 전에는 `failure_manifest`와 `qa_summary`를 먼저 보고, 이후 debug overlay를 눈으로 확인한다

## BDD100K 운영 규칙

- source dataset README는 BDD100K 루트에 유지
- 표준화는 `bdd100k_images_100k/100k`와 `bdd100k_labels/100k`만을 canonical source로 사용한다
- `traffic light`, `traffic sign`는 AIHUB signal source가 담당하므로 BDD canonical output에서 제외한다
- `lane/*`, `area/*` 등은 detector canonical set에서 제외하고 held reason으로만 집계한다
- full run 전에는 `failure_manifest`와 `qa_summary`를 먼저 보고, 이후 debug overlay를 눈으로 확인한다
