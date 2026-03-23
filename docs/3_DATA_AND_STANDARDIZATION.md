# Data And Standardization

## 데이터 소스

- AIHUB
  - `차선-횡단보도 인지 영상(수도권)`
  - `신호등-도로표지판 인지 영상(수도권)`
- BDD100K
  - 7-class OD 보강용
- future sources
  - AIHUB OD source 추가 가능

## AIHUB standardization 원칙

- 원본 dataset는 intact 유지
- output은 `seg_dataset/pv26_aihub_standardized` 아래 별도 생성
- 원본 좌표계 유지
- image는 hardlink 우선, 실패 시 copy
- det label과 scene label을 분리 저장
- meta 리포트를 함께 저장

## 현재 canonical output

```text
seg_dataset/pv26_aihub_standardized/
  images/<split>/*
  labels_det/<split>/*.txt
  labels_scene/<split>/*.json
  meta/
    class_map_det.yaml
    class_map_scene.yaml
    conversion_report.json
    conversion_report.md
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

- runtime raw: `800x600`
- network input: `800x608`
- online policy: resize + pad 또는 canonical pad-only policy를 일관되게 적용
- 학습과 추론은 같은 preprocessing contract를 공유해야 한다.

## AIHUB 문서/실데이터 운영 규칙

- source dataset README는 각 원본 루트에 유지
- standardization output meta가 dataset understanding의 1차 레퍼런스다
- debug overlay는 사람이 conversion 품질을 빠르게 보는 QA 수단이다
