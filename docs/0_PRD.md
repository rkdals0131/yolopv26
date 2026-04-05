# PV26 PRD

## 문서 목적

이 문서는 현재 `yolopv26` 저장소의 최상위 제품 요구사항 문서다. 이 문서와 같은 번호 체계의 문서들이 이전 pivot 문서 세트를 완전히 대체한다.

## 프로젝트 목표

- PV26의 V1 목표는 단일 모델 학습 저장소를 만드는 것이다.
- 모델은 다음을 동시에 다룬다.
  - 7-class object detection
  - traffic light 4-bit attribute prediction
  - lane polyline prediction
  - stop-line polyline prediction
  - crosswalk polygon prediction
- 실제 구현은 `Ultralytics YOLO v26 nano`의 공식 pretrained backbone/neck를 최대한 유지하면서, PV26 전용 head와 loss를 붙이는 방향으로 간다.

## 제품 범위

- 입력
  - standardized dataset raw image는 다양한 원본 해상도를 허용한다.
  - vehicle camera reference frame은 `800x600`이다.
  - 학습/추론 network input은 `800x608`으로 고정한다.
  - loader는 dataset raw에서 `800x608`으로 직접 변환한다.
  - runtime camera가 이미 `800x600`이면 같은 transform은 `800x600 -> 800x608` pad-only로 축약된다.
- 출력
  - detector class: `vehicle / bike / pedestrian / traffic_cone / obstacle / traffic_light / sign`
  - traffic light attribute: `red / yellow / green / arrow`
  - lane geometry: `white_lane / yellow_lane / blue_lane`
  - lane subtype metadata: `solid / dotted`
  - extra geometry: `stop_line`, `crosswalk`

## 데이터 범위

- OD
  - 기본 7-class OD는 BDD100K와 AIHUB traffic 계열을 함께 사용한다.
  - AIHUB OD source로 `도로장애물·표면 인지 영상(수도권)`을 함께 사용한다.
  - 이 source는 V1에서 `traffic_cone / obstacle` 보강만 담당한다.
  - `Person / Manhole / Pothole on road / Filled pothole`는 V1 detector supervision 범위에서 제외한다.
- traffic light
  - detection은 generic `traffic_light` bbox다.
  - 상태는 bbox에 종속된 4-bit attribute로 예측한다.
  - AIHUB traffic source가 핵심 supervision source다.
- lane
  - lane 학습과 추론은 AIHUB 기준 포맷을 중심으로 설계한다.
  - AIHUB의 색상, 타입, 정지선, 횡단보도 richness를 유지한다.

## 비목표

- V1에서 차선을 segmentation mask나 dense row-anchor task로 다시 바꾸지 않는다.
- V1에서 traffic light를 class explosion 방식으로 `tl_red_arrow`, `tl_green_arrow`처럼 detector class에 다 넣지 않는다.
- V1에서 raw dataset 자체를 오프라인 리사이즈하여 새 canonical image set으로 다시 굽지 않는다.
- V1에서 backbone/neck를 scratch로 처음부터 재학습하는 것을 기본 경로로 삼지 않는다.

## 핵심 결정

- backbone/neck는 `yolo26n` pretrained trunk reuse가 기본이다.
- custom task head는 PV26 쪽에서 새로 만든다.
- AIHUB standardization은 raw-space canonical dataset을 만든다.
- `800x608` letterbox/pad transform은 loader 단계에서 온라인으로 적용한다.
- 구현 전에 문서를 먼저 고정하고, 문서와 상태 tracker를 계속 갱신하면서 개발한다.

## 성공 조건

- AIHUB standardized scene/det dataset에서 loader가 안정적으로 sample을 뽑는다.
- loader sample contract와 transform contract가 문서와 코드에서 동일하다.
- pretrained trunk를 부분 로드한 PV26 model이 forward/backward를 통과한다.
- target encoder와 loss가 lane/TL/OD/stop-line/crosswalk를 동시에 처리한다.
- small regression dataset에서 loss가 정상적으로 감소한다.
- 문서와 구현 상태가 일치한다.

## 문서 맵

- [1_DEVELOPMENT_PHILOSOPHY.md](1_DEVELOPMENT_PHILOSOPHY.md)
- [2_SYSTEM_ARCHITECTURE.md](2_SYSTEM_ARCHITECTURE.md)
- [3_DATA_AND_STANDARDIZATION.md](3_DATA_AND_STANDARDIZATION.md)
- [3A_RAW_DATASET_LAYOUTS.md](3A_RAW_DATASET_LAYOUTS.md)
- [4_MODEL_ARCHITECTURE.md](4_MODEL_ARCHITECTURE.md)
- [4A_SAMPLE_AND_TRANSFORM_CONTRACT.md](4A_SAMPLE_AND_TRANSFORM_CONTRACT.md)
- [5_TARGETS_AND_LOSS.md](5_TARGETS_AND_LOSS.md)
- [6_TRAINING_AND_EVALUATION.md](6_TRAINING_AND_EVALUATION.md)
- [7_IMPLEMENTATION_PLAN.md](7_IMPLEMENTATION_PLAN.md)
- [8_TEST_PLAN_AND_CHECKLIST.md](8_TEST_PLAN_AND_CHECKLIST.md)
- [9_EXECUTION_STATUS.md](9_EXECUTION_STATUS.md)
- [10_GROUPED_OD_BOOTSTRAP_PLAN.md](10_GROUPED_OD_BOOTSTRAP_PLAN.md)
- [11_GIT_BRANCH_WORKFLOW.md](11_GIT_BRANCH_WORKFLOW.md)
- [12_PV26_LANE_ARCHITECTURE_AND_DATASET_REVIEW.md](12_PV26_LANE_ARCHITECTURE_AND_DATASET_REVIEW.md)
- [12A_GPT_REVIEW_ON_LANE_ARCH.md](12A_GPT_REVIEW_ON_LANE_ARCH.md)
- [13_ROAD_MARKING_HEAD_REWRITE_DESIGN.md](13_ROAD_MARKING_HEAD_REWRITE_DESIGN.md)
- [13A_ROAD_MARKING_HEAD_REWRITE_CHECKLIST.md](13A_ROAD_MARKING_HEAD_REWRITE_CHECKLIST.md)
- [14_TENSORBOARD_SURFACE.md](14_TENSORBOARD_SURFACE.md)
