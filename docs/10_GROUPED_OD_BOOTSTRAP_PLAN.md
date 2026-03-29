# Grouped OD Bootstrap Plan

## scope

- 이 문서는 OD bootstrap 계획만 다룬다.
- lane / stop-line / crosswalk 계획은 여기서 다루지 않는다.
- 목표는 `7-class detector` 문제를 `3개의 별도 독립 YOLO v26 모델`로 풀어내는 것이다.
- 이 3개 모델은 최종 배포 모델이 아니라 `데이터셋 전처리용 bootstrap detector`다.

## core decision

- YOLO v26 아키텍처는 수정하지 않는다.
- detector head를 쪼개지 않는다.
- 독립적인 YOLO v26 detector model을 `3개` 만든다.
- 각 모델은 자기에게 지정된 class와 지정된 dataset으로 먼저 파인튜닝한다.
- 이 단계에서 각 dataset의 원본 OD label은 ground truth로 취급한다.
- raw source label이 우선이다.
- calibration 단계는 별도 경로로 유지하고, teacher val split 기준 class policy를 보정한다.
- teacher train config는 `yolo26n`과 `yolo26s`를 둘 다 제공한다.

## three independent models

### 1. mobility model

- base: `yolo26n` 기본, `yolo26s` 대안 config도 제공
- classes: `vehicle`, `bike`, `pedestrian`
- first fine-tune dataset: `bdd100k_det_100k`

### 2. signal model

- base: `yolo26n` 기본, `yolo26s` 대안 config도 제공
- classes: `traffic_light`, `sign`
- first fine-tune dataset: `aihub_traffic_seoul`

### 3. obstacle model

- base: `yolo26n` 기본, `yolo26s` 대안 config도 제공
- classes: `traffic_cone`, `obstacle`
- first fine-tune dataset: `aihub_obstacle_seoul`

## training rule

- 각 모델은 official YOLO v26 detector training path를 그대로 사용한다.
- 각 모델은 자기 담당 class만 학습한다.
- 각 모델은 자기에게 지정된 dataset으로만 1차 파인튜닝을 수행한다.
- 각 모델 학습에서는 objectness negative를 활성화한다.
- 각 모델은 자기 담당 class에 대해 background와 foreground를 적극적으로 구분하도록 학습한다.
- mobility model은 BDD class만 학습한다.
- signal model은 AIHUB traffic class만 학습한다.
- obstacle model은 AIHUB obstacle class만 학습한다.

## initial class policy

- 초기 버전부터 class별 policy template을 두고 calibration run에서 score/NMS/min-size를 teacher val split 기준으로 보정한다.
- 최소한 아래는 초기에 바로 둔다.
  - class별 score threshold
  - class별 NMS 기준
  - class별 minimum box size
- `traffic_light`, `sign`, `traffic_cone`, `obstacle`는 같은 threshold로 밀지 않는다.
- obstacle 계열은 더 보수적으로 잡는다.
- calibration 결과와 `hard_negative_manifest.json`은 다음 보정 run의 입력으로 재사용한다.

## box provenance requirement

각 box에는 provenance를 반드시 남긴다.

- `label_origin`
  - `raw_source` 또는 `bootstrap`
- `teacher_name`
  - `mobility`, `signal`, `obstacle` 중 하나
- `confidence`
  - detector confidence
- `model_version`
  - checkpoint version 또는 exported model version
- `run_id`
  - sweep run id
- `created_at`
  - 생성 시각

이 정보는 나중에 teacher별 FP 추적, class별 품질 분석, pseudo-label 삭제/복구에 사용한다.

## obstacle guardrail

- obstacle model은 처음부터 별도 경계 정책을 둔다.
- obstacle은 teacher가 놓치거나 과검출하기 쉬운 class로 취급한다.
- obstacle 계열에는 아래를 기본으로 넣는다.
  - 보수적인 threshold
  - review 비중 확대
  - hard negative 장면 우선 확인
- obstacle과 traffic_cone은 같은 그룹 안에 있어도 같은 품질 정책으로 다루지 않는다.

## obstacle source mapping

AIHUB obstacle source의 canonical detector mapping은 이미 고정돼 있다.

- `Traffic cone -> traffic_cone`
- `Animals(Dolls) -> obstacle`
- `Garbage bag & sacks -> obstacle`
- `Construction signs & Parking prohibited board -> obstacle`
- `Box -> obstacle`
- `Stones on road -> obstacle`
- 아래 항목은 detector canonical output에서 제외한다.
  - `Person`
  - `Manhole`
  - `Pothole on road`
  - `Filled pothole`

즉 obstacle 정보의 출처는 AIHUB obstacle 단일 source이고,
그 source 내부의 지정 클래스들을 `obstacle`로 리맵한 것이다.
`traffic_cone`은 별도 class로 유지한다.

## preprocessing project goal

- 이 프로젝트의 목적은 detector 3개를 서비스에 그대로 붙이는 것이 아니다.
- 이 프로젝트의 목적은 exhaustive OD label set을 만드는 것이다.
- 그 exhaustive OD set으로 PV26 detection head가 background를 더 잘 보도록 다시 학습시키는 것이 최종 목표다.

## full-dataset sweep plan

1. mobility model을 `bdd100k_det_100k`로 먼저 파인튜닝한다.
2. signal model을 `aihub_traffic_seoul`로 먼저 파인튜닝한다.
3. obstacle model을 `aihub_obstacle_seoul`로 먼저 파인튜닝한다.
4. 그 다음 전체 dataset image pool을 하나의 전처리 대상 집합으로 모은다.
5. mobility model을 전체 image pool에 대해 전수 실행한다.
6. signal model을 전체 image pool에 대해 전수 실행한다.
7. obstacle model을 전체 image pool에 대해 전수 실행한다.
8. 세 모델의 출력을 하나의 OD 결과로 합친다.
9. 합쳐진 결과를 canonical scene의 exhaustive OD pseudo-label로 기록한다.
10. 이 exhaustive OD set을 사용해 PV26 detector를 다시 학습한다.

## default execution mode

기본 실행 방식은 `model-centric`이다.

### model-centric

- 전체 image 목록에 대해 mobility model을 먼저 전부 실행한다.
- 그 다음 signal model을 전부 실행한다.
- 그 다음 obstacle model을 전부 실행한다.
- 마지막에 결과를 merge한다.
- 모델 로드/언로드 오버헤드를 줄인다.
- 배치 처리와 resume 관리가 쉽다.
- teacher별 산출물 저장과 provenance 추적이 쉽다.

### image-centric

- image 한 장을 읽고 model 3개를 순차 실행하는 방식은 기본 경로로 쓰지 않는다.
- 모델을 반복해서 켜고 끄는 비용이 커질 수 있다.
- 초기 개발과 디버그용 보조 방식으로만 남긴다.

## merge rule

- raw source label이 우선이다.
- bootstrap detector 출력은 빈 class supervision 구간을 채우는 용도로 사용한다.
- source label과 bootstrap 결과가 충돌하면 source label을 기준으로 본다.
- teacher별 provenance는 merge 이후에도 유지한다.
- low-quality prediction filter는 class별 policy와 review 결과를 기반으로 조정한다.

## final output

- 산출물은 `exhaustive OD pseudo-label dataset`이다.
- 이 산출물은 PV26 detector 재학습 입력으로 사용한다.
- 최종 배포 후보는 bootstrap detector 3개가 아니라, exhaustive OD set으로 다시 학습한 PV26 detector다.

## implemented scope

1. mobility / signal / obstacle detector용 train spec 분리
2. class별 threshold / NMS / min-size policy 정의
3. provenance 포함 detector output schema 정의
4. full-dataset sweep runner 작성
5. detector output merge rule 작성
6. exhaustive OD pseudo-label dataset materialization
7. exhaustive OD set을 사용한 PV26 detector 재학습 경로 작성
