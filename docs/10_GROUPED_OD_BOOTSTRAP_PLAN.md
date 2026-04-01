# Grouped OD Bootstrap Plan

## scope

- 이 문서는 OD bootstrap의 설계 결정과 현재 기본값만 다룬다.
- lane / stop-line / crosswalk 계획은 여기서 다루지 않는다.
- 목표는 `7-class detector` 문제를 `3개의 별도 독립 YOLO v26 teacher`로 풀고, 그 결과로 exhaustive OD pseudo-label을 만드는 것이다.
- 이 teacher 3개는 최종 배포 모델이 아니라 전처리용 bootstrap detector다.

## core decision

- YOLO v26 아키텍처는 수정하지 않는다.
- detector head를 쪼개지 않는다.
- 독립적인 YOLO v26 detector model을 `3개` 유지한다.
- 각 모델은 자기에게 지정된 class와 지정된 dataset으로 먼저 파인튜닝한다.
- 이 단계에서 각 dataset의 원본 OD label은 ground truth로 취급한다.
- raw source label이 우선이다.
- calibration 단계는 별도 경로로 유지하고, teacher val split 기준 class policy를 보정한다.

## current defaults at HEAD

- mobility teacher default는 `yolo26s`다.
- signal teacher default는 `yolo26s`다.
- obstacle teacher default는 `yolo26m`다.
- mobility / signal / obstacle 모두 size variant preset은 계속 남겨두되, 현재 sweep baseline은 위 default를 따른다.
- sweep default execution mode는 `model-centric`이다.
- sweep은 calibration 산출물인 `class_policy.yaml`을 읽는다.
- calibration은 `hard_negative_manifest.json`을 다음 보정 run의 입력으로 재사용한다.

## teacher split

### 1. mobility

- classes: `vehicle`, `bike`, `pedestrian`
- first fine-tune dataset: `bdd100k_det_100k`
- current default base model: `yolo26s`

### 2. signal

- classes: `traffic_light`, `sign`
- first fine-tune dataset: `aihub_traffic_seoul`
- current default base model: `yolo26s`

### 3. obstacle

- classes: `traffic_cone`, `obstacle`
- first fine-tune dataset: `aihub_obstacle_seoul`
- current default base model: `yolo26m`

## training rule

- 각 teacher는 official YOLO v26 detector training path를 그대로 사용한다.
- 각 teacher는 자기 담당 class만 학습한다.
- 각 teacher는 자기에게 지정된 dataset으로만 1차 파인튜닝한다.
- 각 teacher 학습에서는 objectness negative를 활성화한다.
- 각 teacher는 자기 담당 class에 대해 background와 foreground를 적극적으로 구분하도록 학습한다.

## calibration policy

- class별 policy template은 유지한다.
- 최소한 아래는 class별로 분리한다.
  - score threshold
  - NMS 기준
  - minimum box size
- `traffic_light`, `sign`, `traffic_cone`, `obstacle`는 같은 threshold로 묶지 않는다.
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

이 정보는 teacher별 FP 추적, class별 품질 분석, pseudo-label 삭제/복구에 사용한다.

## obstacle guardrail

- obstacle teacher는 별도 경계 정책을 둔다.
- obstacle은 teacher가 놓치거나 과검출하기 쉬운 class로 취급한다.
- obstacle 계열에는 아래를 기본으로 유지한다.
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

즉 obstacle 정보의 출처는 AIHUB obstacle 단일 source이고, 그 source 내부의 지정 클래스들을 `obstacle`로 리맵한 것이다.
`traffic_cone`은 별도 class로 유지한다.

## bootstrap objective

- 이 프로젝트의 목적은 teacher 3개를 서비스에 직접 붙이는 것이 아니다.
- 목적은 exhaustive OD label set을 만드는 것이다.
- 그 exhaustive OD set으로 PV26 detection head가 background를 더 잘 보도록 다시 학습시키는 것이 최종 목표다.

## merge rule

- raw source label이 우선이다.
- bootstrap detector 출력은 빈 class supervision 구간을 채우는 용도로 사용한다.
- source label과 bootstrap 결과가 충돌하면 source label을 기준으로 본다.
- teacher별 provenance는 merge 이후에도 유지한다.
- low-quality prediction filter는 class별 policy와 review 결과를 기반으로 조정한다.

## final output

- 산출물은 `exhaustive OD pseudo-label dataset`이다.
- 이 산출물은 PV26 detector 재학습 입력으로 사용한다.
- 최종 배포 후보는 bootstrap teacher 3개가 아니라, exhaustive OD set으로 다시 학습한 PV26 detector다.

## status note

- 아래 항목들은 이미 HEAD에서 구현되어 있다.
  - teacher train spec 분리
  - class별 threshold / NMS / min-size policy 정의
  - provenance 포함 detector output schema
  - model-centric sweep runner
  - detector output merge rule
  - exhaustive OD pseudo-label dataset materialization
  - exhaustive OD set을 사용하는 PV26 재학습 경로
