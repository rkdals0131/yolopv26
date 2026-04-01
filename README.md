# YOLOPV26

YOLOPV26는 `7-class OD`, 신호등 속성, 차선, 정지선, 횡단보도를 함께 학습하는 2D 인지 학습/전처리 저장소다.

아래 명령은 모두 이 저장소 루트에서 실행한다.

## 바로 쓰는 명령어

### 1. PV26 기본 학습

```bash
python3 tools/run_pv26_train.py --preset default
```

- `default` preset은 `seg_dataset/pv26_exhaustive_od_lane_dataset`를 입력으로 쓰는 최종 merged dataset 전용이다.
- canonical-only 상태라면 먼저 `2번`부터 `8번`까지 실행해 exhaustive OD + lane dataset을 만들어야 한다.
- `stage_1_frozen_trunk_warmup -> stage_2_partial_unfreeze -> stage_3_end_to_end_finetune` 3단계를 자동으로 수행한다.
- 출력 위치: `runs/pv26_exhaustive_od_lane_train/<meta_run_name>/`

### 2. OD bootstrap 소스 준비

```bash
python -m tools.od_bootstrap prepare-sources
```

- AIHUB와 BDD100K 원본을 bootstrap용 canonical bundle과 image list로 정리한다.
- 출력 위치: `seg_dataset/pv26_od_bootstrap/`

### 3. teacher 모델 학습용 데이터셋 생성

```bash
python -m tools.od_bootstrap build-teacher-datasets
```

- `mobility`, `signal`, `obstacle` teacher 모델용 학습 데이터셋을 만든다.
- 출력 위치: `seg_dataset/pv26_od_bootstrap/teacher_datasets/`

### 4. teacher 모델 학습

mobility와 signal은 기본적으로 `yolo26s`를, obstacle은 기본적으로 `yolo26m`을 사용한다.

```bash
python -m tools.od_bootstrap train --teacher mobility
python -m tools.od_bootstrap train --teacher signal
python -m tools.od_bootstrap train --teacher obstacle
```

- 출력 위치: `runs/od_bootstrap/train/<teacher>/`

### 5. teacher 모델 평가

```bash
python -m tools.od_bootstrap eval --teacher mobility
python -m tools.od_bootstrap eval --teacher signal
python -m tools.od_bootstrap eval --teacher obstacle
```

- 출력 위치: `runs/od_bootstrap/eval/<teacher>/`

### 6. 클래스 정책 보정

```bash
python -m tools.od_bootstrap calibrate
```

- teacher validation prediction을 기준으로 클래스 정책을 보정한다.
- 출력 위치: `runs/od_bootstrap/calibration/default/`

### 7. exhaustive OD 생성

```bash
python -m tools.od_bootstrap build-exhaustive-od
```

- 세 teacher를 순차 적용해서 exhaustive OD dataset을 만든다.
- 출력 위치: `seg_dataset/pv26_od_bootstrap/exhaustive_od/<run_id>/`

### 8. 최종 병합 데이터셋 생성

```bash
python -m tools.od_bootstrap build-final-dataset
```

- exhaustive OD 결과와 lane canonical 데이터를 합쳐 최종 학습 데이터셋을 만든다.
- 출력 위치: `seg_dataset/pv26_exhaustive_od_lane_dataset/`

### 9. exhaustive OD + lane 데이터로 PV26 학습

```bash
python3 tools/run_pv26_train.py --preset default
```

- bootstrap 결과를 포함한 최종 병합 데이터셋으로 PV26를 다시 학습한다.

## 어떤 때 무엇을 실행하면 되는가

- `seg_dataset/pv26_exhaustive_od_lane_dataset`가 이미 있으면 `1번`만 실행하면 된다.
- AIHUB나 BDD100K 원본만 있으면 `2번`부터 시작해서 `8번`까지 실행해 최종 merged dataset을 만든 뒤 `1번`을 실행하면 된다.
- exhaustive OD supervision까지 만들려면 `2번`부터 `9번` 순서대로 진행하면 된다.

## 주요 출력 경로

- `runs/pv26_exhaustive_od_lane_train/`: PV26 기본 학습 결과
- `runs/od_bootstrap/`: teacher 모델 학습, 평가, calibration 결과
- `seg_dataset/pv26_od_bootstrap/canonical/`: AIHUB/BDD bootstrap canonical 결과
- `seg_dataset/pv26_od_bootstrap/`: bootstrap 중간 산출물
- `seg_dataset/pv26_exhaustive_od_lane_dataset/`: 최종 병합 학습 데이터셋

## 저장소 구성

- [model/](model/): 전처리, 데이터 로딩, 인코딩, 헤드, 학습, 평가 코드
- [tools/](tools/): 실제 실행용 스크립트
- [config/](config/): PV26 기본 학습 설정
- [docs/](docs/): 설계 문서와 작업 기록
- [test/](test/): 테스트 코드

## 참고 문서

- [docs/0_PRD.md](docs/0_PRD.md): 저장소 목표와 전체 범위
- [docs/3_DATA_AND_STANDARDIZATION.md](docs/3_DATA_AND_STANDARDIZATION.md): 데이터 구조와 전처리 방향
- [docs/3A_RAW_DATASET_LAYOUTS.md](docs/3A_RAW_DATASET_LAYOUTS.md): 원본 데이터셋 배치와 로컬 레이아웃
- [docs/4_MODEL_ARCHITECTURE.md](docs/4_MODEL_ARCHITECTURE.md): 모델 구조
- [docs/5_TARGETS_AND_LOSS.md](docs/5_TARGETS_AND_LOSS.md): 타깃 인코딩과 loss 설계
- [docs/6_TRAINING_AND_EVALUATION.md](docs/6_TRAINING_AND_EVALUATION.md): 학습/평가 정책

## 실행 전 메모

- 의존성 설치: `pip install -r requirements.txt`
- bootstrap preset 값은 `tools/od_bootstrap/presets.py`에서 확인할 수 있다.
- bootstrap 전체 흐름만 따로 보고 싶으면 [tools/od_bootstrap/README.md](tools/od_bootstrap/README.md)를 보면 된다.
