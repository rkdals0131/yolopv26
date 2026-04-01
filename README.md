# YOLOPV26

YOLOPV26는 `7-class OD`, 신호등 속성, 차선, 정지선, 횡단보도를 함께 학습하는 2D 인지 학습/전처리 저장소다.

아래 명령은 모두 이 저장소 루트에서 실행한다.

## 바로 쓰는 명령어

### 1. PV26 기본 학습

```bash
python3 tools/run_pv26_train.py --config config/pv26_meta_train.default.yaml
```

- `finalize`까지 끝난 최종 merged dataset 기준으로 바로 학습을 시작한다.
- `stage_1_frozen_trunk_warmup -> stage_2_partial_unfreeze -> stage_3_end_to_end_finetune` 3단계를 자동으로 수행한다.
- 출력 위치: `runs/pv26_exhaustive_od_lane_train/<meta_run_name>/`

### 2. AIHUB 원본 전처리

```bash
python3 tools/run_aihub_standardize.py
```

- AIHUB 원본을 PV26 canonical 형식으로 변환한다.
- 출력 위치: `seg_dataset/pv26_aihub_standardized/`

### 3. BDD100K 원본 전처리

```bash
python3 tools/run_bdd100k_standardize.py
```

- BDD100K 원본을 PV26 canonical 형식으로 변환한다.
- 출력 위치: `seg_dataset/pv26_bdd100k_standardized/`

### 4. OD bootstrap 소스 준비

```bash
python3 tools/od_bootstrap/preprocess/run_prepare_sources.py --config tools/od_bootstrap/config/preprocess/sources.default.yaml
```

- bootstrap용 canonical bundle과 image list를 만든다.
- 출력 위치: `seg_dataset/pv26_od_bootstrap/`

### 5. teacher 모델 학습용 데이터셋 생성

```bash
python3 tools/od_bootstrap/preprocess/run_build_teacher_datasets.py --config tools/od_bootstrap/config/preprocess/teacher_datasets.default.yaml
```

- `mobility`, `signal`, `obstacle` teacher 모델용 학습 데이터셋을 만든다.
- 출력 위치: `seg_dataset/pv26_od_bootstrap/teacher_datasets/`

### 6. teacher 모델 학습

아래 예시는 `yolo26s` 기준이다. 필요하면 같은 위치의 `yolo26n` 설정 파일로 바꿔서 실행하면 된다.

```bash
python3 tools/od_bootstrap/train/run_train_teacher.py --config tools/od_bootstrap/config/train/mobility_yolo26s.default.yaml
python3 tools/od_bootstrap/train/run_train_teacher.py --config tools/od_bootstrap/config/train/signal_yolo26s.default.yaml
python3 tools/od_bootstrap/train/run_train_teacher.py --config tools/od_bootstrap/config/train/obstacle_yolo26s.default.yaml
```

- 출력 위치: `runs/od_bootstrap/train/<teacher>/`

### 7. teacher 모델 평가

```bash
python3 tools/od_bootstrap/eval/run_teacher_checkpoint_eval.py --config tools/od_bootstrap/config/eval/mobility_checkpoint_eval.default.yaml
python3 tools/od_bootstrap/eval/run_teacher_checkpoint_eval.py --config tools/od_bootstrap/config/eval/signal_checkpoint_eval.default.yaml
python3 tools/od_bootstrap/eval/run_teacher_checkpoint_eval.py --config tools/od_bootstrap/config/eval/obstacle_checkpoint_eval.default.yaml
```

- 출력 위치: `runs/od_bootstrap/eval/<teacher>/`

### 8. 클래스 정책 보정

```bash
python3 tools/od_bootstrap/calibration/run_calibrate_class_policy.py --config tools/od_bootstrap/config/calibration/class_policy.default.yaml
```

- teacher validation prediction을 기준으로 클래스 정책을 보정한다.
- 출력 위치: `runs/od_bootstrap/calibration/default/`

### 9. 전체 OD sweep

```bash
python3 tools/od_bootstrap/sweep/run_model_centric_sweep.py --config tools/od_bootstrap/config/sweep/model_centric.default.yaml
```

- 세 teacher를 순차 적용해서 exhaustive OD dataset을 만든다.
- 출력 위치: `seg_dataset/pv26_od_bootstrap/exhaustive_od/<run_id>/`

### 10. 최종 병합 데이터셋 생성

```bash
python3 tools/od_bootstrap/finalize/run_build_exhaustive_od_lane_dataset.py --config tools/od_bootstrap/config/finalize/pv26_exhaustive_od_lane.default.yaml
```

- exhaustive OD 결과와 lane canonical 데이터를 합쳐 최종 학습 데이터셋을 만든다.
- 출력 위치: `seg_dataset/pv26_exhaustive_od_lane_dataset/`

### 11. exhaustive OD + lane 데이터로 PV26 학습

```bash
python3 tools/run_pv26_train.py --config config/pv26_meta_train.default.yaml
```

- bootstrap 결과를 포함한 최종 병합 데이터셋으로 PV26를 다시 학습한다.

## 어떤 때 무엇을 실행하면 되는가

- 이미 canonical dataset이 준비되어 있으면 `1번`만 실행하면 된다.
- AIHUB나 BDD100K 원본만 있으면 `2번`, `3번`으로 전처리한 뒤 `1번`을 실행하면 된다.
- exhaustive OD supervision까지 만들려면 `4번`부터 `11번` 순서대로 진행하면 된다.

## 주요 출력 경로

- `runs/pv26_meta_train/`: PV26 기본 학습 결과
- `runs/od_bootstrap/`: teacher 모델 학습, 평가, calibration 결과
- `seg_dataset/pv26_aihub_standardized/`: AIHUB 전처리 결과
- `seg_dataset/pv26_bdd100k_standardized/`: BDD100K 전처리 결과
- `seg_dataset/pv26_od_bootstrap/`: bootstrap 중간 산출물
- `seg_dataset/pv26_exhaustive_od_lane_dataset/`: 최종 병합 학습 데이터셋

## 저장소 구성

- [model/](model/): 전처리, 데이터 로딩, 인코딩, 헤드, 학습, 평가 코드
- [tools/](tools/): 실제 실행용 스크립트
- [config/](config/): PV26 기본 학습 설정
- [tools/od_bootstrap/config/](tools/od_bootstrap/config/): bootstrap 관련 설정
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
- 경로와 세부 파라미터는 각 설정 파일에서 먼저 확인하는 편이 가장 안전하다.
- bootstrap 전체 흐름만 따로 보고 싶으면 [tools/od_bootstrap/README.md](tools/od_bootstrap/README.md)를 보면 된다.
