# YOLOPV26

YOLOPV26는 `7-class OD`, 신호등 속성, 차선, 정지선, 횡단보도를 함께 학습하는 2D 인지 학습/전처리 저장소다.

## 가장 먼저 실행

이 저장소를 처음 켰으면 다른 명령보다 먼저 아래 interactive launcher를 실행한다.

```bash
python3 tools/check_env.py
```

- 이게 기본 진입점이다.
- 실행하자마자 runtime, raw dataset root, canonical 처리 수, teacher/calibration/final dataset 상태를 자동 스캔해서 보여준다.
- 숫자/영문 키로 다음 작업을 바로 실행할 수 있다.
- `D` 메뉴로 `stage_3` 기준 peak VRAM stress probe를 바로 돌려 볼 수 있다.
- 직접 `python -m ...` 명령을 치기 전에 현재 상태와 다음 추천 액션을 여기서 먼저 확인하면 된다.
- `H`를 누르면 README 요약과 config 파일 위치를 볼 수 있다.
- TUI 안에서 하이퍼파라미터를 직접 수정하지는 않는다.

아래 명령은 모두 이 저장소 루트에서 실행한다.

## 시작 전에 먼저 확인

### 0. 환경 점검

```bash
python3 tools/check_env.py
```

- 위 명령이 이 저장소의 기본 launcher다.
- 아래 direct command 목록은 launcher 밖에서 수동 실행하고 싶을 때 본다.

자동화나 비대화형 체크는 아래 strict 명령을 사용한다.

```bash
python3 tools/check_env.py --strict --check-yolo-runtime
```

- `torch`, `torchvision`, `ultralytics`, YOLO26 runtime이 현재 환경에서 실제로 로드되는지 먼저 확인한다.
- 이 단계가 깨지면 dataset을 아무리 맞춰도 teacher 학습이나 PV26 학습이 바로 시작되지 않는다.

### 0A. 원본 데이터셋 배치 확인

- AIHUB와 BDD100K 원본 폴더 구조는 [docs/3A_RAW_DATASET_LAYOUTS.md](docs/3A_RAW_DATASET_LAYOUTS.md)를 기준으로 맞춘다.
- 기본 preset은 `seg_dataset/AIHUB`, `seg_dataset/BDD100K`를 가정한다.
- 경로는 [config/user_paths.yaml](config/user_paths.yaml)에서 먼저 수정한다.
- bootstrap 숫자 파라미터는 [config/od_bootstrap_hyperparameters.yaml](config/od_bootstrap_hyperparameters.yaml)에서 수정한다.
- PV26 train 숫자 파라미터는 [config/pv26_train_hyperparameters.yaml](config/pv26_train_hyperparameters.yaml)에서 수정한다.
- 출력 경로를 바꾸면 bootstrap 단계 사이 연결도 같이 바뀌어야 한다. 이게 귀찮으면 기본값을 그대로 쓰는 편이 낫다.
- preset 로직이 실제로 이 두 파일을 어떻게 읽는지 확인하려면 [tools/od_bootstrap/presets.py](tools/od_bootstrap/presets.py)와 [tools/pv26_train/cli.py](tools/pv26_train/cli.py)를 보면 된다.

## 바로 쓰는 명령어

### 1. PV26 기본 학습

```bash
python3 tools/run_pv26_train.py --preset default
```

- `default` preset은 `seg_dataset/pv26_exhaustive_od_lane_dataset`를 입력으로 쓰는 최종 merged dataset 전용이다.
- canonical-only 상태라면 먼저 `2번`부터 `8번`까지 실행해 exhaustive OD + lane dataset을 만들어야 한다.
- `stage_1_frozen_trunk_warmup -> stage_2_partial_unfreeze -> stage_3_end_to_end_finetune -> stage_4_lane_family_finetune` 4단계를 자동으로 수행한다.
- lane head는 모델 소스의 최신 seg-first 경로를 기본으로 사용한다. 학습 숫자와 guard는 [config/pv26_train_hyperparameters.yaml](config/pv26_train_hyperparameters.yaml)에서 조절한다.
- train/validation epoch 진행은 live progress로 표시되고, elapsed/ETA와 rolling timing profile을 함께 보여준다.
- 출력 위치: `runs/pv26_exhaustive_od_lane_train/<meta_run_name>/`

기존 불완전 run을 같은 디렉터리에서 정확히 이어서 실행하려면 아래처럼 `--resume-run`을 사용한다.

```bash
python3 tools/run_pv26_train.py --resume-run runs/pv26_exhaustive_od_lane_train/<meta_run_name>
```

기존 run을 source로 삼아 새 derived run을 만들고 특정 stage window만 다시 학습하려면 아래처럼 `--derive-run`을 사용한다.

```bash
python3 tools/run_pv26_train.py --derive-run runs/pv26_exhaustive_od_lane_train/<source_run_name> --start-stage stage_3_end_to_end_finetune --end-stage stage_3_end_to_end_finetune
```

현재 stage 3 경로의 peak VRAM 상한만 빠르게 확인하려면 아래 direct probe도 사용할 수 있다.

```bash
python3 tools/run_pv26_train.py --preset default --stage3-vram-stress --stress-batch-size 4 --stress-iters 16
```

phase 1-4 각각의 local batch 하한/상한을 한 번에 확인하려면 CUDA가 보이는 터미널에서 아래 sweep을 실행한다. `ceiling_observed=false`면 해당 phase는 입력한 후보 중 OOM이 안 났다는 뜻이라 `max_ok_batch_size`는 확정 상한이 아니라 확인된 하한이다.

```bash
python3 tools/run_pv26_train.py --preset default --phase-vram-sweep --stress-batch-sizes 1,2,4,6,8,12 --stress-iters 8
```

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

- [model/](model/): PV26 runtime data/net/engine 코드
- [tools/](tools/): stable entrypoint(`check_env.py`, `run_pv26_train.py`)와 package-based internal tooling(`tools/check_env/`, `tools/pv26_train/`, `tools/od_bootstrap/source/aihub/`, `tools/od_bootstrap/source/shared/`, `tools/od_bootstrap/teacher/runtime/`)
- [docs/](docs/): 번호가 붙은 설계/상태 문서
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
- interactive 상태 허브: `python3 tools/check_env.py`
- 환경 점검: `python3 tools/check_env.py --strict --check-yolo-runtime`
- raw dataset 경로, run/output 경로는 `config/user_paths.yaml`
- bootstrap 전처리/teacher/calibration/exhaustive 숫자 파라미터는 `config/od_bootstrap_hyperparameters.yaml`
- PV26 학습 숫자 파라미터는 `config/pv26_train_hyperparameters.yaml`
- 코드 안에서 빠르게 조절 지점을 찾고 싶으면 `tools/od_bootstrap/presets.py`와 `tools/run_pv26_train.py`에서 `USER CONFIG`, `HYPERPARAMETERS`, `PHASE HYPERPARAMETERS`를 검색
- bootstrap 전체 흐름만 따로 보고 싶으면 [tools/od_bootstrap/README.md](tools/od_bootstrap/README.md)를 보면 된다.
