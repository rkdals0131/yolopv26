# Modal 학습 빠른 가이드 (PV26)

이 문서는 `tools/train/modal_train_pv26.py`를 기준으로, Modal에서 PV26 학습을 돌릴 때 사용자에게 필요한 행동만 간단히 정리합니다.

## 1) 사전 준비
- Modal 로그인/설정:
  - `modal setup`
- 학습 코드:
  - `tools/train/train_pv26.py`
- Modal 래퍼:
  - `tools/train/modal_train_pv26.py`

## 2) 데이터셋 tar 준비 규격
기본 기대 이름:
- 볼륨 내 디렉토리: `pv26_v1_bdd_full`
- 볼륨 내 tar: `pv26_v1_bdd_full.tar`

tar 내부 권장 최상위 구조:
```text
pv26_v1_bdd_full/
  images/train/*.jpg
  images/val/*.jpg
  labels_det/train/*.txt
  labels_det/val/*.txt
  labels_seg_da/train/*.png
  labels_seg_da/val/*.png
  labels_seg_rm_lane_marker/train/*.png
  labels_seg_rm_lane_marker/val/*.png
  labels_seg_rm_road_marker_non_lane/train/*.png
  labels_seg_rm_road_marker_non_lane/val/*.png
  labels_seg_rm_stop_line/train/*.png
  labels_seg_rm_stop_line/val/*.png
  meta/split_manifest.csv
```

로컬에서 tar 생성 예시:
```bash
tar -cf pv26_v1_bdd_full.tar -C datasets pv26_v1_bdd_full
```

## 3) Modal Volume 업로드
기본 볼륨 이름:
- 데이터: `pv26-datasets`
- 산출물: `pv26-artifacts`

볼륨 목록 확인:
```bash
modal volume list
```

볼륨이 없으면 생성:
```bash
modal volume create pv26-datasets
modal volume create pv26-artifacts
```

업로드:
```bash
modal volume put pv26-datasets ./pv26_v1_bdd_full.tar /pv26_v1_bdd_full.tar
```

확인:
```bash
modal volume ls pv26-datasets /
```

## 4) 실행 방식
### A. Python 진입점으로 실행
```bash
python tools/train/modal_train_pv26.py train \
  --run-name exp_modal_a10g
```

### B. modal run으로 실행
```bash
modal run tools/train/modal_train_pv26.py \
  --run-name exp_modal_a10g
```

학습 하이퍼파라미터(`epochs`, `batch-size`, `workers`, `lr`)는
`tools/train/modal_train_pv26.py`의 아래 상수에서 수정합니다.
- `DEFAULT_EPOCHS`
- `DEFAULT_BATCH_SIZE`
- `DEFAULT_WORKERS`
- `DEFAULT_LR`

`device`는 `tools/train/train_pv26.py` 기본값(`auto`)을 사용합니다.

## 5) 스크립트가 내부에서 하는 일
`train_remote` 동작 순서:
1. 데이터 볼륨에서 `pv26_v1_bdd_full/meta/split_manifest.csv` 존재 확인
2. 없으면 `pv26_v1_bdd_full.tar`를 찾아 자동 압축해제
3. 압축해제 후 manifest 재확인
4. `tools/train/train_pv26.py`를 원격에서 실행
5. 체크포인트/로그를 `pv26-artifacts` 볼륨에 저장

기본 출력 루트:
- `/vol/artifacts/runs/pv26_train`

## 6) 체크포인트/TensorBoard 로그 확인
원격 산출물 확인:
```bash
modal volume ls pv26-artifacts /runs/pv26_train
```

로컬로 내려받기:
```bash
modal volume get pv26-artifacts /runs/pv26_train ./runs_modal
```

로컬 TensorBoard:
```bash
tensorboard --logdir ./runs_modal
```

작업 종료 후 볼륨이 더 이상 필요 없으면 삭제:
```bash
modal volume delete pv26-artifacts
modal volume delete pv26-datasets
```
주의: `delete`는 볼륨 데이터를 영구 삭제합니다.

## 7) 자주 쓰는 환경변수 (선택)
- `PV26_MODAL_APP_NAME` (기본: `pv26-train`)
- `PV26_MODAL_DATASET_VOLUME` (기본: `pv26-datasets`)
- `PV26_MODAL_ARTIFACT_VOLUME` (기본: `pv26-artifacts`)
- `PV26_MODAL_GPU` (기본: `A10G`)
- `PV26_MODAL_TIMEOUT_SEC` (기본: `86400`)
