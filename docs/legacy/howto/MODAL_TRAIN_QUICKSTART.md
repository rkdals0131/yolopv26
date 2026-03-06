> **Archived (2026-03-05)** — kept for history; may be outdated.  
> Canonical docs: `docs/PV26_PRD.md`, `docs/PV26_DATASET_CONVERSION_SPEC.md`, `docs/PV26_DATASET_SOURCES_AND_MAPPING.md`, `docs/PV26_IMPLEMENTATION_STATUS.md`.

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
2. `pv26_v1_bdd_full.tar`가 있으면 컨테이너 로컬 SSD(`/tmp`)로 복사
3. 로컬 SSD(`/tmp`)에서 압축해제 후 로컬 경로를 학습에 사용
4. tar가 없고 볼륨에 디렉토리만 있으면 디렉토리를 `/tmp`로 복사해서 사용
5. `tools/train/train_pv26.py`를 원격에서 실행
6. 체크포인트/로그를 `pv26-artifacts` 볼륨에 저장

기본 출력 루트:
- `/vol/artifacts/runs/pv26_train`

## 6) 체크포인트/TensorBoard 로그 확인
기본 설정(`AUTO_DOWNLOAD_ARTIFACTS=True`)이면 학습 성공 후 아래 경로로 자동 동기화됩니다.
- `runs/pv26_train/<run-name>`

학습 중 자동 동기화 정책(`tools/train/modal_train_pv26.py`):
- TensorBoard(`tb`): 매 epoch(정확히는 `latest.pt` 갱신 감지 시)마다 동기화
- 체크포인트(`latest.pt`, `best.pt`): `SYNC_EVERY_N_EPOCHS`마다 동기화

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

`tools/train/modal_train_pv26.py` 사용자 수정 블록 예시:
```python
GPU_NAME             = os.getenv("PV26_MODAL_GPU", "A10G")  # 예: "A10G", "L4", "A100"
```

## 8) Modal GPU 요약 (대략)
아래는 선택 판단용 요약입니다. 실제 가격/가용성은 수시로 바뀌므로 실행 전에 공식 페이지를 확인하세요.

지원 GPU 타입(문서 기준):
- `T4`, `L4`, `A10`, `A100`(`A100-40GB`, `A100-80GB`), `L40S`, `H100`, `H200`, `B200`

대략 성능 그룹:
- 최고 성능/대형 학습: `B200`, `H200`, `H100`
- 균형형 학습: `A100-80GB`, `A100-40GB`
- 중간급 학습/추론: `L40S`, `A10`, `L4`
- 저비용 테스트/가벼운 추론: `T4`

대략 가격(Modal Pricing, GPU Tasks):
- `B200`: `$0.001736/s` (약 `$6.2496/hr`)
- `H200`: `$0.001261/s` (약 `$4.5396/hr`)
- `H100`: `$0.001097/s` (약 `$3.9492/hr`)
- `A100-80GB`: `$0.000694/s` (약 `$2.4984/hr`)
- `A100-40GB`: `$0.000583/s` (약 `$2.0988/hr`)
- `L40S`: `$0.000542/s` (약 `$1.9512/hr`)
- `A10`: `$0.000306/s` (약 `$1.1016/hr`)
- `L4`: `$0.000222/s` (약 `$0.7992/hr`)
- `T4`: `$0.000164/s` (약 `$0.5904/hr`)

공식 참고:
- Pricing: `https://modal.com/pricing`
- GPU docs: `https://modal.com/docs/guide/gpu`
