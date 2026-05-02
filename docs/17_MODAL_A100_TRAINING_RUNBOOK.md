# Modal A100 학습 실행 절차

이 문서는 현재 `yolopv26` 레포의 unified roadmark native 구조를 Modal A100-40GB 환경에서 실행하기 위한 절차입니다.

목표는 다음입니다.

1. 로컬의 `pv26_exhaustive_od_lane_dataset`을 압축합니다.
2. 압축 파일을 Modal Volume에 업로드합니다.
3. Modal에서 Volume의 archive를 로컬 SSD(`/local`)로 압축 해제합니다.
4. 데이터셋 layout을 검증합니다.
5. 로컬 학습 코드와 같은 entrypoint를 Modal 컨테이너 안에서 실행합니다.
6. 학습 결과를 Modal runs Volume에 보존합니다.

---

## 0. 현재 확인된 로컬 상태와 바로 실행 순서

이 문서는 아래 현재 상태를 기준으로 작성되었습니다.

```text
repo: /home/kai/yolopv26
dataset: /home/kai/yolopv26/seg_dataset/pv26_exhaustive_od_lane_dataset
dataset size: 107G
sample_count: 354350
local archive: /home/kai/yolopv26/pv26_exhaustive_od_lane_dataset.tar.zst
local archive status: 현재 없음. 먼저 압축 생성 필요.
Modal dataset volume: pv26-dataset-archives
Modal runs volume: pv26-training-runs
Modal archive path: /pv26_exhaustive_od_lane_dataset.tar.zst
Modal train preset: pv26_unified_roadmark_segfirst_a100
Modal GPU: A100-40GB
```

필수 dataset directory는 현재 모두 존재하는 것으로 확인되었습니다.

```text
OK images/train
OK images/val
OK images/test
OK labels_scene/train
OK labels_scene/val
OK labels_scene/test
OK labels_det/train
OK labels_det/val
OK labels_det/test
OK meta
```

아래 block을 위에서부터 그대로 실행하시면 됩니다.

```bash
cd /home/kai/yolopv26

# 0. 로컬 preflight: 경로/파일명/script constant가 서로 맞는지 확인합니다.
.venv/bin/python modal/local_preflight.py

# 1. archive 생성, archive contract 검증, Modal Volume 생성, 업로드 확인을 한 번에 수행합니다.
.venv/bin/python modal/prepare_dataset_volume.py --create-archive --ensure-volumes --upload

# 2. 업로드 확인
.venv/bin/modal volume ls pv26-dataset-archives /

# 3. Modal A100 환경에서 archive/extract/layout/CUDA를 먼저 검증합니다.
.venv/bin/modal run modal/check.py

# 4. 검증 통과 후 detached 학습 시작
.venv/bin/modal run --detach modal/train.py

# 5. 결과 Volume 확인
.venv/bin/modal volume ls pv26-training-runs /
.venv/bin/modal volume ls pv26-training-runs /pv26_unified_roadmark_segfirst_a100
```

`modal/local_preflight.py`, `modal/check.py`, `modal/train.py`는 모두 내부에서 hardcoded dataset 이름, archive 파일명, Volume mount 경로, preset 이름이 서로 일치하는지 검사합니다. 맞으면 `OK ...` 로그를 찍고 다음 단계로 넘어가며, 틀리면 즉시 실패합니다.
`modal/prepare_dataset_volume.py`는 로컬 archive 생성부터 Modal Volume 업로드 확인까지 같은 계획값을 사용해 단계별 `OK` 로그를 찍습니다.

---

## 1. 현재 코드 구조

Modal 실행 파일은 `modal/` 디렉토리에 있습니다.

| 파일 | 역할 |
| --- | --- |
| `modal/constants.py` | GPU, CPU, memory, Volume 이름, archive 경로, preset 이름을 정의합니다. CLI 인자 대신 이 파일을 수정합니다. |
| `modal/local_preflight.py` | 로컬에서 dataset 경로, archive 파일명, Modal constant 일치 여부를 확인하고 다음 실행 command를 출력합니다. |
| `modal/prepare_dataset_volume.py` | 로컬 archive 생성, archive top-level 검증, Modal Volume 생성, 업로드 확인을 수행합니다. |
| `modal/check.py` | Modal 환경에서 dataset archive 존재 여부, 압축 해제, layout, CUDA 장치를 확인합니다. |
| `modal/train.py` | Modal 환경에서 dataset archive를 압축 해제하고 PV26 학습을 실행합니다. |
| `modal/dataset_archive.py` | `.tar.zst`, `.tar.gz`, `.tar`, `.zip` 압축 해제와 dataset layout 검증을 담당합니다. |
| `modal/sdk_import.py` | 레포 안의 `modal/` 디렉토리와 외부 Modal SDK 이름 충돌을 피하기 위한 import helper입니다. |

현재 학습 command는 Modal 컨테이너 내부에서 다음과 같이 실행됩니다.

```bash
python3 -m tools.pv26_train.cli --preset pv26_unified_roadmark_segfirst_a100
```

현재 preset은 `tools/pv26_train/scenarios.py`에 정의되어 있습니다.

---

## 2. 현재 아키텍처 상태

현재 `PV26Heads`는 한 모델에서 다음 출력을 모두 냅니다.

```text
YOLO26 trunk P2/P3/P4/P5
 ├─ P3/P4/P5      -> OD head + traffic-light attr head
 └─ P2/P3/P4/P5  -> roadmark_joint_native
                    ├─ lane
                    ├─ stop_line
                    └─ crosswalk
```

출력 contract는 다음입니다.

| 출력 | shape |
| --- | --- |
| `det` | `float32[B, Q_det, 12]` |
| `tl_attr` | `float32[B, Q_det, 4]` |
| `lane` | `float32[B, 24, 38]` |
| `stop_line` | `float32[B, 8, 9]` |
| `crosswalk` | `float32[B, 8, 33]` |

따라서 Modal에서 학습이 시작되면 구조상 OD, TL attr, lane, stopline, crosswalk가 모두 같은 모델에서 학습됩니다.
`modal/train.py`는 학습 시작 직전에 실제 preset을 다시 로드해서 `task_mode=roadmark_joint`, run root, run name prefix, phase별 loss weight를 확인합니다.
단, 이것은 **학습 경로가 연결되었다는 뜻**이지, 성능이 충분하다는 뜻은 아닙니다. 실제 품질은 Modal run 결과로 확인해야 합니다.

---

## 3. 로컬 데이터셋 준비 상태

현재 로컬 데이터셋 경로는 다음입니다.

```text
/home/kai/yolopv26/seg_dataset/pv26_exhaustive_od_lane_dataset
```

현재 확인된 상태입니다.

```text
크기: 107G
sample_count: 354350
archive: /home/kai/yolopv26/pv26_exhaustive_od_lane_dataset.tar.zst 는 아직 없음
```

필수 directory는 모두 존재해야 합니다.

```text
images/train
images/val
images/test
labels_scene/train
labels_scene/val
labels_scene/test
labels_det/train
labels_det/val
labels_det/test
meta
```

확인 command:

```bash
cd /home/kai/yolopv26

DATA=seg_dataset/pv26_exhaustive_od_lane_dataset

du -sh "$DATA"

for d in \
  images/train images/val images/test \
  labels_scene/train labels_scene/val labels_scene/test \
  labels_det/train labels_det/val labels_det/test \
  meta
 do
   test -e "$DATA/$d" && echo "OK      $d" || echo "MISSING $d"
 done
```

---

## 4. Modal 설정값

현재 `modal/constants.py` 기준 설정입니다.

```python
APP_NAME = "pv26-unified-roadmark-segfirst-a100"
GPU_TYPE = "A100-40GB"
MODAL_CPU_CORES = 24
MODAL_MEMORY_MB = 98_304          # 96 GiB
MODAL_EPHEMERAL_DISK_MB = 786_432 # 768 GiB local SSD quota

DATA_VOLUME_NAME = "pv26-dataset-archives"
RUNS_VOLUME_NAME = "pv26-training-runs"

DATASET_ARCHIVE_IN_VOLUME = "/volumes/pv26_dataset_archives/pv26_exhaustive_od_lane_dataset.tar.zst"
LOCAL_DATASET_ROOT = "/local/pv26_exhaustive_od_lane_dataset"
TRAIN_PRESET = "pv26_unified_roadmark_segfirst_a100"
```

주의할 점:

- dataset은 Modal app upload에 포함하지 않습니다.
- `seg_dataset/**`, `runs/**`, `.git/**`, `.venv/**`는 Modal 코드 upload에서 제외됩니다.
- dataset은 반드시 Modal Volume에 압축 파일 형태로 따로 올립니다.
- `.tar.zst` 대신 `.zip`을 쓰려면 `DATASET_ARCHIVE_IN_VOLUME` 파일명도 `.zip`으로 바꾸셔야 합니다.

---

## 5. 로컬 preflight

압축 전에 한 번 실행합니다. 이 스크립트는 Modal SDK를 호출하지 않고 로컬 파일 상태와 script constant만 확인합니다.

```bash
cd /home/kai/yolopv26

.venv/bin/python modal/local_preflight.py
```

현재 정상 출력의 핵심은 다음입니다.

```text
[modal-local-preflight] OK constants ...
[modal-local-preflight] OK dataset status ... sample_count: 354350
[modal-local-preflight] ARCHIVE_MISSING ...
```

`ARCHIVE_MISSING`은 현재 정상입니다. 다음 단계에서 archive를 생성하면 됩니다.

---

## 6. 데이터셋 압축

권장 archive 이름은 다음입니다.

```text
pv26_exhaustive_od_lane_dataset.tar.zst
```

권장 command:

```bash
cd /home/kai/yolopv26

tar --zstd -cf pv26_exhaustive_od_lane_dataset.tar.zst \
  -C seg_dataset \
  --exclude='pv26_exhaustive_od_lane_dataset/debug_vis_lane_audit' \
  pv26_exhaustive_od_lane_dataset
```

`debug_vis_lane_audit`는 학습에 필요하지 않으므로 제외합니다.
원하시면 제외 옵션 없이 전체를 압축하셔도 됩니다.

압축 결과 확인:

```bash
ls -lh pv26_exhaustive_od_lane_dataset.tar.zst

tar --zstd -tf pv26_exhaustive_od_lane_dataset.tar.zst | head -20
```

archive 내부 최상위가 반드시 다음처럼 보여야 합니다.

```text
pv26_exhaustive_od_lane_dataset/
pv26_exhaustive_od_lane_dataset/images/
pv26_exhaustive_od_lane_dataset/labels_scene/
pv26_exhaustive_od_lane_dataset/labels_det/
pv26_exhaustive_od_lane_dataset/meta/
```

위 과정을 수동으로 나누지 않으려면 다음 command를 사용합니다.

```bash
cd /home/kai/yolopv26

.venv/bin/python modal/prepare_dataset_volume.py --create-archive
```

이 스크립트는 archive 파일명이 `pv26_exhaustive_od_lane_dataset.tar.zst`인지, archive 최상위가 `pv26_exhaustive_od_lane_dataset/`인지 확인한 뒤에만 `OK archive ready ...`를 출력합니다.

---

## 7. Modal Volume 생성

Modal CLI는 현재 로컬 venv 기준 `modal client version: 1.3.4`로 확인되었습니다.

먼저 로그인 또는 profile 설정이 되어 있어야 합니다.
이미 Modal을 쓰신 적이 있으면 보통 그대로 동작합니다.

Volume 생성:

```bash
cd /home/kai/yolopv26

.venv/bin/modal volume create pv26-dataset-archives
.venv/bin/modal volume create pv26-training-runs
```

이미 존재하면 에러가 날 수 있습니다. 그 경우 `volume list`로 확인하시면 됩니다.

```bash
.venv/bin/modal volume list
```

권장 자동 확인 command:

```bash
cd /home/kai/yolopv26

.venv/bin/python modal/prepare_dataset_volume.py --ensure-volumes
```

이 스크립트는 `modal volume list --json`으로 두 Volume 이름을 확인하고, 없으면 생성한 뒤 다시 확인합니다.

---

## 8. Dataset archive를 Modal Volume에 업로드

`modal/constants.py`는 Volume 내부 archive 경로를 다음으로 기대합니다.

```text
/pv26_exhaustive_od_lane_dataset.tar.zst
```

업로드 command:

```bash
cd /home/kai/yolopv26

.venv/bin/modal volume put -f \
  pv26-dataset-archives \
  pv26_exhaustive_od_lane_dataset.tar.zst \
  /pv26_exhaustive_od_lane_dataset.tar.zst
```

업로드 확인:

```bash
.venv/bin/modal volume ls pv26-dataset-archives /
```

예상 상태:

```text
pv26_exhaustive_od_lane_dataset.tar.zst
```

권장 자동 업로드 command:

```bash
cd /home/kai/yolopv26

.venv/bin/python modal/prepare_dataset_volume.py --create-archive --ensure-volumes --upload
```

이 스크립트는 업로드 후 `modal volume ls pv26-dataset-archives / --json` 결과에 `pv26_exhaustive_od_lane_dataset.tar.zst`가 확인되어야 `OK uploaded archive ...`를 출력합니다.

---

## 9. Modal 환경 check 실행

학습 전에 반드시 check를 먼저 실행합니다.

```bash
cd /home/kai/yolopv26

.venv/bin/modal run modal/check.py
```

이 command가 하는 일은 다음입니다.

1. A100-40GB Modal function을 띄웁니다.
2. `pv26-dataset-archives` Volume을 mount합니다.
3. `/volumes/pv26_dataset_archives/pv26_exhaustive_od_lane_dataset.tar.zst` 존재 여부를 확인합니다.
4. archive를 `/local/pv26_exhaustive_od_lane_dataset`로 압축 해제합니다.
5. 필수 dataset directory들을 확인합니다.
6. CUDA 사용 가능 여부와 GPU 이름을 출력합니다.

성공 시 출력에는 대략 다음 항목이 포함됩니다.

```text
cuda_available: True
cuda_device: NVIDIA A100 ...
dataset_root: /local/pv26_exhaustive_od_lane_dataset
layout: images/train, labels_scene/train, labels_det/train, ...
```

여기서 실패하면 학습을 시작하지 마시고 archive 경로와 layout부터 고쳐야 합니다.

---

## 10. Modal 학습 실행

check가 통과하면 학습을 실행합니다.

```bash
cd /home/kai/yolopv26

.venv/bin/modal run --detach modal/train.py
```

`--detach`를 붙이는 이유는 로컬 터미널 연결이 끊겨도 Modal run이 계속 돌게 하기 위해서입니다.

`modal/train.py`가 하는 일은 다음입니다.

1. A100-40GB, CPU 24 cores, memory 96 GiB, local SSD 768 GiB function을 띄웁니다.
2. repo 코드를 `/root/yolopv26`에 올립니다.
3. dataset archive를 Volume에서 확인합니다.
4. archive를 `/local/pv26_exhaustive_od_lane_dataset`로 압축 해제합니다.
5. dataset layout을 확인합니다.
6. Modal 컨테이너 안에서 다음 config를 작성합니다.

```yaml
pv26_train:
  dataset_root: /local/pv26_exhaustive_od_lane_dataset
  run_root: /root/yolopv26/runs/pv26_unified_roadmark_segfirst_a100
```

7. 실제 preset을 로드해서 다음 항목을 확인합니다.

```text
dataset.root == /local/pv26_exhaustive_od_lane_dataset
run.run_root == /root/yolopv26/runs/pv26_unified_roadmark_segfirst_a100
run.run_name_prefix == pv26_unified_roadmark_segfirst_a100
train_defaults.task_mode == roadmark_joint
phase 1~3: det/tl_attr/lane/stop_line loss > 0
phase 2~4: crosswalk loss > 0
phase 4: det/tl_attr loss == 0
```

8. 다음 학습 command를 실행합니다.

```bash
python3 -m tools.pv26_train.cli --preset pv26_unified_roadmark_segfirst_a100
```

9. 학습 중 매 epoch마다 고정 preview sample 12개에 대해 lane-family 방식의 comparison grid를 생성합니다.
10. 학습이 끝나거나 예외가 발생해도 `runs` 결과를 `pv26-training-runs` Volume으로 복사합니다.
11. `runs_volume.commit()`을 호출하여 결과를 보존합니다.

결과 복사는 기존 `/volumes/pv26_training_runs/pv26_unified_roadmark_segfirst_a100` 내용을 삭제하지 않고 병합합니다. 재실행 시 이전 timestamp run이 지워지지 않습니다.

---

## 11. 매 epoch comparison grid

Modal A100 preset은 `preview.epoch_comparison_grid=True`로 설정되어 있습니다.
따라서 각 phase의 매 epoch 종료 후 다음 경로에 grid가 생성됩니다.

```text
runs/.../phase_<N>/epoch_comparison_grids/epoch_<EEE>/comparison_grid.png
```

현재 설정은 lane-family에서 쓰던 방식과 맞춰 다음과 같습니다.

```text
sample_count: 12
columns: 3
every_n_epochs: 1
```

즉 12개 고정 sample을 `ground_truth | prediction` pair tile로 만들고, 이를 3열 x 4행 grid로 합칩니다.
각 epoch directory에는 개별 sample별 파일도 같이 남습니다.

```text
epoch_comparison_grids/
  manifest.json
  epoch_001/
    comparison_grid.png
    summary.json
    01__<sample_id>/
      ground_truth.png
      prediction.png
      comparison.png
```

이 artifact는 학습을 멈추게 하는 필수 경로가 아닙니다. overlay rendering 문제가 생기면 log에 `epoch comparison grid failed ...`를 남기고 학습은 계속 진행합니다.

---

## 12. checkpoint 저장 정책

Modal A100 preset은 epoch별 checkpoint spam을 막기 위해 다음처럼 설정되어 있습니다.

```text
checkpoint_every: 0
```

따라서 `epoch_001.pt`, `epoch_002.pt` 같은 epoch checkpoint는 저장하지 않습니다.
대신 checkpoint directory에는 다음 계열만 유지합니다.

| 파일 | 의미 |
| --- | --- |
| `last.pt` | latest checkpoint이자 기존 resume 기준 checkpoint입니다. |
| `best.pt` | phase selection objective 기준 best입니다. |
| `best_detector.pt` | `val.metrics.detector.map50_95` 기준 OD best입니다. |
| `best_traffic_light.pt` | `val.metrics.traffic_light.combo_accuracy` 기준 TL attr best입니다. |
| `best_lane.pt` | `val.metrics.lane.f1` 기준 lane best입니다. |
| `best_stop_line.pt` | `val.metrics.stop_line.f1` 기준 stopline best입니다. |
| `best_crosswalk.pt` | `val.metrics.crosswalk.f1` 기준 crosswalk best입니다. |

`summary.json`에는 `checkpoint_paths.task_best`와 `task_best_metrics`가 같이 기록됩니다.

---

## 13. 로그 확인

`--detach` 실행 후 Modal CLI가 app/function run 정보를 출력합니다.
Modal dashboard에서도 확인할 수 있습니다.

Volume 상태 확인:

```bash
.venv/bin/modal volume ls pv26-training-runs /
.venv/bin/modal volume ls pv26-training-runs /pv26_unified_roadmark_segfirst_a100
```

결과 다운로드 예시:

```bash
cd /home/kai/yolopv26

mkdir -p modal_downloads

.venv/bin/modal volume get \
  pv26-training-runs \
  /pv26_unified_roadmark_segfirst_a100 \
  modal_downloads/pv26_unified_roadmark_segfirst_a100
```

---

## 14. 학습 속도 튜닝 기준

학습 log에는 iteration profile이 나옵니다.

주요 항목은 다음입니다.

| 항목 | 의미 |
| --- | --- |
| `wait` | dataloader에서 다음 batch를 기다린 시간입니다. 낮을수록 좋습니다. |
| `load` | batch를 device로 옮기고 prepare하는 시간입니다. |
| `fwd` | forward 시간입니다. |
| `loss` | loss 계산 시간입니다. |
| `bwd` | backward/optimizer 시간입니다. |
| `total` 또는 `iter_ms` | 전체 iteration 시간입니다. |

판단 기준:

```text
wait_ratio = wait_sec / iteration_sec
```

| wait ratio | 해석 |
| --- | --- |
| `< 3~5%` | data pipeline이 충분합니다. |
| `5~10%` | 괜찮지만 조정 여지가 있습니다. |
| `10~20%` | GPU가 데이터를 기다리는 중입니다. |
| `20%+` | worker, prefetch, CPU, SSD 병목 가능성이 큽니다. |

초기값은 다음 조합입니다.

```text
batch_size: 64
num_workers: 12
prefetch_factor: 4
cpu: 24
memory: 96 GiB
```

만약 `wait`가 높으면 다음 순서로 조정합니다.

1. `num_workers`를 12 → 16으로 올립니다.
2. 그래도 `wait`가 높고 CPU 사용률이 높으면 Modal `MODAL_CPU_CORES`를 32로 올립니다.
3. RAM이 높거나 p99 iteration이 튀면 `prefetch_factor`를 4 → 2로 낮춥니다.
4. OOM이면 `batch_size`를 64 → 48 또는 32로 낮춥니다.

현재 source constant 위치:

- Modal resource: `modal/constants.py`
- batch/worker/prefetch preset: `tools/pv26_train/scenarios.py`

---

## 15. 실패 시 먼저 볼 것

### archive가 없다고 나오는 경우

확인:

```bash
.venv/bin/modal volume ls pv26-dataset-archives /
```

`pv26_exhaustive_od_lane_dataset.tar.zst`가 root에 있어야 합니다.

### dataset layout missing이 나오는 경우

archive 내부 최상위 directory가 잘못되었을 가능성이 큽니다.
다음처럼 archive 내부가 dataset root를 포함해야 합니다.

```text
pv26_exhaustive_od_lane_dataset/images/train
pv26_exhaustive_od_lane_dataset/labels_scene/train
pv26_exhaustive_od_lane_dataset/labels_det/train
pv26_exhaustive_od_lane_dataset/meta
```

### local SSD 부족

`modal/constants.py`에서 다음 값을 늘립니다.

```python
MODAL_EPHEMERAL_DISK_MB = 786_432
```

Modal 문서 기준 `ephemeral_disk`는 MiB 단위이며, 기본 per-container disk quota는 512 GiB입니다. 현재 값은 768 GiB입니다.

### CPU/RAM 부족

`modal/constants.py`에서 다음 값을 조정합니다.

```python
MODAL_CPU_CORES = 24
MODAL_MEMORY_MB = 98_304
```

Modal `memory`는 MB 단위입니다.

### 학습은 시작했는데 OD/TL 또는 lane-family loss가 비정상인 경우

먼저 `runs/.../history/train_steps.jsonl`에서 다음을 확인합니다.

- `losses.det`
- `losses.tl_attr`
- `losses.lane`
- `losses.stop_line`
- `losses.crosswalk`
- `assignment.det`
- `assignment.lane`
- `timing.wait_sec`
- `timing.iteration_sec`

이번 구조 이식은 성능 보장이 아니라 실행 가능한 unified architecture 검증 단계입니다. 따라서 첫 run의 판정은 다음 순서로 합니다.

1. non-finite loss 없이 도는지
2. OD/TL loss가 실제로 켜져 있는지
3. lane/stopline/crosswalk loss가 finite인지
4. validation에서 OD/TL과 roadmark metric이 모두 기록되는지
5. lane F1이 기존 0.25~0.30 천장에 다시 걸리는지, 아니면 다른 양상을 보이는지

---

## 16. 전체 command 요약

현재 상태에서 실제로 실행할 전체 command입니다. 그대로 복사해서 위에서부터 진행하시면 됩니다.

```bash
cd /home/kai/yolopv26

# 0. 로컬 상태와 script constant 확인
.venv/bin/python modal/local_preflight.py

# 1. dataset archive 생성, Volume 생성, upload 확인
.venv/bin/python modal/prepare_dataset_volume.py --create-archive --ensure-volumes --upload

# 2. 업로드 확인
.venv/bin/modal volume ls pv26-dataset-archives /

# 3. Modal 원격 check
.venv/bin/modal run modal/check.py

# 4. detached 학습 시작
.venv/bin/modal run --detach modal/train.py

# 5. 결과 Volume 확인
.venv/bin/modal volume ls pv26-training-runs /
.venv/bin/modal volume ls pv26-training-runs /pv26_unified_roadmark_segfirst_a100
```

결과를 로컬로 내려받을 때는 다음을 실행합니다.

```bash
cd /home/kai/yolopv26

mkdir -p modal_downloads

.venv/bin/modal volume get \
  pv26-training-runs \
  /pv26_unified_roadmark_segfirst_a100 \
  modal_downloads/pv26_unified_roadmark_segfirst_a100
```
