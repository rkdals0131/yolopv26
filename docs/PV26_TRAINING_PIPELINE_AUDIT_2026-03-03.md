# PV26 학습 코드/구조 전수 점검 브리핑 (2026-03-03)

## TL;DR

- "랩탑에서 1 epoch"과 "Modal GPU에서 1 epoch" 시간이 비슷하게 나오는 것은, 현재 구조상 충분히 발생할 수 있다.
- 원인 1순위는 **DataLoader/전처리 병목**이다. 특히 학습 hot path에서 마스크에 대해 `np.unique()` 기반 검증을 매 샘플마다 반복한다.
- 원인 2순위는 **VAL 메트릭 계산이 비효율적**이라(샘플 단위 H2D 재복사 + `.item()` 동기화), epoch wall-time(= train+val)을 크게 잡아먹을 수 있다.
- Modal 래퍼의 기본값은 "정밀 프로파일링" 쪽으로 설정되어 있어(`--profile-sync-cuda`), throughput 비교/실학습에서 오버헤드가 생길 수 있다.

## 범위

다음 파일들을 읽고 구조/병목을 점검했다.

- `tools/train/train_pv26.py` (train/val 루프, 프로파일링, 체크포인트)
- `pv26/torch_dataset.py` (manifest 기반 Dataset, letterbox/증강)
- `pv26/masks.py` (mask validator, ignore mask)
- `pv26/criterion.py` (OD/DA/RM loss, Ultralytics E2E loss 연동)
- `pv26/multitask_model.py` (YOLO26 trunk + segmentation heads)
- `tools/train/modal_train_pv26.py` (Modal 원격 실행 래퍼, dataset stage)
- `pv26/validate_dataset.py` (오프라인 데이터셋 검증 도구)
- `tools/train/common.py` (IoU/mAP 및 유틸)

## 파이프라인 구조 요약

- Dataset: `Pv26ManifestDataset`가 `meta/split_manifest.csv`를 읽고, `__getitem__`에서 이미지/마스크/라벨을 로드한 뒤 letterbox(기본 960x544) 적용 후 텐서로 변환해 반환한다.
  - 이미지: `uint8 [3,H,W]` (정규화는 GPU에서 수행)
  - 마스크: `uint8` `{0,1,255(ignore)}`
  - partial label은 `has_*` 플래그와 "all 255 ignore" 마스크로 표현
- Collate: `tools/train/train_pv26.py::_collate_with_images()`가 워커 프로세스에서 배치 텐서(이미지/마스크/has_*)를 만들고, Ultralytics loss용 flat target(`det_tgt_*`)도 함께 구성한다.
- Train loop: `train_one_epoch()`가
  - dataloader wait time(`wait`)
  - H2D + normalize + channels_last(`h2d`)
  - forward/loss/backward/optimizer 단계별 시간
  을 `[profile][train]` 로그로 출력한다.
- Model/Loss:
  - 기본 `--arch yolo26n`: `PV26MultiHeadYOLO26` + `PV26Criterion(od_loss_impl="ultralytics_e2e")`.
  - `--arch stub`: tiny backbone + dense detector head + dense OD loss(파이썬 루프 기반)라서 성능/스케일 관점에서는 비교 기준으로 부적절하다.

## 주요 병목 및 근거

### 1) Dataset letterbox 경로에서 마스크 검증이 과도하게 비싸다

`pv26/torch_dataset.py::_letterbox_mask_u8()`가 마스크를 letterbox할 때,
입력/출력 각각에 대해 `validate_binary_mask_u8()`를 호출한다.

- 위치:
  - `pv26/torch_dataset.py`의 `_letterbox_mask_u8()` 내부
  - 입력 검증: `validate_binary_mask_u8(mask_u8, ...)`
  - 출력 검증: `validate_binary_mask_u8(out, ...)`
- 문제:
  - `pv26/masks.py::validate_binary_mask_u8()`는 `np.unique(mask)`를 호출한다.
  - 이 검증은 마스크 전체를 스캔하며, 720x1280 같은 큰 배열에서는 비용이 크다.
  - 학습 중에는 샘플당 DA 1 + RM 3 = 최대 4개의 마스크에 대해 letterbox를 수행하므로,
    검증 호출이 매우 높은 빈도로 반복된다.

#### 간단 측정(로컬에서 일부 샘플로 측정)

아래는 `datasets/pv26_v1_bdd_full` 기준으로,
하나의 720x1280 DA 마스크에 대해 반복 측정한 대략 값이다(환경/IO에 따라 달라질 수 있음).

- DA mask 로드(PIL->np): 평균 약 3ms
- `validate_binary_mask_u8(np.unique)`만: 평균 약 9ms

또한 `Pv26ManifestDataset.__getitem__` 자체가 랜덤 샘플 기준 평균 약 79ms/샘플 수준으로 측정되었다.
즉, 지금 구조에서는 GPU가 빨라져도 loader/전처리가 병목이 되기 쉽다.

참고: 데이터 정합성 검증은 이미 `pv26/validate_dataset.py`에 별도 도구가 있으므로,
학습 hot path에서는 검증을 끄거나, 디버그 플래그로 gating하는 편이 일반적으로 맞다.

### 2) Validation 메트릭 계산 경로가 불필요한 H2D 및 동기화를 유발한다

`tools/train/train_pv26.py::validate()`는 loss 계산을 위해 `target_batch`를 GPU로 옮긴 뒤에도,
IoU 메트릭 업데이트에서 CPU 마스크를 다시 `.to(device)`로 샘플 단위 전송한다.

- 위치:
  - DA IoU: `update_binary_iou(..., da_mask_cpu[i].to(device=device))`
  - RM IoU: `update_binary_iou(..., rm_mask_cpu[i, c].to(device=device))`
- 결과:
  - 이미 배치 단위로 한 번 옮긴 마스크를 다시 옮기는 형태라 H2D 오버헤드가 누적된다.
  - `tools/train/common.py::update_binary_iou()`가 `.sum().item()`를 사용해 통계를 갱신하므로,
    GPU 동기화가 자주 발생할 수 있다.

1 epoch wall-time을 "train+val"로 잡아 비교하면, VAL 메트릭 경로가 GPU 개선 효과를 희석시키는 케이스가 흔하다.

### 3) Modal 기본값이 throughput 측정/실학습에 불리한 설정을 포함한다

`tools/train/modal_train_pv26.py` 기본값에는 아래가 포함된다.

- `DEFAULT_PROFILE_SYNC_CUDA=True`로 `--profile-sync-cuda`를 항상 전달
  - 이는 `train_one_epoch()`에서 스텝마다 여러 번 `torch.cuda.synchronize()`를 호출하게 만들어
    비동기 overlap을 깨고 처리량을 떨어뜨릴 수 있다.
- 1 epoch처럼 짧은 런에서는
  - dataset stage(copy/extract)
  - `torch.compile` 초기 오버헤드
  등이 섞여 "학습 본체 처리량" 차이가 잘 안 보일 수 있다.

## 빠른 확인법(실제 병목 판별)

학습 로그의 `[profile][train]` 라인을 보면 대략 판별이 가능하다.

- `wait=...ms (wait_pct=...%)`가 크면 DataLoader/IO가 병목일 확률이 높다.
- `thr=...img/s`가 낮고 `wait_p90/p99`가 크면 "파일 IO/전처리" 쪽이 터진다.
- `device=cuda:0` 여부는 `train_pv26.py` 시작 로그에서 확인한다.
- VAL 포함 비교를 할 때는, train thr과 val 처리 시간을 분리해서 본다.

## 개선 우선순위(효과 큰 순)

1. 학습 hot path에서 마스크 `np.unique` 검증 제거 또는 디버그 플래그로 gating
   - 데이터 정합성은 `pv26/validate_dataset.py`로 사전에 보장하는 방향이 일반적.
2. `validate()`에서 IoU 계산 시 "CPU 마스크를 샘플 단위로 GPU로 재전송"하는 구조 제거
   - 이미 `target_batch`에 GPU 마스크가 존재하므로 이를 활용하거나, 전부 CPU로 내려서 CPU에서 처리 등 명확한 방향으로 정리.
3. Modal 래퍼의 `DEFAULT_PROFILE_SYNC_CUDA` 기본값을 throughput 목적에서는 끄고, 프로파일링 목적에서만 켜기
4. 중기 과제: 작은 파일 다발 IO를 줄이기
   - RM 3채널을 하나의 3채널 PNG로 합치기
   - DA+RM을 하나로 패킹하거나 shard(LMDB/WebDataset)로 묶기
   - 고정 입력 해상도(960x544)를 전제로 한다면 letterbox를 오프라인으로 미리 적용

## 메모

이 문서는 "코드 구조와 병목 후보"를 브리핑하는 목적이며, 실제 속도 개선 폭은
데이터 저장 매체(SSD/NFS), CPU 코어 수, DataLoader workers 설정, 배치 크기, GPU 종류에 따라 달라진다.

