# PV26 Training Pipeline Compendium (2026-03-03)

주의:
- 이 문서는 **코드 수정 없이** “무엇이 비효율인지”와 “어떤 해결 방향이 있는지”를 최대한 구체적으로 설명합니다.
- 성능은 머신(CPU/SSD/NFS), DataLoader workers, batch size, GPU, PyTorch/driver 버전 등에 크게 의존합니다. 여기서 말하는 “병목”은 **코드 구조상 병목이 되기 쉬운 지점**과 **현재 코드에 존재하는 확실한 비효율**을 구분해서 서술합니다.

---

## 0. 문제 배경: 왜 “Modal GPU = 내 랩탑” 1 epoch 시간이 나올 수 있나

사용자가 관측한 현상은 다음 형태입니다.

- 로컬 랩탑에서 PV26를 1 epoch 학습하는 데 걸리는 시간과,
- Modal에서 GPU(A10G 등)를 빌려 1 epoch 학습하는 데 걸리는 시간이
- **큰 차이가 없었다**

직관적으로는 GPU가 있으면 epoch 시간이 확 줄어야 할 것 같지만, 실제로는 아래 조건에서 “GPU를 써도 속도가 안 나오는” 케이스가 흔합니다.

1. **데이터 로딩/전처리(=CPU+IO)가 병목**일 때
- GPU는 계산을 빨리 끝내고, 다음 batch를 기다리면서 놀게 됩니다.
- DataLoader의 “wait time(다음 batch를 받기까지 기다리는 시간)”이 커집니다.

2. **validation(VAL) 경로가 비효율**이고 “epoch time”을 train+val로 재면
- train만 빨라져도 val이 느리면 epoch 총 시간이 크게 안 줄 수 있습니다.
- 특히 per-sample `.item()` sync와 CPU↔GPU 재복사 같은 패턴은 VAL 시간을 불필요하게 늘립니다.

3. **프로파일링/동기화 옵션이 기본으로 켜져** throughput을 깎을 때
- `torch.cuda.synchronize()`가 자주 들어가면 비동기 overlap이 깨지면서 처리량이 감소합니다.
- “정밀 타이밍”이 목적이면 필요하지만, “최대 처리량”이 목적이면 기본은 꺼져 있어야 하는 옵션입니다.

4. **짧은 런(1 epoch)**에서는 초기 오버헤드가 큰 비중을 차지할 때
- Modal 쪽에서는 dataset stage(복사/압축해제) + 환경 초기화가 섞일 수 있습니다.
- `torch.compile`은 첫 몇 step에서 컴파일 비용이 들어갑니다.
- 즉, 1 epoch 비교만으로 “steady-state throughput”을 판단하기 어렵습니다.

요약하면, GPU가 있어도 “계산”이 아니라 “데이터”나 “검증/프로파일링”에 시간이 쓰이면 epoch 시간이 비슷하게 나올 수 있습니다.

---

## 1. 현재 코드 베이스에서의 학습 파이프라인 구성(파일 맵)

학습에 직접 관여하는 주요 파일은 아래입니다.

- 학습 엔트리포인트(로컬/원격 공통): `tools/train/train_pv26.py`
- Modal 원격 실행 래퍼: `tools/train/modal_train_pv26.py`
- Dataset/전처리: `pv26/torch_dataset.py`
- Mask 유틸/검증: `pv26/masks.py`
- Loss(criterion): `pv26/criterion.py`
- Model: `pv26/multitask_model.py`
- Validation metric 유틸: `tools/train/common.py`
- 오프라인 데이터셋 검증 도구: `pv26/validate_dataset.py`

이 문서에서 다루는 “비효율”은 주로 다음 구간에 존재합니다.

- Dataset `__getitem__`가 수행하는 IO/PIL/NumPy/검증/letterbox
- Collate가 worker에서 수행하는 batch 구성 비용
- Train/Val 루프의 device 이동, sync, metric 계산 방식
- Criterion 내부의 GPU→CPU sync를 유발하는 Python bool 분기
- Profiling 기본값이 학습 기본 경로에 주는 오버헤드
- torch.compile과 fullgraph 관점에서 graph break를 만드는 패턴(hook, Python side effect)
- Modal 래퍼가 짧은 런에서 비용을 키우는 staging 정책

---

## 2. PV26 데이터/태스크 계약(왜 마스크가 255(ignore)를 가지는가)

PV26 학습은 멀티태스크를 전제로 합니다.

- OD(Detection)
- DA(Drivable Area) = binary segmentation
- RM(Road Marking) = 3채널 sigmoid segmentation

데이터셋 정책은 “부분 라벨(partial label) 지원”입니다.

- 어떤 샘플은 DA만 있고 OD/RM이 없을 수 있습니다.
- 어떤 샘플은 RM 일부 채널만 supervision이 있을 수 있습니다.
- 이런 경우 “라벨이 없는 태스크/채널을 0으로 채우면” 오학습이 생기므로,
  라벨 부재는 “전 픽셀 ignore(255)” + `has_* = 0`으로 표현하고 loss에서 제외합니다.

이 정책은 문서로도 정리되어 있고(`docs/DATASET_PROFILE_ANALYSIS.md`), 코드로는 아래가 핵심입니다.

- ignore 값: `pv26/masks.py`의 `IGNORE_VALUE = 255`
- Dataset은 `has_*==0`이면 `make_all_ignore_mask()`로 255만 채워 반환
- Criterion은 `mask != 255`로 valid 픽셀만 골라 loss 계산(DA/RM)

중요 포인트:
- 이 정책 자체는 “정확도/학습 안정성” 관점에서 매우 타당합니다.
- 다만 이 정책을 구현하는 방식(특히 마스크 검증/resize)을 학습 hot path에서 과하게 수행하면,
  “성능(throughput)”을 크게 갉아먹을 수 있습니다.

---

## 3. 실행 흐름 상세(Train/Val)

### 3.1 로컬/일반 실행: `tools/train/train_pv26.py`

핵심 인자/기본값(요약이 아니라 실제 성능 영향 포인트):

- `--device auto`: CUDA 가능하면 GPU로, 아니면 CPU
- `--compile` 기본 ON + `--compile-mode default` 기본
- `--amp` 기본 ON(CUDA일 때만)
- `--workers 6`, `--prefetch-factor 4`, `--persistent-workers` 기본 ON
- `--profile-every 10` 기본(0이면 비활성화)
- `--profile-sync-cuda` 옵션이 켜지면 step 타이밍 측정 구간마다 `torch.cuda.synchronize()`가 들어감
- `--progress` 기본 ON(tqdm)

Train loop는 `train_one_epoch()` 내부에서 아래 단계를 반복합니다.

1. `next(loader_it)`로 batch 수령
2. `_prepare_images_for_model()`로 이미지 H2D 및 정규화/`channels_last`
3. `_move_prepared_batch_to_device()`로 타깃 텐서 이동
4. `optimizer.zero_grad(set_to_none=True)`
5. autocast(AMP) 컨텍스트에서 forward + loss
6. backward + optimizer step
7. 프로파일링 윈도우마다 stage time을 출력

여기서 “DataLoader 병목” 여부는 `t_wait`(다음 batch를 받기까지의 대기 시간)를 보면 상당 부분 판단이 됩니다.

### 3.2 Modal 실행: `tools/train/modal_train_pv26.py`

Modal 래퍼는 “원격 컨테이너에서 train_pv26.py를 실행하는 것”에 더해 다음 일을 합니다.

- dataset을 Modal Volume에서 컨테이너 로컬 SSD(`/tmp/pv26_dataset_cache`)로 복사
- tar가 있으면 tar를 복사한 뒤 extract(기본 정책은 매번 기존 로컬 디렉토리 삭제 후 재추출)
- artifact를 다른 Volume에 저장하고, 로컬로 주기적으로 sync(체크포인트/텐서보드)
- 기본값으로 `--profile-sync-cuda`를 켜서 정밀 프로파일링 형태로 실행

짧은(1 epoch) 런에서는 dataset stage의 복사/압축해제, 로그 sync 등이 “epoch time”에 섞여 들어가
순수 학습 throughput 비교가 더 어려워질 수 있습니다.

---

## 4. 데이터 경로 상세: `pv26/torch_dataset.py`

### 4.1 Dataset 계약: manifest-driven

`Pv26ManifestDataset`는 `meta/split_manifest.csv`가 source of truth입니다.

- `__init__`에서 manifest를 읽고, split별 row 리스트를 구성
- `__getitem__`에서 row를 기반으로 파일을 열어 샘플을 생성

### 4.2 `__getitem__`에서 실제로 하는 일(샘플 1개당)

샘플당 수행 작업(현재 구현 기준):

- RGB 이미지: `Image.open(...)` + `convert("RGB")`
- Det: `det_relpath`의 YOLO txt를 read/parse (`_load_yolo_txt`)
- DA mask: png read (있으면), 없으면 all-255 mask 생성
- RM masks 3장: 각각 png read (있으면), 없으면 all-255 mask 생성
- Letterbox(기본 960x544):
  - 이미지: bilinear resize + padding
  - 마스크: nearest resize + padding
  - det bbox: 좌표 스케일/패딩 반영
- Online augmentation(옵션):
  - color jitter(PIL Enhance)
  - hflip(NumPy flip + det cx 변환)
- 최종적으로 NumPy array로 변환 후 torch tensor로 변환

즉, 샘플 1개당 “작은 파일 여러 개”를 열고(PIL), 리사이즈/패딩하고(이미지+마스크), NumPy→torch 변환까지 모두 수행합니다.

이 방식 자체는 이해하기 쉽고 디버깅이 쉽습니다.
하지만 throughput 관점에서는 “IO+PIL+NumPy+검증”이 매우 쉽게 병목이 됩니다.

---

## 5. Collate 상세: `_collate_with_images` (DataLoader worker에서 실행)

`tools/train/train_pv26.py::_collate_with_images()`는 worker 프로세스에서 아래를 수행합니다.

- `images = torch.stack([s.image ...])`
- `da_mask/rm_mask` stack
- `has_*` 텐서화
- detection의 per-sample `[N,5]`를 그대로 리스트로 보관(`det_yolo`)
- `det_label_scope` 문자열 리스트
- `sample_id` 문자열 리스트
- Ultralytics E2E loss용 flat target 생성:
  - `det_tgt_batch_idx`
  - `det_tgt_cls`
  - `det_tgt_bboxes`
  - `det_scope_code`(full/subset/none를 0/1/2로 인코딩)

장점:
- main thread가 GPU 연산에 집중하도록 “배치 구성”을 worker에서 함.
- Ultralytics loss에 필요한 target을 train loop에서 매번 만들지 않음.

단점/개선 여지(GPT Pro 지적):
- train path에서 굳이 필요 없는 “문자열/리스트 메타데이터”까지 같이 싣고 감.
- eval/val에서는 필요하지만 train에서는 불필요한 항목이 있어, collate를 분리하면 train 경로가 가벼워질 수 있음.

---

## 6. Train loop 프로파일링 구조(이미 내장된 계측)

`train_one_epoch()`는 stage time을 다음처럼 나눠서 측정합니다.

- `wait`: dataloader에서 다음 batch를 받기까지 걸린 시간
- `h2d`: 이미지/타깃을 GPU로 옮기고 정규화/channels_last 등을 수행한 시간(이름은 h2d지만 내부에 normalize 포함)
- `fwd`: forward 시간
- `loss`: criterion 시간
- `bwd`: backward 시간
- `opt`: optimizer step 시간

추가로 profiling 윈도우에서 다음을 출력합니다.

- `thr` / `thr_no_wait`: 이미지 처리량(대기 포함/미포함)
- `wait_p50/p90/p99`: wait time 분포
- CPU rss/load
- CUDA mem stats
- `nvidia-smi` 간이 유틸리티

이 계측은 병목을 매우 빠르게 가르는 데 도움이 되지만, “항상 켜져 있으면” 오버헤드가 됩니다(아래 10절 참조).

---

## 7. Criterion(Loss) 상세와 GPU sync 문제: `pv26/criterion.py`

PV26Criterion은 멀티태스크 loss를 합산합니다.

- OD:
  - `od_loss_impl="dense"`: dense YOLO-style loss(파이썬 루프가 많아 성능용이 아님)
  - `od_loss_impl="ultralytics_e2e"`: Ultralytics E2ELoss를 사용(현재 기본 경로)
- DA:
  - BCE-with-logits, `mask != 255`만 valid로 취급
- RM:
  - per-channel focal + dice, `mask != 255`만 valid로 취급

### 7.1 GPT Pro가 지적한 “남은 GPU sync”는 실제로 존재한다

현재 criterion에는 아래 형태의 분기가 존재합니다.

- `if not bool(valid_code.all()): ...`
- `if bool(m.any()): ...`
- `if not bool(keep.any()): ...`

이 패턴은 “CUDA 텐서를 Python bool로 평가”하는 순간 CPU↔GPU 동기화가 발생할 수 있고,
`torch.compile`/CUDA graphs 관점에서도 불리합니다.

또한 `_od_loss_ultralytics()` 일부 분기에서는 `int(has_det[i].item())`처럼 `.item()` 기반 분기가 있습니다.
이는 특정 상황에서 sync를 유발할 수 있습니다.

### 7.2 해결 방향(원칙)

핵심은 다음 두 가지입니다.

1. hot path에서 “GPU 텐서를 Python bool로 바꾸는 분기”를 없애기
2. “빈 텐서/비감독 배치” 같은 케이스를 branchless reduction으로 처리하기

GPT Pro 문서에 있는 branchless 예시는 방향성이 맞습니다.

DA/RM 예시:

```python
# DA: keep는 [B] bool
keep_f = keep.to(dtype=per_sample.dtype)
return (per_sample * keep_f).sum() / keep_f.sum().clamp_min(1.0)

# RM: keep는 [B,C] bool
keep_f = keep.to(dtype=per_channel.dtype)
return (per_channel * keep_f).sum() / keep_f.sum().clamp_min(1.0)
```

OD/ultralytics 예시(핫패스 검사 축소 + 마스킹 기반):

```python
keep_mask = has_det.ne(0)
if det_scope_code is not None:
    keep_mask = keep_mask & det_scope_code.ne(2)  # none 제외
keep_idx = keep_mask.nonzero(as_tuple=False).squeeze(1)

# old_to_new로 batch idx 재매핑 후, m.any() 같은 python 분기 제거
new_idx = old_to_new[src_old]
m = new_idx.ge(0)
batch_idx_t = new_idx[m].to(dtype=torch.float32)
...
```

주의:
- “검사 축소”는 데이터가 collate 단계에서 이미 정제된다는 전제가 필요합니다.
- 안전성을 유지하려면 “오프라인 validator” 또는 “디버그 모드에서만 엄격 체크” 같은 대체 장치가 있어야 합니다.

---

## 8. Validation 경로 상세와 비효율: `tools/train/train_pv26.py::validate`

validation은 아래 일을 한 번에 합니다.

- forward + loss 계산
- DA/RM IoU 메트릭 계산
- OD mAP50 계산

### 8.1 확실한 비효율 1: 마스크를 이미 GPU로 옮겨놓고도, 샘플 단위로 다시 GPU로 옮김

`validate()`는 처음에 다음을 수행합니다.

- `target_batch = _move_prepared_batch_to_device(target_batch_cpu, device=device)`

즉, `target_batch["da_mask"]`, `target_batch["rm_mask"]`는 이미 device로 올라가 있습니다.

그런데 IoU 업데이트에서는 다음이 있습니다.

- `da_mask_cpu[i].to(device=device)`
- `rm_mask_cpu[i, c_idx].to(device=device)`

이건 “이미 올린 걸 또 올리는” 형태라 그대로 손해입니다.

### 8.2 확실한 비효율 2: per-sample metric 계산이 `.item()`를 통해 잦은 sync를 유발

`update_binary_iou()`는 아래 형태로 누적합니다.

- `int((pred_valid & tgt_valid).sum().item())`
- `int((pred_valid | tgt_valid).sum().item())`

validation에서 샘플마다 이것을 호출하면, GPU에서 계산한 sum을 CPU에서 `.item()`로 받는 순간
동기화가 반복될 수 있습니다.

### 8.3 해결 방향(원칙)

1. validation에서는 이미 올라간 device mask를 재사용한다.
- `da_mask_dev = target_batch["da_mask"]`
- `rm_mask_dev = target_batch["rm_mask"]`

2. IoU 통계는 가능한 한 “배치 단위 reduction”으로 계산하고, CPU로 넘기는 것은 배치당 1회(또는 몇 회)로 줄인다.

GPT Pro가 제시한 DA batch-wise 예시는 합리적입니다.

```python
da_mask = target_batch["da_mask"]
has_da = target_batch["has_da"].bool()
valid_da = (da_mask != 255) & has_da[:, None, None]
pred_da = preds.da[:, 0] > 0
tgt_da = da_mask == 1
inter = (((pred_da & tgt_da) & valid_da).sum())
union = (((pred_da | tgt_da) & valid_da).sum())
supervised = has_da.sum()
# inter/union/supervised를 CPU로 가져오는 횟수를 최소화
```

RM도 채널별로 같은 방식으로 배치 reduction이 가능합니다.

추가 고려(정확도/정합성):
- `has_*`가 0인 샘플은 valid mask가 전부 False가 되도록 보장해야 합니다.
- ignore(255) 픽셀은 valid에서 제외해야 합니다.

---

## 9. Dataset/Loader가 병목이 되는 “가장 현실적인 이유”: IO + PIL + NumPy + (추가) 검증 비용

GPT Pro 문서에서도 dataset이 병목이라고 했고, audit 문서에서는 더 구체적으로 “마스크 검증(np.unique)이 hot path에 있다”를 강조했습니다.

### 9.1 확실한 비효율: 마스크 검증이 학습 hot path에 포함되어 있음(`np.unique`)

`pv26/torch_dataset.py::_letterbox_mask_u8()` 내부에서:

- 입력 마스크에 대해 `validate_binary_mask_u8(mask_u8, ...)`
- letterbox 출력 마스크에 대해 `validate_binary_mask_u8(out, ...)`

즉, 마스크 1장당 검증이 2번, RM이 3장 + DA 1장 = 최대 4장이라면 “샘플당 최대 8회”가 실행될 수 있습니다.

그리고 `pv26/masks.py::validate_binary_mask_u8()`는 내부에서 `np.unique(mask)`를 수행합니다.
이건 마스크 전체를 스캔하는 O(HW) 연산이며, 720x1280 같은 큰 마스크에서는 상당히 비쌉니다.

audit 문서에서 로컬 기준으로 측정한 대략 값(환경에 따라 달라짐):

- DA mask 로드(PIL->np): 평균 ~3ms
- `validate_binary_mask_u8(np.unique)`만: 평균 ~9ms
- Dataset `__getitem__` 평균: ~79ms/샘플

즉, GPU가 아무리 빨라도 loader가 1초에 샘플을 충분히 공급하지 못하면 `wait`가 증가하고 throughput이 떨어집니다.

### 9.2 “부분 라벨 샘플”에서도 불필요한 리사이즈/검증 비용을 지불할 가능성

현재 구현은 `has_*==0`이면 원본 해상도에서 `make_all_ignore_mask(in_h, in_w)`를 만들고,
그 이후 letterbox 단계에서 “그 all-255 마스크”를 resize+pad(그리고 검증)합니다.

이건 정확성은 유지하지만, 성능 관점에서는 “어차피 255만 있는 마스크”에 대해
리사이즈/검증을 수행하는 비용이 됩니다.

즉, “비감독 채널”일수록 오히려 더 싸게 만들어야 하는데, 현재는 구조상 그렇지 않습니다.

### 9.3 해결 방향(원칙)

1. 데이터 정합성 검증은 학습 hot path가 아니라, 오프라인 validator로 강제한다.
- 이미 `pv26/validate_dataset.py`가 존재합니다.
- 학습 중에는 검증을 끄거나, 디버그 플래그로만 켜는 게 throughput에 유리합니다.

2. “all-255 ignore 마스크”는 letterbox 결과 크기로 바로 생성해 resize 자체를 피한다.
- 예: output size가 544x960이면, unsupervised 채널은 처음부터 그 크기의 ignore mask를 만들고 넘어가는 방식.

3. 장기적으로는 오프라인 캐시/샤딩이 가장 큰 ROI다.
- letterbox 결과(960x544)를 미리 저장
- det target도 텍스트가 아니라 구조화된 형태로 미리 저장
- 다수의 작은 파일을 shard(LMDB/WebDataset tar/pt shard)로 묶어 IO 효율을 올림

---

## 10. Profiling/로그가 기본 경로에서 유발하는 오버헤드

GPT Pro 문서의 주장:
- profiling 기본값은 “꺼짐”이 맞다.

현재 구조를 보면 이 주장은 타당합니다.

- `--profile-every` 기본값이 10이라, 기본 실행에서도 주기적으로 프로파일 로그를 찍습니다.
- profiling window에서 `nvidia-smi`를 subprocess로 호출합니다.
- 그리고 train loop에서는 CUDA device일 때 `torch.cuda.mem_get_info()`를 매 batch에서 호출합니다(프로파일링 여부와 무관하게).

이 계측 자체는 “디버깅/분석”에는 좋습니다.
하지만 “실학습/throughput”이 목적이면 기본은 꺼져 있어야 하고, 필요할 때만 켜는 구조가 맞습니다.

Modal 래퍼는 더 강하게 profiling 중심 세팅입니다.

- `DEFAULT_PROFILE_SYNC_CUDA=True`로 `--profile-sync-cuda`가 기본으로 전달됩니다.
- 이 옵션이 켜지면, `train_one_epoch()`가 stage time 측정마다 `torch.cuda.synchronize()`를 호출합니다.
- 정밀 타이밍은 얻지만, throughput은 떨어질 수 있습니다.

---

## 11. torch.compile / fullgraph 관점 이슈(특히 hook)

현재 기본은 `torch.compile(mode="default")`입니다.
이 선택은 GPT Pro 문서에서도 “균형이 좋다”는 평가를 했고, 일반적으로도 타당합니다.

하지만 compile/fullgraph 관점에서 “수상한” 패턴이 하나 있습니다.

- `PV26MultiHeadYOLO26`는 backbone feature(P3)를 얻기 위해 forward hook을 등록합니다.
- hook은 Python dict(`self._feat`)에 Tensor를 저장하는 side-effect를 가집니다.

이 패턴은 eager에서는 잘 동작할 수 있지만,
`fullgraph=True`나 CUDA graph 캡처 관점에서는 graph break의 단서가 될 수 있습니다.

따라서:
- `--compile-fullgraph` 옵션을 둔 것은 매우 좋은 “진단 도구”입니다.
- fullgraph에서 문제가 터지면, 첫 번째 용의자는 hook+dict side-effect입니다.

중장기 해결 방향(원칙):
- hook 없이, trunk wrapper가 필요한 feature를 명시적으로 반환하거나
- segmentation head용 feature extraction을 “그래프 친화적인 방식”으로 재구성합니다.

---

## 12. Collate 분리 + drop_last 논의(성능 vs 실험/재현성)

GPT Pro 문서의 제안:
- `_collate_train` / `_collate_eval` 분리
- train loader에서 `drop_last=True` 권장(compile/CUDA graphs 친화)

이 제안은 “compile을 적극적으로 쓰고, shape 안정성을 최대화한다”는 목표에서는 타당합니다.

다만 다음 트레이드오프를 반드시 인지해야 합니다.

- `drop_last=True`는 epoch마다 학습에 사용되는 샘플 수가 줄어듭니다.
- 학습 로그/지표 비교 시 “같은 epoch”의 의미가 달라질 수 있습니다.

따라서 현실적인 운영 방법은 보통 아래 중 하나입니다.

1. throughput/compile 안정성 측정용 벤치에서는 `drop_last=True`
2. 최종 학습(정확도 우선)에서는 `drop_last=False`로 전체 데이터를 최대한 사용

collate 분리는 성능뿐 아니라 구조 분리에도 도움이 됩니다.

- train에서는 “tensor-only” target만 남기면 pinning/전송이 단순해지고 메모리도 줄 수 있습니다.
- eval/val에서는 sample_id, det_yolo 같은 메타가 필요할 수 있습니다.

---

## 13. “전부 다 적용하면 좋아지나?”에 대한 정리(합집합 관점)

두 문서의 제안은 서로 모순되지 않습니다.
대부분의 제안은 서로 독립적이거나, 시너지가 있습니다.

대표적인 시너지:

- Dataset 경량화(검증/불필요 resize 제거)로 `wait` 감소
- Validation 경량화(재복사 제거 + batch-wise metric)로 epoch 총시간 감소
- Criterion sync 제거로 `loss` 구간과 compile 경로 안정화
- Profiling 기본 off로 steady-state throughput 개선 및 비교의 신뢰성 상승
- Collate 분리로 train path 메모리/CPU 부담 감소
- 오프라인 캐시/샤딩으로 “더 빠른 GPU로 갈수록 더 심해지는 병목”을 선제 제거

주의가 필요한 항목:

- “핫패스 검사 제거”는 오프라인 검증이 확실히 돌아간다는 전제가 있어야 합니다.
- `drop_last=True`는 실험 설계/재현성에 영향을 줍니다.
- hook 제거/compile 친화 리팩터는 효과가 있을 수 있지만 작업량이 큽니다(ROI를 따져 단계적으로).

---

## 14. 개선안 총정리(두 문서의 합집합, + 실행/검증 방법 포함)

아래는 “무엇이 비효율인지”와 “해결 방향”을 항목별로 묶어 정리한 합집합입니다.
각 항목은 가능한 한 “왜 느려지는지”와 “무슨 형태로 고치는지”를 함께 적습니다.

### 14.1 Dataset hot path의 마스크 검증(np.unique) 제거/게이팅

비효율:
- 학습 중 매 샘플, 매 마스크 letterbox에서 `validate_binary_mask_u8()`가 호출되고, 이는 `np.unique`를 수행합니다.

왜 문제인가:
- CPU에서 대규모 메모리 스캔이 반복되어 DataLoader throughput을 직접 깎습니다.
- 빠른 GPU일수록 `wait`가 커지며 GPU가 놀 가능성이 큽니다.

해결 방향:
- 오프라인 validator(`pv26/validate_dataset.py`)를 “데이터 생성 파이프라인의 필수 단계”로 두고,
  학습 hot path에서는 검증을 끄거나 디버그 옵션일 때만 실행합니다.

검증 방법:
- 학습 로그의 `[profile][train]`에서 `wait_pct`, `thr(img/s)`가 개선되는지 확인합니다.
- `workers`를 늘려도 `wait`가 줄지 않으면 여전히 dataset이 병목일 가능성이 큽니다.

### 14.2 비감독(all-255) 마스크의 불필요 letterbox/검증/resize 제거

비효율:
- `has_*==0`인 채널도 원본 해상도 ignore 마스크를 만든 뒤 letterbox 단계에서 resize+검증을 수행합니다.

왜 문제인가:
- supervision이 없는 채널은 정보량이 0인데 비용은 지불합니다.

해결 방향:
- output(960x544) 크기의 ignore 마스크를 바로 생성해, resize/검증 자체를 건너뜁니다.

검증 방법:
- `profile`에서 `wait` 감소(로더가 더 빨리 batch 공급)로 나타날 가능성이 큽니다.

### 14.3 Validation에서 device mask 재사용 + batch-wise IoU 누적

비효율:
- `target_batch`를 이미 device로 옮겼는데, IoU 계산에서 CPU 마스크를 샘플 단위로 다시 `.to(device)` 합니다.
- `update_binary_iou()`가 `.item()` 기반 누적이라 sync가 잦습니다.

왜 문제인가:
- epoch wall-time을 train+val로 측정하면 val이 발목 잡습니다.
- 특히 장기 학습(수십~수백 epoch)에서는 val 누적 시간이 커집니다.

해결 방향:
- `da_mask_dev/rm_mask_dev`를 `target_batch`에서 직접 쓰도록 경로를 정리합니다.
- 가능한 한 배치 단위로 reduction 후 CPU로 이동 횟수를 줄입니다.

검증 방법:
- val 단계의 wall-time이 줄고, GPU util이 더 안정적으로 나오는지 확인합니다.

### 14.4 Criterion 내부의 GPU sync 제거(branchless reduction)

비효율:
- `bool(tensor.any())` / `bool(tensor.all())` / `.item()` 기반 분기가 hot path에 존재합니다.

왜 문제인가:
- CPU-GPU 동기화가 들어가면 throughput과 compile/CUDA graph에 불리합니다.

해결 방향:
- keep mask 기반의 branchless reduction으로 바꿉니다.
- “검사”는 디버그/오프라인 단계로 이동합니다.

검증 방법:
- `loss` stage time이 감소하는지, `torch.compile` 성능이 안정화되는지 확인합니다.

### 14.5 Train/Eval collate 분리 + train drop_last 옵션(조건부)

비효율:
- train path에서 불필요한 메타데이터(문자열 리스트 등)를 같이 싣고 갑니다.

왜 문제인가:
- 메모리/CPU 오버헤드가 누적됩니다.
- compile/CUDA graphs는 shape 안정성이 중요합니다.

해결 방향:
- train은 tensor-only target 중심의 collate로 최소화합니다.
- throughput/compile 벤치에서는 `drop_last=True`를 고려합니다.

검증 방법:
- `wait`/`h2d`가 줄어드는지, 컴파일 안정성이 올라가는지 확인합니다.

### 14.6 Dataset 오프라인 캐시/샤딩(가장 큰 ROI 가능)

비효율:
- 샘플당 다수 파일 open + PIL resize + numpy 변환 + torch 변환을 반복합니다.
- det txt를 매번 파싱합니다.

왜 문제인가:
- 더 빠른 GPU로 갈수록 데이터 파이프라인이 먼저 병목이 됩니다.
- 작은 파일 다발 IO는 네트워크/NFS 환경에서 특히 치명적입니다.

해결 방향:
- 960x544 letterbox 결과를 미리 저장(이미지/마스크).
- det targets(`det_tgt_*`)도 미리 직렬화(텍스트 파싱 제거).
- LMDB/WebDataset tar/pt shard로 묶어서 IO 효율을 크게 개선합니다.
- online augmentation은 color jitter / hflip 정도로 최소화합니다.

검증 방법:
- `workers`를 크게 올리지 않아도 `wait`가 낮고 `thr(img/s)`가 올라가야 합니다.

### 14.7 Profiling 기본값을 throughput 친화로(기본 OFF)

비효율:
- 기본 실행에서 profiling/logging이 주기적으로 돌면서 오버헤드가 됩니다.
- Modal은 기본 `--profile-sync-cuda`로 동기화 오버헤드를 넣습니다.

해결 방향:
- regular training 기본값은 profiling off(`profile_every=0`)로 둡니다.
- `mem_get_info`, `nvidia-smi` 등도 profiling window에서만 수행합니다.
- Modal 기본값도 throughput 모드에서는 sync를 끕니다.

검증 방법:
- 동일 설정에서 처리량이 개선되는지 확인합니다.

### 14.8 torch.compile / fullgraph 진단 및 hook 리팩터(중장기)

비효율:
- forward hook + Python dict side-effect는 compile/fullgraph에서 graph break 요인이 될 수 있습니다.

해결 방향:
- `--compile-fullgraph`로 문제 재현/진단을 먼저 합니다.
- 중장기적으로 hook 없는 feature extraction 구조로 바꿉니다.

검증 방법:
- fullgraph에서 에러/graph break가 줄고 compile 성능이 안정화되는지 확인합니다.

### 14.9 TF32 관련(낮은 우선순위, 업그레이드 시 정리)

현황:
- 현재는 `allow_tf32=True`를 사용합니다.

해결 방향:
- PyTorch 향후 버전에서 권장 API(`set_float32_matmul_precision` 등)로 정리할 수 있습니다.

검증 방법:
- 정확도/속도 트레이드오프를 별도 실험으로 확인합니다.

---

## 15. “무엇부터 손댈지”를 결정하는 실전 판별 가이드(프로파일 로그 중심)

이 섹션은 “요약”이 아니라, 실제로 다음 액션을 고를 때 필요한 판별 규칙을 구체적으로 적습니다.

1. `[profile][train]`에서 `wait_pct`가 크다
- DataLoader/IO/전처리가 병목일 확률이 큼.
- 우선순위: dataset hot path 경량화(검증/resize/캐시/샤딩) 계열.

2. `wait`는 낮은데 `loss` 시간이 비정상적으로 크다
- criterion 내부 sync/분기/불필요 검사가 병목일 수 있음.
- 우선순위: criterion bool 분기 제거, ultralytics loss target 경로 최적화.

3. “train만 재면 빨라졌는데 epoch time은 그대로”
- val이 병목일 가능성이 큼.
- 우선순위: validate의 재복사 제거, batch-wise metric, `.item()` 감소.

4. Modal에서만 유독 느리고, 로그에 dataset stage(copy/extract)가 많이 보인다
- 원격 환경의 dataset staging이 비용일 수 있음.
- 우선순위: 실험 설계(2epoch 이상, stage 제외) 또는 staging 캐시 정책.

5. `torch.compile` 켜면 첫 epoch만 유독 느리다
- compile warmup/컴파일 비용이 섞인 것일 수 있음.
- 우선순위: 첫 epoch 제외한 steady-state 측정 또는 `--no-compile`로 분리 측정.

---

## 16. 참고 링크(입력 문서가 인용한 PyTorch 문서)

GPT Pro 문서가 인용한 참고 링크는 아래입니다(원문 링크 유지).

- [1] https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html
- [2] https://docs.pytorch.org/docs/stable/generated/torch.compile.html
- [3] https://docs.pytorch.org/docs/stable/data.html
- [4] https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/compile/programming_model.fullgraph_true.html
- [5] https://docs.pytorch.org/docs/stable/notes/cuda.html

