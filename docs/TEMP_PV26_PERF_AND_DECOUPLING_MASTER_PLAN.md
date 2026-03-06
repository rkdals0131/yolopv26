# TEMP_PV26 성능/구조 개선 마스터 플랜

이 문서는 현재 PV26 학습 경로의 병목과 구조 의존성을 분석한 뒤, **선택 가능한 개선안들을 전부 실행하는 것을 전제로** 한 단계별 실행 계획서다.  
목표는 단순 미세 최적화가 아니라, 다음 세 가지를 함께 달성하는 것이다.

1. 학습 step time, 특히 seg loss/backward 구간을 유의미하게 줄이기
2. compile / graph capture 친화적인 구조로 정리하기
3. `Ultralytics` 내부 구조, `forward hook`, `self._feat` side effect에 대한 결합을 낮추기

추가 전제:

- 최종 학습 대상 데이터는 **BDD + ETRI + RLMD + WOD(Waymo)**를 모두 포함하는 PV26 v1 멀티소스 셋이다.
- 현재 로컬에 존재하는 PV26 v1 루트는 분리되어 있다.
  - `/home/user1/Storage/seg_dataset/pv26_v1_bdd_full`
  - `/home/user1/Storage/seg_dataset/pv26_v1_etri`
  - `/home/user1/Storage/seg_dataset/pv26_v1_rlmd`
  - `/home/user1/Storage/seg_dataset/pv26_v1_waymo_minimal_1ctx`
- 현재 학습 코드는 `--dataset-root` 하나만 받는 단일-manifest 경로이므로, **최종 멀티소스 학습은 merged root 또는 multi-root loader 없이는 직접 실행할 수 없다**: `tools/train/train_pv26.py:42`, `pv26/torch_dataset.py:255`, `pv26/torch_dataset.py:269`

---

## 1. 현재 상태에서 확인된 사실

### 1.0 데이터셋 범위와 현재 제약

- 각 PV26 v1 dataset manifest에는 `source` 컬럼이 존재하므로, 멀티소스 통합 후에도 source-aware 분석이 가능하다.
- 확인된 source별 루트:
  - `pv26_v1_bdd_full`
  - `pv26_v1_etri`
  - `pv26_v1_rlmd`
  - `pv26_v1_waymo_minimal_1ctx`
- 현재 `Pv26ManifestDataset`는 단일 `dataset_root/meta/split_manifest.csv`만 읽는다: `pv26/torch_dataset.py:255`, `pv26/torch_dataset.py:269`
- 따라서 앞으로의 성능 최적화는 최종적으로 **“멀티소스 통합 입력 경로”** 위에서 다시 검증해야 한다.

### 1.1 입력/출력 해상도

- 현재 기본 학습 입력은 letterbox 기준 `960x544`이다: `pv26/torch_dataset.py:26`, `pv26/torch_dataset.py:30`
- DA/RM/lane-subclass logits는 모두 최종적으로 입력 해상도 `H x W`로 복원된다: `pv26/multitask_model.py:182`, `pv26/multitask_model.py:187`, `pv26/multitask_model.py:188`, `pv26/multitask_model.py:189`
- YOLO26 기반 경로도 동일하게 full-resolution logits를 반환한다: `pv26/multitask_model.py:275`, `pv26/multitask_model.py:295`, `pv26/multitask_model.py:296`, `pv26/multitask_model.py:297`

### 1.2 RM 경로 구조

- RM binary logits와 lane-subclass logits는 서로 다른 `RoadMarkingHeadDeconv` 인스턴스를 사용한다: `pv26/multitask_model.py:176`, `pv26/multitask_model.py:177`, `pv26/multitask_model.py:269`, `pv26/multitask_model.py:270`
- 즉, 같은 `p3_head` 입력에서 deconv stack을 두 번 돈다: `pv26/multitask_model.py:293`, `pv26/multitask_model.py:296`, `pv26/multitask_model.py:297`
- 로컬 확인 결과 현재 YOLO26 경로에서 `p3_head`는 `(B, 64, 68, 120)`이고, RM decoder의 최종 full-res hidden은 16채널이다. 배치 128 기준 decoder 하나당 full-res activation만 fp16 약 `1.99 GiB` 수준이다.

### 1.3 loss / profiling 동작

- stage profiling은 `--profile-sync-cuda`가 꺼져 있으면 CUDA sync 없이 `time.perf_counter()`로 측정된다: `tools/train/train_pv26.py:155`, `tools/train/train_pv26.py:161`, `tools/train/train_pv26.py:799`, `tools/train/train_pv26.py:834`, `tools/train/train_pv26.py:842`, `tools/train/train_pv26.py:846`, `tools/train/train_pv26.py:850`, `tools/train/train_pv26.py:855`
- Modal H100/A100 래퍼 기본값은 `DEFAULT_PROFILE_SYNC_CUDA = False`다: `tools/train/modal_h100_train_pv26.py:83`, `tools/train/modal_a100_train_pv26.py:83`
- lane-subclass loss는 의도상 positive pixel만 supervise하지만, 실제 구현은 full-image `F.cross_entropy(..., reduction="none")`를 먼저 계산한 뒤 mask한다: `pv26/criterion.py:471`, `pv26/criterion.py:474`, `pv26/criterion.py:479`, `pv26/criterion.py:481`
- RM loss 역시 full-image BCE/sigmoid/focal/dice를 채널 전체에 대해 계산한다: `pv26/criterion.py:430`, `pv26/criterion.py:436`, `pv26/criterion.py:437`, `pv26/criterion.py:439`, `pv26/criterion.py:445`

### 1.4 compile / graph 친화성

- 현재 `torch.compile()`은 model에만 적용되고 criterion은 eager 상태다: `tools/train/train_pv26.py:747`, `tools/train/train_pv26.py:765`, `tools/train/train_pv26.py:1295`
- `PV26MultiHeadYOLO26`는 `forward hook`으로 backbone feature를 캡처하고 `self._feat` dict를 side effect 저장소로 사용한다: `pv26/multitask_model.py:232`, `pv26/multitask_model.py:234`, `pv26/multitask_model.py:237`, `pv26/multitask_model.py:246`, `pv26/multitask_model.py:277`, `pv26/multitask_model.py:290`

### 1.5 Ultralytics 결합 지점

- 모델 생성 시 `DetectionModel`과 `DEFAULT_CFG`를 직접 import/use 한다: `pv26/multitask_model.py:225`, `pv26/multitask_model.py:226`, `pv26/multitask_model.py:228`, `pv26/multitask_model.py:230`
- feature 추출은 `module index 4 == backbone P3`, `preds["one2many"]["feats"][0] == head P3` 라는 내부 구조 가정에 의존한다: `pv26/multitask_model.py:241`, `pv26/multitask_model.py:242`, `pv26/multitask_model.py:243`, `pv26/multitask_model.py:246`, `pv26/multitask_model.py:286`, `pv26/multitask_model.py:293`
- criterion 역시 `ultralytics.utils.loss.E2ELoss`와 raw `ultra_det_model`에 직접 결합된다: `pv26/criterion.py:39`, `pv26/criterion.py:60`, `pv26/criterion.py:63`, `pv26/criterion.py:70`, `pv26/criterion.py:76`
- train script가 `model.det_model`을 직접 criterion에 넘기고 있어 학습 경로도 Ultralytics 내부 객체를 관통한다: `tools/train/train_pv26.py:1250`, `tools/train/train_pv26.py:1254`

---

## 2. 최종 결정

이번 작업은 아래 항목을 **전부 실행**하되, 한 번에 뒤섞지 않고 **위험도/파급도 순서로 분해**해서 진행한다.

0. 멀티소스 입력 경로 확정
1. 정확한 baseline 재측정
2. 저위험 hot-path 최적화
3. criterion tensorization + compile 친화화
4. RM decoder 공유
5. segmentation 출력 해상도 축소
6. Ultralytics / hook / side-effect 결합 해소
7. 최종적으로 whole-step compile / graph 가능성 재평가

핵심 원칙은 다음과 같다.

- 각 phase는 **독립적으로 검증 가능**해야 한다.
- 각 phase 뒤에는 **속도 + 정확도 + 메모리**를 반드시 다시 측정한다.
- 구조 변경과 성능 변경을 구분해서, 성능 regression의 원인을 추적 가능하게 유지한다.

---

## 3. 실행 순서 요약

| Phase | 목적 | 난이도 | 기대 효과 |
|---|---|---:|---:|
| M0 | 멀티소스 입력 경로 확정 | 중간 | 최종 실험 기반 확보 |
| 0 | 정확한 baseline/profiling 확보 | 낮음 | 분석 정확도 확보 |
| 1 | 저위험 hot-path 정리 | 낮음 | 소폭~중간 |
| 2A/2B/2C | criterion tensorization / compile 친화화 | 중간 | 중간 |
| 3 | RM decoder 공유 | 중간 | 큼 |
| 4 | seg 출력 해상도 절감 | 높음 | 매우 큼 |
| 5 | Ultralytics 결합 해소 | 높음 | 구조 안정성 / 장기 효과 큼 |
| 6 | 통합 최적화 + rollout | 중간 | 최종 수확 |

---

## 4. Phase M0 — 멀티소스 입력 경로 확정

### 4.1 목적

최종 학습은 BDD-only가 아니라 BDD + ETRI + RLMD + WOD를 모두 쓰므로, 먼저 trainer가 소비할 **단일 입력 경로**를 정의해야 한다.

### 4.2 현재 상태

- 현재 코드는 단일 `--dataset-root`만 받는다: `tools/train/train_pv26.py:42`
- 현재 dataset class도 단일 manifest root만 읽는다: `pv26/torch_dataset.py:255`, `pv26/torch_dataset.py:269`
- 즉 아래 둘 중 하나가 선행되어야 한다.
  1. merged PV26 v1 root 생성
  2. multi-root / multi-manifest dataset loader 도입

### 4.3 권장 방향

1. **단기**
   - merged root를 만들거나, 최소한 train용 merged manifest를 만든다.
2. **중기**
   - `Pv26ManifestDataset`를 multi-root aware 형태로 확장하거나, wrapper dataset을 만든다.
3. **장기**
   - source별 sampling weight / curriculum / ablation이 가능하도록 source metadata를 first-class로 다룬다.

### 4.4 수용 기준

- 최종 학습이 BDD/ETRI/RLMD/WOD를 함께 consume 가능
- 실험 로그에 source mix를 기록 가능
- Phase 0 baseline이 “최종 데이터 구성” 위에서 다시 측정 가능

---

## 5. Phase 0 — Baseline과 측정 체계 고정

### 5.1 목적

현재 `loss=150ms`, `bwd=350ms` 같은 수치는 방향성은 유효해 보여도, 기본 설정에서는 async timing이므로 stage attribution이 섞였을 수 있다. 먼저 **동일 조건의 정밀 baseline**을 만든다.

주의:

- BDD-only baseline은 참고값으로는 의미가 있지만, 최종 비교 기준은 멀티소스 baseline이어야 한다.
- 따라서 가능하면 Phase M0 이후에 **combined baseline**을 잡고, 필요하면 BDD-only는 legacy reference로만 남긴다.

### 5.2 작업 항목

- `--profile-sync-cuda`를 켠 상태로 30~50 step 재측정한다: `tools/train/train_pv26.py:161`
- `profile_every`, batch size, image size, compile on/off, AMP on/off를 문서화한다: `tools/train/train_pv26.py:155`, `tools/train/train_pv26.py:1298`
- 다음 지표를 같은 로그 포맷으로 기록한다.
  - total step ms
  - wait / h2d / fwd / loss / bwd / opt ms
  - throughput(img/s)
  - GPU free memory low-watermark
  - train loss breakdown (`od`, `da`, `rm`, `rm_lane_subclass`)

### 5.3 산출물

- 실험 비교 표 1개
- “변경 전 baseline” 로그 스냅샷 1개

### 5.4 수용 기준

- 같은 설정으로 2회 반복했을 때 total step ms 오차가 허용 범위 내(예: ±5%)인지 확인
- async / sync 측정 차이를 명시적으로 기록

---

## 6. Phase 1 — 저위험 hot-path 최적화

이 phase는 **모델 의미를 거의 바꾸지 않는 작업**만 포함한다.

### 6.1 same-size final interpolate 제거

#### 배경

- DA head는 stride-8 feature를 3번 업샘플한 뒤 항상 `F.interpolate(..., size=out_size)`를 한 번 더 호출한다: `pv26/multitask_model.py:113`, `pv26/multitask_model.py:121`
- RM head도 동일 패턴이다: `pv26/multitask_model.py:144`, `pv26/multitask_model.py:150`
- 현재 기본 입력 `960x544`에서는 이미 최종 feature가 목표 크기에 도달하므로 마지막 interpolate가 no-op성 커널일 가능성이 높다.

#### 작업

- `if logits.shape[-2:] != out_size:` 조건부 interpolate로 변경

#### 대상 파일

- `pv26/multitask_model.py`
- `tests/test_multitask_model.py`

#### 수용 기준

- 출력 shape 불변
- smoke test / unit test 통과
- total step ms 또는 fwd ms가 소폭이라도 개선

### 6.2 lane-subclass sparse positive-pixel CE

#### 배경

- 현재는 full-image CE 후 valid mask reduction이다: `pv26/criterion.py:474`, `pv26/criterion.py:479`, `pv26/criterion.py:481`
- BDD lane subclass는 line 기반으로 매우 sparse하다: `pv26/bdd.py:390`, `pv26/bdd.py:396`
- 다만 `sparse_pos`는 eager에서는 이득일 수 있어도, compile 관점에서는 `N_pos` 동적 shape 때문에 불리할 수 있다.

#### 작업

- `dense_masked`와 `sparse_pos` 두 구현을 모두 유지할 수 있게 helper 경계를 먼저 정리한다.
- eager hot-path에서는 valid positive pixel만 gather해 `[N_pos, C]` CE 수행하는 `sparse_pos`를 우선 실험한다.
- sample-wise mean semantics는 유지
- `N_pos == 0`인 sample은 현재와 동일하게 0-loss 처리

#### 대상 파일

- `pv26/criterion.py`
- `tests/test_criterion.py`

#### 수용 기준

- 기존 구현과 supervised positive pixel 기준 의미가 같음
- `rm_lane_subclass` loss 수치가 허용 오차 내에서 유지
- `loss` 구간 ms가 개선
- compile 적용 여부는 Phase 2C에서 별도로 결정하며, 이 patch에서 강제하지 않는다.

### 6.3 `_od_loss_ultralytics()` full-batch fast path

#### 배경

- 현재 keep sample이 full batch여도 `index_select()`로 `boxes`, `scores`, `feats`를 모두 복사한다: `pv26/criterion.py:200`, `pv26/criterion.py:205`, `pv26/criterion.py:206`, `pv26/criterion.py:207`
- detection targets는 collate 단계에서 이미 flat tensor로 준비된다: `tools/train/train_pv26.py:262`, `tools/train/train_pv26.py:308`, `tools/train/train_pv26.py:313`, `tools/train/train_pv26.py:314`, `tools/train/train_pv26.py:315`

#### 작업

- `idx == arange(bsz)`인 경우 raw preds / raw flat target을 그대로 쓰는 fast path 추가
- full batch supervised + no subset/none 조건을 빠르게 판별

#### 대상 파일

- `pv26/criterion.py`
- `tests/test_criterion.py`

#### 수용 기준

- detection loss 수치 불변
- `od` 구간의 불필요한 batch copy 제거

### 6.4 Phase 1 종료 조건

- Phase 0 baseline 대비 total step time 개선 여부를 수치로 기록
- 정확도 계약이 바뀌지 않았음을 unit/integration test로 확인

---

## 7. Phase 2 — criterion tensorization과 compile 친화화

이 phase는 seg loss 블록을 **Python orchestration보다 tensor compute 중심**으로 재구성하는 단계다.

### 7.1 목적

- criterion 전체 또는 seg loss 서브블록에 `torch.compile()`을 적용할 수 있게 만들기
- graph break 가능성을 줄이기
- pointwise/reduction fusion이 잘 일어나도록 계산 경계를 정리하기

### 7.2 작업 항목

#### 7.2A seg loss pure tensor helper 분리

- `_da_loss()`, `_rm_loss()`, `_rm_lane_subclass_loss()`를 순수 tensor helper로 분해
- batch normalization / default fill / python branching을 loss 진입 전에 끝내기: `pv26/criterion.py:78`, `pv26/criterion.py:84`, `pv26/criterion.py:90`
- 이 단계에서는 numerics와 eager 동작을 유지하고, compile은 아직 붙이지 않는다.

#### 7.2B DA/RM seg loss compile 적용

- model만 compile하는 현재 구조를 확장해 criterion 또는 seg-loss callable도 compile 가능하게 설계: `tools/train/train_pv26.py:747`, `tools/train/train_pv26.py:765`, `tools/train/train_pv26.py:1295`
- 초기 타겟은 whole criterion이 아니라 DA/RM seg loss 블록이다.
- lane-subclass는 여기서 compile 대상에 강제로 포함하지 않는다.

#### 7.2C lane-subclass dense vs sparse compile 판단

- `rm_lane_subclass_loss_impl = dense_masked | sparse_pos` 같은 선택 지점을 남긴다.
- eager와 compile에서 동일 구현을 강제하지 않고, 실제 벤치 결과로 결정한다.
- 목표는 “lane-subclass가 빠른 구현”이 아니라 “전체 train step에 유리한 구현”을 선택하는 것이다.

#### 7.2.4 data/target 준비 경계 명확화

- `_move_prepared_batch_to_device()` 이후에는 criterion이 이미 정규화된 tensor batch만 받도록 규약 강화: `tools/train/train_pv26.py:371`, `tools/train/train_pv26.py:833`
- `PV26Criterion.forward()`의 “prepared batch vs raw samples” 이중 경로를 장기적으로 분리: `pv26/criterion.py:78`, `pv26/criterion.py:81`

### 7.3 권장 구현 방향

- `PV26PreparedBatch` dataclass 또는 typed mapping 도입
- seg loss compile 플래그를 별도 CLI 옵션으로 도입
- lane-subclass는 구현 선택 플래그를 남겨 eager/compile 실험을 분리
- compile 실패 시 eager fallback은 유지

### 7.4 대상 파일

- `pv26/criterion.py`
- `tools/train/train_pv26.py`
- 필요 시 `pv26/__init__.py`
- `tests/test_criterion.py`

### 7.5 수용 기준

- eager / compile 양쪽에서 수치 일관성 유지
- graph break / compile failure가 발생해도 fallback 안전
- sync profile 기준 loss ms 또는 total ms가 개선
- lane-subclass는 `dense_masked`와 `sparse_pos` 중 더 빠른 쪽을 실측으로 선택

---

## 8. Phase 3 — RM decoder 공유

이 phase는 **계산량 자체를 줄이는 첫 구조 변경**이다.

### 8.1 배경

- 현재 `rm_head`, `rm_lane_subclass_head`가 각각 별도 deconv decoder를 가진다: `pv26/multitask_model.py:269`, `pv26/multitask_model.py:270`
- 실제로 비싼 부분은 full-resolution까지 올리는 decoder path이며, 마지막 prediction head는 상대적으로 싸다.

### 8.2 목표 구조

- shared RM decoder 1개
- 그 위에 `1x1 conv` branch 2개
  - binary RM logits head (`3ch`)
  - lane-subclass logits head (`K+1 ch`)

### 8.3 작업 항목

- `RoadMarkingHeadDeconv`를 “decoder + pred” 결합형에서 “decoder trunk + prediction heads” 분리형으로 재설계
- `PV26MultiHead` / `PV26MultiHeadYOLO26` 모두 shared decoder 사용으로 이행
- output contract는 그대로 유지 (`rm`, `rm_lane_subclass`)

### 8.4 대상 파일

- `pv26/multitask_model.py`
- `tests/test_multitask_model.py`
- 필요 시 `tests/test_criterion.py`

### 8.5 수용 기준

- 출력 shape/contract 불변
- RM / lane-subclass 학습이 모두 정상
- sync profile 기준 bwd ms와 total step ms가 의미 있게 감소

### 8.6 주요 리스크

- 두 task가 decoder representation을 공유하면서 최적점이 달라질 수 있음
- mitigation:
  - branch-specific stem 1개를 얕게 두는 옵션 확보
  - 초기엔 fully shared, 필요 시 “shared trunk + tiny branch adapter”로 후퇴 가능하게 설계

---

## 9. Phase 4 — Segmentation 출력 해상도 축소

이 phase가 가장 큰 성능 레버지만, 정확도와 loss contract에 영향을 줄 수 있어 가장 조심해서 진행한다.

### 9.1 목적

- full-resolution seg logits 계산을 줄여 activation traffic과 backward cost를 근본적으로 낮춘다.

### 9.2 기본 전략

- 1차 목표는 `1/2 resolution`
- DA와 RM binary는 동일 scale 적용 가능
- lane-subclass는 1/2부터 시작하고, 필요하면 추가 조정

### 9.3 설계 원칙

- **학습 해상도**와 **평가/시각화 해상도**를 분리한다
- 모델 내부 logits는 저해상도일 수 있지만, eval path는 필요 시 full-resolution로 upsample 가능해야 한다

### 9.4 작업 항목

#### 8.4.1 해상도 제어 옵션 도입

- 예시:
  - `seg_output_stride = 1` (현재)
  - `seg_output_stride = 2`
  - 필요 시 `seg_output_stride = 4`

#### 8.4.2 target downsample 정책 정의

- DA:
  - binary mask downsample 시 ignore 보존 정책 필요
- RM binary:
  - thin structure 손실을 줄이기 위해 max/OR 성격 downsample 또는 nearest 기반 정책 비교
- lane-subclass:
  - background/ignore/positive 충돌 시 우선순위 정책 명시
- 이 정책은 criterion 내부가 아니라 **batch-prep 경계**에서 처리한다.
- 즉, criterion은 이미 `seg_output_stride`에 맞게 shape가 정리된 tensor target만 받는다.

#### 8.4.3 criterion shape contract 재설계

- 현재 `_da_loss()`, `_rm_loss()`, `_rm_lane_subclass_loss()`는 logits/mask 동일 spatial shape를 사실상 전제로 한다: `pv26/criterion.py:407`, `pv26/criterion.py:422`, `pv26/criterion.py:451`
- 따라서 학습 시 target resize 책임은 batch-prep/helper 쪽에 고정하고, criterion은 그 contract만 소비한다.

#### 8.4.4 validation/eval 경로 동시 수정

- `validate()`는 reduced-resolution logits를 그대로 full-resolution target과 비교하지 않도록 같은 phase 안에서 함께 수정한다.
- eval에서는 upsample 후 threshold/argmax를 적용해 현재 지표 계약을 유지한다.
- Phase 4는 model/train/eval contract를 한 commit 단위로 같이 움직이는 것을 원칙으로 한다.

### 9.5 권장 순서

1. shared RM decoder 먼저 반영
2. DA/RM/lane-subclass 모두 `1/2 resolution` 실험
3. 정확도 손실이 크면 DA만 먼저 축소하고 RM은 유지하는 split option 검토

### 9.6 대상 파일

- `pv26/multitask_model.py`
- `pv26/criterion.py`
- `pv26/torch_dataset.py` 또는 별도 batch target resize helper
- `tools/train/train_pv26.py`
- `tests/test_criterion.py`
- `tests/test_multitask_model.py`

### 9.7 수용 기준

- total step time과 bwd ms가 Phase 3 대비 유의미하게 감소
- val 지표 악화가 허용 범위 내
- target resize 정책이 문서화되어 재현 가능

---

## 10. Phase 5 — Hook / `self._feat` / Ultralytics 구조 의존성 제거

이 phase는 성능뿐 아니라 **장기 유지보수성**을 위한 핵심 구조 개선이다.

### 10.1 제거 대상

- `forward hook` 기반 feature capture: `pv26/multitask_model.py:234`, `pv26/multitask_model.py:246`
- `self._feat` mutable side-effect state: `pv26/multitask_model.py:232`, `pv26/multitask_model.py:252`, `pv26/multitask_model.py:277`, `pv26/multitask_model.py:290`
- Ultralytics 내부 module index / output dict 구조에 대한 직접 가정: `pv26/multitask_model.py:241`, `pv26/multitask_model.py:243`, `pv26/multitask_model.py:301`
- criterion이 raw `ultra_det_model`과 `E2ELoss`에 직접 결합된 구조: `pv26/criterion.py:63`, `pv26/criterion.py:76`, `tools/train/train_pv26.py:1254`

### 10.1.1 유지해야 할 외부 계약

- migration 중간 단계에서도 `PV26MultiHeadOutput.det`의 외부 contract는 최대한 유지한다.
- pretrained loading과 criterion 초기화가 즉시 깨지지 않도록 `model.det_model` 접근점은 단계적으로만 이동한다.
- Phase 5는 “성능/구조 개선”이지 주변 배선 전체를 한 번에 갈아엎는 작업이 아니다.

### 10.2 목표 구조

#### 9.2.1 내부 추상화 계층 도입

- 예시 dataclass:
  - `PV26DetFeatures`
    - `det_train_out`
    - `det_eval_out`
    - `p3_backbone`
    - `p3_head`
- 예시 protocol/interface:
  - `PV26DetBackend`
    - `infer_feature_shapes()`
    - `forward_features(x) -> PV26DetFeatures`
    - `build_loss_adapter()`

#### 9.2.2 loss adapter 분리

- `PV26Criterion`은 raw Ultralytics 객체 대신 **det loss adapter**에 의존
- adapter가 Ultralytics `E2ELoss`와 prediction schema 차이를 내부에서 흡수

### 10.3 권장 migration 단계

#### 단계 A — 결합점 캡슐화

- `pv26/multitask_model.py` 내부에 박혀 있는 Ultralytics extraction 코드를 별도 backend adapter 파일로 이동
- `PV26MultiHeadYOLO26`는 adapter가 반환한 explicit features만 사용
- 이 단계에서는 동작을 바꾸지 않고 adapter shell만 도입한다.
- `model.det_model`과 기존 pretrained loading 경로는 유지한다.

#### 단계 B — side-effect 제거

- hook 제거
- `self._feat` 제거
- adapter가 한 번의 forward에서 명시적으로 `p3_backbone`, `p3_head`, `det_out`를 함께 반환
- 이 단계에서도 외부 output contract는 바꾸지 않는다.

#### 단계 C — criterion decoupling

- `PV26Criterion(..., ultra_det_model=...)` 패턴 제거: `pv26/criterion.py:40`, `pv26/criterion.py:64`
- 대신 `det_loss_backend` 또는 `det_loss_adapter` 주입
- train script에서 `model.det_model` 직접 전달 제거: `tools/train/train_pv26.py:1250`, `tools/train/train_pv26.py:1254`
- 여기서만 `model.det_model` 의존을 명시적으로 정리하고 migration path를 문서화한다.

#### 단계 D — backend-neutral core

- 장기적으로 `PV26MultiHeadYOLO26`라는 이름 자체보다, `PV26MultiHeadDetBacked` 혹은 유사한 backend-neutral model 계층 고려
- stub / Ultralytics / 향후 다른 detection trunk가 동일 contract를 구현하도록 정리

### 10.4 구현 후보 파일

- 신규 예시:
  - `pv26/det_backends.py`
  - `pv26/det_loss_backends.py`
- 수정:
  - `pv26/multitask_model.py`
  - `pv26/criterion.py`
  - `tools/train/train_pv26.py`
  - `tests/test_multitask_model.py`
  - `tests/test_criterion.py`

### 10.5 수용 기준

- `pv26/multitask_model.py`에 `register_forward_hook()`가 더 이상 없음
- `self._feat` mutable dict가 제거됨
- criterion이 raw Ultralytics model 없이 동작 가능
- Ultralytics output schema 변화가 생겨도 adapter 한 곳만 수정하면 되도록 경계가 정리됨
- migration 중간 단계에서도 pretrained loading과 학습 entrypoint가 깨지지 않음

### 10.6 기대 효과

- compile / CUDA graph 친화성 개선
- 디버깅 난이도 감소
- trunk 교체 비용 감소
- upstream Ultralytics 변경 충격 완화

---

## 11. Phase 6 — 통합 compile / graph 재평가

이 phase는 앞 단계들이 끝난 뒤에만 의미가 있다.

### 11.1 전제 조건

- hook / side effect 제거 완료
- seg loss가 pure tensor callable로 분리 완료
- output shape가 충분히 정적

### 11.2 작업 항목

- model-only compile vs model+criterion compile 비교
- compile mode A/B:
  - `default`
  - `max-autotune`
  - 필요 시 `reduce-overhead`
- graph capture 후보 구간 검토
  - seg loss block
  - 가능하면 train step 일부

### 11.3 대상 파일

- `tools/train/train_pv26.py`
- `pv26/multitask_model.py`
- `pv26/criterion.py`

### 11.4 수용 기준

- compile이 안정적으로 동작
- graph break 빈도 감소
- total step ms / throughput / memory가 최종 목표 범위에 근접

---

## 12. 테스트/검증 계획

### 12.1 기존 테스트 활용

- `tests/test_criterion.py`
- `tests/test_multitask_model.py`

### 12.2 추가해야 할 테스트

- conditional interpolate가 shape/값 contract를 깨지 않는지 확인
- sparse lane-subclass CE가 positive-pixel 의미를 유지하는지 확인
- lane-subclass `dense_masked`와 `sparse_pos`가 toy case에서 기대 semantics를 유지하는지 확인
- shared RM decoder 도입 후 output shape 및 backward 가능 여부 확인
- reduced-resolution seg target resize 정책과 eval upsample 경로를 함께 테스트
- det backend adapter가 explicit feature contract를 안정적으로 반환하고, fake backend로도 단위 테스트 가능한지 확인
- det loss adapter가 full-batch / subset / none 케이스를 올바르게 처리하는지 확인

### 12.3 실험 매트릭스

각 phase 뒤에 최소한 아래 조합을 기록한다.

- compile off / on
- sync profile on
- 고정 batch size
- 동일 train subset
- 동일 seed

### 12.4 최종 비교표에 포함할 지표

- total step ms
- fwd/loss/bwd/opt ms
- throughput(img/s)
- max allocated / free memory low-watermark
- det mAP
- DA IoU
- RM channel IoU
- lane-subclass metric(현재 운영 중인 metric 기준)

---

## 13. 리스크와 대응

### 13.1 정확도 하락

- 가장 큰 리스크는 seg resolution 축소
- 대응:
  - `1/2 resolution`부터 시작
  - DA와 RM을 분리 적용 가능한 옵션 설계
  - val metric gating 없이는 merge 금지

### 13.2 compile 불안정 / graph break

- 대응:
  - seg loss 블록부터 국소 적용
  - eager fallback 유지
  - compile 이전에 side-effect 제거부터 진행

### 13.3 Ultralytics 업스트림 변경

- 대응:
  - adapter layer에 결합점 집중
  - `module index`, `one2many.feats` 가정은 adapter 내부로만 제한

### 13.4 구조 변경으로 인한 디버깅 난이도 증가

- 대응:
  - phase별로 독립 commit/PR 단위 유지
  - phase마다 baseline 비교표 업데이트

### 13.5 멀티소스 데이터 혼합으로 인한 해석 왜곡

- ETRI/RLMD/WOD는 supervision coverage가 BDD와 다르다.
- 예:
  - ETRI: OD 없음
  - RLMD: OD/DA 없음
  - WOD: detection subset + lane-subclass/stop-line 제한
- 대응:
  - 전체 평균뿐 아니라 source별 지표 분리
  - source mix를 baseline 문서에 항상 기록
  - sampling 정책 변경 시 baseline 재측정

---

## 14. 실제 실행 권장 순서

아래 순서로 진행한다.

1. **Phase M0**
   - 멀티소스 입력 경로 확정
   - source mix 확인
2. **Phase 0**
   - sync profiling baseline 확정
3. **Phase 1**
   - same-size interpolate 제거
   - lane-subclass sparse CE
   - OD full-batch fast path
4. **Phase 2A/2B/2C**
   - criterion tensorization
   - DA/RM seg loss compile 시도
   - lane-subclass는 dense vs sparse를 실측으로 결정
5. **Phase 3**
   - RM shared decoder
6. **Phase 4**
   - `1/2 resolution` seg 실험
   - batch-prep target resize + validation/eval patch 동시 반영
7. **Phase 5**
   - 5A: det backend adapter shell 도입
   - 5B: hook / `self._feat` 제거
   - 5C: det loss adapter 도입 + `model.det_model` 의존 정리
8. **Phase 6**
   - 통합 compile / graph 재평가
9. **최종**
   - full benchmark + quality gate + rollout

---

## 15. 완료 조건

다음이 만족되면 이번 대규모 작업을 완료로 본다.

- 멀티소스(BDD/ETRI/RLMD/WOD) 기준 baseline과 최종값이 모두 남아 있음
- sync profile 기준 total step time이 baseline 대비 유의미하게 감소
- RM/lane-subclass가 shared decoder 구조에서 안정적으로 학습
- seg reduced-resolution 실험 결과가 수용 가능한 정확도/속도 trade-off를 보임
- `pv26/multitask_model.py`에서 hook / `self._feat` side effect 제거 완료
- `PV26Criterion`이 raw Ultralytics model 없이 동작 가능한 구조로 정리
- compile 적용 범위가 확대되어도 안정적으로 fallback 가능
- 테스트 / benchmark / 문서가 모두 현재 구조를 반영

---

## 16. 요약

이번 작업의 본질은 “loss 함수 몇 줄 최적화”가 아니라, 아래 세 가지를 순차적으로 해내는 것이다.

1. **full-resolution segmentation compute를 줄이기**
2. **중복 decoder를 제거하기**
3. **Ultralytics 내부 구현에 종속된 경계를 정리하기**

따라서 실행은 반드시 **측정 → 저위험 정리 → 구조 절감 → 의존성 분리 → 통합 최적화** 순서로 간다.  
이 순서를 지키되, 특히 다음 세 가지를 강하게 고정한다.

1. lane-subclass eager 최적화와 compile 최적화는 같은 patch/같은 판단으로 묶지 않는다.
2. reduced-resolution 도입 시 target resize는 batch-prep contract로 처리하고, validation/eval 수정까지 같은 phase에서 끝낸다.
3. Ultralytics decoupling은 외부 contract를 유지한 채 5A/5B/5C 순서로 migration한다.
