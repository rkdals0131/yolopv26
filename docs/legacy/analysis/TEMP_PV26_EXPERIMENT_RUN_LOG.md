# TEMP_PV26 실험 결과 정리

기준 날짜: `2026-03-07`  
기준 데이터셋: `datasets/pv26_v1_merged_all`  
기준 샘플 수: `train=74,372`, `val=12,033`  
기준 실행 조건:

- `batch_size=10`
- `max_train_batches=200`
- `max_val_batches=1`
- `profile_every=20`
- `profile_sync_cuda=true`
- `train_drop_last=true`
- AMP on
- 비교는 warm-up 이후 `40~200 step` window 기준

## 1. 최종 채택 설정

최종 권장 설정은 아래 조합이다.

```bash
python tools/train/train_pv26.py \
  --dataset-root datasets/pv26_v1_merged_all \
  --arch yolo26n \
  --epochs 1 \
  --max-train-batches 200 \
  --max-val-batches 1 \
  --profile-every 20 \
  --profile-sync-cuda \
  --train-drop-last \
  --no-progress \
  --no-tensorboard
```

현재 코드 기본값:

- `compile=False`
- `compile_seg_loss=True`
- `seg_output_stride=2`

최종 권장 이유:

- model `torch.compile`은 forward는 빨라져도 backward가 악화되어 전체 step이 느려졌다.
- `DA/RM seg-loss`만 compile하는 경로는 안정적으로 전체 step time을 줄였다.
- `seg_output_stride=2`는 가장 큰 실측 이득을 만들었다.

## 2. 최종 결과

최종 baseline:

- run: `phase6_baseline_eager`
- config: `--no-compile --compile-seg-loss --seg-output-stride 2`
- steady-state 평균:
  - `total=263.6ms`
  - `fwd=80.5ms`
  - `loss=31.0ms`
  - `bwd=140.2ms`
  - `throughput=37.9 img/s`

Phase 0 eager baseline 대비:

- `353.1ms -> 263.6ms`
- step time `-25.3%`
- throughput `28.3 -> 37.9 img/s`
- throughput `+33.9%`

## 3. Phase별 요약

| ID | Phase | Change | total ms | fwd ms | loss ms | bwd ms | thr img/s | Verdict |
|---|---|---|---:|---:|---:|---:|---:|---|
| `P0-002` | 0 | merged eager baseline | 353.1 | 112.6 | 47.7 | 179.1 | 28.3 | baseline |
| `P0-003` | 0 | merged model compile baseline | 382.4 | 69.4 | 42.5 | 257.7 | 26.1 | reject |
| `P1-004` | 1 | conditional interpolate + OD fast path | 349.1 | 113.3 | 43.7 | 177.5 | 28.6 | keep, low gain |
| `P1-002` | 1 | lane-subclass `sparse_pos` | 353.3 | 117.6 | 41.6 | 179.4 | 28.3 | reject as default |
| `P2-001` | 2A | prepared-batch tensorization | 350.3 | 113.7 | 43.9 | 178.2 | 28.5 | keep, neutral |
| `P2-002` | 2B | seg-loss compile only | 303.8 | 107.7 | 28.5 | 155.2 | 32.9 | keep |
| `P3-001` | 3 | shared RM decoder | 304.8 | 101.5 | 30.5 | 159.1 | 32.8 | keep, structural |
| `P4-001` | 4 | `seg_output_stride=2` | 265.4 | 80.7 | 31.1 | 140.7 | 37.6 | keep |
| `P5-001` | 5A | det backend adapter shell | - | - | - | - | - | keep, structural |
| `P5-002` | 5B | hook / `self._feat` 제거 | - | - | - | - | - | keep, structural |
| `P5-003` | 5C | det loss adapter decoupling | - | - | - | - | - | keep, structural |
| `P6-001` | 6 | model compile + seg-loss compile | 282.8 | 60.5 | 31.5 | 179.5 | 35.4 | reject |
| `P6-003` | 6 | final eager baseline | 263.6 | 80.5 | 31.0 | 140.2 | 37.9 | final |

## 4. 주요 판단

### 4.1 채택

- `conditional final interpolate`
- `_od_loss_ultralytics` full-batch fast path
- `PV26PreparedBatch` 기반 hot path
- `DA/RM seg-loss compile`
- `shared RM decoder`
- `seg_output_stride=2` + batch-prep target resize + eval upsample
- `det backend adapter`
- hook / `self._feat` 제거
- `det loss adapter` 분리

### 4.2 기각

- lane-subclass `sparse_pos`를 기본 구현으로 채택하는 것
  - eager 전체 step 이득이 없었음
- model 전체 `torch.compile`
  - forward는 빨라졌지만 backward가 악화됨
  - 최종 config에서도 eager보다 `+7.3%` 느렸음

## 5. Phase 4 / Phase 6 추가 메모

### 5.1 `seg_output_stride=2`

실측상 가장 큰 레버였다.

- Phase 3 대비:
  - `304.8ms -> 265.4ms`
  - `-12.9%`
- DA/RM/lane-subclass target resize는 criterion 내부가 아니라 batch-prep 경계에서 처리
- validation은 full-res target과 맞추기 위해 logits를 full-res로 올린 뒤 metric 계산

### 5.2 model compile 재평가

최종 구조 정리 이후에도 model compile은 채택하지 않았다.

- eager final: `263.6ms`, `37.9 img/s`
- model compile: `282.8ms`, `35.4 img/s`
- forward:
  - `80.5ms -> 60.5ms`
- backward:
  - `140.2ms -> 179.5ms`

결론:

- compile 후보는 model 전체가 아니라 국소 블록이어야 한다.
- 현재는 `seg-loss compile only`가 최적이다.

## 6. 구조 변경 커밋 기록

주요 커밋:

- `7222d09` `pv26: skip redundant final interpolate in seg heads`
- `e320588` `docs: refine PV26 rollout plan for execution`
- `019dace` `criterion: harden merged OD path and keep lane loss variants`
- `d671b27` `criterion: introduce typed prepared batch hot path`
- `72e22ce` `criterion: restore pin-memory support for prepared batches`
- `b322371` `criterion: add optional compiled seg loss block`
- `9e5ab34` `model: share road-marking decoder trunk`
- `d3bdd19` `train: add half-res segmentation path`
- `2887148` `model: add detection backend adapter shell`
- `66ae3bd` `model: remove hook-based det feature capture`
- `5948e0d` `criterion: decouple ultralytics detection loss`

## 7. 코드 기본값 반영 내용

`tools/train/train_pv26.py` 기본값은 아래처럼 바뀌었다.

- `--compile`: off
- `--compile-seg-loss`: on
- `--seg-output-stride`: `2`

추가로 backend adapter 도입 후 깨졌던 trunk/head optimizer 그룹 분리도 다시 맞췄다.

## 8. 후속 후보

남은 후보는 아래 정도다.

- `compile_mode=max-autotune` 추가 비교
- eval 경로의 compile / upsample warm-up 비용 별도 최적화
- source별 metric 분리 기록
- `seg_output_stride`를 task별로 분리하는 ablation
