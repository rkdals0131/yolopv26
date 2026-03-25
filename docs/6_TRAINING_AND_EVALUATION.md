# Training And Evaluation

## training strategy

- step 1
  - [4A_SAMPLE_AND_TRANSFORM_CONTRACT.md](4A_SAMPLE_AND_TRANSFORM_CONTRACT.md) 기준 standardized dataset loader 구현
- step 2
  - sample contract를 encoded batch contract로 바꾸는 target encoder 구현
- step 3
  - pretrained trunk + custom heads 구성
- step 4
  - multitask loss 연결
- step 5
  - mini overfit smoke
- step 6
  - full training wiring

## optimizer / schedule 원칙

- new head는 trunk보다 높은 learning rate를 쓴다.
- freeze stage에서는 head 위주로 먼저 수렴시킨다.
- unfreeze는 단계적으로 한다.

## recommended stage schedule

1. stage 0
   - shape / target / loss NaN check
2. stage 1
   - trunk freeze, new heads warm-up
3. stage 2
   - neck + upper backbone unfreeze
4. stage 3
   - full fine-tune

## sampler

- dataset-balanced sampler 사용
- initial ratio
  - BDD100K `30%`
  - AIHUB traffic `30%`
  - AIHUB obstacle `15%`
  - AIHUB lane `25%`
- validation은 balanced sampler를 재사용하지 않고 sequential eval loader를 사용한다.

## eval metrics

- detector
  - mAP
  - class AP
- traffic light
  - bit-level AP/F1
  - decoded combination accuracy
- lane
  - point distance
  - color accuracy
  - type accuracy
- stop-line
  - point distance
  - angle error
- crosswalk
  - polygon IoU
  - vertex distance

## smoke criteria

- forward 성공
- backward 성공
- loss finite
- tiny subset overfit 가능
- debug sample visualization 확인 가능
- loader output이 sample contract와 정확히 일치
- encoded batch가 loss spec shape와 정확히 일치

## full-train 진입 조건

- loader와 encoder가 stable
- pretrained partial load 성공
- lane/TL/OD multitask loss가 동시에 finite
- mini smoke에서 명백한 shape bug가 없음

## current runtime status

- trainer skeleton은 `encoded batch -> trunk -> heads -> loss -> backward -> optimizer.step`까지 지원한다.
- trainer는 dataset-balanced batch sampler helper를 지원한다.
- trainer는 step history 요약과 JSONL logging을 지원한다.
- trainer는 `run_manifest.json`, live step/epoch JSONL, TensorBoard scalar logging, rolling timing profile(`wait/load/fwd/loss/bwd`, mean/p50/p99, ETA)를 지원한다.
- trainer는 checkpoint save/load를 지원한다.
- trainer는 full epoch fit loop, val loop, best/last checkpoint, run summary 출력을 지원한다.
- trainer는 AMP, grad accumulation, grad clip, auto resume, non-finite/OOM guard를 지원한다.
- `tools/run_pv26_pilot_train.py`, `tools/run_pv26_tiny_overfit_smoke.py`는 CLI 옵션 대신 파일 상단 config block을 직접 수정하는 방식으로 실행한다.
- evaluator skeleton은 batch-level loss summary와 GT row count summary를 지원한다.
- evaluator는 raw model output을 postprocess prediction bundle로 decode하는 `predict_batch` runtime을 지원한다.
- evaluator는 validation에서 loss/metrics/prediction bundle을 single forward path로 묶어 사용한다.
- evaluator는 batch-level detector AP50/precision/recall, TL bit F1/combo accuracy, lane family matching metrics를 지원한다.
- postprocess는 `torchvision.ops.batched_nms` 사용 가능 시 우선 사용하고, 불가능하면 pure PyTorch NMS fallback을 사용한다.
- tiny overfit smoke는 canonical train batch 2개 기준으로 실제 loss 감소를 확인했다.
- epoch fit smoke는 canonical source 기준으로 checkpoint resume 가능한 run summary를 확인했다.
- detector assignment는 task-aligned assigner 기준으로 통합 완료다.
- lane family Hungarian matching도 통합 완료다.
