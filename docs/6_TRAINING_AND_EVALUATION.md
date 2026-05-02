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
  - tiny overfit regression
- step 6
  - full training wiring

## optimizer / schedule 원칙

- new head는 trunk보다 높은 learning rate를 쓴다.
- freeze stage에서는 head 위주로 먼저 수렴시킨다.
- unfreeze는 단계적으로 한다.
- backbone variant는 `n/s`를 모두 지원하되, 현재 추천 경로는 `yolo26s`다.

## recommended stage schedule

1. stage 1
   - trunk freeze, new heads warm-up
2. stage 2
   - neck + upper backbone unfreeze
3. stage 3
   - full fine-tune
4. stage 4
   - lane-family late fine-tune
   - lane / stop-line / crosswalk metric 회복 및 보정

## phase-specific selection

- 모든 phase의 기본 selection path는 `selection_metrics.phase_objective`다.
- phase objective는 validation metric 기반 composite score이며, lane / stop-line / crosswalk sparse support는 reliability weight로 완화한다.
- best checkpoint와 early exit는 같은 objective를 보되, stop 판정은 `patience + min_delta_abs`를 기본으로 사용한다.
- training preset은 필요하면 phase별 selection override를 실험용으로 허용하지만, shipped preset은 전 phase에서 composite objective를 쓴다.
- 식, 계수, shipped 기본값, `mode=max`, `min_delta_abs` tuning 기준은 [5_TARGETS_AND_LOSS.md](5_TARGETS_AND_LOSS.md)에 구현 기준으로 정리한다.

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

## test-first criteria

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
- tiny regression에서 명백한 shape bug가 없음

## current runtime status

- trainer skeleton은 `encoded batch -> trunk -> heads -> loss -> backward -> optimizer.step`까지 지원한다.
- trainer는 dataset-balanced batch sampler helper를 지원한다.
- trainer는 step history 요약과 JSONL logging을 지원한다.
- trainer는 `run_manifest.json`, live step/epoch JSONL, TensorBoard scalar logging, rolling timing profile(`wait/load/fwd/loss/bwd`, mean/p50/p99, ETA)를 지원한다.
- trainer는 train/validation epoch 양쪽 모두 live progress 표시를 지원하고, phase/epoch/iter/epoch start/elapsed/ETA/timing 정보를 runtime에서 바로 노출한다.
- trainer는 checkpoint save/load를 지원한다.
- trainer는 full epoch fit loop, val loop, best/last checkpoint, run summary 출력을 지원한다.
- trainer는 AMP, grad accumulation, grad clip, auto resume, non-finite/OOM guard를 지원한다.
- trainer preset은 stage 1~4 phase chain을 기준으로 확장된다.
- `tools/run_pv26_train.py`는 현재 `default` preset 하나만 지원한다. legacy/dev preset과 legacy dataset mapping key는 더 이상 지원하지 않는다.
- exact in-place resume는 `python3 tools/run_pv26_train.py --resume-run runs/pv26_exhaustive_od_lane_train/<meta_run_name>` 경로를 기준으로 유지한다.
- derived retrain/fine-tune는 `python3 tools/run_pv26_train.py --derive-run runs/pv26_exhaustive_od_lane_train/<source_run_name> --start-stage <STAGE> --end-stage <STAGE>` 경로를 기준으로 유지한다.
- derived run은 source run의 checkpoint를 seed로 쓰되, epoch/sampler/loss/freeze 숫자 파라미터는 현재 preset + user YAML을 그대로 다시 읽는다.
- `tools/check_env.py` interactive launcher는 `stage_3` peak VRAM stress probe를 제공하고, batch size / short iter 수를 받아 현재 backbone/stage 경로로 메모리 상한을 빠르게 확인할 수 있다.
- direct CLI probe는 `python3 tools/run_pv26_train.py --preset default --stage3-vram-stress --stress-stage <STAGE> --stress-batch-size <BATCH> --stress-iters <ITERS>` 형식으로 유지한다.
- phase별 batch 후보를 한 번에 확인할 때는 `python3 tools/run_pv26_train.py --preset default --phase-vram-sweep --stress-batch-sizes 1,2,4,6,8,12 --stress-iters 8`를 사용한다. 출력의 `ceiling_observed=false`는 OOM/non-finite failure를 아직 못 만났다는 뜻이며, `max_ok_batch_size`는 확정 상한이 아니라 확인된 하한이다.
- `tools/run_pv26_train.py`는 phase별 summary JSON과 `runs/pv26_exhaustive_od_lane_train/` 계열 산출물을 쓴다.
- phase summary와 run manifest는 backbone variant, resolved head channels, phase selection metric 같은 late-stage 판단 정보를 함께 남기는 방향을 따른다.
- `tiny overfit regression`은 `model.engine.trainer.run_pv26_tiny_overfit()` helper와 unit test로 검증한다.
- evaluator skeleton은 batch-level loss summary와 GT row count summary를 지원한다.
- evaluator는 raw model output을 postprocess prediction bundle로 decode하는 `predict_batch` runtime을 지원한다.
- evaluator는 validation에서 loss/metrics/prediction bundle을 single forward path로 묶어 사용한다.
- evaluator는 batch-level detector AP50/precision/recall, TL bit F1/combo accuracy, lane family matching metrics를 지원한다.
- postprocess는 `torchvision.ops.batched_nms` 사용 가능 시 우선 사용하고, 불가능하면 pure PyTorch NMS fallback을 사용한다.
- tiny overfit regression은 canonical train batch 2개 기준으로 실제 loss 감소를 확인했다.
- epoch fit regression은 canonical source 기준으로 checkpoint resume 가능한 run summary를 확인했다.
- detector assignment는 task-aligned assigner 기준으로 통합 완료다.
- lane family Hungarian matching도 통합 완료다.
