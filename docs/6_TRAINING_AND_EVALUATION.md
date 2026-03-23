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
  - BDD100K `35%`
  - AIHUB traffic `35%`
  - AIHUB lane `30%`

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
- evaluator skeleton은 batch-level loss summary와 GT row count summary를 지원한다.
- tiny overfit smoke는 canonical train batch 2개 기준으로 실제 loss 감소를 확인했다.
- detector assignment는 task-aligned assigner 기준으로 통합 완료다.
- lane family Hungarian matching도 통합 완료다.
