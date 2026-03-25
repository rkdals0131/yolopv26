# Targets And Loss

## 목표

- detector, TL attr, lane, stop-line, crosswalk를 하나의 학습 파이프라인에서 다룬다.
- partial label source를 손상시키지 않는다.
- sample/transform contract는 [4A_SAMPLE_AND_TRANSFORM_CONTRACT.md](4A_SAMPLE_AND_TRANSFORM_CONTRACT.md)를 기준으로 한다.

## task summary

- OD
  - 7-class detection
- TL attr
  - 4-bit sigmoid attribute
- lane
  - 16-point polyline
- stop-line
  - 4-point polyline
- crosswalk
  - 8-point polygon

## internal target encoding

- encoded batch는 transformed network pixel space를 입력으로 받는다.
- detector prediction slot 수는 `Q_det`, GT detection row 수는 `N_gt_det`로 분리한다.
- lane
  - objectness
  - color logits 3
  - type logits 2
  - points 16
  - visibility logits 16
- stop-line
  - objectness
  - points 4
- crosswalk
  - objectness
  - polygon points 8

## matching policy

- detector
  - task-aligned assigner runtime 통합 완료
  - real trunk path에서는 feature-shape metadata를 이용해 anchor grid를 만들고 assignment를 계산한다.
  - synthetic unit smoke는 metadata가 없을 때만 `prefix positive fallback`을 사용한다.
- lane family
  - Hungarian matching runtime 통합 완료
  - lane, stop-line, crosswalk 모두 query-to-GT assignment를 cost matrix 기반으로 계산한다.

## loss summary

```text
L_total = λ_det * L_det
        + λ_tl * L_tl
        + λ_lane * L_lane
        + λ_stop * L_stop
        + λ_cross * L_cross
```

## detector loss

- type
  - YOLO detect loss
- terms
  - box
  - obj
  - cls
- partial-det policy
  - sample별 `det_supervised_class_mask`를 따른다.
  - `det_allow_background_negatives=False`인 sample은 unmatched query를 background negative로 쓰지 않는다.
  - cls BCE는 source가 담당하는 class channel에 대해서만 계산한다.
  - 즉 AIHUB traffic은 `traffic_light/sign`만, BDD100K는 `vehicle/bike/pedestrian`만 detector class supervision에 관여한다.

## TL attr loss

- type
  - masked sigmoid focal BCE
- 적용 대상
  - detector assignment 기준 matched `traffic_light` positive
  - valid AIHUB car signal only
- 결합 규칙
  - detector positive가 가리키는 GT index에서 TL bits를 읽는다.
  - GT class가 `traffic_light`가 아니면 TL attr loss는 없다.
  - GT class가 `traffic_light`여도 `valid_mask["tl_attr"] = False`면 loss는 없다.
  - `tl_attr_gt_bits`는 GT-aligned tensor고, prediction-aligned target은 assignment 이후 loss 내부에서 만들어진다.
- bit weights
  - red `1.0`
  - yellow `2.5`
  - green `1.0`
  - arrow `1.8`

## lane loss

- objectness `1.0`
- color CE `1.0`
- type CE `0.5`
- points L1 `5.0`
- visibility BCE `1.0`
- smoothness `0.25`

## stop-line loss

- objectness `1.0`
- points L1 `6.0`
- straightness `0.5`

## crosswalk loss

- objectness `1.0`
- polygon L1 `4.0`
- shape regularizer `0.5`

## dataset masking

- BDD100K
  - det on
  - detector unmatched negative off
  - detector class supervision은 `vehicle / bike / pedestrian` only
  - tl attr off
  - lane off
  - stop-line off
  - crosswalk off
- AIHUB traffic
  - det on
  - detector unmatched negative off
  - detector class supervision은 `traffic_light / sign` only
  - tl attr valid-mask only
  - lane family off
- AIHUB lane
  - det off
  - tl attr off
  - lane/stop-line/crosswalk on

## current status

- spec는 [../model/loss/spec.py](../model/loss/spec.py)에 반영돼 있다.
- smoke/runtime loss는 [../model/loss/runtime.py](../model/loss/runtime.py)에 반영돼 있다.
- current runtime은 finite loss와 backward smoke를 목표로 한다.
- detector matching은 task-aligned assigner 기준으로 동작한다.
- TL attr supervision은 matched detector positive의 GT index를 재사용한다.
- lane family는 Hungarian matching 기준으로 objectness와 geometry target을 query에 재배치한다.
- inference postprocess는 raw detector slot output을 prediction bundle로 decode한다.

## 구현 우선순위

1. full-epoch trainer wiring
2. export / ROS 정교화
3. dataset-level metric aggregation 정교화

## raw model output contract

- detector raw output은 `B x Q_det x (4 bbox + 1 obj + 7 cls)`다.
- TL attr raw output은 `B x Q_det x 4`다.
- 두 raw output은 같은 detector slot index를 공유한다.

## export / ROS prediction bundle

- postprocess 이후 prediction bundle은 `box_xyxy + score + class_id + class_name + tl_attr_scores`다.
- `traffic_light` prediction만 `tl_attr_scores`를 의미 있게 사용한다.
- export와 ROS message는 이 bundle을 기준으로 설계한다.
