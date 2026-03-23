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
  - upstream YOLO assignment
- lane family
  - Hungarian matching

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
  - tl attr off
  - lane off
  - stop-line off
  - crosswalk off
- AIHUB traffic
  - det on
  - tl attr valid-mask only
  - lane family off
- AIHUB lane
  - det off
  - tl attr off
  - lane/stop-line/crosswalk on

## current status

- spec는 [../model/loss/spec.py](../model/loss/spec.py)에 반영돼 있다.
- 실제 runtime loss 구현은 아직 없다.

## 구현 우선순위

1. target encoder
2. masked loss
3. Hungarian matcher
4. smoke backward

## inference output contract

- detector output의 기본 묶음은 `bbox + det score + class logits`다.
- `traffic_light` prediction에는 여기에 `red/yellow/green/arrow` score가 추가로 붙는다.
- export와 ROS message는 이 묶음을 기준으로 설계한다.
