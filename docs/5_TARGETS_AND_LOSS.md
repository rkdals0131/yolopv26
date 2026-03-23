# Targets And Loss

## 목표

- detector, TL attr, lane, stop-line, crosswalk를 하나의 학습 파이프라인에서 다룬다.
- partial label source를 손상시키지 않는다.

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
  - matched `traffic_light` positive
  - valid AIHUB car signal only
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

- spec는 [model/loss/spec.py](/home/user1/ROS2_Workspace/ros2_ws/src/YOLOpv26/model/loss/spec.py)에 반영돼 있다.
- 실제 runtime loss 구현은 아직 없다.

## 구현 우선순위

1. target encoder
2. masked loss
3. Hungarian matcher
4. smoke backward
