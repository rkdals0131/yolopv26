# Signal Attr Teacher Plan

## 3줄 요약

- `best_signal.pt`는 `traffic_light`와 `sign` box를 내는 OD teacher로만 쓰고, red/yellow/green/arrow 상태를 예측하는 teacher로 해석하지 않는다.
- TL attribute는 별도 sidecar인 `best_signal_attr.pt`가 traffic-light ROI crop에서 4-bit `red/yellow/green/arrow`를 예측하는 방식으로 만든다.
- Attr label은 AIHUB signal annotation 중 `traffic_worker.py` 정책을 통과한 row에서만 만들고 accepted traffic-light box에만 적용한다. 이 문서 pass에서는 teacher 학습 코드, checkpoint, evaluator, pseudo-label builder를 만들지 않는다.

## 이번 문서에서 확정한 것

| 항목 | 결정 |
| --- | --- |
| 현재 signal teacher | `best_signal.pt`는 `traffic_light`, `sign` box만 담당 |
| 금지 해석 | `best_signal.pt`를 TL color/arrow state teacher로 쓰지 않음 |
| attr artifact | `best_signal_attr.pt` |
| teacher 형태 | traffic-light ROI crop 기반 4-bit multi-label classifier |
| output bits | `red`, `yellow`, `green`, `arrow` |
| integration | accepted traffic-light box에만 attr teacher 적용 |

## 현재 코드의 기준점

Canonical bit 이름은 [common/pv26_schema.py](../common/pv26_schema.py)의 `TL_BITS`와 같다.

```text
red, yellow, green, arrow
```

AIHUB traffic source의 현재 raw-label collapse policy는 [tools/od_bootstrap/source/aihub/traffic_worker.py](../tools/od_bootstrap/source/aihub/traffic_worker.py)에 있다.

현재 policy는 다음 조건에서만 `tl_attr_valid=1`을 낸다.

1. raw traffic-light `type`이 `car`다.
2. raw `attribute` map이 있다.
3. `x_light`가 on이 아니다.
4. `red`, `yellow`, `green` 중 on인 base color가 0개 또는 1개다.
5. `arrow`는 `left_arrow` 또는 `others_arrow`가 on이면 1이다.

이 policy에서 `off`, `arrow`, `red+arrow`, `yellow+arrow`, `green+arrow` 같은 조합은 valid가 된다. `red+yellow`, `red+green`, `yellow+green`처럼 base color가 2개 이상 켜지면 invalid다.

`best_signal_attr.pt`는 이 정책과 충돌하면 안 된다. 다르게 collapse하려면 새 정책 이름과 migration 문서를 먼저 만들어야 한다.

## Teacher input

Attr teacher의 입력은 traffic-light ROI crop이다.

허용 box source:

- training: AIHUB signal raw label의 GT traffic-light box
- robustness eval: accepted `best_signal.pt` traffic-light box
- pseudo-label materialization: accepted `best_signal.pt` traffic-light box

금지:

- full image 하나에서 scene-level TL state를 직접 예측하지 않는다.
- sign box에는 attr teacher를 적용하지 않는다.
- rejected traffic-light OD candidate에는 attr teacher를 적용하지 않는다.
- OD teacher의 class confidence를 attr confidence로 재사용하지 않는다.

Crop 규칙은 구현 때 config로 고정한다. config 없이 padding, resize, normalization을 코드에 흩뿌리면 안 된다.

## Training label source

Training label은 AIHUB signal raw annotation에서만 만든다.

사용 가능한 row:

- `traffic_worker.py` 기준으로 `tl_attr_valid=1`
- bbox가 유효함
- ROI crop을 만들 수 있음

학습에서 제외할 row:

- `non_car_traffic_light`
- `missing_attribute_map`
- `x_light_active`
- `multi_color_active`
- `traffic_light_invalid_bbox`

제외 row는 버리지 않고 split별 reason count로 남긴다.

## Output contract

Accepted attr prediction은 canonical traffic-light scene row를 채운다.

```json
{
  "id": 0,
  "detection_id": 3,
  "bbox": [100.0, 120.0, 140.0, 180.0],
  "tl_bits": {"red": 1, "yellow": 0, "green": 0, "arrow": 0},
  "tl_attr_valid": 1,
  "collapse_reason": "valid",
  "meta": {
    "label_origin": "teacher_pseudo",
    "teacher_name": "signal_attr",
    "checkpoint": "best_signal_attr.pt",
    "threshold_policy": "signal_attr_v1"
  }
}
```

Rejected attr prediction은 detection은 유지하되 attr만 invalid로 둔다.

```json
{
  "id": 0,
  "detection_id": 3,
  "bbox": [100.0, 120.0, 140.0, 180.0],
  "tl_bits": {"red": 0, "yellow": 0, "green": 0, "arrow": 0},
  "tl_attr_valid": 0,
  "collapse_reason": "signal_attr_teacher_low_confidence"
}
```

Accepted OD와 accepted attr은 별개다. Traffic-light box가 OD label로 accepted되어도 attr audit가 실패하면 attr은 invalid로 둔다.

## Collapse reason vocabulary

구현 pass는 아래 vocabulary를 기본으로 쓴다.

```text
valid
signal_attr_teacher_low_confidence
signal_attr_teacher_ambiguous_bits
signal_attr_teacher_invalid_roi
signal_attr_teacher_missing_checkpoint
signal_attr_teacher_not_run
```

AIHUB raw label에서 온 invalid reason은 기존 worker vocabulary를 유지한다.

```text
non_car_traffic_light
missing_attribute_map
x_light_active
multi_color_active
traffic_light_invalid_bbox
```

새 reason을 추가하면 attr teacher report와 scene schema docs에 같이 추가한다.

## Integration 순서

Pseudo-label materialization은 항상 두 단계다.

1. `best_signal.pt`로 `traffic_light`, `sign` OD candidate를 만든다.
2. OD audit를 통과한 `traffic_light` box에만 `best_signal_attr.pt`를 적용한다.

`sign`에는 2단계를 적용하지 않는다. OD audit에서 떨어진 traffic-light candidate에도 2단계를 적용하지 않는다.

[19_LANE_VAL_OD_TEACHER_EVALSET.md](19_LANE_VAL_OD_TEACHER_EVALSET.md)의 `pv26_eval_lane_val_odpseudo_v1`은 이 sidecar teacher가 있어도 그대로 `has_tl_attr=0`이다. TL attr을 켜려면 새 source key 또는 v2 dataset 계약이 필요하다.

## Metrics

`best_signal_attr.pt`를 usable artifact로 보려면 최소 report가 있어야 한다.

- bit별 precision/recall/F1/AP
- exact 4-bit combo accuracy
- combo별 support count
- collapse reason count
- invalid ROI count
- GT box 입력 성능
- accepted OD box 입력 성능

GT box 성능만 통과하고 accepted OD box 성능이 없으면 pseudo-label teacher로 사용할 수 없다. 그 경우는 crop classifier prototype일 뿐이다.

## 사용 가능 판정

TL attr teacher를 PV26 pseudo-label pipeline에 연결하려면 아래가 모두 필요하다.

1. AIHUB TL bit extraction unit test가 있다.
2. ROI crop bounds/padding/resize test가 있다.
3. `best_signal_attr.pt` checkpoint와 train manifest가 있다.
4. held-out AIHUB signal validation report가 있다.
5. GT box와 accepted OD box 두 입력 조건의 metric이 모두 있다.
6. prediction JSON이 canonical `traffic_lights[]` row를 채우는 integration test가 있다.
7. attr-disabled eval root에서 `has_tl_attr=0`과 no TL attr metric이 유지되는 regression test가 있다.

하나라도 빠지면 TL attr pseudo label을 생성하지 않는다.
