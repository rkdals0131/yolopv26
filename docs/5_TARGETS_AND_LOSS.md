# Targets And Loss

## goals

- detector, TL attr, lane, stop-line, crosswalk를 하나의 학습 파이프라인에서 다룬다.
- partial label source를 손상시키지 않는다.
- sample/transform contract는 [4A_SAMPLE_AND_TRANSFORM_CONTRACT.md](4A_SAMPLE_AND_TRANSFORM_CONTRACT.md)를 기준으로 한다.
- best checkpoint 선택과 phase 종료 기준을 현재 runtime 구현과 정확히 일치하게 문서화한다.

## current implementation snapshot

- runtime loss는 [../model/engine/loss.py](../model/engine/loss.py)에 구현돼 있다.
- detector matching은 task-aligned assigner, lane family는 Hungarian matching으로 고정돼 있다.
- default meta-train selection path는 `selection_metrics.phase_objective`다.
- shipped preset의 selection mode는 `max`다. 즉 값이 클수록 더 좋은 epoch로 취급한다.
- shipped preset의 phase 종료 기준은 `patience + min_delta_abs`다.
- legacy config 호환을 위해 `min_delta_abs`가 없는 phase는 `min_improvement_pct` 상대 개선율 기준으로 fallback한다.

## task summary

- OD
  - 7-class detection
- TL attr
  - 4-bit sigmoid attribute
- lane
  - 16 anchor-row x / visibility
- stop-line
  - 2 endpoints + width
- crosswalk
  - 4-corner quad

## internal target encoding

- encoded batch는 transformed network pixel space를 입력으로 받는다.
- detector prediction slot 수는 `Q_det`, GT detection row 수는 `N_gt_det`로 분리한다.
- lane
  - objectness
  - color logits 3
  - type logits 2
  - anchor-row x 16
  - visibility logits 16
- stop-line
  - objectness
  - endpoints 2
  - width 1
- crosswalk
  - objectness
  - quad corners 4

## matching policy

- detector
  - task-aligned assigner runtime 통합 완료
  - real trunk path에서는 feature-shape metadata를 이용해 anchor grid를 만들고 assignment를 계산한다.
  - metadata가 없거나 assigner가 실패하면 `PV26DetAssignmentUnavailable`를 올린다.
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

## task-wise loss coefficients

### detector loss

- type
  - YOLO detect loss
- terms
  - box
  - obj
  - cls
- partial-det policy
  - sample별 `det_supervised_class_mask`를 따른다.
  - `det_allow_objectness_negatives=False`인 sample은 unmatched query를 objectness background negative로 쓰지 않는다.
  - `det_allow_unmatched_class_negatives=True`인 sample은 unmatched query에서도 source-owned class channel의 zero-target BCE를 계산한다.
  - matched cls BCE와 unmatched negative cls BCE는 분리 정규화한다.
  - unmatched negative cls term에는 `0.1` weight를 적용한다.

### TL attr loss

- type
  - masked sigmoid focal BCE
- 적용 대상
  - detector assignment 기준 matched `traffic_light` positive
  - valid AIHUB car signal only
- bit weights
  - red `1.0`
  - yellow `2.5`
  - green `1.0`
  - arrow `1.8`

### lane loss

- objectness `1.0`
- color CE `1.0`
- type CE `0.5`
- anchor x SmoothL1 `5.0`
- visibility BCE `1.0`
- smoothness `0.25`
- visibility TV `0.1`

### stop-line loss

- objectness `1.0`
- endpoints SmoothL1 `6.0`
- width SmoothL1 `1.0`
- angle/length `0.5`

### crosswalk loss

- objectness `1.0`
- corner SmoothL1 `4.0`
- shape regularizer `0.5`

## stage-aware loss policy

현재 loss spec에 들어간 stage별 task loss weight는 아래와 같다.

| stage | det | tl_attr | lane | stop_line | crosswalk | 의도 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `stage_1_frozen_trunk_warmup` | 1.0 | 0.5 | 2.0 | 1.5 | 0.0 | 새 head와 lane geometry를 빠르게 안정화 |
| `stage_2_partial_unfreeze` | 1.0 | 0.5 | 1.5 | 1.0 | 0.5 | detector와 lane family를 함께 다시 맞춤 |
| `stage_3_end_to_end_finetune` | 1.0 | 0.5 | 1.0 | 1.0 | 0.75 | 전체 multitask 균형 조정 |
| `stage_4_lane_family_finetune` | 0.0 | 0.0 | 1.5 | 1.25 | 1.0 | lane family late fine-tune 집중 |

주의:

- 위 표는 loss 가중치다.
- best checkpoint와 early exit는 아래의 `phase objective`를 사용한다.
- loss weight와 selection objective는 같은 stage 목적을 향하지만, 같은 숫자를 재사용하지는 않는다.

## dataset masking

- BDD100K
  - det on
  - detector objectness negative off
  - detector unmatched class negative on
  - detector class supervision은 `vehicle / bike / pedestrian` only
  - tl attr off
  - lane off
  - stop-line off
  - crosswalk off
- AIHUB traffic
  - det on
  - detector objectness negative off
  - detector unmatched class negative on
  - detector class supervision은 `traffic_light / sign` only
  - tl attr valid-mask only
  - lane family off
- AIHUB obstacle
  - det on
  - detector objectness negative off
  - detector unmatched class negative on
  - detector class supervision은 `traffic_cone / obstacle` only
  - tl attr off
  - lane family off
- AIHUB lane
  - det off
  - tl attr off
  - lane/stop-line/crosswalk on

## current phase selection and early exit

### default shipped config

현재 shipped preset의 기본 selection 설정은 아래다.

```yaml
selection:
  metric_path: selection_metrics.phase_objective
  mode: max
  eps: 1.0e-8
```

의미:

- `metric_path`
  - epoch summary에서 어떤 scalar를 best/early-exit monitor로 읽을지 정한다.
  - shipped preset은 `selection_metrics.phase_objective`를 사용한다.
- `mode`
  - `max`면 값이 클수록 좋다.
  - `min`이면 값이 작을수록 좋다.
  - shipped preset은 `phase_objective`가 utility score이므로 `max`를 쓴다.
- `eps`
  - relative improvement fallback 계산의 분모 하한이다.
  - `min_delta_abs`를 쓰는 shipped preset에서는 영향이 거의 없다.
  - 특별한 이유가 없으면 `1e-8` 그대로 둔다.

### phase objective component formulas

현재 runtime은 validation metric에서 아래 component score를 계산한다.

```text
det_score
  = 0.70 * detector.map50
  + 0.30 * detector.f1

tl_score
  = 0.60 * traffic_light.mean_f1
  + 0.40 * traffic_light.combo_accuracy

lane_score
  = 0.55 * lane.f1
  + 0.20 * (1 - clamp01(lane.mean_point_distance / 40))
  + 0.15 * lane.color_accuracy
  + 0.10 * lane.type_accuracy

stop_score
  = 0.55 * stop_line.f1
  + 0.25 * (1 - clamp01(stop_line.mean_point_distance / 40))
  + 0.20 * (1 - clamp01(stop_line.mean_angle_error / 30))

cross_score
  = 0.45 * crosswalk.f1
  + 0.35 * crosswalk.mean_polygon_iou
  + 0.20 * (1 - clamp01(crosswalk.mean_vertex_distance / 60))
```

여기서 `clamp01(x) = min(max(x, 0), 1)`다.

즉:

- point distance가 0이면 distance score는 1.0이다.
- lane/stop-line 평균 거리 40px에서 distance score는 0.0에 닿는다.
- stop-line 평균 angle error 30deg에서 angle score는 0.0에 닿는다.
- crosswalk 평균 vertex distance 60px에서 vertex score는 0.0에 닿는다.

### support-aware reliability

lane family는 sparse epoch 흔들림을 줄이기 위해 reliability를 곱한다.

```text
support(task) = tp + fn
reliability(task) = min(1, sqrt(support / ref))
```

현재 ref는 아래로 고정돼 있다.

- lane: `300`
- stop_line: `80`
- crosswalk: `80`

주의:

- detector와 traffic_light는 reliability를 `1.0`으로 둔다.
- lane/stop_line/crosswalk는 weight가 0보다 크더라도 `support == 0`이면 objective 합산에서 제외된다.
- 제외는 0점 취급이 아니라 “이번 epoch composite 계산에서 빠짐”이다.

### stage objective weights

phase objective는 stage마다 아래 weight를 사용한다.

| stage | detector | traffic_light | lane | stop_line | crosswalk |
| --- | ---: | ---: | ---: | ---: | ---: |
| `stage_1_frozen_trunk_warmup` | 0.25 | 0.05 | 0.45 | 0.25 | 0.00 |
| `stage_2_partial_unfreeze` | 0.25 | 0.05 | 0.40 | 0.20 | 0.10 |
| `stage_3_end_to_end_finetune` | 0.30 | 0.05 | 0.35 | 0.18 | 0.12 |
| `stage_4_lane_family_finetune` | 0.00 | 0.00 | 0.50 | 0.30 | 0.20 |

실제 계산은 아래와 같다.

```text
effective_weight_i
  = raw_weight_i                      for detector / traffic_light
  = raw_weight_i * reliability_i     for lane / stop_line / crosswalk

phase_objective
  = sum(score_i * effective_weight_i) / sum(effective_weight_i)
```

즉:

- lane family support가 충분하면 raw weight와 거의 같게 작동한다.
- support가 작으면 그 task의 영향이 자동으로 줄어든다.
- stage 4는 detector/TL을 objective에서 완전히 뺀다.

## shipped early-exit rule

### stop state fields

현재 runtime은 epoch마다 `phase_transition`에 아래 값을 남긴다.

- `current_metric_value`
- `best_metric_value`
- `current_phase_objective`
- `best_phase_objective`
- `plateau_count`
- `last_improvement_abs`
- `last_improvement_pct`
- `selection_metric_path`
- `selection_mode`
- `min_epochs`
- `max_epochs`
- `patience`
- `min_delta_abs`
- `improvement_policy`

### absolute-delta mode

shipped preset은 모든 phase에 `min_delta_abs`가 들어 있으므로 아래 로직을 쓴다.

```text
1. epoch < min_epochs 인 동안은 plateau stop을 걸지 않는다.
2. current_metric > best_metric 이면 best는 항상 갱신한다.   (mode=max 기준)
3. 하지만 improvement_abs = current_metric - previous_best 가
   min_delta_abs 미만이면 plateau_count는 reset하지 않고 +1 한다.
4. improvement_abs >= min_delta_abs 이면 plateau_count를 0으로 reset한다.
5. improvement가 전혀 없으면 plateau_count를 +1 한다.
6. epoch >= max_epochs 이면 무조건 종료한다.
7. epoch >= min_epochs 이고 plateau_count >= patience 이면 `reason=plateau`로 종료한다.
```

중요:

- best checkpoint 갱신과 plateau reset은 같은 조건이 아니다.
- 즉 “조금이라도 나아진 checkpoint”는 best로 저장될 수 있지만, `min_delta_abs`를 못 넘기면 phase는 plateau 쪽으로 계속 진행될 수 있다.

### legacy fallback mode

`min_delta_abs`가 없는 phase는 legacy relative improvement 모드로 fallback한다.

```text
improvement_pct
  = ((current - previous_best) / max(abs(previous_best), eps)) * 100    for mode=max
```

이때 plateau reset 기준은 `improvement_pct >= min_improvement_pct`다.

## shipped phase defaults

현재 `config/pv26_train_hyperparameters.yaml` 기본값은 아래다.

| stage | min_epochs | max_epochs | patience | min_delta_abs | legacy min_improvement_pct |
| --- | ---: | ---: | ---: | ---: | ---: |
| `stage_1_frozen_trunk_warmup` | 2 | 6 | 2 | 0.003 | 1.0 |
| `stage_2_partial_unfreeze` | 4 | 10 | 3 | 0.003 | 0.5 |
| `stage_3_end_to_end_finetune` | 21 | 48 | 8 | 0.0025 | 0.25 |
| `stage_4_lane_family_finetune` | 12 | 40 | 10 | 0.002 | 0.25 |

의도:

- stage 1/2는 warm-up과 partial unfreeze를 짧게 끝낸다.
- stage 3는 shipped exhaustive train split 약 25.2만 장, `batch_size=12`, `train_batches=2048` 기준으로 최소 약 `2.05x` exposure를 확보한다.
- stage 4는 lane-only sampler로 lane family를 의도적으로 오래 밀어붙인다.
- 현재 lane split 3만 장 기준 stage 4 min 12 epoch는 약 `26.2x` exposure다.
- lane split이 추후 약 33만 장까지 커져도 stage 4 min 12 epoch는 약 `2.38x` exposure를 확보한다.

추가로 shipped preset의 phase override는 아래다.

| stage | batch_size | trunk_lr | head_lr | 특징 |
| --- | ---: | ---: | ---: | --- |
| `stage_1_frozen_trunk_warmup` | 32 | 5e-5 | 3e-3 | head warm-up |
| `stage_2_partial_unfreeze` | 24 | 3e-5 | 8e-4 | partial unfreeze |
| `stage_3_end_to_end_finetune` | 12 | 1e-5 | 4e-4 | full fine-tune |
| `stage_4_lane_family_finetune` | 32 | 0.0 | 2e-4 | lane-only sampler / lane-family heads only |

## tuning guide

### `selection.metric_path`

추천:

- 기본은 `selection_metrics.phase_objective`
- detector만 강하게 보려는 실험이면 `val.metrics.detector.map50_95`
- lane family만 단순 F1로 빠르게 비교하려면 `val.metrics.lane_family.mean_f1`

영향:

- path를 바꾸면 best checkpoint 저장과 early exit 판단이 둘 다 그 scalar를 따라간다.
- shipped preset처럼 `phase_objective`를 쓰면 multi-task 균형을 본다.
- raw task metric으로 바꾸면 특정 head만 밀어주는 실험이 된다.

### `selection.mode`

- `max`
  - utility, F1, mAP, accuracy 같이 “클수록 좋은 값”에 쓴다.
- `min`
  - loss, error 같이 “작을수록 좋은 값”에 쓴다.

기본:

- `selection_metrics.phase_objective`는 반드시 `max`

### `min_delta_abs`

의미:

- 현재 best보다 얼마나 더 좋아져야 plateau_count를 reset할지 정하는 절대 개선폭
- 현재 objective scale은 대체로 `0.0 ~ 1.0` 범위다

해석:

- `0.005`
  - stage 1처럼 metric 출렁임이 큰 구간에 적합
  - “0.5%p 정도는 올라야 의미 있는 진전”으로 간주
- `0.003`
  - stage 3/4처럼 late fine-tune에서 작은 개선도 반영하고 싶을 때 적합

조정 가이드:

- 너무 빨리 멈춘다
  - `min_delta_abs`를 낮춘다. 예: `0.005 -> 0.003`
  - 또는 `patience`를 늘린다. 예: `5 -> 7`
- 너무 오래 안 멈춘다
  - `min_delta_abs`를 높인다. 예: `0.003 -> 0.006`
  - 또는 `patience`를 줄인다. 예: `8 -> 5`
- validation 노이즈가 심하다
  - 먼저 `patience`를 올리고, 그 다음 `min_delta_abs`를 약간 올린다

### `patience`

의미:

- `min_epochs`를 지난 뒤 “의미 있는 개선이 없던 epoch”를 몇 번까지 허용할지

해석:

- 작게 잡으면 phase가 빨리 끝난다
- 크게 잡으면 노이즈를 더 견디지만 runtime이 길어진다

조정 가이드:

- head warm-up이 자주 출렁이면 `+1 ~ +2`
- late fine-tune에서 이미 충분히 안정적이면 `-1 ~ -2`

### `min_epochs`

의미:

- plateau stop을 걸기 시작하는 최소 epoch

권장:

- warm-up phase는 `2 ~ 4`가 보통 충분하다.
- end-to-end phase는 현재 batch budget 기준 sample exposure로 잡는 편이 낫다.
- shipped preset은 stage 3 `min_epochs=21`로 fully-unfrozen 구간에서 train split을 대략 두 번 보게 맞춘다.

### `max_epochs`

의미:

- hard stop 상한

조정 가이드:

- `patience`를 키웠다면 `max_epochs`도 같이 늘려야 late improvement를 살릴 수 있다

### `min_improvement_pct`

- legacy fallback 전용이다.
- 새 shipped preset에서는 `min_delta_abs`가 있으므로 실제 stop 판정에는 사용되지 않는다.
- 오래된 user YAML과 run resume 호환 때문에 field 자체는 남겨둔다.

## what is implemented vs next

### implemented now

- task-aligned detector assignment
- Hungarian lane-family assignment
- stage-aware loss weights
- stage-aware composite objective
- support-aware reliability for lane family
- `selection_metrics.phase_objective` default monitor
- `patience + min_delta_abs` early-exit policy
- legacy `% improvement` fallback compatibility

### next improvements

- EMA / slope 기반 plateau 판정
- divergence guard
- task floor / canary guard
- stage 4 BN drift 방지 정책
- YAML에서 objective weights / support ref를 직접 열지 여부 재검토

## raw model output contract

- detector raw output은 `B x Q_det x (4 bbox + 1 obj + 7 cls)`다.
- TL attr raw output은 `B x Q_det x 4`다.
- 두 raw output은 같은 detector slot index를 공유한다.

## export / ROS prediction bundle

- postprocess 이후 prediction bundle은 `box_xyxy + score + class_id + class_name + tl_attr_scores`다.
- `traffic_light` prediction만 `tl_attr_scores`를 의미 있게 사용한다.
- export와 ROS message는 이 bundle을 기준으로 설계한다.
