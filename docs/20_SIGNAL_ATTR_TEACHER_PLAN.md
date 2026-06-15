# Signal Attr Teacher Plan

## 3줄 요약

- `best_signal.pt`는 `traffic_light`와 `sign` box를 내는 OD teacher로만 쓰고, red/yellow/green/arrow 상태를 예측하는 teacher로 해석하지 않는다.
- v1 attr teacher는 full-image YOLO custom head가 아니라 accepted traffic-light ROI crop을 입력으로 받는 작은 sidecar classifier `best_signal_attr.pt`다.
- Sidecar는 OD policy/NMS/audit를 통과한 `traffic_light` box에만 적용한다. `sign`, rejected TL candidate, raw candidate에는 적용하지 않는다.

## 이번 문서에서 확정한 것

| 항목 | 결정 |
| --- | --- |
| 현재 signal teacher | `best_signal.pt`는 `traffic_light`, `sign` box만 담당 |
| 금지 해석 | `best_signal.pt`를 TL color/arrow state teacher로 쓰지 않음 |
| attr artifact | `best_signal_attr.pt` |
| v1 teacher 형태 | traffic-light ROI crop classifier |
| 권장 내부 head | `base_color` softmax + `arrow` sigmoid |
| 외부 output contract | PV26 canonical `red`, `yellow`, `green`, `arrow` bits |
| integration | OD audit를 통과한 accepted `traffic_light` box에만 적용 |
| source key policy | attr-enabled materialization은 새 v2 또는 `*_attrpseudo_v1` key 사용 |

이 작업은 [18_ETRI_KCITY_CAMERA_TO_PV26_LABELS.md](18_ETRI_KCITY_CAMERA_TO_PV26_LABELS.md)와 [19_LANE_VAL_OD_TEACHER_EVALSET.md](19_LANE_VAL_OD_TEACHER_EVALSET.md)를 기다릴 필요가 거의 없는 병렬 stream이다. 공통으로 맞출 것은 source key, `traffic_lights[].detection_id`, `labels_det` row order, `tl_attr_valid`, metric enable/disable, manifest failure contract뿐이다.

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

## 왜 v1은 ROI classifier인가

목표는 pseudo-label teacher다. 입력은 이미 `best_signal.pt`가 찾고 OD audit가 keep한 traffic-light ROI다. 따라서 v1에는 full-image detector 재설계가 필요 없다.

권장 v1 구조:

```text
SignalAttrCropClassifier
  input: cropped traffic-light ROI
  backbone: small CNN / MobileNetV3 / ResNet18급 / EfficientNet-B0급
  head_base_color: off / red / yellow / green logits
  head_arrow: binary logit
  output adapter: PV26 red/yellow/green/arrow bits
```

Loss:

```text
base_color_loss = CrossEntropyLoss(weighted)
arrow_loss = BCEWithLogitsLoss(pos_weight=...)
total_loss = base_color_loss + lambda_arrow * arrow_loss
```

문서상 output은 4-bit이지만 내부 head를 꼭 4-bit multi-label로 만들 필요는 없다. `red/yellow/green`은 현재 policy상 mutual exclusive에 가깝고, `arrow`만 독립 binary로 두는 편이 불가능한 base-color 조합을 줄인다.

YOLO/PV26 custom head 통합은 runtime one-pass model이 필요해진 뒤에 고려한다. 지금 custom head부터 붙이면 dataset target encoding, loss, postprocess, export, evaluator, checkpoint compatibility를 동시에 바꿔야 해서 teacher pipeline에는 과하다.

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
- invalid/empty/out-of-bound crop을 black crop 등으로 대체해 classifier에 넣지 않는다.

Crop 규칙은 단일 config artifact로 고정한다.

```yaml
input_size: 128
padding_ratio: 0.15
min_crop_side_px: 4
clip_to_image: true
color_space: rgb
normalization: imagenet
interpolation: bilinear
```

config 없이 padding, resize, normalization을 코드 여러 곳에 흩뿌리면 안 된다.

## Training dataset

Training label은 AIHUB signal raw annotation에서만 만든다.

예상 root:

```text
seg_dataset/pv26_signal_attr_crops/
  images/train/*.jpg
  images/val/*.jpg
  labels/train.jsonl
  labels/val.jsonl
  meta/
    signal_attr_dataset_manifest.json
    rejected_rows.jsonl
    crop_config.yaml
```

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
- `invalid_crop`

제외 row는 버리지 않고 split별 reason count로 남긴다.

Label row 최소 형식:

```json
{
  "sample_id": "sample_id",
  "source_image_path": "absolute/or/provenance/path.jpg",
  "bbox": [100.0, 120.0, 140.0, 180.0],
  "crop_path": "images/train/sample_id.jpg",
  "base_color": "red",
  "arrow": 1,
  "tl_bits": {"red": 1, "yellow": 0, "green": 0, "arrow": 1},
  "source_dataset": "aihub_traffic_seoul",
  "collapse_reason": "valid"
}
```

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

Accepted OD와 accepted attr은 별개다. Traffic-light box가 OD label로 accepted되어도 attr audit가 실패하면 detection row는 유지하고 attr만 invalid로 둔다.

`traffic_lights[].detection_id`는 최종 accepted detection list의 row index다. `labels_det` row order와 scene `detections[]` order가 다르면 안 된다.

## Collapse reason vocabulary

구현 pass는 아래 vocabulary를 기본으로 쓴다.

```text
valid
signal_attr_teacher_low_confidence
signal_attr_teacher_ambiguous_bits
signal_attr_teacher_invalid_roi
signal_attr_teacher_nonfinite_logits
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
invalid_crop
```

새 reason을 추가하면 attr teacher report와 scene schema docs에 같이 추가한다.

## Threshold policy

`threshold_policy="signal_attr_v1"`는 이름만 두면 안 된다. release 전에 아래를 고정한다.

- base-color confidence threshold
- arrow positive threshold
- arrow ambiguity band
- low-confidence reason mapping
- allowed combo set
- calibration split과 최소 support count
- nonfinite logit 처리

`missing_checkpoint`, OOM, batch-level fatal error는 per-ROI low confidence가 아니다. attr-enabled build에서는 ready manifest를 쓰지 않고 fail한다.

## Integration 순서

Pseudo-label materialization은 항상 두 단계다.

1. `best_signal.pt`로 `traffic_light`, `sign` OD candidate를 만든다.
2. OD policy/NMS/audit를 통과한 `traffic_light` box에만 `best_signal_attr.pt`를 적용한다.

개념 흐름:

```text
best_signal.pt
  -> traffic_light / sign OD predictions
  -> OD policy / NMS / audit keep
  -> accepted traffic_light boxes only
  -> best_signal_attr.pt ROI crop classifier
  -> traffic_lights[] sidecar rows
  -> PV26 canonical tl_bits / tl_attr_valid
```

`sign`에는 2단계를 적용하지 않는다. OD audit에서 떨어진 traffic-light candidate에도 2단계를 적용하지 않는다.

[19_LANE_VAL_OD_TEACHER_EVALSET.md](19_LANE_VAL_OD_TEACHER_EVALSET.md)의 `pv26_eval_lane_val_odpseudo_v1`은 이 sidecar teacher가 있어도 그대로 `has_tl_attr=0`이다. TL attr을 켜려면 아래처럼 새 source key 또는 v2 dataset 계약이 필요하다.

```text
pv26_eval_lane_val_odpseudo_v1
  - OD teacher agreement only
  - has_tl_attr=0

pv26_eval_lane_val_odpseudo_attr_v2
  - OD teacher agreement
  - signal_attr sidecar agreement
  - tl_attr can be enabled
```

## Exhaustive OD 연결

실용적인 우선순위는 `signal_attr`를 먼저 만들고 exhaustive OD sweep에 sidecar hook을 붙이는 것이다.

권장 옵션:

```yaml
exhaustive_od:
  signal_attr_sidecar:
    enabled: true
    checkpoint_path: runs/signal_attr/best_signal_attr.pt
    crop_config_path: config/signal_attr_crop.yaml
    threshold_policy: signal_attr_v1
    apply_to_classes: ["traffic_light"]
    require_checkpoint: true
```

기존 `pv26_exhaustive_*` key의 의미를 조용히 바꾸지 않는다. attr-enabled 산출물은 새 key를 쓴다.

예:

```text
pv26_exhaustive_bdd100k_det_100k_attrpseudo_v1
pv26_exhaustive_aihub_traffic_seoul_attrpseudo_v1
pv26_exhaustive_aihub_obstacle_seoul_attrpseudo_v1
```

또는 기존 naming을 유지해야 하면 `*_v2`를 쓴다. 중요한 점은 기존 source key의 `tl_attr` 의미를 변경하지 않는 것이다.

Attr-enabled source의 scene `tasks.has_tl_attr`는 positive-content convention을 따른다.

```text
tasks.has_tl_attr = int(any traffic_lights row has tl_attr_valid == 1)
```

Sidecar 실행 여부와 checkpoint provenance는 manifest에 따로 둔다.

```json
{
  "signal_attr_sidecar": {
    "status": "completed",
    "checkpoint": "best_signal_attr.pt",
    "roi_count": 12345,
    "valid_attr_count": 11000,
    "invalid_attr_count": 1345
  }
}
```

## Metrics

`best_signal_attr.pt`를 usable artifact로 보려면 최소 report가 있어야 한다.

- bit별 precision/recall/F1/AP
- base-color accuracy
- arrow precision/recall/F1
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
4. crop config artifact가 있다.
5. held-out AIHUB signal validation report가 있다.
6. GT box와 accepted OD box 두 입력 조건의 metric이 모두 있다.
7. accepted OD box 성능이 release threshold를 통과했다.
8. prediction JSON이 canonical `traffic_lights[]` row를 채우는 integration test가 있다.
9. missing checkpoint, OOM, nonfinite logits가 ready manifest로 넘어가지 않는 failure test가 있다.
10. attr-disabled eval root에서 `has_tl_attr=0`과 no TL attr metric이 유지되는 regression test가 있다.

하나라도 빠지면 TL attr pseudo label을 생성하지 않는다.

## 필수 테스트

- `test_signal_attr_dataset_uses_only_traffic_worker_valid_rows`
- `test_signal_attr_label_extractor_matches_traffic_worker_policy`
- `test_signal_attr_crop_rejects_empty_or_nonfinite_roi`
- `test_signal_attr_model_outputs_canonical_tl_bits`
- `test_signal_attr_sidecar_runs_only_after_od_policy_keep`
- `test_signal_attr_sidecar_does_not_apply_to_sign`
- `test_signal_attr_sidecar_does_not_apply_to_rejected_traffic_light_candidate`
- `test_signal_attr_low_confidence_keeps_detection_and_invalidates_attr_only`
- `test_signal_attr_detection_id_matches_final_detections_row_order`
- `test_signal_attr_enabled_build_fails_on_missing_checkpoint`
- `test_signal_attr_nonfinite_logits_fail_or_reject_with_closed_reason`
- `test_exhaustive_attr_v2_uses_new_source_key_not_mutating_v1_semantics`
- `test_signal_attr_requires_gt_and_accepted_od_box_reports_for_release`
