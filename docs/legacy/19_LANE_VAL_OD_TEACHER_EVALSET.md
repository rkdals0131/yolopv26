# Lane Validation OD Teacher Eval Set

## 3줄 요약

- `pv26_eval_lane_val_odpseudo_v1`은 기존 lane validation split만 base image set으로 쓰는 eval-only root다.
- OD label은 `best_mobility.pt`, `best_signal.pt`, `best_obstacle.pt`가 만든 candidate 중 policy/audit를 통과한 것만 쓰며, metric 의미는 human GT 성능이 아니라 teacher-pseudo agreement다.
- TL attribute는 v1에서 항상 off다. `best_signal.pt`를 red/yellow/green/arrow teacher로 해석하지 않고, attr sidecar는 별도 v2 key에서만 켠다.

## 이번 문서에서 확정한 것

| 항목 | 결정 |
| --- | --- |
| base image set | 기존 lane validation split만 사용 |
| 추가 label | audit를 통과한 OD teacher pseudo label |
| 제외 source | ETRI KCity, LiDAR, BDD 추가 split, AIHUB traffic/obstacle raw split 직접 유입 |
| TL attr | 항상 off, `tl_attr_valid=0`, `has_tl_attr=0` |
| OD metric 의미 | human GT 성능이 아니라 teacher-pseudo agreement |
| train 사용 | 금지. v1은 eval-only |

ETRI KCity camera 변환은 [18_ETRI_KCITY_CAMERA_TO_PV26_LABELS.md](18_ETRI_KCITY_CAMERA_TO_PV26_LABELS.md)의 별도 source다. ETRI sample은 이 eval root에 들어오면 안 된다.

## Source key와 runtime 등록

구현 pass에서 등록할 key는 아래 하나로 고정한다.

```text
pv26_eval_lane_val_odpseudo_v1
```

현재 loader는 [common/pv26_schema.py](../common/pv26_schema.py)의 `SOURCE_MASK_BY_DATASET`에 없는 key를 거부한다. 또한 `det=True` source는 `DET_SUPERVISION_BY_DATASET`에도 등록되어야 한다.

등록할 task mask:

```python
SOURCE_MASK_BY_DATASET["pv26_eval_lane_val_odpseudo_v1"] = {
    "det": True,
    "lane": True,
    "stop_line": True,
    "crosswalk": True,
    "tl_attr": False,
}
```

등록할 detector supervision policy:

```python
DET_SUPERVISION_BY_DATASET["pv26_eval_lane_val_odpseudo_v1"] = {
    "class_names": OD_CLASSES,
    "allow_objectness_negatives": False,
    "allow_unmatched_class_negatives": False,
}
```

teacher miss, skipped image, failed checkpoint, audit 보류를 true negative로 학습시키면 안 되므로 초기 policy는 보수적으로 둔다.

이 key는 `DATASET_GROUP_BY_KEY`에 넣지 않는다. train config나 dataloader가 `pv26_eval_lane_val_odpseudo_v1`의 `train` record를 발견하면 fail해야 한다.

## Scene task flag 의미

Scene `tasks.has_*`는 positive-content flag로 유지한다. "teacher/audit가 실행됨"과 "positive row가 있음"을 같은 필드에 넣지 않는다.

```json
{
  "has_det": 1,
  "has_lane": 1,
  "has_stop_line": 0,
  "has_crosswalk": 1,
  "has_tl_attr": 0
}
```

특정 scene에 accepted OD pseudo label이 0개면 `has_det=0`이 될 수 있다. 그래도 `det=True` source이므로 `labels_det/val/<sample_id>.txt`는 반드시 존재해야 하며, 빈 txt는 "세 teacher run과 audit가 정상 완료됐고 accepted box가 0개"라는 뜻이다.

감사 완료 여부는 manifest sample row에 따로 둔다.

```json
{
  "teacher_run_status": "completed",
  "det_file_status": "empty",
  "accepted_detection_count": 0,
  "candidate_count_by_teacher": {"mobility": 0, "signal": 0, "obstacle": 0},
  "failure_count": 0
}
```

주의: 일부 legacy helper는 `tasks.has_det=0`인데 det file이 있으면 stale det로 처리한다. 이 root는 canonical loader/evaluator 경로로 검증하거나, 해당 helper를 source-mask 기반으로 고친 뒤 사용한다.

## Base sample 규칙

입력 sample은 기존 lane validation split에서만 온다.

필수 조건:

- split은 `val`이다.
- image는 기존 lane validation image다.
- lane, stop-line, crosswalk label은 기존 lane scene label에서 온다.
- sample id는 base lane validation sample id를 유지한다.
- output sample id set과 count는 base validation set과 정확히 같다.

금지 조건:

- ETRI KCity image를 추가하지 않는다.
- LiDAR 또는 LiDAR-derived label을 추가하지 않는다.
- teacher가 발견한 object 때문에 새 image를 eval root에 추가하지 않는다.
- train/test split image를 `val`로 복사하지 않는다.

## OD pseudo label source

OD pseudo label은 세 teacher checkpoint만 사용한다.

| teacher | checkpoint | 허용 PV26 class |
| --- | --- | --- |
| mobility | `best_mobility.pt` | `vehicle`, `bike`, `pedestrian` |
| signal | `best_signal.pt` | `traffic_light`, `sign` |
| obstacle | `best_obstacle.pt` | `traffic_cone`, `obstacle` |

`best_signal.pt`는 traffic-light/sign box teacher다. red/yellow/green/arrow teacher가 아니다.

Teacher output을 scene에 넣으려면 policy/audit를 통과해야 한다. audit 전 raw prediction은 label이 아니라 candidate다.

Accepted detection row 최소 형식:

```json
{
  "id": 12,
  "class_name": "traffic_light",
  "bbox": [100.0, 120.0, 140.0, 180.0],
  "score": 0.91,
  "meta": {
    "label_origin": "teacher_pseudo",
    "teacher_name": "signal",
    "checkpoint": "best_signal.pt",
    "threshold_policy": "audit_gated_v1"
  }
}
```

`detections[]`, `labels_det`, `traffic_lights[].detection_id`는 같은 accepted detection list에서 생성한다. row order가 달라지면 loader가 crash하지 않고 잘못된 TL placeholder를 box에 붙일 수 있다.

## labels_det 파일 규칙

`det=True` source로 등록할 예정이므로 모든 scene은 대응되는 `labels_det/val/<sample_id>.txt`를 가진다.

- accepted detection이 1개 이상이면 YOLO 5-column row를 쓴다.
- accepted detection이 0개면 빈 txt 파일을 쓴다.
- 파일이 없는 상태는 "0개"가 아니라 "det label 생성 실패"다.

Empty txt는 아래 상태가 모두 manifest에 있을 때만 허용한다.

- `teacher_run_status="completed"`
- 세 teacher checkpoint가 모두 resolved됨
- image decode 성공
- candidate-level nonfinite bbox/score는 rejected reason으로 기록됨
- batch/model-level nonfinite tensor/logits/postprocess corruption은 없음
- OOM/fatal exception count가 0

missing checkpoint, OOM, model output NaN/Inf, image decode failure, postprocess exception은 empty txt로 위장하지 않는다. 이런 경우 ready manifest를 쓰면 안 된다.

## Rejected candidate 보존 규칙

Teacher raw prediction 중 accepted label이 아닌 것은 항상 `meta/rejected_detections.jsonl`에 per-candidate로 남긴다. scene-level `held_annotations`와 teacher별 audit report는 보조 요약으로만 쓴다.

각 rejected row는 최소한 아래 join key를 가진다.

```json
{
  "sample_id": "sample_id",
  "teacher_name": "signal",
  "class_name": "traffic_light",
  "score": 0.12,
  "bbox": [1.0, 2.0, 3.0, 4.0],
  "reason": "teacher_score_below_threshold"
}
```

reason은 고정 vocabulary를 쓴다.

```text
teacher_score_below_threshold
teacher_nms_suppressed
manual_audit_rejected
class_policy_rejected
invalid_bbox
nonfinite_prediction
unsupported_teacher_class
teacher_failure
```

`nonfinite_prediction`은 candidate-level bbox/score 문제에만 쓴다. 이 경우 해당 candidate를 reject하고 teacher run 자체가 정상이라면 build는 계속할 수 있다. batch/model-level tensor corruption, nonfinite logits 폭주, postprocess 전체 실패는 `teacher_failure`이며 ready manifest를 금지한다.

새 reason을 추가하려면 audit report schema에 먼저 추가한다.

## Traffic-light attribute 규칙

이 eval root는 TL attr을 평가하지 않는다.

Traffic-light box에는 compatibility placeholder를 넣을 수 있다.

```json
{
  "id": 7,
  "detection_id": 12,
  "bbox": [100.0, 120.0, 140.0, 180.0],
  "tl_bits": {"red": 0, "yellow": 0, "green": 0, "arrow": 0},
  "tl_attr_valid": 0,
  "collapse_reason": "odpseudo_tl_attr_placeholder"
}
```

금지:

- `has_tl_attr=1`로 올리지 않는다.
- TL attr AP/F1/combo accuracy를 reported metric으로 내보내지 않는다.
- `best_signal.pt` box confidence를 TL state confidence로 재해석하지 않는다.
- image color heuristic으로 red/yellow/green/arrow를 만들지 않는다.

현재 [model/engine/metrics.py](../model/engine/metrics.py)의 `summarize_pv26_metrics()`는 `traffic_light` metric block을 항상 반환한다. 구현 pass에서는 evaluator config나 report wrapper로 attr-disabled eval에서 이 block을 생략하거나 아래처럼 disabled로 표시해야 한다.

```json
{
  "traffic_light": {
    "disabled": true,
    "reason": "source_mask.tl_attr=false"
  }
}
```

TL attr은 [20_SIGNAL_ATTR_TEACHER_PLAN.md](20_SIGNAL_ATTR_TEACHER_PLAN.md)의 sidecar teacher가 생긴 뒤 별도 eval root 또는 v2 key에서만 켠다.

## Manifest 규칙

구현 pass는 deterministic manifest를 만든다.

`teacher_manifest.json` 또는 `conversion_manifest.json`은 builder provenance용이다. `final_dataset_manifest.json`은 loader가 discovered records와 path/order를 대조하는 용도다. 두 역할을 섞지 않는다.

필수 top-level fields:

```json
{
  "dataset_key": "pv26_eval_lane_val_odpseudo_v1",
  "split": "val",
  "status": "ready",
  "metric_semantics": "teacher_pseudo_agreement",
  "sample_count": 123,
  "failure_count": 0,
  "nonfinite_candidate_count": 0,
  "nonfinite_fatal_count": 0,
  "oom_count": 0,
  "teacher_checkpoints": {},
  "audit_policy": "audit_gated_v1",
  "samples": []
}
```

`status="ready"`인 release manifest는 `sample_count > 0`이어야 한다. `sample_count=0`은 schema smoke나 failed dry-run에는 가능하지만 release-ready artifact로 쓰지 않는다.

각 sample row는 loader/final-dataset 관례와 맞게 아래 필드를 가진다.

```json
{
  "final_sample_id": "sample_id",
  "source_dataset_key": "pv26_eval_lane_val_odpseudo_v1",
  "split": "val",
  "source_kind": "lane_val_odpseudo",
  "scene_path": "absolute_path_written_by_builder",
  "image_path": "absolute_path_written_by_builder",
  "det_path": "absolute_path_written_by_builder",
  "source_scene_path": "original_lane_scene_path",
  "source_image_path": "original_lane_image_path",
  "source_det_path": null,
  "teacher_run_status": "completed",
  "accepted_detection_count": 0,
  "det_file_status": "empty"
}
```

`source_kind="lane_val_odpseudo"`는 eval-only provenance metadata다. 기존 `tools/od_bootstrap/build/final_dataset.py`가 이 kind를 제한하면, 이 root는 별도 builder가 `final_dataset_manifest.json`을 직접 쓰거나 해당 builder를 명시적으로 확장한 뒤 사용한다.

Manifest sample order는 `(split, final_sample_id)`로 고정한다. Random order, filesystem discovery order, teacher output order는 허용하지 않는다.

## 사용 가능 판정

이 eval root를 release 가능한 v1으로 보려면 아래가 모두 필요하다.

1. base sample count와 sample id set이 기존 lane validation split과 정확히 같다.
2. ETRI, LiDAR, train/test split path가 manifest에 0개다.
3. `SOURCE_MASK_BY_DATASET`와 `DET_SUPERVISION_BY_DATASET`가 둘 다 등록되어 있다.
4. 모든 scene에 image, scene, det path가 존재한다.
5. 모든 scene의 `source.dataset`은 `pv26_eval_lane_val_odpseudo_v1`이다.
6. 모든 scene의 `source.split`은 `val`이다.
7. 모든 scene의 `tasks.has_tl_attr`는 0이다.
8. missing teacher checkpoint, OOM, nonfinite model output, image decode failure가 empty txt로 materialized되지 않는다.
9. teacher checkpoint path, model version, threshold, NMS policy가 manifest에 기록돼 있다.
10. teacher별/class별 random overlay audit bundle이 있다.
11. evaluator report가 TL attr metric을 reported metric으로 내보내지 않는다.
12. report title/metadata에 `metric_semantics="teacher_pseudo_agreement"`가 포함된다.
13. train config/dataloader가 이 source의 train record를 거부한다.

하나라도 빠지면 이 root는 실험 산출물이지 고정 eval set이 아니다.

## 필수 테스트

- `test_new_source_keys_register_source_mask_and_det_supervision_policy`
- `test_loader_accepts_empty_det_file_for_detector_supervised_new_source`
- `test_loader_rejects_missing_det_file_for_detector_supervised_new_source`
- `test_lane_val_odpseudo_preserves_base_val_sample_ids_and_count`
- `test_lane_val_odpseudo_rejects_non_val_or_foreign_source_paths`
- `test_lane_val_odpseudo_missing_checkpoint_fails_before_writing_ready_manifest`
- `test_lane_val_odpseudo_teacher_oom_does_not_materialize_empty_labels_as_success`
- `test_lane_val_odpseudo_nonfinite_teacher_output_rejected_with_reason`
- `test_lane_val_odpseudo_rejected_candidates_are_joinable_by_sample_and_teacher`
- `test_lane_val_odpseudo_empty_accepted_detections_requires_completed_teacher_status`
- `test_lane_val_odpseudo_manifest_order_is_deterministic`
- `test_lane_val_odpseudo_disables_tl_attr_metrics_in_evaluator_report`
- `test_lane_val_odpseudo_metric_report_declares_teacher_pseudo_agreement`
- `test_eval_only_odpseudo_source_rejected_from_train_split`
