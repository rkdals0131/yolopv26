# Lane Validation OD Teacher Eval Set

## 3줄 요약

- `pv26_eval_lane_val_odpseudo_v1`은 기존 lane validation split만 base image set으로 쓰는 eval-only root다.
- OD label은 `best_mobility.pt`, `best_signal.pt`, `best_obstacle.pt`가 만든 pseudo label 중 audit를 통과한 것만 쓰며, metric 의미는 human GT 성능이 아니라 teacher-pseudo agreement다.
- TL attribute는 항상 off(`has_tl_attr=0`)이고 ETRI, LiDAR, BDD 추가 split, AIHUB raw split 직접 유입, train 용도 사용은 모두 금지한다. 이 문서 pass에서는 source key 등록, teacher 실행, evaluator 수정, manifest 생성 코드를 하지 않는다.

## 이번 문서에서 확정한 것

| 항목 | 결정 |
| --- | --- |
| base image set | 기존 lane validation split만 사용 |
| 추가 label | audit를 통과한 OD teacher pseudo label |
| 제외 source | ETRI KCity, LiDAR, BDD 추가 split, AIHUB traffic/obstacle raw split 직접 유입 |
| TL attr | 항상 off, `has_tl_attr=0` |
| OD metric 의미 | human GT 성능이 아니라 teacher-pseudo agreement |
| train 사용 | 금지. v1은 eval-only |

ETRI KCity camera 변환은 [18_ETRI_KCITY_CAMERA_TO_PV26_LABELS.md](18_ETRI_KCITY_CAMERA_TO_PV26_LABELS.md)의 별도 source다. ETRI sample은 이 eval root에 들어오면 안 된다.

## Source key

구현 pass에서 등록할 key는 아래 하나로 고정한다.

```text
pv26_eval_lane_val_odpseudo_v1
```

현재 loader는 [common/pv26_schema.py](../common/pv26_schema.py)의 `SOURCE_MASK_BY_DATASET`에 없는 key를 거부한다. 구현 pass에서는 이 key를 먼저 등록해야 한다.

등록할 task mask:

```python
{
    "det": True,
    "lane": True,
    "stop_line": True,
    "crosswalk": True,
    "tl_attr": False,
}
```

scene `tasks`는 아래 값을 따른다.

```json
{
  "has_det": 1,
  "has_lane": 1,
  "has_stop_line": 1,
  "has_crosswalk": 1,
  "has_tl_attr": 0
}
```

만약 특정 scene에 accepted OD pseudo label이 0개여도 `has_det=1`을 유지한다. 의미는 "모든 OD teacher pass와 audit를 실행했고 살아남은 box가 0개"다.

## Base sample 규칙

입력 sample은 기존 lane validation split에서만 온다.

필수 조건:

- split은 `val`이다.
- image는 기존 lane validation image다.
- lane, stop-line, crosswalk label은 기존 lane scene label에서 온다.
- sample id는 base lane validation sample id를 유지한다.

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

teacher output을 scene에 넣으려면 audit를 통과해야 한다. audit 전 raw prediction은 label이 아니라 candidate다.

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

## labels_det 파일 규칙

`det=True` source로 등록할 예정이므로 모든 scene은 대응되는 `labels_det/val/<sample_id>.txt`를 가진다.

- accepted detection이 1개 이상이면 YOLO 5-column row를 쓴다.
- accepted detection이 0개면 빈 txt 파일을 쓴다.
- 파일이 없는 상태는 "0개"가 아니라 "det label 생성 실패"다.

이 규칙은 [tools/od_bootstrap/build/final_dataset.py](../tools/od_bootstrap/build/final_dataset.py)의 final dataset publication 경로가 `det=True` source에 대해 det label file 존재를 요구하기 때문에 필요하다.

## Rejected candidate 보존 규칙

teacher raw prediction 중 accepted label이 아닌 것은 조용히 버리지 않는다. 아래 중 하나에 남긴다.

- `meta/rejected_detections.jsonl`
- scene-level `held_annotations`
- teacher별 audit report

reason은 고정 vocabulary를 쓴다.

```text
teacher_score_below_threshold
teacher_nms_suppressed
manual_audit_rejected
class_policy_rejected
invalid_bbox
unsupported_teacher_class
```

새 reason을 추가하려면 audit report schema에 먼저 추가한다.

## Traffic-light attribute 규칙

이 eval root는 TL attr을 평가하지 않는다.

Traffic-light box에는 compatibility placeholder를 넣는다.

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
- TL attr AP/F1/combo accuracy를 보고하지 않는다.
- `best_signal.pt` box confidence를 TL state confidence로 재해석하지 않는다.
- image color heuristic으로 red/yellow/green/arrow를 만들지 않는다.

TL attr은 [20_SIGNAL_ATTR_TEACHER_PLAN.md](20_SIGNAL_ATTR_TEACHER_PLAN.md)의 sidecar teacher가 생긴 뒤 별도 eval root 또는 v2 key에서만 켠다.

## Manifest 규칙

구현 pass는 deterministic manifest를 만든다.

필수 top-level fields:

```json
{
  "dataset_key": "pv26_eval_lane_val_odpseudo_v1",
  "split": "val",
  "sample_count": 0,
  "teacher_checkpoints": {},
  "audit_policy": "audit_gated_v1",
  "samples": []
}
```

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
  "source_det_path": null
}
```

Manifest sample order는 `(split, final_sample_id)`로 고정한다. Random order, filesystem discovery order, teacher output order는 허용하지 않는다.

## 사용 가능 판정

이 eval root를 release 가능한 v1으로 보려면 아래가 모두 필요하다.

1. base sample count가 기존 lane validation split count와 정확히 같다.
2. ETRI, LiDAR, train/test split path가 manifest에 0개다.
3. 모든 scene에 image, scene, det path가 존재한다.
4. 모든 scene의 `source.dataset`은 `pv26_eval_lane_val_odpseudo_v1`이다.
5. 모든 scene의 `source.split`은 `val`이다.
6. 모든 scene의 `tasks.has_tl_attr`는 0이다.
7. teacher checkpoint path, model version, threshold, NMS policy가 manifest에 기록돼 있다.
8. teacher별/class별 random overlay audit bundle이 있다.
9. evaluator smoke가 `tl_attr` metric 없이 통과한다.

하나라도 빠지면 이 root는 실험 산출물이지 고정 eval set이 아니다.
