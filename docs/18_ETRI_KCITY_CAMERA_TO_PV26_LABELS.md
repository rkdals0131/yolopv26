# ETRI KCity Camera To PV26 Labels

## 3줄 요약

- v1 입력은 `ETRI/Multi Camera Semantic Segmentation`의 KCity `leftImg`와 그 sample에 대응하는 semantic label뿐이다.
- 첫 구현 milestone은 label materialization이 아니라 `leftImg` pairing, split, raw class inventory, exclusion reason을 증명하는 dry-run이다.
- PV26 label materialization은 source key, det supervision policy, `tasks.has_*` 의미, manifest failure contract, semantic-to-label 알고리즘이 닫힌 뒤에만 한다.

## 이번 문서에서 확정한 것

| 항목 | 결정 |
| --- | --- |
| 입력 데이터 | ETRI `Multi Camera Semantic Segmentation` / KCity / `leftImg` |
| 제외 데이터 | `rightImg`, `MonoCamera`, LiDAR, camera-LiDAR fusion |
| v1 source key | `etri_kcity_multicamera_leftimg` |
| 목적 | 별도 canonical scene root 생성. 현 final dataset root에는 섞지 않음 |
| 첫 산출물 | dry-run inventory + exclusion manifest |
| traffic-light attribute | raw state 검증 전까지 placeholder, `tl_attr_valid=0`, `has_tl_attr=0` |
| raw class mapping | 원시 클래스 인벤토리와 명시 매핑표 없이는 어떤 클래스도 추정 매핑하지 않음 |

ETRI KCity camera 변환은 [19_LANE_VAL_OD_TEACHER_EVALSET.md](19_LANE_VAL_OD_TEACHER_EVALSET.md)의 lane validation OD pseudo eval set과 분리한다. 두 데이터셋을 같은 root, 같은 manifest, 같은 source key로 섞으면 안 된다.

## Runtime 등록 계약

현재 loader는 [common/pv26_schema.py](../common/pv26_schema.py)의 `SOURCE_MASK_BY_DATASET`에 등록된 `source.dataset`만 받는다. 또한 `det=True` source는 같은 파일의 `DET_SUPERVISION_BY_DATASET`에도 등록되어야 한다. `SOURCE_MASK_BY_DATASET`만 추가하면 indexing 뒤 `dataset[0]` 또는 target encoding 단계에서 실패한다.

구현 pass에서 등록할 key는 아래 하나로 고정한다.

```text
etri_kcity_multicamera_leftimg
```

등록할 task mask:

```python
SOURCE_MASK_BY_DATASET["etri_kcity_multicamera_leftimg"] = {
    "det": True,
    "tl_attr": False,
    "lane": True,
    "stop_line": True,
    "crosswalk": True,
}
```

등록할 detector supervision policy:

```python
DET_SUPERVISION_BY_DATASET["etri_kcity_multicamera_leftimg"] = {
    "class_names": OD_CLASSES,
    "allow_objectness_negatives": False,
    "allow_unmatched_class_negatives": False,
}
```

초기에는 `allow_objectness_negatives=False`를 유지한다. semantic mapping gap, missed component, unsafe class hold를 "진짜 배경 negative"로 학습시키면 안 된다.

`DATASET_GROUP_BY_KEY`에는 바로 넣지 않는다. train 사용을 승인하기 전까지 이 source는 standalone loader/eval 검증 대상으로만 둔다.

## Scene task flag 의미

현재 코드 관례에서 `tasks.has_*`는 "positive row가 존재한다"에 가깝다. 반면 새 builder에는 "audited/materialized completed" 상태도 필요하다. 두 의미를 섞지 않는다.

Scene `tasks`는 positive-content flag로 둔다.

```json
{
  "has_det": 0,
  "has_lane": 1,
  "has_stop_line": 0,
  "has_crosswalk": 1,
  "has_tl_attr": 0
}
```

감사 또는 파일 생성 완료 여부는 scene task flag가 아니라 manifest에 둔다.

```json
{
  "det_materialization_status": "completed",
  "det_file_status": "empty",
  "geometry_materialization_status": "completed"
}
```

`det=True` source이므로 accepted detection이 0개인 scene도 `labels_det/<split>/<sample_id>.txt` 파일을 만든다. 빈 파일은 "detector annotation 생성을 정상 완료했고 accepted object가 0개"라는 뜻이다. 파일이 없는 상태는 "0 detections"가 아니라 build failure다.

주의: `tools/od_bootstrap/build/image_list.py`와 `teacher_dataset.py`의 legacy validation은 `tasks.has_det=0`인데 det file이 있으면 stale det로 볼 수 있다. 이 root를 해당 helper에 넣기 전에는 helper를 source-mask 기반으로 고치거나 별도 builder를 써야 한다.

## 입력 선택 규칙

ETRI v1 dry-run은 아래 조건을 모두 만족하는 sample만 inventory에 올린다.

1. raw image path가 KCity `leftImg` 계열이다.
2. 같은 sample을 설명하는 semantic label이 존재한다.
3. image size를 실제 파일 또는 raw metadata에서 확인한다.
4. split이 명확하다. raw split이 없으면 converter config에서 `train`, `val`, `test` 중 하나로 명시해야 한다.

아래 조건 중 하나라도 참이면 sample 전체를 v1 출력에서 제외하고 manifest에 제외 사유를 남긴다.

- path가 `rightImg` 또는 `MonoCamera`다.
- LiDAR annotation 또는 LiDAR-derived label이다.
- image와 label의 sample id가 일치하지 않는다.
- image size를 확인할 수 없다.
- split이 비어 있거나 converter config에서 결정되지 않았다.

## 출력 root와 manifest 역할

ETRI v1 산출물 root는 별도 root로 둔다.

```text
seg_dataset/pv26_etri_kcity_leftimg/
  images/<split>/
  labels_scene/<split>/
  labels_det/<split>/
  meta/
    raw_class_inventory.json
    class_mapping_table.json
    held_label_report.json
    conversion_manifest.json
    final_dataset_manifest.json
```

`conversion_manifest.json`은 converter provenance, dry-run, exclusion/failure 요약용이다. `final_dataset_manifest.json`은 [model/data/dataset.py](../model/data/dataset.py)의 loader order/path validation용이다. 두 파일을 같은 의미로 쓰지 않는다.

`status="ready"` manifest는 아래 조건일 때만 쓴다.

```json
{
  "dataset_key": "etri_kcity_multicamera_leftimg",
  "status": "ready",
  "sample_count": 123,
  "failure_count": 0,
  "raw_scan_ignored_count_by_reason": {
    "rightImg": 123,
    "MonoCamera": 456,
    "lidar_annotation": 10
  },
  "candidate_excluded_count_by_reason": {
    "missing_semantic_label": 0,
    "image_label_sample_id_mismatch": 0,
    "invalid_split": 0
  },
  "held_count_by_reason": {},
  "nonfinite_count": 0,
  "oom_count": 0,
  "samples": []
}
```

`raw_scan_ignored_count_by_reason`은 v1 scope 밖인 `rightImg`, `MonoCamera`, LiDAR 계열을 정상적으로 무시했다는 audit count다. 이 count가 0일 필요는 없다. 반면 `candidate_excluded_count_by_reason`은 `leftImg` candidate가 pairing/split/metadata 문제로 빠진 release-blocking count이며, ready artifact에서는 0이어야 한다.

`sample_count=0`, image decode failure, nonfinite geometry, invalid split, missing semantic label, output write failure가 있으면 release artifact가 아니다.

## OD class mapping 원칙

PV26 detector class는 [common/pv26_schema.py](../common/pv26_schema.py)의 `OD_CLASSES` 7개뿐이다.

```text
vehicle, bike, pedestrian, traffic_cone, obstacle, traffic_light, sign
```

ETRI raw class를 PV26 class로 쓰려면 `meta/class_mapping_table.json`에 raw class별 row가 있어야 한다. row가 없으면 무조건 `held_annotations`로 보낸다. 추정 매핑은 금지한다.

매핑표 row 형식은 최소한 아래 필드를 가진다.

```json
{
  "raw_class": "raw_label_name",
  "target": "vehicle",
  "task": "det",
  "decision": "map",
  "reason": "raw class is an explicit vehicle object"
}
```

보류 row는 아래처럼 쓴다.

```json
{
  "raw_class": "raw_label_name",
  "target": null,
  "task": null,
  "decision": "hold",
  "reason": "ambiguous_between_bike_vehicle_pedestrian"
}
```

금지 규칙:

- 사람처럼 보인다는 이유만으로 rider bundle을 `pedestrian`으로 쪼개지 않는다.
- two-wheel 객체가 motor vehicle인지 bicycle인지 불명확하면 `bike`나 `vehicle`로 넣지 않는다.
- road surface paint나 drivable-area class를 `obstacle`로 넣지 않는다.
- sign/light/cone 여부가 raw class에서 명확하지 않은 small infrastructure class는 매핑하지 않는다.
- traffic-light color나 arrow state를 image color heuristic으로 만들지 않는다.

## Semantic label materialization gate

ETRI raw label 형식이 semantic-only mask인지, instance id가 있는 mask인지, polygon/JSON인지 먼저 확인한다. 이 확인 전에는 `labels_det`, `lanes`, `stop_lines`, `crosswalks`를 만들지 않는다.

semantic-only mask에서 OD bbox를 만들 경우 아래 정책을 문서와 코드에 먼저 고정한다.

- class union bbox를 한 개 만드는 것은 금지한다.
- connected component bbox를 쓸지, instance id를 쓸지, raw polygon을 쓸지 명시한다.
- min area, min side, max aspect ratio, clipping, occlusion/merge handling, NaN/Inf rejection 기준을 둔다.
- invalid bbox는 `labels_det`에 쓰지 않고 `held_annotations`와 `held_label_report.json`에 남긴다.

Lane/stop-line/crosswalk도 마찬가지다. skeletonization, contour extraction, point resampling, min length/area, visibility 기본값을 정하기 전에는 raw geometry를 PV26 label로 승격하지 않는다.

## Lane, Stop-Line, Crosswalk mapping

Lane family mapping은 [tools/od_bootstrap/source/aihub/lane_worker.py](../tools/od_bootstrap/source/aihub/lane_worker.py)의 현재 관례를 따른다.

Lane:

- `class_name`은 `white_lane`, `yellow_lane`, `blue_lane` 중 하나만 허용한다.
- raw color가 white/yellow/blue 중 하나로 확정되지 않으면 `held_annotations`로 보낸다.
- `source_style`은 `solid`, `dotted`, 또는 `null`만 허용한다.
- `points`는 2개 이상이어야 한다.
- visibility가 raw에 있으면 보존하고, 없으면 `visibility_source="pseudo"`로 전부 visible 처리한다.

Stop-line:

- `stop_lines[]` item은 `points` 2개 이상이어야 한다.
- `p1`은 첫 점, `p2`는 마지막 점으로 둔다.
- 점이 2개 미만이면 `held_annotations` reason은 `stop_line_requires_two_points`다.

Crosswalk:

- `crosswalks[]` item은 polygon point 3개 이상이어야 한다.
- 점이 3개 미만이면 `held_annotations` reason은 `crosswalk_requires_three_points`다.

## Traffic-Light attribute 정책

ETRI raw label이 traffic-light box를 명시하고 bbox가 유효하면 detector class `traffic_light` row로 넣는다. bbox가 유효하지 않으면 `held_annotations`에 `traffic_light_invalid_bbox`로 남긴다. Traffic-light state는 box와 별도 label이다.

`traffic_lights[].detection_id`는 반드시 최종 `labels_det` row index를 가리킨다. scene `detections[]`, `labels_det`, `traffic_lights[].detection_id`는 같은 accepted detection list에서 한 번에 생성해야 한다. row order를 따로 sort하거나 class별 concat하면 silent mismatch가 생긴다.

ETRI v1에서 TL attribute는 아래 조건을 모두 만족할 때만 유효 label로 승격한다.

1. raw annotation 안에 red/yellow/green/arrow 상태가 구조화된 field로 존재한다.
2. field 의미가 dataset 문서 또는 raw sample audit로 확인된다.
3. AIHUB traffic policy와 동등한 collapse rule을 정의한다.
4. `tl_attr_valid=1` sample overlay를 수동 audit한다.

위 조건을 하나라도 만족하지 못하면 traffic-light row는 placeholder다.

```json
{
  "detection_id": 0,
  "bbox": [100.0, 120.0, 140.0, 180.0],
  "tl_bits": {"red": 0, "yellow": 0, "green": 0, "arrow": 0},
  "tl_attr_valid": 0,
  "collapse_reason": "etri_tl_attr_unlabeled"
}
```

placeholder row가 있더라도 scene task는 `has_tl_attr=0`이다.

## held_annotations 규칙

ETRI converter는 안전하지 않은 것을 삭제하지 않는다. 매핑하지 않은 raw annotation은 scene의 `held_annotations`에 남긴다.

최소 형식:

```json
{
  "raw_class": "raw_label_name",
  "reason": "unmapped_or_unsafe_etri_class",
  "raw_attributes": {},
  "raw_geometry_type": "polygon"
}
```

`held_label_report.json`에는 reason별 count와 raw_class별 count를 둘 다 기록한다. audit에서 held count가 0이라고 가정하면 안 된다.

## 사용 가능 판정

ETRI source를 학습 또는 평가에 쓰려면 아래가 모두 통과해야 한다.

1. `leftImg`만 들어간 dry-run manifest가 있다.
2. `raw_class_inventory.json`과 `class_mapping_table.json`이 있다.
3. semantic-to-bbox/vector materialization policy가 문서화되어 있다.
4. mapped/held count summary가 있다.
5. OD/lane/stop-line/crosswalk/held overlay를 random sample로 검토했다.
6. `SOURCE_MASK_BY_DATASET`와 `DET_SUPERVISION_BY_DATASET`가 둘 다 등록되어 있다.
7. canonical dataset loader가 전체 scene을 `dataset[0]`까지 읽는다.
8. every `det=True` scene has `labels_det/<split>/<sample_id>.txt`, including empty files.
9. `rightImg`, `MonoCamera`, LiDAR path가 output manifest에 0개임을 manifest audit로 증명했다.

하나라도 빠지면 ETRI는 "준비 중 source"이지 PV26 train/eval source가 아니다.

## 필수 테스트

- `test_new_source_keys_register_source_mask_and_det_supervision_policy`
- `test_etri_dry_run_includes_only_leftimg_paths`
- `test_etri_converter_requires_semantic_label_pair`
- `test_etri_converter_rejects_image_label_sample_id_mismatch`
- `test_etri_unmapped_raw_class_goes_to_held_annotations_and_report`
- `test_etri_invalid_bbox_goes_to_held_not_labels_det`
- `test_etri_tl_attr_placeholder_never_sets_has_tl_attr`
- `test_etri_writes_empty_det_file_when_no_accepted_detection`
- `test_etri_materialization_fails_release_on_zero_samples`
- `test_loader_rejects_tl_placeholder_detection_id_not_matching_labels_det_order`
