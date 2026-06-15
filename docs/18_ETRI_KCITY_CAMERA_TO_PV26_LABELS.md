# ETRI KCity Camera To PV26 Labels

## 3줄 요약

- v1 입력은 `ETRI/Multi Camera Semantic Segmentation`의 KCity `leftImg`와 그 sample에 대응하는 semantic label뿐이다.
- `rightImg`, `MonoCamera`, 모든 LiDAR, camera/LiDAR fusion 산출물, 그리고 명시 매핑표로 증명되지 않은 raw class는 PV26 label로 추정 변환하지 않는다.
- 산출물은 별도 canonical scene root로 만들고 TL attribute는 raw state 검증 전까지 `has_tl_attr=0` placeholder로 둔다. 이 문서 pass에서는 runtime code, schema key, loader, evaluator를 수정하지 않는다.

## 이번 문서에서 확정한 것

| 항목 | 결정 |
| --- | --- |
| 입력 데이터 | ETRI `Multi Camera Semantic Segmentation` / KCity / `leftImg` |
| 제외 데이터 | `rightImg`, `MonoCamera`, LiDAR, camera-LiDAR fusion |
| 목적 | PV26 canonical scene 변환 계획 |
| 통합 위치 | 현재 final dataset 또는 unified validation/test set에 넣지 않음 |
| traffic-light attribute | raw label이 검증되기 전까지 항상 placeholder, `has_tl_attr=0` |
| raw class mapping | 원시 클래스 인벤토리와 명시 매핑표 없이는 어떤 클래스도 추정 매핑하지 않음 |

ETRI KCity camera 변환은 [19_LANE_VAL_OD_TEACHER_EVALSET.md](19_LANE_VAL_OD_TEACHER_EVALSET.md)의 lane validation OD pseudo eval set과 분리한다. 두 데이터셋을 같은 root, 같은 manifest, 같은 source key로 섞으면 안 된다.

## 현재 코드 기준 제약

현재 loader는 [common/pv26_schema.py](../common/pv26_schema.py)의 `SOURCE_MASK_BY_DATASET`에 등록된 `source.dataset`만 받는다. [model/data/dataset.py](../model/data/dataset.py)는 scene을 읽을 때 다음 조건을 강제한다.

- `source.dataset`은 비어 있으면 안 되고 `SOURCE_MASK_BY_DATASET`에 있어야 한다.
- `source.split`이 있으면 `labels_scene/<split>/` 디렉터리명과 같아야 한다.
- `image.file_name`은 경로가 아니라 basename이어야 한다.
- `image.height`, `image.width`는 양의 정수여야 한다.
- `lanes`, `stop_lines`, `crosswalks`가 있으면 list여야 한다.
- `labels_det` row는 YOLO `class_id cx cy w h` 5-column 형식이고 좌표는 normalized 값이어야 한다.

따라서 구현 pass는 먼저 schema에 source key를 추가해야 한다. 등록할 key는 아래 하나로 고정한다.

```text
etri_kcity_multicamera_leftimg
```

등록할 task mask는 아래처럼 고정한다.

```python
{
    "det": True,
    "tl_attr": False,
    "lane": True,
    "stop_line": True,
    "crosswalk": True,
}
```

`det=True`로 등록한 뒤에는 final dataset publication 경로에서 scene마다 `labels_det/<split>/<sample_id>.txt` 파일이 필요하다. accepted detection이 0개인 scene도 파일을 생략하지 말고 빈 txt 파일을 만든다. 빈 파일은 "detector annotation을 확인했고 accepted object가 0개"라는 뜻이다.

## 입력 선택 규칙

ETRI v1 converter는 아래 조건을 모두 만족하는 sample만 처리한다.

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

## 출력 root

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
```

이 root는 current final dataset root인 `seg_dataset/pv26_exhaustive_od_lane_dataset/`와 다르다. 구현자가 final dataset에 합치려면 별도 계획과 별도 audit가 필요하다.

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

ETRI v1에서 TL attribute는 아래 조건을 모두 만족할 때만 유효 label로 승격한다.

1. raw annotation 안에 red/yellow/green/arrow 상태가 구조화된 field로 존재한다.
2. field 의미가 dataset 문서 또는 raw sample audit로 확인된다.
3. AIHUB traffic policy와 동등한 collapse rule을 정의한다.
4. `tl_attr_valid=1` sample overlay를 수동 audit한다.

위 조건을 하나라도 만족하지 못하면 모든 traffic-light row는 placeholder다.

```json
{
  "detection_id": 0,
  "bbox": [100.0, 120.0, 140.0, 180.0],
  "tl_bits": {"red": 0, "yellow": 0, "green": 0, "arrow": 0},
  "tl_attr_valid": 0,
  "collapse_reason": "etri_tl_attr_unlabeled"
}
```

placeholder row가 하나라도 있더라도 scene task는 `has_tl_attr=0`이다.

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

1. `leftImg`만 들어간 dry run manifest가 있다.
2. `raw_class_inventory.json`과 `class_mapping_table.json`이 있다.
3. mapped/held count summary가 있다.
4. OD/lane/stop-line/crosswalk/held overlay를 random sample로 검토했다.
5. source key 등록 후 canonical dataset loader가 전체 scene을 읽는다.
6. `rightImg`, `MonoCamera`, LiDAR path가 output manifest에 0개임을 grep 또는 manifest audit로 증명했다.

하나라도 빠지면 ETRI는 "준비 중 source"이지 PV26 train/eval source가 아니다.
