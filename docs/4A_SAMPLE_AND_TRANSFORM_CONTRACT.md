# Sample And Transform Contract

## 목적

- loader가 반환하는 sample dictionary를 구현 전에 고정한다.
- raw-space standardized dataset과 network-space training target의 경계를 고정한다.
- target encoder, loss, evaluator가 모두 같은 좌표계 규약을 보게 만든다.

## canonical sizes

- raw standardized dataset
  - 원본 해상도 유지
- runtime raw contract
  - `800x600`
- network input contract
  - `800x608`
- loader와 infer transform은 모두 `800x600 -> 800x608` letterbox/pad 규약을 공유한다.

## sample dictionary contract

```python
sample = {
    "image": torch.FloatTensor[3, 608, 800],
    "det_targets": {
        "boxes_xyxy": torch.FloatTensor[N_det, 4],
        "classes": torch.LongTensor[N_det],
    },
    "tl_attr_targets": {
        "bits": torch.FloatTensor[N_det, 4],
        "is_traffic_light": torch.BoolTensor[N_det],
        "collapse_reason": list[str],
    },
    "lane_targets": {
        "lanes": list[dict],
        "stop_lines": list[dict],
        "crosswalks": list[dict],
    },
    "source_mask": {
        "det": bool,
        "tl_attr": bool,
        "lane": bool,
        "stop_line": bool,
        "crosswalk": bool,
    },
    "valid_mask": {
        "det": torch.BoolTensor[N_det],
        "tl_attr": torch.BoolTensor[N_det],
        "lane": torch.BoolTensor[N_lane],
        "stop_line": torch.BoolTensor[N_stop],
        "crosswalk": torch.BoolTensor[N_cross],
    },
    "meta": {
        "sample_id": str,
        "dataset_key": str,
        "split": str,
        "image_path": str,
        "raw_hw": tuple[int, int],
        "network_hw": tuple[int, int],
        "transform": dict,
    },
}
```

## field rules

- `image`
  - dtype는 `float32`
  - shape는 항상 `[3, 608, 800]`
  - range는 `[0.0, 1.0]`
  - channel order는 `RGB`
- `det_targets`
  - 좌표계는 transformed network pixel space다.
  - format은 `xyxy`
  - `boxes_xyxy` shape는 `[N_det, 4]`
  - `classes` shape는 `[N_det]`
- `tl_attr_targets`
  - `det_targets`와 index가 1:1 정렬된다.
  - `bits[i]`는 `det_targets[i]`의 GT에 대한 TL bits다.
  - non-traffic-light row는 `bits[i] = [0, 0, 0, 0]`로 둔다.
  - 실제 loss 참여 여부는 `valid_mask["tl_attr"][i]`가 결정한다.
- `lane_targets["lanes"]`
  - 각 원소는 `{"points_xy": torch.FloatTensor[P, 2], "color": int, "lane_type": int}`다.
  - `P`는 transformed network space의 ragged polyline point 수다.
- `lane_targets["stop_lines"]`
  - 각 원소는 `{"points_xy": torch.FloatTensor[P, 2]}`다.
- `lane_targets["crosswalks"]`
  - 각 원소는 `{"points_xy": torch.FloatTensor[P, 2]}`다.
- `source_mask`
  - dataset source가 해당 task supervision을 제공하는지 나타낸다.
  - BDD100K는 `det=True`만 켠다.
  - AIHUB traffic은 `det=True`, `tl_attr=True`만 켠다.
  - AIHUB lane은 `lane=True`, `stop_line=True`, `crosswalk=True`만 켠다.
- `valid_mask`
  - source가 task를 제공해도 샘플별 invalid case는 여기서 끊는다.
  - TL invalid case 예시는 `non_car`, `x_light_active`, `multi_color_active`다.
- `meta["transform"]`
  - `scale`
  - `pad_left`
  - `pad_top`
  - `pad_right`
  - `pad_bottom`
  - `resized_hw`

## empty-case rules

- detection object가 하나도 없는 sample
  - `boxes_xyxy`는 shape `[0, 4]`
  - `classes`는 shape `[0]`
  - `valid_mask["det"]`는 shape `[0]`
- TL target이 없는 sample
  - `bits`는 shape `[0, 4]` 또는 detection row와 정렬된 zero-row다.
  - `valid_mask["tl_attr"]`는 모든 row에서 `False`
- lane family annotation이 없는 sample
  - 해당 list는 빈 list
  - 대응 `valid_mask`는 shape `[0]`
- sample dict는 empty-case에서도 키를 절대 생략하지 않는다.

## encoded batch contract

- loader 다음 단계인 target encoder는 ragged sample dict를 fixed-shape batch contract로 바꾼다.
- encoded batch는 loss spec과 정확히 일치해야 한다.

```python
encoded_batch = {
    "image": torch.FloatTensor[B, 3, 608, 800],
    "det": "... YOLO detector target tensor ...",
    "tl_attr": torch.FloatTensor[B, N_det, 4],
    "lane": torch.FloatTensor[B, 12, 54],
    "stop_line": torch.FloatTensor[B, 6, 9],
    "crosswalk": torch.FloatTensor[B, 4, 17],
    "mask": dict,
    "meta": list[dict],
}
```

## transform contract

- raw image size를 `(H_raw, W_raw)`라 둔다.
- network target size를 `(H_net, W_net) = (608, 800)`으로 둔다.
- scale은 아래로 고정한다.

```text
r = min(W_net / W_raw, H_net / H_raw)
W_resize = round(W_raw * r)
H_resize = round(H_raw * r)
pad_w = W_net - W_resize
pad_h = H_net - H_resize
pad_left = floor(pad_w / 2)
pad_right = pad_w - pad_left
pad_top = floor(pad_h / 2)
pad_bottom = pad_h - pad_top
```

- bbox transform

```text
x' = x * r + pad_left
y' = y * r + pad_top
```

- polyline/polygon point transform도 동일하다.
- inverse mapping은 아래다.

```text
x_raw = (x' - pad_left) / r
y_raw = (y' - pad_top) / r
```

## clipping and filtering rules

- transformed 좌표는 `[0, W_net - 1]`, `[0, H_net - 1]` 범위로 clip한다.
- bbox는 clip 이후 `width <= 1` 또는 `height <= 1`이면 drop한다.
- polyline/polygon은 clip 이후 unique point가 2개 미만이면 invalid 처리한다.
- invalid 처리된 원소는 삭제보다 `valid_mask=False`가 우선이다.
- dataset source가 task 자체를 제공하지 않는 경우는 `source_mask=False`가 우선이다.

## TL binding contract

- TL attr supervision은 detector assignment 결과를 재사용한다.
- detector positive 하나는 정확히 하나의 matched GT index를 갖는다.
- 그 GT index의 class가 `traffic_light`면 같은 index에서 TL bits를 읽는다.
- `valid_mask["tl_attr"][gt_idx]`가 `False`면 attr loss를 완전히 skip한다.
- inference output은 아래 구조를 기본으로 한다.

```python
prediction = {
    "box_xyxy": [x1, y1, x2, y2],
    "score": float,
    "class_id": int,
    "class_name": str,
    "tl_attr_scores": {
        "red": float,
        "yellow": float,
        "green": float,
        "arrow": float,
    },
}
```

- `class_name != "traffic_light"`면 `tl_attr_scores`는 downstream에서 무시한다.

## implementation rules

- loader는 sample contract를 만든다.
- target encoder는 encoded batch contract를 만든다.
- loss는 encoded batch contract만 읽는다.
- evaluator와 ROS/export layer는 `meta["transform"]`를 이용해 raw-space 복원을 수행한다.
