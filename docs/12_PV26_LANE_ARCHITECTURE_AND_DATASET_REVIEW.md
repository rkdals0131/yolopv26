# PV26 Lane Dataset And Architecture Review

## Scope

- 이 문서는 `pv26`의 lane family만 다룬다.
- inspected dataset root는 `seg_dataset/pv26_exhaustive_od_lane_dataset`이다.
- 이 root에서 `det` label이 보이지 않는 점은 이번 리뷰의 문제로 취급하지 않는다.
  - 사용자 전제: `det`은 다른 환경에서 이미 학습 및 정상 판정 완료.
- 따라서 아래 평가는 `lane`, `stop_line`, `crosswalk` 데이터와 `lane family head` 구조에 집중한다.

## Executive Summary

- 현재 lane dataset root는 `aihub_lane_seoul` 57,700 샘플로 구성되어 있다.
- lane raw label은 충분히 구조화되어 있지만, 학습 직전 인코딩 과정에서 `visibility`가 사실상 버려진다.
- 현재 `yolopv26`는 `yolo26s.pt`의 backbone + neck를 trunk로 재사용하고, detect head와 lane family head를 별도로 붙인다.
- `YOLO` backbone을 쓰는 방향 자체는 맞다.
- 하지만 현재 lane family head는 spatial feature map을 직접 디코딩하지 않고, multi-scale feature를 global average pooling한 뒤 MLP query head로 lane을 회귀한다.
- 이 구조는 긴 선형 구조물인 lane에 불리하다.
- 특히 현재 구현은
  - spatial layout 손실,
  - 고정 query 수 상한,
  - raw visibility 손실,
  - lane count overflow 시 GT truncation
  문제를 가진다.
- 결론적으로:
  - `YOLO trunk reuse`는 맞는 방향
  - `현재 lane head 설계`는 lane-friendly하다고 보기 어렵다

## 1. Current Lane Dataset Status

### 1.1 Inspected root

- dataset root: `seg_dataset/pv26_exhaustive_od_lane_dataset`
- manifest: `seg_dataset/pv26_exhaustive_od_lane_dataset/meta/final_dataset_summary.json`
- source dataset key: `aihub_lane_seoul`

### 1.2 Sample counts

| Item | Count |
| --- | ---: |
| Total samples | 57,700 |
| Train | 30,000 |
| Val | 27,700 |
| Source datasets present | 1 |

Observed manifest summary:

```json
{
  "sample_count": 57700,
  "dataset_counts": {
    "aihub_lane_seoul": 57700
  }
}
```

### 1.3 Geometry label counts

| Geometry | Total count | Samples containing at least one |
| --- | ---: | ---: |
| Lanes | 238,180 | 57,415 |
| Stop lines | 6,188 | 5,255 |
| Crosswalks | 9,350 | 6,617 |

Per-split geometry counts:

| Split | Lanes | Stop lines | Crosswalks |
| --- | ---: | ---: | ---: |
| Train | 111,922 | 2,858 | 4,144 |
| Val | 126,258 | 3,330 | 5,206 |

### 1.4 Lane count distribution per image

| Metric | Lanes | Stop lines | Crosswalks |
| --- | ---: | ---: | ---: |
| Mean | 4.1279 | 0.1072 | 0.1620 |
| Median | 4 | 0 | 0 |
| P90 | 7 | 0 | 1 |
| P95 | 7 | 1 | 1 |
| Max | 20 | 5 | 5 |
| Zero-count samples | 285 | 52,445 | 51,083 |

Interpretation:

- lane은 대부분 이미지에 존재한다.
- stop_line / crosswalk는 sparse label이다.
- lane count는 평균 4.13개지만, 최대 20개까지 등장한다.

### 1.5 Lane class distribution

Current schema lane class set:

- `white_lane`
- `yellow_lane`
- `blue_lane`

Observed counts:

| Lane class | Count | Ratio |
| --- | ---: | ---: |
| `white_lane` | 198,977 | 83.54% |
| `yellow_lane` | 35,518 | 14.91% |
| `blue_lane` | 3,685 | 1.55% |

Observation:

- `white_lane`가 매우 우세하다.
- `blue_lane`는 극소수 클래스다.

### 1.6 Lane type distribution

Current schema lane type set:

- `solid`
- `dotted`

Observed counts:

| Lane type | Count | Ratio |
| --- | ---: | ---: |
| `solid` | 114,205 | 47.95% |
| `dotted` | 123,975 | 52.05% |

Class and type joint distribution:

| Class + Type | Count |
| --- | ---: |
| `white_lane | solid` | 79,728 |
| `white_lane | dotted` | 119,249 |
| `yellow_lane | solid` | 32,757 |
| `yellow_lane | dotted` | 2,761 |
| `blue_lane | solid` | 1,720 |
| `blue_lane | dotted` | 1,965 |

Observation:

- `yellow_lane`는 `solid` 편향이 강하다.
- `white_lane`는 `dotted` 비중이 높다.
- `blue_lane`는 수가 매우 적어서 class imbalance 대응이 필요할 수 있다.
- **user addition**: `blue_lane`은 버스전용차로라서 현재 있는 데이터셋으로 조금만 학습해도 어차피 이 모델이 사용될 곳에서는 크게 영향을 미치지 않으므로 중요성이 매우 낮다. 해당 문제는 무시한다. 

### 1.7 Query-cap overflow

Current model contract:

- lane query count: `12`
- stop-line query count: `6`
- crosswalk query count: `4`

Observed overflow:

| Condition | Overflow sample count |
| --- | ---: |
| lane GT count > 12 | 109 |
| stop-line GT count > 6 | 0 |
| crosswalk GT count > 4 | 14 |

Max GT count per image:

| Geometry | Max GT count |
| --- | ---: |
| Lanes | 20 |
| Stop lines | 5 |
| Crosswalks | 5 |

Important implication:

- 현재 lane encoder는 `sorted_rows[:LANE_QUERY_COUNT]`만 사용하므로 query cap을 넘는 lane GT는 학습 시 잘린다.
- 즉 일부 장면에서는 GT 전체가 supervision에 반영되지 않는다.

## 2. Lane Label Shape And Example

### 2.1 Raw scene JSON shape

원본 lane label은 `labels_scene/<split>/*.json` 안에 들어 있으며, 각 lane item은 대체로 아래 필드를 가진다.

- `class_name`
- `id`
- `meta.dataset_label`
- `meta.raw_color`
- `meta.raw_type`
- `points`
- `source_style`
- `visibility`

Actual example:

```json
{
  "class_name": "yellow_lane",
  "id": 0,
  "meta": {
    "dataset_label": "traffic_lane",
    "raw_color": "yellow",
    "raw_type": "solid"
  },
  "points": [
    [833.0, 449.0],
    [755.0, 383.0]
  ],
  "source_style": "solid",
  "visibility": [1, 1]
}
```

Raw geometry scene also contains:

- `lanes`
- `stop_lines`
- `crosswalks`
- `detections`

Example inspected sample had:

- `detections: []`
- non-empty `lanes`
- non-empty `crosswalks`

This document does not treat the empty `detections` field as a problem.

### 2.2 Training-time encoded lane target

현재 학습 타깃은 raw JSON을 그대로 쓰지 않고, fixed-size vector로 변환한다.

Lane vector dimension:

- objectness: `1`
- lane color one-hot: `3`
- lane type one-hot: `2`
- 16-point polyline: `32`
- 16-point visibility slots: `16`
- total: `54`

Encoded layout:

| Slice | Meaning |
| --- | --- |
| `[0]` | lane objectness |
| `[1:4]` | lane color one-hot |
| `[4:6]` | lane type one-hot |
| `[6:38]` | 16 x 2 resampled polyline |
| `[38:54]` | 16-point visibility |

Current contract:

- lane count는 최대 `12`
- 각 lane은 16-point polyline으로 resampled
- point order는 `y` descending으로 정렬 후 resample

This is important:

- raw label의 `visibility`는 존재한다.
- 하지만 현재 loader는 lane row에 `visibility`를 싣지 않는다.
- encoder는 visibility slice를 raw label에서 읽지 않고, valid row에 대해 `[38:54] = 1.0`으로 채운다.

즉, 현재 visibility supervision은 실질적으로 `all visible`이다.

Practical meaning:

- 가려진 lane point와 실제 visible point를 구분하는 supervision이 없다.
- 모델 출력에는 visibility slot이 있지만, raw annotation semantics를 보존하지 못한다.

### 2.3 Current label conversion behavior

현재 lane label은 아래 과정을 거친다.

1. raw `points`를 network space로 transform
2. `class_name`을 `white/yellow/blue` index로 변환
3. `source_style` 또는 `meta.raw_type`을 `solid/dotted` index로 변환
4. lane point를 `y` descending으로 정렬
5. 16 point로 resample
6. fixed query slot에 채움

장점:

- shape가 고정되어 batch 처리와 loss 구현이 쉽다.
- polyline regression contract가 명확하다.

단점:

- raw visibility 손실
- query budget 초과 시 GT truncation
- long-range curved lane의 세부 형상 손실 가능

## 3. Current `yolo26s.pt` Trunk Architecture

### 3.1 High-level structure

현재 `yolopv26`는 `Ultralytics YOLO v26` pretrained model을 불러와:

- official `backbone`
- official `neck`

를 trunk로 재사용한다.

구체적으로는:

- layer `0..22`: trunk
- layer `23`: original detect head

그리고 PV26는 detect head 직전의 multi-scale feature source를 그대로 가져온다.

Current detected source indices:

- `[16, 19, 22]`

These are the three pyramid levels used by the original YOLO detect head.

### 3.2 Inspected layer graph of local `yolo26s.pt`

Local model inspection result:

| Layer index | Op | From | Role |
| --- | --- | --- | --- |
| 0 | `Conv` | `-1` | stem |
| 1 | `Conv` | `-1` | downsample |
| 2 | `C3k2` | `-1` | backbone block |
| 3 | `Conv` | `-1` | downsample |
| 4 | `C3k2` | `-1` | backbone block |
| 5 | `Conv` | `-1` | downsample |
| 6 | `C3k2` | `-1` | backbone block |
| 7 | `Conv` | `-1` | downsample |
| 8 | `C3k2` | `-1` | backbone block |
| 9 | `SPPF` | `-1` | receptive field expansion |
| 10 | `C2PSA` | `-1` | high-level context block |
| 11 | `Upsample` | `-1` | FPN top-down |
| 12 | `Concat` | `[-1, 6]` | fuse with mid feature |
| 13 | `C3k2` | `-1` | fused neck block |
| 14 | `Upsample` | `-1` | FPN top-down |
| 15 | `Concat` | `[-1, 4]` | fuse with shallow feature |
| 16 | `C3k2` | `-1` | P3 output source |
| 17 | `Conv` | `-1` | PAN downsample |
| 18 | `Concat` | `[-1, 13]` | PAN fuse |
| 19 | `C3k2` | `-1` | P4 output source |
| 20 | `Conv` | `-1` | PAN downsample |
| 21 | `Concat` | `[-1, 10]` | PAN fuse |
| 22 | `C3k2` | `-1` | P5 output source |
| 23 | `Detect` | `[16, 19, 22]` | original YOLO detect head |

Interpretation:

- layer `0..10`은 backbone + high-level context block
- layer `11..22`는 FPN/PAN style neck
- layer `16/19/22`가 P3/P4/P5 feature source

### 3.3 Feature shapes used by PV26

At input size `608 x 800`, inspected pyramid shapes are:

| Source layer | Level | Shape |
| --- | --- | --- |
| 16 | P3 | `[B, 128, 76, 100]` |
| 19 | P4 | `[B, 256, 38, 50]` |
| 22 | P5 | `[B, 512, 19, 25]` |

So the current `yolo26s.pt` trunk contract is:

- P3: `128 x 76 x 100`
- P4: `256 x 38 x 50`
- P5: `512 x 19 x 25`

### 3.4 How PV26 reuses it

Current PV26 flow is:

```text
input image
  -> YOLO26s stem/backbone/neck trunk
  -> feature maps from layers 16 / 19 / 22
  -> det heads on each scale
  -> tl_attr heads on each scale
  -> lane family head on pooled multi-scale embedding
```

This means:

- detector / tl_attr path is spatial
- lane family path is not truly spatial after pooling

## 4. Current PV26 Lane Family Architecture

### 4.1 Current head split

현재 `PV26Heads`는 두 종류의 head를 가진다.

1. spatial scale heads
   - `det_heads`
   - `tl_attr_heads`
2. pooled query MLP heads
   - `lane_head`
   - `stop_line_head`
   - `crosswalk_head`

Current implementation conceptually:

```text
P3, P4, P5 features
  -> det head per scale
  -> tl_attr head per scale
  -> global average pool each scale
  -> concatenate pooled vectors
  -> MLP
  -> fixed number of lane / stop_line / crosswalk queries
```

### 4.2 Why this is attractive

- 구현이 단순하다.
- output tensor shape가 고정이라 loss와 export가 쉽다.
- detector trunk reuse와 학습 stage 분리가 간단하다.

### 4.3 Why this is weak for lanes

lane은 object box와 다르다.

lane detection needs:

- 긴 구조의 연속성
- 위치별 방향 변화
- near/far scale consistency
- occlusion-aware visibility
- thin structure localization

현재 head는 이런 성질을 직접 모델링하지 않는다.

## 5. Problematic Points In The Current Architecture

### 5.1 Spatial layout is discarded too early

현재 lane family는 각 scale feature를 `mean(H, W)`로 압축한 뒤 MLP로 예측한다.

문제:

- lane은 위치가 의미의 핵심이다.
- global pooled vector는
  - 어느 row에 lane이 있었는지,
  - 좌우 어디를 지나가는지,
  - 중간에 가려졌는지,
  - 곡률이 어떻게 변하는지
  같은 spatial signal을 강하게 잃는다.

결과:

- lane처럼 길고 얇은 구조물에는 불리하다.
- detector처럼 local spatial evidence를 쓰는 경로와 lane 경로의 inductive bias 차이가 너무 크다.

### 5.2 Fixed small query budget causes GT truncation

Current caps:

- lanes: `12`
- stop lines: `6`
- crosswalks: `4`

Observed real data:

- max lane count per image: `20`
- lane overflow samples: `109`
- crosswalk overflow samples: `14`

문제:

- query cap을 넘는 GT는 학습에 반영되지 않는다.
- scene complexity가 큰 장면에서 recall ceiling이 생긴다.

### 5.3 Raw visibility annotation is not preserved

Raw lane label has:

- `visibility`

But current path does:

- loader: points, color, type만 유지
- encoder: visibility slice를 전부 `1.0`으로 채움

문제:

- occlusion reasoning이 사라진다.
- visibility output dimension이 의미 없는 채널이 되기 쉽다.
- “가려져서 안 보이는 lane”과 “없는데 hallucination한 lane”을 구분하는 supervision이 약하다.

### 5.4 Polyline encoding is compressed into a single query vector

Current contract:

- one lane = one 54-d vector

문제:

- row-wise evidence accumulation이 없다.
- local-to-global correction 경로가 없다.
- long lane, heavily curved lane, partially visible lane에서 회귀 난도가 높다.

### 5.5 Lane family fine-tune stage cannot fix the main bottleneck

Current training stages include late `lane_family_only` fine-tuning.

이건 좋은 실험 장치이지만, lane path의 핵심 병목이 `pooled embedding bottleneck`이라면:

- stage 4는 학습 정책 보정일 뿐
- 구조적 spatial bottleneck 자체는 해결하지 못한다

즉, fine-tune policy는 보조 수단이지 핵심 해법이 아니다.

## 6. What Is Lane-Friendly?

### 6.1 Core definition

`lane-friendly`한 구조는 lane을 object box처럼 다루지 않고, 다음 성질을 보존한다.

- high-resolution spatial evidence
- row-wise continuity
- multi-scale fusion
- local geometry + global context 동시 사용
- visibility / occlusion supervision 보존
- variable lane count 또는 충분한 query budget

### 6.2 Minimum conditions for a lane-friendly head

최소한 아래는 있어야 한다.

1. lane head가 spatial feature map 위에서 동작해야 한다.
2. lane point 또는 row anchor 예측이 image row와 직접 연결되어야 한다.
3. visibility가 raw annotation semantics를 유지해야 한다.
4. lane 수 상한이 데이터 분포보다 너무 낮지 않아야 한다.

### 6.3 Typical lane-friendly design patterns

#### A. Row-wise / anchor-based lane head

예시:

- 각 image row 또는 anchor row에서 lane x-position 예측
- lane existence + type + color + visibility 동시 예측

장점:

- lane의 vertical continuity와 잘 맞는다.
- 구현이 비교적 단순하다.
- long thin structure에 강하다.

#### B. Spatial token + transformer decoder

예시:

- P3/P4/P5 feature를 token화
- positional encoding 유지
- learned query가 spatial token에 cross-attention

장점:

- query-based 설계를 유지하면서도 spatial evidence를 잃지 않는다.
- 현재 fixed-query 설계를 완전히 버리지 않고 업그레이드 가능하다.

주의:

- 그냥 pooled embedding에 MLP를 붙이는 것과는 다르다.
- 핵심은 query가 spatial map과 상호작용해야 한다는 점이다.

#### C. High-resolution segmentation / lane mask branch

예시:

- lane heatmap 또는 instance-like lane mask
- 이후 polyline fitting/postprocess

장점:

- lane의 thin geometry를 직접 spatially supervised 가능
- occlusion과 continuity를 dense하게 배울 수 있음

주의:

- 현재 export/runtime contract와 차이가 커질 수 있다.

### 6.4 What is lane-friendly for this repository specifically?

현재 repo 제약을 감안하면 가장 현실적인 lane-friendly 방향은 아래다.

#### Recommended practical path

1. `yolo26s.pt` trunk는 유지
2. lane family head만 교체
3. P3 중심의 high-resolution feature를 직접 사용
4. P4/P5는 context 보조로 사용
5. global average pooling-only path는 제거
6. query-based를 유지하더라도 spatial cross-attention 기반으로 변경
7. raw `visibility`를 실제 supervision으로 연결
8. lane query count를 최소 `16` 이상으로 재검토하거나 dynamic cap 도입

#### Minimum-change option

가장 작은 변경으로 lane-friendly 쪽으로 가려면:

- trunk는 그대로 둔다
- lane head만 spatial decoder로 교체한다
- output contract는 가능한 한 유지한다
  - lane: fixed point polyline
  - stop-line: fixed point polyline
  - crosswalk: fixed point polygon

이 접근은 runtime/export 파이프라인 충격을 줄인다.

## 7. Recommended Architecture Direction

### 7.1 Short-term recommendation

- `YOLO26s trunk reuse` 유지
- lane family MLP head 제거
- lane family spatial decoder 도입
- raw visibility 반영
- lane query cap 재설정

### 7.2 Medium-term recommendation

- lane head를 `spatial token + query decoder` 또는 `row-anchor` 방식으로 전환
- stop-line / crosswalk는 lane과 분리된 geometry head로 유지 가능
- lane family metric을 early architecture decision 기준으로 승격

### 7.3 Explicit do-not-do

현재 상태에서 아래는 권장하지 않는다.

- pooled embedding MLP를 더 깊게 만드는 것만으로 해결하려는 접근
- late-stage fine-tune만으로 lane 품질을 해결하려는 접근
- visibility supervision 없이 visibility output만 유지하는 접근

이들은 구조적 병목을 건드리지 못한다.

## 8. Final Assessment

### 8.1 What is correct

- `YOLO` backbone + neck를 trunk로 재사용하는 방향은 맞다.
- `yolo26s.pt`를 pretrained trunk로 쓰는 선택도 합리적이다.
- detect/tl_attr와 lane family를 task별 head로 분리한 방향도 맞다.

### 8.2 What is not good enough

- 현재 lane family는 trunk feature를 lane-friendly하게 읽지 못한다.
- spatial signal이 pooling에서 과도하게 사라진다.
- raw visibility를 잃는다.
- fixed query cap이 실제 dataset 상한보다 낮다.

### 8.3 Bottom line

한 문장으로 요약하면:

> `YOLO26s trunk reuse`는 맞지만, 현재 `lane head`는 lane-friendly하지 않다.

가장 중요한 이유는:

- lane이 필요한 spatial continuity를 직접 다루지 못하기 때문이다.

## Appendix A. Code Pointers

- trunk adapter: `model/net/trunk.py`
- PV26 heads: `model/net/heads.py`
- schema: `common/pv26_schema.py`
- sample loader: `model/data/dataset.py`
- lane target encoder: `model/data/target_encoder.py`
- training stage policy: `model/engine/trainer.py`

## Appendix B. Key Facts Used In This Review

- `yolo26s.pt` local inspection:
  - total layers: `24`
  - detect head index: `23`
  - feature source indices: `[16, 19, 22]`
- feature shapes at `608x800`:
  - P3: `[1, 128, 76, 100]`
  - P4: `[1, 256, 38, 50]`
  - P5: `[1, 512, 19, 25]`
- dataset totals:
  - total samples: `57,700`
  - total lanes: `238,180`
  - total stop lines: `6,188`
  - total crosswalks: `9,350`

## Appendix C. Self-Contained `yolo26s.pt` Trunk Spec

### C.1 Layer-by-layer table at input `608 x 800`

이 표는 로컬에서 실제 `yolo26s.pt`를 로드해 forward한 결과다.

| Index | Block | From | Output shape |
| --- | --- | --- | --- |
| 0 | `Conv` | `-1` | `[1, 32, 304, 400]` |
| 1 | `Conv` | `-1` | `[1, 64, 152, 200]` |
| 2 | `C3k2` | `-1` | `[1, 128, 152, 200]` |
| 3 | `Conv` | `-1` | `[1, 128, 76, 100]` |
| 4 | `C3k2` | `-1` | `[1, 256, 76, 100]` |
| 5 | `Conv` | `-1` | `[1, 256, 38, 50]` |
| 6 | `C3k2` | `-1` | `[1, 256, 38, 50]` |
| 7 | `Conv` | `-1` | `[1, 512, 19, 25]` |
| 8 | `C3k2` | `-1` | `[1, 512, 19, 25]` |
| 9 | `SPPF` | `-1` | `[1, 512, 19, 25]` |
| 10 | `C2PSA` | `-1` | `[1, 512, 19, 25]` |
| 11 | `Upsample` | `-1` | `[1, 512, 38, 50]` |
| 12 | `Concat` | `[-1, 6]` | `[1, 768, 38, 50]` |
| 13 | `C3k2` | `-1` | `[1, 256, 38, 50]` |
| 14 | `Upsample` | `-1` | `[1, 256, 76, 100]` |
| 15 | `Concat` | `[-1, 4]` | `[1, 512, 76, 100]` |
| 16 | `C3k2` | `-1` | `[1, 128, 76, 100]` |
| 17 | `Conv` | `-1` | `[1, 128, 38, 50]` |
| 18 | `Concat` | `[-1, 13]` | `[1, 384, 38, 50]` |
| 19 | `C3k2` | `-1` | `[1, 256, 38, 50]` |
| 20 | `Conv` | `-1` | `[1, 256, 19, 25]` |
| 21 | `Concat` | `[-1, 10]` | `[1, 768, 19, 25]` |
| 22 | `C3k2` | `-1` | `[1, 512, 19, 25]` |
| 23 | `Detect` | `[16, 19, 22]` | tuple output |

### C.2 What this means structurally

이 표를 한 줄로 읽으면:

```text
image
  -> stride-2 stem
  -> backbone downsampling stages
  -> SPPF
  -> C2PSA
  -> FPN top-down fusion
  -> PAN bottom-up fusion
  -> Detect([16, 19, 22])
```

즉 `yolo26s.pt`는:

- backbone-only 모델이 아니라
- backbone + neck + detect head까지 포함된 end-to-end detector

이고, 현재 PV26는 여기서:

- original detect head는 버리고
- detect head가 원래 참조하던 `16 / 19 / 22` feature map만 재사용한다

### C.3 PV26가 실제로 가져오는 feature

PV26 trunk adapter가 사용하는 source feature는 아래 3개다.

| Source | Semantic level | Shape | Stride |
| --- | --- | --- | ---: |
| Layer 16 | P3 | `[B, 128, 76, 100]` | 8 |
| Layer 19 | P4 | `[B, 256, 38, 50]` | 16 |
| Layer 22 | P5 | `[B, 512, 19, 25]` | 32 |

이 3개가 PV26 custom head의 유일한 입력이다.

## Appendix D. Self-Contained PV26 Head Spec

### D.1 Exact module shapes

Current `PV26Heads` has two head families.

#### Spatial prediction heads

For each of `P3 / P4 / P5`:

```text
feature [B, C, H, W]
  -> Conv3x3(C -> C)
  -> BatchNorm2d(C)
  -> SiLU
  -> Conv1x1(C -> out_dim)
  -> flatten(HW)
  -> transpose
  -> [B, HW, out_dim]
```

Applied to:

- detection head with `out_dim = 12`
- traffic-light attribute head with `out_dim = 4`

#### Query MLP heads

After pooling:

```text
P3 mean pool -> [B, 128]
P4 mean pool -> [B, 256]
P5 mean pool -> [B, 512]
concat -> [B, 896]
  -> Linear(896 -> 896)
  -> SiLU
  -> Linear(896 -> query_count * vector_dim)
  -> reshape
```

Applied to:

- lane head: `12 * 54`
- stop-line head: `6 * 9`
- crosswalk head: `4 * 17`

### D.2 End-to-end tensor flow

With `yolo26s` at input `608 x 800`:

#### Detector path

```text
P3 [B,128,76,100] -> [B,7600,12]
P4 [B,256,38,50]  -> [B,1900,12]
P5 [B,512,19,25]  -> [B,475,12]
concat            -> [B,9975,12]
```

#### Traffic-light attribute path

```text
P3 [B,128,76,100] -> [B,7600,4]
P4 [B,256,38,50]  -> [B,1900,4]
P5 [B,512,19,25]  -> [B,475,4]
concat            -> [B,9975,4]
```

#### Lane family path

```text
P3 [B,128,76,100] --mean(H,W)--> [B,128]
P4 [B,256,38,50]  --mean(H,W)--> [B,256]
P5 [B,512,19,25]  --mean(H,W)--> [B,512]
concat                            [B,896]
lane MLP                          [B,12,54]
stop-line MLP                     [B,6,9]
crosswalk MLP                     [B,4,17]
```

### D.3 Why the bottleneck is obvious from the tensor flow

문서만 보고도 현재 병목이 보이도록 다시 적으면:

- detector path는 끝까지 `[H, W]`가 남아 있다
- lane family path는 head에 들어가기 전에 이미 `[B, 896]` 단일 vector가 된다

즉:

```text
detector:   spatial map -> spatial logits
lane path:  spatial map -> single pooled vector -> geometry regression
```

이 차이가 현재 구조의 핵심 문제다.

lane은:

- 어느 row에 존재하는지
- 좌우 어디를 지나는지
- 중간에 가려졌는지
- 커브가 어떻게 변하는지

가 중요한데, 현재 lane path는 head 직전에 그 정보를 대부분 평균 내버린다.

### D.4 Why this matters more for lanes than for boxes

object detection box는 한 객체당:

- center
- width
- height

중심의 compact representation이 잘 맞는다.

하지만 lane은:

- long-range polyline
- thin structure
- row-wise continuity
- multi-point visibility

를 다뤄야 해서 compact pooled vector 하나로 회귀하기 어렵다.

그래서 현재 구조는:

- box-like inductive bias에는 강하지만
- lane-like inductive bias에는 약하다

### D.5 Architecture takeaway

현재 구조를 아주 짧게 쓰면:

```text
YOLO26s trunk reuse: correct
Lane head after global pooling: weak
```

따라서 구조 수정 우선순위는:

1. trunk 교체가 아니라
2. lane family head를 spatially aware decoder로 바꾸는 것

이다.
