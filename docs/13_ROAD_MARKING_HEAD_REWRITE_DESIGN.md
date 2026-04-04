# Road-Marking Head Rewrite Design

## 문서 목적

- 이 문서는 `12_PV26_LANE_ARCHITECTURE_AND_DATASET_REVIEW.md`와 `12A_GPT_REVIEW_ON_LANE_ARCH.md`를 바탕으로, 다음 구현 wave에서 적용할 `lane family -> road marking head` 개편안을 결정 완료 상태로 고정한다.
- 이 문서는 future-state 설계 문서다.
- 현재 구현 설명은 계속 [4_MODEL_ARCHITECTURE.md](4_MODEL_ARCHITECTURE.md), [5_TARGETS_AND_LOSS.md](5_TARGETS_AND_LOSS.md), [6_TRAINING_AND_EVALUATION.md](6_TRAINING_AND_EVALUATION.md), [9_EXECUTION_STATUS.md](9_EXECUTION_STATUS.md), [12_PV26_LANE_ARCHITECTURE_AND_DATASET_REVIEW.md](12_PV26_LANE_ARCHITECTURE_AND_DATASET_REVIEW.md)를 기준으로 읽는다.

## 문제 정의

- 현재 lane / stop_line / crosswalk는 모두 pooled pyramid embedding 위의 query MLP head를 사용한다.
- 이 구조는 lane에는 명백히 부적합하고, stop_line / crosswalk에도 돌아갈 수는 있지만 geometry-aware하다고 보기 어렵다.
- lane은 row-wise continuity와 visibility supervision을 직접 모델링해야 한다.
- stop_line / crosswalk는 lane처럼 row-anchor가 필요하지는 않지만, 위치 / 각도 / 길이 / 사변형 왜곡을 끝까지 spatial feature 위에서 읽어야 한다.

요약하면:

- lane: lane 전용 해법이 필요하다
- stop_line / crosswalk: road-marking geometry 전용 해법이 필요하다
- 세 task를 pooled vector 하나에 억지로 묶는 것은 끝낸다

## 설계 범위

### in scope

- shared spatial fusion stem
- lane row-anchor rewrite
- stop_line spatial geometry rewrite
- crosswalk spatial geometry rewrite
- internal target / loss / matching / postprocess contract 변경
- checkpoint migration policy
- export metadata 변경

### out of scope

- trunk 교체
- detector head 재설계
- traffic-light attr head 재설계
- dataset canonical format 재생성
- segmentation-first 전환
- public prediction schema를 `points_xy` 밖으로 바꾸는 일

## 고정 원칙

1. trunk는 `yolo26s.pt`를 권장 기본값으로 유지하되, `yolo26n.pt` / `yolo26s.pt` variant compatibility는 계속 유지한다.
2. `P3 / P4 / P5`를 spatial하게 fuse한 뒤 각 task가 그 spatial memory를 읽는다.
3. lane과 road marking은 같은 pooled vector를 공유하지 않는다.
4. lane은 `row-anchor x / visibility`로 바꾼다.
5. stop_line은 `2 endpoints + width`로 바꾼다.
6. crosswalk는 `4-corner quad`로 바꾼다.
7. public output은 계속 `points_xy`다.
8. pre-rewrite checkpoint exact resume은 지원하지 않는다.

## 목표 아키텍처

```text
image
  -> YOLO26 trunk reuse
  -> P3 / P4 / P5
  -> channel projection to 256 each
  -> upsample P4/P5 to P3 resolution
  -> concat
  -> shared spatial fusion stem [B, 256, 76, 100]
      -> lane memory [B, 256, 76, 100]
          -> lane query decoder
          -> lane row-anchor head
      -> geometry memory [B, 256, 76, 100]
          -> stop_line small query decoder
          -> stop_line line head
          -> crosswalk small query decoder
          -> crosswalk quad head
```

## Shared Spatial Fusion Stem

- 입력 feature:
  - `P3`: stride 8
  - `P4`: stride 16
  - `P5`: stride 32
- 각 scale은 `1x1 conv`로 `256` 채널로 projection한다.
- `P4`는 `x2`, `P5`는 `x4` bilinear upsample로 `P3` 해상도에 맞춘다.
- 세 scale을 concat한 뒤 `3x3 conv -> BN -> SiLU -> 3x3 conv -> BN -> SiLU`로 fuse한다.
- shared fusion stem output은 `[B, 256, 76, 100]`로 고정한다.
- 이후 branch는 다음 두 개로만 나눈다.
  - `lane_memory`
  - `geometry_memory`

## Lane Branch

### lane memory

- shared fusion stem에서 `3x3 conv -> BN -> SiLU -> 3x3 conv -> BN -> SiLU`를 한 번 더 적용해 lane 전용 memory를 만든다.
- lane memory shape는 `[B, 256, 76, 100]`를 유지한다.

### lane query decoder

- query count는 `24`로 고정한다.
- learned query embedding `24 x 256`을 사용한다.
- decoder block 수는 `2`로 고정한다.
- 각 decoder block은
  - self-attention
  - cross-attention to flattened lane memory tokens
  - FFN
  순서의 standard transformer decoder block을 사용한다.
- positional encoding은 `P3` grid 기준의 2D sine-cosine encoding을 사용한다.

### lane internal contract

- lane output shape는 `B x 24 x 38`이다.
- lane row layout:

| Slice | Meaning |
| --- | --- |
| `[0]` | objectness logit |
| `[1:4]` | color logits (`white / yellow / blue`) |
| `[4:6]` | type logits (`solid / dotted`) |
| `[6:22]` | `16` anchor-row x coordinates |
| `[22:38]` | `16` anchor-row visibility logits |

- 추가 `start/end row` 보조값은 넣지 않는다.
- lane span은 visibility가 담당한다.

### lane anchor rows

- anchor row count는 `16`이다.
- anchor rows는 network-space `y` 좌표 기준 `linspace(607.0, 0.0, 16)`로 고정한다.
- 순서는 bottom-to-top이다.
- anchor는 raw-space가 아니라 letterbox 이후 network-space 기준으로 정의한다.

### lane target encoding

- GT lane point는 `y` descending으로 정렬한다.
- 각 anchor row마다 GT polyline과 교차하는 segment를 찾아 `x`를 선형 보간한다.
- 해당 anchor row를 덮는 segment가 없으면 `visibility = 0`, `x = 0`으로 채우고 regression mask에서 제외한다.
- visibility source policy는 아래 둘 중 하나만 허용한다.
  - source annotation이 실제 visibility를 제공하면 standardization 단계부터 canonical scene에 그대로 보존한다.
  - source annotation이 실제 visibility를 제공하지 않으면 canonical scene에는 `point-span 기반 pseudo-visibility`만 기록한다.
- downstream 문서와 코드에서는 위 둘을 구분해서 부른다.
  - 실제 source visibility가 있으면 `raw visibility`
  - 실제 source visibility가 없고 geometry에서 만든 값이면 `pseudo-visibility`
- source visibility가 있으면 같은 segment에서 visibility도 선형 보간한 뒤 `>= 0.5`를 visible로 본다.
- source visibility가 없으면 point span 안에서만 `visibility = 1`로 처리한다.
- `x` loss는 `GT visibility = 1` anchor에서만 계산한다.
- invisible anchor의 `x` 값은 contract padding일 뿐 의미 있는 supervision 대상이 아니다.

### lane matching / loss

- Hungarian matching은 유지한다.
- lane cost:
  - visible-anchor masked `x` L1
  - visibility BCE
  - color CE cost
  - type CE cost
- lane loss:
  - objectness BCE
  - color CE
  - type CE
  - visible-anchor masked SmoothL1 on `x`
  - visibility BCE
  - second-difference smoothness regularizer on predicted anchor sequence
  - visibility total-variation regularizer on anchor visibility logits

### lane span continuity rule

- lane span은 visibility가 담당한다.
- 대신 구현은 아래 두 개를 같이 쓴다.
  - training: visibility total-variation regularizer
  - postprocess: 가장 긴 contiguous visible run만 채택
- 따라서 `1, 1, 0, 1, 0, 1` 같은 fragmented visibility는 학습과 decode 양쪽에서 억제한다.

### lane postprocess

- `visibility sigmoid > lane_visibility_threshold` anchor만 active anchor로 사용한다.
- active anchor는 threshold 통과 anchor 전체가 아니라, threshold 통과 anchor 중 가장 긴 contiguous run만 사용한다.
- contiguous run 기준 active anchor가 `2`개 미만이면 해당 query는 버린다.
- active anchor의 `(x, y_anchor)`를 raw-space로 inverse transform해서 `points_xy`로 내보낸다.
- lane dedupe는 다음 세 조건을 함께 본다.
  - visible-anchor overlap ratio
  - active anchor mean `x` distance
  - color/type agreement
- 중복으로 판단되면 score가 낮은 query를 제거한다.

## Stop-Line Branch

### geometry memory

- shared fusion stem에서 `3x3 conv -> BN -> SiLU -> 3x3 conv -> BN -> SiLU`를 적용해 road-marking geometry memory를 만든다.
- geometry memory shape는 `[B, 256, 76, 100]`를 유지한다.
- stop_line과 crosswalk는 이 memory를 공유하되, query embedding과 predictor는 분리한다.

### stop_line decoder

- query count는 `8`로 고정한다.
- learned query embedding `8 x 256`을 사용한다.
- decoder block 수는 `1`로 고정한다.
- positional encoding은 lane branch와 같은 2D sine-cosine encoding을 사용한다.

### stop_line internal contract

- stop_line output shape는 `B x 8 x 6`이다.

| Slice | Meaning |
| --- | --- |
| `[0]` | objectness logit |
| `[1]` | `x1` |
| `[2]` | `y1` |
| `[3]` | `x2` |
| `[4]` | `y2` |
| `[5]` | width |

- endpoint는 network-space 좌표다.
- endpoint order는 `x` ascending 기준으로 정렬한다.
- `x`가 사실상 같으면 `y` descending endpoint를 첫 번째 endpoint로 둔다.
- `width`는 stop line의 full thickness를 나타내는 internal geometry 보조값이다.
- `width`는 public output에 직접 노출하지 않는다.

### stop_line target encoding

- GT stop line points는 left-to-right로 정렬한다.
- polyline 전체를 대표하는 line segment는 first point / last point로 잡는다.
- `width`는 GT polyline point가 `3`개 이상일 때만 supervised target으로 계산한다.
- supervised width는 point-to-segment mean perpendicular distance의 `2x`로 잡는다.
- GT polyline point가 `2`개뿐이면 predicted `width`는 auxiliary prior only로 취급하고 matching/loss에서 제외한다.
- default prior width는 network-space `6.0`을 사용하되, 이 값은 target label이 아니라 decode/dedupe 보조 prior다.
- encoded batch는 stop_line row별 width-valid mask를 별도로 유지한다.

### stop_line matching / loss

- stop_line cost:
  - endpoint L1
  - width-valid row에서만 width L1
  - line angle / line length cost
- stop_line loss:
  - objectness BCE
  - endpoint SmoothL1
  - width-valid row에서만 width SmoothL1
  - line angle / line length regularizer

### stop_line postprocess

- score threshold를 넘는 query만 사용한다.
- `(x1, y1) -> (x2, y2)` centerline을 `4`개 등간격 point로 resample해서 public `points_xy`를 만든다.
- `width`는 public output에 넣지 않고, dedupe나 later visualization 보조값으로만 남긴다.
- dedupe는 centerline distance + angle difference 기준으로 처리한다.

## Crosswalk Branch

### crosswalk decoder

- query count는 `8`로 고정한다.
- learned query embedding `8 x 256`을 사용한다.
- decoder block 수는 `1`로 고정한다.

### crosswalk internal contract

- crosswalk output shape는 `B x 8 x 9`이다.

| Slice | Meaning |
| --- | --- |
| `[0]` | objectness logit |
| `[1:9]` | 4-corner quad coordinates |

- quad corner order는 clockwise다.
- first corner는 network-space top-left corner다.
- 이후 순서는 clockwise로 고정한다.

### crosswalk target encoding

- GT crosswalk polygon은 clockwise로 정렬한다.
- starting point는 top-left corner로 맞춘다.
- 원본 polygon point 수가 4가 아니면 min-area-rect 기반 4-corner quad로 근사하지 않고, 현재 canonical contour를 `4`개 corner로 arc-length downsample한다.
- 이 설계의 목적은 axis-aligned box가 아니라 perspective-aware quad를 유지하는 데 있다.

### crosswalk matching / loss

- crosswalk cost:
  - ordered corner L1
  - polygon IoU cost
- crosswalk loss:
  - objectness BCE
  - corner SmoothL1
  - convexity / edge-length consistency regularizer

### crosswalk postprocess

- score threshold를 넘는 query만 사용한다.
- quad corner를 그대로 `points_xy`에 넣는다.
- dedupe는 quad IoU 기반으로 처리한다.

## Public Output Rule

- lane public output:
  - `score`
  - `class_name`
  - `lane_type`
  - `points_xy`
- stop_line public output:
  - `score`
  - `points_xy`
- crosswalk public output:
  - `score`
  - `points_xy`

즉:

- internal contract는 바뀐다
- evaluator / metrics가 읽는 postprocess output surface는 유지한다

## Checkpoint / Resume Policy

- 이 rewrite는 architecture break다.
- pre-rewrite run의 `--resume-run` exact resume은 지원하지 않는다.
- exact resume 지원 범위는 rewrite 이후 같은 architecture generation 내부로 제한한다.
- old checkpoint reuse는 아래만 허용한다.
  - trunk weights
  - detector head
  - TL attr head
  - shape-compatible shared blocks
- lane / stop_line / crosswalk predictor는 새 generation에서 random init을 기본으로 한다.
- migration load는 strict `load_state_dict()`가 아니라 shape-aware partial load를 별도 경로로 제공해야 한다.
- migration helper는 existing `load_matching_state_dict()` 재사용을 기본 경로로 한다.

## Export / Runtime Rule

- current raw-head export는 crosswalk tensor와 metadata가 빠져 있으므로, rewrite generation에서는 raw-head export contract 자체를 갱신한다.
- TorchScript raw-head export metadata는 det / tl_attr / lane / stop_line / crosswalk shape를 모두 반영해야 한다.
- 이 wave에서는 postprocess 포함 export contract를 새로 만들지 않는다.
- 즉 export는 계속 raw heads를 내보내되, 누락된 crosswalk를 포함한 complete raw-head surface로 정리한다.
- ROS/export consumer는 internal tensor shape를 직접 읽지 않고, postprocess 이후 public `points_xy` surface만 읽는 것을 기준으로 한다.
- public output schema를 먼저 깨는 방향은 허용하지 않는다.

## 구현 순서

1. lane row-anchor rewrite
2. stop_line spatial geometry rewrite
3. crosswalk quad rewrite
4. checkpoint/export hardening

순서를 이대로 고정하는 이유:

- lane이 가장 큰 병목이다.
- stop_line / crosswalk도 pooled MLP를 벗어나야 하지만 lane과 같은 표현으로 묶을 이유는 없다.
- shared geometry memory는 좋지만, final predictor는 task-specific로 분리하는 편이 더 맞다.

## 설계 결론

- lane은 `row-anchor x / visibility`
- stop_line은 `2 endpoints + width`
- crosswalk는 `4-corner quad`
- 공통점은 `P3 중심 shared spatial fusion stem`과 `query decoder 기반 spatial reading`
- 차이점은 final geometry parameterization이다

따라서 이 rewrite는 다음 한 줄로 요약된다.

> pooled lane-family MLP를 폐기하고, `shared spatial fusion stem + lane row-anchor branch + task-specific road-marking geometry heads`로 재구성한다.
