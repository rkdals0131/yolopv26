# PV26 Lane Head 개편 제안 보고서

## 1. 요약

현재 PV26에서 우선적으로 바꿔야 할 대상은 trunk가 아니라 lane head다. 현재 trunk는 YOLO26s의 P3/P4/P5 feature를 재사용하고 있고 detector / tl_attr branch는 끝까지 spatial feature를 읽는다. 반면 lane family는 각 scale feature를 global average pooling으로 압축한 뒤 `[B, 896]` 벡터에서 고정 query MLP로 lane geometry를 예측한다. 따라서 병목은 trunk 이전이 아니라, lane 경로가 spatial map을 head 직전에 버리는 구조에 있다.

이 보고서의 결론은 다음 한 문장으로 정리된다.

> **`YOLO26s trunk reuse`는 유지하고, lane만 `P3 중심 spatial decoder + row-anchor x/visibility head`로 바꾸는 것이 현재 코드베이스에서 가장 현실적이고 효과적인 해법이다.** 

## 2. 왜 그렇게 판단했는가

데이터 분포만 봐도 lane은 별도 취급이 필요하다. 현재 inspected dataset은 57,700장이고 lane은 총 238,180개로 대부분 이미지에 존재한다. 반면 stop_line과 crosswalk는 훨씬 sparse하다. lane 수는 평균 4.13개지만 최대 20개까지 나온다. 즉 lane은 “부가 태스크”가 아니라, 이 데이터셋에서 가장 자주 등장하고 구조적으로도 가장 까다로운 geometry다. 

또 lane은 bounding box와 성질이 다르다. 긴 선형 구조, row-wise continuity, 곡률 변화, 가림 여부, near/far consistency가 핵심이다. 2024 survey도 lane detection의 핵심 설계축으로 lane modeling과 global context supplementation을 강조한다. UFLD는 row-based selecting formulation을 제안했고, LSTR과 Laneformer는 transformer로 long and thin structure와 global context를 읽는 방향을 택했다. 즉 최근 lane 연구의 흐름은 “lane을 box처럼 단순 압축해서 회귀하지 말라”는 쪽에 가깝다. 

## 3. 현재 코드의 핵심 문제

### 3.1 Spatial 정보가 너무 일찍 사라진다

현재 lane family path는 `P3/P4/P5 -> mean(H,W) -> concat -> MLP` 구조다. detector는 spatial map을 유지한 채 logits를 내보내는데, lane은 head 직전에 이미 단일 벡터가 된다. 이 차이 때문에 lane에 필요한 위치 정보, row별 존재 정보, 곡률 변화 정보가 과도하게 평균된다. trunk를 바꿔도 이 문제는 남는다. 

### 3.2 Visibility supervision이 사실상 죽어 있다

raw lane label에는 `visibility`가 존재하지만, 현재 training-time encoded target에서는 visibility slice가 사실상 `all visible`로 채워진다. 이 상태에서는 occlusion reasoning이 사라지고, “가려져서 안 보이는 lane”과 “원래 없는데 hallucination한 lane”을 구분하기 어렵다. visibility output 채널이 있어도 supervision이 죽어 있으니 학습 이득이 제한적일 수밖에 없다. 

### 3.3 Query cap이 실제 데이터 분포보다 낮다

현재 contract는 lane query 12개인데 실제 이미지당 최대 lane 수는 20개다. 그 결과 lane overflow sample이 109개 있고, crosswalk도 max 5인데 query 4라 overflow가 14개 있다. lane encoder가 cap을 넘는 GT를 잘라 쓰는 구조라면 복잡한 장면에서 recall ceiling이 생기는 건 자연스럽다.

### 3.4 Lane representation이 lane에 최적화되어 있지 않다

현재는 한 lane을 54차원 벡터 하나로 표현한다. 여기에는 objectness, color, type, 16-point polyline, 16-point visibility slot이 들어가지만, 실제로는 한 번에 자유곡선을 회귀하는 셈이라 local-to-global correction 경로가 없다. heavily curved lane, partial visibility, long-range lane일수록 난도가 높다. lane_family_only fine-tune도 이 구조적 bottleneck을 없애지는 못한다. 

## 4. 제안 아키텍처

내가 추천하는 구조는 **`spatial query + row-anchor lane head`**다. 이는 DETR류의 set prediction 장점과 row-anchor lane modeling의 장점을 합치는 절충안이다. DETR은 learned query와 bipartite matching이 set prediction에 유효함을 보여줬고, lane 쪽에서는 UFLD / LaneATT / CLRNet이 row-anchor, lane-specific pooling, high/low-level refinement의 효과를 보여줬다. transformer 기반 lane 모델인 LSTR, Laneformer, LDTR도 attention 자체보다 “lane-aware한 spatial interaction”이 중요하다는 점을 보여준다. 

권장 구조는 다음과 같다.

1. trunk는 그대로 둔다.
2. `P3`를 중심으로 `P4/P5`를 upsample해서 fuse하고 lane 전용 spatial feature를 만든다.
3. learned lane query는 이 spatial token과 cross-attention한다.
4. lane당 아래 출력을 예측한다.
   - objectness
   - lane color
   - lane type
   - 16개 fixed anchor row에서의 `x`
   - 16개 fixed anchor row에서의 `visibility`
   - 필요하면 시작/종료 row 보조값

이 구조의 핵심은 **query를 유지하되 spatial map과 상호작용하게 만들고, geometry parameterization은 free-form point regression에서 row-anchor 방식으로 바꾸는 것**이다. 

## 5. 왜 trunk는 유지하고 lane head만 바꾸는가

shared encoder에 task-specific head를 붙이는 방향 자체는 잘못이 아니다. YOLOP와 HybridNets도 shared encoder 위에 detection / lane / other task를 위한 별도 decoder를 둔다. 따라서 “YOLO 계열 trunk를 버리고 lane 전용 backbone으로 갈아타자”보다는, **현재 trunk를 유지한 채 lane head만 lane-friendly하게 바꾸는 편이 리스크 대비 효과가 좋다.** 지금 저장소도 이미 P3/P4/P5라는 괜찮은 multi-scale feature를 가지고 있고, 문제는 그 feature를 lane path에서 pooling으로 버리는 데 있기 때문이다.

## 6. 왜 pure segmentation을 1순위로 두지 않는가

lane segmentation branch는 분명 강한 대안이다. YOLOP와 HybridNets도 lane segmentation 계열을 사용하고, segmentation 방식은 thin geometry supervision에 유리하다. 다만 현재 repo는 fixed query / polyline output contract를 중심으로 묶여 있다. 이 상태에서 pure segmentation으로 가면 export, postprocess, runtime contract를 모두 흔들 가능성이 크다. 그래서 학술적으로는 좋은 카드지만, **현재 저장소와 대회 일정 기준으로는 1순위보다 2순위 카드**라고 본다. 먼저 spatial query + row-anchor로 가고, 필요하면 auxiliary heatmap branch를 training-only로 추가하는 편이 더 안전하다. 

## 7. loss / matching / postprocess 권고

row-anchor 표현으로 바꾸면 loss도 함께 바뀌어야 한다. objectness는 BCE/Focal, color/type은 CE, geometry는 `visibility=1`인 anchor에서만 x regression을 계산하는 masked L1 또는 SmoothL1이 자연스럽다. 여기에 curvature smoothness regularizer를 더하면 lane continuity를 더 잘 보존할 수 있다. 이때 raw visibility를 실제 supervision으로 살리는 것이 중요하다.

matching과 confidence도 lane 전용으로 보는 편이 낫다. LaneIoU/CLRerNet은 많은 경우 “맞는 lane 위치는 이미 예측 안에 있는데 confidence와 assignment가 lane metric에 잘 안 맞는 것”이 문제일 수 있음을 보여준다. 그래서 후속 단계에서는 lane similarity 기반 matching cost나 confidence target을 도입할 가치가 있다. 

postprocess에서는 lane dedupe가 필요하다. query 수를 20까지 늘리고 spatial decoder를 쓰면 recall은 좋아질 수 있지만, 비슷한 lane이 여러 query로 중복 예측될 가능성도 커진다. visible anchor overlap, 평균 x distance, type/color 일치 여부를 기준으로 낮은 score query를 제거하는 방식이 실용적이다.

## 8. stop_line / crosswalk는 lane과 분리해서 보자

lane은 길고 얇으며 row-wise continuity가 중요하지만, stop_line은 짧은 선분이고 crosswalk는 polygon/area 성격이 강하다. 현재 데이터 분포에서도 lane은 dense하고 stop_line / crosswalk는 sparse하다. 그래서 lane 때문에 모든 geometry를 같은 parameterization으로 몰아넣기보다, **lane만 row-anchor로 바꾸고 stop_line / crosswalk는 기존 polyline / polygon 계열 head를 유지하는 편**이 더 합리적이다. 

## 9. 구현 우선순위

### 1단계: 가장 빠른 효과
- pooled lane MLP를 spatial decoder로 교체
- lane query 12 -> 20
- raw visibility 보존
- lane dedupe 추가

### 2단계: 표현 자체 교체
- target encoder를 16 fixed rows 기준으로 재작성
- visible-anchor masked regression 도입
- anchor row 기반 postprocess 작성

### 3단계: 추가 성능 향상
- training-only auxiliary lane heatmap branch
- 필요하면 detector object token을 lane decoder에 넣는 object-aware context 확장
- heavy attention이 부담되면 deformable attention 검토

이 순서는 구조 리스크를 낮추면서도 성능 개선 가능성이 큰 쪽부터 적용하는 방식이다. Laneformer와 LDTR은 object-aware / deformable attention 계열이 후속 강화 포인트가 될 수 있음을 보여준다. 

## 10. 실전 관점에서의 최종 판단

대회 규정상 Camera는 대수 제한이 없고, LiDAR는 1대, GPS는 1개까지 허용된다. 또 경기 배점은 종합주행성능 500점, 조향성능 300점, 가속성능 200점이다. 즉 이 대회에서는 “lane benchmark 수치만 높은 모델”보다, **실제 주행에서 안정적으로 쓰이고 지연이 낮은 lane branch**가 더 중요하다. 그 맥락에서도 trunk 전체를 갈아엎기보다, 현재 YOLO trunk를 유지하면서 lane head만 lane 전용 구조로 바꾸는 선택이 더 현실적이다. 

## 11. 최종 한 줄 결론

**YOLO26s trunk reuse는 유지하고, lane만 `P3 중심 spatial decoder + 20 query + row-anchor x/visibility head`로 교체하라.**
