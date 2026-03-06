> **Archived (2026-03-05)** — kept for history; may be outdated.  
> Canonical docs: `docs/PV26_PRD.md`, `docs/PV26_DATASET_CONVERSION_SPEC.md`, `docs/PV26_DATASET_SOURCES_AND_MAPPING.md`, `docs/PV26_IMPLEMENTATION_STATUS.md`.

# PV26 Lane Marker vs Lane Subclass — A(추론 게이팅) vs B(배경 포함 학습) 정리

목표: **lane(차선/차선마킹)의 존재를 찾는 head**와 **lane의 종류(white/yellow, solid/dashed)를 찾는 head**를 분리해 둔 상태에서,
배경(background)을 어떻게 학습/추론해야 “얇은 선(thin structure)”에서 안정적으로 동작하는지 결정하기 위한 문서입니다.

---

## 0) 용어를 쉬운 말로 정리

- **lane head**: “이 픽셀이 lane 마킹인가?”를 예측하는 head  
  - 이 레포에서는 `rm` head의 `rm_lane_marker` 채널을 보통 lane head로 봅니다. (`pv26/multitask_model.py:176`)
- **lane-subclass head**: “lane이라면 무슨 종류인가?”를 예측하는 head  
  - 이 레포에서는 `rm_lane_subclass` head가 해당합니다. (`pv26/multitask_model.py:270`)
- **배경(background)**: lane이 아닌 모든 픽셀(도로, 차, 하늘, 드라이버블 영역 등 전부 포함)
- **희소(sparse)**: lane 픽셀은 이미지 전체에서 비율이 아주 낮음(대부분은 배경)

핵심 직관:
- lane-subclass는 **lane이 “있을 때만” 의미가 있는 정보**입니다.
- 그래서 “배경을 얼마나/어떻게 학습하느냐”가 성능과 안정성에 큰 영향을 줍니다.

---

## 1) 현재 코드가 실제로 하는 일(사실)

### 1-1. 구조: lane과 subclass는 “병렬”로 나옴 (게이팅 없음)

모델은 같은 feature에서
`rm_head`(lane_marker 포함 3채널)와 `rm_lane_subclass_head`(bg+subclass 채널)를 **별도 head로 병렬 출력**합니다.

- `pv26/multitask_model.py:176` (`self.rm_head = ...`)
- `pv26/multitask_model.py:177` (`self.rm_lane_subclass_head = ...`)
- YOLO26 기반 모델도 동일하게 병렬 출력: `pv26/multitask_model.py:296`, `pv26/multitask_model.py:297`

즉, **네트워크 내부에서** “lane marker가 1인 위치에서만 subclass를 내라” 같은 논리 결합은 없습니다.

### 1-2. 학습(loss): lane은 배경 포함 학습, subclass는(현재) 배경을 거의 학습하지 않음

- DA loss는 valid 픽셀 전체(0/1)에서 BCE를 계산합니다. (`pv26/criterion.py:414`, `pv26/criterion.py:417`)
- RM loss도 valid 픽셀 전체(0/1)에서 focal+dice를 계산합니다. (`pv26/criterion.py:431`, `pv26/criterion.py:436`)
  - 그래서 **lane head는 배경(0)도 학습합니다.**

반면 lane-subclass loss는 현재 구현에서:
- **양성(1..K) 픽셀만** CE를 걸고,
- 배경(0)은 loss에서 마스킹합니다. (`pv26/criterion.py:474`)

이 설정은 의도가 분명합니다:
> “lane-subclass는 lane 위에서만 의미가 있으니, 배경에서 뭘 예측하든 상관없다(대신 lane 위에서만 잘 맞춰라)”

단, 이 의도를 만족하려면 **추론에서 반드시 게이팅(마스킹)** 을 해줘야 합니다(아래 A 옵션).

---

## 2) A vs B: 무엇이 다른가? (한 문장 요약)

- **A (게이팅 전제)**:  
  “lane head가 lane 위치를 결정하고, lane-subclass는 *그 lane 위에서만* 종류를 맞춘다.”

- **B (단독 segmentation)**:  
  “lane-subclass head 자체가 배경까지 포함해, lane 여부 + 종류를 한 번에(혹은 거의) 해결한다.”

둘 다 가능하지만, **use case**와 **데이터셋 구성**에 따라 정답이 달라집니다.

---

## 3) 옵션 A — 추론에서 lane으로 게이팅(추천 방향: 현재 요구와 잘 맞음)

### 3-1. 언제 A가 맞나?

아래 조건이면 A가 자연스럽습니다(지금 질문의 전제와 거의 동일):

- “lane-subclass는 lane 위에서만 유효” (독립적으로 필요하지 않음)
- 여러 데이터셋 호환을 위해 **lane 라벨만 있는 데이터**도 포함됨
- 최종 산출물은 polyline(또는 lane 인스턴스)이고, subclass는 그 위에서 **비율/다수결 룰**로 결정할 계획

### 3-2. A에서 학습은 어떻게 해야 하나?

권장:
1. **lane head(`rm_lane_marker`)는 배경 포함으로 강하게 학습**  
   - 지금처럼 (0/1) 전체 픽셀에서 focal+dice/BCE로 학습하는 게 일반적입니다. (`pv26/criterion.py:431`)
2. **lane-subclass는 lane-positive 영역 중심으로 학습**  
   - “lane인 곳에서만 white/yellow/solid/dashed를 맞추기”  
   - 구현 방식:
     - (가장 단순) GT subclass(1..K) 픽셀에만 CE를 건다. (현재와 유사) (`pv26/criterion.py:474`)
     - (더 계층적으로) GT lane_marker==1인 위치에서만 CE를 건다.
3. lane-subclass 배경(0)은 **굳이 강하게 학습하지 않아도 됨**  
   - 어차피 추론에서 lane 밖은 무조건 background로 “결정”할 것이기 때문
   - 불균형 때문에 오히려 배경 학습을 강하게 넣으면 subclass가 죽는 경우가 많습니다.

### 3-3. A에서 추론은 어떻게 해야 하나? (핵심)

요점: lane-subclass 맵을 “그대로” 쓰지 말고, **lane 맵 위에서만 사용**합니다.

#### Hard gating (가장 쉬움)

1) lane head 확률로 lane 픽셀을 결정:
- `lane_mask = (sigmoid(rm_lane_logit) > thres)`

2) lane-subclass는 그 안에서만 클래스 선택:
- `lane_subclass_cls[~lane_mask] = 0 (bg)`

#### Soft gating (확률적으로 더 깔끔)

계층적 확률로 해석:
- `P(lane_type=k) = P(lane) * P(type=k | lane)`
- `P(bg) = 1 - P(lane)`

실무적으로는:
- lane-subclass logits에서 “bg 채널”은 신뢰하지 않고(학습 안 했으면 더더욱),  
  subclass(1..K)만 softmax해서 `P(type|lane)`로 쓰는 식이 안전합니다.

### 3-4. A의 장단점

장점
- 극단적인 class imbalance(배경 압도)를 lane-subclass에서 피할 수 있음
- “lane이 있어야 subclass가 의미 있다”는 실제 논리를 그대로 반영
- 데이터셋 호환이 좋음(서브클래스 없는 샘플은 `has_rm_lane_subclass=0`로 loss에서 제외 가능)

단점
- lane head가 틀리면 subclass도 같이 틀리거나 누락(계층적 의존)
  - 하지만 질문의 전제(“lane head를 강하게 잘 만들겠다”)와 잘 맞음

---

## 4) 옵션 B — lane-subclass가 배경까지 포함해 단독으로 깨끗한 segmentation을 하도록 학습

### 4-1. 언제 B가 맞나?

- lane head가 불안정하거나, lane head 없이도 subclass 맵이 직접 쓰여야 하는 제품 요구가 있음
- 디버그/시각화/후처리에서 “subclass 맵만 봐도 lane이 깨끗하게 나와야” 한다는 강한 요구가 있음

질문에서 “독립적인 lane-subclass use case는 없다”고 했으니, **B는 필수는 아닐 가능성이 큽니다.**

### 4-2. B에서 학습은 어떻게 해야 하나? (난이도 포인트)

B의 핵심 난점은 하나입니다:
> 배경 픽셀 비율이 너무 커서, 그냥 full-image CE를 하면 subclass가 쉽게 망가진다.

그래서 B를 하려면 아래 중 하나는 거의 필수입니다.

- **가중치(class weight)**: bg 가중치를 매우 낮추거나 subclass 가중치를 높임
- **focal CE**: 쉬운 배경을 덜 벌점 주도록(“어려운 픽셀에 집중”)
- **negative sampling**: 배경 픽셀을 전부 쓰지 말고 일부만 샘플링
- **two-term loss**: (양성 CE 평균) + (배경 CE 평균 * 작은 계수)처럼 균형을 맞춤

### 4-3. B의 장단점

장점
- lane-subclass 맵 하나만 봐도 “배경/종류”가 일관되게 나올 수 있음
- lane head 품질에 대한 의존이 줄어듦

단점
- 불균형 튜닝이 필요하고, 잘못하면 “전부 bg” 혹은 “전부 subclass”로 무너질 수 있음
- lane head와 역할이 겹치면서 학습이 비효율적일 수 있음(중복 학습)

---

## 5) 질문에 대한 결론(추천)

질문에서 명시한 요구/전제를 그대로 따르면:

1) **lane head는 배경 포함으로 강하게 학습한다**  
   - 현재 구현도 그렇게 되어 있음 (`pv26/criterion.py:431`)

2) **lane-subclass는 “lane 위에서만 유효”하게 설계/학습/추론한다**  
   - 학습: lane-positive 영역 중심(현재처럼 양성만 CE 또는 GT lane_marker==1에서만 CE)
   - 추론: 반드시 `rm_lane_marker`로 게이팅해서 lane 밖 subclass는 무시/0으로 처리

이게 옵션 A이며, “여러 데이터셋 호환 + subclass는 lane 조건부”라는 목표에 가장 맞습니다.

### 한 문장으로:
**lane은 “존재/형태”를 튼튼하게, subclass는 “lane 위에서만 타입”을 맞추게 하고, 최종 결과는 polyline 위에서 집계한다.**

---

## 6) 체크리스트(의사결정용)

아래에 “예”가 많으면 A가 유리합니다.

- [ ] lane-subclass를 lane 없이 단독으로 쓸 일은 없다
- [ ] lane-only 데이터셋이 섞인다(= subclass 라벨이 없는 샘플이 많다)
- [ ] 최종은 polyline이며 subtype은 그 위에서 집계/룰로 결정한다
- [ ] thin structure에서 subclass 학습이 불안정해지는 걸 피하고 싶다

B가 필요한 경우는 주로 “subclass 맵이 제품에서 직접 쓰여야 하는” 요구가 있을 때입니다.

---

## 7) 다음 액션 제안(원하면 코드로 반영)

1) **추론/후처리에서 subclass 게이팅을 명시적으로 구현**  
   - (디버그 렌더러 포함) `rm_lane_prob`로 `lane_cls`를 마스킹하는 모드 추가
2) **학습에서 subclass loss를 GT lane_marker==1 영역으로 제한**(원하는 계층성 강화)  
   - 데이터셋이 일관되면(현재 BDD 변환 기준은 일관), 안정적
3) (선택) subclass에 “약한 배경 규제”를 넣고 싶으면  
   - 배경 픽셀을 아주 조금만 샘플링해서 negative term을 추가(불균형 폭발 방지)
