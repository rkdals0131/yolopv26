# Model Architecture

## 핵심 방향

- trunk는 `Ultralytics YOLO v26` 공식 pretrained model을 기준으로 한다.
- PV26 runtime은 `yolo26n.pt`와 `yolo26s.pt`를 모두 수용한다.
- 권장 backbone 경로는 작은 객체 recall과 전체 표현 용량을 고려해 `yolo26s.pt`다.
- backbone과 neck는 가능한 한 공식 구조와 채널 구성을 유지한다.
- PV26 requirement는 custom head와 loss로 해결한다.

## trunk reuse 원칙

- load source
  - official `yolo26n.pt`
  - official `yolo26s.pt`
- reuse 대상
  - backbone
  - neck
- 새로 초기화할 대상
  - detector class head
  - traffic light attr head
  - lane head
  - stop-line head
  - crosswalk head

## 왜 이 방향인가

- COCO pretrained trunk는 저수준/중간 수준 시각 표현을 이미 학습했다.
- PV26는 detector class와 geometry task가 달라 head는 교체가 필요하다.
- trunk를 scratch로 재학습하는 것보다 수렴 시작점이 낫다.

## PV26 model block 구성

```text
input image
  -> YOLOv26 backbone
  -> YOLOv26 neck
  -> det head
  -> tl attr head
  -> lane head
  -> stop-line head
  -> crosswalk head
```

## detector head

- output
  - `Q_det` detector slot별 bbox
  - `Q_det` detector slot별 obj
  - `Q_det` detector slot별 7-class logits
- detector class는 generic `traffic_light`를 유지한다.
- `Q_det`는 prediction slot 수를 의미하고, GT row 수와 다르다.
- 현재 `608x800` 입력 기준 pyramid spatial shape는 아래와 같다.
  - P3: `76 x 100`
  - P4: `38 x 50`
  - P5: `19 x 25`
- channel count는 backbone variant에 따라 달라진다.
  - `yolo26n`: P3/P4/P5 = `64 / 128 / 256`
  - `yolo26s`: P3/P4/P5 = `128 / 256 / 512`
- 즉 PV26 runtime은 fixed spatial contract 위에서 variant-dependent channel contract를 가진다.
- 현재 custom head skeleton의 raw slot count는 `9975`다.

## traffic light attr head

- output
  - `Q_det` detector slot별 `red`
  - `Q_det` detector slot별 `yellow`
  - `Q_det` detector slot별 `green`
  - `Q_det` detector slot별 `arrow`
- detector의 `traffic_light` positive에만 의미가 있다.
- 표현은 4 independent sigmoid logits이다.
- 학습 결합 기준은 detector assignment 결과다.
  - detector가 matched positive로 선택한 GT index를 그대로 TL attr supervision index로 사용한다.
  - matched GT class가 `traffic_light`가 아니면 TL attr loss는 적용하지 않는다.
  - matched GT가 `traffic_light`여도 AIHUB valid mask가 `false`면 TL attr loss는 적용하지 않는다.
- raw model output과 export bundle은 분리해서 다룬다.
  - raw output은 `Q_det` slot aligned logits다.
  - export bundle은 `traffic_light bbox + det confidence + tl bit scores` 묶음이다.

## lane family heads

- lane head
  - fixed query count `12`
  - color logits
  - type logits
  - 16-point polyline
  - visibility logits
- stop-line head
  - fixed query count `6`
  - 4-point polyline
- crosswalk head
  - fixed query count `4`
  - 8-point polygon
- 현재 skeleton 구현은 pooled pyramid embedding 위의 query MLP head다.
- runtime은 trunk adapter가 resolve한 실제 pyramid channel을 head construction에 전달한다.

## 구현 규칙

- trunk adapter를 먼저 만든다.
- trunk adapter는 backbone variant, source weights, resolved pyramid channels를 함께 노출한다.
- pretrained loading은 partial load로 구현한다.
- head 모듈은 trunk와 분리된 클래스로 둔다.
- 최소 regression 기준은 `forward -> loss -> backward`가 된다.
- sample과 transform contract는 [4A_SAMPLE_AND_TRANSFORM_CONTRACT.md](4A_SAMPLE_AND_TRANSFORM_CONTRACT.md)를 기준으로 한다.

## stage-wise training policy

1. head warm-up
   - trunk freeze
2. neck + upper backbone unfreeze
3. end-to-end fine-tune
4. lane-family late fine-tune
   - lane / stop-line / crosswalk metric 중심 보정
   - detector / TL attr은 유지 또는 약한 가중치로만 관여

## 현재 구현 선택

- Ultralytics wrapper를 그대로 사용하되, detect head 직전까지를 trunk adapter로 분리한다.
- partial load helper는 key 이름과 tensor shape가 모두 맞는 항목만 자동으로 로드한다.
- phase 4에서는 lane family 개선을 위해 late-stage selection 기준을 lane family metric 쪽으로 전환할 수 있게 설계한다.
