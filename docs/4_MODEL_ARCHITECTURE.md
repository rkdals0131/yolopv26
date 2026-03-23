# Model Architecture

## 핵심 방향

- trunk는 `Ultralytics YOLO v26 nano` 공식 pretrained model을 기준으로 한다.
- backbone과 neck는 가능한 한 공식 구조와 채널 구성을 유지한다.
- PV26 requirement는 custom head와 loss로 해결한다.

## trunk reuse 원칙

- load source
  - official `yolo26n.pt`
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
  -> YOLOv26n backbone
  -> YOLOv26n neck
  -> det head
  -> tl attr head
  -> lane head
  -> stop-line head
  -> crosswalk head
```

## detector head

- output
  - bbox
  - obj
  - 7-class logits
- detector class는 generic `traffic_light`를 유지한다.

## traffic light attr head

- output
  - `red`
  - `yellow`
  - `green`
  - `arrow`
- detector의 `traffic_light` positive에만 의미가 있다.
- 표현은 4 independent sigmoid logits이다.

## lane family heads

- lane head
  - fixed query count
  - color logits
  - type logits
  - 16-point polyline
  - visibility logits
- stop-line head
  - fixed query count
  - 4-point polyline
- crosswalk head
  - fixed query count
  - 8-point polygon

## 구현 규칙

- trunk adapter를 먼저 만든다.
- pretrained loading은 partial load로 구현한다.
- head 모듈은 trunk와 분리된 클래스로 둔다.
- smoke 기준은 `forward -> loss -> backward`가 된다.

## stage-wise training policy

1. head warm-up
   - trunk freeze
2. neck + upper backbone unfreeze
3. end-to-end fine-tune

## 미정 항목

- lane/stop/crosswalk head의 query count 최종값
- detect branch와 TL attr branch의 구체적 결합 방식
- ultralytics codebase를 wrapper로 쓸지, trunk만 가져와 local module로 감쌀지
