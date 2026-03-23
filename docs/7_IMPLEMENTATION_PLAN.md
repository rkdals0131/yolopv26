# Implementation Plan

## phase 0 완료

- AIHUB standardization
- source README
- conversion report
- debug overlay
- loss design spec

## phase 1 next

### 1. training sample contract

- standardized output를 읽는 loader contract 정의
- sample dictionary schema 정의
- image, det, tl bits, lane family fields를 통일

### 2. online transform

- `800x600 -> 800x608` preprocessing contract 구현
- train/infer shared transform 작성
- raw-space label를 transformed-space target으로 변환

### 3. target encoder

- detector target encoder
- TL bit target encoder
- lane vector encoder
- stop-line encoder
- crosswalk encoder

### 4. trunk adapter

- official `yolo26n.pt` load path 결정
- backbone/neck parameter mapping 작성
- partial load verification 추가

### 5. custom heads

- det head
- TL attr head
- lane head
- stop-line head
- crosswalk head

### 6. loss runtime

- spec를 실제 loss module로 구현
- masking / matching / normalization 구현

### 7. trainer skeleton

- dataset-balanced sampler
- stage-wise freeze policy
- optimizer group 분리
- logging / checkpoint

### 8. evaluator skeleton

- detector metrics
- TL bit metrics
- lane family metrics

## implementation order

1. loader
2. transform
3. target encoder
4. trunk adapter
5. heads
6. loss
7. trainer
8. evaluator

## 문서 업데이트 규칙

- 각 phase 시작 시 `9_EXECUTION_STATUS.md`를 먼저 갱신
- phase 완료 후 test 결과를 status에 기록
- 설계 변경이 생기면 해당 번호 문서를 먼저 수정
