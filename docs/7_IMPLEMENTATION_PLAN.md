# Implementation Plan

## phase 0 완료

- AIHUB standardization
- source README
- conversion report
- debug overlay
- loss design spec

## phase 1 next

### 1. training sample runtime

- 문서 기준은 [4A_SAMPLE_AND_TRANSFORM_CONTRACT.md](4A_SAMPLE_AND_TRANSFORM_CONTRACT.md)로 고정
- loader runtime은 이 sample dictionary schema를 정확히 구현
- image, det, tl bits, lane family fields를 contract에 맞춰 materialize

### 2. online transform

- 문서 기준은 [4A_SAMPLE_AND_TRANSFORM_CONTRACT.md](4A_SAMPLE_AND_TRANSFORM_CONTRACT.md)로 고정
- variable dataset raw -> `800x608` preprocessing contract를 코드로 구현
- `800x600 -> 800x608`은 vehicle camera reference 입력일 때의 특수 케이스로 취급
- train/infer shared transform 작성
- raw-space label를 transformed-space sample target으로 변환

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
- 현재 skeleton 완료
  - stage config
  - optimizer groups
  - 1-step train runtime
  - dataset-balanced batch sampler helper
  - checkpoint save/load
  - history summary / JSONL logging

### 8. evaluator skeleton

- detector metrics
- TL bit metrics
- lane family metrics
- 현재 skeleton 완료
  - batch loss summary
  - GT count summary
  - raw output -> prediction bundle postprocess decode

## implementation order

1. loader
2. transform
3. target encoder
4. trunk adapter
5. heads
6. loss
7. trainer
8. evaluator
9. tiny overfit

## tiny overfit status

- 완료
  - canonical loader batch 2-sample mixed overfit
  - repeated train-step loss 감소 확인
  - smoke command 추가

## 문서 업데이트 규칙

- 각 phase 시작 시 `9_EXECUTION_STATUS.md`를 먼저 갱신
- phase 완료 후 test 결과를 status에 기록
- 설계 변경이 생기면 해당 번호 문서를 먼저 수정

## 이번 phase에서 이미 고정된 항목

- loader sample dictionary key/shape/dtype
- variable dataset raw -> `800x608` transform 수식
- vehicle camera reference `800x600`은 pad-only 특수 케이스
- lane/stop/crosswalk query count `12 / 6 / 4`
- detector assignment 기반 TL attr binding
