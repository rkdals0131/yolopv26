# PV26 Development Philosophy

## 기본 원칙

- 문서 우선
  - 구현 전에 계약을 문서로 먼저 고정한다.
  - 구현 중 변경되면 문서를 즉시 수정한다.
- raw-space 보존
  - 데이터 표준화는 원본 좌표계를 유지한다.
  - 학습 입력 크기 변환은 loader/transform에서 처리한다.
- pretrained trunk 우선
  - trunk는 최대한 공식 `yolo26n` pretrained를 재사용한다.
  - custom requirement는 head와 target/loss에서 해결한다.
- partial label 존중
  - BDD는 detector 중심
  - AIHUB traffic은 detector + TL attr 중심
  - AIHUB lane은 geometry 중심
- test-first
  - 큰 학습 전에 small fixture, tiny subset, overfit regression부터 통과시킨다.

## 코드 철학

- trunk와 head를 분리한다.
- parsing, encoding, model, loss, training loop, eval를 섞지 않는다.
- 데이터 포맷 변환과 학습 입력 변환을 분리한다.
- 사람이 디버그 가능한 산출물을 남긴다.
  - source inventory
  - conversion report
  - debug overlay

## 문서 운영 원칙

- `0_PRD.md`는 제품 범위를 정의한다.
- `9_EXECUTION_STATUS.md`는 살아있는 tracker다.
- 구현 단계가 바뀌면 status와 checklist를 먼저 바꾼다.
- obsolete 문서는 남겨두지 않는다.

## 품질 기준

- 최소 기준
  - py_compile 통과
  - unit test 통과
  - regression run 통과
- 구현 승인 기준
  - 문서와 코드가 모순되지 않을 것
  - loader/target/loss가 NaN 없이 동작할 것
  - pretrained trunk loading 전략이 재현 가능할 것

## 하지 않을 것

- 문서와 다른 임시 구현을 조용히 넣지 않는다.
- full dataset train 전에 giant refactor를 반복하지 않는다.
- raw dataset를 여러 버전으로 중복 생성하지 않는다.
- “일단 돌아가게”를 이유로 contract를 흐리지 않는다.
