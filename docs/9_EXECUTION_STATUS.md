# Execution Status

## 운영 규칙

- 이 문서는 구현 중 항상 갱신하는 live tracker다.
- 코드 변경 전에 해당 phase와 next action을 맞춘다.
- 테스트가 끝나면 결과와 날짜를 기록한다.

## 현재 기준

- 날짜: `2026-03-23`
- phase: `phase 1 pre-loader`
- current focus: `loader + target encoder 설계 진입 전 문서 재정리 완료`

## 완료된 항목

- [x] old pivot 문서 세트 제거
- [x] docs 전면 개정
- [x] `model/` 루트로 패키지 정리
- [x] `aihub_common.py + aihub_standardize.py` 구조 정리
- [x] AIHUB standardization pipeline 구현
- [x] source README 자동 생성
- [x] source inventory / conversion report 구현
- [x] debug overlay 구현
- [x] loss spec 코드/문서 반영
- [x] unit test 통과
- [x] real-data smoke 통과
- [x] git commit 생성

## 다음 작업

- [ ] standardized dataset loader 구현
- [ ] online resize/pad transform 구현
- [ ] target encoder 구현
- [ ] pretrained YOLOv26n trunk adapter 구현
- [ ] custom heads 구현
- [ ] multitask loss runtime 구현

## 최근 검증

- [x] `python3 -m unittest discover -s test -v`
- [x] `python3 -m model.preprocess.aihub_standardize --workers 2 --max-samples-per-dataset 4 --debug-vis-count 2`

## 최근 결정

- trunk는 official pretrained `yolo26n` 기준
- head는 PV26 custom implementation
- raw-space standardized dataset 유지
- `800x608` transform은 loader 단계 온라인 적용
- lane 학습/추론 계약은 AIHUB 기준 유지
