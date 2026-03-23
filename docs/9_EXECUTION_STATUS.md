# Execution Status

## 운영 규칙

- 이 문서는 구현 중 항상 갱신하는 live tracker다.
- 코드 변경 전에 해당 phase와 next action을 맞춘다.
- 테스트가 끝나면 결과와 날짜를 기록한다.

## 현재 기준

- 날짜: `2026-03-23`
- phase: `phase 2 loader-runtime`
- current focus: `loader runtime 완료, target encoder 구현 진입`

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
- [x] sample contract 문서 고정
- [x] transform contract 문서 고정
- [x] detector-TL attr binding 규약 고정
- [x] query count 문서/코드 sync 정리
- [x] docs 상대경로 링크 정리
- [x] `N_gt_det / Q_det` 용어 분리
- [x] dataset raw / vehicle camera reference / network input 용어 분리
- [x] task별 invalid 처리 우선순위 고정
- [x] raw output / export bundle 계층 분리
- [x] BDD100K det-only canonical pipeline 구현
- [x] canonical dataset loader runtime 구현
- [x] shared online letterbox transform 구현
- [x] ragged sample collate 구현
- [x] unit test 통과
- [x] real-data smoke 통과
- [x] git commit 생성

## 다음 작업

- [ ] standardized dataset loader 구현
- [ ] target encoder 구현
- [ ] pretrained YOLOv26n trunk adapter 구현
- [ ] custom heads 구현
- [ ] multitask loss runtime 구현

## 최근 검증

- [x] `python3 -m unittest discover -s test -v`
- [x] `python3 -m model.preprocess.aihub_standardize --workers 2 --max-samples-per-dataset 4 --debug-vis-count 2`
- [x] `python3 -m model.preprocess.bdd100k_standardize --workers 8 --max-samples-per-split 2 --debug-vis-count 2`
- [x] `python3 -m unittest discover -s test -p 'test_pv26_loader.py' -v`
- [x] docs sync test 추가 후 `python3 -m unittest discover -s test -v` 재통과

## 최근 결정

- trunk는 official pretrained `yolo26n` 기준
- head는 PV26 custom implementation
- raw-space standardized dataset 유지
- `800x608` transform은 loader 단계 온라인 적용
- lane 학습/추론 계약은 AIHUB 기준 유지
- loader sample contract는 `image / det_targets / tl_attr_targets / lane_targets / source_mask / valid_mask / meta`로 고정
- lane/stop/crosswalk query count는 `12 / 6 / 4`로 고정
- TL attr supervision은 detector assignment 결과를 재사용
- `N_gt_det`는 GT row count, `Q_det`는 detector prediction slot count로 고정
- standardized dataset loader는 variable dataset raw에서 `800x608`으로 직접 transform한다
- loader 전에 BDD100K도 canonical standardization 레이어로 맞춘다
- BDD100K는 `det only` source로 쓰고 `trafficLightColor`는 scene hint로만 보존한다
- loader collate는 image만 stack하고 ragged target은 list 형태로 유지한다
