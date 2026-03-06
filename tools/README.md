# Tools Catalog

스크립트를 목적별 하위 디렉토리로 분류해 관리합니다.

## data_analysis
- `tools/data_analysis/bdd/`: BDD100K 분석/표준화 스크립트
- `tools/data_analysis/etri/`: ETRI 분석/표준화 스크립트
- `tools/data_analysis/rlmd/`: RLMD 분석/표준화 스크립트
- `tools/data_analysis/wod/`: Waymo v2 분석/추출/표준화 스크립트
- `tools/data_analysis/run_multidataset_normalize_interactive.py`: BDD/ETRI/RLMD/WOD 선택형 인터랙티브 일괄 실행기

### bdd
- `tools/data_analysis/bdd/convert_bdd_type_a.py`: BDD100K -> PV26 Type-A 변환
- `tools/data_analysis/bdd/validate_pv26_dataset.py`: 변환 결과 무결성 검사
- `tools/data_analysis/bdd/pv26_qc_report.py`: split/channel 통계 리포트 생성
- `tools/data_analysis/bdd/run_bdd100k_normalize_interactive.py`: 변환/검증/QC 인터랙티브 실행기
- `tools/data_analysis/bdd/dataset_label_inventory.py`: 데이터셋 라벨/채널 인벤토리 분석

### etri
- `tools/data_analysis/etri/convert_etri_type_a.py`: ETRI(Mono+Multi) polygon JSON -> PV26 Type-A 변환

### rlmd
- `tools/data_analysis/rlmd/convert_rlmd_type_a.py`: RLMD(1080p + AC labeled) -> PV26 Type-A 변환

### wod
- `tools/data_analysis/wod/extract_wod_v2_sample.py`: Waymo v2 parquet 샘플 추출
- `tools/data_analysis/wod/convert_wod_type_a.py`: Waymo v2 parquet -> PV26 Type-A 변환(minimal 우선)

## debug
- `tools/debug/render_pv26_debug_masks.py`: DA/Lane 마스크 디버그 시각화
- `tools/debug/render_weights_example.py`: pv2/pv26 weight 출력 시각화
- `tools/debug/compare_weight_io.py`: best.pt vs yolopv2.pt I/O/feature 비교

## train
- `tools/train/train_pv26.py`: 학습 실행 스크립트(옵션/실행 흐름)
- `tools/train/common.py`: 학습 공통 함수(device/seed/metric/decode 유틸)
- `tools/train/modal_train_pv26.py`: Modal 원격 학습 래퍼(dir 우선 + tar 자동해제)
