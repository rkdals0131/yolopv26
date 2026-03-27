# OD Bootstrap Preprocess

이 디렉토리는 `od_bootstrap` 전처리 전용이다.

구성:
- `run_prepare_sources.py`: BDD100K, AIHUB traffic, AIHUB obstacle canonicalization
- `run_build_teacher_datasets.py`: mobility, signal, obstacle teacher dataset materialization
- `sources.py`: raw source 준비와 canonical bundle 생성
- `teacher_dataset.py`: canonical output에서 teacher 학습용 YOLO dataset 생성

bootstrap 범위:
- 포함: `bdd100k_det_100k`, `aihub_traffic_seoul`, `aihub_obstacle_seoul`
- 제외: `aihub_lane_seoul`

teacher dataset:
- mobility: `vehicle`, `bike`, `pedestrian`
- signal: `traffic_light`, `sign`
- obstacle: `traffic_cone`, `obstacle`
