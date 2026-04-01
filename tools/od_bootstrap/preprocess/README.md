# OD Bootstrap Preprocess

이 디렉토리는 `od_bootstrap` 전처리 전용이다.

구성:
- `run_prepare_sources.py`: BDD100K와 AIHUB canonical source bundle 생성, bootstrap image list 생성
- `run_build_teacher_datasets.py`: mobility, signal, obstacle teacher dataset materialization
- `run_generate_debug_vis.py`: canonical / teacher-dataset / exhaustive debug-vis 생성
- `sources.py`: raw source 준비와 canonical bundle 생성
- `teacher_dataset.py`: canonical output에서 teacher 학습용 YOLO dataset 생성
- `debug_vis.py`: canonical, teacher, exhaustive debug overlay 렌더링 헬퍼

출력 규약:
- bootstrap image list는 sweep collision 방지를 위해 `sample_id`와 `sample_uid`를 함께 기록한다
- teacher dataset 단계는 run-scoped `data.yaml`을 만들지 않는다
- teacher/eval 실행 시 필요한 `data.yaml`은 각 run output 아래에서 별도로 생성된다

bootstrap 범위:
- 포함: `bdd100k_det_100k`, `aihub_traffic_seoul`, `aihub_obstacle_seoul`
- 제외: `aihub_lane_seoul`

teacher dataset:
- mobility: `vehicle`, `bike`, `pedestrian`
- signal: `traffic_light`, `sign`
- obstacle: `traffic_cone`, `obstacle`

runtime:
- `run_build_teacher_datasets.py`는 sample 단위 멀티스레드(`runtime.workers`)로 image/link copy와 label materialization을 병렬 처리한다
- 진행 로그는 stderr로 출력되며 `runtime.log_every` 샘플마다 throughput과 detection 누계를 갱신한다

related smoke helpers:
- `../smoke/run_build_smoke_image_list.py`: bootstrap image list smoke subset 생성
- `../smoke/run_render_smoke_review.py`: final dataset smoke review bundle 렌더링
- `../smoke/run_audit_teacher_checkpoints.py`: teacher checkpoint alias / scale audit
