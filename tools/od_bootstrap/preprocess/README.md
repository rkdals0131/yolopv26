# OD Bootstrap Preprocess

이 디렉토리는 bootstrap 전처리 구현 메모다. 공식 진입점은 `python -m tools.od_bootstrap`다.

구성:
- `../cli.py`: `prepare-sources`, `build-teacher-datasets`, `generate-debug-vis` 서브커맨드 제공
- `../data/source_prep.py`: raw source 준비와 canonical bundle 생성
- `../data/teacher_dataset.py`: canonical output에서 teacher 학습용 YOLO dataset 생성
- `../data/debug_vis.py`: canonical, teacher, exhaustive debug overlay 렌더링 헬퍼
- `../data/sample_manifest.py`: bootstrap 이미지 샘플 manifest 선택
- `../data/review.py`: final dataset review bundle 렌더링
- `../data/checkpoint_audit.py`: teacher checkpoint alias / scale audit

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
- `build-teacher-datasets`는 sample 단위 멀티스레드(`runtime.workers`)로 image/link copy와 label materialization을 병렬 처리한다
- 진행 로그는 stderr로 출력되며 `runtime.log_every` 샘플마다 throughput과 detection 누계를 갱신한다

related review helpers:
- `../data/sample_manifest.py`: bootstrap image list selection helper
- `../data/review.py`: final dataset review bundle 렌더링
- `../data/checkpoint_audit.py`: teacher checkpoint alias / scale audit
