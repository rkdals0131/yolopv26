# OD Bootstrap

`tools/od_bootstrap/`는 `7-class OD exhaustive supervision`을 만들기 위한 전처리, teacher 학습, calibration, exhaustive OD materialization, 최종 병합 파이프라인이다.

범위:
- teacher 학습 대상 source는 `bdd100k_det_100k`, `aihub_traffic_seoul`, `aihub_obstacle_seoul`
- teacher 기본 모델은 mobility/signal은 `yolo26s.pt`, obstacle은 `yolo26m.pt`다
- `PV26Heads`, `PV26Trainer`, `run_pv26_train.py`는 teacher 학습에 쓰지 않는다
- lane teacher는 없다
- lane은 exhaustive OD materialization과 분리해서 final dataset 단계에서 canonical AIHUB output을 그대로 합친다

구조:
- `data/`
  - tracked Python package다. repo-root artifact용 `.gitignore`의 `/data/`와는 별개다
  - raw BDD/AIHUB를 canonicalize
  - bootstrap용 image list 생성
  - teacher train dataset 3개 생성
  - debug-vis 생성
  - 주요 모듈: `source_prep.py`, `teacher_dataset.py`, `debug_vis.py`, `sample_manifest.py`, `review.py`, `checkpoint_audit.py`
- `teacher/`
  - `mobility`, `signal`, `obstacle` teacher를 direct Ultralytics YOLO로 파인튜닝
  - checkpoint evaluation
  - class policy calibration
- `cli.py`
  - `python -m tools.od_bootstrap` 단일 진입점
- `presets.py`
  - bootstrap preset 생성
  - checked-in YAML 대신 code preset만 유지

기본 실행 순서:
1. `python -m tools.od_bootstrap prepare-sources`
2. `python -m tools.od_bootstrap build-teacher-datasets`
3. `python -m tools.od_bootstrap train --teacher mobility`
4. `python -m tools.od_bootstrap train --teacher signal`
5. `python -m tools.od_bootstrap train --teacher obstacle`
6. `python -m tools.od_bootstrap eval --teacher mobility`
7. `python -m tools.od_bootstrap eval --teacher signal`
8. `python -m tools.od_bootstrap eval --teacher obstacle`
9. `python -m tools.od_bootstrap calibrate`
10. `python -m tools.od_bootstrap build-exhaustive-od`
11. `python -m tools.od_bootstrap build-final-dataset`
12. `python3 tools/run_pv26_train.py --preset default`

출력:
- canonical intermediate: `seg_dataset/pv26_od_bootstrap/canonical/`
- bootstrap image list: `seg_dataset/pv26_od_bootstrap/meta/bootstrap_image_list.jsonl`
- teacher datasets: `seg_dataset/pv26_od_bootstrap/teacher_datasets/`
- teacher runs: `runs/od_bootstrap/train/<teacher>/`
- teacher eval: `runs/od_bootstrap/eval/<teacher>/`
- calibrated class policy: `runs/od_bootstrap/calibration/default/class_policy.yaml`
- calibration report: `runs/od_bootstrap/calibration/default/calibration_report.json`
- hard-negative regression set: `runs/od_bootstrap/calibration/default/hard_negative_manifest.json`
- exhaustive OD run metadata: `runs/od_bootstrap/<run_id>/`
- exhaustive OD dataset: `seg_dataset/pv26_od_bootstrap/exhaustive_od/<run_id>/`
- final merged dataset: `seg_dataset/pv26_exhaustive_od_lane_dataset/`
- PV26 final train outputs: `runs/pv26_exhaustive_od_lane_train/<meta_run_name>/`

provenance 필드:
- `label_origin`
- `teacher_name`
- `confidence`
- `model_version`
- `run_id`
- `created_at`

최종 PV26 학습:
- final merged dataset root는 `tools/run_pv26_train.py --preset default`가 사용하는 `seg_dataset/pv26_exhaustive_od_lane_dataset`다
- exhaustive OD source key 3개와 canonical AIHUB output을 같이 사용한다
- 목표는 `PV26 7-class OD head`가 `allow_objectness_negatives=True` 조건의 exhaustive OD supervision을 받도록 만드는 것이다

참고:
- `build-exhaustive-od` preset은 calibration 산출물인 `class_policy.yaml`을 읽는다
- calibration preset은 `hard_negative_manifest.json`을 재사용할 수 있다
- final dataset preset은 exhaustive OD 최신 run을 자동으로 집어온다
- materialized exhaustive OD 산출물은 `sample_uid = <dataset_key>__<split>__<sample_id>` 기준으로 파일명을 만든다
