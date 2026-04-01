# OD Bootstrap

`tools/od_bootstrap/`는 `7-class OD exhaustive supervision`을 만들기 위한 전처리, teacher 학습, calibration, sweep, 최종 병합 파이프라인이다.

범위:
- teacher 학습 대상 source는 `bdd100k_det_100k`, `aihub_traffic_seoul`, `aihub_obstacle_seoul`
- teacher 기본 모델은 mobility/signal은 `yolo26s.pt`, obstacle은 `yolo26m.pt`다
- `yolo26n.pt` 프리셋도 같은 디렉토리에 남아 있지만, 현재 default는 아니다
- `PV26Heads`, `PV26Trainer`, `run_pv26_train.py`는 teacher 학습에 쓰지 않는다
- lane teacher는 없다
- lane은 bootstrap sweep과 분리해서 finalize 시점에 canonical AIHUB output을 그대로 합친다

구조:
- `preprocess/`
  - raw BDD/AIHUB를 canonicalize
  - bootstrap용 image list 생성
  - teacher train dataset 3개 생성
  - debug-vis 생성 CLI 포함
- `train/`
  - `mobility`, `signal`, `obstacle` teacher를 direct Ultralytics YOLO로 파인튜닝
- `eval/`
  - teacher checkpoint에 대해 `predict`/`val` 기반 checkpoint evaluation 실행
- `calibration/`
  - teacher val split prediction을 기준으로 `class_policy.yaml`을 자동 보정
  - class별 false positive 장면을 `hard_negative_manifest.json`으로 남겨 다음 calibration의 회귀 세트로 재사용
- `sweep/`
  - 세 teacher를 `model-centric`으로 전체 OD image list에 순차 적용
  - raw source label 우선 정책으로 teacher prediction을 합쳐 `7-class exhaustive OD dataset` 생성
  - 모든 bootstrap box에는 provenance를 남김
- `finalize/`
  - exhaustive OD dataset과 canonical AIHUB output을 합쳐 최종 `pv26_exhaustive_od_lane_dataset` 생성
- `smoke/`
  - bootstrap image-list subset, checkpoint audit, smoke review bundle 생성

기본 실행 순서:
1. `python3 tools/od_bootstrap/preprocess/run_prepare_sources.py --config tools/od_bootstrap/config/preprocess/sources.default.yaml`
2. `python3 tools/od_bootstrap/preprocess/run_build_teacher_datasets.py --config tools/od_bootstrap/config/preprocess/teacher_datasets.default.yaml`
3. `python3 tools/od_bootstrap/train/run_train_teacher.py --config tools/od_bootstrap/config/train/mobility_yolo26s.default.yaml`
4. `python3 tools/od_bootstrap/train/run_train_teacher.py --config tools/od_bootstrap/config/train/signal_yolo26s.default.yaml`
5. `python3 tools/od_bootstrap/train/run_train_teacher.py --config tools/od_bootstrap/config/train/obstacle_yolo26m.default.yaml`
6. `python3 tools/od_bootstrap/eval/run_teacher_checkpoint_eval.py --config tools/od_bootstrap/config/eval/mobility_checkpoint_eval.default.yaml`
7. `python3 tools/od_bootstrap/eval/run_teacher_checkpoint_eval.py --config tools/od_bootstrap/config/eval/signal_checkpoint_eval.default.yaml`
8. `python3 tools/od_bootstrap/eval/run_teacher_checkpoint_eval.py --config tools/od_bootstrap/config/eval/obstacle_checkpoint_eval.default.yaml`
9. `python3 tools/od_bootstrap/calibration/run_calibrate_class_policy.py --config tools/od_bootstrap/config/calibration/class_policy.default.yaml`
10. `python3 tools/od_bootstrap/sweep/run_model_centric_sweep.py --config tools/od_bootstrap/config/sweep/model_centric.default.yaml`
11. `python3 tools/od_bootstrap/finalize/run_build_exhaustive_od_lane_dataset.py --config tools/od_bootstrap/config/finalize/pv26_exhaustive_od_lane.default.yaml`
12. `python3 tools/run_pv26_train.py --config config/pv26_meta_train.default.yaml`

출력:
- canonical intermediate: `seg_dataset/pv26_od_bootstrap/canonical/`
- bootstrap image list: `seg_dataset/pv26_od_bootstrap/meta/bootstrap_image_list.jsonl`
- teacher datasets: `seg_dataset/pv26_od_bootstrap/teacher_datasets/`
- teacher runs: `runs/od_bootstrap/train/<teacher>/`
- teacher eval: `runs/od_bootstrap/eval/<teacher>/`
- calibrated class policy: `runs/od_bootstrap/calibration/default/class_policy.yaml`
- calibration report: `runs/od_bootstrap/calibration/default/calibration_report.json`
- hard-negative regression set: `runs/od_bootstrap/calibration/default/hard_negative_manifest.json`
- sweep run metadata: `runs/od_bootstrap/<run_id>/`
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
- final merged dataset root는 `config/pv26_meta_train.default.yaml`의 `dataset.root`다
- exhaustive OD source key 3개와 canonical AIHUB output을 같이 사용한다
- 목표는 `PV26 7-class OD head`가 `allow_objectness_negatives=True` 조건의 exhaustive OD supervision을 받도록 만드는 것이다

참고:
- sweep 기본 config는 calibration 산출물인 `class_policy.yaml`을 읽는다
- calibration 기본 config는 `hard_negative_manifest.json`을 재사용할 수 있다
- finalize는 `tools/od_bootstrap/config/finalize/pv26_exhaustive_od_lane.default.yaml`의 `latest` 입력을 사용해 exhaustive OD 최신 run을 집어온다
- manifest-only 검증이 필요하면 `tools/od_bootstrap/config/sweep/model_centric.dryrun.yaml`을 사용한다
- materialized exhaustive OD 산출물은 `sample_uid = <dataset_key>__<split>__<sample_id>` 기준으로 파일명을 만든다
