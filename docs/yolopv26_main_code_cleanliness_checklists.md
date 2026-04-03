# YOLOPV26 main 코드 청결성 수정 체크리스트

이 문서는 [리포트](yolopv26_main_code_cleanliness_report.md)의 지적사항을 빠짐없이 액션 아이템으로 옮긴 체크리스트다.
정렬 기준은 중요도 순서이며, 아래로 갈수록 우선순위가 낮을 뿐 누락된 항목은 없다.

## 1순위. `tools/od_bootstrap/source/` 경계 정리와 private cross-import 제거

- [x] `tools/od_bootstrap/source/bdd100k.py`가 `tools/od_bootstrap/source/aihub.py`의 private/internal helper(`LiveLogger`, `_iter_task_chunks`, `_parallel_chunk_size`, `_default_workers`, `_generate_debug_vis`, `_link_or_copy`, `_write_json`, `_write_text`, `_bbox_to_yolo_line`, `_counter_to_dict`)를 직접 import하지 않도록 정리한다.
- [x] `tools/od_bootstrap/source/aihub.py`와 `tools/od_bootstrap/source/bdd100k.py`를 형제 pipeline coordinator로 분리하고, 서로의 implementation detail에 의존하지 않게 만든다.
- [x] `tools/od_bootstrap/source/aihub_lane_worker.py`, `tools/od_bootstrap/source/aihub_traffic_worker.py`, `tools/od_bootstrap/source/aihub_obstacle_worker.py`가 `raw_common.py`나 `aihub_worker_common.py`의 private helper(`_base_scene`, `_counter_to_dict`, `_sample_id`, `_extract_annotations`, `_safe_slug`, `_normalize_text`, `_load_json`)를 직접 import하지 않도록 정리한다.
- [x] `tools/od_bootstrap/source/` 아래에 `shared_io.py`, `shared_parallel.py`, `shared_scene.py`, `shared_summary.py` 또는 동등한 public shared 모듈을 두고, cross-module 재사용이 필요한 helper는 underscore를 떼서 승격한다.
- [x] `tools/od_bootstrap/source/types.py`를 pure types 모듈로 되돌리고, 기본 경로 상수나 implementation defaults는 `constants.py` 또는 별도 defaults 모듈로 분리한다.
- [x] `shared_parallel.py`로 승격된 `LiveLogger`/parallel chunking 위에, existing output summary skeleton을 `shared_resume.py`로, BDD README/tree/source inventory render를 `shared_source_meta.py`로 재배치한다.
- [x] source pipeline 공통 출력 패턴 중 debug-vis manifest write helper를 `shared_debug.build_debug_vis_manifest()`와 typed manifest rows로 shared public API에 재배치한다.
- [x] `tools/od_bootstrap/source/aihub.py: run_standardization()`의 큰 orchestration 흐름을 단계별 helper로 쪼개 회귀 위험을 줄인다.
- [x] `tools/od_bootstrap/source/bdd100k.py: run_standardization()`의 큰 orchestration 흐름을 단계별 helper로 쪼개 회귀 위험을 줄인다.

## 2순위. `tools/run_pv26_train.py` 분해

- [x] `tools/run_pv26_train.py`를 stable thin facade로 유지하고, CLI / import surface(`load_meta_train_scenario`, `load_meta_train_resume_scenario`, `run_stage3_vram_stress`, `run_meta_train_scenario`, `main`)는 그대로 노출한다.
- [x] preset 조립, scenario 로딩, scenario snapshot, resume recovery를 `tools/pv26_train_scenario.py` 계열로 분리한다.
- [x] phase transition / runtime orchestration은 `tools/pv26_train_runtime.py` 계열로 분리하고, stage 3 VRAM stress probe는 `tools/pv26_train_stress.py` 계열로 분리한다.
- [x] extraction regression gate를 `test/test_run_pv26_train.py`, `test/test_portability_runtime.py`, `test/test_docs_sync.py`로 고정한다.
- [x] `tools/run_pv26_train.py`가 `tools/pv26_train_config.py`와 `tools/pv26_train_artifacts.py`에서 underscore alias를 대량으로 끌어오는 구조를 정리하고, local helper와 외부 public API의 경계를 명확히 한다.
- [x] `tools/run_pv26_train.py`의 meta-train preset assembly와 runtime execution을 서로 다른 모듈로 분리한다.
- [x] `tools/run_pv26_train.py`의 `site.addsitedir(REPO_ROOT)` 의존을 줄이거나 제거해 packaging/entrypoint 경계를 명확히 한다.
- [x] `tools/run_pv26_train.py: run_meta_train_scenario()`를 phase/helper 단위로 쪼개 회귀 위험을 줄인다.

## 3순위. 공통 helper 추출과 중복 제거

- [x] `deep_merge_mappings`를 `common.user_config` public helper로 승격하고 `tools/pv26_train_config.py`가 이를 재사용하도록 정리한다.
- [x] `resolve_latest_root`, `resolve_optional_path`를 `common.paths` 공용 helper로 두고 `tools/od_bootstrap/build/`, `tools/pv26_train_config.py`가 이를 재사용하도록 정리한다.
- [x] `common.io.write_jsonl`을 추가하고 build 내부의 안전한 JSONL/JSON writer call-site가 이를 재사용하도록 정리한다.
- [ ] repo 전반에 흩어진 `now_iso`, `timestamp_token`, `write_json`, `append_jsonl`를 공통화한다.
- [ ] `common/`에 이미 받아줄 자리가 있는데도 로컬 중복으로 남아 있는 `write_json`, `append_jsonl`, `now_iso`, `timestamp_token`를 흡수한다.
- [ ] `common/`은 대공사 대상이 아니라 새 shared helper를 받아주는 착지점으로 사용한다.
- [ ] repo-wide truly common helper는 `common/`으로 올리고, bootstrap 내부에서만 재사용하는 helper는 `tools/od_bootstrap/...` shared 모듈에 두는 기준을 명확히 한다.
- [x] `common/io.py`, `common/paths.py` 확장 방향으로 정리하고 build/PV26 call-site가 그 공용 helper를 재사용하게 맞춘다.
- [x] `build/sweep.py`, `build/debug_vis.py`, `build/teacher_dataset.py`, `source/prepare.py`가 semantics-compatible `common.io` helper(`now_iso`, `timestamp_token`, `write_json`)를 재사용하도록 정리한다.
- [x] `tools/od_bootstrap/build/` 내부의 `write_json`, `resolve_latest_root`, `resolve_optional_path` 같은 low-level IO/path helper 중 공통화 가능한 부분을 shared helper로 정리한다.
- [ ] `link_or_copy`류 helper는 symlink fallback, hardlink/copy, overwrite 금지, existing이면 skip 같은 정책 차이를 유지하고, low-level atomic/json/helper만 공통화한다.
- [x] source pipeline 쪽에서 existing output summary skeleton, debug-vis manifest write, README/tree markdown render 같은 공통 출력 패턴을 재사용 가능한 helper로 정리한다.

## 4순위. `model/engine/` 내부 API 경계 정리

- [x] `model/engine/trainer.py`가 `_trainer_checkpoint.py`, `_trainer_epochs.py`, `_trainer_fit.py`, `_trainer_io.py`, `_trainer_reporting.py`, `_trainer_step.py`의 private helper를 대량 re-export/alias 하는 구조를 줄인다.
- [ ] `model/engine/`에서 public surface와 internal surface를 분리하고, underscore helper는 실제로 파일 내부 전용이 되도록 정리한다.
- [ ] `model/engine/` 안의 private cross-import mesh를 줄이고, shared internal API가 필요하면 public shared 모듈로 승격한다.
- [x] `model/engine/loss.py`와 `model/engine/postprocess.py`의 `_make_anchor_grid`, `_decode_anchor_relative_boxes` 중복을 공용 geometry helper로 정리한다.
- [x] `model/engine/trainer.py`와 `model/engine/evaluator.py`의 `_move_to_device` 중복을 공용 batch helper로 정리한다.
- [x] `model/engine/_trainer_epochs.py`와 `model/engine/evaluator.py`의 `_raw_batch_for_metrics`, `_augment_lane_family_metrics` 중복을 공용 helper로 정리한다.
- [x] anchor grid 생성, anchor-relative box decode 같은 engine 공통 geometry/helper를 `model/engine/_det_geometry.py` shared 모듈로 재배치한다.
- [x] raw batch unwrap, lane family metric augmentation, move-to-device 같은 engine 공통 batch/helper를 `model/engine/batch.py`로 재배치한다.
- [ ] `model/engine/_trainer_epochs.py`에 몰린 epoch-level runtime/reporting helper를 정리해 파일 덩치를 줄인다.
- [x] `model/engine/_loss_spec.py: build_loss_spec()`의 큰 spec builder 흐름을 단계별 helper/contract로 분리해 회귀 범위를 줄인다.

## 5순위. manifest row와 느슨한 dict 계약 축소

- [ ] `tools/od_bootstrap/build/debug_vis.py`, `teacher_dataset.py`, `exhaustive_od.py`, `final_dataset.py`, `sweep.py`에서 JSON manifest row를 `dict[str, Any]` 중심으로 느슨하게 돌리는 구조를 줄인다.
- [x] `debug_vis.py`, `teacher_dataset.py`에서 `DebugVisItemRow`/`TeacherDatasetManifestRow` 수준의 `TypedDict`를 도입한다.
- [x] 최소한 `ExhaustiveSampleRow`, `FinalDatasetSampleRow`, `TeacherPredictionRow` 수준의 `TypedDict`를 더 도입한다.
- [ ] manifest row, sample row, prediction row, summary row에서 key typo, optional field 누락, value 타입 drift, 경로/string 혼합을 정적 계약으로 잡을 수 있게 만든다.
- [ ] source/build 파이프라인에서 자주 오가는 JSON row 계약을 명시적인 타입으로 치환해 IDE 추적성과 리팩토링 안전성을 높인다.

## 6순위. teacher runtime과 `ultralytics_runner.py` 정리

- [ ] `tools/od_bootstrap/teacher/ultralytics_runner.py`에서 dataloader kwargs, progress helper, tensorboard helper, resume helper, artifact refresh helper, callback builder, trainer subclass를 한 파일에서 모두 감싸는 구조를 분해한다.
- [x] `tools/od_bootstrap/teacher/ultralytics_runner.py`가 `runtime_progress.py`, `runtime_tensorboard.py`, `runtime_resume.py`, `runtime_artifacts.py`의 private helper를 alias import하는 구조를 public shared API 중심으로 바꾼다.
- [x] `tools/od_bootstrap/teacher/build_teacher_runtime_callbacks()`의 과도한 dependency injection 인자를 `TeacherRuntimeSupport` 객체로 묶는다.
- [x] `tools/od_bootstrap/teacher/calibrate.py: calibrate_class_policy_scenario()`의 큰 orchestration 흐름을 단계별 helper로 쪼갠다.
- [x] `tools/od_bootstrap/teacher/ultralytics_runner.py: _make_teacher_trainer()`의 큰 orchestration 흐름을 단계별 helper로 쪼갠다.

## 7순위. teacher runtime과 PV26 trainer runtime의 공통 runtime helper 정리

- [ ] teacher runtime과 PV26 trainer runtime에 중복된 `format_duration`, `sync_timing_device`, tensorboard writer build, tensorboard scalar write, timing profile 요약, progress rendering helper를 공용 runtime helper 계층으로 정리한다.
- [ ] `common/train_runtime.py` 또는 동등한 공용 레이어를 두고 duration formatting, device sync timing, scalar flatten/write, rolling timing summary, progress helper를 모은다.

## 8순위. `tools/check_env.py` 파일 경계 정리

- [x] `tools/check_env.py`에서 env check, manifest parsing, workspace scan, action catalog, blockers/advisory 계산, rich TUI render를 `check_env_scan.py`, `check_env_actions.py`, `check_env_tui.py`로 분리한다.
- [x] `tools/check_env.py`의 input handling, subprocess action launch, resume candidate handling을 `check_env_launch.py` 레이어로 추가 분리한다.
- [x] `tools/check_env.py`를 entrypoint/compat facade로 더 경량화하고 `check_env_launch.py` 파일 경계까지 정리한다.

## 9순위. `tools/od_bootstrap/build/` 마감 정리

- [ ] `tools/od_bootstrap/build/`의 manifest/helper naming consistency를 맞춘다.
- [ ] `tools/od_bootstrap/build/`는 구조 자체보다 manifest typing, low-level IO helper 정리, naming consistency에 집중해서 손본다.

## 가드레일

- `model/data/target_encoder.py::_resample_points`와 `model/engine/metrics.py::_resample_points`는 backend가 다르므로 무리한 구현 통합 대신 공용 계약과 테스트 공유 수준까지만 맞춘다.
- source worker의 lane/obstacle/traffic annotation parsing은 skeleton/helper 수준까지만 공통화하고, 실제 annotation semantics 해석은 worker별로 분리 유지한다.
- teacher runtime과 PV26 trainer runtime의 전체 runner abstraction은 통합하지 않고, tensorboard/progress/timing helper 수준만 공유한다.
- `model/engine/loss.py`는 multitask loss 응집 자체를 깨기보다, anchor/geometry helper 중복만 공용화하는 방향으로 손본다.
- `model/`이나 `tools/od_bootstrap/` 상위 폴더를 다시 뒤집는 top-level 구조 개편은 하지 않는다.
- backend, 프레임워크, annotation semantics가 다른 모든 중복을 억지로 하나로 합치지 않는다.
- bootstrap 내부 전용 helper까지 전부 `common/`으로 몰아넣지 않는다.
- 현재 config 체계를 다시 흔드는 리팩토링은 우선순위에서 제외한다.
