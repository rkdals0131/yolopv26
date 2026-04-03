# PV26 System Architecture

## 저장소 구조

```text
common/
  boxes.py
  io.py
  overlay.py
  paths.py
  pv26_schema.py
model/
  data/
    dataset.py
    preview.py
    sampler.py
    target_encoder.py
    transform.py
  net/
    heads.py
    trunk.py
  engine/
    batch.py
    det_geometry.py
    trainer_progress.py
    trainer_reporting.py
    trainer_runtime.py
    _loss_spec.py
    _det_geometry.py
    _trainer_checkpoint.py
    _trainer_epochs.py
    _trainer_fit.py
    _trainer_io.py
    _trainer_progress.py
    _trainer_reporting.py
    _trainer_step.py
    evaluator.py
    loss.py
    metrics.py
    postprocess.py
    spec.py
    trainer.py
tools/
  check_env/
    __init__.py
    actions.py
    launch.py
    scan.py
    tui.py
  check_env.py
  pv26_train/
    __init__.py
    artifacts.py
    cli.py
    config.py
    runtime.py
    scenario.py
    scenarios.py
  run_pv26_train.py
  pv26_train/artifacts.py
  pv26_train/config.py
  od_bootstrap/
    source/
      aihub/
        __init__.py
        debug.py
        lane_worker.py
        obstacle_worker.py
        reports.py
        source_meta.py
        traffic_worker.py
        worker_common.py
        workers.py
      shared/
        __init__.py
        debug.py
        io.py
        parallel.py
        raw.py
        reports.py
        resume.py
        scene.py
        source_meta.py
        summary.py
      aihub.py
      aihub_debug.py
      aihub_lane_worker.py
      aihub_obstacle_worker.py
      aihub_reports.py
      aihub_source_meta.py
      aihub_traffic_worker.py
      aihub_worker_common.py
      bdd100k.py
      constants.py
      defaults.py
      prepare.py
      raw_common.py
      shared_debug.py
      shared_io.py
      shared_parallel.py
      shared_raw.py
      shared_reports.py
      shared_scene.py
      shared_summary.py
      types.py
    build/
      checkpoint_audit.py
      debug_vis.py
      exhaustive_od.py
      final_dataset.py
      image_list.py
      review.py
      sample_manifest.py
      sweep.py
      teacher_dataset.py
    teacher/
      calibrate.py
      eval.py
      policy.py
      runtime/
        __init__.py
        artifacts.py
        callbacks.py
        progress.py
        resume.py
        tensorboard.py
        trainer.py
      train.py
      ultralytics_runner.py
    cli.py
    presets.py
test/
docs/
```

## 아키텍처 레이어

1. raw dataset layer
   - AIHUB raw dataset
   - BDD100K raw dataset
2. bootstrap pipeline layer
   - raw canonicalization
   - teacher dataset materialization
   - exhaustive OD materialization
   - final dataset build
3. runtime data layer
   - canonical outputs -> training sample runtime
   - variable dataset raw -> `800x608` online resize/pad
4. network layer
   - pretrained YOLOv26n backbone/neck
   - PV26 custom heads
5. engine layer
   - multitask loss
   - metrics / postprocess
   - trainer / evaluator

- `model/net`은 `trunk.py`와 `heads.py`로 분리돼 있고, `model/engine/trainer.py`는 `_trainer_*` helper들에 구현을 위임한다.
- `model/engine/det_geometry.py`, `model/engine/trainer_progress.py`, `model/engine/trainer_reporting.py`, `model/engine/trainer_runtime.py`는 public/shared engine helper surface고, underscore 파일은 implementation detail로 유지한다.
- `tools/run_pv26_train.py`는 stable thin entrypoint이고, 실제 구현은 `tools/pv26_train/` 패키지에 둔다.
- `tools/check_env.py`는 stable thin entrypoint이고, 실제 구현은 `tools/check_env/` 패키지에 둔다.
- `tools/od_bootstrap/source`는 `aihub/`와 `shared/` 패키지에 실제 구현을 두고, `bdd100k.py` / `prepare.py`가 coordinator 역할을 맡는다.
- `tools/od_bootstrap/teacher`는 `runtime/` 패키지에 runtime helper family를 두고, `ultralytics_runner.py`는 thin orchestration facade로 유지한다.

## 데이터 흐름

```text
AIHUB raw
  -> tools.od_bootstrap.source.aihub / bdd100k
  -> canonical bundle
  -> teacher dataset build
  -> teacher train / eval / calibrate
  -> build-exhaustive-od
  -> build-final-dataset
  -> model.data dataset
  -> model.net
  -> model.engine
```

## 현재 구현된 것

- bootstrap source prep / canonicalization pipeline
- bootstrap teacher dataset materialization
- bootstrap exhaustive OD materialization
- bootstrap final dataset build
- canonical dataset loader runtime
- shared online letterbox transform runtime
- target encoder runtime
- Ultralytics YOLO26 trunk adapter baseline
- PV26 custom heads skeleton
- multitask loss runtime
- task-aligned detector assignment runtime
- lane family Hungarian matching runtime
- trainer skeleton runtime
- evaluator skeleton runtime
- tiny overfit runtime
- source README generation
- source inventory / conversion report
- debug overlay generation
- loss design spec document + code mirror

## 아직 구현되지 않은 것

- export / ROS prediction bundle 정교화

## 운영 규칙

- `model/data`는 runtime dataset, transform, preview, sampler, target encoding을 다룬다.
- `model/net`은 trunk/head 구조를 다룬다.
- `model/engine`은 loss / metrics / postprocess / trainer / evaluator를 다룬다.
- `tools/od_bootstrap/source`는 raw canonicalization과 source typing을 다룬다.
- `tools/od_bootstrap/build`는 teacher dataset, exhaustive OD, final dataset, review/debug tooling을 다룬다.
