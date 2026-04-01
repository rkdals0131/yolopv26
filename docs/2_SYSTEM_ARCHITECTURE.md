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
    evaluator.py
    loss.py
    metrics.py
    postprocess.py
    trainer.py
tools/
  check_env.py
  run_pv26_train.py
  od_bootstrap/
    source/
      aihub.py
      bdd100k.py
      prepare.py
      raw_common.py
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
