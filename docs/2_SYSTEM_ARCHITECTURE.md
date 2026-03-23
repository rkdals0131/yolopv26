# PV26 System Architecture

## 저장소 구조

```text
model/
  encoding/
    pv26_target_encoder.py
  loading/
    pv26_loader.py
    transform.py
  heads/
    pv26_heads.py
  trunk/
    ultralytics_yolo26.py
  preprocess/
    aihub_common.py
    aihub_standardize.py
    bdd100k_standardize.py
  viz/
    overlay.py
  loss/
    spec.py
tools/
test/
docs/
```

## 아키텍처 레이어

1. source dataset layer
   - AIHUB raw dataset
   - BDD100K raw dataset
2. standardization layer
   - AIHUB raw -> canonical scene JSON / det label / meta report
   - BDD100K raw -> canonical scene JSON / det label / meta report
3. loading layer
   - canonical outputs -> training sample runtime
   - variable dataset raw -> `800x608` online resize/pad
4. target encoding layer
   - detector target
   - TL 4-bit target
   - lane/stop-line/crosswalk vector target
5. model layer
   - pretrained YOLOv26n backbone/neck
   - PV26 custom heads
6. loss layer
   - multitask loss and partial-label masking
7. training/eval layer
   - sampler, schedule, logging, checkpoint, metrics

## 데이터 흐름

```text
AIHUB raw
  -> aihub_standardize
  -> standardized scene/det/meta
  -> dataset loader
  -> online resize/pad
  -> target encoder
  -> PV26 model
  -> multitask loss
  -> trainer / evaluator
```

## 현재 구현된 것

- AIHUB standardization pipeline
- BDD100K detection-only standardization pipeline
- canonical dataset loader runtime
- shared online letterbox transform runtime
- target encoder runtime
- Ultralytics YOLO26 trunk adapter baseline
- PV26 custom heads skeleton
- source README generation
- source inventory / conversion report
- debug overlay generation
- loss design spec document + code mirror

## 아직 구현되지 않은 것

- real multitask loss implementation
- trainer/evaluator

## 운영 규칙

- `preprocess/`는 raw data와 canonical data 사이 계약만 다룬다.
- `viz/`는 human QA를 위한 오버레이와 리포트만 다룬다.
- `loss/`는 target/loss contract를 다룬다.
- model trunk/head, training loop, evaluation 코드는 이후 새 모듈로 추가한다.
