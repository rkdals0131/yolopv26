# Implementation Plan

## phase 0 완료

- AIHUB standardization
- source README
- conversion report
- debug overlay
- loss design spec

## phase 1 implementation record

- 아래 항목은 원래 phase 1 scope였고, 현재는 구현 완료된 runtime 기록으로 남긴다.
- loader / transform / target encoder / trunk / heads / loss / trainer / evaluator는 모두 `model/`과 `tools/` 경로에서 연결돼 있다.

### 0. full-run hardening

- implemented: AIHUB / BDD standardization resume scan
- implemented: failure manifest / QA summary
- implemented: `tools/od_bootstrap/source` shared helper split and `aihub` / `bdd100k` orchestration cleanup
- implemented: `tools/run_pv26_train.py` orchestration split into `tools/pv26_train_config.py` and `tools/pv26_train_artifacts.py`
- implemented: `model/engine` trainer helper split into `_trainer_checkpoint.py`, `_trainer_epochs.py`, `_trainer_fit.py`, `_trainer_io.py`, `_trainer_reporting.py`, `_trainer_step.py`
- implemented: trainer AMP / grad accumulation / grad clip
- implemented: trainer auto resume / non-finite / OOM guard
- implemented: train command
- implemented: AIHUB `도로장애물·표면 인지 영상(수도권)` det-only source integration
  - scope is `traffic_cone / obstacle` only
  - `Person / Manhole / Pothole on road / Filled pothole` stay excluded from standardization output
  - loader det supervision follows the same scope

### 1. training sample runtime

- implemented via the [4A_SAMPLE_AND_TRANSFORM_CONTRACT.md](4A_SAMPLE_AND_TRANSFORM_CONTRACT.md) contract
- loader runtime materializes the sample dictionary schema exactly
- image, det, tl bits, lane family fields are materialized to contract shape

### 2. online transform

- implemented via the [4A_SAMPLE_AND_TRANSFORM_CONTRACT.md](4A_SAMPLE_AND_TRANSFORM_CONTRACT.md) contract
- variable dataset raw -> `800x608` preprocessing is encoded in code
- `800x600 -> 800x608` remains the vehicle camera reference special case
- train/infer shared transform is in place
- raw-space labels are converted into transformed-space sample targets

### 3. target encoder

- implemented:
  - detector target encoder
  - TL bit target encoder
  - lane vector encoder
  - stop-line encoder
  - crosswalk encoder

### 4. trunk adapter

- implemented:
  - official `yolo26n.pt` load path
  - backbone/neck parameter mapping
  - partial load verification

### 5. custom heads

- implemented:
  - det head
  - TL attr head
  - lane head
  - stop-line head
  - crosswalk head

### 6. loss runtime

- implemented:
  - spec translated into the runtime loss module
  - masking / matching / normalization logic

### 7. trainer skeleton

- implemented:
  - dataset-balanced sampler
  - stage-wise freeze policy
  - optimizer group separation
  - logging / checkpointing
  - stage config
  - optimizer groups
  - 1-step train runtime
  - dataset-balanced batch sampler helper
  - checkpoint save/load
  - history summary / JSONL logging
  - full epoch fit loop
  - val loop
  - best / last checkpoint write
  - run summary output
  - AMP
  - grad accumulation
  - grad clip
  - auto resume
  - non-finite / OOM guard

### 8. evaluator skeleton

- implemented:
  - detector metrics
  - TL bit metrics
  - lane family metrics
  - batch loss summary
  - GT count summary
  - raw output -> prediction bundle postprocess decode
  - batch-level detector/TL/lane family metric summary

## current runtime snapshot

- the end-to-end loop is closed from offline standardization through loader / train / loss / inference / evaluation
- tiny overfit regression and full epoch fit regression both exist and are used as regression gates
- OD bootstrap is implemented as a separate pipeline with `prepare-sources / build-teacher-datasets / train / eval / calibrate / build-exhaustive-od / build-final-dataset` stages
- `tools/run_pv26_train.py` stays a single CLI entrypoint, but preset/config handling and manifest/writeout concerns are split into helper modules
- `model/engine/trainer.py` is orchestration-only; step / epoch / fit / checkpoint / reporting logic lives in helper modules
- `2026-04-03` team wave added `model/engine/batch.py`, promoted `deep_merge_mappings` reuse through `common.user_config`, and extracted source shared helpers into `tools/od_bootstrap/source/shared_resume.py` and `shared_source_meta.py`
- `2026-04-03` follow-up wave added `model/engine/_det_geometry.py`, split `tools/check_env.py` into scan/actions/tui companions, tightened teacher runtime imports around public/shared helpers, and expanded `common.paths` / `common.io` reuse across bootstrap build call sites

## priority-2b extraction review boundary

- `tools/run_pv26_train.py` should remain the stable thin facade for the CLI and the import surface exercised by `test/test_run_pv26_train.py`.
- completed on `2026-04-03`: `tools/run_pv26_train.py` no longer mass-imports underscore helpers from `tools/pv26_train_config.py` and `tools/pv26_train_artifacts.py`; internal reads go through public module APIs and the old underscore import surface survives only as local compatibility wrappers.
- scenario / resume recovery responsibilities are the next extraction target:
  - preset lookup and scenario validation
  - scenario snapshot materialization for new runs
  - `meta_manifest.json` loading, snapshot restore, and legacy resume compatibility checks
- runtime / stress responsibilities are the next extraction target:
  - exact-resume CLI dispatch
  - stage 3 VRAM stress configuration / probe / summary assembly
  - keeping the normal meta-train path and stress path on the same user-facing CLI contract
- the regression bar for this split is fixed to:
  - `test/test_run_pv26_train.py`
  - `test/test_portability_runtime.py`
  - `test/test_docs_sync.py`

## implementation order

1. loader
2. transform
3. target encoder
4. trunk adapter
5. heads
6. loss
7. trainer
8. evaluator
9. tiny overfit

## tiny overfit status

- 완료
  - canonical loader batch 2-sample mixed overfit
  - repeated train-step loss 감소 확인
  - tiny overfit command 추가
  - epoch fit regression command 추가

## train status

- 완료
  - train command 추가
  - fit resume path 지원
  - canonical subset 기준 hardening regression test 추가
  - `tools/run_pv26_train.py` config/artifact split 반영

## 문서 업데이트 규칙

- 각 phase 시작 시 `9_EXECUTION_STATUS.md`를 먼저 갱신
- phase 완료 후 test 결과를 status에 기록
- 설계 변경이 생기면 해당 번호 문서를 먼저 수정

## 이번 phase에서 이미 고정된 항목

- loader sample dictionary key/shape/dtype
- variable dataset raw -> `800x608` transform 수식
- vehicle camera reference `800x600`은 pad-only 특수 케이스
- lane/stop/crosswalk query count `12 / 6 / 4`
- detector assignment 기반 TL attr binding
