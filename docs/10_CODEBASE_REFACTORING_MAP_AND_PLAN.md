# PV26 Codebase Refactoring Map And Plan

## 목적

이 문서는 repo 전체 리팩토링의 active map이다. `docs/2_SYSTEM_ARCHITECTURE.md`는 현재 구조 요약만 유지하고, 이 문서는 디렉토리별 책임, public/internal boundary, baseline drift, refactor wave 순서, 금지사항을 고정한다.

리팩토링 목표는 모델 동작을 바꾸는 것이 아니라 이미 있는 계약을 더 명확한 module boundary와 테스트로 보호하는 것이다. baseline test drift와 private helper import boundary를 먼저 정렬했고, 이번 wave는 active runtime이 아닌 실험 probe code를 retired evidence로 내리는 데 집중한다.

## 현재 디렉토리 지도

### `common/`

- `pv26_schema.py`: dataset/task schema와 canonical field 이름을 고정한다.
- `io.py`, `paths.py`: repo-wide JSON/JSONL/path helper의 public surface다.
- `config_coercion.py`, `user_config.py`, `scalars.py`, `task_mode.py`: config coercion, deep merge, scalar normalization, task-mode parsing을 담당한다.
- `geometry.py`, `boxes.py`, `overlay.py`: 순수 geometry/box/overlay helper를 담당한다.
- `train_runtime.py`: duration, progress segment, tensorboard scalar writer, rolling timing status처럼 trainer와 bootstrap trainer가 공유하는 runtime helper를 담당한다.

금지: source별 overwrite policy, dataset publish policy, timestamp timezone policy처럼 call-site마다 의미가 다른 helper를 무리하게 공통화하지 않는다.

### `model/data/`

- `dataset.py`: canonical dataset index, sample load, source mask, valid mask, ragged sample contract를 담당한다.
- `transform.py`: online letterbox, train augmentation, image/geometry transform을 담당한다.
- `sampler.py`: train/eval sampler와 task-positive sampler 계약을 담당한다.
- `target_encoder.py`, `roadmark_v2_targets.py`: raw sample을 loss/trainer 입력 tensor contract로 encode한다.
- `preview.py`: training/evaluation preview overlay를 담당한다.

금지: dataset loader가 training phase policy나 model architecture flag를 직접 해석하지 않는다.

### `model/net/`

- `trunk.py`: Ultralytics YOLO26 adapter, detect-source P3/P4/P5 trunk, roadmark-source P2/P3/P4/P5 trunk, feature-channel inference를 담당한다.
- `heads.py`: `PV26Heads` public constructor와 detector/TL/lane-family head wiring을 담당한다. 현재 `PV26Heads`의 runtime contract는 P2/P3/P4/P5 4-level 입력이다.
- `lane_head_*`, `stopline_head_*`, `crosswalk_head_*`, `roadmark_*`: roadmark family별 내부 head 구현이다.

금지: 새 head family를 추가할 때 기존 `PV26Heads.describe()` / export metadata / loss spec sync를 우회하지 않는다.

### `model/engine/`

- public helper surface:
  - `batch.py`: batch device move, raw batch extraction/merge, lane-family metric augmentation.
  - `det_geometry.py`: detector geometry helper public surface.
  - `trainer_progress.py`, `trainer_reporting.py`: progress/reporting helper surface.
  - `train_summary.py`: train summary schema/helper surface.
  - `loss.py`, `metrics.py`, `postprocess.py`, `spec.py`, `trainer.py`, `evaluator.py`: stable engine runtime.
- internal implementation surface:
  - `_trainer_checkpoint.py`, `_trainer_epochs.py`, `_trainer_fit.py`, `_trainer_io.py`, `_trainer_progress.py`, `_trainer_reporting.py`, `_trainer_step.py`.

원칙: repo 외부 entrypoint는 underscore 모듈을 직접 import하지 않는다. 필요한 helper는 `batch.py`, `trainer_reporting.py`, `trainer_progress.py`, `train_summary.py` 같은 public/shared module로 승격한다.

### `tools/`

stable entrypoint:
- `tools/check_env.py`
- `tools/run_pv26_train.py`
- `python -m tools.od_bootstrap`
- `tools/model_export/`

durable analysis:
- `tools/analyze_pv26_run.py`

package internals:
- `tools/check_env/`
- `tools/pv26_train/`
- `tools/od_bootstrap/source/aihub/`
- `tools/od_bootstrap/source/shared/`
- `tools/od_bootstrap/build/`
- `tools/od_bootstrap/teacher/runtime/`

retired experimental probes:
- `tools/probe_pv26_*.py`
- `tools/run_pv26_lane60_probe.py`
- `tools/evaluate_pv26_lane60_checkpoint.py`
- `tools/analyze_pv26_lane60_prediction_filters.py`
- `tools/analyze_pv26_lane_repairability_model_replay.py`
- `tools/replay_pv26_lane_point_repair.py`
- `tools/interpolate_pv26_checkpoints.py`
- `tools/merge_pv26_lane_family_task_heads.py`

원칙: probe 계열은 public API가 아니며 active runtime surface도 아니다. stable runtime이 아니고, README/architecture/training docs가 현재 실행 surface로 안내하지 않으며, 테스트가 probe 내부 함수를 직접 pin하는 경우 삭제 대상이다. 과거 실험 증거는 `docs/legacy/`, `docs/00B_STATUS_HISTORY.md`, 필요한 status 요약에 남기고 code/test surface에서는 제거한다.

### `tools/modal/`

- `check.py`, `local_preflight.py`: remote training preflight와 local readiness check를 담당한다.
- `prepare_dataset_volume.py`, `dataset_archive.py`: dataset volume/archive surface를 담당한다.
- `train.py`, `constants.py`, `sdk_import.py`: Modal training execution, constants, SDK import guard를 담당한다.

원칙: Modal surface는 remote train preflight/volume/archive/preset verification에 집중한다. local trainer policy를 Modal-only helper에 복제하지 않는다.

### `test/`

- contract/core tests: schema, loader, transform, sampler, heads, loss, metrics, evaluator, trainer.
- runtime/stable tools tests: `tools/check_env.py`, `tools/run_pv26_train.py`, `tools.od_bootstrap`, durable analysis, portability.
- runtime tests: check_env, run_pv26_train, portability, prepared dataset e2e.
- doc-sync tests: README/docs/current architecture와 active refactor map의 핵심 문구.

원칙: refactor slice마다 owning test를 먼저 실행하고, 마지막에 full suite로 baseline green을 확인한다.

## Public / Internal Boundary

stable public surface:
- `model.data` package exports
- `model.net.PV26Heads`
- `model.net.build_yolo26_trunk`
- `model.net.build_yolo26_roadmark_trunk`
- `model.engine.__all__`
- `tools/check_env.py`
- `tools/run_pv26_train.py`
- `python -m tools.od_bootstrap`
- `tools/model_export/`

durable but narrow public surface:
- `tools/analyze_pv26_run.py`

internal surface:
- `model.engine._trainer_*`
- `tools/pv26_train/*` implementation modules behind `tools/run_pv26_train.py`
- `tools/check_env/*` implementation modules behind `tools/check_env.py`
- `tools/od_bootstrap/*` implementation modules behind `python -m tools.od_bootstrap`

retired tool boundary:
- raw-batch merge has a public home at `model.engine.batch.merge_raw_batches`; durable analysis and stable tools use that instead of importing `model.engine._trainer_epochs._merge_raw_batches`.
- retired probe modules must not be reintroduced as compatibility wrappers or hidden helper layers. If an idea becomes current product behavior, implement it through maintained model/data/engine/runtime modules with product tests.

Refactor gate: private core helper imports must remain absent from stable tools, and retired probe imports must remain absent from tests.

## Known Baseline Drift Before Refactor

2026-06-02 baseline command:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python -m unittest discover -s test -p 'test_*.py'
```

Observed result before code repair: `534` tests ran with `2` failures and `2` errors.

- `test_pv26_tiny_overfit.py`: old 3-level `build_yolo26n_trunk()` / `PV26Heads((64, 128, 256))` path conflicts with the current P2/P3/P4/P5 `PV26Heads` contract.
- `test_pv26_trunk_features.py`: old P3/P4/P5 assertion conflicts with the current roadmark trunk P2/P3/P4/P5 path.
- `test_pv26_runtime_sanity.py`: sampler fail-fast assertion expects the stale `balanced sampler found no eligible samples` string instead of current `task-positive multi sampler found no positive samples...`.
- `test_pv26_train_infer_e2e.py`: random one-epoch synthetic e2e asserts lane-family positive predictions, which is not a stable contract. The stable contract is schema presence, finite losses/metrics, and predict path execution.

These are test/contract alignment fixes, not architecture changes.

## Current Refactor Status

- 2026-06-02: documentation-first map added on `refactor/tools-expired-probe-prune`.
- 2026-06-02: baseline drift 4건은 current contract 기준으로 repaired. P2/P3/P4/P5 roadmark trunk, task-positive sampler fail-fast, and stable prepared dataset e2e schema/finite-metrics contract are now test-pinned.
- 2026-06-02: raw-batch merge helper moved to `model.engine.batch.merge_raw_batches`; stable tools and durable analysis no longer import `model.engine._trainer_epochs._merge_raw_batches`.
- 2026-06-02: obsolete probe/lane60 replay/checkpoint-surgery code was retired from active `tools/`, and probe-owned tests were removed. Historical metrics and command evidence remain in status/history docs.
- 2026-06-02: `modal/` was moved under `tools/modal/`; Modal A100 runbook commands, import guards, and module layout tests now track the tools-owned package location.
- 2026-06-02: `tools.modal.sdk_import` now skips `repo_root/tools` while importing the external Modal SDK, so `tools/modal/` cannot shadow top-level `modal` when a launcher prepends the tools directory to `sys.path`.
- 2026-06-02: first repo-grounded E2E connection audit was recorded below, from raw source standardization through final dataset publication, PV26 loader/target encoding, model heads, loss, postprocess, trainer, and tests.
- 2026-06-02: target encoder dense `roadmark_v2` payloads are now pinned against the keys/shapes/dtypes consumed by seg-first lane, stop_line, and crosswalk loss branches.
- 2026-06-02: worker-side encoded collation now has a regression guard showing `collate_pv26_encoded_batch()` emits the same nested seg-first `roadmark_v2` payload as manual encode.
- 2026-06-02: trainer/evaluator `prepare_batch()` now has regression guards for rehydrating seg-first dense targets from encoded eval batches with `_raw_batch` while preserving the same metrics raw supervision bundle and aligned `meta`.
- 2026-06-02: train config -> DataLoader helper propagation is now pinned for encoded-batch flags, worker settings, prefetch behavior, and task-positive sampler options.
- 2026-06-02: phase trainer factory propagation is now pinned from train config and phase policy into `PV26Heads`, `PV26MultiTaskLoss`, `PV26Trainer`, scheduler setup, and evaluator postprocess defaults.
- 2026-06-02: phase execution I/O is now pinned from phase overrides into loader/trainer/fit options, weights-only handoff, checkpoint preview bundle generation, and manifest-ready phase result fields.
- 2026-06-02: meta-train runtime manifest lifecycle is now pinned for selected phase windows, active phase state writes, skipped/completed phase status, lineage preservation, and final checkpoint handoff.
- 2026-06-02: resume CLI handoff is now pinned from manifest selected phase window and lineage seed checkpoint into runtime meta-train options.
- 2026-06-02: final-dataset manifest rows are now checked by `PV26CanonicalDataset` against discovered records, image paths, det paths, and source trace metadata before training can read the published dataset.
- 2026-06-02: `lane_only` target encoding now preserves semantic-less lane geometry rows as query supervision while leaving color/type one-hot slots empty, keeping `lane_supervised_count` aligned with `lane_valid` and query objectness.
- 2026-06-02: train runtime backbone/head-channel handoff now prefers the roadmark trunk and reconstructs P2/P3/P4/P5 head channels from detect-only adapter channels.
- 2026-06-02: `check_env --check-yolo-runtime` now validates the current roadmark trunk source/stride/channel shape instead of only the legacy detect trunk load.
- 2026-06-02: PV26 TorchScript export head-channel inference now reconstructs the current P2/P3/P4/P5 contract and uses the roadmark trunk while keeping detector metadata on P3/P4/P5.
- 2026-06-02: `tools/model_export/pv26_torchscript.py` now reads the loss spec through public `model.engine.spec`, and module-layout guards reject private `model.engine._*` imports from stable tools.
- 2026-06-02: head `describe()` metadata, evaluator `prediction_shapes`, and trainer checkpoint metadata now pin det/TL/lane/stop_line/crosswalk raw-head shapes against the current loss spec.
- 2026-06-02: postprocess letterbox inverse is pinned for detector boxes and lane-family point outputs so evaluator/runtime predictions stay in raw image coordinates.
- 2026-06-02: exact checkpoint resume now rejects mismatched format, spec version, and head summary shape metadata; weights-only migration remains the shape-aware path for older or incompatible checkpoints.
- 2026-06-02: loader TL attribute linkage now rejects duplicate or out-of-range `traffic_lights[].detection_id` values before scene TL labels can drift away from `labels_det` rows.
- 2026-06-02: loader detector targets are pinned to `labels_det` YOLO rows; `labels_scene.detections` remains descriptive/provenance data for reports, review overlays, and exhaustive materialization, not the train target source.
- 2026-06-02: BDD/AIHUB source-standardization resume now rejects stale canonical scenes whose task masks or stale `labels_det` files no longer match the source kind's det-only, lane-only, or traffic/TL contract.
- 2026-06-02: teacher-dataset and bootstrap image-list bridges now reject canonical scenes whose explicit `tasks.has_det` disagrees with `labels_det` presence before teacher filtering or exhaustive OD materialization can reuse stale source labels.
- 2026-06-02: `tools/analyze_pv26_run.py` phase overview now preserves trainer-selected `best_epoch`/selection mode instead of recomputing best rows by max `phase_objective`.
- Remaining risk: future experiment ideas must enter through maintained runtime modules, not by recreating one-off probe helper meshes.

## E2E Pipeline Connection Audit

이 audit는 cleanup/refactor 중 model shape와 data I/O를 바꾸지 않기 위한 current-code 연결표다. 새 helper를 만들기보다 현재 entrypoint, 산출물, 소비자를 명시해 다음 wave에서 회귀를 좁게 잡는다.

### 1. Raw source -> canonical source

- Entry: `python -m tools.od_bootstrap prepare-sources` -> `tools/od_bootstrap/cli.py` -> `tools/od_bootstrap/source/prepare.py`.
- BDD path: `source/bdd100k.py`가 BDD image/json pair를 발견하고, `bdd100k_det_100k` scene을 detector-only source로 쓴다.
- AIHUB path: `source/aihub/pipeline.py`가 lane/obstacle/traffic roots를 archive-or-extracted tree로 정규화하고, dataset kind별 worker로 dispatch한다.
- Output contract: bootstrap output root 아래 `meta/source_prep_manifest.json`, `meta/bootstrap_image_list.jsonl`, `canonical/bdd100k_det_100k/`, `canonical/aihub_standardized/`를 쓴다. Each canonical dataset root owns `images/<split>/`, `labels_scene/<split>/*.json`, optional `labels_det/<split>/*.txt`, conversion/report/debug-vis artifacts; the bootstrap image list carries downstream `sample_uid`, `dataset_key`, `split`, `dataset_root`, `source_name`, and optional `det_path`.
- Shape/label policy: BDD는 vehicle/bike/pedestrian det-only, AIHUB lane은 lane/stop_line/crosswalk geometry-only, AIHUB traffic/obstacle은 det/TL attr 또는 obstacle det source다. 이 분리는 `scene["tasks"]`, `source.dataset`, `labels_det` 존재 여부로 downstream에 전달된다.
- Worker task-mask policy: source workers own the first canonical task masks. BDD and AIHUB obstacle/traffic must emit detector-only or detector-plus-TL scenes with `labels_det`; AIHUB lane must emit lane/stop/crosswalk scenes with no `labels_det`, no detections, and no TL/sign objects.
- Row-order policy: detector-supervised source standardizers write `labels_det` rows in the same order as `labels_scene.detections[].id`; AIHUB traffic `traffic_lights[].detection_id` points at that same row-order id before loader TL attributes are attached.
- Primary tests: `test_aihub_standardize.py`, `test_bdd100k_standardize.py`, `test/od_bootstrap/test_preprocess_sources.py`, `test/od_bootstrap/test_shared_source_helpers.py`, `test/od_bootstrap/test_aihub_workers.py`.

### 2. Canonical source -> teacher/exhaustive/final dataset

- Teacher dataset entry: `build-teacher-datasets` -> `tools/od_bootstrap/build/teacher_dataset.py`.
- Teacher I/O: canonical `labels_scene`를 source dataset key로 필터링하고, `labels_det` YOLO rows를 teacher-local class ids로 remap해 `images/<split>/`, `labels/<split>/`, `meta/teacher_dataset_manifest.json`를 쓴다. Teacher detector labels also come from `labels_det`, not `labels_scene.detections`; each manifest row records `source_scene_path`, `source_image_path`, `source_label_path`, output image/label paths, and source-derived `sample_uid`.
- Exhaustive OD entry: `build-exhaustive-od` -> `tools/od_bootstrap/build/exhaustive_od.py`.
- Exhaustive I/O: `bootstrap_image_list.jsonl` entries, calibrated class policy, teacher predictions를 읽고 raw source detections plus bootstrap detections를 materialize해 exhaustive `labels_scene`, `labels_det`, `images`, `meta/materialization_manifest.json`를 쓴다. Each manifest row records source `labels_scene`, source image, optional source `labels_det`, and output scene/image/det paths. Raw source `detections[].id` must already match row order before bootstrap detections are appended so the exhaustive `labels_det` row/class-id order remains foreign-key compatible.
- Final dataset entry: `build-final-dataset` -> `tools/od_bootstrap/build/final_dataset.py`.
- Final I/O: latest exhaustive OD dataset과 canonical AIHUB lane scenes를 staging root에 hardlink/copy하고 duplicate `final_sample_id`, image filename, and required det paths for detector-supervised sources를 검증한 뒤 atomic publish한다. Each final manifest row records source scene/image/optional det paths plus published scene/image/det paths, so `PV26CanonicalDataset` records can be traced back to exhaustive or lane source files. Output은 PV26 training root인 `pv26_exhaustive_od_lane_dataset` shape: `images/`, `labels_scene/`, optional lane-only `labels_det/`, `meta/final_dataset_manifest.json`, stats/summary/publish marker.
- Loader manifest guard: when `meta/final_dataset_manifest.json` is present, `PV26CanonicalDataset` must reject any mismatch between manifest rows and discovered `labels_scene` records, published image paths, optional det paths, and source trace fields. This keeps training from silently consuming a manually edited or partially stale final dataset.
- Primary tests: `test/od_bootstrap/test_teacher_dataset.py`, `test/od_bootstrap/test_sweep_runner.py`, `test/od_bootstrap/test_final_dataset.py`, `test/od_bootstrap/test_checkpoint_eval.py`, `test/od_bootstrap/test_build_debug_vis.py`.

### 3. Final dataset -> loader/target encoder

- Loader entry: `model.data.PV26CanonicalDataset`.
- Loader I/O: scans `labels_scene/**/*.json`, resolves `images/<split>/<scene.image.file_name>`, optional `labels_det/<split>/<sample>.txt`, and creates `SampleRecord`.
- Detection contract: if `SOURCE_MASK_BY_DATASET[dataset_key]["det"]` is true, missing `labels_det` is a runtime error. YOLO rows are converted back to raw xyxy, then letterboxed into network xyxy.
- Scene detection contract: `labels_scene.detections` is retained for provenance, statistics, debug/review overlays, and exhaustive OD materialization. Loader detector target classes/boxes must come from `labels_det`, so scene detection rows must not be silently substituted as training targets.
- TL linkage contract: `traffic_lights[].detection_id` is a foreign key into the sample's `labels_det` row order. IDs must be unique and must reference an existing detection row before TL bits are attached to encoded detections.
- Geometry contract: lane/stop_line/crosswalk points are letterboxed into `NETWORK_HW` space, clipped, validity-masked by unique point count, and optionally augmented only for train split.
- Encoder entry: `model.data.target_encoder.encode_pv26_batch`.
- Encoder shape contract: images `[B,3,608,800]`; det targets `[B,N_gt,4]`; TL bits `[B,N_gt,4]`; lane `[B,24,38]`; stop_line `[B,8,9]`; crosswalk `[B,8,33]`; dense roadmark targets under `roadmark_v2` with `[B,C,152,200]`-class spatial shapes. The final dataset publication fixture now pins those encoded query/vector shapes against `PV26Heads.describe()` and a real `PV26Heads` raw forward shape.
- Dense roadmark aux contract: `include_lane_segfirst_targets=True` must emit the full `roadmark_v2` key set consumed by seg-first lane, stop_line, and crosswalk losses, including row/seed targets, masks, offsets, HAF/axis targets, and ignore maps with stable shape/dtype.
- Worker collation contract: `collate_pv26_encoded_batch()` is the train-loader encoded-batch path and must match manual `encode_pv26_batch(..., include_lane_segfirst_targets=True)` for raw heads, metadata, and nested dense roadmark aux payloads. `collate_pv26_encoded_eval_batch()` must match the same encoded payload while preserving a no-image `_raw_batch` with raw supervision and aligned `meta` for metrics.
- Primary tests: `test_pv26_loader.py`, `test_pv26_target_encoder.py`, `test_pv26_transform_roundtrip.py`, `test_lane_segfirst_vectorizer.py`, `test_pv26_balanced_sampler.py`, and `test/od_bootstrap/test_final_dataset.py::test_final_dataset_publication_loads_through_pv26_dataset_and_encoder`.

### 4. Encoded batch -> model/loss/postprocess/trainer

- Model entry: `model.net.PV26Heads` with P2/P3/P4/P5 feature pyramid.
- Model shape contract: detector/TL heads use P3/P4/P5 and emit `det [B,9975,12]`, `tl_attr [B,9975,4]`; roadmark heads use current architecture and emit lane/stop_line/crosswalk plus dense roadmark keys expected by loss/postprocess.
- Export contract: PV26 TorchScript export must instantiate `PV26Heads` with P2/P3/P4/P5 checkpoint head channels and a roadmark trunk source pyramid, but export detector feature metadata remains P3/P4/P5 and must match exported `det`/`tl_attr` row counts before artifact metadata is written.
- Reporting/checkpoint contract: `PV26Heads.describe()` carries query/vector dimensions, evaluator summaries report `prediction_shapes` for all raw heads, and trainer checkpoints preserve generation/spec/head metadata. Exact resume rejects generation/format/spec/head-summary mismatches; weights-only migration is the shape-aware compatibility path.
- Loss entry: `model.engine.loss.PV26MultiTaskLoss`.
- Loss I/O: consumes the heads dict and encoded batch dict; source masks and `det_supervised_class_mask` prevent partial-label sources from becoming false negatives. A single synthetic seg-first guard now feeds an actual `PV26Heads` raw output through `PV26MultiTaskLoss` and `postprocess_pv26_batch` on the same encoded batch.
- Postprocess entry: `model.engine.postprocess.postprocess_pv26_batch`.
- Postprocess I/O: decodes detector anchors and roadmark heads back through `meta.transform` into raw image coordinates for evaluator/runtime predictions. Detector query count must match `det_feature_shapes` / `det_feature_strides` metadata before anchor decode. The regression guard covers detector boxes plus lane/stop_line/crosswalk points under non-identity letterbox metadata.
- Trainer entry: `tools/run_pv26_train.py` -> `tools/pv26_train/cli.py` -> `model.engine.PV26Trainer`.
- Trainer I/O: preset resolves dataset roots/run roots; dataloaders can encode in workers; heads receive encoded context when supported; summaries/checkpoints/history are written under run root.
- Backbone/head-channel contract: train runtime must build the roadmark trunk for current `PV26Heads`; if a compatibility adapter exposes only detect P3/P4/P5 channels, `_resolve_head_channels()` must reconstruct the P2/P3/P4/P5 tuple before head construction.
- Loader factory contract: `_build_phase_train_loaders()` must propagate train/val encoded-batch flags, worker settings, prefetch behavior, and task-positive sampler options to `build_pv26_train_dataloader()` / `build_pv26_eval_dataloader()`; the helpers own encoded worker collation and eval raw-bundle preservation.
- Trainer factory contract: `_build_phase_trainer()` must pass architecture/head flags into `PV26Heads`, loss/task/distill options into `PV26MultiTaskLoss`, runtime optimizer/AMP/non-finite/PCGrad options into `PV26Trainer`, and phase max-epochs/schedule into the scheduler while installing the resolved postprocess config as evaluator defaults.
- Phase execution contract: `_execute_phase()` must apply phase overrides before constructing loaders/trainer, pass phase/runtime/history/PCGrad settings into `trainer.fit()`, load the previous phase best checkpoint as weights-only handoff when no local last checkpoint exists, generate best/last preview bundles when checkpoint files exist, and return manifest-ready checkpoint/selection/postprocess/head-channel/run-summary fields. The recorded `head_channels` must come from the constructed `PV26Heads.in_channels`, not only from the backbone-variant fallback.
- Meta manifest contract: `run_meta_train_scenario()` must write selected-window exclusions as skipped phases, set/clear active phase state around each executing phase, preserve lineage, update completed phase results, propagate previous best checkpoints between phases, and report completed/skipped counts plus final checkpoint path.
- Resume CLI contract: `tools/run_pv26_train.py --resume-run` must preserve manifest `selected_phase_window.selected_phase_indices`, `lineage`, and `lineage.seed_checkpoint_path` as `run_meta_train_scenario()` runtime options.
- Prepare-batch bridge: trainer/evaluator raw batches are encoded with seg-first dense targets when the active head needs them; encoded eval batches with `_raw_batch` can be rehydrated for seg-first loss while preserving the same raw supervision bundle for metrics, keeping `_raw_batch` unmoved/no-image and `meta` aligned.
- Primary tests: `test_roadmark_native_contract.py`, `test_pv26_heads.py`, `test_pv26_loss_runtime.py`, `test_pv26_postprocess.py`, `test_pv26_trainer.py`, `test_pv26_evaluator.py`, `test_pv26_train_infer_e2e.py`, `test_pv26_runtime_sanity.py`, `test_model_export.py`.

### Regression Risk Register

- Dataset key drift: `SOURCE_MASK_BY_DATASET`, teacher specs, exhaustive dataset key mapping, sampler groups, and docs must move together. Guarded by loader/sampler/final-dataset tests.
- Shape drift: `model.engine.spec`, `model.data.target_encoder`, `model.net.heads`, loss, postprocess, export metadata, evaluator summaries, and checkpoint metadata must stay synchronized. Guarded by loss spec, target encoder, heads, loss runtime, postprocess, export, evaluator, and trainer tests.
- Raw head contract drift: `PV26Heads.describe()`, emitted `det`/`tl_attr` tensor dims, lane-family query/vector dims, and detector feature metadata must remain derived from the same loss-spec class/bit/query contract consumed by loss, postprocess, evaluator summaries, checkpoints, and export metadata. Guarded by heads raw-contract, postprocess metadata, evaluator shape, trainer checkpoint, and model-export tests.
- Trunk/head channel handoff drift: roadmark trunk builders must expose the P2/P3/P4/P5 source indices, strides, and channel tuple expected by `PV26Heads` before any real forward pass; default `yolo26s` remains `(128,128,256,512)` and `yolo26n` remains `(64,64,128,256)`. Guarded by trunk metadata tests, train runtime backbone/head-channel tests, check_env runtime contract tests, and export metadata tests.
- Checkpoint exact-resume drift: trainer checkpoints must persist checkpoint format, architecture generation, spec version, and every raw-head query/vector dimension in `head_summary`; exact resume must reject any stale head-summary field and reserve shape-aware partial loading for weights-only migration. Guarded by trainer checkpoint metadata and exact-resume rejection tests.
- Optional-det drift: lane-only samples legitimately lack `labels_det`; det-supervised sources must fail if det labels are missing. Guarded by loader and final dataset publication->loader->encoder tests.
- Lane-only semantic drift: `lane_only` mode must not drop geometry-only lane rows simply because color/type semantics are absent; `lane_supervised_count` and `lane_valid`/query objectness must stay aligned. Guarded by target encoder and loss runtime tests.
- Final publish det drift: final dataset publication must not publish detector-supervised exhaustive OD samples without `labels_det`; otherwise the final root can be invalid before loader/runtime sees it. Guarded by final dataset missing-det publication tests.
- Final-manifest loader drift: final dataset publication manifest rows must match the actual records that `PV26CanonicalDataset` discovers, including image/det paths and source trace metadata, before train/eval dataloaders can read the dataset. Guarded by loader manifest drift tests and final dataset publication-through-loader tests.
- Scene-det substitution drift: descriptive `labels_scene.detections` must not become model-facing detector targets; otherwise review/provenance rows can override the normalized YOLO rows consumed by train/eval. Guarded by loader scene-detection mismatch tests.
- Source det row-order drift: BDD and AIHUB detector-supervised source standardizers must keep `labels_det` rows aligned with `labels_scene.detections[].id`, and AIHUB traffic TL attributes must reference those same ids. Guarded by BDD/AIHUB standardization row-order tests and loader TL linkage tests.
- Source resume task drift: resume must not reuse stale canonical scene files whose `tasks`, lane-family fields, traffic fields, or stale `labels_det` files contradict the source kind. Guarded by BDD/AIHUB standardization resume tests.
- Canonical det-label task drift: teacher datasets and bootstrap image lists must reject canonical scenes whose explicit `tasks.has_det` no longer matches `labels_det` presence, otherwise teacher filtering and exhaustive OD materialization can silently preserve stale or missing source labels. Guarded by teacher dataset and image-list bridge tests.
- Exhaustive row-order drift: exhaustive OD writes `labels_det` from `final_scene.detections` order, so raw detection ids must be contiguous row-order ids before bootstrap boxes are appended and class ids must follow the same scene-detection order. Guarded by exhaustive materialization id/order tests.
- TL/det linkage drift: traffic-light attribute labels must stay keyed to actual detector rows; duplicate or out-of-range detection ids must fail at loader boundary. Guarded by loader malformed traffic-light payload tests.
- Coordinate drift: source raw xyxy/points, online letterbox network space, and postprocess inverse raw space must remain separate. Guarded by loader, transform roundtrip, target encoder, and postprocess tests that cover both detector boxes and lane-family points.
- Encoded eval collation drift: eval dataloaders must emit encoded loss inputs that match manual seg-first encoding while retaining no-image `_raw_batch` raw supervision in the same sample order for metrics. Guarded by target encoder eval collation tests and evaluator/trainer rehydration tests.
- Detector feature metadata drift: `det`, `tl_attr`, `det_feature_shapes`, and `det_feature_strides` must describe the same detector query layout before postprocess anchor decode. Guarded by postprocess metadata-required and query-count drift tests.
- Export metadata drift: TorchScript metadata must reject detector feature shape/stride rows that disagree with exported `det` and `tl_attr` raw head rows before publishing an artifact. Guarded by model export metadata tests.
- Prepare-batch raw-bundle drift: evaluator/trainer must keep encoded eval `_raw_batch` as the same unmoved no-image supervision bundle while regenerating seg-first dense targets and preserving `meta` alignment. Guarded by evaluator/trainer prepare-batch rehydration tests.
- Runtime entrypoint drift: stable commands remain `tools/check_env.py`, `tools/run_pv26_train.py`, `python -m tools.od_bootstrap`, `tools/model_export/`, and `tools/modal/`; retired probes must not re-enter active tools/tests. Guarded by docs/module layout tests.
- Modal SDK shadow drift: because the repo owns `tools/modal/`, Modal launchers that prepend `repo_root/tools` must still import the external top-level `modal` SDK, not the local tools package. Guarded by Modal SDK import shadow tests.
- Durable analysis drift: `tools/analyze_pv26_run.py` must preserve trainer summary `best_epoch`, `best_metric_value`, and selection mode when exporting phase overview CSVs; min-selection loss phases must not be reinterpreted as max `phase_objective` phases. Guarded by analyze-run I/O tests.
- `check_env --check-yolo-runtime` must validate the same roadmark trunk source count, strides, and P2/P3/P4/P5 channels that train/export require; a detect-only trunk load is not enough evidence for current PV26 runtime compatibility.

## Refactor Waves

### Wave 1. Documentation-first map

- Add this document as the active refactor map.
- Pin it from `test/test_docs_sync.py`.
- No production code changes in this wave.

### Wave 2. Baseline repair

- Update old 3-level trunk/head tests to `build_yolo26_roadmark_trunk()` and 4-level channel inference.
- Update sampler fail-fast test to the current task-positive multi sampler error contract.
- Update one-epoch e2e to validate schema, finite metrics, and predict path without requiring lane-family positives from a random short run.

### Wave 3. Boundary hardening

- Reduce external imports from `model.engine._trainer_epochs`.
- Keep reusable raw-batch merge behavior at `model.engine.batch.merge_raw_batches`; future one-off experiment code must not recreate retired probe-local helper surfaces.
- Keep `model.engine.__all__` curated and underscore-free.

### Wave 4. Tools/probe retirement

- Keep README focused on stable entrypoints only.
- Keep `tools/analyze_pv26_run.py` durable analysis.
- Retire obsolete probe scripts, lane60 replay helpers, and checkpoint-surgery utilities from active `tools/`.
- Preserve past experiment evidence in `docs/legacy/`, `docs/00B_STATUS_HISTORY.md`, and concise active status summaries.
- Remove tests that import retired probe internals; they are not product/runtime contract tests.

### Wave 5. Guard deleted probe surface

- Assert `tools/probe_pv26_*.py` and retired lane60/replay/checkpoint-surgery scripts are absent.
- Assert tests do not import `tools.probe_*`, `tools.run_pv26_lane60_probe`, or retired replay/checkpoint-surgery modules.
- Keep stable entrypoint smoke coverage for `check_env`, `run_pv26_train`, `tools.od_bootstrap`, and product runtime modules.

### Wave 6. Final docs/status sync

- README remains stable-entrypoint oriented.
- `docs/2_SYSTEM_ARCHITECTURE.md` remains current structure summary.
- This document records remaining refactor risks.
- `docs/9_EXECUTION_STATUS.md` gets a short status entry.
- `docs/00B_STATUS_HISTORY.md` and `docs/legacy/` stay historical and are not rewritten for cleanup aesthetics.

## Post-retirement read-only scan

2026-06-02 scan after full-suite green:

- Retired tool names still appear in `docs/00B_STATUS_HISTORY.md`, `docs/legacy/`, `docs/00A_CURRENT_STATUS.md`, and `docs/temp_gpt_pro_handoff_20260528.md`. This is expected historical evidence. Future doc cleanup may add clearer archival labels or move temporary handoff docs, but must not rewrite metrics history.
- `docs/00C_NEXT_GATES.md` still uses the generic word "probe" for historical negative evidence and planning notes. That is not an active command surface; avoid mass-editing the ledger unless a later docs-only pass narrows wording without changing experiment meaning.
- A few private-module imports remain intentionally outside the retired-probe scope: internal trainer tests import trainer internals to pin checkpoint/fit/step behavior. Stable tools should use public engine facades; do not broaden test-internal private import cleanup without a separate boundary plan and tests.
- README/training docs use "probe" for stable VRAM stress checks through `tools/run_pv26_train.py`. That is a stable train entrypoint, not retired `tools/probe_pv26_*` code.

## 금지사항

- Do not amend the previous cleanup commit with new refactor work.
- Do not touch generated/data/experiment outputs: `.venv`, `.omx`, `runs`, `seg_dataset`, downloaded weights.
- Do not change model architecture, loss behavior, or training behavior while baseline tests are red.
- Do not reintroduce retired probe files as wrappers, aliases, or new public helper modules.
- Do not rewrite `00B_STATUS_HISTORY.md` or `docs/legacy/` as part of cleanup.
- Do not broaden helper public APIs unless at least two maintained call sites need the same behavior and tests can pin the contract.

## Verification Plan

문서/계약:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python -m unittest discover -s test -p 'test_docs_sync.py' -v
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python -m unittest discover -s test -p 'test_module_layout_compat.py' -v
```

baseline repair:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python -m unittest discover -s test -p 'test_pv26_tiny_overfit.py' -v
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python -m unittest discover -s test -p 'test_pv26_trunk_features.py' -v
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python -m unittest discover -s test -p 'test_pv26_runtime_sanity.py' -v
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python -m unittest discover -s test -p 'test_pv26_train_infer_e2e.py' -v
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python -m unittest discover -s test -p 'test_*.py'
```

refactor slices:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python -m py_compile $(git ls-files '*.py')
```

entrypoint smoke:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python3 tools/check_env.py --strict --check-yolo-runtime
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python3 tools/run_pv26_train.py --help
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python -m tools.od_bootstrap --help
```
