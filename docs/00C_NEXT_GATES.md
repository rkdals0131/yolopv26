# 00C. Next Gates

> 이 문서는 다음 실행 순서와 금지사항을 관리한다.
> 새 실험을 끝내면 [history/README.md](history/README.md)에 결과를 추가하고, 이 문서의 gate를 갱신한다.

## 0. 현재 기준

- 기준 branch는 `develop` / `origin/develop` `9001b54`이다.
- 최종 success는 broader validation에서 `lane`, `stop_line`, `crosswalk` F1이 모두 `>= 0.60`인 것이다.
- 현재 best surface는 single raw checkpoint가 아니다. Router/composite/TTA/postprocess가 섞인 retained runtime surface다.
- Practical playback utility는 benchmark success가 아니다.
- former 00B top-level file은 제거됐다. closed experiment와 negative-result ledger는 `docs/history/00B_*.md`에서 찾는다.

## 1. Gate Order

| Gate | 상태 | 다음 행동 |
| --- | --- | --- |
| G0 docs/runtime surface cleanup | active | stale 00B/9/10/11 참조 제거, active docs를 짧게 유지, 로컬 링크/파일 존재만 확인. |
| G1 maintained runtime smoke | next | `check_env`, train CLI smoke, export metadata tests를 현재 runtime surface 기준으로 통과시킨다. |
| G2 export/TorchScript | next | current PV26 raw/dense head output names, metadata, roadmark trunk P2/P3/P4/P5 contract를 검증한다. |
| G3 lane-family metric improvement | blocked on new signal | 기존 closed axis 반복 없이 lane/stop-line을 동시에 올릴 새 instance-quality/candidate contract가 필요하다. |
| G4 source contract implementation | active / parallel | Signal attr, lane-val OD pseudo, ETRI dry-run을 서로 다른 track으로 진행한다. 공통 source/manifest/loader/test contract만 공유한다. |
| G5 ROS2 runtime check | active in `pv26_ros_runtime` | exported artifact를 `kaiev26_msgs/Perception2DFrame`으로 변환할 때 latency, GPU memory, frame rate, ID/class/geometry schema, overlay/sample replay를 확인한다. |

## 2. Do Not Repeat

반복 금지의 상세 근거는 [history/README.md](history/README.md)에서 관련 chunk를 찾아 확인한다. Top-level에서는 family만 유지한다.

- `phase_objective > 0.60`을 task별 F1 0.6 success로 해석하지 않는다.
- broader validation 없이 exact-only gain을 deployment/export default로 승격하지 않는다.
- GradScaler health gate 없이 PV26 long-run default AMP를 되살리지 않는다.
- stop-line score/length/threshold-only, PCA endpoint trim, component split, selector-map gate, center-rank margin, presence/emit gate, HAF quality/threshold/verifier sweep은 반복하지 않는다.
- lane row-scan threshold/cost, support substitution, duplicate suppression, risk-bucket sampler, row-anchor pressure, segment-continuity-only, conditional-row verifier, row-link/anchor-vote, area-ROI same-signal verifier sweep은 반복하지 않는다.
- crosswalk simple threshold/aspect/hull sweep만으로 all-task success를 주장하지 않는다.
- `best_signal.pt`를 traffic-light red/yellow/green/arrow attr teacher로 쓰지 않는다.
- ETRI KCity, lane-val OD pseudo eval root, signal attr teacher를 한 source key나 한 root로 섞지 않는다.
- live experiment branch를 시도마다 만들지 않는다.

## 3. Metric Gate

Metric 실험을 재개하려면 아래 순서를 지킨다.

1. 먼저 history에서 같은 family가 닫혔는지 확인한다.
2. smoke/fixed exact gate에서 TP/FP/FN이 기존 retained reference보다 나아야 한다.
3. exact-only positive면 broader-val512 또는 chunked deterministic aggregate를 확인한다.
4. lane/stop-line/crosswalk 중 하나를 올리면서 다른 task를 깨면 partial-positive로만 기록한다.
5. 결과는 `docs/history/` chunk에 원래 상황, 바꾼 것, 실제 metric/artifact 변화, 다음에 하지 말 것을 남긴다.

## 4. Source Contract Gate

`18`, `19`, `20` 문서는 implementation이 아니라 contract다. 구현하려면 각 source별로 아래를 함께 끝내야 한다.

- `common/pv26_schema.py` source key와 task mask.
- canonical scene/det/TL attr row schema.
- deterministic manifest and audit report.
- loader/final dataset guard.
- random overlay audit bundle.
- targeted unit tests.

Specifics:

- ETRI KCity v1은 `leftImg` only다. `rightImg`, `MonoCamera`, LiDAR, fusion output은 제외한다.
- `pv26_eval_lane_val_odpseudo_v1`은 eval-only다. train에 쓰지 않고 TL attr은 off다.
- `best_signal_attr.pt`는 ROI crop 기반 sidecar teacher가 필요하다. `best_signal.pt`는 box teacher다.

### 4A. Parallel Track Orchestration

G4는 세 독립 project track으로 나눈다. 한 track의 artifact를 다른 track의 input으로 기다리지 않는다.

| Track | 상태 | 첫 구현 범위 | 주요 파일 | 검증 |
| --- | --- | --- | --- | --- |
| A. `signal_attr` sidecar | implementation in progress | canonical AIHUB crop dataset, crop classifier train/eval, attrpseudo exhaustive sidecar hook | `tools/od_bootstrap/signal_attr/`, `tools/od_bootstrap/build/exhaustive_od.py`, `tools/check_env/actions.py` | `test_signal_attr_*`, `test_exhaustive_od_materialization_can_apply_signal_attr_sidecar` |
| B. lane-val OD pseudo eval | ready with source/loader guard | `pv26_eval_lane_val_odpseudo_v1` source registration, eval-only builder, rejected/accepted candidate manifest | `common/pv26_schema.py`, `tools/od_bootstrap/build/lane_val_odpseudo.py`, `model/engine/metrics.py` | `test_lane_val_odpseudo_preserves_base_val_sample_ids_and_count`, `test_lane_val_odpseudo_missing_checkpoint_fails_before_writing_ready_manifest`, `test_lane_val_odpseudo_disables_tl_attr_metrics_in_evaluator_report` |
| C. ETRI KCity leftImg | dry-run ready only | `leftImg` raw scan, image/semantic-label pairing, raw class inventory, ignored vs excluded manifest | `tools/od_bootstrap/source/etri_kcity/`, `common/pv26_schema.py`, `test/od_bootstrap/test_etri_kcity_dry_run.py` | `test_etri_dry_run_includes_only_leftimg_paths`, `test_etri_dry_run_manifest_separates_raw_scan_ignored_from_candidate_excluded`, `test_etri_materialization_fails_release_on_zero_samples` |

Shared rules:

- Add any new loader source key to both `SOURCE_MASK_BY_DATASET` and `DET_SUPERVISION_BY_DATASET`.
- Do not add eval-only or dry-run-only keys to `DATASET_GROUP_BY_KEY` until train usage is explicitly approved.
- Keep `tasks.has_*` as positive-content flags. Put audit/materialization completion in manifests.
- Empty `labels_det` means completed detector materialization with zero accepted boxes. Missing `labels_det` means failure.
- `traffic_lights[].detection_id` must match final `labels_det` row order.
- `best_signal.pt` remains a box teacher; `best_signal_attr.pt` is the only TL state sidecar.
- Existing `pv26_exhaustive_*` source-key semantics must not be silently changed. Attr pseudo outputs need new `*_attrpseudo_v1` or `*_v2` keys.

Immediate next work:

1. Build teacher datasets through `check_env` action `2`; this materializes all four teacher datasets, including canonical AIHUB `signal_attr` crops.
2. Train and evaluate `best_signal_attr.pt` as the fourth teacher through `check_env` actions `4A` and `7A` (`train/eval --teacher signal_attr`).
3. Use `A` for exhaustive OD once OD teachers, calibration, and signal_attr eval are ready; `build-exhaustive-od` auto-enables the sidecar when the default `best_signal_attr.pt` and eval report exist.
4. Keep ETRI KCity at dry-run only; do not create PV26 labels until raw semantic format and materialization policy are audited.

## 5. Export / ROS Gate

Export candidate가 생기면 아래를 확인한다.

- TorchScript artifact and adjacent metadata are written next to checkpoint; metadata includes the final artifact SHA256.
- Metadata output names match actual prediction tensors.
- Roadmark trunk/head channel contract is P2/P3/P4/P5.
- Postprocess output stays in raw image coordinates for det/lane/stop_line/crosswalk.
- ROS2 or sample replay sees the same output schema and practical latency/memory bounds.
- ROS adapter 코드는 `yolopv26`에 넣지 않는다. `pv26_ros_runtime`이 metadata output name/shape/class order를 검증하고 `/perception/{left,right}_wide/frame_2d`를 발행한다.
- Plan A artifact와 Plan B는 공통 evaluator 계약이 확정된 뒤 동일 dataset, rosbag,
  metric 정의로 비교한다. 이 모델 저장소는 외부 orchestration 문서 경로에 의존하지 않는다.

## 6. Documentation Rule

- Top-level docs stay short and current.
- Current status belongs in `00A_CURRENT_STATUS.md`; next work and gates belong in this file.
- Detailed failed attempts stay in `docs/history/`.
- Legacy design originals stay in `docs/legacy/`.
- Do not reintroduce per-run status docs unless there is an active multi-day execution that cannot fit in `00A`/`00C`.
- A doc link in README/PRD/active docs must point to an existing file.
