# 00C. Next Gates

> 이 문서는 다음 실행 순서와 금지사항을 관리한다.
> 새 실험을 끝내면 [history/README.md](history/README.md)에 결과를 추가하고, 이 문서의 gate를 갱신한다.

## 0. 현재 기준

- 기준 branch는 `develop` / `origin/develop` `ef18498`이다.
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
| G4 source contract implementation | optional | ETRI, lane-val OD pseudo, signal attr teacher 중 하나를 고르면 source key/schema/manifest/loader/test를 함께 구현한다. |
| G5 ROS2 runtime check | after export candidate | latency, GPU memory, frame rate, output schema, overlay/sample replay를 확인한다. |

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

## 5. Export / ROS Gate

Export candidate가 생기면 아래를 확인한다.

- TorchScript artifact and adjacent metadata are written next to checkpoint.
- Metadata output names match actual prediction tensors.
- Roadmark trunk/head channel contract is P2/P3/P4/P5.
- Postprocess output stays in raw image coordinates for det/lane/stop_line/crosswalk.
- ROS2 or sample replay sees the same output schema and practical latency/memory bounds.

## 6. Documentation Rule

- Top-level docs stay short and current.
- Current status belongs in `00A_CURRENT_STATUS.md`; next work and gates belong in this file.
- Detailed failed attempts stay in `docs/history/`.
- Legacy design originals stay in `docs/legacy/`.
- Do not reintroduce per-run status docs unless there is an active multi-day execution that cannot fit in `00A`/`00C`.
- A doc link in README/PRD/active docs must point to an existing file.
