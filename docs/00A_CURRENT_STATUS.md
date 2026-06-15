# 00A. Current Status

> 다음 작업자는 이 문서를 먼저 읽는다.
> 상세 실패 이력은 [history/README.md](history/README.md), 다음 실행 gate는 [00C_NEXT_GATES.md](00C_NEXT_GATES.md)를 본다.

## 0. 2026-06-15 기준

- 현재 공식 작업선은 `develop` / `origin/develop`의 `ef18498` (`Align TorchScript export with PV26 runtime heads`)이다.
- `origin/main`은 solid runtime-contract 기준선 `364c019`에 머문다. 로컬 `main`은 `be5106d`로 오래된 상태라 current 판단 기준으로 쓰지 않는다.
- live `exp/lane-family-f1/*` branch는 남기지 않는다. 보존 anchor는 tag `archive/lane-family-current-frontier-20260602`, `archive/lane-family-router-best-20260529`다.
- former 00B top-level file은 제거했다. 과거 실험 ledger는 `docs/history/00B_*.md`에 번호 범위별로 보존한다.
- old refactor map, branch workflow, execution status 문서는 active docs surface에서 빠졌다. 현재 상태는 이 문서, 다음 행동은 `00C_NEXT_GATES.md`가 소유한다.

## 1. 현재 결론

PV26의 exhaustive OD + lane-family 통합 학습/평가 경로는 구현되어 있지만, 최종 benchmark는 아직 통과하지 못했다. Success는 broader validation에서 `lane`, `stop_line`, `crosswalk` F1이 모두 `>= 0.60`인 것이다.

| Surface | Broader lane / stop_line / crosswalk F1 | 판단 |
| --- | --- | --- |
| Best two-checkpoint/router stop-line tradeoff | `0.5628 / 0.5309 / 0.6187` | stop-line 최고 tradeoff지만 single checkpoint success가 아니다. |
| Retained lane-preserving runtime composite | `0.5628 / 0.5164 / 0.6187` | main에 올린 solid runtime contract 계열. |
| Best single trained stop/cross lane-frozen composite | `0.5571 / 0.5278 / 0.6142` | crosswalk는 넘지만 lane/stop-line 미달. |
| Current learned lane replay frontier | `0.5851 / 0.5302 / 0.5969` | lane이 가장 앞섰지만 all-task gate는 실패. |

정성 playback은 유용도 `60-65 / 100` 정도로 기록되어 있다. 이것은 product utility 신호일 뿐 benchmark success로 바꾸지 않는다.

## 2. Runtime Surface

현재 유지하는 실행 표면:

- `python3 tools/check_env.py`
- `python3 tools/check_env.py --strict --check-yolo-runtime`
- `python3 tools/run_pv26_train.py --preset default`
- `python3 tools/run_pv26_train.py --resume-run ...`
- `python3 tools/run_pv26_train.py --derive-run ...`
- `python -m tools.od_bootstrap ...`
- `tools/model_export/pv26_torchscript.py`
- `tools/analyze_pv26_run.py`
- `tools/modal/`

최근 반영된 runtime 기준:

- TorchScript export는 current PV26 raw heads와 동기화되어야 한다. `det`, `tl_attr`, `lane`, `stop_line`, `crosswalk` legacy heads뿐 아니라 현재 dense/aux heads 중 실제 prediction에 있는 output names만 metadata에 기록한다.
- YOLO26 roadmark trunk는 4-level P2/P3/P4/P5 contract를 쓴다. `yolo26s` default channel은 `(128, 128, 256, 512)`다.
- teacher bootstrap은 `check_env`에서 실행 가능한 방향으로 정리됐다.
- teacher train 기본 batch는 local dense-head headroom 기준으로 낮춰졌다: mobility/signal `20`, obstacle `10`.
- rich progress bar는 optional dependency/fallback을 안전하게 처리한다.
- stale/malformed data handoff는 source, loader, transform, batch, loss, postprocess, export boundary에서 fail-fast해야 한다.

Maintained ownership checkpoints:

- `model.engine.batch.merge_raw_batches`, `model.engine.det_geometry`, `model.engine.train_summary`, `model.engine.trainer_progress`, `model.engine.trainer_runtime` are public/shared runtime surfaces.
- `tools/check_env/launch.py` owns launcher/input/resume flow behind the `tools/check_env.py` facade.
- `tools/od_bootstrap/teacher/runtime/trainer.py` and `tools/od_bootstrap/teacher/runtime/progress.py` own teacher runtime/progress helpers.
- `common.io.write_json(overwrite=False)`, `common.io.write_json(default=str)`, `common.io.write_text(...)`, and `common.io.write_jsonl(...)` are the documented JSON/text/JSONL emission helpers.
- teacher summary/report JSON serialization must go through the common helper argument path, not ad hoc direct `json.dumps(..., default=str)` call-sites.

## 3. Dataset And Source Status

현재 schema에 등록된 source key만 loader가 받는다:

- `pv26_exhaustive_bdd100k_det_100k`
- `pv26_exhaustive_aihub_traffic_seoul`
- `pv26_exhaustive_aihub_obstacle_seoul`
- `aihub_traffic_seoul`
- `aihub_obstacle_seoul`
- `aihub_lane_seoul`
- `bdd100k_det_100k`

새 문서 `18`, `19`, `20`은 구현 완료가 아니라 계약 고정 문서다.

- [18_ETRI_KCITY_CAMERA_TO_PV26_LABELS.md](18_ETRI_KCITY_CAMERA_TO_PV26_LABELS.md): ETRI KCity `leftImg` 변환 계약. 아직 source key가 코드에 등록되지 않았다.
- [19_LANE_VAL_OD_TEACHER_EVALSET.md](19_LANE_VAL_OD_TEACHER_EVALSET.md): lane validation + OD teacher pseudo eval root 계약. train source가 아니다.
- [20_SIGNAL_ATTR_TEACHER_PLAN.md](20_SIGNAL_ATTR_TEACHER_PLAN.md): `best_signal.pt`는 box teacher이고, TL attr은 별도 `best_signal_attr.pt` sidecar가 필요하다는 경계.

최종 dataset count checkpoint는 `seg_dataset/pv26_exhaustive_od_lane_dataset/meta/final_dataset_stats.json` 기준이다.

| Target | Positive images | Instances | 판단 |
| --- | ---: | ---: | --- |
| traffic_light det | `80,916` | `233,902` | 전체 support는 있지만 close/medium-plus bucket이 얇다. |
| traffic-light medium_plus | `5,910` | `7,552` | close-range weakness 우선 수집 후보. |
| any lane | `132,097` | - | 총량은 충분하지만 색/유형 imbalance가 남아 있다. |
| stop_line | `18,797` | `25,435` | 낮은 support, 아직 주요 bottleneck. |
| crosswalk | `23,343` | `38,709` | 낮은 support지만 hull decode에서는 partial pass. |
| vehicle det | `283,624` | `2,355,495` | 가장 강한 OD support. |

## 4. Active Docs Surface

- [0_PRD.md](0_PRD.md): 저장소 목표와 문서 맵.
- [00A_CURRENT_STATUS.md](00A_CURRENT_STATUS.md): 현재 snapshot.
- [00C_NEXT_GATES.md](00C_NEXT_GATES.md): 다음 gate와 금지사항.
- [history/README.md](history/README.md): split history index.
- [1_DEVELOPMENT_PHILOSOPHY.md](1_DEVELOPMENT_PHILOSOPHY.md): 운영 철학.
- [2_SYSTEM_ARCHITECTURE.md](2_SYSTEM_ARCHITECTURE.md): package/runtime 구조.
- [5_TARGETS_AND_LOSS.md](5_TARGETS_AND_LOSS.md): target/loss/selection contract.
- [6_TRAINING_AND_EVALUATION.md](6_TRAINING_AND_EVALUATION.md): stage schedule, sampler, eval 정책.
- [8_TEST_PLAN_AND_CHECKLIST.md](8_TEST_PLAN_AND_CHECKLIST.md): 검증 기준.
- [18_ETRI_KCITY_CAMERA_TO_PV26_LABELS.md](18_ETRI_KCITY_CAMERA_TO_PV26_LABELS.md): ETRI source 계약.
- [19_LANE_VAL_OD_TEACHER_EVALSET.md](19_LANE_VAL_OD_TEACHER_EVALSET.md): lane-val OD pseudo eval 계약.
- [20_SIGNAL_ATTR_TEACHER_PLAN.md](20_SIGNAL_ATTR_TEACHER_PLAN.md): traffic-light attr teacher 계약.
- [legacy/](legacy/): 긴 과거 설계 원문.

## 5. Operating Rules

- closed negative experiment를 반복하지 않는다. 먼저 [history/README.md](history/README.md)에서 관련 번호 범위를 찾는다.
- 새 source key는 `common/pv26_schema.py`, loader/final dataset manifest, docs, tests가 함께 움직일 때만 추가한다.
- ETRI, lane-val OD pseudo, signal attr teacher는 서로 다른 source/eval/teacher 계약이다. 한 root나 source key로 섞지 않는다.
- branch-per-attempt는 금지한다. 긴 연구 방향이 갈라질 때만 branch를 만들고, 고정 anchor는 tag로 남긴다.
- run artifact는 checkpoint/summary/manifest/compact CSV 위주로 남기고, negative probe weight와 TensorBoard bulk는 유지하지 않는다.
