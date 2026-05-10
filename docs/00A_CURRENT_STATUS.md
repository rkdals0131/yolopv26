# 00A. Current Status

> 다음 작업자는 이 문서를 먼저 읽는다.
> 상세 실패 이력은 `00B_STATUS_HISTORY.md`, 다음 실행 gate는 `00C_NEXT_GATES.md`를 본다.

## 1. 한 줄 결론

PV26은 exhaustive OD + lane-family 통합 학습 경로와 derived fine-tune 경로가 구현되어 있고, lane-family는 exact epoch-2 replay 기준 `phase_objective=0.6088677363`까지 확인됐다.

이 60% 돌파는 raw model만으로 만든 결론이 아니고, F1 자체가 0.6을 넘었다는 뜻도 아니다. core-centerline/refinement checkpoint 위에 small-fragment FP를 제거하는 postprocess geometry filters가 붙어서 만든 partial success다. 다음 목표를 더 엄격하게 잡는다면 `phase_objective`가 아니라 lane/stop/cross F1 자체를 0.6 이상으로 끌어올리는 것이다.

Active goal:

- broader validation에서 lane / stop-line / crosswalk F1이 모두 `>= 0.60`인 checkpoint + postprocess/preprocess/runtime contract를 만든다.
- exact epoch-2 subset이나 `phase_objective` 단독 통과는 중간 신호일 뿐 최종 성공으로 보지 않는다.
- 실험은 branch/worktree 단위로 분리하고, 한 worktree는 한 축만 바꾼다.

## 2. 현재 기준 artifact

Run:

`runs/pv26_exhaustive_od_lane_train/lane60_core_centerline_refine_cross_retain_from_exhaustive_od_lane_default_20260505_032217_default_20260510_003412`

남긴 핵심 파일:

- checkpoint: `phase_4/checkpoints/best.pt`
- final exact eval: `analysis_exports/exact_checkpoint_eval_final_geometry_filters_maskaware_epoch2/summary.json`
- visual check: `analysis_exports/geometry_visual_compare_gt_best_current_epoch2/gt_best_pregeom_current_geometry_grid.png`

Final exact epoch-2 result:

| Metric | Value |
| --- | ---: |
| objective | `0.6088677363246272` |
| lane F1 | `0.526695898890895` |
| stop-line F1 | `0.4482758620689655` |
| crosswalk F1 | `0.5853658536585366` |
| support lane/stop/cross | `2390 / 60 / 81` |

F1 기준 gap:

- lane: `0.5267 -> 0.6000`, `+0.0733` 필요.
- stop-line: `0.4483 -> 0.6000`, `+0.1517` 필요.
- crosswalk: `0.5854 -> 0.6000`, `+0.0146` 필요.
- 따라서 F1 0.6+ 목표의 병목은 stop-line, 그 다음 lane이다. crosswalk는 거의 도달했지만 broader validation에서 유지되는지 확인해야 한다.

Broader-val512 replay result:

- artifact: `analysis_exports/broader_val512_final_geometry_filters_epoch2/summary.json`
- objective: `0.5943438312141003`
- lane / stop-line / crosswalk F1: `0.5101 / 0.4083 / 0.5854`
- support lane / stop / cross: `9477 / 271 / 395`
- 판단: exact epoch-2의 geometry-filter gain은 더 넓은 slice에서도 완전히 사라지지는 않았지만, objective 0.6과 task별 F1 0.6 목표에는 미달이다. 다음 축은 stop-line first가 맞다.

## 3. Active docs surface

Current status set:

- `00A_CURRENT_STATUS.md`: 지금 어디인지.
- `00B_STATUS_HISTORY.md`: 어떻게 여기까지 왔는지.
- `00C_NEXT_GATES.md`: 다음에 할 것과 하지 말 것.

Core contract docs:

- `0_PRD.md`: 범위와 문서 맵.
- `1_DEVELOPMENT_PHILOSOPHY.md`: 작업 철학.
- `2_SYSTEM_ARCHITECTURE.md`: 현재 package/runtime 구조.
- `5_TARGETS_AND_LOSS.md`: target/loss/selection contract.
- `6_TRAINING_AND_EVALUATION.md`: stage schedule, sampler, eval 정책.
- `8_TEST_PLAN_AND_CHECKLIST.md`: 검증 기준.
- `9_EXECUTION_STATUS.md`: 기존 live tracker. 긴 구현 체크리스트 성격이라 점진적으로 status set과 분리한다.
- `11_GIT_BRANCH_WORKFLOW.md`: branch/worktree 운영.
- `17_MODAL_A100_TRAINING_RUNBOOK.md`: Modal A100 절차서.

Legacy 원문:

- 긴 run analysis와 probe 원문은 `docs/legacy/`에 보존한다.
- root docs는 현재 상태와 절차를 빠르게 찾는 표면으로 유지한다.

## 4. 현재 traffic-light 상태

Lane60 fine-tune은 traffic light를 학습한 run이 아니다.

- phase 4 lane60은 `det=0`, `tl_attr=0`이고 det/tl source samples가 없다.
- traffic light evidence는 이전 phase 3 joint run에서 봐야 한다.

현재 판단:

- traffic light box는 나오지만 class detection은 moderate 수준이다.
- matched traffic light box 안에서 attribute combo accuracy는 `~0.81-0.82`까지 나온다.
- end-to-end로 "신호등이 잘 된다"라고 말하려면 detector recall/precision이 먼저 올라야 한다.

가능한 후속 axis:

- traffic source 중심 sampler.
- `det + tl_attr`만 켜는 fine-tune.
- 필요하면 detector loss에서 traffic_light class만 강제 supervised class로 좁히는 config hook 추가.
