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
- 판단: exact epoch-2의 geometry-filter gain은 더 넓은 slice에서도 완전히 사라지지는 않았지만, objective 0.6과 task별 F1 0.6 목표에는 미달이다.

Stop-line Gate 2 follow-up:

- 여러 worktree에서 threshold, sampler, mask loss, mask vectorizer, center target, dense geometry target, selector center, center stem, component fitting, endpoint supervision, feature isolation을 한 축씩 시험했다.
- best decoder-only 후보는 original checkpoint + `component_pca_full_mask080_score094`로 broader-val512 stop-line F1 `0.4699`, TP/FP/FN `113 / 97 / 158`이다.
- 이 값은 기준선 `0.4083`보다 낫지만, 목표 `0.60`까지는 아직 멀다.
- stop-line dense mask pixel F1은 val128 probe에서 `0.6107`까지 나오지만, center heatmap F1은 `0.1385`라서 mask 존재보다 endpoint/geometry/selector 복원이 병목이다.
- 이후 component split, center-cell geometry mask, half-length scale/loss/log target, learned query-vector proposal, endpoint-delta target/readout도 각각 별도 branch에서 닫았다. 모두 stop-line F1 0.6 path가 아니었다.
- query-vector proposal short run은 vector-only exact val128 epoch1/2 stop-line F1이 모두 `0.0000`이고, threshold를 `0.10`까지 낮춰도 TP/FP/FN `0 / 0 / 55`였다. append mode도 mask baseline 수준에 머물렀다.
- selector-map을 component gate/anchor에 opt-in으로 연결한 read-only decode probe도 stop-line F1 `0.2062`로 `stop_mask_only` `0.2593`보다 낮았다. centerline selector map을 단순 component 선택에 쓰는 후처리만으로는 0.6 path가 아니다.
- endpoint-delta target/readout short run은 val128 epoch1/2 stop-line F1이 모두 `0.0000`이고 best objective도 `0.5880863169`라 기준선보다 낮았다. 새 dense geometry channel을 바로 decode에 쓰는 형태도 현재는 0.6 path가 아니다.
- 남은 stop-line 방향은 새 query row나 endpoint-delta channel 추가가 아니라, predicted center/proposal reliability를 먼저 올리거나 PCA/component-fit weak-positive를 넘어서는 다른 geometry recovery contract를 찾는 것이다.

Lane Gate 3 dense-map probe:

- command: `python3 tools/probe_pv26_lane60_dense_maps.py --checkpoint .../phase_4/checkpoints/best.pt --preset default --phase-index 4 --max-val-batches 128 --device auto`
- lane centerline core best pixel F1은 `0.5729`이고, lane support best pixel F1은 `0.7971`이다.
- 결론: support map은 이미 충분히 강하고, lane은 vectorizer만의 문제가 아니라 centerline core 품질이 아직 0.6 직전에서 막혀 있다.
- centerline-to-vector recovery audit은 같은 vectorizer에 GT centerline을 넣으면 broader-val512 epoch2 lane F1 `0.6630`까지 복구됨을 보였다. 반면 current predicted centerline은 best threshold `0.35`에서도 lane F1 `0.5169`다.
- predicted semantic attrs를 GT로 바꿔도 `pred_full`과 geometry F1은 같으므로, 현 lane 병목은 color/type attr가 아니라 predicted centerline coverage/quality다.
- core target width를 `1 -> 3`으로 넓힌 `core_centerline_refine_core_width3` short probe는 exact val128 epoch2 `phase_objective=0.6002`까지 올랐지만 lane/stop/cross F1은 `0.5232 / 0.4348 / 0.5854`다. lane F1도 기준 exact `0.5267`보다 낮아서 target-width-only widening은 0.6 path가 아니다.
- centerline error-bucket audit val512는 5092개 supervised lane 중 `recall@0.45 < 0.25`가 291개임을 보였다. 가장 약한 bucket은 `bottom_y < 0.50` miss rate `0.1329`, near-vertical `0.1290`, right-side `x >= 0.66` `0.0821`, left-side `x < 0.33` `0.0705`다.
- BCE-focus calibration은 broader-val512 lane F1을 `0.5101 -> 0.5344`로 올렸지만 stop-line/crosswalk가 내려갔고, centerline-core pixel F1도 `0.5729 -> 0.5680`으로 낮아졌다. goal success가 아니라 lane-vectorized metric partial-positive다.
- BCE-focus + PCA stop-line decoder integration best는 broader-val512 lane/stop/cross F1 `0.5344 / 0.4583 / 0.5741`이고, stop-balance + PCA replay best도 `0.5372 / 0.4528 / 0.5812`에 그쳤다.
- 다음 lane 축은 vectorizer rewrite나 threshold sweep이 아니라 side/truncated/near-vertical lane의 predicted centerline core recall을 올리는 한 축이다. stop-line을 재개한다면 dense signal을 line geometry로 바꾸는 readout/target contract 쪽으로 제한한다.

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
