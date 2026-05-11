# 00A. Current Status

> 다음 작업자는 이 문서를 먼저 읽는다.
> 상세 실패 이력은 `00B_STATUS_HISTORY.md`, 다음 실행 gate는 `00C_NEXT_GATES.md`를 본다.

## 1. 한 줄 결론

PV26은 exhaustive OD + lane-family 통합 학습 경로와 derived fine-tune 경로가 구현되어 있고, lane-family는 exact epoch-2 replay 기준 `phase_objective=0.6088677363`, exact postprocess threshold probe 기준 `0.6115270643`까지 확인됐다.

이 60% 돌파는 raw model만으로 만든 결론이 아니고, 세 task F1이 모두 0.6을 넘었다는 뜻도 아니다. core-centerline/refinement checkpoint 위에 small-fragment FP를 제거하는 postprocess geometry filters가 붙어서 만든 partial success다. Gate 4 exact probe에서는 crosswalk F1이 `0.6027`까지 올라갔지만, broader validation 통과 전에는 최종 성공으로 보지 않는다.

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

Gate 4 exact crosswalk threshold candidate:

- artifact: `analysis_exports/crosswalk_postprocess_thresholds_val128_epoch2/summary.json`
- top objective variant: `lane_obj_0.35__cross_mask_0.40__cross_area_32`
- objective: `0.6115270642619861`
- lane / stop-line / crosswalk F1: `0.5326 / 0.4483 / 0.6027`
- crosswalk TP/FP/FN: `44 / 21 / 37`
- broader-val512 crosswalk-only replay: baseline lane/stop/cross F1 `0.5101 / 0.4083 / 0.5854`, candidate `0.5101 / 0.4083 / 0.5845`.
- 판단: exact val128에서는 crosswalk gap을 닫았지만 broader-val512에서 유지되지 않았다. `cross_mask=0.40`, `cross_area=32` threshold tightening은 Gate 4 success가 아니다.

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
- selector/row/x dense-map audit은 val128 epoch2에서 stop-line mask F1 `0.6646`, row-mask F1 `0.6727`, x-mask F1 `0.7482`를 보였지만 selector-mask F1은 `0.3119`, selector-centerline F1은 `0.4259`에 그쳤다. x projection과 mask signal은 남아 있지만, selector/proposal readout이 full line segment로 결합되지 못하는 쪽이 병목이다.
- row/x projection을 직접 span proposal로 바꾸는 read-only probe도 baseline exact val128 stop-line F1 `0.4483`을 넘지 못했다. best fallback은 `0.4310`, best replacement는 `0.3146`이고, row/x decoder는 512 samples 중 29~30 samples에서만 line을 만들었다. 기존 row/x 신호를 후처리 span으로 꺼내는 방식도 0.6 path가 아니다.
- row/x/mask를 학습 target부터 묶는 간단한 rowx-band selector contract도 짧은 run에서 stop-line을 회복하지 못했다. `core_centerline_refine_stop_selector_rowx_band` exact val128 epoch2는 `phase_objective=0.6036`, lane/stop/cross F1 `0.5260 / 0.4144 / 0.5854`, stop-line TP/FP/FN `23 / 28 / 37`로 기준 `0.5267 / 0.4483 / 0.5854`보다 stop-line이 낮다. epoch1 crosswalk task-best `0.6790`은 joint goal signal이 아니다.
- GT-center + angle-anchored mask extent diagnostic은 half-length scalar 대신 mask extent로 길이를 읽으면 headroom이 있음을 보였다. val128에서 predicted angle + mask extent는 stop-line F1 `0.6126`, TP/FP/FN `34 / 17 / 26`까지 올라가지만, broader-val512에서는 `0.5361`, TP/FP/FN `130 / 84 / 141`로 목표 미달이다. GT angle upper bound도 broader-val512 `0.5608`라서 최종 성공은 아니지만, 다음 production 후보는 half-length scalar가 아니라 center proposal + angle-anchored mask extent readout이다.
- production predicted proposal + angle-anchored mask extent readout도 exact val128에서 PCA reference를 넘지 못했다. best `pred_selector_top1_s060_mask050_band4`는 stop-line F1 `0.5085`, TP/FP/FN `30 / 28 / 30`으로 baseline `0.4483`보다 낫지만 prior PCA val128 reference `0.5133`보다 낮다. broader-val512로 확장하지 않는다.
- proposal recall audit은 broader-val512에서 GT 주변 local proposal signal은 남아 있지만 top-k ranking이 약하다는 쪽을 보였다. `max(center, selector)` map은 `max_r8 >= 0.6`이 `208/271`이지만 top3-r8 hit는 `137/271`, raw rank top3는 `14/271`뿐이다. 다음 stop-line 축은 local score 존재 여부보다 proposal ranking/competition 또는 candidate validation contract다.
- candidate-pool audit은 top10/top20 후보 안에 valid stop-line segment headroom이 있음을 보였다. broader-val512 oracle-positive selection은 stop-line F1 `0.6517`, TP/FP/FN `131 / 0 / 140`까지 가능하지만, production score/length filters best는 `0.4371`, TP/FP/FN `106 / 108 / 165`로 PCA broader reference `0.4699`보다 낮다. 즉 candidate pool은 있으나 non-GT FP suppression signal이 아직 없다.
- hard-negative proposal ranking loss short run은 proposal competition을 직접 누르는 opt-in loss를 시험했지만 exact val128에서 확장 조건을 만들지 못했다. stage4-only 2epoch, train512/val128, weight `0.25` best는 lane/stop/cross F1 `0.4754 / 0.4918 / 0.5767`, stop-line TP/FP/FN `30 / 32 / 30`이다. stop-line은 baseline `0.4483`보다 높지만 angle-mask production `0.5085`와 PCA val128 reference `0.5133`보다 낮고 lane/crosswalk가 같이 내려갔다. 같은 checkpoint의 candidate-pool production best도 stop-line F1 `0.4286`이라 broader-val512로 확장하지 않는다.
- endpoint-delta target/readout short run은 val128 epoch1/2 stop-line F1이 모두 `0.0000`이고 best objective도 `0.5880863169`라 기준선보다 낮았다. 새 dense geometry channel을 바로 decode에 쓰는 형태도 현재는 0.6 path가 아니다.
- heatmap-support geometry target fill은 center heatmap support 전체에 offset/angle/half-length target을 채우는 opt-in target 계약을 시험했지만 exact val128 epoch2 stop-line F1이 `0.2338`로 무너졌다. dense stop-line mask F1도 `0.4854`, center heatmap F1도 `0.0815`로 기준 `0.6118 / 0.1396`보다 낮아 같은 형태로 반복하지 않는다.
- row-center auxiliary는 row selector에 centerline-row pressure를 추가해 predicted center/proposal reliability를 올리는지 봤다. exact val128 epoch2 objective `0.6060`, lane/stop/cross F1 `0.5265 / 0.4248 / 0.5818`로 기준 exact `0.6089`, `0.5267 / 0.4483 / 0.5854`보다 낮아서 row-center-aux-only도 0.6 path가 아니다.
- read-only component/readout audit은 broader-val512 GT 271개 중 production TP `98`, anchorless component fit close `123`, anchored fit close `119`를 보였다. GT tube의 mask/center signal은 각각 `223/271`, `220/271`에서 `>=0.50`로 남아 있지만, production FN 173개 중 anchorless fit으로 새로 40px 안에 들어오는 것은 34개뿐이다. 따라서 단순 anchor swap/no-anchor PCA만으로는 0.6 path가 아니다.
- git branch history의 older stop-line oracle/repair 계열도 같이 보면, GT target mask oracle은 stop-line F1 `0.8228`까지 가능했고 GT-overlap oracle은 `component_fit_or_endpoint_error=97` 중 `50`개를 40px 안으로 복구했다. 하지만 그 뒤 non-oracle core-row trim, high-confidence cleanup/PCA endpoint trim, split fit은 각각 기준 PCA reference를 넘지 못했다. oracle headroom은 실제지만, 단순 row-band/core-row, high-confidence subcomponent cleanup, PCA endpoint quantile trim, component split fitting만으로는 회수되지 않는다.
- fit-far visual audit은 production FN, GT tube mask/center `>=0.50`, no-anchor distance `>40px` bucket 상위 18개를 렌더링했다. 18개 모두 production stop-line은 1개씩 있고, 14개는 component_count도 1이라 "아예 안 나옴"보다 single connected component 안에서 wrong line segment를 읽는 문제가 강하다.
- component-conditioned local extraction probe는 predicted component 안에서 center/selector/fused score로 local support를 골라 다시 fit했지만 exact val128 stop-line F1이 기준 `0.4483`을 넘지 못했다. best replacement는 `0.4464`, append-top2는 TP를 늘리는 대신 FP가 크게 늘어 best `0.4054`였다.
- component-split readout probe는 component 내부 multi-line 후보를 pair/Hough-like split으로 만들었지만 exact val128에서 기준을 크게 밑돌았다. best append-top2는 stop-line F1 `0.3421`, TP/FP/FN `26 / 66 / 34`로 baseline TP를 유지하는 대신 FP를 크게 늘렸고, replacement 계열은 TP를 잃어 `0.2857` 이하로 떨어졌다.
- 남은 stop-line 방향은 새 query row, endpoint-delta channel, local score window, pairwise component split, selector-map gate, row/x span proposal, simple rowx-band selector target, 단순 predicted center/selector threshold proposal이 아니다. candidate false-positive를 직접 supervise하거나, mask extent를 읽기 전에 FP 후보를 걸러내는 instance proposal validation contract를 새로 세워야 한다.

Lane Gate 3 dense-map probe:

- command: `python3 tools/probe_pv26_lane60_dense_maps.py --checkpoint .../phase_4/checkpoints/best.pt --preset default --phase-index 4 --max-val-batches 128 --device auto`
- lane centerline core best pixel F1은 `0.5729`이고, lane support best pixel F1은 `0.7971`이다.
- 결론: support map은 이미 충분히 강하고, lane은 vectorizer만의 문제가 아니라 centerline core 품질이 아직 0.6 직전에서 막혀 있다.
- centerline-to-vector recovery audit은 같은 vectorizer에 GT centerline을 넣으면 broader-val512 epoch2 lane F1 `0.6630`까지 복구됨을 보였다. 반면 current predicted centerline은 best threshold `0.35`에서도 lane F1 `0.5169`다.
- predicted semantic attrs를 GT로 바꿔도 `pred_full`과 geometry F1은 같으므로, 현 lane 병목은 color/type attr가 아니라 predicted centerline coverage/quality다.
- core target width를 `1 -> 3`으로 넓힌 `core_centerline_refine_core_width3` short probe는 exact val128 epoch2 `phase_objective=0.6002`까지 올랐지만 lane/stop/cross F1은 `0.5232 / 0.4348 / 0.5854`다. lane F1도 기준 exact `0.5267`보다 낮아서 target-width-only widening은 0.6 path가 아니다.
- centerline error-bucket audit val512는 5092개 supervised lane 중 `recall@0.45 < 0.25`가 291개임을 보였다. 가장 약한 bucket은 `bottom_y < 0.50` miss rate `0.1329`, near-vertical `0.1290`, right-side `x >= 0.66` `0.0821`, left-side `x < 0.33` `0.0705`다.
- side-band centerline BCE positive weighting은 exact val128 lane F1을 `0.5331`로 소폭 올렸지만 phase objective `0.6083`은 기준 `0.6089`보다 낮고 stop-line F1도 `0.4348`로 후퇴했다. lane centerline-core pixel F1도 `0.5705`로 기준 `0.5729`보다 낮아서 side-BCE-only는 0.6 path가 아니다.
- side-band centerline probability margin loss는 exact val128 epoch2 objective를 `0.6097`로 아주 조금 올렸고 lane/stop/cross F1은 `0.5352 / 0.4522 / 0.5854`였다. 하지만 lane centerline-core pixel F1은 `0.5573`으로 기준 `0.5729`보다 더 낮아서 side-margin-only도 centerline 병목 해결이 아니다.
- geometry-risk recall loss는 side/truncated/near-vertical lane을 instance bucket으로 찍어 exact val128 lane F1을 `0.5405`까지 올렸지만 phase objective `0.6086`은 기준보다 낮고 stop-line F1도 `0.4348`로 후퇴했다. lane centerline-core pixel F1도 `0.5572`라 기준 `0.5729`보다 낮아서 risk-recall-only도 centerline 병목 해결이 아니다.
- geometry-risk local Tversky loss는 같은 risk instance 주변 false-positive를 같이 벌주도록 local support mask를 추가했다. exact val128 epoch2 objective는 `0.6089`, lane/stop/cross F1은 `0.5306 / 0.4522 / 0.5854`였고 lane centerline-core pixel F1은 `0.5738`이다. 기준 `0.5729` 대비 gain이 `+0.0009`뿐이고 vectorized lane F1도 recall-only보다 낮아서 broader-val512로 확장하지 않는다.
- lane negative-pixel probability margin은 explicit negative target pixels에서 centerline confidence를 낮추는 반대 압력을 시험했다. exact val128 epoch2 objective `0.6058`, lane/stop/cross F1 `0.5261 / 0.4348 / 0.5854`로 기준 미달이고, lane centerline-core pixel F1도 `0.5736`으로 기준 대비 `+0.0007`뿐이라 broader-val512로 확장하지 않는다.
- support-bridge postprocess는 support를 lane source로 대체하지 않고 centerline binary의 짧은 gap만 high-confidence support 안에서 closing하는 contract를 시험했다. 같은 decode probe baseline lane F1 `0.5222` 대비 best bridge는 `0.4758`이라 recall을 잃었고, support bridge/closing-only도 0.6 path가 아니다.
- endpoint coverage loss는 lane의 visible endpoint heatmap을 추가하고 endpoint positive에서 centerline confidence를 직접 올리는 opt-in loss를 시험했다. exact val128 epoch2 objective `0.6035`, lane/stop/cross F1 `0.5212 / 0.4348 / 0.5854`로 기준 미달이고, lane centerline-core pixel F1도 `0.5670`으로 기준 `0.5729`보다 낮아서 endpoint-only coverage도 0.6 path가 아니다.
- row-scan vectorizer는 connected-component별 vectorization 대신 row cluster track을 opt-in으로 이어서 centerline fragment continuity를 복구했다. exact val128 epoch2 objective는 `0.6144`, lane/stop/cross F1은 `0.5522 / 0.4483 / 0.5854`이고, broader-val512 objective는 `0.5981`, lane/stop/cross F1은 `0.5279 / 0.4083 / 0.5854`다. 18-sample visual comparison grid(`analysis_exports/row_scan_visual_compare_epoch2/row_scan_component_comparison_grid.png`)에서도 일부 side/fragment lane 복구는 보이지만 sample 10처럼 extra/zig track over-link risk가 남아, opt-in partial-positive이지 deployment default는 아니다.
- row-scan geometry guard probe는 length/bottom/gap/dx/turn-angle guard를 exact val128 epoch2에서 비교했다. best lane F1은 `row_gap24_row_dx12`의 `0.5526`으로 기존 row-scan `0.5522` 대비 `+0.0004`뿐이고 FP가 `486 -> 512`로 늘었다. turn-angle guard는 FP를 줄였지만 TP를 더 잃어 best `0.5365`라 default 승격 path가 아니다.
- row-scan residual filter export는 broader-val512 lane TP/FP/FN `4153 / 2105 / 5324`를 남겼다. FN은 left `46.9%`, truncated bottom `<0.50` `27.9%`, aspect `>=3` `65.9%`에 몰리고, FP는 side `74.8%`와 right `40.6%` 비중이 높다.
- residual local separation loss는 left/truncated/high-aspect GT core를 올리고 주변 ring negative를 누르는 opt-in target/loss를 시험했다. exact val128 epoch2 lane F1은 `0.5476`으로 기준 `0.5267`보다 높았지만, stop-line F1은 `0.4310`으로 기준 `0.4483`보다 낮고 phase objective도 `0.6085`로 기준 `0.6089`보다 낮다. lane partial-positive일 뿐 채택/확장하지 않는다.
- BCE-focus calibration은 broader-val512 lane F1을 `0.5101 -> 0.5344`로 올렸지만 stop-line/crosswalk가 내려갔고, centerline-core pixel F1도 `0.5729 -> 0.5680`으로 낮아졌다. goal success가 아니라 lane-vectorized metric partial-positive다.
- BCE-focus + PCA stop-line decoder integration best는 broader-val512 lane/stop/cross F1 `0.5344 / 0.4583 / 0.5741`이고, stop-balance + PCA replay best도 `0.5372 / 0.4528 / 0.5812`에 그쳤다.
- 다음 lane 축은 residual-risk local loss를 더 키우는 방향이 아니라, predicted centerline evidence를 instance 단위로 안정화하거나 row-scan partial-positive를 stop-line/crosswalk 목표와 같이 끌어올리는 contract로 좁힌다. stop-line을 재개한다면 dense signal을 line geometry로 바꾸는 readout/target contract 쪽으로 제한한다.
- Gate 4 crosswalk postprocess probe는 exact val128에서 crosswalk F1을 `0.5854 -> 0.6027`로 올렸지만, broader-val512에서는 `0.5845`로 기준 `0.5854`보다 낮았다. crosswalk stricter component threshold는 exact-only partial-positive로 보관하고 채택하지 않는다.

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
