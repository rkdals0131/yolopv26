# 00A. Current Status

> 다음 작업자는 이 문서를 먼저 읽는다.
> 상세 실패 이력은 `00B_STATUS_HISTORY.md`, 다음 실행 gate는 `00C_NEXT_GATES.md`를 본다.

## 1. 한 줄 결론

PV26은 exhaustive OD + lane-family 통합 학습 경로와 derived fine-tune 경로가 구현되어 있고, lane-family는 exact epoch-2 runtime-TTA probe 기준 `phase_objective=0.6296149306`, broader-val512 runtime/postprocess composite replay 기준 `0.6216194906`까지 확인됐다.

이 60% objective 돌파는 raw model만으로 만든 결론이 아니고, 세 task F1이 모두 0.6을 넘었다는 뜻도 아니다. core-centerline/refinement checkpoint 위에 row-scan/tangent-link vectorizer, small-fragment FP postprocess filters, lane-head transplant, flip-centerline TTA, 또는 hull-based crosswalk decode가 붙어서 만든 partial success다. Broader-val512 composite에서도 objective는 `0.6216`까지 올라갔지만 lane/stop-line/crosswalk F1은 `0.5577 / 0.4235 / 0.6187`이라 최종 성공으로 보지 않는다.

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

Current broader-composite F1 기준 gap:

- lane: `0.5577 -> 0.6000`, `+0.0423` 필요.
- stop-line: `0.4235 -> 0.6000`, `+0.1765` 필요.
- crosswalk: `0.6187`, broader-val512에서 `0.6000` 이상 통과.
- 따라서 F1 0.6+ 목표의 병목은 stop-line, 그 다음 lane이다. crosswalk는 hull decode로 broader pass를 만들었지만, 이 자체는 opt-in postprocess partial-positive이고 lane/stop-line 실패를 가리지 않는다.

Gate 4 exact crosswalk threshold candidate:

- artifact: `analysis_exports/crosswalk_postprocess_thresholds_val128_epoch2/summary.json`
- top objective variant: `lane_obj_0.35__cross_mask_0.40__cross_area_32`
- objective: `0.6115270642619861`
- lane / stop-line / crosswalk F1: `0.5326 / 0.4483 / 0.6027`
- crosswalk TP/FP/FN: `44 / 21 / 37`
- broader-val512 crosswalk-only replay: baseline lane/stop/cross F1 `0.5101 / 0.4083 / 0.5854`, candidate `0.5101 / 0.4083 / 0.5845`.
- 판단: exact val128에서는 crosswalk gap을 닫았지만 broader-val512에서 유지되지 않았다. `cross_mask=0.40`, `cross_area=32` threshold tightening은 Gate 4 success가 아니다.

Gate 4 broader crosswalk shape sweep:

- artifact: `analysis_exports/crosswalk_broader_shape_sweep_val512_epoch2/summary.json`
- variants: `64` crosswalk-only object/mask/component-area/polygon-area/aspect/top-k variants.
- best crosswalk F1 variant: `cross_aspect_2.0`, lane/stop/cross F1 `0.5101 / 0.4083 / 0.5960`, cross TP/FP/FN `239 / 168 / 156`.
- best objective variant: `cross_mask_0.70`, objective `0.5970434363`, lane/stop/cross F1 `0.5101 / 0.4083 / 0.5887`.
- 판단: broader-val512에서 crosswalk F1 `>=0.60` variant가 `0/64`다. 단순 crosswalk threshold/shape postprocess는 Gate 4 success path가 아니다.

Current crosswalk hull decode replay:

- artifact exact: `analysis_exports/crosswalk_hull_decode_val128_epoch2/summary.json`
- artifact broader: `analysis_exports/crosswalk_hull_decode_val512_epoch2/summary.json`
- opt-in postprocess: `crosswalk_polygon_mode=hull`
- exact val128 hull result: objective `0.6246557098`, lane/stop/cross F1 `0.5633 / 0.4483 / 0.5988`.
- broader-val512 hull result: objective `0.6127256636`, lane/stop/cross F1 `0.5407 / 0.4083 / 0.6187`, cross TP/FP/FN `232 / 123 / 163`.
- 판단: rectangle/aspect threshold sweep은 실패했지만, minimum-area rectangle 대신 convex hull을 쓰는 representation-aware decode는 broader-val512 crosswalk gap을 닫았다. 하지만 lane과 stop-line은 여전히 목표 미달이라 all-task success/default가 아니다.

Broader-val512 replay result:

- artifact: `analysis_exports/broader_val512_final_geometry_filters_epoch2/summary.json`
- objective: `0.5943438312141003`
- lane / stop-line / crosswalk F1: `0.5101 / 0.4083 / 0.5854`
- support lane / stop / cross: `9477 / 271 / 395`
- 판단: exact epoch-2의 geometry-filter gain은 더 넓은 slice에서도 완전히 사라지지는 않았지만, objective 0.6과 task별 F1 0.6 목표에는 미달이다.

Current best broader lane replay:

- artifact: `analysis_exports/lane_row_scan_tangent_link_val512_epoch2/summary.json`
- experiment: `core_centerline_refine_row_scan_tangent_link`
- objective: `0.6027206157496527`
- lane / stop-line / crosswalk F1: `0.5407 / 0.4083 / 0.5854`
- support lane / stop / cross: `9477 / 271 / 395`
- 판단: row-scan보다 lane continuity와 objective는 올랐지만 stop-line과 crosswalk가 목표 미달이라 success/default가 아니다.

Current best broader runtime/postprocess composite by objective:

- artifact: `runs/pv26_exhaustive_od_lane_train/lane60_lane_flip_tta_audit_20260512/analysis_exports/broader_val512_current_best_epoch2/summary.json`
- experiment: `core_centerline_refine_row_scan_tangent_link`
- checkpoint composition: original `best.pt` as base, lane head from segment-MIL lane-head-only `best_lane.pt`, stop-line and crosswalk heads from original `best.pt`.
- runtime/postprocess: average only `lane_seg_centerline_logits` from the normal image and horizontal-flip image, then keep stop-line/crosswalk outputs from the normal pass.
- evaluator-only overrides: stop-line `mask=0.80`, stop-line `min_instance_score=0.94`, stop-line `presence=0.0`, crosswalk `polygon_mode=hull`.
- objective: `0.6216194905914751`
- lane / stop-line / crosswalk F1: `0.5577 / 0.4235 / 0.6187`
- TP/FP/FN lane: `4518 / 2206 / 4959`
- TP/FP/FN stop-line: `101 / 105 / 170`
- TP/FP/FN crosswalk: `232 / 123 / 163`
- support lane / stop / cross: `9477 / 271 / 395`
- 판단: flip-centerline TTA recovers a real but small broader lane gain over the same transplanted composite (`0.5480 -> 0.5577`) while preserving stop-line `0.4235` and hull crosswalk `0.6187`. This is a new objective best but still not all-task success because lane and stop-line remain below `0.60`.

Current best broader task-balance replay:

- branch/worktree: `exp/lane-family-f1/stopline-projcomp-flip-composite`.
- code commit: none; artifact-only replay using the existing projection-competition CSV tool and the current flip-centerline reference row.
- artifact: `runs/pv26_exhaustive_od_lane_train/stopline_projcomp_flip_composite_20260513/analysis_exports/val512_epoch2/summary.json`.
- changed axis: keep current `flip_centerline_avg` lane and hull crosswalk metrics, then replay projection-competition stop-line predictions from the same checkpoint/candidate pool.
- lane / stop-line / crosswalk F1: `0.5577 / 0.5164 / 0.6187`.
- stop-line TP/FP/FN: `126 / 91 / 145`.
- lane-family mean/min F1: `0.5643 / 0.5164`.
- 판단: this is a better task-balance lower bound than the objective-best runtime composite, but it still fails all-task `0.60`: lane needs `+0.0423` and stop-line needs `+0.0836`.

Latest stop-line no-oracle axis-offset budget:

- branch/worktree: `exp/lane-family-f1/stopline-axis-offset-budget`.
- code commit: `68585c1`.
- artifact: `runs/pv26_exhaustive_od_lane_train/stopline_axis_offset_budget_20260513/analysis_exports/val512_epoch2/summary.json`.
- changed axis: keep the projection-competition reference fixed, then replace only positive-no-oracle local candidates with GT-only axis-projection oracle variants to separate along-line center error, normal error, and length error.
- local positive-no-oracle rows: `51`; axis-dominant rows: `49`; abs-normal-offset q50/q90: `2.31px / 13.47px`; abs-along-offset q50/q90: `68.59px / 137.64px`.
- projection reference stop-line F1: `0.5164`, TP/FP/FN `126 / 91 / 145`.
- axis-projection + fixed minlen best (`minlen=240`) stop-line F1: `0.5547`, TP/FP/FN `137 / 86 / 134`.
- axis-projection + GT-length oracle stop-line F1: `0.6559`, TP/FP/FN `162 / 61 / 109`, close to full GT-midpoint+GT-length oracle `0.6599`.
- 판단: no-oracle local candidates are mostly wrong along the stop-line axis, not off the line. But fixed-length axis shift is still below `0.60`; a real production path needs a no-GT signal for both along-axis midpoint shift and stop-line extent/length, not another selector-only or symmetric extension sweep.

Latest stop-line axis-projected offset readout:

- branch/worktree: `exp/lane-family-f1/stopline-axis-projected-offset-readout`.
- code commit: `107a0f5`.
- artifact exact: `runs/pv26_exhaustive_od_lane_train/stopline_axis_projected_offset_readout_20260513/analysis_exports/val128_epoch2/summary.json`.
- changed axis: keep the same predicted proposal + angle-mask extent readout, but project the existing predicted center offset onto the predicted stop-line angle axis before decoding.
- exact val128 existing pred-offset reference: stop-line F1 `0.5085`, TP/FP/FN `30 / 28 / 30`, mean point distance `13.62`.
- exact val128 axis-projected offset: stop-line F1 `0.5085`, TP/FP/FN `30 / 28 / 30`, mean point distance `13.51`.
- 판단: current model center-offset already gives the same matched set after axis projection; the tiny point-distance improvement is not a stop-line F1 path. Do not broaden or repeat this as a `top_k`/threshold sweep.

Latest stop-line fragment axis contract:

- branch/worktree: `exp/lane-family-f1/stopline-fragment-axis-contract`.
- model-side code commit: `6cc3f52`; low-disk metric-only helper commit: `ae1fbc2`.
- changed axis: train the fragment-center offset as a stop-line-axis scalar instead of 2D xy offset, and decode fragment extent with the same axis projection.
- exact val128 metric-only run completed without checkpoint/TensorBoard writes; run size was `2.5MB`, checkpoint paths were `null`, and skipped steps were `0`.
- epoch1 lane/stop/cross F1: `0.5081 / 0.0625 / 0.6835`, objective `0.5351`.
- epoch2 lane/stop/cross F1: `0.5225 / 0.1905 / 0.5714`, objective `0.5522`.
- 판단: this closes the axis-scalar fragment contract as a performance negative. It fixed the disk-full observability problem, but the actual stop-line result is far below tangent-link exact `0.4483`, PCA val128 `0.5133`, angle-mask production `0.5085`, and projection-competition broader reference `0.5164`; do not broaden or repeat it as an aux-weight/top-k/min-score/epoch sweep.

Latest stop-line axis-support span audit:

- branch/worktree: `exp/lane-family-f1/stopline-axis-support-span-audit`.
- code commit: `3856bbe`.
- artifact: `runs/pv26_exhaustive_od_lane_train/stopline_axis_support_span_audit_20260513/analysis_exports/val512_epoch2/summary.json`.
- changed axis: keep the projection-competition reference fixed, then replace positive-no-oracle local candidates with a no-GT same-axis support span built from same-sample high-score candidates.
- projection-competition reference: stop-line F1 `0.5164`, TP/FP/FN `126 / 91 / 145`.
- best support-span replay (`minmembers=8`): stop-line F1 `0.5085`, TP/FP/FN `119 / 78 / 152`.
- support span length ratio q50 improves from `0.544` to `1.013`, but midpoint distance q50 only moves `68.59px -> 61.92px` and q90 worsens to `215.68px`.
- 판단: same-axis support span reduces FP only by dropping too many TP; it is not the missing no-GT midpoint/extent signal. Do not repeat as a top-k/min-score/member-count/angle/normal-threshold sweep.

Latest lane FN recovery audit:

- branch/worktree: `exp/lane-family-f1/lane-fn-nearby-fp-recovery-audit`.
- code commit: `978fc88`.
- artifact: `runs/pv26_exhaustive_od_lane_train/lane_fn_recovery_audit_20260513/analysis_exports/broader_val512_epoch2/summary.json`.
- changed axis: no training and no production decoder change; replay the current flip-centerline broader composite, then inspect each missed GT lane for predicted centerline evidence on the GT polyline and nearby unmatched row-scan tracks.
- baseline lane/stop/cross F1: `0.5577 / 0.4235 / 0.6187`.
- lane TP/FP/FN: `4518 / 2206 / 4959`.
- FN evidence: `2066 / 4959` missed lanes have `gt_center_point_mean >= 0.30`; `1363 / 4959` have `>= 0.50`; `1698 / 4959` have an unmatched predicted lane within `120px`; `2588 / 4959` satisfy `center_mean >= 0.30` or unmatched distance `<=120px`.
- no-new-FP upper-bound control: recovering the `center_mean >= 0.30 or unmatched <=120px` subset would imply lane F1 `0.7564`, but this is diagnostic only and uses GT to count recoverable FNs.
- 판단: lane still has recall-side headroom that is not explained by another FP selector threshold. The next lane branch should convert this into a recall-preserving decoder/model-side instance recovery contract; do not claim this read-only upper-bound as production lane success.

Latest lane FN joint-strata audit:

- branch/worktree: `exp/lane-family-f1/lane-fn-joint-strata-audit`.
- code commit: `13bb290`.
- artifact: `runs/pv26_exhaustive_od_lane_train/lane_fn_joint_strata_audit_20260513/analysis_exports/broader_val512_epoch2/summary.json`.
- changed axis: read existing broader-val512 FN rows and split recovery headroom by GT centerline evidence and nearest unmatched predicted track distance.
- lane F1 `0.60` at current FP requires `489` recovered FN.
- `center_mean >= 0.50 AND unmatched <=80px`: `563` FN, no-new-FP upper-bound lane F1 `0.6062`.
- `center_mean >= 0.50 AND unmatched <=120px`: `836` FN, no-new-FP upper-bound lane F1 `0.6285`.
- `center_mean >= 0.50 WITHOUT unmatched <=120px`: `527` FN, no-new-FP upper-bound lane F1 `0.6032`.
- `unmatched <=120px WITHOUT center_mean >=0.50`: `862` FN, no-new-FP upper-bound lane F1 `0.6306`.
- 판단: the lane budget is not one bucket. Both nearby-track repair and centerline-only generation are individually large enough on GT-labeled upper bounds, but the centerline-snap smoke shows that simple local x-snapping is not the production contract.

Latest lane FN pair-geometry audit:

- branch/worktree: `exp/lane-family-f1/lane-fn-pair-geometry-audit`.
- code commit: `941ddd8`.
- artifact: `runs/pv26_exhaustive_od_lane_train/lane_fn_pair_geometry_audit_20260513/analysis_exports/broader_val512_epoch2/summary.json`.
- changed axis: replay the same current flip-centerline composite, but add nearest-prediction geometry features for each FN: length ratio, angle error, center offset, endpoint distance, sample distance, and overlap.
- `center>=0.50 AND unmatched<=80px`: count `563`, nearest distance q50 `53.25px`, length ratio q50 `0.974`, angle error q50 `1.27deg`, center distance q50 `50.11px`, y-overlap q50 `0.905`.
- `center>=0.50 AND unmatched<=120px`: count `836`, nearest distance q50 `64.31px`, length ratio q50 `0.881`, angle error q50 `1.53deg`, center distance q50 `59.98px`, y-overlap q50 `0.858`.
- `center>=0.50 WITHOUT unmatched<=120px`: count `527`, nearest distance q50 `186.96px`, length ratio q50 `2.091`, center distance q50 `182.80px`, y-overlap q50 `0.332`.
- 판단: the strongest nearby-track bucket is not primarily an angle or length-ratio failure. It is mostly a center/position offset around the match threshold. The centerline-only bucket is a different mechanism and likely needs new instance generation, not repair of the current nearest track.

Latest lane lateral-duplicate budget audit:

- branch/worktree: `exp/lane-family-f1/lane-lateral-duplicate-budget-audit`.
- code commit: `50038b9`.
- artifact: `runs/pv26_exhaustive_od_lane_train/lane_lateral_duplicate_budget_20260513/analysis_exports/broader_val512_epoch2/summary.json`.
- changed axis: no decoder change; compute the FP budget a lateral-duplicate style recovery would have if it recovered GT-labeled nearby-track FN buckets.
- baseline lane TP/FP/FN/F1: `4518 / 2206 / 4959 / 0.5577`.
- `unmatched<=80 and center>=0.50`: `563` recoverable FN; no-added-FP upper-bound F1 `0.6062`, but only `172` added FP can be tolerated.
- `unmatched<=120 and center>=0.50`: `836` recoverable FN; no-added-FP upper-bound F1 `0.6285`, with `809` added FP tolerance.
- `unmatched<=120 any center`: `1698` recoverable FN; no-added-FP upper-bound F1 `0.6946`, with `2821` added FP tolerance. If all current FP were duplicated and this whole bucket were recovered, the oracle-budget F1 is still `0.6184`.
- 판단: this is not production success, but it says a one-axis lateral-duplicate smoke is not mathematically dead if it targets a large nearby-track bucket. The next implementation must prove TP recovery and added-FP cost together; do not turn this into an offset/radius sweep.

Latest lane centerline-duplicate smoke:

- branch/worktree: `exp/lane-family-f1/lane-centerline-duplicate-smoke`.
- code commit: `61be845`.
- artifact smoke val4: `runs/pv26_exhaustive_od_lane_train/lane_centerline_duplicate_smoke_20260513/analysis_exports/smoke_val4_epoch2_t030/summary.json`.
- changed axis: keep the original row-scan-tangent track and emit a centerline-translated duplicate only when the translation moves the track.
- smoke val4 result: lane F1 `0.5594`, TP/FP/FN `40 / 17 / 46`; stop-line/crosswalk `0.0000 / 0.5455`.
- comparison: row-scan-tangent smoke reference was `0.5899`, TP/FP/FN `41 / 12 / 45`; replacement centerline translation was `0.5674`, TP/FP/FN `40 / 15 / 46`.
- 판단: preserving the original track avoids replacing it, but the duplicate adds FP without recovering TP. Do not broaden this branch or repeat as a duplicate offset/radius sweep.

Latest lane soft-ridge recovery readout smoke:

- branch/worktree: `exp/lane-family-f1/lane-fn-nearby-fp-recovery-audit`.
- code commit: `ee85fd8`.
- artifact: `runs/pv26_exhaustive_od_lane_train/lane_soft_ridge_recovery_audit_20260513/analysis_exports/smoke_val4_epoch2_t030/summary.json`.
- changed axis: add opt-in `row_scan_tangent_soft_ridge`, which selects per-row centerline probability ridge peaks before tangent linking, then replay it with the current flip-centerline broader composite settings at `lane_obj_threshold=0.30`.
- smoke baseline row-scan-tangent lane F1: `0.5899`, TP/FP/FN `41 / 12 / 45`.
- soft-ridge smoke lane F1: `0.5429`, TP/FP/FN `38 / 16 / 48`.
- 판단: soft-ridge peak picking loses TP and adds FP even on val4 smoke. Do not broaden this readout to val512 or repeat it as a lane threshold sweep without a new non-GT signal that explains how ridge candidates avoid this regression.

Latest lane centerline-snap recovery readout smoke:

- branch/worktree: `exp/lane-family-f1/lane-unmatched-track-pair-audit`.
- code commit: `43735cf`.
- artifact: `runs/pv26_exhaustive_od_lane_train/lane_centerline_snap_recovery_audit_20260513/analysis_exports/smoke_val4_epoch2_t030/summary.json`.
- changed axis: add opt-in `row_scan_tangent_centerline_snap`, which preserves row-scan-tangent instance topology and only snaps existing track x coordinates to local same-row centerline peaks.
- prior row-scan-tangent smoke lane F1: `0.5899`, TP/FP/FN `41 / 12 / 45`.
- centerline-snap smoke lane F1: `0.5674`, TP/FP/FN `40 / 15 / 46`.
- 판단: centerline snapping is less damaging than global soft-ridge peak generation but still loses TP and adds FP relative to the same smoke reference. Do not broaden this readout to val512 or repeat it as a snap-radius sweep without a new FP-control signal.

Latest lane track-level translation readout smoke:

- branch/worktree: `exp/lane-family-f1/lane-track-translation-readout`.
- code commit: `01b3ac1`.
- artifact: `runs/pv26_exhaustive_od_lane_train/lane_track_translation_readout_20260513/analysis_exports/smoke_val4_epoch2_t030/summary.json`.
- changed axis: add opt-in `row_scan_tangent_centerline_translate`, which preserves row-scan-tangent instance topology and y coordinates, then chooses one integer x offset for the whole decoded track by mean centerline score.
- prior row-scan-tangent smoke lane F1: `0.5899`, TP/FP/FN `41 / 12 / 45`.
- track-translation smoke lane F1: `0.5674`, TP/FP/FN `40 / 15 / 46`.
- 판단: track-level uniform translation does not rescue the pair-geometry center-offset bucket. It matches the centerline-snap regression pattern, so do not broaden this readout to val512 or repeat it as a translation-radius/offset sweep without a materially new non-GT FP-control signal.

Latest lane raw-vectorizer drop audit:

- branch/worktree: `exp/lane-family-f1/lane-raw-vectorizer-drop-audit`.
- code commit: `0545315`.
- artifact: `runs/pv26_exhaustive_od_lane_train/lane_raw_vectorizer_drop_audit_20260513/analysis_exports/broader_val512_epoch2/summary.json`.
- changed axis: no production decoder change; for every FN lane, compare final predictions to raw row-scan-tangent vectorizer candidates before bbox-area/aspect geometry filters.
- broader val512 audit baseline in this replay: lane/stop/cross F1 `0.5534 / 0.4235 / 0.6187`, lane TP/FP/FN `4564 / 2453 / 4913`.
- all FN: raw vectorizer already has a `<=40px` candidate for `854 / 4913`; `497` of those fail bbox-area and `37` fail aspect.
- `center>=0.50 without unmatched<=120px`: count `512`, raw vectorizer `<=40px` count `207`, area-filter drops `119`, aspect drops `14`, pass-filter/assignment cases `82`.
- 판단: center-only FN is not pure candidate-generation absence. This justified one guarded area-filter rescue probe, but not a blind bbox-area/aspect sweep because many misses are raw `80/120px` or pass-filter assignment cases.

Latest lane guarded area-rescue readout:

- branch/worktree: `exp/lane-family-f1/lane-guarded-area-rescue-readout`.
- code commit: `a1efacd`.
- artifact: `runs/pv26_exhaustive_od_lane_train/lane_guarded_area_rescue_readout_20260513/analysis_exports/val128_epoch2_rescue1_area1024/summary.json`.
- changed axis: add an opt-in decoder override that can rescue at most one lane candidate per sample when it fails only bbox-area filtering, has bbox area at least `1024px`, and still passes aspect.
- raw-audit replay reference on val128: lane F1 `0.5797`, TP/FP/FN `1204 / 560 / 1186`.
- guarded area-rescue val128 result: lane F1 `0.5716`, TP/FP/FN `1247 / 726 / 1143`; stop-line/crosswalk retained at `0.4364 / 0.5988`.
- 판단: area rescue recovers `43` lane TP but adds `166` FP, so it lowers F1 and should not be broadened to val512. Do not repeat this as a max-per-sample/min-area/bbox-filter sweep without a new FP-control signal.

Latest lane area-rescue center-score gate:

- branch/worktree: `exp/lane-family-f1/lane-area-rescue-center-score-gate`.
- code commit: `4482443`.
- artifact exact val128: `runs/pv26_exhaustive_od_lane_train/lane_area_rescue_center_score_gate_20260513/analysis_exports/val128_epoch2_center075/summary.json`.
- artifact broader val512: `runs/pv26_exhaustive_od_lane_train/lane_area_rescue_center_score_gate_20260513/analysis_exports/val512_epoch2_center075/summary.json`.
- changed axis: keep area rescue opt-in, then require rescued lane candidates to have vectorizer track-level `lane_centerline_track_mean >= 0.75`.
- exact val128 result: lane F1 `0.5846`, TP/FP/FN `1239 / 610 / 1151`; stop-line/crosswalk `0.4483 / 0.5988`.
- broader val512 result: lane F1 `0.5548`, TP/FP/FN `4644 / 2621 / 4833`; stop-line/crosswalk `0.4083 / 0.6187`.
- 판단: centerline mean gating fixes part of the val128 FP problem versus ungated area rescue, but broader val512 still stays below the current broader lane best `0.5577`. This is weak partial/negative evidence, not a default or a threshold-sweep path.

Latest lane area-rescue center-q10 gate:

- branch/worktree: `exp/lane-family-f1/lane-area-rescue-center-q10-gate`.
- code commit: `2d16971`.
- artifact smoke val4: `runs/pv26_exhaustive_od_lane_train/lane_area_rescue_center_q10_gate_20260513/analysis_exports/smoke_val4_epoch2_center075_q10060/summary.json`.
- changed axis: keep the previous area rescue and `lane_centerline_track_mean >= 0.75`, then also require `lane_centerline_track_q10 >= 0.60` as a stricter track-wide FP-control signal.
- smoke val4 result: lane F1 `0.5816`, TP/FP/FN `41 / 14 / 45`; stop-line/crosswalk `0.0000 / 0.5455`.
- comparison to center-mean smoke: lane F1 regressed from `0.5972` to `0.5816`, moving TP/FP/FN from `43 / 15 / 43` to `41 / 14 / 45`.
- 판단: q10 gating removes only one FP while losing two TP on the first smoke gate. Do not broaden this branch to val128/val512 or repeat area rescue as a q10/quantile threshold sweep without a materially new recall-preserving signal.

Latest lane row-scan hysteresis readout:

- branch/worktree: `exp/lane-family-f1/lane-row-scan-hysteresis-readout`.
- code commit: `ac4d498`.
- artifact smoke val4: `runs/pv26_exhaustive_od_lane_train/lane_row_scan_hysteresis_readout_20260513/analysis_exports/smoke_val4_epoch2_t030/summary.json`.
- changed axis: generate row-scan-tangent candidates from a lower centerline threshold, but keep only low-threshold connected components that contain a high-confidence seed.
- smoke val4 result: lane F1 `0.5652`, TP/FP/FN `39 / 13 / 47`; stop-line/crosswalk `0.0000 / 0.5455`.
- 판단: hysteresis is less damaging than soft-ridge peak generation, but it still loses two TP versus the same row-scan-tangent smoke reference and stays below the centerline-snap/translation smoke. Do not broaden or repeat as a low/high threshold sweep without a new instance-level recall-preserving signal.

Latest stop-line recovery-budget audit:

- branch/worktree: `exp/lane-family-f1/stopline-recovery-budget-audit`.
- code commit: `ff70095`.
- artifact: `runs/pv26_exhaustive_od_lane_train/stopline_recovery_budget_audit_20260513/analysis_exports/projection_competition_val512/summary.json`.
- reference: projection-competition readout `proj_comp_length_s090_top2_second_frag5`, stop-line F1 `0.5164`, TP/FP/FN `126 / 91 / 145`.
- candidate-manifest buckets: positive top-oracle `113`, positive misrank `29`, positive no-oracle `62`, GT-negative candidate-bearing `165`.
- target math: at current FP, stop-line needs `+30` recovered TP to reach F1 `0.60`; FP-only recovery would need removing `68 / 91` FP without losing TP.
- selector-only ceiling: recovering all `29` positive-misrank samples gives stop-line F1 `0.5996`, still just below target; misrank plus one no-oracle recovery gives `0.6023`.
- no-oracle upper bound: recovering all `62` positive-no-oracle samples gives F1 `0.6836` and could tolerate up to `76` added FP while staying at `>=0.60`.
- 판단: another selector/logistic/photometric/projection-threshold sweep is not the next useful stop-line axis. The next stop-line branch must add a no-GT candidate-generation or midpoint-recovery signal that reaches currently no-oracle positives while controlling added FP.

Latest stop-line no-oracle fragment-extension budget:

- branch/worktree: `exp/lane-family-f1/stopline-mask-midpoint-recovery-readout`.
- code commit: `fd6fab0`.
- artifact: `runs/pv26_exhaustive_od_lane_train/stopline_mask_midpoint_recovery_readout_20260513/analysis_exports/no_oracle_extension_budget_val512_epoch2/summary.json`.
- changed axis: replace only positive-no-oracle samples in the projection-competition replay with simple min-length extensions of current candidate fragments; compare top, longest-high-score, and nearest-GT-oracle candidate selectors.
- baseline projection competition remains best: stop-line F1 `0.5164`, TP/FP/FN `126 / 91 / 145`.
- best extension variants reach only `0.4713`, TP/FP/FN `119 / 115 / 152`.
- 판단: positive no-oracle is not solved by simply extending short fragments around their current midpoint. Do not implement or sweep a min-length fragment-extension postprocess without a new no-GT centering/candidate-generation signal.

Latest stop-line no-oracle proposal-recall bucket audit:

- branch/worktree: `exp/lane-family-f1/stopline-mask-midpoint-recovery-readout`.
- code commit: `07218b9`.
- artifact: `runs/pv26_exhaustive_od_lane_train/stopline_mask_midpoint_recovery_readout_20260513/analysis_exports/no_oracle_proposal_recall_bucket_val512_epoch2/summary.json`.
- changed axis: no new model run; join existing val512 per-GT proposal recall with candidate-manifest failure buckets.
- positive-no-oracle GT count in the joined max-source rows: `69`.
- max-source local signal: `max_r8 >= 0.6` is `44 / 69`, `max_r4 >= 0.6` is `36 / 69`.
- top-k proximity: `top20_hit_r8` is `61 / 69`, while `top1_hit_r8` is only `12 / 69` and `top3_hit_r8` is `27 / 69`.
- 판단: many no-oracle positives still have dense `max(center, selector)` signal near the GT center. The failure is not primarily dense-map absence; it is current top candidate selection / midpoint centering / geometry decode. The next stop-line branch should generate candidates from local max-source neighborhoods or learn a richer emit/select contract, not stretch selected fragments.

Latest stop-line no-oracle anchor-shift audit:

- branch/worktree: `exp/lane-family-f1/stopline-mask-midpoint-recovery-readout`.
- code commit: `0670701`.
- artifact: `runs/pv26_exhaustive_od_lane_train/stopline_mask_midpoint_recovery_readout_20260513/analysis_exports/no_oracle_anchor_shift_val512_epoch2/summary.json`.
- changed axis: replace positive-no-oracle samples by shifting current candidate segments from decoded center toward the proposal-cell anchor; no model run and no production change.
- baseline projection competition remains best: stop-line F1 `0.5164`, TP/FP/FN `126 / 91 / 145`.
- all anchor-shift variants collapse to F1 `0.4475`, TP/FP/FN `113 / 121 / 158`.
- 판단: the no-oracle gap is not solved by a simple proposal-anchor vs decoded-anchor correction. Next stop-line work needs a richer local-neighborhood candidate geometry signal, not anchor shifting.

Latest stop-line no-oracle local proposal geometry audit:

- branch/worktree: `exp/lane-family-f1/stopline-mask-midpoint-recovery-readout`.
- code commit: `eee5085`.
- artifact: `runs/pv26_exhaustive_od_lane_train/stopline_mask_midpoint_recovery_readout_20260513/analysis_exports/no_oracle_local_proposal_geometry_val512_epoch2/summary.json`.
- changed axis: join positive-no-oracle GT rows to the nearest exported max-source proposal candidate, testing whether local proposal cells near GT already decode usable geometry.
- positive-no-oracle GT rows: `69`; `top20_hit_r8` is `61`, and nearest exported candidate is within r8 for `51`.
- local candidate quality: `nearest_gt_distance <= 40` is `0 / 51`; q50 local candidate nearest distance `95.36px`, midpoint distance `68.59px`, length ratio `0.5437`, angle error `2.59deg`.
- 판단: local score/proposal cells are often near GT, but the exported candidate geometry still does not become matched stop-line segments. The next stop-line edit must create a new local geometry readout from the score island or change the emit/select/readout contract; just selecting local top20 cells is not enough.

Latest stop-line no-oracle local recenter budget:

- branch/worktree: `exp/lane-family-f1/stopline-mask-midpoint-recovery-readout`.
- code commit: `063c2fa`, hardened by `0281346` to prefer per-sample affine for proposal-cell-to-raw recentering.
- artifact: `runs/pv26_exhaustive_od_lane_train/stopline_mask_midpoint_recovery_readout_20260513/analysis_exports/no_oracle_local_recenter_budget_val512_epoch2/summary.json`.
- changed axis: replay GT-joined local proposal candidates with progressively stronger oracles: raw local row, proposal-anchor recenter, min-length recenter, GT-length oracle, and GT-midpoint+GT-length oracle.
- method check: selected local samples with per-sample affine `47 / 47`; sample-affine fallback count `0`.
- baseline projection competition: stop-line F1 `0.5164`, TP/FP/FN `126 / 91 / 145`.
- local raw/anchor/minlen/GT-length all stay below baseline: best non-midpoint oracle is `0.4939`, TP/FP/FN `122 / 101 / 149`.
- GT-midpoint+GT-length oracle opens the budget: F1 `0.6599`, TP/FP/FN `163 / 60 / 108`.
- 판단: local angle evidence is usable only if the center/midpoint is correct. Length or anchor correction is not enough; the next production branch must infer the missing stop-line midpoint/center without GT.

Latest stop-line score-island midpoint readout:

- branch/worktree: `exp/lane-family-f1/stopline-mask-midpoint-recovery-readout`.
- code commit: `b5f29f5`.
- artifact: `runs/pv26_exhaustive_od_lane_train/stopline_mask_midpoint_recovery_readout_20260513/analysis_exports/score_island_midpoint_readout_val128_epoch2/summary.json`.
- changed axis: add opt-in score-island weighted center mode to the existing predicted proposal + angle-mask extent readout; no GT centers, no training, stop-line predictions only.
- exact val128 baseline stop-line F1: `0.4483`, TP/FP/FN `26 / 30 / 34`.
- existing selector-center reference remains best in this replay: `pred_selector_top1_s060_mask050_band4` stop-line F1 `0.5085`, TP/FP/FN `30 / 28 / 30`.
- best island variant reaches only `0.4354`, TP/FP/FN `32 / 55 / 28`; wider island variants regress to `0.4218`.
- 판단: score-island centroiding changes center placement slightly but increases FP and does not recover the midpoint budget. Do not broaden to val512 or turn this into a radius/relative-threshold sweep without a new non-GT FP-control signal.

Latest stop-line score-island linefit readout:

- branch/worktree: `exp/lane-family-f1/stopline-score-island-linefit-readout`.
- code commit: `e89db20`.
- artifact: `runs/pv26_exhaustive_od_lane_train/stopline_score_island_linefit_readout_20260513/analysis_exports/val128_epoch2/summary.json`.
- changed axis: fit both center and axis from the local max(center, selector) score island, then reuse the existing mask-extent line generation; no GT center/angle, no training.
- exact val128 baseline stop-line F1: `0.4483`, TP/FP/FN `26 / 30 / 34`.
- existing selector-center reference remains best in this replay: `pred_selector_top1_s060_mask050_band4` stop-line F1 `0.5085`, TP/FP/FN `30 / 28 / 30`.
- score-island linefit reaches only `0.4110`, TP/FP/FN `30 / 56 / 30`.
- 판단: score-island linefit recovers no extra TP over selector-center and adds too many FP. Do not broaden to val512 or repeat as an island radius/relative-threshold/linefit sweep without a materially new FP-control signal.

Current best exact lane-retention probe:

- artifact: `runs/pv26_exhaustive_od_lane_train/lane60_core_centerline_refine_row_scan_tangent_segment_mil_lane_head_only_from_lane60_core_centerline_refine_cross_retain_from_exhaustive_od_lane_default_20260505_032217_default_20260510_003412_default_20260511_230022/phase_4/history/epochs.jsonl`
- experiment: `core_centerline_refine_row_scan_tangent_segment_mil_lane_head_only`
- objective: `0.6193428422373919`
- lane / stop-line / crosswalk F1: `0.5660 / 0.4483 / 0.5926`
- lane TP/FP/FN: `1162 / 554 / 1228`
- stop TP/FP/FN: `26 / 30 / 34`
- cross TP/FP/FN: `48 / 33 / 33`
- 판단: exact val128에서는 tangent-link exact reference `0.6187165763`을 아주 조금 넘는 새 best지만, lane gain은 `+0.0027` 수준이고 stop-line은 여전히 `0.4483`이라 broader-val512 확장 조건으로 보지 않는다.

Latest lane instance-validator probe:

- artifact: `runs/pv26_exhaustive_od_lane_train/lane60_core_centerline_refine_row_scan_tangent_instance_validator_from_lane60_core_centerline_refine_cross_retain_from_exhaustive_od_lane_default_20260505_032217_default_20260510_003412_default_20260512_193428/phase_4/history/epochs.jsonl`
- validator-logit audit: `runs/pv26_exhaustive_od_lane_train/lane60_core_centerline_refine_row_scan_tangent_instance_validator_from_lane60_core_centerline_refine_cross_retain_from_exhaustive_od_lane_default_20260505_032217_default_20260510_003412_default_20260512_193428/analysis_exports/validator_logits_on_tangent_candidates_val128_epoch2/summary.json`
- experiment: `core_centerline_refine_row_scan_tangent_instance_validator`
- changed axis: add an opt-in `lane_seg_instance_validator_logits` head, supervise predicted centerline components as valid/invalid candidates, then decode with `row_scan_tangent_instance_validator`.
- best exact val128 epoch2 objective: `0.6142642145`
- lane / stop-line / crosswalk F1: `0.5523 / 0.4522 / 0.5854`
- lane TP/FP/FN: `1099 / 491 / 1291`
- stop TP/FP/FN: `26 / 29 / 34`
- validator-logit audit on tangent-link candidates: heldout baseline lane F1 `0.5576`; best validator-only row threshold `0.5577`; logistic all-feature threshold `0.5649`; oracle TP-only selector `0.6344`.
- 판단: stop-line is slightly above tangent-link exact `0.4483`, but lane regresses below tangent-link `0.5633`, row-distribution `0.5659`, and segment-MIL lane-head-only `0.5660`; objective is also below all current exact references. The first validator contract is wired and runnable, and the oracle audit says lane still has FP-selector headroom, but the learned validator logits do not provide a recall-preserving production gate. This `weight=0.35`, threshold `0.45` axis is closed.

Latest lane upper-trunk capacity probe:

- artifact: `runs/pv26_exhaustive_od_lane_train/lane60_core_centerline_refine_row_scan_tangent_upper_trunk_from_lane60_core_centerline_refine_cross_retain_from_exhaustive_od_lane_default_20260505_032217_default_20260510_003412_default_20260512_113104/phase_4/history/epochs.jsonl`
- experiment: `core_centerline_refine_row_scan_tangent_upper_trunk`
- changed axis: keep the core target, `row_scan_tangent` vectorizer, stop-line/crosswalk settings, sampler, and train/val volume fixed, but reopen `lane_family_plus_upper_trunk` with trunk LR `2e-6` and head LR `1e-4`.
- best exact val128 epoch2 objective: `0.6169717875`
- lane / stop-line / crosswalk F1: `0.5597 / 0.4561 / 0.5854`
- lane TP/FP/FN: `1115 / 479 / 1275`
- stop TP/FP/FN: `26 / 28 / 34`
- cross TP/FP/FN: `48 / 35 / 33`
- 판단: stop-line is slightly higher than tangent-link exact `0.4483`, but lane falls below tangent-link `0.5633`, row-distribution `0.5659`, and segment-MIL lane-head-only `0.5660`; objective is also below the current exact references. Upper-trunk capacity/freeze-scope-only is not a broader-val512 expansion path.

Latest lane flip-consistency training probe:

- branch/worktree: `exp/lane-family-f1/lane-flip-consistency-row-scan-tangent`.
- code commits: `2dcd776` adds the opt-in train-time flip-consistency loss; `e19dbd1` adds the row-scan-tangent probe preset and single-axis guard test.
- artifact: `runs/pv26_exhaustive_od_lane_train/lane60_core_centerline_refine_row_scan_tangent_flip_consistency_from_lane60_core_centerline_refine_cross_retain_from_exhaustive_od_lane_default_20260505_032217_default_20260510_003412_default_20260513_002056/phase_4/history/epochs.jsonl`
- experiment: `core_centerline_refine_row_scan_tangent_flip_consistency`
- changed axis: keep `row_scan_tangent`, loss weights, freeze policy, stop-line/crosswalk settings, and sampler fixed, then add only `lane_flip_consistency_weight=0.25`.
- exact val128 epoch2 objective: `0.5991166581`
- lane / stop-line / crosswalk F1: `0.5542 / 0.3966 / 0.5548`
- task-best F1: lane `0.5542` at epoch2, stop-line `0.3966` at epoch2, crosswalk `0.6711` at epoch1.
- skipped steps: `0`.
- 판단: runtime is stable, but the exact metric regresses below tangent-link reference `0.6187`, `0.5633 / 0.4483 / 0.5854`; the epoch1 crosswalk spike is not enough because the best objective checkpoint misses all three task targets. Flip-consistency-regularizer-only is not a broader-val512 expansion path.

Latest lane endpoint-extension readout probe:

- branch/worktree: `exp/lane-family-f1/lane-endpoint-extension-readout`.
- code commit: `8f5c3a0` adds a read-only fixed-distance endpoint-extension probe on top of the row-scan-tangent readout.
- artifact: `runs/pv26_exhaustive_od_lane_train/lane_endpoint_extend_val128_epoch2/summary.json`.
- changed axis: keep the checkpoint, row-scan-tangent decoder, stop-line/crosswalk predictions, sampler, and validation slice fixed, then extend decoded lane polyline endpoints by fixed raw-pixel distances before matching.
- exact val128 baseline objective and lane/stop/cross F1: `0.6187165763`, `0.5633 / 0.4483 / 0.5854`.
- best extension by objective: `top32`, objective `0.5998086929`, lane/stop/cross F1 `0.5503 / 0.4483 / 0.5854`.
- lane TP/FP/FN baseline: `1121 / 469 / 1269`; `top32`: `1095 / 495 / 1295`.
- 판단: fixed top/bottom endpoint extension hurts lane TP and FP at exact val128, so it is not a broader-val512 expansion path. Do not repeat as a distance sweep unless a new conditioning signal decides when extension is safe.

Latest lane positive-core flip-consistency training probe:

- branch/worktree: `exp/lane-family-f1/lane-positive-core-flip-consistency`.
- code commit: `572f8a4` adds `lane_flip_consistency_mask_mode` and a row-scan-tangent positive-core consistency preset.
- artifact: `runs/pv26_exhaustive_od_lane_train/lane60_core_centerline_refine_row_scan_tangent_positive_flip_consistency_from_lane60_core_centerline_refine_cross_retain_from_exhaustive_od_lane_default_20260505_032217_default_20260510_003412_default_20260513_012145/phase_4/history/epochs.jsonl`.
- changed axis: keep row-scan-tangent, checkpoint, task weights, sampler, and consistency weight fixed, then restrict flip-consistency MSE to supervised `lane_seg_centerline_core` positive pixels.
- exact val128 epoch2 objective: `0.5981680601`
- lane / stop-line / crosswalk F1: `0.5513 / 0.3966 / 0.5548`
- task-best F1: lane `0.5513` epoch2, stop-line `0.3966` epoch2, crosswalk `0.6711` epoch1.
- skipped steps: `0`.
- 판단: runtime is stable, but positive-core-only consistency does not rescue the global flip-consistency regression. It is below tangent-link `0.6187`, `0.5633 / 0.4483 / 0.5854` and slightly below global flip-consistency objective `0.5991166581`, so do not broaden to val512.

Latest lane soft-skeleton topology-loss probe:

- branch/worktree: `exp/lane-family-f1/lane-centerline-soft-skeleton`.
- code commit: `e7a6ebf` adds opt-in soft-skeleton/clDice-style centerline topology loss and a row-scan-tangent probe preset.
- artifact: `runs/pv26_exhaustive_od_lane_train/lane60_core_centerline_refine_row_scan_tangent_soft_skeleton_from_lane60_core_centerline_refine_cross_retain_from_exhaustive_od_lane_default_20260505_032217_default_20260510_003412_default_20260513_020600/phase_4/history/epochs.jsonl`.
- changed axis: keep row-scan-tangent, checkpoint, task weights, sampler, stop-line, and crosswalk fixed, then add only `lane_segfirst_soft_skeleton_weight=0.25` with `6` skeleton iterations.
- exact val128 epoch2 objective: `0.6152102115`
- lane / stop-line / crosswalk F1: `0.5609 / 0.4348 / 0.5854`
- lane TP/FP/FN: `1114 / 468 / 1276`
- task-best F1: lane `0.5609` epoch2, stop-line `0.4348` epoch2, crosswalk `0.6790` epoch1.
- skipped steps: `0`.
- 판단: runtime is stable, but topology-loss-only does not reach the tangent-link exact reference `0.6187165763`, `0.5633 / 0.4483 / 0.5854`; it also underperforms the earlier soft-shell auxiliary lane result. Do not broaden to val512 or repeat this as a weight/iteration sweep.

Latest lane instance-embedding row-link probe:

- branch/worktree: `exp/lane-family-f1/lane-embedding-row-scan-link`.
- code commit: `7b84900` adds an opt-in dense lane instance embedding head, `lane_seg_instance_id` target, embedding pull/push loss, and `row_scan_tangent_embedding` decoder.
- artifact: `runs/pv26_exhaustive_od_lane_train/lane60_core_centerline_refine_row_scan_tangent_embedding_link_from_lane60_core_centerline_refine_cross_retain_from_exhaustive_od_lane_default_20260505_032217_default_20260510_003412_default_20260513_024937/phase_4/history/epochs.jsonl`.
- changed axis: keep row-scan-tangent, checkpoint, task weights, sampler, stop-line, and crosswalk fixed, then add only `lane_segfirst_loss_weights.instance_embedding=0.35` and decode with `row_scan_tangent_embedding`.
- exact val128 epoch2 objective: `0.6150797781`.
- lane / stop-line / crosswalk F1: `0.5594 / 0.4348 / 0.5854`.
- lane TP/FP/FN: `1114 / 479 / 1276`.
- task-best F1: lane `0.5594` epoch2, stop-line `0.4348` epoch2, crosswalk `0.6790` epoch1.
- skipped steps: `0`.
- 판단: runtime and checkpoint handoff are stable, but the embedding-link axis is below tangent-link exact `0.6187165763`, `0.5633 / 0.4483 / 0.5854`; it also fails to beat soft-skeleton or segment-MIL lane-head-only exact references. Do not broaden to val512 or repeat as an embedding distance/weight sweep.

Latest lane instance safety-gate replay:

- artifact: `runs/pv26_exhaustive_od_lane_train/lane60_lane_instance_safety_gate_replay_20260513/analysis_exports/broader_val512_flip_centerline_avg_safety_gate_epoch2/summary.json`.
- source features: `lane60_lane_flip_instance_evidence_20260512` broader flip-centerline instance evidence CSV.
- changed axis: keep the current flip-centerline row-scan candidates fixed, then preserve each sample's top-K candidates by `pred_index` while applying the existing logistic row gate to the rest.
- best heldout variant: `keep_topk=0`, threshold `0.1535836312`, lane F1 `0.5650`, TP/FP/FN `2142 / 818 / 2480`.
- full-split lane F1 for the same selected variant: `0.5705`, TP/FP/FN `4406 / 1564 / 5071`.
- `keep_topk=1..8` all reduce heldout/full lane F1 versus logistic-only because FP comes back faster than TP.
- 판단: top-K safety fallback does not solve the logistic gate's recall tradeoff and remains below the earlier full split-count logistic diagnostic `0.5738`, current lane oracle `0.6457`, and the `0.60` target. Do not convert this into a production safety gate.

Latest lane instance score-gate replay:

- branch/worktree: `exp/lane-family-f1/lane-instance-validator-score-rank`.
- code commit: `7975e77` adds an opt-in `row_scan_tangent_instance_score_gate` decoder that generates row-scan tangent candidates from the unmasked centerline, then filters completed lane candidates by their mean `instance_validator` score.
- replay artifact: `runs/pv26_exhaustive_od_lane_train/lane60_lane_instance_score_gate_replay_20260513/analysis_exports/instance_score_gate_replay_val128_epoch2/summary.json`.
- source checkpoint: previous `core_centerline_refine_row_scan_tangent_instance_validator` best checkpoint, evaluated with `core_centerline_refine_row_scan_tangent_instance_score_gate`.
- exact val128 epoch2 objective: `0.6143914132`.
- lane / stop-line / crosswalk F1: `0.5529 / 0.4522 / 0.5854`.
- 판단: candidate-level score gating avoids the hard pixel-mask recall cut in code, but the metric is effectively flat against hard validator `0.6142642145`, `0.5523 / 0.4522 / 0.5854`, and still below tangent-link exact `0.6187165763`, `0.5633 / 0.4483 / 0.5854`. Do not launch full training or sweep validator score thresholds from this checkpoint.

Latest task-head merge with segment-MIL lane + rank-stop head:

- artifact exact: `runs/pv26_exhaustive_od_lane_train/lane60_task_head_merge_segment_mil_rank_stop_source_cross_20260512/analysis_exports/exact_val128_segment_mil_epoch2/summary.json`
- artifact broader: `runs/pv26_exhaustive_od_lane_train/lane60_task_head_merge_segment_mil_rank_stop_source_cross_20260512/analysis_exports/broader_val512_segment_mil_stop_pca_hull_epoch2/summary.json`
- composition: start from the segment-MIL lane-head-only checkpoint, replace lane weights from its `best_lane.pt`, replace stop-line weights from the proposal-rank `best_stop_line.pt`, and keep crosswalk from the source `best.pt`.
- exact val128 objective: `0.6248193729`
- exact lane / stop-line / crosswalk F1: `0.5660 / 0.4918 / 0.5926`
- exact TP/FP/FN lane: `1162 / 554 / 1228`
- exact TP/FP/FN stop-line: `30 / 32 / 30`
- exact TP/FP/FN crosswalk: `48 / 33 / 33`
- broader-val512 objective: `0.6129801219`
- broader lane / stop-line / crosswalk F1: `0.5480 / 0.3976 / 0.6185`
- broader TP/FP/FN lane: `4457 / 2333 / 5020`
- broader TP/FP/FN stop-line: `100 / 132 / 171`
- broader TP/FP/FN crosswalk: `231 / 121 / 164`
- 판단: exact subset에서는 stop-line head transplant가 좋아 보였지만, broader-val512에서는 stop-line FP가 늘어 current broader best objective `0.6216194906`와 stop-line `0.4235`보다 낮다. Task-head composition is exact partial-positive but broader negative; it is not an all-task success path.

Latest lane-head transplant onto original stop/cross base:

- artifact exact: `runs/pv26_exhaustive_od_lane_train/lane60_lane_head_transplant_original_stop_pca_20260512/analysis_exports/exact_val128_row_scan_tangent_epoch2/summary.json`
- artifact broader: `runs/pv26_exhaustive_od_lane_train/lane60_lane_head_transplant_original_stop_pca_20260512/analysis_exports/broader_val512_stop_pca_hull_epoch2/summary.json`
- composition: base original `best.pt`, lane head from segment-MIL lane-head-only `best_lane.pt`, stop-line/crosswalk heads from original `best.pt`.
- exact val128 objective: `0.6193289563`
- exact lane / stop-line / crosswalk F1: `0.5660 / 0.4483 / 0.5854`
- broader-val512 objective: `0.6176617972`
- broader lane / stop-line / crosswalk F1: `0.5480 / 0.4235 / 0.6187`
- broader TP/FP/FN stop-line: `101 / 105 / 170`
- 판단: this was the cleaner broader composite before flip-centerline TTA and remains the base checkpoint composition for the current runtime best. By itself, lane and stop-line still fail the `>=0.60` goal.

Latest stop-line component-topology validator replay:

- artifact: `runs/pv26_exhaustive_od_lane_train/lane60_core_centerline_refine_cross_retain_from_exhaustive_od_lane_default_20260505_032217_default_20260510_003412/analysis_exports/stopline_component_topology_rich_val512_epoch2/summary.json`
- branch: `exp/lane-family-f1/stopline-component-topology-audit`
- changed axis: keep the checkpoint fixed, add predicted-mask component topology features to the stop-line candidate CSV, and replay the existing gap4/top50 rich-validator half-split task probe.
- candidate rows: `15629`; oracle-positive rows: `4683`; oracle-positive rate: `0.2996`.
- standard candidate reference: baseline stop-line F1 `0.4083`, score-threshold production `0.4371`, gap4/top50 oracle `0.6877`.
- held-out rich-logistic task replay: baseline `0.3877`, component-topology rich logistic `0.4126`, `selector_r4_max` threshold `0.4537`.
- 판단: component topology has some offline signal but does not beat the existing selector-only replay, PCA broader reference `0.4699`, or the 0.6 target. Treat it as read-only negative evidence, not a production decoder or model-side validator success.

Latest stop-line sample-tree gate audit:

- artifact: `runs/pv26_exhaustive_od_lane_train/lane60_core_centerline_refine_cross_retain_from_exhaustive_od_lane_default_20260505_032217_default_20260510_003412/analysis_exports/stopline_gap4_sample_tree_gate_audit_val512_epoch2/summary.json`
- branch: `exp/lane-family-f1/stopline-sample-tree-gate-audit`
- changed axis: extend the existing gap4/top50 sample-gate diagnostic with a shallow greedy decision-tree gate over candidate-bearing sample aggregate features.
- train surrogate selection F1: logistic `0.6559`, tree `0.5933`.
- held-out surrogate selection F1: emit-all `0.4320`, logistic `0.5732`, tree `0.5543`.
- held-out tree TP/FP/FN: `51 / 70 / 12`; held-out logistic TP/FP/FN: `47 / 54 / 16`.
- learned tree rule collapsed to `mask_r4_max_max > 0.9962455`.
- 판단: a nonlinear sample gate does not beat the prior logistic surrogate and still is not actual task F1. This closes "make the CSV sample gate slightly more nonlinear" as a useful next stop-line path.

Latest stop-line GT sample-gate oracle audit:

- artifact: `runs/pv26_exhaustive_od_lane_train/lane60_core_centerline_refine_cross_retain_from_exhaustive_od_lane_default_20260505_032217_default_20260510_003412/analysis_exports/stopline_oracle_sample_gate_val512_epoch2/summary.json`
- branch: `exp/lane-family-f1/stopline-oracle-sample-gate-audit`
- changed axis: keep the checkpoint fixed, then use GT stop-line sample presence only to suppress emissions on negative samples while comparing non-oracle score/top-k ranking against local feature ranking.
- baseline stop-line F1: `0.4083`, TP/FP/FN `98 / 111 / 173`.
- score-threshold production reference: `0.4371`, TP/FP/FN `106 / 108 / 165`.
- GT sample gate best: `gt_sample_gate_max_top10_score` stop-line F1 `0.4758`, TP/FP/FN `113 / 91 / 158`.
- GT sample gate with score `>=0.80`: `0.4743`, TP/FP/FN `106 / 70 / 165`.
- GT sample gate + local feature ranking: `gt_sample_gate_gap4_max_top50_selector_r4` stop-line F1 `0.4800`, TP/FP/FN `114 / 90 / 157`; `mask_r4_max` ranking is lower at `0.4589`, TP/FP/FN `109 / 95 / 162`.
- gap4/top50 oracle-positive upper bound: `0.6877`, TP/FP/FN `142 / 0 / 129`.
- 판단: perfect sample-level emission plus the best tested local feature rank only reaches `0.4800`, barely above the PCA broader reference `0.4699` and still far below `0.60`. Stop-line is not solved by sample emission gating or shallow local-feature reranking on the current candidate rows; candidate generation/readout geometry remains the blocker.

Latest stop-line hard-negative sampler probe:

- artifact: `runs/pv26_exhaustive_od_lane_train/lane60_core_centerline_refine_row_scan_tangent_stopline_hard_negative_sampler_from_lane60_core_centerline_refine_cross_retain_from_exhaustive_od_lane_default_20260505_032217_default_20260510_003412_default_20260512_204742/phase_4/history/epochs.jsonl`
- branch: `exp/lane-family-f1/stopline-candidate-manifest-audit`
- changed axis: keep row-scan/tangent, core lane target, loss weights, and freeze policy fixed, but add a `stopline_negative` task-positive bucket that samples stop-line-capable lane-source images with no stop-line GT.
- best exact val128 epoch2 objective: `0.6063`
- lane / stop-line / crosswalk F1: `0.5475 / 0.4561 / 0.5644`
- TP/FP/FN stop-line: `26 / 28 / 34`
- 판단: stop-line is only `+0.0078` over tangent-link exact `0.4483`, while lane, crosswalk, and objective are below current exact references. Hard-negative exposure alone is not the missing selector/readout contract and is not expanded to broader-val512.

Latest stop-line candidate manifest failure-mode audit:

- artifact: `runs/pv26_exhaustive_od_lane_train/lane60_lane_head_transplant_original_stop_pca_20260512/analysis_exports/stopline_current_candidate_manifest_failure_audit_val512_epoch2/summary.json`
- tool: `tools/analyze_pv26_stopline_candidate_manifest.py`
- gap4/max candidate-bearing samples: `369`
- buckets: GT-negative candidate-bearing `165`, positive top-oracle `113`, positive misrank `29`, positive no-oracle `62`.
- positive rates: top-oracle `0.5539`, has-oracle `0.6961`.
- positive no-oracle nearest-distance bins: `40_80=28`, `gte80=34`; there are no `<40` cases in that bucket.
- 판단: sampler-only negative exposure is not enough, and selector-only ranking can recover only the `29` misrank samples. A large part of the stop-line gap is still candidate generation / midpoint proposal recovery, not just production thresholding.

Latest enriched stop-line candidate manifest audit:

- artifact: `runs/pv26_exhaustive_od_lane_train/lane60_lane_head_transplant_original_stop_pca_20260512/analysis_exports/stopline_current_candidate_manifest_failure_audit_enriched_val512_epoch2/summary.json`
- changed axis: add GT/candidate length ratios and bucket-level numeric quantiles to the manifest analyzer.
- `positive_no_oracle` top score q50 is `0.9998`, nearest GT distance q50 is `89.98px`, and nearest candidate length ratio q50 is only `0.066`.
- 판단: the failing positive no-oracle samples are not low-score candidates waiting for a ranker. They are high-confidence, short fragments whose centers are far from the GT midpoint. Next stop-line work must change center proposal / extent recovery, not another score or sample-gate threshold.

Latest learned stop-line fragment-to-center extent probe:

- branch/worktree: `exp/lane-family-f1/stopline-fragment-extent-recovery`.
- code commit: `64c7d70` adds an opt-in `stop_line_fragment_center_offset` head, fragment-line targets, fragment auxiliary loss, and fragment-extent postprocess decode.
- artifact: `runs/pv26_exhaustive_od_lane_train/lane60_core_centerline_refine_stop_fragment_extent_from_lane60_core_centerline_refine_cross_retain_from_exhaustive_od_lane_default_20260505_032217_default_20260510_003412_default_20260513_033355/phase_4/history/epochs.jsonl`.
- changed axis: keep checkpoint, sampler, freeze policy, and task weights fixed, then train a dense fragment pixel -> full stop-line center/angle/extent recovery contract with `stopline_fragment_extent_aux_weight=0.75` and `stop_line_fragment_extent_enabled=true`.
- best exact val128 objective: `0.5499257237` at epoch1.
- epoch1 lane / stop-line / crosswalk F1: `0.5147 / 0.0267 / 0.6625`.
- epoch2 lane / stop-line / crosswalk F1: `0.5222 / 0.0519 / 0.5389`.
- epoch2 stop-line TP/FP/FN: `4 / 90 / 56`.
- fragment-disabled replay on the same checkpoints: epoch1 stop-line F1 `0.1233`, epoch2 stop-line F1 `0.2517`.
- skipped steps: `0`.
- 판단: runtime and shape-aware handoff are stable, but the learned fragment-to-center extent head collapses stop-line task F1 far below tangent-link exact `0.4483`, PCA val128 `0.5133`, and read-only fragment-extent replay `0.4918`. Disabling the new decode recovers some F1 but not the baseline, so both the decode path and the auxiliary-trained checkpoint are weak. Do not broaden to val512 or repeat as fragment top-k/min-score/aux-weight sweep.

Latest stop-line fragment-union readout replay:

- branch/worktree: `exp/lane-family-f1/stopline-fragment-union-readout`.
- code commit: `0b8aa6f` adds `tools/probe_pv26_stopline_fragment_union_readout.py` and regression coverage.
- follow-up code commit: `ed9fc12` promotes the same readout into an opt-in model-output postprocess/evaluator path on the experiment branch.
- artifact: `runs/pv26_exhaustive_od_lane_train/lane60_lane_head_transplant_original_stop_pca_20260512/analysis_exports/stopline_fragment_union_readout_val512_epoch2/summary.json`.
- productionized replay artifact: `runs/pv26_exhaustive_od_lane_train/lane60_lane_head_transplant_original_stop_pca_20260512/analysis_exports/stopline_fragment_union_postprocess_val512_epoch2/summary.json`.
- changed axis: keep the current-composite candidate CSV fixed, then cluster same-line high-score short stop-line fragments and emit the merged segment. GT is used only for evaluation, not prediction.
- support accounting: total stop-line support `271`, candidate-bearing GT `257`, missing-GT FN add-on `14`.
- best broader-val512 variant: `union_a12_o36_s080_c2_fallback_top`.
- lane / stop-line / crosswalk F1: `0.5480 / 0.4948 / 0.6187`.
- TP/FP/FN stop-line: `120 / 94 / 151`.
- non-fallback best: `union_a16_o48_s080_c2`, stop-line F1 `0.4936`, TP/FP/FN `116 / 83 / 155`.
- productionized replay result: same broader-val512 stop-line F1 `0.4948453608`, TP/FP/FN `120 / 94 / 151`, lane/crosswalk F1 `0.5480 / 0.6187`, phase objective `0.6373915315`.
- 판단: fragment union is the first current-composite stop-line replay in this lane to beat score-threshold production `0.4371`, PCA broader reference `0.4699`, and GT sample-gate + same-row feature rank `0.4800`, and the opt-in evaluator path reproduces the CSV result. It is still below `0.60`, remains an experiment-branch opt-in rather than a default decoder, and should not trigger another threshold-only sweep.

Latest stop-line fragment follow-up replays:

- seed-extension branch/worktree: `exp/lane-family-f1/stopline-fragment-seed-extend`.
- seed-extension code commit: `900c9b8`.
- seed-extension artifact: `runs/pv26_exhaustive_od_lane_train/lane60_lane_head_transplant_original_stop_pca_20260512/analysis_exports/stopline_fragment_seed_extend_readout_val512_epoch2/summary.json`.
- seed-extension changed axis: preserve high-score union seed clusters, then let lower-score same-line fragments extend those clusters without creating standalone predictions.
- seed-extension result: best `seed_extend_s080_e065_c2_fallback_top`, stop-line F1 `0.4742`, TP/FP/FN `115 / 99 / 156`.
- length-competition branch/worktree: `exp/lane-family-f1/stopline-fragment-length-competition`.
- length-competition code commit: `1619fd9`.
- length-competition audit commit: `206dc26`.
- length-competition artifact: `runs/pv26_exhaustive_od_lane_train/lane60_lane_head_transplant_original_stop_pca_20260512/analysis_exports/stopline_fragment_length_competition_readout_val512_epoch2/summary.json`.
- length-competition delta audit artifact: `runs/pv26_exhaustive_od_lane_train/lane60_lane_head_transplant_original_stop_pca_20260512/analysis_exports/stopline_fragment_readout_delta_val512_epoch2/summary.json`.
- length-competition changed axis: keep union groups, let high-confidence single candidates compete with them, and rank the one emitted stop-line by no-GT length evidence.
- length-competition result: best `length_comp_single090_length`, lane / stop-line / crosswalk F1 `0.5480 / 0.5031 / 0.6187`, stop-line TP/FP/FN `121 / 89 / 150`.
- delta audit: union vs length labels are `fp_removed=3`, `tp_added=2`, `tp_lost=1`, `same=363`.
- multi-instance branch/worktree: `exp/lane-family-f1/stopline-fragment-multi-instance`.
- multi-instance code commit: `4c4dc91`.
- multi-instance second-gate commit: `e795098`.
- multi-instance artifact: `runs/pv26_exhaustive_od_lane_train/lane60_lane_head_transplant_original_stop_pca_20260512/analysis_exports/stopline_fragment_multi_instance_readout_val512_epoch2/summary.json`.
- multi-instance gated artifact: `runs/pv26_exhaustive_od_lane_train/lane60_lane_head_transplant_original_stop_pca_20260512/analysis_exports/stopline_fragment_multi_instance_gated_readout_val512_epoch2/summary.json`.
- multi-instance changed axis: keep fragment-union grouping but allow up to two predictions per sample to test whether the one-stop-line emit cap hides valid additional GT stop-lines.
- multi-instance result: best `multi_a16_o48_s080_c2_top2_fallback`, lane / stop-line / crosswalk F1 `0.5480 / 0.5040 / 0.6187`, stop-line TP/FP/FN `125 / 100 / 146`.
- multi-instance gated result: best `multi_a16_o48_s080_c2_top2_second_lenratio070`, lane / stop-line / crosswalk F1 `0.5480 / 0.5061 / 0.6187`, stop-line TP/FP/FN `124 / 95 / 147`.
- projection-split branch/worktree: `exp/lane-family-f1/stopline-fragment-projection-split`.
- projection-split code commit: `a3636aa`.
- projection-split artifact: `runs/pv26_exhaustive_od_lane_train/lane60_lane_head_transplant_original_stop_pca_20260512/analysis_exports/stopline_fragment_projection_split_readout_val512_epoch2/summary.json`.
- projection-split changed axis: split over-merged same-line fragment-union groups by large projection gaps before merging, then apply the same conservative second-instance gate.
- projection-split result: best `proj_split_a16_o48_gap320_c2_top2_second_lenratio070`, lane / stop-line / crosswalk F1 `0.5480 / 0.5112 / 0.6187`, stop-line TP/FP/FN `125 / 93 / 146`.
- projection-competition code commit: `e00df5f`.
- projection-competition artifact: `runs/pv26_exhaustive_od_lane_train/lane60_lane_head_transplant_original_stop_pca_20260512/analysis_exports/stopline_fragment_projection_competition_readout_val512_epoch2/summary.json`.
- projection-competition changed axis: let projection-split union groups and high-confidence single candidates compete by no-GT length evidence, while retaining a conservative second-fragment gate.
- projection-competition result: best `proj_comp_length_s090_top2_second_frag5`, lane / stop-line / crosswalk F1 `0.5480 / 0.5164 / 0.6187`, stop-line TP/FP/FN `126 / 91 / 145`.
- projection selector-audit branch/worktree: `exp/lane-family-f1/stopline-projection-selector-audit`.
- projection selector-audit code commits: `4d16e2f`, `d3871ee`.
- projection selector-audit artifact: `runs/pv26_exhaustive_od_lane_train/lane60_lane_head_transplant_original_stop_pca_20260512/analysis_exports/stopline_projection_competition_selector_audit_val512_epoch2/summary.json`.
- projection selector-audit result: best single-feature no-GT gate is `component_svd_thickness_mean >= 4.681936370001899`, stop-line F1 `0.5250`, TP/FP/FN `126 / 83 / 145`; GT-presence oracle control reaches `0.5575` but is not a production selector. Multifeature logistic overfits: train F1 `0.6635`, held-out F1 `0.4369` vs held-out baseline `0.4848`, full replay F1 `0.5336`.
- projection photometric-selector branch/worktree: `exp/lane-family-f1/stopline-photometric-selector-audit`.
- projection photometric-selector code commit: `44f4f4e`.
- projection photometric-selector artifact: `runs/pv26_exhaustive_od_lane_train/lane60_lane_head_transplant_original_stop_pca_20260512/analysis_exports/stopline_projection_photometric_selector_audit_val512_epoch2/summary.json`.
- projection photometric-selector result: best raw-image photometric rule is `photo_abs_contrast_side12 >= 8.923120498657227`, stop-line F1 `0.5217`, TP/FP/FN `126 / 86 / 145`; adding photometric features does not beat the prior thickness gate (`0.5250`). Logistic with photometric features overfits harder: train F1 `0.6832`, held-out F1 `0.4433` vs held-out baseline `0.4848`, full replay F1 `0.5442`.
- row/x candidate-consistency branch/worktree: `exp/lane-family-f1/stopline-rowx-candidate-consistency-audit`.
- row/x candidate-consistency code commit: `4c7c164`.
- row/x candidate-consistency artifacts: `runs/pv26_exhaustive_od_lane_train/stopline_rowx_candidate_consistency_audit_20260513/analysis_exports/{val128_epoch2,val512_epoch2}/summary.json`.
- row/x candidate-consistency result: val128 held-out looked promising (`baseline 0.5161`, row/x rich logistic `0.5667`, selector_r4 `0.5846`), but val512 held-out stayed weak (`baseline 0.3877`, row/x rich logistic `0.4087`, selector_r4 `0.4259`) and did not beat PCA/projection references.
- normal-support recenter branch/worktree: `exp/lane-family-f1/stopline-normal-support-recenter-readout`.
- normal-support recenter code commit: `6898a5f`.
- normal-support recenter artifacts: `runs/pv26_exhaustive_od_lane_train/stopline_normal_support_recenter_readout_20260513/analysis_exports/{smoke_val4_epoch2,val128_epoch2}/summary.json`.
- normal-support recenter result: exact val128 baseline is stop-line F1 `0.4483`, TP/FP/FN `26 / 30 / 34`; selector-center angle-mask reference is `0.5085`, `30 / 28 / 30`; best normal-scan variants regress to `0.4354`, `32 / 55 / 28`.
- raw-edge recenter branch/worktree: `exp/lane-family-f1/stopline-raw-edge-recenter-readout`.
- raw-edge recenter code commit: `5abfb41`.
- raw-edge recenter artifacts: `runs/pv26_exhaustive_od_lane_train/stopline_raw_edge_recenter_readout_20260513/analysis_exports/{smoke_val4_epoch2,val128_epoch2}/summary.json`.
- raw-edge recenter result: exact val128 baseline is stop-line F1 `0.4483`, TP/FP/FN `26 / 30 / 34`; selector-center angle-mask reference is `0.5085`, `30 / 28 / 30`; raw-edge normal scan regresses to `0.2993`, `22 / 65 / 38`, and the offset-penalty variant regresses further to `0.2585`, `19 / 68 / 41`.
- 판단: low-score extension regresses and is closed. Length competition is a small partial-positive over fragment union (`0.4948 -> 0.5031`, `+1 TP`, `-5 FP`) but the delta audit shows the gain is confined to six samples and mostly fallback suppression, not broad geometry recovery. Multi-instance and second-instance gating are weak partial positives. Projection splitting gives a real FP improvement, and projection-competition is the current ungated local stop-line readout reference (`0.5164`). The selector audits can suppress some FP but either stay weak (`0.5250` single-feature, `0.5217` photometric) or fail held-out generalization (`0.4369` exported-feature logistic, `0.4433` photometric logistic), so they are not deployable selector paths. Row/x projection consistency also fails to become a stronger held-out selector on val512. Normal-support recentering adds TP on exact val128 but doubles FP enough to fall below baseline, and raw-edge contrast recentering collapses TP while adding FP, so neither is a midpoint recovery path. Do not repeat these as feature-rank, single-score, top-K, second-prediction threshold, projection-gap, projection-competition length/min-score, projection-competition single-feature selector, exported-feature logistic selector, raw-image photometric selector, row/x consistency selector, normal-support recenter, or raw-edge recenter sweeps. The next stop-line step must improve candidate generation/midpoint recovery or introduce a materially new no-GT signal, while preserving crosswalk hull retention and lane composition explicitly.

Latest stop-line center-rank margin probe:

- artifact: `runs/pv26_exhaustive_od_lane_train/lane60_core_centerline_refine_row_scan_tangent_stop_center_rank_margin_from_lane60_core_centerline_refine_cross_retain_from_exhaustive_od_lane_default_20260505_032217_default_20260510_003412_default_20260512_212813/phase_4/history/epochs.jsonl`
- experiment: `core_centerline_refine_row_scan_tangent_stop_center_rank_margin`
- changed axis: add opt-in `stopline_center_rank_margin_weight=0.35`, which pushes the GT stop-line midpoint proposal above top hard-negative center/selector proposals.
- best exact val128 epoch2 objective: `0.6170`
- lane / stop-line / crosswalk F1: `0.5605 / 0.4602 / 0.5854`
- TP/FP/FN stop-line: `26 / 27 / 34`
- 판단: valid runtime and a tiny stop-line gain, but still below tangent-link exact objective `0.6187`, segment-MIL lane-head-only exact objective `0.6193`, and geometry-validator stop-line `0.4655`. Do not broaden; margin-only midpoint ranking is not enough.

Latest stop-line center-rank margin readout replay:

- artifact: `runs/pv26_exhaustive_od_lane_train/lane60_core_centerline_refine_row_scan_tangent_stop_center_rank_margin_from_lane60_core_centerline_refine_cross_retain_from_exhaustive_od_lane_default_20260505_032217_default_20260510_003412_default_20260512_212813/analysis_exports/stopline_pred_angle_mask_extent_val128_epoch2/summary.json`
- tool: `tools/probe_pv26_stopline_pred_angle_mask_extent.py`
- changed axis: keep the trained center-rank checkpoint fixed, then replay production-style predicted-proposal + angle-anchored mask-extent stop-line readout.
- best exact val128 variant: `pred_selector_top1_s060_mask050_band4`
- lane / stop-line / crosswalk F1: `0.5252 / 0.5042 / 0.5854`
- TP/FP/FN stop-line: `30 / 29 / 30`
- 판단: readout replay recovers stop-line over the checkpoint baseline `0.4602`, but it still misses the prior angle-mask production exact reference `0.5085` and PCA val128 reference `0.5133`. Do not broaden this checkpoint through angle-mask replay.

Latest center-rank proposal recall audit:

- artifact: `runs/pv26_exhaustive_od_lane_train/lane60_core_centerline_refine_row_scan_tangent_stop_center_rank_margin_from_lane60_core_centerline_refine_cross_retain_from_exhaustive_od_lane_default_20260505_032217_default_20260510_003412_default_20260512_212813/analysis_exports/stopline_proposal_recall_val128_epoch2/summary.json`
- tool: `tools/probe_pv26_stopline_proposal_recall.py`
- changed axis: keep the center-rank checkpoint fixed and measure whether GT stop-line centers moved into better proposal-map top-k positions.
- exact val128 `max` source: `max_r8 >= 0.6` is `50/60`, top3-hit-r8 is `39/60`, top10-hit-r8 is `52/60`, raw rank top3 is `11/60`.
- prior original-checkpoint val128 `max` source: `50/60`, `37/60`, `53/60`, raw rank top3 `9/60`.
- 판단: center-rank margin only gives tiny top3/raw-rank movement and does not improve local recall or top10 recall. This does not recover the positive no-oracle bucket and is not a candidate-generation fix.

Latest lane flip-TTA audit:

- artifact exact: `runs/pv26_exhaustive_od_lane_train/lane60_lane_flip_tta_audit_20260512/analysis_exports/exact_val128_current_best_epoch2/summary.json`
- artifact broader: `runs/pv26_exhaustive_od_lane_train/lane60_lane_flip_tta_audit_20260512/analysis_exports/broader_val512_current_best_epoch2/summary.json`
- branch: `exp/lane-family-f1/lane-flip-tta-audit`
- changed axis: keep checkpoint, stop-line/crosswalk outputs, stop-line `mask=0.80`, and crosswalk hull decode fixed; run a horizontal-flip forward pass and merge only the lane centerline logits by averaging.
- exact val128 objective: `0.6296149306`
- exact lane / stop-line / crosswalk F1: `0.5854 / 0.4364 / 0.5988`
- broader-val512 objective: `0.6216194906`
- broader lane / stop-line / crosswalk F1: `0.5577 / 0.4235 / 0.6187`
- broader lane TP/FP/FN: `4518 / 2206 / 4959`
- 판단: flip centerline averaging is a valid runtime/postprocess partial-positive: it improves broader lane F1 by `+0.0098` and objective by about `+0.0040` over the prior transplanted composite. It does not close lane 0.6, and it does nothing for the stop-line bottleneck.

Latest segment-MIL lane-head-only + flip-TTA replay:

- artifact exact: `runs/pv26_exhaustive_od_lane_train/lane60_lane_flip_tta_on_segment_mil_lane_head_only_20260512/analysis_exports/exact_val128_epoch2/summary.json`
- artifact broader: `runs/pv26_exhaustive_od_lane_train/lane60_lane_flip_tta_on_segment_mil_lane_head_only_20260512/analysis_exports/broader_val512_epoch2/summary.json`
- changed axis: keep the `core_centerline_refine_row_scan_tangent_segment_mil_lane_head_only` checkpoint fixed, then apply only `flip_centerline_avg` lane TTA with stop-line `mask=0.80` and crosswalk hull decode.
- exact val128 `flip_centerline_avg`: objective `0.6298`, lane/stop/cross F1 `0.5854 / 0.4364 / 0.6061`.
- broader-val512 `flip_centerline_avg`: objective `0.6207`, lane/stop/cross F1 `0.5577 / 0.4184 / 0.6185`.
- 판단: exact looks slightly stronger than the segment-MIL baseline, but broader does not beat the current broader best objective `0.6216`, and stop-line is lower than the current broader best `0.4235`. Do not promote this combination.

Latest lane geometry-filter probe:

- artifact: `runs/pv26_exhaustive_od_lane_train/lane60_lane_head_transplant_original_stop_pca_20260512/analysis_exports/lane_geometry_filter_probe_val128_epoch2/summary.json`
- changed axis: keep the lane-head transplant checkpoint fixed and test stricter lane bbox area/aspect filters on exact val128 before any broader replay.
- baseline lane F1: `0.5660`, TP/FP/FN `1162 / 554 / 1228`.
- best stricter candidate by objective, `lane_bbox_area_8192`, lane F1: `0.5580`, TP/FP/FN `1097 / 445 / 1293`.
- 판단: stricter geometry filters reduce FP but lose too many TP. This is not a lane 0.6 path and is not broadened.

Latest stop-line flip-TTA audit:

- artifact smoke: `runs/pv26_exhaustive_od_lane_train/lane60_stopline_flip_tta_audit_20260512/analysis_exports/smoke_val4_current_best_epoch2/summary.json`
- branch: `exp/lane-family-f1/stopline-flip-tta-audit`
- changed axis: keep checkpoint, lane/crosswalk outputs, stop-line `mask=0.80`, and crosswalk hull decode fixed; run a horizontal-flip forward pass and merge only stop-line dense score maps and/or geometry maps by averaging.
- smoke val4 variants: `baseline`, `flip_stop_score_avg`, `flip_stop_geometry_avg`, `flip_stop_all_avg`.
- smoke val4 result: all four variants are identical at objective `0.6395`, lane/stop-line/crosswalk F1 `0.5507 / 0.0000 / 0.5455`, stop-line TP/FP/FN `0 / 3 / 2`.
- 판단: this has no stop-line signal even as a smoke probe, so it was not expanded to val128/broader. Do not treat simple stop-line flip score/geometry averaging as a current improvement path.

Latest lane flip-consistency instance-evidence audit:

- artifact exact: `runs/pv26_exhaustive_od_lane_train/lane60_lane_flip_instance_evidence_20260512/analysis_exports/exact_val128_current_best_epoch2/summary.json`
- artifact broader: `runs/pv26_exhaustive_od_lane_train/lane60_lane_flip_instance_evidence_20260512/analysis_exports/broader_val512_current_best_epoch2/summary.json`
- artifact exact flip-centerline-avg follow-up: `runs/pv26_exhaustive_od_lane_train/lane60_lane_flip_instance_evidence_20260512/analysis_exports/exact_val128_flip_centerline_avg_instance_gate_epoch2/summary.json`
- artifact broader flip-centerline-avg follow-up: `runs/pv26_exhaustive_od_lane_train/lane60_lane_flip_instance_evidence_20260512/analysis_exports/broader_val512_flip_centerline_avg_instance_gate_epoch2/summary.json`
- branch: `exp/lane-family-f1/lane-flip-instance-evidence`
- changed axis: keep the current transplanted composite and row-scan tangent lane decode fixed, add normal/flip centerline agreement features to the existing read-only lane instance evidence validator, then replay the same row gate on top of the existing `flip_centerline_avg` runtime lane baseline.
- exact val128 heldout result: logistic instance gate lane F1 `0.6002` vs heldout baseline `0.5843`, but stop-line/crosswalk heldout F1 `0.4667 / 0.5952`.
- broader val512 heldout result: logistic instance gate lane F1 `0.5660` vs heldout baseline `0.5473`; full split-count replay implies lane F1 `0.5682`, TP/FP/FN `4332 / 1439 / 5145`, compared with baseline `0.5480`, `4457 / 2333 / 5020`.
- no-flip broader ablation: `runs/pv26_exhaustive_od_lane_train/lane60_lane_flip_instance_evidence_20260512/analysis_exports/broader_val512_no_flip_ablation_epoch2/summary.json`; logistic heldout lane F1 `0.5663`, full split-count lane F1 `0.5675`.
- flip-centerline-avg follow-up: exact heldout logistic lane F1 `0.6125`, but broader heldout lane F1 only `0.5694`; broader full split-count lane F1 is `0.5738`, TP/FP/FN `4455 / 1596 / 5022`, with stop-line/crosswalk unchanged at `0.4235 / 0.6187`.
- flip-only single features are weaker than the logistic mix on broader heldout: `flip_center_point_mean` lane F1 `0.5587`, `center_consensus_point_mean` `0.5583`.
- 판단: lane instance gating has real row-level signal and can suppress many lane FP, and the strongest read-only lane diagnostic is now `0.5738` full split-count on top of `flip_centerline_avg`. It still trades away TP/FN, leaves broader lane below `0.60`, and does not address stop-line. It is read-only post-hoc evidence, not a production decoder or all-task success.

Latest lane instance oracle-selection audit:

- artifact: `runs/pv26_exhaustive_od_lane_train/lane60_lane_instance_oracle_selection_20260512/analysis_exports/broader_val512_flip_centerline_avg_oracle_selection_retention_epoch2/summary.json`
- branch: `exp/lane-family-f1/lane-instance-oracle-selection-audit`
- changed axis: keep the current `flip_centerline_avg` runtime lane baseline plus stop-line/crosswalk retention overrides fixed, then add a GT-matching oracle selector that keeps only row-scan lane predictions already matched as TP. This is read-only diagnostic evidence, not production decode.
- broader full baseline: lane/stop/cross F1 `0.5577 / 0.4235 / 0.6187`, lane TP/FP/FN `4518 / 2206 / 4959`.
- oracle TP-only selector: lane/stop/cross F1 `0.6457 / 0.4235 / 0.6187`, lane TP/FP/FN `4518 / 0 / 4959`.
- learned logistic heldout reference under the same retention config: lane F1 `0.5700`, TP/FP/FN `2173 / 830 / 2449`, below the oracle selector and below lane `0.60`.
- 판단: current row-scan/flip candidate set has enough matched lane predictions for a perfect selector to exceed lane `0.60`; the lane gap is primarily FP suppression / instance selection, not only missing lane candidates. But the available logistic/post-hoc selector still falls far short, and stop-line remains unchanged at `0.4235`, so all-task success is still blocked.

Latest lane row-scan duplicate suppression audit:

- artifact: `runs/pv26_exhaustive_od_lane_train/lane60_lane_duplicate_suppression_20260512/analysis_exports/broader_val512_flip_centerline_avg_duplicate_d24_retention_epoch2/summary.json`
- branch: `exp/lane-family-f1/lane-row-scan-duplicate-suppression`
- changed axis: keep the current `flip_centerline_avg` runtime lane baseline, stop-line retention override, and crosswalk hull decode fixed, then suppress only same-class/same-type lane rows whose mean point distance is `<=24` pixels.
- broader full baseline: lane/stop/cross F1 `0.5577 / 0.4235 / 0.6187`, lane TP/FP/FN `4518 / 2206 / 4959`.
- duplicate suppression: lane/stop/cross F1 `0.5578 / 0.4235 / 0.6187`, lane TP/FP/FN `4518 / 2205 / 4959`.
- held-out duplicate suppression similarly removes only one FP: lane F1 `0.5562` vs baseline `0.5561`, TP/FP/FN `2212 / 1120 / 2410`.
- 판단: the oracle selector gap is not explained by near-duplicate row-scan emissions. A simple same-schema geometric dedupe is a no-op for the current broader FP problem, so the next lane path still needs a stronger instance selection/readout contract rather than distance-threshold dedupe.

Latest lane per-sample top-k cap audit:

- artifact: `runs/pv26_exhaustive_od_lane_train/lane60_lane_sample_topk_cap_20260512/analysis_exports/broader_val512_flip_centerline_avg_sample_top5_logistic_retention_epoch2/summary.json`
- branch: `exp/lane-family-f1/lane-sample-topk-cap-audit`
- changed axis: keep the same `flip_centerline_avg` runtime lane baseline plus stop-line/crosswalk retention overrides fixed, then cap each sample to its top 5 row-scan lane predictions by the existing logistic evidence score.
- broader full result: lane/stop/cross F1 `0.5618 / 0.4235 / 0.6187`, lane TP/FP/FN `4490 / 2016 / 4987`, vs baseline `0.5577`, `4518 / 2206 / 4959`.
- broader held-out result: lane F1 `0.5596`, TP/FP/FN `2201 / 1043 / 2421`, vs held-out baseline `0.5561`, `2212 / 1121 / 2410`.
- 판단: per-sample cap removes some FP but also drops TP. The gain is small and remains below the stronger logistic threshold diagnostic (`0.5700` held-out) and far below the oracle TP-only selector (`0.6457` full). Simple sample-level emission capping is not enough to recover the lane oracle gap.

Git-history backfill: closed architecture probes not in the current code line:

- `exp/lane-family-f1/lane-centerline-dilated-context`: centerline branch dilation was operational but exact objective `0.6155`, lane/stop/cross F1 `0.5617 / 0.4348 / 0.5854`, below tangent-link and segment-MIL lane-head-only references.
- `exp/lane-family-f1/lane-support-conditioned-centerline`: detached support-conditioned centerline refinement was operational but exact objective `0.6152`, lane/stop/cross F1 `0.5607 / 0.4348 / 0.5854`, below the same references.
- `exp/lane-family-f1/stopline-center-stem`: wiring the unused stop-line `center_stem` into center outputs was a valid architecture cleanup but exact stop-line F1 fell to `0.3704`, TP/FP/FN `20 / 28 / 40`.
- 판단: these are retained as negative branch evidence, not merged code. The next lane/stop-line axis should not repeat receptive-field widening, support-conditioned centerline refinement, or stop-line center-stem cleanup.

Latest lane vectorizer scope probe:

- artifact: `analysis_exports/lane_row_scan_tangent_component_val128_epoch2/summary.json`
- experiment: `core_centerline_refine_row_scan_tangent_component`
- objective: `0.6106257251980678`
- lane / stop-line / crosswalk F1: `0.5305 / 0.4483 / 0.5854`
- lane TP/FP/FN: `1030 / 463 / 1360`
- 판단: component 내부로만 tangent row-scan 연결을 제한하면 baseline보다는 lane이 조금 오르지만 current tangent-link exact reference `0.5633`, `1121 / 469 / 1269`에 크게 못 미친다. over-link를 줄이는 대신 TP를 많이 잃는 쪽이라 broader-val512로 확장하지 않는다.

Latest lane row-scan link oracle audit:

- artifact: `analysis_exports/lane_tangent_oracle_val128_epoch2/summary.json`
- experiment: `core_centerline_refine_row_scan_tangent_link`
- read-only variants: keep the checkpoint fixed, then replace only selected seg-first lane maps before postprocess.
- baseline exact val128 lane F1: `0.5633`, lane TP/FP/FN `1121 / 469 / 1269`.
- `gt_tangent_axis` lane F1: `0.5611`, lane TP/FP/FN `1116 / 472 / 1274`.
- `gt_centerline_core` lane F1: `0.6778`, lane TP/FP/FN `1243 / 35 / 1147`.
- `gt_centerline_core_gt_tangent` lane F1: `0.6781`, lane TP/FP/FN `1244 / 35 / 1146`.
- 판단: tangent-axis oracle alone does not rescue row-scan linking; GT centerline core does. The next lane axis should target predicted centerline coverage/quality as an instance-level contract, not another tangent/link cost sweep.

Latest lane centerline instance-balance probe:

- artifact: `runs/pv26_exhaustive_od_lane_train/lane60_core_centerline_refine_row_scan_tangent_instance_balance_from_lane60_core_centerline_refine_cross_retain_from_exhaustive_od_lane_default_20260505_032217_default_20260510_003412_default_20260512_071343/phase_4/history/epochs.jsonl`
- experiment: `core_centerline_refine_row_scan_tangent_instance_balance`
- changed axis: add `lane_seg_centerline_instance_weight` and opt-in `lane_segfirst_instance_centerline_weight=0.35`.
- best exact val128 epoch2 objective: `0.6161312489`
- lane / stop-line / crosswalk F1: `0.5654 / 0.4310 / 0.5854`
- lane TP/FP/FN: `1171 / 581 / 1219`
- 판단: lane F1 is slightly above tangent-link exact `0.5633`, but below segment-MIL lane-head-only `0.5660`; stop-line regresses below tangent-link `0.4483`. Instance-balanced positive centerline loss alone is not a broader-val512 expansion path.

Latest lane soft-instance shell probe:

- artifact: `runs/pv26_exhaustive_od_lane_train/lane60_core_centerline_refine_row_scan_tangent_soft_instance_shell_from_lane60_core_centerline_refine_cross_retain_from_exhaustive_od_lane_default_20260505_032217_default_20260510_003412_default_20260512_093000/phase_4/history/epochs.jsonl`
- experiment: `core_centerline_refine_row_scan_tangent_soft_instance_shell`
- changed axis: keep the core centerline target and add a separate instance-balanced soft-shell BCE around each lane centerline with `lane_segfirst_soft_instance_centerline_weight=0.35`.
- best exact val128 epoch2 objective: `0.6159063607`
- lane / stop-line / crosswalk F1: `0.5641 / 0.4348 / 0.5854`
- lane TP/FP/FN: `1144 / 522 / 1246`
- stop TP/FP/FN: `25 / 30 / 35`
- 판단: lane F1 is only noise-level above tangent-link exact `0.5633` and below segment-MIL lane-head-only `0.5660`; stop-line still regresses below tangent-link `0.4483`. Soft-shell instance-balanced side supervision is not a broader-val512 expansion path.

Latest lane centerline soft-ignore probe:

- artifact: `runs/pv26_exhaustive_od_lane_train/lane60_core_centerline_refine_row_scan_tangent_soft_ignore_band_from_lane60_core_centerline_refine_cross_retain_from_exhaustive_od_lane_default_20260505_032217_default_20260510_003412_default_20260512_095254/phase_4/history/epochs.jsonl`
- experiment: `core_centerline_refine_row_scan_tangent_soft_ignore_band`
- changed axis: keep `lane_segfirst_centerline_target_mode=core`, but exclude non-core pixels with `lane_seg_centerline_soft >= 0.20` from centerline BCE/Dice/Focal loss.
- best exact val128 epoch2 objective: `0.6088389162`
- lane / stop-line / crosswalk F1: `0.5360 / 0.4348 / 0.5854`
- lane TP/FP/FN: `1032 / 429 / 1358`
- stop TP/FP/FN: `25 / 30 / 35`
- 판단: soft-shell negative masking cuts FP but loses too much lane TP, far below tangent-link exact `0.5633`, instance-balance `0.5654`, soft-instance shell `0.5641`, and segment-MIL lane-head-only `0.5660`. Soft-band ignore/threshold-only masking is not a broader-val512 expansion path.

Latest lane centerline threshold oracle audit:

- artifact: `analysis_exports/lane_centerline_threshold_oracle_val128_epoch2/summary.json`
- script: `tools/probe_pv26_lane_centerline_threshold_oracle.py`
- experiment: fixed `core_centerline_refine_row_scan_tangent_link` checkpoint, then compare global `lane_obj_threshold` candidates against a per-sample dense-core oracle threshold.
- best global result: `global_t020`, lane/stop/cross F1 `0.5641 / 0.4483 / 0.5988`, lane TP/FP/FN `1153 / 545 / 1237`.
- current reference-like `global_t045`: lane/stop/cross F1 `0.5633 / 0.4483 / 0.5988`, lane TP/FP/FN `1121 / 469 / 1269`.
- `sample_oracle_dense_core`: lane/stop/cross F1 `0.5482 / 0.4483 / 0.5988`, lane TP/FP/FN `1074 / 454 / 1316`, selected-threshold mean/min/max `0.6380 / 0.2000 / 0.8000`.
- 판단: global threshold calibration adds only noise-level lane gain over the current tangent-link exact reference and still stays below the exact lane-head-only best `0.5660`. Sample-wise dense-core thresholding is worse. Lane threshold-only calibration/readout is closed; the next lane axis needs a stronger predicted-centerline instance contract rather than another threshold sweep.

Latest stop-line freeze-policy probe:

- artifact: `runs/pv26_exhaustive_od_lane_train/lane60_core_centerline_refine_row_scan_tangent_stop_geometry_validator_stop_head_only_from_lane60_core_centerline_refine_cross_retain_from_exhaustive_od_lane_default_20260505_032217_default_20260510_003412_default_20260511_231815/phase_4/history/epochs.jsonl`
- experiment: `core_centerline_refine_row_scan_tangent_stop_geometry_validator_stop_head_only`
- objective: `0.6179851856657268`
- lane / stop-line / crosswalk F1: `0.5640 / 0.4386 / 0.5926`
- lane TP/FP/FN: `1121 / 464 / 1269`
- stop TP/FP/FN: `25 / 29 / 35`
- cross TP/FP/FN: `48 / 33 / 33`
- 판단: stop-line head만 업데이트해도 geometry-validator stop-line gain이 보존되지 않았다 (`0.4655 -> 0.4386`). head isolation은 lane/crosswalk retention에는 도움되지만 stop-line rescue가 아니므로 broader-val512로 확장하지 않는다.

Latest stop-line candidate-select contract probe:

- artifact: `runs/pv26_exhaustive_od_lane_train/lane60_core_centerline_refine_row_scan_tangent_stop_candidate_select_gap4_from_lane60_core_centerline_refine_cross_retain_from_exhaustive_od_lane_default_20260505_032217_default_20260510_003412_default_20260512_080843/phase_4/history/epochs.jsonl`
- experiment: `core_centerline_refine_row_scan_tangent_stop_candidate_select_gap4`
- changed axis: supervise denser `gap4/top50` stop-line candidates into `stop_line_candidate_validator_logits` and `stop_line_presence_logits`, then read with `stop_line_component_gate_source=max_validator`.
- best exact val128 objective: epoch1 `0.5980293261`; epoch2 dropped to `0.5855145029`.
- epoch2 lane / stop-line / crosswalk F1: `0.5612 / 0.0000 / 0.5854`
- epoch2 stop TP/FP/FN: `0 / 0 / 60`
- 판단: runtime and plumbing are valid, but the candidate-select + presence/max-validator gate suppresses stop-line emission completely. This is below tangent-link, PCA, angle-mask, task-head merge, and geometry-validator references, so broader-val512로 확장하지 않는다.

Latest stop-line candidate-select gate replay:

- artifact default: `runs/pv26_exhaustive_od_lane_train/lane60_core_centerline_refine_row_scan_tangent_stop_candidate_select_gap4_from_lane60_core_centerline_refine_cross_retain_from_exhaustive_od_lane_default_20260505_032217_default_20260510_003412_default_20260512_080843/analysis_exports/candidate_select_gate_replay_default_val128_epoch2/summary.json`
- artifact no-presence: `runs/pv26_exhaustive_od_lane_train/lane60_core_centerline_refine_row_scan_tangent_stop_candidate_select_gap4_from_lane60_core_centerline_refine_cross_retain_from_exhaustive_od_lane_default_20260505_032217_default_20260510_003412_default_20260512_080843/analysis_exports/candidate_select_gate_replay_max_validator_no_presence_val128_epoch2/summary.json`
- changed axis: evaluator-only replay of the same `best.pt`; no model/loss/sampler changes.
- default `max_validator + presence=0.35`: lane/stop/cross F1 `0.5540 / 0.0000 / 0.5590`, stop TP/FP/FN `0 / 0 / 60`.
- `max_validator + presence=0.0`: lane/stop/cross F1 `0.5540 / 0.4483 / 0.5590`, stop TP/FP/FN `26 / 30 / 34`.
- `center + presence=0.0`: lane/stop/cross F1 `0.5540 / 0.4386 / 0.5590`, stop TP/FP/FN `25 / 29 / 35`.
- 판단: 0-emission failure is mainly the sample-level presence gate, not the `max_validator` component gate alone. But removing presence only recovers the old tangent-link-level stop-line F1 and crosswalk on `best.pt` is weak, so this is diagnostic evidence, not a 0.6 path.

Latest preprocessing/runtime probe:

- artifact: `runs/pv26_exhaustive_od_lane_train/lane60_core_centerline_refine_row_scan_tangent_no_aug_from_lane60_core_centerline_refine_cross_retain_from_exhaustive_od_lane_default_20260505_032217_default_20260510_003412_default_20260512_044501/phase_4/history/epochs.jsonl`
- experiment: `core_centerline_refine_row_scan_tangent_no_aug`
- changed axis: keep `row_scan_tangent` and loss/postprocess contract fixed, disable stage-4 `train_augmentation`.
- objective: `0.6093231418573035`
- lane / stop-line / crosswalk F1: `0.5595 / 0.4298 / 0.5476`
- lane TP/FP/FN: `1109 / 465 / 1281`
- stop TP/FP/FN: `26 / 35 / 34`
- cross TP/FP/FN: `46 / 41 / 35`
- 판단: augmentation-off wiring and runtime are valid, but exact epoch2 is below tangent-link reference `0.6187`, `0.5633 / 0.4483 / 0.5854`. Crosswalk task-best reached `0.6788` at epoch1 but the selected epoch2 checkpoint loses crosswalk retention, so broader-val512로 확장하지 않는다.

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
- candidate-pool audit은 top10/top20 후보 안에 valid stop-line segment headroom이 있음을 보였다. broader-val512 oracle-positive selection은 stop-line F1 `0.6517`, TP/FP/FN `131 / 0 / 140`까지 가능하지만, production score/length filters best는 `0.4371`, TP/FP/FN `106 / 108 / 165`로 PCA broader reference `0.4699`보다 낮다. 즉 candidate pool은 있으나 score/length만으로는 FP suppression signal이 부족하다.
- candidate generation gap audit은 top-k NMS gap을 `10 -> 6/4`로 줄이고 top-k를 `20 -> 50`까지 넓히면 oracle-positive headroom이 더 있음을 보였다. broader-val512 `gap4_oracle_max_top50_positive`는 stop-line F1 `0.6877`, TP/FP/FN `142 / 0 / 129`로 기존 oracle top20 `0.6517`, `131 / 0 / 140`보다 높다. 하지만 production `gap4_max_top50_score_s080`은 기존 `max_top10_score_s080`과 같은 `0.4371`, TP/FP/FN `106 / 108 / 165`에 머문다. 결론은 dense candidate pool을 더 넓히는 것만으로는 production path가 아니고, denser 후보를 고를 task-aware selector/readout contract가 따로 필요하다는 것이다.
- hard-negative proposal ranking loss short run은 proposal competition을 직접 누르는 opt-in loss를 시험했지만 exact val128에서 확장 조건을 만들지 못했다. stage4-only 2epoch, train512/val128, weight `0.25` best는 lane/stop/cross F1 `0.4754 / 0.4918 / 0.5767`, stop-line TP/FP/FN `30 / 32 / 30`이다. stop-line은 baseline `0.4483`보다 높지만 angle-mask production `0.5085`와 PCA val128 reference `0.5133`보다 낮고 lane/crosswalk가 같이 내려갔다. 같은 checkpoint의 candidate-pool production best도 stop-line F1 `0.4286`이라 broader-val512로 확장하지 않는다.
- candidate scalar-feature validator audit은 existing CSV feature만으로 후보 row label을 어느 정도 분리할 수 있음을 보였다. val512 half-split logistic은 test AUC `0.7894`, AP `0.5699`, candidate-level F1 `0.5651`; score 단일 feature도 test candidate-level F1 `0.5812`다. 하지만 이건 candidate-row 분류이고, 같은 scalar score/length production replay는 stop-line task F1 `0.4371`에 머물렀다. 따라서 scalar CSV classifier를 production validator로 보지 않고, 실제 instance/task-aware validator head가 필요하다.
- candidate rich-feature validator audit은 proposal 좌표, decoded center 좌표, center/selector/mask point/window probability까지 CSV에 추가하면 row-level separability가 더 좋아짐을 보였다. val512 half-split logistic은 33개 feature에서 test AUC/AP/F1 `0.8149 / 0.5950 / 0.6175`, oracle-best row F1 `0.6387`이고, 단일 feature로도 `selector_r4_max`가 row F1 `0.6109`를 냈다. 그러나 candidate-pool task replay 자체는 그대로이고 best production stop-line F1은 여전히 `0.4371`이다. 결론은 "rich map-local signal은 model-side candidate instance validator 후보를 정당화한다"이지 "CSV threshold decoder가 성공했다"가 아니다.
- candidate rich-validator held-out task replay는 row signal이 task selection으로 일부 옮겨지지만 gate를 통과하지 못함을 보였다. val512 앞 256 batch에서 threshold를 맞추고 뒤 256 batch를 평가하면 held-out baseline stop-line F1 `0.3877` 대비 rich logistic `0.4231`, `selector_r4_max` `0.4259`까지 오른다. 하지만 PCA broader reference `0.4699`에도 못 미치고, 0.6 목표와는 거리가 크다. 따라서 rich CSV/logistic threshold replay는 production path가 아니며, stop-line을 계속한다면 단순 score replay가 아니라 실제 model-side instance validator loss/head가 필요하다.
- gap4/top50 denser pool 위의 rich-selector held-out task replay는 약한 개선만 만들었다. 같은 val512 half split에서 `--rich-validator-min-gap 4.0 --rich-validator-top-k 50`을 쓰면 held-out baseline stop-line F1 `0.3877` 대비 rich logistic task threshold `0.4190`, rich logistic row threshold `0.4298`, `selector_r4_max` task threshold `0.4537`까지 오른다. 기존 gap10/top20 `selector_r4_max` `0.4259`보다는 높지만 PCA broader reference `0.4699`와 0.6 목표에는 못 미친다. denser 후보와 rich local feature는 selector/readout contract의 전제 evidence일 뿐, CSV threshold replay 자체는 production path가 아니다.
- selector feature-patch validator audit은 dense `stop_line_selector_feature`에서 proposal/decoded center 주변 128-channel patch mean을 꺼내 289-feature logistic replay를 시험했지만 gap4/top50 held-out task F1이 `0.3982`에 그쳤다. 같은 split의 held-out baseline `0.3877`보다는 아주 조금 높지만 기존 `selector_r4_max` task threshold `0.4558`보다 낮고 PCA broader reference `0.4699`와도 멀다. 따라서 raw selector feature embedding을 offline logistic/MLP로 더 키우는 방향은 현재 evidence만으로 production path가 아니다.
- gap4/top50 candidate feature rank diagnostic은 positive sample 안에서는 local mask/selector 신호가 oracle 후보를 꽤 잘 올린다는 것을 보였다. `mask_r4_max`는 oracle-positive가 있는 142 sample 중 top1 hit `116/142` (`0.8169`), top3 hit `124/142` (`0.8732`)다. 하지만 후보가 있는 sample은 369개이고 그중 227개는 oracle-positive가 없는 negative candidate sample이다. 즉 다음 stop-line 병목은 "positive sample 안에서 어떤 후보가 1등인가"만이 아니라, "후보가 있는 sample에서 emit/no-emit을 어떻게 가르는가"다. 단순 ranking feature나 threshold replay는 production path가 아니다.
- gap4/top50 sample-level emit gate audit은 이 premise를 더 좁혔다. 같은 candidate rows에서 held-out emit-all surrogate selection F1은 `0.4320`이고, sample logistic gate는 held-out sample F1 `0.6585`, surrogate selection F1 `0.5732`까지 올렸다. 하지만 이것은 oracle-label surrogate이고 task F1이나 production decoder가 아니므로, 다음 stop-line 축은 이를 실제 task replay 또는 model-side emit gate로 검증해야 한다.
- gap4/top50 sample-gate actual task replay는 surrogate gain이 evaluator task F1로 전이되지 않음을 보였다. 같은 threshold `0.2471111762446392`에서 train stop-line F1은 `0.4269 -> 0.4640`으로 올랐지만 held-out은 `0.3877 -> 0.3843`으로 낮아졌다. Full val512 합산도 stop-line TP/FP/FN `102 / 106 / 169`, F1 `0.4259`로 PCA broader reference `0.4699`와 0.6 목표에 못 미친다. CSV logistic sample gate는 production path가 아니다.
- gap4/top50 baseline-preserving rescue replay는 sample-gate 후보를 baseline에 fallback/append하면 train overfit을 줄일 수 있는지 봤지만 역시 gate를 넘지 못했다. Full val512 합산 기준 best rescue는 `baseline_or_gate` stop-line F1 `0.4165`, TP/FP/FN `106 / 132 / 165`이고, `baseline_plus_gate_max2`는 `0.4141`, `100 / 112 / 171`이다. Baseline `0.4083`보다는 약간 높지만 기존 sample-gate actual replay `0.4259`, score-threshold production `0.4371`, PCA broader reference `0.4699`보다 낮다. baseline 보존형 sample-gate rescue도 production path가 아니다.
- model-side candidate instance validator head는 opt-in 구현과 runtime smoke는 통과했지만 exact val128 gate에서 실패했다. Direct `validator` gate는 epoch1/2 stop-line F1이 모두 `0.0000`이고, 같은 epoch2 checkpoint를 baseline `center` gate로 되돌려도 stop-line F1 `0.4211`로 기준 `0.4483`보다 낮다. cold validator gate를 바로 production gate로 쓰는 방식과 현재 hard-negative auxiliary 조합은 0.6 path가 아니다.
- candidate validator calibrated replay는 direct collapse를 일부 회복했지만 exact val128 gate를 넘지 못했다. Same checkpoint에서 `product_validator` + fallback best는 lane/stop/cross F1 `0.5271 / 0.4918 / 0.5854`, stop-line TP/FP/FN `30 / 32 / 30`이다. direct validator `0.0000`과 center-gate replay `0.4211`보다는 낫지만 PCA val128 reference `0.5133`과 angle-mask production `0.5085`보다 낮으므로 broader-val512로 확장하지 않는다.
- candidate validator warm-bias smoke는 direct dense validator gate 실패가 단순 초기 bias 문제인지 확인했다. bias `2.0`과 `4.0` 모두 train8/val4 smoke에서 stop-line F1 `0.0000`, TP/FP/FN `0 / 2 / 2`라 exact로 확장하지 않는다.
- stop-line proposal competition loss는 center/selector proposal map을 GT center heatmap distribution 쪽으로 직접 경쟁시키는 opt-in loss를 시험했지만 exact val128 gate에서 실패했다. Best epoch2 objective는 `0.6051`, lane/stop/cross F1은 `0.5552 / 0.4107 / 0.5854`, stop-line TP/FP/FN은 `23 / 29 / 37`이다. Prior row-scan reference `0.6144`, `0.5522 / 0.4483 / 0.5854`보다 objective와 stop-line이 낮고 loss 계산 비용도 커서 proposal-distribution-loss-only는 0.6 path가 아니다.
- modern task-head merge replay는 retrain 없이 row-scan lane decode와 proposal-rank stop-line head를 결합하면 exact val128 objective가 `0.6196`까지 오름을 보였다. Best exact merge는 lane/stop/cross F1 `0.5522 / 0.4918 / 0.5854`, stop-line TP/FP/FN `30 / 32 / 30`이다. 그러나 broader-val512에서는 objective `0.5977`, lane/stop/cross F1 `0.5279 / 0.4079 / 0.5854`, stop-line TP/FP/FN `103 / 131 / 168`로 row-scan/PCA integration보다 낫지 않다. Crosswalk head까지 transplant한 merge도 exact crosswalk가 `0.5535`로 내려가므로 사용하지 않는다.
- stop-line candidate-assignment loss는 기존 center/selector top-k 후보를 GT endpoint segment에 직접 assign하고 true/false score, hard-negative rank, normalized geometry loss를 주는 opt-in loss로 시험했지만 exact val128에서 실패했다. Best objective는 epoch1 `0.5939`이고, epoch2 lane/stop/cross F1은 `0.5555 / 0.0571 / 0.5818`, stop-line TP/FP/FN은 `2 / 8 / 58`이다. Runtime은 `skipped_steps=0`로 깨끗했으므로 구현 실패가 아니라 valid negative evidence다. 같은 center/selector top-k 후보 위에 assignment-loss만 얹는 방식은 0.6 path가 아니다.
- endpoint-delta target/readout short run은 val128 epoch1/2 stop-line F1이 모두 `0.0000`이고 best objective도 `0.5880863169`라 기준선보다 낮았다. 새 dense geometry channel을 바로 decode에 쓰는 형태도 현재는 0.6 path가 아니다.
- heatmap-support geometry target fill은 center heatmap support 전체에 offset/angle/half-length target을 채우는 opt-in target 계약을 시험했지만 exact val128 epoch2 stop-line F1이 `0.2338`로 무너졌다. dense stop-line mask F1도 `0.4854`, center heatmap F1도 `0.0815`로 기준 `0.6118 / 0.1396`보다 낮아 같은 형태로 반복하지 않는다.
- row-center auxiliary는 row selector에 centerline-row pressure를 추가해 predicted center/proposal reliability를 올리는지 봤다. exact val128 epoch2 objective `0.6060`, lane/stop/cross F1 `0.5265 / 0.4248 / 0.5818`로 기준 exact `0.6089`, `0.5267 / 0.4483 / 0.5854`보다 낮아서 row-center-aux-only도 0.6 path가 아니다.
- read-only component/readout audit은 broader-val512 GT 271개 중 production TP `98`, anchorless component fit close `123`, anchored fit close `119`를 보였다. GT tube의 mask/center signal은 각각 `223/271`, `220/271`에서 `>=0.50`로 남아 있지만, production FN 173개 중 anchorless fit으로 새로 40px 안에 들어오는 것은 34개뿐이다. 따라서 단순 anchor swap/no-anchor PCA만으로는 0.6 path가 아니다.
- git branch history의 older stop-line oracle/repair 계열도 같이 보면, GT target mask oracle은 stop-line F1 `0.8228`까지 가능했고 GT-overlap oracle은 `component_fit_or_endpoint_error=97` 중 `50`개를 40px 안으로 복구했다. 하지만 그 뒤 non-oracle core-row trim, high-confidence cleanup/PCA endpoint trim, split fit은 각각 기준 PCA reference를 넘지 못했다. oracle headroom은 실제지만, 단순 row-band/core-row, high-confidence subcomponent cleanup, PCA endpoint quantile trim, component split fitting만으로는 회수되지 않는다.
- fit-far visual audit은 production FN, GT tube mask/center `>=0.50`, no-anchor distance `>40px` bucket 상위 18개를 렌더링했다. 18개 모두 production stop-line은 1개씩 있고, 14개는 component_count도 1이라 "아예 안 나옴"보다 single connected component 안에서 wrong line segment를 읽는 문제가 강하다.
- component-conditioned local extraction probe는 predicted component 안에서 center/selector/fused score로 local support를 골라 다시 fit했지만 exact val128 stop-line F1이 기준 `0.4483`을 넘지 못했다. best replacement는 `0.4464`, append-top2는 TP를 늘리는 대신 FP가 크게 늘어 best `0.4054`였다.
- component-split readout probe는 component 내부 multi-line 후보를 pair/Hough-like split으로 만들었지만 exact val128에서 기준을 크게 밑돌았다. best append-top2는 stop-line F1 `0.3421`, TP/FP/FN `26 / 66 / 34`로 baseline TP를 유지하는 대신 FP를 크게 늘렸고, replacement 계열은 TP를 잃어 `0.2857` 이하로 떨어졌다.
- component proposal readout probe는 predicted mask component마다 center/selector/max proposal 하나를 고르고 predicted angle + mask extent로 segment를 읽는 decode-only contract를 시험했다. exact val128 best는 selector/max component proposal stop-line F1 `0.4308`, TP/FP/FN `28 / 42 / 32`로 baseline `0.4483`, TP/FP/FN `26 / 30 / 34`보다 낮다. recall은 `+2 TP`지만 FP가 `+12`라 PCA val128 reference `0.5133`과 angle-mask production `0.5085`에 못 미쳤다.
- mask-ridge readout probe는 center/selector top-k를 쓰지 않고 predicted mask component의 distance-transform ridge에서 center/axis를 읽는 decode-only contract를 시험했다. exact val128 best `ridge_horiz_component_band3_mask080`는 stop-line F1 `0.4538`, TP/FP/FN `27 / 32 / 33`으로 baseline `0.4483`보다 `+0.0055`뿐이고 PCA val128 `0.5133`, angle-mask production `0.5085`보다 낮아 broader-val512로 확장하지 않는다.
- line-support readout probe는 top-k center/selector proposal을 predicted angle + mask extent로 segment화한 뒤 segment 위의 mask/center/selector support를 다시 점수화했다. exact val128 best는 `max_top20_line_fused_gate_m045` stop-line F1 `0.4651`, TP/FP/FN `30 / 39 / 30`으로 baseline `0.4483`, `26 / 30 / 34`보다 높지만 PCA val128 `0.5133`, angle-mask production `0.5085`, task-head merge `0.4918`보다 낮다. line-support reranking만으로는 broader-val512로 확장하지 않는다.
- candidate consensus readout은 top-k 후보끼리 가까운 segment agreement가 high-score isolated FP를 걸러내는지 봤지만 exact val128 best consensus `max_top20_consensus_d48_c3` stop-line F1은 `0.4706`, TP/FP/FN `28 / 31 / 32`다. Baseline `0.4483`보다는 높지만 `max_top10_score_s080`/angle-mask production `0.5085`, PCA val128 `0.5133`, task-head merge `0.4918`보다 낮아 broader-val512로 확장하지 않는다. Oracle-positive selection은 stop-line F1 `0.7368`이지만 production signal이 아니다.
- lane-context readout은 `row_scan_tangent` lane prediction과 stop-line 후보의 교차/거리 feature로 isolated FP를 줄일 수 있는지 봤지만 exact val128 best lane-context `max_top10_lane_cross48_c1` stop-line F1은 `0.4696`, TP/FP/FN `27 / 28 / 33`이다. Baseline `0.4483`보다는 높지만 score-threshold `0.5085`, PCA `0.5133`, task-head merge `0.4918`보다 낮아 broader-val512로 확장하지 않는다.
- crosswalk-context readout은 predicted crosswalk polygon proximity가 stop-line FP를 줄이는지 봤지만 exact val128에서 더 나빴다. `max_top10_crosswalk_context` stop-line F1은 `0.3810`, TP/FP/FN `28 / 59 / 32`이고 `max_top20_crosswalk_near48_c1`은 `0.2105`, TP/FP/FN `10 / 25 / 50`이다. Predicted roadmark context만으로는 candidate selector가 되지 않아 broader-val512로 확장하지 않는다.
- delayed candidate-validator auxiliary는 dense validator map을 바로 production gate로 쓰지 않고 center gate를 유지한 채 hard-negative aux만 학습한 뒤 validator map mixing을 replay했지만 exact val128 gate를 넘지 못했다. Training best epoch2는 lane/stop/cross F1 `0.5546 / 0.4425 / 0.5854`, stop-line TP/FP/FN `25 / 28 / 35`이고, same-checkpoint replay best `product_validator_top3_s040_mask050_band4_fallback`도 stop-line F1 `0.4793`, TP/FP/FN `29 / 32 / 31`로 angle-mask `0.5085`, PCA `0.5133`, task-head merge `0.4918`보다 낮다.
- model-side presence emit gate는 sample-level `stop_line_presence_logits`와 postprocess emission threshold를 학습/적용했지만 exact val128 stop-line F1을 `0.4112`, TP/FP/FN `22 / 25 / 38`로 낮췄다. FP는 조금 줄었지만 TP가 더 줄어 tangent-link exact `0.4483`, PCA/angle-mask references보다 낮아 broader-val512로 확장하지 않는다.
- proposal-stat emit gate는 stop-line mask/center/selector/row/x dense-map max/mean statistics를 `stop_line_presence_logits`에 더했지만 exact val128 stop-line F1이 `0.4074`, TP/FP/FN `22 / 26 / 38`로 더 낮아졌다. Dense proposal-map presence evidence도 TP를 회복하지 못하므로 broader-val512로 확장하지 않는다.
- 남은 stop-line 방향은 새 query row, endpoint-delta channel, local score window, pairwise component split, component별 proposal readout, mask-ridge readout-only, top-k line-support reranking, candidate agreement/consensus-only, lane-context readout-only, crosswalk-context readout-only, selector-map gate, row/x span proposal, simple rowx-band selector target, 단순 predicted center/selector threshold proposal, proposal-distribution KL loss-only, dense candidate-validator aux/map-mixing, sample-gate threshold/replay/rescue, presence-only/proposal-stat emit gate, mask-wide angle-field auxiliary, stopline_mask_angle_aux_weight-only가 아니다. stop-line을 계속한다면 current center/selector map을 섞는 보정이 아니라 후보 생성/readout contract 자체를 바꾸거나, lane instance stability로 돌아가 stop-line/crosswalk retention을 같이 보는 쪽으로 좁힌다.

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
- row-scan tangent-link vectorizer는 predicted `tangent_axis`를 row-cluster linking cost에만 쓰는 opt-in decode 축이다. exact val128 epoch2 objective는 `0.6187`, lane/stop/cross F1은 `0.5633 / 0.4483 / 0.5854`이고, broader-val512 objective는 `0.6027`, lane/stop/cross F1은 `0.5407 / 0.4083 / 0.5854`다. row-scan보다 lane continuity는 실제로 좋아졌지만 stop-line/crosswalk는 그대로라 all-task 0.6 success가 아니며, tangent-link-only cost sweep은 반복하지 않는다.
- row-scan tangent-link + stop-line `mask=0.80` integration exact replay는 objective `0.6171`, lane/stop/cross F1 `0.5633 / 0.4364 / 0.5854`로 tangent-link 단독보다 stop-line이 내려갔다. 이 결합은 broader로 확장하지 않는다.
- row-scan + stop-line PCA-threshold integration replay는 row-scan lane gain과 stop-line `mask=0.80`, `min_instance_score=0.94` override를 같이 적용했다. exact val128 lane/stop/cross F1은 `0.5522 / 0.4364 / 0.5854`, broader-val512는 `0.5279 / 0.4235 / 0.5854`이고 broader objective는 `0.6019`다. objective만 보면 0.6을 넘지만 task별 F1 목표와 PCA-only stop-line reference에는 미달이라 success/default가 아니다.
- row-scan geometry guard probe는 length/bottom/gap/dx/turn-angle guard를 exact val128 epoch2에서 비교했다. best lane F1은 `row_gap24_row_dx12`의 `0.5526`으로 기존 row-scan `0.5522` 대비 `+0.0004`뿐이고 FP가 `486 -> 512`로 늘었다. turn-angle guard는 FP를 줄였지만 TP를 더 잃어 best `0.5365`라 default 승격 path가 아니다.
- row-scan residual filter export는 broader-val512 lane TP/FP/FN `4153 / 2105 / 5324`를 남겼다. FN은 left `46.9%`, truncated bottom `<0.50` `27.9%`, aspect `>=3` `65.9%`에 몰리고, FP는 side `74.8%`와 right `40.6%` 비중이 높다.
- residual local separation loss는 left/truncated/high-aspect GT core를 올리고 주변 ring negative를 누르는 opt-in target/loss를 시험했다. exact val128 epoch2 lane F1은 `0.5476`으로 기준 `0.5267`보다 높았지만, stop-line F1은 `0.4310`으로 기준 `0.4483`보다 낮고 phase objective도 `0.6085`로 기준 `0.6089`보다 낮다. lane partial-positive일 뿐 채택/확장하지 않는다.
- BCE-focus calibration은 broader-val512 lane F1을 `0.5101 -> 0.5344`로 올렸지만 stop-line/crosswalk가 내려갔고, centerline-core pixel F1도 `0.5729 -> 0.5680`으로 낮아졌다. goal success가 아니라 lane-vectorized metric partial-positive다.
- BCE-focus + PCA stop-line decoder integration best는 broader-val512 lane/stop/cross F1 `0.5344 / 0.4583 / 0.5741`이고, stop-balance + PCA replay best도 `0.5372 / 0.4528 / 0.5812`에 그쳤다.
- lane instance evidence validator audit은 row-scan lane TP/FP를 map-local evidence로 어느 정도 분리했다. val128 held-out logistic row AUC/AP는 `0.8407 / 0.9312`지만, held-out task replay lane F1은 `0.5691 -> 0.5749`로 `+0.0058`뿐이고 TP를 `566 -> 537`로 잃었다. post-hoc instance evidence threshold는 0.6 path가 아니다.
- row-scan tangent-stability short run은 audit에서 신호가 있던 tangent alignment를 training-side로 당겨 봤지만 exact val128 gate를 통과하지 못했다. `tangent` loss를 `0.35 -> 1.0`으로 올린 best epoch2는 lane/stop/cross F1 `0.5555 / 0.4348 / 0.5854`, objective `0.6122`이고 dense lane centerline-core F1은 `0.5765`다. 기존 row-scan exact `0.5522 / 0.4483 / 0.5854`, objective `0.6144`, dense core `0.5729` 대비 lane/core gain은 작고 stop-line이 내려가므로 broader-val512로 확장하지 않는다.
- row-scan dynamic hard-negative margin loss는 current high-confidence predicted centerline 후보 중 GT support 밖 top-k만 margin 아래로 누르는 training-side FP suppression을 시험했다. exact val128 epoch2는 lane/stop/cross F1 `0.5556 / 0.4348 / 0.5854`, objective `0.6121`, lane TP/FP/FN `1104 / 480 / 1286`이다. prior row-scan exact objective `0.6144`와 stop-line `0.4483`보다 낮고 tangent-stability와 같은 stop-line regression을 보이므로 broader-val512로 확장하지 않는다.
- row-scan centerline focal short run은 centerline BCE/Dice 위에 focal 항을 추가했지만 exact val128 gate를 넘지 못했다. epoch2 objective는 `0.6134`, lane/stop/cross F1은 `0.5551 / 0.4522 / 0.5854`, lane TP/FP/FN `1103 / 481 / 1287`, stop-line TP/FP/FN `26 / 29 / 34`다. lane/stop-line task F1은 prior row-scan보다 조금 높지만 selection objective가 `0.6144`를 넘지 못해 broader-val512로 확장하지 않는다.
- row-scan risk-bucket sampler는 residual FN에서 보인 left/truncated/high-aspect lane을 `lane_risk` task-positive bucket으로 직접 노출했다. exact val128 epoch2 objective는 `0.5957`, lane/stop/cross F1은 `0.5494 / 0.3964 / 0.5478`, lane TP/FP/FN은 `1081 / 464 / 1309`다. lane FP는 줄었지만 recall이 안 올라가고 stop-line/crosswalk retention을 잃으므로 sampler-exposure-only 축은 0.6 path가 아니다.
- row-anchor recall loss는 visible GT anchor-row x 위치 주변 centerline logit을 직접 올리는 opt-in auxiliary로 row-scan predicted centerline evidence를 안정화하려 했지만 exact val128 gate를 넘지 못했다. epoch2 objective는 `0.6106`, lane/stop/cross F1은 `0.5512 / 0.4310 / 0.5854`, lane TP/FP/FN은 `1141 / 609 / 1249`다. Lane TP는 row-scan reference보다 늘었지만 FP가 더 커지고 stop-line이 내려가므로 broader-val512로 확장하지 않는다.
- row-anchor local contrast loss는 같은 anchor row의 GT-near band 밖 high-logit negative를 같이 누르며 positive-only FP 증가를 줄였지만 exact val128 gate를 넘지 못했다. epoch2 objective는 `0.6134`, lane/stop/cross F1은 `0.5559 / 0.4483 / 0.5854`, lane TP/FP/FN은 `1133 / 553 / 1257`이다. Positive-only 대비 FP는 줄었지만 prior row-scan objective `0.6144`보다 낮고 lane gain도 exact-only small gain이라 broader-val512로 확장하지 않는다.
- inter-lane gap margin loss는 같은 row에서 인접 GT lane 사이 gap을 negative로 눌러 row-scan over-link/side FP를 줄이는 training-side auxiliary를 시험했다. exact val128 epoch2 objective는 `0.6122`, lane/stop/cross F1은 `0.5560 / 0.4348 / 0.5854`, lane TP/FP/FN은 `1104 / 477 / 1286`, stop-line TP/FP/FN은 `25 / 30 / 35`다. Prior row-scan보다 lane F1은 조금 높지만 objective `0.6144`와 stop-line `0.4483`을 넘지 못해 broader-val512로 확장하지 않는다.
- segment continuity contrast loss는 visible GT lane anchor 사이 segment를 샘플링해 centerline positive를 당기고 양옆 normal-offset negative margin을 누르는 training-side continuity contract를 시험했다. exact val128 epoch2 objective는 `0.6127`, lane/stop/cross F1은 `0.5594 / 0.4348 / 0.5854`, lane TP/FP/FN은 `1142 / 551 / 1248`, stop-line TP/FP/FN은 `25 / 30 / 35`다. 최근 lane-loss 후보 중 lane F1은 가장 높지만 objective `0.6144`와 stop-line `0.4483`을 넘지 못해 broader-val512로 확장하지 않는다.
- row-scan-tangent balanced-retain은 stop-line/crosswalk loss weight를 같이 올려 tangent-link lane gain을 보존하면서 다른 roadmark를 회복하려 했지만 exact val128 gate를 넘지 못했다. epoch2 objective는 `0.6147`, lane/stop/cross F1은 `0.5607 / 0.4310 / 0.5854`, lane TP/FP/FN은 `1117 / 477 / 1273`, stop-line TP/FP/FN은 `25 / 31 / 35`다. Crosswalk task-best는 epoch1 `0.6748`이지만 최종 best objective와 stop-line은 tangent-link exact reference `0.6187`, `0.5633 / 0.4483 / 0.5854`보다 낮아 broader-val512로 확장하지 않는다.
- stop-line local-centerline selector target은 center/selector target을 GT center 주변 local centerline으로 좁혀 proposal map ranking을 개선하려 했지만 exact val128 gate를 넘지 못했다. epoch2 objective는 `0.6167`, lane/stop/cross F1은 `0.5607 / 0.4522 / 0.5854`, stop-line TP/FP/FN은 `26 / 29 / 34`다. Stop-line은 tangent-link exact `0.4483`보다 `+0.0039`뿐이고 objective/lane은 `0.6187`, `0.5633`보다 낮아 broader-val512로 확장하지 않는다.
- stop-line geometry-aware candidate validator는 current top-k 후보를 predicted endpoint geometry로 GT segment에 라벨링하는 opt-in validator loss를 시험했지만 gate를 넘지 못했다. exact val128 epoch2 objective는 `0.6182`, lane/stop/cross F1은 `0.5607 / 0.4655 / 0.5854`, stop-line TP/FP/FN은 `27 / 29 / 33`이다. Stop-line은 local-centerline selector보다 높지만 objective/lane은 tangent-link exact `0.6187`, `0.5633`보다 낮고, same-checkpoint replay에서 validator-map variants는 baseline `0.4655`보다 모두 낮아 broader-val512로 확장하지 않는다.
- stop-line mask-wide angle-field auxiliary는 existing `stop_line_angle` map을 stop-line mask support 전체에서 sign-invariant axis cosine loss로 supervised했지만 gate를 넘지 못했다. exact val128 epoch2 objective는 `0.6152`, lane/stop/cross F1은 `0.5606 / 0.4348 / 0.5854`, stop-line TP/FP/FN은 `25 / 30 / 35`이다. 이는 tangent-link `0.6187`, `0.5633 / 0.4483 / 0.5854`와 PCA/angle-mask/task-head-merge stop-line references보다 낮아 broader-val512로 확장하지 않는다.
- stop-line hard-negative sampler는 `stopline_negative` task-positive bucket으로 GT-negative lane-source samples를 직접 노출했지만 exact val128 epoch2 objective `0.6063`, lane/stop/cross F1 `0.5475 / 0.4561 / 0.5644`에 그쳤다. Stop-line은 tiny gain이지만 lane/crosswalk/objective retention을 잃으므로 broader-val512로 확장하지 않는다.
- stop-line candidate manifest failure-mode audit은 gap4/max candidate-bearing samples `369`개를 `GT-negative 165 / positive top-oracle 113 / positive misrank 29 / positive no-oracle 62`로 분해했다. Positive no-oracle bucket은 nearest candidate distance가 `40_80=28`, `gte80=34`라 selector-only로는 회복할 수 없고, midpoint proposal/candidate generation 문제가 남아 있다.
- learned stop-line fragment-to-center extent head는 high-confidence short fragment에서 full segment를 직접 복원하려 했지만 exact val128 stop-line F1이 epoch1/2 `0.0267 / 0.0519`로 무너졌다. Fragment decode를 끈 replay도 stop-line F1 `0.1233 / 0.2517`에 그쳐 기존 기준을 회복하지 못한다. Target/head wiring은 안정적이지만 production readout으로 확장할 신호가 없어 val512로 넓히지 않는다.
- stop-line center-rank margin은 GT midpoint proposal logit을 hard-negative proposal보다 높이는 opt-in loss로 no-oracle bucket을 겨냥했지만 exact val128 epoch2 objective `0.6170`, lane/stop/cross F1 `0.5605 / 0.4602 / 0.5854`에 그쳤다. Stop-line은 tangent-link 대비 `+0.0119`지만 objective/lane은 current exact references보다 낮고 geometry-validator stop-line `0.4655`도 못 넘어 broader-val512로 확장하지 않는다.
- denser stop-line candidate-select contract는 gap4/top50 후보를 validator/presence로 직접 학습하고 `max_validator` readout을 썼지만 stop-line emit이 완전히 꺼졌다. exact val128 epoch2 objective는 `0.5855`, lane/stop/cross F1은 `0.5612 / 0.0000 / 0.5854`, stop-line TP/FP/FN은 `0 / 0 / 60`이다. Same-axis threshold/longer run이 아니라 emit/select contract 재설계 없이는 반복하지 않는다.
- stop-line geometry-validator + stop-line-head-only retention schedule은 trunk, detector/TL, lane, crosswalk heads를 고정하고 stop-line head만 업데이트했지만 stop-line을 회복하지 못했다. exact val128 epoch2 objective는 `0.6180`, lane/stop/cross F1은 `0.5640 / 0.4386 / 0.5926`, stop-line TP/FP/FN은 `25 / 29 / 35`다. Geometry-validator reference `0.4655`, tangent-link reference `0.4483`보다 stop-line이 낮아 broader-val512로 확장하지 않는다.
- row-scan tangent support gate는 centerline을 support로 대체하지 않고 low-support centerline pixels만 opt-in으로 제외하는 readout을 시험했지만 exact val128에서 tangent-link와 사실상 같았다. objective는 `0.6187189441`이고 lane/stop/cross F1은 `0.5633 / 0.4483 / 0.5854`, lane TP/FP/FN은 `1121 / 469 / 1269`다. Gain은 `+0.000002` objective 수준이라 broader-val512로 확장하지 않는다.
- row-scan tangent segment MIL은 GT lane segment 위 centerline evidence를 positive-only로 올리는 training-side instance contract를 시험했다. exact val128 epoch2 objective는 `0.6161803030`, lane/stop/cross F1은 `0.5655 / 0.4348 / 0.5854`, lane TP/FP/FN은 `1161 / 555 / 1229`다. Lane F1은 tangent-link reference보다 `+0.0022` 높지만 FP도 늘고 stop-line이 내려가 joint objective gate를 못 넘으므로 broader-val512로 확장하지 않는다.
- row-scan tangent segment MIL + lane-head-only retention schedule은 trunk, detector/TL, stop-line, crosswalk heads를 고정하고 lane head만 업데이트했다. exact val128 epoch2 objective는 `0.6193428422`, lane/stop/cross F1은 `0.5660 / 0.4483 / 0.5926`, lane TP/FP/FN은 `1162 / 554 / 1228`이다. Exact-only best를 만들었지만 tangent-link 대비 objective gain은 `+0.000626`로 작고 stop-line 목표는 전혀 닫히지 않아 broader-val512로 확장하지 않는다.
- row-scan segment-continuity contrast + lane-head-only retention schedule은 segment-continuity의 stop/cross corruption을 freeze로 막으면 lane gain이 남는지 봤지만 exact val128 gate를 넘지 못했다. Epoch2 objective는 `0.6157263716`, lane/stop/cross F1은 `0.5590 / 0.4483 / 0.5926`, lane TP/FP/FN은 `1141 / 551 / 1249`이다. Segment-continuity full-head `0.5594`보다 lane이 낮고, tangent-link `0.6187`, segment-MIL lane-head-only `0.6193` objective도 못 넘어 broader-val512로 확장하지 않는다.
- row-scan tangent row-distribution loss는 같은 row의 GT core centerline pixels를 column distribution으로 정규화해 centerline logits에 직접 row-wise pressure를 줬다. Exact val128 epoch2 objective는 `0.6174723013`, lane/stop/cross F1은 `0.5659 / 0.4483 / 0.5854`, lane TP/FP/FN은 `1127 / 466 / 1263`이다. Tangent-link 대비 lane은 `+0.0026`이고 stop/cross는 보존됐지만 objective는 `0.6187`보다 낮고 segment-MIL lane-head-only exact best `0.6193`, `0.5660 / 0.4483 / 0.5926`도 넘지 못해 broader-val512로 확장하지 않는다.
- row-scan tangent segment-MIL + row-distribution 조합은 segment-level positive evidence와 row-wise distribution pressure를 같은 lane-head-only retention schedule에 같이 얹어 봤지만 exact val128 gate를 넘지 못했다. Epoch2 objective는 `0.6186643159`, lane/stop/cross F1은 `0.5621 / 0.4483 / 0.5926`, lane TP/FP/FN은 `1152 / 557 / 1238`이다. Segment-MIL lane-head-only exact best `0.6193`, lane `0.5660`보다 낮고 row-distribution-only lane `0.5659`도 못 넘어 broader-val512로 확장하지 않는다.
- 다음 lane 축은 row-scan/tangent-link 후처리 cost sweep, threshold integration, residual-risk local loss weight, post-hoc row-scan evidence threshold, tangent-loss-only 강화, dynamic hard-negative margin-only, centerline-focal-only, risk-bucket sampler-only, row-anchor-positive-only, row-anchor-contrast-only, inter-lane gap margin-only, segment-continuity-contrast-only, segment-continuity + lane-head-only retention, retention-balance loss-weight-only, segment-MIL-positive-only, row-distribution-only, segment-MIL + row-distribution 조합, lane-head-only retention schedule을 반복하는 방향이 아니라, predicted centerline evidence를 더 명시적인 instance 단위 contract로 안정화하거나 tangent-link lane partial-positive를 stop-line/crosswalk 목표와 같이 끌어올리는 contract로 좁힌다. stop-line을 재개한다면 current center/selector top-k 후보 위 validator/assignment/candidate-select loss, gap/top-k-only 후보 pool 확장, stop-line-head-only freeze schedule, lane/crosswalk context-only filtering이 아니라 emit을 죽이지 않는 새로운 selector/readout contract로 제한한다.
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
