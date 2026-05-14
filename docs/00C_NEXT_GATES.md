# 00C. Next Gates

> 이 문서는 다음 실행 순서와 금지사항을 관리한다.
> 새 실험을 끝내면 `00B_STATUS_HISTORY.md`에 결과를 추가하고, 이 문서의 gate를 갱신한다.

## 1. 지금 하지 말 것

- 60% 돌파를 "raw model solved"로 표현하지 않는다.
- `phase_objective=0.6089`를 F1 0.6 달성으로 표현하지 않는다.
- weak-positive 후처리 조합의 broader objective `0.6086`을 task별 F1 0.6 success로 표현하지 않는다.
- final geometry filters를 broader val 없이 deployment default로 승격하지 않는다.
- lane-family 개선을 또 긴 same-axis run 하나로 확인하려고 하지 않는다.
- GradScaler health gate 없이 PV26 long-run AMP default를 되살리지 않는다.
- run artifact를 몇 GB씩 남기는 방식으로 실험하지 않는다. 핵심 checkpoint, exact eval summary, 비교 grid만 남긴다.
- traffic light 상태를 lane60 결과로 판단하지 않는다.
- stop-line micro target/loss/sampler 축을 같은 형태로 반복하지 않는다.
- `stopline_negative` hard-negative sampler-only를 stop-line production fix로 weight/epoch sweep하지 않는다.
- support map을 lane centerline 대체물처럼 쓰는 실험을 반복하지 않는다.
- support map으로 centerline component를 단순 bridge/closing하는 후처리만 반복하지 않는다.
- row-scan tangent support gate를 support-threshold sweep으로 반복하지 않는다.
- learned query-vector proposal-only를 objective만 보고 확장하지 않는다.
- selector-map component gate를 stop-line readout fix로 반복하지 않는다.
- endpoint-delta channel 추가 + direct decode를 같은 형태로 반복하지 않는다.
- heatmap-support geometry target fill을 stop-line reliability fix로 반복하지 않는다.
- no-anchor PCA 또는 current-anchor swap만 stop-line readout fix로 반복하지 않는다.
- component 내부 high-score point-pair split만 stop-line readout fix로 반복하지 않는다.
- GT-overlap oracle headroom을 production decode 후보로 착각하지 않는다. row-band/core-row trim, high-confidence subcomponent cleanup, PCA endpoint quantile trim, component split fitting은 같은 family로 반복하지 않는다.
- score/length-only candidate filter를 stop-line production fix로 반복하지 않는다.
- oracle-positive candidate selection을 production decode 성공으로 표현하지 않는다.
- hard-negative proposal ranking loss-only를 stop-line production fix로 longer run 확장하지 않는다.
- candidate-row scalar classifier AUC/AP를 stop-line task success로 표현하지 않는다.
- rich candidate-row classifier/window-feature threshold를 stop-line task success나 production decoder로 표현하지 않는다.
- rich candidate-validator held-out threshold replay를 stop-line production decoder나 gate success로 표현하지 않는다.
- gap4/top50 rich-selector held-out threshold replay를 stop-line production decoder나 gate success로 표현하지 않는다.
- stop-line selector feature-patch offline validator replay를 model-side candidate validator success로 표현하지 않는다.
- stop-line component-topology rich-validator replay를 production decoder나 model-side validator success로 표현하지 않는다.
- gap4/top50 positive-sample candidate-rank hit rate를 stop-line task F1이나 production decoder evidence로 표현하지 않는다.
- gap4/top50 sample-level emit-gate surrogate F1을 stop-line task F1이나 production decoder evidence로 표현하지 않는다.
- gap4/top50 sample-tree/nonlinear gate surrogate를 stop-line task F1이나 production decoder evidence로 표현하지 않는다.
- GT sample-gate oracle-only replay를 stop-line production decoder나 0.6 path로 표현하지 않는다.
- gap4/top50 sample-gate actual task replay의 train-only gain을 stop-line production fix로 표현하거나 threshold sweep으로 반복하지 않는다.
- gap4/top50 sample-gate 후보를 baseline에 fallback/append하는 rescue replay를 stop-line production fix로 반복하지 않는다.
- model-side stop-line presence/emit gate를 presence-only loss/threshold sweep으로 반복하지 않는다.
- model-side stop-line proposal-stat emit gate를 dense-stat presence/head/threshold sweep으로 반복하지 않는다.
- cold candidate validator gate 또는 현재 top-k hard-negative auxiliary 조합을 stop-line production fix로 longer run 확장하지 않는다.
- candidate-validator warm-bias-only를 direct dense gate rescue로 exact/broader run 확장하지 않는다.
- calibrated validator map-mixing replay를 stop-line production fix로 longer run 확장하지 않는다.
- delayed dense candidate-validator auxiliary와 validator map-mixing을 stop-line production fix로 longer run 확장하지 않는다.
- proposal-distribution KL loss-only를 stop-line production fix로 longer run 확장하지 않는다.
- exact objective `0.6196`만 보고 task-head merge replay를 lane-family 0.6 success로 표현하지 않는다.
- exact objective `0.6248`만 보고 segment-MIL lane + proposal-rank stop-line task-head merge를 lane-family 0.6 success로 표현하지 않는다.
- segment-MIL lane-head-only checkpoint에 flip-centerline TTA를 얹은 exact objective `0.6298`만 보고 current broader best로 승격하지 않는다.
- broader FP suppression premise 없이 exact task-best head transplant만 반복하지 않는다.
- crosswalk retention 근거 없이 focal-cross task head를 transplant하지 않는다.
- current center/selector top-k 후보 위의 candidate-assignment loss-only를 stop-line production fix로 longer run 확장하지 않는다.
- current center/selector top-k 후보 위의 geometry-aware candidate-validator loss-only를 stop-line production fix로 longer run 확장하지 않는다.
- stop-line positive no-oracle samples를 score/rank threshold 문제로만 해석하지 않는다. enriched manifest 기준 top score는 포화됐고 length ratio/center distance가 병목이다.
- stop-line center-rank margin-only를 `weight/topk/margin` sweep으로 반복하지 않는다.
- stop-line center-rank checkpoint를 angle-anchored mask-extent replay로 broader 확장하지 않는다.
- stop-line center-rank checkpoint의 proposal-recall top3 소폭 개선을 candidate-generation recovery로 표현하지 않는다.
- denser gap4/top50 candidate-select loss와 presence/max-validator gate 조합을 stop-line production fix로 longer run 확장하지 않는다.
- candidate-select checkpoint에서 presence threshold만 낮추거나 끄는 rescue replay를 stop-line production fix로 반복하지 않는다.
- stopline local-x auxiliary-only를 weight/schedule sweep으로 반복하지 않는다. Same-val64 smoke에서 stop-line F1은 `0.2000 -> 0.2222`로 작게 올랐지만 lane F1 `0.5469 -> 0.5196`, objective `0.6050 -> 0.5901`로 내려갔다.
- stop-line center target을 `centerline` mode로 바꾸는 target-mode-only 축을 반복하지 않는다. Same-val64 smoke에서 stop-line F1 `0.2000 -> 0.1455`, lane F1 `0.5469 -> 0.5202`, objective `0.6050 -> 0.5812`로 내려갔다.
- stop-line-head-only freeze schedule로 geometry-validator loss를 LR/epoch sweep하지 않는다.
- side-band centerline BCE positive weighting만으로 lane 0.6 path를 다시 찾지 않는다.
- side-band centerline probability margin loss만으로 lane 0.6 path를 다시 찾지 않는다.
- side/truncated/near-vertical geometry-risk recall-only loss만으로 lane 0.6 path를 다시 찾지 않는다.
- side/truncated/near-vertical geometry-risk local Tversky loss만으로 lane 0.6 path를 다시 찾지 않는다.
- lane negative-pixel probability margin loss만으로 lane 0.6 path를 다시 찾지 않는다.
- lane endpoint coverage loss만으로 lane 0.6 path를 다시 찾지 않는다.
- support-conditioned lane endpoint extension을 endpoint support threshold/step/max-length sweep으로 반복하지 않는다. Val4에서 `8` lanes / `24` points를 실제로 움직였지만 lane TP/FP/FN은 `40 / 11 / 46`으로 base task-mask variant와 같았다.
- residual-risk core/ring local separation loss weight만 키워 lane 0.6 path를 다시 찾지 않는다.
- row-scan lane instance evidence AUC/AP 또는 post-hoc threshold replay를 lane task success로 표현하지 않는다.
- logistic/tangent-alignment 같은 post-hoc row-scan evidence threshold를 production filter로 반복하지 않는다.
- row-scan dynamic hard-negative margin-only를 lane FP suppression fix로 longer run 확장하지 않는다.
- row-scan centerline-focal-only를 lane calibration fix로 longer run 확장하지 않는다.
- row-scan residual-risk bucket sampler-only를 lane recall fix로 longer run 확장하지 않는다.
- row-scan row-anchor positive recall loss-only를 lane evidence stability fix로 longer run 확장하지 않는다.
- row-scan row-anchor local contrast를 gap/top-k/weight sweep만으로 반복하지 않는다.
- row-scan inter-lane gap margin을 min-gap/top-k/weight sweep만으로 반복하지 않는다.
- row-scan segment-continuity + lane-head-only retention을 LR/epoch/weight sweep만으로 반복하지 않는다.
- row-scan tangent component-limited readout을 lane over-link fix로 반복하지 않는다.
- row-scan tangent train-augmentation-off를 lane/stop-line/crosswalk retention fix로 반복하지 않는다.
- row-scan GT tangent-axis replacement나 tangent-link cost/oracle sweep을 lane 0.6 path로 반복하지 않는다.
- row-scan tangent centerline snap을 snap-radius/window sweep으로 반복하지 않는다.
- lane centerline instance-balanced positive loss를 weight sweep으로 반복하지 않는다.
- lane soft-instance centerline shell auxiliary를 weight/radius/sigma sweep만으로 반복하지 않는다.
- lane soft-skeleton/clDice-style topology loss를 weight/iteration sweep만으로 반복하지 않는다.
- lane dense instance-embedding row-link/loss를 embedding distance/weight sweep만으로 반복하지 않는다.
- lane global/sample centerline threshold-only calibration을 lane 0.6 path로 반복하지 않는다.
- lane flip-consistency regularizer를 weight/mask-mode sweep만으로 반복하지 않는다.
- `core_centerline_refine_row_scan_tangent_positive_flip_consistency` restricts the same flip-consistency pressure to GT lane centerline-core positives, but exact val128 still misses the gate: objective `0.5981680601`, lane/stop/cross `0.5513 / 0.3966 / 0.5548`, skipped steps `0`.
- flip-centerline TTA의 max/union/threshold sweep만으로 lane 0.6 path를 찾지 않는다.
- normal/flip lane instance evidence를 같은 post-hoc threshold/logistic replay로 반복하지 않는다.
- lane oracle TP-only selector를 production lane success로 표현하지 않는다.
- lane row-scan same-schema duplicate suppression을 distance-threshold sweep으로 반복하지 않는다.
- lane row-scan per-sample top-k cap을 count/cap sweep으로 반복하지 않는다.
- lane logistic instance gate에 per-sample top-K safety fallback을 붙이는 K/threshold sweep을 반복하지 않는다.
- lane instance-validator logits의 threshold/weight/logistic replay를 같은 checkpoint에서 반복하지 않는다.
- lane instance-validator candidate score-gate를 같은 checkpoint에서 threshold/statistic sweep으로 반복하지 않는다.
- lane vectorizer semantic vote mode를 class/type vote weighting sweep으로 반복하지 않는다. Current flip-centerline composite에서 `component`, `centerline`, `centerline_excess`, `component_core` 모두 exact val128 lane TP/FP/FN/F1 `1200 / 510 / 1190 / 0.5854`로 동일했다.
- lane task-mask context gate is a fixed partial-positive, not a new sweep family. `flip_centerline_avg_lane_cross_comp050` improves broader val512 lane F1 `0.5577 -> 0.5628` while preserving stop-line/crosswalk, but lane still needs `+0.0372`; do not repeat it as source/strength/mask-threshold tuning.
- flip-centerline average 위의 post-hoc row gate exact `0.6125`를 lane-family success로 표현하거나 같은 threshold replay를 반복하지 않는다.
- lane dual-checkpoint centerline averaging을 checkpoint/weight/threshold sweep으로 반복하지 않는다.
- lane support-conditioned endpoint extension is also closed at smoke. Branch `exp/lane-family-f1/lane-endpoint-support-extension-smoke` / commit `1cc9c5a` extended `8` lanes and added `24` endpoint points, improving lane mean point distance `13.3898 -> 12.5226`, but lane TP/FP/FN/F1 stayed exactly `40 / 11 / 46 / 0.5839` against the fixed task-mask reference. Do not broaden or repeat as endpoint support threshold, step, or max-length tuning without a new assignment-moving signal.
- lane centerline thinning before vectorization is closed as smoke-negative. Branch `exp/lane-family-f1/lane-centerline-thinning-smoke` / commit `6c3767b` applied fixed Zhang-Suen thinning after the task-mask lane gate, but lane TP/FP/FN/F1 regressed `40 / 11 / 46 / 0.5839 -> 35 / 14 / 51 / 0.5185` and mean point distance worsened `13.3898 -> 15.3421`. Do not broaden or repeat as thinning threshold, morphology, skeletonization-kernel, or ridge-width tuning without a new TP-preserving assignment signal.
- lane component-polyfit vectorization is closed as smoke-negative. Branch `exp/lane-family-f1/lane-component-polyfit-smoke` / commit `b8f8ec8` changed only the lane vectorizer readout after the fixed task-mask dense variant, but lane TP/FP/FN/F1 regressed `40 / 11 / 46 / 0.5839 -> 32 / 12 / 54 / 0.4923`. Do not broaden or repeat as polynomial degree, row-stride, component-size, or component-fit smoothing tuning without a new TP-preserving assignment signal.
- lane row-scan tangent soft-ridge readout을 `threshold/peak-distance` sweep으로 반복하지 않는다.
- lane row-scan tangent centerline translation을 translation-radius/offset sweep으로 반복하지 않는다.
- lane ranked local centerline snapping을 ranker/radius/local-snap sweep으로 반복하지 않는다.
- coherent/affine centerline-peak lane repair도 affine/local-snap/radius sweep으로 반복하지 않는다. Val4에서 geometry는 움직였지만 lane TP/FP/FN/F1이 `39 / 17 / 47 / 0.5493`으로 flat이었다.
- exported repairability ranker 위의 simple centerline translation을 top-K/radius/local-offset sweep으로 반복하지 않는다.
- lane residual repairability-ranker gate를 top-K/score-threshold/residual-threshold sweep으로 반복하지 않는다.
- lane area-rescue repairability-ranker gate를 top-K/score-threshold/min-area/min-centerline/max-per-sample sweep으로 반복하지 않는다.
- lane no-GT feature-space kNN residual templates를 k/top-K/feature-distance/weighting sweep으로 반복하지 않는다.
- row-scan tangent segment-MIL + row-distribution 조합을 같은 lane-head-only retention schedule에서 weight-only로 반복하지 않는다.
- lane bottom-anchor offset auxiliary-only를 weight/LR/epoch/freeze-policy sweep으로 반복하지 않는다.
- row-scan tangent upper-trunk unfreeze를 LR/schedule/capacity-only sweep으로 반복하지 않는다.
- lane-head transplant checkpoint에서 bbox area/aspect만 강화하는 geometry-filter sweep을 lane 0.6 path로 반복하지 않는다.
- lane raw-vectorizer drop audit을 blind bbox-area/aspect 완화 sweep으로 해석하지 않는다.
- lane guarded area-rescue와 center-score-gated area-rescue를 `max_per_sample`/min-area/min-centerline/bbox-filter sweep으로 반복하지 않는다.
- lane centerline-branch dilated context를 dilation/depth/gate-init sweep으로 반복하지 않는다.
- detached support-conditioned centerline refinement를 longer run/LR/gate sweep으로 반복하지 않는다.
- stop-line center-stem wiring cleanup을 stop-line rescue path로 반복하지 않는다.
- stop-line P4 coarse-context fusion을 random-fusion warm-start/longer-run/LR sweep으로 반복하지 않는다.
- stop-line zero-gated P4/P5 coarse-context residual을 gate-init/LR/longer-run/projector sweep으로 반복하지 않는다.
- stop-line mask-wide angle-field auxiliary 또는 `stopline_mask_angle_aux_weight`만으로 stop-line production fix를 반복하지 않는다.
- stop-line dense score/geometry flip TTA averaging을 같은 형태로 val128/broader 확장하지 않는다.
- stop-line scale dense TTA averaging을 scale-factor/score-map/geometry-map subset sweep으로 반복하지 않는다. Val4 smoke에서 score/geometry/all averaging 모두 stop-line TP/FP/FN `0 / 3 / 2`로 reference와 완전히 같았다.
- stop-line normal/flip decoded-candidate agreement를 agreement-distance/top-k/point-average sweep으로 반복하지 않는다.
- stop-line normal/flip decoded-candidate union을 flip-only/normal+flip top-k/component-count sweep으로 반복하지 않는다.
- exact val128 crosswalk threshold pass를 broader-val Gate 4 success로 표현하지 않는다.
- Gate 4 crosswalk isolation을 볼 때 lane threshold까지 같이 바꾼 top-objective variant를 먼저 broader default 후보로 삼지 않는다.
- `cross_mask=0.40`, `cross_area=32` stricter crosswalk component threshold를 broader-val success 없이 default/export 후보로 반복하지 않는다.
- broader crosswalk object/mask/component-area/polygon-area/aspect/top-k threshold sweep을 postprocess-only success path로 반복하지 않는다.
- `crosswalk_polygon_mode=hull` broader crosswalk pass를 all-task success나 deployment default로 표현하지 않는다.
- hull/aspect/threshold 조합 sweep을 같은 representation 안에서 반복하지 않는다.
- exact val128 `stopline_negative` sampler tiny gain을 broader expansion이나 stop-line 0.6 path로 표현하지 않는다.
- 단순 predicted center/selector/max proposal threshold + angle-mask extent readout을 stop-line production fix로 반복하지 않는다.
- predicted mask distance-transform ridge/medial readout-only를 stop-line production fix로 반복하지 않는다.
- top-k proposal을 line-support score로 rerank하는 readout만 stop-line production fix로 반복하지 않는다.
- predicted proposal에서 시작하는 gap-tolerant mask-strip extent readout을 `top_k/mask_threshold/max_gap/band` sweep으로 반복하지 않는다.
- learned dense fragment-to-center offset/extent head를 `aux_weight/top_k/min_score/epoch` sweep으로 반복하지 않는다.
- stop-line fragment seed-extension을 extender score sweep으로 반복하지 않는다.
- fragment-union length/component-length competition을 feature-rank 또는 single-score sweep으로 반복하지 않는다.
- fragment-union multi-instance를 top-K/angle/offset sweep으로 반복하지 않는다.
- fragment-union second-instance gate를 score/fragment-count/length-ratio sweep으로 반복하지 않는다.
- fragment-union projection-gap split을 gap/angle/offset sweep으로 반복하지 않는다.
- fragment-union projection competition을 length/min-score/rank-feature sweep으로 반복하지 않는다.
- fragment-union projection competition selector를 single-feature threshold/gate sweep으로 반복하지 않는다.
- fragment-union projection competition selector를 exported-feature logistic/post-hoc gate sweep으로 반복하지 않는다.
- fragment-union projection competition selector를 raw-image photometric/contrast gate sweep으로 반복하지 않는다.
- row/x projection consistency를 candidate-pool threshold/logistic selector sweep으로 반복하지 않는다.
- stop-line recovery-budget audit의 oracle TP budget을 production decoder success로 표현하지 않는다.
- stop-line positive-misrank selector-only recovery를 0.6 path로 반복하지 않는다.
- stop-line sample_id temporal adjacency를 frame-gap/sequence-prefix/score/temporal-smoothing sweep으로 반복하지 않는다.
- stop-line positive-no-oracle 후보를 current fragment midpoint 기준 min-length extension으로 회복하는 sweep을 반복하지 않는다.
- stop-line no-oracle 후보를 proposal-cell anchor shift만으로 회복하는 sweep을 반복하지 않는다.
- stop-line no-oracle 후보를 GT 근처 local top20 proposal cell 선택만으로 회복한다고 가정하지 않는다.
- stop-line no-oracle 후보를 local proposal anchor + length/extent repair만으로 회복하는 sweep을 반복하지 않는다.
- stop-line score-island weighted center를 radius/relative-threshold/min-score sweep으로 반복하지 않는다.
- stop-line normal-support recenter를 radius/step/support-weight/offset-penalty sweep으로 반복하지 않는다.
- stop-line raw-edge recenter를 contrast/length-weight/radius/step/offset-penalty sweep으로 반복하지 않는다.
- stop-line raw-image axis-stripe midpoint/extent readout을 top-k/confidence/side-band/smoothing/contrast-span/threshold sweep으로 반복하지 않는다.
- stop-line score-island linefit을 island-radius/relative-threshold/linefit-mode sweep으로 반복하지 않는다.
- stop-line axis-projection oracle을 production readout으로 쓰거나 fixed-minlen sweep만으로 반복하지 않는다.
- stop-line no-GT feature ridge/linear geometry correction을 alpha/feature-subset/score-threshold sweep으로 반복하지 않는다.
- stop-line predicted center-offset을 angle axis로 projection하는 readout을 top-k/threshold sweep으로 반복하지 않는다.
- stop-line same-axis support span을 top-k/min-score/member-count/angle/normal-threshold sweep으로 반복하지 않는다.
- stop-line axis-profile midpoint-symmetric extent를 top-k/proposal-threshold/normal-band sweep으로 반복하지 않는다.
- stop-line axis-profile score weighting을 center/selector/fused profile-source나 top-k/threshold/band sweep으로 반복하지 않는다.
- stop-line dense Hough/global line-vote readout을 angle-bin/rho-bin/normal-band/mask/proposal-threshold sweep으로 반복하지 않는다.
- candidate agreement/consensus-only를 stop-line production fix로 반복하지 않는다.
- lane-context readout-only를 stop-line production fix로 반복하지 않는다.
- predicted lane-crossing 기반 stop-line midpoint/extent reconstruction을 distance/margin/top-k sweep으로 반복하지 않는다.
- crosswalk-context readout-only를 stop-line production fix로 반복하지 않는다.
- stop-line fit-far visual audit를 task-mask competition strength/source sweep으로 반복하지 않는다. Crosswalk-only suppression gives only exact val128 `+1 TP`, and crosswalk+lane suppression collapses recall.
- geometry-labeled candidate-validator map replay를 stop-line production readout fix로 반복하지 않는다.
- live teacher-cache distill plumbing pass를 stop-line F1 개선이나 all-task gate progress로 표현하지 않는다. It only proves the opt-in runtime path can load a frozen teacher and attach cache during a real train step.
- same-checkpoint stop-line-only teacher-cache self-distill을 weight/epoch/batch-size/EMA sweep으로 반복하지 않는다. Standard exact-val128 replay regressed lane/stop-line/crosswalk against the source merged checkpoint.
- restored fragment-union/projection-split/projection-competition tooling을 축 재개 신호로 해석하지 않는다. It is only a reproducibility surface for the already-closed/readout-reference family.

## 2. Top-level goal: lane-family F1 0.6+

목표:

- 최종 목표를 `phase_objective`가 아니라 lane / stop-line / crosswalk F1 자체의 0.6+ 달성으로 둔다.
- 최소 기준은 broader validation에서 세 task 모두 F1 `>= 0.60`이다.
- 중간 기준으로 mean F1 `>= 0.60`을 볼 수는 있지만, stop-line이 낮은 상태에서 평균만 넘기는 것은 성공으로 보지 않는다.

현재 broader best composite by objective 기준:

| Task | Current F1 | Gap to 0.6 | 우선순위 |
| --- | ---: | ---: | --- |
| lane | `0.5628` | `+0.0372` | 2 |
| stop-line | `0.4235` | `+0.1765` | 1 |
| crosswalk | `0.6187` | pass | retention |

Known broader task-balance replay:

| Task | F1 | Gap to 0.6 | Source |
| --- | ---: | ---: | --- |
| lane | `0.5628` | `+0.0372` | flip-centerline + fixed crosswalk-mask lane gate |
| stop-line | `0.5164` | `+0.0836` | projection-competition replay |
| crosswalk | `0.6187` | pass | hull crosswalk |

해석:

- stop-line이 가장 큰 병목이다. 기존 stop-line reweight/retention은 일부 개선을 만들었지만 0.6까지는 멀다.
- Projection-competition stop-line replay can be combined with the current crosswalk-mask lane gate and hull crosswalk reference as an artifact-only lower bound: lane/stop-line/crosswalk F1 `0.5628 / 0.5164 / 0.6187`, mean/min F1 `0.5659 / 0.5164`. This improves task balance but still leaves lane and stop-line below `0.60`, so the next work is not another recombination of known partial positives.
- Projection replay tooling and candidate-pool manifest generation are restored for reproducibility. Regenerate the replay input with `tools/probe_pv26_stopline_candidate_pool.py --dataset-root <pv26_exhaustive_od_lane_dataset> --proposal-min-gap 4` when needed, but treat that as tooling only. Current `develop` can regenerate a non-empty exact-val128 manifest and projection-competition replay reaches exact stop-line F1 `0.5167`, TP/FP/FN `31 / 29 / 29`; this is not a new stop-line branch and does not reopen projection-competition or candidate-row feature sweeps.
- Candidate-pool generation can now add `--projection-competition-replay` to write the fixed projection-competition reference in the same run. Use it as a reproduction gate only; the exact val128 audit still shows top-oracle/misrank/no-oracle split `32 / 3 / 15` among GT-positive candidate-bearing samples, so the next stop-line branch must change candidate-generation/midpoint recovery or add a materially different FP-control signal.
- Lane composite replay tooling is also restored for reproducibility. Use `tools/probe_pv26_lane_flip_tta.py` with `core_centerline_refine_row_scan_tangent_link` and the retained merged checkpoint when a branch needs the current composite baseline, but treat this as a reproduction gate only. It does not reopen flip max/union, task-mask source/strength, smoothing, scale, shift, photometric, thinning, polyfit, or other closed lane TTA/vectorizer sweeps.
- projection-competition reference 기준 stop-line은 F1 `0.5164`, TP/FP/FN `126 / 91 / 145`다. 현재 FP를 유지하면 F1 `0.60`에 `+30` TP가 필요하고, all positive-misrank `29`개를 oracle로 모두 회복해도 F1 `0.5996`으로 모자란다. 따라서 selector-only path는 FP 제거를 같이 하거나 no-oracle positive를 최소 1개 이상 회복해야 한다.
- score-island midpoint and score-island linefit readouts are both closed. Weighted midpoint reached only exact val128 F1 `0.4354`, and local score-island center+axis linefit reached only `0.4110` against the existing selector-center reference `0.5085`. The next stop-line branch needs a different candidate-generation/readout contract with explicit FP control, not another local-island radius or threshold sweep.
- stop-line no-oracle axis-offset budget shows the remaining local-candidate geometry error is mostly along the stop-line axis: `49 / 51` selected local positive-no-oracle rows are axis-dominant, abs-normal-offset q50/q90 is only `2.31px / 13.47px`, while abs-along-offset q50/q90 is `68.59px / 137.64px`. Axis-projection + GT-length oracle reaches stop-line F1 `0.6559`, TP/FP/FN `162 / 61 / 109`, almost matching full GT-midpoint+GT-length oracle `0.6599`; fixed minlen after axis projection peaks at only `0.5547`, TP/FP/FN `137 / 86 / 134`. Therefore the next production stop-line branch must infer both along-axis midpoint shift and extent/length from no-GT support; do not repeat axis oracle as production or reduce it to a min-length sweep.
- stop-line axis-projected predicted-offset readout is also closed as flat. Exact val128 `axisproj_selector_top1_s060_mask050_band4` matches the existing pred-offset selector reference exactly on stop-line F1 and TP/FP/FN: `0.5085`, `30 / 28 / 30`; mean point distance only moves `13.62 -> 13.51`. Existing center-offset predictions do not contain enough extra no-GT along-axis recovery signal under this readout.
- stop-line axis-profile proposal-cell readout is closed as a tie, not a new path. Exact val128 best `axis_profile_cell_top1_s060_mask050_band4` and `axis_profile_offset_top1_s060_mask050_band4` both reach stop-line F1 `0.5085`, TP/FP/FN `30 / 28 / 30`, matching the existing predicted angle/mask-extent and axis-projected-offset references but not beating PCA val128 `0.5133` or broader projection-competition `0.5164`. Do not broaden or repeat it as a proposal-source/top-k/mask-threshold/normal-band sweep.
- stop-line symmetric axis-profile readout is closed as negative, not just flat. Exact val128 `axis_profile_sym_cell_top1_s060_mask050_band4` drops to stop-line F1 `0.2203`, TP/FP/FN `13 / 45 / 47`, while the existing profile top1 remains `0.5085`, `30 / 28 / 30`. Forcing the proposal cell to be the segment midpoint is not a no-GT center recovery path.
- stop-line axis-window recenter is also closed as negative. Branch `exp/lane-family-f1/stopline-axis-window-recenter` / commit `e4b6dd2` slid the candidate center along the predicted axis using mask/proposal line support, but exact val128 reached only stop-line F1 `0.4306`, TP/FP/FN `31 / 53 / 29`, below baseline `0.4483` and the selector reference `0.5085`. Do not repeat as axis-window radius, step, scoring length, top-K, proposal-threshold, or fallback tuning.
- Stop-line dense-Hough readout is also closed as negative. Branch `exp/lane-family-f1/stopline-dense-hough-readout` / commit `64c8439` globally voted over predicted dense mask/proposal/angle support without GT anchors, but exact val128 best Hough reached only stop-line F1 `0.4000`, TP/FP/FN `24 / 36 / 36`, below baseline `0.4483`, `26 / 30 / 34`, and the existing selector reference `0.5085`, `30 / 28 / 30`. Do not repeat this as angle-bin, rho-bin, normal-band, mask-threshold, proposal-threshold, or top-K tuning without a materially new FP-control/candidate-generation signal.
- Stop-line scale dense TTA is closed at smoke. Branch `exp/lane-family-f1/stopline-scale-dense-tta-smoke` / commit `26a7f7e` averaged resized-scale stop-line score maps, geometry maps, or both with scales `0.875` and `1.125`, but val4 stop-line TP/FP/FN stayed `0 / 3 / 2` for every scale variant, matching the current reference. Do not broaden or repeat as scale-factor/map-subset tuning without a new no-GT stop-line signal.
- Stop-line live teacher-cache distill is executable, but same-checkpoint self-distill is closed negative. Branch `exp/lane-family-f1/stopline-live-distill-smoke` added opt-in frozen-checkpoint teacher-cache plumbing and passed a real CUDA one-batch smoke with `status ok`, `successful_batches 1`, `loss_total_mean 17.4609`. Follow-up branch `exp/lane-family-f1/stopline-distill-short` tested stop-line-only self-distill and standard exact-val128 rejected it: source lane/stop/cross `0.5445 / 0.4483 / 0.5854`, distill best `0.4835 / 0.3774 / 0.5036`. Do not broaden or repeat as a simple distill weight/schedule sweep.
- stop-line axis score-profile weighting is also closed as an exact-only positive. Selector-profile top1 improved exact val128 to stop-line F1 `0.5254`, TP/FP/FN `31 / 27 / 29`, but broader val512 fell to `0.4303`, `105 / 112 / 166`; best broader profile variant was mask-profile top3 at only `0.4531`, `111 / 108 / 160`. Do not repeat this as center/selector/fused profile-source or axis-profile top-k/threshold/band tuning.
- `exp/lane-family-f1/stopline-fragment-axis-contract` is now closed as negative. Commit `6cc3f52` made fragment-center supervision axis-scalar and aligned fragment decode with axis projection; commit `ae1fbc2` added a metric-only low-disk probe path. The exact val128 metric-only run completed, but epoch2 lane/stop/cross F1 was only `0.5225 / 0.1905 / 0.5714`, objective `0.5522`, with skipped steps `0` and checkpoint paths `null`. Do not broaden or repeat this as an aux-weight/top-k/min-score/epoch sweep.
- `exp/lane-family-f1/stopline-temporal-context-audit` tested whether `sample_id` frame adjacency can act as a no-GT FP gate on the archived exact val128 detector-context candidate rows. It is closed as weak: gap `100` keeps only `32%` of positive samples and `28.57%` of no-oracle positives, while gap `10000` recovers positives only by keeping `70.27%` of negatives. Do not repeat as a frame-gap/sequence-prefix/score/temporal-smoothing sweep on the same sparse validation candidate rows.
- Same-axis support span is also closed. It improves local length ratio q50 from `0.544` to `1.013`, but midpoint distance q50 only moves `68.59px -> 61.92px` and q90 worsens to `215.68px`. Val512 replay stays below the projection-competition reference: best support-span F1 `0.5085`, TP/FP/FN `119 / 78 / 152`, versus reference `0.5164`, `126 / 91 / 145`. The next stop-line branch still needs a new no-GT centering signal, not member-count or span-threshold tuning.
- Raw-image axis-stripe midpoint/extent is also closed as negative. Exact val128 raw-stripe replay collapses stop-line F1 from baseline `0.4483`, TP/FP/FN `26 / 30 / 34`, to `0.0685`, `5 / 81 / 55`, with `0` raw-improved-to-positive rows and nearest-GT distance q50 worsening `36.18px -> 156.92px`. Do not repeat this as a raw-stripe top-k, confidence, side-band, smoothing, contrast-span, or threshold sweep.
- Stop-line flip-consensus readout is closed as an exact-only small positive. Candidate-level normal/flip agreement improves exact val128 stop-line F1 only `0.4483 -> 0.4667`, TP/FP/FN `26 / 30 / 34 -> 28 / 32 / 32`; point averaging improves mean distance `17.92 -> 14.35` but not enough matched lines. It stays below PCA val128 `0.5133`, angle-mask production `0.5085`, and broader projection-competition `0.5164`, so do not broaden or repeat as agreement-distance/top-k/point-average tuning without a new TP-preserving candidate-generation signal.
- Stop-line flip-union readout is also closed as a FP-heavy negative. Flip-only and normal+flip union variants add some TP but expand FP faster: `flip_only_top1` reaches only stop-line F1 `0.4306`, TP/FP/FN `31 / 53 / 29`, while `normal_flip_union_top2` raises TP to `37` but FP to `136`, dropping F1 to `0.3176`. This is not a TP-preserving candidate-generation signal, so do not broaden or repeat as flip-only/normal+flip top-k or component-count tuning without a new FP-control signal.
- Stop-line lane-crossing extent readout is closed as a TP-collapse negative. Exact val128 `max_top20_lane_extent48_c2` emits the same `27` stop-lines as `max_top20_lane_cross48_c2`, but stop-line TP/FP/FN/F1 moves from `15 / 12 / 45 / 0.3448` to `3 / 24 / 57 / 0.0690`; even baseline is much higher at `26 / 30 / 34 / 0.4483`. Do not repeat this as a lane-crossing distance, margin, or top-k sweep.
- Stop-line fit-far visual audit found retained dense signal but no production-ready readout premise. Val128 has `51 / 60` GT stop-lines with tube mask max `>=0.50` and `50 / 60` with center max `>=0.50`; no-anchor fit is close for `34 / 60`, but task-mask competition only moves exact stop-line from `0.4483`, `26 / 30 / 34` to `0.4615`, `27 / 30 / 33` at best. Crosswalk+lane suppression collapses recall to `0.1429` or lower. Do not broaden or repeat as task-mask competition strength/source/mask-threshold sweeps.
- Stop-line no-GT feature geometry regression is also closed as an overfit premise. On candidate-bearing val128 rows, fixed `max_top10_score_s080` top1 selection had heldout stop-line F1 `0.5938`, TP/FP/FN `19 / 15 / 11`, but ridge-predicted midpoint/length correction fell to `0.0938`, `3 / 31 / 27`, despite train split improving to `0.6667`. Do not convert this train-only gain into a model-side readout or repeat as ridge-alpha/feature-subset/score-threshold tuning.
- Stop-line proposal-island midpoint is also closed as negative. A fixed max(center, selector) connected-island centroid readout on exact val128 moves stop-line TP/FP/FN/F1 from baseline `26 / 30 / 34 / 0.4483` to `30 / 46 / 30 / 0.4412`; fallback is identical. The island source improves matched geometry distance but pays too much FP, so do not repeat it as proposal-source, island-radius, relative-threshold, top-k, or fallback tuning without a new FP-control signal.
- Stop-line proposal-island baseline-absent replay is also closed as negative. Keeping baseline outputs when the baseline already emits a stop-line and using the same proposal-island readout only on baseline-absent samples lowered exact val128 stop-line F1 to `0.4265`, TP/FP/FN `29 / 47 / 31`, worse than both baseline and plain proposal-island. Do not repeat this as baseline-present/absent fallback tuning; a future stop-line branch needs a materially new FP-control/candidate-generation signal.
- Stop-line P4 context head is closed as insufficient architecture-only evidence. Branch `exp/lane-family-f1/stopline-p4-context-head` / commit `1cb0bd7` fused P2/P3/P4 in the stop-line head and passed unit wiring tests, but one-epoch same-evaluator val128 reached only lane/stop/cross F1 `0.4952 / 0.1980 / 0.6588`, objective `0.5642`, while the baseline evaluator row was `0.5267 / 0.0000 / 0.5854`, objective `0.5763`. It creates some stop-line TP but lowers objective and stays far below known stop-line references, so do not broaden it without a materially new initialization or FP-control premise.
- Stop-line zero-gated coarse context is also closed as negative. Branch `exp/lane-family-f1/stopline-zero-gated-coarse-context` / commit `b495721` kept the P2/P3 stop-line path intact and added P4/P5 through a near-zero residual gate, but one-epoch val128 reached only lane/stop/cross F1 `0.5073 / 0.1765 / 0.6628`, objective `0.5730`, with stop-line TP/FP/FN `9 / 38 / 46`. Safer initialization did not fix the missing no-GT midpoint/extent signal, so do not repeat it as a gate-init, LR, longer-run, projector, or delayed-unfreeze sweep.
- lane은 row-scan/tangent-link로 broader `0.5407`, segment-MIL lane-head-only composite로 `0.5480`, flip-centerline TTA로 `0.5577`, fixed crosswalk-mask lane gate로 `0.5628`까지 올랐지만 아직 0.6에는 부족하다. GT tangent-axis read-only oracle은 exact val128 lane F1을 `0.5633 -> 0.5611`로 올리지 못했고, GT centerline-core oracle은 `0.6778`까지 올렸다. Centerline threshold audit도 best global threshold `0.5641`, sample-wise dense-core oracle `0.5482`에 그쳐 scalar calibration-only가 0.6 path가 아님을 확인했다. Instance-balanced positive core, soft-shell, soft-skeleton topology, and dense instance-embedding row-link losses produced only tiny or negative exact movement with stop-line regression. 다음 lane 축은 tangent/link cost, threshold sweep, flip max/union sweep, task-mask source/strength sweep, centerline auxiliary weight/iteration sweep, or embedding-distance/weight sweep이 아니라 더 강한 predicted centerline evidence 또는 instance-level recovery다.
- lane dual-checkpoint centerline averaging is also closed at smoke. Averaging the current lane-head transplant with the original tangent-link checkpoint lowered val4 lane F1 versus `flip_centerline_avg`: `0.5899` -> `0.5672` for dual+flip and `0.5263` for dual-normal. The older checkpoint does not provide complementary TP recovery under fixed centerline-logit averaging.
- lane instance evidence broader val512 audit은 heldout baseline lane F1 `0.5473`에서 logistic gate `0.5660`까지 올렸고, full split-count로는 `0.5480 -> 0.5682`에 해당한다. 그러나 no-flip ablation도 heldout `0.5663`, full split-count `0.5675`로 거의 같아서 flip-specific gain은 없다. Flip-centerline average 위에 같은 gate를 얹으면 exact heldout lane F1은 `0.6125`까지 오르지만 broader heldout은 `0.5694`, full split-count는 `0.5738`에 그친다. 이것은 post-hoc row gate이고 recall을 줄이며 stop-line은 그대로 `0.4235` 수준이라 all-task success가 아니다. Follow-up oracle TP-only selector shows the same current candidate set could reach full lane F1 `0.6457` if all FP were removed while TP stayed fixed, so lane has selector headroom. Same-schema d24 duplicate suppression removes only one broader FP (`2206 -> 2205`), so the oracle gap is not a near-duplicate emission problem. Per-sample top5 logistic cap gives only a small full lane gain (`0.5577 -> 0.5618`) while losing TP (`4518 -> 4490`), so the oracle gap is not just sample over-emission count either. A recall-safety replay that always preserves each sample's top-K predictions before applying the logistic gate also fails: best heldout is `keep_topk=0`, full lane F1 `0.5705`, and every `keep_topk=1..8` is lower. Candidate-level validator score-gate on the learned validator checkpoint is also flat: exact val128 objective `0.6143914132`, lane/stop/cross `0.5529 / 0.4522 / 0.5854`, below tangent-link exact `0.6187165763`, `0.5633 / 0.4483 / 0.5854`. Fixed-distance endpoint extension is also closed: exact val128 best `top32` lowered lane F1 `0.5633 -> 0.5503` and moved TP/FP/FN `1121 / 469 / 1269 -> 1095 / 495 / 1295`. Positive-core-only flip consistency is also closed: exact val128 best objective `0.5981680601`, lane/stop/cross `0.5513 / 0.3966 / 0.5548`. Soft-skeleton topology loss is closed too: exact objective `0.6152102115`, lane/stop/cross `0.5609 / 0.4348 / 0.5854`, below tangent-link `0.6187165763`. Dense instance-embedding row-link is also closed as a standalone axis: exact objective `0.6150797781`, lane/stop/cross `0.5594 / 0.4348 / 0.5854`. 다음 lane 작업은 이 신호를 반복 threshold/dedupe/top-k cap/safety-gate/score-gate/endpoint-extension/flip-consistency/topology-loss/embedding-link sweep으로 쓰는 게 아니라 recall-preserving model-side/decoder-side instance-stability contract로 바꾸는 경우에만 진행한다.
- Lane anchor-offset instance auxiliary is closed as broader-val512 negative. Branch `exp/lane-family-f1/lane-anchor-offset-instance-head` / commit `edbd0dc` added an opt-in bottom-anchor offset target/head/loss, but the 2-epoch broader probe reached only phase objective `0.5921307966`, lane/stop/cross F1 `0.5097 / 0.4025 / 0.5741`, and mean/min F1 `0.4954 / 0.4025`. Do not repeat as anchor-offset weight, LR, epoch-count, or freeze-policy tuning.
- Lane legacy row-head fallback/union is closed at smoke. Branch `exp/lane-family-f1/lane-legacy-row-head-union-smoke-v2` / commit `a1e0771` showed `legacy_only` val4 lane TP/FP/FN `0 / 0 / 86`, while `baseline_legacy_union` and `flip_centerline_avg_legacy_union` were exactly identical to their references. Do not repeat as legacy row-head threshold, top-K, dedupe-distance, or union tuning.
- lane FN recovery audit on the current flip-centerline broader composite shows recall-side headroom: `1363 / 4959` missed GT lanes have `gt_center_point_mean >= 0.50`, `1698 / 4959` have an unmatched predicted lane within `120px`, and the diagnostic no-new-FP upper bound for `center_mean >=0.30 or unmatched <=120px` is lane F1 `0.7564`. This is read-only GT-labeled evidence, not production success. The next lane branch should target recall-preserving instance recovery from existing centerline evidence/nearby partial tracks, not another FP selector threshold.
- The lane FN joint-strata audit shows lane F1 `0.60` needs `489` recovered FN at current FP; `center>=0.50 and unmatched<=80px` has `563` FN (`0.6062` no-new-FP upper bound), while `center>=0.50 without unmatched<=120px` has `527` FN (`0.6032`) and `unmatched<=120px without center>=0.50` has `862` FN (`0.6306`). This means both track repair and centerline-only generation are large enough as GT-labeled buckets, but neither is production evidence by itself.
- Pair-geometry audit shows the strongest nearby-track bucket is mostly a position/center-offset problem, not an angle or length-ratio problem: `center>=0.50 and unmatched<=80px` has length ratio q50 `0.974`, angle q50 `1.27deg`, y-overlap q50 `0.905`, but center distance q50 `50.11px`, just beyond the `40px` match threshold. The center-only bucket is different: no near unmatched track, length ratio q50 `2.091`, center distance q50 `182.80px`, y-overlap q50 `0.332`, so it likely needs instance generation rather than current-track repair.
- Lateral-duplicate budget is read-only oracle planning evidence, not production success. It shows `unmatched<=120 any center` has `1698` recoverable FN and can tolerate up to `2821` added FP for lane F1 `0.60`; even duplicating all current FP while recovering that full bucket gives oracle-budget F1 `0.6184`. However the tighter `unmatched<=80 and center>=0.50` bucket can tolerate only `172` added FP. A next lane implementation may test one fixed duplicate-style smoke only if it reports TP recovery and added FP together; do not expand this into an offset/radius sweep.
- Lane FP-repair oracle is new read-only planning evidence, not production success. It shows the current broader-val512 unmatched-track bucket is even stronger if existing FP can be repaired into TP: `unmatched<=120 any center` maps `1698` FN rows to `1272` unique unmatched predictions and gives oracle lane F1 `0.7148`; the tight `unmatched<=80 and center>=0.50` bucket maps `563` rows to `533` unique predictions and still gives oracle F1 `0.6235`. The next lane branch may target a no-GT FP-to-TP repair or model-side instance-alignment contract, but must not repeat duplicate append, translation radius, residual append, or threshold gating as if this oracle were production evidence.
- Lane repairable-unmatched feature audit narrows the repair premise: among `2206` unmatched predictions, `526` are tight repair targets (`<=80 and center>=0.50`) and `1393` are broad repair targets (`<=120 any center`). The best single no-GT feature is `pred_polyline_length`, with tight-label AUC/AP `0.7205 / 0.4278` and broad-label AUC/AP `0.6614 / 0.7368`; `pred_center_point_mean` is similar but lower (`0.7106` tight AUC, `0.6295` broad AUC). Repairable tracks are longer and stronger on centerline/support, but tight-label precision at the positive-count cutoff is only `0.4563`, so do not convert this into a single-feature threshold gate.
- Lane repairability model replay is positive planning evidence, still not production success. A 2-fold out-of-fold no-GT logistic ranker over unmatched-track/context features gives tight-label AUC/AP `0.7629 / 0.4742`; tight top-750 replay nearly reaches lane `0.60` (`0.5985`, `330` repairs) and top-1000 reaches `0.6066` only with low precision `0.3960`. The broad label is more actionable: AUC/AP `0.6821 / 0.7491`, top-500 selects `405 / 500` repairable rows and gives oracle-repair lane F1 `0.6077`, while top-1000 gives `0.6528`. Commit `885948d` removes the GT-derived `sample_unmatched_pred_count` feature before exporting ranker parameters, so future readouts may use the parameter artifact as a no-GT scorer premise. The next lane branch may use this as a budgeted broad repair-ranker premise, but it must implement a real no-GT geometry/instance-alignment repair and report actual TP/FP/FN. Do not count the replay itself as success or repeat it as a threshold/top-K gate sweep.
- The first fixed-ranker geometry transfer smoke is also closed flat. `exp/lane-family-f1/lane-ranked-translate-repair-smoke` used the exported broad ranker and a val-size-scaled top-500/2048 budget, but the selected four val4 rows all had `dx=0`, `selected_moved_count=0`, and lane TP/FP/FN/F1 stayed `41 / 12 / 45 / 0.5899`. Do not repeat this as a ranker top-K, translation-radius, or local x-offset sweep; future repair work needs a materially different instance-alignment or instance-generation contract that actually moves geometry and then reports TP/FP/FN movement.
- The ranked local 2D snap follow-up also closes this centerline-snapping family. It did move all selected rows on val4 (`selected_moved_count=4`, moved points `76`), but lane TP/FP/FN/F1 stayed exactly `41 / 12 / 45 / 0.5899`. Geometry movement alone is not useful unless it crosses matching boundaries, so do not broaden or repeat as ranker/radius/local-snap tuning without a new TP/FP/FN movement signal.
- The ranked component-row projection follow-up is also closed flat. It selected and moved all `4` val4 repair rows (`35` moved points), but lane TP/FP/FN/F1 stayed exactly `41 / 12 / 45 / 0.5899`. Do not repeat component-row/path projection as a connected-component, row-projection, or path-following sweep without a new signal that first changes assignment metrics.
- `exp/lane-family-f1/lane-repair-geometry-export` added point JSON columns to the lane FN/unmatched repair audit, and `exp/lane-family-f1/lane-point-repair-replay` proved the actual metric mechanics with an oracle-only replay. Val128 oracle replacement of selected unmatched prediction points moved lane TP/FP/FN/F1 from `1200 / 510 / 1190 / 0.5854` to `1503 / 207 / 887 / 0.7332`, with `322 / 510` selected candidates, `304` unique targets, and `18` duplicate targets. This is not production success because it copies GT geometry and leaves stop-line at `0.4483`. The next lane branch must infer replacement geometry from no-GT signals and report actual TP/FP/FN; do not repeat this as another GT-point oracle, aggregate-distance claim, ranker top-K sweep, or centerline translation-radius sweep.
- `exp/lane-family-f1/lane-point-repair-regression-premise` tested the simplest no-GT geometry inference after the oracle replay: two-fold ridge prediction of one translation vector from prediction-side features on the `322` oracle-selected val128 candidates. It only raised 40px-close rows `24 -> 44`, while q50/q90 distance worsened `65.31 / 106.31 -> 66.89 / 110.11` and improved/worsened rows were split `159 / 163`. Do not convert this into live decoder integration or repeat it as l2/feature/threshold tuning without a materially new geometry signal.
- `exp/lane-family-f1/lane-point-repair-polyline-residual-premise` tested a stronger no-GT geometry premise: two-fold ridge prediction of full resampled lane polylines from prediction-side features and predicted track shape on the same `322` oracle-selected val128 candidates. Best `absolute_polyline_ridge` raised close rows `23 -> 56`, with distance q50/q90 `65.95 / 107.25 -> 65.25 / 105.88` and improved/worsened `172 / 150`. This is better than one-vector translation but still weak artifact-only evidence; do not integrate it as a live decoder or repeat it as ridge/output-point-count tuning until a new no-GT selection/alignment signal can move actual TP/FP/FN.
- `exp/lane-family-f1/lane-repairability-polyline-selection-premise` added the missing no-GT selection check before the polyline geometry premise. Applying the exported broad repairability scorer to val128 point-export rows and keeping the scaled top-116 gives broad-label precision `96 / 116 = 0.8276` and tight positives `60`, but the best held-out geometry variant only raises close rows `8 -> 20`; q50/q90 worsens from `65.06 / 148.49` to `77.60 / 183.02`. The scorer can find plausible repair targets, but current no-GT geometry reconstruction does not support live repair. Do not repeat this as top-k/ridge tuning; the next lane repair path needs a new alignment signal, not another scorer-plus-regressor replay.
- `exp/lane-family-f1/lane-knn-residual-premise` tested a non-parametric alternative to ridge/polyline regression: transfer geometry residual templates from `k=5` feature-space neighbors after the same top-116 broad repairability selection. It raised close rows only `8 -> 21`, while q50/q90 worsened from `65.06 / 148.49` to `75.17 / 212.43` and worsened rows outnumbered improved rows (`66 / 50`) for the best close-count variant. This is a weak offline premise, not live lane TP/FP/FN evidence. Do not repeat it as a k/top-K/feature-distance/weighting sweep.
- The fixed duplicate-style smoke is now closed negative: preserving the original row-scan-tangent track and adding a centerline-translated copy gives lane F1 `0.5594`, TP/FP/FN `40 / 17 / 46`, below both the row-scan-tangent smoke reference `0.5899`, `41 / 12 / 45` and replacement centerline translation `0.5674`, `40 / 15 / 46`. Do not broaden or repeat preserved translation duplicates as an offset/radius sweep.
- Residual centerline-component append is closed as val128 negative. It recovered `+20 TP` but added `+101 FP`, moving lane F1 `0.5739 -> 0.5669`, TP/FP/FN `1175 / 530 / 1215 -> 1195 / 631 / 1195`. Do not repeat this as residual centerline/support threshold, component-size, coverage-width, min-length, or per-sample cap sweeps without a new no-GT FP-control signal.
- Residual baseline-proximity/support gating is also closed as val128 negative. The fixed gate looked good on smoke (`0.5588 -> 0.5755`, TP/FP/FN `38 / 12 / 48 -> 40 / 13 / 46`) but exact val128 still fell below baseline (`0.5739 -> 0.5706`, `1175 / 530 / 1215 -> 1190 / 591 / 1200`). Do not repeat residual append as proximity, length, support, component, or per-sample threshold sweeps without a stronger new FP-control signal.
- Residual repairability-ranker gating is also closed as val128 negative. The exported broad repairability scorer selected only `27 / 121` residual candidates, but only `6` were matched TP rows and `21` were FP rows. Exact val128 lane F1 moved `0.5739 -> 0.5730`, TP/FP/FN `1175 / 530 / 1215 -> 1181 / 551 / 1209`; stop-line/crosswalk were unchanged at `0.4364 / 0.5854`. Do not repeat this as a top-K, score-threshold, or residual-candidate threshold sweep without a new no-GT FP-control/alignment signal.
- Residual repairability-gated replacement is also closed at smoke. Replacing the nearest baseline lane with the fixed selected residual row kept lane TP/FP/FN unchanged at `38 / 12 / 48`, F1 `0.5588`, even though the selected residual row was matched as a TP under append accounting. Do not broaden or repeat this as a nearest-distance, top-K, repairability-threshold, or residual-component sweep without a new no-GT alignment signal.
- The first production-like readout attempt from that evidence, `row_scan_tangent_soft_ridge` at `lane_obj_threshold=0.30`, failed val4 smoke: lane F1 moved `0.5899 -> 0.5429`, TP/FP/FN `41 / 12 / 45 -> 38 / 16 / 48`. Do not broaden it to val512 or repeat it as a peak/threshold sweep.
- The seeded low-threshold follow-up, `row_scan_tangent_hysteresis`, also failed val4 smoke: lane F1 `0.5652`, TP/FP/FN `39 / 13 / 47`. It reduces FP relative to the row-scan-tangent smoke but loses more TP, so do not repeat it as a low/high threshold or hysteresis-seed sweep without a new instance-level recall-preserving signal.
- The constrained topology-preserving follow-up, `row_scan_tangent_centerline_snap`, also failed val4 smoke: lane F1 moved `0.5899 -> 0.5674`, TP/FP/FN `41 / 12 / 45 -> 40 / 15 / 46`. Do not broaden it to val512 or repeat it as a snap-radius sweep unless a materially new non-GT FP-control signal is added.
- Row-scan-tangent global row assignment is also closed as negative. Replacing greedy per-cluster linking with per-row Hungarian track-to-cluster assignment fixes a synthetic conflict test but fails the real val4 smoke: lane F1 `0.5152`, TP/FP/FN `34 / 12 / 52`, versus row-scan-tangent reference `0.5899`, `41 / 12 / 45`; stop-line/crosswalk also stay unusable at `0.0000 / 0.4000`. Do not broaden or repeat this as a global/Hungarian row-assignment variant without a new recall-preserving signal.
- Single-scale centerline TTA is also closed at smoke. Averaging a scale-resized input pass back into only `lane_seg_centerline_logits` lost lane TP in both directions: scale `0.875` `flip_scale_centerline_avg` was `0.5606`, TP/FP/FN `37 / 9 / 49`, and scale `1.125` was `0.5778`, `39 / 10 / 47`, versus the existing `flip_centerline_avg` reference `0.5899`, `41 / 12 / 45`. Do not broaden or repeat as scale-factor/interpolation/scale-weight sweeps without a new recall-preserving signal.
- Fixed input-shift centerline TTA is also closed at smoke. Shifting the input by `+/-8px`, inverse-shifting only `lane_seg_centerline_logits`, and averaging with flip-centerline TTA lowered lane F1 to `0.5652`, TP/FP/FN `39 / 13 / 47`, versus the existing `flip_centerline_avg` reference `0.5899`, `41 / 12 / 45`. Do not broaden or repeat as shift-size, shift-direction, interpolation, or shift-weight sweeps without a new TP-preserving signal.
- Photometric centerline TTA is also closed at smoke. Brightness gains `0.90,1.10` averaged into only `lane_seg_centerline_logits` lowered `flip_photometric_centerline_avg` lane F1 to `0.5672`, TP/FP/FN `38 / 10 / 48`, versus the existing `flip_centerline_avg` reference `0.5899`, `41 / 12 / 45`. The lower FP is not useful because it loses three TP; do not broaden or repeat as brightness, contrast, gain, or averaging-weight sweeps without a new TP-preserving signal.
- Row-scan x-coordinate smoothing is also closed at smoke. Fixed-window smoothing of decoded `row_scan_tangent` track x positions lowered `flip_centerline_avg_smooth_x` lane F1 to `0.5038`, TP/FP/FN `33 / 12 / 53`, versus the existing `flip_centerline_avg` reference `0.5899`, `41 / 12 / 45`. Do not broaden or repeat as smoothing window, kernel, endpoint-preservation, or smoothing-weight sweeps without a new TP-preserving signal.
- The area-rescue center-score gate is also closed as weak partial/negative evidence. It improved exact val128 over ungated area rescue (`0.5716 -> 0.5846`) but broader val512 lane F1 stayed at `0.5548`, below the current broader best `0.5628`, with stop-line still `0.4083`. The stricter q10 follow-up is also closed at smoke: adding `lane_centerline_track_q10 >= 0.60` moved val4 lane F1 `0.5972 -> 0.5816`, TP/FP/FN `43 / 15 / 43 -> 41 / 14 / 45`. Do not repeat it as a `min_centerline`, q10/quantile, min-area, or max-per-sample sweep without a new recall-preserving signal beyond track statistics.
- Area-rescue repairability-ranker gating is also closed as effectively flat. A fixed exported broad repairability ranker gate looked plausible on smoke (`0.5899 -> 0.6000`, TP/FP/FN `41 / 12 / 45 -> 42 / 12 / 44`), but exact val128 only moved lane F1 `0.5854 -> 0.5860`, TP/FP/FN `1200 / 510 / 1190 -> 1214 / 539 / 1176`, while stop-line/crosswalk stayed `0.4364 / 0.5988`. The selected area-rescue rows were only `14` matched TP versus `29` FP, so do not repeat this as a repairability top-K, score threshold, min-area, min-centerline, or max-per-sample sweep without a new no-GT FP-control/alignment signal.
- The track-level uniform translation follow-up, `row_scan_tangent_centerline_translate`, failed the same val4 smoke gate: lane F1 `0.5674`, TP/FP/FN `40 / 15 / 46`. This closes simple center-offset repair as a radius/offset sweep; a future lane branch needs a different instance-generation or FP-control signal.
- Raw-vectorizer drop audit showed a non-repeated lane follow-up existed: in broader val512, `854 / 4913` FNs already had raw row-scan-tangent candidates within `40px` before geometry filtering, and `497` of those failed bbox-area. The guarded area-rescue follow-up is now closed negative: val128 lane F1 moved `0.5797 -> 0.5716`, TP/FP/FN `1204 / 560 / 1186 -> 1247 / 726 / 1143`. The next lane branch should not relax bbox filters again unless it first adds a materially new FP-control signal.
- crosswalk는 opt-in hull decode로 broader-val512 `0.6187`까지 올라 현재 gap은 닫혔다. 다음 stop-line/lane work에서는 이 crosswalk retention을 유지하는지 확인한다.

실험 원칙:

- 한 번에 한 축만 바꾼다.
- exact epoch-2 subset만으로 성공 판정하지 않는다.
- metric은 task F1, TP/FP/FN, support를 같이 본다.
- comparison grid는 metric 보조 증거로만 쓴다.
- run artifact는 `best.pt`, exact eval summary, comparison grid 정도만 남긴다.
- Git branch/worktree를 파서 실험할 때도 한 worktree는 한 가설만 소유한다.
- `/tmp` 안의 임시 산출물은 삭제 가능하다. 그 밖 경로에서는 삭제하지 않고 삭제후보 폴더로 이동만 허용한다.

## 3. Worktree experiment protocol

목적:

- F1 0.6+까지 가는 후보를 architecture / postprocess / preprocess-runtime 축으로 나누되, 서로의 결과가 섞이지 않게 한다.

Branch/worktree 규칙:

- branch 이름은 `exp/lane-family-f1/<axis>-<short-hypothesis>` 형식으로 쓴다.
- worktree 경로는 repo 밖 sibling 경로를 쓴다. 예: `<repo-sibling>/yolopv26-exp-stopline-decoder`.
- 한 worktree에서 동시에 두 축을 바꾸지 않는다.
- positive result만 merge 대상으로 본다. negative result도 `00B_STATUS_HISTORY.md`에 남겨 같은 가설을 반복하지 않는다.
- develop 승격 전에는 exact replay와 broader-val replay를 모두 통과해야 한다.

초기 실험 lane:

| Lane | 첫 질문 | 주요 파일 |
| --- | --- | --- |
| postprocess | final geometry filters가 broader-val에서도 precision gain을 유지하는가 | `model/engine/postprocess.py`, `tools/evaluate_pv26_lane60_checkpoint.py`, `tools/analyze_pv26_lane60_prediction_filters.py` |
| stop-line architecture | stop-line F1 병목이 decoder/target/loss 어느 쪽인가 | `model/net/stopline_head_line.py`, `model/engine/loss.py`, `model/engine/postprocess.py`, `tools/run_pv26_lane60_probe.py` |
| lane architecture | centerline recall 부족인지 vectorizer recovery 부족인지 분리 가능한가 | `model/net/lane_head_segfirst.py`, `model/engine/loss.py`, `tools/probe_pv26_lane60_dense_maps.py` |
| sampler/feeder | stop-line/crosswalk positive exposure와 validation support가 충분히 안정적인가 | `model/data/sampler.py`, `tools/pv26_train/cli.py`, `config/pv26_train_hyperparameters.yaml` |

첫 실행 순서:

1. current `develop`에서 Gate 1 broader-val replay를 기준선으로 확정한다.
2. 기준선 summary를 `00B_STATUS_HISTORY.md`에 추가한다.
3. 기준선 이후부터 worktree를 나눠 stop-line / lane / sampler-feeder 후보를 한 축씩 실험한다.

## 4. Gate 1: final geometry filters broader validation replay

목적:

- exact epoch-2 subset에서 `0.6088677363` objective와 lane/stop/cross F1 `0.5267 / 0.4483 / 0.5854`를 만든 geometry filters가 더 넓은 validation slice에서도 유효한지 확인한다.

현재 결과:

- val512 replay에서 objective는 `0.5943438312141003`이다.
- lane / stop-line / crosswalk F1은 `0.5101 / 0.4083 / 0.5854`다.
- support는 `9477 / 271 / 395`다.
- 결론은 partial success다. geometry-filter gain은 완전히 사라지지 않았지만, objective 0.6과 task별 F1 0.6 모두 아직 통과하지 못했다.

확인할 것:

- 같은 checkpoint와 same postprocess config로 larger validation slice objective/F1 replay.
- lane/stop/cross task별 TP/FP/FN 변화.
- small-fragment filter가 recall을 과하게 깎는 장면이 있는지 comparison grid 확인.

성공 기준:

- objective gain이 exact epoch-2에만 과적합된 현상이 아니어야 한다.
- lane recall 손실이 과도하면 filter threshold를 deployment default로 승격하지 않는다.
- F1 0.6+ plan의 baseline으로 쓸 broader-val lane/stop/cross F1을 확정한다.

Gate 상태:

- 기준선은 확정됐다: broader-val512 F1 `0.5101 / 0.4083 / 0.5854`.
- deployment/export default 승격은 보류한다.
- 다음 실행은 Gate 2 stop-line first다.

## 5. Gate 2: stop-line first improvement axis

목적:

- F1 0.6+ 목표의 최대 gap인 stop-line을 먼저 올린다.

현재 진단:

- `exp/lane-family-f1/stopline-diagnostics` worktree에서 val512 TP/FP/FN feature export를 완료했다.
- stop-line TP/FP/FN은 `98 / 111 / 173`이다.
- TP와 FP score median이 `0.9646 / 0.9609`로 겹쳐서 score threshold만으로 분리하기 어렵다.
- FN bbox area median은 `2275.0`으로 TP median `1364.4`보다 작지 않다. area/aspect filter 강화는 recall 손실 위험이 크다.
- stop-line mask pixel F1은 val128 probe에서 `0.6107`까지 나오지만, stop-line center heatmap F1은 `0.1385`에 그쳤다.
- decoder-only PCA component 후보는 broader-val512 stop-line F1을 `0.4699`까지 올렸지만 목표 `0.60`에는 아직 멀다.
- endpoint proposal, feature isolation, wider mask target, selector decode, sampler/loss 단일 축은 모두 0.6 path가 아니었다.
- component split, center-cell geometry mask, half-length scale/loss/log target, learned query-vector proposal-only도 0.6 path가 아니었다.
- proposal/readout oracle은 GT center/length reconstruction 기준 val128 stop-line F1 `1.0000`을 냈다. 따라서 evaluator representation보다 predicted center/proposal/readout contract가 병목이다.
- predicted half-length는 단순 scale 문제가 아니다. x128 scaling도 stop-line F1 `0.4500`에 그쳤고, log target은 exact val128 stop-line F1 `0.4211`로 후퇴했다.
- learned vector proposal short run은 vector-only exact val128 epoch1/2 stop-line F1이 모두 `0.0000`이었다. threshold를 낮춰도 TP가 없어서 broader-val이나 long run으로 확장하지 않는다.
- selector-map component gate read-only probe도 val128 stop-line F1 `0.2062`로 `stop_mask_only` `0.2593`보다 낮았다. selector map을 단순 component 선택에 쓰는 후처리만으로는 0.6 path가 아니다.
- selector/row/x dense-map audit은 val128 epoch2에서 stop-line mask F1 `0.6646`, row-mask F1 `0.6727`, x-mask F1 `0.7482`를 보였지만 selector-mask F1은 `0.3119`, selector-centerline F1은 `0.4259`였다. mask와 x projection은 남아 있지만 selector/proposal readout이 full line segment로 결합되지 못한다.
- row/x projection을 직접 span proposal로 바꾸는 read-only probe도 exact val128 baseline stop-line F1 `0.4483`보다 낮았다. best fallback은 `0.4310`, best replacement는 `0.3146`이고 row/x decoder는 512 samples 중 `29~30`개에서만 line을 만들었다.
- rowx-band selector target + selector component gate short run도 exact val128 epoch2 stop-line F1 `0.4144`, TP/FP/FN `23 / 28 / 37`로 기준 `0.4483`, `26 / 30 / 34`보다 낮았다. `phase_objective=0.6036`은 stop-line success가 아니다.
- GT-center + angle-anchored mask extent diagnostic은 exact val128에서 predicted angle variant stop-line F1 `0.6126`, broader-val512에서 `0.5361`을 냈다. 이는 production success는 아니지만, half-length scalar보다 mask extent length readout이 더 유망하다는 upper-bound다.
- production predicted center/selector proposal + angle-anchored mask extent readout은 exact val128에서 best `0.5085`, TP/FP/FN `30 / 28 / 30`에 그쳐 prior PCA val128 reference `0.5133`을 넘지 못했다. baseline `0.4483`보다는 낫지만 broader-val512로 확장하지 않는다.
- stop-line flip TTA score/geometry averaging smoke는 val4에서 baseline과 완전히 동일했다: lane/stop/cross F1 `0.5507 / 0.0000 / 0.5455`, stop-line TP/FP/FN `0 / 3 / 2`. 같은 averaging contract는 val128/broader 확장 후보가 아니다.
- proposal recall audit은 GT 주변 local score와 top-k ranking을 분리했다. broader-val512 `max(center, selector)` source는 `max_r8 >= 0.6`이 `208/271`이지만 top3-hit-r8은 `137/271`, raw rank top3는 `14/271`뿐이다. local signal은 남아 있으나 proposal competition/ranking이 약하다.
- candidate-pool audit은 broader-val512에서 oracle-positive top20 selection stop-line F1 `0.6517`, TP/FP/FN `131 / 0 / 140`까지 가능함을 보였다. 하지만 best production score/length filter는 `0.4371`, TP/FP/FN `106 / 108 / 165`로 PCA reference `0.4699`보다 낮다. 후보 pool은 있으나 score/length만으로는 FP suppression signal이 부족하다.
- candidate generation gap audit은 broader-val512에서 top-k NMS gap을 줄이고 더 넓은 pool을 만들면 oracle-positive headroom이 늘어남을 보였다. `gap6_oracle_max_top20_positive`는 stop-line F1 `0.6683`, TP/FP/FN `136 / 0 / 135`, `gap4_oracle_max_top50_positive`는 `0.6877`, TP/FP/FN `142 / 0 / 129`다. 그러나 production `gap6/gap4` score-threshold variants는 기존 best `0.4371`, TP/FP/FN `106 / 108 / 165`와 동일하다. gap/top-k-only pool widening은 production fix가 아니며 denser candidates를 고르는 task-aware selector/readout이 별도로 필요하다.
- hard-negative proposal ranking loss short run은 exact val128 epoch2 lane/stop/cross F1 `0.4754 / 0.4918 / 0.5767`, stop-line TP/FP/FN `30 / 32 / 30`에 그쳤다. stop-line은 baseline `0.4483`보다 높지만 angle-mask production `0.5085`와 PCA val128 reference `0.5133`보다 낮고, same-checkpoint candidate-pool production best도 `0.4286`이라 같은 loss-only family로 확장하지 않는다.
- candidate scalar-feature validator audit은 val512 candidate-row half split에서 logistic test AUC/AP `0.7894 / 0.5699`, candidate-level F1 `0.5651`을 보였다. score-only도 candidate-level F1 `0.5812`다. 하지만 candidate-row 분류는 task F1이 아니며 같은 scalar family의 production replay는 broader-val512 stop-line F1 `0.4371`에 머물렀다.
- candidate rich-feature validator audit은 proposal/decoded center 좌표와 center/selector/mask local window features까지 넣으면 val512 candidate-row logistic test AUC/AP/F1 `0.8149 / 0.5950 / 0.6175`, oracle-best row F1 `0.6387`까지 오른다. `selector_r4_max` 단일 feature도 row F1 `0.6109`다. 하지만 task replay는 변하지 않으므로 production success가 아니라 model-side candidate instance validator의 전제 evidence다.
- stop-line recovery-budget audit은 projection-competition reference `0.5164`, TP/FP/FN `126 / 91 / 145`에서 target F1 `0.60`까지 필요한 최소 회복량을 고정했다. Current FP에서 `+30` TP가 필요하고, FP-only path는 `68 / 91` FP 제거가 필요하다. Positive-misrank `29`개만 모두 회복해도 `0.5996`이라 selector/ranker-only는 수학적으로도 한 끗 부족하다. Positive no-oracle `62`개를 모두 회복하는 oracle upper bound는 F1 `0.6836`이고 added FP `76`까지 허용하므로, 다음 stop-line branch는 no-oracle candidate generation 또는 midpoint recovery를 겨냥해야 한다.
- stop-line no-oracle fragment-extension budget은 current positive-no-oracle samples를 top/longest/nearest-GT-oracle fragment의 min-length extension으로 대체해도 baseline projection competition `0.5164`, TP/FP/FN `126 / 91 / 145`보다 모두 낮았다. Best extension은 F1 `0.4713`, TP/FP/FN `119 / 115 / 152`다. 따라서 no-oracle 문제는 단순 짧은 fragment 길이 부족이 아니라 current midpoint/centering 자체가 틀어진 문제로 본다.
- stop-line no-oracle proposal-recall bucket audit은 positive-no-oracle GT `69`개 중 max-source `max_r8 >= 0.6`이 `44`, `top20_hit_r8`이 `61`개임을 보였다. 그러나 `top1_hit_r8`은 `12`, `top3_hit_r8`은 `27`개뿐이다. 즉 dense `max(center, selector)` 신호는 상당수 남아 있지만 현재 top candidate 선택과 geometry decode가 GT center 주변 신호를 full segment로 바꾸지 못한다.
- stop-line no-oracle anchor-shift audit은 current no-oracle candidates를 proposal-cell anchor 쪽으로 되돌려도 baseline projection competition `0.5164`, TP/FP/FN `126 / 91 / 145`보다 모두 낮았다. 모든 anchor-shift variant는 F1 `0.4475`, TP/FP/FN `113 / 121 / 158`이었다. 따라서 문제는 decoded-center offset만이 아니라 local-neighborhood 후보 geometry 자체다.
- stop-line no-oracle local proposal geometry audit은 positive-no-oracle max-source GT `69`개 중 `top20_hit_r8` `61`, nearest exported candidate within r8 `51`임을 보였지만, local candidate 중 `nearest_gt_distance <= 40`은 `0`개였다. Local q50은 proposal distance `2px`, max_r8 `0.9952`, candidate nearest distance `95.36px`, midpoint distance `68.59px`, length ratio `0.5437`이다. 즉 GT 근처 local cell은 있어도 현재 decode geometry가 full matched segment로 바뀌지 않는다.
- stop-line no-oracle local recenter budget은 selected local samples `47 / 47`에서 per-sample affine을 썼고 fallback `0`으로 local raw/anchor/minlen/GT-length variants가 모두 projection-competition reference `0.5164`보다 낮음을 보였다. Best non-midpoint oracle은 `local_anchor_gt_length_oracle` F1 `0.4939`, TP/FP/FN `122 / 101 / 149`다. 반면 GT midpoint+GT length oracle은 F1 `0.6599`, TP/FP/FN `163 / 60 / 108`이므로, local angle evidence는 center가 맞을 때 쓸 수 있고 핵심 병목은 midpoint/center recovery다.
- stop-line no-oracle axis-offset budget은 그 midpoint/center recovery를 더 좁혔다. Local selected rows `51`개 중 `49`개는 법선보다 축 방향 offset이 크고, normal q50/q90은 `2.31px / 13.47px`라 후보가 대체로 같은 line 위에 있다. Axis-only GT projection + GT length oracle은 F1 `0.6559`, TP/FP/FN `162 / 61 / 109`로 full GT-midpoint+GT-length oracle `0.6599`와 거의 같다. 하지만 axis-only projection + fixed minlen sweep은 `minlen=240`의 F1 `0.5547`이 최고라 목표에 못 미친다. 다음 branch는 no-GT along-axis center/extent inference가 필요하다.
- stop-line axis-projected predicted-offset readout은 이 along-axis center premise의 첫 no-GT probe였지만 exact val128에서 기존 full pred-offset과 같은 matched set이다: both `0.5085`, TP/FP/FN `30 / 28 / 30`. Mean point distance만 `13.62 -> 13.51`로 줄어 F1 gate를 못 넘으므로 broader-val512로 확장하지 않는다.
- stop-line axis-profile readout은 proposal-cell anchor로 mask profile의 midpoint/length를 다시 읽어봤지만 exact val128 best가 `0.5085`, TP/FP/FN `30 / 28 / 30`으로 같은 matched set에 머문다. Existing angle-mask/axis-projected readout과 동률일 뿐 PCA val128 `0.5133`이나 broader projection-competition `0.5164`를 넘지 못하므로 proposal-source/top-k/mask-threshold/normal-band sweep으로 반복하지 않는다.
- stop-line axis-window recenter는 predicted axis 위 고정 window로 center를 재선택했지만 exact val128 F1 `0.4306`, TP/FP/FN `31 / 53 / 29`로 baseline/selector references보다 낮다. Axis-window radius/step/scoring-length/top-K/proposal-threshold/fallback sweep으로 반복하지 않는다.
- stop-line dense-Hough global line vote는 predicted dense mask/proposal/angle support에서 GT 없이 선분을 뽑았지만 exact val128 best Hough F1 `0.4000`, TP/FP/FN `24 / 36 / 36`으로 baseline `0.4483`와 selector reference `0.5085`보다 낮다. Angle-bin/rho-bin/normal-band/mask/proposal-threshold/top-K sweep으로 반복하지 않는다.
- candidate rich-validator held-out task replay는 row signal이 task selection으로 일부 옮겨짐을 보였지만 gate를 통과하지 못했다. val512 앞 half에서 threshold를 맞추고 뒤 half를 평가하면 held-out baseline stop-line F1 `0.3877`에서 rich logistic `0.4231`, `selector_r4_max` `0.4259`로 오른다. 하지만 PCA broader reference `0.4699`보다 낮고 목표 `0.60`과는 멀다.
- selector feature-patch validator replay는 dense `stop_line_selector_feature`의 proposal/decoded local 128-channel patch mean까지 넣어도 gap4/top50 held-out rich logistic task F1 `0.3982`, TP/FP/FN `44 / 49 / 84`에 그쳤다. 기존 `selector_r4_max` threshold replay `0.4558`, PCA broader reference `0.4699`보다 낮아서 raw selector embedding patch를 offline validator로 키우는 방향은 현재 production path가 아니다.
- model-side candidate instance validator head는 opt-in 구현과 smoke는 통과했지만 exact val128에서 실패했다. Direct `validator` gate는 epoch1/2 stop-line F1 `0.0000`이고, 같은 epoch2 checkpoint를 `center` gate로 되돌려도 stop-line F1 `0.4211`로 기준 `0.4483`보다 낮다.
- candidate validator calibrated replay는 direct collapse를 일부 회복했지만 gate를 넘지 못했다. Same checkpoint에서 `product_validator` + fallback best는 exact val128 lane/stop/cross F1 `0.5271 / 0.4918 / 0.5854`, stop-line TP/FP/FN `30 / 32 / 30`이다. direct validator `0.0000`과 center-gate replay `0.4211`보다는 낫지만 PCA val128 reference `0.5133`과 angle-mask production `0.5085`보다 낮으므로 broader-val512로 확장하지 않는다.
- candidate validator warm-bias smoke는 bias `2.0`과 `4.0` 모두 train8/val4 stop-line F1 `0.0000`, TP/FP/FN `0 / 2 / 2`라 direct dense validator gate 실패가 단순 초기 logit bias 문제만은 아님을 보였다.
- proposal competition loss는 center/selector proposal map을 GT center heatmap distribution 쪽으로 누르는 opt-in KL loss를 시험했지만 exact val128 epoch2 objective `0.6051`, lane/stop/cross F1 `0.5552 / 0.4107 / 0.5854`, stop-line TP/FP/FN `23 / 29 / 37`에 그쳤다. Prior row-scan reference `0.6144`, `0.5522 / 0.4483 / 0.5854`보다 objective와 stop-line이 낮고, dense KL loss 비용도 커서 같은 loss-only family로 확장하지 않는다.
- modern task-head merge replay는 row-scan lane decode와 proposal-rank stop-line head를 결합하면 exact val128 objective `0.6196`, lane/stop/cross F1 `0.5522 / 0.4918 / 0.5854`, stop-line TP/FP/FN `30 / 32 / 30`까지 올릴 수 있음을 보였다. 그러나 broader-val512에서는 objective `0.5977`, lane/stop/cross F1 `0.5279 / 0.4079 / 0.5854`, stop-line TP/FP/FN `103 / 131 / 168`이라 prior row-scan broader reference를 넘지 못했다. Focal-cross head transplant도 exact crosswalk F1 `0.5535`로 retention을 잃으므로 반복하지 않는다.
- segment-MIL lane + proposal-rank stop-line task-head merge는 exact val128 objective를 `0.6248`까지 올리고 lane/stop/cross F1 `0.5660 / 0.4918 / 0.5926`을 만들었지만, broader-val512에서는 objective `0.6130`, lane/stop/cross F1 `0.5480 / 0.3976 / 0.6185`, stop-line TP/FP/FN `100 / 132 / 171`로 current broader best objective `0.6231`와 stop-line `0.4235`보다 낮다. Exact-val task-head recombination is not a broader stop-line fix.
- candidate-assignment loss는 existing center/selector top-k 후보를 GT endpoint segment에 직접 assign하는 opt-in loss를 시험했지만 exact val128에서 stop-line이 무너졌다. Best objective는 epoch1 `0.5939`이고 epoch2 lane/stop/cross F1은 `0.5555 / 0.0571 / 0.5818`, stop-line TP/FP/FN `2 / 8 / 58`이다. Runtime은 `skipped_steps=0`라 valid negative evidence이며, 같은 top-k 후보 위 loss-only assignment는 반복하지 않는다.
- endpoint-delta target/readout short run도 val128 epoch1/2 stop-line F1이 모두 `0.0000`이었다. 새 dense endpoint channel을 바로 decode source로 쓰면 TP가 사라지므로 같은 형태로 확장하지 않는다.
- heatmap-support geometry target fill은 exact val128 epoch2 stop-line F1 `0.2338`, TP/FP/FN `18 / 76 / 42`로 기준보다 크게 낮고, dense stop-line mask/center F1도 `0.4854 / 0.0815`로 후퇴했다. center heatmap support 전체에 geometry target을 채우는 방식은 0.6 path가 아니다.
- row-center auxiliary short run은 exact val128 epoch2 stop-line F1 `0.4248`로 기준 `0.4483`보다 낮았다. row selector에 centerline-row pressure만 추가하는 방식도 0.6 path가 아니다.
- component/readout audit은 broader-val512 GT `271`개 중 production TP `98`, anchorless component fit close `123`, anchored fit close `119`를 보였다. GT tube mask/center max가 `>=0.50`인 GT는 `223 / 220`개지만, production FN 중 anchorless fit으로 새로 close가 되는 것은 `34`개뿐이다. 단순 no-anchor PCA/anchor swap은 0.6 path가 아니다.
- older branch history의 GT target mask oracle은 stop-line F1 `0.8228`까지 가능했고, GT-overlap oracle은 `component_fit_or_endpoint_error=97` 중 `50`개를 oracle-trimmed pixels로 40px 안에 복구했다. 따라서 vectorizer/evaluator 자체가 hard blocker라는 해석은 약하다.
- 하지만 non-oracle repair follow-up은 닫혔다. core-row trim best는 val128 stop-line F1 `0.4833`으로 PCA reference `0.5133` 미달, high-confidence cleanup은 broader-val512 `0.4667`로 PCA reference `0.4699` 미달, split fit은 val128 `0.4957 / 0.4786`로 PCA reference를 못 넘었다.
- fit-far visual audit은 production FN, GT tube mask/center `>=0.50`, no-anchor distance `>40px` bucket 상위 18개를 렌더링했다. 18개 모두 production stop-line은 1개씩 있고, 14개는 component_count도 1이라 single connected component 안에서 wrong line segment를 읽는 문제가 강하다.
- local component extraction probe는 center/selector/fused score로 component 내부 local support를 골라 다시 fit했지만 exact val128 stop-line F1이 기준 `0.4483`을 넘지 못했다. best replacement는 `0.4464`, append-top2 best는 FP 증가 때문에 `0.4054`다.
- component-split readout probe는 pair/Hough-like 후보를 component 내부에서 만들었지만 exact val128 best append-top2도 stop-line F1 `0.3421`, TP/FP/FN `26 / 66 / 34`로 baseline보다 낮았다. replacement 계열은 best `0.2857`로 TP를 크게 잃었다.
- component proposal readout probe는 predicted mask component마다 center/selector/max proposal 하나를 고르고 predicted angle + mask extent로 segment를 읽었지만 exact val128 best stop-line F1은 `0.4308`, TP/FP/FN `28 / 42 / 32`로 baseline `0.4483`, TP/FP/FN `26 / 30 / 34`보다 낮았다. component별 proposal 제한은 recall을 조금 올렸지만 FP를 더 많이 늘린다.
- mask-ridge readout probe는 center/selector top-k 대신 predicted mask component의 distance-transform ridge에서 center/axis를 직접 읽었지만 exact val128 best stop-line F1은 `0.4538`, TP/FP/FN `27 / 32 / 33`에 그쳤다. Baseline `0.4483`보다 작은 gain이지만 PCA val128 `0.5133`과 angle-mask production `0.5085`를 넘지 못해 broader-val512로 확장하지 않는다.
- line-support readout probe는 top-k center/selector proposal을 predicted angle + mask extent로 segment화하고 segment 위 mask/center/selector support로 rerank했다. exact val128 best stop-line F1은 `0.4651`, TP/FP/FN `30 / 39 / 30`으로 baseline `0.4483`, `26 / 30 / 34`보다 높지만 PCA val128 `0.5133`, angle-mask production `0.5085`, task-head merge `0.4918`보다 낮아 broader-val512로 확장하지 않는다.
- candidate consensus readout은 top-k 후보끼리 가까운 segment agreement로 isolated FP를 누르는 read-only selector를 시험했지만 exact val128 best production consensus stop-line F1은 `0.4706`, TP/FP/FN `28 / 31 / 32`다. Baseline `0.4483`은 넘지만 `max_top10_score_s080`/angle-mask production `0.5085`, PCA val128 `0.5133`, task-head merge `0.4918`보다 낮고, oracle-positive `0.7368`은 production evidence가 아니다.
- lane-context readout은 `row_scan_tangent` lane prediction과 stop-line 후보 segment의 교차/거리 feature로 isolated FP를 줄이는지 봤지만 exact val128 best lane-context stop-line F1은 `0.4696`, TP/FP/FN `27 / 28 / 33`이다. Baseline `0.4483`은 넘지만 score-threshold `0.5085`, PCA val128 `0.5133`, task-head merge `0.4918`보다 낮아 broader-val512로 확장하지 않는다.
- crosswalk-context readout은 predicted crosswalk polygon proximity로 stop-line 후보를 정렬/필터링했지만 exact val128에서 기준보다 낮았다. `max_top10_crosswalk_context` stop-line F1은 `0.3810`, TP/FP/FN `28 / 59 / 32`이고 `max_top20_crosswalk_near48_c1`은 `0.2105`, TP/FP/FN `10 / 25 / 50`이다. Predicted crosswalk context는 FP를 줄이지 못하거나 recall을 무너뜨리므로 broader-val512로 확장하지 않는다.
- delayed candidate-validator auxiliary는 center gate를 유지한 채 dense validator hard-negative aux를 학습하고 post-train validator map mixing을 replay했지만 exact val128 gate를 넘지 못했다. Training epoch2 lane/stop/cross F1은 `0.5546 / 0.4425 / 0.5854`이고 replay best `product_validator_top3_s040_mask050_band4_fallback` stop-line F1은 `0.4793`, TP/FP/FN `29 / 32 / 31`로 task-head merge `0.4918`, angle-mask `0.5085`, PCA `0.5133`보다 낮다.
- local-centerline selector target은 center target을 `local_union`, selector target을 `local_centerline`으로 바꿔 center/selector proposal map ranking을 직접 바꿔 봤지만 exact val128 gate를 넘지 못했다. Epoch2 lane/stop/cross F1은 `0.5607 / 0.4522 / 0.5854`, objective `0.6167`, stop-line TP/FP/FN `26 / 29 / 34`이다. Stop-line gain은 tangent-link exact `0.4483` 대비 `+0.0039`뿐이고 objective/lane이 낮아 broader-val512로 확장하지 않는다.
- geometry-aware candidate validator는 current top-k center/selector 후보를 predicted endpoint segment distance로 라벨링하는 opt-in loss를 시험했지만 exact val128 gate를 넘지 못했다. Epoch2 objective는 `0.6182`, lane/stop/cross F1은 `0.5607 / 0.4655 / 0.5854`, stop-line TP/FP/FN은 `27 / 29 / 33`이다. Stop-line은 tangent-link exact `0.4483`과 local-centerline selector `0.4522`보다 높지만 objective/lane은 tangent-link exact `0.6187`, `0.5633`보다 낮다. Same-checkpoint replay에서도 baseline stop-line F1 `0.4655`가 최고이고 validator-map variants는 `blend_validator` `0.3973`, `product_validator` `0.3889`, `max_validator` `0.3692`, direct `validator` `0.3077`로 모두 낮아 broader-val512로 확장하지 않는다.
- denser candidate-select contract는 gap4/top50 후보를 직접 candidate-validator/presence supervision으로 학습하고 `max_validator` gate로 읽었지만 stop-line emit이 완전히 꺼졌다. Exact val128 epoch2 objective는 `0.5855`, lane/stop/cross F1은 `0.5612 / 0.0000 / 0.5854`, stop-line TP/FP/FN은 `0 / 0 / 60`이다. Runtime은 정상이고 `skipped_steps=0`이므로 valid negative evidence다. Same `best.pt` evaluator replay shows the direct suppressor is the presence gate: default `max_validator + presence=0.35` gives stop-line F1 `0.0000`, while `max_validator + presence=0.0` restores only `0.4483`, TP/FP/FN `26 / 30 / 34`. This recovery is baseline-level, so 같은 candidate-select + presence/max-validator 조합이나 presence-threshold rescue를 longer run/sweep으로 반복하지 않는다.
- stop-line-head-only geometry-validator schedule은 trunk, detector/TL, lane, crosswalk heads를 고정하고 stop-line head만 업데이트했지만 exact val128 gate를 넘지 못했다. Epoch2 objective는 `0.6180`, lane/stop/cross F1은 `0.5640 / 0.4386 / 0.5926`, stop-line TP/FP/FN은 `25 / 29 / 35`이다. Geometry-validator stop-line reference `0.4655`와 tangent-link stop-line `0.4483`보다 낮아 broader-val512로 확장하지 않는다.
- model-side presence emit gate는 sample-level `stop_line_presence_logits`와 postprocess emission threshold를 학습/적용했지만 exact val128 gate를 넘지 못했다. Epoch2 objective는 `0.6102`, lane/stop/cross F1은 `0.5606 / 0.4112 / 0.5854`, stop-line TP/FP/FN은 `22 / 25 / 38`이다. FP는 조금 줄었지만 TP가 더 줄어 tangent-link exact `0.4483`, PCA/angle-mask references보다 낮아 broader-val512로 확장하지 않는다.
- proposal-stat emit gate는 `stop_line_presence_logits`에 stop-line mask/center/selector/row/x dense-map max/mean statistics를 더했지만 exact val128 gate를 넘지 못했다. Epoch2 objective는 `0.6120`, lane/stop/cross F1은 `0.5611 / 0.4074 / 0.5854`, stop-line TP/FP/FN은 `22 / 26 / 38`이다. Presence-only보다 FP가 하나 늘고 TP는 회복되지 않아 broader-val512로 확장하지 않는다.
- mask-wide angle-field auxiliary는 existing `stop_line_angle` map을 stop-line mask support 전체에서 supervised했지만 exact val128 gate를 넘지 못했다. Epoch2 objective는 `0.6152`, lane/stop/cross F1은 `0.5606 / 0.4348 / 0.5854`, stop-line TP/FP/FN은 `25 / 30 / 35`이다. Tangent-link exact `0.6187`, `0.5633 / 0.4483 / 0.5854`와 PCA `0.5133`, angle-mask production `0.5085`, task-head merge `0.4918` stop-line references보다 낮아 broader-val512로 확장하지 않는다.
- lane centerline threshold oracle은 fixed tangent-link checkpoint에서 global lane threshold와 sample-wise dense-core oracle threshold를 replay했지만 lane 0.6 path가 아니었다. Best global `global_t020` lane F1은 `0.5641`, TP/FP/FN `1153 / 545 / 1237`로 current `global_t045`-like `0.5633`, `1121 / 469 / 1269`와 noise-level 차이고, `sample_oracle_dense_core`는 `0.5482`, `1074 / 454 / 1316`로 더 낮았다. Threshold-only calibration/readout을 반복하지 않는다.
- lane soft-instance shell auxiliary는 core centerline target을 유지하고 각 lane 주변 soft-shell에 instance-balanced BCE를 추가했지만 exact val128 gate를 넘지 못했다. Epoch2 objective는 `0.6159063607`, lane/stop/cross F1은 `0.5641 / 0.4348 / 0.5854`, lane TP/FP/FN은 `1144 / 522 / 1246`이다. Tangent-link보다 lane F1은 noise-level로 높지만 objective와 stop-line은 낮고 segment-MIL lane-head-only exact best `0.5660 / 0.4483 / 0.5926`에도 못 미쳐 broader-val512로 확장하지 않는다.
- lane soft-ignore band는 core centerline target을 유지하되 soft-shell pixels를 centerline BCE/Dice/Focal negative에서 제외했지만 exact val128 gate를 넘지 못했다. Epoch2 objective는 `0.6088389162`, lane/stop/cross F1은 `0.5360 / 0.4348 / 0.5854`, lane TP/FP/FN은 `1032 / 429 / 1358`이다. FP는 tangent-link보다 줄었지만 TP를 더 크게 잃어 lane recall이 무너졌으므로 soft-band ignore/threshold-only masking은 반복하지 않는다.

후보:

- stop-line은 GT-center angle-mask extent upper-bound를 봤지만, 단순 predicted selector/center proposal readout은 PCA reference를 못 넘었다.
- stop-line proposal map은 GT 근처 local signal이 남아 있지만 low-top-k ranking이 약하다.
- stop-line candidate pool에는 valid segment가 있고, gap/top-k를 넓힌 denser pool에는 oracle headroom이 더 있다. 그러나 score/length production, held-out task replay, gap4/top50 rich-selector held-out replay, component-topology rich-validator replay, positive-sample rank diagnostic, cold/warm dense validator gate training, delayed dense candidate-validator auxiliary, simple calibrated map mixing, existing top-k candidate-assignment loss-only, geometry-aware candidate-validator loss-only, denser candidate-select + presence/max-validator gate, stop-line-head-only freeze schedule, model-side presence-only emit gate, proposal-stat emit gate, mask-wide angle-field auxiliary, candidate consensus-only, lane/crosswalk context-only readout 모두 실패했다. gap4/top50 `selector_r4_max` held-out stop-line F1은 `0.4537`로 prior held-out `0.4259`보다 높지만 PCA broader reference `0.4699`와 목표 `0.60`을 넘지 못했다. Component-topology rich logistic도 held-out stop-line F1 `0.4126`, TP/FP/FN `46 / 49 / 82`로 selector-only replay보다 낮다. Feature-rank diagnostic에서는 `mask_r4_max`가 positive sample 내부 oracle 후보를 top1 `116/142`로 잘 올렸지만, candidate-bearing sample `369`개 중 `227`개는 oracle-positive가 없는 negative sample이다. Sample-level emit gate diagnostic은 held-out surrogate selection F1을 emit-all `0.4320`에서 logistic gate `0.5732`로 올렸지만, shallow tree gate는 `0.5543`으로 logistic보다 낮고 actual held-out task replay도 baseline stop-line F1 `0.3877`에서 `0.3843`으로 낮아졌다. GT sample-gate oracle replay는 GT sample presence에 current-row score/local-feature ranking을 붙여도 best `selector_r4_max` stop-line F1 `0.4800`, TP/FP/FN `114 / 90 / 157`에 그쳐 `0.60`과는 멀고 PCA broader reference `0.4699`를 아주 조금 넘는 정도다. Baseline-preserving rescue replay도 full val512 best rescue `0.4165`로 기존 sample-gate actual `0.4259`, score-threshold production `0.4371`, PCA broader reference `0.4699`를 넘지 못했다. Presence-only model-side emit gate도 exact stop-line F1을 `0.4112`로 낮췄고, dense proposal-stat presence gate도 `0.4074`로 더 낮았으며, mask-wide angle-field auxiliary도 `0.4348`로 tangent-link/PCA/angle-mask references를 넘지 못했다. Denser candidate-select exact run은 stop-line emit 자체를 `0 / 0 / 60`으로 꺼버렸다. `stopline_negative` hard-negative sampler-only short run도 exact val128 stop-line F1을 `0.4561`로 조금 올렸을 뿐 lane/crosswalk/objective를 `0.5475 / 0.5644 / 0.6063`으로 낮춰 expansion gate를 못 넘었다. Fragment-union CSV replay and its opt-in evaluator postprocess replay are the first broader positives in this local family: stop-line F1 `0.4948`, TP/FP/FN `120 / 94 / 151`, beating PCA broader `0.4699` and GT sample-gate + same-row feature rank `0.4800`, but still below `0.60`. Follow-up seed-extension regressed to `0.4742`, and length-competition only nudged the readout to `0.5031`, TP/FP/FN `121 / 89 / 150`; delta audit shows only `6/369` candidate-bearing samples changed (`fp_removed=3`, `tp_added=2`, `tp_lost=1`). Multi-instance fragment-union top2 reaches `0.5040`, TP/FP/FN `125 / 100 / 146`, but it buys `+4 TP` versus length competition with `+11 FP`; second-instance gating improves the trade to `0.5061`, TP/FP/FN `124 / 95 / 147`. Projection-gap split reached `0.5112`, TP/FP/FN `125 / 93 / 146`; projection competition is the current ungated local stop-line readout reference at `0.5164`, TP/FP/FN `126 / 91 / 145`. Single-feature projection selector audit reaches only `0.5250`, TP/FP/FN `126 / 83 / 145`; raw-image photometric selector reaches only `0.5217`, TP/FP/FN `126 / 86 / 145`; multifeature logistic reaches full `0.5336` and photometric logistic full `0.5442`, but both fail held-out (`0.4369` and `0.4433` vs baseline `0.4848`). Row/x candidate consistency also fails as a stronger held-out selector: val512 held-out baseline is `0.3877`, row/x rich logistic `0.4087`, and selector_r4 `0.4259`. None is a deployable selector path. 현 방식의 candidate validator/assignment/select/agreement/context/logistic/tree-sample-gate/rescue/presence-stat/mask-angle-field/topology-column/rowx-consistency/read-only sample oracle/read-only feature-rank readout과 sampler-only negative exposure는 닫고, 다음 stop-line 후보는 fragment-union top-k/min-gap/min-score knob sweep, length-feature micro-sweep, multi-instance top-K sweep, second-instance gate sweep, projection-gap sweep, projection-competition length/min-score sweep, projection-competition single-feature selector sweep, projection-competition exported-feature logistic sweep, projection-competition raw-image photometric sweep, or row/x consistency logistic sweep이 아니라 midpoint/candidate generation recovery 또는 FP를 더 크게 줄이는 새 no-GT readout contract다.
- Detector-context FP audit is also closed. Predicted `traffic_light` / `sign` proximity suppresses emissions but destroys recall: exact val128 `max_top10_signal_context_c1` stop-line F1 is only `0.1538`, TP/FP/FN `6 / 12 / 54`, versus baseline `0.4483`, `26 / 30 / 34`, and score-threshold reference `0.5085`, `30 / 28 / 30`. Candidate-level signal-near gating preserves only `25 / 156` top10/gap10 oracle-positive rows (`0.160` recall). Do not repeat detector signal proximity as a radius, score, class-weight, or top-k sweep unless a new TP-preserving signal is added.
- Current-composite candidate manifest failure-mode audit refines the split: gap4/max candidate-bearing samples `369`개 중 GT-negative `165`, positive top-oracle `113`, positive misrank `29`, positive no-oracle `62`다. Positive has-oracle rate is `0.6961`; no-oracle nearest-distance bins are `40_80=28`, `gte80=34`. Therefore selector/ranker work can only address the `29` misrank samples unless it also changes center proposal/candidate generation. The next stop-line branch must explicitly target midpoint proposal recovery or a new non-top-k candidate generation/readout contract; do not run another threshold/ranker-only branch.
- Enriched manifest geometry makes the same point sharper: `positive_no_oracle` top score q50 is `0.9998`, nearest GT distance q50 is `89.98px`, and nearest candidate length ratio q50 is `0.066`. The issue is high-confidence short fragments far from the GT midpoint, not a low-score ranker threshold.
- Fragment extent recovery tested a production-style predicted proposal + gap-tolerant mask-strip readout. It lifted exact val128 stop-line F1 from baseline `0.4483` to `0.4918`, but stayed below prior PCA `0.5133` and predicted angle-mask production `0.5085` references. It is not a broader-val512 expansion path.
- Learned dense fragment-to-center offset/extent head is also closed. Exact val128 best objective was only `0.5499257237`, and stop-line F1 collapsed to epoch1/2 `0.0267 / 0.0519` with skipped steps `0`. Fragment-disabled replay recovers only to stop-line F1 `0.1233 / 0.2517`, so this is both a weak auxiliary-trained checkpoint and a bad production decode. This is valid negative evidence, not a runtime failure; do not broaden or repeat as an aux-weight/top-k/min-score sweep.
- Fragment-union readout over the current-composite candidate CSV is a partial broader positive, and the opt-in model-output postprocess/evaluator replay reproduces it: best `union_a12_o36_s080_c2_fallback_top` reaches stop-line F1 `0.4948`, TP/FP/FN `120 / 94 / 151`; non-fallback CSV `union_a16_o48_s080_c2` reaches `0.4936`, TP/FP/FN `116 / 83 / 155`. Seed-extension follow-up is negative at `0.4742`, while length-competition is a tiny partial-positive at `0.5031`, TP/FP/FN `121 / 89 / 150`. Delta audit says this changes only `6/369` candidate-bearing samples, so it is fallback suppression rather than broad geometry recovery. Multi-instance top2 reaches `0.5040`, TP/FP/FN `125 / 100 / 146`, but the added recall is nearly cancelled by FP; second-instance gating improves it only to `0.5061`, TP/FP/FN `124 / 95 / 147`. Projection-gap split reaches `0.5112`, TP/FP/FN `125 / 93 / 146`; projection competition is the current ungated local stop-line readout reference at `0.5164`, TP/FP/FN `126 / 91 / 145`. Single-feature projection selector audit reaches `0.5250`, TP/FP/FN `126 / 83 / 145`; raw-image photometric selector reaches `0.5217`, TP/FP/FN `126 / 86 / 145`; multifeature logistic full replay reaches `0.5336` and photometric logistic full replay reaches `0.5442`, but held-out logistic regresses to `0.4369` / `0.4433` from baseline `0.4848`. None is enough for the all-task goal.
- The first midpoint-rank training attempt, `core_centerline_refine_row_scan_tangent_stop_center_rank_margin`, is valid but insufficient: exact val128 objective `0.6170`, lane/stop/cross F1 `0.5605 / 0.4602 / 0.5854`. It is below tangent-link and segment-MIL exact references, and the stop-line gain does not beat geometry-validator `0.4655`. Do not broaden or repeat as a margin-only sweep.
- The same checkpoint with predicted-proposal + angle-anchored mask-extent replay reaches best exact stop-line F1 `0.5042`, TP/FP/FN `30 / 29 / 30`, but remains below prior angle-mask production `0.5085` and PCA val128 `0.5133`. This closes center-rank-margin + existing readout replay as an expansion path.
- Center-rank proposal recall also does not show enough candidate-generation recovery. Exact val128 `max` source stayed at `max_r8 >= 0.6` `50/60`, top3-hit-r8 only moved `37/60 -> 39/60`, top10-hit-r8 fell `53/60 -> 52/60`, and raw rank top3 moved `9/60 -> 11/60`.
- Score-island weighted center readout is also closed. It changed the existing predicted proposal + angle-mask extent replay to use a weighted local `max(center, selector)` island center, but exact val128 best island stop-line F1 was only `0.4354`, TP/FP/FN `32 / 55 / 28`, below baseline `0.4483` and the existing selector-center reference `0.5085`.
- Normal-support recenter readout is also closed. Sliding predicted proposal centers along the angle normal to maximize local mask support also reaches only exact val128 stop-line F1 `0.4354`, TP/FP/FN `32 / 55 / 28`, so the extra recall is cancelled by FP growth.
- Raw-edge recenter readout is closed too. Using raw-image edge/contrast as the normal-scan center score collapses exact val128 stop-line F1 to `0.2993` and `0.2585`, TP/FP/FN `22 / 65 / 38` and `19 / 68 / 41`, below both baseline `0.4483` and selector-center reference `0.5085`.
- PCA component 후보는 weak-positive reference로 보관하되, deployment default 승격 후보로 보지 않는다.

성공 기준:

- stop-line F1이 broader-val에서 의미 있게 상승해야 한다.
- lane/crosswalk F1이 0.6 목표에서 멀어질 정도로 무너지면 실패다.
- exact subset에서만 좋아지는 stop-line threshold tweak은 채택하지 않는다.

Gate 상태:

- partial weak-positive only. broader-val512 ungated stop-line reference는 projection-competition fragment-union CSV replay 기준 `0.5164`, val-selected single-feature selector audit best는 `0.5250`, multifeature logistic full replay는 `0.5336`이지만 held-out은 baseline보다 나쁘다. Opt-in evaluator/postprocess로 재현된 fragment-union 기준은 `0.4948`이다. 모두 목표 미달이다.
- segment-MIL lane + proposal-rank stop-line head merge는 exact val128에서는 좋아졌지만 broader-val512 stop-line FP가 늘어 current broader best를 넘지 못했다. Task-head merge는 새 broader FP-suppression premise 없이는 닫힌 축이다.
- same-family micro experiments는 중단한다.
- `exp/lane-family-f1/stopline-fragment-axis-contract` is no longer active. The low-disk exact run completed and failed the exact gate: epoch2 objective `0.5522`, lane/stop/cross F1 `0.5225 / 0.1905 / 0.5714`, stop-line TP/FP/FN `8 / 16 / 52`. Do not broaden or rerun this axis as a fragment-center aux, offset-loss-mode, top-k/min-score, or longer-epoch sweep.
- Fit-far visual + task-mask competition is now closed as a weak exact-only premise: crosswalk-only suppression changes stop-line TP/FP/FN only `26 / 30 / 34 -> 27 / 30 / 33`, and lane-inclusive suppression destroys recall. Since no materially different stop-line premise is available from this visual pass, prefer lane instance-stability work while preserving the current stop-line/crosswalk contract.
- 다음 stop-line 실행은 current center/selector top-k 위 loss-only assignment나 dense validator aux, stop-line-head-only schedule, presence-only/proposal-stat emit gate나 mask-wide angle-field auxiliary, learned dense fragment-to-center extent head, fragment-union top-k/min-gap/min-score/angle/offset sweep, seed-extension score sweep, length/component-length feature-rank sweep, multi-instance top-K sweep, second-instance threshold sweep, projection-gap sweep, projection-competition length/min-score/rank-feature sweep, projection-competition single-feature selector sweep, projection-competition exported-feature logistic sweep, projection-competition raw-image photometric sweep, row/x consistency logistic sweep, normal-support recenter sweep, raw-edge recenter sweep이 아니다. Fragment-union은 opt-in evaluator path까지 `0.4948`로 재현됐고 length-competition CSV replay는 `0.5031`로만 소폭 올랐으며 delta audit상 변경 샘플은 `6/369`뿐이다. Multi-instance top2는 `0.5040`으로 더 높지만 `+4 TP`와 함께 `+11 FP`를 만들고, second-instance gating은 `0.5061`로만 오른다. Projection-gap split은 `0.5112`까지 올랐고 projection competition은 `0.5164`까지 올랐으며 single-feature selector audit은 `0.5250`, raw-image photometric selector는 `0.5217`, multifeature logistic full replay는 `0.5336`, photometric logistic full replay는 `0.5442`까지 올랐지만 held-out이 무너지고 아직 stop-line `0.60`과는 멀다. 따라서 다음 축은 `0.5442` 같은 full-replay overfit을 넘기는 게 아니라 held-out을 보존하는 candidate-generation/midpoint recovery 또는 기존 exported/photometric feature와 다른 FP suppression premise여야 한다. top10/top20 및 gap4/top50 후보 pool은 oracle headroom을 보였고 rich local features는 row-level signal을 보였으며 sample-level emit gate는 surrogate selection F1 개선을 보였지만, tree gate는 logistic보다 낮고 actual held-out task replay와 baseline-preserving rescue replay는 실패했으며 GT sample-gate + same-row feature-rank oracle도 best `0.4800`으로 0.6에는 멀다. Model-side presence-only/proposal-stat emit gate와 mask-wide angle-field auxiliary도 exact stop-line F1을 낮췄고, center-rank margin/readout/proposal-recall, learned fragment-to-center extent head, normal-support recenter, raw-edge recenter도 candidate-generation recovery를 만들지 못했다. held-out task replay, gap4/top50 rich-selector held-out replay, component-topology rich-validator replay, positive-sample rank diagnostic, sample-level emit-gate surrogate, sample-tree gate surrogate, GT sample-gate oracle-only replay, same-row feature-rank oracle replay, sample-gate actual task replay, sample-gate baseline rescue replay, model-side presence-only emit gate, proposal-stat emit gate, mask-wide angle-field auxiliary, cold/warm dense validator gate, delayed dense validator auxiliary, calibrated map mixing, proposal-distribution KL loss, task-head merge replay, candidate-assignment loss-only, geometry-aware candidate-validator loss-only, stop-line-head-only freeze schedule, center-rank margin-only, center-rank angle-mask replay, center-rank proposal-recall-only, learned fragment-to-center extent-only, seed-extension replay, length-competition feature-rank replay, multi-instance top-K replay, second-instance score/fragment-count/length-ratio gate replay, projection-gap split replay, projection-competition feature-rank replay, projection-competition single-feature selector replay, projection-competition exported-feature logistic replay, projection-competition raw-image photometric replay, normal-support recenter replay, raw-edge recenter replay, consensus-only readout, lane-context readout-only, crosswalk-context readout-only, local-centerline selector target은 PCA/angle-mask reference와 all-task 0.6 gate를 넘지 못했다. 단순 half-length target/loss/readout scalar, learned query-vector proposal-only, endpoint-delta direct decode, heatmap-support geometry fill, row-center auxiliary-only, selector-map threshold/gate-only, row/x span proposal-only, rowx-band selector target + selector component gate, 단순 predicted center/selector/max proposal + angle-mask extent readout, component별 proposal readout, mask-ridge readout-only, top-k line-support reranking, normal-support recenter radius/step/support-weight-only, raw-edge recenter contrast/length-weight/radius/step/offset-penalty-only, candidate gap/top-k-only widening, candidate agreement/consensus-only, lane-context readout-only, crosswalk-context readout-only, local-centerline selector target-only, presence-only emit loss/threshold, proposal-stat presence head/threshold, stopline_mask_angle_aux_weight-only, dense fragment-center offset/extent aux-only, geometry-aware candidate-validator map replay, stop-line-head-only geometry-validator schedule, no-anchor PCA/anchor swap-only, PCA threshold/top-k-only, row-band/core-row trim, high-confidence cleanup, PCA endpoint quantile trim, local center/selector/fused window extraction, append-top2, pair/Hough-like component split readout, score/length-only candidate filter, scalar/rich/component-topology candidate-row classifier as production, rich held-out threshold replay as production, gap4/top50 rich-selector threshold replay as production, component-topology rich-validator threshold replay as production, positive-sample rank hit rate as production, sample-level emit-gate surrogate as production, sample-tree gate surrogate as production, GT sample-gate oracle as production, same-row feature-rank oracle as production, sample-gate train-only task gain as production, sample-gate baseline fallback/append rescue as production, hard-negative proposal ranking loss-only, proposal-distribution KL loss-only, direct/cold/warm dense validator gate, delayed dense validator aux/map-mixing, calibrated validator map-mixing replay, current top-k candidate-assignment loss-only, crosswalk-retention 없는 task-head transplant, oracle-positive candidate selection as production은 반복하지 않는다.

## 6. Gate 3: lane recall without fragment FP

목적:

- lane F1을 0.6까지 끌어올리되, final geometry filters가 제거한 small-fragment FP를 다시 만들지 않는다.

후보:

- core centerline + gated refinement는 유지한다.
- lane centerline recall 부족인지, vectorizer recovery 부족인지 broader-val dense-map PR과 vectorizer audit으로 분리한다.
- current dense-map probe 기준 lane centerline core best pixel F1은 `0.5729`, lane support best pixel F1은 `0.7971`이다.
- 따라서 첫 후보는 support가 아니라 centerline-core 품질이다.
- centerline-to-vector recovery audit 기준, GT centerline oracle은 broader-val512 epoch2 lane F1 `0.6630`까지 복구하지만 current predicted centerline은 best threshold `0.35`에서도 `0.5169`다.
- predicted attrs oracle도 geometry F1을 올리지 못했으므로 lane 병목은 semantic attr가 아니라 predicted centerline coverage/quality다.
- `core_centerline_refine_core_width3`는 exact val128 epoch2 `phase_objective=0.6002`까지 갔지만 lane/stop/cross F1 `0.5232 / 0.4348 / 0.5854`로 기준선 미달이다. target-width-only widening은 반복하지 않는다.
- centerline error-bucket audit val512 기준 miss는 side/truncated/near-vertical lane에 몰린다. `bottom_y < 0.50` miss rate `0.1329`, near-vertical `0.1290`, right-side `0.0821`, left-side `0.0705`, center x-band `0.0317`이다.
- `core_centerline_refine_side_bce_focus`는 exact val128 lane F1을 `0.5331`로 소폭 올렸지만 phase objective `0.6083`이 기준 `0.6089`보다 낮고 stop-line F1도 `0.4348`로 내려갔다. centerline-core pixel F1도 `0.5705`로 기준 `0.5729`보다 낮아 side-BCE-only는 0.6 path가 아니다.
- `core_centerline_refine_side_margin`은 exact val128 epoch2 objective `0.6097`와 lane/stop/cross F1 `0.5352 / 0.4522 / 0.5854`로 기준을 근소하게 넘었지만, dense-map PR에서 lane centerline-core F1이 `0.5573`으로 기준 `0.5729`보다 크게 낮아졌다. centerline 병목을 직접 푼 신호가 아니므로 broader-val512로 확장하지 않는다.
- `core_centerline_refine_geometry_risk_recall`은 side/truncated/near-vertical lane을 target risk bucket으로 찍고 recall-only loss를 추가했다. exact val128 epoch2 lane/stop/cross F1은 `0.5405 / 0.4348 / 0.5854`였지만, phase objective `0.6086`은 기준보다 낮고 lane centerline-core F1도 `0.5572`로 기준 `0.5729`보다 낮다. broader-val512로 확장하지 않는다.
- `core_centerline_refine_geometry_risk_local_tversky`는 risk instance 주변 local support에서 false-positive를 같이 벌주는 Tversky loss를 추가했다. exact val128 epoch2 objective는 `0.6089`, lane/stop/cross F1은 `0.5306 / 0.4522 / 0.5854`였고 lane centerline-core F1은 `0.5738`로 기준보다 `+0.0009`뿐이다. centerline 병목 해결 신호로 보기에는 너무 작고 recall-only보다 vectorized lane F1도 낮아서 broader-val512로 확장하지 않는다.
- `core_centerline_refine_negative_margin`은 explicit lane negative pixels에서 centerline probability를 margin 아래로 누르는 loss를 추가했다. exact val128 epoch2 objective는 `0.6058`, lane/stop/cross F1은 `0.5261 / 0.4348 / 0.5854`이고 lane centerline-core F1은 `0.5736`으로 기준보다 `+0.0007`뿐이다. exact metric이 기준 미달이고 dense gain도 noise-level이라 broader-val512로 확장하지 않는다.
- `lane-centerline-support-bridge`는 support를 대체 source로 쓰지 않고 centerline binary의 short gap만 support 안에서 closing했다. 같은 val128 decode probe baseline lane F1 `0.5222` 대비 best bridge `lane_bridge_s080_i2`는 lane F1 `0.4758`로 낮고, stronger closing은 `0.4075`까지 무너졌다. broader-val512로 확장하지 않는다.
- `core_centerline_refine_endpoint_coverage`는 lane visible endpoint heatmap과 endpoint-positive centerline BCE를 추가했다. exact val128 epoch2 objective는 `0.6035`, lane/stop/cross F1은 `0.5212 / 0.4348 / 0.5854`이고 lane centerline-core F1도 `0.5670`으로 기준 `0.5729`보다 낮다. broader-val512로 확장하지 않는다.
- `core_centerline_refine_row_scan_vectorizer`는 centerline binary를 바꾸지 않고 row cluster track을 이어 vectorize했다. exact val128 epoch2 lane/stop/cross F1은 `0.5522 / 0.4483 / 0.5854`, broader-val512는 `0.5279 / 0.4083 / 0.5854`로 lane gain이 유지됐다. 다만 objective `0.5981`과 stop-line F1 `0.4083` 때문에 goal success는 아니며, 18-sample visual audit에서도 sample 10 over-link risk가 남아 deployment default로 보지 않는다.
- `core_centerline_refine_row_scan_tangent_link`는 predicted `tangent_axis`를 row-cluster link assignment에만 쓰는 opt-in vectorizer 축이다. exact val128 epoch2 objective는 `0.6187`, lane/stop/cross F1은 `0.5633 / 0.4483 / 0.5854`이고, broader-val512 objective는 `0.6027`, lane/stop/cross F1은 `0.5407 / 0.4083 / 0.5854`다. row-scan보다 lane/objective는 올라 current lane replay reference로 볼 수 있지만, stop-line/crosswalk가 목표 미달이라 success/default는 아니다.
- `core_centerline_refine_row_scan_tangent_component`는 tangent row-scan을 connected component 내부로 제한해 global over-link를 줄이는 read-only vectorizer scope probe다. exact val128 objective는 `0.6106257252`, lane/stop/cross F1은 `0.5305 / 0.4483 / 0.5854`, lane TP/FP/FN은 `1030 / 463 / 1360`이다. Baseline보다는 lane이 조금 높지만 tangent-link reference `0.5633`, `1121 / 469 / 1269`에 크게 못 미쳐 broader-val512로 확장하지 않는다.
- `core_centerline_refine_row_scan_tangent_link` + stop-line `mask=0.80` exact integration은 objective `0.6171`, lane/stop/cross F1 `0.5633 / 0.4364 / 0.5854`로 tangent-link 단독보다 stop-line이 내려갔다. 같은 PCA-threshold 결합은 broader로 확장하지 않는다.
- broader-val512 composite lower-bound는 row-scan tangent lane + stop-line `mask=0.80`, `min_instance_score=0.94` + crosswalk `polygon_mode=hull`을 묶어 objective `0.6165`, lane/stop/cross F1 `0.5407 / 0.4235 / 0.6187`을 만들었다. Segment-MIL lane-head-only checkpoint에 같은 stop/cross overrides를 묶으면 objective는 `0.6167526016`, lane/stop/cross F1은 `0.5480 / 0.4184 / 0.6185`가 된다. Transplanting only the segment-MIL lane head onto the original stop/cross base reaches objective `0.6176617972`, lane/stop/cross F1 `0.5480 / 0.4235 / 0.6187`; adding flip-centerline averaging lifts it to objective `0.6216194906`, lane/stop/cross F1 `0.5577 / 0.4235 / 0.6187`; adding the fixed crosswalk-mask lane gate is the current objective-best at `0.6230558331`, lane/stop/cross F1 `0.5628 / 0.4235 / 0.6187`. The artifact-only projection-competition + current lane/crosswalk task-balance replay reaches lane/stop/cross F1 `0.5628 / 0.5164 / 0.6187`, mean/min F1 `0.5659 / 0.5164`, but still leaves lane and stop-line below `0.60`.
- `core_centerline_refine_row_scan_tangent_support_gate`는 low-support centerline pixels를 opt-in으로 제외했지만 exact val128에서 tangent-link와 사실상 같았다. Objective는 `0.6187189441`, lane/stop/cross F1은 `0.5633 / 0.4483 / 0.5854`, lane TP/FP/FN은 `1121 / 469 / 1269`이다. Support gate가 meaningful FP/TP trade-off를 만들지 못했으므로 broader-val512로 확장하지 않는다.
- `core_centerline_refine_row_scan_tangent_segment_mil`은 GT lane segment 위 centerline logits를 positive-only로 올렸지만 exact val128 gate를 넘지 못했다. Epoch2 objective는 `0.6161803030`, lane/stop/cross F1은 `0.5655 / 0.4348 / 0.5854`, lane TP/FP/FN은 `1161 / 555 / 1229`이다. Lane F1은 tangent-link보다 조금 높지만 lane FP와 stop-line regression이 같이 늘어 broader-val512로 확장하지 않는다.
- `core_centerline_refine_row_scan_tangent_segment_mil_lane_head_only`는 segment-MIL 신호를 lane head에만 적용해 stop/cross head corruption을 막는 retention schedule을 시험했다. Epoch2 exact objective는 `0.6193428422`, lane/stop/cross F1은 `0.5660 / 0.4483 / 0.5926`, lane TP/FP/FN은 `1162 / 554 / 1228`이다. Broader-val512 full-checkpoint composite replay with stop threshold + hull gives objective `0.6167526016`, lane/stop/cross F1 `0.5480 / 0.4184 / 0.6185`. Transplanting only its lane head onto the original stop/cross base is slightly cleaner at objective `0.6176617972`, flip-centerline TTA lifts that same runtime composite to objective `0.6216194906`, and the fixed crosswalk-mask lane gate reaches objective `0.6230558331`, lane/stop/cross F1 `0.5628 / 0.4235 / 0.6187`, but still not all-task success.
- Stricter lane geometry filters on the lane-head transplant checkpoint do not recover lane `0.6` even on exact val128. Baseline lane F1 is `0.5660`, TP/FP/FN `1162 / 554 / 1228`; the best stricter candidate by objective, `lane_bbox_area_8192`, falls to lane F1 `0.5580`, TP/FP/FN `1097 / 445 / 1293`. Do not broaden bbox area/aspect-only filtering.
- `core_centerline_refine_row_scan_segment_continuity_lane_head_only`는 segment-continuity side-negative loss를 lane head에만 적용해 stop/cross retention을 보존하는지 봤지만 exact val128 gate를 넘지 못했다. Epoch2 objective는 `0.6157263716`, lane/stop/cross F1은 `0.5590 / 0.4483 / 0.5926`, lane TP/FP/FN은 `1141 / 551 / 1249`이다. Tangent-link objective `0.6187`, segment-MIL lane-head-only `0.6193`, segment-continuity full-head lane F1 `0.5594`를 모두 넘지 못해 broader-val512로 확장하지 않는다.
- `core_centerline_refine_row_scan_tangent_no_aug`는 `row_scan_tangent` 계약을 유지한 채 stage-4 train augmentation만 껐지만 exact val128 gate를 넘지 못했다. Epoch2 objective는 `0.6093231419`, lane/stop/cross F1은 `0.5595 / 0.4298 / 0.5476`, lane TP/FP/FN은 `1109 / 465 / 1281`, stop-line TP/FP/FN은 `26 / 35 / 34`다. Tangent-link exact reference `0.6187`, `0.5633 / 0.4483 / 0.5854`보다 모두 낮아 broader-val512로 확장하지 않는다.
- `row-scan-stop-pca-integration`은 row-scan lane vectorizer와 stop-line `mask=0.80`, `min_instance_score=0.94` override를 같은 checkpoint replay에 묶었다. exact val128 lane/stop/cross F1은 `0.5522 / 0.4364 / 0.5854`, broader-val512는 `0.5279 / 0.4235 / 0.5854`이고 broader objective는 `0.6019`다. objective-only partial-positive지만 task별 F1 목표와 PCA-only stop-line reference를 못 넘으므로 success/default가 아니다.
- `lane-row-scan-geometry-guard`는 length/bottom/gap/dx/turn-angle guard를 exact val128에서 비교했다. best lane F1은 `row_gap24_row_dx12`의 `0.5526`으로 기존 row-scan `0.5522` 대비 noise-level이고 FP가 늘었다. turn-angle guard는 FP를 줄였지만 TP를 더 잃어 best `0.5365`에 그쳤다.
- `lane-row-scan-residual-buckets`는 broader-val512 row-scan residual lane TP/FP/FN `4153 / 2105 / 5324`를 확인했다. FN은 left `46.9%`, truncated `<0.50` `27.9%`, aspect `>=3` `65.9%`에 몰리고, FP는 side `74.8%`와 right `40.6%` 비중이 높다.
- `core_centerline_refine_residual_local_separation`은 residual-risk GT core positive와 local ring negative margin을 같이 줬다. exact val128 epoch2 lane/stop/cross F1은 `0.5476 / 0.4310 / 0.5854`로 lane은 기준보다 올랐지만 stop-line과 objective `0.6085`가 기준 `0.6089`보다 낮다. broader-val512로 확장하지 않는다.
- `core_centerline_refine_bce_focus`는 lane F1을 broader-val512 `0.5101 -> 0.5344`로 올렸지만, centerline-core pixel F1은 `0.5680`으로 기준선보다 낮고 stop-line/crosswalk가 내려갔다.
- BCE-focus + PCA stop-line decoder integration audit best는 lane/stop/cross F1 `0.5344 / 0.4583 / 0.5741`로 partial-positive지만 목표 미달이다.
- BCE-focus stop-balance broader-val512는 `0.5372 / 0.4041 / 0.5812`이고 PCA replay best도 `0.5372 / 0.4528 / 0.5812`라 stop-line 병목을 못 풀었다.
- `lane-instance-evidence-validator-audit`은 row-scan lane predictions의 centerline/support/tangent map-local evidence를 feature로 뽑아 train/held-out replay를 했다. val128 held-out logistic row AUC/AP는 `0.8407 / 0.9312`지만, held-out lane F1은 `0.5691 -> 0.5749`로 `+0.0058`뿐이고 TP를 `566 -> 537`로 잃었다. row-level separability는 있지만 post-hoc threshold filter는 0.6 path가 아니다.
- `core_centerline_refine_row_scan_tangent_stability`는 tangent loss를 `0.35 -> 1.0`으로 올려 training-side instance evidence를 강화해 봤지만 exact val128에서 확장 조건을 못 만들었다. epoch2 lane/stop/cross F1은 `0.5555 / 0.4348 / 0.5854`, objective `0.6122`이고 dense lane centerline-core F1은 `0.5765`다. prior row-scan exact `0.5522 / 0.4483 / 0.5854`, objective `0.6144`, dense core `0.5729` 대비 lane/core gain은 작고 stop-line이 내려가므로 broader-val512로 확장하지 않는다.
- `core_centerline_refine_row_scan_hard_negative_margin`은 current high-confidence predicted centerline 후보 중 GT support 밖 top-k만 margin 아래로 누르는 dynamic hard-negative loss를 시험했다. exact val128 epoch2 lane/stop/cross F1은 `0.5556 / 0.4348 / 0.5854`, objective `0.6121`, lane TP/FP/FN `1104 / 480 / 1286`이다. prior row-scan objective `0.6144`보다 낮고 stop-line이 `0.4483 -> 0.4348`로 내려가므로 broader-val512로 확장하지 않는다.
- `core_centerline_refine_row_scan_centerline_focal`은 centerline BCE/Dice에 focal 항을 더했지만 exact val128 gate를 넘지 못했다. epoch2 lane/stop/cross F1은 `0.5551 / 0.4522 / 0.5854`, objective `0.6134`, lane TP/FP/FN `1103 / 481 / 1287`이다. prior row-scan보다 lane/stop-line task F1은 아주 조금 높지만 objective `0.6144`를 넘지 못하므로 broader-val512로 확장하지 않는다.
- `core_centerline_refine_row_scan_risk_bucket_sampler`는 residual FN bucket(left/truncated/high-aspect)을 `lane_risk` task-positive sampler로 직접 노출했지만 exact val128 gate를 넘지 못했다. epoch2 lane/stop/cross F1은 `0.5494 / 0.3964 / 0.5478`, objective `0.5957`, lane TP/FP/FN `1081 / 464 / 1309`이다. lane FP는 줄어도 recall이 올라가지 않고 stop-line/crosswalk retention이 깨지므로 broader-val512로 확장하지 않는다.
- `core_centerline_refine_row_scan_row_anchor_recall`은 visible GT anchor-row x 위치 주변 centerline logit에 positive pressure를 줬지만 exact val128 gate를 넘지 못했다. epoch2 lane/stop/cross F1은 `0.5512 / 0.4310 / 0.5854`, objective `0.6106`, lane TP/FP/FN `1141 / 609 / 1249`이다. Lane TP는 늘었지만 FP가 크게 늘고 stop-line도 내려가므로 positive-anchor-only loss는 broader-val512로 확장하지 않는다.
- `core_centerline_refine_row_scan_row_anchor_contrast`는 같은 anchor row에서 GT-near band 밖 high-logit negative를 같이 누르도록 바꿨다. exact val128 epoch2 lane/stop/cross F1은 `0.5559 / 0.4483 / 0.5854`, objective `0.6134`, lane TP/FP/FN `1133 / 553 / 1257`이다. Positive-only보다 FP와 stop-line은 회복했지만 prior row-scan objective `0.6144`를 넘지 못하므로 broader-val512로 확장하지 않는다.
- `core_centerline_refine_row_scan_interlane_gap_margin`은 같은 anchor row에서 인접 GT lane 사이 gap을 negative margin으로 눌러 over-link/side FP를 줄이는 축을 시험했다. exact val128 epoch2 lane/stop/cross F1은 `0.5560 / 0.4348 / 0.5854`, objective `0.6122`, lane TP/FP/FN `1104 / 477 / 1286`, stop-line TP/FP/FN `25 / 30 / 35`이다. Lane F1은 prior row-scan `0.5522`보다 높지만 stop-line과 objective가 낮아 broader-val512로 확장하지 않는다.
- `core_centerline_refine_row_scan_segment_continuity_contrast`는 visible GT lane segment 사이 centerline positive와 양옆 normal-offset negative margin을 같이 주는 training-side continuity loss를 시험했다. exact val128 epoch2 lane/stop/cross F1은 `0.5594 / 0.4348 / 0.5854`, objective `0.6127`, lane TP/FP/FN `1142 / 551 / 1248`, stop-line TP/FP/FN `25 / 30 / 35`이다. Lane F1은 최근 lane-loss 후보 중 가장 높지만 stop-line regression과 objective miss 때문에 broader-val512로 확장하지 않는다.
- `core_centerline_refine_row_scan_tangent_balanced_retain`은 tangent-link lane replay 위에서 stop-line/crosswalk retention을 loss-weight balance로 회복하려 했지만 exact val128 gate를 넘지 못했다. epoch2 objective는 `0.6147`, lane/stop/cross F1은 `0.5607 / 0.4310 / 0.5854`, lane TP/FP/FN `1117 / 477 / 1273`, stop-line TP/FP/FN `25 / 31 / 35`이다. Crosswalk task-best는 epoch1 `0.6748`까지 올랐지만 final best objective와 lane/stop-line은 tangent-link exact reference `0.6187`, `0.5633 / 0.4483 / 0.5854`보다 낮으므로 broader-val512로 확장하지 않는다.
- support map을 단순 대체하거나 직접 residual input으로 넣는 방식은 이미 negative evidence가 있으므로 반복하지 않는다.
- `lane-row-scan-link-oracle-audit`은 same-checkpoint read-only oracle로 tangent/link axis를 분리했다. Exact val128에서 `gt_tangent_axis`는 lane F1 `0.5633 -> 0.5611`로 개선이 없고, `gt_centerline_core`는 `0.6778`, lane TP/FP/FN `1243 / 35 / 1147`까지 올라간다. Tangent/linking direction이 아니라 predicted centerline core coverage/quality가 dominant lane headroom이다.
- `core_centerline_refine_row_scan_tangent_instance_balance`는 lane instance별 centerline-core positive loss를 균등화했지만 exact val128에서 objective `0.6161312489`, lane/stop/cross F1 `0.5654 / 0.4310 / 0.5854`에 그쳤다. Tangent-link보다 lane TP는 늘었지만 FP도 `469 -> 581`로 늘고 stop-line이 내려가므로 broader-val512로 확장하지 않는다.
- `core_centerline_refine_row_scan_tangent_instance_validator`는 predicted centerline component를 valid/invalid candidate로 학습하는 새 opt-in head/loss/readout contract다. Contract wiring과 smoke는 성공했지만 exact val128 epoch2 objective는 `0.6142642145`, lane/stop/cross F1은 `0.5523 / 0.4522 / 0.5854`, lane TP/FP/FN은 `1099 / 491 / 1291`이다. Stop-line은 tangent-link보다 조금 높지만 lane이 tangent-link `0.5633`, row-distribution `0.5659`, segment-MIL lane-head-only `0.5660`보다 낮아 broader-val512로 확장하지 않는다.
- `core_centerline_refine_row_scan_tangent_soft_ignore_band`는 soft-shell pixels를 core-centerline negative에서 제외했지만 exact val128에서 objective `0.6088389162`, lane/stop/cross F1 `0.5360 / 0.4348 / 0.5854`로 크게 낮았다. FP reduction보다 TP/FN regression이 커서 broader-val512로 확장하지 않는다.
- `core_centerline_refine_row_scan_tangent_row_distribution`은 row별 GT core pixels를 column distribution으로 정규화하는 opt-in loss를 시험했다. Exact val128 epoch2 objective는 `0.6174723013`, lane/stop/cross F1은 `0.5659 / 0.4483 / 0.5854`, lane TP/FP/FN은 `1127 / 466 / 1263`이다. Tangent-link보다 lane은 아주 조금 높고 stop/cross는 보존됐지만 objective가 낮고 segment-MIL lane-head-only best를 넘지 못해 broader-val512로 확장하지 않는다.
- `core_centerline_refine_row_scan_tangent_segment_mil_row_distribution`은 segment-MIL positive evidence와 row-distribution pressure를 lane-head-only retention schedule에 결합했지만 exact val128 gate를 넘지 못했다. Epoch2 objective는 `0.6186643159`, lane/stop/cross F1은 `0.5621 / 0.4483 / 0.5926`, lane TP/FP/FN은 `1152 / 557 / 1238`이다. Segment-MIL lane-head-only best와 row-distribution-only lane F1을 모두 넘지 못해 broader-val512로 확장하지 않는다.
- `core_centerline_refine_row_scan_tangent_upper_trunk`는 현재 tangent-link 계약을 유지한 채 `lane_family_plus_upper_trunk`만 작은 LR로 열었지만 exact val128 gate를 넘지 못했다. Epoch2 objective는 `0.6169717875`, lane/stop/cross F1은 `0.5597 / 0.4561 / 0.5854`, lane TP/FP/FN은 `1115 / 479 / 1275`, stop-line TP/FP/FN은 `26 / 28 / 34`이다. Stop-line은 tangent-link보다 조금 높지만 lane/objective가 segment-MIL lane-head-only, row-distribution-only, tangent-link references를 넘지 못해 broader-val512로 확장하지 않는다.
- `core_centerline_refine_row_scan_tangent_dilated_context`는 centerline branch receptive field를 넓혔지만 exact objective `0.6155`, lane/stop/cross F1 `0.5617 / 0.4348 / 0.5854`에 그쳐 tangent-link and segment-MIL references를 넘지 못했다. Dilation/context widening alone is closed.
- `core_centerline_refine_row_scan_tangent_support_conditioned`는 strong support map을 detached feature로 centerline refinement에 넣었지만 exact objective `0.6152`, lane/stop/cross F1 `0.5607 / 0.4348 / 0.5854`에 그쳤다. Support-conditioned centerline refinement alone is closed.
- `core_centerline_refine_row_scan_tangent_flip_consistency`는 train-time horizontal flip consistency regularizer를 `lane_flip_consistency_weight=0.25`로 추가했지만 exact val128 gate를 넘지 못했다. Epoch2 objective는 `0.5991166581`, lane/stop/cross F1은 `0.5542 / 0.3966 / 0.5548`이고 skipped step은 `0`이다. Runtime은 안정적이지만 tangent-link reference `0.6187`, `0.5633 / 0.4483 / 0.5854`보다 모두 낮으므로 broader-val512로 확장하지 않는다.
- `lane_endpoint_extend` readout은 decoded lane polyline의 top/bottom endpoint를 fixed raw-pixel distance로 연장했지만 exact val128에서 baseline보다 낮았다. Best extension by objective는 `top32`이고 objective `0.5998086929`, lane/stop/cross F1 `0.5503 / 0.4483 / 0.5854`다. Baseline tangent-link `0.6187165763`, `0.5633 / 0.4483 / 0.5854`보다 낮아 broader-val512로 확장하지 않는다.
- 새 lane training/readout axis는 stop-line plan과 섞지 않고 별도 short run으로 보되, row-scan/tangent-link 후처리 cost sweep / GT tangent-axis oracle sweep / instance-balanced positive centerline weight-only / instance-validator `weight=0.35`, threshold `0.45` / soft-instance centerline shell auxiliary-only / soft-band ignore threshold-only / row-distribution-only / segment-MIL + row-distribution 조합 / component-limited row-scan readout / hard-negative-only / tangent-only / residual-risk-weight-only / centerline-focal-only / risk-bucket sampler-only / row-anchor-positive-only / row-anchor-contrast-only / inter-lane-gap-margin-only / segment-continuity-contrast-only / segment-continuity + lane-head-only retention / segment-MIL-positive-only / retention-balance loss-weight-only / lane-head-only retention schedule family / upper-trunk capacity/freeze-scope-only / centerline-branch dilation-only / support-conditioned centerline refinement-only / flip-consistency regularizer-only / fixed-distance endpoint-extension readout / logistic-gate top-K safety fallback / anchor-offset auxiliary-only / legacy row-head fallback-union-only는 닫힌 축으로 취급한다.

성공 기준:

- lane F1이 올라야 하고, FP 감소만으로 recall이 무너지는 개선은 실패다.
- broader-val comparison grid에서 긴 실제 차선이 빠지는 장면이 늘면 실패다.

다음 실행:

- `exp/lane-family-f1/lane-centerline-core-calibration`, `exp/lane-family-f1/lane-bce-stopline-pca-integration`, `exp/lane-family-f1/lane-bce-stopline-balance`는 partial/negative evidence로 보관한다.
- `exp/lane-family-f1/stopline-vector-proposal-readout`은 learned query-vector proposal negative evidence로 보관한다.
- `exp/lane-family-f1/stopline-selector-component-gate`는 selector-map component gate negative evidence로 보관한다.
- `exp/lane-family-f1/stopline-selector-dense-audit`은 mask/row/x signal은 남아 있지만 selector/proposal readout contract가 약하다는 read-only evidence로 보관한다.
- `exp/lane-family-f1/stopline-rowx-proposal-readout`은 row/x projection-only span proposal negative evidence로 보관한다.
- `exp/lane-family-f1/stopline-selector-rowx-band-contract`는 simple rowx-band selector target + selector gate negative evidence로 보관한다.
- `exp/lane-family-f1/stopline-rowx-candidate-consistency-audit`은 row/x projection agreement가 val128에서는 좋아 보였지만 val512 held-out에서 PCA/projection references를 못 넘는다는 read-only negative evidence로 보관한다.
- `exp/lane-family-f1/stopline-normal-support-recenter-readout`은 local mask support로 center normal-coordinate를 재조정해도 exact val128에서 FP가 커져 baseline보다 낮아진다는 read-only negative evidence로 보관한다.
- `exp/lane-family-f1/stopline-raw-edge-recenter-readout`은 raw-image edge/contrast로 center normal-coordinate를 재조정하면 exact val128에서 TP가 줄고 FP가 커져 baseline보다 크게 낮아진다는 read-only negative evidence로 보관한다.
- `exp/lane-family-f1/stopline-projcomp-flip-composite`는 known best lane/cross partial-positive와 projection-competition stop-line readout을 합쳐도 broader task F1이 `0.5628 / 0.5164 / 0.6187`에 그친다는 artifact-only task-balance evidence로 보관한다.
- `exp/lane-family-f1/restore-stopline-projection-tools` restores the fragment-union / projection-split / projection-competition probe scripts and tests onto current develop so the task-balance reference can be reproduced without old worktrees. This is tooling only; it does not reopen projection-competition sweeps.
- `exp/lane-family-f1/stopline-angle-mask-extent-diagnostic`은 GT center upper-bound에서 angle-anchored mask extent가 half-length scalar보다 낫지만 broader-val512 목표에는 아직 모자란다는 read-only evidence로 보관한다.
- `exp/lane-family-f1/stopline-centerline-endpoint-offset`은 endpoint-delta target/readout negative evidence로 보관한다.
- `exp/lane-family-f1/stopline-heatmap-geometry-support`는 heatmap-support geometry target-fill negative evidence로 보관한다.
- `exp/lane-family-f1/stopline-row-center-aux`는 row-center auxiliary-only negative evidence로 보관한다.
- `exp/lane-family-f1/stopline-readout-component-audit`은 dense mask/center signal이 남아 있어도 단순 component PCA/anchor swap만으로는 부족하다는 read-only evidence로 보관한다.
- `exp/lane-family-f1/stopline-fit-far-visual-audit`은 high-signal FN bucket의 visual evidence로 보관한다.
- `exp/lane-family-f1/stopline-local-component-extraction`은 local center/selector/fused window extraction negative evidence로 보관한다.
- `exp/lane-family-f1/stopline-component-split-readout`은 pairwise component split readout negative evidence로 보관한다.
- `exp/lane-family-f1/stopline-component-proposal-readout`은 component별 proposal 제한이 recall보다 FP를 더 늘린다는 read-only negative evidence로 보관한다.
- `exp/lane-family-f1/stopline-mask-ridge-readout`은 predicted mask ridge/medial geometry가 tiny exact gain만 만들고 PCA/angle-mask reference를 못 넘는 read-only negative evidence로 보관한다.
- `exp/lane-family-f1/stopline-line-support-readout`은 segment line-support reranking이 baseline은 조금 넘지만 PCA/angle-mask/head-merge reference를 못 넘는 read-only partial/negative evidence로 보관한다.
- `exp/lane-family-f1/stopline-candidate-consensus-readout`은 candidate agreement/consensus-only selector가 baseline은 조금 넘지만 stronger stop-line references를 못 넘는 read-only partial/negative evidence로 보관한다.
- `exp/lane-family-f1/stopline-lane-context-readout`은 row-scan-tangent lane context가 baseline은 조금 넘지만 stronger stop-line references를 못 넘는 read-only partial/negative evidence로 보관한다.
- `exp/lane-family-f1/stopline-crosswalk-context-readout`은 predicted crosswalk proximity가 stop-line candidate selector로는 baseline보다 나쁘다는 read-only negative evidence로 보관한다.
- `exp/lane-family-f1/stopline-detector-context-fp-audit`은 predicted traffic-light/sign proximity가 stop-line candidate selector로는 recall을 크게 죽여 baseline보다 훨씬 낮다는 read-only negative evidence로 보관한다.
- `exp/lane-family-f1/stopline-candidate-validator-audit`은 hard-negative proposal ranking loss-only negative evidence로 보관한다.
- `exp/lane-family-f1/stopline-candidate-feature-audit`은 scalar candidate-row feature separability evidence로 보관한다.
- `exp/lane-family-f1/stopline-candidate-rich-feature-audit`은 selector/mask/center local window features가 candidate-row separability를 강화한다는 read-only evidence로 보관한다.
- `exp/lane-family-f1/stopline-candidate-rich-validator-replay`는 rich row signal이 held-out task selection으로는 약하게만 옮겨져 PCA reference를 넘지 못한다는 partial/negative evidence로 보관한다.
- `exp/lane-family-f1/stopline-component-topology-audit`은 predicted-mask component topology columns를 rich-validator replay에 더해도 held-out task F1이 `0.4126`으로 selector-only replay와 PCA reference를 못 넘는다는 read-only negative evidence로 보관한다.
- `exp/lane-family-f1/stopline-sample-tree-gate-audit`은 gap4/top50 sample gate를 shallow tree로 비선형화해도 held-out surrogate selection F1 `0.5543`으로 logistic `0.5732`보다 낮다는 read-only negative evidence로 보관한다.
- `exp/lane-family-f1/stopline-candidate-validator-calibrated-replay`는 model-side validator checkpoint의 direct collapse를 product/fallback replay로 일부 회복해도 PCA/angle-mask reference를 넘지 못한다는 negative evidence로 보관한다.
- `exp/lane-family-f1/stopline-candidate-competition-calibration`의 modern task-head merge replay는 exact objective를 `0.6196`까지 올리지만 broader-val512에서 `0.5977`로 내려가고 all-task F1 0.6과 stop-line PCA/angle-mask reference를 못 넘는 composition partial/negative evidence로 보관한다.
- `exp/lane-family-f1/task-head-merge-segment-mil-rank-stop`은 segment-MIL lane/cross retention과 proposal-rank stop-line head를 합치면 exact objective는 `0.6248`까지 오르지만 broader-val512 stop-line F1이 `0.3976`으로 내려간다는 composition partial/negative evidence로 보관한다.
- `exp/lane-family-f1/lane-head-transplant-original-stop-pca`는 segment-MIL lane head만 original stop/cross base에 이식하면 broader objective `0.6177`까지 오르지만 lane/stop-line F1 `0.5480 / 0.4235`라 task-F1 0.6 success는 아니라는 composition partial-positive evidence로 보관한다.
- `exp/lane-family-f1/lane-flip-tta-audit`은 current transplanted composite 위에서 flip-centerline averaging이 broader objective `0.6216`, lane/stop/cross F1 `0.5577 / 0.4235 / 0.6187`로 small lane gain을 만들지만 lane/stop-line 0.6 success는 아니라는 runtime/postprocess partial-positive evidence로 보관한다.
- `exp/lane-family-f1/lane-task-mask-context-gate`는 flip-centerline average 위에 fixed crosswalk-mask lane gate를 얹으면 broader objective `0.6231`, lane/stop/cross F1 `0.5628 / 0.4235 / 0.6187`로 small FP-control gain을 만들지만 lane/stop-line 0.6 success는 아니라는 runtime/postprocess partial-positive evidence로 보관한다.
- `exp/lane-family-f1/lane-fn-nearby-fp-recovery-audit`은 current flip-centerline composite의 missed GT lanes가 existing centerline evidence 또는 nearby unmatched tracks로 recover 가능한지 나눈 read-only evidence다. Broader val512에서 `1363` FNs have GT-line center mean `>=0.50`, `1698` FNs have unmatched prediction `<=120px`, and the best diagnostic no-new-FP upper-bound is lane F1 `0.7564`; production success가 아니므로 다음에는 실제 decoder/model-side recovery contract로 검증해야 한다.
- 같은 branch의 `row_scan_tangent_soft_ridge` smoke는 probability ridge peak picking이 기준선보다 나쁨을 보였다. Val4 lane F1은 `0.5899 -> 0.5429`, TP/FP/FN은 `41 / 12 / 45 -> 38 / 16 / 48`이므로 broader replay 없이 negative evidence로 보관한다.
- `exp/lane-family-f1/lane-guarded-area-rescue-readout`은 raw vectorizer에서 area-filter drop이 실제로 존재하더라도 단순 guarded rescue는 FP를 너무 많이 늘린다는 negative evidence로 보관한다. Val128 lane F1은 `0.5797 -> 0.5716`, TP/FP/FN은 `1204 / 560 / 1186 -> 1247 / 726 / 1143`이다.
- `exp/lane-family-f1/lane-area-rescue-center-score-gate`는 area-rescue candidate에 track centerline mean gate를 추가하면 exact val128은 `0.5846`까지 회복하지만 broader val512 lane F1 `0.5548`로 current broader best `0.5628`를 넘지 못한다는 weak partial/negative evidence로 보관한다.
- `exp/lane-family-f1/lane-flip-consistency-row-scan-tangent`는 train-time flip consistency regularizer가 stable하게 돌지만 exact val128 objective `0.5991`, lane/stop/cross `0.5542 / 0.3966 / 0.5548`로 tangent-link reference를 못 넘는다는 negative evidence로 보관한다.
- `exp/lane-family-f1/lane-endpoint-extension-readout`은 fixed-distance lane endpoint extension이 exact val128에서 lane TP를 잃고 FP를 늘린다는 negative evidence로 보관한다.
- `exp/lane-family-f1/stopline-candidate-instance-assignment`은 current center/selector top-k 후보 위의 direct candidate assignment loss가 exact val128 stop-line을 회복하지 못한다는 negative evidence로 보관한다.
- `exp/lane-family-f1/stopline-geometry-candidate-validator`는 predicted endpoint geometry로 candidate-validator labels를 만들면 stop-line은 일부 오르지만 objective/lane과 validator-map replay가 기준을 못 넘는 partial/negative evidence로 보관한다.
- `exp/lane-family-f1/stopline-head-only-geometry-validator`는 geometry-validator supervision을 stop-line head에만 적용해도 stop-line gain이 보존되지 않는다는 negative evidence로 보관한다.
- `exp/lane-family-f1/stopline-presence-emit-gate`는 sample-level model-side presence emit gate가 구현/런타임은 통과하지만 exact stop-line F1을 낮춘다는 negative evidence로 보관한다.
- `exp/lane-family-f1/stopline-proposal-stat-emit-gate`는 dense proposal-map statistics를 sample-level presence gate에 더해도 TP가 회복되지 않고 exact stop-line F1이 더 낮아진다는 negative evidence로 보관한다.
- `exp/lane-family-f1/stopline-gap4-baseline-rescue-replay`는 gap4/top50 sample-gated 후보를 baseline에 fallback/append해도 full val512 stop-line F1이 `0.4165` 이하라 기존 sample-gate/score-threshold/PCA reference를 못 넘는다는 negative evidence로 보관한다.
- `exp/lane-family-f1/stopline-candidate-manifest-audit`은 current-composite candidate rows/visual buckets를 traceable하게 만들었고, 후속 `stopline_negative` sampler-only short run이 exact val128 objective `0.6063`, lane/stop/cross F1 `0.5475 / 0.4561 / 0.5644`로 expansion gate를 못 넘는다는 negative evidence로 보관한다. The added failure-mode audit shows positive no-oracle `62` is larger than positive misrank `29`, so the next stop-line axis should not be selector-only.
- `exp/lane-family-f1/stopline-fragment-axis-contract` is negative evidence. Commit `6cc3f52` changed the opt-in fragment-center offset loss from 2D xy SmoothL1 to along-axis scalar SmoothL1 and aligned fragment decode with the same axis projection; commit `ae1fbc2` added metric-only low-disk probing. The completed exact val128 metric-only result was objective `0.5522`, lane/stop/cross F1 `0.5225 / 0.1905 / 0.5714`, so do not broaden or repeat this branch.
- `core_centerline_refine_row_scan_tangent_stop_center_rank_margin` is stored on the same branch as a valid but insufficient midpoint-rank loss attempt: exact val128 `0.5605 / 0.4602 / 0.5854`, objective `0.6170`, no broader expansion.
- `stopline_pred_angle_mask_extent_val128_epoch2` on that center-rank checkpoint is also negative: best readout replay stop-line F1 `0.5042`, below prior exact readout references.
- `lane60_lane_flip_tta_on_segment_mil_lane_head_only_20260512` is negative broader evidence: exact `0.5854 / 0.4364 / 0.6061`, but broader `0.5577 / 0.4184 / 0.6185` with objective `0.6207`, below the current broader best.
- `exp/lane-family-f1/stopline-center-stem`은 unused center stem을 center outputs에 연결해도 exact stop-line F1이 `0.3704`로 떨어진다는 architecture-cleanup negative evidence로 보관한다.
- `exp/lane-family-f1/lane-row-scan-tangent-no-aug`는 stage-4 train augmentation을 끄는 preprocessing/runtime 단일축이 exact val128에서도 tangent-link reference를 못 넘는다는 negative evidence로 보관한다.
- `exp/lane-family-f1/lane-row-scan-link-oracle-audit`은 GT tangent-axis replacement가 lane F1을 올리지 못하고 GT centerline-core replacement만 큰 headroom을 보인다는 read-only bottleneck evidence로 보관한다.
- `exp/lane-family-f1/lane-centerline-instance-balance`는 instance-balanced positive centerline loss가 tiny lane gain만 만들고 FP/stop-line regression을 일으킨다는 weak partial/negative evidence로 보관한다.
- `exp/lane-family-f1/lane-soft-instance-shell`은 instance-balanced soft-shell centerline auxiliary가 tiny lane gain만 만들고 objective/stop-line/segment-MIL reference를 넘지 못한다는 weak partial/negative evidence로 보관한다.
- `exp/lane-family-f1/lane-centerline-soft-ignore`는 soft-shell negative masking이 lane FP를 줄여도 TP를 더 크게 잃는다는 negative evidence로 보관한다.
- `exp/lane-family-f1/lane-row-distribution-centerline`은 row-wise centerline distribution loss가 exact lane을 아주 조금 올려도 objective/segment-MIL gate를 넘지 못한다는 weak partial/negative evidence로 보관한다.
- `exp/lane-family-f1/lane-segment-mil-row-distribution`은 segment-MIL positive evidence와 row-distribution pressure를 합쳐도 exact segment-MIL lane-head-only best를 넘지 못한다는 negative evidence로 보관한다.
- `exp/lane-family-f1/stopline-overlap-fit-oracle`은 GT mask/vectorizer headroom과 GT-overlap oracle recovery evidence로 보관한다.
- `exp/lane-family-f1/stopline-component-core-fit`, `exp/lane-family-f1/stopline-component-cleanup-fit`, `exp/lane-family-f1/stopline-component-split-fit`은 oracle headroom을 non-oracle trimming/splitting으로 회수하지 못한 negative evidence로 보관한다.
- `exp/lane-family-f1/lane-vectorizer-recovery-audit`은 vectorizer headroom / predicted centerline bottleneck evidence로 보관한다.
- `exp/lane-family-f1/lane-centerline-core-width3`은 centerline target-width-only negative evidence로 보관한다.
- `exp/lane-family-f1/lane-centerline-error-buckets`은 missed-centerline bucket evidence로 보관한다.
- `exp/lane-family-f1/lane-centerline-side-bucket-weight`는 side-band BCE-only partial/negative evidence로 보관한다.
- `exp/lane-family-f1/lane-centerline-side-margin`은 side-band probability margin partial/negative evidence로 보관한다.
- `exp/lane-family-f1/lane-centerline-geometry-risk-recall`은 side/truncated/near-vertical geometry-risk recall-only partial/negative evidence로 보관한다.
- `exp/lane-family-f1/lane-centerline-geometry-risk-local-tversky`는 geometry-risk local false-positive penalty partial/negative evidence로 보관한다.
- `exp/lane-family-f1/lane-centerline-negative-margin`은 negative-pixel probability margin partial/negative evidence로 보관한다.
- `exp/lane-family-f1/lane-centerline-support-bridge`는 support-bridge/closing-only postprocess negative evidence로 보관한다.
- `exp/lane-family-f1/lane-centerline-endpoint-coverage`는 endpoint-only coverage loss negative evidence로 보관한다.
- `exp/lane-family-f1/lane-row-scan-vectorizer`는 broader-val lane partial-positive로 develop에 opt-in 승격했다.
- `exp/lane-family-f1/row-scan-stop-pca-integration`은 row-scan lane gain과 stop-line PCA-threshold override를 묶어도 task별 F1 목표에는 못 미친다는 integration partial/negative evidence로 보관한다.
- `exp/lane-family-f1/lane-row-scan-geometry-guard`는 row-scan micro-guard negative evidence로 보관한다.
- `exp/lane-family-f1/lane-row-scan-residual-buckets`는 row-scan 후 residual이 left/truncated/high-aspect FN과 side/right FP에 남는다는 read-only evidence로 보관한다.
- `exp/lane-family-f1/lane-residual-local-separation`은 lane partial-positive지만 stop-line/objective regression 때문에 negative evidence로 보관한다.
- `exp/lane-family-f1/lane-instance-evidence-validator-audit`은 row-scan instance map-local evidence가 row-level로는 분리되지만 held-out task F1 gain이 noise-level이라는 read-only evidence로 보관한다.
- `exp/lane-family-f1/lane-tangent-stability-row-scan`은 tangent-loss-only training-side reinforcement가 dense/lane gain을 조금 만들지만 stop-line/objective regression을 일으킨다는 partial/negative evidence로 보관한다.
- `exp/lane-family-f1/lane-row-scan-hard-negative-margin`은 dynamic top-k FP suppression이 tiny lane FP reduction만 만들고 row-scan objective/stop-line retention을 넘지 못한다는 partial/negative evidence로 보관한다.
- `exp/lane-family-f1/lane-row-scan-centerline-focal`은 centerline-focal-only calibration이 tiny task-F1 gain은 만들지만 row-scan objective gate를 넘지 못한다는 partial/negative evidence로 보관한다.
- `exp/lane-family-f1/lane-row-scan-risk-bucket-sampler`은 residual-risk lane oversampling이 lane recall을 못 올리고 stop-line/crosswalk retention을 깨는 negative evidence로 보관한다.
- `exp/lane-family-f1/lane-row-anchor-recall`은 direct anchor-row positive pressure가 lane TP를 늘려도 FP와 stop-line regression 때문에 row-scan gate를 넘지 못한다는 negative evidence로 보관한다.
- `exp/lane-family-f1/lane-row-anchor-contrast`는 anchor-row local negative pressure가 positive-only FP를 줄이지만 objective gate를 넘지 못하는 partial/negative evidence로 보관한다.
- `exp/lane-family-f1/lane-interlane-gap-margin`은 row-scan inter-lane gap negative margin이 tiny lane F1 gain을 만들지만 objective/stop-line gate를 넘지 못한다는 partial/negative evidence로 보관한다.
- `exp/lane-family-f1/lane-segment-continuity-contrast`는 segment-level continuity positive/side-negative loss가 lane F1을 더 올려도 row-scan objective/stop-line gate를 넘지 못한다는 partial/negative evidence로 보관한다.
- `exp/lane-family-f1/lane-segment-continuity-lane-head-retain`은 segment-continuity side-negative loss를 lane-head-only freeze와 묶어도 tangent-link/segment-MIL lane-head-only gate를 넘지 못한다는 negative evidence로 보관한다.
- `exp/lane-family-f1/lane-row-scan-tangent-link`은 tangent-guided row-cluster linking이 broader lane/objective를 올리지만 stop-line/crosswalk 목표를 풀지 못한다는 partial-positive evidence로 보관한다.
- `exp/lane-family-f1/lane-row-scan-component-tangent`는 connected-component 내부로 tangent row-scan을 제한하면 over-link보다 TP 손실이 더 커진다는 negative evidence로 보관한다.
- `exp/lane-family-f1/lane-tangent-support-gate`는 row-scan tangent readout에서 low-support centerline pixels를 제외해도 exact metric이 사실상 변하지 않는 no-op/negative evidence로 보관한다.
- `exp/lane-family-f1/lane-row-scan-tangent-balanced-retain`은 crosswalk task-best가 살아도 loss-weight balance만으로 tangent-link objective와 stop-line을 보존하지 못한다는 training-side negative evidence로 보관한다.
- `exp/lane-family-f1/lane-segment-mil-broader-composite`는 segment-MIL lane-head-only exact best가 broader objective를 조금 올리지만 stop-line을 회복하지 못한다는 broader partial-positive/negative evidence로 보관한다.
- `exp/lane-family-f1/lane-row-scan-tangent-upper-trunk`는 upper-trunk capacity/freeze-scope-only가 stop-line을 아주 조금 올려도 lane/objective를 기존 exact references보다 낮춘다는 negative evidence로 보관한다.
- `exp/lane-family-f1/lane-centerline-dilated-context`는 centerline branch dilation/context widening이 exact tangent-link and segment-MIL references를 못 넘는다는 architecture negative evidence로 보관한다.
- `exp/lane-family-f1/lane-support-conditioned-centerline`은 strong support map을 detached centerline refinement feature로 써도 exact lane/objective/stop-line retention을 못 올린다는 architecture negative evidence로 보관한다.
- `exp/lane-family-f1/lane-instance-oracle-selection-audit`은 current row-scan/flip candidate set이 oracle TP-only selection으로 full lane F1 `0.6457`까지 갈 수 있지만 learned/post-hoc gate는 heldout `0.5700`에 그친다는 read-only selector-headroom evidence로 보관한다.
- `exp/lane-family-f1/lane-row-scan-duplicate-suppression`은 same-schema row-scan duplicate suppression이 broader val512에서 FP를 하나만 줄인 no-op/negative evidence로 보관한다.
- `exp/lane-family-f1/lane-sample-topk-cap-audit`은 per-sample top5 logistic cap이 broader full lane F1을 `0.5618`까지만 올리고 TP를 잃는 weak partial/negative evidence로 보관한다.
- `exp/lane-family-f1/lane-positive-core-flip-consistency`는 flip-consistency pressure를 GT centerline-core positive pixels로 제한해도 global flip-consistency regression을 회수하지 못한다는 negative evidence로 보관한다.
- row-scan visual audit artifact는 `analysis_exports/row_scan_visual_compare_epoch2/row_scan_component_comparison_grid.png`와 `manifest.json`이다. stop-line fit-far visual audit artifact는 `analysis_exports/stopline_fit_far_visual_audit_val512_epoch2/stopline_fit_far_bucket_grid.png`와 `manifest.json`이다. 판정은 partial-pass/caution이므로, 다음 lane 한 축은 row-scan/tangent-link 후처리 cost sweep, GT tangent-axis oracle/cost sweep, row-scan + stop-line PCA threshold-only integration, component-limited tangent row-scan readout, residual-risk local loss weight 조절, instance-balanced positive centerline weight-only, soft-instance centerline shell auxiliary-only, soft-band ignore threshold-only, row-distribution-only, segment-MIL + row-distribution 조합, upper-trunk capacity/freeze-scope-only, post-hoc instance evidence threshold, oracle TP-only selector replay, duplicate-suppression distance sweep, sample top-k cap sweep, tangent-loss-only 강화, dynamic hard-negative margin-only, centerline-focal-only, risk-bucket sampler-only, row-anchor-positive-only, row-anchor-contrast-only, inter-lane gap margin-only, segment-continuity-contrast-only, segment-continuity + lane-head-only retention, retention-balance loss-weight-only, flip-consistency mask/weight-only가 아니라 predicted centerline evidence를 더 명시적인 training-side instance 단위로 안정화하거나 stop-line/crosswalk retention을 같이 보는 contract다. stop-line을 재개한다면 cold validator/map-mixing/current top-k assignment loss가 아니라 후보 생성/readout contract를 바꿔야 한다. PCA threshold/top-k, stop-line weight-only, component split, center-cell geometry-mask, half-length inference-scale, half-length loss-weight-only, log-target-only, learned query-vector proposal-only, selector-map component gate/threshold-only, row/x span proposal-only, rowx-band selector target + selector component gate, 단순 predicted center/selector/max proposal + angle-mask extent readout, mask-ridge readout-only, top-k line-support reranking, candidate agreement/consensus-only, score/length-only candidate pool filter, scalar/rich candidate-row classifier as production, rich held-out threshold replay as production, sample-gate baseline rescue replay as production, hard-negative proposal ranking loss-only, direct/cold validator gate, calibrated validator map-mixing replay, current top-k hard-negative auxiliary, current top-k candidate-assignment loss-only, endpoint-delta direct decode, heatmap-support geometry fill, row-center auxiliary-only, no-anchor PCA/anchor swap-only, local center/selector/fused window extraction, append-top2, point-pair component split readout, row-scan length/bottom/gap/dx/turn-angle guard, row-scan/tangent-link cost sweep, GT tangent-axis oracle/cost sweep, component-limited tangent row-scan, lane support-substitution, lane threshold-only sweep, semantic attr oracle, duplicate-suppression distance sweep, sample top-k cap sweep, lane target-width-only widening, side-band BCE-only, side-band margin-only, geometry-risk recall-only, geometry-risk local-Tversky-only, negative-pixel margin-only, support-bridge/closing-only, endpoint-only coverage, residual-risk local separation weight-only, post-hoc row-scan evidence threshold, oracle TP-only selector as production, tangent-loss-only reinforcement, dynamic hard-negative margin-only, centerline-focal-only, risk-bucket sampler-only, row-anchor-positive-only, row-anchor-contrast-only, inter-lane-gap-margin-only, segment-continuity-contrast-only, segment-continuity + lane-head-only retention, retention-balance loss-weight-only, instance-balanced positive centerline weight-only, soft-instance centerline shell auxiliary-only, soft-band ignore threshold-only, row-distribution-only, segment-MIL + row-distribution 조합, upper-trunk capacity/freeze-scope-only, flip-consistency mask/weight-only, lane FP-repair oracle as production, single-feature repairability threshold/gate, duplicate append/translation-radius replay는 반복하지 않는다.
- val128 dense-map PR, exact task F1, broader-val replay, visual audit 순서로 통과시킨다.

## 7. Gate 4: crosswalk retention to 0.6

목적:

- crosswalk F1을 `0.5854` 근처에서 안정적으로 0.6 이상으로 넘긴다.

후보:

- exact val128 threshold probe는 완료됐다. `cross_mask=0.40`, `cross_area=32`가 crosswalk F1을 `0.5854 -> 0.6027`로 올렸다.
- top objective variant는 `lane_obj_0.35__cross_mask_0.40__cross_area_32`이고 objective `0.6115270643`, lane/stop/cross F1 `0.5326 / 0.4483 / 0.6027`이다.
- crosswalk-only candidate는 `lane_obj_0.45__cross_mask_0.40__cross_area_32`이고 objective `0.6098888876`, lane/stop/cross F1 `0.5267 / 0.4483 / 0.6027`이다.
- broader-val512 crosswalk-only replay는 실패했다. baseline lane/stop/cross F1은 `0.5101 / 0.4083 / 0.5854`, candidate는 `0.5101 / 0.4083 / 0.5845`다.
- broader-val512 crosswalk-only shape sweep은 object/mask/component-area/polygon-area/aspect/top-k `64`개 variant를 확인했지만 crosswalk F1 `>=0.60` variant가 `0/64`였다. Best crosswalk F1은 `cross_aspect_2.0`의 `0.5960`, TP/FP/FN `239 / 168 / 156`이고, best objective는 `cross_mask_0.70`의 objective `0.5970`, crosswalk F1 `0.5887`이다.
- row-scan tangent + stop-line threshold + `cross_aspect_2.0` composite에서도 crosswalk는 `0.5960`으로 유지됐지만 stop-line은 `0.4235`, lane은 `0.5407`라 all-task target에는 실패했다.
- `crosswalk_polygon_mode=hull` replay는 broader-val512 crosswalk F1을 `0.5854 -> 0.6187`로 올렸다. Same checkpoint/same lane/stop-line 기준 lane/stop/cross F1은 `0.5407 / 0.4083 / 0.6187`이고, stop thresholds까지 묶은 current composite는 `0.5407 / 0.4235 / 0.6187`이다.
- crosswalk-heavy loss 재시도는 이미 negative evidence가 있으므로 기본 후보가 아니다.
- crosswalk를 재개한다면 simple threshold/shape postprocess나 같은 hull/aspect sweep이 아니라 training-side retention evidence나 새 representation-aware decode contract를 먼저 요구한다.

성공 기준:

- crosswalk F1 `>=0.60`이 broader-val에서 유지되어야 한다.
- lane/stop-line 목표를 희생하는 crosswalk-only gain은 채택하지 않는다.
- broader-val512 crosswalk pass만으로 deployment/default 승격하지 않는다. lane/stop-line F1도 함께 `>=0.60`이어야 한다.

Gate 상태:

- crosswalk는 opt-in hull decode 아래 broader-val512 partial-positive/pass다.
- all-task gate는 아직 실패다. 현재 active bottleneck은 stop-line first, lane second다.

## 8. Gate 5: export/TorchScript

목적:

- `phase_4/checkpoints/best.pt`를 실제 export/runtime 후보로 만들 수 있는지 확인한다.

확인할 것:

- raw-head export metadata.
- postprocess contract.
- geometry filters가 export/ROS runtime에서 동일하게 적용되는지.
- output schema가 기존 prediction bundle과 호환되는지.

## 9. Gate 6: ROS2 realtime check

목적:

- 학습 metric 후보가 실제 ROS2 runtime에서 쓸 수 있는지 확인한다.

확인할 것:

- latency.
- GPU memory.
- frame-rate.
- prediction artifact shape.
- comparison overlay 또는 sample replay.

## 10. Gate 7: traffic-light selective fine-tune

Lane60 fine-tune과 분리한다.

가능한 시작점:

- phase 3 joint checkpoint.
- current merged develop 코드.

실험 형태:

- traffic source 중심 sampler.
- `det + tl_attr` loss만 켜기.
- 필요하면 detector supervised class를 `traffic_light`로 좁히는 config hook 추가.

판단 기준:

- traffic-light box recall/precision과 attribute combo accuracy를 분리해서 본다.
- attribute가 좋아도 box 검출이 약하면 end-to-end traffic-light 성공으로 보지 않는다.

## 11. 문서 갱신 규칙

실험을 끝낼 때마다 다음 네 가지를 남긴다:

- 원래 상황.
- 바꾼 것.
- 실제 metric/artifact 변화.
- 다음에 하지 말 것과 다음 gate.
