# 00C. Next Gates

> 이 문서는 다음 실행 순서와 금지사항을 관리한다.
> 새 실험을 끝내면 `00B_STATUS_HISTORY.md`에 결과를 추가하고, 이 문서의 gate를 갱신한다.

## 1. 지금 하지 말 것

- 60% 돌파를 "raw model solved"로 표현하지 않는다.
- `phase_objective=0.6089`를 F1 0.6 달성으로 표현하지 않는다.
- final geometry filters를 broader val 없이 deployment default로 승격하지 않는다.
- lane-family 개선을 또 긴 same-axis run 하나로 확인하려고 하지 않는다.
- GradScaler health gate 없이 PV26 long-run AMP default를 되살리지 않는다.
- run artifact를 몇 GB씩 남기는 방식으로 실험하지 않는다. 핵심 checkpoint, exact eval summary, 비교 grid만 남긴다.
- traffic light 상태를 lane60 결과로 판단하지 않는다.
- stop-line micro target/loss/sampler 축을 같은 형태로 반복하지 않는다.
- support map을 lane centerline 대체물처럼 쓰는 실험을 반복하지 않는다.
- support map으로 centerline component를 단순 bridge/closing하는 후처리만 반복하지 않는다.
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
- side-band centerline BCE positive weighting만으로 lane 0.6 path를 다시 찾지 않는다.
- side-band centerline probability margin loss만으로 lane 0.6 path를 다시 찾지 않는다.
- side/truncated/near-vertical geometry-risk recall-only loss만으로 lane 0.6 path를 다시 찾지 않는다.
- side/truncated/near-vertical geometry-risk local Tversky loss만으로 lane 0.6 path를 다시 찾지 않는다.
- lane negative-pixel probability margin loss만으로 lane 0.6 path를 다시 찾지 않는다.
- lane endpoint coverage loss만으로 lane 0.6 path를 다시 찾지 않는다.
- residual-risk core/ring local separation loss weight만 키워 lane 0.6 path를 다시 찾지 않는다.
- exact val128 crosswalk threshold pass를 broader-val Gate 4 success로 표현하지 않는다.
- Gate 4 crosswalk isolation을 볼 때 lane threshold까지 같이 바꾼 top-objective variant를 먼저 broader default 후보로 삼지 않는다.
- `cross_mask=0.40`, `cross_area=32` stricter crosswalk component threshold를 broader-val success 없이 default/export 후보로 반복하지 않는다.
- 단순 predicted center/selector/max proposal threshold + angle-mask extent readout을 stop-line production fix로 반복하지 않는다.

## 2. Top-level goal: lane-family F1 0.6+

목표:

- 최종 목표를 `phase_objective`가 아니라 lane / stop-line / crosswalk F1 자체의 0.6+ 달성으로 둔다.
- 최소 기준은 broader validation에서 세 task 모두 F1 `>= 0.60`이다.
- 중간 기준으로 mean F1 `>= 0.60`을 볼 수는 있지만, stop-line이 낮은 상태에서 평균만 넘기는 것은 성공으로 보지 않는다.

현재 exact epoch-2 기준:

| Task | Current F1 | Gap to 0.6 | 우선순위 |
| --- | ---: | ---: | --- |
| lane | `0.5267` | `+0.0733` | 2 |
| stop-line | `0.4483` | `+0.1517` | 1 |
| crosswalk | `0.5854` | `+0.0146` | 3 |

해석:

- stop-line이 가장 큰 병목이다. 기존 stop-line reweight/retention은 일부 개선을 만들었지만 0.6까지는 멀다.
- lane은 postprocess geometry filters로 크게 올랐지만 아직 0.6에는 부족하다. 더 강한 raw centerline recall 또는 vectorizer-level recovery가 필요하다.
- crosswalk는 0.6에 가장 가깝다. 새 training axis보다 먼저 broader validation에서 유지되는지 확인한다.

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
- worktree 경로는 repo 밖 sibling 경로를 쓴다. 예: `/home/kai/yolopv26-exp-stopline-decoder`.
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
- proposal recall audit은 GT 주변 local score와 top-k ranking을 분리했다. broader-val512 `max(center, selector)` source는 `max_r8 >= 0.6`이 `208/271`이지만 top3-hit-r8은 `137/271`, raw rank top3는 `14/271`뿐이다. local signal은 남아 있으나 proposal competition/ranking이 약하다.
- candidate-pool audit은 broader-val512에서 oracle-positive top20 selection stop-line F1 `0.6517`, TP/FP/FN `131 / 0 / 140`까지 가능함을 보였다. 하지만 best production score/length filter는 `0.4371`, TP/FP/FN `106 / 108 / 165`로 PCA reference `0.4699`보다 낮다. 후보 pool은 있으나 non-GT FP suppression signal이 없다.
- hard-negative proposal ranking loss short run은 exact val128 epoch2 lane/stop/cross F1 `0.4754 / 0.4918 / 0.5767`, stop-line TP/FP/FN `30 / 32 / 30`에 그쳤다. stop-line은 baseline `0.4483`보다 높지만 angle-mask production `0.5085`와 PCA val128 reference `0.5133`보다 낮고, same-checkpoint candidate-pool production best도 `0.4286`이라 같은 loss-only family로 확장하지 않는다.
- endpoint-delta target/readout short run도 val128 epoch1/2 stop-line F1이 모두 `0.0000`이었다. 새 dense endpoint channel을 바로 decode source로 쓰면 TP가 사라지므로 같은 형태로 확장하지 않는다.
- heatmap-support geometry target fill은 exact val128 epoch2 stop-line F1 `0.2338`, TP/FP/FN `18 / 76 / 42`로 기준보다 크게 낮고, dense stop-line mask/center F1도 `0.4854 / 0.0815`로 후퇴했다. center heatmap support 전체에 geometry target을 채우는 방식은 0.6 path가 아니다.
- row-center auxiliary short run은 exact val128 epoch2 stop-line F1 `0.4248`로 기준 `0.4483`보다 낮았다. row selector에 centerline-row pressure만 추가하는 방식도 0.6 path가 아니다.
- component/readout audit은 broader-val512 GT `271`개 중 production TP `98`, anchorless component fit close `123`, anchored fit close `119`를 보였다. GT tube mask/center max가 `>=0.50`인 GT는 `223 / 220`개지만, production FN 중 anchorless fit으로 새로 close가 되는 것은 `34`개뿐이다. 단순 no-anchor PCA/anchor swap은 0.6 path가 아니다.
- older branch history의 GT target mask oracle은 stop-line F1 `0.8228`까지 가능했고, GT-overlap oracle은 `component_fit_or_endpoint_error=97` 중 `50`개를 oracle-trimmed pixels로 40px 안에 복구했다. 따라서 vectorizer/evaluator 자체가 hard blocker라는 해석은 약하다.
- 하지만 non-oracle repair follow-up은 닫혔다. core-row trim best는 val128 stop-line F1 `0.4833`으로 PCA reference `0.5133` 미달, high-confidence cleanup은 broader-val512 `0.4667`로 PCA reference `0.4699` 미달, split fit은 val128 `0.4957 / 0.4786`로 PCA reference를 못 넘었다.
- fit-far visual audit은 production FN, GT tube mask/center `>=0.50`, no-anchor distance `>40px` bucket 상위 18개를 렌더링했다. 18개 모두 production stop-line은 1개씩 있고, 14개는 component_count도 1이라 single connected component 안에서 wrong line segment를 읽는 문제가 강하다.
- local component extraction probe는 center/selector/fused score로 component 내부 local support를 골라 다시 fit했지만 exact val128 stop-line F1이 기준 `0.4483`을 넘지 못했다. best replacement는 `0.4464`, append-top2 best는 FP 증가 때문에 `0.4054`다.
- component-split readout probe는 pair/Hough-like 후보를 component 내부에서 만들었지만 exact val128 best append-top2도 stop-line F1 `0.3421`, TP/FP/FN `26 / 66 / 34`로 baseline보다 낮았다. replacement 계열은 best `0.2857`로 TP를 크게 잃었다.

후보:

- stop-line은 GT-center angle-mask extent upper-bound를 봤지만, 단순 predicted selector/center proposal readout은 PCA reference를 못 넘었다.
- stop-line proposal map은 GT 근처 local signal이 남아 있지만 low-top-k ranking이 약하다.
- stop-line candidate pool에는 valid segment가 있지만 score/length-only filter와 dense proposal ranking loss-only는 reference를 넘지 못했다. 다음 stop-line 축은 candidate validator/classifier처럼 candidate instance 자체를 분류하는 target이거나, Gate 3 lane predicted centerline instance stability로 돌아가는 쪽이다.
- PCA component 후보는 weak-positive reference로 보관하되, deployment default 승격 후보로 보지 않는다.

성공 기준:

- stop-line F1이 broader-val에서 의미 있게 상승해야 한다.
- lane/crosswalk F1이 0.6 목표에서 멀어질 정도로 무너지면 실패다.
- exact subset에서만 좋아지는 stop-line threshold tweak은 채택하지 않는다.

Gate 상태:

- partial weak-positive only. broader-val512 best stop-line F1은 `0.4699`이고 목표 미달이다.
- same-family micro experiments는 중단한다.
- 다음 stop-line 실행은 단순 threshold/top-k decode가 아니라 proposal reliability를 직접 올리거나 검증하는 contract여야 한다. top10/top20 후보 pool은 recall headroom을 보였지만 FP 후보를 제어하지 못하면 실패다. 단순 half-length target/loss/readout scalar, learned query-vector proposal-only, endpoint-delta direct decode, heatmap-support geometry fill, row-center auxiliary-only, selector-map threshold/gate-only, row/x span proposal-only, rowx-band selector target + selector component gate, 단순 predicted center/selector/max proposal + angle-mask extent readout, no-anchor PCA/anchor swap-only, PCA threshold/top-k-only, row-band/core-row trim, high-confidence cleanup, PCA endpoint quantile trim, local center/selector/fused window extraction, append-top2, pair/Hough-like component split readout, score/length-only candidate filter, hard-negative proposal ranking loss-only, oracle-positive candidate selection as production은 반복하지 않는다.

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
- `lane-row-scan-geometry-guard`는 length/bottom/gap/dx/turn-angle guard를 exact val128에서 비교했다. best lane F1은 `row_gap24_row_dx12`의 `0.5526`으로 기존 row-scan `0.5522` 대비 noise-level이고 FP가 늘었다. turn-angle guard는 FP를 줄였지만 TP를 더 잃어 best `0.5365`에 그쳤다.
- `lane-row-scan-residual-buckets`는 broader-val512 row-scan residual lane TP/FP/FN `4153 / 2105 / 5324`를 확인했다. FN은 left `46.9%`, truncated `<0.50` `27.9%`, aspect `>=3` `65.9%`에 몰리고, FP는 side `74.8%`와 right `40.6%` 비중이 높다.
- `core_centerline_refine_residual_local_separation`은 residual-risk GT core positive와 local ring negative margin을 같이 줬다. exact val128 epoch2 lane/stop/cross F1은 `0.5476 / 0.4310 / 0.5854`로 lane은 기준보다 올랐지만 stop-line과 objective `0.6085`가 기준 `0.6089`보다 낮다. broader-val512로 확장하지 않는다.
- `core_centerline_refine_bce_focus`는 lane F1을 broader-val512 `0.5101 -> 0.5344`로 올렸지만, centerline-core pixel F1은 `0.5680`으로 기준선보다 낮고 stop-line/crosswalk가 내려갔다.
- BCE-focus + PCA stop-line decoder integration audit best는 lane/stop/cross F1 `0.5344 / 0.4583 / 0.5741`로 partial-positive지만 목표 미달이다.
- BCE-focus stop-balance broader-val512는 `0.5372 / 0.4041 / 0.5812`이고 PCA replay best도 `0.5372 / 0.4528 / 0.5812`라 stop-line 병목을 못 풀었다.
- support map을 단순 대체하거나 직접 residual input으로 넣는 방식은 이미 negative evidence가 있으므로 반복하지 않는다.
- 새 lane training axis는 stop-line plan과 섞지 않고 별도 short run으로 본다.

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
- `exp/lane-family-f1/stopline-angle-mask-extent-diagnostic`은 GT center upper-bound에서 angle-anchored mask extent가 half-length scalar보다 낫지만 broader-val512 목표에는 아직 모자란다는 read-only evidence로 보관한다.
- `exp/lane-family-f1/stopline-centerline-endpoint-offset`은 endpoint-delta target/readout negative evidence로 보관한다.
- `exp/lane-family-f1/stopline-heatmap-geometry-support`는 heatmap-support geometry target-fill negative evidence로 보관한다.
- `exp/lane-family-f1/stopline-row-center-aux`는 row-center auxiliary-only negative evidence로 보관한다.
- `exp/lane-family-f1/stopline-readout-component-audit`은 dense mask/center signal이 남아 있어도 단순 component PCA/anchor swap만으로는 부족하다는 read-only evidence로 보관한다.
- `exp/lane-family-f1/stopline-fit-far-visual-audit`은 high-signal FN bucket의 visual evidence로 보관한다.
- `exp/lane-family-f1/stopline-local-component-extraction`은 local center/selector/fused window extraction negative evidence로 보관한다.
- `exp/lane-family-f1/stopline-component-split-readout`은 pairwise component split readout negative evidence로 보관한다.
- `exp/lane-family-f1/stopline-candidate-validator-audit`은 hard-negative proposal ranking loss-only negative evidence로 보관한다.
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
- `exp/lane-family-f1/lane-row-scan-geometry-guard`는 row-scan micro-guard negative evidence로 보관한다.
- `exp/lane-family-f1/lane-row-scan-residual-buckets`는 row-scan 후 residual이 left/truncated/high-aspect FN과 side/right FP에 남는다는 read-only evidence로 보관한다.
- `exp/lane-family-f1/lane-residual-local-separation`은 lane partial-positive지만 stop-line/objective regression 때문에 negative evidence로 보관한다.
- row-scan visual audit artifact는 `analysis_exports/row_scan_visual_compare_epoch2/row_scan_component_comparison_grid.png`와 `manifest.json`이다. stop-line fit-far visual audit artifact는 `analysis_exports/stopline_fit_far_visual_audit_val512_epoch2/stopline_fit_far_bucket_grid.png`와 `manifest.json`이다. 판정은 partial-pass/caution이므로, 다음 lane 한 축은 row-scan 후처리 조절이나 residual-risk local loss weight 조절이 아니라 predicted centerline evidence를 instance 단위로 안정화하거나 stop-line/crosswalk retention을 같이 보는 contract다. stop-line을 재개한다면 candidate validator/classifier처럼 candidate instance 자체를 분류하는 target이 필요하다. PCA threshold/top-k, stop-line weight-only, component split, center-cell geometry-mask, half-length inference-scale, half-length loss-weight-only, log-target-only, learned query-vector proposal-only, selector-map component gate/threshold-only, row/x span proposal-only, rowx-band selector target + selector component gate, 단순 predicted center/selector/max proposal + angle-mask extent readout, score/length-only candidate pool filter, hard-negative proposal ranking loss-only, endpoint-delta direct decode, heatmap-support geometry fill, row-center auxiliary-only, no-anchor PCA/anchor swap-only, local center/selector/fused window extraction, append-top2, point-pair component split readout, row-scan length/bottom/gap/dx/turn-angle guard, lane support-substitution, lane threshold-only sweep, semantic attr oracle, lane target-width-only widening, side-band BCE-only, side-band margin-only, geometry-risk recall-only, geometry-risk local-Tversky-only, negative-pixel margin-only, support-bridge/closing-only, endpoint-only coverage, residual-risk local separation weight-only는 반복하지 않는다.
- val128 dense-map PR, exact task F1, broader-val replay, visual audit 순서로 통과시킨다.

## 7. Gate 4: crosswalk retention to 0.6

목적:

- crosswalk F1을 `0.5854` 근처에서 안정적으로 0.6 이상으로 넘긴다.

후보:

- exact val128 threshold probe는 완료됐다. `cross_mask=0.40`, `cross_area=32`가 crosswalk F1을 `0.5854 -> 0.6027`로 올렸다.
- top objective variant는 `lane_obj_0.35__cross_mask_0.40__cross_area_32`이고 objective `0.6115270643`, lane/stop/cross F1 `0.5326 / 0.4483 / 0.6027`이다.
- crosswalk-only candidate는 `lane_obj_0.45__cross_mask_0.40__cross_area_32`이고 objective `0.6098888876`, lane/stop/cross F1 `0.5267 / 0.4483 / 0.6027`이다.
- broader-val512 crosswalk-only replay는 실패했다. baseline lane/stop/cross F1은 `0.5101 / 0.4083 / 0.5854`, candidate는 `0.5101 / 0.4083 / 0.5845`다.
- crosswalk-heavy loss 재시도는 이미 negative evidence가 있으므로 기본 후보가 아니다.
- crosswalk를 재개한다면 stricter component threshold가 아니라 recall-safe shape filtering이나 training-side retention evidence를 먼저 요구한다.

성공 기준:

- crosswalk F1 `>=0.60`이 broader-val에서 유지되어야 한다.
- lane/stop-line 목표를 희생하는 crosswalk-only gain은 채택하지 않는다.
- exact val128 기준으로는 partial-positive일 뿐이고, broader-val512 통과 전에는 deployment/default 승격하지 않는다.

Gate 상태:

- exact-only partial-positive, broader-val512 failure.
- 현재 active bottleneck은 다시 stop-line first, lane second다.

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
