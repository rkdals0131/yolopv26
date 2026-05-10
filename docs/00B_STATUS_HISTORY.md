# 00B. Status History

> 이 문서는 오답노트다.
> 원래 상황, 바꾼 것, 실제 결과, 다음에 하지 말 것을 압축해서 남긴다.

## 1. 2026-04-06 run: metric collapse를 숫자로 본 시점

상황:

- shipped schedule은 `head warmup -> partial unfreeze -> full finetune -> lane-family late finetune`이었다.
- phase 1/2는 detector만 약간 개선되고 lane/stop/cross는 0에 가까웠다.
- phase 3에서도 lane family는 충분히 올라오지 않았다.
- phase 4는 trunk를 고정하고 lane-family head만 큰 노출량으로 밀었다.

결과:

- phase 3 best objective는 `0.2821`, lane F1 최고는 `0.0124`뿐이었다.
- phase 4 epoch 1이 best였고, 이후 lane TP가 0으로 붕괴했다.
- train loss는 내려갔지만 validation lane metric은 사라졌다.

판단:

- "loss는 내려가는데 metric contract를 만족하지 못한다"는 것이 핵심이었다.
- phase objective floor와 no-match 처리, phase 4 진입 조건, lane-family metric contract를 먼저 의심해야 한다.
- 데이터만 더 넣거나 같은 phase 4를 길게 돌리는 방식은 우선순위가 낮다.

원문: `legacy/16_PV26_RUN_ANALYSIS_20260406_201012.md`

## 2. Strict runtime cleanup: 조용히 지나가는 실패를 줄이는 방향

상황:

- training/runtime/resume/tooling에 best-effort fallback이 많았다.
- 특히 non-finite/OOM/assigner 오류 skip, checkpoint/config 자동 대체, 빈 자료구조 fallback이 run 해석을 흐렸다.

정리 원칙:

- core training/loss/resume/export는 strict fail-fast가 기본이다.
- preview/report/check_env 같은 도구만 제한적으로 best-effort를 허용한다.
- optional I/O와 required I/O를 함수/호출 경로에서 분리한다.

하지 말 것:

- broken batch를 조용히 skip해서 run을 살리는 방향.
- source checkpoint나 config가 없을 때 비슷한 다른 것을 자동으로 집어드는 방향.
- training correctness와 visualization/report convenience를 같은 fallback으로 묶는 방향.

원문: `legacy/15_Exception_remove.md`

## 3. 2026-05-02 full-run 실패: AMP collapse

상황:

- `python3 tools/run_pv26_train.py --preset default`로 capped long-run을 시작했다.
- phase schedule은 stage 1/2 짧게, stage 3/4는 capped `30~50` epoch 구조였다.

결과:

- `exhaustive_od_lane_default_20260502_193106` run은 폐기한다.
- phase 1 후반부터 AMP GradScaler가 scale `0.0`을 기록했고, phase 2/3에서는 대부분의 train step이 scale `0.0`으로 남았다.
- validation은 detector/lane/stop-line TP가 거의 없고 crosswalk FP가 폭주했다.

수정:

- PV26 shipped local long-run 기본값은 `amp=false`로 되돌렸다.
- `train_batches`, sampled step history, checkpoint 저장 정책을 조정해서 긴 run이 disk/history bloat로 죽지 않게 했다.
- `task_positive_task=multi:lane,stopline,crosswalk`와 fail-fast sampler 정책을 명확히 했다.

하지 말 것:

- GradScaler health gate 없이 PV26 long-run default를 다시 AMP로 돌리지 않는다.
- comparison grid artifact만 보고 training이 정상이라고 단정하지 않는다.

## 4. 2026-05-05 long run: 실제로 작동하는 baseline

상황:

- 3일 20시간 가까운 run에서 처음으로 "의도한 구조가 작동한다"는 lane-family baseline이 나왔다.
- source run: `runs/pv26_exhaustive_od_lane_train/exhaustive_od_lane_default_20260505_032217`.

결과:

- phase 4 epoch 18이 best로 선택됐다.
- objective `0.5272`, lane/stop/cross F1 `0.3643 / 0.3556 / 0.5165`.
- phase 4 epoch 37까지 가도 더 좋아지지 않았고, 일부 task는 오히려 흔들렸다.

판단:

- architecture가 완전히 틀린 상태는 아니었다.
- 하지만 같은 축으로 4일을 더 돌린다고 60~70%로 점프할 근거도 없었다.
- 다음 작업은 "긴 run 재시도"가 아니라 checkpoint에서 어떤 신호가 부족한지 줄이는 probe여야 했다.

## 5. Lane-family 60% probe: training-side 개선과 후처리 개선 분리

1차 probe:

- threshold-only val128 sweep은 작은 gain만 줬다.
- upper-trunk unfreeze는 heads-only보다 확실히 낫지 않았다.
- short loss rebalance는 stop-line을 올릴 수 있었지만 lane이 계속 낮았다.

2차 architecture/target probe:

- lane centerline target을 soft에서 core로 바꾸는 것이 첫 유의미한 개선이었다.
- core target은 objective를 dense-sharpen range `~0.5308`에서 `0.5530`으로 올렸다.
- centerline-only gated refinement가 objective를 `0.5591`까지 올렸다.
- merged task-head seed와 cross-adapted head merge가 `0.5609`까지 nudged했다.

falsified:

- support-map substitution은 shortcut이 아니었다.
- low-LR retention, Dice-heavy centerline, stop/cross reweight, upper-trunk reopen, crosswalk isolator는 60% path가 아니었다.
- longer same-axis continuation은 epoch 2 근처에서 peak를 찍고 흔들렸다.

후처리/evaluator premise:

- standalone evaluator가 fresh validation subset을 보고 있던 문제가 있었다.
- `tools/evaluate_pv26_lane60_checkpoint.py --validation-epoch`로 sampler를 advance해서 training best epoch와 같은 subset을 replay하도록 고쳤다.
- exact epoch-2 support는 lane/stop/cross `2390 / 60 / 81`이다.

최종 결과:

- selected `best.pt` exact epoch-2 baseline: objective `0.5611`, lane/stop/cross F1 `0.4435 / 0.3946 / 0.4111`.
- score/component filtering 후: objective `0.5732`, lane/stop/cross F1 `0.4429 / 0.4275 / 0.4953`.
- final geometry filters 후: objective `0.6088677363`, lane/stop/cross F1 `0.5267 / 0.4483 / 0.5854`.

해석:

- 60% 돌파는 real checkpoint + necessary decode cleanup의 결과다.
- model raw output만 강해졌다고 해석하면 안 된다.
- 작은 lane fragments, 약한 stop-line fragments, 작은/flat crosswalk polygons를 줄이는 geometry filters가 metric을 크게 움직였다.
- broader validation replay 없이 deployment/export default로 고정하지 않는다.

원문:

- `legacy/18_PV26_LANE_FAMILY_THRESHOLD_SWEEP_20260509.md`
- `legacy/19_PV26_LANE60_PROBES_20260509.md`
- `legacy/20_PV26_LANE60_CONTINUATION_20260509.md`

## 6. 2026-05-10 Gate 1: broader-val replay로 60% 착시를 줄인 시점

상황:

- exact epoch-2 subset에서 final geometry filters가 objective `0.6088677363`을 만들었다.
- 하지만 다음 목표는 `phase_objective`가 아니라 broader validation에서 lane / stop-line / crosswalk F1 모두 `0.60+`다.
- 그래서 같은 checkpoint와 같은 postprocess config를 validation 512 batch로 다시 replay했다.

결과:

- artifact: `runs/pv26_exhaustive_od_lane_train/lane60_core_centerline_refine_cross_retain_from_exhaustive_od_lane_default_20260505_032217_default_20260510_003412/analysis_exports/broader_val512_final_geometry_filters_epoch2/summary.json`
- objective `0.5943438312141003`
- lane F1 `0.5100986992613896`, TP/FP/FN `3902 / 1920 / 5575`, support `9477`
- stop-line F1 `0.4083333333333333`, TP/FP/FN `98 / 111 / 173`, support `271`
- crosswalk F1 `0.5853658536585366`, TP/FP/FN `216 / 127 / 179`, support `395`

판단:

- exact subset의 0.6089는 broader-val 기준으로는 유지되지 않았다.
- 그래도 geometry filters가 완전히 깨진 신호는 아니다. lane precision은 `0.6702`, crosswalk precision은 `0.6297`로 FP 억제 효과가 남아 있다.
- 병목은 더 명확해졌다. stop-line은 precision `0.4689`, recall `0.3616`, F1 `0.4083`으로 F1 0.6 목표에서 가장 멀다.
- lane은 precision보다 recall이 더 큰 문제다. lane recall은 `0.4117`이고, FN이 `5575`다.

하지 말 것:

- exact epoch-2 objective `0.6089`만으로 deployment/export default라고 말하지 않는다.
- broader-val F1 0.6 미달 상태에서 export/ROS gate로 넘어가지 않는다.
- 다음 실험에서 stop-line과 lane 개선을 같은 worktree에 섞지 않는다.

다음:

- Gate 2는 stop-line first로 간다.
- 별도 worktree에서 stop-line decoder/target/loss 또는 stop-line-heavy short fine-tune을 한 축씩 실험한다.
- lane recall 실험은 Gate 2 결과와 분리해서 진행한다.

## 7. 2026-05-10 Gate 2 seed: stop-line feature audit

상황:

- `exp/lane-family-f1/stopline-diagnostics` worktree를 만들고 dataset/weight symlink를 연결했다.
- 같은 checkpoint와 val512 slice에서 TP/FP/FN shape feature를 export했다.
- artifact: branch-local `runs/lane_family_f1/stopline_filter_features_val512_epoch2/summary.json`

결과:

- stop-line counts는 TP/FP/FN `98 / 111 / 173`이다.
- score 분포는 TP median `0.9646`, FP median `0.9609`로 거의 겹친다.
- bbox area는 TP median `1364.4`, FP median `980.5`, FN median `2275.0`이다.
- bbox aspect는 TP median `29.25`, FP median `19.80`, FN median `22.16`이다.

판단:

- 단순 score threshold는 좋은 다음 축이 아니다. TP와 FP score가 너무 가깝다.
- bbox area/aspect filter도 단독으로는 위험하다. FN에도 큰 stop-line이 많아서 recall을 더 깎을 수 있다.
- Gate 2의 우선 가설은 postprocess threshold보다 stop-line decoder/target/recall 문제다.

하지 말 것:

- stop-line F1을 score threshold tweak만으로 해결하려고 하지 않는다.
- area/aspect threshold를 올리는 실험은 FN 손실을 먼저 계산하지 않고 채택하지 않는다.

## 8. 2026-05-10 Gate 2 stop-line worktree trail: many narrow axes, no 0.6 path yet

상황:

- Gate 2는 broader-val512 기준 stop-line F1 `0.4083`을 먼저 올리는 것이었다.
- worktree는 `exp/lane-family-f1/stopline-*` 형태로 분리했고, 한 worktree는 한 가설만 소유했다.

실험 축:

- score/threshold-only와 mask-only postprocess.
- centerline target, dense geometry target, selector-center target, center stem.
- mask loss, wider mask target, positive/weighted sampler.
- mask vectorizer, component PCA fit, core-row fit, cleanup fit.
- endpoint mask weighting, endpoint proposal head, feature isolation.

실제 결과:

- 단순 score threshold는 TP/FP score 분포가 겹쳐서 버렸다.
- sampler/loss/target 단일 축은 stop-line을 0.6 근처로 끌어올리지 못했다.
- endpoint mask weighting은 broader-val512 stop-line F1 `0.4226 / 0.4237`로 decoder-only PCA 후보보다 나빴다.
- direct selector decode는 val128 stop-line F1 `0.0635` 수준이라 fallback이 아니었다.
- core-row fit은 val128 PCA reference `0.5133`보다 낮은 `0.4833`이었다.
- cleanup fit은 val128에서 거의 동률이었지만 val512에서 `0.4667`로 PCA reference `0.4699`보다 낮았다.
- wider stop-line mask target은 short train epoch2에서 lane/stop/cross F1 `0.5254 / 0.3091 / 0.5854`로 실패했다.
- feature isolation은 phase objective만 `0.6019`까지 갔고 task F1은 lane/stop/cross `0.5260 / 0.4107 / 0.5854`라서 성공이 아니었다.
- endpoint proposal도 phase objective만 `0.6030`까지 갔고 task F1은 lane/stop/cross `0.5260 / 0.4404 / 0.5854`로 exact baseline stop-line `0.4483`보다 낮았다.

가장 강한 stop-line 중간 후보:

- original checkpoint + decoder-only `component_pca_full_mask080_score094`.
- broader-val512 lane/stop/cross F1 `0.5101 / 0.4699 / 0.5854`.
- stop-line TP/FP/FN `113 / 97 / 158`.

판단:

- stop-line은 "mask가 전혀 없다"보다 "mask에서 정확한 선분 geometry로 복원하는 계약"이 더 큰 병목이다.
- PCA component fit은 weak-positive지만 목표까지는 `+0.13` 정도 남아 있다.
- 같은 종류의 micro target/loss/sampler를 더 반복하는 것은 우선순위가 낮다.

하지 말 것:

- `phase_objective > 0.60`을 task F1 0.6 달성으로 해석하지 않는다.
- stop-line mask pixel 품질만 보고 vector F1이 해결됐다고 말하지 않는다.
- endpoint/feature-isolation 단일 축은 같은 형태로 반복하지 않는다.

## 9. 2026-05-10 Gate 3 lane dense-map split: support is strong, centerline core is the lane bottleneck

상황:

- Gate 2 stop-line micro axes가 0.6 path를 만들지 못했다.
- 다음으로 lane F1 `0.5101`의 병목이 dense-map 자체인지 vectorizer recovery인지 분리했다.

실행:

- command: `python3 tools/probe_pv26_lane60_dense_maps.py --checkpoint runs/pv26_exhaustive_od_lane_train/lane60_core_centerline_refine_cross_retain_from_exhaustive_od_lane_default_20260505_032217_default_20260510_003412/phase_4/checkpoints/best.pt --preset default --phase-index 4 --max-val-batches 128 --device auto`
- processed batches: `128`, batch size `4`.

결과:

- lane centerline core best pixel F1 `0.5729` at threshold `0.9`, precision/recall `0.5148 / 0.6457`.
- lane support best pixel F1 `0.7971` at threshold `0.9`, precision/recall `0.7527 / 0.8470`.
- lane centerline soft best pixel F1 `0.3608`, recall `0.2347`.
- stop-line mask best pixel F1 `0.6107`, but stop-line center heatmap best pixel F1 `0.1385`.
- crosswalk mask best pixel F1 `0.8606`, but crosswalk center best pixel F1 `0.1324`.

판단:

- lane support map은 이미 강해서 support-map substitution류를 다시 반복할 이유가 없다.
- lane은 centerline core pixel F1이 아직 `0.57`대라 vectorizer만 탓할 수 없다.
- stop-line/crosswalk는 mask pixel map과 center/geometry map의 격차가 커서 vector/center 복원이 병목이라는 기존 판단을 강화한다.

다음:

- Gate 3는 lane centerline-core 품질을 올리는 한 축 실험으로 간다.
- 후보는 centerline-core loss/target calibration 또는 centerline-to-vector recovery audit이다.
- support map을 직접 centerline 대체물로 쓰거나 residual로 넣는 방식은 이미 negative evidence가 있으므로 반복하지 않는다.

## 10. 2026-05-10/11 branch-local follow-ups: partial lane lift, stop-line still below target

상황:

- `develop`의 Gate 2/3 이후 여러 branch/worktree에서 한 축씩 추가 실험했다.
- 이 구간의 목적은 broader-val512 기준 lane / stop-line / crosswalk F1을 모두 `0.60+`로 올릴 수 있는 후보를 찾는 것이었다.
- 성공 기준은 여전히 task별 F1이지 `phase_objective`가 아니다.

Lane branch 결과:

- `exp/lane-family-f1/lane-centerline-core-calibration`은 `core_centerline_refine_bce_focus`로 centerline BCE/Dice 비율만 `2/2 -> 3/1`로 바꾼 단일 축이다.
- exact val128 epoch2 lane/stop/cross F1은 `0.5451 / 0.4348 / 0.5854`, broader-val512 F1은 `0.5344 / 0.4025 / 0.5741`이다.
- lane은 broader-val 기준선 `0.5101`보다 올랐지만, stop-line과 crosswalk가 내려갔다.
- dense-map PR에서 lane centerline core F1은 `0.5680`으로 기준선 `0.5729`보다 낮다. 이 gain은 centerline pixel map 자체의 개선이라기보다 vectorized metric partial-positive로 본다.
- BCE-focus checkpoint decode audit의 best proxy는 `stop_mask_only`였지만 lane/stop/cross F1 `0.5471 / 0.2569 / 0.6545`에 그쳤다. lane threshold tightening, support-as-centerline, support-blend는 lane F1을 개선하지 못했다.
- `exp/lane-family-f1/lane-bce-stopline-pca-integration`에서 BCE-focus checkpoint 위에 PCA stop-line decoder 후보를 얹었을 때 best broader-val512 F1은 `0.5344 / 0.4583 / 0.5741`이다.
- `exp/lane-family-f1/lane-bce-stopline-balance`는 task weight를 lane `2.0`, stop-line `2.25`, crosswalk `1.75`로 바꿨지만 broader-val512 F1은 `0.5372 / 0.4041 / 0.5812`이고, PCA replay best도 `0.5372 / 0.4528 / 0.5812`였다.

Stop-line branch 결과:

- `exp/lane-family-f1/stopline-component-split-fit`은 component row-band split이 val128 PCA reference stop-line F1 `0.5133`보다 낮은 `0.4957` 또는 `0.4786`에 그쳐 broader-val512로 확장하지 않았다.
- `exp/lane-family-f1/stopline-geometry-center-mask`는 geometry regression을 center cell에만 걸었지만 exact val128 epoch2 lane/stop/cross F1 `0.5154 / 0.2609 / 0.5697`로 stop-line이 무너졌다.
- `exp/lane-family-f1/stopline-proposal-readout-diagnostic`은 GT stop-line replacement와 GT center/length reconstruction이 val128 stop-line F1 `1.0000`을 낼 수 있음을 보였다. evaluator/readout representation 자체가 0.6을 막고 있지는 않다.
- `exp/lane-family-f1/stopline-gt-center-readout`은 predicted offset 또는 predicted angle만 섞은 variant가 stop-line F1 `1.0000`을 유지하지만, predicted half-length가 들어가면 invalid readout이 58개로 늘고 stop-line F1 `0.0645`로 무너짐을 보였다.
- `exp/lane-family-f1/stopline-half-length-scale-audit`에서 predicted half-length x128 best는 stop-line F1 `0.4500`이다. baseline `0.4483`과 사실상 같고 pred-vs-GT Spearman은 `-0.204`라 단순 scale 문제가 아니다.
- `exp/lane-family-f1/stopline-half-length-loss-boost`는 half-length loss weight `16x`에도 exact val128 epoch2 stop-line F1 `0.4576`에 그쳤다.
- `exp/lane-family-f1/stopline-half-length-log-target`은 log target으로 exact val128 epoch2 stop-line F1 `0.4211`까지 후퇴했다.
- `exp/lane-family-f1/stopline-vector-proposal-readout`은 learned query-vector proposal-only short run이다. vector-only exact val128 epoch1/2 stop-line F1은 모두 `0.0000`이고, threshold를 `0.10`까지 낮춰도 TP/FP/FN `0 / 0 / 55`였다. append mode는 mask baseline과 같은 stop-line F1 `0.2000`에 머물렀고, 같은 checkpoint의 `stop_mask_only`는 `0.2778`이었다.

판단:

- Lane BCE-focus는 보관할 partial-positive다. 다만 전체 목표를 달성하지 못했고 centerline pixel PR 개선도 아니다.
- PCA component decoder는 stop-line weak-positive reference지만 threshold/top-k 확장만으로는 0.6까지 못 간다.
- stop-line의 hard blocker는 dense signal을 valid line geometry로 바꾸는 instance readout/target contract다.
- learned query-vector proposal-only, half-length scalar 변형, component split, center-cell geometry-mask, stop-line weight-only는 반복하지 않는다.

다음:

- 다음 stop-line 축은 dense mask/centerline signal을 line geometry로 복원하는 target/readout contract다.
- stop-line을 잠시 보류한다면 lane axis는 support substitution이나 threshold sweep이 아니라 centerline-to-vector recovery error bucket 또는 새로운 centerline-core 품질 가설로 제한한다.

## 11. 2026-05-11 Gate 2 revisit: selector-map component gate is negative

맥락:

- stop-line loss는 `stop_line_selector_map_logits`를 centerline target으로 학습시키지만, production `_stopline_mask_to_polyline`은 component gate에서 주로 row selector와 center heatmap을 쓴다.
- 그래서 새 training 없이 selector map을 component 선택/anchor source에 opt-in으로 연결하면 dense centerline signal이 line geometry readout을 개선하는지 확인했다.

변경:

- branch: `exp/lane-family-f1/stopline-selector-component-gate`
- `PV26PostprocessConfig.stop_line_component_gate_source`를 추가했다. 기본값은 기존 동작인 `center`다.
- `selector` mode는 selector map을 component gate와 anchor source로 쓰고 row selector override를 끈다.
- `max` mode는 selector map과 center heatmap의 max map을 component gate로 쓴다.
- `tools/probe_pv26_lane60_decode_variants.py`에 `stop_selector_gate`, `stop_selector_gate_obj030`, `stop_selector_gate_mask030`, `stop_gate_max` variants와 `--output-json`을 추가했다.

검증:

- output: `runs/pv26_exhaustive_od_lane_train/lane60_core_centerline_refine_cross_retain_from_exhaustive_od_lane_default_20260505_032217_default_20260510_003412/analysis_exports/decode_variants_selector_gate_val128_epoch2.json`
- baseline proxy lane/stop/cross F1은 `0.5222 / 0.2000 / 0.6667`이다.
- `stop_mask_only`는 lane/stop/cross F1 `0.5222 / 0.2593 / 0.6667`, stop-line TP/FP/FN `14 / 39 / 41`이다.
- `stop_selector_gate`, `stop_selector_gate_obj030`, `stop_selector_gate_mask030`은 모두 lane/stop/cross F1 `0.5222 / 0.2062 / 0.6667`, stop-line TP/FP/FN `10 / 32 / 45`다.
- `stop_gate_max`는 lane/stop/cross F1 `0.5222 / 0.1980 / 0.6667`, stop-line TP/FP/FN `10 / 36 / 45`다.

판단:

- selector map을 단순히 component gate/anchor로 연결해도 stop-line recall이 살아나지 않는다.
- FP는 baseline보다 줄지만 TP도 유지되지 않아서 `stop_mask_only`보다 약하다.
- 이 결과는 "selector map을 쓰지 않아서 생긴 후처리 bug"가 아니라, dense signal을 valid line geometry로 바꾸는 더 명시적인 target/readout 계약이 필요하다는 쪽을 강화한다.

하지 말 것:

- selector-map component gate를 broader-val512나 long run으로 확장하지 않는다.
- row selector를 끄고 selector map으로 component만 고르는 후처리 sweep을 반복하지 않는다.

다음:

- 다음 stop-line 축은 selector map gate가 아니라 train-time target과 geometry readout이 직접 맞는 contract여야 한다.
- 가능한 후보는 component fit 후처리보다 endpoint/length를 component 또는 centerline pixels에서 직접 supervise/recover하는 방식이다.

## 12. 2026-05-11 Gate 2 revisit: endpoint-delta target/readout is negative

맥락:

- selector-map component gate가 실패한 뒤, stop-line centerline pixels에서 양 끝점 delta를 직접 회귀하면 component mask를 valid line segment로 복원할 수 있는지 확인했다.
- 목적은 half-length scalar 대신 `(start_x, start_y, end_x, end_y)` delta를 dense stop-line head에서 예측하고, opt-in decoder가 component anchor에서 이 값을 읽는 것이었다.

변경:

- branch: `exp/lane-family-f1/stopline-centerline-endpoint-offset`
- `StopLineDenseLocalHead`에 `stop_line_endpoint_delta` 4ch output을 추가했다.
- `build_stopline_mask_targets`와 batch encoder가 centerline pixels에 endpoint deltas를 기록하게 했다.
- stop-line mask loss에 `stopline_endpoint_delta_aux_weight`를 추가하고 기본값은 `0.0`으로 유지했다.
- `PV26PostprocessConfig.stop_line_endpoint_delta_decode`를 추가하고 기본값은 `False`로 유지했다.
- `core_centerline_refine_stop_endpoint_delta` probe는 endpoint delta aux weight `2.0`과 endpoint decode를 켠다.

검증:

- output run: `runs/pv26_exhaustive_od_lane_train/lane60_core_centerline_refine_stop_endpoint_delta_from_lane60_core_centerline_refine_cross_retain_from_exhaustive_od_lane_default_20260505_032217_default_20260510_003412_default_20260511_021151`
- epoch1 objective `0.5880863169`, lane/stop/cross F1 `0.5124 / 0.0000 / 0.6790`, stop-line TP/FP/FN `0 / 13 / 55`.
- epoch2 objective `0.5758300830`, lane/stop/cross F1 `0.5261 / 0.0000 / 0.5854`, stop-line TP/FP/FN `0 / 13 / 60`.
- best objective는 epoch1의 `0.5880863169`로 기준 exact epoch2 `0.6088677363`과 broader-val512 `0.5943438312`보다 낮다.

판단:

- endpoint-delta channel 자체는 구현/학습/디코드 단위 테스트를 통과했다.
- 하지만 newly initialized endpoint geometry를 바로 decode source로 쓰면 stop-line TP가 사라진다.
- 현재 병목은 endpoint 표현이 없어서가 아니라 predicted anchor/geometry reliability가 부족한 쪽이다.

하지 말 것:

- endpoint-delta channel 추가 + direct decode를 같은 형태로 long run이나 broader-val512로 확장하지 않는다.
- `phase_objective`가 일부 유지된 것만 보고 stop-line 개선으로 해석하지 않는다. stop-line F1은 `0.0000`이다.

다음:

- stop-line을 재개한다면 predicted center/proposal reliability를 먼저 올리거나 PCA/component-fit weak-positive를 넘어서는 다른 geometry recovery contract가 필요하다.
- stop-line을 잠시 보류한다면 lane axis는 support substitution이나 threshold sweep이 아니라 centerline-to-vector recovery error bucket 또는 새로운 centerline-core 품질 가설로 제한한다.

## 13. 2026-05-11 Gate 3 lane vectorizer recovery audit: vectorizer has headroom, predicted centerline is limiting

맥락:

- dense-map probe는 lane support map이 강하고 centerline core pixel F1이 `0.5729`에 머무른다는 것을 보였다.
- 아직 남은 질문은 "centerline만 맞으면 현재 seg-first vectorizer가 0.6 lane F1을 복구할 수 있는가"였다.
- 새 학습 없이 validation epoch2 sampler를 맞춰 current predicted maps, GT centerline oracle, GT attrs oracle을 같은 vectorizer/geometry filters로 비교했다.

변경:

- branch: `exp/lane-family-f1/lane-vectorizer-recovery-audit`
- `tools/probe_pv26_lane60_lane_vectorizer_recovery.py`를 추가했다.
- variants: `pred_full`, `pred_centerline_gt_attrs`, `gt_centerline_pred_attrs`, `gt_centerline_gt_attrs`.
- thresholds: `0.35 / 0.45 / 0.55 / 0.65 / 0.75 / 0.85`.
- `--validation-epoch`으로 training replay와 같은 validation sampler subset을 맞춘다.

검증:

- output: `runs/pv26_exhaustive_od_lane_train/lane60_core_centerline_refine_cross_retain_from_exhaustive_od_lane_default_20260505_032217_default_20260510_003412/analysis_exports/lane_vectorizer_recovery_val512_validation_epoch2`
- command: `python3 tools/probe_pv26_lane60_lane_vectorizer_recovery.py --checkpoint .../phase_4/checkpoints/best.pt --source-run ... --lane60-experiment core_centerline_refine_cross_retain --phase-index 4 --max-val-batches 512 --validation-epoch 2 --batch-size 4 --device cuda:0 --output-dir .../analysis_exports/lane_vectorizer_recovery_val512_validation_epoch2`
- `gt_centerline_gt_attrs@0.35`: lane F1 `0.6630`, precision/recall `0.9621 / 0.5058`, TP/FP/FN `4793 / 189 / 4684`, strict F1 `0.6485`.
- `gt_centerline_pred_attrs@0.35`: lane F1 `0.6630`, precision/recall `0.9621 / 0.5058`, TP/FP/FN `4793 / 189 / 4684`, strict F1 `0.6126`.
- `pred_full@0.35`: lane F1 `0.5169`, precision/recall `0.6677 / 0.4217`, TP/FP/FN `3996 / 1989 / 5481`, strict F1 `0.4900`.
- `pred_full@0.45`: lane F1 `0.5101`, precision/recall `0.6702 / 0.4117`, TP/FP/FN `3902 / 1920 / 5575`, strict F1 `0.4855`.

판단:

- 현재 vectorizer/geometry filters는 GT centerline이 주어지면 broader-val512 epoch2에서 lane F1 `0.6630`까지 복구한다.
- predicted attrs를 GT로 바꿔도 geometry F1은 그대로라서 color/type attr가 lane geometry F1 병목은 아니다.
- threshold를 `0.45 -> 0.35`로 낮춰도 `0.5101 -> 0.5169`만 오른다. threshold-only sweep은 0.6 path가 아니다.
- predicted centerline과 GT centerline 사이의 gap은 lane F1 기준 `0.1461`이다. Gate 3 병목은 vectorizer rewrite보다 predicted centerline coverage/quality다.

다음:

- 다음 lane 축은 centerline-core recall/coverage를 올리는 target/loss/calibration 쪽으로 제한한다.
- support map substitution, semantic attr oracle, threshold sweep, vectorizer rewrite부터 시작하는 실험은 반복하지 않는다.

## 14. 2026-05-11 Gate 3 lane target calibration: centerline core width3 is negative

맥락:

- vectorizer recovery audit 이후, predicted centerline coverage를 직접 올리는 가장 작은 target-side 가설을 확인했다.
- 기존 core target은 `LaneSegFirstTargetConfig.centerline_core_width=1`로 고정되어 있었다.
- 이 실험은 target renderer의 core width만 opt-in train config로 열고, `core_centerline_refine_cross_retain` 설정에서 `lane_segfirst_centerline_core_width=3`만 추가했다.

변경:

- branch: `exp/lane-family-f1/lane-centerline-core-width3`
- `lane_segfirst_centerline_core_width` train default를 추가했다.
- encoded train/eval dataloader, raw-batch trainer/evaluator path 모두 같은 `LaneSegFirstTargetConfig`를 사용하게 했다.
- `tools/run_pv26_lane60_probe.py`에 `core_centerline_refine_core_width3` experiment를 추가했다.

검증:

- output run: `runs/pv26_exhaustive_od_lane_train/lane60_core_centerline_refine_core_width3_from_lane60_core_centerline_refine_cross_retain_from_exhaustive_od_lane_default_20260505_032217_default_20260510_003412_default_20260511_030257`
- command: `python3 tools/run_pv26_lane60_probe.py --source-run .../lane60_core_centerline_refine_cross_retain_from_exhaustive_od_lane_default_20260505_032217_default_20260510_003412 --experiment core_centerline_refine_core_width3 --epochs 2 --train-batches 512 --val-batches 128 --batch-size 4 --device cuda:0 --run-root runs/pv26_exhaustive_od_lane_train`
- epoch1: `phase_objective=0.5781`, lane/stop/cross F1 `0.5252 / 0.2000 / 0.6790`.
- epoch2: `phase_objective=0.6002`, lane/stop/cross F1 `0.5232 / 0.4348 / 0.5854`.
- 기준 exact epoch2는 `phase_objective=0.6089`, lane/stop/cross F1 `0.5267 / 0.4483 / 0.5854`.

판단:

- phase objective만 보면 epoch2가 `0.6002`까지 올라가지만, task별 F1 목표는 실패다.
- lane F1은 기준 `0.5267`보다 낮고, stop-line도 기준 `0.4483`보다 낮다.
- core target width-only widening은 predicted centerline coverage 병목을 해결하지 못했다.
- exact val128에서 기준선도 못 넘었으므로 broader-val512나 long run으로 확장하지 않는다.

다음:

- centerline target 폭만 넓히는 실험은 반복하지 않는다.
- 다음 lane 축은 단순 target-width 조절이 아니라 missed-centerline error bucket을 보거나, recall을 올리면서 vectorized FP를 늘리지 않는 구조/학습 신호를 새로 잡아야 한다.

## 15. 2026-05-11 Gate 3 lane centerline error buckets: misses concentrate in truncated/side/near-vertical lanes

맥락:

- target-width-only widening이 실패했으므로, 다음 lane axis를 새로 추측하기 전에 predicted centerline miss가 어떤 lane에 몰리는지 확인했다.
- 목적은 GT lane마다 current centerline probability를 core target 위에서 샘플링하고, 위치/길이/형태/color/type bucket별 `recall@0.45`, `recall@0.90`, miss/dead rate를 남기는 것이다.

변경:

- branch: `exp/lane-family-f1/lane-centerline-error-buckets`
- `tools/probe_pv26_lane60_centerline_error_buckets.py`를 추가했다.
- 출력은 per-lane CSV `lane_centerline_lane_rows.csv`, bucket summary CSV `lane_centerline_bucket_summary.csv`, `summary.json`이다.

검증:

- val128 output: `runs/pv26_exhaustive_od_lane_train/lane60_core_centerline_refine_cross_retain_from_exhaustive_od_lane_default_20260505_032217_default_20260510_003412/analysis_exports/lane_centerline_error_buckets_val128_validation_epoch2`
- val512 output: `runs/pv26_exhaustive_od_lane_train/lane60_core_centerline_refine_cross_retain_from_exhaustive_od_lane_default_20260505_032217_default_20260510_003412/analysis_exports/lane_centerline_error_buckets_val512_validation_epoch2`
- command: `python3 tools/probe_pv26_lane60_centerline_error_buckets.py --checkpoint .../phase_4/checkpoints/best.pt --source-run ... --lane60-experiment core_centerline_refine_cross_retain --phase-index 4 --max-val-batches 512 --validation-epoch 2 --batch-size 4 --device cuda:0 --output-dir .../analysis_exports/lane_centerline_error_buckets_val512_validation_epoch2`
- val512 supervised lanes: `5092`.
- total `recall@0.45 < 0.25`: `291 / 5092`.
- total `dead@0.45`: `54 / 5092`.

주요 val512 buckets:

- `bottom_band <0.50`: lanes `519`, `recall@0.45=0.6677`, miss rate `0.1329`, `recall@0.90=0.5125`.
- `slope_band near_vertical`: lanes `31`, `recall@0.45=0.6611`, miss rate `0.1290`, `recall@0.90=0.4485`.
- `x_band >=0.66`: lanes `1316`, `recall@0.45=0.7500`, miss rate `0.0821`, `recall@0.90=0.6068`.
- `x_band <0.33`: lanes `1631`, `recall@0.45=0.7610`, miss rate `0.0705`, `recall@0.90=0.6265`.
- `x_band 0.33-0.66`: lanes `2145`, `recall@0.45=0.8260`, miss rate `0.0317`, `recall@0.90=0.6997`.

판단:

- centerline miss는 전체적으로 균일하지 않다. center lane보다 side lane이 약하고, bottom이 낮은 truncated lane과 near-vertical lane이 더 약하다.
- 단순 target-width 조절보다, 이 bucket들의 recall을 올리면서 side false-positive fragments를 늘리지 않는 신호가 필요하다.

다음:

- 다음 lane training axis는 side/truncated/near-vertical lane recall을 겨냥한다.
- threshold-only, target-width-only, semantic attr oracle, support substitution은 반복하지 않는다.

## 16. 2026-05-11 Gate 3 lane side-band BCE weighting: small lane-vector lift, no centerline-core gain

맥락:

- centerline error-bucket audit에서 center x-band보다 side lanes가 약했다.
- 그래서 전체 lane loss ratio나 target width를 다시 흔들지 않고, centerline core positive 중 좌우 x-band만 BCE에서 더 강하게 보는지 확인했다.
- 이 실험은 truncated/near-vertical 판별까지 섞지 않고 side-band weighting만 한 단일 축이다.

변경:

- branch: `exp/lane-family-f1/lane-centerline-side-bucket-weight`
- `PV26MultiTaskLoss`에 opt-in `lane_segfirst_centerline_side_positive_weight`와 `lane_segfirst_centerline_side_band_fraction`을 추가했다. 기본값은 기존 동작과 같은 `1.0 / 0.33`이다.
- `core_centerline_refine_side_bce_focus` probe는 outer 33% x-band의 centerline-core positive BCE weight만 `2.0`으로 올린다.

검증:

- output run: `runs/pv26_exhaustive_od_lane_train/lane60_core_centerline_refine_side_bce_focus_from_lane60_core_centerline_refine_cross_retain_from_exhaustive_od_lane_default_20260505_032217_default_20260510_003412_default_20260511_033647`
- command: `python3 tools/run_pv26_lane60_probe.py --source-run .../lane60_core_centerline_refine_cross_retain_from_exhaustive_od_lane_default_20260505_032217_default_20260510_003412 --experiment core_centerline_refine_side_bce_focus --epochs 2 --train-batches 512 --val-batches 128 --batch-size 4 --device cuda:0 --run-root runs/pv26_exhaustive_od_lane_train`
- epoch1: `phase_objective=0.5812`, lane/stop/cross F1 `0.5195 / 0.2000 / 0.6790`.
- epoch2: `phase_objective=0.6083`, lane/stop/cross F1 `0.5331 / 0.4348 / 0.5854`.
- 기준 exact epoch2는 `phase_objective=0.6089`, lane/stop/cross F1 `0.5267 / 0.4483 / 0.5854`.
- dense-map PR on best checkpoint: lane centerline-core best pixel F1 `0.5705` at threshold `0.9`, precision/recall `0.4912 / 0.6803`; lane support best pixel F1 `0.7979`.
- 기준 dense-map PR은 lane centerline-core F1 `0.5729`, lane support F1 `0.7971`이었다.

판단:

- vectorized lane F1은 `+0.0064` 올랐지만, phase objective와 stop-line F1이 기준보다 낮다.
- centerline-core pixel F1도 개선되지 않았으므로, 이 gain은 목표한 side-centerline quality 개선이 아니다.
- exact gate에서 기준선을 명확히 넘지 못했으므로 broader-val512로 확장하지 않는다.

하지 말 것:

- side-band BCE positive weighting만 같은 형태로 반복하지 않는다.
- lane vector F1 소폭 상승만 보고 0.6 path로 확장하지 않는다.

다음:

- side/truncated/near-vertical miss는 여전히 유효한 진단이지만, 단순 BCE 가중치가 아니라 instance/geometry-aware recall 신호가 필요하다.

## 17. 2026-05-11 Gate 3 lane side-band margin loss: exact objective nudged, centerline-core regressed

맥락:

- side-band BCE weighting이 vectorized lane F1만 소폭 올리고 centerline-core dense PR을 개선하지 못했다.
- 다음으로 side-band positive centerline pixels에 `sigmoid(logit) >= margin`을 직접 요구하는 opt-in margin loss를 확인했다.
- 목적은 BCE 재가중치보다 더 직접적으로 side centerline probability를 끌어올릴 수 있는지 보는 것이었다.

변경:

- branch: `exp/lane-family-f1/lane-centerline-side-margin`
- `PV26MultiTaskLoss`에 opt-in `lane_segfirst_centerline_side_margin_weight`, `lane_segfirst_centerline_side_margin`, `lane_segfirst_centerline_side_margin_band_fraction`을 추가했다. 기본값은 기존 동작과 같은 비활성 상태다.
- `core_centerline_refine_side_margin` probe는 outer 33% x-band centerline-core positive pixels에 probability margin `0.9`, loss weight `1.0`을 적용한다.

검증:

- output run: `runs/pv26_exhaustive_od_lane_train/lane60_core_centerline_refine_side_margin_from_lane60_core_centerline_refine_cross_retain_from_exhaustive_od_lane_default_20260505_032217_default_20260510_003412_default_20260511_035507`
- command: `python3 tools/run_pv26_lane60_probe.py --source-run .../lane60_core_centerline_refine_cross_retain_from_exhaustive_od_lane_default_20260505_032217_default_20260510_003412 --experiment core_centerline_refine_side_margin --epochs 2 --train-batches 512 --val-batches 128 --batch-size 4 --device cuda:0 --run-root runs/pv26_exhaustive_od_lane_train`
- epoch1: `phase_objective=0.5867`, lane/stop/cross F1 `0.5409 / 0.2000 / 0.6790`.
- epoch2: `phase_objective=0.6097`, lane/stop/cross F1 `0.5352 / 0.4522 / 0.5854`.
- 기준 exact epoch2는 `phase_objective=0.6089`, lane/stop/cross F1 `0.5267 / 0.4483 / 0.5854`.
- dense-map PR on best checkpoint: lane centerline-core best pixel F1 `0.5573` at threshold `0.9`, precision/recall `0.4502 / 0.7313`; lane support best pixel F1 `0.7977`.

판단:

- exact objective와 vectorized lane/stop-line F1은 기준보다 아주 조금 올랐다.
- 하지만 centerline-core pixel F1이 `0.5729 -> 0.5573`으로 내려갔다. recall은 올라갔지만 precision이 크게 떨어져 side/fragment FP 위험이 커진 형태다.
- centerline 병목을 직접 푼 신호가 아니므로 broader-val512로 확장하지 않는다.

하지 말 것:

- side-band probability margin loss만 같은 형태로 반복하지 않는다.
- exact objective `+0.0008`만 보고 lane 0.6 path로 확장하지 않는다.

다음:

- side/truncated/near-vertical miss 진단은 유지하되, probability를 더 밀어붙이는 방식보다 instance/geometry-aware recall 신호가 필요하다.

## 18. 2026-05-11 Gate 3 lane geometry-risk recall: vector F1 nudged, dense centerline-core regressed

맥락:

- error-bucket audit은 side, truncated, near-vertical lane이 centerline miss에 취약하다고 봤다.
- side-BCE와 side-margin은 x-band probability만 밀어붙여 centerline-core PR을 개선하지 못했다.
- 그래서 좌표 band만이 아니라 lane instance geometry bucket을 target에 남기고, 해당 core pixels에 recall-only loss를 걸었다.

변경:

- branch: `exp/lane-family-f1/lane-centerline-geometry-risk-recall`
- `render_lane_segfirst_targets`가 side/truncated/near-vertical lane core pixels를 `lane_seg_centerline_geometry_risk`로 내보내게 했다.
- `PV26MultiTaskLoss`에 opt-in `lane_segfirst_geometry_risk_recall_weight`를 추가했다. 기본값은 기존 동작과 같은 `0.0`이다.
- `core_centerline_refine_geometry_risk_recall` probe는 geometry-risk recall weight `0.75`를 켠다.

검증:

- output run: `runs/pv26_exhaustive_od_lane_train/lane60_core_centerline_refine_geometry_risk_recall_from_lane60_core_centerline_refine_cross_retain_from_exhaustive_od_lane_default_20260505_032217_default_20260510_003412_default_20260511_041807`
- command: `python3 tools/run_pv26_lane60_probe.py --source-run .../lane60_core_centerline_refine_cross_retain_from_exhaustive_od_lane_default_20260505_032217_default_20260510_003412 --experiment core_centerline_refine_geometry_risk_recall --epochs 2 --train-batches 512 --val-batches 128 --batch-size 4 --device cuda:0 --run-root runs/pv26_exhaustive_od_lane_train`
- epoch1: `phase_objective=0.5852`, lane/stop/cross F1 `0.5400 / 0.2000 / 0.6790`.
- epoch2: `phase_objective=0.6086`, lane/stop/cross F1 `0.5405 / 0.4348 / 0.5854`.
- 기준 exact epoch2는 `phase_objective=0.6089`, lane/stop/cross F1 `0.5267 / 0.4483 / 0.5854`.
- dense-map PR on best checkpoint: lane centerline-core best pixel F1 `0.5572` at threshold `0.9`, precision/recall `0.4479 / 0.7370`; lane support best pixel F1 `0.7978`.

판단:

- exact lane F1은 기준보다 `+0.0138` 올랐지만, phase objective와 stop-line F1은 기준보다 낮다.
- centerline-core pixel F1은 `0.5729 -> 0.5572`로 내려갔다. recall은 올랐지만 precision이 더 크게 떨어져 fragment/FP 위험이 커진 형태다.
- 이 축도 centerline 병목 해결이 아니라 vectorized metric partial-positive다. broader-val512로 확장하지 않는다.

하지 말 것:

- geometry-risk recall-only loss를 같은 형태로 반복하지 않는다.
- exact lane F1 `0.5405`만 보고 0.6 path로 확장하지 않는다.

다음:

- side/truncated/near-vertical miss 진단은 유지하되, recall-only positive pressure만으로는 부족하다.
- 다음 lane 축은 risk lane의 false-positive/fragment 제어까지 같이 갖는 더 정밀한 instance/geometry-aware contract여야 한다.

## 19. 2026-05-11 Gate 3 lane geometry-risk local Tversky: centerline-core gain is noise-level

맥락:

- geometry-risk recall-only loss는 exact lane F1을 올렸지만 dense centerline-core F1을 크게 낮췄다.
- 이번 실험은 같은 side/truncated/near-vertical risk instance 주변 local support에서 false-positive까지 같이 벌주는 Tversky term을 추가했다.
- 목적은 risk lane recall pressure를 유지하면서 fragment/FP 위험을 줄일 수 있는지 보는 것이었다.

변경:

- branch: `exp/lane-family-f1/lane-centerline-geometry-risk-local-tversky`
- `render_lane_segfirst_targets`가 `lane_seg_centerline_geometry_risk_support`를 추가로 내보내게 했다.
- `PV26MultiTaskLoss`에 opt-in `lane_segfirst_geometry_risk_local_tversky_weight`를 추가했다. 기본값은 기존 동작과 같은 `0.0`이다.
- `core_centerline_refine_geometry_risk_local_tversky` probe는 geometry-risk local Tversky weight `0.75`를 켠다.

검증:

- output run: `runs/pv26_exhaustive_od_lane_train/lane60_core_centerline_refine_geometry_risk_local_tversky_from_lane60_core_centerline_refine_cross_retain_from_exhaustive_od_lane_default_20260505_032217_default_20260510_003412_default_20260511_043546`
- command: `python3 tools/run_pv26_lane60_probe.py --source-run .../lane60_core_centerline_refine_cross_retain_from_exhaustive_od_lane_default_20260505_032217_default_20260510_003412 --experiment core_centerline_refine_geometry_risk_local_tversky --epochs 2 --train-batches 512 --val-batches 128 --batch-size 4 --device cuda:0 --run-root runs/pv26_exhaustive_od_lane_train`
- epoch1: `phase_objective=0.5798`, lane/stop/cross F1 `0.5160 / 0.2000 / 0.6790`.
- epoch2: `phase_objective=0.6089`, lane/stop/cross F1 `0.5306 / 0.4522 / 0.5854`.
- 기준 exact epoch2는 `phase_objective=0.6089`, lane/stop/cross F1 `0.5267 / 0.4483 / 0.5854`.
- dense-map PR on best checkpoint: lane centerline-core best pixel F1 `0.5738` at threshold `0.9`, precision/recall `0.5138 / 0.6498`; lane support best pixel F1 `0.7971`.
- 기준 dense-map PR은 lane centerline-core F1 `0.5729`, lane support F1 `0.7971`이었다.

판단:

- exact objective와 stop-line F1은 기준보다 극소폭 높지만, lane F1 gain은 `+0.0039`뿐이다.
- dense centerline-core F1도 기준 대비 `+0.0009`라 실질적인 centerline 병목 해결 신호로 보기 어렵다.
- recall-only probe의 exact lane F1 `0.5405`보다 낮으므로 risk-lane vectorized metric 측면에서도 더 약하다.
- broader-val512로 확장하지 않는다.

하지 말 것:

- geometry-risk local Tversky loss만 같은 형태로 반복하지 않는다.
- exact objective `+0.00004` 또는 dense F1 `+0.0009` 수준의 노이즈를 lane 0.6 path로 취급하지 않는다.

다음:

- side/truncated/near-vertical miss 진단은 유지하되, BCE weight-only, margin-only, recall-only, local false-positive penalty-only는 닫는다.
- 다음 lane 축은 risk bucket을 다시 누르는 게 아니라 prediction confidence/fragment separation 자체를 바꾸는 contract여야 한다.

## 20. 2026-05-11 Gate 2 stop-line heatmap-support geometry target: dense maps regress

맥락:

- stop-line decoder는 predicted center/proposal 위치에서 offset/angle/half-length를 읽는다.
- 기존 dense target은 center heatmap 주변을 positive로 만들지만 geometry target은 floor center cell 한 곳에만 기록한다.
- 이번 실험은 center peak가 주변 support로 움직여도 geometry가 supervised되도록 heatmap support 전체에 geometry target을 채우는 opt-in target 계약을 확인했다.

변경:

- branch: `exp/lane-family-f1/stopline-heatmap-geometry-support`
- `build_stopline_dense_targets`에 opt-in `geometry_target_mode=heatmap_support`를 추가했다. 기본값은 기존 동작인 `center_cell`이다.
- trainer/evaluator encode path가 `PV26MultiTaskLoss.stopline_geometry_target_mode`를 raw-batch encoding에 전달하게 했다.
- `core_centerline_refine_stop_heatmap_geometry_support` probe는 dataloader pre-encoding을 끄고 `stopline_geometry_target_mode=heatmap_support`를 켠다.

검증:

- output run: `runs/pv26_exhaustive_od_lane_train/lane60_core_centerline_refine_stop_heatmap_geometry_support_from_lane60_core_centerline_refine_cross_retain_from_exhaustive_od_lane_default_20260505_032217_default_20260510_003412_default_20260511_045646`
- command: `python3 tools/run_pv26_lane60_probe.py --source-run .../lane60_core_centerline_refine_cross_retain_from_exhaustive_od_lane_default_20260505_032217_default_20260510_003412 --experiment core_centerline_refine_stop_heatmap_geometry_support --epochs 2 --train-batches 512 --val-batches 128 --batch-size 4 --device cuda:0 --run-root runs/pv26_exhaustive_od_lane_train`
- epoch1: `phase_objective=0.5594`, lane/stop/cross F1 `0.5116 / 0.1071 / 0.6707`.
- epoch2: `phase_objective=0.5618`, lane/stop/cross F1 `0.5178 / 0.2338 / 0.5476`, stop-line TP/FP/FN `18 / 76 / 42`.
- 기준 exact epoch2는 `phase_objective=0.6089`, lane/stop/cross F1 `0.5267 / 0.4483 / 0.5854`.
- dense-map PR on best checkpoint: stop-line mask best pixel F1 `0.4854`, stop-line center best pixel F1 `0.0815`, lane centerline-core best pixel F1 `0.5728`.
- 기준 dense-map PR은 stop-line mask F1 `0.6118`, stop-line center F1 `0.1396`, lane centerline-core F1 `0.5729`였다.

판단:

- exact stop-line F1이 `0.4483 -> 0.2338`로 크게 후퇴했다.
- dense stop-line mask와 center heatmap도 같이 후퇴했다. geometry target support를 넓힌 것이 readout reliability를 올린 게 아니라 dense stop-line maps 자체를 망가뜨린 형태다.
- broader-val512로 확장하지 않는다.

하지 말 것:

- heatmap-support geometry target fill을 같은 형태로 반복하지 않는다.
- center/proposal reliability 문제를 geometry target 위치 확장만으로 해결하려고 하지 않는다.

다음:

- stop-line을 재개한다면 target support를 넓히는 방향보다, predicted center/proposal confidence를 분리해 안정화하거나 PCA/component-fit weak-positive를 넘어서는 다른 geometry recovery contract를 찾아야 한다.
