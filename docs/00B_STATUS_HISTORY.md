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

