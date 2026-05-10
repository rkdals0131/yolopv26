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

확인할 것:

- 같은 checkpoint와 same postprocess config로 larger validation slice objective/F1 replay.
- lane/stop/cross task별 TP/FP/FN 변화.
- small-fragment filter가 recall을 과하게 깎는 장면이 있는지 comparison grid 확인.

성공 기준:

- objective gain이 exact epoch-2에만 과적합된 현상이 아니어야 한다.
- lane recall 손실이 과도하면 filter threshold를 deployment default로 승격하지 않는다.
- F1 0.6+ plan의 baseline으로 쓸 broader-val lane/stop/cross F1을 확정한다.

## 5. Gate 2: stop-line first improvement axis

목적:

- F1 0.6+ 목표의 최대 gap인 stop-line을 먼저 올린다.

후보:

- stop-line dense mask/center heatmap diagnostics를 broader-val에서 다시 export한다.
- stop-line FP/FN을 feature audit으로 나누고, component/aspect/score filter가 TP를 자르는지 확인한다.
- stop-line-only 또는 stop-line-heavy short fine-tune을 하되 lane/crosswalk regression을 같은 eval에서 같이 본다.
- 필요하면 stop-line geometry head의 target/loss/decoder contract를 재검토한다.

성공 기준:

- stop-line F1이 broader-val에서 의미 있게 상승해야 한다.
- lane/crosswalk F1이 0.6 목표에서 멀어질 정도로 무너지면 실패다.
- exact subset에서만 좋아지는 stop-line threshold tweak은 채택하지 않는다.

## 6. Gate 3: lane recall without fragment FP

목적:

- lane F1을 0.6까지 끌어올리되, final geometry filters가 제거한 small-fragment FP를 다시 만들지 않는다.

후보:

- core centerline + gated refinement는 유지한다.
- lane centerline recall 부족인지, vectorizer recovery 부족인지 broader-val dense-map PR과 vectorizer audit으로 분리한다.
- support map을 단순 대체하거나 직접 residual input으로 넣는 방식은 이미 negative evidence가 있으므로 반복하지 않는다.
- 새 lane training axis는 stop-line plan과 섞지 않고 별도 short run으로 본다.

성공 기준:

- lane F1이 올라야 하고, FP 감소만으로 recall이 무너지는 개선은 실패다.
- broader-val comparison grid에서 긴 실제 차선이 빠지는 장면이 늘면 실패다.

## 7. Gate 4: crosswalk retention to 0.6

목적:

- crosswalk F1을 `0.5854` 근처에서 안정적으로 0.6 이상으로 넘긴다.

후보:

- 먼저 broader-val에서 현재 crosswalk F1이 유지되는지 확인한다.
- crosswalk-heavy loss 재시도는 이미 negative evidence가 있으므로 기본 후보가 아니다.
- 필요하면 crosswalk polygon area/aspect thresholds의 recall 손실을 audit한다.

성공 기준:

- crosswalk F1 `>=0.60`이 broader-val에서 유지되어야 한다.
- lane/stop-line 목표를 희생하는 crosswalk-only gain은 채택하지 않는다.

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
