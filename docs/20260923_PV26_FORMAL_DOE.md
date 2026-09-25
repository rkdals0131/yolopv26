# PV26 후속 파인튜닝 DoE

작성: 2026-09-23. 이 문서는 **실험 설계와 자동 실행 설정**이다. 80k 이후 FC 정책 비교는 4,000 step, 기존 F 6후보 탐색은 1,000 step까지 완료했다. 아래의 새 DoE 87회는 별도 실험이다.

## 판정과 현재 6후보의 위치

기존 F 6후보는 joint/roadmark-only 2방식과 LR 3수준을 빠르게 훑는 레시피 탐색이다. joint에서 body·detector head·roadmark LR이 함께 바뀌며 roadmark-only에서는 body와 detector head가 학습되지 않는다. 게다가 roadmark-only는 roadmark 원본만 소비한다. **따라서 이 6개에서 각 LR의 독립 효과나 학습 범위×LR 상호작용을 추정할 수 없다.** 이 결과를 정식 DoE의 완료라고 부르지 않는다.

DoE는 모든 가능한 방법의 단순 곱집합이 아니다. 같은 데이터와 업데이트 수에서 비교할 수 있는 요인들을 블록 안에서 교차시키고, 다른 의미를 가진 방법은 별도 블록에서 비교한다. [NIST의 실험 설계 안내](https://www.itl.nist.gov/div898/handbook/pri/section3/pri33.htm)는 screening과 상호작용 추정을 구분하고, [교락 설명](https://www.itl.nist.gov/div898/handbook/pri/section3/pri3343.htm)은 함께 움직이는 요인들의 효과를 분리할 수 없음을 보여준다. 이 원칙을 현재 PV26 코드가 지원하는 요인에 적용했다.

### 공통 출발점과 응답

- 모든 후보는 [80k latest 가중치](../runs/20260922_1607_joint_lr3x_80k/checkpoints/latest.pt)에서 **새 optimizer**로 시작한다. 검출 손실 정책은 4,000-step 개발 비교에서 선정한 restart로 고정한다.
- train/val membership은 80k 실행의 저장 목록을 재사용한다. 개발 평가는 두 출처에서 1,024장씩 같은 목록으로 계산한다. 최종 [수동주행 MCAP](20260923_MCAP_FINAL_EVALUATION_PROTOCOL.md)은 모든 요인 선택·threshold 조정에서 제외한다.
- 주 응답은 **흰 차선 F1과 노란 차선 F1의 평균**이다. 표본 수가 많은 흰 차선만으로 연구 방향을 고르지 않기 위한 선택이다. 도로표식 전체 F1, 색상별 P/R, 정지선 F1·오탐, 신호등 종류별 F1/recall과 처리시간도 별도로 보고한다.
- 신호등을 낮춰 얻은 차선 개선은 최종 채택 후보로 자동 승격하지 않는다. 현재 제품 요구사항에는 허용 가능한 신호등 하락폭이 없어 숫자를 임의로 만들지 않는다. trade-off는 그대로 표시하고 확인이 필요한 결과로 둔다.
- 각 셀은 seed 26·27·28로 반복한다. 원본 목록은 고정하되 sampler draw와 증강은 seed별로 달라진다. 한 seed의 짧은 구간 성능을 정식 효과로 단정하지 않는다.

## 교차 블록과 별도 레시피 블록

| 블록 | 요인과 수준 | 셀 × seed | 무엇을 읽는가 |
| --- | --- | ---: | --- |
| A: LR | joint의 body LR 0.3/1/3배 × roadmark LR 0.3/1/3배. detector head LR은 3e-4 고정 | 3×3×3 = **27** | body와 차선 decoder LR의 주효과·교차 효과·비선형 변화 |
| B: 최적화·gradient | AdamW cosine / Schedule-Free AdamW × sum / PCGrad / GradNorm. microbatch 8, logical 32 고정 | 2×3×3 = **18** | optimizer **레시피**와 gradient 전략의 교차 효과 |
| C: 학습 노출·손실 | traffic:roadmark 2:1 / 1:1 / 1:2 × roadmark loss weight 0.5 / 1 / 2 | 3×3×3 = **27** | roadmark 노출과 loss 비중의 주효과·교차 효과 |
| D: 별도 optimizer 방식 | AdamW cosine / AdamW constant / Prodigy d_coef 0.3 / 1 / 3 | 5×3 = **15** | 구조가 다른 optimizer 레시피의 조건부 비교. 완전 교차 효과로 해석하지 않음 |

첫 rung은 **총 87회의 서로 구분되는 학습 실행**이다. A/B/C의 72회가 교차 블록이고 D의 15회는 중첩된 optimizer 레시피 비교다. 기존 6후보와 FC 두 후보는 별도 pilot이며 87에 중복 계산하지 않는다.

### A: 학습률 설계

기준 LR은 body 3e-5, detector head 3e-4, roadmark 9e-4다. A는 head를 3e-4로 고정하고 body와 roadmark만 각각 0.3/1/3배로 바꾼다. 3×3 전체 셀을 채워 body LR과 roadmark LR이 서로 보완하거나 충돌하는지 살핀다. 기존 FJ-L/M/H는 head LR도 함께 바뀌어 A의 대각선 자료로 재사용하지 않는다. FD-L/M/H는 입력 source 구성과 감독 태스크가 달라 A의 셀이 아니다.

### B: optimizer 레시피와 gradient

AdamW cosine은 현재 대조군이다. Schedule-Free는 constant LR, 50-step warmup 및 학습 자료의 BN 보정이 포함된 별도 **레시피**다. 따라서 B에서 추정하는 주효과는 optimizer와 schedule/평균화의 묶음이지 AdamW와 Schedule-Free 알고리즘만의 순수 효과가 아니다. 같은 body/head/roadmark 초기 LR, logical batch 32, microbatch 8을 두 레시피에서 고정한다.

각 레시피에 sum, PCGrad, GradNorm alpha=1.5를 모두 교차시킨다. PCGrad는 충돌 gradient를 처리하고 [원 논문](https://papers.neurips.cc/paper_files/paper/2020/file/3fe78a8acf5fda99de95303940a2420c-Paper.pdf), GradNorm은 태스크의 gradient 크기를 맞춘다([원 논문](https://proceedings.mlr.press/v80/chen18a.html)). 두 방법의 추가 역전파와 실패·OOM을 점수 0으로 대체하지 않고 status=failed로 구분한다. 이전 초기학습 search의 GradNorm 실패를 이 실험의 열등한 정확도라고 해석하지 않는다.

### C: 노출과 손실의 교차

source sampling 비율과 roadmark loss weight는 결과적으로 같은 gradient 기여를 만들 수 있지만, 데이터의 **어떤 장면을 얼마나 보는지**와 **본 뒤 얼마만큼 반영하는지**가 다른 요인이다. C의 9셀을 채워 두 조절 방식의 교차 효과를 확인한다. 총 image draws와 logical batch는 고정하고 실제 traffic/roadmark draw 수를 함께 보고한다.

### D: 비교 가능성의 경계

AdamW cosine을 D 안에 다시 포함시켜 같은 시기의 기준으로 삼는다. AdamW constant는 일정의 효과가 optimizer와 분리되는 보조 비교다. 현행 Prodigy 구현은 AdamW의 body/head/roadmark LR group 비율을 사용하지 않고 d_coef를 새로 추정한다. 그래서 Prodigy 세 수준을 A나 B의 LR 배수에 억지로 끼우지 않는다. Schedule-Free의 train/eval weights와 BN 정책도 B 안에서 함께 평가한다.

Weight decay, batch, sampler, photometric 증강, soft target, 보조 geometry loss, 새 모델 head는 이번 87회에 섞지 않는다. A/B/C의 상호작용과 실제 실패 사례를 보고 별도 블록을 설계한다. 한 블록의 승자를 다른 블록의 모든 셀에 소급 적용했다고 주장하지 않는다.

## 무작위 순서, 단계 예산, 분석

후보 실행 순서는 블록별 고정 난수로 한 번 섞어 파일에 저장했다. 동일 seed의 9셀을 시간순으로 몰아 GPU 온도·장치 상태와 요인 수준이 겹치지 않게 한다. 저장한 candidate 이름, 요인 수준, seed, 시작 checkpoint, index 경로와 stage2 설정은 각 run_config.json 및 search_results.json에서 확인할 수 있다.

각 후보는 처음부터 cosine horizon을 12,000 step으로 고정하고 300→1,000→4,000→12,000 step의 **누적** rung을 사용한다. 300-step은 실행 가능성과 큰 퇴행을 보는 screening이다. 교차 효과를 계산하려는 rung에서는 셀×seed가 모두 끝나야 한다. 일부 후보를 조기에 제외한 다음 rung은 **후보 비교 단계**이며 원래 블록의 완전 교차 효과로 분석하지 않는다. [Hyperband 원 논문](https://www.jmlr.org/papers/v18/16-558.html)의 자원 배분 아이디어를 쓰되, early stopping이 효과 추정의 균형을 깨뜨리는 점을 명시한다.

분석은 모든 셀을 같은 seed 세 개에서 얻었을 때 cell 평균, 요인별 주효과, 두 주효과를 뺀 교차 잔차를 계산한다. 각 셀과 같은 seed의 중심 설정을 쌍으로 비교한다. 3개 seed의 분산 범위를 그대로 보여주며 p-value나 일반화 확신도를 만들지 않는다. 조건별 예측 오류는 동일한 개발 이미지에서 확인한다. 최종 MCAP으로 효과를 선택하지 않는다.

단일 RTX 4060에서 블록 하나씩 직렬로 실행한다. 첫 rung 87회를 모두 끝내면 최소 수 시간의 GPU 시간과 각 run의 약 0.4GiB checkpoint·index snapshot이 든다. 현재 runs의 여유 공간은 이후 시작 시 재확인한다. B의 PCGrad/GradNorm은 더 느리거나 실패할 수 있으므로 고정 소요 시간을 약속하지 않는다.

## 자동 실행

[생성 스크립트](../tools/generate_pv26_stage2_doe.py)는 위 4개 블록의 구체적 후보와 seed를 [config](../config/)에 기록한다. 기존 [자동 학습 스크립트](../tools/run_pv26_method_search.py)는 한 블록의 후보를 직렬 실행하고 80k 가중치에서 시작하며, 같은 개발 목록으로 두 태스크를 평가한다. [분석 스크립트](../tools/analyze_pv26_stage2_doe.py)는 동일 rung의 셀×seed 완료 여부, 주효과와 교차 잔차를 문서로 쓴다.

[블록 실행 스크립트](../tools/run_pv26_stage2_doe.py)는 네 블록을 한 GPU에서 순서대로 실행하고 각 블록이 끝나면 분석 Markdown을 저장한다. 기본값은 첫 300-step rung이고 저장 위치는 runs/20260923_stage2_doe_*다.

~~~bash
cd /home/kai/yolopv26
.venv/bin/python tools/generate_pv26_stage2_doe.py
.venv/bin/python tools/run_pv26_stage2_doe.py --through-images 9600
.venv/bin/python tools/run_pv26_method_search.py --base-config config/pv26_stage2.yaml --search-config config/pv26_stage2_doe_lr.yaml --through-images 9600
.venv/bin/python tools/analyze_pv26_stage2_doe.py --run runs/20260923_stage2_doe_lr --resource-images 9600
~~~

B/C/D는 각각 pv26_stage2_doe_optimizer_gradient.yaml, pv26_stage2_doe_balance.yaml, pv26_stage2_doe_optimizer_nested.yaml을 --search-config에 사용한다. 위 명령 중 블록 실행 스크립트는 A/B/C/D를 모두 실행하므로 한 블록만 시험할 때에는 뒤의 개별 명령을 사용한다. 각 파일의 experiment.name이 별도 runs 경로를 정한다. 같은 스크립트에 --through-images 32000 등을 주면 저장된 후보가 이어지고, 검토 후 --promote 이름...으로 다음 rung의 후보를 제한할 수 있다.

현재 문서 작성 시점에 87회의 결과는 없다. 새 블록의 완료 수와 분석 파일은 각 runs 디렉터리에서만 확인한다. **DoE를 설계·생성한 것과 DoE의 효과가 검증된 것은 서로 다른 상태다.**
