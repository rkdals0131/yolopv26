## 결론

**다음 본학습의 기준 후보는 `adamw_lr_3x`로 잡는 것이 맞습니다.** `schedulefree_10x`는 비교 후보로 남길 만합니다. 다만 이번 결과를 설명하는 데는 optimizer 종류보다 먼저 확인해야 할 변수가 있습니다.

특히 새로 확인한 핵심은 **정답 마스크에서도 발생하는 선 복원 손실**, **이미지별 detector loss 정규화**, **source별로 분리된 BN 재추정 배치**입니다. 이 셋은 현재 문서의 원인 분석을 상당히 보완합니다.

코드와 결과 JSON을 대조했고 GradNorm 오류와 PCGrad 계산 및 정답 마스크 복원은 CPU에서 직접 시험했습니다. 체크포인트가 없어 실제 모델의 재추론과 BN 재평가는 수행하지 않았습니다.

[전체 검토문 Markdown](sandbox:/mnt/data/pv26_training_review_20260922.md) · [재현 코드와 결과 및 GradNorm 최소 패치](sandbox:/mnt/data/pv26_review_evidence_20260922.zip)

---

## 1. 완료된 네 후보를 비교하면

아래는 모두 **28,800-image 학습 예산과 8,192장 전체 validation**입니다. 출처는 각 후보의 `validation_full_latest.json`입니다.

| 후보                 |  Step |      신호등 F1 | 도로표식 line F1 |           Q |   도로표식 FP |
| ------------------ | ----: | ----------: | -----------: | ----------: | --------: |
| **AdamW LR 3×**    |   900 | **0.57538** |  **0.40222** | **0.48880** |    11,749 |
| Schedule-Free 10×  |   900 |     0.52297 |      0.38480 |     0.45388 | **9,989** |
| Roadmark loss 0.5× |   900 | **0.59260** |      0.02959 |     0.31109 |   395,706 |
| Logical batch 16   | 1,800 |     0.58293 |      0.05102 |     0.31698 |   240,963 |

공동학습 후보로는 AdamW 3×가 가장 설득력 있습니다. 신호등 F1만 보면 다른 두 후보가 조금 높지만 도로표식 오탐이 너무 많습니다. 특히 `roadmark_loss_0p5x`의 초기 결과를 근거로 “두 태스크의 균형이 좋아졌다”고 설명했던 부분은 최종 결과를 반영해 수정해야 합니다.

그렇다고 AdamW가 SF보다 모든 면에서 좋은 것은 아닙니다.

| 세부 지표          |   AdamW 3× |      SF 10× |
| -------------- | ---------: | ----------: |
| 신호등 precision  |     0.7195 |  **0.7361** |
| 신호등 recall     | **0.4794** |      0.4056 |
| 보행자 신호등 recall | **0.2187** |      0.0883 |
| 도로표식 pixel F1  |     0.5084 |  **0.5135** |
| 노란 차선 line F1  |     0.3256 |  **0.3635** |
| 정지선 line F1    |    0.01310 | **0.01921** |

**AdamW는 더 많이 찾아서 aggregate F1이 높고 SF는 상대적으로 보수적인 검출 결과입니다.** 같은 precision 또는 같은 FP/image 조건에서 recall을 비교하면 차이가 달라질 수 있습니다.

정지선은 둘 다 별도 개선이 필요합니다. AdamW의 정지선 TP/FP/FN은 **19/2,771/92**이고 SF는 **15/1,436/96**입니다. SF의 정지선 F1 우위도 TP가 많아서가 아니라 FP가 적어서 나타납니다.

따라서 현재 판단은 다음과 같습니다.

> **AdamW 3×를 다음 학습의 기준으로 채택하되 SF 10×와의 비교는 BN 및 operating threshold를 맞춘 후 확정한다. 정지선 성능은 Q와 분리해서 관리한다.**

나머지 다섯 후보의 최종 결과는 이번 판단에 포함하지 않았습니다.

---

## 2. 분석 문서에서 바꿔야 할 원인 해석

기존 문서는 가설과 관측을 꽤 잘 구분하고 있습니다. 특히 WD의 직접 shrink가 작다는 계산이나 source 비율과 loss weight를 구분한 설명은 좋습니다. 보완할 부분은 주로 **제목의 강도와 빠진 대안 설명**입니다.

| 실험                               | 유지할 수 있는 설명                           | 보완할 설명                                                             |
| -------------------------------- | ------------------------------------- | ------------------------------------------------------------------ |
| Baseline / LR 0.3×               | 초기 도로표식 FP가 많고 낮은 LR에서 신호등 recall이 낮다 | “새 head의 적응이 느리다” 외에 eval BN과 confidence 보정 상태도 원인 후보입니다.          |
| 전체 LR 3×                         | TP 증가와 FP 급감은 실제 개선이다                 | 특징의 구분 능력 개선인지 전체 logit 이동인지 BN 변화인지 분리되지 않았습니다.                   |
| Decoder LR 1e-3 / 2e-3           | 이 seed의 초기 결과는 기본 3e-3보다 낮다           | 이 세 점으로 decoder 9e-3의 효과나 전체 LR 3×의 원인을 설명할 수 없습니다.                |
| Constant                         | 첫 rung에서는 R이 오르고 D가 내려간다              | 후반 LR 노출이 다른 조기 수렴 비교입니다. detector 내부 loss 일정도 그대로 남아 있습니다.        |
| WD 0 / 1e-3                      | 현재 조합에서의 점수 차이는 확인된다                  | 과적합이나 최적 WD의 근거로는 부족합니다. BN과 bias까지 decay하는 정책도 포함된 비교입니다.         |
| Augmentation off                 | 증강을 유지할 실용적 근거는 있다                    | 일반화 개선뿐 아니라 학습 중 BN 분포 변화도 가능합니다. 수평 반전과 광도 증강의 효과도 분리되지 않았습니다.    |
| Source 비율 두 종                    | 노출 비율에 따른 D/R 변화                      | 양성 support와 gradient 분산 및 BN 구성의 변화입니다. task loss 계수 변경과 동치가 아닙니다. |
| Roadmark loss 0.5× / 2×          | 공유 gradient의 상대 비중이 바뀐다               | 0.5×의 초기 동반 개선을 장기적인 균형 개선으로 확장하면 안 됩니다. 최종 roadmark FP가 매우 큽니다.   |
| Least-used-of-two                | 파일 방문 편차 감소                           | 장면 다양성이나 희귀 양성의 균형을 보장하지 않습니다.                                     |
| Batch 16 / 64 / 64+LR2× / Micro8 | 각각의 전체 조합에 대한 관측                      | update 횟수와 moment의 시간축 및 BN physical batch가 동시에 변합니다.              |
| PCGrad                           | 이번 구현과 micro10 조합의 성능                 | 작은 음의 cosine만으로 투영의 영향도 작다고 설명하기 어렵습니다. 아래 계산이 중요합니다.              |
| GradNorm 두 종                     | 구현 오류로 중단                             | 성능과 alpha 우열에 대한 관측은 없습니다.                                         |
| SF 세 종                           | 높은 LR에서 결과가 회복된다                      | 평균 iterate의 적응 지연과 BN 재추정 영향을 분리해야 합니다.                            |
| Prodigy 세 종                      | 현재 wrapper 설정에서 R이 낮다                 | 공통 `d`와 모듈별 LR 비율 제거가 함께 바뀐 비교입니다.                                 |

또한 `roadmark_loss_0p5x` 같은 실험에서 **학습 로그의 가중된 loss와 validation의 가중 전 loss를 그대로 나누어 train/eval gap을 해석하면 안 됩니다.** 비교 전에 loss 정의를 맞춰야 합니다.

---

## 3. Optimizer보다 먼저 확인할 세 가지

### 3.1 정답 마스크를 그대로 넣어도 선 평가에서 손실이 발생합니다

업로드된 두 roadmark 예시에 대해 다음을 실행했습니다.

`정답 polyline → 실제 학습용 raster 생성 → ±20 logits → 실제 후처리 → 실제 8px 매칭`

| 예시                             | 정답 선 | 복원한 선 | TP | FP | FN |       F1 |
| ------------------------------ | ---: | ----: | -: | -: | -: | -------: |
| `roadmark_train_16616773.json` |    4 |     4 |  3 |  1 |  1 | **0.75** |
| `roadmark_val_11990481.json`   |    6 |     6 |  6 |  0 |  0 |  **1.0** |

첫 번째 예시는 **학습 target을 정확히 맞췄는데도 FP와 FN이 하나씩 발생**합니다.

이 결과가 전체 데이터의 성능이나 모델의 수학적 상한이라는 뜻은 아닙니다. 다만 현재의 line F1에는 모델 학습 오차 외에 **rasterization과 추적 및 매칭 과정의 손실**이 포함된다는 실제 사례입니다.

관련 코드는 다음입니다.

* `code/model/engine/postprocess.py:59–215`
* `code/model/engine/geometry_metrics.py:55–118`

현재 후처리는 흰 차선과 노란 차선을 row 방향으로 추적하고 정지선은 column 방향으로 추적합니다. `max_link=3`, `max_gap=3`, `min_points=6`입니다.

현재 축소율 0.625와 stride 4에서는 **출력 한 칸이 원본 6.4px**입니다. 8px 매칭 허용치는 약 1.25칸에 해당합니다. 선의 기울기와 길이 및 양자화에 따라 모델이 target을 잘 맞춰도 line matching에서 손해를 볼 수 있습니다.

가장 먼저 할 것은 **전체 라벨의 정답 마스크 round-trip**입니다. 클래스와 선 길이 및 기울기별로 복원 TP/FP/FN과 fragmentation을 집계하면 됩니다. 추가 학습 없이 표현과 후처리의 문제를 분리할 수 있습니다.

### 3.2 Detector loss가 공식 batch loss와 같은 목표는 아닙니다

`code/model/engine/loss.py:85–99`에서는 공식 criterion을 **이미지 한 장씩 호출한 후 평균**합니다.

대조한 Ultralytics **v8.4.115** criterion은 전달받은 batch에 대해 `target_scores.sum()`을 정규화 분모로 사용합니다. 음성 이미지의 분모는 최소값 1이 됩니다. 따라서 현재의 이미지별 정규화 평균은 batch 전체의 합으로 정규화하는 것과 다릅니다. 

BCE 부분만 개념적으로 쓰면 현재 방식은 다음 형태입니다.

$$
L_{\mathrm{current}}
=
\frac1B\sum_i
\frac{A_i}{\max(S_i,1)}
$$

반면 batch 전체 정규화는 전역 배율을 제외하면 다음 형태입니다.

$$
L_{\mathrm{pooled}}
\propto
\frac{\sum_i A_i}{\max(\sum_i S_i,1)}
$$

여기서 \(A_i\)는 해당 이미지의 BCE 합이고 \(S_i\)는 assignment의 target-score 합입니다.

현재 방식은 같은 출력이 주어졌을 때 microbatch 분할에 따른 objective 변화를 줄이는 장점이 있습니다. **구현 오류라고 단정할 부분은 아닙니다.** 그러나 음성 이미지가 72.3%인 상황에서는 양성과 음성의 상대적 기여를 반드시 알아야 합니다.

다음 통계를 먼저 기록하는 편이 좋습니다.

> 이미지별 det loss와 foreground assignment 수 및 target-score 합을 양성/음성으로 나누고 각 그룹의 shared gradient norm을 비교한다.

이 진단 없이 “양성이 적으니 낮은 LR에서 적응이 늦다”만으로 detector의 동작을 설명하면 중요한 변수를 놓칩니다.

### 3.3 BN 재추정은 ‘64장이 적다’보다 배치 구성이 더 구체적인 문제입니다

`code/tools/pv26_train/cli.py:342–353`을 보면 SF의 BN 재추정은 source별 32장과 batch size 8을 사용합니다.

그런데 subset이 source별로 이어 붙여지므로 실제로는 다음과 같습니다.

> Traffic-only 4 batches → Roadmark-only 4 batches

학습 sampler는 source를 섞습니다. 즉 **학습과 재추정의 physical batch 구성이 다릅니다.**

BatchNorm의 running variance는 학습 배치에서 계산한 분산을 갱신해 유지합니다. source-pure 배치의 분산을 평균한 값과 서로 다른 source가 섞인 분포의 분산은 source 간 평균 차이 때문에 달라질 수 있습니다. ([PyTorch Docs][1])

이를 보이는 CPU 반례도 실행했습니다. 평균이 0과 10인 두 source를 각각 pure batch로 넣으면 running variance가 약 **1.1429**였고 합친 표본의 분산은 **26.4127**이었습니다. 실제 PV26에서 그 정도의 오차가 난다는 측정은 아닙니다.

여기서 중요한 구분이 있습니다. 현재 재추정은 `momentum=None`이므로 **“마지막 roadmark source가 traffic 통계를 덮어쓴다”는 설명은 맞지 않습니다.** 핵심은 처리 순서 자체보다 서로 다른 source를 같은 batch에 넣느냐입니다.

최소 대조는 다음이면 됩니다.

| 대조                                  | 바꾸는 것                       |
| ----------------------------------- | --------------------------- |
| 같은 64장으로 source-pure 대 mixed batch  | 표본 수를 고정하고 배치 구성만 비교        |
| Mixed batch 유지 후 32→128→512장/source | 표본 수에 따른 수렴 확인              |
| AdamW와 SF에 동일한 재추정 적용               | optimizer 차이와 평가 프로토콜 차이 분리 |

“BN이 119개이므로 64장이 부족하다”는 식으로 충분성을 판단하기보다는 실제 지표와 통계의 안정성을 확인하는 것이 맞습니다.

---

## 4. AdamW 전체 LR 3×의 FP 급감을 어떻게 분리할 것인가

1차 고정 원장에서 baseline과 AdamW 3×의 변화는 큽니다.

| 지표                | Baseline | AdamW 3× |
| ----------------- | -------: | -------: |
| Roadmark TP       |      137 |      199 |
| Roadmark FP       |   26,789 |      423 |
| Roadmark pixel F1 |   0.0128 |   0.4696 |
| Roadmark BCE      |   0.9870 |   0.0199 |

이는 단순한 line matching의 우연한 변화로 보기 어렵습니다. 하지만 **좋은 특징을 학습했다는 설명만으로 원인을 확정할 수도 없습니다.** BN 변화나 logit 보정만으로 BCE와 고정 threshold의 pixel F1 및 line F1이 함께 바뀔 수 있습니다.

### 먼저 checkpoint를 고정한 진단

같은 checkpoint에 대해 stored BN과 재추정 BN을 비교하고 클래스별 foreground/background logit 분포를 봐야 합니다. 동시에 pixel PR/AP와 threshold에 따른 ridge 수 및 선 조각 수를 기록합니다.

판단은 이렇게 할 수 있습니다.

**PR 곡선은 비슷한데 threshold 조정으로 격차가 크게 닫히면** confidence 보정과 operating point 설명이 강해집니다. **PR 자체와 선의 기하학적 품질이 함께 좋아지면** 특징 구분 능력 개선의 근거가 강해집니다.

Threshold 선택은 별도 calibration 장면에서 해야 합니다. 비교 대상 validation에서 각 후보의 최고 F1을 찾아 다시 같은 validation 점수로 보고하면 선택 편향이 추가됩니다.

### 그다음 모듈 LR 2×2 실험

최소한 아래 네 조합이 필요합니다. 숫자는 기본 LR 대비 배율입니다.

| 조합                | Backbone | Detector | Roadmark decoder |
| ----------------- | -------: | -------: | ---------------: |
| Baseline          |        1 |        1 |                1 |
| Decoder만 증가       |        1 |        1 |                3 |
| Pretrained 모듈만 증가 |        3 |        3 |                1 |
| 전체 증가             |        3 |        3 |                3 |

이 네 개로 **decoder의 빠른 적응과 공유/pretrained 모듈의 변화 및 상호작용**을 우선 분리할 수 있습니다. 이후 필요할 때 backbone과 detector를 따로 나누면 됩니다.

Sparse-target prior를 반영한 decoder bias 초기화도 별도 대조로는 가치가 있습니다. 다만 그것을 BCE+Dice의 최적 초기화라고 전제하면 안 됩니다. FP를 줄이는 대신 초기 positive 학습을 늦출 가능성도 함께 측정해야 합니다.

---

## 5. PCGrad: 투영식은 맞지만 ‘약한 충돌’ 해석은 보완해야 합니다

### 구현 자체

두 태스크가 공통으로 사용하는 공간에서 현재의 대칭 투영식은 PCGrad의 기본 연산에 맞습니다. 논리 batch 전체에서 태스크별 gradient를 누적한 뒤 투영하는 구조도 적절합니다. 두 태스크에서는 상대 태스크의 순서를 섞는 문제가 여러 태스크일 때처럼 복잡하지 않습니다. ([arXiv][2])

다만 현재 `_gradient_geometry()`는 **양쪽 gradient가 모두 존재하는 파라미터의 교집합**에서 내적과 norm을 계산합니다. 태스크 전용 head는 투영하지 않습니다.

이를 “틀린 PCGrad”라고 부를 근거는 부족하지만 공유 공간을 명시해야 합니다. 가능하면 매번 동적으로 교집합을 만들기보다 모델 구조에 따른 고정된 shared parameter 집합을 정의하는 편이 해석하기 쉽습니다.

### 초기 로그를 실제 식에 대입하면

초기 값은 다음입니다.

$$
\|g_D\|=312.095,\quad
\|g_R\|=0.35343,\quad
\cos(g_D,g_R)=-0.01893
$$

두 norm의 비는 약 **883배**입니다.

현재 대칭 투영 후 합산식은 공유 공간에서 다음처럼 정리됩니다.

$$
g_{\mathrm{PC}}
=
\left(1-c\frac{\|g_R\|}{\|g_D\|}\right)g_D
+
\left(1-c\frac{\|g_D\|}{\|g_R\|}\right)g_R
$$

따라서 초기 로그에서는 다음과 같습니다.

$$
g_{\mathrm{PC}}
\approx
1.0000214g_D+17.7181g_R
$$

**작은 음의 cosine이어도 작은 태스크의 합성 계수는 크게 달라질 수 있습니다.**

다만 전체 raw 합산 gradient 대비 변화 norm은 약 **1.89%**입니다. 실제 AdamW 업데이트가 17.7배 변했다거나 전체 네트워크의 roadmark loss weight를 17.7배로 바꿨다는 뜻은 아닙니다. 공유 공간의 clipping 이전 계산입니다. 이 계산은 실제 함수를 호출해 검산했습니다.

따라서 문서의 제목은 다음 정도가 정확합니다.

> “기록된 두 시점의 각도 충돌은 작지만 norm 불균형에 따른 투영 효과가 존재하며 microbatch 교란 때문에 단독 성능 이득은 확인되지 않았다.”

### OOM과 BN

이 부분은 구현이 잘 처리하고 있습니다. 재시도 전에 RNG와 model buffer를 복원하므로 **실패한 forward의 BN 통계가 그대로 남아 누적됐다는 비판은 해당하지 않습니다.**

실제 교란은 성공한 학습의 physical batch입니다.

* 기존: `20 + 12`
* PCGrad: `10 + 10 + 10 + 2`

일반 합산을 micro10으로 맞춘 대조가 필요합니다. 다음 비교를 새로 구성한다면 모두 micro8 같은 공통값으로 맞추는 방법도 좋습니다.

BN freeze를 시험할 경우에는 trainer가 매 step `model.train()`을 호출한다는 점을 반영해야 합니다. 시작 전에 한 번 `BN.eval()`을 호출하는 방식으로는 정책이 유지되지 않습니다.

---

## 6. GradNorm: 최소 수정과 loss 일정 문제

### 오류 수정

`trainer.py:493–501`은 다음처럼 고치면 해당 오류를 해결할 수 있습니다.

```python
for parameter, det_gradient, road_gradient in zip(
    parameters, det_gradients, road_gradients
):
    if det_gradient is None and road_gradient is None:
        parameter.grad = None
    elif det_gradient is None:
        parameter.grad = weights[1] * road_gradient
    elif road_gradient is None:
        parameter.grad = weights[0] * det_gradient
    else:
        parameter.grad = (
            weights[0] * det_gradient
            + weights[1] * road_gradient
        )
```

원본은 수정하지 않고 별도 patch로 만들었습니다. **양쪽 None / det만 None / road만 None / 둘 다 존재**의 네 조합과 GradNorm weight-loss의 유한한 미분을 CPU에서 확인했습니다.

여기서는 `None`을 유지하는 것이 중요합니다. 0 tensor를 넣으면 optimizer가 해당 파라미터를 건너뛰는 대신 momentum이나 weight decay를 적용하는 경로가 될 수 있습니다.

### 상대 학습률의 의미가 어긋나는 이유

GradNorm은 각 태스크의 초기 loss 대비 감소율과 공유 파라미터에서의 gradient norm을 이용해 가중치를 조절합니다. 원 논문에서도 측정할 공유 파라미터 집합 \(W\)를 정의합니다. ([arXiv][3])

현재 detector는 다음 구조입니다.

$$
L_D(t)=a(t)L_{\mathrm{o2m}}(t)+b(t)L_{\mathrm{o2o}}(t)
$$

그런데 `pv26.py:145`에서 one2one branch의 feature는 detach되어 있습니다. 따라서 공유 본체가 받는 gradient는 다음입니다.

$$
\nabla_W L_D(t)
=
a(t)\nabla_W L_{\mathrm{o2m}}(t)
$$

반면 GradNorm이 진도를 판단하는 loss에는 one2one 항도 들어 있습니다. 여기에 \(a(t)\)가 0.8에서 0.1로 감소합니다.

즉 현재 GradNorm은 **one2one head의 진도와 branch 가중치 일정 변화를 공유 표현의 학습 진도에 섞어 해석**할 수 있습니다.

우선 o2m/o2o loss와 계수 및 shared norm과 learned weight를 함께 기록해야 합니다. 그다음 “원래 일정을 유지한 GradNorm”과 “branch 계수를 고정한 대조”를 비교하는 것이 좋습니다.

Shared-o2m만 진도 측정에 사용하는 설계도 가능하지만 이는 변형으로 명시해야 합니다. **전체 detector loss를 단순히 \(a(t)\)로 나누는 보정은 one2one 항 때문에 적절하지 않습니다.**

---

## 7. Schedule-Free와 Prodigy 및 Muon의 공정한 비교

### Schedule-Free

현재 `optimizer.train()/eval()` 전환은 연결되어 있습니다. 단순한 모드 전환 누락이 아닙니다.

SF는 학습 iterate와 평가 iterate가 다르므로 평가 weights에 맞는 BN이 필요합니다. 공식 구현도 이 전환과 BN 처리를 별도로 안내합니다. ([GitHub][4])

최소 실험은 **같은 SF checkpoint의 평가 weights를 그대로 유지한 채 BN 재추정만 바꾸는 것**입니다. 앞서 제시한 source-pure/mixed 대조와 장수 대조면 충분합니다.

저장 경로도 한 가지 확인할 필요가 있습니다. 임의 시점 checkpoint를 저장할 때 평가 weights로 전환하더라도 BN은 이전 학습 iterate의 통계일 수 있습니다. 반면 현재 최종 validation 경로는 BN 재추정 후 저장하므로 **이번 최종 결과가 그 문제로 잘못됐다고 단정할 수는 없습니다.**

평가 직후 저장한 checkpoint를 standalone 평가로 다시 읽었을 때 지표가 같은지 확인하고 BN을 어느 weight step에서 추정했는지만 함께 남기면 됩니다.

### Prodigy

현재 비교는 **“모듈별 LR을 튜닝한 AdamW 레시피”와 “공통 `d`를 쓰는 Prodigy 레시피”의 비교**입니다. 실용적인 설정 비교로는 유효하지만 optimizer 자체의 우열을 분리한 실험은 아닙니다.

또한 공식 Prodigy 구현은 서로 다른 nonzero group LR을 허용하지 않습니다. 따라서 기존 `1:10:30`을 group `lr`에 그대로 복원하는 변경은 적절하지 않습니다. 

더 유용한 비교는 다음 세 방향입니다.

| 비교                                        | 알아내는 것                                       |
| ----------------------------------------- | -------------------------------------------- |
| Uniform-LR AdamW 대 공통-d Prodigy           | 모듈별 LR 정책 차이를 줄인 비교                          |
| 모듈별 독립 Prodigy 인스턴스                       | Pretrained 모듈과 새 decoder가 다른 적응 스케일을 필요로 하는가 |
| Backbone/detector AdamW + decoder Prodigy | 새 decoder의 LR 적응에만 Prodigy가 도움이 되는가          |

독립 인스턴스는 고정된 1:10:30 복원과는 다른 변형입니다.

`d(t)`와 그룹별 실제 update/weight norm 및 clipping을 기록해야 합니다. `slice_p=11`과 bias correction 및 WD도 명시해야 합니다. 분석 문서가 checkpoint에서 읽었다고 보고한 `d` 값은 이번 압축에 checkpoint가 없으므로 직접 재확인하지 못했습니다.

### Muon

**비교할 가치는 있습니다. CNN에 적용할 수 없다는 판단은 맞지 않습니다.** 원저자 구현은 convolution update를 `[C_out, C_in × kH × kW]`로 펼쳐 처리합니다. Hidden weight에 Muon을 사용하고 최종 출력층과 bias 등의 파라미터에 일반 optimizer를 사용하는 구성도 명시합니다. 

PV26에서는 다음 구성이 출발점으로 적절합니다.

> Hidden convolution에는 Muon을 적용하고 input stem과 최종 출력층 및 BN affine과 bias는 auxiliary AdamW에 남긴다.

학습률 숫자를 AdamW와 같게 두는 것이 공정성을 뜻하지는 않습니다. 동일한 tuning budget 안에서 Muon LR과 pretrained/new module의 배율을 탐색해야 합니다. Shape scaling과 momentum 및 Newton–Schulz 설정은 고정해야 합니다.

추가로 공식 참고 구현 일부는 `grad=None`을 zero로 바꿉니다. 이 partial-label 모델에서는 비활성 파라미터의 moment/decay 정책이 달라질 수 있으므로 사용할 구현의 None 처리도 명시해야 합니다. 

다만 우선순위는 아래의 데이터와 평가 실험보다 뒤입니다. 지금은 Muon을 넣기 전에 원인을 더 싸게 분리할 수 있는 항목이 많습니다.

---

## 8. 작은 신호등과 희귀 정지선

### 작은 신호등: source 비율보다 source 내부 표본을 바꿔야 합니다

차량 신호등의 76%가 짧은 변 8px 미만이라는 관측은 중요합니다. 다만 **stride보다 작아서 positive assignment가 무조건 사라진다는 설명은 부정확합니다.** 사용 버전인 Ultralytics v8.4.115에는 작은 GT의 후보 anchor 영역을 확장하는 처리가 이미 있습니다. 

먼저 object 크기별 assigned foreground 수와 recall을 확인해야 합니다.

우선순위는 **traffic 내부의 양성과 장면 sampling → 해상도 → P2 head**로 보겠습니다. Traffic 비율을 늘려도 내부의 72.3% 음성 비율이 유지되면 필요한 양성 노출이 효율적으로 증가하지 않습니다.

Vehicle/pedestrian 양성과 tiny object 및 hard negative를 장면 단위로 나누어 sampling하고 검증 분포는 그대로 유지하는 편이 좋습니다. 음성을 전부 제거하는 것은 오탐 억제 학습을 잃으므로 피해야 합니다.

해상도는 현재 입력의 padding부터 볼 수 있습니다.

| 입력 \(H\times W\) | 실제 원영상 축소율 | 현재 대비 입력 pixel 수 |
| ---------------- | ---------: | ---------------: |
| 608×800          |      0.625 |            1.00× |
| 576×1024         |      0.800 |          약 1.21× |
| 736×1280         |      1.000 |          약 1.94× |

현재 608×800 안의 실제 영상은 450×800입니다. 따라서 직사각 입력을 이용하면 계산량 증가에 비해 실제 신호등 해상도를 꽤 높일 수 있습니다. 위 pixel 비율은 FPS 예측은 아니므로 실제 latency와 VRAM은 측정해야 합니다.

P2 head는 그다음 별도 대조가 좋습니다. 현재는 feature index와 channel 구성이 고정되어 있으므로 stride 설정만 바꾸는 것으로 끝나지 않습니다.

### 정지선: 먼저 0.0216%의 분모를 바로잡아야 합니다

질문의 **0.0216%는 정지선 채널의 valid pixel 대비 양성 비율**입니다. “모든 roadmark 양성 pixel 중 정지선 비율”은 아닙니다.

Census로 계산한 전체 양성 pixel 중 정지선 비중은 약 **2.38%**입니다. 그래도 희귀하다는 판단은 유지됩니다. 학습 정지선 positive image는 336/4096입니다.

Roadmark 16장을 독립 추출한다고 단순화하면 정지선이 한 장도 없을 확률은 약 **25.4%**입니다.

현재 Dice도 클래스 균등 평균이 아닙니다. 양성이 있는 **image-class 쌍 전체를 합쳐 평균**하며 학습 데이터의 그러한 쌍에서 정지선 비중은 약 6.21%입니다. 이것이 실제 gradient 기여율과 같다는 뜻은 아니지만 평균 방식의 불균형은 분명합니다.

추천하는 수정은 먼저 클래스별 positive-image Dice 평균을 구한 뒤 클래스 가중 평균을 내는 방식입니다. 정지선 positive sampling과 BCE `pos_weight`는 별도 실험으로 분리하는 편이 좋습니다.

특히 prevalence의 역수인 약 4,627을 곧바로 `pos_weight`에 넣지는 않겠습니다. 이미 AdamW 정지선 precision이 **0.00681**인 상황에서는 positive 강조만 크게 늘리면 FP 문제가 악화될 수 있습니다.

평가도 정지선 TP/FP/FN과 FP/image 및 정지선이 없는 장면의 오탐률을 따로 봐야 합니다. Micro-F1은 희귀 GT의 미검출에는 둔감하지만 희귀 클래스 FP가 폭증하면 크게 악화됩니다. 따라서 “micro가 정지선을 전혀 반영하지 않는다”보다 **희귀 클래스의 누락과 오탐을 비대칭적으로 반영한다**는 설명이 정확합니다.

---

## 9. Successive halving과 Q

현재 runner는 순차 실행하는 **synchronous successive halving**입니다. ASHA가 아닙니다. 이 부분은 기존 문서가 이미 올바르게 정정했습니다.

좋은 점은 전체 horizon을 처음부터 정하고 승격 시 이어서 학습한다는 것입니다. Rung마다 cosine을 다시 시작하는 비교는 아닙니다.

문제는 조기 평가의 support입니다. 첫 rung의 roadmark 정지선 GT는 **2개**입니다. 이 상태에서는 micro-F1뿐 아니라 macro-F1으로 바꾸어도 정지선 순위의 불확실성이 큽니다.

Q는 유지하더라도 함께 볼 지표를 늘리는 편이 좋습니다.

$$
D_{\rm macro}
=
\frac{F1_{\rm vehicle}+F1_{\rm pedestrian}}2
$$

$$
R_{\rm macro}
=
\frac{F1_{\rm white}+F1_{\rm yellow}+F1_{\rm stop}}3
$$

태스크별 macro와 현재 aggregate F1을 함께 보고 희귀 클래스의 support도 표시해야 합니다. 실제 운영 우선순위가 정해져 있다면 그에 맞춘 선택 기준을 사전에 정하면 됩니다.

조기 탈락 위험을 줄이려면 baseline과 학습 동작이 다른 후보 일부를 끝까지 보존하고 다른 시작 예산을 둔 대조도 필요합니다. 현재 결과만으로 낮은 LR이나 SF 1×가 최종 예산에서도 나쁠지는 알 수 없습니다. 이런 early-stopping 기반 자원 배분은 최종 성능과 조기 성능의 관계에 의존합니다. ([arXiv][5])

또한 8,192장은 같은 validation pool의 확대입니다. **별도의 촬영 구간 holdout과 threshold calibration 분리**가 필요합니다. Seed 비교는 같은 seed끼리 짝지어 하고 영상 간 상관을 고려해 scene 단위 bootstrap을 사용하는 편이 적절합니다.

Runner에는 재개 시 확인할 구체적인 경로도 있습니다. `run_pv26_method_search.py:48–61`은 남은 steps를 `summary.global_step`으로 계산하지만 실제 재개는 checkpoint step에서 시작합니다. 중단 후 checkpoint가 summary보다 앞서면 최종 planned horizon 아래의 중간 예산에서는 목표를 초과할 수 있습니다. 이번 완료된 네 run에서 실제 초과를 관측한 것은 아닙니다. 재개 기준을 checkpoint와 sampler position에 맞추면 됩니다.

---

## 10. 다음 실험은 이 순서로 하겠습니다

공통으로 초기 state와 paired seed 및 데이터 노출과 augmentation 정책을 고정해야 합니다. 학습 비교에서는 logical batch와 physical batch 및 BN 정책과 precision 및 clipping도 맞춥니다. 아래 seed 수는 탐색을 위한 권장치이며 통계적 보증은 아닙니다.

| 순위     | 가설과 바꿀 변수                                                               | 핵심 고정 조건                                     | 판단 metric                                         | Seed                          |
| ------ | ----------------------------------------------------------------------- | -------------------------------------------- | ------------------------------------------------- | ----------------------------- |
| **1**  | 정답 raster와 선 복원 사이에 손실이 있다. 전체 GT round-trip과 trace 설정 진단               | 모델 제외. 원본 8px 평가 기준 유지                       | 클래스·길이·기울기별 TP/FP/FN과 fragmentation 및 coverage    | 학습 0                          |
| **2**  | BN과 threshold가 optimizer 격차 일부를 설명한다. Pure/mixed와 32/128/512장-source 대조 | 같은 checkpoint와 평가 weights. 별도 calibration 장면 | Pixel PR/AP와 line F1 및 FP/image                   | 추가 학습 0. Calibration 표본 선택 2회 |
| **3**  | 이미지별 det 정규화가 음성의 상대 기여를 크게 만든다. 통계 수집 후 pooled 대조                      | 같은 BN과 logical batch. 가능하면 동일 출력부터 비교        | 양성/음성 loss·norm과 tiny/ped recall 및 FP/image       | 진단 0. 학습 대조 3                 |
| **4**  | LR 3× 이득이 decoder 또는 pretrained 모듈에 있다. 위 2×2 LR 실험                     | 초기 state와 전체 horizon 및 sampling 고정           | D/R와 PR 및 background logits와 실제 update norm       | 3                             |
| **5**  | AdamW 3× 우위가 seed와 긴 예산에도 유지된다. AdamW3/SF10/baseline 비교                 | 공통 BN·threshold 정책과 scene holdout            | Paired D/R 차이와 클래스별 PR                            | 3. 근접하면 5                     |
| **6**  | 파일 균등화보다 양성·장면 stratification이 효과적이다. Source 내부 sampler 변경              | Source 1:1과 loss 및 구조 유지                     | Tiny/ped/stop recall과 음성 장면 FP 및 고유 scene support | 3                             |
| **7**  | 신호등 해상도 부족의 영향이 크다. 직사각 고해상도 후 별도 P2 대조                                 | Sampling과 loss 고정. 후처리 물리 기준 관리              | 크기 구간별 AP/recall과 latency 및 VRAM                  | Pilot 1. 유망 대조 3              |
| **8**  | 정지선 loss 평균 방식이 부족하다. Class-balanced Dice와 제한된 BCE 가중치 분리               | Sampler 고정. BN과 threshold 정책 공통              | Stop PR와 FP/image 및 위치·각도 오차와 D 저하                | 3                             |
| **9**  | Task-gradient 방식의 효과가 BN 교란과 다르다. Sum/PCGrad/수정 GradNorm 비교             | 공통 microbatch와 고정 shared W. Branch 일정 별도 대조  | D/R와 weight·norm·cosine 및 clipping·update 기록      | 3                             |
| **10** | 모듈 역할별 optimizer 적합성이 있다. 공정한 Prodigy 대조 또는 Muon+aux AdamW              | 동일 tuning budget과 노출 및 None·decay·BN 정책      | D/R와 희귀 클래스 성능 및 동일 GPU 처리량                       | 탐색 1. 선발 조합 3                 |

10번은 Prodigy와 Muon의 모든 조합을 돌리라는 뜻은 아닙니다. 앞선 실험 이후에도 optimizer 차이가 주요 불확실성으로 남을 때 비교 축 하나를 선택하는 것이 좋습니다.

추가 검증에 필요한 자료는 **300-step baseline/AdamW3/SF10 및 완료 후보의 최종 checkpoint**, **샘플 목록이 가리키는 전체 label JSON과 scene/sequence 정보**, **이미지별 raw prediction 또는 roadmark logits**, **step별 branch loss와 gradient 및 update 통계**입니다. SF checkpoint에는 optimizer state와 BN buffer 및 BN을 추정한 weight step이 함께 필요합니다.

**지금의 다음 학습 기준은 AdamW 3×입니다. 추가 분석 예산은 정답 마스크 복원과 BN·threshold 대조부터 쓰는 것이 가장 효율적입니다.** 이 결과가 나오면 “LR을 키워서 학습이 좋아졌다”는 설명을 decoder 적응과 공유 특징 및 평가 보정 중 어느 쪽인지 훨씬 정확하게 좁힐 수 있습니다.

[1]: https://docs.pytorch.org/docs/2.10/generated/torch.nn.BatchNorm2d.html "BatchNorm2d — PyTorch 2.10 documentation"
[2]: https://arxiv.org/abs/2001.06782?utm_source=chatgpt.com "Gradient Surgery for Multi-Task Learning"
[3]: https://arxiv.org/html/1711.02257 "GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks"
[4]: https://github.com/facebookresearch/schedule_free "GitHub - facebookresearch/schedule_free: Schedule-Free Optimization in PyTorch · GitHub"
[5]: https://arxiv.org/abs/1603.06560?utm_source=chatgpt.com "Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization"
