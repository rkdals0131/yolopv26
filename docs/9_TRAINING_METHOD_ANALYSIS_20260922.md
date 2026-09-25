# PV26 학습 방법론 27개 실험 분석

분석 대상은 `20260922_full_method_search_v2`다. **1차 27개 설정의 결과는 확정됐고, 28,800장까지 연장하는 승격 실험은 진행 중이다.** 이 문서는 2026-09-22 12:09 KST까지 확인한 1차 결과와 승격 결과를 설명한다. 27개 중 25개는 평가를 완료했고, GradNorm 두 설정은 구현 오류로 실패했다. 진행 중인 후보의 최종 결과를 예측값으로 채우지 않았다.

현재 가장 설득력 있는 관찰은 **큰 학습률을 사용한 두 설정에서 도로표식 배경 오탐이 크게 줄었다**는 것이다. AdamW 전체 LR 3×와 Schedule-Free 10×는 같은 9,600장 노출에서 선 F1이 각각 0.3390, 0.3602였다. 기본 설정은 0.0100이었다. 반면 신호등은 두 설정 모두 기본 설정보다 낮았다. 단일 총점보다 이 태스크 간 차이를 봐야 한다.

그렇다고 모든 차이를 optimizer의 우열로 설명할 수는 없다. 실제 데이터의 음성 영상 비율, 아주 작은 신호등, 희귀 정지선, 새 decoder와 사전학습 본체의 비대칭, BatchNorm, 물리 배치 변경이 결과에 함께 작용했다. 아래에서 **관측한 결과**, **코드로 확인한 작동 원리**, **아직 검증하지 않은 원인 가설**을 구분한다.

## 1. 이번 비교의 정확한 조건

| 항목 | 실제 조건 |
| --- | --- |
| 코드 | `47ea4db6dfecc6d88ef1c90250e90704f746b89e` |
| 원격 실행 | `kai:/home/kai/yolopv26/runs/20260922_full_method_search_v2/` |
| 초기화 | `yolo26s.pt`에서 전이 가능한 가중치 이식, 새 출력층·decoder 초기화, seed 26 |
| 학습 목록 | traffic 4,096장 + roadmark 4,096장 |
| 검증 목록 | 각 source의 별도 Validation split에서 4,096장씩 |
| 1차·승격 순위 평가 | 위 검증 목록에서 source별 고정 128장, 합계 256장 |
| 최종 확장 평가 | 학습 horizon을 마친 후보의 저장된 검증 목록 8,192장 전체 |
| 입력 | 608×800 letterbox, BF16 |
| 기본 배치 | 논리 32, 물리 20; 논리 배치 한 번을 20+12장으로 처리 |
| 기본 LR | backbone/neck `1e-4`, detector head `1e-3`, roadmark decoder `3e-3` |
| 기본 최적화 | AdamW, cosine, weight decay `1e-4`, gradient clipping 5 |
| 1차 예산 | 모든 후보에 누적 9,600장 노출 |
| 승격 예산 | 상위 9개에 누적 28,800장 노출 |
| step 수 | batch 16: 600→1,800, batch 32: 300→900, batch 64: 150→450 |
| 반복 | 이번 v2는 seed 하나. 앞선 수동 3-seed 실험은 조건이 달라 반복으로 합산하지 않음 |

여기서 장수는 **중복을 포함한 학습 입력 횟수**다. source별 4,096장의 서로 다른 파일을 9,600장 새로 확보했다는 뜻이 아니다. 일반 1:1 후보는 첫 단계에서 source별 4,800회를 뽑는다.

27개 run의 저장된 train/val index를 각각 비교했으며, 모든 후보가 같은 순서의 표본 목록을 사용함을 확인했다. 다만 학습 순서는 sampler와 source 비율에 따라 달라진다. `augment=false`는 같은 draw에서 영상 변환을 제거한다.

첫 7개 후보의 1차 학습은 user1 RTX 4060 Laptop에서 수행했고 나머지는 KAI RTX 4060에서 수행했다. 따라서 **호스트를 넘나드는 시간 비교는 optimizer 속도 비교로 사용할 수 없다.** 같은 seed와 같은 소스가 CUDA 계산까지 bit-exact한 결과를 보장하지도 않는다. 수천분의 몇 수준의 점수 차이는 반복 전에는 순위 이상의 의미를 부여하지 않는다.

## 2. 모델 구조가 만드는 학습 비대칭

현재 본체는 약 1,006만 parameter이며, 이전 legacy PV26의 여러 head 구성이 아니라 아래의 집중 모델이다.

```text
RGB → YOLO26-s backbone/neck (9,016,000 parameters)
        ├─ P3/P4/P5 → two-class detector (933,412)
        │                ├─ one2many: 공유 본체에도 gradient 전달
        │                └─ one2one: feature.detach(), 추론에서 사용하는 검출 branch
        └─ P2/P3/P4 → 새 roadmark decoder (107,331)
                         └─ white / yellow / stop-line, stride-4 logit map
```

코드 근거: [PV26 모델](../model/net/pv26.py), [손실](../model/engine/loss.py).

### 2.1 사전학습 본체와 새 decoder에는 같은 LR이 같은 의미가 아니다

backbone/neck와 shape-compatible detector 파라미터는 사전학습 지식을 가지고 출발한다. roadmark decoder는 새로 시작한다. 새 decoder의 낮은 LR은 배경·선 구분을 익히는 속도를 제한할 수 있지만, 본체까지 크게 움직이면 기존 검출 특징을 훼손하거나 assignment와 confidence를 흔들 수 있다. 이 구조 때문에 **도로표식이 빨리 좋아지는 LR과 신호등이 안정적으로 좋아지는 LR이 다를 수 있다.**

decoder는 전체 파라미터의 약 1.1%지만, 그 loss는 P2/P3/P4를 통해 공유 본체에도 전달된다. 작은 decoder라는 사실이 detector와 독립적으로 학습된다는 뜻은 아니다.

### 2.2 두 태스크가 같은 영상의 정답을 함께 보는 학습이 아니다

traffic 영상은 신호등만, roadmark 영상은 도로표식만 감독한다. 다른 태스크의 label은 없는 것으로 처리하며 배경 label로 바꾸지 않는다. 따라서 source 비율은 정답 노출뿐 아니라 공유 특징이 경험하는 영상 분포도 바꾼다.

손실은 태스크별 유효량으로 각각 정규화된다. traffic을 2배 넣었다고 `det + roadmark`에서 det 항의 계수가 자동으로 2배가 되는 것은 아니다. source 비율은 표본 다양성·gradient 추정 분산·BN 분포를 바꾸고, `roadmark_loss_weight`는 loss 계수를 직접 바꾼다. 두 실험은 별개의 질문이다.

### 2.3 검출 loss 숫자가 공유 본체의 영향력을 그대로 나타내지 않는다

설치된 Ultralytics E2ELoss와 `set_progress()`에 따르면 one2many/one2one 가중치는 초기 `0.8/0.2`, horizon의 1/3 지점 `0.5667/0.4333`, 종료 지점 `0.1/0.9`로 변한다. one2one feature는 detach되어 공유 본체로 역전파되지 않는다.

따라서 일정이 진행될수록 detector의 공유 본체 감독은 loss의 branch 가중치 측면에서 감소한다. roadmark는 별도의 같은 decay가 없다. **300→900 step의 det loss를 고정된 동일 목적함수 값처럼 해석하면 안 되며**, constant LR나 Schedule-Free도 검출 loss의 이 내부 일정은 계속 사용한다. 이번 Schedule-Free는 학습률 decay가 없다는 뜻이지 전체 학습이 종료 horizon과 무관하다는 뜻은 아니다.

### 2.4 BatchNorm은 별도의 변동 원인이다

모델에는 BatchNorm 모듈이 119개 있다. joint forward에서는 두 source 영상 모두 decoder를 지나므로, roadmark label이 없는 traffic 영상도 decoder의 BN running statistics에 영향을 준다. 손실 masking이 BN 업데이트를 masking하지는 않는다.

논리 batch가 같아도 microbatch를 20, 16, 10, 8로 바꾸면 BN이 보는 묶음과 running-stat 갱신 횟수가 바뀐다. 따라서 gradient accumulation의 loss 정규화가 올바르더라도 모델 업데이트가 완전히 같지는 않다. 이는 batch 및 PCGrad 실험을 해석할 때 실제로 작용한 조건 차이다. BN이 성능 차이의 주원인인지는 별도 진단이 필요하다.

## 3. 실제 사용한 데이터의 분포

저장된 index의 라벨 16,384개(train/val × 두 source × 4,096)를 현재 decoder로 읽어 집계했다. 이미지 전체를 재학습하거나 GPU 추론하지는 않았다. 네 대표 원본 영상도 직접 확인했다. 집계 코드와 결과는 문서 끝의 분석 산출물 경로에 있다.

### 3.1 신호등: 많은 음성 영상, 작은 객체, 적은 보행자 표본

| 항목 | 학습 traffic 4,096장 | 검증 traffic 4,096장 |
| --- | ---: | ---: |
| 대상 신호등이 없는 영상 | 2,961장, **72.29%** | 2,171장, **53.00%** |
| 차량 신호등 객체 | 2,022 | 4,500 |
| 보행자 신호등 객체 | 285 | 974 |
| 차량 신호등이 있는 영상 | 1,066 | 1,832 |
| 보행자 신호등이 있는 영상 | 246 | 677 |
| 차량 신호등의 축소 후 짧은 변 중앙값 | **5.00 px** | **5.00 px** |
| 차량 신호등의 짧은 변 <8 px 비율 | 76.01% | 73.38% |
| 보행자 신호등의 짧은 변 중앙값 | 8.75 px | 7.81 px |

모두 1280×720 영상이다. 608×800 입력에서는 영상 내용이 450×800으로 축소되고 나머지는 padding이다. 원본 box에 resize scale 0.625를 적용해 위 크기를 계산했다.

detector는 stride 8/16/32의 특징을 사용한다. 도로표식에는 stride 4의 P2가 있지만 신호등 검출에는 없다. 짧은 변이 5px인 대상은 가장 세밀한 검출 특징의 한 cell보다 작다. 검출 불가능하다는 뜻은 아니지만, 작은 위치 오차에 IoU 0.5 판정이 민감하며 이미지 해상도와 특징 표현의 제약을 optimizer만으로 해소하기 어렵다.

기본 논리 batch에서 traffic 16장 중 신호등 양성 영상은 평균 약 4.4장이고, 보행자 신호등이 있는 영상은 약 1장이다. source 전체를 균등하게 보는 sampler는 이 희귀 클래스를 따로 보강하지 않는다. 학습은 검증보다 음성 영상 비중도 높다. 높은 precision과 낮은 recall, 보행자 성능 정체를 설명하는 유력한 데이터 측 원인이다.

여기서 음성은 모델의 두 대상 클래스가 없다는 뜻이다. 해당 영상에 표지판·버스용 신호 등이 있을 수 있다. 미상 종류 때문에 감독에서 제외된 영상은 이번 선택된 목록에서는 0장이었다.

### 3.2 도로표식: 흰 선 중심의 극단적으로 희소한 표적

| 항목 | 흰 차선 | 노란 차선 | 정지선 |
| --- | ---: | ---: | ---: |
| 학습 polyline 수 | 14,840 | 1,958 | **366** |
| 양성 학습 영상 수 | 3,963 | 1,111 | **336** |
| 학습에서 클래스별 유효 pixel 중 양성 비율 | **0.7916%** | **0.0934%** | **0.0216%** |
| 검증 4,096장의 polyline 수 | 15,879 | 1,855 | **111** |
| 양성 검증 영상 수 | 4,046 | 1,259 | **93** |
| 순위 평가 128장의 polyline 수 | 496 | 54 | **2** |

미상 차선 색상 때문에 흰/노란 차선 감독이 꺼진 영상은 이번 목록에서 0장이었다. 위 픽셀 비율은 실제 stride-4 rasterizer에서 padding을 제외한 유효 영역 기준이다.

BCE는 유효한 모든 픽셀에, Dice는 양성 표적이 있는 이미지·클래스에 적용된다. 정지선이 없는 정상 영상도 BCE 음성 학습에 기여한다. Dice가 희소 표적을 돕지만, 정지선의 적은 장면 다양성과 많은 음성 노출을 없애주지는 않는다. 이번 종합 점수만으로 정지선 품질을 판단하기 어렵다.

차선 평가 단위는 라벨 polyline이다. 연속 차로 하나나 주행 가능성의 평가가 아니다. 점선 조각과 긴 선, 가림에 의해 분리된 선을 어떻게 이어주는지도 F1에 영향을 준다.

### 3.3 파일 개수와 독립 장면 개수는 다르다

`--sample-limit`은 전체 파일을 무작위로 뽑지 않고 **정렬 후 앞 4,096개**를 고른다.

| source/split | 실제 선택 범위 |
| --- | --- |
| traffic train | 첫 archive group, `s01000200` → `s01005308` |
| traffic val | 첫 validation archive group, `s01000100` → `s01040257` |
| roadmark train | 첫 archive group, `16608323` → `16627408` |
| roadmark val | 첫 validation archive group, `10002688` → `14103434` |

source별 train/val sample ID 중복은 0개다. 그러나 파일 ID와 archive group만으로 촬영 경로·시퀀스가 분리됐다고 증명할 수 없다. 연속 프레임의 높은 상관이나 지역 편향은 가능성이 있고, 실제 촬영 구간의 교집합은 아직 조사되지 않았다. 이전 설명의 “같은 촬영 그룹이어서”라는 단정도 이 근거만으로는 성립하지 않는다.

직접 본 roadmark train/val 대표 영상에는 계기판·보닛 노출량, 카메라 시점, 역광이 달랐다. traffic 대표 영상에는 신호등이 없는 도로와 작은 신호등이 있는 교차로가 있었다. **도메인 차이를 의심할 사례는 존재하지만 네 장만으로 분포 전체를 대표한다고 주장하지 않는다.**

## 4. 점수의 의미와 비교표

`D`는 신호등 전체 F1(confidence 0.25, box IoU 0.5), `R`은 도로표식 polyline 전체 F1(원본 좌표 평균 거리 허용 8px), `Q=(D+R)/2`는 이번 탐색의 승격 점수다. 두 F1 모두 클래스별 macro 평균이 아니라 TP/FP/FN 합계로 계산된다. 따라서 Q의 태스크 가중치는 동등하지만 각 태스크 내부의 희귀 클래스 영향은 동등하지 않다.

도로표식은 sigmoid 0.5를 넘는 ridge를 추적하고 최소 6개 점 등을 만족해야 선으로 출력한다. stride-4 한 픽셀은 원본 약 6.4px에 해당한다. logit의 작은 이동이 threshold를 넘는 배경 ridge 수를 크게 바꿀 수 있어, 학습 loss 변화와 선 F1 변화가 비례하지 않는다. [평가](../model/engine/evaluation.py), [후처리](../model/engine/postprocess.py), [선 매칭](../model/engine/geometry_metrics.py).

아래는 **모두 9,600장 노출, 동일 256장 검증** 결과다. ΔQ는 이 v2 기본 설정과의 차이다. “종료”는 이 예산에서 승격되지 않았다는 뜻이며, 장기 학습에서도 열등하다는 결론은 아니다.

| # | 설정 | D | R | Q | ΔQ | 1차 판정 |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | `adamw_cosine_baseline` | 0.5704 | 0.0100 | 0.2902 | +0.0000 | 승격 |
| 2 | `adamw_lr_0p3x` | 0.4876 | 0.0101 | 0.2489 | -0.0413 | 종료 |
| 3 | `adamw_lr_3x` | 0.4556 | 0.3390 | 0.3973 | +0.1071 | 승격 |
| 4 | `adamw_roadmark_1x` | 0.5169 | 0.0039 | 0.2604 | -0.0298 | 종료 |
| 5 | `adamw_roadmark_2x` | 0.5358 | 0.0087 | 0.2723 | -0.0180 | 종료 |
| 6 | `adamw_constant` | 0.5147 | 0.0219 | 0.2683 | -0.0219 | 종료 |
| 7 | `weight_decay_0` | 0.5039 | 0.0550 | 0.2794 | -0.0108 | 종료 |
| 8 | `weight_decay_1e3` | 0.5169 | 0.0080 | 0.2624 | -0.0278 | 종료 |
| 9 | `augmentation_off` | 0.4286 | 0.0134 | 0.2210 | -0.0692 | 종료 |
| 10 | `traffic_ratio_2to1` | 0.5532 | 0.0226 | 0.2879 | -0.0023 | 승격 |
| 11 | `roadmark_ratio_2to1` | 0.4809 | 0.0323 | 0.2566 | -0.0336 | 종료 |
| 12 | `roadmark_loss_0p5x` | 0.5831 | 0.0176 | 0.3003 | +0.0101 | 승격 |
| 13 | `roadmark_loss_2x` | 0.5353 | 0.0441 | 0.2897 | -0.0005 | 승격 |
| 14 | `least_used_of_two` | 0.5374 | 0.0227 | 0.2801 | -0.0101 | 승격 |
| 15 | `logical_batch_16` | 0.5461 | 0.0476 | 0.2969 | +0.0066 | 승격 |
| 16 | `logical_batch_64` | 0.5512 | 0.0378 | 0.2945 | +0.0043 | 승격 |
| 17 | `logical_batch_64_lr_2x` | 0.4715 | 0.0367 | 0.2541 | -0.0361 | 종료 |
| 18 | `microbatch_8` | 0.5126 | 0.0228 | 0.2677 | -0.0225 | 종료 |
| 19 | `pcgrad` | 0.5179 | 0.0180 | 0.2679 | -0.0223 | 종료 |
| 20 | `gradnorm_alpha_0p5` | — | — | — | — | 구현 오류 |
| 21 | `gradnorm_alpha_1p5` | — | — | — | — | 구현 오류 |
| 22 | `schedulefree_1x` | 0.2432 | 0.0071 | 0.1252 | -0.1651 | 종료 |
| 23 | `schedulefree_3x` | 0.2536 | 0.0148 | 0.1342 | -0.1560 | 종료 |
| 24 | `schedulefree_10x` | 0.4262 | 0.3602 | 0.3932 | +0.1030 | 승격 |
| 25 | `prodigy_dcoef_0p3` | 0.5535 | 0.0044 | 0.2789 | -0.0113 | 종료 |
| 26 | `prodigy_dcoef_1` | 0.5434 | 0.0030 | 0.2732 | -0.0170 | 종료 |
| 27 | `prodigy_dcoef_3` | 0.4647 | 0.0099 | 0.2373 | -0.0529 | 종료 |

## 5. 27개 설정별 해석

### 01. `adamw_cosine_baseline` — 신호등은 먼저 적응했고 도로표식은 오탐이 지배

**관측:** D=0.5704, R=0.0100. 도로표식 TP 137, FP 26,789, FN 415이며 precision은 0.51%, recall은 24.82%다. 선을 전혀 찾지 못한 것이 아니라 잘못 출력한 선이 너무 많다. 검증 roadmark BCE 0.9870, Dice loss 0.8283도 낮은 품질과 일치한다.

**해석:** 검출에 전이 가능한 특징과 별도 객체 detector가 있는 반면, 희소 centerline을 읽는 decoder는 새로 학습한다. 초기 수백 update에서 신호등과 roadmark의 진도가 다른 것은 구조상 자연스럽다. 낮은 R을 “차선을 모두 놓쳤다”로 읽으면 진단을 틀리게 한다. 우선 설명해야 할 현상은 배경 출력 억제 실패다.

**남은 가설:** train/eval BN 차이, 선택된 train/val 장면 차이, 충분히 학습되지 않은 logit calibration 중 어느 쪽이 FP를 크게 만드는지 분리되지 않았다. 900-step baseline이 완료되기 전에는 LR 3× 대비 최종 우열도 확정할 수 없다.

### 02. `adamw_lr_0p3x` — 신호등 학습 지연, 도로표식 개선은 거의 없음

LR은 `3e-5 / 3e-4 / 9e-4`다. D=0.4876, R=0.0101. 기본값 대비 신호등 precision은 0.664→0.776으로 오르지만 recall은 0.500→0.355로 낮아진다. 단순 발산이 아니라 **보수적으로 적게 검출하는 상태**다.

새 검출 출력층과 decoder를 짧은 예산에 적응시키기에는 update가 작았을 가능성이 높다. 특히 72.3% 음성 traffic 영상과 작은 양성 box는 낮은 confidence 상태가 지속되기 쉬운 조건이다. R의 +0.00018 차이는 도로표식에 유리하다는 증거로 볼 수준이 아니다. 더 긴 horizon에서의 이점은 이번 조기 종료로 확인하지 못했다.

### 03. `adamw_lr_3x` — 도로표식 배경 억제가 실제로 개선된 가장 강한 후보

LR은 `3e-4 / 3e-3 / 9e-3`다. D=0.4556, R=0.3390. roadmark TP는 137→199, FP는 **26,789→423**, FN은 415→353으로 변했다. pixel F1도 0.0128→0.4696이며 BCE는 0.9870→0.0199다. 단순 선 매칭의 우연한 변화만으로 설명할 수 없는 개선이다.

**유력한 설명:** 새 decoder와 공유 특징이 배경과 중심선을 구분하는 쪽으로 더 빨리 이동했다. threshold를 넘는 배경 ridge가 사라진 효과가 특히 크다. 동시에 detector precision과 recall이 모두 하락했으므로 본체·검출 head의 큰 update는 초기 신호등에 부담이었다.

**확정하지 못한 부분:** decoder LR `9e-3`만 올린 대조군은 없다. 따라서 이득이 decoder에서 왔는지 backbone LR 상승이 필수였는지 분리할 수 없다. 900 step에서 D/R 모두 더 개선됐다는 후속 결과는 초기 신호등 손해의 일부가 회복 가능했음을 보여준다(6절).

### 04. `adamw_roadmark_1x` — decoder LR `1e-3`은 이 예산에서 느림

이름의 1×는 `head_lr=1e-3` 기준이다. **v2 기본 decoder `3e-3`의 1/3**이다. backbone·detector LR은 기본값이다. D=0.5169, R=0.0039, roadmark FP 38,486이다.

새 decoder의 작은 LR에서 배경 억제가 더 늦는다는 설명과 일치한다. 신호등도 낮아졌다는 점은 공유 본체를 통해 decoder 학습 상태가 detector에 영향을 줄 수 있음을 보여주는 관찰이지만, 구체적인 gradient 간섭이 원인이라는 증명은 아니다. “detector LR을 안 바꿨으니 D는 같아야 한다”는 전제는 joint 구조에서 성립하지 않는다.

### 05. `adamw_roadmark_2x` — `1e-3`보다 낫지만 기본 `3e-3`에는 못 미침

decoder LR만 `2e-3`이다. D=0.5358, R=0.0087. decoder LR `1e-3`의 FP 38,486에서 26,114로 줄었지만 기본값 대비 R은 낮다. 세 수준 `1e-3→2e-3→3e-3`의 D/R은 이 seed에서 함께 상승했다.

이 순서는 낮은 decoder LR의 적응 지연 가설을 지지한다. 다만 `9e-3` 단독 대조군이 없어 LR 3× 후보의 큰 개선을 이어 설명하기에는 부족하다. 세 점으로 최적 LR 곡선을 확정하지 않는다.

### 06. `adamw_constant` — roadmark는 약간 더 학습됐으나 detector가 손해

D=0.5147, R=0.0219. 같은 초기 LR에서 cosine만 제거했다. 기본 cosine은 300/900 step 지점에 LR이 초기값의 약 75%다. 따라서 constant는 후반부에 상대적으로 더 크게 update했다.

R의 상승과 BCE 0.4925는 더 큰 후반 update가 decoder의 배경 억제를 도왔을 가능성과 일치한다. 반대로 D가 하락해 Q는 낮다. 하지만 아직 horizon의 1/3이라 cosine의 마지막 수렴 구간 효과는 비교되지 않았다. 이는 **조기 예산에서의 결과**이지 LR schedule이 원천적으로 우월하다는 검증이 아니다.

### 07. `weight_decay_0` — decoder 이득과 detector 손실이 공존, 과적합 단정 불가

D=0.5039, R=0.0550. roadmark TP 192, FP 6,238로 기본보다 좋다. detector recall은 0.500→0.392로 줄어 Q는 낮아졌다. 그래서 “정규화가 없으면 두 태스크 모두 나빠진다”는 해석은 틀린다.

**가능한 설명:** 새 decoder에 대한 축소 제약 제거가 빠른 fitting을 도왔거나, 작은 수치 차이가 BN·비선형 출력의 다른 경로로 이어졌을 수 있다. 실제 train 전체 성능과 독립 검증 learning curve가 없으므로 과적합이 확인됐다고 말할 수 없다.

또한 이 짧은 일정에서 기본 WD의 직접 shrink는 작다. 큰 F1 변화를 모두 regularization의 직접 효과로 돌리는 설명은 신뢰도가 낮다.

### 08. `weight_decay_1e3` — 현재 설정에서는 악화, “너무 강한 WD”는 아직 가설

D=0.5169, R=0.0080으로 기본보다 모두 낮다. 새 decoder 학습을 제약했을 가능성은 있지만, 이 수준이 절대적으로 너무 크다고 단정할 수 없다.

구체적으로 300-step cosine 동안 decoder의 decoupled decay만 계산한 총 shrink는 `1 − ∏(1 − lr_t × 0.001) ≈ 0.0822%`다. 기본 WD `1e-4`에서는 약 0.0082%다. 이 직접 크기는 작고, AdamW 경로의 증폭·BN·단일 seed 변동이 섞일 수 있다. 반복 없이 WD `1e-4`가 최적이라고 선언하지 않는다. 현 구현은 BN·bias도 같은 optimizer group의 WD에 포함한다.

### 09. `augmentation_off` — 신호등 일반화/억제에 불리한 신호

D=0.4286, R=0.0134다. 신호등 precision은 0.664→0.424, recall은 0.500→0.434로 모두 하락했다. 단순히 덜 적극적인 검출이 된 것이 아니라 false positive도 문제가 됐다.

밝기·대비 ±10% 및 수평 반전을 제거하면 선택된 좁은 영상 구간의 외형에 더 의존할 수 있다. 작은 신호등, 배경 표지판, 시점·노출 변화가 있는 검증에서 검출 confidence가 덜 안정적이었을 가능성이 있다. **이번 v2에서는 신호등에 증강을 유지할 근거가 있다.**

R은 오히려 소폭 상승했다. 그러므로 같은 증강이 roadmark에 반드시 좋다고 단정할 수 없고, 수평 반전과 광도 증강을 한 번에 껐기 때문에 각각의 기여도 모른다. 앞선 v1의 augmentation-off 고득점은 다른 기본 LR·코드 조건이므로 이번 원인 설명에 혼합하지 않는다.

### 10. `traffic_ratio_2to1` — 더 많은 traffic이 신호등 F1 상승으로 직결되지 않음

노출은 대략 traffic 6,400 / roadmark 3,200회다. D=0.5532, R=0.0226. 기본 대비 D는 낮고 R은 높으며 Q 차이는 -0.0023으로 작다.

traffic 노출을 늘려도 그 안의 72.3% 음성 비율은 유지된다. 보행자·작은 양성 객체에 집중하는 변경이 아니며, 태스크별 평균 loss의 det 계수도 2배가 아니다. 공유 본체와 BN이 traffic 도메인에 더 맞춰지고, gradient 분산과 샘플 이력이 달라진다. 이것이 단순한 “traffic을 더 봤으니 D가 오른다” 예측이 빗나갈 수 있는 이유다. R 상승은 관측됐지만 그 기전을 현재 결과로 확정하기 어렵다.

### 11. `roadmark_ratio_2to1` — 도로표식 개선과 신호등 희생이 기대 방향으로 나타남

노출은 대략 traffic 3,200 / roadmark 6,400회다. D=0.4809, R=0.0323이다. 도로표식 TP가 137→188, FP가 26,789→10,898로 개선됐지만 detector recall은 0.500→0.380으로 낮아졌다.

roadmark 표본을 더 보는 것은 희소 선 학습을 도왔을 가능성이 있다. 반면 이미 적은 신호등 양성 노출을 더 줄여 detector의 적응이 느려진 것으로 해석할 수 있다. 다만 source 비율 변경에 수반되는 BN 분포 변화도 분리되지 않았다. Q는 낮지만 “두 태스크 모두 나빠졌다”는 해석은 정확하지 않다.

### 12. `roadmark_loss_0p5x` — 기본보다 두 F1이 함께 조금 높아진 후보

D=0.5831, R=0.0176으로 둘 다 기본보다 높다. source 노출은 1:1 그대로이고 roadmark BCE+Dice 계수만 0.5다.

**유력한 가설:** 새 decoder가 초기 공유 특징에 요구하는 변화가 줄어 detector와의 균형이 나아졌을 수 있다. 더 안정적인 공유 특징이 roadmark eval에도 도움이 될 수 있다. 다만 Q 이득 +0.0101은 단일 seed의 작은 차이이며, R 절대값은 여전히 매우 낮다.

주의할 점은 AdamW에서 loss를 0.5배 하는 것이 update를 0.5배 하는 것과 같지 않다는 것이다. 분자·분모의 적응적 정규화 때문에 task-exclusive 파라미터는 단순 scaling이 상당 부분 상쇄될 수 있고, 공유 파라미터에서의 방향·상대 비중과 clipping이 더 중요하다. 로그의 roadmark loss가 절반으로 보이는 것도 가중치 반영 결과이지 학습 품질이 두 배 좋아진 증거가 아니다.

### 13. `roadmark_loss_2x` — 태스크 간 교환관계가 더 명확

D=0.5353, R=0.0441, Q=0.2897로 기본 Q와 거의 같다. roadmark TP 193, FP 8,013으로 좋아졌고 신호등 recall은 감소했다.

동일한 표본에서 roadmark가 공유 파라미터의 합성 gradient에 더 크게 반영되므로 관측 방향을 설명하기 쉽다. 다만 gradient norm과 실제 update 비율을 전 step 기록하지 않았으므로 “정확히 두 배 강해졌다”는 말은 성립하지 않는다. 신호등을 더 중요시하는 선택 기준이면 기본이, roadmark를 더 중요시하면 이 후보가 선호될 수 있다.

### 14. `least_used_of_two` — 파일 방문 균형은 개선, 장면·클래스 균형은 별개

D=0.5374, R=0.0227이다. 동일 4,096-file pool과 seed/draw 규칙에서 4,800회 선택 시 고유 방문율은 기존 계산상 traffic 69.07→82.76%, roadmark 68.53→82.32%였고 최대 노출 횟수는 6~7→3회였다. sampler가 의도한 노출 편차 감소는 실제 선택열로 확인됐다.

하지만 자주 놓치는 **양성 클래스**를 더 뽑거나 **독립 장면**을 균등하게 고르는 방법은 아니다. 음성 영상과 서로 비슷한 프레임도 똑같이 균형화한다. 따라서 방문율 상승이 D 상승으로 직결되지 않는 것이 이상하지 않다. 이번에는 R은 개선되고 D는 낮아졌다. 표본 다양성 이득과 유용한 양성의 반복 학습 감소를 분리하는 기록이 없어 원인을 하나로 고정하지 않는다.

사용 횟수는 loss가 아니라 draw 이력 기준이다. 낮은 loss의 쉬운 표본을 선호하는 curriculum과는 다르며, 이번 실행에 loss 기반 선택 실험은 없다.

### 15. `logical_batch_16` — 더 많은 update와 바뀐 BN이 함께 작용

같은 9,600장을 600번 update했다. D=0.5461, R=0.0476. Q는 기본보다 +0.0066이다. batch 32의 300 update보다 optimizer가 두 배 자주 반응해 새 decoder의 적응을 도왔을 가능성이 있다.

그러나 물리 batch도 20→16으로 바뀌어 baseline의 20+12 묶음이 단일 16장 묶음으로 바뀐다. gradient 잡음만의 실험이 아니다. 같은 이미지 수라도 AdamW moment의 시간 단위, WD 적용 횟수, BN 업데이트가 달라진다. “작은 배치가 더 좋은 일반화”라는 일반명제를 입증하지 않는다.

### 16. `logical_batch_64` — 더 적은 update인데도 R 상승, 단순 update 횟수 설명만으로 부족

9,600장을 150번 update, microbatch 16×4 누적했다. D=0.5512, R=0.0378, Q는 +0.0043이다. noise가 작은 논리 gradient가 유리했을 가능성이 있지만, batch 16도 R이 올랐으므로 **R 상승을 update 횟수 하나로 설명할 수 없다.**

두 후보가 공통으로 물리 batch 16을 쓴다는 점도 중요하다. BN 묶음 차이나 한 seed의 변동이 공통 효과일 수 있다. KAI 학습구간 처리량은 약 50.4 img/s로 다른 일반 후보와 비슷했다. 큰 batch가 뚜렷한 속도 이득까지 보인 것은 아니다.

### 17. `logical_batch_64_lr_2x` — 선형 LR scaling은 이 조건에서 실패

LR은 `2e-4 / 2e-3 / 6e-3`다. 비교 대상은 같은 batch 64의 #16이다. D는 0.5512→0.4715, R은 0.0378→0.0367로 하락했다. 신호등 recall이 0.422→0.349로 감소했다.

전이학습 backbone, 새 head, AdamW, sparse label 혼합에서는 batch를 두 배 키웠다고 LR 두 배가 정답이 아니다. 더 큰 update가 detector의 안정적인 적응에 불리했을 가능성이 높다. 이 결과는 이번 linear-scaling 조합에 대한 반증이며, batch 64 자체를 기각할 근거는 아니다.

### 18. `microbatch_8` — VRAM은 줄지만 품질이 그대로라는 보장은 없음

논리 batch 32는 유지하고 물리 8×4로 바꿨다. D=0.5126, R=0.0228이다. loss 정규화는 논리 batch 기준을 보존하지만 BN 통계는 보존하지 않는다. 그래서 이 결과를 accumulation 구현 오류로 단정할 수 없다.

이번 KAI run의 학습 처리량은 약 51.8 img/s, peak allocated 메모리는 3,092,450,816 bytes(약 3.09GB)로 자원 측 장점이 있다. 다만 기본 D를 유지하지 못했으므로 “메모리만 절약하는 완전히 동등한 설정”이라고 취급하면 안 된다. 품질 차이는 BN, BF16 연산 순서, 최적화 경로가 후보 원인이다.

### 19. `pcgrad` — 약한 충돌을 줄이는 대가가 관측 이득보다 큼

D=0.5179, R=0.0180이다. KAI에서 초기 OOM 1회로 물리 batch가 **20→10**이 됐고, 학습 처리량은 약 **39.0 img/s**였다. 일반 후보 약 50 img/s보다 느렸다.

기록된 초기 shared-gradient cosine은 -0.0189, 300 step에서는 -0.0261이었다. norm은 초기 det 312.1 / roadmark 0.3534, 마지막 det 6.16 / roadmark 1.06이었다. 두 기록은 **거의 직교인 약한 음의 정렬**이며 큰 방향 충돌의 증거는 아니다. 전 step 로그가 없어 충돌 빈도를 추정할 수 없다.

PCGrad는 음의 내적 성분을 투영으로 제거하지만, 희귀 양성을 늘리거나 작은 객체의 해상도를 개선하거나 BN을 바로잡지는 않는다. 현재 구현은 양쪽 task gradient가 존재하는 파라미터의 내적·norm으로 투영하므로, 전체 벡터에서 norm을 정의하는 변형과도 구별해야 한다. 이번 결과는 이 구현과 microbatch 10을 포함한 조합의 결과다. **PCGrad 단독 효과를 판단하려면 일반 합산도 microbatch 10으로 맞춘 대조군이 필요하다.** [원 논문](https://arxiv.org/abs/2001.06782).

### 20. `gradnorm_alpha_0p5` — 실패 이유는 알고리즘 성능이 아닌 `None` 처리

최종 F1이 없다. `_assign_task_gradients()`에서 `det_gradient is None`일 때 `weights[1] * road_gradient`를 수행하지만, **양쪽 gradient가 모두 None인 파라미터**를 처리하지 못해 `Tensor * NoneType`로 중단됐다. 초기 OOM 뒤 microbatch 10으로 줄고 첫 update가 성공한 로그는 있지만 정확한 실패 step은 로그에 남지 않았다.

이 모델에서는 양쪽 None이 정상일 수 있다. Ultralytics bbox loss는 foreground assignment가 있을 때만 계산되고, roadmark loss는 detector 전용 regression head를 사용하지 않는다. 72.3%의 traffic 음성 비율은 이런 비활성 경로가 나올 수 있는 실질적인 조건이다. 다만 실패 당시 파라미터 이름과 batch가 저장되지 않아 그 정확한 파라미터를 지목할 수는 없다.

alpha 0.5가 약해서 학습이 실패했다는 해석은 근거가 없다. 나는 이 조합의 처리 누락을 만들었고, 정상적으로 지원해야 할 무감독 파라미터 경로를 검증하지 못했다. 이번 문서화에서는 코드를 수정하거나 재실행하지 않았다.

### 21. `gradnorm_alpha_1p5` — 같은 구현 오류, alpha 우열 판단 불가

#20과 동일한 위치와 오류로 실패했다. alpha는 상대적으로 느린 태스크에 목표 gradient norm을 얼마나 강하게 배분할지 조절하지만, 이번에는 그 효과를 평가할 run을 완료하지 못했다.

수정 후에도 검토할 모델 측 관계가 있다. detector loss는 one2many/one2one 가중치가 일정에 따라 변하므로, 최초 loss 대비 감소율이 순수한 “학습 진도”만 나타내지 않는다. 본 구현의 GradNorm은 그 감소율과 공유 gradient norm을 사용한다. alpha 비교는 정상 run과 이 내부 일정의 영향을 함께 봐야 한다. 두 실패를 0점으로 넣거나 GradNorm이 나쁘다고 순위를 매기지 않는다. [GradNorm 원 논문](https://proceedings.mlr.press/v80/chen18a.html).

### 22. `schedulefree_1x` — 새 head의 적응과 평가 평균 가중치 사이 지연 가능성

LR `1e-4 / 1e-3 / 3e-3`, 50-step warmup, 외부 LR decay 없음. D=0.2432, R=0.0071로 낮다. 검출 FP 354, TP 72이며 precision은 0.169다. recall 0.434는 기본의 0.500보다 낮지만 precision 붕괴가 특히 크다.

학습과 평가에 다른 iterate를 쓰는 Schedule-Free에서는 새 head의 초기 나쁜 상태가 평가 평균에 영향을 줄 수 있다. 큰 LR을 필요로 하는 경우가 있다는 공식 안내와도 일치한다. 실제 train 마지막 det loss는 0.7567인데 검증은 4.7089다. **초기 평균 가중치의 적응 지연, train/val 분포 차이, BN 재추정 오차가 후보 원인**이다. train/eval loss는 표본과 모드도 달라 이 차이만으로 특정 원인을 입증하지는 않는다.

평가 시 `optimizer.eval()`을 호출하고, 학습 표본 32장/source로 BN을 reset 후 재추정한다. 따라서 단순 모드 전환 누락은 현재 코드의 설명이 아니다. 단 64장으로 119개 BN을 재추정하는 충분성은 확인되지 않았다. [공식 구현의 평가·BN 안내](https://github.com/facebookresearch/schedule_free).

### 23. `schedulefree_3x` — 약간 회복했지만 낮은 precision을 해결하지 못함

LR `3e-4 / 3e-3 / 9e-3`. D=0.2536, R=0.0148이다. #22보다 roadmark F1과 검증 det loss는 개선됐지만 detector precision 0.181은 여전히 낮다.

같은 초기 LR의 AdamW 3×(#03)와 비교하면 Schedule-Free는 평균 방식·warmup·평가 BN 재추정까지 함께 바뀐다. 따라서 이 차이는 optimizer update 공식 하나의 효과로 분리되지 않는다. 동일 LR 예산에서 이 Schedule-Free recipe는 충분히 적응하지 못했을 가능성이 크지만, 느리게 좋아질 후보를 300 step에서 탈락시킨 영향도 남는다.

### 24. `schedulefree_10x` — 큰 LR이 도로표식 배경 억제를 회복

LR `1e-3 / 1e-2 / 3e-2`. D=0.4262, R=0.3602다. roadmark TP 192, FP 322, pixel F1 0.4836으로 **1차 도로표식 최고**다. #22와 같은 평균·BN 평가 경로를 쓰면서 이 수준까지 오른 것은 “Schedule-Free는 이 CNN에 작동하지 않는다”는 결론을 반박한다.

큰 update가 새 decoder를 초기 불량 평균에서 빠르게 벗어나게 했을 가능성이 있다. 한편 detector recall은 0.313으로 낮아 기본 D보다 떨어진다. 이 조합에서 큰 LR이 roadmark에 효과적이라는 것은 확인됐지만, 높은 LR 자체와 Schedule-Free의 기여는 AdamW 10× 대조군이 없어 분리되지 않는다. 900-step 결과는 6절에 별도로 비교한다.

### 25. `prodigy_dcoef_0p3` — 공유 LR 척도가 새 decoder에는 작았을 가능성

D=0.5535, R=0.0044다. 체크포인트에서 확인한 300-step `d`는 **9.8298e-5**이며, 세 optimizer group 모두 `lr=1`, 같은 `d`를 사용했다.

본 구현은 AdamW용 group LR을 제거한 뒤 Prodigy를 생성한다. 즉 기본의 backbone:detector:roadmark LR 비율 `1:10:30`을 유지하지 않는다. 추정된 공통 척도가 기존 본체에 가까운 크기로 형성되면 새 decoder는 상대적으로 느릴 수 있다. train roadmark Dice loss도 약 0.7721로 일반 후보보다 높아 적응 지연 가설을 뒷받침한다.

`d`는 적응적 update의 scale이지 AdamW의 LR과 동일한 양이 아니다. 따라서 정확히 30배 느리다고 계산해서는 안 된다. 다만 **자동 LR이 태스크·모듈별 최적 LR을 자동으로 따로 찾은 실험은 아니라는 것**은 코드와 state에서 확정된다. [Prodigy 공식 안내](https://github.com/konstmish/prodigy).

### 26. `prodigy_dcoef_1` — coefficient를 올려도 `d`가 비례해서 커지지는 않음

D=0.5434, R=0.0030. `d=1.0693e-4`로 d_coef 0.3의 약 1.09배뿐이다. d_coef가 3.33배라는 이유로 실제 update scale도 그만큼 커졌다고 말할 수 없다. 학습 경로가 달라지면 추정기 입력도 달라진다.

train roadmark Dice loss는 #25보다 낮은 약 0.6485지만 validation line FP는 41,315로 더 많다. 이것은 train objective 개선이 eval 배경 억제로 바로 이어지지 않았다는 관찰이다. 공유 scale, BN, 도메인 차이와 출력 threshold를 함께 의심해야 한다. Prodigy 원리 자체가 실패했다는 증거는 아니다.

### 27. `prodigy_dcoef_3` — roadmark fitting은 빨라졌으나 detector 성능은 하락

D=0.4647, R=0.0099. `d=1.8328e-4`로 #26보다 약 1.71배다. train roadmark Dice loss는 약 0.5950으로 더 낮아졌고 line FP도 16,376으로 줄었지만, D와 Q가 하락했다.

공통 scale을 키워 새 decoder를 더 빨리 움직이려다 사전학습 검출 경로에도 부담을 준다는 설명과 일치한다. 그러나 높은 d_coef가 detector를 손상시켰다는 직접 weight-update 측정은 없다. 이번 Prodigy 세 결과가 보여주는 것은 **모듈별 LR 비율을 없앤 현재 조합이 이 짧은 전이학습 예산에서 좋은 균형을 찾지 못했다**는 사실이다. 세 후보 모두 `slice_p=11`, constant schedule을 사용했다.

## 6. 완료된 승격 실험은 별도로 비교

다음 두 후보는 누적 28,800장, 900 step을 완료했다. 같은 예산·같은 평가셋끼리만 비교한다. 나머지 승격 7개는 이 문서의 관찰 시점에 최종 결과가 미확정이다.

| 후보 | 256장 D | 256장 R | Q | 검증 목록 전체 8,192장 D | 전체 R | 전체 D/R 평균 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| AdamW LR 3× | 0.6028 | 0.4174 | 0.5101 | **0.5754** | **0.4022** | **0.4888** |
| Schedule-Free 10× | 0.5758 | 0.3966 | 0.4862 | 0.5230 | 0.3848 | 0.4539 |

1차에서 Schedule-Free 10×가 roadmark만은 앞섰으나, 900 step에서는 AdamW 3×가 전체 line F1과 signal F1 모두 앞섰다. 하지만 클래스별 결론은 다르다.

| 8,192장 평가의 클래스별 F1 | AdamW LR 3× | Schedule-Free 10× |
| --- | ---: | ---: |
| 차량 신호등 | 0.6175 | 0.5778 |
| 보행자 신호등 | **0.3247** | 0.1559 |
| 흰 차선 | **0.4536** | 0.4075 |
| 노란 차선 | 0.3256 | **0.3635** |
| 정지선 | 0.0131 | **0.0192** |

Schedule-Free는 노란 선과 정지선 F1이 약간 높지만, 정지선 절대 성능은 둘 다 낮다. AdamW는 정지선 TP 19/GT 111, FP 2,771이고, Schedule-Free는 TP 15, FP 1,436이다. Schedule-Free의 더 높은 정지선 F1은 **더 많이 찾은 결과가 아니라 오탐을 줄인 결과**다. 두 방식 모두 정지선 품질을 확보했다고 할 수 없다.

보행자 신호등 recall은 AdamW 0.2187, Schedule-Free 0.0883이다. training에 보행자 객체가 285개뿐이었고, 평가 데이터에서 더 작은 보행자 신호등 비율이 높다는 점이 공통 한계다. optimizer 간 차이의 원인은 아직 분리되지 않았다.

전체 검증 결과는 `latest`, `best`, `best_roadmark` 세 역할 모두 확인됐으며, 이 두 run에서는 각각 같은 최종 성능이었다. **전체 검증은 저장된 4,096장/source 목록 전체라는 뜻이며 원본 Validation 전체 57,700장도, 별도 test set도 아니다.** 순위용 256장을 포함하므로 독립 test 성능으로 표시하지 않는다.

## 7. 어떤 설명에 더 무게를 둬야 하나

### 7.1 가장 강한 근거: 도로표식 성능은 배경 오탐 억제에 크게 좌우됐다

큰 LR 두 후보는 TP 증가와 함께 수만 개 FP를 수백 개로 줄였고, pixel F1과 BCE도 함께 개선됐다. 따라서 단지 metric 하나가 우연히 오른 결과는 아니다. 현재 decoder의 sparse-target 학습과 eval 출력 안정화가 중요한 축이라는 해석은 강하다.

다만 **배경 logit calibration이 좋아졌다는 관찰**과 **그 원인이 BN이라는 주장**은 구별해야 한다. 여러 run은 train 마지막 BCE가 약 0.02인데 검증 BCE는 0.2~1 이상이다. 이 격차는 BN 또는 도메인 차이를 조사할 이유이지, BN 결함의 증명은 아니다. 같은 train subset을 eval mode로 평가하고 train-only BN 재계산 전후를 비교하는 진단이 원인을 가르는 최소 후속 작업이다.

### 7.2 신호등은 최적화보다 데이터·해상도의 제한도 큼

작은 box, 음성 영상의 과다, 적은 보행자 표본은 실제로 집계된 사실이다. LR·task balancing은 이 조건에서 gradient를 바꾸지만, 없는 다양성이나 입력 해상도를 만들어주지 않는다. 보행자·작은 객체별 recall이 개선되지 않으면 총 D 상승만으로 목적을 달성했다고 평가하지 않는 편이 타당하다.

### 7.3 손실 가중치·source 비율·head LR은 서로 대체재가 아님

| 조정 | 직접 바뀌는 것 | 함께 바뀔 수 있는 것 |
| --- | --- | --- |
| source 비율 | 어느 영상의 정답을 얼마나 보는가 | 양성 빈도, gradient 분산, BN 도메인 |
| task loss weight | 공유 gradient를 합칠 때의 상대 크기 | 방향, clipping, AdamW moment |
| head LR | 해당 모듈의 optimizer update scale | 다음 step의 공유 특징 gradient |
| microbatch | 한 forward에서 BN이 보는 영상 묶음 | running statistics, 메모리, BF16 수치 경로 |

예를 들어 roadmark 비율 2×와 loss 2×는 R을 모두 올렸지만 D 감소 크기는 달랐다. 같은 이름의 “roadmark 강화”로 묶어 한 가지 법칙을 만들면 안 된다.

## 8. 실험 설계와 이전 설명에서 바로잡을 부분

1. **ASHA를 실행한 것은 아니다.** [현재 runner](../tools/run_pv26_method_search.py)는 전체 후보가 한 예산을 마친 뒤 정렬하여 다음 예산을 주는 순차 successive halving이다. 비동기 ASHA나 자동 candidate 생성·Bayesian search는 구현하지 않았다. 27개는 사람이 정한 설정 목록이며 전 조합 factorial 탐색도 아니다. [ASHA 논문](https://arxiv.org/abs/1810.05934)과 현재 구현을 구분한다.
2. **조기 탈락의 타당성은 검증하지 못했다.** 탈락한 후보를 28,800장까지 모두 학습한 대조군이 없으므로 늦게 좋아지는 방법을 놓쳤는지 모른다. 명목상 27개 모두 최대 예산을 주는 것보다 training image 노출을 약 44.4% 절약하는 설계지만, 그만큼 최종 순위를 정확히 보존했다는 증거는 없다. 평가 비용은 이 절약 계산에서 제외된다.
3. **승격 Q는 실험자가 정한 선택 정책이다.** 두 태스크 micro-F1의 동등 평균이고, 희귀 클래스 보호나 차량용 합격 기준은 아니다. #10과 #13처럼 기본 Q와 차이가 작은 후보 사이의 순위를 통계적 사실로 과장하지 않는다.
4. **GNS를 이용해 batch를 자동 선택한 것은 아니다.** 예전 별도 진단의 aggregate 약 1,087.8, 개별 유효 추정 7/12는 저장돼 있지만 이전 checkpoint와 이전 측정 방식이다. 측정 후 코드가 train-mode·논리 손실 정규화 등으로 수정됐고 해당 방식의 새 결과는 확인되지 않았다. 작은 차이의 분모로 큰 값을 만들 수 있어 이 수치로 “적정 batch 1,088”을 제시하면 안 된다. 현재 BN·태스크 혼합에서도 단순 잡음 모델의 가정이 얼마나 맞는지 확인해야 한다. [GNS 논문](https://arxiv.org/abs/1812.06162).
5. **Muon 미실험 사유를 잘못 설명했다.** 내가 PyTorch 내장 구현의 2D 제한을 Muon 자체의 CNN 비호환으로 확대했다. 원저자의 구현과 글은 4D convolution update의 마지막 세 차원을 펴는 방법을 명시한다. PV26에 적용 가능한 연구 경로가 있으며, 이번 matrix에서 빠진 것은 적용 불가능해서가 아니라 구현·실험을 하지 않았기 때문이다. [원저자 설명](https://kellerjordan.github.io/posts/muon/), [convolution 처리 코드](https://raw.githubusercontent.com/KellerJordan/Muon/master/muon.py).
6. **clipping 빈도·실제 update/weight 비율은 측정되지 않았다.** 현재 로그에는 모든 step의 clipping 전 norm과 AdamW update 크기가 없다. 큰 초기 task-gradient norm이나 loss 숫자로 빈도를 추정하지 않는다.
7. **이전 v1의 우열을 v2에 대입하지 않는다.** 예를 들어 augmentation off와 batch-64 LR scaling에 대한 이전 설명은 v2에서 방향이 달라졌다. baseline decoder LR·코드·평가 조건이 바뀌었으므로 그 차이를 방법론의 모순이나 재현 성공으로 부르지 않는다.

수십 후보의 결과를 작은 동일 검증셋으로 반복 선택하면 검증셋에 대한 선택 편향도 생긴다. 현재 결과를 다음 후보를 고르는 자료로 사용하고, 촬영 구간 분리 평가와 여러 seed의 짝지은 비교로 큰 차이를 확인하는 것이 적절하다. [Tuning Playbook](https://github.com/google-research/tuning_playbook).

## 9. 현재 결과가 지지하는 후속 순서

**첫째, 진행 중인 9개 승격 후보를 같은 28,800장 예산으로 마친 뒤 비교해야 한다.** 특히 baseline의 긴 예산 결과가 없으면 큰 LR의 이득이 최종 품질인지 빠른 수렴인지 구분할 수 없다.

**둘째, GradNorm 두 실패는 고친 뒤 같은 조건으로 다시 평가해야 한다.** 필요한 수정은 양쪽 gradient가 없는 parameter를 해당 update에서 제외하는 정상 경로 처리다. 이는 방법론에 유리하게 loss를 바꾸는 일이 아니라 미완료 비교를 가능하게 하는 수정이다.

**셋째, 큰 도로표식 차이에 대해서는 BN과 decoder LR을 분리하는 것이 정보량이 크다.** 같은 checkpoint의 BN 재추정 전후와 decoder `9e-3` 단독 후보가 최소 대조다. BN 진단이 원인을 설명한다면 optimizer 순위보다 먼저 그 평가·학습 조건을 맞춰야 한다.

**넷째, 희귀 클래스와 작은 객체를 평가·노출 설계에 반영해야 한다.** 표본 파일 수보다 장면 단위 분리와 보행자/정지선 support가 중요하다. two-choice가 해결한 노출 횟수 편차와 실제 유용 정보량의 편차는 다르다.

**다섯째, 미실험 방법을 성능 열세로 처리하지 않는다.** Muon, 무작위 비복원 shuffle, loss 기반 curriculum, 실제 ASHA의 자원 효율, 방법 간 조합은 이번 27개 결과로 판정할 수 없다. 현재 수치로 가장 강한 후보는 AdamW LR 3×이지만, 전체 방법론의 최종 우승자나 배포 권고로 확정할 단계는 아니다.

## 10. 근거와 재확인 위치

- 실험 원본: `kai:/home/kai/yolopv26/runs/20260922_full_method_search_v2/`
- 1차 결과의 고정 원장: `search_results.json → rungs[0].results`.
- 후보의 실제 설정: `trials/<name>/run_config.json`. 재생성되는 YAML보다 이 실행 snapshot을 기준으로 해석했다.
- 중간·최종 지표: `validation.json`, `summary.json`, `validation_full_latest.json` 등. 최신 summary는 승격 후 1차 값을 덮어쓰므로 같은 예산 비교에는 rung 기록을 사용했다.
- 학습 loss·microbatch·OOM·GradNorm traceback: `tmux.log`. 첫 7개 로컬 trial의 상세 로그는 이 원격 로그에 포함되지 않는다.
- 분석 시점의 소형 결과 사본: `/home/user1/Storage/ROS2_Workspace_offload/yolopv26/20260922_method_analysis/snapshot/`.
- 실제 표본 목록과 label census: 같은 분석 폴더의 `train_samples.jsonl`, `val_samples.jsonl`, `census.py`, `data_census.json`.
- Prodigy `d`는 KAI의 각 `checkpoints/latest.pt` optimizer group에서 CPU로 읽었다. 이 값은 300-step 결과이며 step별 변화 곡선은 보존되지 않았다.
- 구조·정규화: [모델](../model/net/pv26.py), [피더와 표적](../model/data/dataset.py), [손실](../model/engine/loss.py), [트레이너](../model/engine/trainer.py), [optimizer·평가 연결](../tools/pv26_train/cli.py).

이번 작업은 기존 결과·소스·라벨 분석과 문서화다. 학습 코드, KAI 설정, 실행 중인 trial은 변경하지 않았고 로컬 GPU 학습도 실행하지 않았다. 900-step 결과가 아직 없는 후보는 1차 결과만으로 설명했다.
