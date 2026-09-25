# PV26 80k 이후 연구 및 실험 설계

작성·자료 확인: 2026-09-23. 최초 연구·설계 후 구현 진행 상태를 10절에 기록했다. **장기 학습이나 지정 MCAP 최종 성능 평가는 아직 실행하지 않았다.**

## 1. 다음 실행자가 먼저 읽을 내용

공통 출발점은 80,000 step의 latest.pt다. 먼저 현재 선 추출의 격자 효과와 모델의 미검출을 분리하고, 동일 체크포인트의 작은 학습률 파인튜닝을 기준 실험으로 만든다. 데이터·손실·최적화 방법은 하나씩 비교하고, 선 단위 표현이나 시간 모델은 확인된 한계를 해결하는 별도 실험으로 진행한다.

처음 계획의 F 6후보만으로는 정식 DoE를 완성할 수 없다는 사용자 지적에 따라 [교차 요인·반복 seed를 포함한 DoE](20260923_PV26_FORMAL_DOE.md)를 별도로 설계하고 자동 실행 파일을 생성했다. F 6후보는 pilot 결과로만 취급한다.

사용자가 최종 기준으로 지정한 기록은 다음이다.

~~~
/home/kai/KAI_ws/records/260920/rosbag2_2026_09_20-14_07_38/
~~~

이 bag은 **최종 평가 전용**이다. 라벨 작성, 영상 identity·timestamp, 신호등 검출+SignalAttr 연결, 차선·정지선·시간 지표와 평가 공개 순서는 [MCAP 최종 평가 설계](20260923_MCAP_FINAL_EVALUATION_PROTOCOL.md)에 정의했다. 최종 bag에 맞춘 threshold 튜닝·추가 학습·BN adaptation은 하지 않는다.

처음 요청의 범위는 조사·실험 설계·문서화까지였고, 이후 사용자가 구현을 요청했다. 구현 경과는 10절에 기록한다. 장기 실험·라벨 작업·평가 실행은 결과와 필요한 자원을 기준으로 진행한다.

### 출발점과 우선순위

1. 현재 PV26를 보존하고 재현 가능한 개발 기준선을 만든다.
2. 선 추출만 바꾼 비교와 파인튜닝 범위·학습률 비교를 먼저 한다.
3. 신호등 상태는 제품 의미를 학습한 SignalAttr가 필요하므로 별도 학습·평가를 설계한다.
4. 신호등 성능을 악화시키지 않는 후보 중 차선·정지선 품질과 처리시간을 비교한다.
5. 개발 검증에서 후보를 확정한 후 기준 모델과 같은 SignalAttr를 붙여 최종 bag을 평가한다.

정확도 합격선은 현재 PRD에 수치로 정해져 있지 않다. 아래 숫자는 명시한 경우를 제외하면 **탐색 시작값·실험 예산 제안**이다. 실차 안전 기준이나 이미 검증한 최적값으로 취급하지 않는다.

## 2. 확인한 코드·가중치·데이터 상태

### 2.1 기준 모델

~~~
/home/kai/yolopv26/runs/20260922_1607_joint_lr3x_80k/checkpoints/latest.pt
~~~

2026-09-23 확인한 저장소 HEAD는 47ea4db6dfecc6d88ef1c90250e90704f746b89e였다. config/pv26.yaml의 80k·LR 변경과 model/engine/evaluation.py의 persistent worker 수정은 미커밋 상태였다. 다음 에이전트는 현재 diff를 확인하고 보존해야 한다. HEAD 문자열만으로 실제 실행 소스가 완전히 재현된다고 주장하지 않는다.

| 항목 | 확인 결과 |
| --- | --- |
| 모델 | YOLO26-s 공유 body/neck + 차량·보행자 2-class detector + 3-channel roadmark decoder |
| 입력 / 도로표식 출력 | 608×800 / 152×200, stride 4 |
| decoder 폭 | 64 |
| 학습 | joint 80,000 step 완료, 종료 요청 아님, 기록상 skipped update·OOM retry 0 |
| optimizer / schedule | AdamW / cosine |
| 최초 LR | body 3e-4, detector head 3e-3, roadmark 9e-3 |
| 완료 checkpoint의 실제 LR | 세 group 모두 0.0 |
| 배치 / 정밀도 | logical 32, microbatch 20, BF16 |
| 데이터 | train: traffic 150,000 + roadmark 251,081; saved val: 30,000 + 27,700 |
| 감독 | source kind별 partial label. 미라벨 태스크를 negative로 처리하지 않음 |
| 증강 | 좌우 반전, brightness·contrast 변화 |
| 도로표식 loss | 1-output-pixel 중심선 target의 BCE + Dice |
| 신호등 상태 | PV26 box 이후 별도의 SignalAttr crop classifier |

출처: 해당 run의 run_config.json·summary.json·latest.pt, [모델](../model/net/pv26.py), [데이터](../model/data/dataset.py), [손실](../model/engine/loss.py), [학습 CLI](../tools/pv26_train/cli.py). optimizer의 backbone group은 detector.model[:-1]이며 이름과 달리 neck도 포함한다.

### 2.2 실제 전체 AIHub 검증 결과

각 가중치에서 57,700장의 저장된 validation 전체를 평가한 수치다. MCAP 성능이나 독립 촬영 구간 일반화 수치가 아니다.

| 가중치 | step | 신호등 검출 F1 | 도로표식 선분 F1 | 흰 차선 F1 | 노란 차선 F1 | 정지선 F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| latest | 80,000 | 0.76838 | 0.65299 | 0.69453 | 0.54856 | 0.35710 |
| best | 54,000 | 0.77016 | 0.64008 | 0.68939 | 0.52378 | 0.31116 |
| best_roadmark | 63,600 | 0.76556 | 0.64861 | 0.69214 | 0.53770 | 0.35282 |

latest의 선분 precision / recall은 흰 차선 0.8229 / 0.6008, 노란 차선 0.5678 / 0.5306, 정지선 0.2694 / 0.5294다. 흰 차선에서는 누락 개선 여지가 있고, 정지선은 오탐도 큰 연구 대상이다. 매칭 성공 선의 평균 위치 오차는 각각 2.27 / 2.72 / 3.01 원본 pixel이다. 이 평균이 작은 것은 미검출 선의 품질을 보장하지 않는다.

후반부 주기 검증은 63,600 step의 도로표식 F1 0.6605, 80,000 step의 0.6543이었다. 같은 recipe를 더 오래 실행하면 반드시 개선된다는 근거는 없다. 동시에 cosine 후반의 작은 LR도 영향을 주므로 구조적 한계나 과적합으로 단정하지 않는다.

관련 산출물: [run 요약](../runs/20260922_1607_joint_lr3x_80k/summary.json), [전체 latest 검증](../runs/20260922_1607_joint_lr3x_80k/validation_full_latest.json), [6장 비교 이미지](../runs/20260922_1607_joint_lr3x_80k/inference_lane_samples_20260923/contact_sheet.png).

### 2.3 자글거림의 확인 사실과 가설

[현재 decoder](../model/engine/postprocess.py)는 확률맵의 정수 위치에서 ridge를 찾고 그 점들을 연결한다. 1280폭 AIHub 영상은 800으로 줄어들어 출력 한 칸이 원본 약 6.4px이고, 최종 MCAP의 800폭 입력에서는 약 4px다. 각 점에 모델 불확실성과 격자 양자화가 함께 나타날 수 있다.

- 확인: 현재 추출 좌표가 grid center로 제한된다.
- 가설: 격자 사이 좌표 추정이나 국소 곡선 fitting으로 일부 자글거림이 줄 수 있다.
- 미확인: 전체 오류 중 양자화·확률맵 noise·잘못된 line association 각각의 기여도.
- 주의: 시각적으로 매끈해져도 곡률·끝점·작은 정지선이 지워지면 실제 품질은 악화될 수 있다.

### 2.4 SignalAttr는 별도 기준선이 필요

[기본 포함 가중치](../models/signal_attr/README.md)는 legacy_arrow 의미이며 제품 추론에서 state_valid=false가 된다. 이번 run의 PV26 학습으로 SignalAttr까지 학습된 것은 아니다. 조사 시작 시 runs 안에는 제품용 SignalAttr 장기 완료 run이 없었고, 이후 50-step S1을 10절에 기록했다.

이미 생성한 제품 crop은 runs/20260922_011743_signal_crops에 train 203,496개, val 50,661개가 있다. manifest의 state_semantics는 left_arrow, **all_off_is_valid=true**다. 예전 소규모 보고서의 off 제외 정책과 다르다. 새 실행은 이 실제 snapshot 정책을 명시적으로 계승하고, 바꾸는 실험은 별도 dataset/run으로 수행한다.

train의 희귀 조합은 arrow 단독 146개, yellow+arrow 40개이고 off는 56,800개다. 이 수는 개별 crop 수이며 독립 주행 장면 수가 아니다. 균등 sampling에서 희귀 crop을 반복 노출하는 것이 일반화 개선을 보장하지 않는다.

### 2.5 이전 방법론 탐색을 읽는 법

[기존 search.yaml](../runs/20260922_full_method_search_v2/search.yaml)은 27개 후보를 9,600→28,800 image draws로 비교했다. logical batch 32이면 300→900 step이다. 초기학습에서는 adamw_lr_3x가 마지막 ranking의 선두였고, GradNorm 두 후보는 status=failed였다. 실패는 성능 열세와 다르다.

기존 [search runner](../tools/run_pv26_method_search.py)는 신호등·차선 F1 평균으로 초기학습 후보를 정렬했다. 후속 구현에서는 stage2 base config의 initial_checkpoint·data.index_run을 읽어 새 run을 시작하고, 모든 후보를 같은 joint 개발 목록으로 평가하는 모드를 추가했다. 이전 탐색의 평균 점수 방식은 기존 config에서 유지하며, stage2 config는 두 태스크 점수를 따로 보고 자동 단일 승자를 선언하지 않는다.

## 3. 문헌 조사와 채택 판단

1차 출처인 원 논문·학회 공개본·저자 저장소를 확인했다. 2026년 연구까지 검색했으며 전수 문헌 목록이라는 뜻은 아니다. 각 방법의 published score/FPS는 데이터·backbone·장치·평가 방식이 달라 PV26 지표와 직접 비교하지 않는다. 논문 제목의 아이디어가 현재 코드에 구현됐다는 의미도 아니다.

| 근거 | 핵심 내용 | 이 연구에서의 사용과 한계 |
| --- | --- | --- |
| [R01 AdamW](https://arxiv.org/abs/1711.05101) | adaptive optimizer와 weight decay를 분리 | 현행 대조군 유지. optimizer 교체 효과와 LR·WD 효과를 분리 |
| [R02 LP-FT, ICLR 2022](https://arxiv.org/abs/2202.10054) | head 정렬 후 전체 fine-tuning으로 feature distortion 완화 | decoder-only→joint의 근거. 분류 연구를 차선 개선 증거로 직접 대입하지 않음 |
| [R03 L2-SP, ICML 2018](https://proceedings.mlr.press/v80/li18a.html) | 0 대신 pretrained weights를 기준으로 regularization | joint가 신호등을 잊을 때 제한적 후보. 현재 코드에는 없음 |
| [R04 Learning without Forgetting](https://arxiv.org/abs/1606.09282) | 기존 모델의 출력 지식으로 이전 태스크 보존 | 신호등 teacher distillation 후보. 원래 신호등 GT가 있으므로 우선 GT replay를 비교 |
| [R05 PCGrad, NeurIPS 2020](https://papers.neurips.cc/paper_files/paper/2020/file/3fe78a8acf5fda99de95303940a2420c-Paper.pdf) | 충돌하는 태스크 gradient의 projection | 충돌·trade-off가 반복되는 경우 비교. 추가 backward 비용 포함 |
| [R06 GradNorm, ICML 2018](https://proceedings.mlr.press/v80/chen18a.html) | 학습 속도와 gradient 크기에 따른 loss balance | 현행 구현 재사용 가능하나 이전 failed 원인과 구현 범위를 먼저 확인 |
| [R07 Schedule-Free](https://arxiv.org/abs/2405.15682), [공식 구현](https://github.com/facebookresearch/schedule_free) | 일정 대신 iterate averaging을 활용 | 현행 후보. train/eval weights와 BN 보정을 함께 지켜야 함 |
| [R08 Prodigy](https://arxiv.org/abs/2306.06101) | adaptive step-size 추정 | 후순위 비교. 현재 구현은 group별 LR 비율을 제거하므로 AdamW와 단일 변수 비교가 아님 |
| [R09 Focal Loss, ICCV 2017](https://openaccess.thecvf.com/content_ICCV_2017/papers/Lin_Focal_Loss_for_ICCV_2017_paper.pdf) | 쉬운 negative의 loss 비중 완화 | roadmark 배경 불균형 후보. detector의 공식 loss까지 동시에 바꾸지 않음 |
| [R10 clDice, CVPR 2021](https://arxiv.org/abs/2003.07311) | skeleton과 mask의 일치로 연결성 학습 | 같은 선 내부의 단절 가설에 한정. 인접 차선·GT에서 분리한 선을 이어버릴 위험 |
| [R11 CLRNet, CVPR 2022](https://arxiv.org/abs/2203.10350) | 여러 feature 수준에서 선을 보정하고 Line IoU 학습 | instance 표현이 필요한 구조 실험. 현재 3-class mask에 loss 이름만 붙일 수 없음 |
| [R12 CLRerNet, WACV 2024](https://openaccess.thecvf.com/content/WACV2024/papers/Honda_CLRerNet_Improving_Confidence_of_Lane_Detection_With_LaneIoU_WACV_2024_paper.pdf) | LaneIoU로 위치와 confidence·assignment 관계 개선 | instance head에서 검토. 현재 mean ridge score는 calibrated lane confidence가 아님 |
| [R13 BézierLaneNet, CVPR 2022](https://arxiv.org/abs/2203.02431) | parametric Bézier 곡선 표현 | 곡선 가설 비교. 숨은 선의 임의 연장, 분기, 짧은 정지선은 별도 처리 필요 |
| [R14 UFLDv2](https://arxiv.org/abs/2206.07389) | row/column hybrid anchor와 ordinal 위치 분류 | 빠른 위치 표현 후보. 고정 lane slot·횡방향 정지선·색상 계약은 추가 설계 필요 |
| [R15 LaneTCA](https://arxiv.org/abs/2408.13852) | 인접·누적 시간 context 결합 | 영상 학습 자료를 확보한 후. 현재 shuffled image 학습만으로 구현되지 않음 |
| [R16 SIGMA-Lane, 2026](https://arxiv.org/abs/2608.16338) | 가림으로 temporal state가 오염되는 경로를 gate로 제어 | arXiv는 ECCV 2026 accepted로 표기. 복잡도·export·학습자료 부담으로 첫 라운드 후순위 |
| [R17 Cold Diffusion Lane, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Huang_When_Anchors_Meet_Cold_Diffusion_A_Multi-Stage_Approach_to_Lane_ICCV_2025_paper.html) | 여러 단계·해상도에서 lane refinement | 조사 범위에 포함하되 반복 계산을 실시간 예산과 비교해야 해 첫 구현에서 제외 |
| [R18 FDA Traffic Light, ICCVW 2025](https://openaccess.thecvf.com/content/ICCV2025W/2COOOL/html/Gakhar_Fourier_Domain_Adaptation_for_Traffic_Light_Detection_in_Adverse_Weather_ICCVW_2025_paper.html) | Fourier 기반 악천후 domain augmentation | 검출 연구이며 상태 색 보존은 별도 확인 필요. 최종 bag을 style donor로 쓰지 않음 |
| [R19 Driving-Oriented Metrics, CVPR 2022](https://arxiv.org/abs/2203.16851) | 전통 차선 지표와 후단 주행 영향의 차이 | F1만으로 차량 사용 가능성을 판단하지 않는 근거. 이 연구에 제어 simulator를 새로 만들지 않음 |
| [R20 Hyperband](https://www.jmlr.org/papers/v18/16-558.html) | 작은 자원으로 후보를 선별하고 유망 후보에 증액 | 기존 successive-halving 개념 재사용. 새로운 head의 느린 초기학습을 성숙 모델과 조기 탈락 비교하지 않음 |
| [R21 SWA](https://arxiv.org/abs/1803.05407) | 여러 학습 가중치 평균으로 일반화 개선 | optimizer 연구 후의 가중치 평균 후보. 공간적 선 smoothing과 다른 효과 |

우선순위는 논문 발표연도가 아니라 현재 실패·변경 비용·추론 부담으로 정한다. 새 코드를 실제 도입할 때 해당 버전의 라이선스·의존성과 TensorRT/TorchScript 지원을 확인한다. 이번 조사에서는 외부 모델·데이터를 다운로드하거나 설치하지 않았다.

## 4. 개발 데이터와 비교 원칙

### 4.1 자료 역할

| 자료 | 역할 |
| --- | --- |
| 기존 saved train index | 기본 supervised 학습 pool과 source membership 기준 |
| 기존 AIHub saved validation 전체 | 이미 사용한 개발 평가. 전체 candidate 비교의 기준 |
| 개발 subset | scene·source·클래스·작은 신호등 등 조건이 포함되는 고정 선별 집합 |
| 추가 현장 개발 기록 | domain adaptation·temporal 학습·threshold calibration. 최종 bag과 다른 주행 |
| 사용자 지정 260920 14:07:38 bag | 최종 평가만 |

현재 27만 장 등의 수는 샘플 수이지 scene 수가 아니다. 원본 metadata의 촬영 구간 정보로 같은 구간·인접 프레임을 묶고 train/dev 분리와 좌우 중복을 확인한다. filename 순서나 임의의 접두부를 촬영 구간의 증거로 간주하지 않는다. scene ID가 없으면 그 한계를 기록하고 기존 split을 과장하지 않는다.

기존 모델이 학습한 프레임 일부를 새 holdout으로 떼어내도 현 checkpoint에는 이미 노출된 자료다. 원래 train 이미지는 개발용 추가 진단으로 쓸 수 있지만 독립 holdout이라고 부르지 않는다.

### 4.2 동일 조건 비교

- 같은 start checkpoint, seed, image draw budget, source ratio, 평가 membership, threshold로 한 축씩 바꾼다. seed 26으로 선별하고 유망 후보·대조군을 27·28에서도 paired 반복하는 것을 제안한다.
- 논리 배치 변경은 update 수와 image draws를 함께 기록한다. microbatch 변경은 BN 통계를 바꿀 수 있으므로 단순 누적배치 동치라고 간주하지 않는다.
- frozen decoder 실험은 도로표식 이미지만 소비하고 joint는 traffic도 소비한다. 총 draws를 맞춘 주 비교와 태스크별 실제 draws를 모두 보고한다. 같은 epoch라고 표시하지 않는다.
- 대조군과 후보의 신호등 검출, 선분, 상태, wall time을 함께 기록한다. 느린 방법은 같은 draws와 같은 wall-time 시점의 결과를 구분한다.
- 후보를 결합할 때는 base / A / B / A+B로 보완·간섭을 확인한다. 모든 후보의 Cartesian product를 돌리지 않는다.
- 주기 subset은 source당 1,024장 현행 기준을 시작점으로 하되 희귀 상태·정지선·작은 신호등의 support를 확인한다. 포함 사례가 적으면 해당 조건을 별도 stratum으로 보고한다.
- 구현 오류·nonfinite·OOM으로 실패한 run과 낮은 metric의 run을 구분한다. 실패 후보를 성능 순위의 최하위 점수로 대체하지 않는다.

### 4.3 후보 채택 기준

PRD의 신호등 우선순위를 유지한다. line F1과 signal F1의 단순 평균이 올라가더라도 신호등이 악화되면 개선 후보로 승격하지 않는다.

개발 단계에서는 신호등 종류별 F1/recall과 상태 오류, 도로표식 클래스별 F1/coverage/위치, 시간 비용을 나란히 보고 Pareto 후보를 남긴다. 새 비열등 허용폭은 정하지 않는다. 작은 변동은 paired 반복과 장면 단위 불확실성으로 확인하고, 판정이 불확실하면 그 사실을 남긴다. 최종 bag 결과에 맞춰 metric 가중치나 허용 손실을 바꾸지 않는다.

## 5. 실험군과 하이퍼파라미터

### 5.1 P: 학습 없는 선 추출 비교

기준 model을 고정해 **개발 영상**의 동일 logits에서 비교한다. 대규모 logits corpus를 만들지 말고 필요한 개발 subset만 cache하거나 같은 프로세스에서 후보 decoder에 전달한다.

| ID | 변경 | 바꾸지 않는 것 | 채택 판단 |
| --- | --- | --- | --- |
| P0 | 현재 ridge 정수 좌표 추출 | 모든 기본값 | 기준선 |
| P1 | 각 ridge 주변 국소 확률로 transverse subpixel 좌표 추정 | 검출한 trace·클래스·support·threshold | 위치 잔차와 자글거림 개선, coverage 유지 |
| P2 | P1 trace 내부의 국소 2차 곡선 fitting | 서로 다른 trace를 병합하거나 끝점을 외삽하지 않음 | 곡선·정지선·짧은 선 왜곡 확인 |
| P3 | P0/P1에 클래스별 threshold calibration | model weights | 개발 PR 곡선상의 누락·오탐 균형 |

P1은 peak 주변 한 grid-cell 이내의 local expectation 또는 3점 peak interpolation 중 하나를 먼저 시험한다. 두 차선 사이 전체 행의 soft-argmax는 차선 중간을 반환할 수 있으므로 사용하지 않는다. peak support가 없는 좌표를 만들어 coverage를 늘리지 않는다.

P2의 fitting 강도는 개발 자료의 위치·끝점 오차로 고른다. 전체 차선을 하나의 저차 다항식으로 강제하거나 근거 없는 gap 연결을 추가하지 않는다. 현행 AIHub에는 점선도 연속 중심선으로 주석한 사례가 있으므로 모든 점선 공백을 끊는 새 목표를 도입하지 않는다. 영상에 선을 굵게 그려 매끈하게 보이게 하는 것은 개선 실험이 아니다.

P3의 시작 threshold 후보는 0.35 / 0.50 / 0.65다. 현재 0.50을 중심으로 약한·강한 검출의 영향을 볼 탐색값이며, 최종 bag을 보고 추가 조정하지 않는다. weight 변화와 threshold 변화의 효과를 각각 보고한다.

### 5.2 F: 학습 범위·학습률

모든 새 run은 latest.pt의 model weights를 읽고 optimizer·schedule을 새로 만든다. 완료된 80k run의 resume로 새로운 LR 실험을 만들지 않는다.

LR 기준은 기존 최초 LR의 0.1배인 body 3e-5 / detector head 3e-4 / roadmark 9e-4다. 현재 익힌 표현을 작은 변화부터 보정하려는 출발점으로, 이전 0-LR checkpoint의 배수가 아니다. 이를 중심으로 약 3배씩 양쪽을 탐색한다.

| ID | 학습 범위 | body LR | detector LR | roadmark LR | 지원 상태 |
| --- | --- | ---: | ---: | ---: | --- |
| FJ-L | joint | 9e-6 | 9e-5 | 2.7e-4 | 현행 CLI 가능 |
| FJ-M | joint 대조군 | 3e-5 | 3e-4 | 9e-4 | 현행 CLI 가능 |
| FJ-H | joint | 9e-5 | 9e-4 | 2.7e-3 | 현행 CLI 가능 |
| FD-L | roadmark decoder만 | frozen | frozen | 2.7e-4 | stage=roadmark 가능 |
| FD-M | roadmark decoder만 | frozen | frozen | 9e-4 | stage=roadmark 가능 |
| FD-H | roadmark decoder만 | frozen | frozen | 2.7e-3 | stage=roadmark 가능 |
| FS | 유망 FD로 먼저 정렬 후 joint | FJ 승자 LR | FJ 승자 LR | FJ 승자 LR | 두 run의 단계별 연결로 가능; 총 예산 맞춤 |
| FN | 유망 joint에서 BN 통계 고정 | 동일 | 동일 | 동일 | freeze policy 구현 필요 |

FD는 detector 전체를 eval/frozen으로 하므로 신호등 body·head와 BN을 보존한다. 현재 stage=roadmark 검증은 신호등을 아예 계산하지 않으므로 최종 비교에서는 joint inference로 모델을 읽고 **원래 두 source의 동일 개발 평가 목록**으로 평가해야 한다. FD summary의 detector 점수를 joint 점수로 사용하면 안 된다.

FS는 첫 4k decoder-only + 8k joint를 초기 설계로 둔다. 같은 12k joint와 total draws, 태스크별 draws, wall time을 같이 비교한다. 분류의 LP-FT 논문 결과가 FS 개선을 보장하지는 않는다.

현재 CLI로 두 run을 연결하면 stage뿐 아니라 optimizer와 LR schedule도 다시 시작된다. FS는 우선 이 전체 recipe의 비교로 보고한다. freeze 자체의 기여를 주장하려면 동일한 4k+8k optimizer 재시작을 하는 joint→joint 대조를 추가한다. 단일 run의 LR 일정을 유지한 채 unfreeze하는 방식은 별도 구현·비교다.

### 5.3 검출 loss 일정은 LR 일정과 별개

현재 PV26FocusedLoss.set_progress는 검출 one-to-many / one-to-one loss 비율도 run progress에 따라 바꾼다. 설치된 Ultralytics 8.4.115는 시작 0.8 / 0.2, 마지막 0.1 / 0.9를 쓴다. 새 --initial-checkpoint 실행은 global_step=0이므로 **LR뿐 아니라 이 비율도 시작 상태로 돌아간다**.

첫 joint 비교에 두 대조군을 둔다.

- FC-reset: 현재 CLI 그대로 검출 비율을 새 run의 0.8 / 0.2부터 전이한다.
- FC-mature: 80k 완료 시점의 0.1 / 0.9를 고정하고 LR schedule만 새로 시작한다.

두 run은 같은 FJ-M LR·data·budget으로 비교한다. FC-mature는 아직 설정 항목이 없으므로 구현이 필요하다. 선택한 loss 일정은 이후 모든 joint 후보에서 공통으로 사용한다. optimizer나 loss 변경의 효과를 이 비율 reset과 섞지 않는다. resume 시에는 결정한 정책과 진행 상태를 그대로 복원한다.

### 5.4 O: optimizer·정규화·태스크 균형

F 실험의 유망 범위를 고른 뒤 아래를 순차 비교한다.

| 축 | 초기 탐색값 / 대조 | 조건·비용 |
| --- | --- | --- |
| optimizer | AdamW cosine → AdamW constant → Schedule-Free AdamW | LR를 각 방법에 맞게 개발 검증. 같은 raw LR가 같은 효과라고 가정하지 않음 |
| weight decay | 1e-4 대조, 0 / 1e-3 | 현재 trainable parameter group 구성 유지. bias/BN decay 제외는 별도 변화 |
| logical batch | 32 대조, 16 / 64 | microbatch는 VRAM 내에서 고정해 가능한 한 BN 효과 분리; image budget 일치 |
| task loss 비중 | det=1, roadmark 0.5 / 1 / 2 | source ratio를 동시에 바꾸지 않음 |
| source 비율 | traffic:roadmark 1:1 대조, 2:1 / 1:2 | 신호등 악화 여부와 실제 태스크별 노출량 보고 |
| PCGrad | sum 대조, PCGrad | shared body에서 충돌·trade-off를 실제 train batch로 진단한 경우 |
| GradNorm | sum 대조, alpha 0.5 / 1.5 | 이전 failed 오류부터 재현·수정 여부 확인; 시간 비용과 aux state 복구 |
| Prodigy | d_coef 0.3 / 1 / 3 | 현재 코드는 group LR 분리를 없앰. fine-tune 1차 후보에서 제외하고 독립 recipe로 분류 |
| weights averaging | 현행 best/latest → EMA 또는 SWA 중 한 방법 | BN 재계산은 train 자료만. Schedule-Free와 중복 결합부터 시작하지 않음 |

공유 gradient의 음수 cosine 비율만으로 PCGrad를 선택하지 않는다. supervised loss·norm·실제 신호등/차선 성능의 충돌이 함께 나타나는지 본다. 현행 전략이 논문의 shared-parameter 선택·정규화와 동일한지도 확인한다.

Schedule-Free는 optimizer.train()/eval()과 evaluation weights에 맞는 BN 보정까지 현행 경로를 재사용한다. FC-mature 또는 새 BN 정책을 붙일 때 이 순서를 깨지 않는다.

신호등을 잊는 현상이 나타나면 원본 traffic GT replay와 작은 body LR를 먼저 비교한다. 그래도 남으면 L2-SP 또는 frozen baseline detector의 distillation을 **한 방법씩** 비교한다. teacher도 final bag으로 만들거나 calibrate하지 않는다.

### 5.5 D: 데이터 구성·증강

| ID | 비교 내용 | 감독·평가 조건 |
| --- | --- | --- |
| D0 | 기존 train 전체와 source 1:1 | 기준 |
| D1 | source ratio 2:1 / 1:2 | O의 loss-weight 실험과 분리 |
| D2 | roadmark 내 노란 차선·정지선·교차로/가림 사례 비중 조절 | train 라벨·장면 또는 train-only mining으로 선정 |
| D3 | 작은 신호등·보행자·희귀 상태의 sampling 보강 | target 누락 라벨과 unreadable 구분 |
| D4 | 별도 주행 train/dev 데이터 혼합 | 최종 bag 및 같은 주행의 다른 카메라 제외 |
| D5 | brightness/contrast 대조 + 측정 가능한 blur/JPEG/scale 변화 | 신호등 색과 화살표 의미를 보존 |

rare/common 균형은 자연 분포, 빈도의 제곱근 역비례, 균등을 후보로 둔다. 같은 희귀 이미지의 복제 수만 늘리는 방식과 새 독립 장면 추가를 구분한다. 증강률과 강도는 개발 자료에서 원본 신호가 사라지지 않는 범위를 확인해 정한다.

카메라의 800×600과 기존 1280×720 소스는 종횡비·시야·대상 크기가 다르다. 같은 letterbox가 이를 완전히 없애지 않는다. 별도 현장 개발 영상으로 개선 여부를 판단하고, 최종 bag의 모습을 보고 augmentation이나 ROI를 맞추지 않는다.

crop 확대·고해상도 입력은 tiny signal recall 후보이나 별도 지연·VRAM 실험이다. 단순 upsample은 없는 광학 세부를 복원하지 않는다. 새 crop은 박스·점열·감독 mask를 같은 좌표 변환으로 처리해야 한다.

SignalAttr에는 좌우 반전을 무조건 적용하지 않는다. 좌회전 화살표가 우회전 모양으로 바뀌기 때문이다. 색상 hue를 크게 바꾸거나 색을 제거하는 증강도 점등 상태 라벨과 모순될 수 있다. detector의 flip 가능 여부와 상태 crop의 정책을 분리한다.

FDA와 pseudo-labeling은 별도 현장 **개발** 자료가 있고 명확한 domain gap을 확인했을 때 후순위로 검토한다. FDA의 traffic-light detection 결과는 색상·좌회전 판독 보존을 증명하지 않는다.

현재 FocusedSource는 kind가 하나이며 한 이미지에 두 태스크 GT를 모두 주는 새 형식을 직접 지원하지 않는다. 신규 joint-labeled 주행 데이터 도입은 양쪽 supervision mask와 evaluator를 함께 확장해야 한다. 미라벨 채널을 배경으로 채우지 않는다.

### 5.6 L: 학습 목표·공간 표현

| ID | 가설과 변경 | 반드시 확인할 부작용 | 구현 |
| --- | --- | --- | --- |
| L0 | 현행 BCE+Dice, 1-pixel raster | 기준 | 있음 |
| L1 | line 주변 거리 기반 soft target | 선이 두꺼워져 merge·FP가 늘어나는지 | target 생성 변경 |
| L2 | BCE 대신 focal 성분, Dice 유지 | 작은/약한 true line recall 감소 | loss 변경 |
| L3 | tangent 또는 local offset 보조 회귀 | 다중 선이 같은 cell에 있을 때 모순 감독, 끝점 drift | auxiliary head+target 필요 |
| L4 | 같은 trace 내부의 clDice 계열 보조 | 인접 차선·GT의 분리 구간을 잘못 연결 | differentiable loss 필요 |
| L5 | instance 좌표와 LaneIoU/Line IoU | assignment·confidence 및 정지선 표현 | instance head 실험에 포함 |

L1의 초기 공간 폭은 output grid 기준 sigma 0.5 / 1.0으로 제안한다. 1-cell 중심선의 양자화 민감도를 완화하는 비교이며 GT를 평가 시 넓히는 것이 아니다. 원본 annotation의 연결·분리와 visibility를 유지한다. 현재 데이터가 연속 중심선으로 표현한 점선을 새 paint-segmentation 목표로 바꾸지 않는다.

L2는 gamma=0 대조와 gamma=2를 시작점으로 한다. 불균형 효과와 class weight 효과를 한 번에 섞지 않는다. L3/L4의 보조 loss 계수는 동일 train batch에서 주 loss와 gradient 규모를 본 뒤 한 자리수 배율 범위에서 정한다. 정규화가 다른 논문의 coefficient를 그대로 복사하지 않는다.

현재 decoder는 detach→NumPy→SciPy를 포함한다. decode_roadmark_points 또는 match_roadmark_lines를 그대로 training loss에 넣으면 미분 경로가 끊긴다. L3~L5는 raw GT 기하에 대한 PyTorch 표현과 assignment를 설계해야 한다.

### 5.7 M: 모델 표현과 시간 정보

| 우선순위 | 후보 | 필요한 변경과 비교 |
| --- | --- | --- |
| 먼저 | 기존 stride-4 맵 + P1/P2 | weights 변경 없이 양자화 기여 분리 |
| 다음 | 맵 + subpixel offset 또는 tangent | 기존 logits head 보존, 보조 출력·consumer·export 버전 명시 |
| 다음 | 더 세밀한 decoder 또는 input 해상도 | 위치·recall과 GPU/CPU 시간의 실제 trade-off |
| 조건부 | CLRNet/CLRerNet 계열 instance refinement | 기존 YOLO body 재사용 가능성을 확인; 새 head warmup과 같은 학습 예산 대조 |
| 조건부 | Bézier / hybrid row-column anchor | 곡선·분기·가림·정지선을 별도 검증; 외부 class contract 유지 |
| 후속 | causal temporal association·filter | 학습된 temporal model 전에 reference와 비교; 상태 reset·지연·ghost line 측정 |
| 후속 | LaneTCA / SIGMA-Lane 계열 | 독립 sequence train/dev, 카메라별 state, causal mask, export 및 4060 시간 측정 |
| 첫 라운드 제외 | cold diffusion·3D/BEV topology 전체 교체 | 반복 연산·3D GT·후단 책임이 현행 2D 관측 계약보다 큼 |

새 head는 성숙한 기존 head와 300-step 결과만으로 비교하지 않는다. body를 고정해 새 출력의 학습이 실제 진행되는 예산을 확보한 다음 joint 예산을 더하고, 사용한 총 자원을 모두 기록한다. 나중에 더 나은 구조를 고른 경우 기존 pretrained YOLO26-s부터의 재학습을 대조할 수 있다. 무작위 초기화 전체 재학습은 초기 라운드의 기본값이 아니다.

시간 정보를 쓰는 방법은 프레임 안의 자글거림과 프레임 사이 흔들림을 구분한다. 미래 frame으로 smoothing한 결과를 online inference라고 보고하지 않는다. 좌우 camera state를 공유하거나 재생 시작에 지난 run의 state를 이어쓰지 않는다.

외부 관측 의미는 유지한다. 내부 표현·TorchScript tensor 계약이 바뀌면 기존 checkpoint loader·export metadata·inference decoder와 영향을 받는 소비자를 함께 갱신하고, 구형 가중치는 명시적인 구형 경로로 읽는다. 일괄 strict=False로 누락 weight를 숨기지 않는다.

## 6. SignalAttr 실험 설계

PV26 신호등 박스 비교와 상태 판독 비교를 분리한 뒤 연결한다.

| ID | 내용 | 목적 |
| --- | --- | --- |
| S0 | 현재 포함 legacy 가중치의 출력 의미 확인 | 제품 state_valid=false 기준 확인; 상태 성능 후보가 아님 |
| S1 | 제품 crop, legacy weights 초기화, natural sampling | 의미가 바뀐 상태 head의 자연분포 대조 |
| S2 | 같은 조건, balanced sampling | 희귀 상태와 false positive trade-off |
| S3 | S1/S2 유망 sampling에서 LR 비교 | 현재 1e-3 기준 주변 3e-4 / 1e-3 / 3e-3 |
| S4 | 유망 설정 + detector crop 오차를 반영한 crop jitter | GT crop→predicted crop 성능 저하 완화 |
| S5 | 필요 시 state head 재초기화 또는 random init 대조 | legacy arrow 의미의 잔존 영향 확인 |

config의 logical 256 / microbatch 128, 128×128 crop, padding 0.15, WD 1e-4를 시작 대조로 둔다. classifier만의 속도와 분포가 달라 PV26 step 수를 그대로 예산으로 쓰지 않는다. 203,496개의 train crop을 기준으로 1 / 3 / 6회분 draws가 약 795 / 2,385 / 4,770 update(logical 256)에 해당한다. 이는 replacement sampling의 노출량 환산이며 고유 표본을 전부 방문한다는 뜻은 아니다.

crop snapshot은 all_off_is_valid=true이므로 S1~S4는 이를 고정한다. 읽을 수 없는 상태는 valid off와 다르다. off 제외 정책을 연구하려면 실제 주석 의미를 확인한 별도 데이터 비교로 분리한다.

legacy 초기화와 상태 의미 변경이 중요한 비교 항목이다. 장기 학습이 됐다는 이유만으로 정확한 left_arrow 의미를 얻었다고 가정하지 않는다. 차량·보행자별 검증, 원형 green↔left_arrow, red+arrow 조합, state_valid 비율을 확인한다.

S4는 개발 train 영상의 detector 예측 box를 GT와 연결하거나 GT box에 제한된 perturbation을 주어 실제 ROI 오차를 반영한다. final bag 예측 crop을 train에 넣지 않는다. 원본 image/scene별 train/dev 분리를 유지한다.

제품용 승자를 정한 후에는 같은 SignalAttr를 baseline PV26와 후보 PV26에 붙여 검출기 변화만 비교한다. 그 다음 같은 검출기에서 SignalAttr 후보를 비교한다. 최종 조합의 승리를 어느 구성요소의 효과인지 구분할 수 있게 한다.

## 7. 실행 순서·예산

### 7.1 추천 순서

1. **준비:** 저장장치, 기준 weights, saved train/val membership, 개발 subset, 제품 SignalAttr 데이터 의미를 확인한다. final bag의 라벨 준비는 예측 결과와 독립적으로 진행할 수 있다.
2. **P 비교:** P0~P3를 개발 영상에 적용해 자글거림의 추출 경로 기여를 확인한다. 신호등 threshold와 학습 weights는 고정한다.
3. **FC 비교:** 검출 loss reset/mature를 FJ-M에서 비교한다. 하나를 subsequent joint 실험의 공통 정책으로 둔다.
4. **F 비교:** 6개 LR·freeze 후보를 같은 예산으로 선별한다. 대조군을 계속 남겨 추가 학습 자체의 효과를 측정한다.
5. **D/L/O 비교:** 개발 오류에 직접 대응하는 후보를 각 family에서 하나씩 먼저 붙인다. trade-off가 없는 개선만 조합한다.
6. **S 비교:** SignalAttr는 별도 실험으로 평가하고 같은 GPU 실행은 직렬화한다.
7. **반복 확인:** 유망 recipe와 같은 예산 대조군을 추가 seed에서 반복하고 전체 AIHub 개발 validation으로 비교한다.
8. **최종 후보 확정:** model·SignalAttr·threshold·선 추출·시간 처리·평가 protocol을 확정한다.
9. **최종 MCAP:** 기준과 사전 선택 후보를 동일 protocol로 비교하고 결과·오류 사례·미측정 조건을 보고한다.

FC/P 결과가 나쁘면 더 복잡한 방법을 자동으로 추가하지 않는다. 실패 원인과 중심 동작부터 확인한다. FD에서 충분한 개선을 얻으면 joint 구조 변경을 필수 단계로 강제하지 않는다.

### 7.2 초기 자원 배분 제안

| 단계 | budget | 해석 |
| --- | --- | --- |
| 초기 진단 | 후보당 1,000 update = 32,000 draws, batch 32 | 학습·비정상·퇴행 확인. 느린 후보를 성능만으로 확정 탈락시키지 않음 |
| 1차 선별 | 후보당 누적 4,000 update = 128,000 draws | 주기 검증 추세와 대표 조건 비교 |
| 유망 후보 | 대조군 포함 누적 12,000 update = 384,000 draws | 짧은 순위가 유지되는지 확인 |
| 반복 | 유망 후보와 대조군을 seed 27·28에서 같은 총예산 | paired 차이와 일관성 |
| 구조 교체 | 새 head warmup을 별도로 산정 | 성숙 head와 동일 warmup step을 강제하지 않음 |

동일 family의 cosine max_steps는 **처음부터 12,000으로 고정**한다. --steps로 1k·4k에서 중간 정지하고 동일 run을 resume한다. 살아남은 후보의 max_steps를 4k→12k로 바꾸면 초반 LR 자체가 달라지므로 동일 schedule 비교가 아니다. FC도 같은 horizon을 따른다.

6개 F 후보를 4k까지, 그중 대조군 포함 3개를 12k까지 돌리면 총 48,000 update다. 이는 batch 32에서 1,536,000 draws다. 이 예산은 연구 제안이며 지금 실행하지 않았다.

실제 완료 run의 마지막 1,000 step에서 load+update 중앙값은 약 0.624초였고, 전체 validation 57,700장은 가중치당 약 641~643초였다. 동일 속도를 단순 적용하면 48k update는 약 8.3 GPU시간이다. 여기에 주기 검증·저장·데이터 준비·마지막 full validation이 추가된다. 현행 CLI는 완료 시 latest/best/best_roadmark를 모두 full 평가하므로 run당 약 32분이 추가될 수 있다. FD·PCGrad·새 head·SignalAttr의 시간은 이 값을 그대로 적용하지 않는다.

core F 탐색은 대략 반나절 규모, 추가 seed와 D/L/O는 별도 예산으로 계획한다. 이것은 총 소요시간 보장이 아니다. 첫 실제 후보의 속도와 개선량을 보고 총 GPU시간을 다시 산정한다. broad grid를 전부 실행하는 것이 완료 조건은 아니다.

## 8. 구현 인수인계: 현재 가능·추가 필요

| 항목 | 실제 상태 / 다음 수정 지점 |
| --- | --- |
| checkpoint weights로 새 학습 | run_pv26_train.py --initial-checkpoint, TUI K 지원. E는 기존 optimizer·schedule 재개. stage2 전용 config 추가 |
| LR·stage·source ratio | config와 cli.py 지원. 공통 mutable config/pv26.yaml 대신 실험별 별도 config |
| frozen backbone 범위 | stage=roadmark는 detector 전체+BN 고정. neck만·BN만 고정 등은 추가 필요 |
| 검출 loss 성숙 단계 유지 | detector_loss_schedule: restart/mature 지원. 기존 run 기본은 restart |
| 동일 saved index로 새 run | stage2 config의 data.index_run 또는 --index-run을 새 run에 snapshot. single-stage도 원본 joint 목록에서 해당 source만 선택 |
| 전체 태스크의 동일 평가 | --evaluate-only + --eval-stage joint + --eval-index-run 지원. 기존 geometry/evaluation 재사용 |
| warm-start search | run_pv26_method_search.py가 stage2 base의 initial_checkpoint·index_run을 보존하고 공통 joint 개발 평가를 수행 |
| search 선택 기준 | stage2는 signal/roadmark 점수를 별도 보고하고 후보를 자동 탈락시키지 않음. 검토 후 --promote로 다음 rung 후보 지정 가능 |
| class/scene sampling | source weight와 least_used_of_two는 있음. source 내부 strata sampling은 추가 필요 |
| 새로운 loss/offset/instance | dataset targets·model·loss·decode·export를 함께 수정 |
| MCAP inference | predict_pv26_mcap.py의 오프라인 streaming adapter 추가. 별도 개발 bag 두 영상으로 확인 |
| 상태 연결 | predict_pv26.py --signal-checkpoint / SignalAttrRuntime 있음 |
| SignalAttr 의미 | 제품용 checkpoint 학습 필요. legacy 의미를 metadata 문자열만 바꿔서 승격하지 않음 |
| 자글거림·시간 metric/AP | subpixel·국소 보정 좌표와 클래스별 threshold의 실제 평가 경로 추가. 공간·시간 전용 metric/AP는 미구현 |
| artifact 저장 | kai의 runs 하위를 사용. --artifact-root는 별도 root가 필요한 환경에서만 선택 |

새 run은 저장된 입력 index를 읽고 stage가 해당하는 source만 선택한 뒤 자체 train_samples.jsonl/val_samples.jsonl로 snapshot한다. 현재 _configuration의 기존 artifact 거부와 충돌하지 않는다. 새 corpus 형식·범용 manifest 체계를 만들 필요는 없다.

새 checkpoint 로드 시 추가 head만 초기화하고 기존 weights의 예상 대응을 명시한다. resume는 model·optimizer·scheduler·sampler·BN·loss policy를 복구한다. 이 경계의 호환성 확인을 내부 함수마다 반복하지 않는다.

### 실제 지원하는 CLI 모양

아래 명령은 **구현된 경로의 구조 예시**다. 저장소와 적절한 실험 output 경로를 준비한 후 사용한다. 이 문서의 BN freeze policy는 아직 CLI flag가 아니다.

~~~bash
cd /home/kai/yolopv26
.venv/bin/python tools/run_pv26_train.py \
  --config config/pv26_stage2.yaml \
  --output-dir runs/20260923_stage2_fj_m \
  --steps 1000 --seed 26
~~~

output은 이 호스트의 runs 하위에 둔다. 사용자는 kai 환경에 별도 외장 SSD가 없으며 이 경로를 사용할 것을 명시했다. _output_directory는 기본으로 REPO_ROOT/runs 하위인지를 확인한다.

기존 원본 _source_records는 source별 파일 탐색을 수행한다. --sample-limit은 정렬된 앞부분을 고르므로 scene을 대표하는 subset sampler로 사용하지 않는다. 외부 saved index 전달을 구현한 뒤 기존 membership을 활용하면 반복 탐색도 줄일 수 있다.

## 9. 완료 조건과 다음 에이전트 전달문

연구 라운드가 끝났다고 보고하려면 다음 결과가 있어야 한다.

- 어떤 축을 실제 비교했고 어떤 축은 근거 부족·비용·개선 부재로 보류했는지 구분.
- 기준 모델과 후보의 같은 평가 목록·threshold·관측 의미에서의 비교.
- 차선 시각 품질 개선과 신호등 검출·상태 유지, 추론 시간 사이의 trade-off.
- 제품 SignalAttr와 PV26 연결 결과. GT crop의 높은 점수만으로 대체하지 않음.
- 최종 bag의 GT가 있는 부분의 정량 결과와 전 시퀀스의 오류·지연 분석.
- 다른 날·장소·상태에 대해서는 무엇이 미측정인지 명시.

다음 에이전트에 그대로 전달할 작업 요약:

> 80k latest.pt를 기준으로 이 문서의 FC/P/F부터 구현·실험한다. 먼저 실제 artifact 저장소와 saved index 입력, 검출 loss 일정, stage가 달라도 동일한 두 태스크 평가를 해결한다. 기존 CLI·evaluator를 확장하고 별도 실험 harness를 만들지 않는다. 최종 260920 14:07:38 MCAP은 학습·튜닝에서 제외한다. SignalAttr는 제품 crop 의미로 학습한 후보를 사용한다. 개발 자료로 후보·threshold를 고른 뒤 별도 MCAP protocol대로 기준 모델과 최종 후보를 비교한다. 원본 run·bag·기존 미커밋 수정을 보존한다.

## 10. 구현 진행과 실제 확인 결과

사용자의 후속 구현 요청으로 다음 항목을 추가했다. 구현 전의 계획 숫자·후보 설명은 실험 가설이며 아래 결과와 구분한다.

| 경로 | 완료 및 확인 범위 |
| --- | --- |
| stage2 preset | config/pv26_stage2.yaml이 latest.pt와 저장된 train/val 목록, 12k cosine horizon, FJ-M LR를 지정 |
| FC | restart/mature 검출 loss 정책을 새 run 설정에 저장. 80k의 0.1/0.9 비율을 mature에서 유지 |
| F | --index-run, stage/LR override, single-stage index snapshot 구현. 실제 1-step CPU FD와 BF16 GPU joint smoke 완료 |
| 자동 탐색 | 기존 runner를 stage2 checkpoint·고정 데이터 목록·공통 joint 평가·중간 rung 재개에 연결. FC 두 후보의 300/1,000/4,000-step rung 완료. F 여섯 후보는 restart 정책으로 시작 |
| 공통 평가 | single-stage checkpoint를 --eval-stage joint와 --eval-index-run으로 신호등+도로표식 같은 목록에서 평가. 실제 CPU 개발 표본으로 확인 |
| P1/P2/P3 | grid/subpixel/smooth 좌표와 흰·노란·정지선별 threshold를 이미지 추론·평가에 연결. P2는 연속 구간의 중간만 국소 2차 다항식으로 보정하고 양끝은 보존 |
| MCAP 라벨·입력·점수 | prepare_pv26_mcap_labels.py로 2Hz+이벤트 프레임 추출, predict_pv26_mcap.py로 현재 FocusedPerception 관측 JSONL 게시, evaluate_pv26_mcap.py로 같은 프레임의 박스·상태·선분 결합. 다른 날짜 개발 bag의 카메라 프레임과 실제 AIHub 라벨로 각 경로를 시험 |

P1 진단은 80k 가중치와 AIHub roadmark validation의 전체 목록에 균등하게 걸친 128장을 사용했다. 동일 데이터·모델·기본 threshold에서 grid F1은 0.62905, subpixel F1은 0.63090이었다. 흰 차선은 0.65306→0.65546, 노란 차선 0.62000→0.62000, 정지선 0.25000→0.25000이었다. 매칭 성공 선 평균 위치 오차는 흰 차선 2.146→2.051px, 노란 차선 2.839→2.767px, 정지선 2.792→2.855px였다. 6장 비교에서도 subpixel은 선 개수를 그대로 두고 좌표를 약 1.3~2.3px 움직였으나 장면별 위치 효과는 섞여 있었다.

현재 주기 검증과 같은 2,048장(traffic·roadmark 각 1,024장)을 80k 동일 가중치로 다시 평가했다. 0.65는 앞선 128장 개발 표본에서 선택한 탐색값이므로, 이 2,048장도 독립 최종자료가 아니라 더 넓은 개발 평가다.

| 선 추출 / 공통 threshold | signal F1 | 선분 전체 F1 | 흰 차선 | 노란 차선 | 정지선 |
| --- | ---: | ---: | ---: | ---: | ---: |
| grid / 0.50, 저장된 80k 주기 검증 | 0.77212 | 0.65431 | 0.69163 | 0.57642 | 0.31250 |
| grid / 0.65 | 0.77212 | 0.65824 | 0.69387 | 0.58518 | 0.31792 |
| subpixel / 0.65 | 0.77212 | 0.65990 | 0.69601 | 0.58645 | 0.31214 |
| smooth / 0.65 | 0.77212 | 0.65966 | 0.69601 | 0.58391 | 0.31792 |

국소 보정은 F1의 일관된 우위를 만들지 못했다. 정지선과 노란 차선의 차이가 있어 기본값 grid / 0.50을 유지하고 후보로만 남긴다. 처리시간 진단은 1280×720 AIHub 두 장을 800×600으로 준비한 뒤 RTX 4060에서 동일 프로세스의 4회 반복 중앙값이다. SignalAttr, JPEG 읽기, ROS·LiDAR·후단은 포함하지 않는다. grid 0.50 27.7ms, grid 0.65 26.5ms, subpixel 0.65 27.5ms, smooth 0.65 31.1ms였다. 반복 4회만의 결과이므로 운영 p95나 33ms 목표 충족 증거가 아니다.

관련 test/test_focused_*, SignalAttr, MCAP scorer 테스트 37개가 통과했다. GPU smoke는 임시 64×64·logical 2·1 step으로 코드 연결만 확인했으며 실제 608×800 장기 학습·품질 확인이 아니다. 임시 GPU 출력은 실행 종료 시 정리했다. 지정 최종 MCAP에는 새 모델 추론을 실행하지 않았다.

기존 9월 19일 MCAP에서 100장을 PV26+50-step SignalAttr로 읽었으나 첫 구간에는 검출된 신호등이 0개였다. 그래서 이 결과만으로 기록 신호등 상태 평가를 완료했다고 하지 않는다. 별도의 실제 AIHub 신호등 이미지에서는 PV26 박스 2개에 제품용 SignalAttr의 유효 상태가 붙는 경로를 확인했다.

SignalAttr S1 자연분포는 기존 제품 crop에서 50/5,000 step을 진행했다. 개발 crop 1,024장의 상태 macro F1은 0.66753, state_valid coverage는 0.83203이었다. 차량용 left_arrow F1 0.640(양성 67), 보행자 green F1 0.250(양성 7) 등 희귀 상태 불확실성이 크다. 이 초기 수치를 최종 품질로 해석하지 않는다. run은 runs/20260923_signal_attr_stage2_natural에 보존했다. checkpoints는 약 3.8MiB지만 자체 data_snapshot이 약 352MiB여서 실행 전체가 약 356MiB였다. 사용자의 저장 위치 정정에 따라 남은 학습도 runs 아래에서 진행할 수 있다.

사용자 요청에 따라 후속 실험 산출물은 kai의 runs에 저장한다. FC 두 정책은 같은 시작 가중치, 저장된 train/val 목록, seed 26, 개발 traffic/roadmark 각 1,024장에서 비교했다.

| 누적 step | restart 신호등 / 선분 F1 | mature 신호등 / 선분 F1 |
| ---: | ---: | ---: |
| 300 | 0.77247 / 0.65401 | 0.77187 / 0.65798 |
| 1,000 | 0.77721 / 0.65550 | 0.77377 / 0.65585 |
| 4,000 | 0.77476 / 0.65724 | 0.77371 / 0.64986 |

300 step에서는 성능 우위가 갈렸으나 4,000 step에서는 restart가 두 지표에서 모두 높았다. 따라서 F 여섯 후보의 공통 검출 loss 정책으로 restart를 선택했다. 이것은 AIHub 개발 평가의 선택이며 지정 MCAP 최종 평가 결과는 아니다. runs/20260923_stage2_fc/search_results.json에 단계별 원자료가 있다.

F 여섯 후보의 첫 300-step rung도 동일한 개발 traffic/roadmark 각 1,024장으로 완료했다. 아래는 장기 승격 판정이 아닌 초기 관찰이다. fd 후보의 검출기 전체가 고정돼 있어 신호등 F1은 80k 기준값을 유지했다.

| 후보 | 신호등 F1 | 선분 F1 |
| --- | ---: | ---: |
| fj_l | 0.77167 | 0.65508 |
| fj_m | 0.77247 | 0.65401 |
| fj_h | 0.76073 | 0.65288 |
| fd_l | 0.77212 | 0.65422 |
| fd_m | 0.77212 | 0.65508 |
| fd_h | 0.77212 | 0.65405 |

누적 1,000-step rung에서 fj_l은 신호등/선분 F1 0.77201/0.65467, fj_m 0.77721/0.65550, fj_h 0.79012/0.64268, fd_l 0.77212/0.65288, fd_m 0.77212/0.65310, fd_h 0.77212/0.65603이었다. 특히 fj_h는 신호등이 오르고 차선이 떨어져 단일 평균 점수로 선택하면 안 되는 사례다. 결과 파일은 runs/20260923_stage2_f/search_results.json이다.

이 여섯 레시피는 독립적인 LR·학습법 효과를 추정할 수 없으므로 [3-seed 교차 요인 DoE](20260923_PV26_FORMAL_DOE.md)를 추가했다. 2026-09-23 현재 첫 300-step screening이 runs/20260923_stage2_doe_*에서 백그라운드로 진행 중이며, 87개 결과가 나올 때까지 교차 효과를 완료했다고 말하지 않는다.

남은 구현은 세분화된 D/L/O/M 조건부 후보, 장기 GPU 비교, 제품 SignalAttr 장기 학습, 최종 MCAP 정답 라벨·정량 평가다. MCAP scorer는 ignore_regions가 지정된 프레임을 전체 제외하고 그 수를 보고하므로, 실제 최종 라벨에서 그런 프레임이 생기면 부분 영역 처리 계약을 먼저 완성해야 한다. 시간 안정성·GT crop 상태 단독 지표도 별도로 남아 있다. stage2 자동 탐색은 먼저 FC 두 후보, 다음 F 여섯 후보를 runs 아래에서 각 rung별로 수행한다. 각 rung의 신호등·차선 결과를 보고 장기 승격 대상을 선택한다.
